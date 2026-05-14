from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..engine.analytics import Analytics
from ..engine.dqn_agent import DQNAgent
from ..engine.environment import Environment
from ..engine.logic_engine import filter_actions
from ..engine.ml_model import EnvPredictor
from ..engine.rl_agent import RLAgent
from ..engine.search import ClassicSearch

router = APIRouter()

# ---------------------------------------------------------------------------
# Shared simulation state (per-process; replace with Redis for multi-worker)
# ---------------------------------------------------------------------------
_env = Environment()
_agent_ql = RLAgent(n_states=_env.n_states, n_actions=_env.n_actions)
_agent_dqn = DQNAgent(n_actions=_env.n_actions)
_analytics = Analytics()
_predictor = EnvPredictor()
_search = ClassicSearch(_env)
_agent_type: str = "qlearning"  # "qlearning" | "dqn"

_ML_RETRAIN_INTERVAL = 50


def _active_agent() -> RLAgent | DQNAgent:
    return _agent_ql if _agent_type == "qlearning" else _agent_dqn


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
def status():
    return {
        "status": "ok",
        "agent_type": _agent_type,
        "ml_trained": _predictor.is_trained,
    }


class AgentTypeBody(BaseModel):
    type: str  # "qlearning" | "dqn"


@router.post("/simulation/agent-type")
def set_agent_type(body: AgentTypeBody):
    global _agent_type
    if body.type not in ("qlearning", "dqn"):
        raise HTTPException(400, detail="type must be 'qlearning' or 'dqn'")
    _agent_type = body.type
    # Reset the newly activated agent so it starts fresh
    _active_agent().reset()
    _analytics.reset()
    _env.reset()
    return {"agent_type": _agent_type}


@router.post("/simulation/reset")
def reset_simulation():
    _env.reset()
    _agent_ql.reset()
    _agent_dqn.reset()
    _analytics.reset()
    return {"message": "Simulation reset", "agent_type": _agent_type}


@router.get("/simulation/stats")
def get_stats():
    return _analytics.summary()


@router.get("/simulation/comparison")
def get_comparison():
    """Compare RL (current run) against Greedy and A* on fresh environments."""
    import pandas as pd

    ep_rewards = _analytics._episode_rewards
    rl_avg = round(pd.Series(ep_rewards[-10:]).mean(), 2) if ep_rewards else 0.0
    search_results = _search.evaluate(episodes=10)

    return {
        "rl": {
            "avg_reward_last10": rl_avg,
            "total_episodes": _analytics.episode,
            "episode_rewards": ep_rewards[-20:],
            "agent_type": _agent_type,
        },
        "greedy": {"avg_reward": search_results["greedy"]},
        "astar": {"avg_reward": search_results["astar"]},
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "step":
                agent = _active_agent()
                state = _env.get_state()

                # 1. Logic Engine — filter forbidden actions via predicate rules
                allowed = filter_actions(state, list(range(_env.n_actions)))

                # 2. ML Predictor — next-condition probabilities
                ml_probs = _predictor.predict_proba(state, allowed[0])

                # 3. Agent — epsilon-greedy within allowed actions
                action = agent.select_action(state, allowed)

                next_state, reward, done = _env.step(action)
                agent.learn(state, action, reward, next_state, done)
                _analytics.log(state, action, reward)

                # 4. Periodic ML retraining
                n_records = len(_analytics._records)
                if n_records >= 20 and n_records % _ML_RETRAIN_INTERVAL == 0:
                    import numpy as np

                    df = _analytics.to_dataframe()
                    X = df[["position", "resources", "env_condition", "action"]].values[:-1]
                    y = df["env_condition"].values[1:]
                    _predictor.train(X, y)

                # 5. DQN-specific telemetry
                dqn_info: dict = {}
                if _agent_type == "dqn":
                    dqn_info = {
                        "dqn_loss": _agent_dqn.last_loss,
                        "dqn_buffer": _agent_dqn.buffer_size,
                    }

                await websocket.send_json({
                    "state": state,
                    "action": int(action),
                    "allowed_actions": allowed,
                    "reward": reward,
                    "done": done,
                    "episode": _analytics.episode,
                    "total_reward": _analytics.total_reward(),
                    "epsilon": round(agent.epsilon, 4),
                    "ml_probs": ml_probs,
                    "ml_trained": _predictor.is_trained,
                    "agent_type": _agent_type,
                    **dqn_info,
                })

                if done:
                    _env.reset()
                    _analytics.next_episode()

    except WebSocketDisconnect:
        pass
