from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..engine.analytics import Analytics
from ..engine.environment import Environment
from ..engine.logic_engine import filter_actions
from ..engine.ml_model import EnvPredictor
from ..engine.rl_agent import RLAgent
from ..engine.search import ClassicSearch

router = APIRouter()

# Shared simulation state (per-process; replace with Redis for multi-worker)
_env = Environment()
_agent = RLAgent(n_states=_env.n_states, n_actions=_env.n_actions)
_analytics = Analytics()
_predictor = EnvPredictor()
_search = ClassicSearch(_env)

# Retrain the ML predictor every N steps
_ML_RETRAIN_INTERVAL = 50


@router.get("/status")
def status():
    return {"status": "ok", "ml_trained": _predictor.is_trained}


@router.post("/simulation/reset")
def reset_simulation():
    _env.reset()
    _agent.reset()
    _analytics.reset()
    return {"message": "Simulation reset"}


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
            "episode_rewards": ep_rewards[-20:],  # last 20 for chart
        },
        "greedy": {
            "avg_reward": search_results["greedy"],
        },
        "astar": {
            "avg_reward": search_results["astar"],
        },
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "step":
                state = _env.get_state()

                # 1. Logic Engine: filter forbidden actions via predicate rules
                allowed = filter_actions(state, list(range(_env.n_actions)))

                # 2. ML Predictor: get next-condition probabilities for chosen action hint
                ml_probs = _predictor.predict_proba(state, allowed[0])

                # 3. RL Agent: pick best action among allowed (epsilon-greedy)
                action = _agent.select_action(state, allowed)

                next_state, reward, done = _env.step(action)
                _agent.learn(state, action, reward, next_state)
                _analytics.log(state, action, reward)

                # 4. Periodic ML retraining from episode history
                n_records = len(_analytics._records)
                if n_records >= 20 and n_records % _ML_RETRAIN_INTERVAL == 0:
                    import numpy as np

                    df = _analytics.to_dataframe()
                    X = df[["position", "resources", "env_condition", "action"]].values[:-1]
                    y = df["env_condition"].values[1:]
                    _predictor.train(X, y)

                await websocket.send_json({
                    "state": state,
                    "action": int(action),
                    "allowed_actions": allowed,
                    "reward": reward,
                    "done": done,
                    "episode": _analytics.episode,
                    "total_reward": _analytics.total_reward(),
                    "epsilon": round(_agent.epsilon, 4),
                    "ml_probs": ml_probs,
                    "ml_trained": _predictor.is_trained,
                })

                if done:
                    _env.reset()
                    _analytics.next_episode()

    except WebSocketDisconnect:
        pass
