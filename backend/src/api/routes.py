from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..engine.analytics import Analytics
from ..engine.dqn_agent import DQNAgent
from ..engine.environment import Environment
from ..engine.logic_engine import (
    engine_backend,
    filter_actions,
    filter_actions_fast,
    get_active_rules,
    get_active_rules_fast,
)
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

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent.parent / "data"
_AUTO_SAVE_EVERY: int = 1   # default: save after every completed episode
_last_saved_episode: int = 0


def _active_agent() -> RLAgent | DQNAgent:
    return _agent_ql if _agent_type == "qlearning" else _agent_dqn


def _do_save() -> dict:
    """Flush CSVs and model checkpoints to DATA_DIR."""
    _analytics.save_csv(DATA_DIR)
    _agent_ql.save(DATA_DIR / "qtable.npy")
    _agent_dqn.save(DATA_DIR / "dqn_checkpoint.pt")
    _predictor.save(DATA_DIR / "ml_model.joblib")
    return {"saved": True, "episode": _analytics.episode}


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
def status():
    return {
        "status": "ok",
        "agent_type": _agent_type,
        "ml_trained": _predictor.is_trained,
        "logic_engine": engine_backend(),
    }


class AgentTypeBody(BaseModel):
    type: str  # "qlearning" | "dqn"


@router.post("/simulation/agent-type")
def set_agent_type(body: AgentTypeBody):
    global _agent_type, _last_saved_episode
    if body.type not in ("qlearning", "dqn"):
        raise HTTPException(400, detail="type must be 'qlearning' or 'dqn'")

    # 1. Flush current session to CSV before switching so no data is lost
    _do_save()
    _last_saved_episode = _analytics.episode

    # 2. Switch agent type and reset only the newly activated agent
    #    (the old agent keeps its Q-table / DQN weights in memory)
    _agent_type = body.type
    _active_agent().reset()

    # 3. Reset environment to start a clean episode; analytics carries over
    _env.reset()

    return {"agent_type": _agent_type}


@router.post("/simulation/reset")
def reset_simulation():
    global _last_saved_episode
    _env.reset()
    _agent_ql.reset()
    _agent_dqn.reset()
    _analytics.reset()
    _last_saved_episode = 0
    # CSV files are intentionally preserved — history accumulates across resets
    return {"message": "Simulation reset", "agent_type": _agent_type}


@router.post("/simulation/reset-full")
def reset_full():
    """Wipe ALL persisted training state and restart from zero.

    Deletes Q-table, DQN checkpoint, ML model, and both CSV log files so the
    next session starts with a completely untrained agent.
    """
    global _last_saved_episode, _predictor
    # Reset every in-memory component
    _env.reset()
    _agent_ql.reset()
    _agent_dqn.reset()
    _analytics.reset()
    _predictor = EnvPredictor()   # fresh, untrained instance
    _last_saved_episode = 0

    # Remove all persisted files
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in [
        "qtable.npy",
        "qtable_meta.npy",
        "dqn_checkpoint.pt",
        "ml_model.joblib",
        "episode_rewards.csv",
        "training_log.csv",
    ]:
        p = DATA_DIR / fname
        if p.exists():
            p.unlink()

    return {"ok": True, "message": "Reset completo — todos los datos de entrenamiento eliminados"}


@router.post("/simulation/save")
def manual_save():
    """Manually trigger a save of all training state."""
    global _last_saved_episode
    result = _do_save()
    _last_saved_episode = _analytics.episode
    return result


class AutoSaveBody(BaseModel):
    every: int  # number of episodes between auto-saves (1–20)


@router.post("/simulation/autosave-interval")
def set_autosave_interval(body: AutoSaveBody):
    """Configure how often the server auto-saves (in completed episodes)."""
    global _AUTO_SAVE_EVERY
    if not (1 <= body.every <= 20):
        raise HTTPException(400, detail="every must be between 1 and 20")
    _AUTO_SAVE_EVERY = body.every
    return {"auto_save_every": _AUTO_SAVE_EVERY}


@router.get("/simulation/stats")
def get_stats():
    summary = _analytics.summary()
    summary["last_saved_episode"] = _last_saved_episode
    summary["auto_save_every"]    = _AUTO_SAVE_EVERY
    return summary


@router.get("/simulation/comparison")
def get_comparison():
    """Compare RL (current run) against Greedy and A* on fresh environments."""
    ep_rewards = [r["total_reward"] for r in _analytics._episode_rewards]
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


@router.get("/training/history")
def get_training_history():
    """Return the last 300 episode_rewards.csv entries for the learning curve.

    Capped to avoid sending thousands of rows to the browser after very long
    training sessions, which causes recharts to render huge SVG paths.
    """
    ep_path = DATA_DIR / "episode_rewards.csv"
    if not ep_path.exists():
        return {"episodes": []}
    try:
        df = pd.read_csv(ep_path)
        return {"episodes": df.tail(300).to_dict("records")}
    except Exception:
        return {"episodes": []}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global _last_saved_episode

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "step":
                agent = _active_agent()
                state = _env.get_state()

                # 1. Logic Engine — O(1) LUT lookup (precomputed for all 165 states)
                allowed = filter_actions_fast(state)

                # 2. ML Predictor — next-condition probabilities
                ml_probs = _predictor.predict_proba(state, allowed[0])

                # 3. Agent — epsilon-greedy within allowed actions
                action = agent.select_action(state, allowed)

                next_state, reward, done = _env.step(action)
                agent.learn(state, action, reward, next_state, done)
                _analytics.log(state, action, reward)

                # 4. Periodic ML retraining
                # Use the accumulated step counter (not len(_records) which is
                # now capped by save_csv to avoid unbounded memory growth).
                total_steps = _analytics._step_count
                if total_steps >= 20 and total_steps % _ML_RETRAIN_INTERVAL == 0:
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

                # 6 & 7. Heavy optional fields — sent every 5 steps only.
                # q_values (4 floats) and rules_info (10 objects × 4 fields)
                # update slowly and are non-critical for real-time display.
                # This cuts WebSocket payload by ~60% on the off-steps.
                send_heavy = (total_steps % 5 == 0)

                q_values = agent.get_q_values(state) if send_heavy else None
                rules_info = get_active_rules_fast(state) if send_heavy else None

                # Determine explore/exploit mode (always needed for DecisionAnatomy)
                _q = q_values if q_values is not None else agent.get_q_values(state)
                explore = agent.epsilon > 0 and (
                    action not in allowed
                    or _q[action] < max(_q[a] for a in allowed)
                )
                mode = "exploración" if explore else "explotación"

                msg: dict = {
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
                    "logic_engine": engine_backend(),
                    "decision_mode": mode,
                    **dqn_info,
                }
                if q_values is not None:
                    msg["q_values"] = q_values
                if rules_info is not None:
                    msg["rules_info"] = rules_info

                await websocket.send_json(msg)

                if done:
                    _env.reset()
                    _analytics.next_episode(
                        epsilon=round(agent.epsilon, 4),
                        agent_type=_agent_type,
                    )
                    # Auto-save every N completed episodes
                    if _analytics.episode - _last_saved_episode >= _AUTO_SAVE_EVERY:
                        _do_save()
                        _last_saved_episode = _analytics.episode

    except WebSocketDisconnect:
        pass
