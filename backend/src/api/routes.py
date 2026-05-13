from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..engine.analytics import Analytics
from ..engine.environment import Environment
from ..engine.rl_agent import RLAgent
from ..engine.search import ClassicSearch

router = APIRouter()

# Shared simulation state (per-process; replace with Redis for multi-worker)
_env = Environment()
_agent = RLAgent(n_states=_env.n_states, n_actions=_env.n_actions)
_analytics = Analytics()
_search = ClassicSearch(_env)


@router.get("/status")
def status():
    return {"status": "ok"}


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
    rl_rewards = _analytics.total_reward()
    search_rewards = _search.evaluate(episodes=10)
    return {"rl": rl_rewards, "classic_search": search_rewards}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "step":
                state = _env.get_state()
                action = _agent.select_action(state)
                next_state, reward, done = _env.step(action)
                _agent.learn(state, action, reward, next_state)
                _analytics.log(state, action, reward)

                await websocket.send_json({
                    "state": state,
                    "action": int(action),
                    "reward": reward,
                    "done": done,
                    "episode": _analytics.episode,
                    "total_reward": _analytics.total_reward(),
                    "epsilon": round(_agent.epsilon, 4),
                })

                if done:
                    _env.reset()
                    _analytics.next_episode()

    except WebSocketDisconnect:
        pass
