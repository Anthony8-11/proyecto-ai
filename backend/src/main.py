from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import (
    DATA_DIR,
    _agent_dqn,
    _agent_ql,
    _analytics,
    _do_save,
    _predictor,
    router,
)
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: restore persisted training state ────────────────────
    _agent_ql.load(DATA_DIR / "qtable.npy")
    _agent_dqn.load(DATA_DIR / "dqn_checkpoint.pt")
    _predictor.load(DATA_DIR / "ml_model.joblib")

    # Seed episode_rewards from CSV so EpisodeChart shows historical data
    historical = _analytics.load_csv(DATA_DIR)
    if historical:
        _analytics._episode_rewards = historical
        _analytics._saved_episodes_idx = len(historical)
        _analytics.episode = historical[-1]["episode_num"] + 1

    yield

    # ── Shutdown: flush any unsaved state ────────────────────────────
    _do_save()


app = FastAPI(title="Proyecto IA — RL System", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
