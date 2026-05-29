"""Analytics: logs every step and computes learning metrics.

Design note
-----------
``summary()`` is called every 3 seconds while the simulation is running.
The original implementation rebuilt a full Pandas DataFrame from all records
on each call.  After a long session (10 000+ steps) this was measurably slow.

The current implementation maintains *running accumulators* so that
``summary()`` runs in O(1) regardless of session length.  A Pandas DataFrame
is still built on demand via ``to_dataframe()`` — used only for the ML
retraining path which fires at most once every 50 steps.
"""

from pathlib import Path

import pandas as pd


class Analytics:
    def __init__(self):
        self.episode = 1
        self._records: list[dict] = []
        self._episode_rewards: list[dict] = []   # list[dict] (not list[float])
        self._current_episode_reward = 0.0
        self._episode_start_idx = 0              # index into _records for current episode
        self._saved_records_idx = 0              # records already flushed to CSV
        self._saved_episodes_idx = 0             # episode entries already flushed to CSV

        # ── Running accumulators (keep summary() at O(1)) ──────────────
        self._step_count: int = 0
        self._resource_sum: float = 0.0
        self._action_counts: dict[int, int] = {}
        self._condition_counts: dict[int, int] = {}

    # ------------------------------------------------------------------
    def log(self, state: dict, action: int, reward: float):
        self._records.append({
            "episode": self.episode,
            "position": state["position"],
            "resources": state["resources"],
            "env_condition": state["env_condition"],
            "action": action,
            "reward": reward,
        })
        self._current_episode_reward += reward

        # Update O(1) accumulators
        self._step_count += 1
        self._resource_sum += state["resources"]
        self._action_counts[action] = self._action_counts.get(action, 0) + 1
        cond = state["env_condition"]
        self._condition_counts[cond] = self._condition_counts.get(cond, 0) + 1

    def next_episode(self, epsilon: float = 1.0, agent_type: str = "qlearning") -> None:
        steps_this_ep = len(self._records) - self._episode_start_idx
        self._episode_rewards.append({
            "episode_num": self.episode,
            "total_reward": round(self._current_episode_reward, 2),
            "steps": steps_this_ep,
            "epsilon": round(epsilon, 4),
            "agent_type": agent_type,
        })
        self._episode_start_idx = len(self._records)
        self._current_episode_reward = 0.0
        self.episode += 1

    def total_reward(self) -> float:
        return round(self._current_episode_reward, 2)

    def reset(self):
        self.__init__()

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return session statistics in O(1) using running accumulators.

        No DataFrame is created here — all values are maintained incrementally
        in ``log()`` and ``next_episode()``.
        """
        ep_totals = [r["total_reward"] for r in self._episode_rewards]
        last10 = ep_totals[-10:]
        mean_last10 = round(sum(last10) / len(last10), 2) if last10 else 0

        last50 = self._episode_rewards[-50:]
        return {
            "steps": self._step_count,
            "episodes": self.episode,
            "episode_rewards": [r["total_reward"] for r in last50],
            "episode_agents":  [r["agent_type"]   for r in last50],
            "mean_reward_last10": mean_last10,
            "action_distribution": dict(self._action_counts),
            "avg_resources": (
                round(self._resource_sum / self._step_count, 2)
                if self._step_count else 0
            ),
            "condition_distribution": dict(self._condition_counts),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    # Maximum number of step records kept in memory after a CSV flush.
    # This is the window used by the ML retraining call in routes.py.
    # All older records are safely stored in training_log.csv on disk.
    _RECORDS_MEMORY_CAP = 1_000

    def save_csv(self, data_dir: Path) -> None:
        """Append unsaved records and episode rewards to CSV files.

        After flushing, trims the in-memory ``_records`` list to
        ``_RECORDS_MEMORY_CAP`` entries.  Older records are on disk;
        the running accumulators keep summary() correct regardless.
        """
        data_dir.mkdir(parents=True, exist_ok=True)

        # Flush unsaved step records
        new_records = self._records[self._saved_records_idx:]
        if new_records:
            log_path = data_dir / "training_log.csv"
            df = pd.DataFrame(new_records)
            df.to_csv(
                log_path, index=False, mode="a",
                header=not log_path.exists(),
            )
            self._saved_records_idx = len(self._records)

        # Flush unsaved episode rewards
        new_episodes = self._episode_rewards[self._saved_episodes_idx:]
        if new_episodes:
            ep_path = data_dir / "episode_rewards.csv"
            ep_df = pd.DataFrame(new_episodes)
            ep_df.to_csv(
                ep_path, index=False, mode="a",
                header=not ep_path.exists(),
            )
            self._saved_episodes_idx = len(self._episode_rewards)

        # ── Trim records to cap in-memory footprint ───────────────────
        # Everything older than the cap is already on disk.
        keep = self._RECORDS_MEMORY_CAP
        if len(self._records) > keep:
            discard = len(self._records) - keep
            self._records = self._records[discard:]
            self._saved_records_idx = max(0, self._saved_records_idx - discard)
            self._episode_start_idx = max(0, self._episode_start_idx - discard)

    def load_csv(self, data_dir: Path) -> list[dict]:
        """Load historical episode rewards from CSV. Returns list of dicts."""
        ep_path = data_dir / "episode_rewards.csv"
        if not ep_path.exists():
            return []
        try:
            df = pd.read_csv(ep_path)
            required = {"episode_num", "total_reward", "steps", "epsilon", "agent_type"}
            if not required.issubset(df.columns):
                return []  # schema mismatch — ignore stale file
            return df.to_dict("records")
        except Exception:
            return []
