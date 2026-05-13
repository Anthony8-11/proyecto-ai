"""Pandas-based analytics: logs every step and computes learning metrics."""

import pandas as pd


class Analytics:
    def __init__(self):
        self.episode = 1
        self._records: list[dict] = []
        self._episode_rewards: list[float] = []
        self._current_episode_reward = 0.0

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

    def next_episode(self):
        self._episode_rewards.append(self._current_episode_reward)
        self._current_episode_reward = 0.0
        self.episode += 1

    def total_reward(self) -> float:
        return round(self._current_episode_reward, 2)

    def reset(self):
        self.__init__()

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        if not self._records:
            return {"steps": 0, "episodes": 0, "episode_rewards": []}

        df = pd.DataFrame(self._records)
        action_dist = df["action"].value_counts().to_dict()

        return {
            "steps": len(df),
            "episodes": self.episode,
            "episode_rewards": self._episode_rewards[-50:],  # last 50
            "mean_reward_last10": round(
                pd.Series(self._episode_rewards[-10:]).mean(), 2
            ) if self._episode_rewards else 0,
            "action_distribution": {int(k): int(v) for k, v in action_dist.items()},
            "avg_resources": round(df["resources"].mean(), 2),
            "condition_distribution": df["env_condition"].value_counts().to_dict(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)
