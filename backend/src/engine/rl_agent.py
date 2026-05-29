from pathlib import Path

import numpy as np


class RLAgent:
    """Tabular Q-Learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state: dict, allowed_actions: list[int] | None = None) -> int:
        actions = allowed_actions if allowed_actions is not None else list(range(self.n_actions))
        if np.random.random() < self.epsilon:
            return int(np.random.choice(actions))
        q_vals = self.q_table[state["index"]]
        return int(max(actions, key=lambda a: q_vals[a]))

    def learn(
        self,
        state: dict,
        action: int,
        reward: float,
        next_state: dict,
        done: bool = False,
    ) -> None:
        s, s_ = state["index"], next_state["index"]
        td_target = reward if done else reward + self.gamma * np.max(self.q_table[s_])
        self.q_table[s, action] += self.alpha * (td_target - self.q_table[s, action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values(self, state: dict) -> list[float]:
        """Return Q-values for the current state (one per action)."""
        return [round(float(v), 3) for v in self.q_table[state["index"]]]

    def reset(self):
        self.epsilon = 1.0
        self.q_table[:] = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save Q-table and epsilon to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self.q_table)
        meta_path = Path(str(path).replace(".npy", "_meta.npy"))
        np.save(str(meta_path), np.array([self.epsilon]))

    def load(self, path: Path) -> bool:
        """Load Q-table and epsilon. Returns True on success."""
        try:
            if not path.exists():
                return False
            loaded = np.load(str(path))
            if loaded.shape != (self.n_states, self.n_actions):
                return False  # stale file from different env config
            self.q_table = loaded
            meta_path = Path(str(path).replace(".npy", "_meta.npy"))
            if meta_path.exists():
                self.epsilon = float(np.load(str(meta_path))[0])
            return True
        except Exception:
            return False
