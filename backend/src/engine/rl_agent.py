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

    def learn(self, state: dict, action: int, reward: float, next_state: dict):
        s, s_ = state["index"], next_state["index"]
        td_target = reward + self.gamma * np.max(self.q_table[s_])
        self.q_table[s, action] += self.alpha * (td_target - self.q_table[s, action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        self.epsilon = 1.0
        self.q_table[:] = 0
