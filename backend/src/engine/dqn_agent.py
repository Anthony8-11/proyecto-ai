"""Deep Q-Network (DQN) agent.

Architecture
------------
- QNetwork: 3-input MLP (position / resources / env_condition, all normalised)
  with two 64-unit hidden layers and a linear output of shape (n_actions,).
- ReplayBuffer: fixed-size circular deque for experience replay.
- DQNAgent: epsilon-greedy exploration, batch training against a frozen target
  network that syncs every `target_update_freq` gradient steps.

The public interface mirrors RLAgent exactly so routes.py can switch between
them transparently.
"""

import collections
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """MLP that maps a normalised state vector to one Q-value per action."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Circular experience replay buffer."""

    def __init__(self, capacity: int = 10_000):
        self._buf: deque = collections.deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
class DQNAgent:
    """Deep Q-Network agent with experience replay and target network.

    State encoding
    --------------
    [position / 4, resources / 10, env_condition / 2]  → float32 in [0, 1]
    """

    STATE_DIM = 3
    _NORM = np.array([4.0, 10.0, 2.0], dtype=np.float32)

    def __init__(
        self,
        n_actions: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 100,
        replay_capacity: int = 10_000,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self._epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._replay_capacity = replay_capacity

        self._device = torch.device("cpu")
        self._build_networks()

        self._step_count = 0
        self._last_loss: float = 0.0

    # ------------------------------------------------------------------
    def _build_networks(self) -> None:
        self._online = QNetwork(self.STATE_DIM, self.n_actions).to(self._device)
        self._target = QNetwork(self.STATE_DIM, self.n_actions).to(self._device)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()
        self._optimizer = optim.Adam(self._online.parameters(), lr=1e-3)
        self._loss_fn = nn.MSELoss()
        self._buffer = ReplayBuffer(self._replay_capacity)

    def _encode(self, state: dict) -> np.ndarray:
        return np.array(
            [state["position"], state["resources"], state["env_condition"]],
            dtype=np.float32,
        ) / self._NORM

    # ------------------------------------------------------------------
    def select_action(self, state: dict, allowed_actions: list[int] | None = None) -> int:
        actions = allowed_actions if allowed_actions is not None else list(range(self.n_actions))
        if random.random() < self.epsilon:
            return int(random.choice(actions))
        x = torch.tensor(self._encode(state), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_vals = self._online(x).squeeze(0).numpy()
        return int(max(actions, key=lambda a: float(q_vals[a])))

    def learn(
        self,
        state: dict,
        action: int,
        reward: float,
        next_state: dict,
        done: bool = False,
    ) -> None:
        self._buffer.push(self._encode(state), action, reward, self._encode(next_state), done)
        self._step_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if len(self._buffer) < self.batch_size:
            return

        # --- sample and convert batch ---
        batch = self._buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s_t = torch.tensor(np.array(states), dtype=torch.float32)
        a_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        r_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        ns_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        d_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Current Q-values for taken actions
        q_current = self._online(s_t).gather(1, a_t)

        # Bellman target using frozen target network
        with torch.no_grad():
            q_next = self._target(ns_t).max(dim=1, keepdim=True)[0]
            q_target = r_t + self.gamma * q_next * (1.0 - d_t)

        loss = self._loss_fn(q_current, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding updates on early episodes
        nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=10.0)
        self._optimizer.step()
        self._last_loss = float(loss.item())

        # Sync target network
        if self._step_count % self.target_update_freq == 0:
            self._target.load_state_dict(self._online.state_dict())

    def get_q_values(self, state: dict) -> list[float]:
        """Return the online network's Q-value estimates for the current state."""
        x = torch.tensor(self._encode(state), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return [round(float(v), 3) for v in self._online(x).squeeze(0).numpy()]

    def reset(self) -> None:
        self.epsilon = self._epsilon_start
        self._step_count = 0
        self._last_loss = 0.0
        self._build_networks()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save network weights, optimizer state and training counters."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict": self._online.state_dict(),
                "target_state_dict": self._target.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self._step_count,
            },
            str(path),
        )

    def load(self, path: Path) -> bool:
        """Load checkpoint. Returns True on success."""
        try:
            if not path.exists():
                return False
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            required = {
                "online_state_dict", "target_state_dict",
                "optimizer_state_dict", "epsilon", "step_count",
            }
            if not required.issubset(checkpoint.keys()):
                return False
            self._online.load_state_dict(checkpoint["online_state_dict"])
            self._target.load_state_dict(checkpoint["target_state_dict"])
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = float(checkpoint["epsilon"])
            self._step_count = int(checkpoint["step_count"])
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    @property
    def last_loss(self) -> float:
        return round(self._last_loss, 6)

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)
