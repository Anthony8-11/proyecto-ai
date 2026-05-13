import random
from dataclasses import dataclass, field


@dataclass
class State:
    position: int       # 0..N_POSITIONS-1
    resources: int      # 0..MAX_RESOURCES
    env_condition: int  # 0=normal, 1=alert, 2=critical


N_POSITIONS = 5
MAX_RESOURCES = 10
CONDITIONS = 3

# Actions: 0=move, 1=allocate_resources, 2=wait, 3=react
N_ACTIONS = 4


class Environment:
    n_states: int = N_POSITIONS * (MAX_RESOURCES + 1) * CONDITIONS
    n_actions: int = N_ACTIONS

    def __init__(self):
        self._state = self._random_state()
        self._step_count = 0

    # ------------------------------------------------------------------
    def reset(self) -> State:
        self._state = self._random_state()
        self._step_count = 0
        return self._state

    def get_state(self) -> dict:
        s = self._state
        return {
            "position": s.position,
            "resources": s.resources,
            "env_condition": s.env_condition,
            "index": self._state_index(s),
        }

    def step(self, action: int) -> tuple[dict, float, bool]:
        self._step_count += 1
        self._apply_dynamic_event()
        reward = self._apply_action(action)
        done = self._step_count >= 200 or self._state.resources <= 0
        return self.get_state(), reward, done

    # ------------------------------------------------------------------
    def _apply_action(self, action: int) -> float:
        s = self._state
        if action == 0:   # move
            s.position = (s.position + 1) % N_POSITIONS
            return 10.0 if s.env_condition == 0 else -5.0
        elif action == 1: # allocate resources
            if s.resources < MAX_RESOURCES:
                s.resources = min(s.resources + 2, MAX_RESOURCES)
                return 10.0
            return -5.0
        elif action == 2: # wait
            return 5.0 if s.env_condition == 0 else -5.0
        elif action == 3: # react
            if s.env_condition > 0:
                s.env_condition = max(s.env_condition - 1, 0)
                s.resources = max(s.resources - 1, 0)
                return 10.0
            return -5.0
        return 0.0

    def _apply_dynamic_event(self):
        s = self._state
        roll = random.random()
        if roll < 0.05:    # 5% critical event
            s.env_condition = 2
            s.resources = max(s.resources - 3, 0)
        elif roll < 0.20:  # 15% alert
            s.env_condition = min(s.env_condition + 1, 2)
        elif roll < 0.40:  # 20% gradual recovery
            s.env_condition = max(s.env_condition - 1, 0)

    @staticmethod
    def _random_state() -> State:
        return State(
            position=random.randint(0, N_POSITIONS - 1),
            resources=random.randint(3, MAX_RESOURCES),
            env_condition=0,
        )

    @staticmethod
    def _state_index(s: State) -> int:
        return s.position * (MAX_RESOURCES + 1) * CONDITIONS + s.resources * CONDITIONS + s.env_condition
