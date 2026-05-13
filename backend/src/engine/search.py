"""Classic search algorithms (A* and Greedy) for comparison against RL."""

import heapq
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class _Node:
    priority: float
    state: Any = field(compare=False)
    cost: float = field(compare=False, default=0.0)
    reward: float = field(compare=False, default=0.0)


def _heuristic(state: dict) -> float:
    """Simple heuristic: prefer low condition + high resources."""
    return (2 - state["env_condition"]) + state["resources"] * 0.5


class ClassicSearch:
    def __init__(self, env):
        self._env = env

    def greedy_action(self, state: dict) -> int:
        """Pick the action whose heuristic value is highest (one-step lookahead)."""
        best_action, best_h = 0, -float("inf")
        for action in range(self._env.n_actions):
            # Estimate: prefer reacting when condition is high, moving otherwise
            h = _heuristic(state)
            if action == 3 and state["env_condition"] > 0:
                h += 5
            elif action == 1 and state["resources"] < 5:
                h += 3
            elif action == 0 and state["env_condition"] == 0:
                h += 2
            if h > best_h:
                best_h, best_action = h, action
        return best_action

    def astar_action(self, state: dict, depth: int = 3) -> int:
        """Best-first expansion up to `depth` steps; returns first action of best path."""
        frontier: list[_Node] = []
        for action in range(self._env.n_actions):
            priority = -(_heuristic(state) + action * 0.1)
            heapq.heappush(frontier, _Node(priority, (state, action, action, 0), 0, 0))

        best_action, best_score = 0, -float("inf")
        visited = 0
        while frontier and visited < 50:
            node = heapq.heappop(frontier)
            visited += 1
            cur_state, first_action, _, cur_depth = node.state
            score = -node.priority
            if score > best_score:
                best_score, best_action = score, first_action
            if cur_depth < depth:
                for next_action in range(self._env.n_actions):
                    h = _heuristic(cur_state) + random.uniform(0, 0.5)
                    heapq.heappush(
                        frontier,
                        _Node(-(h + score), (cur_state, first_action, next_action, cur_depth + 1), 0, 0),
                    )
        return best_action

    def evaluate(self, episodes: int = 10) -> float:
        """Run Greedy for N episodes and return average total reward."""
        total = 0.0
        for _ in range(episodes):
            self._env.reset()
            ep_reward = 0.0
            for _ in range(200):
                state = self._env.get_state()
                action = self.greedy_action(state)
                _, reward, done = self._env.step(action)
                ep_reward += reward
                if done:
                    break
            total += ep_reward
        self._env.reset()
        return round(total / episodes, 2)
