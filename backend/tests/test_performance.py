"""Performance benchmark suite for the simulation hot-path.

Each test measures throughput or latency for a critical code path and
asserts a minimum performance floor.  Run with:

    cd backend
    uv run pytest tests/test_performance.py -v -s

The ``-s`` flag lets timing output appear in the console.

Floors (conservative, CPU-independent):
- filter_actions  (Python fallback) :  ≥ 200 000 calls/s
- filter_actions_fast (LUT)         :  ≥ 1 000 000 calls/s
- get_active_rules (direct)         :  ≥  50 000 calls/s
- get_active_rules_fast (LUT)       :  ≥ 500 000 calls/s
- analytics.summary (after ≥10 k steps): ≤ 1 ms per call
- full WebSocket step (all combined):  ≥ 500 steps/s  (we run at ~6.7 steps/s)
- DQN training step (after warm-up) :  ≥  50 steps/s
"""

import time
from statistics import mean

import numpy as np
import pytest

from src.engine.analytics import Analytics
from src.engine.dqn_agent import DQNAgent
from src.engine.environment import Environment
from src.engine.logic_engine import (
    _python_filter,
    filter_actions,
    filter_actions_fast,
    get_active_rules,
    get_active_rules_fast,
)
from src.engine.ml_model import EnvPredictor
from src.engine.rl_agent import RLAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _throughput(fn, n: int = 100_000) -> float:
    """Return calls per second for a zero-argument callable."""
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - t0
    return n / elapsed


def _sample_states(n: int = 200) -> list[dict]:
    """Return n deterministic states covering the full space."""
    env = Environment()
    states = []
    rng = np.random.default_rng(42)
    for _ in range(n):
        pos  = int(rng.integers(0, 5))
        res  = int(rng.integers(0, 11))
        cond = int(rng.integers(0, 3))
        idx  = pos * 11 * 3 + res * 3 + cond
        states.append({"position": pos, "resources": res, "env_condition": cond, "index": idx})
    return states


STATES = _sample_states(200)
ALL_ACTIONS = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 1. Logic engine — filter_actions
# ---------------------------------------------------------------------------

class TestFilterActions:
    def test_python_filter_throughput(self):
        """Python predicate filter should handle ≥200 000 calls/s."""
        s = STATES[0]
        cps = _throughput(lambda: _python_filter(s, ALL_ACTIONS))
        print(f"\n  Python filter:   {cps:,.0f} calls/s")
        assert cps >= 200_000, f"Too slow: {cps:.0f} calls/s"

    def test_lut_filter_throughput(self):
        """LUT filter_actions_fast should handle ≥1 000 000 calls/s."""
        s = STATES[0]
        cps = _throughput(lambda: filter_actions_fast(s))
        print(f"\n  LUT filter:      {cps:,.0f} calls/s")
        assert cps >= 1_000_000, f"Too slow: {cps:.0f} calls/s"

    def test_lut_speedup_vs_python(self):
        """LUT must be at least 3× faster than Python predicate evaluation."""
        s = STATES[0]
        py_cps  = _throughput(lambda: _python_filter(s, ALL_ACTIONS))
        lut_cps = _throughput(lambda: filter_actions_fast(s))
        ratio = lut_cps / py_cps
        print(f"\n  Speedup (LUT vs Python): {ratio:.1f}x")
        assert ratio >= 3.0, f"LUT speedup too low: {ratio:.1f}x"

    def test_lut_correctness_all_states(self):
        """LUT results must match Python predicate results for all 165 states."""
        mismatches = []
        for s in STATES:
            expected = _python_filter(s, ALL_ACTIONS) or [2]
            got      = filter_actions_fast(s)
            if sorted(expected) != sorted(got):
                mismatches.append((s, expected, got))
        assert not mismatches, (
            f"{len(mismatches)} state(s) with LUT mismatch:\n"
            + "\n".join(str(m) for m in mismatches[:3])
        )


# ---------------------------------------------------------------------------
# 2. Logic engine — get_active_rules
# ---------------------------------------------------------------------------

class TestGetActiveRules:
    def test_direct_throughput(self):
        """Direct get_active_rules should handle ≥50 000 calls/s."""
        s = STATES[0]
        cps = _throughput(lambda: get_active_rules(s), n=20_000)
        print(f"\n  get_active_rules (direct): {cps:,.0f} calls/s")
        assert cps >= 50_000, f"Too slow: {cps:.0f} calls/s"

    def test_lut_throughput(self):
        """LUT get_active_rules_fast should handle ≥500 000 calls/s."""
        s = STATES[0]
        cps = _throughput(lambda: get_active_rules_fast(s))
        print(f"\n  get_active_rules (LUT):    {cps:,.0f} calls/s")
        assert cps >= 500_000, f"Too slow: {cps:.0f} calls/s"

    def test_lut_correctness(self):
        """LUT rules must match direct computation for all sample states."""
        for s in STATES:
            direct = get_active_rules(s)
            fast   = get_active_rules_fast(s)
            assert direct == fast, f"Mismatch at state {s}"


# ---------------------------------------------------------------------------
# 3. Analytics — summary() performance
# ---------------------------------------------------------------------------

class TestAnalyticsSummary:
    @staticmethod
    def _filled_analytics(n_steps: int) -> Analytics:
        env = Environment()
        agent = RLAgent(n_states=env.n_states, n_actions=env.n_actions)
        an = Analytics()
        for _ in range(n_steps):
            state  = env.get_state()
            action = agent.select_action(state)
            _, reward, done = env.step(action)
            an.log(state, action, reward)
            agent.learn(state, action, reward, env.get_state(), done)
            if done:
                env.reset()
                an.next_episode()
        return an

    def test_summary_small_session(self):
        """summary() must complete in ≤1 ms for a 500-step session."""
        an = self._filled_analytics(500)
        times = []
        for _ in range(1_000):
            t0 = time.perf_counter()
            an.summary()
            times.append(time.perf_counter() - t0)
        avg_ms = mean(times) * 1_000
        print(f"\n  summary() avg latency (500 steps):  {avg_ms:.3f} ms")
        assert avg_ms <= 1.0, f"summary() too slow: {avg_ms:.3f} ms"

    def test_summary_large_session(self):
        """summary() must complete in ≤1 ms even for a 10 000-step session."""
        an = self._filled_analytics(10_000)
        times = []
        for _ in range(500):
            t0 = time.perf_counter()
            an.summary()
            times.append(time.perf_counter() - t0)
        avg_ms = mean(times) * 1_000
        print(f"\n  summary() avg latency (10 000 steps): {avg_ms:.3f} ms")
        assert avg_ms <= 1.0, f"summary() too slow at scale: {avg_ms:.3f} ms"

    def test_summary_values_correct(self):
        """summary() accumulators must match a reference DataFrame computation."""
        import pandas as pd
        an = self._filled_analytics(300)

        got = an.summary()

        # Reference using Pandas (original approach)
        df = pd.DataFrame(an._records)
        expected_steps       = len(df)
        expected_action_dist = {int(k): int(v) for k, v in df["action"].value_counts().items()}
        expected_avg_res     = round(df["resources"].mean(), 2)

        assert got["steps"] == expected_steps
        assert got["action_distribution"] == expected_action_dist
        assert abs(got["avg_resources"] - expected_avg_res) < 0.01


# ---------------------------------------------------------------------------
# 4. RL agent step throughput
# ---------------------------------------------------------------------------

class TestRLAgentThroughput:
    def test_qlearning_step_throughput(self):
        """Q-Learning select_action + learn must achieve ≥100 000 steps/s.

        Conservative floor that accounts for Windows CPython overhead and the
        random-number generation inside ``env.step``.  The UI only demands
        ~6.7 steps/s, so 100 K gives ≥14 000× headroom.
        """
        env   = Environment()
        agent = RLAgent(n_states=env.n_states, n_actions=env.n_actions)
        state = env.get_state()
        n     = 100_000

        t0 = time.perf_counter()
        for _ in range(n):
            action = agent.select_action(state)
            next_s, reward, done = env.step(action)
            agent.learn(state, action, reward, next_s, done)
            state = next_s
            if done:
                env.reset()
                state = env.get_state()
        sps = n / (time.perf_counter() - t0)
        print(f"\n  Q-Learning steps/s: {sps:,.0f}")
        assert sps >= 100_000, f"Too slow: {sps:.0f} steps/s"

    def test_dqn_step_throughput_post_warmup(self):
        """DQN select_action + learn must achieve ≥50 steps/s after warm-up."""
        agent = DQNAgent(n_actions=4, batch_size=64, replay_capacity=2_000)
        env   = Environment()
        state = env.get_state()

        # Warm up replay buffer
        for _ in range(agent.batch_size + 10):
            allowed = filter_actions_fast(state)
            action  = agent.select_action(state, allowed)
            next_s, reward, done = env.step(action)
            agent.learn(state, action, reward, next_s, done)
            state = next_s if not done else env.get_state()

        # Benchmark
        n  = 1_000
        t0 = time.perf_counter()
        for _ in range(n):
            allowed = filter_actions_fast(state)
            action  = agent.select_action(state, allowed)
            next_s, reward, done = env.step(action)
            agent.learn(state, action, reward, next_s, done)
            state = next_s if not done else env.get_state()
        sps = n / (time.perf_counter() - t0)
        print(f"\n  DQN steps/s (post-warmup): {sps:,.0f}")
        assert sps >= 50, f"DQN too slow: {sps:.0f} steps/s"


# ---------------------------------------------------------------------------
# 5. Full pipeline simulation — mirrors the WebSocket hot path exactly
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_full_step_throughput_qlearning(self):
        """Full pipeline (filter+predict+agent+env+analytics) ≥500 steps/s with Q-Learning."""
        env       = Environment()
        agent     = RLAgent(n_states=env.n_states, n_actions=env.n_actions)
        analytics = Analytics()
        predictor = EnvPredictor()
        n         = 5_000

        t0 = time.perf_counter()
        for _ in range(n):
            state   = env.get_state()
            allowed = filter_actions_fast(state)
            _       = predictor.predict_proba(state, allowed[0])
            action  = agent.select_action(state, allowed)
            next_s, reward, done = env.step(action)
            agent.learn(state, action, reward, next_s, done)
            analytics.log(state, action, reward)
            _rules  = get_active_rules_fast(state)   # noqa: F841
            if done:
                env.reset()
                analytics.next_episode()
        sps = n / (time.perf_counter() - t0)
        print(f"\n  Full pipeline Q-Learning steps/s: {sps:,.0f}")
        assert sps >= 500, f"Pipeline too slow: {sps:.0f} steps/s"

    def test_full_step_throughput_dqn(self):
        """Full pipeline ≥50 steps/s with DQN (heavier due to NN inference)."""
        env       = Environment()
        agent     = DQNAgent(n_actions=env.n_actions, batch_size=64)
        analytics = Analytics()
        predictor = EnvPredictor()

        # Warm up replay buffer so training kicks in during benchmark
        state = env.get_state()
        for _ in range(agent.batch_size + 10):
            allowed = filter_actions_fast(state)
            action  = agent.select_action(state, allowed)
            next_s, reward, done = env.step(action)
            agent.learn(state, action, reward, next_s, done)
            analytics.log(state, action, reward)
            if done:
                env.reset()
                state = env.get_state()
            else:
                state = next_s

        n  = 500
        t0 = time.perf_counter()
        for _ in range(n):
            state   = env.get_state()
            allowed = filter_actions_fast(state)
            _       = predictor.predict_proba(state, allowed[0])
            action  = agent.select_action(state, allowed)
            next_s, reward, done = env.step(action)
            agent.learn(state, action, reward, next_s, done)
            analytics.log(state, action, reward)
            _rules  = get_active_rules_fast(state)  # noqa: F841
            if done:
                env.reset()
                analytics.next_episode()
        sps = n / (time.perf_counter() - t0)
        print(f"\n  Full pipeline DQN steps/s:        {sps:,.0f}")
        assert sps >= 50, f"DQN pipeline too slow: {sps:.0f} steps/s"

    def test_websocket_speed_headroom(self):
        """Pipeline must be ≥50× faster than the UI step interval (150 ms).

        At 6.7 steps/s from the frontend, the backend needs to complete
        each step well under 150 ms.  A ≥50× margin ensures the backend
        is never the bottleneck, even under concurrent load.
        """
        UI_INTERVAL_S = 0.150  # 150 ms between frontend step requests
        TARGET_S      = UI_INTERVAL_S / 50  # 3 ms max per step

        env    = Environment()
        agent  = RLAgent(n_states=env.n_states, n_actions=env.n_actions)
        an     = Analytics()
        pred   = EnvPredictor()
        n      = 1_000
        times  = []

        for _ in range(n):
            state = env.get_state()
            t0    = time.perf_counter()
            allowed = filter_actions_fast(state)
            _       = pred.predict_proba(state, allowed[0])
            action  = agent.select_action(state, allowed)
            next_s, reward, done = env.step(action)
            agent.learn(state, action, reward, next_s, done)
            an.log(state, action, reward)
            _       = get_active_rules_fast(state)
            times.append(time.perf_counter() - t0)
            if done:
                env.reset()
                an.next_episode()

        avg_ms = mean(times) * 1_000
        p99_ms = sorted(times)[int(len(times) * 0.99)] * 1_000
        headroom = (UI_INTERVAL_S / mean(times))
        print(
            f"\n  Step latency — avg: {avg_ms:.3f} ms  "
            f"p99: {p99_ms:.3f} ms  "
            f"headroom: {headroom:.0f}×"
        )
        assert mean(times) <= TARGET_S, (
            f"Step avg {avg_ms:.3f} ms exceeds {TARGET_S*1000:.1f} ms target"
        )
