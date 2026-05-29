import numpy as np

from src.engine.dqn_agent import DQNAgent
from src.engine.environment import Environment
from src.engine.logic_engine import engine_backend, filter_actions
from src.engine.ml_model import EnvPredictor
from src.engine.rl_agent import RLAgent
from src.engine.search import ClassicSearch


def test_environment_reset():
    env = Environment()
    state = env.reset()
    assert 0 <= state.position < 5
    assert 0 <= state.resources <= 10


def test_environment_step_returns_reward():
    env = Environment()
    env.reset()
    state = env.get_state()
    next_state, reward, done = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_rl_agent_learns():
    env = Environment()
    agent = RLAgent(n_states=env.n_states, n_actions=env.n_actions)
    state = env.get_state()
    action = agent.select_action(state)
    next_state, reward, _ = env.step(action)
    agent.learn(state, action, reward, next_state)
    assert agent.epsilon < 1.0


def test_logic_engine_filters_critical():
    critical_state = {"position": 2, "resources": 5, "env_condition": 2, "index": 0}
    allowed = filter_actions(critical_state, list(range(4)))
    assert 0 not in allowed  # R2: no moverse en crítico
    assert 2 not in allowed  # R5: no esperar en crítico


def test_logic_engine_minimum_resources():
    low_resource_state = {"position": 1, "resources": 1, "env_condition": 0, "index": 0}
    allowed = filter_actions(low_resource_state, list(range(4)))
    assert 0 not in allowed  # R1: no moverse sin recursos mínimos


def test_logic_engine_always_returns_action():
    impossible_state = {"position": 0, "resources": 0, "env_condition": 2, "index": 0}
    allowed = filter_actions(impossible_state, list(range(4)))
    assert len(allowed) >= 1  # siempre debe haber al menos una acción


def test_rl_agent_respects_allowed_actions():
    """Agent must only pick actions from the allowed subset."""
    env = Environment()
    agent = RLAgent(n_states=env.n_states, n_actions=env.n_actions)
    state = env.get_state()
    allowed = [1, 3]  # restrict to allocate_resources and react
    for _ in range(30):
        action = agent.select_action(state, allowed_actions=allowed)
        assert action in allowed


def test_logic_engine_integrates_with_agent():
    """The full pipeline: filter → agent select → learn should not crash."""
    env = Environment()
    agent = RLAgent(n_states=env.n_states, n_actions=env.n_actions)
    for _ in range(20):
        state = env.get_state()
        allowed = filter_actions(state, list(range(env.n_actions)))
        action = agent.select_action(state, allowed_actions=allowed)
        assert action in allowed
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        if done:
            env.reset()


def test_env_predictor_trains_and_predicts():
    """EnvPredictor must train on sufficient data and return valid probs."""
    predictor = EnvPredictor()
    assert not predictor.is_trained
    # Minimal synthetic training data (30 rows)
    rng = np.random.default_rng(0)
    X = rng.integers(0, [5, 11, 3, 4], size=(30, 4))
    y = rng.integers(0, 3, size=30)
    predictor.train(X, y)
    assert predictor.is_trained
    state = {"position": 2, "resources": 5, "env_condition": 1, "index": 0}
    probs = predictor.predict_proba(state, action=1)
    assert set(probs.keys()).issubset({0, 1, 2})
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_classic_search_evaluate_does_not_modify_env():
    """evaluate() must use fresh envs and leave the live env untouched."""
    env = Environment()
    env.reset()
    original_state = env.get_state()
    search = ClassicSearch(env)
    results = search.evaluate(episodes=3)
    after_state = env.get_state()
    # Live env position and resources should be unchanged
    assert after_state["position"] == original_state["position"]
    assert after_state["resources"] == original_state["resources"]
    assert "greedy" in results and "astar" in results


# ---------------------------------------------------------------------------
# DQN tests
# ---------------------------------------------------------------------------

def test_dqn_selects_action_within_allowed():
    """DQN must only choose actions from the allowed subset."""
    agent = DQNAgent(n_actions=4)
    state = {"position": 1, "resources": 5, "env_condition": 0, "index": 0}
    allowed = [1, 3]
    for _ in range(30):
        action = agent.select_action(state, allowed_actions=allowed)
        assert action in allowed, f"Got {action}, expected one of {allowed}"


def test_dqn_trains_after_buffer_warm_up():
    """DQN loss must become non-zero once the replay buffer has enough samples."""
    agent = DQNAgent(n_actions=4, batch_size=16, replay_capacity=500)
    env = Environment()
    state = env.get_state()
    # Feed batch_size + 1 transitions to guarantee at least one training step
    for _ in range(agent.batch_size + 1):
        allowed = filter_actions(state, list(range(env.n_actions)))
        action = agent.select_action(state, allowed)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        if done:
            env.reset()
        state = env.get_state()
    assert agent.last_loss > 0.0, "Loss should be positive after training"
    assert agent.buffer_size >= agent.batch_size


def test_dqn_epsilon_decays():
    """Epsilon must strictly decrease after each learn() call."""
    agent = DQNAgent(n_actions=4, epsilon_decay=0.9)
    env = Environment()
    initial_eps = agent.epsilon
    state = env.get_state()
    for _ in range(5):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
    assert agent.epsilon < initial_eps


def test_dqn_reset_restores_initial_state():
    """reset() must bring epsilon and buffer back to initial conditions."""
    agent = DQNAgent(n_actions=4, batch_size=4, replay_capacity=50)
    env = Environment()
    state = env.get_state()
    for _ in range(20):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
    assert agent.epsilon < 1.0
    agent.reset()
    assert agent.epsilon == 1.0
    assert agent.buffer_size == 0
    assert agent.last_loss == 0.0


# ---------------------------------------------------------------------------
# Prolog backend tests
# ---------------------------------------------------------------------------

def test_prolog_backend_is_active():
    """SWI-Prolog must be the active logic engine (not the Python fallback)."""
    assert engine_backend() == "prolog", (
        "Expected SWI-Prolog via pyswip. "
        "Install SWI-Prolog and ensure pyswip is in the dependencies."
    )


def test_prolog_rules_match_python_fallback():
    """Prolog rules must produce the same allowed-action sets as the Python predicates.

    Runs 200 random states and compares both backends. Any divergence means
    rules.pl is out of sync with the Python fallback in logic_engine.py.
    """
    import random
    from src.engine.logic_engine import _prolog_filter, _python_filter  # type: ignore[attr-defined]

    rng = random.Random(42)
    mismatches = []
    for _ in range(200):
        state = {
            "position": rng.randint(0, 4),
            "resources": rng.randint(0, 10),
            "env_condition": rng.randint(0, 2),
            "index": 0,
        }
        available = list(range(4))
        prolog_result = sorted(_prolog_filter(state, available))
        python_result = sorted(_python_filter(state, available))
        if prolog_result != python_result:
            mismatches.append((state, prolog_result, python_result))

    assert not mismatches, (
        f"{len(mismatches)} state(s) where Prolog and Python disagree:\n"
        + "\n".join(str(m) for m in mismatches[:5])
    )
