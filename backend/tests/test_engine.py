from src.engine.environment import Environment
from src.engine.logic_engine import filter_actions
from src.engine.rl_agent import RLAgent


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
