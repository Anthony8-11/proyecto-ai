"""Predicate-logic constraint engine.

Each rule is a callable (State dict → bool) that returns True when the
action associated with it is FORBIDDEN.  The engine filters the action
space before the agent selects an action.
"""

MIN_RESOURCES = 2
CRITICAL = 2
ALERT = 1


def _r(state: dict) -> int:
    return state["resources"]


def _c(state: dict) -> int:
    return state["env_condition"]


def _p(state: dict) -> int:
    return state["position"]


# action index → list[forbidden_predicate]
RULES: dict[int, list] = {
    # move (0)
    0: [
        lambda s: _r(s) < MIN_RESOURCES,           # R1: no moverse sin recursos
        lambda s: _c(s) == CRITICAL,               # R2: no moverse en estado crítico
    ],
    # allocate_resources (1)
    1: [
        lambda s: _r(s) >= 10,                     # R3: no asignar si ya al máximo
        lambda s: _c(s) == CRITICAL and _r(s) < 1, # R4: bloqueo total en crítico sin recursos
    ],
    # wait (2)
    2: [
        lambda s: _c(s) == CRITICAL,               # R5: esperar en crítico es inválido
        lambda s: _r(s) <= 1 and _c(s) >= ALERT,  # R6: no esperar con recursos casi agotados en alerta
    ],
    # react (3)
    3: [
        lambda s: _c(s) == 0,                      # R7: reaccionar sin amenaza es inútil
        lambda s: _r(s) < 1,                       # R8: no se puede reaccionar sin recursos
        lambda s: _p(s) == 0 and _c(s) < ALERT,   # R9: posición 0 solo reacciona bajo alerta+
        lambda s: _r(s) > 8 and _c(s) == 0,       # R10: recursos abundantes + calma = no reaccionar
    ],
}


def filter_actions(state: dict, available: list[int]) -> list[int]:
    """Return the subset of available actions not forbidden by any rule."""
    allowed = []
    for action in available:
        forbidden = any(pred(state) for pred in RULES.get(action, []))
        if not forbidden:
            allowed.append(action)
    # Guarantee at least one action (fallback: wait or react)
    return allowed if allowed else [2]
