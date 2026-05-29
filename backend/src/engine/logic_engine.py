"""Predicate-logic constraint engine backed by SWI-Prolog via pyswip.

Rules are declared in ``rules.pl`` using standard ISO Prolog syntax.
The Python layer loads that file **once** at import time and queries it
for every simulation step.

If SWI-Prolog / pyswip is unavailable the module falls back transparently
to equivalent pure-Python predicates so the rest of the system keeps
working without any code change.

Public interface
----------------
filter_actions(state, available) -> list[int]
    Return the subset of *available* action indices that are not
    forbidden by any predicate rule.

engine_backend() -> str
    Returns ``"prolog"`` when SWI-Prolog is active, ``"python"`` otherwise.
"""

from __future__ import annotations

from pathlib import Path

MIN_RESOURCES = 2
CRITICAL = 2
ALERT = 1

# Action index → Prolog atom  (must match the atoms in rules.pl)
_ACTION_ATOMS: dict[int, str] = {
    0: "mover",
    1: "asignar",
    2: "esperar",
    3: "reaccionar",
}

_RULES_PL = Path(__file__).parent / "rules.pl"

# ---------------------------------------------------------------------------
# Prolog backend — initialised once at module import
# ---------------------------------------------------------------------------
_prolog = None
_prolog_ok: bool = False

try:
    from pyswip import Prolog as _SWIProlog  # type: ignore[import]

    _prolog = _SWIProlog()
    _prolog.consult(str(_RULES_PL))
    _prolog_ok = True
except Exception:
    # SWI-Prolog not installed or pyswip not available → use Python fallback
    _prolog_ok = False


def _prolog_filter(state: dict, available: list[int]) -> list[int]:
    """Query rules.pl via SWI-Prolog to determine which actions are allowed."""
    pos = state["position"]
    rec = state["resources"]
    cond = state["env_condition"]
    allowed = []
    for action in available:
        atom = _ACTION_ATOMS[action]
        goal = f"accion_permitida({atom}, {pos}, {rec}, {cond})"
        if list(_prolog.query(goal)):  # type: ignore[union-attr]
            allowed.append(action)
    return allowed


# ---------------------------------------------------------------------------
# Pure-Python fallback — same 10 rules encoded as lambdas
# ---------------------------------------------------------------------------
def _r(s: dict) -> int:
    return s["resources"]


def _c(s: dict) -> int:
    return s["env_condition"]


def _p(s: dict) -> int:
    return s["position"]


_PY_RULES: dict[int, list] = {
    # mover (0)
    0: [
        lambda s: _r(s) < MIN_RESOURCES,            # R1
        lambda s: _c(s) == CRITICAL,                 # R2
    ],
    # asignar (1)
    1: [
        lambda s: _r(s) >= 10,                       # R3
        lambda s: _c(s) == CRITICAL and _r(s) < 1,  # R4
    ],
    # esperar (2)
    2: [
        lambda s: _c(s) == CRITICAL,                 # R5
        lambda s: _r(s) <= 1 and _c(s) >= ALERT,    # R6
    ],
    # reaccionar (3)
    3: [
        lambda s: _c(s) == 0,                        # R7
        lambda s: _r(s) < 1,                         # R8
        lambda s: _p(s) == 0 and _c(s) < ALERT,     # R9
        lambda s: _r(s) > 8 and _c(s) == 0,         # R10
    ],
}


def _python_filter(state: dict, available: list[int]) -> list[int]:
    allowed = []
    for action in available:
        forbidden = any(pred(state) for pred in _PY_RULES.get(action, []))
        if not forbidden:
            allowed.append(action)
    return allowed


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def filter_actions(state: dict, available: list[int]) -> list[int]:
    """Return the subset of *available* actions not forbidden by any rule.

    Dispatches to the SWI-Prolog engine (``rules.pl``) when pyswip is
    available; otherwise falls back to the equivalent pure-Python predicates.

    Always returns **at least one action** — defaults to ``[2]`` (esperar)
    if all actions are prohibited by the rules.
    """
    if _prolog_ok:
        allowed = _prolog_filter(state, available)
    else:
        allowed = _python_filter(state, available)
    return allowed if allowed else [2]


def engine_backend() -> str:
    """Return which backend is powering the constraint engine.

    Returns
    -------
    ``"prolog"``
        SWI-Prolog is active and ``rules.pl`` has been loaded.
    ``"python"``
        Fallback pure-Python predicates are being used.
    """
    return "prolog" if _prolog_ok else "python"


# ---------------------------------------------------------------------------
# Rule metadata — for frontend display
# ---------------------------------------------------------------------------
RULE_LABELS: list[dict] = [
    {"id": "R1",  "action": 0, "desc": "No redirigir · capacidad insuficiente"},
    {"id": "R2",  "action": 0, "desc": "No redirigir · fallo de red activo"},
    {"id": "R3",  "action": 1, "desc": "No activar · generadores al máximo"},
    {"id": "R4",  "action": 1, "desc": "No activar · fallo sin capacidad"},
    {"id": "R5",  "action": 2, "desc": "No monitorear · fallo de red activo"},
    {"id": "R6",  "action": 2, "desc": "No monitorear · sobrecarga sin capacidad"},
    {"id": "R7",  "action": 3, "desc": "No cortar · red estable"},
    {"id": "R8",  "action": 3, "desc": "No cortar · sin capacidad"},
    {"id": "R9",  "action": 3, "desc": "No cortar · sector industrial sin alerta"},
    {"id": "R10", "action": 3, "desc": "No cortar · alta capacidad en red estable"},
]

# Predicates in the same order as RULE_LABELS (used regardless of Prolog)
_RULE_PREDICATES = [
    lambda s: _r(s) < MIN_RESOURCES,             # R1
    lambda s: _c(s) == CRITICAL,                  # R2
    lambda s: _r(s) >= 10,                        # R3
    lambda s: _c(s) == CRITICAL and _r(s) < 1,   # R4
    lambda s: _c(s) == CRITICAL,                  # R5
    lambda s: _r(s) <= 1 and _c(s) >= ALERT,     # R6
    lambda s: _c(s) == 0,                         # R7
    lambda s: _r(s) < 1,                          # R8
    lambda s: _p(s) == 0 and _c(s) < ALERT,      # R9
    lambda s: _r(s) > 8 and _c(s) == 0,          # R10
]


def get_active_rules(state: dict) -> list[dict]:
    """Return all 10 rules annotated with whether they are blocking in *state*.

    Each item: ``{id, action, desc, active}``  where ``active=True`` means
    the rule is currently firing (blocking that action in this state).
    """
    result = []
    for meta, pred in zip(RULE_LABELS, _RULE_PREDICATES):
        result.append({**meta, "active": bool(pred(state))})
    return result


# ---------------------------------------------------------------------------
# Pre-computed lookup tables for all 165 reachable states
#
# The state space is tiny: 5 positions × 11 resource levels × 3 conditions
# = 165 states.  Both filter_actions and get_active_rules are pure functions
# of state, so we can precompute every result at module load time and serve
# them in O(1) via dict lookup instead of re-evaluating predicates on every
# WebSocket step.
#
# The LUT is built using the Python predicates.  The test
# ``test_prolog_rules_match_python_fallback`` verifies that Prolog and Python
# produce identical results for all states, so this is safe even when Prolog
# is the active backend.
# ---------------------------------------------------------------------------

_N_POSITIONS   = 5
_MAX_RESOURCES = 10
_N_CONDITIONS  = 3
_ALL_ACTIONS   = [0, 1, 2, 3]


def _build_lut() -> tuple[dict[int, list[int]], dict[int, list[dict]]]:
    """Return (filter_lut, rules_lut) keyed by state index."""
    filter_lut: dict[int, list[int]] = {}
    rules_lut:  dict[int, list[dict]] = {}
    for pos in range(_N_POSITIONS):
        for res in range(_MAX_RESOURCES + 1):
            for cond in range(_N_CONDITIONS):
                idx = (
                    pos * (_MAX_RESOURCES + 1) * _N_CONDITIONS
                    + res * _N_CONDITIONS
                    + cond
                )
                s = {
                    "position": pos,
                    "resources": res,
                    "env_condition": cond,
                    "index": idx,
                }
                allowed = _python_filter(s, _ALL_ACTIONS)
                filter_lut[idx] = allowed if allowed else [2]
                rules_lut[idx] = [
                    {**meta, "active": bool(pred(s))}
                    for meta, pred in zip(RULE_LABELS, _RULE_PREDICATES)
                ]
    return filter_lut, rules_lut


_FILTER_LUT, _RULES_LUT = _build_lut()


def filter_actions_fast(state: dict) -> list[int]:
    """O(1) lookup for the standard all-actions case.

    Equivalent to ``filter_actions(state, [0, 1, 2, 3])`` but uses the
    precomputed table instead of re-evaluating predicates or running Prolog
    queries.  Use this in the hot WebSocket step path.
    """
    return _FILTER_LUT[state["index"]]


def get_active_rules_fast(state: dict) -> list[dict]:
    """O(1) lookup returning the annotated rule list for *state*.

    Equivalent to ``get_active_rules(state)`` but served from the precomputed
    table.  Use this in the hot WebSocket step path.
    """
    return _RULES_LUT[state["index"]]
