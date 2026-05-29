"""ML model that predicts environment condition transitions.

Trains incrementally on episode history stored by Analytics.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class EnvPredictor:
    def __init__(self):
        self._model = RandomForestClassifier(n_estimators=50, random_state=42)
        self._trained = False
        self._le = LabelEncoder().fit([0, 1, 2])

    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray):
        """X: [[position, resources, env_condition, action]], y: next env_condition."""
        if len(X) < 20:
            return
        self._model.fit(X, y)
        self._trained = True

    def predict_proba(self, state: dict, action: int) -> dict[str, float]:
        """Return probability distribution over next env_condition values."""
        if not self._trained:
            return {0: 0.6, 1: 0.3, 2: 0.1}
        x = np.array([[state["position"], state["resources"], state["env_condition"], action]])
        probs = self._model.predict_proba(x)[0]
        classes = self._model.classes_
        return {int(c): round(float(p), 3) for c, p in zip(classes, probs)}

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save trained model to disk (no-op if not yet trained)."""
        if not self._trained:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "le": self._le, "trained": self._trained},
            str(path),
        )

    def load(self, path: Path) -> bool:
        """Load model from disk. Returns True on success."""
        try:
            if not path.exists():
                return False
            data = joblib.load(str(path))
            if not isinstance(data, dict) or "model" not in data:
                return False
            self._model = data["model"]
            self._le = data["le"]
            self._trained = data.get("trained", True)
            return True
        except Exception:
            return False
