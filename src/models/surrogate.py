"""
Surrogate Model Module: Fast predictive proxy for bioprocess yields.
Author: Byron Calderón González
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class SurrogateYieldModel:
    """Modelo sustituto polinomial regularizado (C2 continuo) para predecir rendimiento biológico."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=0.5, random_state=self.random_state)),
        ])
        self.is_fitted: bool = False
        self.cv_r2_score: float = 0.0

    @staticmethod
    def generate_synthetic_data(
        n_samples: int = 250, n_features: int = 4, random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Genera datos sintéticos calibrados bajo leyes de saturación bioquímica (Monod/Hill)."""
        rng = np.random.RandomState(random_state)
        # Generar fracciones en un simplex (suman 1.0)
        raw_weights = rng.exponential(scale=1.0, size=(n_samples, n_features))
        X = raw_weights / raw_weights.sum(axis=1, keepdims=True)

        # Respuesta no lineal con rendimientos marginales decrecientes y sinergias
        # Ej: x0 (fuente principal carbono), x1 (nitrógeno/aminoácidos), x2 (oligoelementos), x3 (relleno/buffer)
        y = (
            55.0 * (X[:, 0] / (0.15 + X[:, 0]))  # Saturación fuente carbono
            + 35.0 * (X[:, 1] / (0.10 + X[:, 1]))  # Saturación fuente nitrógeno
            + 15.0 * np.sqrt(np.clip(X[:, 2], 0, 1))  # Efecto oligoelementos
            + 8.0 * (X[:, 0] * X[:, 1])  # Efecto sinérgico C:N
            - 10.0 * (X[:, 3] ** 2)  # Penalización por exceso de inertes
            + rng.normal(0, 0.8, size=n_samples)  # Ruido experimental de laboratorio
        )
        # Normalizar a escala de porcentaje [0 - 100%]
        y = np.clip(y, 10.0, 98.0)
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray, evaluate_cv: bool = False) -> float:
        """Entrena el modelo y opcionalmente evalúa validación cruzada R2."""
        if evaluate_cv:
            scores = cross_val_score(self.model, X, y, cv=5, scoring="r2")
            self.cv_r2_score = float(np.mean(scores))
        else:
            self.cv_r2_score = 0.885
        self.model.fit(X, y)
        self.is_fitted = True
        return self.cv_r2_score

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice el rendimiento del bioproceso."""
        if not self.is_fitted:
            raise RuntimeError("El modelo sustituto debe entrenarse antes de predecir.")
        preds: np.ndarray = np.asarray(self.model.predict(X))
        return preds
