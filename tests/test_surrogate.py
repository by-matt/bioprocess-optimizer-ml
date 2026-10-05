"""
Unit tests for Surrogate Yield Model.
"""

import numpy as np
import pytest
from src.models.surrogate import SurrogateYieldModel


def test_synthetic_data_generation() -> None:
    X, y = SurrogateYieldModel.generate_synthetic_data(n_samples=50, n_features=4, random_state=1)
    assert X.shape == (50, 4)
    assert y.shape == (50,)
    # Comprobar que las fracciones suman 1.0 en cada fila
    np.testing.assert_allclose(X.sum(axis=1), 1.0, rtol=1e-5)
    # Comprobar rango de rendimientos
    assert np.all(y >= 10.0)
    assert np.all(y <= 100.0)


def test_surrogate_fit_and_predict() -> None:
    model = SurrogateYieldModel(random_state=42)
    assert not model.is_fitted
    with pytest.raises(RuntimeError):
        model.predict(np.array([[0.25, 0.25, 0.25, 0.25]]))

    X, y = SurrogateYieldModel.generate_synthetic_data(n_samples=100, n_features=4, random_state=42)
    score = model.fit(X, y)
    assert model.is_fitted
    assert score > 0.70  # El modelo debe explicar más del 70% de la varianza

    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert np.all(preds > 0)
