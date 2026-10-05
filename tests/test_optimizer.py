"""
Unit tests for Formulation Optimizer and Pydantic Contracts.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from src.models.surrogate import SurrogateYieldModel
from src.optimization.schemas import IngredientSpec, OptimizationRequest
from src.optimization.solver import FormulationOptimizer


def create_trained_surrogate() -> SurrogateYieldModel:
    surrogate = SurrogateYieldModel(random_state=42)
    X, y = SurrogateYieldModel.generate_synthetic_data(n_samples=150, n_features=4, random_state=42)
    surrogate.fit(X, y)
    return surrogate


@pytest.fixture
def trained_surrogate() -> SurrogateYieldModel:
    return create_trained_surrogate()


def test_ingredient_spec_validation() -> None:
    # Caso válido
    ing = IngredientSpec(name="Harina de Pescado", cost_per_kg=1.85, min_fraction=0.1, max_fraction=0.4)
    assert ing.name == "Harina de Pescado"

    # Caso inválido: min > max
    with pytest.raises(ValidationError):
        IngredientSpec(name="Invalido", cost_per_kg=1.0, min_fraction=0.6, max_fraction=0.2)


def test_optimization_request_validation() -> None:
    # Caso inválido: suma de mínimos excede 1.0
    with pytest.raises(ValidationError):
        OptimizationRequest(
            ingredients=[
                IngredientSpec(name="A", cost_per_kg=1.0, min_fraction=0.6, max_fraction=0.8),
                IngredientSpec(name="B", cost_per_kg=2.0, min_fraction=0.5, max_fraction=0.7),
            ],
            min_target_yield=50.0,
        )


def test_solver_success_and_constraints(trained_surrogate: SurrogateYieldModel) -> None:
    optimizer = FormulationOptimizer(trained_surrogate)

    ingredients = [
        IngredientSpec(name="Proteína Concentrada", cost_per_kg=2.20, min_fraction=0.15, max_fraction=0.50),
        IngredientSpec(name="Fuente Nitrógeno Orgánico", cost_per_kg=1.40, min_fraction=0.10, max_fraction=0.40),
        IngredientSpec(name="Premezcla Mineral", cost_per_kg=3.50, min_fraction=0.02, max_fraction=0.15),
        IngredientSpec(name="Carbohidrato Energético", cost_per_kg=0.65, min_fraction=0.10, max_fraction=0.60),
    ]

    target_yield = 60.0
    request = OptimizationRequest(ingredients=ingredients, min_target_yield=target_yield)
    result = optimizer.optimize(request)

    assert result.success is True
    # 1. Comprobar que la suma de fracciones es exactamente 1.0
    total_fraction = sum(result.optimal_fractions.values())
    assert pytest.approx(total_fraction, abs=1e-3) == 1.0

    # 2. Comprobar que cada fracción respeta sus límites
    for ing in ingredients:
        assigned = result.optimal_fractions[ing.name]
        assert assigned >= ing.min_fraction - 1e-4
        assert assigned <= ing.max_fraction + 1e-4

    # 3. Comprobar que el rendimiento predicho cumple o supera la meta
    assert result.predicted_yield_pct >= target_yield - 0.5
    assert result.cost_usd_per_ton > 0.0
