"""
Formulation Optimizer Solver using SciPy SLSQP and Surrogate Model.
Author: Byron Calderón González
"""

import numpy as np
from scipy.optimize import minimize

from src.models.surrogate import SurrogateYieldModel
from src.optimization.schemas import OptimizationRequest, OptimizationResult


class FormulationOptimizer:
    """Motor de optimización no lineal para minimización de costo por tonelada."""

    def __init__(self, surrogate: SurrogateYieldModel) -> None:
        if not surrogate.is_fitted:
            raise ValueError("El modelo sustituto suministrado debe estar previamente entrenado.")
        self.surrogate = surrogate

    def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        """Ejecuta la optimización con restricciones duras."""
        costs = np.array([ing.cost_per_kg for ing in request.ingredients], dtype=np.float64)

        # Condición inicial factible: punto medio normalizado dentro de límites
        mins = np.array([ing.min_fraction for ing in request.ingredients], dtype=np.float64)
        maxs = np.array([ing.max_fraction for ing in request.ingredients], dtype=np.float64)
        mids = (mins + maxs) / 2.0
        initial_guess = mids / np.sum(mids)

        # Límites por variable
        bounds = [(ing.min_fraction, ing.max_fraction) for ing in request.ingredients]

        # Restricción 1 (Igualdad): La suma de las fracciones debe ser exactamente 1.0
        equality_constraint = {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
        }

        # Restricción 2 (Desigualdad): Rendimiento predicho >= meta mínima
        def yield_constraint(w: np.ndarray) -> float:
            pred = self.surrogate.predict(w.reshape(1, -1))[0]
            return float(pred - request.min_target_yield)

        inequality_constraint = {
            "type": "ineq",
            "fun": yield_constraint,
        }

        # Función objetivo: Costo unitario en USD/kg
        def cost_objective(w: np.ndarray) -> float:
            return float(np.dot(w, costs))

        result = minimize(
            fun=cost_objective,
            x0=initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=[equality_constraint, inequality_constraint],
            options={"ftol": 1e-4, "eps": 1e-3, "maxiter": 300, "disp": False},
        )

        success = bool(result.success)
        optimal_w = result.x if success else initial_guess
        # Normalizar para evitar desvíos microscópicos de punto flotante
        optimal_w = optimal_w / np.sum(optimal_w)

        cost_kg = float(np.dot(optimal_w, costs))
        cost_ton = cost_kg * 1000.0
        pred_yield = float(self.surrogate.predict(optimal_w.reshape(1, -1))[0])

        fractions_dict: dict[str, float] = {
            ing.name: round(float(w), 4)
            for ing, w in zip(request.ingredients, optimal_w, strict=False)
        }

        return OptimizationResult(
            success=success,
            status_message=str(result.message) if success else f"No convergió: {result.message}",
            optimal_fractions=fractions_dict,
            cost_usd_per_kg=round(cost_kg, 4),
            cost_usd_per_ton=round(cost_ton, 2),
            predicted_yield_pct=round(pred_yield, 2),
            iterations=int(result.nit),
        )
