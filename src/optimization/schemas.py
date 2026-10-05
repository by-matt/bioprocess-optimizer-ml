"""
Pydantic v2 Schemas and Data Contracts for Bioprocess Optimization.
"""

from typing import Dict, List
from pydantic import BaseModel, Field, model_validator


class IngredientSpec(BaseModel):
    name: str = Field(..., description="Nombre comercial o químico del ingrediente")
    cost_per_kg: float = Field(..., gt=0.0, description="Costo unitario en USD por kilogramo")
    min_fraction: float = Field(
        0.0, ge=0.0, le=1.0, description="Fracción mínima de inclusión en la formulación (0.0 a 1.0)"
    )
    max_fraction: float = Field(
        1.0, ge=0.0, le=1.0, description="Fracción máxima de inclusión en la formulación (0.0 a 1.0)"
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "IngredientSpec":
        if self.min_fraction > self.max_fraction:
            raise ValueError(
                f"min_fraction ({self.min_fraction}) no puede ser mayor que max_fraction ({self.max_fraction})"
            )
        return self


class OptimizationRequest(BaseModel):
    ingredients: List[IngredientSpec] = Field(
        ..., min_length=2, description="Lista de ingredientes a formular (mínimo 2)"
    )
    min_target_yield: float = Field(
        ..., ge=0.0, le=100.0, description="Rendimiento biológico mínimo objetivo (%)"
    )

    @model_validator(mode="after")
    def validate_sum_bounds(self) -> "OptimizationRequest":
        min_sum = sum(ing.min_fraction for ing in self.ingredients)
        max_sum = sum(ing.max_fraction for ing in self.ingredients)
        if min_sum > 1.0:
            raise ValueError(
                f"La suma de fracciones mínimas ({min_sum:.2f}) excede 1.0 (imposible formular)."
            )
        if max_sum < 1.0:
            raise ValueError(
                f"La suma de fracciones máximas ({max_sum:.2f}) es menor que 1.0 (imposible formular)."
            )
        return self


class OptimizationResult(BaseModel):
    success: bool
    status_message: str
    optimal_fractions: Dict[str, float] = Field(
        ..., description="Fracciones óptimas asignadas a cada ingrediente (suman 1.0)"
    )
    cost_usd_per_kg: float
    cost_usd_per_ton: float
    predicted_yield_pct: float
    iterations: int
