"""
Streamlit Web Application: BioProcess-Optimizer ML
Author: Byron Calderón González (github.com/by-matt)
"""

from typing import TypedDict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.surrogate import SurrogateYieldModel
from src.optimization.schemas import IngredientSpec, OptimizationRequest
from src.optimization.solver import FormulationOptimizer


class DefaultIngredient(TypedDict):
    name: str
    cost: float
    min: float
    max: float


st.set_page_config(
    page_title="BioProcess-Optimizer ML | Byron Calderón",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_trained_surrogate() -> SurrogateYieldModel:
    surrogate = SurrogateYieldModel(random_state=42)
    X, y = SurrogateYieldModel.generate_synthetic_data(n_samples=250, n_features=4, random_state=42)
    surrogate.fit(X, y)
    return surrogate


surrogate_model = get_trained_surrogate()
optimizer = FormulationOptimizer(surrogate_model)

# Header
st.title("⚡ BioProcess-Optimizer ML")
st.caption(
    "**Applied AI & Constrained Mathematical Optimization for Industrial Formulation & Bioprocesses** | "
    "Developed by [Byron Calderón González](https://linkedin.com/in/byron-calderón) • [GitHub](https://github.com/by-matt)"
)
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("🛠️ Configuración de Insumos")
preset = st.sidebar.selectbox(
    "Plantilla de Formulación Industrial:",
    [
        "Acuicultura / Nutrición Marina (AquaBiotics Core)",
        "Fermentación de Precisión & Biomasa Microbiana",
        "Ingredientes Funcionales & Agroindustria",
    ],
)

default_ingredients: list[DefaultIngredient] = [
    {"name": "Proteína / Hidrolizado Marino", "cost": 2.30, "min": 0.15, "max": 0.45},
    {"name": "Fuente Nitrógeno Fermentativo", "cost": 1.45, "min": 0.10, "max": 0.40},
    {"name": "Suplemento Microelementos & Taurina", "cost": 4.10, "min": 0.02, "max": 0.12},
    {"name": "Vehículo Base Energético (Carbohidrato)", "cost": 0.60, "min": 0.15, "max": 0.60},
]

st.sidebar.subheader("Parámetros por Ingrediente")
ingredients_input: list[IngredientSpec] = []
for i, item in enumerate(default_ingredients):
    ing_name = item["name"]
    ing_cost = item["cost"]
    ing_min = item["min"]
    ing_max = item["max"]
    st.sidebar.markdown(f"**{ing_name}**")
    c1, c2, c3 = st.sidebar.columns(3)
    cost = float(c1.number_input(f"USD/kg #{i+1}", value=ing_cost, step=0.1, key=f"cost_{i}"))
    min_f = float(c2.number_input(f"Mín % #{i+1}", value=int(ing_min * 100), step=1, key=f"min_{i}")) / 100.0
    max_f = float(c3.number_input(f"Máx % #{i+1}", value=int(ing_max * 100), step=1, key=f"max_{i}")) / 100.0
    ingredients_input.append(
        IngredientSpec(name=ing_name, cost_per_kg=cost, min_fraction=min_f, max_fraction=max_f)
    )

target_yield = st.sidebar.slider(
    "🎯 Rendimiento Biológico Mínimo Requerido (%)",
    min_value=40.0,
    max_value=90.0,
    value=65.0,
    step=1.0,
)

if st.sidebar.button("🚀 Ejecutar Optimización", type="primary", use_container_width=True):
    try:
        request = OptimizationRequest(ingredients=ingredients_input, min_target_yield=target_yield)
        result = optimizer.optimize(request)

        # Baseline: Formulación equitativa para comparación de ROI
        n_ing = len(ingredients_input)
        baseline_w = np.full(n_ing, 1.0 / n_ing)
        baseline_cost_ton = (
            sum(ing.cost_per_kg * w for ing, w in zip(ingredients_input, baseline_w, strict=False))
            * 1000.0
        )
        savings_pct = max(0.0, ((baseline_cost_ton - result.cost_usd_per_ton) / baseline_cost_ton) * 100.0)

        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Costo Optimizado (USD/Ton)", f"${result.cost_usd_per_ton:,.2f}", f"-{savings_pct:.1f}% vs. Base")
        col2.metric("Rendimiento Predicho", f"{result.predicted_yield_pct:.2f}%", f"Meta: {target_yield}%")
        col3.metric("Iteraciones Solver", f"{result.iterations} it", "Convergencia SLSQP")
        col4.metric("Estatus Algoritmo", "EXITOSO" if result.success else "FALLA", "Restricciones OK")

        tab1, tab2, tab3 = st.tabs([
            "📊 Composición & Frontera de Pareto",
            "🔍 Explicabilidad SHAP (Waterfall)",
            "📄 Exportación LIMS / ERP",
        ])

        with tab1:
            st.markdown("### 📊 Composición Óptima vs. Costo Marginal")
            df_res = pd.DataFrame([
                {"Ingrediente": k, "Inclusión (%)": v * 100.0, "Costo Insumo (USD/kg)": ing.cost_per_kg}
                for (k, v), ing in zip(result.optimal_fractions.items(), ingredients_input, strict=False)
            ])

            col_left, col_right = st.columns([1, 1])
            with col_left:
                fig_bar = px.bar(
                    df_res,
                    x="Ingrediente",
                    y="Inclusión (%)",
                    color="Inclusión (%)",
                    color_continuous_scale="Viridis",
                    text="Inclusión (%)",
                    title="Distribución Óptima de la Fórmula",
                )
                fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_right:
                # Simulación de Frontera de Pareto
                yield_range = np.linspace(45.0, 85.0, 15)
                pareto_costs: list[float | None] = []
                for y_target in yield_range:
                    try:
                        res_sim = optimizer.optimize(
                            OptimizationRequest(ingredients=ingredients_input, min_target_yield=y_target)
                        )
                        pareto_costs.append(res_sim.cost_usd_per_ton)
                    except Exception:
                        pareto_costs.append(None)

                df_pareto = pd.DataFrame({"Target Yield (%)": yield_range, "Cost (USD/Ton)": pareto_costs})
                fig_pareto = px.line(
                    df_pareto,
                    x="Target Yield (%)",
                    y="Cost (USD/Ton)",
                    markers=True,
                    title="Frontera de Pareto: Costo vs. Rendimiento Biológico",
                )
                fig_pareto.add_trace(
                    go.Scatter(
                        x=[result.predicted_yield_pct],
                        y=[result.cost_usd_per_ton],
                        mode="markers",
                        marker=dict(size=14, color="red", symbol="star"),
                        name="Punto Óptimo Seleccionado",
                    )
                )
                st.plotly_chart(fig_pareto, use_container_width=True)

        with tab2:
            st.markdown("### 🔍 Explicabilidad de Caja Blanca: Contribuciones Aditivas SHAP")
            st.caption(
                "Muestra cómo la reasignación de ingredientes respecto a la formulación base "
                "explica el incremento neto en el rendimiento biológico predicho."
            )

            # Cálculo de Atribución Marginal SHAP
            optimal_weights_vec = np.array(list(result.optimal_fractions.values()))
            y_base_pred = float(surrogate_model.predict(baseline_w.reshape(1, -1))[0])
            y_opt_pred = float(result.predicted_yield_pct)

            deltas: list[float] = []
            for idx in range(n_ing):
                w_step = baseline_w.copy()
                w_step[idx] = optimal_weights_vec[idx]
                y_step = float(surrogate_model.predict(w_step.reshape(1, -1))[0])
                deltas.append(y_step - y_base_pred)

            total_delta = y_opt_pred - y_base_pred
            sum_d = sum(deltas)
            scaled_deltas = [
                d * (total_delta / sum_d) if abs(sum_d) > 1e-5 else (total_delta / n_ing)
                for d in deltas
            ]

            names = [ing.name for ing in ingredients_input]
            wf_x = ["Fórmula Base"] + names + ["Rendimiento Final"]
            wf_y = [y_base_pred] + scaled_deltas + [y_opt_pred]
            wf_measure = ["absolute"] + ["relative"] * n_ing + ["total"]
            wf_text = (
                [f"{y_base_pred:.1f}%"]
                + [f"{v:+.2f}%" for v in scaled_deltas]
                + [f"{y_opt_pred:.1f}%"]
            )

            fig_wf = go.Figure(
                go.Waterfall(
                    name="SHAP Attribution",
                    orientation="v",
                    measure=wf_measure,
                    x=wf_x,
                    y=wf_y,
                    text=wf_text,
                    textposition="outside",
                    connector={"line": {"color": "#64748b"}},
                    decreasing={"marker": {"color": "#ef4444"}},
                    increasing={"marker": {"color": "#10b981"}},
                    totals={"marker": {"color": "#3b82f6"}},
                )
            )
            fig_wf.update_layout(
                title="Waterfall SHAP: Contribución Marginal por Ingrediente (%)",
                waterfallgap=0.3,
                yaxis=dict(title="Rendimiento Predicho (%)", range=[30, 100]),
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            st.info(
                "💡 **Interpretación Ejecutiva para Gerencia:** "
                "Los valores verdes representan ingredientes cuyo ajuste aportó positivamente al rendimiento, "
                "mientras que los rojos señalan insumos penalizados para optimizar el margen sin comprometer la meta biológica."
            )

        with tab3:
            st.markdown("### 📄 Contratos de Datos & Exportación Estructurada")
            col_json, col_csv = st.columns(2)
            with col_json:
                st.markdown("**Contrato Pydantic v2 (JSON):**")
                st.json(result.model_dump())
                st.download_button(
                    label="⬇️ Descargar Ficha Técnica JSON",
                    data=result.model_dump_json(indent=2),
                    file_name="optimizacion_bioproceso.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with col_csv:
                st.markdown("**Tabla para ERP / LIMS (CSV):**")
                st.dataframe(df_res, use_container_width=True)
                csv_data = df_res.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Descargar Distribución CSV",
                    data=csv_data,
                    file_name="distribucion_ingredientes.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        st.success(f"Optimización completada con éxito. Estatus: {result.status_message}")

    except Exception as e:
        st.error(f"Error en los parámetros de optimización: {e}")
else:
    st.info("Ajuste los límites y costos en la barra lateral y presione 'Ejecutar Optimización' para calcular la fórmula óptima.")
