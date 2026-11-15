"""Bloque del agente supervisado (modo automático) para las páginas de clasificación y regresión."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.components.shared import ejecutar, obtener_controlador, pie_de_ejecucion
from frontend.components.theme import AMARILLO, CIAN, ESCALA_CONTINUA, MAGENTA, SECUENCIA, grafico

TITULOS = {"classification": "clasificación", "regression": "regresión"}
CLAVES = {"classification": "agente_clasificacion", "regression": "agente_regresion"}
AUTOMATICO = "(que decida el agente)"


def seccion_agente(tipo: str, df: pd.DataFrame) -> None:
    """Controles + informe del agente; la ejecución pasa por el coordinador (tarea ``<tipo>.auto``)."""
    clave = CLAVES[tipo]
    ctrl = obtener_controlador()
    with st.container(border=True):
        st.markdown(f"**Agente inteligente de {TITULOS[tipo]}**")
        st.caption(
            "Opcional. El LLM decide qué variable predecir, qué columnas excluir y qué modelos "
            "validar; Python entrena con las clases del framework en varias particiones y mide. "
            "Gemini solo recibe perfiles y métricas agregadas, nunca las filas de los datos."
        )
        a0, a1, a2, a3 = st.columns([1.3, 1, 1, 1])
        opciones = [AUTOMATICO, *map(str, df.columns)]
        eda = ctrl.eda_result
        sugerido = str(eda["result"]["target"] if eda is not None else st.session_state.get("eda_objetivo"))
        eleccion = a0.selectbox("Variable objetivo", opciones,
                                index=opciones.index(sugerido) if sugerido in opciones else 0,
                                key=f"{clave}_objetivo", help="Por defecto se usa la elegida en el EDA, si existe.")
        objetivo = None if eleccion == AUTOMATICO else {str(c): c for c in df.columns}[eleccion]
        repeticiones = a1.slider("Particiones de validación", 2, 10, 5, key=f"{clave}_rep")
        prueba = a2.slider("Prueba (%)", 10, 50, 25, 5, key=f"{clave}_test")
        a3.write("")
        if a3.button("Ejecutar agente", type="primary", width="stretch", key=f"{clave}_btn"):
            resultado = ejecutar(
                lambda: ctrl.run_auto(tipo, target=objetivo, repetitions=repeticiones, test_size=prueba / 100),
                "El agente está eligiendo el objetivo y validando modelos...", "No se pudo ejecutar el agente: ")
            if resultado is not None:
                st.session_state[clave] = resultado

    ejecucion = st.session_state.get(clave)
    if ejecucion is not None:
        with st.expander("Informe del agente", expanded=True):
            _informe(ejecucion, tipo, clave)


def _informe(ejecucion: dict, tipo: str, clave: str) -> None:
    agente = ejecucion["result"]
    pie_de_ejecucion(ejecucion)
    if agente["mode"] == "llm":
        st.success(f"Análisis dirigido por el LLM ({agente['model']}).")
        if agente["llm_usage_text"]:
            st.caption(f"Uso del LLM: {agente['llm_usage_text']}")
    else:
        st.warning(f"No se pudo usar el LLM; se usó el respaldo por reglas. Motivo: {agente['fallback_reason']}")

    metrica = agente["metric"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Variable objetivo", str(agente["target"]))
    m2.metric("Predictores", len(agente["predictors"]))
    m3.metric("Mejor modelo", agente["best_name"])
    m4.metric(f"{metrica} (media)", f"{agente['best_mean']:.4f}", f"± {agente['best_std']:.4f}", delta_color="off")
    st.caption(f"Mejor configuración: `{agente['best']}`")
    if agente["llm_choice"] and agente["llm_choice"] != agente["best"]:
        st.warning(f"El LLM eligió `{agente['llm_choice']}`, pero `{agente['best']}` tiene mejor {metrica} medio. "
                   "Revisa su justificación (por ejemplo, estabilidad entre particiones).")
    if agente["excluded"]:
        st.caption(f"Columnas excluidas de los predictores: {', '.join(map(str, agente['excluded']))}.")
    if agente["rows_without_target"]:
        st.caption(f"Se descartaron {agente['rows_without_target']} filas sin valor en el objetivo.")

    st.markdown("**Decisiones del agente**")
    for item in agente["decisions"]:
        st.markdown(f"- **{item['decision']}** — {item['justificacion']}")
    with st.expander("Plan de algoritmos"):
        for item in agente["plan"]:
            icono = "✅" if item.get("ejecutar") else "⏭️"
            st.markdown(f"{icono} **{item['paso']}** ({item['familia']}) — {item['razon']}")
    if agente["analysis"]:
        st.markdown("**Interpretación del LLM**")
        st.write(agente["analysis"])
    st.markdown("**Autoevaluación del agente**")
    (st.info if agente["beats_baseline"] else st.error)("\n".join(f"- {t}" for t in agente["self_evaluation"]))

    ranking = agente["ranking"]
    st.markdown(f"**Ranking de modelos validados** (media en {agente['repetitions']} particiones)")
    columnas_num = [c for c in ranking.columns if c not in ("Modelo", "Algoritmo")]
    st.dataframe(
        ranking.style.format({c: "{:.4f}" for c in columnas_num}, na_rep="-").apply(
            lambda fila: [f"background-color: rgba(252, 238, 10, 0.18); color: {AMARILLO}; font-weight: 700"
                          if fila["Modelo"] == agente["best"] else ""] * len(fila),
            axis=1,
        ),
        width="stretch",
        hide_index=True,
    )

    g1, g2 = st.columns(2)
    orden = ranking["Modelo"].tolist()
    with g1:
        fig_barras = go.Figure(go.Bar(
            x=ranking[metrica], y=ranking["Modelo"], orientation="h",
            error_x=dict(type="data", array=ranking[f"Desv. {metrica}"], color=MAGENTA),
            marker=dict(color=[AMARILLO if m == agente["best"] else CIAN for m in orden]),
            text=ranking[metrica].map(lambda v: f"{v:.4f}"), textposition="inside",
        ))
        fig_barras.update_yaxes(categoryorder="array", categoryarray=orden[::-1])
        fig_barras.update_layout(title=f"{metrica} medio ± desviación", xaxis_title=metrica)
        grafico(fig_barras, key=f"{clave}_barras", height=max(320, 40 * len(orden)))
    with g2:
        fig_caja = px.box(agente["by_seed"], x=metrica, y="Modelo", color="Modelo", points="all",
                          category_orders={"Modelo": orden}, color_discrete_sequence=SECUENCIA,
                          title="Estabilidad entre particiones")
        fig_caja.update_layout(showlegend=False)
        grafico(fig_caja, key=f"{clave}_caja", height=max(320, 40 * len(orden)))

    detalle = agente["best_detail"]
    if tipo == "classification":
        matriz = detalle["confusion_matrix"]
        fig_cm = px.imshow(np.asarray(matriz["matrix"]), x=matriz["labels"], y=matriz["labels"], text_auto=True,
                           color_continuous_scale=ESCALA_CONTINUA,
                           labels={"x": "Predicción", "y": "Real", "color": "Casos"},
                           title="Matriz de confusión del mejor modelo (partición 1)")
        grafico(fig_cm, key=f"{clave}_cm", height=420)
    else:
        y_test, y_pred = np.asarray(detalle["y_test"]), np.asarray(detalle["y_pred"])
        st.caption(f"Referencia: predecir siempre la media daría un RMSE ≈ {detalle['baseline_rmse']:.4f} en la "
                   f"partición 1; el mejor modelo obtiene {detalle['metrics']['RMSE']:.4f}.")
        limites = [float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))]
        fig_rp = px.scatter(x=y_test, y=y_pred, color=np.abs(y_test - y_pred), color_continuous_scale=ESCALA_CONTINUA,
                            labels={"x": "Real", "y": "Predicción", "color": "Error abs."},
                            title="Real vs. predicción del mejor modelo (partición 1)")
        fig_rp.add_scatter(x=limites, y=limites, mode="lines", name="Predicción perfecta",
                           line=dict(color=CIAN, dash="dash", width=2))
        grafico(fig_rp, key=f"{clave}_rp", height=440)

    st.caption(f"Para reproducirlo en el flujo manual, elige '{agente['target']}' como variable a predecir "
               f"y el algoritmo «{agente['best_name']}» con {agente['best_params'] or 'sus parámetros por defecto'}.")
    with st.expander("Bitácora de razonamiento"):
        st.dataframe(agente["log"], width="stretch")
    st.download_button("Descargar ranking (CSV)", data=ranking.to_csv(index=False).encode("utf-8"),
                       file_name=f"ranking_agente_{tipo}.csv", mime="text/csv", key=f"{clave}_descargar")
    st.download_button("Descargar informe del agente (HTML)", data=agente["report_html"].encode("utf-8"),
                       file_name=f"informe_agente_{tipo}.html", mime="text/html", key=f"{clave}_descargar_html")
