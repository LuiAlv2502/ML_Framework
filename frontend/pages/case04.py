"""Interfaz independiente para la demostración del Caso 04."""
from __future__ import annotations

import streamlit as st

from frontend.components.shared import ejecutar, obtener_controlador

st.title("Caso 04")
st.write(
    "Ejecuta tres escenarios de complejidad creciente dentro del MAS existente: perfilado, "
    "cooperación entre agentes y consolidación de alternativas."
)
controlador = obtener_controlador()

if not controlador.has_data:
    st.info("Carga un conjunto de datos para ejecutar la demostración.")
else:
    datos = controlador.active_data()
    objetivos = [
        columna for columna in datos.columns
        if 2 <= datos[columna].nunique(dropna=True) <= 30
    ]
    if not objetivos:
        st.info("El conjunto activo no tiene una variable objetivo con 2 a 30 clases para la comparación.")
    else:
        sugerido = controlador.suggest_target()
        indice_objetivo = objetivos.index(sugerido) if sugerido in objetivos else 0
        with st.form("ejecutar_caso04"):
            objetivo = st.selectbox("Variable objetivo para clasificación", objetivos, index=indice_objetivo)
            semillas = st.multiselect(
                "Semillas para repetir las pruebas",
                [11, 42, 43, 44, 73],
                default=[42, 43, 44],
            )
            ejecutar_demo = st.form_submit_button("Ejecutar los tres escenarios", type="primary")
        if ejecutar_demo:
            resultado = ejecutar(
                lambda: controlador.run_case04_evaluation(objetivo, semillas),
                "Ejecutando escenarios y comparando resultados...",
            )
            if resultado is not None:
                st.session_state["mas_caso04_resultado"] = resultado

resultado_guardado = st.session_state.get("mas_caso04_resultado")
if resultado_guardado:
    st.caption(
        f"Última ejecución · conjunto: {resultado_guardado['dataset']} · "
        f"objetivo: {resultado_guardado['target']} · "
        f"semillas: {', '.join(map(str, resultado_guardado['seeds']))}"
    )
    st.dataframe(
        [
            {
                "Escenario": escenario["scenario"],
                "Complejidad": escenario["complexity"],
                "Tiempo (s)": escenario["duration_s"],
                "Interacciones": escenario["interactions"],
                "Tareas completadas": escenario["tasks_completed"],
                "Errores": escenario["errors"],
                "Métrica de calidad": escenario["quality_metric"] or "No aplica",
                "Calidad media": escenario["quality_mean"],
                "Desviación de calidad": escenario["quality_std"],
            }
            for escenario in resultado_guardado["scenarios"]
        ],
        width="stretch",
        hide_index=True,
    )
    st.markdown("**Comparación centralizada y multiagente del escenario 3**")
    st.dataframe(
        [
            {
                "Enfoque": comparacion["approach"],
                "Tiempo medio (s)": comparacion["duration_s"],
                "Desviación de tiempo (s)": comparacion["duration_std"],
                "Interacciones": comparacion["interactions"],
                "Tareas completadas": comparacion["tasks_completed"],
                "Modelos evaluados": comparacion["models_evaluated"],
                "Errores": comparacion["errors"],
                "F1 macro medio": comparacion["quality"],
                "Desviación F1 macro": comparacion["quality_std"],
                "Resultado elegido": comparacion["selected"],
            }
            for comparacion in (
                resultado_guardado["comparison"]["centralized"],
                resultado_guardado["comparison"]["multiagent"],
            )
        ],
        width="stretch",
        hide_index=True,
    )
    st.markdown("**Resultados por semilla y enfoque**")
    st.dataframe(
        [
            {
                "Semilla": fila["seed"],
                "Enfoque": fila["approach"],
                "Tiempo (s)": fila["duration_s"],
                "Interacciones": fila["interactions"],
                "Tareas completadas": fila["tasks_completed"],
                "Errores": fila["errors"],
                "F1 macro": fila["quality"],
                "Seleccionado": fila["selected"],
            }
            for fila in resultado_guardado["comparison"]["trials"]
        ],
        width="stretch",
        hide_index=True,
    )
    for escenario in resultado_guardado["scenarios"]:
        with st.expander(escenario["scenario"]):
            st.write(f"**Agentes participantes:** {', '.join(escenario['agents']) or 'No aplica'}")
            st.write(f"**Experimentos guardados:** {', '.join(escenario['run_ids']) or 'No registrados'}")
            for corrida in escenario["task_runs"]:
                st.markdown(
                    f"**{corrida['task']} · semilla {corrida['seed'] or 'única'} · "
                    f"política {corrida['policy'] or 'sin coordinación'}**"
                )
                if corrida["candidates"]:
                    st.caption(f"Candidatos: {', '.join(corrida['candidates'])}")
                if corrida["selected_agent"]:
                    st.caption(f"Agente seleccionado: {corrida['selected_agent']}")
                if corrida["decisions"]:
                    st.dataframe(
                        [
                            {
                                "Decisión": decision.get("decision"),
                                "Justificación": decision.get("justification"),
                            }
                            for decision in corrida["decisions"]
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                if corrida["messages"]:
                    st.dataframe(corrida["messages"], width="stretch", hide_index=True)
    st.caption("Las ejecuciones también quedan disponibles en la sección Experimentos.")
