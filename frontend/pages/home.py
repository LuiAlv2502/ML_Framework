"""Página de inicio con el estado general del laboratorio."""
from __future__ import annotations

import streamlit as st

from frontend.components.shared import obtener_controlador

st.title("Laboratorio de análisis de datos")
st.write("Carga un conjunto de datos, prepara sus variables y ejecuta análisis desde las páginas del menú.")

controlador = obtener_controlador()
estado = controlador.mas_status()
datos = estado.get("dataset")
agentes = estado.get("agents", 0)

columnas = st.columns(3)
columnas[0].metric("Agentes disponibles", estado.get("active", 0))
columnas[1].metric("Agentes registrados", agentes)
columnas[2].metric("Tipos de tareas", estado.get("capabilities", 0))

if datos:
    st.subheader("Datos activos")
    resumen = controlador.dataset_summary()
    columnas = st.columns(4)
    columnas[0].metric("Archivo", datos["name"])
    columnas[1].metric("Filas", resumen["rows"])
    columnas[2].metric("Columnas", resumen["columns"])
    columnas[3].metric("Valores faltantes", resumen["nulls_total"])
else:
    st.info("Aún no hay datos cargados. Abre **Datos** para cargar un archivo o probar un conjunto incluido.")

st.subheader("Mensajes recientes")
mensajes = controlador.list_messages(limit=5)
if mensajes:
    st.dataframe(
        [{"Hora": mensaje["timestamp"], "Emisor": mensaje["sender"], "Receptor": mensaje["receiver"],
          "Tarea": mensaje["task"], "Estado": mensaje["status"]}
         for mensaje in mensajes[::-1]],
        width="stretch",
        hide_index=True,
    )
else:
    st.caption("Los mensajes entre componentes aparecerán aquí cuando se ejecuten análisis.")

servicio_ia = estado.get("llm", {})
if servicio_ia.get("available"):
    st.success("El servicio de IA está disponible.")
elif servicio_ia.get("enabled"):
    st.info("El servicio de IA no está disponible; los análisis compatibles pueden usar reglas automáticas.")
else:
    st.caption("El servicio de IA está desactivado; los análisis compatibles pueden usar reglas automáticas.")

ultima_tarea = estado.get("last_task")
if ultima_tarea:
    st.subheader("Último análisis")
    tarea = ultima_tarea.get("task", "—")
    etiquetas_tarea = {
        "eda": "Exploración de datos",
        "clustering": "Análisis de grupos",
        "classification": "Clasificación",
        "regression": "Predicción de valores",
        "classification_benchmark": "Comparación de clasificadores",
        "regression_benchmark": "Comparación de predicciones",
        "clustering_benchmark": "Comparación de grupos",
    }
    st.write(f"Análisis: **{etiquetas_tarea.get(tarea, tarea.replace('.', ' ').replace('_', ' ').capitalize())}**")
    st.write("Estado: **Completado**" if ultima_tarea.get("ok") else "Estado: **Con errores**")
    if ultima_tarea.get("error"):
        st.error(ultima_tarea["error"])
