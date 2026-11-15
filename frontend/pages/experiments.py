"""Historial, consulta, comparación y descarga de experimentos."""
from __future__ import annotations

import streamlit as st

from frontend.components.shared import ejecutar, obtener_controlador

st.title("Experimentos")
st.write("Revisa análisis anteriores, compara sus resultados o descarga un informe.")
controlador = obtener_controlador()

with st.expander("Filtrar historial"):
    tareas_disponibles = ["Todas", "eda", "clustering", "classification", "regression",
                          "classification_benchmark", "regression_benchmark", "clustering_benchmark"]
    tarea = st.selectbox("Tipo de análisis", tareas_disponibles)
    desde = st.text_input("Desde (AAAA-MM-DD)")
    hasta = st.text_input("Hasta (AAAA-MM-DD)")

filtros = {"task": None if tarea == "Todas" else tarea, "since": desde or None, "until": hasta or None}
historial = ejecutar(lambda: controlador.list_experiments(**filtros), "Cargando historial...")
if historial is None:
    st.stop()
if historial.empty:
    st.info("No hay experimentos guardados para esos filtros.")
    st.stop()

st.dataframe(historial, width="stretch", hide_index=True)
ids = historial["id"].astype(str).tolist()
st.subheader("Consultar un experimento")
seleccion = st.selectbox("Experimento", ids)
detalle = ejecutar(lambda: controlador.get_experiment(seleccion), "Cargando experimento...")
if detalle:
    registro = detalle["run"]
    st.write(f"**Tarea:** {registro.get('task', '—')} · **Estado:** {registro.get('status', '—')}")
    st.write(f"**Conjunto:** {(registro.get('dataset') or {}).get('name', '—')}")
    st.json(registro.get("summary") or {})
    if not detalle["metrics"].empty:
        st.subheader("Resultados")
        st.dataframe(detalle["metrics"], width="stretch", hide_index=True)
    with st.expander("Configuración y decisiones"):
        st.json({"config": registro.get("config"), "decisions": registro.get("decisions")})
    if detalle["messages"]:
        with st.expander("Mensajes intercambiados durante el análisis"):
            st.dataframe(
                [{clave: mensaje.get(clave) for clave in
                  ("timestamp", "sender", "receiver", "task", "status", "error")}
                 for mensaje in detalle["messages"]],
                width="stretch",
                hide_index=True,
            )
    formato = st.selectbox("Formato de descarga", ["json", "csv", "html"])
    descarga = ejecutar(lambda: controlador.export_experiment(seleccion, formato), "Preparando descarga...")
    if descarga is not None:
        st.download_button(
            "Descargar experimento",
            data=descarga,
            file_name=f"{seleccion}.{formato}",
            mime={"json": "application/json", "csv": "text/csv", "html": "text/html"}[formato],
        )

st.subheader("Comparar dos experimentos")
if len(ids) >= 2:
    columnas = st.columns(2)
    primero = columnas[0].selectbox("Primer experimento", ids, key="compare_first")
    segundo = columnas[1].selectbox("Segundo experimento", ids, index=1, key="compare_second")
    if st.button("Comparar"):
        resultado = ejecutar(lambda: controlador.compare_experiments(primero, segundo), "Comparando...")
        if resultado is not None:
            st.dataframe(resultado, width="stretch", hide_index=True)
else:
    st.caption("Se necesitan al menos dos experimentos para compararlos.")

with st.expander("Eliminar experimento"):
    eliminar_id = st.selectbox("Experimento a eliminar", ids, key="delete_experiment")
    confirmar = st.checkbox("Confirmo que quiero eliminar permanentemente este experimento")
    if st.button("Eliminar", disabled=not confirmar):
        def _eliminar() -> bool:
            controlador.delete_experiment(eliminar_id)
            return True

        if ejecutar(_eliminar, "Eliminando..."):
            st.success("Experimento eliminado.")
            st.rerun()
