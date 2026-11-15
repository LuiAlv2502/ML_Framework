"""Carga y exploración del conjunto de datos activo."""
from __future__ import annotations

import streamlit as st

from frontend.components.shared import obtener_controlador, obtener_datos, restaurar_csv_original

st.title("Datos")
st.write("Carga tus propios datos o selecciona uno de los conjuntos de ejemplo.")

controlador = obtener_datos()
controlador = controlador or obtener_controlador()
disponibles = controlador.list_datasets()

if disponibles:
    etiquetas = {str(item["name"]): item for item in disponibles}
    nombre = st.selectbox("Conjunto de ejemplo", list(etiquetas))
    if st.button("Cargar conjunto de ejemplo", type="primary"):
        try:
            controlador.load_sample(nombre)
            st.rerun()
        except Exception as error:
            st.error(f"No se pudo cargar el conjunto: {error}")
else:
    st.info("No hay conjuntos de ejemplo en la carpeta de datos. Puedes cargar un archivo desde la barra lateral.")

if controlador.has_data:
    info = controlador.dataset_info()
    resumen = controlador.dataset_summary()
    st.subheader("Conjunto activo")
    columnas = st.columns(4)
    columnas[0].metric("Archivo", info["name"])
    columnas[1].metric("Filas", resumen["rows"])
    columnas[2].metric("Columnas", resumen["columns"])
    columnas[3].metric("Valores faltantes", resumen["nulls_total"])
    st.caption(f"Origen de los datos activos: {info['source']}")

    original = controlador.original_data()
    activo = controlador.active_data()
    if not activo.equals(original) and st.button("Restaurar datos originales"):
        restaurar_csv_original()
        st.rerun()

    st.subheader("Vista previa")
    st.dataframe(controlador.dataset_preview(20), width="stretch", hide_index=True)
    with st.expander("Resumen de columnas"):
        st.dataframe(controlador.dataset_summary()["types"], width="stretch", hide_index=True)
