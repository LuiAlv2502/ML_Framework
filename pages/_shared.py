import pandas as pd
import streamlit as st

from model import DataModel


def obtener_modelo():
    if "model" not in st.session_state:
        st.session_state.model = DataModel()
        st.session_state.archivo_nombre = None

    archivo = st.sidebar.file_uploader("Cargar CSV", type=["csv"])

    if archivo is not None and archivo.name != st.session_state.archivo_nombre:
        df = pd.read_csv(archivo)
        st.session_state.model.cargar_dataframe(df)
        st.session_state.archivo_nombre = archivo.name

    model = st.session_state.model
    filas, columnas = model.dimensiones()

    with st.sidebar:
        if filas > 0:
            st.success(st.session_state.archivo_nombre or "Archivo cargado")
            col1, col2 = st.columns(2)
            col1.metric("Filas", filas)
            col2.metric("Columnas", columnas)
        else:
            st.info("Sube un CSV para comenzar.")

    return model if filas > 0 else None
