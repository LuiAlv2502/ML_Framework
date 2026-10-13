import hashlib
import io

import pandas as pd
import streamlit as st

from model import DataModel


def obtener_modelo():
    if "model" not in st.session_state:
        st.session_state.model = DataModel()
        st.session_state.archivo_nombre = None
    st.session_state.setdefault("archivo_hash", None)

    archivo = st.sidebar.file_uploader("Cargar CSV", type=["csv"])
    archivo_actualizado = False

    if archivo is not None:
        contenido = archivo.getvalue()
        archivo_hash = hashlib.sha256(contenido).hexdigest()
    else:
        contenido = None
        archivo_hash = None

    if archivo is not None and archivo_hash != st.session_state.archivo_hash:
        try:
            df = pd.read_csv(io.BytesIO(contenido))
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.sidebar.error(f"No se pudo leer el CSV: {error}")
            return None

        st.session_state.model.cargar_dataframe(df)
        st.session_state.archivo_nombre = archivo.name
        st.session_state.archivo_hash = archivo_hash
        archivo_actualizado = True

        for clave in (
            "clf_particiones",
            "clf_resultados",
            "clf_comparacion",
            "exp_seleccionado",
        ):
            st.session_state.pop(clave, None)

    model = st.session_state.model
    filas, columnas = model.dimensiones()

    with st.sidebar:
        if filas > 0:
            nombre_archivo = st.session_state.archivo_nombre or "Archivo cargado"
            if archivo_actualizado:
                st.success(f"CSV actualizado: {nombre_archivo}")
            else:
                st.success(nombre_archivo)
            col1, col2 = st.columns(2)
            col1.metric("Filas", filas)
            col2.metric("Columnas", columnas)
        else:
            st.info("Sube un CSV para comenzar.")

    return model if filas > 0 else None
