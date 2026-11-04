import hashlib
import io

import pandas as pd
import streamlit as st

from model import DataModel

CLAVES_RESULTADOS = (
    "clf_particiones",
    "clf_resultados",
    "clf_comparacion",
    "exp_seleccionado",
    "reg_particiones",
    "reg_resultados",
    "reg_benchmark",
    "reg_exp_seleccionado",
    "agente_no_supervisado",
    "no_supervisado_archivo_hash",
)


def limpiar_resultados():
    """Descarta particiones y modelos entrenados cuando cambian los datos."""
    for clave in CLAVES_RESULTADOS:
        st.session_state.pop(clave, None)


def limpiar_estado_eda():
    """Descarta el resultado del EDA y el conjunto derivado del archivo anterior."""
    for clave in (
        "agente_eda",
        "agente_eda_html",
        "eda_archivo_hash",
        "eda_objetivo",
        "modelo_eda",
        "modelo_manual",
    ):
        st.session_state.pop(clave, None)


def seleccionar_modelo_activo(
    fuente,
    modelo_original,
    modelo_manual,
    modelo_eda,
    archivo_hash,
    eda_archivo_hash,
):
    """Selecciona los datos activos sin perder la copia del CSV cargado."""
    if (
        fuente == "agente"
        and modelo_eda is not None
        and archivo_hash is not None
        and eda_archivo_hash == archivo_hash
    ):
        return modelo_eda
    if fuente in ("original", "manual") and modelo_manual is not None:
        return modelo_manual
    return modelo_original


def restaurar_csv_original():
    """Activa una copia limpia del CSV cargado, preservando el modelo canónico."""
    modelo_original = st.session_state.get("model")
    if modelo_original is None:
        raise RuntimeError("No hay un CSV cargado para restaurar.")

    modelo_manual = DataModel()
    modelo_manual.cargar_dataframe(modelo_original.dataframe.dataframe)
    st.session_state["modelo_manual"] = modelo_manual
    st.session_state["modo_datos_activos"] = "original"
    limpiar_resultados()


def obtener_modelo(recomendar_eda=False):
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
        limpiar_resultados()
        limpiar_estado_eda()
        modelo_manual = DataModel()
        modelo_manual.cargar_dataframe(df)
        st.session_state["modelo_manual"] = modelo_manual
        st.session_state["modo_datos_activos"] = "original"

    if "modelo_manual" not in st.session_state and st.session_state.model.dimensiones()[0] > 0:
        modelo_manual = DataModel()
        modelo_manual.cargar_dataframe(st.session_state.model.dataframe.dataframe)
        st.session_state["modelo_manual"] = modelo_manual

    model = seleccionar_modelo_activo(
        st.session_state.get("modo_datos_activos", "original"),
        st.session_state.model,
        st.session_state.get("modelo_manual"),
        st.session_state.get("modelo_eda"),
        st.session_state.get("archivo_hash"),
        st.session_state.get("eda_archivo_hash"),
    )
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
            fuente = st.session_state.get("modo_datos_activos", "original")
            etiquetas_fuente = {
                "original": "CSV original",
                "manual": "Cambios manuales",
                "agente": "Preparación del agente EDA",
            }
            st.caption(f"Datos activos: {etiquetas_fuente.get(fuente, 'CSV original')}")
        else:
            st.info("Sube un CSV para comenzar.")

    if filas == 0:
        return None

    if recomendar_eda and st.session_state.get("modo_datos_activos", "original") != "agente":
        if st.session_state.get("modo_datos_activos") == "manual":
            st.info(
                "Se usarán tus cambios manuales. Si quieres recomendaciones automáticas, "
                "puedes ejecutar el agente EDA."
            )
        else:
            st.info(
                "Puedes revisar los datos y ejecutar el agente EDA para recibir recomendaciones; "
                "también puedes continuar directamente con el CSV original."
            )
        st.page_link("pages/eda.py", label="Revisar el EDA (opcional)", icon=":material/smart_toy:")

    return model
