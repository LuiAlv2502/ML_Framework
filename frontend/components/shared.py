"""Estado compartido entre páginas: controlador por sesión, carga de datos en la barra lateral y mensajes.

Las páginas **solo** hablan con :class:`framework_ia.controller.FrameworkController`; este módulo crea una
instancia por sesión de Streamlit (la agencia rehidrata los agentes persistidos al arrancar).
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

import streamlit as st

from framework_ia.controller import FrameworkController
from framework_ia.services.dataset_service import SOURCES, DatasetService
from framework_ia.utils.exceptions import friendly_message

CLAVES_RESULTADOS = (
    "clf_particiones",
    "clf_resultados",
    "clf_comparacion",
    "clf_benchmark",
    "exp_seleccionado",
    "reg_particiones",
    "reg_resultados",
    "reg_benchmark",
    "reg_exp_seleccionado",
    "agente_no_supervisado",
    "no_supervisado_archivo_hash",
    "agrupamiento_benchmark",
    "agente_clasificacion",
    "agente_regresion",
)


def obtener_controlador() -> FrameworkController:
    """Controlador de la sesión (se crea una sola vez)."""
    if "controller" not in st.session_state:
        st.session_state["controller"] = FrameworkController()
    return st.session_state["controller"]


def limpiar_resultados() -> None:
    """Descarta particiones y modelos entrenados cuando cambian los datos."""
    for clave in CLAVES_RESULTADOS:
        st.session_state.pop(clave, None)


def limpiar_estado_eda() -> None:
    """Descarta el estado de la página EDA ligado al archivo anterior."""
    for clave in ("eda_objetivo", "eda_mensaje"):
        st.session_state.pop(clave, None)


def seleccionar_modelo_activo(fuente, modelo_original, modelo_manual, modelo_eda, archivo_hash, eda_archivo_hash):
    """Selecciona los datos activos sin perder la copia del archivo cargado (regla del backend)."""
    return DatasetService.select_active(fuente, modelo_original, modelo_manual, modelo_eda,
                                        archivo_hash, eda_archivo_hash)


def restaurar_csv_original() -> None:
    """Activa una copia limpia del archivo cargado."""
    obtener_controlador().restore_dataset()
    limpiar_resultados()


def mostrar_error(error: BaseException, prefijo: str = "") -> None:
    """Muestra un error amigable (nunca un traceback)."""
    st.error(f"{prefijo}{friendly_message(error)}")


def ejecutar(funcion: Callable[[], Any], spinner: str = "Procesando...", prefijo: str = "") -> Any:
    """Ejecuta una llamada al controlador mostrando un spinner y un error amigable si falla."""
    try:
        with st.spinner(spinner):
            return funcion()
    except Exception as error:  # el usuario solo ve el mensaje
        mostrar_error(error, prefijo)
        return None


def pie_de_ejecucion(resultado: dict[str, Any] | None) -> None:
    """Indica qué agente atendió la tarea y con qué capacidad (trazabilidad del MAS)."""
    if not resultado:
        return
    texto = (f"Atendido por **{resultado.get('selected_agent')}** · capacidad `{resultado.get('capability')}` · "
             f"política `{resultado.get('policy')}`")
    if resultado.get("run_id"):
        texto += f" · experimento `{resultado['run_id']}`"
    st.caption(texto)


def _cargar_desde_barra_lateral(controlador: FrameworkController) -> bool:
    archivo = st.sidebar.file_uploader("Cargar datos (CSV o Excel)", type=["csv", "xlsx", "xls"])
    if archivo is None:
        return False
    contenido = archivo.getvalue()
    huella = hashlib.sha256(contenido).hexdigest()
    if st.session_state.get("ultimo_archivo_subido") == huella:
        return False
    st.session_state["ultimo_archivo_subido"] = huella
    try:
        controlador.load_dataset(contenido, archivo.name)
    except Exception as error:
        st.sidebar.error(f"No se pudo cargar el archivo: {friendly_message(error)}")
        return False
    limpiar_resultados()
    limpiar_estado_eda()
    return True


def estado_barra_lateral(controlador: FrameworkController, actualizado: bool = False) -> None:
    """Resumen del dataset activo en la barra lateral."""
    with st.sidebar:
        if not controlador.has_data:
            st.info("Cargue un archivo aquí o en la página **Datos** para comenzar.")
            return
        info = controlador.dataset_info()
        (st.success if actualizado else st.caption)(f"Archivo: {info['name']}")
        c1, c2 = st.columns(2)
        c1.metric("Filas", info["rows"])
        c2.metric("Columnas", info["columns"])
        st.caption(f"Datos activos: {SOURCES.get(info['source'], info['source'])}")


def obtener_datos(recomendar_eda: bool = False) -> FrameworkController | None:
    """Barra lateral de carga y estado; devuelve el controlador si hay datos activos."""
    controlador = obtener_controlador()
    actualizado = _cargar_desde_barra_lateral(controlador)
    estado_barra_lateral(controlador, actualizado)
    if not controlador.has_data:
        return None
    fuente = controlador.dataset_info()["source"]
    if recomendar_eda and fuente != "agente":
        if fuente in ("manual", "preprocesado"):
            st.info("Se usarán sus cambios de limpieza/preprocesamiento. Si quiere recomendaciones automáticas, "
                    "puede ejecutar el agente EDA.")
        else:
            st.info("Puede revisar los datos y ejecutar el agente EDA para recibir recomendaciones; "
                    "también puede continuar directamente con el archivo original.")
        try:
            st.page_link("frontend/pages/eda.py", label="Revisar el EDA (opcional)", icon=":material/smart_toy:")
        except Exception:  # la página se ejecuta fuera de st.navigation (por ejemplo, en pruebas)
            pass
    return controlador


# Compatibilidad con el nombre usado por las páginas de los laboratorios.
obtener_modelo = obtener_datos
