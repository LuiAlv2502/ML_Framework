"""Operaciones guiadas para preparar los datos."""
from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.components.shared import ejecutar, obtener_datos, restaurar_csv_original

st.title("Preprocesamiento")
st.write("Agrega operaciones en el orden en que deben aplicarse. El conjunto original se conserva.")
controlador = obtener_datos()
if controlador is None:
    st.info("Carga primero un conjunto de datos desde la barra lateral o la página **Datos**.")
    st.stop()

datos = controlador.active_data()
columnas = list(datos.columns)
operaciones_disponibles = controlador.preprocessing_operations()
opcion = st.selectbox(
    "Operación",
    list(operaciones_disponibles),
    format_func=lambda nombre: operaciones_disponibles[nombre],
)

def _columnas(opcional: bool = True) -> list[Any] | None:
    elegidas = st.multiselect("Columnas", columnas, help="Si no eliges ninguna, se aplicará a las columnas compatibles.")
    return elegidas or (None if opcional else [])

def _formar_operacion(nombre: str) -> dict[str, Any]:
    operacion: dict[str, Any] = {"op": nombre}
    if nombre == "impute":
        operacion["columns"] = _columnas()
        operacion["strategy"] = st.selectbox("Método para completar valores", ["mediana_moda", "media_moda"])
    elif nombre == "drop_na":
        operacion["columns"] = _columnas()
    elif nombre == "fill_na":
        operacion["columns"] = _columnas()
        tipo_valor = st.selectbox("Tipo del valor de reemplazo", ["Texto", "Número entero", "Número decimal"])
        if tipo_valor == "Texto":
            operacion["value"] = st.text_input("Valor de reemplazo")
        elif tipo_valor == "Número entero":
            operacion["value"] = st.number_input("Valor de reemplazo", value=0, step=1)
        else:
            operacion["value"] = st.number_input("Valor de reemplazo", value=0.0, step=0.1)
    elif nombre in ("normalize", "standardize", "encode_onehot", "encode_ordinal"):
        operacion["columns"] = _columnas()
        if nombre == "encode_onehot":
            operacion["drop_first"] = st.checkbox("Omitir una categoría por columna")
    elif nombre in ("drop_columns", "select_features"):
        operacion["columns"] = _columnas(opcional=False)
    elif nombre == "convert_type":
        operacion["column"] = st.selectbox("Columna", columnas)
        operacion["dtype"] = st.selectbox("Tipo nuevo", controlador.preprocessing_types())
    elif nombre == "sort":
        operacion["column"] = st.selectbox("Ordenar por", columnas)
        operacion["ascending"] = st.checkbox("Orden ascendente", value=True)
    return operacion

with st.form("agregar_operacion"):
    operacion = _formar_operacion(opcion)
    agregar = st.form_submit_button("Agregar a la preparación")

if agregar:
    if opcion in ("drop_columns", "select_features") and not operacion.get("columns"):
        st.error("Selecciona al menos una columna.")
    else:
        st.session_state.setdefault("preprocessing_operations", []).append(operacion)

pendientes = st.session_state.setdefault("preprocessing_operations", [])
if pendientes:
    st.subheader("Operaciones en orden")
    st.dataframe(
        [{"#": i, "Operación": operaciones_disponibles.get(item["op"], item["op"]),
          "Parámetros": {k: v for k, v in item.items() if k != "op"}}
         for i, item in enumerate(pendientes, start=1)],
        width="stretch",
        hide_index=True,
    )
    botones = st.columns(2)
    if botones[0].button("Aplicar preparación", type="primary"):
        resultado = ejecutar(lambda: controlador.configure_preprocessing(pendientes), "Preparando los datos...")
        if resultado:
            st.session_state["preprocessing_operations"] = []
            st.success("Preparación aplicada.")
            st.dataframe(resultado["result"]["steps"], width="stretch", hide_index=True)
            st.dataframe(resultado["result"]["comparison"], width="stretch", hide_index=True)
    if botones[1].button("Vaciar lista"):
        st.session_state["preprocessing_operations"] = []
        st.rerun()

if controlador.dataset_info()["source"] != "original" and st.button("Volver al conjunto original"):
    restaurar_csv_original()
    st.rerun()
