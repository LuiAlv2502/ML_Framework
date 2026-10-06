import streamlit as st

from pages._shared import obtener_modelo

st.title("Supervisado - Predicción")

model = obtener_modelo()
if model is None:
    st.stop()

st.info(
    "El módulo de regresión aún no está implementado. Cuando añadas el bloque de regresión, "
    "replica el mismo patrón que la clasificación."
)
