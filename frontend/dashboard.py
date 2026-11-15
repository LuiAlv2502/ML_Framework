"""Punto de entrada del dashboard Streamlit: configura la página, el tema y la navegación."""
from __future__ import annotations

import streamlit as st

from frontend.components.theme import aplicar_estilo

PAGINAS = "frontend/pages"


def main() -> None:
    """Construye la navegación multipágina y ejecuta la página seleccionada."""
    st.set_page_config(page_title="Laboratorio IA", layout="wide")
    aplicar_estilo()

    pagina = st.navigation(
        [
            st.Page(f"{PAGINAS}/eda.py", title="EDA", icon=":material/smart_toy:", default=True),
            st.Page(f"{PAGINAS}/unsupervised.py", title="No supervisado", icon=":material/scatter_plot:"),
            st.Page(f"{PAGINAS}/classification.py", title="Supervisado - Clasificación", icon=":material/category:"),
            st.Page(f"{PAGINAS}/prediction.py", title="Supervisado - Predicción", icon=":material/trending_up:"),
        ]
    )

    with st.sidebar:
        st.markdown("### Laboratorio IA")
        st.caption("Análisis de datos con MVC + Streamlit · Night City edition")

    pagina.run()
