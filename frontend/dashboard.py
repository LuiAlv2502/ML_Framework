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
            st.Page(f"{PAGINAS}/home.py", title="Inicio", icon=":material/home:", default=True),
            st.Page(f"{PAGINAS}/data.py", title="Datos", icon=":material/table_view:"),
            st.Page(f"{PAGINAS}/eda.py", title="EDA", icon=":material/insights:"),
            st.Page(f"{PAGINAS}/preprocessing.py", title="Preprocesamiento", icon=":material/tune:"),
            st.Page(f"{PAGINAS}/unsupervised.py", title="Agrupamiento", icon=":material/scatter_plot:"),
            st.Page(f"{PAGINAS}/classification.py", title="Clasificación", icon=":material/category:"),
            st.Page(f"{PAGINAS}/prediction.py", title="Regresión", icon=":material/trending_up:"),
            st.Page(f"{PAGINAS}/agents.py", title="MAS", icon=":material/groups:"),
            st.Page(f"{PAGINAS}/experiments.py", title="Experimentos", icon=":material/science:"),
        ]
    )

    with st.sidebar:
        st.markdown("### Laboratorio IA")
        st.caption("Framework multiagente + Streamlit")

    pagina.run()
