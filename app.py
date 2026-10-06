import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

st.set_page_config(
    page_title="Laboratorio IA",
    layout="wide",
)

pg = st.navigation(
    [
        st.Page(
            "pages/unsupervised.py",
            title="No supervisado",
            icon=":material/scatter_plot:",
            default=True,
        ),
        st.Page(
            "pages/classification.py",
            title="Supervisado - Clasificación",
            icon=":material/category:",
        ),
        st.Page(
            "pages/prediction.py",
            title="Supervisado - Predicción",
            icon=":material/trending_up:",
        ),
    ]
)

with st.sidebar:
    st.markdown("### Laboratorio IA")
    st.caption("Análisis de datos con MVC + Streamlit")

pg.run()
