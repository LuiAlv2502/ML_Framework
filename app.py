import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

from pages._theme import aplicar_estilo  # noqa: E402

st.set_page_config(
    page_title="Laboratorio IA",
    layout="wide",
)
aplicar_estilo()

pg = st.navigation(
    [
        st.Page(
            "pages/agent_eda.py",
            title="Agente inteligente EDA",
            icon=":material/smart_toy:",
        ),
        st.Page(
            "pages/unsupervised.py",
            title="No supervisado",
            icon=":material/scatter_plot:",
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
    st.caption("Análisis de datos con MVC + Streamlit · Night City edition")

pg.run()
