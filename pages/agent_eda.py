import streamlit as st

from agents import AgenteEDA
from pages._shared import obtener_modelo

st.title("Agente inteligente de EDA")
st.caption(
    "El agente observa los datos, decide qué pasos del EDA son necesarios, los ejecuta, "
    "interpreta cada resultado y gráfico, y da recomendaciones. No modifica los datos cargados."
)

model = obtener_modelo()
if model is None:
    st.stop()

df = model.dataframe.dataframe

c1, c2 = st.columns([2, 1])
objetivo = c1.selectbox(
    "Variable objetivo (opcional)",
    ["(ninguna)", *df.columns.tolist()],
    help="Si se indica, el agente analiza si el problema es de clasificación o regresión.",
)
ejecutar = c2.button("Ejecutar agente", type="primary", width="stretch")

if ejecutar:
    with st.spinner("El agente está analizando los datos..."):
        agente = AgenteEDA(df, objetivo=None if objetivo == "(ninguna)" else objetivo).ejecutar()
        st.session_state["agente_eda"] = agente
        st.session_state["agente_eda_html"] = agente.informe_html()

agente = st.session_state.get("agente_eda")
if agente is None:
    st.info("Presione **Ejecutar agente** para iniciar el análisis.")
    st.stop()

# ---------------------------------------------------------------------------
# Síntesis
# ---------------------------------------------------------------------------
st.subheader("Síntesis del agente")
st.write(agente.sintesis["resumen"])
for recomendacion in agente.sintesis["recomendaciones"]:
    if recomendacion.startswith("[ALTA]"):
        st.error(recomendacion)
    elif recomendacion.startswith("[MEDIA]"):
        st.warning(recomendacion)
    else:
        st.success(recomendacion)
st.caption(agente.sintesis["advertencia"])

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
with st.expander("Plan del agente (qué decidió hacer y por qué)", expanded=True):
    for item in agente.plan:
        icono = "✅" if item["ejecutar"] else "⏭️"
        st.markdown(f"{icono} **{item['paso']}** — {item['razon']}")

# ---------------------------------------------------------------------------
# Pasos
# ---------------------------------------------------------------------------
st.subheader("Pasos del EDA")
iconos = {"alta": "🔴", "media": "🟡", "informativa": "🟢"}
for paso in agente.pasos:
    with st.expander(f"{iconos[paso.severidad]} Paso {paso.numero}. {paso.titulo}"):
        st.markdown(f"**Observación:** {paso.observacion}")
        st.markdown(f"**Razonamiento:** {paso.razonamiento}")
        st.markdown(f"**Acción:** {paso.accion}")
        for nombre, tabla in paso.tablas.items():
            st.markdown(f"*{nombre}*")
            st.dataframe(tabla, width="stretch")
        for titulo, figura in paso.figuras:
            st.markdown(f"*{titulo}*")
            st.pyplot(figura, clear_figure=False)
        st.markdown(f"**Interpretación:** {paso.interpretacion}")
        st.markdown("**Recomendaciones:**")
        for recomendacion in paso.recomendaciones:
            st.markdown(f"- {recomendacion}")

# ---------------------------------------------------------------------------
# Bitácora y descargas
# ---------------------------------------------------------------------------
with st.expander("Bitácora de razonamiento"):
    st.dataframe(agente.bitacora, width="stretch")

d1, d2 = st.columns(2)
d1.download_button(
    "Descargar informe (HTML)",
    data=st.session_state["agente_eda_html"].encode("utf-8"),
    file_name="informe_agente_eda.html",
    mime="text/html",
    width="stretch",
)
d2.download_button(
    "Descargar datos con la limpieza sugerida (CSV)",
    data=agente.datos_limpios.to_csv(index=False).encode("utf-8"),
    file_name="datos_limpios_agente.csv",
    mime="text/csv",
    width="stretch",
)