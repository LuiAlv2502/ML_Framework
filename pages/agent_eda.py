import streamlit as st

from agents import AgenteEDA
from model import DataModel
from pages._shared import limpiar_resultados, obtener_modelo

st.title("Agente inteligente de EDA")
st.caption(
    "Este paso prepara una copia de los datos para los análisis. El CSV original se conserva. "
    "Si usarás clasificación o regresión, selecciona aquí la variable objetivo para evitar que "
    "el agente la considere irrelevante. Los valores faltantes de predictores se conservan aquí "
    "para imputarlos después de separar entrenamiento y prueba."
)

model = obtener_modelo(requiere_eda=False)
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
        modelo_eda = DataModel()
        modelo_eda.cargar_dataframe(agente.datos_para_modelos)
        limpiar_resultados()
        st.session_state["agente_eda"] = agente
        st.session_state["agente_eda_html"] = agente.informe_html()
        st.session_state["modelo_eda"] = modelo_eda
        st.session_state["eda_archivo_hash"] = st.session_state.get("archivo_hash")
        st.session_state["eda_objetivo"] = None if objetivo == "(ninguna)" else objetivo

agente = st.session_state.get("agente_eda")
if agente is None:
    st.info("Presione **Ejecutar agente** para iniciar el análisis.")
    st.stop()

objetivo_actual = None if objetivo == "(ninguna)" else objetivo
if st.session_state.get("eda_objetivo") != objetivo_actual:
    st.session_state.pop("modelo_eda", None)
    st.session_state.pop("eda_archivo_hash", None)
if (
    st.session_state.get("eda_archivo_hash") != st.session_state.get("archivo_hash")
    or st.session_state.get("eda_objetivo") != objetivo_actual
):
    st.warning("El archivo o la variable objetivo cambiaron. Vuelve a ejecutar el EDA antes de usar los modelos.")

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
    "Descargar datos preparados para modelos (CSV)",
    data=agente.datos_para_modelos.to_csv(index=False).encode("utf-8"),
    file_name="datos_preparados_agente.csv",
    mime="text/csv",
    width="stretch",
)