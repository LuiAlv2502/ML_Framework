import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents import AgenteClasificacion, AgenteRegresion, llm
from pages._theme import AMARILLO, CIAN, ESCALA_CONTINUA, MAGENTA, SECUENCIA, grafico

TITULOS = {"clasificacion": "clasificación", "regresion": "regresión"}
AGENTES = {"clasificacion": AgenteClasificacion, "regresion": AgenteRegresion}
AUTOMATICO = "(que decida el agente)"


def seccion_agente(tipo, df):
    """Bloque del agente (controles + informe), separado del flujo manual como en el EDA."""
    clave = f"agente_{tipo}"
    with st.container(border=True):
        st.markdown(f"**Agente inteligente de {TITULOS[tipo]}**")
        st.caption(
            "Opcional. El LLM decide qué variable predecir, qué columnas excluir y qué modelos "
            "validar; Python entrena con las clases del framework en varias particiones y mide. "
            "Gemini solo recibe perfiles y métricas agregadas, nunca las filas del CSV."
        )
        a0, a1, a2, a3 = st.columns([1.3, 1, 1, 1])
        opciones = [AUTOMATICO, *map(str, df.columns)]
        agente_eda = st.session_state.get("agente_eda")
        # El agente EDA puede haber renombrado la columna objetivo.
        sugerido = str(agente_eda.objetivo if agente_eda is not None else st.session_state.get("eda_objetivo"))
        eleccion = a0.selectbox(
            "Variable objetivo",
            opciones,
            index=opciones.index(sugerido) if sugerido in opciones else 0,
            key=f"{clave}_objetivo",
            help="Por defecto se usa la elegida en el EDA, si existe.",
        )
        objetivo = None if eleccion == AUTOMATICO else {str(c): c for c in df.columns}[eleccion]
        repeticiones = a1.slider("Particiones de validación", 2, 10, 5, key=f"{clave}_rep")
        prueba = a2.slider("Prueba (%)", 10, 50, 25, 5, key=f"{clave}_test")
        a3.write("")
        if a3.button("Ejecutar agente", type="primary", width="stretch", key=f"{clave}_btn"):
            with st.spinner("El agente está eligiendo el objetivo y validando modelos..."):
                try:
                    st.session_state[clave] = AGENTES[tipo](
                        df, repeticiones=repeticiones, test_size=prueba / 100, objetivo=objetivo
                    ).ejecutar()
                except Exception as error:
                    st.error(f"No se pudo ejecutar el agente: {type(error).__name__}: {error}")

    agente = st.session_state.get(clave)
    if agente is not None:
        with st.expander("Informe del agente", expanded=True):
            _informe(agente, clave)


def _informe(agente, clave):
    if agente.modo == "llm":
        st.success(f"Análisis dirigido por el LLM ({agente.modelo}).")
        if agente.uso_llm:
            st.caption(f"Uso del LLM: {llm.resumen_uso(agente.uso_llm)}")
    else:
        st.warning(f"No se pudo usar el LLM; se usó el respaldo por reglas. Motivo: {agente.motivo_respaldo}")

    metrica = agente.metrica
    mejor = agente.validaciones[agente.mejor]
    resumen_mejor = mejor["resumen"][metrica]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Variable objetivo", str(agente.objetivo))
    m2.metric("Predictores", len(agente.predictores))
    m3.metric("Mejor modelo", mejor["nombre"])
    m4.metric(
        f"{metrica} (media)",
        f"{resumen_mejor['media']:.4f}",
        f"± {resumen_mejor['desv']:.4f}",
        delta_color="off",
    )
    st.caption(f"Mejor configuración: `{agente.mejor}`")
    if agente.eleccion_llm and agente.eleccion_llm != agente.mejor:
        st.warning(
            f"El LLM eligió `{agente.eleccion_llm}`, pero `{agente.mejor}` tiene mejor {metrica} medio. "
            "Revisa su justificación (por ejemplo, estabilidad entre particiones)."
        )
    if agente.excluidas:
        st.caption(f"Columnas excluidas de los predictores: {', '.join(map(str, agente.excluidas))}.")
    if agente.filas_sin_objetivo:
        st.caption(f"Se descartaron {agente.filas_sin_objetivo} filas sin valor en el objetivo.")

    st.markdown("**Decisiones del agente**")
    for item in agente.decisiones:
        st.markdown(f"- **{item['decision']}** — {item['justificacion']}")

    with st.expander("Plan de algoritmos"):
        for item in agente.plan:
            icono = "✅" if item.get("ejecutar") else "⏭️"
            st.markdown(f"{icono} **{item['paso']}** ({item['familia']}) — {item['razon']}")

    if agente.analisis_llm:
        st.markdown("**Interpretación del LLM**")
        st.write(agente.analisis_llm)

    st.markdown("**Autoevaluación del agente**")
    aviso = st.info if agente.supera_linea_base else st.error
    aviso("\n".join(f"- {texto}" for texto in agente.autoevaluacion))

    ranking = agente.ranking()
    st.markdown(f"**Ranking de modelos validados** (media en {agente.repeticiones} particiones)")
    columnas_num = [c for c in ranking.columns if c not in ("Modelo", "Algoritmo")]
    st.dataframe(
        ranking.style.format({c: "{:.4f}" for c in columnas_num}, na_rep="-").apply(
            lambda fila: [
                f"background-color: rgba(252, 238, 10, 0.18); color: {AMARILLO}; font-weight: 700"
                if fila["Modelo"] == agente.mejor else ""
            ] * len(fila),
            axis=1,
        ),
        width="stretch",
        hide_index=True,
    )

    g1, g2 = st.columns(2)
    orden = ranking["Modelo"].tolist()
    with g1:
        fig_barras = go.Figure(
            go.Bar(
                x=ranking[metrica],
                y=ranking["Modelo"],
                orientation="h",
                error_x=dict(type="data", array=ranking[f"Desv. {metrica}"], color=MAGENTA),
                marker=dict(color=[AMARILLO if m == agente.mejor else CIAN for m in orden]),
                text=ranking[metrica].map(lambda v: f"{v:.4f}"),
                textposition="inside",
            )
        )
        fig_barras.update_yaxes(categoryorder="array", categoryarray=orden[::-1])
        fig_barras.update_layout(title=f"{metrica} medio ± desviación", xaxis_title=metrica)
        grafico(fig_barras, key=f"{clave}_barras", height=max(320, 40 * len(orden)))
    with g2:
        fig_caja = px.box(
            agente.mejores_por_semilla(),
            x=metrica,
            y="Modelo",
            color="Modelo",
            points="all",
            category_orders={"Modelo": orden},
            color_discrete_sequence=SECUENCIA,
            title="Estabilidad entre particiones",
        )
        fig_caja.update_layout(showlegend=False)
        grafico(fig_caja, key=f"{clave}_caja", height=max(320, 40 * len(orden)))

    detalle = agente.resultado_mejor()
    if agente.tipo == "clasificacion":
        metricas = detalle["metricas"]
        etiquetas = metricas["confusion_matrix_labels"]
        matriz = np.asarray(metricas["confusion_matrix"])
        fig_cm = px.imshow(
            matriz,
            x=etiquetas,
            y=etiquetas,
            text_auto=True,
            color_continuous_scale=ESCALA_CONTINUA,
            labels={"x": "Predicción", "y": "Real", "color": "Casos"},
            title="Matriz de confusión del mejor modelo (partición 1)",
        )
        grafico(fig_cm, key=f"{clave}_cm", height=420)
    else:
        y_test, y_pred = detalle["y_test"], detalle["y_pred"]
        base = AgenteRegresion.linea_base_regresion(pd.Series(y_test))
        st.caption(
            f"Referencia: predecir siempre la media daría un RMSE ≈ {base:.4f} en la partición 1; "
            f"el mejor modelo obtiene {detalle['metricas']['RMSE']:.4f}."
        )
        limites = [float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))]
        fig_rp = px.scatter(
            x=y_test,
            y=y_pred,
            color=np.abs(y_test - y_pred),
            color_continuous_scale=ESCALA_CONTINUA,
            labels={"x": "Real", "y": "Predicción", "color": "Error abs."},
            title="Real vs. predicción del mejor modelo (partición 1)",
        )
        fig_rp.add_scatter(
            x=limites, y=limites, mode="lines", name="Predicción perfecta",
            line=dict(color=CIAN, dash="dash", width=2),
        )
        grafico(fig_rp, key=f"{clave}_rp", height=440)

    st.caption(
        f"Para reproducirlo en el flujo manual, elige '{agente.objetivo}' como variable a predecir "
        f"y el algoritmo «{mejor['nombre']}» con {mejor['parametros'] or 'sus parámetros por defecto'}."
    )
    with st.expander("Bitácora de razonamiento"):
        st.dataframe(pd.DataFrame(agente.bitacora), width="stretch")
    st.download_button(
        "Descargar ranking (CSV)",
        data=ranking.to_csv(index=False).encode("utf-8"),
        file_name=f"ranking_agente_{agente.tipo}.csv",
        mime="text/csv",
        key=f"{clave}_descargar",
    )
    st.download_button(
        "Descargar informe del agente (HTML)",
        data=agente.informe_html().encode("utf-8"),
        file_name=f"informe_agente_{agente.tipo}.html",
        mime="text/html",
        key=f"{clave}_descargar_html",
    )
