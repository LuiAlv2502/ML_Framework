import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents import AgenteEDA, llm
from model import DataModel
from pages._shared import (
    limpiar_resultados,
    obtener_modelo,
    restaurar_csv_original,
    seleccionar_modelo_activo,
)
from pages._theme import (
    AMARILLO,
    CIAN,
    ESCALA_DIVERGENTE,
    MAGENTA,
    REJILLA,
    ROJO,
    SECUENCIA,
    VERDE,
    grafico,
)

st.title("Exploración de datos (EDA)")

modelo_base = obtener_modelo()
if modelo_base is None:
    st.stop()

# ---------------------------------------------------------------------------
# Agente inteligente
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Agente inteligente de EDA**")
    st.caption(
        "El EDA manual está disponible aunque no ejecutes el agente. Ejecutarlo es opcional: "
        "si lo haces, su copia preparada será la que usen los modelos. El CSV original se conserva."
    )
    a1, a2 = st.columns([2, 1])
    datos_csv = st.session_state["model"].dataframe.dataframe
    columnas_base = datos_csv.columns.tolist()
    objetivo_previo = st.session_state.get("eda_objetivo")
    objetivo = a1.selectbox(
        "Variable objetivo (opcional)",
        ["(ninguna)", *columnas_base],
        index=columnas_base.index(objetivo_previo) + 1 if objetivo_previo in columnas_base else 0,
        help="Si se indica, el agente analiza si el problema es de clasificación o regresión.",
    )
    objetivo_actual = None if objetivo == "(ninguna)" else objetivo
    a2.write("")
    if a2.button("Ejecutar agente", type="primary", width="stretch"):
        with st.spinner("El agente está analizando los datos..."):
            agente_nuevo = AgenteEDA(datos_csv, objetivo=objetivo_actual).ejecutar()
            modelo_eda = DataModel()
            modelo_eda.cargar_dataframe(agente_nuevo.datos_para_modelos)
            limpiar_resultados()
            st.session_state["agente_eda"] = agente_nuevo
            st.session_state["agente_eda_html"] = agente_nuevo.informe_html()
            st.session_state["modelo_eda"] = modelo_eda
            st.session_state["eda_archivo_hash"] = st.session_state.get("archivo_hash")
            st.session_state["eda_objetivo"] = objetivo_actual
            st.session_state["modo_datos_activos"] = "agente"

agente = st.session_state.get("agente_eda")
if agente is not None and st.session_state.get("eda_objetivo") != objetivo_actual:
    st.info("Cambió la variable objetivo seleccionada; vuelve a ejecutar el agente si quieres actualizar su informe.")

model = seleccionar_modelo_activo(
    st.session_state.get("modo_datos_activos", "original"),
    st.session_state["model"],
    st.session_state.get("modelo_manual"),
    st.session_state.get("modelo_eda"),
    st.session_state.get("archivo_hash"),
    st.session_state.get("eda_archivo_hash"),
)
fuente_activa = st.session_state.get("modo_datos_activos", "original")
etiqueta_fuente = {
    "original": "CSV original",
    "manual": "cambios hechos manualmente",
    "agente": "copia preparada por el agente EDA",
}.get(fuente_activa, "CSV original")
st.caption(f"El EDA muestra los datos activos: **{etiqueta_fuente}**.")

df = model.dataframe.dataframe
numericas = df.select_dtypes(include=np.number).columns.tolist()
categoricas = [c for c in df.columns if c not in numericas]


def _aplicar(operacion, mensaje, guardar_como_manual=True):
    """Ejecuta una operación de limpieza del modelo e invalida resultados previos."""
    try:
        operacion()
    except (KeyError, TypeError, ValueError) as error:
        st.session_state["eda_mensaje"] = ("error", f"No se pudo aplicar: {error}")
    else:
        if guardar_como_manual:
            st.session_state["modelo_manual"] = model
            st.session_state["modo_datos_activos"] = "manual"
        limpiar_resultados()
        st.session_state["eda_mensaje"] = ("ok", mensaje)
    st.rerun()


tab_agente, tab_resumen, tab_calidad, tab_dist, tab_rel, tab_limpieza = st.tabs(
    [
        "Informe del agente",
        "Resumen",
        "Calidad de datos",
        "Distribuciones",
        "Relaciones",
        "Limpieza y transformación",
    ]
)

# ---------------------------------------------------------------------------
# Informe del agente
# ---------------------------------------------------------------------------
with tab_agente:
    if agente is None:
        st.info("Presione **Ejecutar agente** para que realice el EDA automáticamente.")
    else:
        st.subheader("Síntesis del agente")
        if agente.modo != "llm":
            st.warning(f"No se pudo usar el LLM; se usó el análisis por reglas. Motivo: {agente.motivo_respaldo}")
        elif agente.uso_llm:
            st.caption(f"Uso del LLM: {llm.resumen_uso(agente.uso_llm)}")
        st.write(agente.sintesis["resumen"])
        for recomendacion in agente.sintesis["recomendaciones"]:
            if recomendacion.startswith("[ALTA]"):
                st.error(recomendacion)
            elif recomendacion.startswith("[MEDIA]"):
                st.warning(recomendacion)
            else:
                st.success(recomendacion)
        st.caption(agente.sintesis["advertencia"])

        with st.expander("Plan del agente (qué decidió hacer y por qué)", expanded=True):
            for item in agente.plan:
                icono = "✅" if item["ejecutar"] else "⏭️"
                st.markdown(f"{icono} **{item['paso']}** — {item['razon']}")

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

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
with tab_resumen:
    filas, columnas = model.dimensiones()
    nulos_total = int(model.contar_nulos().sum())
    duplicados = int(df.duplicated().sum())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Filas", filas)
    m2.metric("Columnas", columnas)
    m3.metric("Numéricas", len(numericas))
    m4.metric("Valores nulos", nulos_total)
    m5.metric("Filas duplicadas", duplicados)

    c1, c2 = st.columns([1, 3])
    extremo = c1.radio("Mostrar", ["Primeras filas", "Últimas filas"])
    n_filas = c1.slider("Cantidad", 1, max(1, min(100, filas)), min(10, filas))
    with c2:
        vista = (
            model.primeras_filas(n_filas)
            if extremo == "Primeras filas"
            else model.ultimas_filas(n_filas)
        )
        st.dataframe(vista, width="stretch", height=320)

    st.subheader("Tipos de datos")
    tipos = pd.DataFrame(
        {
            "Columna": model.nombres_columnas(),
            "Tipo": model.tipos_datos().astype(str).to_numpy(),
            "Valores únicos": df.nunique().to_numpy(),
            "Nulos": model.contar_nulos().to_numpy(),
        }
    )
    t1, t2 = st.columns([3, 2])
    t1.dataframe(tipos, width="stretch", hide_index=True)
    conteo_tipos = tipos["Tipo"].value_counts().reset_index()
    conteo_tipos.columns = ["Tipo", "Columnas"]
    fig_tipos = px.pie(
        conteo_tipos,
        names="Tipo",
        values="Columnas",
        hole=0.6,
        title="Tipos de columnas",
    )
    fig_tipos.update_traces(marker=dict(line=dict(color="#0A0A12", width=2)))
    with t2:
        grafico(fig_tipos, key="eda_tipos", height=320)

# ---------------------------------------------------------------------------
# Calidad de datos
# ---------------------------------------------------------------------------
with tab_calidad:
    st.subheader("Valores nulos por columna")
    nulos = model.contar_nulos().rename("Nulos").reset_index()
    nulos.columns = ["Columna", "Nulos"]
    nulos["% nulos"] = nulos["Nulos"] / max(len(df), 1) * 100
    if nulos["Nulos"].sum() == 0:
        st.success("El conjunto de datos no tiene valores nulos.")
    fig_nulos = px.bar(
        nulos,
        x="Columna",
        y="% nulos",
        text=nulos["Nulos"],
        color="% nulos",
        color_continuous_scale=[[0, CIAN], [1, ROJO]],
        range_color=[0, max(nulos["% nulos"].max(), 1)],
    )
    fig_nulos.update_yaxes(title="% de nulos")
    grafico(fig_nulos, key="eda_nulos", height=360)

    st.subheader("Registros duplicados")
    filas_dup = model.detectar_duplicados()
    if filas_dup.empty:
        st.success("No hay filas duplicadas.")
    else:
        st.warning(f"{len(filas_dup)} filas involucradas en duplicados.")
        st.dataframe(filas_dup, width="stretch", height=240)

    st.subheader("Valores atípicos (rango intercuartílico)")
    if not numericas:
        st.info("No existen columnas numéricas para detectar valores atípicos.")
    else:
        outliers = model.detectar_outliers()
        resumen_out = pd.DataFrame(
            {
                "Columna": list(outliers),
                "Atípicos": [len(v) for v in outliers.values()],
            }
        )
        resumen_out["% atípicos"] = resumen_out["Atípicos"] / max(len(df), 1) * 100
        o1, o2 = st.columns([2, 1])
        fig_out = px.bar(
            resumen_out.sort_values("Atípicos"),
            x="Atípicos",
            y="Columna",
            orientation="h",
            text="Atípicos",
            color="% atípicos",
            color_continuous_scale=[[0, VERDE], [0.5, AMARILLO], [1, ROJO]],
        )
        with o1:
            grafico(fig_out, key="eda_outliers", height=max(300, 32 * len(numericas)))
        with o2:
            col_out = st.selectbox("Ver detalle de", list(outliers))
            st.dataframe(outliers[col_out], width="stretch", height=300)

# ---------------------------------------------------------------------------
# Distribuciones
# ---------------------------------------------------------------------------
with tab_dist:
    distribucion = model.distribucion_variables()
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Variables numéricas")
        if distribucion["numericas"].empty:
            st.info("Sin variables numéricas.")
        else:
            st.dataframe(
                distribucion["numericas"].style.format("{:.3f}"),
                width="stretch",
            )
    with d2:
        st.subheader("Variables categóricas")
        if distribucion["categoricas"].empty:
            st.info("Sin variables categóricas.")
        else:
            st.dataframe(distribucion["categoricas"], width="stretch")

    with st.expander("Estadísticas descriptivas completas"):
        st.dataframe(model.estadisticas(), width="stretch")

    if numericas:
        st.subheader("Histograma por variable")
        h1, h2, h3 = st.columns([2, 1, 1])
        columna = h1.selectbox("Variable", numericas, key="eda_hist_var")
        bins = h2.slider("Intervalos", 5, 100, 30, key="eda_hist_bins")
        color_por = h3.selectbox(
            "Colorear por",
            ["(ninguno)", *[c for c in categoricas if df[c].nunique() <= 15]],
            key="eda_hist_color",
        )
        fig_hist = px.histogram(
            df,
            x=columna,
            nbins=bins,
            marginal="box",
            color=None if color_por == "(ninguno)" else color_por,
            barmode="overlay",
            opacity=0.85,
        )
        grafico(fig_hist, key="eda_hist", height=450)

        st.subheader("Diagramas de caja")
        escala = st.radio(
            "Escala",
            ["Original", "Estandarizada (z)"],
            horizontal=True,
            help="Estandarizar permite comparar variables con rangos muy distintos.",
        )
        datos_box = df[numericas]
        if escala != "Original":
            datos_box = (datos_box - datos_box.mean()) / datos_box.std(ddof=0).replace(0, 1)
        largo = datos_box.melt(var_name="Variable", value_name="Valor")
        fig_box = px.box(largo, x="Variable", y="Valor", color="Variable", points="outliers")
        fig_box.update_layout(showlegend=False)
        grafico(fig_box, key="eda_box", height=450)

    st.subheader("Frecuencias")
    col_freq = st.selectbox("Columna", df.columns.tolist(), key="eda_freq")
    frecuencia = model.frecuencias()[col_freq]
    tabla_freq = frecuencia.rename("Frecuencia").reset_index()
    tabla_freq.columns = [col_freq, "Frecuencia"]
    tabla_freq[col_freq] = tabla_freq[col_freq].astype(str)
    tabla_freq["%"] = tabla_freq["Frecuencia"] / tabla_freq["Frecuencia"].sum() * 100
    f1, f2 = st.columns([1, 2])
    f1.dataframe(
        tabla_freq.style.format({"%": "{:.2f}"}),
        width="stretch",
        hide_index=True,
        height=380,
    )
    top = tabla_freq.head(30)
    fig_freq = px.bar(
        top,
        x=col_freq,
        y="Frecuencia",
        color="Frecuencia",
        color_continuous_scale=[[0, MAGENTA], [1, AMARILLO]],
        title="Top 30 valores" if len(tabla_freq) > 30 else None,
    )
    fig_freq.update_layout(coloraxis_showscale=False)
    with f2:
        grafico(fig_freq, key="eda_freq_graf", height=380)

# ---------------------------------------------------------------------------
# Relaciones
# ---------------------------------------------------------------------------
with tab_rel:
    if len(numericas) < 2:
        st.info("Se requieren al menos dos columnas numéricas para analizar relaciones.")
    else:
        st.subheader("Mapa de calor de correlaciones")
        corr = model.correlacion()
        fig_corr = go.Figure(
            go.Heatmap(
                z=corr.to_numpy(),
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                colorscale=ESCALA_DIVERGENTE,
                text=corr.round(2).to_numpy(),
                texttemplate="%{text}",
                hovertemplate="%{y} · %{x}<br>r = %{z:.3f}<extra></extra>",
                xgap=2,
                ygap=2,
            )
        )
        fig_corr.update_yaxes(autorange="reversed", showgrid=False)
        fig_corr.update_xaxes(showgrid=False)
        grafico(fig_corr, key="eda_corr", height=max(420, 45 * len(numericas)))

        pares = (
            corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
            .stack()
            .rename("r")
            .reset_index()
        )
        pares.columns = ["Variable 1", "Variable 2", "r"]
        pares["|r|"] = pares["r"].abs()
        with st.expander("Pares con mayor correlación"):
            st.dataframe(
                pares.sort_values("|r|", ascending=False).head(15).style.format(
                    {"r": "{:.3f}", "|r|": "{:.3f}"}
                ),
                width="stretch",
                hide_index=True,
            )

        st.subheader("Dispersión entre dos variables")
        opciones_color = ["(ninguno)", *df.columns.tolist()]
        s1, s2, s3 = st.columns(3)
        eje_x = s1.selectbox("Eje X", numericas, index=0, key="eda_sc_x")
        eje_y = s2.selectbox("Eje Y", numericas, index=1, key="eda_sc_y")
        color_sc = s3.selectbox("Color", opciones_color, key="eda_sc_color")
        fig_sc = px.scatter(
            df,
            x=eje_x,
            y=eje_y,
            color=None if color_sc == "(ninguno)" else color_sc,
            marginal_x="histogram",
            marginal_y="histogram",
            opacity=0.85,
        )
        fig_sc.update_traces(marker=dict(size=8), selector=dict(type="scatter"))
        grafico(fig_sc, key="eda_scatter", height=550)

        st.subheader("Matriz de dispersión")
        seleccion = st.multiselect(
            "Variables",
            numericas,
            default=numericas[: min(5, len(numericas))],
            key="eda_matriz_vars",
        )
        color_matriz = st.selectbox(
            "Color",
            ["(ninguno)", *[c for c in df.columns if df[c].nunique() <= 15]],
            key="eda_matriz_color",
        )
        if len(seleccion) >= 2:
            fig_matriz = px.scatter_matrix(
                df,
                dimensions=seleccion,
                color=None if color_matriz == "(ninguno)" else df[color_matriz].astype(str),
                color_discrete_sequence=SECUENCIA,
            )
            fig_matriz.update_traces(
                diagonal_visible=False,
                showupperhalf=False,
                marker=dict(size=4, opacity=0.7),
            )
            fig_matriz.update_xaxes(gridcolor=REJILLA)
            fig_matriz.update_yaxes(gridcolor=REJILLA)
            grafico(fig_matriz, key="eda_matriz", height=150 * len(seleccion) + 100)
        else:
            st.info("Seleccione al menos dos variables.")

# ---------------------------------------------------------------------------
# Limpieza y transformación
# ---------------------------------------------------------------------------
with tab_limpieza:
    mensaje = st.session_state.pop("eda_mensaje", None)
    if mensaje:
        (st.success if mensaje[0] == "ok" else st.error)(mensaje[1])

    st.caption(
        "Las operaciones manuales pasan a ser los datos usados en todas las páginas. "
        "Al aplicarlas se descartan particiones y modelos ya entrenados."
    )

    l1, l2 = st.columns(2)
    with l1.container(border=True):
        st.markdown("**Valores nulos**")
        if st.button("Imputar (mediana / moda)", width="stretch"):
            _aplicar(model.imputar_nulos, "Nulos imputados con mediana (numéricas) y moda (categóricas).")
        if st.button("Eliminar filas con nulos", width="stretch"):
            _aplicar(model.eliminar_nulos, "Filas con nulos eliminadas.")
        r1, r2 = st.columns([2, 1])
        valor_relleno = r1.text_input("Reemplazar nulos por", value="0")
        if r2.button("Reemplazar", width="stretch"):
            try:
                valor = float(valor_relleno)
            except ValueError:
                valor = valor_relleno
            _aplicar(lambda: model.reemplazar_nulos(valor), f"Nulos reemplazados por {valor!r}.")

    with l2.container(border=True):
        st.markdown("**Duplicados y escalado**")
        if st.button("Eliminar duplicados", width="stretch"):
            _aplicar(model.eliminar_duplicados, "Filas duplicadas eliminadas.")
        if st.button("Normalizar numéricas [0, 1]", width="stretch"):
            _aplicar(model.normalizar, "Variables numéricas normalizadas entre 0 y 1.")
        if st.button("Estandarizar numéricas (z)", width="stretch"):
            _aplicar(model.estandarizar, "Variables numéricas estandarizadas (media 0, desviación 1).")

    l3, l4 = st.columns(2)
    with l3.container(border=True):
        st.markdown("**Columnas**")
        col_eliminar = st.selectbox("Eliminar columna", df.columns.tolist(), key="eda_col_eliminar")
        if st.button("Eliminar columna", width="stretch"):
            _aplicar(lambda: model.eliminar_columna(col_eliminar), f"Columna '{col_eliminar}' eliminada.")

        c_col, c_tipo = st.columns(2)
        col_convertir = c_col.selectbox("Convertir columna", df.columns.tolist(), key="eda_col_tipo")
        tipo_nuevo = c_tipo.selectbox("A tipo", ["float64", "int64", "str", "category"])
        if st.button("Convertir tipo", width="stretch"):
            _aplicar(
                lambda: model.convertir_tipos({col_convertir: tipo_nuevo}),
                f"Columna '{col_convertir}' convertida a {tipo_nuevo}.",
            )

    with l4.container(border=True):
        st.markdown("**Ordenar y filtrar**")
        col_ordenar = st.selectbox("Ordenar por", df.columns.tolist(), key="eda_col_ordenar")
        if st.button("Ordenar ascendente", width="stretch"):
            _aplicar(lambda: model.ordenar(col_ordenar), f"Datos ordenados por '{col_ordenar}'.")

        f1, f2, f3 = st.columns([2, 1, 2])
        col_filtro = f1.selectbox("Filtrar columna", df.columns.tolist(), key="eda_col_filtro")
        operador = f2.selectbox("Op.", ["==", "!=", ">", ">=", "<", "<="])
        valor_filtro = f3.text_input("Valor", key="eda_valor_filtro")
        if valor_filtro:
            try:
                filtrado = model.filtrar(
                    col_filtro, operador, model.convertir_valor(col_filtro, valor_filtro)
                )
                st.caption(f"{len(filtrado)} filas cumplen la condición (vista previa, no modifica los datos).")
                st.dataframe(filtrado.head(50), width="stretch", height=200)
            except (KeyError, TypeError, ValueError) as error:
                st.error(f"No se pudo filtrar: {error}")

    st.divider()
    b1, b2 = st.columns(2)
    if b1.button("Restaurar datos originales", type="primary", width="stretch"):
        _aplicar(
            restaurar_csv_original,
            "Datos restaurados al CSV original.",
            guardar_como_manual=False,
        )
    b2.download_button(
        "Descargar datos actuales (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="datos_eda.csv",
        mime="text/csv",
        width="stretch",
    )
