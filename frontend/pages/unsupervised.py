import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.components.shared import ejecutar, obtener_datos, pie_de_ejecucion
from frontend.components.theme import (
    AMARILLO,
    CIAN,
    MAGENTA,
    REJILLA,
    ROSA_NEON,
    SECUENCIA,
    VERDE,
    brillo,
    grafico,
)

st.title("Agrupamiento")

ctrl = obtener_datos(recomendar_eda=True)
if ctrl is None:
    st.stop()

st.caption("Reducción de dimensionalidad (PCA, t-SNE, UMAP) y agrupamiento (K-Means, HAC). "
           "Todos los cálculos los ejecuta el agente de agrupamiento a través del coordinador.")

df_original = ctrl.active_data()
huella = ctrl.dataset_info()["hash"] + ctrl.dataset_info()["source"]
if st.session_state.get("no_supervisado_archivo_hash") != huella:
    for clave in ("agente_no_supervisado", "agr_reduccion", "agr_cluster", "agr_codo", "agr_dendro",
                  "agrupamiento_benchmark"):
        st.session_state.pop(clave, None)
    st.session_state["no_supervisado_archivo_hash"] = huella

columnas_color = [c for c in df_original.columns if df_original[c].nunique() <= 15]
with st.expander("Variables a usar", expanded=False):
    sugeridas_excluir = ctrl.detect_id_columns()
    columnas_uso = st.multiselect(
        "Columnas incluidas en el análisis",
        df_original.columns.tolist(),
        default=[c for c in df_original.columns if c not in sugeridas_excluir],
        help="Se excluyen por defecto las columnas que parecen identificadores.",
    )
    if sugeridas_excluir:
        st.caption(f"Identificadores detectados y excluidos por defecto: {sugeridas_excluir}")
if not columnas_uso:
    st.warning("Seleccione al menos una columna.")
    st.stop()
columnas = None if len(columnas_uso) == df_original.shape[1] else columnas_uso

# ---------------------------------------------------------------------------
# Agente inteligente
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Agente inteligente no supervisado**")
    st.caption(
        "El análisis manual está disponible aunque no ejecutes el agente. El agente selecciona "
        "análisis permitidos; PCA y los algoritmos de clustering se ejecutan en Python. "
        "Gemini recibe métricas agregadas, no las filas de los datos."
    )
    if st.button("Ejecutar agente", type="primary", key="btn_agente_no_supervisado"):
        resultado = ejecutar(lambda: ctrl.run_auto("clustering"), "El agente está analizando los datos...",
                             "No se pudo ejecutar el agente: ")
        if resultado is not None:
            st.session_state["agente_no_supervisado"] = resultado

tab_agente, tab_reduccion, tab_cluster, tab_bench = st.tabs(
    ["Informe del agente", "Reducción dimensional", "Clustering", "Comparación de configuraciones"]
)


def _color_puntos(columna):
    if columna == "(ninguno)":
        return None
    return df_original[columna].astype(str).to_numpy()


def _dispersion(datos, x, y, z=None, color=None, etiquetas=None, titulo=None):
    if color is not None:
        datos = datos.assign(Grupo=color)
    argumentos = dict(x=x, y=y, labels=etiquetas, title=titulo, color="Grupo" if color is not None else None,
                      color_discrete_sequence=SECUENCIA)
    if z is not None:
        figura = px.scatter_3d(datos, z=z, **argumentos)
        figura.update_traces(marker=dict(size=4, opacity=0.85, line=dict(width=0)))
    else:
        figura = px.scatter(datos, **argumentos)
        figura.update_traces(marker=dict(size=9, opacity=0.85))
        if color is None:
            figura.update_traces(marker=dict(color=CIAN))
    return figura


def _circulo_correlaciones(cargas, etiquetas):
    figura = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 200)
    figura.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines",
                                line=dict(color=ROSA_NEON, width=2, dash="dot"), hoverinfo="skip", showlegend=False))
    for i, (variable, carga) in enumerate(cargas.iterrows()):
        color = SECUENCIA[i % len(SECUENCIA)]
        figura.add_annotation(x=carga["PC1"], y=carga["PC2"], ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                              showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=color)
        figura.add_annotation(x=carga["PC1"], y=carga["PC2"], text=f"<b>{variable}</b>", font=dict(color=color),
                              showarrow=False, yshift=12)
    figura.update_layout(
        xaxis=dict(range=[-1.2, 1.2], title=etiquetas["PC1"], zeroline=True),
        yaxis=dict(range=[-1.2, 1.2], title=etiquetas["PC2"], zeroline=True, scaleanchor="x"),
    )
    return figura


# ---------------------------------------------------------------------------
# Reducción dimensional
# ---------------------------------------------------------------------------
with tab_reduccion:
    r1, r2, r3 = st.columns([2, 1, 2])
    metodo = r1.selectbox("Método", ["PCA", "t-SNE", "UMAP"])
    n = r2.radio("Dimensiones", [2, 3], horizontal=True)
    color_red = r3.selectbox("Colorear por", ["(ninguno)", *columnas_color], key="red_color")

    parametros = {"n_components": n}
    if metodo == "PCA":
        p1, p2 = st.columns(2)
        if p1.checkbox("Variación: whiten=True", help="Escala los componentes a varianza unitaria."):
            parametros["whiten"] = True
        parametros["svd_solver"] = p2.selectbox("Solucionador SVD", ["auto", "full", "randomized"])
    elif metodo == "t-SNE":
        parametros["perplexity"] = st.slider("Perplexity", 5.0, 50.0, 30.0)
    else:
        parametros["n_neighbors"] = st.slider("Vecinos", 2, 50, 15)
        parametros["min_dist"] = st.slider("Distancia mínima", 0.0, 1.0, 0.1)

    if st.button("Ejecutar", type="primary", key="btn_reduccion"):
        clave = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap"}[metodo]
        resultado = ejecutar(lambda: ctrl.run_clustering(clave, parametros, columns=columnas),
                             f"Calculando {metodo}...", "Error: ")
        if resultado is not None:
            st.session_state["agr_reduccion"] = {"metodo": metodo, "n": n, "resultado": resultado}

    guardado = st.session_state.get("agr_reduccion")
    if guardado is not None:
        resultado, metodo_g, n_g = guardado["resultado"], guardado["metodo"], guardado["n"]
        datos_r = resultado["result"]
        pie_de_ejecucion(resultado)
        colores = _color_puntos(color_red)
        if metodo_g == "PCA":
            componentes = datos_r["components"]
            var_exp = datos_r["explained_variance"]
            st.metric("Varianza acumulada", f"{datos_r['metrics']['varianza_acumulada']:.2%}")
            etiquetas = {f"PC{i + 1}": f"PC{i + 1} ({var_exp[i]:.1%})" for i in range(len(var_exp))}
            grafico(_dispersion(componentes, "PC1", "PC2", z="PC3" if n_g == 3 and "PC3" in componentes else None,
                                color=colores, etiquetas=etiquetas, titulo="Plano principal"), height=600)
            c_var, c_circ = st.columns(2)
            with c_var:
                st.subheader("Varianza explicada")
                nombres = [f"PC{i + 1}" for i in range(len(var_exp))]
                fig_var = go.Figure()
                fig_var.add_bar(x=nombres, y=var_exp, name="Individual", marker=dict(color=CIAN),
                                text=[f"{v:.1%}" for v in var_exp], textposition="outside")
                fig_var.add_scatter(x=nombres, y=np.cumsum(var_exp), name="Acumulada", mode="lines+markers",
                                    line=dict(color=AMARILLO, width=3), marker=dict(size=10, symbol="diamond"))
                fig_var.update_yaxes(tickformat=".0%", range=[0, 1.1])
                grafico(brillo(fig_var), height=500)
            with c_circ:
                st.subheader("Círculo de correlaciones")
                if datos_r["loadings"].shape[1] >= 2:
                    grafico(_circulo_correlaciones(datos_r["loadings"], etiquetas), height=500)
        else:
            st.metric("Trustworthiness", f"{datos_r['metrics']['trustworthiness']:.4f}")
            embedding = datos_r["embedding"]
            grafico(_dispersion(embedding, "Dim1", "Dim2", z="Dim3" if n_g == 3 else None, color=colores,
                                titulo=metodo_g), height=600)

# ---------------------------------------------------------------------------
# Informe del agente
# ---------------------------------------------------------------------------
with tab_agente:
    ejecucion = st.session_state.get("agente_no_supervisado")
    if ejecucion is None:
        st.info("Presione **Ejecutar agente** para que realice el análisis no supervisado automáticamente.")
    else:
        agente = ejecucion["result"]
        pie_de_ejecucion(ejecucion)
        if agente["mode"] == "llm":
            st.success(f"Análisis dirigido por el LLM ({agente['model']}).")
        else:
            st.warning(f"No se pudo usar el LLM; se usó el análisis por reglas. Motivo: {agente['fallback_reason']}")

        st.subheader("Decisiones del agente")
        for item in agente["plan"]:
            icono = "✅" if item.get("ejecutar") else "⏭️"
            st.markdown(f"{icono} **{item['paso']}** — {item['razon']}")
        if agente["analysis"]:
            st.subheader("Interpretación del LLM")
            st.write(agente["analysis"])
        if agente["self_evaluation"]:
            st.subheader("Autoevaluación del agente")
            st.info("\n".join(f"- {texto}" for texto in agente["self_evaluation"]))
        if agente["llm_usage_text"]:
            st.caption(f"Uso del LLM: {agente['llm_usage_text']}")

        resultados = agente["results"]
        if "pca" in resultados:
            resultado_pca = resultados["pca"]
            st.subheader("PCA")
            st.metric("Varianza explicada acumulada", f"{resultado_pca['varianza_acumulada']:.2%}")
            componentes = resultado_pca["componentes"]
            if {"PC1", "PC2"} <= set(componentes.columns):
                grafico(brillo(_dispersion(componentes, "PC1", "PC2", etiquetas={"PC1": "PC1", "PC2": "PC2"},
                                           titulo="PCA seleccionado por el agente")), height=500)
            st.markdown("**Variables más influyentes por componente**")
            st.dataframe(pd.DataFrame(resultado_pca["variables_mas_influyentes"]).fillna(""), width="stretch")
        if "codo" in resultados:
            inercias = resultados["codo"]["serie"]
            st.subheader("Método del codo")
            fig_codo = px.line(x=inercias.index, y=inercias.to_numpy(), markers=True,
                               labels={"x": "Número de grupos (k)", "y": "Inercia"},
                               title="Inercia por número de grupos")
            fig_codo.update_traces(line=dict(color=AMARILLO, width=3))
            grafico(brillo(fig_codo), height=420)
        for clave in ("kmeans", "hac"):
            if clave not in resultados:
                continue
            resultado_cluster = resultados[clave]
            st.subheader(resultado_cluster["algoritmo"])
            m1, m2 = st.columns(2)
            m1.metric("Silhouette", f"{resultado_cluster['silhouette']:.4f}")
            m2.metric("Grupos encontrados", resultado_cluster["grupos_obtenidos"])
            tamanos = pd.Series(resultado_cluster["tamanos"], name="Observaciones")
            fig_tamanos = px.bar(x=tamanos.index.astype(str), y=tamanos.to_numpy(),
                                 labels={"x": "Grupo", "y": "Observaciones"}, title="Tamaño de los grupos",
                                 color=tamanos.index.astype(str), color_discrete_sequence=SECUENCIA)
            fig_tamanos.update_layout(showlegend=False)
            grafico(brillo(fig_tamanos), height=360)
            if resultado_cluster["perfil_numerico"]:
                st.markdown("**Promedios numéricos por grupo**")
                st.dataframe(pd.DataFrame.from_dict(resultado_cluster["perfil_numerico"], orient="index"),
                             width="stretch")
            perfil_categorico = resultado_cluster["perfil_categorico"]
            if any(perfil_categorico.values()):
                st.markdown("**Categorías más frecuentes por grupo**")
                st.dataframe(pd.concat({g: pd.DataFrame.from_dict(d, orient="index")
                                        for g, d in perfil_categorico.items() if d}, names=["Grupo", "Variable"]),
                             width="stretch")
            with st.expander("Datos con grupo asignado"):
                st.dataframe(resultado_cluster["resultado"], width="stretch", height=300)
        if not resultados:
            st.info("El agente no ejecutó análisis. Revisa las razones en sus decisiones.")
        with st.expander("Bitácora de razonamiento"):
            st.dataframe(agente["log"], width="stretch")
        st.download_button("Descargar informe del agente (HTML)", data=agente["report_html"].encode("utf-8"),
                           file_name="informe_agente_no_supervisado.html", mime="text/html", width="stretch")

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
with tab_cluster:
    c1, c2, c3, c4 = st.columns(4)
    algoritmo = c1.selectbox("Algoritmo", ["K-Means", "HAC"])
    k = c2.slider("Número de clústeres", 2, 10, 3)
    parametros_c = {"n_clusters": k}
    if algoritmo == "HAC":
        parametros_c["linkage"] = c3.selectbox("Enlace", ["ward", "average", "complete", "single"])
        if parametros_c["linkage"] != "ward":
            parametros_c["metric"] = c4.selectbox("Distancia", ["euclidean", "manhattan", "cosine"])
    else:
        parametros_c["init"] = c3.selectbox("Inicialización", ["k-means++", "random"])
        parametros_c["n_init"] = c4.slider("Reinicios (n_init)", 1, 50, 10)

    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    agrupar = b1.button("Agrupar", type="primary", width="stretch")
    codo = b2.button("Método del codo", width="stretch")
    dendro = b3.button("Dendrograma", width="stretch", disabled=algoritmo != "HAC")

    if codo:
        resultado = ejecutar(lambda: ctrl.run_clustering("elbow", {"k_max": max(k, 10)}, columns=columnas),
                             "Calculando...", "Error: ")
        if resultado is not None:
            st.session_state["agr_codo"] = {"k": k, "resultado": resultado}
    if dendro:
        resultado = ejecutar(lambda: ctrl.run_clustering("dendrogram", {"method": parametros_c["linkage"]},
                                                         columns=columnas), "Construyendo dendrograma...", "Error: ")
        if resultado is not None:
            st.session_state["agr_dendro"] = resultado
    if agrupar:
        clave = "kmeans" if algoritmo == "K-Means" else "hac"
        resultado = ejecutar(lambda: ctrl.run_clustering(clave, parametros_c, columns=columnas),
                             f"Ejecutando {algoritmo}...", "Error: ")
        if resultado is not None:
            st.session_state["agr_cluster"] = resultado

    codo_guardado = st.session_state.get("agr_codo")
    if codo_guardado is not None:
        inercias = codo_guardado["resultado"]["result"]["inertias"]
        fig = go.Figure(go.Scatter(x=inercias["k"], y=inercias["inercia"], mode="lines+markers",
                                   line=dict(color=AMARILLO, width=3),
                                   marker=dict(size=10, color=CIAN, symbol="diamond"), name="Inercia"))
        fig.add_vline(x=codo_guardado["k"], line=dict(color=MAGENTA, dash="dash"))
        fig.update_layout(title="Método del codo", xaxis_title="k", yaxis_title="Inercia")
        fig.update_xaxes(dtick=1)
        grafico(brillo(fig), height=450)

    dendro_guardado = st.session_state.get("agr_dendro")
    if dendro_guardado is not None:
        st.image(dendro_guardado["result"]["image_png"], caption="Dendrograma HAC")

    cluster_guardado = st.session_state.get("agr_cluster")
    if cluster_guardado is not None:
        datos_c = cluster_guardado["result"]
        pie_de_ejecucion(cluster_guardado)
        metricas = datos_c["metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Silhouette", f"{metricas['silhouette']:.4f}", help="Mayor es mejor (−1 a 1).")
        m2.metric("Davies-Bouldin", f"{metricas['davies_bouldin']:.4f}", help="Menor es mejor.")
        m3.metric("Calinski-Harabasz", f"{metricas['calinski_harabasz']:.1f}", help="Mayor es mejor.")
        if "inercia" in metricas:
            m4.metric("Inercia", f"{metricas['inercia']:.1f}")
        else:
            m4.metric("Correlación cofenética", f"{metricas['correlacion_cofenetica']:.4f}")

        proyeccion = datos_c["projection"]
        g1, g2 = st.columns([2, 1])
        fig = px.scatter(proyeccion, x="PC1", y="PC2", color="cluster",
                         title=f"{datos_c['algorithm']} proyectado con PCA", color_discrete_sequence=SECUENCIA,
                         category_orders={"cluster": sorted(proyeccion["cluster"].unique())})
        fig.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=1, color="#0A0A12")))
        centros = datos_c["centers"]
        fig.add_scatter(x=centros["PC1"], y=centros["PC2"], mode="markers+text", text=centros.index,
                        textposition="top center", name="Centros",
                        marker=dict(symbol="x", size=16, color=AMARILLO, line=dict(width=2, color=AMARILLO)))
        with g1:
            grafico(fig, height=520)
        conteo = pd.Series(datos_c["sizes"])
        fig_tam = px.bar(x=conteo.index.astype(str), y=conteo.to_numpy(), color=conteo.index.astype(str),
                         color_discrete_sequence=SECUENCIA, labels={"x": "Clúster", "y": "Observaciones"},
                         title="Tamaño de cada clúster", text=conteo.to_numpy())
        fig_tam.update_layout(showlegend=False)
        with g2:
            grafico(fig_tam, height=520)

        perfil = datos_c["profile"]
        if perfil.shape[1] >= 2:
            st.subheader("Perfil de los clústeres")
            rango = (perfil.max() - perfil.min()).replace(0, 1)
            perfil_norm = (perfil - perfil.min()) / rango
            variables = list(perfil.columns)
            fig_radar = go.Figure()
            for i, (cluster, fila) in enumerate(perfil_norm.iterrows()):
                fig_radar.add_trace(go.Scatterpolar(r=[*fila.to_numpy(), fila.iloc[0]], theta=[*variables, variables[0]],
                                                    name=f"Clúster {cluster}", fill="toself", opacity=0.55,
                                                    line=dict(color=SECUENCIA[i % len(SECUENCIA)], width=2)))
            fig_radar.update_layout(polar=dict(bgcolor="#12121F",
                                               radialaxis=dict(range=[0, 1], gridcolor=REJILLA, showticklabels=False),
                                               angularaxis=dict(gridcolor=REJILLA, linecolor=CIAN)))
            p1, p2 = st.columns([1, 1])
            with p1:
                grafico(fig_radar, height=480)
            with p2:
                st.dataframe(perfil.style.format("{:.2f}"), width="stretch")
        with st.expander("Datos con clúster asignado"):
            st.dataframe(datos_c["data_with_cluster"], width="stretch", height=300)

# ---------------------------------------------------------------------------
# Comparación de configuraciones (benchmark)
# ---------------------------------------------------------------------------
with tab_bench:
    st.caption("Ejecuta, con la misma preparación de datos y semilla, PCA estándar vs. whiten, K-Means para "
               "un rango de k (y con init aleatorio) y HAC con los cuatro enlaces. Métricas: silhouette (↑), "
               "Davies-Bouldin (↓), Calinski-Harabasz (↑), cofenética (HAC), varianza y trustworthiness (PCA).")
    q1, q2, q3, q4 = st.columns(4)
    k_min = q1.number_input("k mínimo", 2, 10, 2)
    k_max = q2.number_input("k máximo", 2, 15, 6)
    hac_k = q3.number_input("k para HAC", 2, 10, 3)
    q4.write("")
    if q4.button("▶ Comparar", type="primary", width="stretch"):
        resultado = ejecutar(
            lambda: ctrl.run_benchmark("clustering", {"params": {"k_min": int(k_min), "k_max": int(k_max),
                                                                 "hac_k": int(hac_k)},
                                                      **({"columns": columnas} if columnas else {})}),
            "Comparando configuraciones...", "No se pudo ejecutar la comparación: ")
        if resultado is not None:
            st.session_state["agrupamiento_benchmark"] = resultado

    bench = st.session_state.get("agrupamiento_benchmark")
    if bench is None:
        st.info("Presione **Comparar** para ejecutar el benchmark de agrupamiento.")
    else:
        datos_b = bench["result"]
        pie_de_ejecucion(bench)
        st.success(f"Mejor configuración de agrupamiento: **{datos_b['best']}** · "
                   f"silhouette {datos_b['metrics']['silhouette']:.4f}")
        tabla = datos_b["table"]
        agrupamiento = tabla[tabla["Tipo"] == "Agrupamiento"].dropna(subset=["silhouette"])
        metrica = st.radio("Métrica", ["silhouette", "davies_bouldin", "calinski_harabasz"], horizontal=True)
        ordenada = agrupamiento.sort_values(metrica, ascending=metrica == "davies_bouldin")
        fig_b = px.bar(ordenada, x=metrica, y="Configuración", color="Algoritmo", orientation="h",
                       color_discrete_map={"K-Means": CIAN, "HAC": MAGENTA},
                       text=ordenada[metrica].map(lambda v: f"{v:.3f}"),
                       category_orders={"Configuración": ordenada["Configuración"].tolist()})
        grafico(fig_b, key="agr_bench", height=max(360, 40 * len(ordenada)))
        formatos = {c: "{:.4f}" for c in tabla.columns if tabla[c].dtype.kind == "f"}
        st.dataframe(tabla.style.format(formatos, na_rep="—")
                     .highlight_max(subset=["silhouette"], props=f"color: #000; background-color: {VERDE}"),
                     width="stretch", hide_index=True)
        st.download_button("Descargar comparación (CSV)", data=tabla.to_csv(index=False).encode("utf-8"),
                           file_name="benchmark_agrupamiento.csv", mime="text/csv")
