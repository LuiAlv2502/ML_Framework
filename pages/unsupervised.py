import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pages._shared import obtener_modelo
from pages._theme import (
    AMARILLO,
    CIAN,
    MAGENTA,
    REJILLA,
    ROSA_NEON,
    SECUENCIA,
    brillo,
    grafico,
)

st.title("Estudio no supervisado")

model = obtener_modelo()
if model is None:
    st.stop()

st.caption("La exploración de datos (EDA) está disponible en su propia página.")

df_original = model.dataframe.dataframe
columnas_color = [c for c in df_original.columns if df_original[c].nunique() <= 15]

tab_reduccion, tab_cluster = st.tabs(["Reducción dimensional", "Clustering"])


def _color_puntos(columna):
    if columna == "(ninguno)":
        return None
    return df_original[columna].astype(str).to_numpy()


def _dispersion(datos, x, y, z=None, color=None, etiquetas=None, titulo=None):
    if color is not None:
        datos = datos.assign(Grupo=color)
    argumentos = dict(
        x=x,
        y=y,
        labels=etiquetas,
        title=titulo,
        color="Grupo" if color is not None else None,
        color_discrete_sequence=SECUENCIA,
    )
    if z is not None:
        figura = px.scatter_3d(datos, z=z, **argumentos)
        figura.update_traces(marker=dict(size=4, opacity=0.85, line=dict(width=0)))
    else:
        figura = px.scatter(datos, **argumentos)
        figura.update_traces(marker=dict(size=9, opacity=0.85))
        if color is None:
            figura.update_traces(marker=dict(color=CIAN))
    return figura


with tab_reduccion:
    r1, r2, r3 = st.columns([2, 1, 2])
    metodo = r1.selectbox("Método", ["PCA", "t-SNE", "UMAP"])
    n = r2.radio("Dimensiones", [2, 3], horizontal=True)
    color_red = r3.selectbox("Colorear por", ["(ninguno)", *columnas_color], key="red_color")

    perplexity = None
    n_neighbors = None
    min_dist = None
    if metodo == "t-SNE":
        perplexity = st.slider("Perplexity", 5.0, 50.0, 30.0)
    elif metodo == "UMAP":
        n_neighbors = st.slider("Vecinos", 2, 50, 15)
        min_dist = st.slider("Distancia mínima", 0.0, 1.0, 0.1)

    if st.button("Ejecutar", type="primary"):
        with st.spinner(f"Calculando {metodo}..."):
            try:
                colores = _color_puntos(color_red)
                if metodo == "PCA":
                    resultado = model.pca(n_components=n)
                    componentes = resultado["componentes"]
                    var_exp = resultado["varianza_explicada"]
                    st.metric("Varianza acumulada", f"{resultado['metrica']:.2%}")

                    etiquetas = {
                        f"PC{i + 1}": f"PC{i + 1} ({var_exp[i]:.1%})" for i in range(n)
                    }
                    fig = _dispersion(
                        componentes,
                        "PC1",
                        "PC2",
                        z="PC3" if n == 3 else None,
                        color=colores,
                        etiquetas=etiquetas,
                        titulo="Plano principal",
                    )
                    grafico(fig, height=600)

                    c_var, c_circ = st.columns(2)
                    with c_var:
                        st.subheader("Varianza explicada")
                        nombres = [f"PC{i + 1}" for i in range(len(var_exp))]
                        acumulada = np.cumsum(var_exp)
                        fig_var = go.Figure()
                        fig_var.add_bar(
                            x=nombres,
                            y=var_exp,
                            name="Individual",
                            marker=dict(color=CIAN),
                            text=[f"{v:.1%}" for v in var_exp],
                            textposition="outside",
                        )
                        fig_var.add_scatter(
                            x=nombres,
                            y=acumulada,
                            name="Acumulada",
                            mode="lines+markers",
                            line=dict(color=AMARILLO, width=3),
                            marker=dict(size=10, symbol="diamond"),
                        )
                        fig_var.update_yaxes(tickformat=".0%", range=[0, 1.1])
                        grafico(brillo(fig_var), height=500)

                    with c_circ:
                        st.subheader("Círculo de correlaciones")
                        cargas = resultado["cargas"]
                        fig_circ = go.Figure()
                        theta = np.linspace(0, 2 * np.pi, 200)
                        fig_circ.add_trace(
                            go.Scatter(
                                x=np.cos(theta),
                                y=np.sin(theta),
                                mode="lines",
                                line=dict(color=ROSA_NEON, width=2, dash="dot"),
                                hoverinfo="skip",
                                showlegend=False,
                            )
                        )
                        for i, (variable, carga) in enumerate(cargas.iterrows()):
                            color = SECUENCIA[i % len(SECUENCIA)]
                            fig_circ.add_annotation(
                                x=carga["PC1"],
                                y=carga["PC2"],
                                ax=0,
                                ay=0,
                                xref="x",
                                yref="y",
                                axref="x",
                                ayref="y",
                                showarrow=True,
                                arrowhead=2,
                                arrowwidth=2,
                                arrowcolor=color,
                            )
                            fig_circ.add_annotation(
                                x=carga["PC1"],
                                y=carga["PC2"],
                                text=f"<b>{variable}</b>",
                                font=dict(color=color),
                                showarrow=False,
                                yshift=12,
                            )
                        fig_circ.update_layout(
                            xaxis=dict(range=[-1.2, 1.2], title=etiquetas["PC1"], zeroline=True),
                            yaxis=dict(
                                range=[-1.2, 1.2],
                                title=etiquetas["PC2"],
                                zeroline=True,
                                scaleanchor="x",
                            ),
                        )
                        grafico(fig_circ, height=500)
                else:
                    if metodo == "t-SNE":
                        emb = model.tsne(n_components=n, perplexity=perplexity)
                    else:
                        emb = model.umap_embedding(
                            n_components=n,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                        )
                    fig = _dispersion(
                        emb,
                        "Dim1",
                        "Dim2",
                        z="Dim3" if n == 3 else None,
                        color=colores,
                        titulo=metodo,
                    )
                    grafico(fig, height=600)
            except Exception as error:
                st.error(f"Error: {error}")

with tab_cluster:
    c1, c2, c3 = st.columns(3)
    algoritmo = c1.selectbox("Algoritmo", ["K-Means", "HAC"])
    k = c2.slider("Número de clústeres", 2, 10, 3)
    enlace = None
    if algoritmo == "HAC":
        enlace = c3.selectbox("Enlace", ["ward", "average", "complete", "single"])

    b1, b2, _ = st.columns([1, 1, 3])
    ejecutar = b1.button("Agrupar", type="primary", width="stretch")
    codo = b2.button("Método del codo", width="stretch")

    if codo:
        with st.spinner("Calculando..."):
            try:
                from sklearn.cluster import KMeans

                matriz, _ = model.dataframe.preparar_datos_numericos()
                k_max = max(k, 10)
                inercias = [
                    KMeans(n_clusters=i, n_init=10, random_state=42).fit(matriz).inertia_
                    for i in range(1, k_max + 1)
                ]
                fig = go.Figure(
                    go.Scatter(
                        x=list(range(1, k_max + 1)),
                        y=inercias,
                        mode="lines+markers",
                        line=dict(color=AMARILLO, width=3),
                        marker=dict(size=10, color=CIAN, symbol="diamond"),
                        name="Inercia",
                    )
                )
                fig.add_vline(x=k, line=dict(color=MAGENTA, dash="dash"))
                fig.update_layout(
                    title="Método del codo",
                    xaxis_title="k",
                    yaxis_title="Inercia",
                )
                fig.update_xaxes(dtick=1)
                grafico(brillo(fig), height=450)
            except Exception as error:
                st.error(f"Error: {error}")

    if ejecutar:
        with st.spinner(f"Ejecutando {algoritmo}..."):
            try:
                if algoritmo == "K-Means":
                    resultado = model.kmeans(n_clusters=k)
                else:
                    resultado = model.hac(n_clusters=k, linkage=enlace)

                df_resultado = resultado["resultado"]
                conteo = df_resultado["cluster"].value_counts().sort_index()

                m1, m2 = st.columns(2)
                m1.metric("Silhouette", f"{resultado['silhouette']:.4f}")
                m2.metric("Clústeres", int(conteo.size))

                from sklearn.decomposition import PCA

                matriz, _ = model.dataframe.preparar_datos_numericos()
                proyeccion = PCA(n_components=2, random_state=42).fit_transform(matriz)
                df_plot = pd.DataFrame(proyeccion, columns=["PC1", "PC2"])
                df_plot["cluster"] = df_resultado["cluster"].astype(str).values

                g1, g2 = st.columns([2, 1])
                fig = px.scatter(
                    df_plot,
                    x="PC1",
                    y="PC2",
                    color="cluster",
                    title=f"{algoritmo} proyectado con PCA",
                    color_discrete_sequence=SECUENCIA,
                    category_orders={"cluster": sorted(df_plot["cluster"].unique())},
                )
                fig.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=1, color="#0A0A12")))
                centros = df_plot.groupby("cluster")[["PC1", "PC2"]].mean()
                fig.add_scatter(
                    x=centros["PC1"],
                    y=centros["PC2"],
                    mode="markers+text",
                    text=centros.index,
                    textposition="top center",
                    marker=dict(symbol="x", size=16, color=AMARILLO, line=dict(width=2, color=AMARILLO)),
                    name="Centros",
                )
                with g1:
                    grafico(fig, height=520)

                fig_tam = px.bar(
                    x=conteo.index.astype(str),
                    y=conteo.to_numpy(),
                    color=conteo.index.astype(str),
                    color_discrete_sequence=SECUENCIA,
                    labels={"x": "Clúster", "y": "Observaciones"},
                    title="Tamaño de cada clúster",
                    text=conteo.to_numpy(),
                )
                fig_tam.update_layout(showlegend=False)
                with g2:
                    grafico(fig_tam, height=520)

                numericas = df_resultado.select_dtypes(include=np.number).columns.drop(
                    "cluster", errors="ignore"
                )
                if len(numericas) >= 2:
                    st.subheader("Perfil de los clústeres")
                    perfil = df_resultado.groupby("cluster")[numericas].mean()
                    rango = (perfil.max() - perfil.min()).replace(0, 1)
                    perfil_norm = (perfil - perfil.min()) / rango
                    fig_radar = go.Figure()
                    for i, (cluster, fila) in enumerate(perfil_norm.iterrows()):
                        color = SECUENCIA[i % len(SECUENCIA)]
                        fig_radar.add_trace(
                            go.Scatterpolar(
                                r=[*fila.to_numpy(), fila.iloc[0]],
                                theta=[*numericas, numericas[0]],
                                name=f"Clúster {cluster}",
                                line=dict(color=color, width=2),
                                fill="toself",
                                opacity=0.55,
                            )
                        )
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor="#12121F",
                            radialaxis=dict(range=[0, 1], gridcolor=REJILLA, showticklabels=False),
                            angularaxis=dict(gridcolor=REJILLA, linecolor=CIAN),
                        )
                    )
                    p1, p2 = st.columns([1, 1])
                    with p1:
                        grafico(fig_radar, height=480)
                    with p2:
                        st.dataframe(perfil.style.format("{:.2f}"), width="stretch")

                with st.expander("Datos con clúster asignado"):
                    st.dataframe(df_resultado, width="stretch", height=300)
            except Exception as error:
                st.error(f"Error: {error}")
