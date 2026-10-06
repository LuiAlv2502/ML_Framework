import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pages._shared import obtener_modelo

st.title("Estudio no supervisado")

model = obtener_modelo()
if model is None:
    st.stop()

tab_eda, tab_reduccion, tab_cluster = st.tabs(
    ["Exploración", "Reducción dimensional", "Clustering"]
)

with tab_eda:
    st.subheader("Vista previa")
    st.dataframe(model.primeras_filas(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Estadísticas descriptivas"):
            st.dataframe(model.estadisticas(), use_container_width=True)
    with col2:
        if st.checkbox("Nulos por columna"):
            st.dataframe(model.contar_nulos(), use_container_width=True)



    df_num = model.dataframe.dataframe.select_dtypes(include=np.number)
    if not df_num.empty:
        st.subheader("Matriz de correlación")
        corr = df_num.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribución por variable")
        columna = st.selectbox("Variable", df_num.columns.tolist())
        fig_hist = px.histogram(df_num, x=columna, nbins=30, marginal="box")
        st.plotly_chart(fig_hist, use_container_width=True)

with tab_reduccion:
    metodo = st.selectbox("Método", ["PCA", "t-SNE", "UMAP"])
    n = st.radio("Dimensiones", [2, 3], horizontal=True)

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
                if metodo == "PCA":
                    resultado = model.pca(n_components=n)
                    componentes = resultado["componentes"]
                    var_exp = resultado["varianza_explicada"]
                    st.success(f"Varianza acumulada: {resultado['metrica']:.2%}")

                    etiquetas = {
                        "PC1": f"PC1 ({var_exp[0]:.1%})",
                        "PC2": f"PC2 ({var_exp[1]:.1%})",
                    }
                    if n == 3:
                        etiquetas["PC3"] = f"PC3 ({var_exp[2]:.1%})"
                        fig = px.scatter_3d(
                            componentes,
                            x="PC1",
                            y="PC2",
                            z="PC3",
                            labels=etiquetas,
                        )
                    else:
                        fig = px.scatter(
                            componentes,
                            x="PC1",
                            y="PC2",
                            labels=etiquetas,
                        )
                    fig.update_traces(marker=dict(size=6, opacity=0.75))
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Varianza explicada")
                    fig_var = px.bar(
                        x=[f"PC{i + 1}" for i in range(len(var_exp))],
                        y=var_exp,
                        labels={"x": "Componente", "y": "Varianza"},
                    )
                    st.plotly_chart(fig_var, use_container_width=True)

                    st.subheader("Círculo de correlaciones")
                    cargas = resultado["cargas"]
                    fig_circ = go.Figure()
                    theta = np.linspace(0, 2 * np.pi, 100)
                    fig_circ.add_trace(
                        go.Scatter(
                            x=np.cos(theta),
                            y=np.sin(theta),
                            mode="lines",
                            line=dict(color="gray"),
                            showlegend=False,
                        )
                    )
                    for variable, carga in cargas.iterrows():
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
                            arrowcolor="#2563eb",
                        )
                        fig_circ.add_annotation(
                            x=carga["PC1"],
                            y=carga["PC2"],
                            text=variable,
                            showarrow=False,
                            yshift=10,
                        )
                    fig_circ.update_layout(
                        xaxis=dict(range=[-1.2, 1.2], title="PC1", zeroline=True),
                        yaxis=dict(
                            range=[-1.2, 1.2],
                            title="PC2",
                            zeroline=True,
                            scaleanchor="x",
                        ),
                        height=600,
                    )
                    st.plotly_chart(fig_circ, use_container_width=True)
                else:
                    if metodo == "t-SNE":
                        emb = model.tsne(n_components=n, perplexity=perplexity)
                    else:
                        emb = model.umap_embedding(
                            n_components=n,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                        )
                    if n == 3:
                        fig = px.scatter_3d(
                            emb,
                            x="Dim1",
                            y="Dim2",
                            z="Dim3",
                            title=metodo,
                        )
                    else:
                        fig = px.scatter(emb, x="Dim1", y="Dim2", title=metodo)
                    fig.update_traces(marker=dict(size=6, opacity=0.75))
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as error:
                st.error(f"Error: {error}")

with tab_cluster:
    algoritmo = st.selectbox("Algoritmo", ["K-Means", "HAC"])
    k = st.slider("Número de clústeres", 2, 10, 3)
    enlace = None
    if algoritmo == "HAC":
        enlace = st.selectbox("Enlace", ["ward", "average", "complete", "single"])

    col1, col2 = st.columns(2)
    ejecutar = col1.button("Agrupar", type="primary")
    codo = col2.button("Método del codo")

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
                fig = px.line(
                    x=list(range(1, k_max + 1)),
                    y=inercias,
                    markers=True,
                    labels={"x": "k", "y": "Inercia"},
                    title="Método del codo",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as error:
                st.error(f"Error: {error}")

    if ejecutar:
        with st.spinner(f"Ejecutando {algoritmo}..."):
            try:
                if algoritmo == "K-Means":
                    resultado = model.kmeans(n_clusters=k)
                else:
                    resultado = model.hac(n_clusters=k, linkage=enlace)

                st.success(f"Silhouette: {resultado['silhouette']:.4f}")
                df_resultado = resultado["resultado"]
                st.dataframe(df_resultado.head(), use_container_width=True)

                from sklearn.decomposition import PCA

                matriz, _ = model.dataframe.preparar_datos_numericos()
                proyeccion = PCA(n_components=2, random_state=42).fit_transform(matriz)
                df_plot = pd.DataFrame(proyeccion, columns=["PC1", "PC2"])
                df_plot["cluster"] = df_resultado["cluster"].astype(str).values

                fig = px.scatter(
                    df_plot,
                    x="PC1",
                    y="PC2",
                    color="cluster",
                    title=f"{algoritmo} proyectado con PCA",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_traces(
                    marker=dict(size=8, line=dict(width=0.5, color="white"))
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as error:
                st.error(f"Error: {error}")
