import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage as scipy_linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score

from ._base import UnsupervisedBase

try:
    import umap
except ImportError:
    umap = None


class ClusteringAnalysis(UnsupervisedBase):
    def hac(self, n_clusters=3, linkage="ward", metric="euclidean", columnas=None):
        matriz, _ = self.preparar_datos_numericos(columnas)
        modelo = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric if linkage != "ward" else "euclidean",
        )
        etiquetas = modelo.fit_predict(matriz)
        resultado = self.dataframe.copy()
        resultado["cluster"] = etiquetas
        enlaces = scipy_linkage(matriz, method=linkage)
        correlacion_cofenetica, _ = cophenet(enlaces, pdist(matriz))
        print("HAC - Correlación cofenética: " f"{correlacion_cofenetica:.4f}")

        return {
            "modelo": modelo,
            "resultado": resultado,
            "correlacion_cofenetica": correlacion_cofenetica,
            "silhouette": silhouette_score(matriz, etiquetas),
        }

    def hac_dendrograma(self, metodo="ward", columnas=None):
        matriz, _ = self.preparar_datos_numericos(columnas)
        enlaces = scipy_linkage(matriz, method=metodo)
        figura, eje = plt.subplots(figsize=(12, 6))
        dendrogram(enlaces, ax=eje)
        eje.set_title(f"Dendrograma HAC (método={metodo})")
        eje.set_xlabel("Índice de muestra")
        eje.set_ylabel("Distancia")
        plt.tight_layout()
        display(figura)
        plt.close(figura)
        return eje

    def kmeans(self, n_clusters=3, init="k-means++", n_init=10, columnas=None):
        matriz, _ = self.preparar_datos_numericos(columnas)
        modelo = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            random_state=42,
        )
        etiquetas = modelo.fit_predict(matriz)
        resultado = self.dataframe.copy()
        resultado["cluster"] = etiquetas
        metrica = silhouette_score(matriz, etiquetas)
        print(f"K-Means - Silhouette score: {metrica:.4f}")

        return {
            "modelo": modelo,
            "resultado": resultado,
            "inercia": modelo.inertia_,
            "silhouette": metrica,
        }

    def kmeans_codo(self, k_max=10, columnas=None):
        if k_max < 2:
            raise ValueError("k_max debe ser mayor o igual que 2.")

        matriz, _ = self.preparar_datos_numericos(columnas)
        inercias = []

        for k in range(1, k_max + 1):
            modelo = KMeans(n_clusters=k, n_init=10, random_state=42)
            modelo.fit(matriz)
            inercias.append(modelo.inertia_)

        figura, eje = plt.subplots(figsize=(8, 6))
        eje.plot(range(1, k_max + 1), inercias, marker="o")
        eje.set_xlabel("Número de clústeres (k)")
        eje.set_ylabel("Inercia")
        eje.set_title("Método del codo")
        plt.tight_layout()
        display(figura)
        plt.close(figura)
        return pd.Series(inercias, index=range(1, k_max + 1), name="inercia")

    def cluster_grafico(self, etiquetas, titulo="Clústeres", modo_3d=False):
        """Proyecta observaciones y las colorea por clúster."""
        matriz, _ = self.preparar_datos_numericos()
        dimensiones = 3 if modo_3d else 2
        proyeccion = PCA(n_components=dimensiones, random_state=42).fit_transform(matriz)

        if modo_3d:
            figura = plt.figure(figsize=(9, 7))
            eje = figura.add_subplot(111, projection="3d")
            puntos = eje.scatter(
                proyeccion[:, 0], proyeccion[:, 1], proyeccion[:, 2],
                c=etiquetas, cmap="tab10", alpha=0.8,
            )
            eje.set_zlabel("Componente principal 3")
        else:
            figura, eje = plt.subplots(figsize=(8, 6))
            puntos = eje.scatter(
                proyeccion[:, 0], proyeccion[:, 1],
                c=etiquetas, cmap="tab10", alpha=0.8,
            )
        eje.set_xlabel("Componente principal 1")
        eje.set_ylabel("Componente principal 2")
        eje.set_title(titulo + (" (3D)" if modo_3d else ""))
        figura.colorbar(puntos, ax=eje, label="Clúster")
        plt.tight_layout()
        display(figura)
        plt.close(figura)
        return eje

    def tsne(self, n_components=2, perplexity=30.0, learning_rate="auto", columnas=None):
        matriz, _ = self.preparar_datos_numericos(columnas)
        modelo = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=42,
            init="pca",
        )
        embedding = modelo.fit_transform(matriz)
        nombres = [f"Dim{i + 1}" for i in range(embedding.shape[1])]
        metrica = trustworthiness(matriz, embedding, n_neighbors=5)
        print(f"t-SNE - Trustworthiness: {metrica:.4f}")
        resultado = pd.DataFrame(embedding, columns=nombres)
        resultado.attrs["modelo"] = modelo
        resultado.attrs["trustworthiness"] = metrica
        return resultado

    def umap_embedding(self, n_components=2, n_neighbors=15, min_dist=0.1, columnas=None):
        if umap is None:
            raise ImportError("Instale umap-learn con: pip install umap-learn")

        matriz, _ = self.preparar_datos_numericos(columnas)
        modelo = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        embedding = modelo.fit_transform(matriz)
        nombres = [f"Dim{i + 1}" for i in range(embedding.shape[1])]
        metrica = trustworthiness(matriz, embedding, n_neighbors=5)
        print(f"UMAP - Trustworthiness: {metrica:.4f}")
        resultado = pd.DataFrame(embedding, columns=nombres)
        resultado.attrs["modelo"] = modelo
        resultado.attrs["trustworthiness"] = metrica
        return resultado

    def embedding_grafico(
        self,
        embedding,
        titulo="Embedding",
        etiquetas=None,
        nombre_etiqueta="Grupo",
        modo_3d=False,
    ):
        dimensiones_requeridas = 3 if modo_3d else 2
        if embedding.shape[1] < dimensiones_requeridas:
            raise ValueError(
                f"La visualización requiere {dimensiones_requeridas} dimensiones."
            )

        columnas = list(embedding.columns)
        if modo_3d:
            figura = plt.figure(figsize=(9, 7))
            eje = figura.add_subplot(111, projection="3d")
        else:
            figura, eje = plt.subplots(figsize=(9, 7))

        if etiquetas is None:
            coordenadas = [embedding[columnas[0]], embedding[columnas[1]]]
            if modo_3d:
                coordenadas.append(embedding[columnas[2]])
            eje.scatter(
                *coordenadas,
                color="#2563eb",
                alpha=0.75,
                edgecolors="white",
                linewidths=0.3,
            )
        else:
            etiquetas = pd.Series(etiquetas, index=embedding.index)
            paleta = plt.get_cmap("tab10")

            for indice, etiqueta in enumerate(sorted(etiquetas.unique())):
                mascara = etiquetas == etiqueta
                coordenadas = [
                    embedding.loc[mascara, columnas[0]],
                    embedding.loc[mascara, columnas[1]],
                ]
                if modo_3d:
                    coordenadas.append(embedding.loc[mascara, columnas[2]])
                eje.scatter(
                    *coordenadas,
                    color=paleta(indice % 10),
                    label=f"{nombre_etiqueta} {etiqueta}",
                    alpha=0.8,
                    edgecolors="white",
                    linewidths=0.3,
                )
            eje.legend(title=nombre_etiqueta, frameon=True)

        eje.set_xlabel(columnas[0])
        eje.set_ylabel(columnas[1])
        if modo_3d:
            eje.set_zlabel(columnas[2])
        eje.set_title(titulo + (" (3D)" if modo_3d else ""))
        eje.grid(alpha=0.2)
        plt.tight_layout()
        display(figura)
        plt.close(figura)
        return eje