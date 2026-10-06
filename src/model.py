import pandas as pd
from unsupervised import ClusteringAnalysis, PCAAnalysis


class _ModelAnalysis(PCAAnalysis, ClusteringAnalysis):
    pass


class DataModel:
    """
    MODEL
    Encapsula el objeto de analisis no supervisado y expone
    operaciones que serán utilizadas por el Controller.
    No depende de ningún archivo en particular: se puede
    construir vacío y cargar cualquier CSV en tiempo de ejecución.
    """

    def __init__(self, ruta_csv=None):
        self.__datos_originales = pd.DataFrame()
        self.__dataframe = _ModelAnalysis(self.__datos_originales.copy())

        if ruta_csv:
            self.cargar_archivo(ruta_csv)

    @property
    def dataframe(self):
        return self.__dataframe

    def cargar_archivo(self, ruta_csv):
        self.__datos_originales = pd.read_csv(ruta_csv)
        self.__dataframe = _ModelAnalysis(
            self.__datos_originales.copy()
        )
        return self.__dataframe

    def cargar_dataframe(self, df):
        self.__datos_originales = df.copy()
        self.__dataframe = _ModelAnalysis(self.__datos_originales.copy())
        return self.__dataframe

    def restaurar(self):
        self.__dataframe = _ModelAnalysis(
            self.__datos_originales.copy()
        )
        return self.__dataframe

    def informacion(self):
        return str(self.__dataframe)

    def mostrar(self):
        return self.__dataframe.mostrar()

    def primeras_filas(self, n=10):
        return self.__dataframe.primeras_filas(n)

    def ultimas_filas(self, n=10):
        return self.__dataframe.ultimas_filas(n)

    def obtener_fila(self, indice):
        return self.__dataframe.obtener_fila(indice)

    def obtener_columna(self, nombre):
        return self.__dataframe.obtener_columna(nombre)

    def convertir_valor(self, columna, valor):
        serie = self.obtener_columna(columna)

        if pd.api.types.is_numeric_dtype(serie):
            try:
                return float(valor)
            except ValueError:
                pass

        return valor

    def dimensiones(self):
        return self.__dataframe.dimensiones()

    def nombres_columnas(self):
        return self.__dataframe.nombres_columnas()

    def tipos_datos(self):
        return self.__dataframe.tipos_datos()

    def contar_nulos(self):
        return self.__dataframe.contar_nulos()

    def eliminar_nulos(self):
        return self.__dataframe.eliminar_nulos()

    def reemplazar_nulos(self, valor=0):
        return self.__dataframe.reemplazar_nulos(valor)

    def ordenar(self, columna):
        return self.__dataframe.ordenar(columna)

    def filtrar(self, columna, operador, valor):
        return self.__dataframe.filtrar(columna, operador, valor)

    def agregar_columna(self, nombre, valor):
        return self.__dataframe.agregar_columna(nombre, valor)

    def eliminar_columna(self, nombre):
        return self.__dataframe.eliminar_columna(nombre)

    def estadisticas(self):
        return self.__dataframe.estadisticas()

    def correlacion(self):
        return self.__dataframe.correlacion()

    def visualizar_datos(self):
        return self.__dataframe.visualizar_datos()

    def frecuencias(self):
        return self.__dataframe.frecuencias()

    def detectar_duplicados(self):
        return self.__dataframe.detectar_duplicados()

    def eliminar_duplicados(self):
        return self.__dataframe.eliminar_duplicados()

    def convertir_tipos(self, conversiones):
        return self.__dataframe.convertir_tipos(conversiones)

    def distribucion_variables(self):
        return self.__dataframe.distribucion_variables()

    def histogramas(self):
        return self.__dataframe.histogramas()

    def boxplots(self):
        return self.__dataframe.boxplots()

    def scatterplots(self):
        return self.__dataframe.scatterplots()

    def mapa_calor(self):
        return self.__dataframe.mapa_calor()

    def detectar_outliers(self):
        return self.__dataframe.detectar_outliers()

    def imputar_nulos(self):
        return self.__dataframe.imputar_nulos()

    def normalizar(self):
        return self.__dataframe.normalizar()

    def estandarizar(self):
        return self.__dataframe.estandarizar()

    def seleccionar_variables(self, columnas):
        return self.__dataframe.seleccionar_variables(columnas)

    def guardar_csv(self, ruta="dataframe_guardado.csv"):
        return self.__dataframe.guardar_csv(ruta)

    def exportar_resultados(self, ruta="resultados_eda.csv"):
        return self.__dataframe.exportar_resultados(ruta)

    # ==========================================================
    # MÉTODOS NO SUPERVISADOS
    # ==========================================================

    def pca(self, n_components=2, whiten=False, svd_solver="auto"):
        return self.__dataframe.pca(n_components, whiten, svd_solver)

    def pca_grafico(self, resultado_pca, modo_3d=False):
        return self.__dataframe.pca_grafico(resultado_pca, modo_3d)

    def pca_circulo_correlaciones(self, resultado_pca):
        return self.__dataframe.pca_circulo_correlaciones(resultado_pca)

    def hac(self, n_clusters=3, linkage="ward", metric="euclidean"):
        return self.__dataframe.hac(n_clusters, linkage, metric)

    def hac_dendrograma(self, metodo="ward"):
        return self.__dataframe.hac_dendrograma(metodo)

    def kmeans(self, n_clusters=3, init="k-means++", n_init=10):
        return self.__dataframe.kmeans(n_clusters, init, n_init)

    def kmeans_codo(self, k_max=10):
        return self.__dataframe.kmeans_codo(k_max)

    def cluster_grafico(self, etiquetas, titulo="Clústeres", modo_3d=False):
        return self.__dataframe.cluster_grafico(etiquetas, titulo, modo_3d)

    def tsne(self, n_components=2, perplexity=30.0, learning_rate="auto"):
        return self.__dataframe.tsne(n_components, perplexity, learning_rate)

    def umap_embedding(self, n_components=2, n_neighbors=15, min_dist=0.1):
        return self.__dataframe.umap_embedding(
            n_components, n_neighbors, min_dist
        )

    def embedding_grafico(
        self,
        embedding,
        titulo="Embedding",
        etiquetas=None,
        nombre_etiqueta="Grupo",
        modo_3d=False,
    ):
        return self.__dataframe.embedding_grafico(
            embedding, titulo, etiquetas, nombre_etiqueta, modo_3d
        )
