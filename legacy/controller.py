class DataController:
    """Coordina las acciones solicitadas entre DataModel y DataView."""

    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.actualizar_columnas(self.model.nombres_columnas())
        self.view.boton_ejecutar.on_click(self.ejecutar)
        self.view.boton_restaurar.on_click(self.restaurar)
        self.view.boton_cargar.on_click(self.cargar_archivo)

    def iniciar(self):
        self.view.mostrar()

    def cargar_archivo(self, _):
        ruta = self.view.ruta_archivo_ingresada()

        def accion():
            if not ruta:
                raise ValueError("Ingrese la ruta de un archivo CSV.")

            self.model.cargar_archivo(ruta)
            self.view.actualizar_columnas(self.model.nombres_columnas())
            self.view.mostrar_resultado(f"Archivo cargado: {ruta}")

        self.view.ejecutar_en_salida(accion)

    def restaurar(self, _):
        self.model.restaurar()
        self.view.actualizar_columnas(self.model.nombres_columnas())
        self.view.ejecutar_en_salida(
            lambda: self.view.mostrar_resultado(
                "Datos restaurados correctamente."
            )
        )

    def ejecutar(self, _):
        self.view.ejecutar_en_salida(self._ejecutar_accion)

    def _ejecutar_accion(self):
        metodo = self.view.metodo.value
        resultado = self._acciones()[metodo]()

        if metodo in {"Agregar columna", "Eliminar columna"}:
            self.view.actualizar_columnas(self.model.nombres_columnas())

        self.view.mostrar_resultado(resultado)

    def _ejecutar_pca(self, variacion=False):
        modo_3d = self.view.visualizacion_3d.value
        resultado = self.model.pca(
            n_components=max(3, self.view.parametro_entero.value) if modo_3d else self.view.parametro_entero.value,
            whiten=variacion,
            svd_solver="randomized" if variacion else "auto",
        )
        self.model.pca_grafico(resultado, modo_3d)
        self.model.pca_circulo_correlaciones(resultado)
        return resultado

    def _ejecutar_clustering(self, algoritmo):
        resultado = algoritmo()
        self.model.cluster_grafico(
            resultado["resultado"]["cluster"],
            "Visualización de clústeres",
            self.view.visualizacion_3d.value,
        )
        return resultado

    def _ejecutar_embedding(self, algoritmo, titulo):
        embedding = algoritmo()
        self.model.embedding_grafico(
            embedding, titulo, modo_3d=self.view.visualizacion_3d.value
        )
        return embedding

    def _ejecutar_umap(self):
        cantidad_clusters = max(2, self.view.parametro_entero.value)
        modo_3d = self.view.visualizacion_3d.value
        embedding = self.model.umap_embedding(
            n_components=3 if modo_3d else 2,
            n_neighbors=cantidad_clusters,
            min_dist=self.view.parametro_decimal.value,
        )
        resultado_kmeans = self.model.kmeans(n_clusters=cantidad_clusters)
        self.model.embedding_grafico(
            embedding,
            "UMAP coloreado por clústeres K-Means",
            resultado_kmeans["resultado"]["cluster"],
            "Clúster",
            modo_3d,
        )
        return embedding

    def _acciones(self):
        columna = self.view.columna.value
        valor = self.view.valor.value

        return {
            "Información": self.model.informacion,
            "Mostrar": self.model.mostrar,
            "Primeras filas": lambda: self.model.primeras_filas(10),
            "Últimas filas": lambda: self.model.ultimas_filas(10),
            "Obtener fila": lambda: self.model.obtener_fila(
                self.view.indice.value
            ),
            "Obtener columna": lambda: self.model.obtener_columna(columna),
            "Dimensiones": self.model.dimensiones,
            "Nombres columnas": self.model.nombres_columnas,
            "Tipos de datos": self.model.tipos_datos,
            "Contar nulos": self.model.contar_nulos,
            "Eliminar nulos": self.model.eliminar_nulos,
            "Reemplazar nulos": lambda: self.model.reemplazar_nulos(
                valor or 0
            ),
            "Ordenar": lambda: self.model.ordenar(columna),
            "Filtrar": lambda: self.model.filtrar(
                columna,
                self.view.operador.value,
                self.model.convertir_valor(columna, valor),
            ),
            "Agregar columna": lambda: self.model.agregar_columna(
                valor or "nueva_columna", 1
            ),
            "Eliminar columna": lambda: self.model.eliminar_columna(columna),
            "Estadísticas": self.model.estadisticas,
            "Correlación": self.model.correlacion,
            "Visualizar datos": self.model.visualizar_datos,
            "Frecuencias": self.model.frecuencias,
            "Detectar duplicados": self.model.detectar_duplicados,
            "Eliminar duplicados": self.model.eliminar_duplicados,
            "Convertir tipos": lambda: self.model.convertir_tipos(
                {columna: "float64"}
            ),
            "Distribución variables": self.model.distribucion_variables,
            "Histogramas": self.model.histogramas,
            "Boxplots": self.model.boxplots,
            "Scatterplots": self.model.scatterplots,
            "Mapa de calor": self.model.mapa_calor,
            "Detectar outliers": self.model.detectar_outliers,
            "Imputar nulos": self.model.imputar_nulos,
            "Normalizar": self.model.normalizar,
            "Estandarizar": self.model.estandarizar,
            "Seleccionar variables": lambda: self.model.seleccionar_variables(
                list(self.view.columnas_multiple.value)
            ),
            "Guardar CSV": lambda: self.model.guardar_csv(
                "dataframe_guardado.csv"
            ),
            "Exportar resultados": lambda: self.model.exportar_resultados(
                "resultados_eda.csv"
            ),
            "PCA": self._ejecutar_pca,
            "PCA (variación)": lambda: self._ejecutar_pca(variacion=True),
            "HAC": lambda: self._ejecutar_clustering(
                lambda: self.model.hac(
                    n_clusters=self.view.parametro_entero.value,
                    linkage=self.view.metodo_enlace.value,
                )
            ),
            "HAC Dendrograma": lambda: self.model.hac_dendrograma(
                metodo=self.view.metodo_enlace.value
            ),
            "K-Means": lambda: self._ejecutar_clustering(
                lambda: self.model.kmeans(
                    n_clusters=self.view.parametro_entero.value
                )
            ),
            "K-Means Codo": lambda: self.model.kmeans_codo(
                k_max=max(2, self.view.parametro_entero.value)
            ),
            "t-SNE": lambda: self._ejecutar_embedding(
                lambda: self.model.tsne(
                    n_components=3 if self.view.visualizacion_3d.value else 2,
                    perplexity=self.view.parametro_decimal.value or 30.0
                ),
                "t-SNE",
            ),
            "UMAP": self._ejecutar_umap,
        }