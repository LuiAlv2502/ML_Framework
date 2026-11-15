import ipywidgets as widgets
from IPython.display import clear_output, display


class DataView:
    """
    VIEW
    Solo contiene componentes visuales.
    No conoce la lógica del DataFrame.
    """

    def __init__(self):
        self.titulo = widgets.HTML("<h2>MVC no supervisado - Jupyter</h2>")

        self.ruta_archivo = widgets.Text(
            description="Archivo:",
            placeholder="Ruta al CSV, ej. ../data/drug200.csv",
            layout=widgets.Layout(width="430px")
        )

        self.boton_cargar = widgets.Button(
            description="Cargar archivo",
            button_style="info"
        )

        self.metodo = widgets.Dropdown(
            options=[
                "Información",
                "Mostrar",
                "Primeras filas",
                "Últimas filas",
                "Obtener fila",
                "Obtener columna",
                "Dimensiones",
                "Nombres columnas",
                "Tipos de datos",
                "Contar nulos",
                "Eliminar nulos",
                "Reemplazar nulos",
                "Ordenar",
                "Filtrar",
                "Agregar columna",
                "Eliminar columna",
                "Estadísticas",
                "Correlación",
                "Visualizar datos",
                "Frecuencias",
                "Detectar duplicados",
                "Eliminar duplicados",
                "Convertir tipos",
                "Distribución variables",
                "Histogramas",
                "Boxplots",
                "Scatterplots",
                "Mapa de calor",
                "Detectar outliers",
                "Imputar nulos",
                "Normalizar",
                "Estandarizar",
                "Seleccionar variables",
                "Guardar CSV",
                "Exportar resultados",
                "PCA",
                "PCA (variación)",
                "HAC",
                "HAC Dendrograma",
                "K-Means",
                "K-Means Codo",
                "t-SNE",
                "UMAP",
            ],
            description="Método:",
            layout=widgets.Layout(width="430px")
        )

        self.columna = widgets.Dropdown(
            options=[],
            description="Columna:",
            layout=widgets.Layout(width="330px")
        )

        self.operador = widgets.Dropdown(
            options=["==", "!=", ">", ">=", "<", "<="],
            description="Operador:"
        )

        self.valor = widgets.Text(
            description="Valor:",
            placeholder="Valor a usar"
        )

        self.indice = widgets.IntText(
            value=0,
            description="Fila:"
        )

        self.columnas_multiple = widgets.SelectMultiple(
            options=[],
            description="Columnas:",
            layout=widgets.Layout(width="350px", height="130px")
        )

        self.parametro_entero = widgets.IntText(
            value=2,
            description="k / n / vecinos:"
        )

        self.parametro_decimal = widgets.FloatText(
            value=0.1,
            description="Perplexity/min_dist:"
        )

        self.metodo_enlace = widgets.Dropdown(
            options=["ward", "average", "complete", "single"],
            description="Enlace HAC:"
        )

        self.visualizacion_3d = widgets.Checkbox(
            value=False,
            description="Visualización 3D"
        )

        self.boton_ejecutar = widgets.Button(
            description="Ejecutar",
            button_style="primary"
        )

        self.boton_restaurar = widgets.Button(
            description="Restaurar datos"
        )

        self.salida = widgets.Output()
        self.campo_columna = widgets.Box([self.columna])
        self.campo_operador = widgets.Box([self.operador])
        self.campo_valor = widgets.Box([self.valor])
        self.campo_indice = widgets.Box([self.indice])
        self.campo_columnas_multiple = widgets.Box([self.columnas_multiple])
        self.campo_parametro_entero = widgets.Box([self.parametro_entero])
        self.campo_parametro_decimal = widgets.Box([self.parametro_decimal])
        self.campo_metodo_enlace = widgets.Box([self.metodo_enlace])
        self.campo_visualizacion_3d = widgets.Box([self.visualizacion_3d])
        self.controles_operacion = widgets.HBox([
            self.campo_columna,
            self.campo_operador,
            self.campo_valor,
            self.campo_indice,
        ])
        self.controles_parametros = widgets.HBox([
            self.campo_parametro_entero,
            self.campo_parametro_decimal,
            self.campo_metodo_enlace,
            self.campo_visualizacion_3d,
        ])

        self.metodo.observe(self._actualizar_controles, names="value")
        self._actualizar_controles()

    def actualizar_columnas(self, columnas):
        self.columna.options = columnas
        self.columnas_multiple.options = columnas

    def ruta_archivo_ingresada(self):
        return self.ruta_archivo.value.strip()

    def _mostrar_control(self, control, visible):
        control.layout.display = "" if visible else "none"

    def _actualizar_controles(self, _=None):
        metodo = self.metodo.value

        usa_columna = {
            "Obtener columna",
            "Ordenar",
            "Filtrar",
            "Eliminar columna",
            "Convertir tipos",
        }
        usa_valor = {"Reemplazar nulos", "Filtrar", "Agregar columna"}
        usa_parametro_entero = {
            "PCA",
            "PCA (variación)",
            "HAC",
            "K-Means",
            "K-Means Codo",
            "UMAP",
        }

        self._mostrar_control(self.campo_columna, metodo in usa_columna)
        self._mostrar_control(self.campo_operador, metodo == "Filtrar")
        self._mostrar_control(self.campo_valor, metodo in usa_valor)
        self._mostrar_control(self.campo_indice, metodo == "Obtener fila")
        self._mostrar_control(
            self.campo_columnas_multiple, metodo == "Seleccionar variables"
        )
        self._mostrar_control(
            self.campo_parametro_entero, metodo in usa_parametro_entero
        )
        self._mostrar_control(
            self.campo_parametro_decimal, metodo in {"t-SNE", "UMAP"}
        )
        self._mostrar_control(
            self.campo_metodo_enlace, metodo in {"HAC", "HAC Dendrograma"}
        )
        self._mostrar_control(
            self.campo_visualizacion_3d,
            metodo in {"PCA", "PCA (variación)", "HAC", "K-Means", "t-SNE", "UMAP"},
        )

        descripciones = {
            "PCA": "Componentes:",
            "PCA (variación)": "Componentes:",
            "HAC": "Clústeres:",
            "K-Means": "Clústeres:",
            "K-Means Codo": "k máximo:",
            "UMAP": "Vecinos:",
        }
        self.parametro_entero.description = descripciones.get(
            metodo, "Parámetro:"
        )

        self.parametro_decimal.description = (
            "Perplexity:" if metodo == "t-SNE" else "Min. distancia:"
        )

    def ejecutar_en_salida(self, accion):
        with self.salida:
            clear_output()
            try:
                accion()
            except Exception as error:
                print(f"ERROR: {type(error).__name__}: {error}")

    def mostrar_resultado(self, resultado):
        if isinstance(resultado, dict):
            for nombre, datos in resultado.items():
                print(f"\n--- {nombre} ---")
                display(datos)
            return

        if isinstance(resultado, str):
            print(resultado)
            return

        display(resultado)

    def mostrar(self):
        display(self.titulo)
        display(widgets.HBox([self.ruta_archivo, self.boton_cargar]))
        display(widgets.HBox([self.metodo, self.boton_ejecutar, self.boton_restaurar]))
        display(self.controles_operacion)
        display(self.controles_parametros)
        display(self.campo_columnas_multiple)
        display(self.salida)
