import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataFrame:
    """
    Clase base para EDA, limpieza y preprocesamiento de un DataFrame.
    """

    def __init__(self, dataframe=None):
        if dataframe is None:
            dataframe = pd.DataFrame()

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("El atributo dataframe debe ser un pandas.DataFrame.")

        self.__dataframe = dataframe.copy()

    # ==========================================================
    # Propiedad
    # ==========================================================

    @property
    def dataframe(self):
        return self.__dataframe

    @dataframe.setter
    def dataframe(self, nuevo_dataframe):
        if not isinstance(nuevo_dataframe, pd.DataFrame):
            raise TypeError("El nuevo valor debe ser un pandas.DataFrame.")

        self.__dataframe = nuevo_dataframe.copy()

    # ==========================================================
    # toString
    # ==========================================================

    def __str__(self):
        return (
            "Clase DataFrame\n"
            f"Filas: {self.__dataframe.shape[0]}\n"
            f"Columnas: {self.__dataframe.shape[1]}\n"
            f"Nombres: {list(self.__dataframe.columns)}"
        )

    # ==========================================================
    # 20 EJERCICIOS PROPUESTOS
    # ==========================================================
    def cargar_csv(self, ruta, separador=",", encoding="utf-8"):
        """
        Carga un archivo CSV y lo asigna al DataFrame.
        """
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No existe el archivo '{ruta}'.")

        self.__dataframe = pd.read_csv(ruta, sep=separador, encoding=encoding)
        return self.__dataframe


    def mostrar(self):
        """Retorna el DataFrame completo."""
        return self.__dataframe

    def primeras_filas(self, n=5):
        """Retorna las primeras n filas."""
        if n < 0:
            raise ValueError("n debe ser mayor o igual que cero.")
        return self.__dataframe.head(n)

    def ultimas_filas(self, n=5):
        """Retorna las últimas n filas."""
        if n < 0:
            raise ValueError("n debe ser mayor o igual que cero.")
        return self.__dataframe.tail(n)

    def recorrer_filas(self):
        """Recorre e imprime todas las filas del DataFrame."""
        for indice, fila in self.__dataframe.iterrows():
            print(f"Índice: {indice}")
            print(fila)
            print("-" * 40)

    def recorrer_columnas(self):
        """Recorre e imprime todas las columnas del DataFrame."""
        for columna in self.__dataframe.columns:
            print(f"Columna: {columna}")
            print(self.__dataframe[columna])
            print("-" * 40)

    def obtener_fila(self, indice):
        """Obtiene una fila mediante su posición numérica."""
        if not isinstance(indice, int):
            raise TypeError("El índice debe ser un número entero.")

        if indice < 0 or indice >= len(self.__dataframe):
            raise IndexError("El índice está fuera del rango del DataFrame.")

        return self.__dataframe.iloc[indice]

    def obtener_columna(self, nombre):
        """Retorna una columna por su nombre."""
        if nombre not in self.__dataframe.columns:
            raise KeyError(f"La columna '{nombre}' no existe.")

        return self.__dataframe[nombre]

    def dimensiones(self):
        """Retorna una tupla con la cantidad de filas y columnas."""
        return self.__dataframe.shape

    def nombres_columnas(self):
        """Retorna una lista con los nombres de las columnas."""
        return list(self.__dataframe.columns)

    def tipos_datos(self):
        """Retorna los tipos de datos de cada columna."""
        return self.__dataframe.dtypes

    def contar_nulos(self):
        """Retorna la cantidad de valores nulos por columna."""
        return self.__dataframe.isnull().sum()

    def eliminar_nulos(self):
        """Elimina las filas que contienen valores nulos."""
        self.__dataframe.dropna(inplace=True)
        self.__dataframe.reset_index(drop=True, inplace=True)
        return self.__dataframe

    def reemplazar_nulos(self, valor=0):
        """Reemplaza los valores nulos por el valor indicado."""
        self.__dataframe.fillna(valor, inplace=True)
        return self.__dataframe

    def ordenar(self, columna):
        """Ordena el DataFrame de forma ascendente por una columna."""
        if columna not in self.__dataframe.columns:
            raise KeyError(f"La columna '{columna}' no existe.")

        self.__dataframe.sort_values(
            by=columna,
            ascending=True,
            inplace=True
        )
        self.__dataframe.reset_index(drop=True, inplace=True)

        return self.__dataframe

    def filtrar(self, columna, operador, valor):
        """
        Filtra registros mediante un operador de comparación.

        Operadores permitidos:
        ==, !=, >, >=, <, <=
        """
        if columna not in self.__dataframe.columns:
            raise KeyError(f"La columna '{columna}' no existe.")

        operadores = {
            "==": self.__dataframe[columna] == valor,
            "!=": self.__dataframe[columna] != valor,
            ">": self.__dataframe[columna] > valor,
            ">=": self.__dataframe[columna] >= valor,
            "<": self.__dataframe[columna] < valor,
            "<=": self.__dataframe[columna] <= valor
        }

        if operador not in operadores:
            raise ValueError(
                "Operador inválido. Use ==, !=, >, >=, < o <=."
            )

        return self.__dataframe[operadores[operador]]

    def agregar_columna(self, nombre, valores):
        """Agrega una nueva columna al DataFrame."""
        if nombre in self.__dataframe.columns:
            raise ValueError(f"La columna '{nombre}' ya existe.")

        if np.isscalar(valores):
            self.__dataframe[nombre] = valores
        else:
            if len(valores) != len(self.__dataframe):
                raise ValueError(
                    "La cantidad de valores debe coincidir con las filas."
                )
            self.__dataframe[nombre] = valores

        return self.__dataframe

    def eliminar_columna(self, nombre):
        """Elimina una columna del DataFrame."""
        if nombre not in self.__dataframe.columns:
            raise KeyError(f"La columna '{nombre}' no existe.")

        self.__dataframe.drop(columns=[nombre], inplace=True)
        return self.__dataframe

    def estadisticas(self):
        """Calcula estadísticas descriptivas."""
        return self.__dataframe.describe(include="all")

    def correlacion(self):
        """Calcula la matriz de correlación de variables numéricas."""
        numericas = self.__dataframe.select_dtypes(include=np.number)

        if numericas.empty:
            raise ValueError("No existen columnas numéricas.")

        return numericas.corr()

    def guardar_csv(self, ruta="dataframe.csv"):
        """Guarda el DataFrame en un archivo CSV."""
        self.__dataframe.to_csv(ruta, index=False)
        return ruta

    # ==========================================================
    # MÉTODOS DE UN EDA COMPLETO
    # ==========================================================

    def visualizar_datos(self):
        """Muestra información general y las primeras filas."""
        print("Primeras filas:")
        print(self.__dataframe.head())
        print("\nDimensiones:")
        print(self.__dataframe.shape)
        print("\nTipos de datos:")
        print(self.__dataframe.dtypes)

        return self.__dataframe.head()

    def frecuencias(self):
        """Calcula frecuencias absolutas por columna."""
        resultado = {}

        for columna in self.__dataframe.columns:
            resultado[columna] = self.__dataframe[
                columna
            ].value_counts(dropna=False)

        return resultado

    def detectar_duplicados(self):
        """Retorna las filas duplicadas."""
        return self.__dataframe[self.__dataframe.duplicated(keep=False)]

    def eliminar_duplicados(self):
        """Elimina registros duplicados."""
        self.__dataframe.drop_duplicates(inplace=True)
        self.__dataframe.reset_index(drop=True, inplace=True)
        return self.__dataframe

    def convertir_tipos(self, conversiones):
        """
        Convierte tipos de datos.

        Ejemplo:
        {"edad": "int64", "fecha": "datetime64[ns]"}
        """
        if not isinstance(conversiones, dict):
            raise TypeError("conversiones debe ser un diccionario.")

        for columna in conversiones:
            if columna not in self.__dataframe.columns:
                raise KeyError(f"La columna '{columna}' no existe.")

        self.__dataframe = self.__dataframe.astype(conversiones)
        return self.__dataframe

    def distribucion_variables(self):
        """Genera estadísticas descriptivas por tipo de variable."""
        resultado = {
            "numericas": pd.DataFrame(),
            "categoricas": pd.DataFrame()
        }

        numericas = self.__dataframe.select_dtypes(include=np.number)
        categoricas = self.__dataframe.select_dtypes(exclude=np.number)

        if not numericas.empty:
            resultado["numericas"] = numericas.describe().T

        if not categoricas.empty:
            resultado["categoricas"] = categoricas.describe().T

        return resultado

    def histogramas(self):
        """Genera histogramas para las variables numéricas."""
        numericas = self.__dataframe.select_dtypes(include=np.number)

        if numericas.empty:
            raise ValueError("No existen columnas numéricas.")

        ejes = numericas.hist(figsize=(12, 8))
        plt.tight_layout()
        plt.show()

        return ejes

    def boxplots(self):
        """Genera diagramas de caja para variables numéricas."""
        numericas = self.__dataframe.select_dtypes(include=np.number)

        if numericas.empty:
            raise ValueError("No existen columnas numéricas.")

        eje = numericas.plot(
            kind="box",
            figsize=(12, 6),
            rot=45
        )
        plt.tight_layout()
        plt.show()

        return eje

    def scatterplots(self):
        """Genera una matriz de gráficos de dispersión."""
        numericas = self.__dataframe.select_dtypes(include=np.number)

        if numericas.shape[1] < 2:
            raise ValueError(
                "Se requieren al menos dos columnas numéricas."
            )

        ejes = pd.plotting.scatter_matrix(
            numericas,
            figsize=(12, 12),
            diagonal="hist"
        )
        plt.tight_layout()
        plt.show()

        return ejes

    def mapa_calor(self):
        """Genera un mapa de calor de correlaciones."""
        matriz = self.correlacion()

        figura, eje = plt.subplots(
            figsize=(10, 8)
        )

        imagen = eje.imshow(
            matriz.values,
            aspect="auto",
            vmin=-1,
            vmax=1
        )

        eje.set_xticks(range(len(matriz.columns)))
        eje.set_yticks(range(len(matriz.index)))
        eje.set_xticklabels(
            matriz.columns,
            rotation=45,
            ha="right"
        )
        eje.set_yticklabels(matriz.index)

        for fila in range(len(matriz.index)):
            for columna in range(len(matriz.columns)):
                eje.text(
                    columna,
                    fila,
                    f"{matriz.iloc[fila, columna]:.2f}",
                    ha="center",
                    va="center"
                )

        figura.colorbar(imagen, ax=eje)
        eje.set_title("Mapa de calor de correlaciones")
        plt.tight_layout()
        plt.show()

        return matriz

    def detectar_outliers(self):
        """
        Detecta valores atípicos mediante el rango intercuartílico.
        """
        numericas = self.__dataframe.select_dtypes(include=np.number)

        if numericas.empty:
            raise ValueError("No existen columnas numéricas.")

        resultado = {}

        for columna in numericas.columns:
            q1 = numericas[columna].quantile(0.25)
            q3 = numericas[columna].quantile(0.75)
            rango = q3 - q1

            limite_inferior = q1 - 1.5 * rango
            limite_superior = q3 + 1.5 * rango

            mascara = (
                (numericas[columna] < limite_inferior)
                | (numericas[columna] > limite_superior)
            )

            resultado[columna] = self.__dataframe.loc[mascara]

        return resultado

    def imputar_nulos(self):
        """
        Imputa nulos con la mediana en variables numéricas
        y con la moda en variables categóricas.
        """
        for columna in self.__dataframe.columns:
            if not self.__dataframe[columna].isnull().any():
                continue

            if pd.api.types.is_numeric_dtype(
                self.__dataframe[columna]
            ):
                valor = self.__dataframe[columna].median()
            else:
                moda = self.__dataframe[columna].mode()
                valor = moda.iloc[0] if not moda.empty else "Desconocido"

            self.__dataframe[columna] = (
                self.__dataframe[columna].fillna(valor)
            )

        return self.__dataframe

    def normalizar(self):
        """
        Normaliza variables numéricas entre 0 y 1.
        """
        numericas = self.__dataframe.select_dtypes(
            include=np.number
        ).columns

        if len(numericas) == 0:
            raise ValueError("No existen columnas numéricas.")

        for columna in numericas:
            minimo = self.__dataframe[columna].min()
            maximo = self.__dataframe[columna].max()

            if maximo == minimo:
                self.__dataframe[columna] = 0.0
            else:
                self.__dataframe[columna] = (
                    self.__dataframe[columna] - minimo
                ) / (maximo - minimo)

        return self.__dataframe

    def estandarizar(self):
        """
        Estandariza variables numéricas con media 0
        y desviación estándar 1.
        """
        numericas = self.__dataframe.select_dtypes(
            include=np.number
        ).columns

        if len(numericas) == 0:
            raise ValueError("No existen columnas numéricas.")

        for columna in numericas:
            media = self.__dataframe[columna].mean()
            desviacion = self.__dataframe[columna].std()

            if desviacion == 0 or pd.isna(desviacion):
                self.__dataframe[columna] = 0.0
            else:
                self.__dataframe[columna] = (
                    self.__dataframe[columna] - media
                ) / desviacion

        return self.__dataframe

    def seleccionar_variables(self, columnas):
        """Retorna un DataFrame con las columnas seleccionadas."""
        if not isinstance(columnas, list):
            raise TypeError("columnas debe ser una lista.")

        inexistentes = [
            columna
            for columna in columnas
            if columna not in self.__dataframe.columns
        ]

        if inexistentes:
            raise KeyError(
                f"Columnas inexistentes: {inexistentes}"
            )

        return self.__dataframe[columnas]

    def exportar_resultados(self, ruta="resultados_eda.csv"):
        """Exporta el DataFrame resultante del EDA."""
        self.__dataframe.to_csv(ruta, index=False)
        return ruta

if __name__ == "__main__":
    datos = {
        "Nombre": ["Ana", "Luis", "Marta", "Luis"],
        "Edad": [22, 25, np.nan, 25],
        "Nota": [85, 90, 78, 90]
    }

    df_pandas = pd.DataFrame(datos)
    df = DataFrame(df_pandas)

    print(df)
    print("\nPrimeras filas:")
    print(df.primeras_filas())
