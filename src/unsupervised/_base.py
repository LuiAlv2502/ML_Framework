import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataframe import DataFrame


class UnsupervisedBase(DataFrame):
    def preparar_datos_numericos(self, columnas=None):
        datos = self.dataframe if columnas is None else self.dataframe[columnas]

        if datos.empty:
            raise ValueError("No hay datos para procesar.")

        datos_codificados = pd.get_dummies(datos, drop_first=True)

        if datos_codificados.empty:
            raise ValueError("No existen columnas utilizables.")

        matriz = StandardScaler().fit_transform(
            datos_codificados.to_numpy(dtype=float)
        )
        return matriz, list(datos_codificados.columns)