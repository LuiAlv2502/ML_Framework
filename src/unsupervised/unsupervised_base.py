import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from Eda.dataframe import DataFrame

class UnsupervisedBase(DataFrame):
    def preparar_datos_numericos(self, columnas=None):
        datos = self.dataframe if columnas is None else self.dataframe[columnas]

        if datos.empty:
            raise ValueError("No hay datos para procesar.")

        datos = datos.copy()
        numericas = datos.select_dtypes(include=np.number).columns.tolist()
        categoricas = [columna for columna in datos.columns if columna not in numericas]
        if numericas:
            datos[numericas] = SimpleImputer(strategy="median").fit_transform(datos[numericas])
        for columna in categoricas:
            moda = datos[columna].mode(dropna=True)
            valor = moda.iloc[0] if not moda.empty else "Desconocido"
            datos[columna] = datos[columna].astype(object).fillna(valor)

        datos_codificados = pd.get_dummies(datos, drop_first=True)

        if datos_codificados.empty:
            raise ValueError("No existen columnas utilizables.")

        matriz = StandardScaler().fit_transform(
            datos_codificados.to_numpy(dtype=float)
        )
        return matriz, list(datos_codificados.columns)