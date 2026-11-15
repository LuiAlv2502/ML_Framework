"""Transformaciones de preprocesamiento reutilizables.

Envuelven los métodos de :class:`framework_ia.ml.eda.DataFrame` (imputar, normalizar, estandarizar, ...) y
agregan las operaciones que faltaban (codificación, exclusión de identificadores, selección de atributos).
Todas reciben un ``pandas.DataFrame`` y devuelven uno **nuevo**: nunca modifican la entrada.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from framework_ia.ml.eda import DataFrame

DTYPES = ("float64", "int64", "str", "category", "datetime64[ns]", "bool")
FILTER_OPERATORS = ("==", "!=", ">", ">=", "<", "<=")


def _numericas(df: pd.DataFrame, columns: list[Any] | None) -> list[Any]:
    numericas = df.select_dtypes(include=np.number).columns.tolist()
    return numericas if not columns else [c for c in columns if c in numericas]


def _sobre_columnas(df: pd.DataFrame, columns: list[Any], metodo: str) -> pd.DataFrame:
    """Aplica un método del DataFrame del framework solo sobre ``columns``."""
    if not columns:
        raise ValueError("No hay columnas numéricas para transformar.")
    datos = df.copy()
    parcial = DataFrame(datos[columns])
    getattr(parcial, metodo)()
    datos[columns] = parcial.dataframe[columns]
    return datos


def impute(df: pd.DataFrame, columns: list[Any] | None = None, strategy: str = "mediana_moda") -> pd.DataFrame:
    """Imputa nulos: mediana o media (numéricas) y moda (categóricas)."""
    columnas = columns or df.columns.tolist()
    datos = df.copy()
    if strategy == "mediana_moda":
        parcial = DataFrame(datos[columnas])
        parcial.imputar_nulos()
        datos[columnas] = parcial.dataframe[columnas]
        return datos
    if strategy != "media_moda":
        raise ValueError("Estrategia de imputación inválida (use mediana_moda o media_moda).")
    for columna in columnas:
        if not datos[columna].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(datos[columna]):
            datos[columna] = datos[columna].fillna(datos[columna].mean())
        else:
            moda = datos[columna].mode()
            datos[columna] = datos[columna].fillna(moda.iloc[0] if not moda.empty else "Desconocido")
    return datos


def drop_na(df: pd.DataFrame, columns: list[Any] | None = None) -> pd.DataFrame:
    """Elimina filas con nulos (en todas las columnas o solo en ``columns``)."""
    if columns:
        return df.dropna(subset=columns).reset_index(drop=True)
    datos = DataFrame(df)
    return datos.eliminar_nulos().copy()


def fill_na(df: pd.DataFrame, value: Any = 0, columns: list[Any] | None = None) -> pd.DataFrame:
    """Reemplaza nulos por un valor constante."""
    if columns:
        datos = df.copy()
        datos[columns] = datos[columns].fillna(value)
        return datos
    return DataFrame(df).reemplazar_nulos(value).copy()


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas duplicadas."""
    return DataFrame(df).eliminar_duplicados().copy()


def normalize(df: pd.DataFrame, columns: list[Any] | None = None) -> pd.DataFrame:
    """Escala min-max a [0, 1] las columnas numéricas indicadas (o todas)."""
    return _sobre_columnas(df, _numericas(df, columns), "normalizar")


def standardize(df: pd.DataFrame, columns: list[Any] | None = None) -> pd.DataFrame:
    """Estandariza (z-score) las columnas numéricas indicadas (o todas)."""
    return _sobre_columnas(df, _numericas(df, columns), "estandarizar")


def drop_columns(df: pd.DataFrame, columns: list[Any]) -> pd.DataFrame:
    """Elimina columnas."""
    datos = DataFrame(df)
    for columna in columns:
        datos.eliminar_columna(columna)
    return datos.dataframe.copy()


def convert_type(df: pd.DataFrame, column: Any, dtype: str) -> pd.DataFrame:
    """Convierte el tipo de una columna."""
    if dtype not in DTYPES:
        raise ValueError(f"Tipo no soportado '{dtype}'. Opciones: {', '.join(DTYPES)}.")
    if dtype == "datetime64[ns]":
        datos = df.copy()
        datos[column] = pd.to_datetime(datos[column], errors="coerce")
        return datos
    if dtype in ("float64", "int64") and not pd.api.types.is_numeric_dtype(df[column]):
        convertida = pd.to_numeric(df[column], errors="coerce")
        if convertida.isna().sum() > df[column].isna().sum():
            raise ValueError(f"'{column}' tiene valores que no son números; no se puede convertir a {dtype}.")
        df = df.assign(**{column: convertida})
    return DataFrame(df).convertir_tipos({column: dtype}).copy()


def sort(df: pd.DataFrame, column: Any, ascending: bool = True) -> pd.DataFrame:
    """Ordena por una columna."""
    if ascending:
        return DataFrame(df).ordenar(column).copy()
    if column not in df.columns:
        raise KeyError(f"La columna '{column}' no existe.")
    return df.sort_values(column, ascending=False).reset_index(drop=True)


def encode_onehot(df: pd.DataFrame, columns: list[Any] | None = None, drop_first: bool = False) -> pd.DataFrame:
    """Codificación one-hot de las columnas categóricas indicadas (o todas)."""
    columnas = columns or df.select_dtypes(exclude=np.number).columns.tolist()
    if not columnas:
        raise ValueError("No hay columnas categóricas para codificar.")
    return pd.get_dummies(df, columns=columnas, drop_first=drop_first, dtype=int)


def encode_ordinal(df: pd.DataFrame, columns: list[Any] | None = None) -> pd.DataFrame:
    """Codificación ordinal (códigos enteros por categoría, -1 para nulos)."""
    columnas = columns or df.select_dtypes(exclude=np.number).columns.tolist()
    if not columnas:
        raise ValueError("No hay columnas categóricas para codificar.")
    datos = df.copy()
    for columna in columnas:
        datos[columna] = datos[columna].astype("category").cat.codes
    return datos


def select_features(df: pd.DataFrame, columns: list[Any]) -> pd.DataFrame:
    """Conserva solo las columnas indicadas."""
    return DataFrame(df).seleccionar_variables(list(columns)).copy()


def standardize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza espacios y símbolos de los nombres de columna por ``_``."""
    nuevos = {c: re.sub(r"[^0-9a-zA-Z_]+", "_", str(c).strip()).strip("_") or str(c) for c in df.columns}
    return df.rename(columns=nuevos)


def detect_id_columns(df: pd.DataFrame) -> list[Any]:
    """Columnas que parecen identificadores (únicas por fila con nombre de ID, consecutivas o texto único)."""
    n = len(df)
    ids = []
    for columna in df.columns:
        serie = df[columna]
        nombre_id = re.search(r"(^id$|_id$|^id_|^unnamed|codigo|customerid)", str(columna), re.I)
        if n > 20 and serie.nunique(dropna=True) == n:
            consecutivo = pd.api.types.is_integer_dtype(serie) and serie.is_monotonic_increasing
            texto_unico = not pd.api.types.is_numeric_dtype(serie)
            if nombre_id or consecutivo or texto_unico:
                ids.append(columna)
        elif nombre_id and serie.nunique(dropna=True) == n:
            ids.append(columna)
    return ids


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las columnas detectadas como identificadores."""
    ids = detect_id_columns(df)
    return df.drop(columns=ids) if ids else df.copy()


def filter_rows(df: pd.DataFrame, column: Any, operator: str, value: Any) -> pd.DataFrame:
    """Filas que cumplen ``column <operator> value`` (vista previa: no modifica ``df``)."""
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]) and isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    return DataFrame(df).filtrar(column, operator, value).copy()


OPERATIONS: dict[str, Callable[..., pd.DataFrame]] = {
    "impute": impute,
    "drop_na": drop_na,
    "fill_na": fill_na,
    "drop_duplicates": drop_duplicates,
    "normalize": normalize,
    "standardize": standardize,
    "drop_columns": drop_columns,
    "convert_type": convert_type,
    "sort": sort,
    "encode_onehot": encode_onehot,
    "encode_ordinal": encode_ordinal,
    "select_features": select_features,
    "standardize_names": standardize_names,
    "drop_id_columns": drop_id_columns,
}

DESCRIPTIONS: dict[str, str] = {
    "impute": "Imputar nulos (mediana o media para numéricas, moda para categóricas)",
    "drop_na": "Eliminar filas con nulos",
    "fill_na": "Reemplazar nulos por un valor",
    "drop_duplicates": "Eliminar filas duplicadas",
    "normalize": "Normalizar numéricas a [0, 1]",
    "standardize": "Estandarizar numéricas (media 0, desviación 1)",
    "drop_columns": "Eliminar columnas",
    "convert_type": "Convertir el tipo de una columna",
    "sort": "Ordenar por una columna",
    "encode_onehot": "Codificación one-hot de categóricas",
    "encode_ordinal": "Codificación ordinal de categóricas",
    "select_features": "Seleccionar atributos (conservar solo esas columnas)",
    "standardize_names": "Estandarizar nombres de columnas",
    "drop_id_columns": "Excluir columnas identificadoras",
}


def apply_operation(df: pd.DataFrame, operation: str, **params: Any) -> pd.DataFrame:
    """Aplica una operación de :data:`OPERATIONS` por nombre."""
    if operation not in OPERATIONS:
        raise ValueError(f"Operación desconocida '{operation}'. Opciones: {', '.join(OPERATIONS)}.")
    return OPERATIONS[operation](df, **params)


__all__ = [
    "DESCRIPTIONS",
    "DTYPES",
    "FILTER_OPERATORS",
    "OPERATIONS",
    "apply_operation",
    "detect_id_columns",
    "filter_rows",
]
