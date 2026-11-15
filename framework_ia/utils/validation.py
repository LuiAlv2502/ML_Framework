"""Validación de entradas: DataFrames, columnas, tipos de tarea, parámetros y archivos.

Cada función lanza :class:`~framework_ia.utils.exceptions.ValidationError` con un mensaje claro en español.
Las especificaciones de parámetros usan el mismo formato que los agentes supervisados:

- ``("entero", minimo, maximo)``
- ``("entero_o_nulo", minimo, maximo)`` (``None`` o ``0`` significan "sin límite")
- ``("real", minimo, maximo)``
- ``("opcion", [valores permitidos])``
- ``("booleano",)``
- ``("lista_reales", minimo, maximo)`` / ``("lista_enteros", minimo, maximo)``
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .exceptions import ValidationError

TASK_TYPES = ("eda", "preprocessing", "clustering", "classification", "regression")
ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")
DEFAULT_MAX_ROWS = 200_000
DEFAULT_MAX_COLUMNS = 500
DEFAULT_MAX_FILE_MB = 50.0
CAPABILITY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")


def validate_dataframe(
    df: Any,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    min_rows: int = 1,
    min_columns: int = 1,
) -> pd.DataFrame:
    """Comprueba que ``df`` sea un DataFrame no vacío, de tamaño razonable y sin columnas duplicadas."""
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("Los datos deben ser una tabla (pandas.DataFrame).")
    if df.empty or len(df) < min_rows:
        raise ValidationError(
            "El conjunto de datos está vacío." if df.empty
            else f"Se requieren al menos {min_rows} filas; hay {len(df)}."
        )
    if df.shape[1] < min_columns:
        raise ValidationError(f"Se requieren al menos {min_columns} columnas; hay {df.shape[1]}.")
    if len(df) > max_rows:
        raise ValidationError(
            f"El conjunto tiene {len(df):,} filas y el máximo permitido es {max_rows:,}. "
            "Use una muestra o ajuste el límite en config.json."
        )
    if df.shape[1] > max_columns:
        raise ValidationError(f"El conjunto tiene {df.shape[1]} columnas; el máximo es {max_columns}.")
    duplicadas = df.columns[df.columns.duplicated()].tolist()
    if duplicadas:
        raise ValidationError(f"Hay columnas con nombre repetido: {duplicadas}. Renómbrelas antes de cargar.")
    return df


def validate_columns(df: pd.DataFrame, columns: Iterable[Any], *, role: str = "columnas") -> list[Any]:
    """Verifica que todas las columnas existan en ``df`` y devuelve la lista."""
    columnas = list(columns)
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ValidationError(f"Las {role} {faltantes} no existen en los datos.")
    return columnas


def validate_target(df: pd.DataFrame, target: Any, *, task: str | None = None) -> Any:
    """Verifica la columna objetivo y su compatibilidad con el tipo de tarea."""
    if target is None or target == "":
        raise ValidationError("Seleccione la variable objetivo.")
    if target not in df.columns:
        raise ValidationError(f"La variable objetivo '{target}' no existe en los datos.")
    serie = df[target]
    if serie.notna().sum() < 2:
        raise ValidationError(f"La variable objetivo '{target}' no tiene suficientes valores.")
    if task == "regression" and not pd.api.types.is_numeric_dtype(serie):
        raise ValidationError(f"'{target}' no es numérica; la regresión requiere un objetivo numérico.")
    if task == "classification":
        clases = serie.nunique(dropna=True)
        if clases < 2:
            raise ValidationError(f"'{target}' tiene una sola clase; no hay nada que distinguir.")
        if clases > 30:
            raise ValidationError(
                f"'{target}' tiene {clases} valores distintos: parece continua. "
                "Para clasificación elija una columna categórica (máximo 30 clases)."
            )
    return target


def validate_task_type(task: str) -> str:
    """Comprueba que el tipo de tarea sea uno de :data:`TASK_TYPES`."""
    if task not in TASK_TYPES:
        raise ValidationError(f"Tipo de tarea inválido '{task}'. Opciones: {', '.join(TASK_TYPES)}.")
    return task


def validate_range(
    name: str, value: Any, minimum: float, maximum: float, *, integer: bool = False
) -> float | int:
    """Valida que ``value`` sea numérico (entero si se pide) y esté en ``[minimum, maximum]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} debe ser numérico.")
    if integer:
        if float(value) != int(value):
            raise ValidationError(f"{name} debe ser entero.")
        value = int(value)
    if not minimum <= value <= maximum:
        raise ValidationError(f"{name} debe estar entre {minimum} y {maximum}; se recibió {value}.")
    return value


def validate_params(params: dict[str, Any] | None, spec: dict[str, tuple], *, owner: str = "") -> dict[str, Any]:
    """Valida un diccionario de hiperparámetros contra una especificación y devuelve una copia limpia."""
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValidationError("Los parámetros deben ser un diccionario.")
    prefijo = f" para {owner}" if owner else ""
    desconocidos = sorted(set(params) - set(spec))
    if desconocidos:
        raise ValidationError(
            f"Parámetros no permitidos{prefijo}: {desconocidos}. Permitidos: {sorted(spec)}."
        )
    limpios: dict[str, Any] = {}
    for nombre, valor in params.items():
        tipo, *rango = spec[nombre]
        if tipo == "opcion":
            if valor not in rango[0]:
                raise ValidationError(f"{nombre} debe ser uno de {rango[0]}.")
        elif tipo == "booleano":
            if not isinstance(valor, bool):
                raise ValidationError(f"{nombre} debe ser verdadero o falso.")
        elif tipo == "entero_o_nulo" and valor in (None, 0):
            valor = None
        elif tipo in ("lista_reales", "lista_enteros"):
            if not isinstance(valor, (list, tuple)) or not valor:
                raise ValidationError(f"{nombre} debe ser una lista no vacía de números.")
            valor = [
                validate_range(nombre, v, rango[0], rango[1], integer=tipo == "lista_enteros") for v in valor
            ]
            if len(set(valor)) != len(valor):
                raise ValidationError(f"Los valores de {nombre} no deben repetirse.")
        else:
            valor = validate_range(nombre, valor, rango[0], rango[1], integer=tipo != "real")
            if tipo == "real":
                valor = float(valor)
        limpios[nombre] = valor
    return limpios


def validate_file(filename: str, size_bytes: int | None = None, *, max_mb: float = DEFAULT_MAX_FILE_MB) -> str:
    """Valida extensión y tamaño de un archivo cargado; devuelve la extensión en minúsculas."""
    if not filename:
        raise ValidationError("El archivo no tiene nombre.")
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Formato '{extension or 'desconocido'}' no soportado. Use: {', '.join(ALLOWED_EXTENSIONS)}."
        )
    if size_bytes is not None:
        if size_bytes == 0:
            raise ValidationError("El archivo está vacío.")
        if size_bytes > max_mb * 1024 * 1024:
            raise ValidationError(f"El archivo pesa {size_bytes / 1024 / 1024:.1f} MB; el máximo es {max_mb} MB.")
    return extension


def detect_encoding(content: bytes) -> str:
    """Devuelve la primera codificación de :data:`ENCODINGS` que decodifica ``content`` sin errores."""
    for codificacion in ENCODINGS:
        try:
            content.decode(codificacion)
        except UnicodeDecodeError:
            continue
        return codificacion
    raise ValidationError("No se pudo determinar la codificación del archivo (pruebe guardarlo como UTF-8).")


def validate_capability(capability: str) -> str:
    """Valida el formato de una capacidad (``dominio.accion``, en minúsculas)."""
    if not isinstance(capability, str) or not CAPABILITY_PATTERN.match(capability):
        raise ValidationError(
            f"Capacidad inválida '{capability}'. Use el formato dominio.accion en minúsculas "
            "(por ejemplo, classification.train)."
        )
    return capability


def validate_name(name: Any, *, what: str = "nombre") -> str:
    """Valida un nombre legible (agente, experimento): 2–60 caracteres alfanuméricos, espacios, _ o -."""
    if not isinstance(name, str) or not re.fullmatch(r"[\wÁÉÍÓÚáéíóúÑñ][\w ÁÉÍÓÚáéíóúÑñ\-.]{1,59}", name.strip()):
        raise ValidationError(
            f"El {what} debe tener entre 2 y 60 caracteres (letras, números, espacios, '_', '-' o '.')."
        )
    return name.strip()


def suggest_task_type(serie: pd.Series) -> str:
    """Sugiere ``classification`` o ``regression`` según el tipo y la cardinalidad de la variable."""
    if pd.api.types.is_numeric_dtype(serie) and serie.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


__all__ = [
    "ALLOWED_EXTENSIONS",
    "TASK_TYPES",
    "detect_encoding",
    "suggest_task_type",
    "validate_capability",
    "validate_columns",
    "validate_dataframe",
    "validate_file",
    "validate_name",
    "validate_params",
    "validate_range",
    "validate_target",
    "validate_task_type",
]
