"""Preprocesamiento configurable del dataset (independiente del EDA) y particiones de entrenamiento/prueba."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from framework_ia.ml.preprocessing import transforms
from framework_ia.utils.exceptions import ValidationError
from framework_ia.utils.logging_config import get_logger
from framework_ia.utils.validation import validate_columns, validate_dataframe, validate_range, validate_target

logger = get_logger("services.preprocessing")

# Parámetros permitidos por operación: nombre -> obligatorio
OPERATION_PARAMS: dict[str, dict[str, bool]] = {
    "impute": {"columns": False, "strategy": False},
    "drop_na": {"columns": False},
    "fill_na": {"value": False, "columns": False},
    "drop_duplicates": {},
    "normalize": {"columns": False},
    "standardize": {"columns": False},
    "drop_columns": {"columns": True},
    "convert_type": {"column": True, "dtype": True},
    "sort": {"column": True, "ascending": False},
    "encode_onehot": {"columns": False, "drop_first": False},
    "encode_ordinal": {"columns": False},
    "select_features": {"columns": True},
    "standardize_names": {},
    "drop_id_columns": {},
}


@dataclass
class PreprocessingResult:
    """Datos transformados y bitácora de cada operación aplicada."""

    data: pd.DataFrame
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def log(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


class PreprocessingService:
    """Valida y aplica una lista ordenada de operaciones de preprocesamiento."""

    @staticmethod
    def available_operations() -> dict[str, str]:
        """Operaciones soportadas con su descripción."""
        return dict(transforms.DESCRIPTIONS)

    @staticmethod
    def validate_operation(df: pd.DataFrame, operation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Comprueba nombre y parámetros de una operación; devuelve ``(nombre, parametros)``."""
        if not isinstance(operation, dict) or "op" not in operation:
            raise ValidationError("Cada operación debe ser un diccionario con la clave 'op'.")
        nombre = operation["op"]
        if nombre not in OPERATION_PARAMS:
            raise ValidationError(f"Operación desconocida '{nombre}'. Opciones: {', '.join(OPERATION_PARAMS)}.")
        parametros = {k: v for k, v in operation.items() if k != "op"}
        permitidos = OPERATION_PARAMS[nombre]
        extra = sorted(set(parametros) - set(permitidos))
        if extra:
            raise ValidationError(f"Parámetros no válidos para '{nombre}': {extra}.")
        faltan = [p for p, obligatorio in permitidos.items() if obligatorio and parametros.get(p) in (None, [], "")]
        if faltan:
            raise ValidationError(f"La operación '{nombre}' requiere: {faltan}.")
        if parametros.get("columns"):
            validate_columns(df, parametros["columns"])
        if parametros.get("column") is not None:
            validate_columns(df, [parametros["column"]], role="columna")
        return nombre, parametros

    def apply(self, df: pd.DataFrame, operations: list[dict[str, Any]]) -> PreprocessingResult:
        """Aplica las operaciones en orden sobre una copia y registra el efecto de cada una."""
        validate_dataframe(df)
        if not operations:
            raise ValidationError("Seleccione al menos una operación de preprocesamiento.")
        datos = df.copy()
        pasos = []
        for posicion, operacion in enumerate(operations, start=1):
            nombre, parametros = self.validate_operation(datos, operacion)
            antes = datos.shape
            nulos_antes = int(datos.isna().sum().sum())
            try:
                datos = transforms.apply_operation(datos, nombre, **parametros)
            except (KeyError, TypeError, ValueError) as error:
                raise ValidationError(f"Paso {posicion} ({nombre}) no se pudo aplicar: {error}") from error
            if datos.empty:
                raise ValidationError(f"Paso {posicion} ({nombre}) dejó el conjunto vacío.")
            pasos.append({
                "paso": posicion,
                "operacion": nombre,
                "descripcion": transforms.DESCRIPTIONS[nombre],
                "parametros": {k: (list(v) if isinstance(v, tuple) else v) for k, v in parametros.items()},
                "filas": f"{antes[0]} → {datos.shape[0]}",
                "columnas": f"{antes[1]} → {datos.shape[1]}",
                "nulos": f"{nulos_antes} → {int(datos.isna().sum().sum())}",
            })
            logger.info("Preprocesamiento %s: %s", nombre, pasos[-1])
        return PreprocessingResult(datos.reset_index(drop=True), pasos)

    @staticmethod
    def compare(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
        """Tabla antes/después por columna (tipo y nulos)."""
        columnas = list(dict.fromkeys([*before.columns, *after.columns]))
        filas = []
        for columna in columnas:
            filas.append({
                "Columna": str(columna),
                "Tipo antes": str(before[columna].dtype) if columna in before else "—",
                "Tipo después": str(after[columna].dtype) if columna in after else "(eliminada)",
                "Nulos antes": int(before[columna].isna().sum()) if columna in before else None,
                "Nulos después": int(after[columna].isna().sum()) if columna in after else None,
            })
        return pd.DataFrame(filas)

    @staticmethod
    def detect_id_columns(df: pd.DataFrame) -> list[Any]:
        """Columnas que parecen identificadores."""
        return transforms.detect_id_columns(df)

    @staticmethod
    def filter_preview(df: pd.DataFrame, column: Any, operator: str, value: Any, limit: int = 50) -> pd.DataFrame:
        """Vista previa de un filtro (no modifica los datos)."""
        if operator not in transforms.FILTER_OPERATORS:
            raise ValidationError(f"Operador inválido. Use {', '.join(transforms.FILTER_OPERATORS)}.")
        try:
            return transforms.filter_rows(df, column, operator, value)
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError(f"No se pudo filtrar: {error}") from error

    @staticmethod
    def partitions(
        df: pd.DataFrame,
        *,
        task: str,
        target: Any,
        features: list[Any] | None,
        test_size: float,
        seeds: list[int],
        stratify: bool = False,
    ) -> list[dict[str, Any]]:
        """Particiones aprendizaje/prueba (una por semilla) con los índices de prueba.

        Usa ``prepare_data`` de las clases supervisadas para que la partición coincida exactamente con la
        que usarán los modelos.
        """
        from framework_ia.ml.classification import ClasificacionModelos
        from framework_ia.ml.regression import RegresionModelos

        if task not in ("classification", "regression"):
            raise ValidationError("Las particiones solo aplican a clasificación o regresión.")
        validate_target(df, target, task=task)
        validate_range("test_size", test_size, 0.05, 0.95)
        if features is not None and not features:
            raise ValidationError("Seleccione al menos una variable predictora.")
        if not seeds:
            raise ValidationError("Indique al menos una semilla.")
        resultado = []
        for semilla in seeds:
            if task == "classification":
                modelo = ClasificacionModelos(df, target=target, feature_columns=features, test_size=test_size,
                                              random_state=int(semilla), stratify=stratify, scale=False)
            else:
                modelo = RegresionModelos(df, target=target, feature_columns=features, test_size=test_size,
                                          random_state=int(semilla), scale=False)
            try:
                split = modelo.prepare_data()
            except (KeyError, ValueError) as error:
                raise ValidationError(f"No se pudo generar la partición: {error}") from error
            resultado.append({
                "seed": int(semilla),
                "test_index": split.X_test.index.tolist(),
                "n_train": len(split.X_train),
                "n_test": len(split.X_test),
            })
        return resultado


__all__ = ["OPERATION_PARAMS", "PreprocessingResult", "PreprocessingService"]
