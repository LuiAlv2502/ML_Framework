"""Agente de regresión: elige una variable numérica continua y valida regresores del framework."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from supervised import RegresionModelos
from supervised.Regression.Regression import NOMBRES as NOMBRES_REGRESION

from .supervised_agent import ARBOLES, DIVISION, PROFUNDIDAD, AgenteSupervisado

MIN_UNICOS_REGRESION = 10


class AgenteRegresion(AgenteSupervisado):
    """Valida regresores con RMSE como métrica principal."""

    TIPO = "regresion"
    PROBLEMA = "regresión"
    CONFIGURACION = {
        "clase": RegresionModelos,
        "metrica": "RMSE",
        "maximizar": False,
        "metricas": ["RMSE", "MAE", "ER", "R2"],
        "algoritmos": {
            "regresion_lineal_multiple": {},
            "lasso": {"alpha": ("real", 0.0001, 10_000.0)},
            "lasso_cv": {"cv": ("entero", 2, 10)},
            "ridge": {"alpha": ("real", 0.0001, 10_000.0)},
            "ridge_cv": {},
            "svr": {
                "kernel": ("opcion", ["rbf", "linear", "poly", "sigmoid"]),
                "C": ("real", 0.01, 10_000.0),
                "epsilon": ("real", 0.0, 100.0),
                "degree": ("entero", 2, 6),
            },
            "decision_tree": {"max_depth": PROFUNDIDAD, "min_samples_split": DIVISION},
            "random_forest": {
                "n_estimators": ARBOLES,
                "max_depth": PROFUNDIDAD,
                "min_samples_split": DIVISION,
            },
        },
        "nombres": dict(NOMBRES_REGRESION),
    }

    def _motivo_por_tipo(self, serie: pd.Series, columna: Any, unicos: int) -> str | None:
        if not pd.api.types.is_numeric_dtype(serie):
            return f"'{columna}' no es numérica; la regresión requiere un objetivo numérico."
        if unicos <= MIN_UNICOS_REGRESION:
            return (
                f"'{columna}' solo tiene {unicos} valores distintos: parece categórica, "
                "más adecuada para clasificación."
            )
        return None

    def _describir_objetivo(self, y: pd.Series) -> dict[str, Any]:
        return {
            "resumen_objetivo": {
                "min": float(y.min()),
                "mediana": float(y.median()),
                "max": float(y.max()),
                "media": float(y.mean()),
                "desviacion": float(y.std()),
            }
        }

    def _comparar_linea_base(self, resumen: dict[str, dict[str, float]], y: pd.Series) -> tuple[bool, str]:
        base = self.linea_base_regresion(y)
        valor = resumen["RMSE"]["media"]
        return valor < base, f"RMSE medio {valor:.4f} frente a {base:.4f} de predecir siempre la media"

    @staticmethod
    def linea_base_regresion(y: pd.Series) -> float:
        """RMSE de predecir siempre la media (referencia mínima)."""
        return float(np.sqrt(((y - y.mean()) ** 2).mean()))
