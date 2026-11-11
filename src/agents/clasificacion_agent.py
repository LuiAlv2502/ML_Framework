"""Agente de clasificación: elige una variable categórica y valida clasificadores del framework."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from supervised import ClasificacionModelos

from .supervisado_agent import ARBOLES, DIVISION, MAX_CLASES, PROFUNDIDAD, AgenteSupervisado


class AgenteClasificacion(AgenteSupervisado):
    """Valida clasificadores con F1 macro como métrica principal y partición estratificada."""

    TIPO = "clasificacion"
    PROBLEMA = "clasificación"
    CONFIGURACION = {
        "clase": ClasificacionModelos,
        "metrica": "f1_macro",
        "maximizar": True,
        "metricas": ["f1_macro", "accuracy", "precision_macro", "recall_macro"],
        "algoritmos": {
            "knn": {"n_neighbors": ("entero", 1, 50)},
            "decision_tree": {"max_depth": PROFUNDIDAD, "min_samples_split": DIVISION},
            "random_forest": {
                "n_estimators": ARBOLES,
                "max_depth": PROFUNDIDAD,
                "min_samples_split": DIVISION,
            },
            "gradient_boosting": {"n_estimators": ARBOLES, "max_depth": ("entero", 1, 10)},
            "adaboost": {"n_estimators": ("entero", 10, 300)},
            "xgboost": {
                "n_estimators": ARBOLES,
                "max_depth": ("entero", 1, 15),
                "learning_rate": ("real", 0.01, 1.0),
            },
        },
        "nombres": {
            "knn": "K vecinos más cercanos",
            "decision_tree": "Árbol de decisión",
            "random_forest": "Bosques aleatorios",
            "gradient_boosting": "Gradient Boosting",
            "adaboost": "AdaBoost",
            "xgboost": "XGBoost",
        },
    }

    def _reiniciar(self) -> None:
        super()._reiniciar()
        self.estratificar = False

    def _motivo_por_tipo(self, serie: pd.Series, columna: Any, unicos: int) -> str | None:
        if unicos < 2:
            return f"'{columna}' tiene un solo valor; no hay clases que distinguir."
        if unicos > MAX_CLASES:
            return (
                f"'{columna}' tiene {unicos} valores distintos: parece continua, no una clase "
                f"(máximo {MAX_CLASES} clases)."
            )
        return None

    def _describir_objetivo(self, y: pd.Series) -> dict[str, Any]:
        conteos = y.value_counts()
        filas_prueba = math.ceil(len(y) * self.test_size)
        filas_entrenamiento = len(y) - filas_prueba
        self.estratificar = bool(
            conteos.min() >= 2 and min(filas_prueba, filas_entrenamiento) >= len(conteos)
        )
        return {
            "clases": int(len(conteos)),
            "linea_base_accuracy": round(float(conteos.iloc[0] / len(y)), 4),
            "proporcion_clase_minoritaria": round(float(conteos.iloc[-1] / len(y)), 4),
            "particion_estratificada": self.estratificar,
        }

    def _argumentos_particion(self) -> dict[str, Any]:
        return {"stratify": self.estratificar}

    def _comparar_linea_base(self, resumen: dict[str, dict[str, float]], y: pd.Series) -> tuple[bool, str]:
        base = float(y.value_counts(normalize=True).iloc[0])
        valor = resumen["accuracy"]["media"]
        return valor > base, (
            f"accuracy medio {valor:.4f} frente a {base:.4f} de predecir siempre la clase mayoritaria"
        )
