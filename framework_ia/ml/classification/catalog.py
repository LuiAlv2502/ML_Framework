"""Catálogo de clasificadores: nombre legible, familia, hiperparámetros permitidos y configuraciones de benchmark.

Es la *lista blanca* que usan los agentes para validar lo que piden el usuario o el LLM.
"""
from __future__ import annotations

from typing import Any

PROFUNDIDAD = ("entero_o_nulo", 1, 30)
DIVISION = ("entero", 2, 20)
ARBOLES = ("entero", 10, 500)

CATALOGO: dict[str, dict[str, Any]] = {
    "knn": {
        "name": "K vecinos más cercanos",
        "family": "vecinos",
        "params": {"n_neighbors": ("entero", 1, 50), "algorithm": ("opcion", ["auto", "ball_tree", "kd_tree", "brute"])},
    },
    "decision_tree": {
        "name": "Árbol de decisión",
        "family": "árbol",
        "params": {"max_depth": PROFUNDIDAD, "min_samples_split": DIVISION},
    },
    "random_forest": {
        "name": "Bosques aleatorios",
        "family": "ensamble bagging",
        "params": {"n_estimators": ARBOLES, "max_depth": PROFUNDIDAD, "min_samples_split": DIVISION},
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "family": "ensamble boosting",
        "params": {"n_estimators": ARBOLES, "max_depth": ("entero", 1, 10)},
    },
    "adaboost": {
        "name": "AdaBoost",
        "family": "ensamble boosting",
        "params": {"n_estimators": ("entero", 10, 300)},
    },
    "xgboost": {
        "name": "XGBoost",
        "family": "ensamble boosting",
        "params": {"n_estimators": ARBOLES, "max_depth": ("entero", 1, 15), "learning_rate": ("real", 0.01, 1.0)},
    },
    "adaboost_grid": {
        "name": "AdaBoost optimizado (GridSearchCV)",
        "family": "ensamble boosting",
        "params": {
            "n_estimators": ("lista_enteros", 1, 1000),
            "learning_rate": ("lista_reales", 0.0001, 10.0),
            "cv": ("entero", 2, 10),
        },
    },
}

# Modelo base de cada algoritmo (configuración estándar) + variaciones de hiperparámetros.
BENCHMARK_CONFIGS: list[dict[str, Any]] = [
    {"algorithm": "knn", "label": "KNN k=5 (estándar)", "params": {}},
    {"algorithm": "knn", "label": "KNN k=3", "params": {"n_neighbors": 3}},
    {"algorithm": "knn", "label": "KNN k=11", "params": {"n_neighbors": 11}},
    {"algorithm": "decision_tree", "label": "Árbol (estándar)", "params": {}},
    {"algorithm": "decision_tree", "label": "Árbol max_depth=4", "params": {"max_depth": 4}},
    {"algorithm": "random_forest", "label": "RF 200 árboles (estándar)", "params": {}},
    {"algorithm": "random_forest", "label": "RF 100 árboles, max_depth=6", "params": {"n_estimators": 100, "max_depth": 6}},
    {"algorithm": "adaboost", "label": "AdaBoost 80 (estándar)", "params": {}},
    {"algorithm": "adaboost", "label": "AdaBoost 200", "params": {"n_estimators": 200}},
    {"algorithm": "xgboost", "label": "XGBoost (estándar)", "params": {}},
    {"algorithm": "xgboost", "label": "XGBoost lr=0.05, 300 árboles",
     "params": {"learning_rate": 0.05, "n_estimators": 300}},
    {"algorithm": "gradient_boosting", "label": "Gradient Boosting (estándar)", "params": {}},
]


def describe_params(params: dict[str, Any]) -> str:
    """Texto breve de los hiperparámetros (``estándar`` si no hay)."""
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items())) if params else "estándar"


__all__ = ["BENCHMARK_CONFIGS", "CATALOGO", "describe_params"]
