"""Catálogo de regresores: nombre legible, familia, hiperparámetros permitidos y configuraciones de benchmark."""
from __future__ import annotations

from typing import Any

from framework_ia.ml.classification.catalog import ARBOLES, DIVISION, PROFUNDIDAD, describe_params

from .regression import FAMILIAS, NOMBRES

FAMILIA_DE = {metodo: familia for familia, metodos in FAMILIAS.items() for metodo in metodos}
FAMILIA_DE["regresion_lineal_simple"] = "Lineales"

CATALOGO: dict[str, dict[str, Any]] = {
    "regresion_lineal_simple": {"params": {}, "requires_column": True},
    "regresion_lineal_multiple": {"params": {}},
    "lasso": {"params": {"alpha": ("real", 0.0001, 10_000.0)}},
    "lasso_cv": {"params": {"cv": ("entero", 2, 10)}},
    "ridge": {"params": {"alpha": ("real", 0.0001, 10_000.0)}},
    "ridge_cv": {"params": {}},
    "svr": {"params": {
        "kernel": ("opcion", ["rbf", "linear", "poly", "sigmoid"]),
        "C": ("real", 0.01, 10_000.0),
        "epsilon": ("real", 0.0, 100.0),
        "degree": ("entero", 2, 6),
    }},
    "decision_tree": {"params": {"max_depth": PROFUNDIDAD, "min_samples_split": DIVISION}},
    "random_forest": {"params": {"n_estimators": ARBOLES, "max_depth": PROFUNDIDAD, "min_samples_split": DIVISION}},
}
for _clave, _info in CATALOGO.items():
    _info["name"] = NOMBRES[_clave]
    _info["family"] = FAMILIA_DE[_clave]

LINEALES = {"regresion_lineal_simple", "regresion_lineal_multiple", "lasso", "lasso_cv", "ridge", "ridge_cv"}

# Configuración estándar de cada algoritmo de las tres familias + variaciones de hiperparámetros.
BENCHMARK_CONFIGS: list[dict[str, Any]] = [
    *({"algorithm": m, "label": f"{NOMBRES[m]} (estándar)", "params": {}}
      for metodos in FAMILIAS.values() for m in metodos),
    {"algorithm": "lasso", "label": "Lasso alpha=1", "params": {"alpha": 1.0}},
    {"algorithm": "ridge", "label": "Ridge alpha=10", "params": {"alpha": 10.0}},
    {"algorithm": "svr", "label": "SVR lineal C=10", "params": {"kernel": "linear", "C": 10.0}},
    {"algorithm": "svr", "label": "SVR rbf C=10", "params": {"C": 10.0}},
    {"algorithm": "decision_tree", "label": "Árbol max_depth=6", "params": {"max_depth": 6}},
    {"algorithm": "random_forest", "label": "RF 300 árboles, max_depth=10",
     "params": {"n_estimators": 300, "max_depth": 10}},
]

__all__ = ["BENCHMARK_CONFIGS", "CATALOGO", "FAMILIA_DE", "LINEALES", "describe_params"]
