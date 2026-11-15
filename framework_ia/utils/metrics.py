"""Métricas comunes de clasificación, regresión y agrupamiento.

Las clases de ``framework_ia.ml`` delegan aquí el cálculo de métricas para que agentes, servicios y
benchmarks usen exactamente las mismas definiciones.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage as scipy_linkage
from scipy.spatial.distance import pdist
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.utils.multiclass import unique_labels

# Métrica principal por dominio: (nombre, mayor_es_mejor)
PRIMARY_METRICS: dict[str, tuple[str, bool]] = {
    "classification": ("f1_macro", True),
    "regression": ("RMSE", False),
    "clustering": ("silhouette", True),
}


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, Any]:
    """Accuracy, precisión/recall/F1 macro, matriz de confusión (con etiquetas) y reporte por clase."""
    labels = unique_labels(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": [str(label) for label in labels],
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """RMSE, MAE, ER (error relativo, como en la guía del curso) y R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    suma_abs = np.sum(np.abs(y_true))
    return {
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "ER": float(np.sum(np.abs(y_true - y_pred)) / suma_abs) if suma_abs else float("nan"),
        "R2": float(r2_score(y_true, y_pred)),
    }


def clustering_metrics(matrix: np.ndarray, labels: Sequence[Any]) -> dict[str, float]:
    """Silhouette, Davies-Bouldin y Calinski-Harabasz (``nan`` si hay menos de 2 grupos o grupos de tamaño n)."""
    etiquetas = np.asarray(labels)
    grupos = len(np.unique(etiquetas))
    if grupos < 2 or grupos >= len(etiquetas):
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
    return {
        "silhouette": float(silhouette_score(matrix, etiquetas)),
        "davies_bouldin": float(davies_bouldin_score(matrix, etiquetas)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, etiquetas)),
    }


def cophenetic_correlation(matrix: np.ndarray, method: str = "ward") -> float:
    """Correlación cofenética del dendrograma construido con ``method``."""
    enlaces = scipy_linkage(matrix, method=method)
    valor, _ = cophenet(enlaces, pdist(matrix))
    return float(valor)


def trustworthiness_score(matrix: np.ndarray, embedding: np.ndarray, n_neighbors: int = 5) -> float:
    """Trustworthiness de una proyección (1 = conserva perfectamente las vecindades)."""
    vecinos = max(1, min(n_neighbors, (len(matrix) - 1) // 2))
    return float(trustworthiness(matrix, embedding, n_neighbors=vecinos))


def is_better(candidate: float, reference: float | None, maximize: bool) -> bool:
    """Indica si ``candidate`` mejora a ``reference`` según el sentido de la métrica (ignora ``nan``)."""
    if candidate is None or (isinstance(candidate, float) and math.isnan(candidate)):
        return False
    if reference is None or (isinstance(reference, float) and math.isnan(reference)):
        return True
    return candidate > reference if maximize else candidate < reference


__all__ = [
    "PRIMARY_METRICS",
    "classification_metrics",
    "clustering_metrics",
    "cophenetic_correlation",
    "is_better",
    "regression_metrics",
    "trustworthiness_score",
]
