"""Benchmark de agrupamiento: compara configuraciones de PCA, K-Means y HAC con métricas homogéneas.

Todas las configuraciones usan la misma matriz preparada (imputación, one-hot y estandarización de
:class:`UnsupervisedBase`) y la misma semilla, de modo que las métricas son comparables:

- Agrupamiento: silhouette (↑), Davies-Bouldin (↓), Calinski-Harabasz (↑); inercia (K-Means) y correlación
  cofenética (HAC).
- Reducción de dimensionalidad (PCA): varianza explicada acumulada y trustworthiness de la proyección.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from framework_ia.utils.metrics import clustering_metrics, trustworthiness_score

from .clustering import ClusteringAnalysis
from .pca import PCAAnalysis

MAX_FILAS_HAC = 5000
LINKAGES = ("ward", "average", "complete", "single")


def default_configs(k_values: range | list[int] = range(2, 7), hac_k: int = 3) -> list[dict[str, Any]]:
    """Modelo base + variaciones: K-Means para cada k, HAC con cada enlace y PCA estándar vs. whiten."""
    configs: list[dict[str, Any]] = [
        {"algorithm": "pca", "label": "PCA 2 comp. (estándar)", "params": {"n_components": 2}},
        {"algorithm": "pca", "label": "PCA 2 comp. whiten + randomized",
         "params": {"n_components": 2, "whiten": True, "svd_solver": "randomized"}},
    ]
    configs += [{"algorithm": "kmeans", "label": f"K-Means k={k}", "params": {"n_clusters": k}} for k in k_values]
    configs.append({"algorithm": "kmeans", "label": f"K-Means k={hac_k} init=random",
                    "params": {"n_clusters": hac_k, "init": "random"}})
    configs += [{"algorithm": "hac", "label": f"HAC {enlace} k={hac_k}",
                 "params": {"n_clusters": hac_k, "linkage": enlace}} for enlace in LINKAGES]
    return configs


class ClusteringBenchmark:
    """Ejecuta varias configuraciones de agrupamiento/reducción sobre los mismos datos."""

    def __init__(self, df: pd.DataFrame, columns: list[Any] | None = None) -> None:
        self.columns = columns
        self._clustering = ClusteringAnalysis(df)
        self._pca = PCAAnalysis(df)
        self.matrix, self.feature_names = self._clustering.preparar_datos_numericos(columns)

    def _pca_fila(self, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        n = min(int(params.get("n_components", 2)), self.matrix.shape[0], self.matrix.shape[1])
        resultado = self._pca.pca(n_components=n, whiten=bool(params.get("whiten", False)),
                                  svd_solver=params.get("svd_solver", "auto"), columnas=self.columns)
        componentes = resultado["componentes"].to_numpy()
        metricas = {
            "varianza_acumulada": float(resultado["metrica"]),
            "trustworthiness": trustworthiness_score(self.matrix, componentes),
        }
        return metricas, resultado

    def _cluster_fila(self, algoritmo: str, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if algoritmo == "kmeans":
            resultado = self._clustering.kmeans(columnas=self.columns, **params)
        else:
            if len(self.matrix) > MAX_FILAS_HAC:
                raise ValueError(f"HAC se limita a {MAX_FILAS_HAC} filas.")
            resultado = self._clustering.hac(columnas=self.columns, **params)
        etiquetas = resultado["resultado"]["cluster"].to_numpy()
        metricas = clustering_metrics(self.matrix, etiquetas)
        metricas["grupos"] = int(len(np.unique(etiquetas)))
        if "inercia" in resultado:
            metricas["inercia"] = float(resultado["inercia"])
        if "correlacion_cofenetica" in resultado:
            metricas["correlacion_cofenetica"] = float(resultado["correlacion_cofenetica"])
        return metricas, resultado

    def run(self, configs: list[dict[str, Any]] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Devuelve ``(tabla, resultados)``; la tabla queda ordenada por silhouette (agrupamiento primero)."""
        configs = default_configs() if configs is None else configs
        filas, resultados = [], {}
        for posicion, config in enumerate(configs):
            algoritmo = config["algorithm"]
            params = dict(config.get("params") or {})
            etiqueta = config.get("label") or f"{algoritmo} {params}"
            if etiqueta in resultados:
                etiqueta = f"{etiqueta} #{posicion + 1}"
            fila: dict[str, Any] = {
                "Configuración": etiqueta,
                "Algoritmo": {"pca": "PCA", "kmeans": "K-Means", "hac": "HAC"}.get(algoritmo, algoritmo),
                "Tipo": "Reducción" if algoritmo == "pca" else "Agrupamiento",
                "Parámetros": ", ".join(f"{k}={v}" for k, v in sorted(params.items())) or "estándar",
            }
            try:
                if algoritmo == "pca":
                    metricas, resultado = self._pca_fila(params)
                elif algoritmo in ("kmeans", "hac"):
                    metricas, resultado = self._cluster_fila(algoritmo, params)
                else:
                    raise ValueError(f"Algoritmo de benchmark desconocido '{algoritmo}'.")
            except (ValueError, TypeError) as error:
                filas.append({**fila, "error": str(error)})
                continue
            filas.append({**fila, **metricas})
            resultados[etiqueta] = {"algorithm": algoritmo, "params": params, "metrics": metricas, "raw": resultado}
        tabla = pd.DataFrame(filas)
        if "silhouette" in tabla:
            tabla = tabla.sort_values(["Tipo", "silhouette"], ascending=[True, False], na_position="last",
                                      ignore_index=True)
        return tabla, resultados


__all__ = ["LINKAGES", "ClusteringBenchmark", "default_configs"]
