"""Ejecuta benchmarks reproducibles y genera resultados y figuras del informe."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from framework_ia.controller import FrameworkController
from framework_ia.persistence.config_store import ConfigStore
from framework_ia.utils.logging_config import close_logging

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "resultados"
FIGURES = OUTPUT / "figures"
SEEDS = [42, 43, 44]
TEST_SIZE = 0.25
CLUSTER_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_table(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, encoding="utf-8-sig")


def _plot_confusion_matrix(result: dict[str, Any]) -> None:
    data = result["confusion_matrix"]
    matrix = np.asarray(data["matrix"])
    labels = [str(label) for label in data["labels"]]
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Observaciones")
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Clase predicha",
        ylabel="Clase real",
        title="Clasificación de medicamentos: matriz de confusión",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(FIGURES / "01_matriz_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_regression_predictions(predictions: pd.DataFrame) -> None:
    actual = predictions["Real"].to_numpy(dtype=float)
    predicted = predictions["Predicción"].to_numpy(dtype=float)
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(actual, predicted, alpha=0.75, edgecolor="white", linewidth=0.4)
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#dc2626", label="Predicción ideal")
    axis.set(
        xlabel="Ingreso anual real",
        ylabel="Ingreso anual predicho",
        title="Regresión RidgeCV: valores reales y predichos",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURES / "02_regresion_real_predicho.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_clusters(result: dict[str, Any]) -> None:
    projection = result["projection"].copy()
    figure, axis = plt.subplots(figsize=(8, 6))
    for cluster in sorted(projection["cluster"].astype(str).unique()):
        group = projection[projection["cluster"].astype(str) == cluster]
        axis.scatter(group.iloc[:, 0], group.iloc[:, 1], label=f"Grupo {cluster}", alpha=0.8, s=35)
    variance = result["projection_variance"]
    axis.set(
        xlabel=f"Componente principal 1 ({variance[0]:.1%})",
        ylabel=f"Componente principal 2 ({variance[1]:.1%})",
        title="Segmentación de clientes con K-Means (k=3)",
    )
    axis.legend(title="Agrupación", frameon=True)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURES / "03_segmentacion_clientes.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_benchmark_comparison(
    classification: pd.DataFrame, regression: pd.DataFrame, clustering: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10, 13))
    specifications = [
        (classification, "f1_macro", "Modelo", "Clasificación · F1 macro (mayor es mejor)", True, "#2563eb"),
        (regression, "RMSE", "Modelo", "Regresión · RMSE (menor es mejor)", False, "#059669"),
        (clustering, "silhouette", "Configuración", "Agrupamiento · Silhouette (mayor es mejor)", True, "#d97706"),
    ]
    for axis, (table, metric, label, title, ascending, color) in zip(axes, specifications):
        valid = table.dropna(subset=[metric]).sort_values(metric, ascending=not ascending).head(6)
        valid = valid.sort_values(metric, ascending=ascending)
        axis.barh(valid[label].astype(str), valid[metric], color=color)
        axis.set_title(title, loc="left")
        axis.set_xlabel(metric)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Comparación de configuraciones", fontsize=15)
    figure.tight_layout()
    figure.savefig(FIGURES / "04_comparacion_benchmarks.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _best_row(table: pd.DataFrame, metric: str, ascending: bool) -> dict[str, Any]:
    valid = table.dropna(subset=[metric])
    if valid.empty:
        raise RuntimeError(f"No hay resultados válidos para la métrica {metric}.")
    return valid.sort_values(metric, ascending=ascending).iloc[0].to_dict()


def _latex_escape(value: str) -> str:
    for character, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        value = value.replace(character, replacement)
    return value


def run() -> dict[str, Any]:
    os.environ.pop("GEMINI_API_KEY", None)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    config = ConfigStore()
    config.set("llm.enabled", False)
    with tempfile.TemporaryDirectory(prefix="framework-ia-experiments-") as temporary:
        temporary_path = Path(temporary)
        config.set("paths.logs_dir", str(temporary_path / "logs"))
        controller = FrameworkController(config=config, persist=False, runs_dir=temporary_path / "runs")
        try:
            controller.load_sample("drug200.csv")
            classification_benchmark = controller.run_benchmark(
                "classification",
                {"target": "Drug", "test_size": TEST_SIZE, "seeds": [SEEDS[0]], "stratify": True},
            )["result"]["table"]
            classification = controller.run_classification(
                target="Drug",
                algorithm="random_forest",
                params={"n_estimators": 200},
                test_size=TEST_SIZE,
                seeds=SEEDS,
            )["result"]

            regression_data = pd.read_csv(ROOT / "data" / "regression_sample.csv")
            controller.load_dataframe(regression_data, "regression_sample.csv")
            regression_benchmark = controller.run_benchmark(
                "regression",
                {"target": "ingreso_anual", "test_size": TEST_SIZE, "seeds": [SEEDS[0]]},
            )["result"]["table"]
            regression = controller.run_regression(
                target="ingreso_anual",
                algorithm="ridge_cv",
                params={"alphas": [0.01, 0.1, 1.0, 10.0, 100.0]},
                test_size=TEST_SIZE,
                seeds=SEEDS,
            )["result"]

            controller.load_sample("Mall_Customers.csv")
            clustering_benchmark = controller.run_benchmark(
                "clustering",
                {
                    "columns": CLUSTER_COLUMNS,
                    "configs": [
                        {"algorithm": "kmeans", "label": f"K-Means k={k}", "params": {"n_clusters": k}}
                        for k in (2, 3, 4)
                    ] + [
                        {"algorithm": "hac", "label": "HAC Ward k=3",
                         "params": {"n_clusters": 3, "linkage": "ward"}},
                        {"algorithm": "pca", "label": "PCA 2 componentes",
                         "params": {"n_components": 2}},
                        {"algorithm": "pca", "label": "PCA 2 componentes con whitening",
                         "params": {"n_components": 2, "whiten": True, "svd_solver": "full"}},
                    ],
                },
            )["result"]["table"]
            clustering = controller.run_clustering(
                "kmeans",
                {"n_clusters": 3, "init": "k-means++", "n_init": 10},
                columns=CLUSTER_COLUMNS,
                projection_dims=2,
            )["result"]
        finally:
            close_logging()

    _write_table(classification_benchmark, OUTPUT / "benchmark_clasificacion.csv")
    _write_table(regression_benchmark, OUTPUT / "benchmark_regresion.csv")
    _write_table(clustering_benchmark, OUTPUT / "benchmark_agrupamiento.csv")
    _plot_confusion_matrix(classification)
    _plot_regression_predictions(regression["experiments"][0]["predictions"])
    _plot_clusters(clustering)
    _plot_benchmark_comparison(classification_benchmark, regression_benchmark, clustering_benchmark)

    best_classification = _best_row(classification_benchmark, "f1_macro", False)
    best_regression = _best_row(regression_benchmark, "RMSE", True)
    best_clustering = _best_row(clustering_benchmark, "silhouette", False)
    summary = {
        "python": platform.python_version(),
        "random_seeds": SEEDS,
        "test_size": TEST_SIZE,
        "benchmark_seed": SEEDS[0],
        "datasets": {
            "classification": {
                "file": "data/drug200.csv",
                "sha256": _sha256(ROOT / "data" / "drug200.csv"),
                "rows": int(len(pd.read_csv(ROOT / "data" / "drug200.csv"))),
                "target": "Drug",
            },
            "regression": {
                "file": "data/regression_sample.csv",
                "sha256": _sha256(ROOT / "data" / "regression_sample.csv"),
                "rows": int(len(regression_data)),
                "target": "ingreso_anual",
            },
            "clustering": {
                "file": "data/Mall_Customers.csv",
                "sha256": _sha256(ROOT / "data" / "Mall_Customers.csv"),
                "rows": int(len(pd.read_csv(ROOT / "data" / "Mall_Customers.csv"))),
                "features": CLUSTER_COLUMNS,
            },
        },
        "selected_models": {
            "classification_benchmark": {
                "model": str(best_classification["Modelo"]),
                "f1_macro": float(best_classification["f1_macro"]),
            },
            "classification_evaluation": {
                "model": "Random Forest (200 árboles)",
                "metrics_by_seed": classification["metrics_table"].to_dict(orient="records"),
                "confusion_matrix_seed_42": classification["confusion_matrix"],
            },
            "regression_benchmark": {
                "model": str(best_regression["Modelo"]),
                "rmse": float(best_regression["RMSE"]),
            },
            "regression_evaluation": {
                "model": "RidgeCV",
                "metrics_by_seed": regression["metrics_table"].to_dict(orient="records"),
                "alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "clustering_benchmark": {
                "configuration": str(best_clustering["Configuración"]),
                "silhouette": float(best_clustering["silhouette"]),
            },
            "clustering_visualization": {
                "model": "K-Means k=3",
                "silhouette": float(clustering["metrics"]["silhouette"]),
                "projection_variance": clustering["projection_variance"],
                "group_sizes": {str(key): int(value) for key, value in clustering["sizes"].items()},
            },
        },
    }
    clasificacion_media = classification["metrics_table"][["accuracy", "f1_macro"]].mean()
    regresion_media = regression["metrics_table"][["RMSE", "R2"]].mean()
    tabla_latex = [
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Tarea y configuración & Resultado principal & Valor \\",
        r"\midrule",
        f"Benchmark de clasificación & {_latex_escape(str(best_classification['Modelo']))} "
        f"(F1 macro) & {best_classification['f1_macro']:.4f} \\\\",
        f"Random Forest, 3 semillas & Exactitud media & {clasificacion_media['accuracy']:.4f} \\\\",
        f"Random Forest, 3 semillas & F1 macro medio & {clasificacion_media['f1_macro']:.4f} \\\\",
        f"Benchmark de regresión & {_latex_escape(str(best_regression['Modelo']))} "
        f"(RMSE) & {best_regression['RMSE']:.2f} \\\\",
        f"RidgeCV, 3 semillas & RMSE medio & {regresion_media['RMSE']:.2f} \\\\",
        f"RidgeCV, 3 semillas & $R^2$ medio & {regresion_media['R2']:.4f} \\\\",
        f"Benchmark de agrupamiento & {_latex_escape(str(best_clustering['Configuración']))} "
        f"(silhouette) & {best_clustering['silhouette']:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (OUTPUT / "resultados_tabla.tex").write_text("\n".join(tabla_latex) + "\n", encoding="utf-8")
    (OUTPUT / "resultados.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Resultados reproducibles generados en {OUTPUT}")
    return summary


if __name__ == "__main__":
    run()
