from pathlib import Path

import pandas as pd

from framework_ia.controller import FrameworkController
from framework_ia.ml.clustering.clustering import ClusteringAnalysis
from framework_ia.ml.clustering.pca import PCAAnalysis

ROOT = Path(__file__).resolve().parents[1]


def test_ridge_cv_recibe_los_alphas_elegidos_desde_el_controlador(tmp_path):
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)
    datos = pd.read_csv(ROOT / "data" / "regression_sample.csv")
    controlador.load_dataframe(datos, "regression_sample.csv")
    alphas = [0.01, 0.1, 1.0, 10.0]

    resultado = controlador.run_regression(
        target="ingreso_anual",
        algorithm="ridge_cv",
        params={"alphas": alphas},
        test_size=0.2,
    )

    detalle = resultado["result"]
    assert detalle["params"]["alphas"] == alphas
    assert detalle["experiments"][0]["alpha_optimo"] in alphas


def test_kmeans_transmite_init_y_n_init_hasta_el_algoritmo(monkeypatch, tmp_path):
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)
    datos = pd.read_csv(ROOT / "data" / "Mall_Customers.csv")
    controlador.load_dataframe(datos, "Mall_Customers.csv")
    original = ClusteringAnalysis.kmeans
    recibidos = {}

    def capturar(self, **params):
        recibidos.update(params)
        return original(self, **params)

    monkeypatch.setattr(ClusteringAnalysis, "kmeans", capturar)
    controlador.run_clustering(
        "kmeans",
        {"n_clusters": 3, "init": "random", "n_init": 4},
        columns=["Annual Income (k$)", "Spending Score (1-100)"],
    )

    assert recibidos["init"] == "random"
    assert recibidos["n_init"] == 4


def test_pca_transmite_whiten_y_svd_solver_hasta_el_algoritmo(monkeypatch, tmp_path):
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)
    datos = pd.read_csv(ROOT / "data" / "Mall_Customers.csv")
    controlador.load_dataframe(datos, "Mall_Customers.csv")
    original = PCAAnalysis.pca
    recibidos = {}

    def capturar(self, **params):
        recibidos.update(params)
        return original(self, **params)

    monkeypatch.setattr(PCAAnalysis, "pca", capturar)
    controlador.run_clustering(
        "pca",
        {"n_components": 2, "whiten": True, "svd_solver": "full"},
        columns=["Annual Income (k$)", "Spending Score (1-100)"],
    )

    assert recibidos["whiten"] is True
    assert recibidos["svd_solver"] == "full"
