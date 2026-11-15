from pathlib import Path

import pandas as pd

from framework_ia.controller import FrameworkController

ROOT = Path(__file__).resolve().parents[1]


def _verificar_tarea(controlador: FrameworkController, resultado: dict) -> None:
    assert resultado["selected_agent"] in resultado["agents_used"]
    assert resultado["candidates"]
    assert resultado["decisions"]
    assert resultado["messages"]
    assert resultado["run_id"]
    assert isinstance(resultado["result"], dict)
    assert controlador.get_experiment(resultado["run_id"])["run"]["status"] == "ok"


def test_flujos_eda_ml_benchmarks_y_persistencia_sin_llm(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)

    controlador.load_sample("drug200.csv")
    eda = controlador.run_eda(target="Drug", use_llm=False, activate=False)
    _verificar_tarea(controlador, eda)

    controlador.load_sample("Mall_Customers.csv")
    agrupamiento = controlador.run_clustering(
        "kmeans",
        {"n_clusters": 3, "init": "k-means++", "n_init": 3},
        columns=["Annual Income (k$)", "Spending Score (1-100)"],
    )
    _verificar_tarea(controlador, agrupamiento)
    assert len(agrupamiento["result"]["labels"]) == 200

    controlador.load_sample("drug200.csv")
    clasificacion = controlador.run_classification(
        target="Drug", algorithm="knn", params={"n_neighbors": 3}, seeds=[42], test_size=0.25,
    )
    _verificar_tarea(controlador, clasificacion)
    assert "accuracy" in clasificacion["result"]["metrics"]

    datos_regresion = pd.read_csv(ROOT / "data" / "regression_sample.csv")
    controlador.load_dataframe(datos_regresion, "regression_sample.csv")
    regresion = controlador.run_regression(
        target="ingreso_anual", algorithm="ridge", params={"alpha": 1.0}, seeds=[42], test_size=0.25,
    )
    _verificar_tarea(controlador, regresion)
    assert "RMSE" in regresion["result"]["metrics"]

    controlador.load_sample("drug200.csv")
    benchmark_clasificacion = controlador.run_benchmark(
        "classification",
        {
            "target": "Drug",
            "seeds": [42],
            "test_size": 0.25,
            "configs": [{"algorithm": "knn", "params": {"n_neighbors": 3}, "label": "KNN prueba"}],
        },
    )
    _verificar_tarea(controlador, benchmark_clasificacion)
    assert not benchmark_clasificacion["result"]["table"].empty

    controlador.load_dataframe(datos_regresion, "regression_sample.csv")
    benchmark_regresion = controlador.run_benchmark(
        "regression",
        {
            "target": "ingreso_anual",
            "seeds": [42],
            "test_size": 0.25,
            "configs": [{"algorithm": "ridge", "params": {"alpha": 1.0}, "label": "Ridge prueba"}],
        },
    )
    _verificar_tarea(controlador, benchmark_regresion)
    assert not benchmark_regresion["result"]["table"].empty

    controlador.load_sample("Mall_Customers.csv")
    benchmark_agrupamiento = controlador.run_benchmark(
        "clustering",
        {"configs": [{"algorithm": "kmeans", "label": "K-Means k=3", "params": {"n_clusters": 3}}]},
    )
    _verificar_tarea(controlador, benchmark_agrupamiento)
    assert not benchmark_agrupamiento["result"]["table"].empty

    historial = controlador.list_experiments()
    assert len(historial) == 7
    assert len(controlador.list_messages()) > 0
