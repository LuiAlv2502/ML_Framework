import pandas as pd
import pytest

from framework_ia.controller import FrameworkController
from framework_ia.utils.exceptions import ValidationError


def test_demostracion_caso04_ejecuta_escenarios_y_limpia_el_agente_temporal(drug, tmp_path):
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)
    controlador.load_dataframe(drug, "drug200.csv")
    datos_antes = controlador.active_data()
    agente_usuario = controlador.create_agent(
        {"agent_type": "knn_auto", "name": "KNN personalizado", "config": {"k_max": 7}},
        persist=False,
    )

    resultado = controlador.run_case04_evaluation("Drug", seeds=[42, 43])

    assert [escenario["complexity"] for escenario in resultado["scenarios"]] == ["Baja", "Media", "Alta"]
    assert resultado["scenarios"][0]["scenario"] == "Agrupamiento inicial"
    assert resultado["scenarios"][0]["quality_metric"] == "silhouette"
    assert all(escenario["run_ids"] for escenario in resultado["scenarios"])
    assert [escenario["trials"] for escenario in resultado["scenarios"]] == [1, 2, 2]
    assert len(resultado["comparison"]["trials"]) == 4
    assert resultado["comparison"]["centralized"]["trials"] == 2
    assert resultado["comparison"]["multiagent"]["trials"] == 2
    assert resultado["comparison"]["centralized"]["interactions"] == 0
    assert resultado["comparison"]["multiagent"]["interactions"] > 0
    assert resultado["comparison"]["centralized"]["quality_metric"] == "f1_macro"
    assert resultado["comparison"]["multiagent"]["quality_metric"] == "f1_macro"
    assert resultado["comparison"]["centralized"]["quality"] == pytest.approx(
        resultado["comparison"]["multiagent"]["quality"]
    )
    assert not any(agente["name"].startswith("Caso04 KNN automático") for agente in controlador.list_agents())
    pd.testing.assert_frame_equal(controlador.active_data(), datos_antes)

    centralizado = controlador.get_experiment(
        resultado["comparison"]["centralized"]["run_ids"][0]
    )["run"]
    agrupamiento = controlador.get_experiment(resultado["scenarios"][0]["run_ids"][0])["run"]
    fanout = controlador.get_experiment(
        resultado["comparison"]["multiagent"]["run_ids"][0]
    )["run"]
    assert agrupamiento["task"] == "clustering.kmeans"
    assert "agente-agrupamiento" in agrupamiento["agents"]
    assert centralizado["task"] == "case04.centralized_classification"
    assert fanout["coordination"]["policy"] == "fanout"
    assert "agente-clasificacion" in fanout["agents"]
    assert any(agente.startswith("knn_auto-") for agente in fanout["agents"])
    assert agente_usuario["agent_id"] not in fanout["agents"]
    assert any(agente["agent_id"] == agente_usuario["agent_id"] for agente in controlador.list_agents())


def test_demostracion_caso04_rechaza_menos_de_cinco_filas_de_entrenamiento(drug, tmp_path):
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)
    controlador.load_dataframe(drug.head(4), "muestra_pequena.csv")

    with pytest.raises(ValidationError, match="5 filas de entrenamiento"):
        controlador.run_case04_evaluation("Drug")


def test_demostracion_caso04_rechaza_semillas_vacias(drug, tmp_path):
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)
    controlador.load_dataframe(drug, "drug200.csv")

    with pytest.raises(ValidationError, match="al menos una semilla"):
        controlador.run_case04_evaluation("Drug", seeds=[])
