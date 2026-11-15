from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from framework_ia.controller import FrameworkController

ROOT = Path(__file__).resolve().parents[1]
NEW_PAGES = ("home.py", "data.py", "preprocessing.py", "agents.py", "experiments.py")


@pytest.mark.parametrize("filename", NEW_PAGES)
def test_paginas_nuevas_abren_con_un_dataset(tmp_path, filename):
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)
    controlador.load_dataframe(
        pd.DataFrame({"grupo": ["a", "b", "a", "c"], "valor": [1.0, 2.0, None, 4.0]}),
        "prueba.csv",
    )
    prueba = AppTest.from_file(str(ROOT / "frontend" / "pages" / filename), default_timeout=15)
    prueba.session_state["controller"] = controlador

    prueba.run()

    assert not prueba.exception


def test_controlador_permite_crear_y_quitar_agente_dinamico(tmp_path):
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)

    creado = controlador.create_agent({
        "agent_type": "knn_auto",
        "name": "KNN dinámico de prueba",
        "capabilities": ["classification.train"],
        "config": {"k_max": 9},
    }, persist=False)

    assert creado["agent_id"] in {a["agent_id"] for a in controlador.list_agents()}
    assert creado["config"] == {"k_max": 9}
    controlador.remove_agent(creado["agent_id"])
    assert creado["agent_id"] not in {a["agent_id"] for a in controlador.list_agents()}


def test_pagina_experimentos_presenta_un_registro_guardado(tmp_path):
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)
    controlador.load_dataframe(pd.DataFrame({"x": [1, 2, 3]}), "datos.csv")
    controlador.experiments.record(
        task="classification",
        dataset=controlador.dataset_info(),
        config={"algorithm": "knn"},
        result={"metrics": {"accuracy": 0.9}},
    )
    prueba = AppTest.from_file(str(ROOT / "frontend" / "pages" / "experiments.py"), default_timeout=15)
    prueba.session_state["controller"] = controlador

    prueba.run()

    assert not prueba.exception
    assert any("classification" in str(element.value) for element in prueba.dataframe)
