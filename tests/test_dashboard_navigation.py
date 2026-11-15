from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from framework_ia.controller import FrameworkController

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
PAGINAS = (
    ("home.py", "Inicio"),
    ("data.py", "Datos"),
    ("eda.py", "EDA"),
    ("preprocessing.py", "Preprocesamiento"),
    ("unsupervised.py", "Agrupamiento"),
    ("classification.py", "Clasificación"),
    ("prediction.py", "Regresión"),
    ("agents.py", "MAS"),
    ("case04.py", "Caso 04"),
    ("experiments.py", "Experimentos"),
)


def test_navegacion_registra_paginas_y_abre_inicio():
    prueba = AppTest.from_file(str(APP)).run()

    assert not prueba.exception
    registro = list(prueba._registered_pages.values())
    assert [pagina["page_name"] for pagina in registro] == [nombre for _, nombre in PAGINAS]
    assert [Path(pagina["script_path"]).name for pagina in registro] == [archivo for archivo, _ in PAGINAS]
    assert prueba.title[0].value == "Laboratorio de análisis de datos"


@pytest.mark.parametrize(("archivo", "_nombre"), PAGINAS)
def test_cada_pagina_abre_con_drug200(archivo, _nombre, tmp_path):
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)
    controlador.load_sample("drug200.csv")
    prueba = AppTest.from_file(str(APP)).run()
    prueba.session_state["controller"] = controlador

    prueba.switch_page(f"frontend/pages/{archivo}").run()

    assert not prueba.exception
