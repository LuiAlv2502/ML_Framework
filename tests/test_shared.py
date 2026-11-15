from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from framework_ia.ml import DataModel
from frontend.components import shared as _shared


def _modelo(datos):
    modelo = DataModel()
    modelo.cargar_dataframe(pd.DataFrame(datos))
    return modelo


def test_seleccionar_modelo_original_no_toma_cambios_manuales_ni_del_agente():
    original = _modelo({"x": [1, 2]})
    manual = _modelo({"x": [1, 2]})
    agente = _modelo({"x": [100, 200]})

    activo = _shared.seleccionar_modelo_activo(
        "original", original, manual, agente, "hash", "hash"
    )

    assert activo.dataframe.dataframe["x"].tolist() == [1, 2]
    assert activo is manual


def test_seleccionar_modelo_manual_usa_los_cambios_manuales():
    original = _modelo({"x": [1, 2]})
    manual = _modelo({"x": [10, 20]})

    activo = _shared.seleccionar_modelo_activo(
        "manual", original, manual, None, "hash", None
    )

    assert activo.dataframe.dataframe["x"].tolist() == [10, 20]


def test_seleccionar_modelo_agente_requiere_hash_del_csv_actual():
    original = _modelo({"x": [1, 2]})
    manual = _modelo({"x": [10, 20]})
    agente = _modelo({"x": [100, 200]})

    activo = _shared.seleccionar_modelo_activo(
        "agente", original, manual, agente, "csv-actual", "csv-anterior"
    )

    assert activo is original


def test_seleccionar_modelo_agente_usa_preparacion_vigente():
    original = _modelo({"x": [1, 2]})
    manual = _modelo({"x": [10, 20]})
    agente = _modelo({"x": [100, 200]})

    activo = _shared.seleccionar_modelo_activo(
        "agente", original, manual, agente, "mismo-csv", "mismo-csv"
    )

    assert activo is agente


def test_restaurar_csv_original_reemplaza_cambios_manuales(monkeypatch, tmp_path):
    from framework_ia.controller import FrameworkController

    controlador = FrameworkController(persist=False, runs_dir=tmp_path)
    controlador.load_dataframe(pd.DataFrame({"x": [1, 2]}), "datos.csv")
    controlador.datasets.set_active(pd.DataFrame({"x": [9]}), "manual")
    monkeypatch.setattr(_shared.st, "session_state", {"controller": controlador})

    _shared.restaurar_csv_original()

    assert controlador.dataset_info()["source"] == "original"
    assert controlador.active_data()["x"].tolist() == [1, 2]
