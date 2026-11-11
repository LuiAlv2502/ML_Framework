from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import DataModel
from pages import _shared


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


def test_restaurar_csv_original_reemplaza_cambios_manuales(monkeypatch):
    original = _modelo({"x": [1, 2]})
    monkeypatch.setattr(
        _shared.st,
        "session_state",
        {"model": original, "modelo_manual": _modelo({"x": [9]}), "modo_datos_activos": "manual"},
    )

    _shared.restaurar_csv_original()

    assert _shared.st.session_state["modo_datos_activos"] == "original"
    assert _shared.st.session_state["modelo_manual"].dataframe.dataframe["x"].tolist() == [1, 2]
    assert original.dataframe.dataframe["x"].tolist() == [1, 2]
