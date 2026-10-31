from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import AgenteEDA


def _ejecutar(df, objetivo=None):
    agente = AgenteEDA(df, objetivo=objetivo).ejecutar()
    agente.cerrar_figuras()
    return agente


def test_agente_no_modifica_los_datos_originales():
    df = pd.read_csv(ROOT / "data" / "diabetes.csv")
    copia = df.copy()

    _ejecutar(df, objetivo="Outcome")

    pd.testing.assert_frame_equal(df, copia)


def test_agente_detecta_ceros_que_son_datos_faltantes():
    df = pd.read_csv(ROOT / "data" / "diabetes.csv")

    agente = _ejecutar(df, objetivo="Outcome")

    sospechosos = agente.percepcion["ceros_sospechosos"]
    assert {"Glucose", "BloodPressure", "Insulin"} <= set(sospechosos)
    assert "Pregnancies" not in sospechosos  # 0 embarazos es un valor válido


def test_agente_omite_pasos_innecesarios_con_su_razon():
    df = pd.read_csv(ROOT / "data" / "drug200.csv")

    agente = _ejecutar(df)

    omitidos = {p["paso"]: p["razon"] for p in agente.plan if not p["ejecutar"]}
    assert "Tratar valores faltantes" in omitidos
    assert "Analizar la variable objetivo" in omitidos
    assert all(razon for razon in omitidos.values())


def test_agente_elimina_identificadores_en_la_copia():
    df = pd.DataFrame({"id": range(40), "x": np.arange(40) % 7, "y": np.arange(40) % 5 * 1.5})

    agente = _ejecutar(df)

    assert "id" not in agente.datos_limpios.columns  # identificador eliminado
    assert "id" in df.columns  # el original no cambia


def test_agente_elimina_duplicados_en_la_copia():
    df = pd.DataFrame({"x": np.arange(40) % 7, "y": np.arange(40) * 1.5})
    df = pd.concat([df, df.iloc[[1, 2]]], ignore_index=True)  # 2 filas duplicadas

    agente = _ejecutar(df)

    assert len(agente.datos_limpios) == 40
    assert len(df) == 42  # el original no cambia


def test_agente_identifica_tipo_de_problema():
    df = pd.read_csv(ROOT / "data" / "drug200.csv")

    agente = _ejecutar(df, objetivo="Drug")

    paso_objetivo = agente.pasos[-1]
    assert "clasificación" in paso_objetivo.observacion


@pytest.mark.parametrize("archivo", ["drug200.csv", "Mall_Customers.csv", "water_potability.csv"])
def test_agente_genera_informe_html(archivo, tmp_path):
    df = pd.read_csv(ROOT / "data" / archivo)
    agente = AgenteEDA(df).ejecutar()

    ruta = agente.guardar_informe(tmp_path / "informe.html")
    contenido = Path(ruta).read_text(encoding="utf-8")
    agente.cerrar_figuras()

    assert "Bitácora de razonamiento" in contenido
    assert "data:image/png;base64" in contenido
    assert len(agente.pasos) >= 4


def test_agente_rechaza_objetivo_inexistente():
    df = pd.read_csv(ROOT / "data" / "drug200.csv")

    with pytest.raises(KeyError):
        AgenteEDA(df, objetivo="no_existe")


