from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from scripts.generate_regression_sample import generate
from scripts.package_project import build_archive


def test_generador_regresion_es_reproducible(tmp_path):
    primero = tmp_path / "primero.csv"
    segundo = tmp_path / "segundo.csv"

    generate(primero, rows=12, seed=7)
    generate(segundo, rows=12, seed=7)

    assert primero.read_bytes() == segundo.read_bytes()
    assert primero.read_text(encoding="utf-8").count("\n") == 13


def test_dataset_sintetico_de_regresion_incluye_objetivo():
    dataset = Path(__file__).resolve().parents[1] / "data" / "regression_sample.csv"
    tabla = pd.read_csv(dataset)

    assert len(tabla) == 300
    assert "ingreso_anual" in tabla.columns


def test_paquete_excluye_entornos_y_archivos_generados(tmp_path):
    raiz = tmp_path / "proyecto"
    (raiz / "src").mkdir(parents=True)
    (raiz / ".venv" / "Scripts").mkdir(parents=True)
    (raiz / "runs" / "run-1").mkdir(parents=True)
    (raiz / "src" / "app.py").write_text("print('ok')", encoding="utf-8")
    (raiz / ".venv" / "Scripts" / "python.exe").write_bytes(b"fake")
    (raiz / "runs" / "run-1" / "run.json").write_text("{}", encoding="utf-8")
    (raiz / "secrets.toml").write_text("token = 'no incluir'", encoding="utf-8")
    (raiz / "framework_ia" / "persistence").mkdir(parents=True)
    (raiz / "framework_ia" / "persistence" / "agents.json").write_text("[]", encoding="utf-8")
    destino = build_archive(tmp_path / "entrega.zip", root=raiz)

    with ZipFile(destino) as archivo:
        nombres = archivo.namelist()

    assert nombres == ["src/app.py"]
