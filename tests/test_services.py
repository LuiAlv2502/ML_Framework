import ast
from pathlib import Path

import pandas as pd
import pytest

from framework_ia.persistence import ConfigStore, ExperimentStore
from framework_ia.services import DatasetService, ExperimentService, PreprocessingService, metrics_table
from framework_ia.utils.exceptions import ValidationError

ROOT = Path(__file__).resolve().parents[1]
DRUG = ROOT / "data" / "drug200.csv"


@pytest.fixture
def servicio():
    servicio = DatasetService(ConfigStore())
    servicio.load_path(DRUG)
    return servicio


# ---------------------------------------------------------------------------
# DatasetService
# ---------------------------------------------------------------------------
def test_dataset_service_carga_y_hash(servicio):
    assert servicio.info.rows == 200 and servicio.info.columns == 6
    assert len(servicio.info.hash) == 64
    assert servicio.source == "original"
    assert servicio.active.shape == (200, 6)


def test_dataset_service_lee_csv_con_punto_y_coma_y_latin1():
    servicio = DatasetService(ConfigStore())
    contenido = "año;valor\n2020;1\n2021;2\n".encode("latin-1")
    servicio.load_bytes(contenido, "datos.csv")
    assert servicio.active.columns.tolist() == ["año", "valor"]


@pytest.mark.parametrize(
    ("contenido", "nombre", "mensaje"),
    [
        (b"a,b\n1,2", "datos.txt", "no soportado"),
        (b"", "datos.csv", "vacío"),
        (b"a,b\n", "datos.csv", "vacío"),
    ],
)
def test_dataset_service_rechaza_archivos_invalidos(contenido, nombre, mensaje):
    with pytest.raises(ValidationError, match=mensaje):
        DatasetService(ConfigStore()).load_bytes(contenido, nombre)


def test_dataset_service_sin_datos_da_error_amigable():
    with pytest.raises(ValidationError, match="No hay datos cargados"):
        DatasetService(ConfigStore()).active


def test_dataset_service_activa_y_restaura(servicio):
    servicio.set_active(servicio.active.head(10), "manual", "prueba")
    assert servicio.active.shape[0] == 10 and servicio.source == "manual"
    assert servicio.original.shape[0] == 200
    servicio.restore()
    assert servicio.active.shape[0] == 200 and servicio.source == "original"
    with pytest.raises(ValidationError):
        servicio.set_active(servicio.active, "desconocido")


def test_dataset_service_resumenes(servicio):
    resumen = servicio.summary()
    assert resumen["rows"] == 200 and "Na_to_K" in resumen["numeric"]
    tablas = servicio.eda_tables()
    assert set(tablas) >= {"nulls", "outliers_summary", "distribution", "statistics", "correlation"}
    assert servicio.suggest_target() == "Drug"
    assert servicio.suggest_task("Drug") == "classification"
    assert servicio.suggest_task("Na_to_K") == "regression"
    assert any(a["name"] == "drug200.csv" for a in servicio.list_available())


def test_select_active_conserva_reglas_de_seleccion():
    assert DatasetService.select_active("agente", "o", "m", "a", "h", "h") == "a"
    assert DatasetService.select_active("agente", "o", "m", "a", "h", "otro") == "o"
    assert DatasetService.select_active("manual", "o", "m", None, "h", None) == "m"


# ---------------------------------------------------------------------------
# PreprocessingService
# ---------------------------------------------------------------------------
def test_preprocesamiento_aplica_operaciones_en_orden():
    df = pd.DataFrame({
        "id": range(30), "edad": [1.0, None] * 15, "grupo": ["a", "b", None] * 10, "y": [0, 1] * 15,
    })
    resultado = PreprocessingService().apply(df, [
        {"op": "drop_id_columns"},
        {"op": "impute"},
        {"op": "encode_onehot", "columns": ["grupo"]},
        {"op": "standardize", "columns": ["edad"]},
    ])
    datos = resultado.data
    assert "id" not in datos.columns
    assert datos.isna().sum().sum() == 0
    assert {"grupo_a", "grupo_b"} <= set(datos.columns)
    assert len(resultado.steps) == 4 and resultado.log["operacion"].tolist()[0] == "drop_id_columns"
    assert df["edad"].isna().any()  # la entrada no se modifica


@pytest.mark.parametrize(
    ("operaciones", "mensaje"),
    [
        ([], "al menos una"),
        ([{"op": "magia"}], "desconocida"),
        ([{"op": "drop_columns"}], "requiere"),
        ([{"op": "drop_columns", "columns": ["zzz"]}], "no existen"),
        ([{"op": "impute", "foo": 1}], "no válidos"),
        ([{"op": "convert_type", "column": "x", "dtype": "int64"}], "no se pudo aplicar"),
    ],
)
def test_preprocesamiento_valida(operaciones, mensaje):
    df = pd.DataFrame({"x": ["1", "a"], "z": [1, 2]})
    with pytest.raises(ValidationError, match=mensaje):
        PreprocessingService().apply(df, operaciones)


def test_preprocesamiento_compare_y_filtro():
    antes = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})
    despues = antes.drop(columns="b").fillna(0)
    tabla = PreprocessingService.compare(antes, despues)
    assert tabla.loc[tabla["Columna"] == "b", "Tipo después"].item() == "(eliminada)"
    filtrado = PreprocessingService.filter_preview(pd.DataFrame({"a": [1, 5, 9]}), "a", ">", "4")
    assert filtrado["a"].tolist() == [5, 9]
    with pytest.raises(ValidationError):
        PreprocessingService.filter_preview(antes, "a", "~", 1)


def test_particiones_coinciden_con_las_semillas(servicio):
    particiones = PreprocessingService.partitions(
        servicio.active, task="classification", target="Drug", features=None,
        test_size=0.25, seeds=[1, 2], stratify=True,
    )
    assert [p["seed"] for p in particiones] == [1, 2]
    assert particiones[0]["n_train"] + particiones[0]["n_test"] == 200
    assert particiones[0]["test_index"] != particiones[1]["test_index"]
    with pytest.raises(ValidationError):
        PreprocessingService.partitions(servicio.active, task="regression", target="Drug",
                                        features=None, test_size=0.25, seeds=[1])


# ---------------------------------------------------------------------------
# ExperimentService
# ---------------------------------------------------------------------------
def test_experiment_service_registra_y_compara(tmp_path):
    servicio = ExperimentService(ExperimentStore(tmp_path))
    resultado = {"table": pd.DataFrame({"Modelo": ["a"], "f1_macro": [0.5]}),
                 "primary_metric": {"name": "f1_macro", "value": 0.5}}
    a = servicio.record(task="classification", dataset={"name": "x.csv"}, config={"k": 1}, result=resultado)
    b = servicio.record(task="classification", dataset={"name": "x.csv"}, config={"k": 2},
                        result={"metrics": {"accuracy": 0.7}, "label": "knn"})
    assert len(servicio.list_frame()) == 2
    comparacion = servicio.compare(a, b)
    assert "config: k" in comparacion["Campo"].tolist()
    with pytest.raises(ValidationError):
        servicio.compare(a, a)
    assert metrics_table({"metrics": {"accuracy": 0.7}, "label": "knn"})["Modelo"].item() == "knn"


# ---------------------------------------------------------------------------
# Restricción de arquitectura: los servicios y la persistencia no dependen de Streamlit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("carpeta", ["services", "persistence", "utils", "ml", "agents"])
def test_backend_no_importa_streamlit(carpeta):
    for archivo in (ROOT / "framework_ia" / carpeta).rglob("*.py"):
        arbol = ast.parse(archivo.read_text(encoding="utf-8"))
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                nombres = [alias.name for alias in nodo.names]
            elif isinstance(nodo, ast.ImportFrom):
                nombres = [nodo.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] == "streamlit" for n in nombres), f"{archivo} importa streamlit"
