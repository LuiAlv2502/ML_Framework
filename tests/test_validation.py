import numpy as np
import pandas as pd
import pytest

from framework_ia.utils.exceptions import (
    CapabilityNotFoundError,
    FrameworkError,
    ValidationError,
    friendly_message,
)
from framework_ia.utils.metrics import (
    classification_metrics,
    clustering_metrics,
    cophenetic_correlation,
    is_better,
    regression_metrics,
)
from framework_ia.utils.serialization import dataframe_from_jsonable, dumps, summarize, to_jsonable
from framework_ia.utils.validation import (
    detect_encoding,
    suggest_task_type,
    validate_capability,
    validate_dataframe,
    validate_file,
    validate_name,
    validate_params,
    validate_range,
    validate_target,
    validate_task_type,
)


# ---------------------------------------------------------------------------
# Validación de DataFrames
# ---------------------------------------------------------------------------
def test_validate_dataframe_acepta_datos_validos():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert validate_dataframe(df) is df


@pytest.mark.parametrize(
    ("datos", "mensaje"),
    [
        ([1, 2, 3], "pandas.DataFrame"),
        (pd.DataFrame(), "vacío"),
        (pd.DataFrame({"a": range(11)}), "máximo permitido"),
    ],
)
def test_validate_dataframe_rechaza_entradas_invalidas(datos, mensaje):
    with pytest.raises(ValidationError, match=mensaje):
        validate_dataframe(datos, max_rows=10)


def test_validate_dataframe_rechaza_columnas_duplicadas():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValidationError, match="repetido"):
        validate_dataframe(df)


def test_validation_error_es_value_error_para_compatibilidad():
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame())


# ---------------------------------------------------------------------------
# Objetivo y tipo de tarea
# ---------------------------------------------------------------------------
def test_validate_target_por_tarea():
    df = pd.DataFrame({"x": range(40), "clase": ["a", "b"] * 20, "texto": ["t"] * 40})
    assert validate_target(df, "clase", task="classification") == "clase"
    assert validate_target(df, "x", task="regression") == "x"
    with pytest.raises(ValidationError, match="no existe"):
        validate_target(df, "zzz")
    with pytest.raises(ValidationError, match="no es numérica"):
        validate_target(df, "clase", task="regression")
    with pytest.raises(ValidationError, match="una sola clase"):
        validate_target(df, "texto", task="classification")
    with pytest.raises(ValidationError, match="parece continua"):
        validate_target(df, "x", task="classification")


def test_validate_task_type():
    assert validate_task_type("clustering") == "clustering"
    with pytest.raises(ValidationError, match="inválido"):
        validate_task_type("magia")


def test_suggest_task_type():
    assert suggest_task_type(pd.Series(np.linspace(0, 1, 50))) == "regression"
    assert suggest_task_type(pd.Series(["a", "b"] * 25)) == "classification"


# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
SPEC = {
    "n": ("entero", 1, 10),
    "alpha": ("real", 0.0, 1.0),
    "depth": ("entero_o_nulo", 1, 5),
    "kernel": ("opcion", ["rbf", "linear"]),
    "flag": ("booleano",),
    "grid": ("lista_enteros", 1, 100),
}


def test_validate_params_limpia_y_convierte():
    limpios = validate_params(
        {"n": 3.0, "alpha": 1, "depth": 0, "kernel": "rbf", "flag": True, "grid": [10, 20]}, SPEC
    )
    assert limpios == {"n": 3, "alpha": 1.0, "depth": None, "kernel": "rbf", "flag": True, "grid": [10, 20]}


@pytest.mark.parametrize(
    ("params", "mensaje"),
    [
        ({"zzz": 1}, "no permitidos"),
        ({"n": 0}, "entre 1 y 10"),
        ({"n": 2.5}, "entero"),
        ({"alpha": "x"}, "numérico"),
        ({"kernel": "poly"}, "uno de"),
        ({"flag": "si"}, "verdadero o falso"),
        ({"grid": [5, 5]}, "repetirse"),
        ({"grid": []}, "lista no vacía"),
    ],
)
def test_validate_params_rechaza_valores_invalidos(params, mensaje):
    with pytest.raises(ValidationError, match=mensaje):
        validate_params(params, SPEC)


def test_validate_range_booleano_no_es_numero():
    with pytest.raises(ValidationError):
        validate_range("x", True, 0, 1)


# ---------------------------------------------------------------------------
# Archivos, capacidades y nombres
# ---------------------------------------------------------------------------
def test_validate_file_extension_y_tamano():
    assert validate_file("datos.CSV", 100) == ".csv"
    assert validate_file("libro.xlsx") == ".xlsx"
    with pytest.raises(ValidationError, match="no soportado"):
        validate_file("datos.txt")
    with pytest.raises(ValidationError, match="vacío"):
        validate_file("datos.csv", 0)
    with pytest.raises(ValidationError, match="máximo"):
        validate_file("datos.csv", 3 * 1024 * 1024, max_mb=1)


def test_detect_encoding():
    assert detect_encoding("a,b\n1,2".encode("utf-8")) == "utf-8"
    assert detect_encoding("año,b\n1,2".encode("latin-1")) == "latin-1"


@pytest.mark.parametrize("capacidad", ["classification.train", "eda.profile", "clustering.k_means"])
def test_validate_capability_valida(capacidad):
    assert validate_capability(capacidad) == capacidad


@pytest.mark.parametrize("capacidad", ["Classification.train", "train", "eda.", "eda profile", 3])
def test_validate_capability_invalida(capacidad):
    with pytest.raises(ValidationError):
        validate_capability(capacidad)


def test_validate_name():
    assert validate_name("  Agente KNN-1 ") == "Agente KNN-1"
    with pytest.raises(ValidationError):
        validate_name("x")
    with pytest.raises(ValidationError):
        validate_name("<script>")


# ---------------------------------------------------------------------------
# Excepciones
# ---------------------------------------------------------------------------
def test_friendly_message():
    assert "classification.train" in friendly_message(CapabilityNotFoundError("classification.train"))
    assert friendly_message(KeyError("col")) == "Dato no encontrado: col"
    assert friendly_message(RuntimeError("")) == "RuntimeError"
    assert isinstance(CapabilityNotFoundError("x.y"), FrameworkError)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def test_metricas_clasificacion_y_regresion():
    clasif = classification_metrics(["a", "b", "a"], ["a", "b", "b"])
    assert clasif["accuracy"] == pytest.approx(2 / 3)
    assert clasif["confusion_matrix_labels"] == ["a", "b"]
    reg = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert reg["MAE"] == pytest.approx(1 / 3)
    assert set(reg) == {"RMSE", "MAE", "ER", "R2"}


def test_metricas_agrupamiento():
    rng = np.random.default_rng(0)
    matriz = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    etiquetas = [0] * 20 + [1] * 20
    metricas = clustering_metrics(matriz, etiquetas)
    assert metricas["silhouette"] > 0.9
    assert metricas["davies_bouldin"] < 0.1
    assert np.isnan(clustering_metrics(matriz, [0] * 40)["silhouette"])
    assert 0 < cophenetic_correlation(matriz) <= 1


def test_is_better():
    assert is_better(0.9, 0.8, maximize=True)
    assert is_better(1.0, 2.0, maximize=False)
    assert not is_better(float("nan"), 0.1, maximize=True)
    assert is_better(0.1, None, maximize=True)


# ---------------------------------------------------------------------------
# Serialización
# ---------------------------------------------------------------------------
def test_to_jsonable_dataframe_ida_y_vuelta():
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan]})
    datos = to_jsonable({"tabla": df, "arr": np.array([1, 2]), "x": np.float64(1.5), "s": {1}})
    assert datos["arr"] == [1, 2] and datos["x"] == 1.5 and datos["s"] == [1]
    assert datos["tabla"]["data"][1][1] is None
    reconstruida = dataframe_from_jsonable(datos["tabla"])
    assert reconstruida.shape == (2, 2)
    assert '"tabla"' in dumps({"tabla": df})


def test_summarize_no_incluye_filas():
    resumen = summarize({"data": pd.DataFrame({"a": range(1000)}), "texto": "x" * 1000})
    assert resumen["data"]["tabla"] == "1000 filas × 1 columnas"
    assert len(resumen["texto"]) < 400
