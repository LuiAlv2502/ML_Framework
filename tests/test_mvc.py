from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Eda.dataframe import DataFrame
from model import DataModel
from supervised import ClasificacionModelos
from unsupervised.unsupervised_base import UnsupervisedBase
from unsupervised.clustering import ClusteringAnalysis
from unsupervised.pca import PCAAnalysis


DATASET = ROOT / "data" / "drug200.csv"


@pytest.fixture
def model():
    return DataModel(DATASET)


def test_cargar_csv_generico(model):
    assert model.dimensiones() == (200, 6)
    assert set(model.nombres_columnas()) == {
        "Age", "Sex", "BP", "Cholesterol", "Na_to_K", "Drug"
    }


def test_restaurar_recupera_datos_originales(model):
    model.eliminar_columna("Drug")
    model.restaurar()

    assert "Drug" in model.nombres_columnas()


def test_imputar_nulos_y_normalizar(tmp_path):
    ruta = tmp_path / "datos.csv"
    pd.DataFrame({"valor": [1.0, None, 3.0], "grupo": ["a", None, "b"]}).to_csv(
        ruta, index=False
    )
    model = DataModel(ruta)
    model.imputar_nulos()
    model.normalizar()

    assert model.contar_nulos().sum() == 0
    assert model.obtener_columna("valor").between(0, 1).all()


def test_pca_devuelve_dos_componentes(model):
    resultado = model.pca(n_components=2)

    assert resultado["componentes"].shape == (200, 2)
    assert len(resultado["varianza_explicada"]) == 2
    assert resultado["varianza_acumulada"][-1] <= 1


def test_pca_y_clustering_separados(model):
    assert issubclass(PCAAnalysis, UnsupervisedBase)
    assert issubclass(ClusteringAnalysis, UnsupervisedBase)
    assert isinstance(model.dataframe, PCAAnalysis)
    assert isinstance(model.dataframe, ClusteringAnalysis)
    assert PCAAnalysis(model.mostrar()).pca()["componentes"].shape == (200, 2)
    assert ClusteringAnalysis(model.mostrar()).kmeans()["resultado"]["cluster"].nunique() == 3


@pytest.mark.parametrize("algoritmo", ["hac", "kmeans"])
def test_clustering_asigna_grupo_y_metrica(model, algoritmo):
    resultado = getattr(model, algoritmo)(n_clusters=3)

    assert "cluster" in resultado["resultado"].columns
    assert resultado["resultado"]["cluster"].nunique() == 3
    assert -1 <= resultado["silhouette"] <= 1


def test_kmeans_reporta_inercia(model):
    resultado = model.kmeans(n_clusters=3)

    assert resultado["inercia"] > 0


@pytest.mark.parametrize(
    ("metodo", "parametros"),
    [
        ("tsne", {"perplexity": 5.0}),
        ("umap_embedding", {"n_neighbors": 5}),
    ],
)
def test_embeddings_tienen_dos_dimensiones(model, metodo, parametros):
    embedding = getattr(model, metodo)(**parametros)

    assert embedding.shape == (200, 2)


def test_pca_rechaza_dataframe_vacio():
    datos = PCAAnalysis(pd.DataFrame())

    with pytest.raises(ValueError, match="No hay datos"):
        datos.pca()


def test_clasificacion_knn_preprocesa_categorias_y_evalua_multiclase(model):
    clasificador = ClasificacionModelos(model.mostrar(), target="Drug")
    split = clasificador.prepare_data()

    assert len(split.X_train) + len(split.X_test) == 200
    assert "Drug" not in split.X_train.columns
    assert set(split.y_train) == set(split.y_test)

    resultado = clasificador.knn(n_neighbors=3)

    assert 0 <= resultado["metricas"]["accuracy"] <= 1
    assert len(resultado["metricas"]["confusion_matrix"]) == 5
    assert resultado["model"].predict(split.X_test).shape == split.y_test.shape


def test_matriz_confusion_expone_etiquetas_en_el_mismo_orden():
    datos = pd.DataFrame(
        {
            "predictor": range(6),
            "objetivo": ["a", "b", "c", "a", "b", "c"],
        }
    )
    clasificador = ClasificacionModelos(datos, target="objetivo")

    metricas = clasificador.evaluar_clasificacion(
        ["a", "b", "a"],
        ["a", "b", "b"],
    )

    assert metricas["confusion_matrix_labels"] == ["a", "b"]
    assert len(metricas["confusion_matrix"]) == len(
        metricas["confusion_matrix_labels"]
    )
    assert all(
        len(fila) == len(metricas["confusion_matrix_labels"])
        for fila in metricas["confusion_matrix"]
    )


def test_clasificacion_xgboost_admite_etiquetas_categoricas(model):
    clasificador = ClasificacionModelos(model.mostrar(), target="Drug")

    resultado = clasificador.xgboost(n_estimators=10)

    assert 0 <= resultado["metricas"]["accuracy"] <= 1


@pytest.mark.parametrize(
    ("metodo", "parametros", "atributos_esperados"),
    [
        ("knn", {"n_neighbors": 3}, {"n_neighbors": 3}),
        (
            "decision_tree",
            {"min_samples_split": 4, "max_depth": 5},
            {"min_samples_split": 4, "max_depth": 5},
        ),
        (
            "random_forest",
            {"n_estimators": 25, "max_depth": 5},
            {"n_estimators": 25, "max_depth": 5},
        ),
        (
            "gradient_boosting",
            {"n_estimators": 25, "max_depth": 1},
            {"n_estimators": 25, "max_depth": 1},
        ),
        ("adaboost", {"n_estimators": 25}, {"n_estimators": 25}),
        (
            "xgboost",
            {"n_estimators": 10, "max_depth": 2},
            {"n_estimators": 10, "max_depth": 2},
        ),
    ],
)
def test_clasificacion_aplica_variaciones_de_parametros(
    model, metodo, parametros, atributos_esperados
):
    clasificador = ClasificacionModelos(model.mostrar(), target="Drug")

    resultado = getattr(clasificador, metodo)(**parametros)
    estimador = resultado["model"].named_steps["modelo"]
    if metodo == "xgboost":
        estimador = estimador.estimator_

    for atributo, esperado in atributos_esperados.items():
        assert getattr(estimador, atributo) == esperado
    assert 0 <= resultado["metricas"]["accuracy"] <= 1


def test_clasificacion_grid_adaboost_acepta_parametros_sin_prefijo(model):
    clasificador = ClasificacionModelos(model.mostrar(), target="Drug")

    resultado = clasificador.adaboost_grid(
        {"n_estimators": [10, 20]},
        cv=2,
    )

    assert resultado["best_params"]["n_estimators"] in {10, 20}
    assert 0 <= resultado["metricas"]["accuracy"] <= 1


def test_comparar_basico_incluye_modelos_del_trabajo(model):
    clasificador = ClasificacionModelos(model.mostrar(), target="Drug")

    resultados = clasificador.comparar_basico()

    assert set(resultados) == {
        "knn",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "adaboost",
        "xgboost",
    }
    assert all("metricas" in resultado for resultado in resultados.values())