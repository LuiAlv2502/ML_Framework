import json

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from framework_ia.persistence import AgentStateStore, ConfigStore, ExperimentStore, ModelStore
from framework_ia.utils.exceptions import PersistenceError, ValidationError


# ---------------------------------------------------------------------------
# ExperimentStore
# ---------------------------------------------------------------------------
def _registro(task="classification", nombre="drug200.csv"):
    return {
        "task": task,
        "dataset": {"name": nombre, "hash": "abc"},
        "config": {"algorithm": "knn"},
        "agents": ["clasificador-1"],
        "decisions": [{"decision": "delegar", "justification": "capacidad"}],
        "best_model": "knn",
        "summary": {"f1_macro": 0.9},
    }


def test_experiment_store_guarda_lista_y_recupera(tmp_path):
    store = ExperimentStore(tmp_path / "runs")
    metricas = pd.DataFrame({"Modelo": ["knn"], "f1_macro": [0.9]})
    mensajes = [{"sender": "coordinator", "receiver": "clasificador-1", "performative": "request"}]
    run_id = store.save(_registro(), metrics=metricas, messages=mensajes)

    assert (tmp_path / "runs" / run_id / "run.json").exists()
    assert (tmp_path / "runs" / run_id / "metrics.csv").exists()
    assert (tmp_path / "runs" / run_id / "messages.jsonl").exists()
    assert json.loads((tmp_path / "runs" / "index.json").read_text(encoding="utf-8"))[0]["id"] == run_id

    datos = store.get(run_id)
    assert datos["run"]["best_model"] == "knn"
    assert datos["metrics"]["f1_macro"].iloc[0] == pytest.approx(0.9)
    assert datos["messages"][0]["sender"] == "coordinator"


def test_experiment_store_filtra(tmp_path):
    store = ExperimentStore(tmp_path)
    store.save(_registro("classification", "a.csv"))
    store.save(_registro("regression", "b.csv"))
    assert len(store.list()) == 2
    assert [r["task"] for r in store.list(task="regression")] == ["regression"]
    assert [r["dataset"]["name"] for r in store.list(dataset="a.csv")] == ["a.csv"]
    assert store.list(since="2999-01-01") == []


@pytest.mark.parametrize("formato", ["json", "csv", "html"])
def test_experiment_store_exporta(tmp_path, formato):
    store = ExperimentStore(tmp_path)
    run_id = store.save(_registro(), metrics=pd.DataFrame({"Modelo": ["knn"], "f1_macro": [0.9]}))
    contenido = store.export(run_id, formato, destination=tmp_path / f"salida.{formato}")
    assert contenido and (tmp_path / f"salida.{formato}").read_bytes() == contenido
    if formato == "json":
        assert json.loads(contenido)["run"]["id"] == run_id


def test_experiment_store_errores_y_borrado(tmp_path):
    store = ExperimentStore(tmp_path)
    with pytest.raises(ValidationError):
        store.save({"dataset": {}})
    with pytest.raises(PersistenceError):
        store.get("no-existe")
    run_id = store.save(_registro())
    with pytest.raises(ValidationError):
        store.export(run_id, "pdf")
    store.delete(run_id)
    assert store.list() == []


# ---------------------------------------------------------------------------
# ModelStore
# ---------------------------------------------------------------------------
def test_model_store_guarda_y_carga(tmp_path):
    store = ModelStore(tmp_path)
    modelo = LinearRegression().fit([[0], [1], [2]], [0, 1, 2])
    model_id = store.save(modelo, "lineal", {"task": "regression"})

    cargado = store.load(model_id)
    assert cargado.predict([[3]])[0] == pytest.approx(3)
    assert store.metadata(model_id)["task"] == "regression"
    assert store.list()[0]["id"] == model_id
    store.delete(model_id)
    with pytest.raises(PersistenceError):
        store.load(model_id)


# ---------------------------------------------------------------------------
# AgentStateStore
# ---------------------------------------------------------------------------
def test_agent_state_store_upsert_y_remove(tmp_path):
    store = AgentStateStore(tmp_path / "agents.json")
    assert store.load() == []
    store.upsert({"name": "knn", "agent_type": "classification", "capabilities": ["classification.train"]})
    store.upsert({"name": "knn", "agent_type": "classification", "priority": 5})
    store.upsert({"name": "otro", "agent_type": "regression"})
    assert [d["name"] for d in store.load()] == ["knn", "otro"]
    assert store.load()[0]["priority"] == 5
    assert store.remove("knn") is True
    assert store.remove("knn") is False
    with pytest.raises(ValidationError):
        store.save([{"name": "sin tipo"}])


def test_agent_state_store_archivo_danado(tmp_path):
    ruta = tmp_path / "agents.json"
    ruta.write_text("{no es json", encoding="utf-8")
    with pytest.raises(PersistenceError):
        AgentStateStore(ruta).load()


# ---------------------------------------------------------------------------
# ConfigStore
# ---------------------------------------------------------------------------
def test_config_store_valores_por_defecto_y_merge(tmp_path):
    archivo = tmp_path / "config.json"
    archivo.write_text(json.dumps({"limits": {"max_rows": 10}, "nuevo": 1}), encoding="utf-8")
    config = ConfigStore(archivo, root=tmp_path)
    assert config.get("limits.max_rows") == 10
    assert config.get("limits.max_columns") == 500
    assert config.get("coordinator.routing.classification.capability") == "classification.train"
    assert config.get("no.existe", "x") == "x"
    assert config.path_for("runs_dir") == tmp_path / "runs"
    config.set("limits.max_rows", 99)
    guardado = config.save(tmp_path / "otra.json")
    assert json.loads(guardado.read_text(encoding="utf-8"))["limits"]["max_rows"] == 99


def test_config_store_archivo_invalido(tmp_path):
    archivo = tmp_path / "config.json"
    archivo.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(PersistenceError):
        ConfigStore(archivo, root=tmp_path)


def test_config_store_del_repositorio_es_valido():
    config = ConfigStore()
    assert config.get("random_state") == 42
    assert config.path_for("data_dir").name == "data"
