"""Prueba clave: un agente creado en tiempo de ejecución es descubierto y usado por el coordinador
sin modificar ningún archivo del coordinador ni de la agencia."""
import hashlib
from pathlib import Path
from typing import Any

from framework_ia.agents import BaseAgent, Message, build_default_agency
from framework_ia.agents import _support
from framework_ia.ml.classification import ClasificacionModelos
from framework_ia.utils.validation import validate_target

ROOT = Path(__file__).resolve().parents[1]
NUCLEO = [
    ROOT / "framework_ia" / "agents" / nombre
    for nombre in ("coordinator_agent.py", "agency.py", "registry.py", "agent_factory.py", "messaging.py",
                   "base_agent.py")
]


class KNNOnlyClassifier(BaseAgent):
    """Agente definido fuera del framework: solo entrena KNN con k=1."""

    default_capabilities = frozenset({"classification.train"})

    def _execute(self, message: Message) -> dict[str, Any]:
        datos = self.require_data(message)
        objetivo = validate_target(datos, message.payload["target"], task="classification")
        resultado = _support.ml_call(ClasificacionModelos(datos, target=objetivo).knn, n_neighbors=1)
        f1 = resultado["metricas"]["f1_macro"]
        return {"label": "knn-only k=1", "metrics": {"f1_macro": f1},
                "primary_metric": {"name": "f1_macro", "value": f1, "maximize": True}}


def _huellas() -> dict[Path, str]:
    return {ruta: hashlib.sha256(ruta.read_bytes()).hexdigest() for ruta in NUCLEO}


def test_agente_dinamico_se_usa_sin_tocar_el_nucleo(drug):
    antes = _huellas()
    agencia = build_default_agency(state_store=False, model_store=False, rehydrate=False)

    # 1) Se registra un tipo nuevo y se crea un agente en caliente con mayor prioridad.
    agencia.factory.register_agent_type("knn_only", KNNOnlyClassifier)
    nuevo = agencia.create_agent({"agent_type": "knn_only", "name": "KNN only", "priority": 90}, persist=False)
    assert nuevo.agent_id in agencia.registry
    assert nuevo in agencia.candidates("classification.train")

    # 2) El coordinador lo descubre por capacidad y le delega.
    respuesta = agencia.request("classification", {"data": drug, "target": "Drug", "algorithm": "knn"})
    assert respuesta.ok, respuesta.error
    resultado = respuesta.payload
    assert resultado["selected_agent"] == nuevo.agent_id
    assert resultado["result"]["label"] == "knn-only k=1"
    assert nuevo.agent_id in resultado["decisions"][2]["justification"]

    # 3) Con fan-out, el coordinador consolida entre el agente nuevo y el original.
    fanout = agencia.request("classification", {"data": drug, "target": "Drug", "algorithm": "knn",
                                                "policy": "fanout"}).payload
    assert set(fanout["agents_used"]) == {nuevo.agent_id, "agente-clasificacion"}

    # 4) Ningún archivo del núcleo cambió.
    assert _huellas() == antes


def test_plugin_descubierto_automaticamente_atiende_tareas(drug):
    agencia = build_default_agency(state_store=False, model_store=False, rehydrate=False)
    assert "knn_auto" in agencia.factory.types
    plugin = agencia.create_agent({"agent_type": "knn_auto", "name": "KNN automático", "priority": 95},
                                  persist=False)
    resultado = agencia.request("classification", {"data": drug, "target": "Drug"}).payload
    assert resultado["selected_agent"] == plugin.agent_id
    assert resultado["result"]["params"]["n_neighbors"] in (3, 5, 7, 9, 11)
