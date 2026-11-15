"""Construcción de la agencia por defecto: registro + fábrica + bus + coordinador + 4 agentes especializados."""
from __future__ import annotations

from typing import Any

from framework_ia.persistence.agent_state_store import AgentStateStore
from framework_ia.persistence.config_store import ConfigStore
from framework_ia.persistence.model_store import ModelStore

from .agency import Agency

# Importar los módulos registra sus tipos en el catálogo de la fábrica (decorador @agent_type).
from .classification_agent import ClassificationAgent  # noqa: F401
from .clustering_agent import ClusteringAgent  # noqa: F401
from .coordinator_agent import CoordinatorAgent  # noqa: F401
from .eda_agent import EDAAgent  # noqa: F401
from .regression_agent import RegressionAgent  # noqa: F401

COORDINATOR_ID = "coordinador"

DEFAULT_AGENTS: list[dict[str, Any]] = [
    {"agent_type": "eda", "name": "Agente EDA", "agent_id": "agente-eda", "priority": 10},
    {"agent_type": "clustering", "name": "Agente de agrupamiento", "agent_id": "agente-agrupamiento", "priority": 10},
    {"agent_type": "classification", "name": "Agente de clasificación", "agent_id": "agente-clasificacion",
     "priority": 10},
    {"agent_type": "regression", "name": "Agente de regresión", "agent_id": "agente-regresion", "priority": 10},
]


def build_default_agency(
    config: ConfigStore | None = None,
    *,
    state_store: AgentStateStore | None | bool = True,
    model_store: ModelStore | None | bool = True,
    rehydrate: bool = True,
    discover_plugins: bool = True,
) -> Agency:
    """Crea una agencia lista para usar.

    ``state_store``/``model_store`` aceptan una instancia, ``True`` (usar las rutas de la configuración) o
    ``False``/``None`` (sin persistencia, útil en pruebas).
    """
    config = config or ConfigStore()
    if state_store is True:
        state_store = AgentStateStore(config.path_for("agents_state"))
    if model_store is True:
        model_store = ModelStore(config.path_for("models_dir"))
    contexto: dict[str, Any] = {"config": config, "model_store": model_store or None}
    agencia = Agency(
        max_retries=int(config.get("coordinator.max_retries", 1)),
        state_store=state_store or None,
        context=contexto,
    )
    if discover_plugins:
        agencia.factory.discover_plugins()
    agencia.coordinator = agencia.factory.create("coordinator", "Coordinador", agent_id=COORDINATOR_ID, priority=100)
    for definicion in DEFAULT_AGENTS:
        agencia.factory.create_from_definition(definicion)
    agencia.rehydration_errors = {}
    if rehydrate and agencia.state_store is not None:
        _, agencia.rehydration_errors = agencia.factory.rehydrate()
    return agencia


__all__ = ["COORDINATOR_ID", "DEFAULT_AGENTS", "build_default_agency"]
