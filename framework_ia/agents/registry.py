"""Registro (comunidad) de agentes: alta, baja, consulta y búsqueda por capacidad.

Es seguro ante hilos y observable: cada cambio se notifica a los suscriptores (el dashboard, la bitácora).
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Callable, Iterable

from framework_ia.utils.exceptions import AgentError
from framework_ia.utils.logging_config import get_logger

from .base_agent import AVAILABLE_STATUSES, AgentStatus, BaseAgent

logger = get_logger("agents.registry")
RegistryListener = Callable[[dict[str, Any]], None]


class AgentRegistry:
    """Comunidad de agentes indexada por ``agent_id``."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._listeners: list[RegistryListener] = []
        self._events: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Observadores
    # ------------------------------------------------------------------
    def subscribe(self, listener: RegistryListener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def _notificar(self, evento: str, agente: BaseAgent, **extra: Any) -> None:
        registro = {"timestamp": datetime.now().isoformat(timespec="seconds"), "event": evento,
                    "agent_id": agente.agent_id, "name": agente.name, **extra}
        with self._lock:
            self._events.append(registro)
            oyentes = list(self._listeners)
        logger.info("Registro: %s", registro)
        for oyente in oyentes:
            try:
                oyente(registro)
            except Exception:
                pass

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    # ------------------------------------------------------------------
    # Alta y baja
    # ------------------------------------------------------------------
    def register(self, agent: BaseAgent) -> BaseAgent:
        """Agrega un agente; el ``agent_id`` y el nombre deben ser únicos."""
        if not isinstance(agent, BaseAgent):
            raise AgentError("Solo se pueden registrar instancias de BaseAgent.")
        with self._lock:
            if agent.agent_id in self._agents:
                raise AgentError(f"Ya existe un agente con id '{agent.agent_id}'.")
            if any(a.name.lower() == agent.name.lower() for a in self._agents.values()):
                raise AgentError(f"Ya existe un agente llamado '{agent.name}'. Elija otro nombre.")
            self._agents[agent.agent_id] = agent
        self._notificar("registered", agent, capabilities=sorted(agent.capabilities))
        return agent

    def unregister(self, agent_id: str) -> BaseAgent:
        """Quita un agente del registro y lo devuelve."""
        with self._lock:
            agente = self._agents.pop(agent_id, None)
        if agente is None:
            raise AgentError(f"No existe el agente '{agent_id}'.")
        agente.set_status(AgentStatus.STOPPED)
        self._notificar("unregistered", agente)
        return agente

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    def get(self, agent_id: str) -> BaseAgent:
        with self._lock:
            agente = self._agents.get(agent_id)
        if agente is None:
            raise AgentError(f"No existe el agente '{agent_id}'.")
        return agente

    def find_by_name(self, name: str) -> BaseAgent | None:
        with self._lock:
            return next((a for a in self._agents.values() if a.name.lower() == name.lower()), None)

    def list(self, *, include_inactive: bool = True) -> list[BaseAgent]:
        with self._lock:
            agentes = list(self._agents.values())
        return agentes if include_inactive else [a for a in agentes if a.is_available]

    def find_by_capability(self, capability: str, *, exclude: Iterable[str] = ()) -> list[BaseAgent]:
        """Agentes **disponibles** que publican ``capability``.

        Orden: primero los no degradados, luego mayor prioridad, mayor tasa de éxito y menor tiempo medio.
        """
        excluidos = set(exclude)
        with self._lock:
            candidatos = [
                a for a in self._agents.values()
                if a.agent_id not in excluidos and capability in a.capabilities and a.status in AVAILABLE_STATUSES
            ]
        return sorted(
            candidatos,
            key=lambda a: (a.status == AgentStatus.DEGRADED, -a.priority, -a.success_rate, a.average_time),
        )

    def capabilities(self) -> dict[str, list[str]]:
        """Mapa capacidad → agentes que la publican (todos los estados)."""
        mapa: dict[str, list[str]] = {}
        for agente in self.list():
            for capacidad in agente.capabilities:
                mapa.setdefault(capacidad, []).append(agente.agent_id)
        return dict(sorted(mapa.items()))

    def update_status(self, agent_id: str, status: AgentStatus | str, *, reason: str = "") -> BaseAgent:
        agente = self.get(agent_id)
        anterior = agente.status
        agente.set_status(status)
        self._notificar("status", agente, previous=anterior.value, status=agente.status.value, reason=reason)
        return agente

    def snapshot(self) -> list[dict[str, Any]]:
        """Descripción de todos los agentes (para el dashboard)."""
        return [a.describe() for a in self.list()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)

    def __contains__(self, agent_id: object) -> bool:
        with self._lock:
            return agent_id in self._agents


__all__ = ["AgentRegistry"]
