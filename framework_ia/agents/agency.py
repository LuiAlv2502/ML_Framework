"""Agencia: administra el ciclo de vida de los agentes, asigna tareas, coordina la cooperación y gestiona fallos.

- **Ciclo de vida**: ``start/stop/pause/resume/restart_agent``; cada cambio queda como ``CONFIRM`` en el bus.
- **Asignación**: busca en el registro los agentes con la capacidad pedida.
- **Cooperación**: política ``fanout`` (todos los candidatos atienden la tarea; el coordinador consolida).
- **Fallos**: reintentos configurables por agente y *fallback* al siguiente candidato; los errores del
  usuario (validación) no se reintentan porque fallarían igual en cualquier agente.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from framework_ia.persistence.agent_state_store import AgentStateStore
from framework_ia.utils.exceptions import CapabilityNotFoundError, TaskExecutionError, ValidationError
from framework_ia.utils.logging_config import get_logger

from .agent_factory import AgentFactory
from .base_agent import AVAILABLE_STATUSES, AgentStatus, BaseAgent
from .messaging import Message, MessageBus, Performative
from .registry import AgentRegistry

logger = get_logger("agents.agency")
POLICIES = ("priority", "fanout")


@dataclass
class DispatchAttempt:
    """Un intento de ejecutar la tarea en un agente."""

    agent_id: str
    attempt: int
    ok: bool
    error: str | None
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "attempt": self.attempt, "ok": self.ok, "error": self.error,
                "elapsed_s": self.elapsed_s}


@dataclass
class DispatchResult:
    """Resultado de una asignación: respuestas exitosas e historial de intentos."""

    capability: str
    policy: str
    responses: list[Message] = field(default_factory=list)
    attempts: list[DispatchAttempt] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    fallbacks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def agents_used(self) -> list[str]:
        return [r.sender for r in self.responses]


class Agency:
    """Posee el registro, la fábrica y el bus, y ofrece las operaciones de alto nivel del MAS."""

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        factory: AgentFactory | None = None,
        bus: MessageBus | None = None,
        *,
        max_retries: int = 1,
        state_store: AgentStateStore | None = None,
        context: dict[str, Any] | None = None,
        agency_id: str = "agencia",
    ) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 5:
            raise ValidationError("max_retries debe ser un entero entre 0 y 5.")
        self.registry = registry or AgentRegistry()
        self.bus = bus or MessageBus()
        self.context: dict[str, Any] = context if context is not None else {}
        self.context.setdefault("agency", self)
        self.factory = factory or AgentFactory(self.registry, self.bus, context=self.context, state_store=state_store)
        self.state_store = state_store or self.factory.state_store
        self.max_retries = max_retries
        self.agency_id = agency_id
        self.coordinator: BaseAgent | None = None

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def _cambiar_estado(self, agent_id: str, estado: AgentStatus, accion: str, motivo: str = "") -> BaseAgent:
        agente = self.registry.update_status(agent_id, estado, reason=motivo)
        self.bus.publish(Message(
            sender=self.agency_id, receiver=agent_id, performative=Performative.CONFIRM,
            task=f"lifecycle.{accion}", payload={"status": estado.value, "reason": motivo},
        ))
        return agente

    def start_agent(self, agent_id: str) -> BaseAgent:
        """Pone en marcha un agente detenido."""
        return self._cambiar_estado(agent_id, AgentStatus.IDLE, "start")

    def stop_agent(self, agent_id: str, reason: str = "") -> BaseAgent:
        """Detiene un agente: deja de recibir tareas hasta que se reinicie."""
        return self._cambiar_estado(agent_id, AgentStatus.STOPPED, "stop", reason)

    def pause_agent(self, agent_id: str, reason: str = "") -> BaseAgent:
        """Pausa un agente temporalmente."""
        return self._cambiar_estado(agent_id, AgentStatus.PAUSED, "pause", reason)

    def resume_agent(self, agent_id: str) -> BaseAgent:
        """Reanuda un agente pausado."""
        agente = self.registry.get(agent_id)
        if agente.status not in (AgentStatus.PAUSED, AgentStatus.IDLE):
            raise ValidationError(f"Solo se pueden reanudar agentes pausados ('{agente.name}' está {agente.status.value}).")
        return self._cambiar_estado(agent_id, AgentStatus.IDLE, "resume")

    def restart_agent(self, agent_id: str) -> BaseAgent:
        """Reinicia un agente (olvida los fallos consecutivos y lo deja disponible)."""
        self.registry.get(agent_id).reset_failures()
        return self._cambiar_estado(agent_id, AgentStatus.IDLE, "restart")

    def create_agent(self, definition: dict[str, Any], *, persist: bool = True) -> BaseAgent:
        """Crea un agente con la fábrica (validación + registro + bus) y opcionalmente lo persiste."""
        return self.factory.create_from_definition(definition, persist=persist and self.state_store is not None)

    def remove_agent(self, agent_id: str, *, forget: bool = True) -> BaseAgent:
        if self.coordinator is not None and agent_id == self.coordinator.agent_id:
            raise ValidationError("No se puede eliminar al agente coordinador.")
        return self.factory.destroy(agent_id, forget=forget)

    # ------------------------------------------------------------------
    # Asignación, cooperación y fallos
    # ------------------------------------------------------------------
    def candidates(self, capability: str, *, exclude: list[str] | tuple[str, ...] = ()) -> list[BaseAgent]:
        return self.registry.find_by_capability(capability, exclude=exclude)

    def _intentar(
        self, agente: BaseAgent, capability: str, payload: dict[str, Any], sender: str,
        correlation_id: str | None, reintentos: int, resultado: DispatchResult,
    ) -> Message | None:
        """Envía la tarea al agente con reintentos; devuelve la respuesta INFORM o ``None``."""
        for intento in range(1, reintentos + 2):
            if agente.status not in AVAILABLE_STATUSES:
                break
            solicitud = Message(sender=sender, receiver=agente.agent_id, performative=Performative.REQUEST,
                                task=capability, payload=payload, correlation_id=correlation_id or "")
            solicitud.meta["attempt"] = intento
            correlation_id = solicitud.correlation_id
            inicio = time.perf_counter()
            respuesta = self.bus.send(solicitud)
            resultado.attempts.append(DispatchAttempt(
                agente.agent_id, intento, respuesta.ok, respuesta.error, round(time.perf_counter() - inicio, 4)
            ))
            if respuesta.ok:
                return respuesta
            if respuesta.payload.get("user_error"):
                raise ValidationError(respuesta.error or "La solicitud no es válida.")
            logger.warning("Intento %s de %s en %s falló: %s", intento, capability, agente.agent_id, respuesta.error)
        return None

    def dispatch(
        self,
        capability: str,
        payload: dict[str, Any],
        *,
        sender: str,
        correlation_id: str | None = None,
        policy: str = "priority",
        max_retries: int | None = None,
        exclude: list[str] | tuple[str, ...] = (),
    ) -> DispatchResult:
        """Asigna una tarea según la política.

        - ``priority``: el mejor candidato; si falla (tras reintentos) pasa al siguiente (*fallback*).
        - ``fanout``: todos los candidatos; devuelve todas las respuestas exitosas.
        """
        if policy not in POLICIES:
            raise ValidationError(f"Política inválida '{policy}'. Opciones: {', '.join(POLICIES)}.")
        reintentos = self.max_retries if max_retries is None else max_retries
        candidatos = self.candidates(capability, exclude=exclude)
        resultado = DispatchResult(capability, policy, candidates=[a.agent_id for a in candidatos])
        if not candidatos:
            self.bus.publish(Message(
                sender=self.agency_id, receiver=sender, performative=Performative.FAILURE, task=capability,
                payload={"reason": "sin candidatos"}, correlation_id=correlation_id or "",
                error=f"Ningún agente activo publica '{capability}'.",
            ))
            raise CapabilityNotFoundError(capability)

        for posicion, agente in enumerate(candidatos):
            respuesta = self._intentar(agente, capability, payload, sender, correlation_id, reintentos, resultado)
            if respuesta is not None:
                resultado.responses.append(respuesta)
                if policy == "priority":
                    return resultado
                continue
            if policy == "priority" and posicion + 1 < len(candidatos):
                siguiente = candidatos[posicion + 1]
                evento = {"from": agente.agent_id, "to": siguiente.agent_id,
                          "reason": resultado.attempts[-1].error if resultado.attempts else "no disponible"}
                resultado.fallbacks.append(evento)
                self.bus.publish(Message(
                    sender=self.agency_id, receiver=sender, performative=Performative.INFORM,
                    task="agency.fallback", payload=evento, correlation_id=correlation_id or "",
                ))

        if not resultado.responses:
            ultimo = resultado.attempts[-1].error if resultado.attempts else "ningún agente estaba disponible"
            raise TaskExecutionError(
                f"La tarea '{capability}' falló en {len(candidatos)} agente(s). Último error: {ultimo}",
                details={"attempts": [a.to_dict() for a in resultado.attempts]},
            )
        return resultado

    # ------------------------------------------------------------------
    # Solicitudes al coordinador
    # ------------------------------------------------------------------
    def request(self, task: str, payload: dict[str, Any], *, sender: str = "controller") -> Message:
        """Envía una solicitud al coordinador a través del bus y devuelve su respuesta."""
        if self.coordinator is None:
            raise ValidationError("La agencia no tiene un agente coordinador.")
        return self.bus.send(Message(sender=sender, receiver=self.coordinator.agent_id,
                                     performative=Performative.REQUEST, task=task, payload=payload))

    def status_summary(self) -> dict[str, Any]:
        """Conteos por estado, número de mensajes y último mensaje (para la página de Inicio)."""
        agentes = self.registry.list()
        por_estado: dict[str, int] = {}
        for agente in agentes:
            por_estado[agente.status.value] = por_estado.get(agente.status.value, 0) + 1
        ultimos = self.bus.history(limit=1)
        return {
            "agents": len(agentes),
            "by_status": por_estado,
            "active": sum(1 for a in agentes if a.is_available),
            "degraded": por_estado.get("degraded", 0),
            "messages": len(self.bus),
            "last_message": ultimos[0].to_dict() if ultimos else None,
            "capabilities": len(self.registry.capabilities()),
        }


__all__ = ["POLICIES", "Agency", "DispatchAttempt", "DispatchResult"]
