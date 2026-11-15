"""Agente Coordinador: recibe la solicitud, analiza la tarea, descubre capacidades, delega, consolida y
registra sus decisiones.

El coordinador **no conoce a los agentes concretos**: traduce la tarea a una capacidad con una tabla de
enrutamiento que es configuración (``coordinator.routing`` en ``config.json``) y pregunta al registro quién
la publica. Por eso un agente nuevo con una capacidad existente se usa sin tocar este archivo.

Políticas cuando varios agentes publican la misma capacidad:

- ``priority``: se delega al mejor candidato (prioridad, éxito, tiempo); si falla, la agencia hace *fallback*.
- ``fanout``: todos los candidatos ejecutan la tarea y se elige el mejor resultado por su métrica principal
  (cada agente informa ``primary_metric = {"name", "value", "maximize"}``).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from framework_ia.persistence.config_store import DEFAULT_CONFIG
from framework_ia.utils.exceptions import AgentError, ValidationError
from framework_ia.utils.metrics import is_better
from framework_ia.utils.validation import CAPABILITY_PATTERN, validate_capability

from .agency import POLICIES, Agency
from .agent_factory import agent_type
from .base_agent import BaseAgent
from .messaging import Message

ROUTING_KEYS = ("capability", "policy")


@dataclass
class CoordinatorResult:
    """Resultado consolidado que el coordinador devuelve al controlador."""

    task: str
    capability: str
    policy: str
    correlation_id: str
    result: dict[str, Any]
    agents_used: list[str]
    selected_agent: str
    candidates: list[str]
    decisions: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        datos = asdict(self)
        datos["result"] = self.result  # se conserva sin copiar (puede contener DataFrames)
        return datos


@agent_type("coordinator", description="Coordina la comunidad: enruta por capacidades, delega y consolida.")
class CoordinatorAgent(BaseAgent):
    """Agente coordinador del sistema multiagente."""

    default_capabilities = frozenset({"coordination.route"})
    supported_capabilities = None

    def __init__(self, name: str = "Coordinador", capabilities: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(name, capabilities, **kwargs)
        config = self.context.get("config")
        enrutamiento = config.get("coordinator.routing") if config is not None else None
        self.routing: dict[str, dict[str, Any]] = enrutamiento or DEFAULT_CONFIG["coordinator"]["routing"]
        politica = config.get("coordinator.default_policy") if config is not None else None
        self.default_policy = politica or "priority"

    @property
    def agency(self) -> Agency:
        agencia = self.context.get("agency")
        if agencia is None:
            raise AgentError("El coordinador no está conectado a una agencia.")
        return agencia

    def can_handle(self, capability: str) -> bool:
        """El coordinador es la puerta de entrada: acepta cualquier tarea y la valida en :meth:`analyze`."""
        return True

    # ------------------------------------------------------------------
    # Flujo del coordinador
    # ------------------------------------------------------------------
    def analyze(self, message: Message) -> tuple[str, str, str]:
        """Determina capacidad y política; devuelve ``(capacidad, politica, justificacion)``."""
        payload = message.payload
        ruta = self.routing.get(message.task, {})
        if payload.get("capability"):
            capacidad = validate_capability(payload["capability"])
            motivo = "La solicitud indica la capacidad explícitamente."
        elif ruta.get("capability"):
            capacidad = ruta["capability"]
            motivo = f"Tabla de enrutamiento (configuración): '{message.task}' → '{capacidad}'."
        elif CAPABILITY_PATTERN.match(message.task):
            capacidad = message.task
            motivo = "La tarea ya es una capacidad (dominio.accion)."
        else:
            raise ValidationError(
                f"Tarea desconocida '{message.task}'. Tareas configuradas: {', '.join(sorted(self.routing))}."
            )
        politica = payload.get("policy") or ruta.get("policy") or self.default_policy
        if politica not in POLICIES:
            raise ValidationError(f"Política inválida '{politica}'. Opciones: {', '.join(POLICIES)}.")
        return capacidad, politica, motivo

    @staticmethod
    def _metrica(resultado: dict[str, Any]) -> tuple[str | None, float | None, bool]:
        principal = resultado.get("primary_metric") or {}
        return principal.get("name"), principal.get("value"), bool(principal.get("maximize", True))

    def consolidate(self, respuestas: list[Message]) -> tuple[Message, list[dict[str, Any]], str]:
        """Elige la mejor respuesta por la métrica principal informada por cada agente."""
        if len(respuestas) == 1:
            return respuestas[0], [], f"Un solo agente respondió ({respuestas[0].sender})."
        alternativas, mejor, valor_mejor = [], None, None
        nombre_metrica, maximizar = None, True
        for respuesta in respuestas:
            nombre, valor, sentido = self._metrica(respuesta.payload)
            nombre_metrica = nombre_metrica or nombre
            maximizar = sentido if nombre else maximizar
            alternativas.append({"agent_id": respuesta.sender, "metric": nombre, "value": valor,
                                 "elapsed_s": respuesta.meta.get("elapsed_s")})
            if mejor is None or is_better(valor, valor_mejor, sentido):
                mejor, valor_mejor = respuesta, valor
        if nombre_metrica is None:
            return respuestas[0], alternativas, "Ningún agente informó métrica: se toma la primera respuesta."
        sentido = "mayor" if maximizar else "menor"
        return mejor, alternativas, (
            f"Se eligió el resultado de '{mejor.sender}' por {sentido} {nombre_metrica} "
            f"({valor_mejor}) entre {len(respuestas)} agentes."
        )

    def _execute(self, message: Message) -> dict[str, Any]:
        inicio = time.perf_counter()
        decisiones: list[dict[str, Any]] = []

        def decidir(decision: str, justificacion: str) -> None:
            decisiones.append({"timestamp": datetime.now().isoformat(timespec="milliseconds"),
                               "decision": decision, "justification": justificacion})

        decidir("Solicitud recibida", f"'{message.sender}' pidió la tarea '{message.task}'.")
        capacidad, politica, motivo = self.analyze(message)
        decidir(f"Capacidad requerida: {capacidad}", motivo)
        t_analisis = time.perf_counter()

        candidatos = self.agency.candidates(capacidad, exclude=[self.agent_id])
        if not candidatos:
            decidir("Sin candidatos", f"Ningún agente disponible publica '{capacidad}'.")
        else:
            detalle = ", ".join(f"{a.agent_id} (prioridad {a.priority}, éxito {a.success_rate:.0%}, "
                                f"{a.status.value})" for a in candidatos)
            decidir(f"Descubrimiento: {len(candidatos)} candidato(s)", f"Registro consultado por capacidad: {detalle}.")
        decidir(f"Política: {politica}",
                "Delegar al mejor candidato con fallback." if politica == "priority"
                else "Todos los candidatos ejecutan la tarea y se consolida el mejor resultado.")

        contenido = {k: v for k, v in message.payload.items() if k not in ROUTING_KEYS}
        try:
            asignacion = self.agency.dispatch(
                capacidad, contenido, sender=self.agent_id, correlation_id=message.correlation_id,
                policy=politica, exclude=[self.agent_id],
            )
        except Exception as error:
            decidir("Delegación fallida", str(error))
            if hasattr(error, "details"):
                error.details = {**getattr(error, "details", {}), "decisions": decisiones}
            raise
        t_delegacion = time.perf_counter()
        for evento in asignacion.fallbacks:
            decidir("Fallback", f"'{evento['from']}' falló ({evento['reason']}); se delegó en '{evento['to']}'.")
        decidir("Delegación completada",
                f"Respondieron: {', '.join(asignacion.agents_used)} en {len(asignacion.attempts)} intento(s).")

        elegida, alternativas, justificacion = self.consolidate(asignacion.responses)
        decidir("Consolidación", justificacion)
        fin = time.perf_counter()

        return CoordinatorResult(
            task=message.task,
            capability=capacidad,
            policy=politica,
            correlation_id=message.correlation_id,
            result=elegida.payload,
            agents_used=asignacion.agents_used,
            selected_agent=elegida.sender,
            candidates=asignacion.candidates,
            decisions=decisiones,
            attempts=[a.to_dict() for a in asignacion.attempts],
            alternatives=alternativas,
            timings={
                "analysis_s": round(t_analisis - inicio, 4),
                "delegation_s": round(t_delegacion - t_analisis, 4),
                "consolidation_s": round(fin - t_delegacion, 4),
                "total_s": round(fin - inicio, 4),
            },
        ).to_dict()


__all__ = ["CoordinatorAgent", "CoordinatorResult"]
