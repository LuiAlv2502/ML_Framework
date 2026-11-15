"""Clase base común de todos los agentes del sistema multiagente.

:meth:`BaseAgent.handle` es un *método plantilla*: valida el mensaje, marca el agente como ocupado, llama
a :meth:`BaseAgent._execute` (lo único que implementa cada agente), captura errores y los convierte en
``FAILURE``, actualiza estadísticas y estado, y devuelve la respuesta.

Los errores del usuario (``ValidationError``) y los de enrutamiento (``AgentError``) se responden con
``FAILURE`` pero **no** degradan al agente; cualquier otra excepción sí cuenta como fallo propio:
1 fallo consecutivo → ``DEGRADED``, :attr:`BaseAgent.fail_after` fallos consecutivos → ``FAILED``.
"""
from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

import pandas as pd

from framework_ia.utils.exceptions import AgentError, FrameworkError, ValidationError, friendly_message
from framework_ia.utils.logging_config import get_logger
from framework_ia.utils.validation import validate_capability, validate_dataframe, validate_name, validate_params

from .messaging import Message, Performative

logger = get_logger("agents")


class AgentStatus(str, Enum):
    """Estados del ciclo de vida de un agente."""

    IDLE = "idle"           # disponible
    BUSY = "busy"           # ejecutando una tarea
    DEGRADED = "degraded"   # falló recientemente; sigue disponible con menor preferencia
    PAUSED = "paused"       # pausado por el operador
    STOPPED = "stopped"     # detenido por el operador
    FAILED = "failed"       # demasiados fallos consecutivos; requiere reinicio


AVAILABLE_STATUSES = frozenset({AgentStatus.IDLE, AgentStatus.DEGRADED})


class BaseAgent(ABC):
    """Agente con identidad, capacidades publicadas, estado, prioridad y estadísticas."""

    agent_type: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    default_capabilities: ClassVar[frozenset[str]] = frozenset()
    #: Capacidades que el tipo sabe atender (``None`` = cualquiera con formato válido).
    supported_capabilities: ClassVar[frozenset[str] | None] = None
    #: Especificación de la configuración aceptada (formato de :func:`validate_params`).
    config_schema: ClassVar[dict[str, tuple]] = {}
    fail_after: ClassVar[int] = 3

    def __init__(
        self,
        name: str,
        capabilities: list[str] | set[str] | None = None,
        *,
        agent_id: str | None = None,
        priority: int = 10,
        config: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = validate_name(name, what="nombre del agente")
        capacidades = set(capabilities) if capabilities else set(self.default_capabilities)
        if not capacidades:
            raise ValidationError("El agente debe publicar al menos una capacidad.")
        for capacidad in capacidades:
            validate_capability(capacidad)
        if self.supported_capabilities is not None:
            no_soportadas = sorted(capacidades - set(self.supported_capabilities))
            if no_soportadas:
                raise ValidationError(
                    f"El tipo '{self.agent_type}' no sabe atender {no_soportadas}. "
                    f"Capacidades soportadas: {sorted(self.supported_capabilities)}."
                )
        if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 100:
            raise ValidationError("La prioridad debe ser un entero entre 0 y 100.")
        self.capabilities: frozenset[str] = frozenset(capacidades)
        self.agent_id = agent_id or f"{self.agent_type}-{uuid.uuid4().hex[:6]}"
        self.priority = priority
        self.config = self.validate_config(config or {})
        self.context: dict[str, Any] = context if context is not None else {}
        self.metadata = dict(metadata or {})
        self.status = AgentStatus.IDLE
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.tasks_received = 0
        self.tasks_succeeded = 0
        self.tasks_failed = 0
        self.tasks_rejected = 0
        self.consecutive_failures = 0
        self.total_time = 0.0
        self.current_task: str | None = None
        self.last_task: str | None = None
        self.last_error: str | None = None
        self.task_log: deque[dict[str, Any]] = deque(maxlen=50)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Configuración y capacidades
    # ------------------------------------------------------------------
    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Valida la configuración contra :attr:`config_schema` (las subclases pueden ampliarla)."""
        return validate_params(config, self.config_schema, owner=f"el agente '{self.agent_type}'")

    def can_handle(self, capability: str) -> bool:
        return capability in self.capabilities

    @property
    def is_available(self) -> bool:
        return self.status in AVAILABLE_STATUSES

    @property
    def success_rate(self) -> float:
        intentos = self.tasks_succeeded + self.tasks_failed
        return 1.0 if intentos == 0 else self.tasks_succeeded / intentos

    @property
    def average_time(self) -> float:
        return self.total_time / self.tasks_received if self.tasks_received else 0.0

    def set_status(self, status: AgentStatus | str) -> None:
        with self._lock:
            self.status = AgentStatus(status)

    def reset_failures(self) -> None:
        """Olvida los fallos consecutivos (usado al reiniciar)."""
        with self._lock:
            self.consecutive_failures = 0
            self.last_error = None

    # ------------------------------------------------------------------
    # Método plantilla
    # ------------------------------------------------------------------
    def handle(self, message: Message) -> Message:
        """Procesa un mensaje y devuelve la respuesta (nunca lanza excepciones)."""
        if message.performative == Performative.QUERY:
            return message.reply(Performative.INFORM, self.describe(), sender=self.agent_id)
        if message.performative != Performative.REQUEST:
            return message.reply(
                Performative.FAILURE, {"error_type": "UnsupportedPerformative", "user_error": False},
                sender=self.agent_id, error=f"El agente no atiende mensajes '{message.performative.value}'.",
            )
        with self._lock:
            if not self.is_available:
                return message.reply(
                    Performative.FAILURE, {"error_type": "AgentUnavailable", "user_error": False},
                    sender=self.agent_id, error=f"El agente '{self.name}' no está disponible ({self.status.value}).",
                )
            if not self.can_handle(message.task):
                return message.reply(
                    Performative.FAILURE, {"error_type": "CapabilityNotPublished", "user_error": False},
                    sender=self.agent_id,
                    error=f"El agente '{self.name}' no publica la capacidad '{message.task}'.",
                )
            estado_previo = self.status
            self.status = AgentStatus.BUSY
            self.current_task = message.task
            self.tasks_received += 1

        inicio = time.perf_counter()
        exito, falla_propia = False, False
        try:
            self.validate_payload(message)
            resultado = self._execute(message)
            if not isinstance(resultado, dict):
                raise TypeError("_execute debe devolver un diccionario.")
            respuesta = message.reply(Performative.INFORM, resultado, sender=self.agent_id)
            exito = True
        except FrameworkError as error:
            falla_propia = not isinstance(error, (ValidationError, AgentError))
            respuesta = message.reply(
                Performative.FAILURE,
                {"error_type": type(error).__name__, "user_error": isinstance(error, ValidationError),
                 **error.details},
                sender=self.agent_id, error=friendly_message(error),
            )
        except Exception as error:  # fallo propio del agente
            falla_propia = True
            logger.exception("El agente %s falló en %s", self.agent_id, message.task)
            respuesta = message.reply(
                Performative.FAILURE, {"error_type": type(error).__name__, "user_error": False},
                sender=self.agent_id, error=f"{type(error).__name__}: {friendly_message(error)}",
            )
        duracion = time.perf_counter() - inicio

        with self._lock:
            self.total_time += duracion
            self.current_task = None
            self.last_task = message.task
            if exito:
                self.tasks_succeeded += 1
                self.consecutive_failures = 0
                self.status = AgentStatus.IDLE
            elif falla_propia:
                self.tasks_failed += 1
                self.consecutive_failures += 1
                self.last_error = respuesta.error
                self.status = (AgentStatus.FAILED if self.consecutive_failures >= self.fail_after
                               else AgentStatus.DEGRADED)
            else:
                self.tasks_rejected += 1
                self.last_error = respuesta.error
                self.status = estado_previo
            self.task_log.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "task": message.task,
                "correlation_id": message.correlation_id,
                "result": "ok" if exito else ("fallo" if falla_propia else "rechazada"),
                "elapsed_s": round(duracion, 4),
            })
        respuesta.meta.update({"elapsed_s": round(duracion, 4), "agent_type": self.agent_type,
                               "agent_name": self.name})
        return respuesta

    def validate_payload(self, message: Message) -> None:
        """Validación previa del contenido del mensaje (las subclases pueden ampliarla)."""

    @abstractmethod
    def _execute(self, message: Message) -> dict[str, Any]:
        """Ejecuta la tarea pedida en ``message.task`` y devuelve un resultado estructurado."""

    # ------------------------------------------------------------------
    # Utilidades para las subclases
    # ------------------------------------------------------------------
    @staticmethod
    def require_data(message: Message, key: str = "data") -> pd.DataFrame:
        """Extrae y valida el DataFrame del contenido del mensaje."""
        datos = message.payload.get(key)
        if datos is None:
            raise ValidationError("La solicitud no incluye datos ('data').")
        return validate_dataframe(datos)

    # ------------------------------------------------------------------
    # Descripción
    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Ficha del agente para el registro y el panel MAS."""
        with self._lock:
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "agent_type": self.agent_type,
                "description": self.description,
                "capabilities": sorted(self.capabilities),
                "status": self.status.value,
                "priority": self.priority,
                "current_task": self.current_task,
                "last_task": self.last_task,
                "tasks_received": self.tasks_received,
                "tasks_succeeded": self.tasks_succeeded,
                "tasks_failed": self.tasks_failed,
                "tasks_rejected": self.tasks_rejected,
                "success_rate": round(self.success_rate, 4),
                "average_time_s": round(self.average_time, 4),
                "last_error": self.last_error,
                "config": dict(self.config),
                "created_at": self.created_at,
                "metadata": dict(self.metadata),
            }

    def to_definition(self) -> dict[str, Any]:
        """Definición mínima para recrear el agente con la fábrica (persistencia)."""
        return {
            "name": self.name,
            "agent_type": self.agent_type,
            "capabilities": sorted(self.capabilities),
            "config": dict(self.config),
            "priority": self.priority,
            "agent_id": self.agent_id,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.agent_id} {sorted(self.capabilities)} {self.status.value}>"


__all__ = ["AVAILABLE_STATUSES", "AgentStatus", "BaseAgent"]
