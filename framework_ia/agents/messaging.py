"""Mensajería entre agentes: mensajes con performativas (estilo FIPA-ACL) y un bus que registra todo.

El :class:`MessageBus` es el medio de transporte: entrega cada ``REQUEST`` al manejador suscrito del
receptor, registra la petición y la respuesta, y conserva el historial completo para el panel MAS y las
evidencias. Ningún agente se llama directamente: todos se comunican a través del bus.
"""
from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from framework_ia.utils.exceptions import ValidationError
from framework_ia.utils.serialization import summarize


class Performative(str, Enum):
    """Intención comunicativa del mensaje."""

    REQUEST = "request"   # pedir la ejecución de una tarea
    INFORM = "inform"     # entregar un resultado
    FAILURE = "failure"   # informar que la tarea falló
    QUERY = "query"       # consultar información (por ejemplo, la descripción de un agente)
    CONFIRM = "confirm"   # confirmar un evento (creación, cambio de estado, ...)


class MessageStatus(str, Enum):
    """Estado del mensaje dentro del bus."""

    CREATED = "created"
    SENT = "sent"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"
    UNDELIVERABLE = "undeliverable"
    LOGGED = "logged"


def _nuevo_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Message:
    """Mensaje entre agentes.

    ``correlation_id`` agrupa todos los mensajes de una misma solicitud (por defecto, el id del mensaje que
    la inició). ``meta`` guarda datos de control (tiempo de ejecución, intento, etc.).
    """

    sender: str
    receiver: str
    performative: Performative
    task: str
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    id: str = field(default_factory=_nuevo_id)
    status: str = MessageStatus.CREATED.value
    timestamp: datetime = field(default_factory=datetime.now)
    error: str | None = None
    in_reply_to: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sender or not self.receiver:
            raise ValidationError("Todo mensaje necesita emisor y receptor.")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValidationError("Todo mensaje necesita una tarea.")
        try:
            self.performative = Performative(self.performative)
        except ValueError as error:
            raise ValidationError(
                f"Performativa inválida '{self.performative}'. Opciones: {[p.value for p in Performative]}."
            ) from error
        if not isinstance(self.payload, dict):
            raise ValidationError("El contenido (payload) del mensaje debe ser un diccionario.")
        self.correlation_id = self.correlation_id or self.id

    def reply(
        self,
        performative: Performative | str,
        payload: dict[str, Any] | None = None,
        *,
        sender: str | None = None,
        error: str | None = None,
    ) -> Message:
        """Construye la respuesta (mismo ``correlation_id``, receptor = emisor original)."""
        return Message(
            sender=sender or self.receiver,
            receiver=self.sender,
            performative=Performative(performative),
            task=self.task,
            payload=payload or {},
            correlation_id=self.correlation_id,
            in_reply_to=self.id,
            error=error,
        )

    @property
    def ok(self) -> bool:
        return self.performative != Performative.FAILURE

    def to_dict(self, *, full_payload: bool = False) -> dict[str, Any]:
        """Representación serializable; por defecto el contenido se resume (sin filas de datos)."""
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "in_reply_to": self.in_reply_to,
            "timestamp": self.timestamp.isoformat(timespec="milliseconds"),
            "sender": self.sender,
            "receiver": self.receiver,
            "performative": self.performative.value,
            "task": self.task,
            "status": self.status,
            "error": self.error,
            "payload": self.payload if full_payload else summarize(self.payload, max_items=12),
            "meta": dict(self.meta),
        }


Handler = Callable[[Message], "Message | None"]
Listener = Callable[[Message], None]


class MessageBus:
    """Transporte y registro de todos los mensajes del sistema (seguro ante hilos)."""

    def __init__(self, max_history: int = 5000) -> None:
        self._handlers: dict[str, Handler] = {}
        self._history: deque[Message] = deque(maxlen=max_history)
        self._listeners: list[Listener] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Suscripciones
    # ------------------------------------------------------------------
    def subscribe(self, agent_id: str, handler: Handler) -> None:
        """Registra el manejador que recibirá los mensajes dirigidos a ``agent_id``."""
        with self._lock:
            self._handlers[agent_id] = handler

    def unsubscribe(self, agent_id: str) -> None:
        with self._lock:
            self._handlers.pop(agent_id, None)

    def is_subscribed(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._handlers

    def add_listener(self, listener: Listener) -> None:
        """Observador que se notifica con cada mensaje registrado (útil para el dashboard o logs)."""
        with self._lock:
            self._listeners.append(listener)

    # ------------------------------------------------------------------
    # Envío
    # ------------------------------------------------------------------
    def _registrar(self, message: Message) -> None:
        with self._lock:
            self._history.append(message)
            oyentes = list(self._listeners)
        for oyente in oyentes:
            try:
                oyente(message)
            except Exception:  # un observador defectuoso no debe romper la comunicación
                pass

    def publish(self, message: Message) -> Message:
        """Registra un mensaje informativo sin entregarlo (eventos, confirmaciones)."""
        message.status = MessageStatus.LOGGED.value
        self._registrar(message)
        return message

    def send(self, message: Message) -> Message:
        """Entrega ``message`` a su receptor y devuelve la respuesta (también registrada).

        Si el receptor no está suscrito, devuelve un ``FAILURE`` generado por el bus.
        """
        message.status = MessageStatus.SENT.value
        self._registrar(message)
        with self._lock:
            manejador = self._handlers.get(message.receiver)
        if manejador is None:
            message.status = MessageStatus.UNDELIVERABLE.value
            respuesta = message.reply(
                Performative.FAILURE,
                {"error_type": "Undeliverable", "user_error": False},
                sender="bus",
                error=f"El receptor '{message.receiver}' no está disponible en el bus.",
            )
            respuesta.status = MessageStatus.FAILED.value
            self._registrar(respuesta)
            return respuesta
        message.status = MessageStatus.DELIVERED.value
        respuesta = manejador(message)
        if respuesta is None:
            message.status = MessageStatus.PROCESSED.value
            respuesta = message.reply(Performative.CONFIRM, {}, sender=message.receiver)
        message.status = MessageStatus.PROCESSED.value if respuesta.ok else MessageStatus.FAILED.value
        respuesta.status = MessageStatus.DELIVERED.value if respuesta.ok else MessageStatus.FAILED.value
        self._registrar(respuesta)
        return respuesta

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    def history(
        self,
        *,
        sender: str | None = None,
        receiver: str | None = None,
        performative: Performative | str | None = None,
        task: str | None = None,
        correlation_id: str | None = None,
        agent: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Mensajes registrados (en orden cronológico) que cumplen todos los filtros indicados."""
        with self._lock:
            mensajes = list(self._history)
        if performative is not None:
            performative = Performative(performative)
        filtrados = [
            m for m in mensajes
            if (sender is None or m.sender == sender)
            and (receiver is None or m.receiver == receiver)
            and (performative is None or m.performative == performative)
            and (task is None or m.task == task)
            and (correlation_id is None or m.correlation_id == correlation_id)
            and (agent is None or agent in (m.sender, m.receiver))
        ]
        return filtrados[-limit:] if limit else filtrados

    def records(self, **filters: Any) -> list[dict[str, Any]]:
        """Historial filtrado como diccionarios serializables."""
        return [m.to_dict() for m in self.history(**filters)]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._history)


__all__ = ["Message", "MessageBus", "MessageStatus", "Performative"]
