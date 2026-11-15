"""Excepciones propias del framework.

Todas heredan de :class:`FrameworkError` y exponen un mensaje pensado para mostrarse al usuario final
(sin trazas). ``ValidationError`` también hereda de ``ValueError`` para que el código existente que captura
``ValueError`` siga funcionando.
"""
from __future__ import annotations

from typing import Any


class FrameworkError(Exception):
    """Error base del framework con un mensaje amigable y detalles opcionales."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    @property
    def user_message(self) -> str:
        """Mensaje listo para mostrarse en la interfaz."""
        return self.message


class ValidationError(FrameworkError, ValueError):
    """Una entrada (datos, parámetros, archivo) no cumple las reglas del framework."""


class AgentError(FrameworkError):
    """Error relacionado con un agente o con la comunidad de agentes."""


class CapabilityNotFoundError(AgentError, LookupError):
    """Ningún agente activo publica la capacidad solicitada."""

    def __init__(self, capability: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            f"No hay ningún agente activo con la capacidad '{capability}'. "
            "Cree o reactive un agente que la publique en el panel MAS.",
            details={"capability": capability, **(details or {})},
        )
        self.capability = capability


class TaskExecutionError(AgentError):
    """Una tarea falló en todos los agentes candidatos (después de reintentos y fallback)."""


class PersistenceError(FrameworkError):
    """No se pudo leer o escribir un recurso persistido."""


def friendly_message(error: BaseException) -> str:
    """Convierte cualquier excepción en un texto breve para el usuario (nunca un traceback)."""
    if isinstance(error, FrameworkError):
        return error.user_message
    if isinstance(error, KeyError):
        texto = str(error.args[0]) if error.args else str(error)
        return f"Dato no encontrado: {texto}"
    if isinstance(error, ImportError):
        return f"Falta una dependencia opcional: {error}"
    texto = str(error).strip()
    return texto or type(error).__name__


__all__ = [
    "AgentError",
    "CapabilityNotFoundError",
    "FrameworkError",
    "PersistenceError",
    "TaskExecutionError",
    "ValidationError",
    "friendly_message",
]
