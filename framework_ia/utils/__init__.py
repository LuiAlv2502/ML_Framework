"""Utilitarios transversales del framework: excepciones, validación, métricas, serialización y logging."""

from .exceptions import (
    AgentError,
    CapabilityNotFoundError,
    FrameworkError,
    PersistenceError,
    TaskExecutionError,
    ValidationError,
    friendly_message,
)
from .logging_config import configure_logging, get_logger

__all__ = [
    "AgentError",
    "CapabilityNotFoundError",
    "FrameworkError",
    "PersistenceError",
    "TaskExecutionError",
    "ValidationError",
    "configure_logging",
    "friendly_message",
    "get_logger",
]
