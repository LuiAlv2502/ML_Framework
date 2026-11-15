"""Configuración de logging del framework (archivo rotativo ``logs/framework.log`` + consola opcional)."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .paths import resolve

LOGGER_NAME = "framework_ia"
FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured_path: Path | None = None


def configure_logging(
    log_dir: str | Path = "logs",
    *,
    level: int = logging.INFO,
    max_bytes: int = 1_000_000,
    backup_count: int = 3,
    console: bool = False,
) -> Path:
    """Configura (una sola vez por ruta) el logger raíz del framework y devuelve la ruta del archivo."""
    global _configured_path
    ruta = resolve(log_dir) / "framework.log"
    logger = logging.getLogger(LOGGER_NAME)
    if _configured_path == ruta:
        return ruta
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    archivo = RotatingFileHandler(ruta, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    archivo.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(archivo)
    if console:
        consola = logging.StreamHandler()
        consola.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(consola)
    logger.setLevel(level)
    logger.propagate = False
    _configured_path = ruta
    return ruta


def get_logger(name: str = "") -> logging.Logger:
    """Devuelve un logger hijo de ``framework_ia`` (por ejemplo ``framework_ia.agents.registry``)."""
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


__all__ = ["configure_logging", "get_logger"]
