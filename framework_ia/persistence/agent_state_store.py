"""Persistencia de la definición de los agentes creados dinámicamente (para rehidratarlos al iniciar)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from framework_ia.utils.exceptions import PersistenceError, ValidationError

REQUIRED_FIELDS = ("name", "agent_type")


class AgentStateStore:
    """Lista de definiciones ``{"name", "agent_type", "capabilities", "config", "priority"}`` en un archivo JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def load(self) -> list[dict[str, Any]]:
        """Devuelve todas las definiciones guardadas (lista vacía si no hay archivo)."""
        with self._lock:
            if not self.path.exists():
                return []
            try:
                datos = json.loads(self.path.read_text(encoding="utf-8") or "[]")
            except ValueError as error:
                raise PersistenceError(f"El archivo de agentes '{self.path.name}' está dañado: {error}") from error
            if not isinstance(datos, list):
                raise PersistenceError("El archivo de agentes debe contener una lista.")
            return datos

    def save(self, definitions: list[dict[str, Any]]) -> None:
        """Reemplaza todas las definiciones."""
        for definicion in definitions:
            faltan = [c for c in REQUIRED_FIELDS if not definicion.get(c)]
            if faltan:
                raise ValidationError(f"La definición de agente no tiene los campos {faltan}.")
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporal = self.path.with_suffix(".tmp")
                temporal.write_text(json.dumps(definitions, ensure_ascii=False, indent=2), encoding="utf-8")
                temporal.replace(self.path)
            except OSError as error:
                raise PersistenceError(f"No se pudo guardar el estado de los agentes: {error}") from error

    def upsert(self, definition: dict[str, Any]) -> None:
        """Agrega o reemplaza (por nombre) una definición."""
        with self._lock:
            definiciones = [d for d in self.load() if d.get("name") != definition.get("name")]
            definiciones.append(definition)
            self.save(definiciones)

    def remove(self, name: str) -> bool:
        """Elimina una definición por nombre; devuelve ``True`` si existía."""
        with self._lock:
            definiciones = self.load()
            restantes = [d for d in definiciones if d.get("name") != name]
            if len(restantes) == len(definiciones):
                return False
            self.save(restantes)
            return True


__all__ = ["AgentStateStore"]
