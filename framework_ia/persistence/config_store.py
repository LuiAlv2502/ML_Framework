"""Lectura de la configuración del framework (``config.json`` o ``config.yaml`` en la raíz del proyecto)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from framework_ia.utils.exceptions import PersistenceError
from framework_ia.utils.paths import PROJECT_ROOT, resolve

DEFAULT_CONFIG: dict[str, Any] = {
    "random_state": 42,
    "paths": {
        "data_dir": "data",
        "runs_dir": "runs",
        "models_dir": "framework_ia/models",
        "logs_dir": "logs",
        "agents_state": "framework_ia/persistence/agents.json",
    },
    "limits": {"max_rows": 200_000, "max_columns": 500, "max_file_mb": 50},
    "coordinator": {
        "default_policy": "priority",
        "max_retries": 1,
        # Tabla tarea -> capacidad requerida. Es configuración: añadir una tarea no exige tocar el coordinador.
        "routing": {
            "eda": {"capability": "eda.profile"},
            "eda_summary": {"capability": "eda.summary"},
            "preprocessing": {"capability": "eda.preprocess"},
            "clustering": {"capability": "clustering.kmeans"},
            "clustering_benchmark": {"capability": "clustering.benchmark"},
            "classification": {"capability": "classification.train"},
            "classification_benchmark": {"capability": "classification.benchmark", "policy": "priority"},
            "regression": {"capability": "regression.train"},
            "regression_benchmark": {"capability": "regression.benchmark", "policy": "priority"},
        },
    },
    "llm": {"enabled": True},
}

CONFIG_FILES = ("config.json", "config.yaml", "config.yml")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    resultado = copy.deepcopy(base)
    for clave, valor in override.items():
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = _merge(resultado[clave], valor)
        else:
            resultado[clave] = copy.deepcopy(valor)
    return resultado


class ConfigStore:
    """Configuración efectiva = valores por defecto + archivo de configuración (si existe)."""

    def __init__(self, path: str | Path | None = None, *, root: Path = PROJECT_ROOT) -> None:
        self.root = root
        self.path = Path(path) if path else self._buscar_archivo()
        self.data = _merge(DEFAULT_CONFIG, self._leer(self.path) if self.path else {})

    def _buscar_archivo(self) -> Path | None:
        for nombre in CONFIG_FILES:
            candidato = self.root / nombre
            if candidato.exists():
                return candidato
        return None

    @staticmethod
    def _leer(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise PersistenceError(f"No existe el archivo de configuración '{path}'.")
        texto = path.read_text(encoding="utf-8")
        try:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml  # type: ignore[import-untyped]
                except ImportError as error:
                    raise PersistenceError("Para usar config.yaml instale PyYAML o use config.json.") from error
                datos = yaml.safe_load(texto) or {}
            else:
                datos = json.loads(texto or "{}")
        except (ValueError, TypeError) as error:
            raise PersistenceError(f"El archivo de configuración '{path.name}' no es válido: {error}") from error
        if not isinstance(datos, dict):
            raise PersistenceError("La configuración debe ser un objeto (clave: valor).")
        return datos

    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor con notación de puntos, por ejemplo ``coordinator.max_retries``."""
        actual: Any = self.data
        for parte in key.split("."):
            if not isinstance(actual, dict) or parte not in actual:
                return default
            actual = actual[parte]
        return copy.deepcopy(actual)

    def set(self, key: str, value: Any) -> None:
        """Modifica un valor en memoria (use :meth:`save` para persistirlo)."""
        partes = key.split(".")
        actual = self.data
        for parte in partes[:-1]:
            actual = actual.setdefault(parte, {})
        actual[partes[-1]] = value

    def path_for(self, key: str) -> Path:
        """Ruta absoluta configurada en ``paths.<key>``."""
        valor = self.get(f"paths.{key}")
        if valor is None:
            raise PersistenceError(f"No hay una ruta configurada para '{key}'.")
        return resolve(valor, self.root)

    def save(self, path: str | Path | None = None) -> Path:
        """Guarda la configuración efectiva en JSON."""
        destino = Path(path) if path else (self.path or self.root / "config.json")
        destino.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.path = destino
        return destino


__all__ = ["DEFAULT_CONFIG", "ConfigStore"]
