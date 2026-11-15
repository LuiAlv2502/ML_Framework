"""Almacenamiento de modelos entrenados con ``joblib`` y sus metadatos en JSON."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from framework_ia.utils.exceptions import PersistenceError
from framework_ia.utils.serialization import to_jsonable


class ModelStore:
    """Guarda ``<model_id>.joblib`` + ``<model_id>.json`` en ``base_dir`` (por defecto ``framework_ia/models``)."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def save(self, model: Any, name: str, metadata: dict[str, Any] | None = None) -> str:
        """Serializa el modelo y devuelve su identificador."""
        nombre = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)[:40] or "modelo"
        model_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{nombre}_{uuid.uuid4().hex[:6]}"
        info = {
            "id": model_id,
            "name": name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "class": type(model).__name__,
            **(metadata or {}),
        }
        with self._lock:
            try:
                joblib.dump(model, self.base_dir / f"{model_id}.joblib")
                (self.base_dir / f"{model_id}.json").write_text(
                    json.dumps(to_jsonable(info), ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except (OSError, TypeError, ValueError) as error:
                raise PersistenceError(f"No se pudo guardar el modelo '{name}': {error}") from error
        return model_id

    def load(self, model_id: str) -> Any:
        """Carga un modelo guardado."""
        ruta = self.base_dir / f"{model_id}.joblib"
        if not ruta.exists():
            raise PersistenceError(f"No existe el modelo '{model_id}'.")
        return joblib.load(ruta)

    def metadata(self, model_id: str) -> dict[str, Any]:
        """Metadatos de un modelo guardado."""
        ruta = self.base_dir / f"{model_id}.json"
        if not ruta.exists():
            raise PersistenceError(f"No existen metadatos para el modelo '{model_id}'.")
        return json.loads(ruta.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        """Metadatos de todos los modelos (más recientes primero)."""
        filas = [json.loads(p.read_text(encoding="utf-8")) for p in self.base_dir.glob("*.json")]
        return sorted(filas, key=lambda f: f.get("created_at", ""), reverse=True)

    def delete(self, model_id: str) -> None:
        """Elimina un modelo y sus metadatos."""
        for sufijo in (".joblib", ".json"):
            (self.base_dir / f"{model_id}{sufijo}").unlink(missing_ok=True)


__all__ = ["ModelStore"]
