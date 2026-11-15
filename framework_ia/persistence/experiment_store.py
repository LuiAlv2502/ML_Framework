"""Almacenamiento de ejecuciones (experimentos) en ``runs/<run_id>/``.

Cada ejecución guarda:

- ``run.json``: id, fecha, dataset y hash, tarea, configuración, agentes participantes, decisiones del
  coordinador, mejor modelo y resumen.
- ``metrics.csv``: tabla de métricas (si la hay).
- ``messages.jsonl``: mensajes intercambiados entre agentes durante la ejecución.

Además mantiene un índice ``runs/index.json`` para listar sin abrir cada carpeta.
"""
from __future__ import annotations

import html
import io
import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from framework_ia.utils.exceptions import PersistenceError, ValidationError
from framework_ia.utils.serialization import to_jsonable

EXPORT_FORMATS = ("json", "csv", "html")
INDEX_FIELDS = ("id", "created_at", "task", "dataset", "best_model", "agents", "status", "duration_s", "summary")


class ExperimentStore:
    """Persistencia de experimentos con índice; segura ante hilos dentro de un mismo proceso."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Índice
    # ------------------------------------------------------------------
    @property
    def index_path(self) -> Path:
        return self.base_dir / "index.json"

    def _leer_indice(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        try:
            datos = json.loads(self.index_path.read_text(encoding="utf-8") or "[]")
        except ValueError as error:
            raise PersistenceError(f"El índice de experimentos está dañado: {error}") from error
        return datos if isinstance(datos, list) else []

    def _escribir_indice(self, indice: list[dict[str, Any]]) -> None:
        temporal = self.index_path.with_suffix(".tmp")
        temporal.write_text(json.dumps(indice, ensure_ascii=False, indent=2), encoding="utf-8")
        temporal.replace(self.index_path)

    # ------------------------------------------------------------------
    # Operaciones
    # ------------------------------------------------------------------
    @staticmethod
    def new_run_id(task: str) -> str:
        """Identificador ordenable por fecha: ``AAAAMMDD-HHMMSS_<tarea>_<8 hex>``."""
        marca = datetime.now().strftime("%Y%m%d-%H%M%S")
        tarea = "".join(c if c.isalnum() else "-" for c in task)[:30]
        return f"{marca}_{tarea}_{uuid.uuid4().hex[:8]}"

    def save(
        self,
        record: dict[str, Any],
        *,
        metrics: pd.DataFrame | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """Guarda una ejecución y devuelve su ``run_id``."""
        if not isinstance(record, dict) or not record.get("task"):
            raise ValidationError("El registro del experimento debe indicar la tarea ('task').")
        with self._lock:
            run_id = record.get("id") or self.new_run_id(str(record["task"]))
            carpeta = self.base_dir / run_id
            carpeta.mkdir(parents=True, exist_ok=True)
            registro = {"created_at": datetime.now().isoformat(timespec="seconds"), **record, "id": run_id}
            registro.setdefault("status", "ok")
            try:
                (carpeta / "run.json").write_text(
                    json.dumps(to_jsonable(registro), ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if metrics is not None and not metrics.empty:
                    metrics.to_csv(carpeta / "metrics.csv", index=False)
                with (carpeta / "messages.jsonl").open("w", encoding="utf-8") as archivo:
                    for mensaje in messages or []:
                        archivo.write(json.dumps(to_jsonable(mensaje), ensure_ascii=False) + "\n")
            except OSError as error:
                raise PersistenceError(f"No se pudo guardar el experimento: {error}") from error
            indice = [fila for fila in self._leer_indice() if fila.get("id") != run_id]
            indice.append({campo: to_jsonable(registro.get(campo)) for campo in INDEX_FIELDS})
            self._escribir_indice(indice)
            return run_id

    def list(
        self,
        *,
        task: str | None = None,
        dataset: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        """Resumen de las ejecuciones (más recientes primero) con filtros opcionales.

        ``since``/``until`` son fechas ISO (``AAAA-MM-DD``) inclusivas.
        """
        with self._lock:
            filas = self._leer_indice()
        if task:
            filas = [f for f in filas if f.get("task") == task]
        if dataset:
            filas = [f for f in filas if (f.get("dataset") or {}).get("name") == dataset]
        if since:
            filas = [f for f in filas if str(f.get("created_at", ""))[:10] >= since[:10]]
        if until:
            filas = [f for f in filas if str(f.get("created_at", ""))[:10] <= until[:10]]
        return sorted(filas, key=lambda f: str(f.get("created_at", "")), reverse=True)

    def get(self, run_id: str) -> dict[str, Any]:
        """Devuelve ``{"run": dict, "metrics": DataFrame, "messages": list}`` de una ejecución."""
        carpeta = self.base_dir / run_id
        archivo = carpeta / "run.json"
        if not archivo.exists():
            raise PersistenceError(f"No existe el experimento '{run_id}'.")
        run = json.loads(archivo.read_text(encoding="utf-8"))
        metricas = pd.read_csv(carpeta / "metrics.csv") if (carpeta / "metrics.csv").exists() else pd.DataFrame()
        mensajes = []
        if (carpeta / "messages.jsonl").exists():
            for linea in (carpeta / "messages.jsonl").read_text(encoding="utf-8").splitlines():
                if linea.strip():
                    mensajes.append(json.loads(linea))
        return {"run": run, "metrics": metricas, "messages": mensajes}

    def delete(self, run_id: str) -> None:
        """Elimina una ejecución y la quita del índice."""
        with self._lock:
            shutil.rmtree(self.base_dir / run_id, ignore_errors=True)
            self._escribir_indice([f for f in self._leer_indice() if f.get("id") != run_id])

    def export(self, run_id: str, formato: str = "json", destination: str | Path | None = None) -> bytes:
        """Exporta una ejecución a JSON, CSV (métricas) o HTML; opcionalmente la escribe en ``destination``."""
        if formato not in EXPORT_FORMATS:
            raise ValidationError(f"Formato de exportación inválido. Opciones: {', '.join(EXPORT_FORMATS)}.")
        datos = self.get(run_id)
        if formato == "json":
            contenido = json.dumps(
                {**datos, "metrics": to_jsonable(datos["metrics"])}, ensure_ascii=False, indent=2
            ).encode("utf-8")
        elif formato == "csv":
            buffer = io.StringIO()
            metricas = datos["metrics"]
            (metricas if not metricas.empty else pd.DataFrame([datos["run"].get("summary") or {}])).to_csv(
                buffer, index=False
            )
            contenido = buffer.getvalue().encode("utf-8")
        else:
            contenido = self._html(datos).encode("utf-8")
        if destination is not None:
            Path(destination).write_bytes(contenido)
        return contenido

    @staticmethod
    def _html(datos: dict[str, Any]) -> str:
        run = datos["run"]
        e = html.escape
        partes = [
            "<!doctype html><html lang='es'><head><meta charset='utf-8'>",
            f"<title>Experimento {e(run['id'])}</title><style>body{{font-family:Arial,sans-serif;max-width:1000px;"
            "margin:30px auto;color:#222}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;"
            "padding:4px 8px}h1,h2{color:#1f4e79}</style></head><body>",
            f"<h1>Experimento {e(run['id'])}</h1>",
            f"<p><b>Fecha:</b> {e(str(run.get('created_at')))} · <b>Tarea:</b> {e(str(run.get('task')))} · "
            f"<b>Estado:</b> {e(str(run.get('status')))}</p>",
            f"<p><b>Dataset:</b> {e(json.dumps(run.get('dataset'), ensure_ascii=False))}</p>",
            f"<p><b>Agentes:</b> {e(', '.join(map(str, run.get('agents') or [])))} · "
            f"<b>Mejor modelo:</b> {e(str(run.get('best_model')))}</p>",
            "<h2>Configuración</h2>",
            f"<pre>{e(json.dumps(run.get('config'), ensure_ascii=False, indent=2))}</pre>",
            "<h2>Decisiones del coordinador</h2><ul>",
        ]
        for decision in run.get("decisions") or []:
            partes.append(
                f"<li><b>{e(str(decision.get('decision')))}</b> — {e(str(decision.get('justification')))}</li>"
            )
        partes.append("</ul>")
        if not datos["metrics"].empty:
            partes.append("<h2>Métricas</h2>" + datos["metrics"].to_html(index=False, border=0))
        partes.append(f"<h2>Mensajes ({len(datos['messages'])})</h2><table><tr><th>Hora</th><th>Emisor</th>"
                      "<th>Receptor</th><th>Performativa</th><th>Tarea</th><th>Estado</th></tr>")
        for mensaje in datos["messages"]:
            partes.append("<tr>" + "".join(
                f"<td>{e(str(mensaje.get(c, '')))}</td>"
                for c in ("timestamp", "sender", "receiver", "performative", "task", "status")
            ) + "</tr>")
        partes.append("</table></body></html>")
        return "\n".join(partes)


__all__ = ["EXPORT_FORMATS", "ExperimentStore"]
