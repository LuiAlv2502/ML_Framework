"""Registro, consulta, comparación y exportación de experimentos."""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

from framework_ia.persistence.experiment_store import ExperimentStore
from framework_ia.utils.exceptions import ValidationError


def metrics_table(result: dict[str, Any]) -> pd.DataFrame:
    """Extrae la tabla de métricas de un resultado de agente (benchmark, entrenamiento o agrupamiento)."""
    if not isinstance(result, dict):
        return pd.DataFrame()
    for clave in ("table", "ranking", "metrics_table"):
        tabla = result.get(clave)
        if isinstance(tabla, pd.DataFrame) and not tabla.empty:
            return tabla.copy()
    metricas = result.get("metrics")
    if isinstance(metricas, dict):
        planas = {k: v for k, v in metricas.items() if isinstance(v, (int, float, str, bool)) or v is None}
        if planas:
            return pd.DataFrame([{"Modelo": result.get("label") or result.get("algorithm") or "resultado",
                                  **planas}])
    return pd.DataFrame()


class ExperimentService:
    """Fachada de :class:`ExperimentStore` con utilidades de comparación."""

    def __init__(self, store: ExperimentStore) -> None:
        self.store = store

    def record(
        self,
        *,
        task: str,
        dataset: dict[str, Any],
        config: dict[str, Any],
        result: dict[str, Any] | None = None,
        agents: list[str] | None = None,
        decisions: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        best_model: str | None = None,
        status: str = "ok",
        error: str | None = None,
        started_at: float | None = None,
    ) -> str:
        """Guarda una ejecución y devuelve su identificador."""
        tabla = metrics_table(result or {})
        resumen = {}
        if result and isinstance(result.get("primary_metric"), dict):
            resumen = {result["primary_metric"].get("name", "metrica"): result["primary_metric"].get("value")}
        registro = {
            "task": task,
            "dataset": dataset,
            "config": config,
            "agents": agents or [],
            "decisions": decisions or [],
            "best_model": best_model,
            "status": status,
            "error": error,
            "summary": resumen,
            "duration_s": round(time.perf_counter() - started_at, 3) if started_at else None,
        }
        return self.store.save(registro, metrics=tabla, messages=messages)

    def list(self, **filters: Any) -> list[dict[str, Any]]:
        return self.store.list(**filters)

    def list_frame(self, **filters: Any) -> pd.DataFrame:
        """Historial como tabla para el dashboard."""
        filas = []
        for run in self.store.list(**filters):
            dataset = run.get("dataset") or {}
            filas.append({
                "id": run.get("id"),
                "fecha": run.get("created_at"),
                "tarea": run.get("task"),
                "dataset": dataset.get("name"),
                "agentes": ", ".join(run.get("agents") or []),
                "mejor modelo": run.get("best_model"),
                "estado": run.get("status"),
                "duración (s)": run.get("duration_s"),
                **{f"métrica: {k}": v for k, v in (run.get("summary") or {}).items()},
            })
        return pd.DataFrame(filas)

    def get(self, run_id: str) -> dict[str, Any]:
        return self.store.get(run_id)

    def export(self, run_id: str, formato: str = "json") -> bytes:
        return self.store.export(run_id, formato)

    def delete(self, run_id: str) -> None:
        self.store.delete(run_id)

    def compare(self, run_a: str, run_b: str) -> pd.DataFrame:
        """Compara la configuración y el resumen de dos ejecuciones lado a lado."""
        if run_a == run_b:
            raise ValidationError("Seleccione dos ejecuciones distintas.")
        a, b = self.store.get(run_a)["run"], self.store.get(run_b)["run"]
        campos = ["task", "created_at", "best_model", "status", "duration_s"]
        filas = [{"Campo": c, run_a: a.get(c), run_b: b.get(c)} for c in campos]
        filas.append({"Campo": "dataset", run_a: (a.get("dataset") or {}).get("name"),
                      run_b: (b.get("dataset") or {}).get("name")})
        filas.append({"Campo": "agentes", run_a: ", ".join(a.get("agents") or []),
                      run_b: ", ".join(b.get("agents") or [])})
        for clave in sorted(set(a.get("summary") or {}) | set(b.get("summary") or {})):
            filas.append({"Campo": f"métrica: {clave}", run_a: (a.get("summary") or {}).get(clave),
                          run_b: (b.get("summary") or {}).get(clave)})
        for clave in sorted(set(a.get("config") or {}) | set(b.get("config") or {})):
            filas.append({"Campo": f"config: {clave}", run_a: str((a.get("config") or {}).get(clave)),
                          run_b: str((b.get("config") or {}).get(clave))})
        return pd.DataFrame(filas)


__all__ = ["ExperimentService", "metrics_table"]
