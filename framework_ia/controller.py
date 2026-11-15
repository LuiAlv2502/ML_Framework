"""Controlador/orquestador del framework: **única puerta de entrada** del frontend.

El frontend (Streamlit) nunca instancia clases de ML ni agentes: llama a este controlador, que

1. valida la solicitud y toma el dataset activo de :class:`DatasetService`;
2. construye un :class:`~framework_ia.agents.Message` y lo envía al :class:`CoordinatorAgent` por el bus;
3. el coordinador descubre por capacidades al agente adecuado y le delega la tarea;
4. guarda la ejecución con :class:`ExperimentService` (configuración, agentes, decisiones, métricas, mensajes);
5. devuelve al frontend solo estructuras serializables o DataFrames de resultados (nunca objetos de sklearn).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from framework_ia.agents import Agency, build_default_agency, llm
from framework_ia.persistence.config_store import ConfigStore
from framework_ia.persistence.experiment_store import ExperimentStore
from framework_ia.services.dataset_service import DatasetService
from framework_ia.services.experiment_service import ExperimentService
from framework_ia.services.preprocessing_service import PreprocessingService
from framework_ia.utils.exceptions import TaskExecutionError, ValidationError
from framework_ia.utils.logging_config import configure_logging, get_logger
from framework_ia.utils.validation import validate_task_type

logger = get_logger("controller")

SUPERVISED = ("classification", "regression")
CLUSTERING_METHODS = ("pca", "kmeans", "hac", "tsne", "umap", "elbow", "dendrogram")
BENCHMARK_TASKS = {"classification": "classification_benchmark", "regression": "regression_benchmark",
                   "clustering": "clustering_benchmark"}


def _publico(payload: dict[str, Any]) -> dict[str, Any]:
    """Configuración registrable de una solicitud (sin los datos)."""
    return {k: v for k, v in payload.items() if k != "data"}


class FrameworkController:
    """Fachada del backend para el dashboard (y para scripts o pruebas)."""

    def __init__(
        self,
        config: ConfigStore | None = None,
        *,
        agency: Agency | None = None,
        persist: bool = True,
        runs_dir: str | Path | None = None,
    ) -> None:
        self.config = config or ConfigStore()
        configure_logging(self.config.path_for("logs_dir"))
        self.datasets = DatasetService(self.config)
        self.preprocessing = PreprocessingService()
        self.experiments = ExperimentService(ExperimentStore(runs_dir or self.config.path_for("runs_dir")))
        self.agency = agency or build_default_agency(self.config, state_store=persist, model_store=persist)
        self.llm_enabled = bool(self.config.get("llm.enabled", True))
        self.eda_result: dict[str, Any] | None = None
        self.eda_hash: str | None = None
        self.last_task: dict[str, Any] | None = None

    # ==================================================================
    # Datos
    # ==================================================================
    @property
    def has_data(self) -> bool:
        return self.datasets.has_data

    def load_dataset(self, content: bytes, filename: str) -> dict[str, Any]:
        """Carga un archivo subido (CSV/Excel) y lo deja como dataset activo."""
        info = self.datasets.load_bytes(content, filename)
        self._reiniciar_resultados()
        return info.to_dict()

    def load_sample(self, name: str) -> dict[str, Any]:
        """Carga uno de los archivos de la carpeta ``data/``."""
        disponibles = {d["name"]: d["path"] for d in self.datasets.list_available()}
        if name not in disponibles:
            raise ValidationError(f"No existe el dataset '{name}' en la carpeta de datos.")
        info = self.datasets.load_path(disponibles[name])
        self._reiniciar_resultados()
        return info.to_dict()

    def load_dataframe(self, df: pd.DataFrame, name: str = "datos") -> dict[str, Any]:
        info = self.datasets.load_dataframe(df, name)
        self._reiniciar_resultados()
        return info.to_dict()

    def _reiniciar_resultados(self) -> None:
        self.eda_result = None
        self.eda_hash = None

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.datasets.list_available()

    def dataset_info(self) -> dict[str, Any]:
        return self.datasets.dataset_dict()

    def active_data(self) -> pd.DataFrame:
        """Copia del dataset activo (para mostrar y graficar)."""
        return self.datasets.active

    def original_data(self) -> pd.DataFrame:
        return self.datasets.original

    def dataset_summary(self) -> dict[str, Any]:
        return self.datasets.summary()

    def dataset_preview(self, n: int = 10, *, tail: bool = False) -> pd.DataFrame:
        return self.datasets.preview(n, tail=tail)

    def eda_tables(self) -> dict[str, Any]:
        return self.datasets.eda_tables()

    def target_summary(self, target: Any) -> dict[str, Any]:
        return self.datasets.target_summary(target)

    def suggest_target(self) -> Any:
        return self.datasets.suggest_target()

    def suggest_task(self, target: Any) -> str:
        return self.datasets.suggest_task(target)

    def restore_dataset(self) -> dict[str, Any]:
        """Vuelve a activar una copia del archivo original."""
        return self.datasets.restore().to_dict()

    def detect_id_columns(self) -> list[Any]:
        return self.preprocessing.detect_id_columns(self.datasets.active)

    def filter_preview(self, column: Any, operator: str, value: Any) -> pd.DataFrame:
        return self.preprocessing.filter_preview(self.datasets.active, column, operator, value)

    def preprocessing_operations(self) -> dict[str, str]:
        return self.preprocessing.available_operations()

    def partitions(self, *, task: str, target: Any, features: list[Any] | None, test_size: float,
                   seeds: list[int], stratify: bool = False) -> list[dict[str, Any]]:
        """Particiones aprendizaje/prueba (índices de prueba por semilla) idénticas a las de los modelos."""
        return self.preprocessing.partitions(self.datasets.active, task=task, target=target, features=features,
                                             test_size=test_size, seeds=seeds, stratify=stratify)

    # ==================================================================
    # Ejecución de tareas a través del coordinador
    # ==================================================================
    def _use_llm(self, use_llm: bool | None) -> bool:
        return self.llm_enabled if use_llm is None else bool(use_llm) and self.llm_enabled

    def run_task(self, task: str, payload: dict[str, Any] | None = None, *, record: bool = True) -> dict[str, Any]:
        """Envía una tarea al coordinador con el dataset activo y devuelve el resultado consolidado."""
        contenido = {"data": self.datasets.active, **(payload or {})}
        inicio = time.perf_counter()
        respuesta = self.agency.request(task, contenido, sender="controller")
        mensajes = self.agency.bus.records(correlation_id=respuesta.correlation_id)
        if not respuesta.ok:
            decisiones = respuesta.payload.get("decisions", [])
            if record:
                self.experiments.record(
                    task=task, dataset=self.datasets.dataset_dict(), config=_publico(contenido),
                    decisions=decisiones, messages=mensajes, status="error", error=respuesta.error,
                    started_at=inicio,
                )
            self.last_task = {"task": task, "ok": False, "error": respuesta.error}
            logger.warning("Tarea %s falló: %s", task, respuesta.error)
            if respuesta.payload.get("user_error"):
                raise ValidationError(respuesta.error or "Solicitud inválida.")
            raise TaskExecutionError(respuesta.error or "La tarea falló.", details={"decisions": decisiones})
        resultado = dict(respuesta.payload)
        resultado["messages"] = mensajes
        detalle = resultado["result"]
        mejor = detalle.get("best") or detalle.get("label") or detalle.get("algorithm")
        if record:
            resultado["run_id"] = self.experiments.record(
                task=task, dataset=self.datasets.dataset_dict(), config=_publico(contenido), result=detalle,
                agents=resultado["agents_used"], decisions=resultado["decisions"], messages=mensajes,
                best_model=str(mejor) if mejor is not None else None, started_at=inicio,
                coordination={
                    "policy": resultado["policy"],
                    "candidates": resultado["candidates"],
                    "selected_agent": resultado["selected_agent"],
                },
            )
        self.last_task = {"task": task, "ok": True, "agent": resultado["selected_agent"],
                          "run_id": resultado.get("run_id")}
        return resultado

    def run_case04_evaluation(
        self, target: Any, seeds: list[int] | tuple[int, ...] | None = None
    ) -> dict[str, Any]:
        """Ejecuta los escenarios integrados del Estudio de Caso 04."""
        from framework_ia.agents.case_study import ejecutar_demostracion_caso04

        return ejecutar_demostracion_caso04(self, target, seeds)

    # ------------------------------------------------------------------
    # EDA y preprocesamiento
    # ------------------------------------------------------------------
    def run_eda(self, target: Any = None, *, use_llm: bool | None = None, activate: bool = True) -> dict[str, Any]:
        """Ejecuta el agente EDA sobre el archivo original; opcionalmente activa su copia preparada."""
        payload = {"target": target, "use_llm": self._use_llm(use_llm)}
        contenido = {"data": self.datasets.original, **payload}
        inicio = time.perf_counter()
        respuesta = self.agency.request("eda", contenido)
        if not respuesta.ok:
            if respuesta.payload.get("user_error"):
                raise ValidationError(respuesta.error or "Solicitud inválida.")
            raise TaskExecutionError(respuesta.error or "El EDA falló.")
        resultado = dict(respuesta.payload)
        mensajes = self.agency.bus.records(correlation_id=respuesta.correlation_id)
        resultado["messages"] = mensajes
        resultado["run_id"] = self.experiments.record(
            task="eda", dataset=self.datasets.dataset_dict(), config=payload, result=resultado["result"],
            agents=resultado["agents_used"], decisions=resultado["decisions"], messages=mensajes,
            started_at=inicio,
            coordination={
                "policy": resultado["policy"],
                "candidates": resultado["candidates"],
                "selected_agent": resultado["selected_agent"],
            },
        )
        self.eda_result = resultado
        self.eda_hash = (self.datasets.info.hash if self.datasets.info else None)
        if activate:
            self.use_eda_preparation()
        self.last_task = {"task": "eda", "ok": True, "agent": resultado["selected_agent"],
                          "run_id": resultado["run_id"]}
        return resultado

    def use_eda_preparation(self) -> dict[str, Any]:
        """Activa la copia preparada por el agente EDA (si corresponde al archivo actual)."""
        if self.eda_result is None or self.datasets.info is None or self.eda_hash != self.datasets.info.hash:
            raise ValidationError("No hay una preparación del agente EDA vigente para este archivo.")
        datos = self.eda_result["result"]["prepared_data"]
        return self.datasets.set_active(datos, "agente", "Copia preparada por el agente EDA").to_dict()

    def clean(self, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Limpieza manual (EDA): aplica las operaciones y deja el resultado como dataset activo."""
        resultado = self.run_task("eda.clean", {"operations": operations}, record=False)
        self.datasets.set_active(resultado["result"]["data"], "manual",
                                 ", ".join(o.get("op", "?") for o in operations))
        return resultado

    def configure_preprocessing(self, operations: list[dict[str, Any]], *, apply: bool = True) -> dict[str, Any]:
        """Preprocesamiento configurable; con ``apply=True`` el resultado pasa a ser el dataset activo."""
        resultado = self.run_task("preprocessing", {"operations": operations})
        if apply:
            self.datasets.set_active(resultado["result"]["data"], "preprocesado",
                                     f"{len(operations)} operaciones de preprocesamiento")
        return resultado

    def correlations(self) -> dict[str, Any]:
        return self.run_task("eda.correlations", record=False)["result"]

    # ------------------------------------------------------------------
    # Aprendizaje automático
    # ------------------------------------------------------------------
    def run_clustering(self, method: str, params: dict[str, Any] | None = None, *, columns: list[Any] | None = None,
                       projection_dims: int = 2, record: bool = True) -> dict[str, Any]:
        """PCA, K-Means, HAC, t-SNE, UMAP, método del codo o dendrograma."""
        if method not in CLUSTERING_METHODS:
            raise ValidationError(f"Método no soportado '{method}'. Opciones: {', '.join(CLUSTERING_METHODS)}.")
        payload = {"params": params or {}, "projection_dims": projection_dims}
        if columns:
            payload["columns"] = list(columns)
        return self.run_task(f"clustering.{method}", payload, record=record)

    def _supervisado(self, task: str, *, target: Any, algorithm: str, params: dict[str, Any] | None,
                     features: list[Any] | None, test_size: float, seeds: list[int] | None, stratify: bool,
                     scale: bool, save_model: bool, label: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"target": target, "algorithm": algorithm, "params": params or {},
                                   "test_size": test_size, "seeds": seeds or [int(self.config.get("random_state", 42))],
                                   "scale": scale, "save_model": save_model}
        if features is not None:
            payload["features"] = list(features)
        if task == "classification":
            payload["stratify"] = stratify
        if label:
            payload["label"] = label
        return self.run_task(task, payload)

    def run_classification(self, *, target: Any, algorithm: str, params: dict[str, Any] | None = None,
                           features: list[Any] | None = None, test_size: float = 0.25,
                           seeds: list[int] | None = None, stratify: bool = True, scale: bool = True,
                           save_model: bool = False, label: str | None = None) -> dict[str, Any]:
        return self._supervisado("classification", target=target, algorithm=algorithm, params=params,
                                 features=features, test_size=test_size, seeds=seeds, stratify=stratify,
                                 scale=scale, save_model=save_model, label=label)

    def run_regression(self, *, target: Any, algorithm: str, params: dict[str, Any] | None = None,
                       features: list[Any] | None = None, test_size: float = 0.25, seeds: list[int] | None = None,
                       scale: bool = True, save_model: bool = False, label: str | None = None) -> dict[str, Any]:
        return self._supervisado("regression", target=target, algorithm=algorithm, params=params,
                                 features=features, test_size=test_size, seeds=seeds, stratify=False,
                                 scale=scale, save_model=save_model, label=label)

    def run_benchmark(self, task: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Benchmark uniforme (modelo base + variaciones) para clasificación, regresión o agrupamiento."""
        if task not in BENCHMARK_TASKS:
            raise ValidationError(f"Benchmark no soportado para '{task}'. Opciones: {', '.join(BENCHMARK_TASKS)}.")
        return self.run_task(BENCHMARK_TASKS[task], dict(config or {}))

    def run_auto(self, task: str, *, target: Any = None, repetitions: int = 5, test_size: float = 0.25,
                 use_llm: bool | None = None) -> dict[str, Any]:
        """Modo automático: el agente del dominio decide qué validar (LLM opcional + respaldo por reglas)."""
        validate_task_type(task)
        if task not in (*SUPERVISED, "clustering"):
            raise ValidationError("El modo automático aplica a clustering, classification o regression.")
        payload: dict[str, Any] = {"use_llm": self._use_llm(use_llm)}
        if task in SUPERVISED:
            payload.update({"target": target, "repetitions": repetitions, "test_size": test_size})
        return self.run_task(f"{task}.auto", payload)

    # ==================================================================
    # Sistema multiagente
    # ==================================================================
    def list_agents(self) -> list[dict[str, Any]]:
        return self.agency.registry.snapshot()

    def agent_types(self) -> list[dict[str, Any]]:
        return self.agency.factory.available_types()

    def capabilities_map(self) -> dict[str, list[str]]:
        return self.agency.registry.capabilities()

    def list_messages(self, **filters: Any) -> list[dict[str, Any]]:
        return self.agency.bus.records(**{k: v for k, v in filters.items() if v not in (None, "")})

    def registry_events(self) -> list[dict[str, Any]]:
        return self.agency.registry.events

    def create_agent(self, definition: dict[str, Any], *, persist: bool = True) -> dict[str, Any]:
        """Valida, crea, registra y (por defecto) persiste un agente nuevo."""
        if definition.get("agent_type") == "coordinator":
            raise ValidationError("Solo puede existir un coordinador.")
        return self.agency.create_agent(definition, persist=persist).describe()

    def probe_agent(self, agent_id: str, capability: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Pone un agente primero temporalmente y comprueba su descubrimiento por el coordinador."""
        agente = self.agency.registry.get(agent_id)
        if capability not in agente.capabilities:
            raise ValidationError(f"El agente '{agente.name}' no tiene asignada la tarea '{capability}'.")
        anteriores = {a.agent_id: a.priority for a in self.agency.registry.find_by_capability(capability)}
        try:
            for candidato in self.agency.registry.find_by_capability(capability):
                candidato.priority = 100 if candidato.agent_id == agent_id else min(candidato.priority, 99)
            return self.run_task(capability, payload)
        finally:
            for candidato_id, prioridad in anteriores.items():
                self.agency.registry.get(candidato_id).priority = prioridad

    def remove_agent(self, agent_id: str) -> None:
        self.agency.remove_agent(agent_id, forget=True)

    def agent_lifecycle(self, agent_id: str, action: str) -> dict[str, Any]:
        """Acciones del ciclo de vida: ``start``, ``stop``, ``pause``, ``resume`` o ``restart``."""
        acciones = {"start": self.agency.start_agent, "stop": self.agency.stop_agent,
                    "pause": self.agency.pause_agent, "resume": self.agency.resume_agent,
                    "restart": self.agency.restart_agent}
        if action not in acciones:
            raise ValidationError(f"Acción inválida '{action}'. Opciones: {', '.join(acciones)}.")
        if self.agency.coordinator is not None and agent_id == self.agency.coordinator.agent_id \
                and action in ("stop", "pause"):
            raise ValidationError("No se puede detener ni pausar al coordinador.")
        return acciones[action](agent_id).describe()

    def mas_status(self) -> dict[str, Any]:
        """Estado general del MAS, del dataset y del LLM (página de Inicio)."""
        return {**self.agency.status_summary(), "llm": self.llm_status(), "dataset": self.dataset_info(),
                "last_task": self.last_task, "rehydration_errors": getattr(self.agency, "rehydration_errors", {}),
                "plugin_errors": dict(self.agency.factory.plugin_errors)}

    def llm_status(self) -> dict[str, Any]:
        motivo = llm.motivo_no_disponible()
        return {"enabled": self.llm_enabled, "available": motivo is None and self.llm_enabled,
                "model": llm.MODELO_POR_DEFECTO,
                "reason": motivo or ("" if self.llm_enabled else "Desactivado en config.json (llm.enabled).")}

    # ==================================================================
    # Experimentos
    # ==================================================================
    def list_experiments(self, **filters: Any) -> pd.DataFrame:
        return self.experiments.list_frame(**{k: v for k, v in filters.items() if v not in (None, "")})

    def get_experiment(self, run_id: str) -> dict[str, Any]:
        return self.experiments.get(run_id)

    def export_experiment(self, run_id: str, formato: str = "json") -> bytes:
        return self.experiments.export(run_id, formato)

    def compare_experiments(self, run_a: str, run_b: str) -> pd.DataFrame:
        return self.experiments.compare(run_a, run_b)

    def delete_experiment(self, run_id: str) -> None:
        self.experiments.delete(run_id)


__all__ = ["BENCHMARK_TASKS", "CLUSTERING_METHODS", "FrameworkController"]
