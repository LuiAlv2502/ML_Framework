"""Escenarios reproducibles para evaluar el Estudio de Caso 04 dentro del MAS."""
from __future__ import annotations

import math
import statistics
import time
import uuid
from typing import TYPE_CHECKING, Any

import pandas as pd

from framework_ia.agents.bootstrap import build_default_agency
from framework_ia.ml.classification import ClasificacionModelos
from framework_ia.utils.exceptions import TaskExecutionError, ValidationError
from framework_ia.utils.validation import validate_target

if TYPE_CHECKING:
    from framework_ia.controller import FrameworkController


DEFAULT_SEEDS = (42, 43, 44)
TEST_SIZE = 0.25


def _validate_seeds(seeds: list[int] | tuple[int, ...] | None) -> list[int]:
    valores = list(DEFAULT_SEEDS if seeds is None else seeds)
    if not valores:
        raise ValidationError("Seleccione al menos una semilla para repetir los experimentos.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in valores):
        raise ValidationError("Las semillas deben ser números enteros.")
    if len(set(valores)) != len(valores):
        raise ValidationError("No repita semillas en una misma evaluación.")
    return valores


def _estadisticas(valores: list[float]) -> tuple[float | None, float | None]:
    if not valores:
        return None, None
    media = statistics.mean(valores)
    desviacion = statistics.stdev(valores) if len(valores) > 1 else 0.0
    return round(media, 6), round(desviacion, 6)


def _traza_mensajes(mensajes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    campos = ("timestamp", "sender", "receiver", "performative", "task", "status", "correlation_id", "error")
    return [{campo: mensaje.get(campo) for campo in campos} for mensaje in mensajes]


def _ejecutar_tarea(
    controlador: FrameworkController, tarea: str, payload: dict[str, Any]
) -> dict[str, Any]:
    inicio = time.perf_counter()
    respuesta = controlador.run_task(tarea, payload)
    duracion = round(time.perf_counter() - inicio, 4)
    metrica = (respuesta.get("result") or {}).get("primary_metric") or {}
    return {
        "task": tarea,
        "result": respuesta,
        "duration_s": duracion,
        "interactions": len(respuesta.get("messages", [])),
        "tasks_completed": len(respuesta.get("agents_used", [])),
        "errors": sum(not intento.get("ok", False) for intento in respuesta.get("attempts", [])),
        "quality_metric": metrica.get("name"),
        "quality": metrica.get("value"),
    }


def _resumen_tarea(tarea: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    respuesta = tarea["result"]
    return {
        "task": tarea["task"],
        "seed": seed,
        "run_id": respuesta.get("run_id"),
        "duration_s": tarea["duration_s"],
        "interactions": tarea["interactions"],
        "tasks_completed": tarea["tasks_completed"],
        "errors": tarea["errors"],
        "quality_metric": tarea["quality_metric"],
        "quality": tarea["quality"],
        "agents": respuesta.get("agents_used", []),
        "selected_agent": respuesta.get("selected_agent"),
        "policy": respuesta.get("policy"),
        "candidates": respuesta.get("candidates", []),
        "alternatives": respuesta.get("alternatives", []),
        "decisions": respuesta.get("decisions", []),
        "messages": _traza_mensajes(respuesta.get("messages", [])),
    }


def _resumir_escenario(
    nombre: str,
    complejidad: str,
    tareas: list[dict[str, Any]],
    ejecuciones: list[dict[str, Any]],
) -> dict[str, Any]:
    calidades = [
        float(tarea["quality"])
        for tarea, _ in ejecuciones
        if tarea["quality"] is not None
    ]
    media, desviacion = _estadisticas(calidades)
    semillas = {seed for _, seed in ejecuciones if seed is not None}
    return {
        "scenario": nombre,
        "complexity": complejidad,
        "trials": len(semillas) if semillas else len(ejecuciones),
        "duration_s": round(sum(tarea["duration_s"] for tarea in tareas), 4),
        "interactions": sum(tarea["interactions"] for tarea in tareas),
        "tasks_completed": sum(tarea["tasks_completed"] for tarea in tareas),
        "errors": sum(tarea["errors"] for tarea in tareas),
        "quality_metric": next(
            (
                tarea["quality_metric"]
                for tarea, _ in ejecuciones
                if tarea["quality_metric"]
            ),
            None,
        ),
        "quality_mean": media,
        "quality_std": desviacion,
        "agents": list(dict.fromkeys(
            agente for tarea in tareas for agente in tarea["result"].get("agents_used", [])
        )),
        "run_ids": [
            tarea["result"].get("run_id")
            for tarea in tareas
            if tarea["result"].get("run_id")
        ],
        "task_runs": [
            _resumen_tarea(tarea, seed=seed)
            for tarea, seed in ejecuciones
        ],
    }


def _evaluacion_centralizada(
    datos: pd.DataFrame, objetivo: Any, k_values: list[int], seed: int
) -> dict[str, Any]:
    """Evalúa las mismas alternativas del MAS sin coordinador, bus ni agentes."""
    modelo = ClasificacionModelos(
        datos,
        target=objetivo,
        test_size=TEST_SIZE,
        random_state=seed,
        scale=True,
        stratify=True,
    )
    fijo = modelo.knn(n_neighbors=5)["metricas"]
    auto_evaluaciones = [
        {"k": k, "metrics": modelo.knn(n_neighbors=k)["metricas"]}
        for k in k_values
    ]
    mejor_auto = max(auto_evaluaciones, key=lambda evaluacion: evaluacion["metrics"]["f1_macro"])
    evaluaciones = [
        {"label": "KNN fijo (k=5)", "metrics": fijo, "k": 5},
        {"label": "KNN automático", **mejor_auto},
    ]
    mejor = max(evaluaciones, key=lambda evaluacion: evaluacion["metrics"]["f1_macro"])
    metricas = mejor["metrics"]
    return {
        "label": mejor["label"],
        "algorithm": "knn",
        "params": {"n_neighbors": mejor["k"]},
        "metrics": {
            nombre: metricas[nombre]
            for nombre in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        },
        "primary_metric": {"name": "f1_macro", "value": metricas["f1_macro"], "maximize": True},
        "evaluations": [
            {
                "label": evaluacion["label"],
                "k": evaluacion["k"],
                "f1_macro": evaluacion["metrics"]["f1_macro"],
            }
            for evaluacion in evaluaciones
        ],
        "models_evaluated": len(k_values) + 1,
    }


def _ejecutar_fanout_aislado(
    controlador: FrameworkController,
    agencia: Any,
    datos: pd.DataFrame,
    objetivo: Any,
    seed: int,
    k_values: list[int],
) -> tuple[dict[str, Any], float]:
    configuracion = {
        "target": objetivo,
        "algorithm": "knn",
        "params": {"n_neighbors": 5},
        "test_size": TEST_SIZE,
        "scale": True,
        "stratify": True,
        "seeds": [seed],
        "policy": "fanout",
    }
    inicio = time.perf_counter()
    respuesta = agencia.request(
        "classification",
        {"data": datos, **configuracion},
        sender="controller",
    )
    mensajes = agencia.bus.records(correlation_id=respuesta.correlation_id)
    duracion = round(time.perf_counter() - inicio, 4)
    if not respuesta.ok:
        decisiones = respuesta.payload.get("decisions", [])
        error = respuesta.error or "La comparación multiagente falló."
        controlador.experiments.record(
            task="classification",
            dataset=controlador.dataset_info(),
            config={**{k: v for k, v in configuracion.items() if k != "data"}, "k_values": k_values},
            decisions=decisiones,
            messages=mensajes,
            status="error",
            error=error,
            started_at=inicio,
        )
        if respuesta.payload.get("user_error"):
            raise ValidationError(error)
        raise TaskExecutionError(error, details={"decisions": decisiones})

    resultado = dict(respuesta.payload)
    resultado["messages"] = mensajes
    detalle = resultado["result"]
    mejor = detalle.get("best") or detalle.get("label") or detalle.get("algorithm")
    resultado["run_id"] = controlador.experiments.record(
        task="classification",
        dataset=controlador.dataset_info(),
        config={**configuracion, "k_values": k_values},
        result=detalle,
        agents=resultado["agents_used"],
        decisions=resultado["decisions"],
        messages=mensajes,
        best_model=str(mejor) if mejor is not None else None,
        started_at=inicio,
        coordination={
            "policy": resultado["policy"],
            "candidates": resultado["candidates"],
            "selected_agent": resultado["selected_agent"],
        },
    )
    return resultado, duracion


def _resumir_comparacion(filas: list[dict[str, Any]], enfoque: str) -> dict[str, Any]:
    propias = [fila for fila in filas if fila["approach"] == enfoque]
    calidad, calidad_std = _estadisticas([float(fila["quality"]) for fila in propias])
    duracion, duracion_std = _estadisticas([float(fila["duration_s"]) for fila in propias])
    interacciones, _ = _estadisticas([float(fila["interactions"]) for fila in propias])
    tareas, _ = _estadisticas([float(fila["tasks_completed"]) for fila in propias])
    return {
        "approach": enfoque,
        "trials": len(propias),
        "duration_s": duracion,
        "duration_std": duracion_std,
        "interactions": interacciones,
        "tasks_completed": tareas,
        "models_evaluated": max((fila["models_evaluated"] for fila in propias), default=0),
        "errors": sum(fila["errors"] for fila in propias),
        "quality": calidad,
        "quality_std": calidad_std,
        "quality_metric": "f1_macro",
        "selected": sorted({fila["selected"] for fila in propias if fila["selected"]}),
        "run_ids": [fila["run_id"] for fila in propias if fila["run_id"]],
    }


def ejecutar_demostracion_caso04(
    controlador: FrameworkController,
    objetivo: Any,
    seeds: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Ejecuta tres escenarios y compara una selección centralizada con el MAS."""
    semillas = _validate_seeds(seeds)
    datos = controlador.active_data()
    validate_target(datos, objetivo, task="classification")
    filas_utiles = int(datos[objetivo].notna().sum())
    filas_entrenamiento = filas_utiles - math.ceil(filas_utiles * TEST_SIZE)
    clases = datos[objetivo].value_counts(dropna=True)
    if filas_entrenamiento < 5:
        raise ValidationError("Se requieren al menos 5 filas de entrenamiento para comparar KNN.")
    if (clases < 2).any():
        raise ValidationError("Cada clase necesita al menos 2 filas para usar una partición estratificada.")
    filas_prueba = math.ceil(filas_utiles * TEST_SIZE)
    if filas_prueba < len(clases) or filas_entrenamiento < len(clases):
        raise ValidationError(
            "La partición estratificada requiere al menos una fila de prueba y de entrenamiento por clase."
        )

    k_max = min(11, filas_entrenamiento)
    k_values = list(range(3, k_max + 1, 2))
    parametros_modelo = {"test_size": TEST_SIZE, "scale": True, "stratify": True}
    nombre_dataset = (controlador.dataset_info() or {}).get("name", "Datos activos")

    columnas_numericas = datos.select_dtypes(include="number").columns.tolist()
    columnas_agrupamiento = [columna for columna in columnas_numericas if columna != objetivo][:2]
    if len(columnas_agrupamiento) < 2:
        raise ValidationError("El escenario inicial de agrupamiento requiere dos columnas numéricas predictoras.")
    simple = _ejecutar_tarea(
        controlador,
        "clustering.kmeans",
        {
            "data": datos,
            "columns": columnas_agrupamiento,
            "params": {"n_clusters": 3, "init": "k-means++", "n_init": 10},
            "projection_dims": 2,
        },
    )
    escenario_simple = _resumir_escenario(
        "Agrupamiento inicial",
        "Baja",
        [simple],
        [(simple, None)],
    )

    perfil = _ejecutar_tarea(
        controlador,
        "eda",
        {"data": datos, "target": objetivo, "use_llm": False},
    )
    preparados = perfil["result"]["result"]["prepared_data"]
    clasificaciones = [
        _ejecutar_tarea(
            controlador,
            "classification",
            {
                **parametros_modelo,
                "data": preparados,
                "target": objetivo,
                "algorithm": "knn",
                "params": {"n_neighbors": 5},
                "seeds": [seed],
            },
        )
        for seed in semillas
    ]
    escenario_cooperativo = _resumir_escenario(
        "Preparación EDA y clasificación",
        "Media",
        [perfil, *clasificaciones],
        list(zip([perfil, *clasificaciones], [None, *semillas])),
    )

    agencia = build_default_agency(
        config=controlador.config,
        state_store=False,
        model_store=False,
        rehydrate=False,
    )
    if "knn_auto" not in agencia.factory.types:
        raise ValidationError("No está disponible el plugin de agente KNN automático requerido por la demostración.")
    plugin = agencia.create_agent(
        {
            "agent_type": "knn_auto",
            "name": f"Caso04 KNN automático {uuid.uuid4().hex[:6]}",
            "config": {"k_max": k_max},
            "priority": 50,
        },
        persist=False,
    )
    comparaciones = []
    tareas_conflicto = []
    ejecuciones_conflicto = []
    try:
        for seed in semillas:
            config_semilla = {**parametros_modelo, "seeds": [seed]}
            inicio_central = time.perf_counter()
            centralizado = _evaluacion_centralizada(datos, objetivo, k_values, seed)
            central_id = controlador.experiments.record(
                task="case04.centralized_classification",
                dataset=controlador.dataset_info(),
                config={
                    "target": objetivo,
                    **config_semilla,
                    "k_values": k_values,
                    "seed": seed,
                },
                result=centralizado,
                best_model=centralizado["label"],
                decisions=[{
                    "decision": "Selección centralizada",
                    "justification": "Se eligió la alternativa con mayor F1 macro entre los mismos modelos KNN.",
                }],
                started_at=inicio_central,
            )
            duracion_central = round(time.perf_counter() - inicio_central, 4)

            fila_central = {
                "seed": seed,
                "approach": "Centralizado",
                "duration_s": duracion_central,
                "interactions": 0,
                "tasks_completed": 2,
                "models_evaluated": centralizado["models_evaluated"],
                "errors": 0,
                "quality_metric": "f1_macro",
                "quality": centralizado["primary_metric"]["value"],
                "selected": centralizado["label"],
                "run_id": central_id,
                "candidates": centralizado["evaluations"],
                "message_trace": [],
            }
            comparaciones.append(fila_central)

            fanout, duracion_mas = _ejecutar_fanout_aislado(
                controlador,
                agencia,
                datos,
                objetivo,
                seed,
                k_values,
            )
            resultado_mas = fanout["result"]
            metrica_mas = (resultado_mas.get("primary_metric") or {}).get("value")
            fila_mas = {
                "seed": seed,
                "approach": "MAS",
                "duration_s": duracion_mas,
                "interactions": len(fanout.get("messages", [])),
                "tasks_completed": len(fanout.get("agents_used", [])),
                "models_evaluated": centralizado["models_evaluated"],
                "errors": sum(not intento.get("ok", False) for intento in fanout.get("attempts", [])),
                "quality_metric": "f1_macro",
                "quality": metrica_mas,
                "selected": resultado_mas.get("label") or resultado_mas.get("algorithm"),
                "run_id": fanout.get("run_id"),
                "candidates": fanout.get("alternatives", []),
                "selected_agent": fanout.get("selected_agent"),
                "candidate_agents": fanout.get("candidates", []),
                "message_trace": _traza_mensajes(fanout.get("messages", [])),
                "decisions": fanout.get("decisions", []),
            }
            comparaciones.append(fila_mas)
            tareas_conflicto.append(_ejecutar_tarea_respuesta(
                fanout,
                duration_s=duracion_mas,
            ))
            ejecuciones_conflicto.append((tareas_conflicto[-1], seed))
    finally:
        agencia.remove_agent(plugin.agent_id, forget=True)

    escenario_conflicto = _resumir_escenario(
        "Selección y resolución de alternativas",
        "Alta",
        tareas_conflicto,
        ejecuciones_conflicto,
    )
    return {
        "dataset": nombre_dataset,
        "target": objetivo,
        "rows": int(len(datos)),
        "columns": int(datos.shape[1]),
        "seeds": semillas,
        "test_size": TEST_SIZE,
        "k_values": k_values,
        "scenarios": [escenario_simple, escenario_cooperativo, escenario_conflicto],
        "comparison": {
            "centralized": _resumir_comparacion(comparaciones, "Centralizado"),
            "multiagent": _resumir_comparacion(comparaciones, "MAS"),
            "trials": comparaciones,
        },
        "runs": {
            "simple": escenario_simple["run_ids"],
            "cooperation": escenario_cooperativo["run_ids"],
            "conflict": escenario_conflicto["run_ids"],
        },
    }


def _ejecutar_tarea_respuesta(respuesta: dict[str, Any], duration_s: float) -> dict[str, Any]:
    """Adapta la respuesta MAS ya ejecutada al formato común de escenarios."""
    metrica = (respuesta.get("result") or {}).get("primary_metric") or {}
    return {
        "task": respuesta.get("task", "classification"),
        "result": respuesta,
        "duration_s": duration_s,
        "interactions": len(respuesta.get("messages", [])),
        "tasks_completed": len(respuesta.get("agents_used", [])),
        "errors": sum(not intento.get("ok", False) for intento in respuesta.get("attempts", [])),
        "quality_metric": metrica.get("name"),
        "quality": metrica.get("value"),
    }
