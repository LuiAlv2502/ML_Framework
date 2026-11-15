"""Agente de clasificación: elige una variable categórica y valida clasificadores del framework."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from framework_ia.ml.classification import ClasificacionModelos
from framework_ia.ml.classification.catalog import BENCHMARK_CONFIGS as CONFIGS_CLASIFICACION
from framework_ia.ml.classification.catalog import CATALOGO as CATALOGO_CLASIFICACION
from framework_ia.ml.classification.catalog import describe_params
from framework_ia.utils.exceptions import ValidationError

from . import _support
from .agent_factory import agent_type
from .messaging import Message
from .supervised_agent import (
    ARBOLES,
    DIVISION,
    MAX_CLASES,
    PROFUNDIDAD,
    AgenteSupervisado,
    SupervisedAgentAdapter,
)


class AgenteClasificacion(AgenteSupervisado):
    """Valida clasificadores con F1 macro como métrica principal y partición estratificada."""

    TIPO = "clasificacion"
    PROBLEMA = "clasificación"
    CONFIGURACION = {
        "clase": ClasificacionModelos,
        "metrica": "f1_macro",
        "maximizar": True,
        "metricas": ["f1_macro", "accuracy", "precision_macro", "recall_macro"],
        "algoritmos": {
            "knn": {"n_neighbors": ("entero", 1, 50)},
            "decision_tree": {"max_depth": PROFUNDIDAD, "min_samples_split": DIVISION},
            "random_forest": {
                "n_estimators": ARBOLES,
                "max_depth": PROFUNDIDAD,
                "min_samples_split": DIVISION,
            },
            "gradient_boosting": {"n_estimators": ARBOLES, "max_depth": ("entero", 1, 10)},
            "adaboost": {"n_estimators": ("entero", 10, 300)},
            "xgboost": {
                "n_estimators": ARBOLES,
                "max_depth": ("entero", 1, 15),
                "learning_rate": ("real", 0.01, 1.0),
            },
        },
        "nombres": {
            "knn": "K vecinos más cercanos",
            "decision_tree": "Árbol de decisión",
            "random_forest": "Bosques aleatorios",
            "gradient_boosting": "Gradient Boosting",
            "adaboost": "AdaBoost",
            "xgboost": "XGBoost",
        },
    }

    def _reiniciar(self) -> None:
        super()._reiniciar()
        self.estratificar = False

    def _motivo_por_tipo(self, serie: pd.Series, columna: Any, unicos: int) -> str | None:
        if unicos < 2:
            return f"'{columna}' tiene un solo valor; no hay clases que distinguir."
        if unicos > MAX_CLASES:
            return (
                f"'{columna}' tiene {unicos} valores distintos: parece continua, no una clase "
                f"(máximo {MAX_CLASES} clases)."
            )
        return None

    def _describir_objetivo(self, y: pd.Series) -> dict[str, Any]:
        conteos = y.value_counts()
        filas_prueba = math.ceil(len(y) * self.test_size)
        filas_entrenamiento = len(y) - filas_prueba
        self.estratificar = bool(
            conteos.min() >= 2 and min(filas_prueba, filas_entrenamiento) >= len(conteos)
        )
        return {
            "clases": int(len(conteos)),
            "linea_base_accuracy": round(float(conteos.iloc[0] / len(y)), 4),
            "proporcion_clase_minoritaria": round(float(conteos.iloc[-1] / len(y)), 4),
            "particion_estratificada": self.estratificar,
        }

    def _argumentos_particion(self) -> dict[str, Any]:
        return {"stratify": self.estratificar}

    def _comparar_linea_base(self, resumen: dict[str, dict[str, float]], y: pd.Series) -> tuple[bool, str]:
        base = float(y.value_counts(normalize=True).iloc[0])
        valor = resumen["accuracy"]["media"]
        return valor > base, (
            f"accuracy medio {valor:.4f} frente a {base:.4f} de predecir siempre la clase mayoritaria"
        )


# ======================================================================
# Adaptador MAS
# ======================================================================
@agent_type("classification", description="Entrena, compara y selecciona clasificadores (KNN, árboles, RF, "
                                          "Gradient Boosting, AdaBoost, XGBoost).")
class ClassificationAgent(SupervisedAgentAdapter):
    """Agente especializado en clasificación (envuelve ``ClasificacionModelos`` y ``AgenteClasificacion``)."""

    TASK = "classification"
    CATALOG = CATALOGO_CLASIFICACION
    BENCHMARK_CONFIGS = CONFIGS_CLASIFICACION
    LEGACY = AgenteClasificacion
    PRIMARY = ("f1_macro", True)
    default_capabilities = frozenset({"classification.train", "classification.benchmark", "classification.auto"})
    supported_capabilities = default_capabilities

    def _modelo(self, comunes: dict[str, Any], semilla: int, payload: dict[str, Any]) -> ClasificacionModelos:
        return _support.ml_call(
            ClasificacionModelos, comunes["data"], target=comunes["target"], test_size=comunes["test_size"],
            random_state=semilla, scale=comunes["scale"], stratify=bool(payload.get("stratify", True)),
            feature_columns=comunes["features"],
        )

    def _train(self, message: Message) -> dict[str, Any]:
        comunes = self._comunes(message)
        payload = message.payload
        algoritmo = payload.get("algorithm")
        parametros = self._parametros(algoritmo, payload.get("params"))
        etiqueta = payload.get("label") or f"{algoritmo} ({describe_params(parametros)})"
        experimentos, pipeline = [], None
        for semilla in comunes["seeds"]:
            modelo = self._modelo(comunes, semilla, payload)
            particion = _support.ml_call(modelo.prepare_data)
            if algoritmo == "adaboost_grid":
                grid = {k: v for k, v in parametros.items() if k != "cv"}
                if not grid:
                    raise ValidationError("Indique los valores de n_estimators y/o learning_rate a probar.")
                resultado = _support.ml_call(modelo.adaboost_grid, grid, cv=parametros.get("cv", 5))
            else:
                resultado = _support.ml_call(getattr(modelo, algoritmo), **parametros)
            pipeline = resultado["model"]
            metricas = resultado["metricas"]
            predicciones = particion.X_test.copy()
            predicciones.insert(0, "Real", particion.y_test.to_numpy())
            predicciones.insert(1, "Predicción", pipeline.predict(particion.X_test))
            predicciones.insert(2, "Correcto", predicciones["Real"].to_numpy() == predicciones["Predicción"].to_numpy())
            experimento = {
                "seed": semilla,
                "n_train": len(particion.X_train),
                "n_test": len(particion.X_test),
                "test_index": particion.X_test.index.tolist(),
                "metrics": {m: metricas[m] for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
                "confusion_matrix": {"labels": metricas["confusion_matrix_labels"],
                                     "matrix": metricas["confusion_matrix"]},
                "report": pd.DataFrame(metricas["classification_report"]).T,
                "predictions": predicciones,
            }
            if "best_params" in resultado:
                experimento["best_params"] = resultado["best_params"]
                experimento["best_cv_score"] = resultado["best_cv_score"]
            experimentos.append(experimento)
        tabla = pd.DataFrame([{"semilla": e["seed"], **e["metrics"]} for e in experimentos])
        medias = tabla.drop(columns="semilla").mean().to_dict()
        model_id = None
        almacen = self.context.get("model_store")
        if payload.get("save_model") and almacen is not None:
            model_id = almacen.save(pipeline, etiqueta, {"task": self.TASK, "target": str(comunes["target"]),
                                                         "algorithm": algoritmo, "params": parametros,
                                                         "metrics": medias, "agent": self.agent_id})
        return {
            "task": "classification.train",
            "algorithm": algoritmo,
            "algorithm_name": self.CATALOG[algoritmo]["name"],
            "family": self.CATALOG[algoritmo]["family"],
            "label": etiqueta,
            "params": parametros,
            "target": comunes["target"],
            "features": comunes["features"],
            "test_size": comunes["test_size"],
            "seeds": comunes["seeds"],
            "experiments": experimentos,
            "metrics": medias,
            "metrics_table": tabla,
            "confusion_matrix": experimentos[0]["confusion_matrix"],
            "model_id": model_id,
            "primary_metric": _support.primary("f1_macro", medias["f1_macro"], True),
        }

    def _extra_benchmark(self, tabla: pd.DataFrame, resultados: dict[str, Any]) -> dict[str, Any]:
        return {"confusion_matrices": {
            etiqueta: {"labels": r["metricas"]["confusion_matrix_labels"], "matrix": r["metricas"]["confusion_matrix"]}
            for etiqueta, r in resultados.items()
        }}

    def _detalle_auto(self, agente: AgenteSupervisado) -> dict[str, Any]:
        metricas = agente.resultado_mejor()["metricas"]
        return {"metrics": {m: metricas[m] for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
                "confusion_matrix": {"labels": metricas["confusion_matrix_labels"],
                                     "matrix": metricas["confusion_matrix"]}}
