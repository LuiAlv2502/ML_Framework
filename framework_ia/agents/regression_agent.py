"""Agente de regresión: elige una variable numérica continua y valida regresores del framework."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from framework_ia.ml.classification.catalog import describe_params
from framework_ia.ml.regression import RegresionModelos
from framework_ia.ml.regression.catalog import BENCHMARK_CONFIGS as CONFIGS_REGRESION
from framework_ia.ml.regression.catalog import CATALOGO as CATALOGO_REGRESION
from framework_ia.ml.regression.catalog import LINEALES
from framework_ia.ml.regression.regression import NOMBRES as NOMBRES_REGRESION
from framework_ia.utils.exceptions import ValidationError

from . import _support
from .agent_factory import agent_type
from .messaging import Message
from .supervised_agent import ARBOLES, DIVISION, PROFUNDIDAD, AgenteSupervisado, SupervisedAgentAdapter

MIN_UNICOS_REGRESION = 10


class AgenteRegresion(AgenteSupervisado):
    """Valida regresores con RMSE como métrica principal."""

    TIPO = "regresion"
    PROBLEMA = "regresión"
    CONFIGURACION = {
        "clase": RegresionModelos,
        "metrica": "RMSE",
        "maximizar": False,
        "metricas": ["RMSE", "MAE", "ER", "R2"],
        "algoritmos": {
            "regresion_lineal_multiple": {},
            "lasso": {"alpha": ("real", 0.0001, 10_000.0)},
            "lasso_cv": {"cv": ("entero", 2, 10)},
            "ridge": {"alpha": ("real", 0.0001, 10_000.0)},
            "ridge_cv": {},
            "svr": {
                "kernel": ("opcion", ["rbf", "linear", "poly", "sigmoid"]),
                "C": ("real", 0.01, 10_000.0),
                "epsilon": ("real", 0.0, 100.0),
                "degree": ("entero", 2, 6),
            },
            "decision_tree": {"max_depth": PROFUNDIDAD, "min_samples_split": DIVISION},
            "random_forest": {
                "n_estimators": ARBOLES,
                "max_depth": PROFUNDIDAD,
                "min_samples_split": DIVISION,
            },
        },
        "nombres": dict(NOMBRES_REGRESION),
    }

    def _motivo_por_tipo(self, serie: pd.Series, columna: Any, unicos: int) -> str | None:
        if not pd.api.types.is_numeric_dtype(serie):
            return f"'{columna}' no es numérica; la regresión requiere un objetivo numérico."
        if unicos <= MIN_UNICOS_REGRESION:
            return (
                f"'{columna}' solo tiene {unicos} valores distintos: parece categórica, "
                "más adecuada para clasificación."
            )
        return None

    def _describir_objetivo(self, y: pd.Series) -> dict[str, Any]:
        return {
            "resumen_objetivo": {
                "min": float(y.min()),
                "mediana": float(y.median()),
                "max": float(y.max()),
                "media": float(y.mean()),
                "desviacion": float(y.std()),
            }
        }

    def _comparar_linea_base(self, resumen: dict[str, dict[str, float]], y: pd.Series) -> tuple[bool, str]:
        base = self.linea_base_regresion(y)
        valor = resumen["RMSE"]["media"]
        return valor < base, f"RMSE medio {valor:.4f} frente a {base:.4f} de predecir siempre la media"

    @staticmethod
    def linea_base_regresion(y: pd.Series) -> float:
        """RMSE de predecir siempre la media (referencia mínima)."""
        return float(np.sqrt(((y - y.mean()) ** 2).mean()))


# ======================================================================
# Adaptador MAS
# ======================================================================
@agent_type("regression", description="Entrena, compara y selecciona regresores (lineales, Lasso/Ridge, "
                                      "SVR, árboles, Random Forest).")
class RegressionAgent(SupervisedAgentAdapter):
    """Agente especializado en regresión (envuelve ``RegresionModelos`` y ``AgenteRegresion``)."""

    TASK = "regression"
    CATALOG = CATALOGO_REGRESION
    BENCHMARK_CONFIGS = CONFIGS_REGRESION
    LEGACY = AgenteRegresion
    PRIMARY = ("RMSE", False)
    default_capabilities = frozenset({"regression.train", "regression.benchmark", "regression.auto"})
    supported_capabilities = default_capabilities

    def _modelo(self, comunes: dict[str, Any], semilla: int, payload: dict[str, Any]) -> RegresionModelos:
        return _support.ml_call(
            RegresionModelos, comunes["data"], target=comunes["target"], test_size=comunes["test_size"],
            random_state=semilla, scale=comunes["scale"], feature_columns=comunes["features"],
        )

    def _train(self, message: Message) -> dict[str, Any]:
        comunes = self._comunes(message)
        payload = message.payload
        algoritmo = payload.get("algorithm")
        crudos = dict(payload.get("params") or {})
        columna = crudos.pop("columna", None)
        parametros = self._parametros(algoritmo, crudos)
        datos = comunes["data"]
        if algoritmo == "regresion_lineal_simple":
            predictores = comunes["features"] or [c for c in datos.columns if c != comunes["target"]]
            if columna not in predictores or not pd.api.types.is_numeric_dtype(datos[columna]):
                raise ValidationError("La regresión simple requiere una variable predictora numérica ('columna').")
            parametros = {"columna": columna}
        etiqueta = payload.get("label") or f"{algoritmo} ({describe_params(parametros)})"
        experimentos, pipeline, resumen_objetivo = [], None, None
        for semilla in comunes["seeds"]:
            modelo = self._modelo(comunes, semilla, payload)
            particion = _support.ml_call(modelo.prepare_data)
            resultado = _support.ml_call(getattr(modelo, algoritmo), **parametros)
            resumen_objetivo = resumen_objetivo or modelo.resumen_variable_objetivo()
            pipeline = resultado["model"]
            indices = particion.X_test.index.to_numpy()
            ids = indices + 1 if np.issubdtype(indices.dtype, np.integer) else indices
            predicciones = pd.DataFrame({"ID": ids, "Real": resultado["y_test"], "Predicción": resultado["y_pred"]})
            predicciones["Residuo"] = predicciones["Real"] - predicciones["Predicción"]
            predicciones["Error absoluto"] = predicciones["Residuo"].abs()
            experimento: dict[str, Any] = {
                "seed": semilla,
                "n_train": len(particion.X_train),
                "n_test": len(particion.X_test),
                "test_index": particion.X_test.index.tolist(),
                "metrics": dict(resultado["metricas"]),
                "predictions": predicciones.sort_values("ID").reset_index(drop=True),
                "baseline_rmse": self.LEGACY.linea_base_regresion(pd.Series(resultado["y_test"])),
            }
            if "alpha_optimo" in resultado:
                experimento["alpha_optimo"] = resultado["alpha_optimo"]
            if algoritmo == "regresion_lineal_simple":
                xs = np.linspace(datos[columna].min(), datos[columna].max(), 200)
                experimento.update({
                    "w0": resultado["w0"], "w1": resultado["w1"],
                    "line": {"x": xs.tolist(), "y": pipeline.predict(pd.DataFrame({columna: xs})).tolist()},
                })
            elif algoritmo in LINEALES:
                coeficientes = RegresionModelos.coeficientes(resultado)
                coeficientes["predictor"] = coeficientes["predictor"].str.replace(
                    r"^(numericas|categoricas)__", "", regex=True)
                experimento["coefficients"] = coeficientes
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
            "task": "regression.train",
            "algorithm": algoritmo,
            "algorithm_name": self.CATALOG[algoritmo]["name"],
            "family": self.CATALOG[algoritmo]["family"],
            "label": etiqueta,
            "params": parametros,
            "target": comunes["target"],
            "features": comunes["features"],
            "test_size": comunes["test_size"],
            "seeds": comunes["seeds"],
            "target_summary": resumen_objetivo,
            "experiments": experimentos,
            "metrics": medias,
            "metrics_table": tabla,
            "model_id": model_id,
            "primary_metric": _support.primary("RMSE", medias["RMSE"], False),
        }

    def _extra_benchmark(self, tabla: pd.DataFrame, resultados: dict[str, Any]) -> dict[str, Any]:
        return {"best_by_family": RegresionModelos.mejor_por_familia(None, tabla)}

    def _detalle_auto(self, agente: AgenteSupervisado) -> dict[str, Any]:
        detalle = agente.resultado_mejor()
        y_test = np.asarray(detalle["y_test"], dtype=float)
        return {"metrics": dict(detalle["metricas"]), "y_test": y_test.tolist(),
                "y_pred": np.asarray(detalle["y_pred"], dtype=float).tolist(),
                "baseline_rmse": self.LEGACY.linea_base_regresion(pd.Series(y_test))}
