from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .._base import SupervisadoBase


# Familias de algoritmos para el benchmarking (Lab 3).
FAMILIAS = {
    "Lineales": [
        "regresion_lineal_multiple",
        "lasso",
        "lasso_cv",
        "ridge",
        "ridge_cv",
    ],
    "Máquinas de soporte vectorial": ["svr"],
    "Árboles": ["decision_tree", "random_forest"],
}

NOMBRES = {
    "regresion_lineal_simple": "Regresión lineal simple",
    "regresion_lineal_multiple": "Regresión lineal múltiple",
    "lasso": "Lasso",
    "lasso_cv": "LassoCV",
    "ridge": "Ridge",
    "ridge_cv": "RidgeCV",
    "svr": "SVR",
    "decision_tree": "Árbol de decisión",
    "random_forest": "Random Forest",
}


class RegresionModelos(SupervisadoBase):
    """Algoritmos de regresión integrados en la clase de Supervisados."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        test_size: float = 0.25,
        random_state: int = 42,
        scale: bool = True,
        feature_columns: list[str] | None = None,
    ) -> None:
        # En regresión la variable objetivo es continua: no se puede estratificar.
        super().__init__(
            df,
            target=target,
            test_size=test_size,
            random_state=random_state,
            scale=scale,
            stratify=False,
            feature_columns=feature_columns,
        )
        if not pd.api.types.is_numeric_dtype(self.dataframe[target]):
            raise TypeError(
                f"La variable objetivo '{target}' debe ser numérica para regresión."
            )

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------
    def evaluar_regresion(self, y_true, y_pred) -> dict[str, float]:
        """RMSE, MAE y ER (como en la guía del profesor) + R²."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        suma_abs = np.sum(np.abs(y_true))
        return {
            "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "ER": float(np.sum(np.abs(y_true - y_pred)) / suma_abs) if suma_abs else float("nan"),
            "R2": float(r2_score(y_true, y_pred)),
        }

    def resumen_variable_objetivo(self) -> dict[str, Any]:
        """Mínimo, cuartiles y máximo de la variable a predecir (sirve para
        interpretar si un RMSE es grande o pequeño)."""
        y = self.dataframe[self.target].dropna()
        return {
            "Mínimo": float(y.min()),
            "Cuartiles": [float(q) for q in y.quantile([0.25, 0.5, 0.75])],
            "Máximo": float(y.max()),
            "Media": float(y.mean()),
        }

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _evaluar_modelo(self, model, columnas: list[str] | None = None) -> dict[str, Any]:
        split = self.prepare_data()
        X_train, X_test = split.X_train, split.X_test
        if columnas is not None:
            X_train, X_test = X_train[columnas], X_test[columnas]

        pipeline = self._build_pipeline(X_train, model)
        pipeline.fit(X_train, split.y_train)
        y_pred = pipeline.predict(X_test)
        return {
            "model": pipeline,
            "metricas": self.evaluar_regresion(split.y_test, y_pred),
            "y_test": split.y_test.to_numpy(),
            "y_pred": y_pred,
        }

    @staticmethod
    def coeficientes(resultado: dict[str, Any]) -> pd.DataFrame:
        """Coeficientes de un modelo lineal (útil para ver qué variables
        Lasso deja en cero)."""
        pipeline = resultado["model"]
        modelo = pipeline.named_steps["modelo"]
        if not hasattr(modelo, "coef_"):
            raise TypeError("El modelo no es lineal: no tiene coeficientes.")
        nombres = pipeline.named_steps["preprocesamiento"].get_feature_names_out()
        return pd.DataFrame(
            {"predictor": nombres, "coef": np.ravel(modelo.coef_)}
        ).sort_values("coef", key=np.abs, ascending=False, ignore_index=True)

    # ------------------------------------------------------------------
    # Familia: modelos lineales
    # ------------------------------------------------------------------
    def regresion_lineal_simple(self, columna: str):
        """Regresión con UNA sola variable predictora: y = w0 + w1·x."""
        split = self.prepare_data()
        if columna not in split.X_train.columns:
            raise KeyError(f"La columna '{columna}' no es una variable predictora.")
        if not pd.api.types.is_numeric_dtype(split.X_train[columna]):
            raise TypeError("La regresión simple requiere una variable numérica.")
        resultado = self._evaluar_modelo(LinearRegression(), columnas=[columna])
        modelo = resultado["model"].named_steps["modelo"]
        resultado["w0"] = float(modelo.intercept_)
        resultado["w1"] = float(modelo.coef_[0])
        return resultado

    def regresion_lineal_multiple(self):
        """Regresión con todas las variables predictoras."""
        return self._evaluar_modelo(LinearRegression())

    def lasso(self, alpha: float = 0.1):
        return self._evaluar_modelo(
            Lasso(alpha=alpha, max_iter=10000, random_state=self.random_state)
        )

    def lasso_cv(self, alphas=None, cv: int = 10):
        """Busca el mejor alpha de Lasso con validación cruzada."""
        alphas = np.logspace(-6, 6, 200) if alphas is None else alphas
        resultado = self._evaluar_modelo(
            LassoCV(alphas=alphas, cv=cv, max_iter=10000, random_state=self.random_state)
        )
        resultado["alpha_optimo"] = float(resultado["model"].named_steps["modelo"].alpha_)
        return resultado

    def ridge(self, alpha: float = 1.0):
        return self._evaluar_modelo(Ridge(alpha=alpha))

    def ridge_cv(self, alphas=None):
        """Busca el mejor alpha de Ridge con validación cruzada."""
        alphas = np.logspace(-10, 2, 200) if alphas is None else alphas
        resultado = self._evaluar_modelo(RidgeCV(alphas=alphas))
        resultado["alpha_optimo"] = float(resultado["model"].named_steps["modelo"].alpha_)
        return resultado

    # ------------------------------------------------------------------
    # Familia: máquinas de soporte vectorial
    # ------------------------------------------------------------------
    def svr(self, kernel: str = "rbf", C: float = 100, epsilon: float = 0.1, degree: int = 3):
        return self._evaluar_modelo(SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree))

    # ------------------------------------------------------------------
    # Familia: árboles
    # ------------------------------------------------------------------
    def decision_tree(self, max_depth: int | None = 3, min_samples_split: int = 2):
        return self._evaluar_modelo(
            DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=self.random_state,
            )
        )

    def random_forest(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
    ):
        return self._evaluar_modelo(
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
            )
        )

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------
    @staticmethod
    def _tabla(resultados: dict[str, dict[str, Any]]) -> pd.DataFrame:
        filas = [{"Modelo": nombre, **res["metricas"]} for nombre, res in resultados.items()]
        return (
            pd.DataFrame(filas)
            .sort_values("RMSE", ignore_index=True)
        )

    def comparar_configuraciones(self, metodo: str, configuraciones: list[tuple[str, dict]]):
        """Ejecuta un mismo algoritmo con varias configuraciones.
        configuraciones = [("nombre", {parametros}), ...]"""
        resultados = {
            nombre: getattr(self, metodo)(**parametros)
            for nombre, parametros in configuraciones
        }
        return self._tabla(resultados), resultados

    def comparar_familia(self, familia: str):
        """Compara los algoritmos de UNA misma familia (configuración estándar)."""
        if familia not in FAMILIAS:
            raise KeyError(f"Familia desconocida. Opciones: {list(FAMILIAS)}")
        resultados = {NOMBRES[m]: getattr(self, m)() for m in FAMILIAS[familia]}
        return self._tabla(resultados), resultados

    def comparar_familias(self):
        """Benchmarking entre familias: compara TODOS los algoritmos y marca
        a qué familia pertenece cada uno."""
        filas, resultados = [], {}
        for familia, metodos in FAMILIAS.items():
            for metodo in metodos:
                res = getattr(self, metodo)()
                resultados[NOMBRES[metodo]] = res
                filas.append({"Familia": familia, "Modelo": NOMBRES[metodo], **res["metricas"]})
        tabla = pd.DataFrame(filas).sort_values("RMSE", ignore_index=True)
        return tabla, resultados

    def mejor_por_familia(self, tabla_familias: pd.DataFrame) -> pd.DataFrame:
        """De la tabla de comparar_familias, deja solo el mejor (menor RMSE) de cada familia."""
        return (
            tabla_familias.sort_values("RMSE")
            .groupby("Familia", as_index=False)
            .first()
            .sort_values("RMSE", ignore_index=True)
        )
