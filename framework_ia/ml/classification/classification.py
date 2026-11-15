from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
	AdaBoostClassifier,
	GradientBoostingClassifier,
	RandomForestClassifier,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from framework_ia.utils.metrics import classification_metrics

from .supervised_base import SupervisadoBase


class _EncodedTargetClassifier(ClassifierMixin, BaseEstimator):
	def __init__(self, estimator: Any) -> None:
		self.estimator = estimator

	def fit(self, X, y):
		self.label_encoder_ = LabelEncoder()
		encoded_y = self.label_encoder_.fit_transform(y)
		self.estimator_ = clone(self.estimator)
		self.estimator_.fit(X, encoded_y)
		self.classes_ = self.label_encoder_.classes_
		return self

	def predict(self, X):
		encoded_predictions = np.asarray(self.estimator_.predict(X), dtype=int)
		return self.label_encoder_.inverse_transform(encoded_predictions)

	def predict_proba(self, X):
		return self.estimator_.predict_proba(X)


class ClasificacionModelos(SupervisadoBase):
	def evaluar_clasificacion(self, y_true, y_pred) -> dict[str, Any]:
		return classification_metrics(y_true, y_pred)

	def _evaluar_modelo(self, model) -> dict[str, Any]:
		fitted_model, split, y_pred = self._fit_predict(model)
		return {
			"model": fitted_model,
			"metricas": self.evaluar_clasificacion(split.y_test, y_pred),
		}

	def knn(self, n_neighbors: int = 5, algorithm: str = "auto"):
		model = KNeighborsClassifier(
			n_neighbors=n_neighbors,
			algorithm=algorithm,
		)
		return self._evaluar_modelo(model)

	def decision_tree(
		self,
		min_samples_split: int = 2,
		max_depth: int | None = None,
	):
		model = DecisionTreeClassifier(
			min_samples_split=min_samples_split,
			max_depth=max_depth,
			random_state=self.random_state,
		)
		return self._evaluar_modelo(model)

	def random_forest(
		self,
		n_estimators: int = 200,
		min_samples_split: int = 2,
		max_depth: int | None = None,
	):
		model = RandomForestClassifier(
			n_estimators=n_estimators,
			min_samples_split=min_samples_split,
			max_depth=max_depth,
			random_state=self.random_state,
		)
		return self._evaluar_modelo(model)

	def gradient_boosting(self, n_estimators: int = 120, max_depth: int = 2):
		model = GradientBoostingClassifier(
			n_estimators=n_estimators,
			max_depth=max_depth,
			random_state=self.random_state,
		)
		return self._evaluar_modelo(model)

	def adaboost(self, n_estimators: int = 80):
		model = AdaBoostClassifier(
			n_estimators=n_estimators,
			random_state=self.random_state,
		)
		return self._evaluar_modelo(model)

	def xgboost(
		self,
		n_estimators: int = 120,
		max_depth: int = 3,
		learning_rate: float = 0.1,
	):
		try:
			from xgboost import XGBClassifier
		except ImportError as error:
			raise ImportError(
				"XGBoost no está instalado. Instálelo con 'pip install xgboost'."
			) from error

		model = _EncodedTargetClassifier(
			XGBClassifier(
				n_estimators=n_estimators,
				max_depth=max_depth,
				learning_rate=learning_rate,
				random_state=self.random_state,
				eval_metric="mlogloss",
			)
		)
		return self._evaluar_modelo(model)

	def adaboost_grid(
		self,
		param_grid: dict[str, Any],
		cv: int = 5,
	):
		split = self.prepare_data()
		if not param_grid:
			raise ValueError("param_grid no puede estar vacío.")

		class_counts = split.y_train.value_counts()
		cv_splits = min(cv, int(class_counts.min()))
		if cv_splits < 2:
			raise ValueError(
				"Cada clase debe tener al menos dos observaciones en entrenamiento para GridSearchCV."
			)

		model = self._build_pipeline(
			split.X_train,
			AdaBoostClassifier(random_state=self.random_state),
		)
		prefixed_grid = {
			key if key.startswith("modelo__") else f"modelo__{key}": values
			for key, values in param_grid.items()
		}
		grid = GridSearchCV(
			model,
			prefixed_grid,
			cv=cv_splits,
			scoring="accuracy",
			n_jobs=-1,
		)
		grid.fit(split.X_train, split.y_train)
		y_pred = grid.predict(split.X_test)
		return {
			"model": grid.best_estimator_,
			"best_params": {
				key.removeprefix("modelo__"): value
				for key, value in grid.best_params_.items()
			},
			"best_cv_score": float(grid.best_score_),
			"metricas": self.evaluar_clasificacion(split.y_test, y_pred),
		}

	def comparar_basico(self):
		return {
			"knn": self.knn(),
			"decision_tree": self.decision_tree(),
			"random_forest": self.random_forest(),
			"gradient_boosting": self.gradient_boosting(),
			"adaboost": self.adaboost(),
			"xgboost": self.xgboost(),
		}

	def benchmark(self, configs: list[dict[str, Any]] | None = None):
		"""Modelo base + configuraciones alternativas con la misma partición y semilla.

		``configs`` es una lista de ``{"algorithm", "params", "label"}`` (por defecto
		``catalog.BENCHMARK_CONFIGS``). Devuelve ``(tabla, resultados)``: la tabla está ordenada por
		F1 macro y ``resultados[label]`` contiene el pipeline y las métricas completas.
		"""
		from .catalog import BENCHMARK_CONFIGS, CATALOGO, describe_params

		configs = BENCHMARK_CONFIGS if configs is None else configs
		if not configs:
			raise ValueError("Indique al menos una configuración para el benchmark.")
		filas, resultados = [], {}
		for posicion, config in enumerate(configs):
			algoritmo = config.get("algorithm")
			if algoritmo not in CATALOGO:
				raise ValueError(f"Algoritmo desconocido '{algoritmo}'. Opciones: {list(CATALOGO)}")
			parametros = dict(config.get("params") or {})
			etiqueta = config.get("label") or f"{algoritmo} ({describe_params(parametros)})"
			if etiqueta in resultados:
				etiqueta = f"{etiqueta} #{posicion + 1}"
			fila = {
				"Modelo": etiqueta,
				"Algoritmo": CATALOGO[algoritmo]["name"],
				"Familia": CATALOGO[algoritmo]["family"],
				"Configuración": "estándar" if not parametros else "variación",
				"Parámetros": describe_params(parametros),
			}
			try:
				if algoritmo == "adaboost_grid":
					cv = parametros.pop("cv", 5)
					resultado = self.adaboost_grid(parametros, cv=cv)
				else:
					resultado = getattr(self, algoritmo)(**parametros)
			except ImportError as error:
				filas.append({**fila, "error": str(error)})
				continue
			resultados[etiqueta] = resultado
			metricas = resultado["metricas"]
			filas.append({
				**fila,
				**{m: metricas[m] for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
			})
		tabla = pd.DataFrame(filas)
		if "f1_macro" in tabla:
			tabla = tabla.sort_values("f1_macro", ascending=False, na_position="last", ignore_index=True)
		return tabla, resultados



    