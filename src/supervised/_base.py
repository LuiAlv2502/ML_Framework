from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataframe import DataFrame


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class SupervisadoBase(DataFrame):
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        test_size: float = 0.25,
        random_state: int = 42,
        scale: bool = True,
        stratify: bool = True,
        feature_columns: list[str] | None = None,
    ) -> None:
        super().__init__(df)

        if target not in self.dataframe.columns:
            raise KeyError(f"La columna objetivo '{target}' no existe.")
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1.")

        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.scale = scale
        self.stratify = stratify
        self.feature_columns = feature_columns

    def prepare_data(self) -> SplitData:
        data = self.dataframe
        if data.empty:
            raise ValueError("No hay datos para procesar.")
        if data[self.target].isna().any():
            raise ValueError("La variable objetivo contiene valores nulos.")

        columns = self.feature_columns or [
            column for column in data.columns if column != self.target
        ]
        if self.target in columns:
            raise ValueError("La variable objetivo no puede ser una característica.")

        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise KeyError(f"Columnas predictoras inexistentes: {missing}")
        if not columns:
            raise ValueError("Debe haber al menos una columna predictora.")

        X = data.loc[:, columns]
        y = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
        )
        return SplitData(X_train, X_test, y_train, y_test)

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = [
            column for column in X.columns if column not in numeric_columns
        ]
        transformers = []

        if numeric_columns:
            numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
            if self.scale:
                numeric_steps.append(("scaler", StandardScaler()))
            transformers.append(
                ("numericas", Pipeline(numeric_steps), numeric_columns)
            )

        if categorical_columns:
            categorical_pipeline = Pipeline(
                [

                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(
                ("categoricas", categorical_pipeline, categorical_columns)
            )

        return ColumnTransformer(transformers, remainder="drop")

    def _build_pipeline(self, X: pd.DataFrame, model: Any) -> Pipeline:
        return Pipeline(
            [
                ("preprocesamiento", self._build_preprocessor(X)),
                ("modelo", model),
            ]
        )

    def _fit_predict(self, model: Any) -> tuple[Pipeline, SplitData, Any]:
        split = self.prepare_data()
        pipeline = self._build_pipeline(split.X_train, model)
        pipeline.fit(split.X_train, split.y_train)
        y_pred = pipeline.predict(split.X_test)
        return pipeline, split, y_pred

    def preparar_datos_numericos(self, columnas=None):
        datos = self.dataframe if columnas is None else self.dataframe[columnas]

        if datos.empty:
            raise ValueError("No hay datos para procesar.")

        datos_codificados = pd.get_dummies(datos, drop_first=True)

        if datos_codificados.empty:
            raise ValueError("No existen columnas utilizables.")

        matriz = StandardScaler().fit_transform(
            datos_codificados.to_numpy(dtype=float)
        )
        return matriz, list(datos_codificados.columns)
