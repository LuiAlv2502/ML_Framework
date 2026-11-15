"""Gestión del dataset activo: carga, validación, hash, copia preparada y resúmenes para el EDA."""
from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from framework_ia.ml.eda import DataFrame
from framework_ia.persistence.config_store import ConfigStore
from framework_ia.utils.exceptions import ValidationError
from framework_ia.utils.logging_config import get_logger
from framework_ia.utils.validation import (
    detect_encoding,
    suggest_task_type,
    validate_dataframe,
    validate_file,
)

logger = get_logger("services.dataset")

SOURCES = {
    "original": "Archivo original",
    "manual": "Cambios manuales (EDA)",
    "agente": "Preparación del agente EDA",
    "preprocesado": "Preprocesamiento configurado",
}


@dataclass
class DatasetInfo:
    """Descripción del dataset activo."""

    name: str
    hash: str
    rows: int
    columns: int
    source: str = "original"
    loaded_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_hash(content: bytes) -> str:
    """SHA-256 del contenido del archivo."""
    return hashlib.sha256(content).hexdigest()


def dataframe_hash(df: pd.DataFrame) -> str:
    """SHA-256 de un DataFrame (independiente de cómo se cargó)."""
    return compute_hash(pd.util.hash_pandas_object(df, index=True).values.tobytes()
                        + "|".join(map(str, df.columns)).encode())


class DatasetService:
    """Mantiene el dataset original y la copia activa que usan los análisis."""

    def __init__(self, config: ConfigStore | None = None) -> None:
        self.config = config or ConfigStore()
        self._original: pd.DataFrame | None = None
        self._active: pd.DataFrame | None = None
        self.info: DatasetInfo | None = None
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------
    @property
    def has_data(self) -> bool:
        return self._active is not None

    @property
    def original(self) -> pd.DataFrame:
        self._requerir()
        return self._original.copy()

    @property
    def active(self) -> pd.DataFrame:
        self._requerir()
        return self._active.copy()

    @property
    def source(self) -> str:
        return self.info.source if self.info else "original"

    def _requerir(self) -> None:
        if self._active is None:
            raise ValidationError("No hay datos cargados. Cargue un CSV en la página Datos.")

    def _limites(self) -> dict[str, Any]:
        return {
            "max_rows": int(self.config.get("limits.max_rows", 200_000)),
            "max_columns": int(self.config.get("limits.max_columns", 500)),
        }

    def _registrar(self, evento: str, detalle: str) -> None:
        self.history.append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                             "event": evento, "detail": detalle})
        logger.info("%s: %s", evento, detalle)

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------
    def read_bytes(self, content: bytes, filename: str) -> pd.DataFrame:
        """Lee CSV o Excel desde bytes con validación de formato, codificación y separador."""
        extension = validate_file(filename, len(content), max_mb=float(self.config.get("limits.max_file_mb", 50)))
        try:
            if extension == ".csv":
                codificacion = detect_encoding(content)
                df = pd.read_csv(io.BytesIO(content), encoding=codificacion)
                if df.shape[1] == 1 and any(s in str(df.columns[0]) for s in (";", "\t", "|")):
                    df = pd.read_csv(io.BytesIO(content), encoding=codificacion, sep=None, engine="python")
            else:
                df = pd.read_excel(io.BytesIO(content))
        except ImportError as error:
            raise ValidationError("Para leer Excel instale 'openpyxl' (pip install openpyxl).") from error
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as error:
            raise ValidationError(f"No se pudo leer '{filename}': {error}") from error
        return validate_dataframe(df, **self._limites())

    def load_bytes(self, content: bytes, filename: str) -> DatasetInfo:
        """Carga un archivo subido por el usuario y lo deja como dataset activo."""
        df = self.read_bytes(content, filename)
        return self._establecer(df, filename, compute_hash(content))

    def load_path(self, path: str | Path) -> DatasetInfo:
        """Carga un archivo del disco (por ejemplo, de ``data/``)."""
        ruta = Path(path)
        if not ruta.exists():
            raise ValidationError(f"No existe el archivo '{ruta}'.")
        return self.load_bytes(ruta.read_bytes(), ruta.name)

    def load_dataframe(self, df: pd.DataFrame, name: str = "datos") -> DatasetInfo:
        """Carga un DataFrame ya construido."""
        validate_dataframe(df, **self._limites())
        return self._establecer(df.copy(), name, dataframe_hash(df))

    def _establecer(self, df: pd.DataFrame, name: str, hash_: str) -> DatasetInfo:
        self._original = df.reset_index(drop=True)
        self._active = self._original.copy()
        self.info = DatasetInfo(name=name, hash=hash_, rows=len(df), columns=df.shape[1])
        self.history = []
        self._registrar("carga", f"{name}: {len(df)} filas × {df.shape[1]} columnas (hash {hash_[:12]}).")
        return self.info

    def list_available(self) -> list[dict[str, Any]]:
        """Archivos CSV/Excel disponibles en la carpeta de datos configurada."""
        carpeta = self.config.path_for("data_dir")
        if not carpeta.exists():
            return []
        archivos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in (".csv", ".xlsx", ".xls"))
        return [{"name": p.name, "path": str(p), "size_kb": round(p.stat().st_size / 1024, 1)} for p in archivos]

    # ------------------------------------------------------------------
    # Copia activa
    # ------------------------------------------------------------------
    def set_active(self, df: pd.DataFrame, source: str, description: str = "") -> DatasetInfo:
        """Reemplaza la copia activa (el original se conserva)."""
        self._requerir()
        if source not in SOURCES:
            raise ValidationError(f"Origen inválido '{source}'. Opciones: {', '.join(SOURCES)}.")
        validate_dataframe(df, **self._limites())
        self._active = df.reset_index(drop=True).copy()
        self.info.source = source
        self.info.rows, self.info.columns = self._active.shape
        self._registrar(source, description or SOURCES[source])
        return self.info

    def restore(self) -> DatasetInfo:
        """Vuelve a activar una copia del original."""
        self._requerir()
        self._active = self._original.copy()
        self.info.source = "original"
        self.info.rows, self.info.columns = self._active.shape
        self._registrar("restauración", "Se restauraron los datos originales.")
        return self.info

    @staticmethod
    def select_active(source: str, original: Any, manual: Any, agent: Any,
                      file_hash: str | None, agent_hash: str | None) -> Any:
        """Elige qué versión usar: la del agente solo si corresponde al archivo actual."""
        if source == "agente" and agent is not None and file_hash is not None and agent_hash == file_hash:
            return agent
        if source in ("original", "manual") and manual is not None:
            return manual
        return original

    # ------------------------------------------------------------------
    # Resúmenes
    # ------------------------------------------------------------------
    def preview(self, n: int = 10, tail: bool = False) -> pd.DataFrame:
        """Primeras (o últimas) ``n`` filas del dataset activo."""
        datos = DataFrame(self._active if self._active is not None else pd.DataFrame())
        return datos.ultimas_filas(n) if tail else datos.primeras_filas(n)

    def summary(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Dimensiones, tipos, nulos y duplicados."""
        datos = self.active if df is None else df
        numericas = datos.select_dtypes(include=np.number).columns.tolist()
        tipos = pd.DataFrame({
            "Columna": [str(c) for c in datos.columns],
            "Tipo": datos.dtypes.astype(str).to_numpy(),
            "Valores únicos": datos.nunique().to_numpy(),
            "Nulos": datos.isna().sum().to_numpy(),
        })
        return {
            "rows": int(datos.shape[0]),
            "columns": int(datos.shape[1]),
            "numeric": numericas,
            "categorical": [c for c in datos.columns if c not in numericas],
            "nulls_total": int(datos.isna().sum().sum()),
            "duplicates": int(datos.duplicated().sum()),
            "types": tipos,
            "memory_kb": round(float(datos.memory_usage(deep=True).sum()) / 1024, 1),
        }

    def eda_tables(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Tablas del EDA manual calculadas con la clase ``DataFrame`` del framework."""
        datos = self.active if df is None else df
        modelo = DataFrame(datos)
        numericas = datos.select_dtypes(include=np.number).columns.tolist()
        nulos = modelo.contar_nulos().rename("Nulos").reset_index()
        nulos.columns = ["Columna", "Nulos"]
        nulos["% nulos"] = nulos["Nulos"] / max(len(datos), 1) * 100
        outliers = modelo.detectar_outliers() if numericas else {}
        return {
            "nulls": nulos,
            "duplicates": modelo.detectar_duplicados(),
            "outliers": outliers,
            "outliers_summary": pd.DataFrame({
                "Columna": list(outliers),
                "Atípicos": [len(v) for v in outliers.values()],
                "% atípicos": [len(v) / max(len(datos), 1) * 100 for v in outliers.values()],
            }),
            "distribution": modelo.distribucion_variables(),
            "statistics": modelo.estadisticas(),
            "frequencies": modelo.frecuencias(),
            "correlation": modelo.correlacion() if len(numericas) >= 2 else pd.DataFrame(),
        }

    def target_summary(self, target: Any) -> dict[str, Any]:
        """Mínimo, cuartiles, máximo y media de una variable numérica (referencia para interpretar errores)."""
        datos = self.active
        if target not in datos.columns:
            raise ValidationError(f"La columna '{target}' no existe.")
        if not pd.api.types.is_numeric_dtype(datos[target]):
            raise ValidationError(f"'{target}' no es numérica.")
        y = datos[target].dropna()
        return {
            "Mínimo": float(y.min()),
            "Cuartiles": [float(q) for q in y.quantile([0.25, 0.5, 0.75])],
            "Máximo": float(y.max()),
            "Media": float(y.mean()),
        }

    def suggest_target(self) -> Any:
        """Sugiere como objetivo la última columna que no parezca identificador."""
        datos = self.active
        for columna in reversed(datos.columns.tolist()):
            if datos[columna].nunique(dropna=True) < len(datos):
                return columna
        return datos.columns[-1]

    def suggest_task(self, target: Any) -> str:
        """``classification`` o ``regression`` según la variable objetivo."""
        datos = self.active
        if target not in datos.columns:
            raise ValidationError(f"La columna '{target}' no existe.")
        return suggest_task_type(datos[target])

    def dataset_dict(self) -> dict[str, Any]:
        """Información serializable del dataset activo (para registrar experimentos)."""
        return self.info.to_dict() if self.info else {}


__all__ = ["SOURCES", "DatasetInfo", "DatasetService", "compute_hash", "dataframe_hash"]
