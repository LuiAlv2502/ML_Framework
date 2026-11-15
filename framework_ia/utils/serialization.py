"""Conversión de resultados (DataFrames, arreglos, figuras, dataclasses) a estructuras JSON-serializables."""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import math
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATAFRAME_TAG = "__dataframe__"


def figure_to_png_bytes(figure: Any, dpi: int = 110) -> bytes:
    """Renderiza una figura de matplotlib a PNG."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()


def figure_to_base64(figure: Any, dpi: int = 110) -> str:
    """Renderiza una figura de matplotlib a PNG codificado en base64."""
    return base64.b64encode(figure_to_png_bytes(figure, dpi)).decode()


def _es_figura(obj: Any) -> bool:
    return type(obj).__name__ == "Figure" and hasattr(obj, "savefig")


def _float(valor: float) -> float | None:
    return None if math.isnan(valor) or math.isinf(valor) else valor


def to_jsonable(obj: Any, *, max_rows: int | None = None) -> Any:
    """Convierte ``obj`` recursivamente a tipos nativos de JSON.

    Los DataFrames se representan como ``{"__dataframe__": True, "columns": [...], "index": [...], "data": [...]}``
    (reversible con :func:`dataframe_from_jsonable`). Los modelos y objetos desconocidos se convierten a texto.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _float(float(obj))
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode()
    if isinstance(obj, pd.DataFrame):
        datos = obj if max_rows is None else obj.head(max_rows)
        return {
            DATAFRAME_TAG: True,
            "columns": [str(c) for c in datos.columns],
            "index": [to_jsonable(i) for i in datos.index],
            "data": [[to_jsonable(v) for v in fila] for fila in datos.itertuples(index=False, name=None)],
            "shape": list(obj.shape),
        }
    if isinstance(obj, pd.Series):
        datos = obj if max_rows is None else obj.head(max_rows)
        return {str(k): to_jsonable(v) for k, v in datos.items()}
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {campo.name: to_jsonable(getattr(obj, campo.name), max_rows=max_rows)
                for campo in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_rows=max_rows) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, max_rows=max_rows) for v in obj]
    if _es_figura(obj):
        return {"__figure_png_base64__": figure_to_base64(obj)}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return to_jsonable(obj.to_dict(), max_rows=max_rows)
        except TypeError:
            pass
    return repr(obj)


def dataframe_from_jsonable(data: dict[str, Any]) -> pd.DataFrame:
    """Reconstruye un DataFrame serializado con :func:`to_jsonable`."""
    if not isinstance(data, dict) or not data.get(DATAFRAME_TAG):
        raise TypeError("El objeto no es un DataFrame serializado.")
    return pd.DataFrame(data["data"], columns=data["columns"], index=data.get("index"))


def summarize(obj: Any, *, max_items: int = 20, max_text: int = 300) -> Any:
    """Resumen compacto para bitácoras y registros de mensajes (no incluye filas completas)."""
    if isinstance(obj, pd.DataFrame):
        return {"tabla": f"{obj.shape[0]} filas × {obj.shape[1]} columnas",
                "columnas": [str(c) for c in obj.columns[:max_items]]}
    if isinstance(obj, pd.Series):
        return {"serie": f"{len(obj)} valores", "nombre": str(obj.name)}
    if isinstance(obj, np.ndarray):
        return {"arreglo": list(obj.shape)}
    if isinstance(obj, dict):
        resumen = {str(k): summarize(v, max_items=max_items, max_text=max_text)
                   for k, v in list(obj.items())[:max_items]}
        if len(obj) > max_items:
            resumen["…"] = f"{len(obj) - max_items} claves más"
        return resumen
    if isinstance(obj, (list, tuple, set)):
        elementos = list(obj)
        resumen = [summarize(v, max_items=max_items, max_text=max_text) for v in elementos[:max_items]]
        if len(elementos) > max_items:
            resumen.append(f"… {len(elementos) - max_items} elementos más")
        return resumen
    if isinstance(obj, str) and len(obj) > max_text:
        return obj[:max_text] + "…"
    if _es_figura(obj):
        return "<figura>"
    convertido = to_jsonable(obj)
    if isinstance(convertido, str) and len(convertido) > max_text:
        return convertido[:max_text] + "…"
    return convertido


def dumps(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` tolerante a los tipos del framework."""
    return json.dumps(to_jsonable(obj), ensure_ascii=False, **kwargs)


__all__ = [
    "DATAFRAME_TAG",
    "dataframe_from_jsonable",
    "dumps",
    "figure_to_base64",
    "figure_to_png_bytes",
    "summarize",
    "to_jsonable",
]
