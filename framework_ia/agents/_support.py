"""Utilidades compartidas por los adaptadores de agentes especializados."""
from __future__ import annotations

import contextlib
import io
from typing import Any, Callable, Iterator

import pandas as pd

from framework_ia.utils.exceptions import FrameworkError, ValidationError, friendly_message
from framework_ia.utils.validation import validate_columns, validate_range

LLM_DISABLED_REASON = "El LLM se desactivó para esta tarea (respaldo por reglas)."


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Silencia los ``print`` informativos de las clases de ML heredadas de los laboratorios."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def ml_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Ejecuta un método de ML y traduce sus errores de datos (KeyError/ValueError/TypeError) a ``ValidationError``."""
    try:
        with quiet():
            return func(*args, **kwargs)
    except FrameworkError:
        raise
    except (KeyError, ValueError, TypeError) as error:
        raise ValidationError(friendly_message(error)) from error


def run_legacy(agent: Any, use_llm: bool) -> Any:
    """Ejecuta un agente de los laboratorios; con ``use_llm=False`` fuerza el respaldo por reglas."""
    with quiet():
        if use_llm:
            return agent.ejecutar()
        agent._reiniciar()
        agent.modo = "reglas"
        agent.motivo_respaldo = LLM_DISABLED_REASON
        agent._ejecutar_reglas()
        return agent


def seeds(payload: dict[str, Any], default: int = 42) -> list[int]:
    """Semillas de las particiones: ``seeds`` (lista) o ``random_state`` (una)."""
    valores = payload.get("seeds")
    if valores is None:
        valores = [payload.get("random_state", default)]
    if not isinstance(valores, (list, tuple)) or not valores or len(valores) > 50:
        raise ValidationError("Indique entre 1 y 50 semillas.")
    return [int(validate_range("semilla", v, 0, 2**32 - 1, integer=True)) for v in valores]


def features(df: pd.DataFrame, payload: dict[str, Any], target: Any) -> list[Any] | None:
    """Variables predictoras validadas (``None`` = todas menos el objetivo)."""
    columnas = payload.get("features")
    if columnas is None:
        return None
    columnas = validate_columns(df, columnas, role="variables predictoras")
    if not columnas:
        raise ValidationError("Seleccione al menos una variable predictora.")
    if target in columnas:
        raise ValidationError("La variable objetivo no puede ser también predictora.")
    return columnas


def primary(name: str, value: Any, maximize: bool) -> dict[str, Any]:
    """Métrica principal que usa el coordinador para consolidar resultados."""
    return {"name": name, "value": None if value is None else float(value), "maximize": maximize}


def use_llm(agent: Any, payload: dict[str, Any]) -> bool:
    """El payload manda; si no lo indica, se usa la configuración del agente (por defecto ``False``)."""
    if "use_llm" in payload:
        return bool(payload["use_llm"])
    return bool(agent.config.get("use_llm", False))


__all__ = ["LLM_DISABLED_REASON", "features", "ml_call", "primary", "quiet", "run_legacy", "seeds", "use_llm"]
