"""Agentes de prueba compartidos por los tests del sistema multiagente."""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from framework_ia.agents import BaseAgent, Message
from framework_ia.utils.exceptions import ValidationError


class EchoAgent(BaseAgent):
    """Devuelve una métrica configurable; sirve para probar enrutamiento y consolidación."""

    agent_type = "echo_test"
    default_capabilities = frozenset({"test.echo"})
    config_schema = {"score": ("real", 0.0, 1.0)}

    def _execute(self, message: Message) -> dict[str, Any]:
        score = self.config.get("score", 0.5)
        return {"echo": message.payload.get("value"), "agent": self.agent_id,
                "primary_metric": {"name": "score", "value": score, "maximize": True}}


class FlakyAgent(BaseAgent):
    """Falla las primeras ``fail_times`` veces que se le pide una tarea."""

    agent_type = "flaky_test"
    default_capabilities = frozenset({"test.echo"})
    config_schema = {"fail_times": ("entero", 0, 100)}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    def _execute(self, message: Message) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self.config.get("fail_times", 1):
            raise RuntimeError(f"fallo simulado {self.calls}")
        return {"echo": message.payload.get("value"), "agent": self.agent_id,
                "primary_metric": {"name": "score", "value": 0.1, "maximize": True}}


class PickyAgent(BaseAgent):
    """Rechaza la solicitud como error del usuario."""

    agent_type = "picky_test"
    default_capabilities = frozenset({"test.echo"})

    def _execute(self, message: Message) -> dict[str, Any]:
        raise ValidationError("dato inválido del usuario")


@pytest.fixture
def drug() -> pd.DataFrame:
    from pathlib import Path

    return pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "drug200.csv")
