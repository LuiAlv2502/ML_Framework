from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import llm


class _RateLimitError(Exception):
    status_code = 429

    def __init__(self, retry_after=None):
        self.response = NS(
            status_code=429,
            headers={} if retry_after is None else {"Retry-After": str(retry_after)},
        )


class _ClienteConLimite:
    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        respuesta = self.respuestas.pop(0)
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta


def test_completar_respeta_retry_after(monkeypatch):
    esperas = []
    monkeypatch.setattr(llm.time, "sleep", esperas.append)
    cliente = _ClienteConLimite([_RateLimitError(2), "ok"])

    assert llm.completar(cliente, reintentos=1) == "ok"
    assert esperas == [2.0]


def test_completar_reintenta_429_y_expone_motivo_accionable(monkeypatch):
    esperas = []
    monkeypatch.setattr(llm.time, "sleep", esperas.append)
    cliente = _ClienteConLimite([_RateLimitError(), _RateLimitError()])

    with pytest.raises(llm.ErrorLLM, match="cuota agotada"):
        llm.completar(cliente, reintentos=1)

    assert esperas == [5]
