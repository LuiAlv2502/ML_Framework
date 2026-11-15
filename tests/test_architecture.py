"""Restricción de arquitectura: el frontend no contiene ML ni importa la capa ``framework_ia.ml``."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = sorted((ROOT / "frontend").rglob("*.py")) + [ROOT / "app.py"]
PROHIBIDOS = ("sklearn", "xgboost", "umap", "scipy", "framework_ia.ml", "framework_ia.agents")


def _imports(ruta: Path) -> list[str]:
    nombres = []
    for nodo in ast.walk(ast.parse(ruta.read_text(encoding="utf-8"))):
        if isinstance(nodo, ast.Import):
            nombres += [alias.name for alias in nodo.names]
        elif isinstance(nodo, ast.ImportFrom):
            nombres.append(nodo.module or "")
    return nombres


@pytest.mark.parametrize("ruta", FRONTEND, ids=lambda r: str(r.relative_to(ROOT)))
def test_frontend_no_importa_ml(ruta):
    prohibidos = [n for n in _imports(ruta) if any(n == p or n.startswith(p + ".") for p in PROHIBIDOS)]
    assert not prohibidos, f"{ruta.name} importa {prohibidos}; use framework_ia.controller"


def test_paginas_usan_el_controlador():
    paginas = sorted((ROOT / "frontend" / "pages").glob("*.py"))
    assert paginas
    for pagina in paginas:
        assert "frontend.components.shared" in _imports(pagina), f"{pagina.name} no usa el controlador"
