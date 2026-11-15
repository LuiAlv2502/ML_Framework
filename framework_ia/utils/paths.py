"""Rutas del proyecto usadas por la configuración y la persistencia."""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


def resolve(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Devuelve ``path`` absoluta; las rutas relativas se interpretan desde la raíz del proyecto."""
    ruta = Path(path)
    return ruta if ruta.is_absolute() else (base / ruta)


__all__ = ["PACKAGE_ROOT", "PROJECT_ROOT", "resolve"]
