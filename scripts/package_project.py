"""Crea un ZIP del proyecto sin entornos virtuales ni archivos locales generados."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git", ".idea", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv",
    "__pycache__", "build", "dist", "htmlcov", "logs", "runs", "framework_ia.egg-info",
}
EXCLUDED_FILES = {".coverage", ".env", "secrets.toml"}


def _include(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part.lower() in EXCLUDED_DIRS or part.lower().endswith(".egg-info") for part in relative.parts):
        return False
    if path.name in EXCLUDED_FILES or path.name.startswith(".coverage") \
            or path.suffix.lower() in {".pyc", ".pyo"}:
        return False
    if path.name.startswith(".env.") and path.name != ".env.example":
        return False
    if relative.parts[:2] == ("framework_ia", "models"):
        return False
    if relative.as_posix() == "framework_ia/persistence/agents.json":
        return False
    return path.is_file()


def build_archive(output: Path, root: Path = ROOT) -> Path:
    """Escribe en ``output`` y devuelve la ruta del ZIP creado."""
    root = root.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for current, directories, filenames in os.walk(root):
            directories[:] = sorted(
                name for name in directories
                if name.lower() not in EXCLUDED_DIRS and not name.lower().endswith(".egg-info")
            )
            for name in sorted(filenames):
                path = Path(current) / name
                if _include(path, root) and path.resolve() != output:
                    archive.write(path, path.relative_to(root))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "framework-ia.zip",
        help="Ruta del archivo ZIP de salida.",
    )
    args = parser.parse_args()
    destino = build_archive(args.output)
    print(f"Paquete creado: {destino}")


if __name__ == "__main__":
    main()
