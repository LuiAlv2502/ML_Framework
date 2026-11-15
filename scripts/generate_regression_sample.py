"""Genera un conjunto sintético y reproducible para probar tareas de regresión."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(output: Path, rows: int = 300, seed: int = 42) -> None:
    rng = random.Random(seed)
    departments = {"Ventas": 2500, "Ingeniería": 7000, "Operaciones": 1000}
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "experiencia_anios",
            "nivel_educativo",
            "horas_semanales",
            "proyectos_completados",
            "departamento",
            "ingreso_anual",
        ])
        for _ in range(rows):
            experience = rng.randint(0, 35)
            education = rng.choice(["secundaria", "licenciatura", "maestria", "doctorado"])
            education_level = {"secundaria": 0, "licenciatura": 1, "maestria": 2, "doctorado": 3}[education]
            hours = rng.randint(20, 60)
            projects = rng.randint(0, 20)
            department = rng.choice(list(departments))
            income = (
                30000
                + 1800 * experience
                + 5000 * education_level
                + 300 * hours
                + 600 * projects
                + departments[department]
                + rng.gauss(0, 5000)
            )
            writer.writerow([experience, education, hours, projects, department, round(income, 2)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "regression_sample.csv")
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows < 10:
        parser.error("--rows debe ser al menos 10.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate(args.output, rows=args.rows, seed=args.seed)
    print(f"Dataset sintético escrito en {args.output} ({args.rows} filas).")


if __name__ == "__main__":
    main()
