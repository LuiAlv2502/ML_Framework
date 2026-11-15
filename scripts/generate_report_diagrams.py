"""Genera diagramas PNG del framework para insertarlos en el informe LaTeX."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "resultados" / "figures"
COLORS = {
    "user": "#e8eef8",
    "ui": "#d9eaf7",
    "control": "#cde8e2",
    "mas": "#fff0c2",
    "agents": "#f8d7da",
    "ml": "#e6def6",
    "store": "#e5e7eb",
    "edge": "#334155",
}


def _node(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    color: str,
    *,
    fontsize: float = 9,
) -> tuple[float, float]:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=color,
        edgecolor="#475569",
        linewidth=1.1,
    )
    axis.add_patch(box)
    axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=fontsize)
    return x + width / 2, y + height / 2


def _arrow(
    axis: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
    *,
    curve: float = 0,
    dashed: bool = False,
    label_offset: tuple[float, float] = (0, 0),
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["edge"],
            "linewidth": 1.15,
            "linestyle": "--" if dashed else "-",
            "connectionstyle": f"arc3,rad={curve}",
        },
    )
    if label:
        x = (start[0] + end[0]) / 2 + label_offset[0]
        y = (start[1] + end[1]) / 2 + label_offset[1]
        axis.text(x, y, label, ha="center", va="center", fontsize=7.5, color="#1e293b",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})


def _finish(axis: Any, title: str, bounds: tuple[float, float, float, float]) -> None:
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_ylim(bounds[2], bounds[3])
    axis.set_aspect("auto")
    axis.axis("off")
    axis.set_title(title, fontsize=14, fontweight="bold", pad=15)


def _save(figure: Any, filename: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def arquitectura() -> None:
    figure, axis = plt.subplots(figsize=(12, 7.2))
    _node(axis, 5.0, 6.25, 2.0, 0.55, "Persona usuaria", COLORS["user"])
    _node(axis, 4.65, 5.15, 2.7, 0.65, "Dashboard Streamlit", COLORS["ui"])
    _node(axis, 4.35, 3.95, 3.3, 0.7, "FrameworkController", COLORS["control"])
    _node(axis, 4.4, 2.7, 3.2, 0.8, "Agencia MAS\ncoordinador · registro · fábrica · bus", COLORS["mas"],
          fontsize=8.5)
    _node(axis, 8.6, 2.7, 3.0, 0.8, "Agentes especializados\nEDA · agrupamiento · clasificación · regresión",
          COLORS["agents"], fontsize=8.2)
    _node(axis, 0.4, 1.15, 3.0, 0.8, "Servicios y persistencia\ndatos · preparación · experimentos",
          COLORS["store"], fontsize=8.0)
    _node(axis, 8.6, 1.15, 3.0, 0.8, "Backend ML\npreprocesamiento y modelos", COLORS["ml"])

    _arrow(axis, (6, 6.25), (6, 5.8))
    _arrow(axis, (6, 5.15), (6, 4.65))
    _arrow(axis, (6, 3.95), (6, 3.5))
    _arrow(axis, (7.6, 3.1), (8.6, 3.1))
    _arrow(axis, (4.55, 4.0), (3.4, 1.95), curve=-0.05)
    _arrow(axis, (10.1, 2.7), (10.1, 1.95))
    _finish(axis, "Arquitectura por capas y flujo multiagente", (-0.05, 12.05, 0.6, 6.95))
    _save(figure, "05_arquitectura_mas.png")


def componentes() -> None:
    figure, axis = plt.subplots(figsize=(12, 7.5))
    _node(axis, 4.7, 6.3, 2.6, 0.6, "Dashboard Streamlit", COLORS["ui"])
    _node(axis, 4.45, 5.15, 3.1, 0.65, "FrameworkController", COLORS["control"])
    _node(axis, 4.25, 3.95, 3.5, 0.7, "Agency\ncoordinador · registro · fábrica · bus", COLORS["mas"],
          fontsize=8.5)
    _node(axis, 4.45, 2.3, 3.1, 0.7, "BaseAgent\ncontrato común", COLORS["agents"])
    agents = [
        _node(axis, 0.15, 0.35, 2.1, 0.7, "CoordinatorAgent", COLORS["mas"], fontsize=7.8),
        _node(axis, 2.5, 0.35, 2.1, 0.7, "EDAAgent", COLORS["agents"]),
        _node(axis, 4.85, 0.35, 2.1, 0.7, "ClusteringAgent", COLORS["agents"], fontsize=7.8),
        _node(axis, 7.2, 0.35, 2.1, 0.7, "ClassificationAgent", COLORS["agents"], fontsize=7.2),
        _node(axis, 9.55, 0.35, 2.1, 0.7, "RegressionAgent", COLORS["agents"], fontsize=7.8),
    ]

    _arrow(axis, (6, 6.3), (6, 5.8))
    _arrow(axis, (6, 5.15), (6, 4.65))
    _arrow(axis, (6, 3.95), (6, 3.0), "orquesta", label_offset=(0.45, 0))
    for index, agent in enumerate(agents):
        endpoint = 5.0 + 0.5 * index
        _arrow(axis, (agent[0], 1.05), (endpoint, 2.3), dashed=True)
    axis.text(6, 0.05, "Las clases especializadas heredan el contrato común de BaseAgent.",
              ha="center", va="center", fontsize=8.2, color="#334155")
    _finish(axis, "Componentes y especialización de agentes", (-0.05, 12.05, -0.15, 7.0))
    _save(figure, "06_componentes_mas.png")


def secuencia() -> None:
    figure, axis = plt.subplots(figsize=(13, 7.5))
    participantes = [
        ("Usuario", 0.6, COLORS["user"]),
        ("Streamlit", 2.4, COLORS["ui"]),
        ("Controller", 4.2, COLORS["control"]),
        ("Agency / bus", 6.0, COLORS["mas"]),
        ("Coordinador", 7.8, COLORS["mas"]),
        ("Registro", 9.6, COLORS["mas"]),
        ("Agente ML", 11.4, COLORS["agents"]),
        ("Experimentos", 13.2, COLORS["store"]),
    ]
    for label, x, color in participantes:
        _node(axis, x - 0.62, 6.0, 1.25, 0.55, label, color, fontsize=8)
        axis.plot([x, x], [0.35, 5.95], linestyle=(0, (3, 3)), color="#94a3b8", linewidth=0.9)
    mensajes = [
        (0.6, 2.4, "1. configura análisis"),
        (2.4, 4.2, "2. tarea + datos"),
        (4.2, 6.0, "3. solicitud"),
        (6.0, 7.8, "4. entrega REQUEST"),
        (7.8, 9.6, "5. consulta capacidad"),
        (7.8, 6.0, "6. candidato(s)"),
        (6.0, 11.4, "7. mensaje de tarea"),
        (11.4, 6.0, "8. resultado / fallo"),
        (6.0, 7.8, "9. consolida"),
        (6.0, 4.2, "10. resultado + decisiones"),
        (4.2, 13.2, "11. persiste ejecución"),
        (4.2, 2.4, "12. muestra métricas"),
    ]
    ys = [5.45, 4.98, 4.51, 4.04, 3.57, 3.1, 2.63, 2.16, 1.69, 1.22, 0.75, 0.28]
    for (sender, receiver, label), y in zip(mensajes, ys):
        _arrow(axis, (sender, y), (receiver, y), label, label_offset=(0, 0.16))
    _finish(axis, "Secuencia de una solicitud de análisis", (-0.1, 13.9, 0, 6.8))
    _save(figure, "07_secuencia_mas.png")


def main() -> None:
    arquitectura()
    componentes()
    secuencia()
    print(f"Diagramas generados en {OUTPUT}")


if __name__ == "__main__":
    main()
