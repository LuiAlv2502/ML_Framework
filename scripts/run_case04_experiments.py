"""Ejecuta y reporta el estudio multiagente reproducible del Caso 04."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn

from framework_ia.controller import FrameworkController
from framework_ia.persistence.config_store import ConfigStore
from framework_ia.utils.logging_config import close_logging

OUTPUT = ROOT / "docs" / "resultados" / "caso04"
DEFAULT_SEEDS = [42, 43, 44]
DATASETS = [
    {"file": "drug200.csv", "target": "Drug", "description": "Medicamentos (cinco clases)"},
    {"file": "diabetes.csv", "target": "Outcome", "description": "Diabetes (clasificación binaria)"},
    {"file": "water_potability.csv", "target": "Potability", "description": "Potabilidad (clasificación binaria)"},
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _number(value: Any, decimals: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"


def _scenario_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for experiment in results:
        for scenario in experiment["scenarios"]:
            rows.append({
                "dataset": experiment["dataset"],
                "target": experiment["target"],
                "scenario": scenario["scenario"],
                "complexity": scenario["complexity"],
                "trials": scenario["trials"],
                "duration_s_total": scenario["duration_s"],
                "interactions_total": scenario["interactions"],
                "tasks_completed_total": scenario["tasks_completed"],
                "errors_total": scenario["errors"],
                "quality_metric": scenario["quality_metric"],
                "quality_mean": scenario["quality_mean"],
                "quality_std": scenario["quality_std"],
            })
    return rows


def _comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for experiment in results:
        for trial in experiment["comparison"]["trials"]:
            rows.append({
                "dataset": experiment["dataset"],
                "target": experiment["target"],
                "seed": trial["seed"],
                "approach": trial["approach"],
                "duration_s": trial["duration_s"],
                "interactions": trial["interactions"],
                "tasks_completed": trial["tasks_completed"],
                "models_evaluated": trial["models_evaluated"],
                "errors": trial["errors"],
                "quality_metric": trial["quality_metric"],
                "quality": trial["quality"],
                "selected": trial["selected"],
                "run_id": trial["run_id"],
            })
    return rows


def _comparison_summary(rows: list[dict[str, Any]], dataset: str, approach: str) -> dict[str, Any]:
    matches = [
        row for row in rows
        if row["dataset"] == dataset and row["approach"] == approach
    ]
    times = [float(row["duration_s"]) for row in matches]
    qualities = [float(row["quality"]) for row in matches]
    return {
        "duration_mean": statistics.mean(times),
        "duration_std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "quality_mean": statistics.mean(qualities),
        "quality_std": statistics.stdev(qualities) if len(qualities) > 1 else 0.0,
        "interactions_mean": statistics.mean(float(row["interactions"]) for row in matches),
        "errors_total": sum(int(row["errors"]) for row in matches),
    }


def _render_report(manifest: dict[str, Any], scenario_rows: list[dict[str, Any]],
                   comparison_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Estudio de Caso 04: Agencia y Sistema Multiagente",
        "",
        f"**Ejecución:** {manifest['generated_at']}  ",
        f"**Python:** {manifest['environment']['python']} · "
        f"**pandas:** {manifest['environment']['pandas']} · "
        f"**scikit-learn:** {manifest['environment']['scikit_learn']}  ",
        f"**Protocolo:** partición estratificada 75/25 · semillas {', '.join(map(str, manifest['seeds']))} · "
        "F1 macro para selección.",
        "",
        "## Resumen",
        "",
        "Este estudio ejecuta tres escenarios de complejidad creciente sobre conjuntos incluidos localmente en "
        "el proyecto. El servicio externo de IA se desactiva; el EDA usa su modo reproducible basado en reglas. "
        "El tercer escenario compara una evaluación centralizada con el mismo conjunto de estrategias KNN que "
        "atienden los agentes. Los resultados son mediciones de esta ejecución, no promesas de rendimiento "
        "generalizable.",
        "",
        "## Fundamentación y mapeo de agencia",
        "",
        "Un agente es una entidad computacional situada en un entorno que percibe y actúa de forma autónoma "
        "para lograr objetivos; un MAS coordina entidades que interactúan para resolver tareas relacionadas "
        "[1, 2]. Autonomía, reactividad, iniciativa y habilidad social se interpretan aquí como propiedades "
        "operacionales, no como una implementación BDI completa.",
        "",
        "| Propiedad o concepto | Implementación observada | Alcance y límite |",
        "|---|---|---|",
        "| Entorno y percepción | Los agentes reciben una tabla y una solicitud estructurada; EDA calcula perfil, nulos, duplicados y tipos. | El entorno es el dataset cargado, el registro y el bus en memoria. |",
        "| Autonomía | Cada agente valida y ejecuta la capacidad que publica sin que el coordinador conozca su clase concreta. | La autonomía está acotada a la tarea delegada; no hay objetivos persistentes. |",
        "| Reactividad | `BaseAgent.handle` responde a solicitudes y transiciones del ciclo de vida. | No hay un bucle sensorial continuo. |",
        "| Proactividad | El coordinador inicia la delegación y la consolidación una vez recibida una solicitud. | Los especialistas no generan tareas espontáneas. |",
        "| Acciones y objetivos | Los agentes ejecutan la capacidad asignada; clasificación maximiza F1 macro y K-Means maximiza silhouette. | El objetivo es local a la solicitud, no persistente. |",
        "| Conocimiento | Cada agente usa su configuración, capacidades, algoritmo y estadísticas operativas. | No hay una ontología ni una base de conocimiento compartida. |",
        "| Habilidad social | Mensajes tipados con emisor, receptor, performativa, tarea y correlación; respuestas registradas por el bus. | Comunicación síncrona local; no es una red distribuida. |",
        "| Estado y memoria de trabajo | Estados de agente, contadores, fallos consecutivos, tareas recientes y registros persistidos por experimento. | La bitácora del agente es acotada; no existe memoria semántica compartida de largo plazo. |",
        "| Coordinación y resolución | Enrutamiento por capacidad; `priority` admite fallback y `fanout` elige por F1 macro. | La selección es arbitraje por métrica, no negociación estratégica. |",
        "",
        "El diseño global es **híbrido**: el coordinador aplica reglas de enrutamiento y políticas; el EDA "
        "construye un plan de análisis; los agentes de aprendizaje ejecutan modelos estadísticos. No se afirma "
        "que sea una arquitectura BDI formal. En una arquitectura reactiva, la acción suele vincularse "
        "directamente con la percepción; la deliberativa mantiene razonamiento explícito sobre objetivos; la "
        "híbrida combina ambos estilos [1, 2].",
        "",
        "### Roles",
        "",
        "| Rol | Capacidad usada en los escenarios | Objetivo |",
        "|---|---|---|",
        "| Coordinador | `coordination.route` | Interpretar la tarea, buscar agentes compatibles, delegar y consolidar. |",
        "| Agente de agrupamiento | `clustering.kmeans` | Formar tres grupos iniciales y medir silhouette. |",
        "| Analista EDA | `eda.profile` | Analizar y preparar una copia de los datos. |",
        "| Clasificador | `classification.train` | Entrenar KNN fijo con la partición solicitada. |",
        "| KNN automático | `classification.train` | Evaluar varios valores impares de k y reportar el mejor F1 macro. |",
        "| Bus y registro | Mensajes correlacionados y registro de ejecuciones | Mantener trazabilidad de solicitudes, respuestas y decisiones. |",
        "",
        "### Arquitectura (diagrama textual)",
        "",
        "```text",
        "Dashboard MAS / script de experimentos",
        "                  |",
        "          FrameworkController",
        "                  | REQUEST(task, payload, correlation_id)",
        "                  v",
        "          Coordinador MAS",
        "       /          |            |           \\",
        "ClusteringAgent EDAAgent ClassificationAgent KNNAutoAgent",
        "       \\          |            |           /",
        "          MessageBus + Registry",
        "                  |",
        "   Servicios ML / ExperimentStore local",
        "```",
        "",
        "### Flujo del escenario cooperativo",
        "",
        "```text",
        "Controlador -> Coordinador: solicitud eda",
        "Coordinador -> EDAAgent: perfil del conjunto",
        "EDAAgent -> Coordinador: perfil + copia preparada",
        "Controlador -> Coordinador: classification (copia preparada, semilla)",
        "Coordinador -> ClassificationAgent: KNN fijo",
        "ClassificationAgent -> Controlador: métricas + traza",
        "```",
        "",
        "### Flujo de decisión / conflicto",
        "",
        "```text",
        "Controlador -> Coordinador: classification, policy=fanout",
        "Coordinador -> ClassificationAgent: KNN k=5",
        "Coordinador -> KNNAutoAgent: KNN k en {3,5,7,9,11} (limitado por tamaño de entrenamiento)",
        "Ambos agentes -> Coordinador: resultado + F1 macro",
        "Coordinador: conserva el mayor F1 macro y registra alternativas y decisión",
        "```",
        "",
        "El fan-out del framework actual es síncrono. El diagrama representa mensajes y arbitraje, no ejecución "
        "paralela sobre procesos o máquinas distintas.",
        "",
        "## Metodología",
        "",
        "### Escenarios",
        "",
        "1. **Baja complejidad — agrupamiento inicial:** el agente K-Means forma tres grupos sobre dos predictores "
        "numéricos; la métrica silhouette describe su cohesión/separación y no usa la variable objetivo.",
        "2. **Complejidad media — cooperación secuencial:** EDA produce una copia preparada que se entrega al "
        "clasificador; se repite el entrenamiento con las semillas indicadas.",
        "3. **Alta complejidad — decisión entre alternativas:** `fanout` ejecuta el clasificador KNN fijo y el "
        "buscador KNN automático. La línea base centralizada ejecuta esas mismas estrategias directamente, "
        "sin coordinador, bus ni agentes.",
        "",
        "### Datos y protocolo",
        "",
        "| Archivo local | Objetivo | Filas | Columnas totales | Huella SHA-256 |",
        "|---|---|---:|---:|---|",
    ]
    for dataset in manifest["datasets"]:
        lines.append(
            f"| `{dataset['file']}` | `{dataset['target']}` | {dataset['rows']} | {dataset['columns']} | "
            f"`{dataset['sha256']}` |"
        )
    lines.extend([
        "",
        "Las huellas identifican los archivos usados, pero no acreditan su procedencia ni licencia. Los CSV "
        "heredados del repositorio no tienen esos metadatos documentados; deben verificarse antes de "
        "redistribuirlos. La partición, el escalado y la semilla se mantienen iguales entre los dos enfoques. "
        "La selección se hace por F1 macro para ponderar todas las clases.",
        "",
        "## Resultados medidos: escenarios",
        "",
        "| Conjunto | Escenario | Complejidad | Corridas | Tiempo total (s) | Interacciones | Tareas completadas | Errores | Métrica de calidad (media ± DE) |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in scenario_rows:
        calidad = "—" if row["quality_mean"] is None else (
            f"{row['quality_metric']}: {_number(row['quality_mean'])} ± {_number(row['quality_std'])}"
        )
        lines.append(
            f"| {row['dataset']} | {row['scenario']} | {row['complexity']} | {row['trials']} | "
            f"{_number(row['duration_s_total'])} | {row['interactions_total']} | "
            f"{row['tasks_completed_total']} | {row['errors_total']} | {calidad} |"
        )
    lines.extend([
        "",
        "## Comparación centralizada y MAS",
        "",
        "| Conjunto | Semilla | Enfoque | Tiempo (s) | Interacciones | Tareas completadas | Modelos evaluados | Errores | F1 macro | Selección |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in comparison_rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['approach']} | {_number(row['duration_s'])} | "
            f"{row['interactions']} | {row['tasks_completed']} | {row['models_evaluated']} | "
            f"{row['errors']} | {_number(row['quality'])} | {row['selected']} |"
        )
    lines.extend([
        "",
        "### Análisis agregado de la comparación",
        "",
        "| Conjunto | Tiempo central (media ± DE) | Tiempo MAS (media ± DE) | Cambio de tiempo MAS (%) | Diferencia F1 macro | Interacciones MAS / prueba | Errores MAS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in manifest["datasets"]:
        nombre = dataset["file"]
        central = _comparison_summary(comparison_rows, nombre, "Centralizado")
        mas = _comparison_summary(comparison_rows, nombre, "MAS")
        cambio_tiempo = (
            100 * (mas["duration_mean"] - central["duration_mean"]) / central["duration_mean"]
            if central["duration_mean"] else 0.0
        )
        diferencia_f1 = mas["quality_mean"] - central["quality_mean"]
        lines.append(
            f"| {nombre} | {_number(central['duration_mean'])} ± {_number(central['duration_std'])} | "
            f"{_number(mas['duration_mean'])} ± {_number(mas['duration_std'])} | "
            f"{_number(cambio_tiempo, 2)}% | {_number(diferencia_f1)} | "
            f"{_number(mas['interactions_mean'], 2)} | {mas['errors_total']} |"
        )
    f1_iguales = all(
        abs(float(fila["quality"]) - float(siguiente["quality"])) < 1e-12
        for fila in comparison_rows
        if fila["approach"] == "Centralizado"
        for siguiente in comparison_rows
        if siguiente["approach"] == "MAS"
        and siguiente["dataset"] == fila["dataset"]
        and siguiente["seed"] == fila["seed"]
    )
    lines.extend([
        "",
        "En esta corrida, la calidad coincide en cada par de conjunto/semilla: "
        + ("sí." if f1_iguales else "no; las diferencias deben interpretarse junto a la tabla por semilla."),
        "Con el mismo conjunto de candidatos, partición y métrica, el valor de F1 debe coincidir; el propósito "
        "de esta comparación es observar el costo de coordinación y las interacciones. El porcentaje de tiempo "
        "puede oscilar con el equipo y con la carga del proceso; solo se repitieron tres semillas.",
        "",
        "La versión centralizada y el MAS entrenan el mismo número de modelos KNN por semilla. El MAS añade "
        "enrutamiento, mensajes, consolidación y persistencia; por ello su tiempo de pared puede ser mayor. "
        "Las métricas de calidad se comparan directamente solo dentro del mismo conjunto y semilla. La "
        "desviación estándar describe variación entre semillas y no es un intervalo de confianza.",
        "",
        "## Discusión crítica",
        "",
        "### Fortalezas",
        "",
        "- Separación entre la interfaz/controlador y agentes especializados; el coordinador enruta por capacidad.",
        "- Cooperación observable mediante una transferencia explícita del resultado EDA a clasificación.",
        "- Resolución explicable por una métrica primaria, con agentes candidatos, alternativas y decisión guardados.",
        "- La misma partición y los mismos parámetros permiten contrastar calidad y sobrecosto operacional entre enfoques.",
        "- El registro de mensajes, intentos y experimentos facilita repetir y auditar cada ejecución.",
        "",
        "### Limitaciones y mejoras",
        "",
        "- El flujo es local y síncrono; no mide latencia de red, concurrencia ni tolerancia a fallos de infraestructura.",
        "- El conflicto se resuelve por máximo F1 macro, no por negociación, utilidad múltiple o preferencias del usuario.",
        "- Las conclusiones dependen de los conjuntos disponibles, de sus licencias aún no confirmadas y de la selección de KNN.",
        "- Para una evaluación más fuerte: añadir particiones externas, pruebas estadísticas pareadas, más algoritmos, "
        "inyección controlada de fallos y medir disponibilidad, consumo de memoria y escalabilidad.",
        "- Las pruebas automáticas del proyecto cubren fallback, reintentos, errores del usuario, estados, mensajes y "
        "creación de plugins; este experimento no introduce fallos artificiales en los resultados de calidad.",
        "",
        "## Reproducibilidad",
        "",
        "Desde la raíz del repositorio, con dependencias instaladas:",
        "",
        "```powershell",
        "python scripts/run_case04_experiments.py",
        "```",
        "",
        "La ejecución desactiva el LLM y regenera este informe y los CSV/JSON en "
        "`docs/resultados/caso04/`. Para usar otras semillas: "
        "`python scripts/run_case04_experiments.py --seeds 11 42 73`.",
        "",
        "## Bibliografía",
        "",
        "1. Wooldridge, M. y Jennings, N. R. (1995). Intelligent agents: Theory and practice. "
        "*The Knowledge Engineering Review*, 10(2), 115–152. "
        "https://doi.org/10.1017/S0269888900008122",
        "2. Jennings, N. R., Sycara, K. y Wooldridge, M. (1998). A roadmap of agent research and development. "
        "*Autonomous Agents and Multi-Agent Systems*, 1, 7–38. https://doi.org/10.1023/A:1010090405266",
        "3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. "
        "*Journal of Machine Learning Research*, 12, 2825–2830.",
        "4. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. "
        "*Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.",
        "5. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster "
        "analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. "
        "https://doi.org/10.1016/0377-0427(87)90125-7",
        "",
    ])
    return "\n".join(lines)


def run_study(
    output_dir: str | Path = OUTPUT,
    *,
    seeds: list[int] | None = None,
    datasets: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Ejecuta el protocolo completo y escribe artefactos de resultados sin imágenes."""
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    datasets = list(DATASETS if datasets is None else datasets)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("Indique una lista no vacía de semillas enteras distintas.")
    if not datasets:
        raise ValueError("Indique al menos un conjunto de datos para evaluar.")

    directorio_salida = Path(output_dir)
    directorio_salida.mkdir(parents=True, exist_ok=True)
    experimentos = []
    metadatos_datasets = []
    config = ConfigStore()
    config.set("llm.enabled", False)

    with tempfile.TemporaryDirectory(prefix="caso04-") as temporal:
        ruta_temporal = Path(temporal)
        config.set("paths.logs_dir", str(ruta_temporal / "logs"))
        controlador = FrameworkController(config=config, persist=False, runs_dir=ruta_temporal / "runs")
        try:
            for definicion in datasets:
                ruta = ROOT / "data" / definicion["file"]
                controlador.load_sample(definicion["file"])
                resultado = controlador.run_case04_evaluation(definicion["target"], seeds)
                resultado["file"] = definicion["file"]
                resultado["description"] = definicion.get("description", definicion["file"])
                experimentos.append(resultado)
                metadatos_datasets.append({
                    "file": definicion["file"],
                    "target": definicion["target"],
                    "rows": resultado["rows"],
                    "columns": resultado["columns"],
                    "sha256": _sha256(ruta),
                })
        finally:
            close_logging()

    manifest = {
        "title": "Estudio de Caso 04: Agencia y Sistema Multiagente",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "seeds": seeds,
        "test_size": 0.25,
        "selection_metric": "f1_macro",
        "datasets": metadatos_datasets,
        "experiments": experimentos,
    }
    escenarios = _scenario_rows(experimentos)
    comparaciones = _comparison_rows(experimentos)
    (directorio_salida / "resultados.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(escenarios).to_csv(
        directorio_salida / "resultados_escenarios.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(comparaciones).to_csv(
        directorio_salida / "comparacion_central_mas.csv", index=False, encoding="utf-8-sig"
    )
    informe = _render_report(manifest, escenarios, comparaciones)
    (directorio_salida / "INFORME_CASO04.md").write_text(informe, encoding="utf-8")
    print(f"Informe generado: {directorio_salida / 'INFORME_CASO04.md'}")
    print(f"Resultados: {directorio_salida / 'resultados.json'}")
    print(pd.DataFrame(escenarios).to_string(index=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta los experimentos reproducibles del Caso 04.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT,
        help="Directorio de salida (predeterminado: docs/resultados/caso04).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Semillas enteras distintas para repetir clasificación (predeterminado: 42 43 44).",
    )
    arguments = parser.parse_args()
    run_study(arguments.output_dir, seeds=arguments.seeds)


if __name__ == "__main__":
    main()
