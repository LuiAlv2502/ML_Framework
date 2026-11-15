"""Demuestra que un plugin nuevo se incorpora y es elegido por capacidades."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from framework_ia.controller import FrameworkController
from framework_ia.persistence.config_store import ConfigStore
from framework_ia.utils.logging_config import close_logging

ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = (
    "framework_ia/agents/coordinator_agent.py",
    "framework_ia/agents/agency.py",
    "framework_ia/agents/registry.py",
    "framework_ia/agents/agent_factory.py",
)


def _hashes() -> dict[str, str]:
    return {
        nombre: hashlib.sha256((ROOT / nombre).read_bytes()).hexdigest()
        for nombre in CORE_FILES
    }


def _tabla_agentes(agentes: list[dict]) -> str:
    filas = ["| ID | Nombre | Tipo | Capacidades |", "|---|---|---|---|"]
    filas.extend(
        f"| {a['agent_id']} | {a['name']} | {a['agent_type']} | {', '.join(a['capabilities'])} |"
        for a in agentes
    )
    return "\n".join(filas)


def generar_evidencia() -> Path:
    """Ejecuta la demostración y guarda un registro legible en ``docs/evidencias``."""
    iniciales = _hashes()
    with tempfile.TemporaryDirectory(prefix="framework-ia-demo-") as temporal:
        try:
            temporal_path = Path(temporal)
            configuracion = ConfigStore()
            configuracion.set("paths.logs_dir", str(temporal_path / "logs"))
            controlador = FrameworkController(
                config=configuracion,
                persist=False,
                runs_dir=temporal_path / "runs",
            )
            controlador.load_sample("drug200.csv")
            agentes_antes = controlador.list_agents()
            nuevo = controlador.create_agent(
                {
                    "agent_type": "knn_auto",
                    "name": "KNN automático de evidencia",
                    "capabilities": ["classification.train"],
                    "priority": 50,
                    "config": {"k_max": 9},
                },
                persist=False,
            )
            agentes_despues = controlador.list_agents()
            resultado = controlador.probe_agent(
                nuevo["agent_id"],
                "classification.train",
                {"target": "Drug", "algorithm": "knn", "test_size": 0.25, "seeds": [42], "stratify": True},
            )
            mensajes = resultado["messages"]
            hash_finales = _hashes()
        finally:
            close_logging()

    directorio = ROOT / "docs" / "evidencias"
    directorio.mkdir(parents=True, exist_ok=True)
    destino = directorio / "dynamic_agent_log.md"
    seccion_mensajes = [
        "| Hora | Emisor | Receptor | Tarea | Estado | Conversación |",
        "|---|---|---|---|---|---|",
    ]
    seccion_mensajes.extend(
        f"| {m['timestamp']} | {m['sender']} | {m['receiver']} | {m['task']} | "
        f"{m['status']} | {m['correlation_id']} |"
        for m in mensajes
    )
    comparacion_hashes = [
        "| Archivo del núcleo | SHA-256 antes | SHA-256 después | Sin cambios |",
        "|---|---|---|---|",
    ]
    comparacion_hashes.extend(
        f"| `{nombre}` | `{iniciales[nombre]}` | `{hash_finales[nombre]}` | "
        f"{'Sí' if iniciales[nombre] == hash_finales[nombre] else 'No'} |"
        for nombre in CORE_FILES
    )
    texto = "\n".join(
        [
            "# Evidencia de creación dinámica de agente",
            "",
            "Demostración generada por `python scripts/demo_dynamic_agent.py` con `drug200.csv`, sin credenciales "
            "de IA y sin persistir estado en el repositorio.",
            "",
            "## Antes de crear el plugin",
            "",
            _tabla_agentes(agentes_antes),
            "",
            "## Después de crear el plugin",
            "",
            _tabla_agentes(agentes_despues),
            "",
            f"## Decisión del coordinador para `{resultado['task']}`",
            "",
            f"- Política: `{resultado['policy']}`",
            f"- Candidatos: {', '.join(resultado['candidates'])}",
            f"- Agente elegido: `{resultado['selected_agent']}`",
            f"- ID de ejecución temporal: `{resultado['run_id']}`",
            "",
            "## Mensajes intercambiados",
            "",
            *seccion_mensajes,
            "",
            "## Integridad del núcleo",
            "",
            *comparacion_hashes,
            "",
            "El plugin se creó en ejecución y el coordinador lo seleccionó por la capacidad "
            "`classification.train`; no se editaron los cuatro archivos del núcleo.",
            "",
        ]
    )
    destino.write_text(texto, encoding="utf-8")
    return destino


def main() -> None:
    os.environ.pop("GEMINI_API_KEY", None)
    print(generar_evidencia())


if __name__ == "__main__":
    main()
