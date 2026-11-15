"""Plugin de ejemplo: un clasificador que solo usa KNN y prueba varios ``k`` por su cuenta.

Este archivo demuestra cómo incorporar un tipo de agente **sin modificar el núcleo**: al iniciar, la
fábrica importa todos los módulos de ``framework_ia/agents/plugins/`` y el decorador ``@agent_type``
registra el tipo. Después puede crearse desde el panel MAS o con
``agency.create_agent({"agent_type": "knn_auto", "name": "KNN automático"})``.
"""
from __future__ import annotations

from typing import Any

from framework_ia.agents import _support
from framework_ia.agents.agent_factory import agent_type
from framework_ia.agents.base_agent import BaseAgent
from framework_ia.agents.messaging import Message
from framework_ia.ml.classification import ClasificacionModelos
from framework_ia.utils.validation import validate_target


@agent_type("knn_auto", description="Plugin de ejemplo: KNN que elige el mejor k (3, 5, 7, 9, 11) por F1 macro.")
class KNNAutoAgent(BaseAgent):
    """Publica ``classification.train``; ignora el algoritmo pedido y busca el mejor ``k`` de KNN."""

    default_capabilities = frozenset({"classification.train"})
    supported_capabilities = frozenset({"classification.train"})
    config_schema = {"k_max": ("entero", 3, 25)}

    def _execute(self, message: Message) -> dict[str, Any]:
        datos = self.require_data(message)
        objetivo = validate_target(datos, message.payload.get("target"), task="classification")
        semilla = _support.seeds(message.payload)[0]
        modelo = _support.ml_call(
            ClasificacionModelos, datos, target=objetivo, random_state=semilla,
            test_size=float(message.payload.get("test_size", 0.25)),
            stratify=bool(message.payload.get("stratify", True)),
            feature_columns=_support.features(datos, message.payload, objetivo),
        )
        mejores, mejor_k = None, None
        for k in range(3, int(self.config.get("k_max", 11)) + 1, 2):
            resultado = _support.ml_call(modelo.knn, n_neighbors=k)
            if mejores is None or resultado["metricas"]["f1_macro"] > mejores["metricas"]["f1_macro"]:
                mejores, mejor_k = resultado, k
        metricas = mejores["metricas"]
        resumen = {m: metricas[m] for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}
        return {
            "task": "classification.train",
            "algorithm": "knn",
            "algorithm_name": "K vecinos más cercanos (k automático)",
            "label": f"knn_auto (k={mejor_k})",
            "params": {"n_neighbors": mejor_k},
            "target": objetivo,
            "metrics": resumen,
            "confusion_matrix": {"labels": metricas["confusion_matrix_labels"], "matrix": metricas["confusion_matrix"]},
            "primary_metric": _support.primary("f1_macro", resumen["f1_macro"], True),
            "agent": self.agent_id,
        }
