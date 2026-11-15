# Evidencia de creación dinámica de agente

Demostración generada por `python scripts/demo_dynamic_agent.py` con `drug200.csv`, sin credenciales de IA y sin persistir estado en el repositorio.

## Antes de crear el plugin

| ID | Nombre | Tipo | Capacidades |
|---|---|---|---|
| coordinador | Coordinador | coordinator | coordination.route |
| agente-eda | Agente EDA | eda | eda.clean, eda.correlations, eda.preprocess, eda.profile, eda.summary |
| agente-agrupamiento | Agente de agrupamiento | clustering | clustering.auto, clustering.benchmark, clustering.dendrogram, clustering.elbow, clustering.hac, clustering.kmeans, clustering.pca, clustering.tsne, clustering.umap |
| agente-clasificacion | Agente de clasificación | classification | classification.auto, classification.benchmark, classification.train |
| agente-regresion | Agente de regresión | regression | regression.auto, regression.benchmark, regression.train |

## Después de crear el plugin

| ID | Nombre | Tipo | Capacidades |
|---|---|---|---|
| coordinador | Coordinador | coordinator | coordination.route |
| agente-eda | Agente EDA | eda | eda.clean, eda.correlations, eda.preprocess, eda.profile, eda.summary |
| agente-agrupamiento | Agente de agrupamiento | clustering | clustering.auto, clustering.benchmark, clustering.dendrogram, clustering.elbow, clustering.hac, clustering.kmeans, clustering.pca, clustering.tsne, clustering.umap |
| agente-clasificacion | Agente de clasificación | classification | classification.auto, classification.benchmark, classification.train |
| agente-regresion | Agente de regresión | regression | regression.auto, regression.benchmark, regression.train |
| knn_auto-c2a315 | KNN automático de evidencia | knn_auto | classification.train |

## Decisión del coordinador para `classification.train`

- Política: `priority`
- Candidatos: knn_auto-c2a315, agente-clasificacion
- Agente elegido: `knn_auto-c2a315`
- ID de ejecución temporal: `20261115-132417_classification-train_0c4dc86e`

## Mensajes intercambiados

| Hora | Emisor | Receptor | Tarea | Estado | Conversación |
|---|---|---|---|---|---|
| 2026-11-15T13:24:16.687 | controller | coordinador | classification.train | processed | 2014987b5ad3 |
| 2026-11-15T13:24:16.687 | coordinador | knn_auto-c2a315 | classification.train | processed | 2014987b5ad3 |
| 2026-11-15T13:24:17.026 | knn_auto-c2a315 | coordinador | classification.train | delivered | 2014987b5ad3 |
| 2026-11-15T13:24:17.026 | coordinador | controller | classification.train | delivered | 2014987b5ad3 |

## Integridad del núcleo

| Archivo del núcleo | SHA-256 antes | SHA-256 después | Sin cambios |
|---|---|---|---|
| `framework_ia/agents/coordinator_agent.py` | `f4aff48c60d258408659c46392f02755906731f41e16b713b144d2d20ca32241` | `f4aff48c60d258408659c46392f02755906731f41e16b713b144d2d20ca32241` | Sí |
| `framework_ia/agents/agency.py` | `064f3c2be995c17a7239b89be1620c90d7b8035fce49e7117ea4596e14d280d8` | `064f3c2be995c17a7239b89be1620c90d7b8035fce49e7117ea4596e14d280d8` | Sí |
| `framework_ia/agents/registry.py` | `44e4c2dc27b35ca80c8789d9981ad2d2aa75c324a9d37d682549b2c3de3b78dd` | `44e4c2dc27b35ca80c8789d9981ad2d2aa75c324a9d37d682549b2c3de3b78dd` | Sí |
| `framework_ia/agents/agent_factory.py` | `ee8b7012969ae532edfdc2f6fb8e8ce567c952ca0b3b2f2cb88ffc78315b33d3` | `ee8b7012969ae532edfdc2f6fb8e8ce567c952ca0b3b2f2cb88ffc78315b33d3` | Sí |

El plugin se creó en ejecución y el coordinador lo seleccionó por la capacidad `classification.train`; no se editaron los cuatro archivos del núcleo.
