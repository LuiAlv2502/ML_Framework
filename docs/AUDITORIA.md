# Auditoría de entrega

## Alcance y validación

Esta auditoría describe el estado de la aplicación y sus entregables después de completar los bloques A–E. La
validación automatizada más reciente registró **225 pruebas aprobadas** y una advertencia conocida de UMAP
relacionada con la configuración de `random_state` y `n_jobs`. La cobertura total de `framework_ia` fue **83 %**;
agentes: **88.6 %**, servicios: **91.0 %**, persistencia: **92.4 %**. `pip check` no encontró dependencias
incompatibles.

El entorno validado fue Windows con Python 3.13.7. `pyproject.toml` declara Python 3.11 como mínimo, pero en esta
entrega no se ejecutó una matriz de pruebas para todas las versiones admitidas.

## Estado de requisitos

| Área | Estado | Evidencia |
|---|---|---|
| Organización en capas y controlador del backend | Completo | `framework_ia/`, `frontend/` y pruebas de integración. |
| Navegación de la aplicación | Completo | Nueve páginas: Inicio, Datos, EDA, Preprocesamiento, Agrupamiento, Clasificación, Regresión, MAS y Experimentos. |
| Clasificación, regresión y agrupamiento | Completo | Algoritmos, métricas y benchmarks conectados al controlador; parámetros verificables desde la aplicación. |
| Sistema multiagente | Completo | Coordinador, agentes especializados, creación de plugins, mensajes y registro de decisiones. |
| Pruebas y cobertura | Completo | `docs/evidencias/pytest_output.txt` y `docs/evidencias/coverage.txt`. |
| Experimentos reproducibles | Completo | Script, CSV de resultados, manifiesto con semillas y hashes, y cuatro figuras en `docs/resultados/`. |
| Informe técnico | Completo | Fuente LaTeX, bibliografía y PDF compilado de seis páginas en `docs/technical_report.pdf`. |
| Guía y dependencias | Completo | README actualizado; versiones de ejecución fijadas en `requirements.txt` y herramientas de desarrollo separadas. |
| Capturas de la interfaz | Pendiente manual | `docs/evidencias/CAPTURAS.md` describe las capturas requeridas; no se incluyen imágenes simuladas. |
| Procedencia y licencias de CSV heredados | Pendiente de confirmación | Los metadatos de origen no están disponibles en el repositorio. |
| Paquete ZIP | Excluido por solicitud | No se generará ni validará un ZIP de entrega. |

## Algoritmos y parámetros

- **Clasificación:** KNN, árbol de decisión, Random Forest, Gradient Boosting, AdaBoost y XGBoost. El catálogo
  permite configurar los parámetros admitidos y comparar modelos con métricas por clase y F1 macro.
- **Regresión:** modelos lineales simples y múltiples, Lasso/LassoCV, Ridge/RidgeCV, SVR, árbol y Random Forest.
  RidgeCV recibe la lista de alphas configurada y se verifica en `tests/test_hyperparameter_wiring.py`.
- **Agrupamiento y reducción:** K-Means, HAC, PCA, t-SNE y UMAP. Las pruebas verifican la propagación de
  `init`/`n_init` en K-Means y `whiten`/`svd_solver` en PCA.
- **Evaluación reproducible:** `scripts/run_experiments.py` compara configuraciones con particiones y semillas
  declaradas; los resultados no sustituyen una validación con datos representativos del uso real.

## Datos y limitaciones

`data/regression_sample.csv` se genera sintéticamente y puede regenerarse con `scripts/generate_regression_sample.py`.
Los demás CSV no incluyen procedencia ni licencia verificadas; no se les atribuye un origen ni se recomienda
redistribuirlos hasta confirmar sus condiciones de uso.

El proveedor Gemini es opcional: las pruebas y los experimentos reproducibles funcionan sin credenciales. Si se
habilita, se debe proporcionar la clave en una variable de entorno y mantenerla fuera del repositorio.
