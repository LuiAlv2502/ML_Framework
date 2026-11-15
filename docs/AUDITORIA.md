# Auditoría inicial del repositorio (Fase 0)

Fecha de la auditoría: 2026-11-15. Línea base: `python -m pytest` → **61 pruebas pasan** (1 advertencia de UMAP
sobre `n_jobs`). Entorno: Windows, Python 3.13.7, `.venv` local (fue necesario instalar `pytest`, que figuraba en
`requirements.txt` pero no estaba instalado en el entorno).

Estados: **Completo** (cumple el requisito tal como lo pide el enunciado), **Parcial** (existe, pero falta una parte
exigida), **Falta** (no existe).

## 1. Módulos del enunciado

| # | Requisito | Archivo / clase que lo cubre | Estado | Observaciones |
|---|-----------|------------------------------|--------|---------------|
| 1.1 | Funcionar como framework (no notebooks sueltos) | `src/` (`EDA`, `supervised`, `unsupervised`, `agents`) | Parcial | Código reutilizable, pero no es un paquete instalable: usa `sys.path.append` y `from supervised import ...`. |
| 1.2 | Estructura de paquetes pedida (`agents/`, `ml/`, `services/`, `persistence/`, `utils/`, `frontend/`, `models/`) | — | Falta | Código plano en `src/`. |
| 1.3 | Modularidad y reutilización | `SupervisadoBase`, `UnsupervisedBase`, `DataFrame`, `AgenteSupervisado` | Parcial | Buena herencia en ML; los agentes EDA y No Supervisado no comparten interfaz con los supervisados. |
| 1.4 | Manejo de errores y validación de entradas | Validaciones dispersas (`ValueError`, `KeyError`) en clases y páginas | Parcial | No hay excepciones propias ni un módulo de validación; algunas páginas muestran el mensaje crudo de la excepción. |
| 2.1 | EDA (perfil, nulos, duplicados, atípicos, distribuciones, correlaciones) | `src/EDA/dataframe.py` (`DataFrame`), `src/agents/eda_agent.py` (`AgenteEDA`), `pages/eda.py` | Completo | Agente con ciclo percibir → planificar → actuar e informe HTML. |
| 2.2 | Preprocesamiento (imputación, codificación, escalado, selección de variables) | `DataFrame.imputar_nulos/normalizar/estandarizar/seleccionar_variables`; `SupervisadoBase._build_preprocessor` | Parcial | Existe, pero no hay un panel de preprocesamiento separado del EDA ni configuración explícita de variables (tipos, exclusión de IDs, objetivo). |
| 3.1 | Agrupamiento: PCA, HAC, K-Means (+ t-SNE, UMAP) | `src/unsupervised/pca.py` (`PCAAnalysis`), `clustering.py` (`ClusteringAnalysis`) | Completo | Métricas: varianza acumulada, silhouette, inercia, cofenética, trustworthiness. |
| 3.2 | Clasificación: KNN, DT, RF, XGBoost, AdaBoost | `src/supervised/Classification/Classification.py` (`ClasificacionModelos`) | Completo | Además Gradient Boosting y AdaBoost con GridSearchCV. |
| 3.3 | Regresión: lineal simple y múltiple, Lasso, LassoCV, Ridge, RidgeCV, SVR, DT, RF | `src/supervised/Regression/Regression.py` (`RegresionModelos`) | Completo | Métricas RMSE, MAE, ER, R². |
| 3.4 | Benchmarking y comparación de modelos | `ClasificacionModelos.comparar_basico`, `RegresionModelos.comparar_familias/mejor_por_familia` | Parcial | No hay una operación uniforme para los tres dominios; no existe comparación formal de configuraciones de agrupamiento. |
| 4.1 | Agente Coordinador | — | Falta | |
| 4.2 | Agentes especializados (EDA, agrupamiento, clasificación, regresión) | `AgenteEDA`, `AgenteNoSupervisado`, `AgenteClasificacion`, `AgenteRegresion` | Parcial | Existen y funcionan (con LLM opcional y respaldo por reglas), pero no comparten una clase base común ni publican capacidades. |
| 4.3 | Comunidad de agentes / registro con capacidades | — | Falta | |
| 4.4 | Creación dinámica de agentes (fábrica) | — | Falta | |
| 4.5 | Mensajería entre agentes | — | Falta | |
| 4.6 | Agencia (ciclo de vida, asignación, cooperación, fallos) | — | Falta | |
| 4.7 | Incorporar un agente nuevo sin modificar la lógica central | — | Falta | |
| 5.1 | Frontend en Streamlit | `app.py`, `pages/*.py`, tema en `pages/_theme.py` y `.streamlit/config.toml` | Parcial | Cuatro páginas (EDA, No supervisado, Clasificación, Predicción). |
| 5.2 | Algoritmos de ML en el backend, no en las páginas | — | Falta | `pages/classification.py` instancia `ClasificacionModelos`; `pages/prediction.py` instancia `RegresionModelos`; `pages/unsupervised.py` importa `sklearn` (KMeans y PCA) dentro de la página; `eda.py` y `unsupervised.py` usan `DataModel` directamente. |
| 5.3 | Páginas de Inicio, Datos, Preprocesamiento, MAS, Experimentos | — | Falta | La carga de datos es solo un `file_uploader` en la barra lateral. |
| 5.4 | Panel MAS y formulario para crear agentes | — | Falta | |
| 6.1 | Pruebas automatizadas | `tests/` (6 archivos, 61 pruebas) | Parcial | No hay pruebas del MAS, de integración extremo a extremo ni de arquitectura. |
| 6.2 | Documentación | `README.md`, `Lab1_Documentation.pdf`, `Lab_2_Classification.pdf` | Parcial | README desactualizado (afirma que la regresión no existe; la estructura descrita no coincide). |

## 2. Entregables

| # | Entregable | Estado | Observaciones |
|---|-----------|--------|---------------|
| E1 | Código fuente del framework organizado por capas | Parcial | Ver 1.1–1.2. |
| E2 | Dashboard Streamlit funcional | Parcial | Ver 5.1–5.4. |
| E3 | Sistema multiagente con coordinador, registro, fábrica, agencia y mensajería | Falta | Ver sección 4. |
| E4 | Benchmarking de agrupamiento, clasificación y regresión | Parcial | Ver 3.4. |
| E5 | Documento técnico en LaTeX | Falta | |
| E6 | Diagramas (arquitectura, clases/componentes, secuencia) | Falta | |
| E7 | README con instalación, ejecución y uso | Parcial | Desactualizado. |
| E8 | `requirements.txt` reproducible | Parcial | Sin versiones fijadas; incluye `ipywidgets`/`ipykernel` (solo para notebooks) y `pytest` mezclado con dependencias de ejecución. |
| E9 | Datasets e instrucciones de obtención | Parcial | 4 CSV en `data/` sin documentar origen ni licencia; no hay un dataset claramente de regresión. |
| E10 | Evidencias de pruebas y de creación dinámica de agentes | Falta | |

## 3. Algoritmos requeridos: configuración estándar y variación desde la UI

"Estándar" = se puede ejecutar con sus valores por defecto; "Variación" = la UI permite cambiar hiperparámetros.

| Algoritmo | Implementación | Estándar en UI | Variación en UI | Estado |
|-----------|----------------|----------------|-----------------|--------|
| KNN | `ClasificacionModelos.knn` | Sí | Sí (`n_neighbors`, `algorithm`) | Completo |
| Árbol de decisión (clasificación) | `ClasificacionModelos.decision_tree` | Sí | Sí (`max_depth`, `min_samples_split`) | Completo |
| Random Forest (clasificación) | `ClasificacionModelos.random_forest` | Sí | Sí (`n_estimators`, `max_depth`, `min_samples_split`) | Completo |
| XGBoost | `ClasificacionModelos.xgboost` | Sí | Sí (`n_estimators`, `max_depth`, `learning_rate`) | Completo |
| AdaBoost | `ClasificacionModelos.adaboost` / `adaboost_grid` | Sí | Sí (`n_estimators`; cuadrícula `n_estimators` × `learning_rate`) | Completo |
| Regresión lineal simple | `RegresionModelos.regresion_lineal_simple` | Sí | Sí (variable predictora) | Completo |
| Regresión lineal múltiple | `RegresionModelos.regresion_lineal_multiple` | Sí | No aplica (sin hiperparámetros) | Completo |
| Lasso | `RegresionModelos.lasso` | Sí | Sí (`alpha`) | Completo |
| LassoCV | `RegresionModelos.lasso_cv` | Sí | Sí (`cv`) | Completo |
| Ridge | `RegresionModelos.ridge` | Sí | Sí (`alpha`) | Completo |
| RidgeCV | `RegresionModelos.ridge_cv` | Sí | No (alphas fijos) | Parcial |
| SVR | `RegresionModelos.svr` | Sí | Sí (`kernel`, `C`, `epsilon`, `degree`) | Completo |
| DecisionTreeRegressor | `RegresionModelos.decision_tree` | Sí | Sí (`max_depth`, `min_samples_split`) | Completo |
| RandomForestRegressor | `RegresionModelos.random_forest` | Sí | Sí (`n_estimators`, `max_depth`, `min_samples_split`) | Completo |
| PCA | `PCAAnalysis.pca` | Sí (2 o 3 componentes) | Parcial: la variante `whiten=True`/`svd_solver` solo existe en el MVC de notebooks | Parcial |
| HAC | `ClusteringAnalysis.hac` | Sí | Sí (`n_clusters`, enlace) | Completo |
| K-Means | `ClusteringAnalysis.kmeans` | Sí | Parcial: solo `n_clusters` (no `init` ni `n_init`) | Parcial |
| t-SNE / UMAP (adicionales) | `ClusteringAnalysis.tsne` / `umap_embedding` | Sí | Sí (`perplexity`; `n_neighbors`, `min_dist`) | Completo |

## 4. Higiene del repositorio

| Problema | Acción en Fase 0 |
|----------|------------------|
| `.gitignore` con solo 2 líneas | Reescrito con entornos, cachés, IDE, cobertura, `.env`, `runs/`, logs e informes generados. |
| `.idea/`, `.coverage` y `notebooks/.ipynb_checkpoints/` versionados | Retirados del índice con `git rm -r --cached`. |
| `.venv/` (~990 MB) en la carpeta del proyecto | Ignorado; el script de entrega (Fase 7) lo excluirá del comprimido. |
| Sin plantilla de variables de entorno | Creado `.env.example` (`GEMINI_API_KEY`, `GEMINI_MODEL`). |
