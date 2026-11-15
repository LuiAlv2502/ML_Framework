# Estudio de Caso 04: Agencia y Sistema Multiagente

**Ejecución:** 2026-11-15T21:14:57+00:00  
**Python:** 3.13.9 · **pandas:** 3.0.5 · **scikit-learn:** 1.9.0  
**Protocolo:** partición estratificada 75/25 · semillas 42, 43, 44 · F1 macro para selección.

## Resumen

Este estudio ejecuta tres escenarios de complejidad creciente sobre conjuntos incluidos localmente en el proyecto. El servicio externo de IA se desactiva; el EDA usa su modo reproducible basado en reglas. El tercer escenario compara una evaluación centralizada con el mismo conjunto de estrategias KNN que atienden los agentes. Los resultados son mediciones de esta ejecución, no promesas de rendimiento generalizable.

## Fundamentación y mapeo de agencia

Un agente es una entidad computacional situada en un entorno que percibe y actúa de forma autónoma para lograr objetivos; un MAS coordina entidades que interactúan para resolver tareas relacionadas [1, 2]. Autonomía, reactividad, iniciativa y habilidad social se interpretan aquí como propiedades operacionales, no como una implementación BDI completa.

| Propiedad o concepto | Implementación observada | Alcance y límite |
|---|---|---|
| Entorno y percepción | Los agentes reciben una tabla y una solicitud estructurada; EDA calcula perfil, nulos, duplicados y tipos. | El entorno es el dataset cargado, el registro y el bus en memoria. |
| Autonomía | Cada agente valida y ejecuta la capacidad que publica sin que el coordinador conozca su clase concreta. | La autonomía está acotada a la tarea delegada; no hay objetivos persistentes. |
| Reactividad | `BaseAgent.handle` responde a solicitudes y transiciones del ciclo de vida. | No hay un bucle sensorial continuo. |
| Proactividad | El coordinador inicia la delegación y la consolidación una vez recibida una solicitud. | Los especialistas no generan tareas espontáneas. |
| Acciones y objetivos | Los agentes ejecutan la capacidad asignada; clasificación maximiza F1 macro y K-Means maximiza silhouette. | El objetivo es local a la solicitud, no persistente. |
| Conocimiento | Cada agente usa su configuración, capacidades, algoritmo y estadísticas operativas. | No hay una ontología ni una base de conocimiento compartida. |
| Habilidad social | Mensajes tipados con emisor, receptor, performativa, tarea y correlación; respuestas registradas por el bus. | Comunicación síncrona local; no es una red distribuida. |
| Estado y memoria de trabajo | Estados de agente, contadores, fallos consecutivos, tareas recientes y registros persistidos por experimento. | La bitácora del agente es acotada; no existe memoria semántica compartida de largo plazo. |
| Coordinación y resolución | Enrutamiento por capacidad; `priority` admite fallback y `fanout` elige por F1 macro. | La selección es arbitraje por métrica, no negociación estratégica. |

El diseño global es **híbrido**: el coordinador aplica reglas de enrutamiento y políticas; el EDA construye un plan de análisis; los agentes de aprendizaje ejecutan modelos estadísticos. No se afirma que sea una arquitectura BDI formal. En una arquitectura reactiva, la acción suele vincularse directamente con la percepción; la deliberativa mantiene razonamiento explícito sobre objetivos; la híbrida combina ambos estilos [1, 2].

### Roles

| Rol | Capacidad usada en los escenarios | Objetivo |
|---|---|---|
| Coordinador | `coordination.route` | Interpretar la tarea, buscar agentes compatibles, delegar y consolidar. |
| Agente de agrupamiento | `clustering.kmeans` | Formar tres grupos iniciales y medir silhouette. |
| Analista EDA | `eda.profile` | Analizar y preparar una copia de los datos. |
| Clasificador | `classification.train` | Entrenar KNN fijo con la partición solicitada. |
| KNN automático | `classification.train` | Evaluar varios valores impares de k y reportar el mejor F1 macro. |
| Bus y registro | Mensajes correlacionados y registro de ejecuciones | Mantener trazabilidad de solicitudes, respuestas y decisiones. |

### Arquitectura (diagrama textual)

```text
Dashboard MAS / script de experimentos
                  |
          FrameworkController
                  | REQUEST(task, payload, correlation_id)
                  v
          Coordinador MAS
       /          |            |           \
ClusteringAgent EDAAgent ClassificationAgent KNNAutoAgent
       \          |            |           /
          MessageBus + Registry
                  |
   Servicios ML / ExperimentStore local
```

### Flujo del escenario cooperativo

```text
Controlador -> Coordinador: solicitud eda
Coordinador -> EDAAgent: perfil del conjunto
EDAAgent -> Coordinador: perfil + copia preparada
Controlador -> Coordinador: classification (copia preparada, semilla)
Coordinador -> ClassificationAgent: KNN fijo
ClassificationAgent -> Controlador: métricas + traza
```

### Flujo de decisión / conflicto

```text
Controlador -> Coordinador: classification, policy=fanout
Coordinador -> ClassificationAgent: KNN k=5
Coordinador -> KNNAutoAgent: KNN k en {3,5,7,9,11} (limitado por tamaño de entrenamiento)
Ambos agentes -> Coordinador: resultado + F1 macro
Coordinador: conserva el mayor F1 macro y registra alternativas y decisión
```

El fan-out del framework actual es síncrono. El diagrama representa mensajes y arbitraje, no ejecución paralela sobre procesos o máquinas distintas.

## Metodología

### Escenarios

1. **Baja complejidad — agrupamiento inicial:** el agente K-Means forma tres grupos sobre dos predictores numéricos; la métrica silhouette describe su cohesión/separación y no usa la variable objetivo.
2. **Complejidad media — cooperación secuencial:** EDA produce una copia preparada que se entrega al clasificador; se repite el entrenamiento con las semillas indicadas.
3. **Alta complejidad — decisión entre alternativas:** `fanout` ejecuta el clasificador KNN fijo y el buscador KNN automático. La línea base centralizada ejecuta esas mismas estrategias directamente, sin coordinador, bus ni agentes.

### Datos y protocolo

| Archivo local | Objetivo | Filas | Columnas totales | Huella SHA-256 |
|---|---|---:|---:|---|
| `drug200.csv` | `Drug` | 200 | 6 | `10caebf3b055e149ef1e7494042031e510b47ba644f3c304bbb86c428fe89b66` |
| `diabetes.csv` | `Outcome` | 768 | 9 | `698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72` |
| `water_potability.csv` | `Potability` | 3276 | 10 | `904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf` |

Las huellas identifican los archivos usados, pero no acreditan su procedencia ni licencia. Los CSV heredados del repositorio no tienen esos metadatos documentados; deben verificarse antes de redistribuirlos. La partición, el escalado y la semilla se mantienen iguales entre los dos enfoques. La selección se hace por F1 macro para ponderar todas las clases.

## Resultados medidos: escenarios

| Conjunto | Escenario | Complejidad | Corridas | Tiempo total (s) | Interacciones | Tareas completadas | Errores | Métrica de calidad (media ± DE) |
|---|---|---|---:|---:|---:|---:|---:|---|
| drug200.csv | Agrupamiento inicial | Baja | 1 | 1.9719 | 4 | 1 | 0 | silhouette: 0.4205 ± 0.0000 |
| drug200.csv | Preparación EDA y clasificación | Media | 3 | 3.4130 | 16 | 4 | 0 | f1_macro: 0.7371 ± 0.1247 |
| drug200.csv | Selección y resolución de alternativas | Alta | 3 | 1.4994 | 18 | 6 | 0 | f1_macro: 0.8287 ± 0.0688 |
| diabetes.csv | Agrupamiento inicial | Baja | 1 | 0.1795 | 4 | 1 | 0 | silhouette: 0.4179 ± 0.0000 |
| diabetes.csv | Preparación EDA y clasificación | Media | 3 | 4.9097 | 16 | 4 | 0 | f1_macro: 0.6998 ± 0.0061 |
| diabetes.csv | Selección y resolución de alternativas | Alta | 3 | 1.0483 | 18 | 6 | 0 | f1_macro: 0.7025 ± 0.0244 |
| water_potability.csv | Agrupamiento inicial | Baja | 1 | 0.4845 | 4 | 1 | 0 | silhouette: 0.3219 ± 0.0000 |
| water_potability.csv | Preparación EDA y clasificación | Media | 3 | 5.5738 | 16 | 4 | 0 | f1_macro: 0.5945 ± 0.0332 |
| water_potability.csv | Selección y resolución de alternativas | Alta | 3 | 1.7298 | 18 | 6 | 0 | f1_macro: 0.5985 ± 0.0262 |

## Comparación centralizada y MAS

| Conjunto | Semilla | Enfoque | Tiempo (s) | Interacciones | Tareas completadas | Modelos evaluados | Errores | F1 macro | Selección |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| drug200.csv | 42 | Centralizado | 0.4511 | 0 | 2 | 6 | 0 | 0.8211 | KNN automático |
| drug200.csv | 42 | MAS | 0.4691 | 6 | 2 | 6 | 0 | 0.8211 | knn_auto (k=3) |
| drug200.csv | 43 | Centralizado | 0.3823 | 0 | 2 | 6 | 0 | 0.7639 | KNN automático |
| drug200.csv | 43 | MAS | 0.5301 | 6 | 2 | 6 | 0 | 0.7639 | knn_auto (k=11) |
| drug200.csv | 44 | Centralizado | 0.3722 | 0 | 2 | 6 | 0 | 0.9010 | KNN automático |
| drug200.csv | 44 | MAS | 0.5002 | 6 | 2 | 6 | 0 | 0.9010 | knn_auto (k=3) |
| diabetes.csv | 42 | Centralizado | 0.3437 | 0 | 2 | 6 | 0 | 0.7141 | KNN automático |
| diabetes.csv | 42 | MAS | 0.3403 | 6 | 2 | 6 | 0 | 0.7141 | knn_auto (k=7) |
| diabetes.csv | 43 | Centralizado | 0.3258 | 0 | 2 | 6 | 0 | 0.7189 | KNN automático |
| diabetes.csv | 43 | MAS | 0.3890 | 6 | 2 | 6 | 0 | 0.7189 | knn_auto (k=3) |
| diabetes.csv | 44 | Centralizado | 0.3209 | 0 | 2 | 6 | 0 | 0.6744 | KNN fijo (k=5) |
| diabetes.csv | 44 | MAS | 0.3190 | 6 | 2 | 6 | 0 | 0.6744 | knn_auto (k=5) |
| water_potability.csv | 42 | Centralizado | 0.5257 | 0 | 2 | 6 | 0 | 0.5685 | KNN automático |
| water_potability.csv | 42 | MAS | 0.5844 | 6 | 2 | 6 | 0 | 0.5685 | knn_auto (k=3) |
| water_potability.csv | 43 | Centralizado | 0.5420 | 0 | 2 | 6 | 0 | 0.6164 | KNN fijo (k=5) |
| water_potability.csv | 43 | MAS | 0.5335 | 6 | 2 | 6 | 0 | 0.6164 | knn_auto (k=5) |
| water_potability.csv | 44 | Centralizado | 0.5869 | 0 | 2 | 6 | 0 | 0.6107 | KNN fijo (k=5) |
| water_potability.csv | 44 | MAS | 0.6119 | 6 | 2 | 6 | 0 | 0.6107 | knn_auto (k=5) |

### Análisis agregado de la comparación

| Conjunto | Tiempo central (media ± DE) | Tiempo MAS (media ± DE) | Cambio de tiempo MAS (%) | Diferencia F1 macro | Interacciones MAS / prueba | Errores MAS |
|---|---:|---:|---:|---:|---:|---:|
| drug200.csv | 0.4019 ± 0.0429 | 0.4998 ± 0.0305 | 24.37% | 0.0000 | 6.00 | 0 |
| diabetes.csv | 0.3301 ± 0.0120 | 0.3494 ± 0.0359 | 5.85% | 0.0000 | 6.00 | 0 |
| water_potability.csv | 0.5515 ± 0.0317 | 0.5766 ± 0.0398 | 4.54% | 0.0000 | 6.00 | 0 |

En esta corrida, la calidad coincide en cada par de conjunto/semilla: sí.
Con el mismo conjunto de candidatos, partición y métrica, el valor de F1 debe coincidir; el propósito de esta comparación es observar el costo de coordinación y las interacciones. El porcentaje de tiempo puede oscilar con el equipo y con la carga del proceso; solo se repitieron tres semillas.

La versión centralizada y el MAS entrenan el mismo número de modelos KNN por semilla. El MAS añade enrutamiento, mensajes, consolidación y persistencia; por ello su tiempo de pared puede ser mayor. Las métricas de calidad se comparan directamente solo dentro del mismo conjunto y semilla. La desviación estándar describe variación entre semillas y no es un intervalo de confianza.

## Discusión crítica

### Fortalezas

- Separación entre la interfaz/controlador y agentes especializados; el coordinador enruta por capacidad.
- Cooperación observable mediante una transferencia explícita del resultado EDA a clasificación.
- Resolución explicable por una métrica primaria, con agentes candidatos, alternativas y decisión guardados.
- La misma partición y los mismos parámetros permiten contrastar calidad y sobrecosto operacional entre enfoques.
- El registro de mensajes, intentos y experimentos facilita repetir y auditar cada ejecución.

### Limitaciones y mejoras

- El flujo es local y síncrono; no mide latencia de red, concurrencia ni tolerancia a fallos de infraestructura.
- El conflicto se resuelve por máximo F1 macro, no por negociación, utilidad múltiple o preferencias del usuario.
- Las conclusiones dependen de los conjuntos disponibles, de sus licencias aún no confirmadas y de la selección de KNN.
- Para una evaluación más fuerte: añadir particiones externas, pruebas estadísticas pareadas, más algoritmos, inyección controlada de fallos y medir disponibilidad, consumo de memoria y escalabilidad.
- Las pruebas automáticas del proyecto cubren fallback, reintentos, errores del usuario, estados, mensajes y creación de plugins; este experimento no introduce fallos artificiales en los resultados de calidad.

## Reproducibilidad

Desde la raíz del repositorio, con dependencias instaladas:

```powershell
python scripts/run_case04_experiments.py
```

La ejecución desactiva el LLM y regenera este informe y los CSV/JSON en `docs/resultados/caso04/`. Para usar otras semillas: `python scripts/run_case04_experiments.py --seeds 11 42 73`.

## Bibliografía

1. Wooldridge, M. y Jennings, N. R. (1995). Intelligent agents: Theory and practice. *The Knowledge Engineering Review*, 10(2), 115–152. https://doi.org/10.1017/S0269888900008122
2. Jennings, N. R., Sycara, K. y Wooldridge, M. (1998). A roadmap of agent research and development. *Autonomous Agents and Multi-Agent Systems*, 1, 7–38. https://doi.org/10.1023/A:1010090405266
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
5. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7
