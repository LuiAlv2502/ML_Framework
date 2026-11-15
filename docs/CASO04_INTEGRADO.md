# Estudio de Caso 04 integrado

El Caso 04 se ejecuta en el proyecto existente: la interfaz está dentro de **MAS > Evaluación** y el
protocolo completo desde la raíz está en:

```powershell
python scripts/run_case04_experiments.py
```

El comando usa agentes predeterminados en una agencia limpia, desactiva el LLM y ejecuta tres conjuntos
locales (`drug200.csv`, `diabetes.csv` y `water_potability.csv`) con las semillas 42, 43 y 44. Para cambiar
las semillas:

```powershell
python scripts/run_case04_experiments.py --seeds 11 42 73
```

## Escenarios de complejidad creciente

1. **Baja — agrupamiento inicial:** el agente de clustering aplica K-Means sobre dos predictores numéricos y
   reporta silhouette.
2. **Media — cooperación secuencial:** EDA prepara una copia y la entrega al agente clasificador; la
   clasificación se repite por semilla.
3. **Alta — resolución de alternativas:** en una agencia temporal y limpia, el coordinador aplica `fanout`
   al clasificador KNN fijo y al plugin `knn_auto`; escoge el resultado de mayor F1 macro. Un flujo
   centralizado evalúa las mismas alternativas sobre la misma partición y semilla, sin coordinador, bus ni
   agentes. La agencia aislada evita que plugins que el usuario haya creado alteren la comparación.

Se miden duración, interacciones, tareas completadas, errores, modelos evaluados, silhouette y F1 macro
cuando aplica. Los mensajes resumidos, candidatos, decisiones, semillas, configuración, entorno y huellas
SHA-256 se incluyen en los resultados; las ejecuciones del dashboard, además, aparecen en **Experimentos**.

## Archivos de salida

El comando regenera, sin imágenes:

- `docs/resultados/caso04/INFORME_CASO04.md`: fundamentos, arquitectura textual, metodología, resultados
  medidos, discusión crítica, limitaciones, mejoras y bibliografía.
- `docs/resultados/caso04/resultados.json`: manifiesto y trazas detalladas por conjunto/semilla.
- `docs/resultados/caso04/resultados_escenarios.csv`: métricas agregadas por escenario.
- `docs/resultados/caso04/comparacion_central_mas.csv`: comparación por semilla y enfoque.

## Interpretación y límites

La comparación de calidad solo es válida dentro del mismo conjunto y semilla. El fan-out actual es
síncrono, así que el tiempo refleja coordinación local y persistencia, no comunicación distribuida ni
paralelismo. La decisión es arbitraje por F1 macro, no negociación estratégica. Las pruebas del proyecto
cubren reintentos, fallback y errores; no se inyectan fallos en las mediciones de calidad.

Los conjuntos heredados no tienen procedencia o licencias documentadas; los hashes permiten identificar
los archivos usados, pero no autorizan su redistribución. El informe generado deja explícito ese límite.
