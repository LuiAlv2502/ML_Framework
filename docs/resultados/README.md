# Experimentos reproducibles

Los resultados se generan desde la raíz del proyecto:

```powershell
python scripts/run_experiments.py
```

El script desactiva el proveedor de IA y guarda ejecuciones y logs temporales fuera del repositorio. Reemplaza los
CSV, el manifiesto y las cuatro imágenes de esta carpeta.

## Protocolo

- **Clasificación:** `drug200.csv`, objetivo `Drug`, benchmarks del catálogo y partición estratificada 75/25 con
  semilla 42. El Random Forest de 200 árboles se evalúa además con semillas 42, 43 y 44.
- **Regresión:** conjunto sintético de 300 filas, objetivo `ingreso_anual`, benchmark 75/25 con semilla 42.
  RidgeCV se evalúa además con semillas 42, 43 y 44 y alphas `[0.01, 0.1, 1, 10, 100]`.
- **Agrupamiento:** `Mall_Customers.csv`, solo `Annual Income (k$)` y `Spending Score (1-100)`. Se comparan
  K-Means para k=2, 3 y 4, HAC Ward para k=3 y dos configuraciones de PCA.

Las comparaciones de clasificación y regresión usan una única partición idéntica por tarea, de forma que cada
configuración del benchmark se compare en las mismas condiciones. La evaluación con tres semillas es un resultado
separado; sus valores permiten observar el efecto de cambiar la partición. En agrupamiento se informa silhouette,
Davies-Bouldin y Calinski-Harabasz cuando aplica. Ninguna métrica permite comparar directamente tareas o datasets
distintos.

## Archivos generados

- `benchmark_clasificacion.csv`, `benchmark_regresion.csv` y `benchmark_agrupamiento.csv`: tablas completas de
  configuraciones y métricas.
- `resultados.json`: protocolo, versión de Python, semillas, hashes SHA-256, conteos y resultados seleccionados.
- `resultados_tabla.tex`: tabla breve incluida por [el informe técnico](../technical_report.tex).
- `figures/01_matriz_confusion.png`: clasificación con Random Forest, semilla 42.
- `figures/02_regresion_real_predicho.png`: RidgeCV, semilla 42.
- `figures/03_segmentacion_clientes.png`: K-Means k=3 con proyección PCA.
- `figures/04_comparacion_benchmarks.png`: clasificación, regresión y agrupamiento.

`drug200.csv` y `Mall_Customers.csv` llegaron sin procedencia ni licencia documentadas; los hashes registran los
archivos empleados, no verifican sus derechos de uso. `regression_sample.csv` es sintético y se puede regenerar con
`python scripts/generate_regression_sample.py`.

Para compilar el PDF incluido en el informe se necesita MiKTeX o una distribución compatible con `pdflatex` y
`bibtex`; ejecute `.\scripts\build_report.ps1` desde la raíz del proyecto.

## Estudio de Caso 04

El protocolo de agencia y sistemas multiagentes se ejecuta aparte, sin alterar los benchmarks existentes:

```powershell
python scripts/run_case04_experiments.py
```

Genera el informe textual, un manifiesto con trazas y hashes, una tabla de métricas por escenario y una
comparación centralizada/MAS por semilla en `docs/resultados/caso04/`. La ejecución predeterminada usa
`drug200.csv`, `diabetes.csv` y `water_potability.csv` con semillas 42, 43 y 44. Puede cambiar las semillas
con `--seeds 11 42 73`.
