# Conjuntos de datos

Los archivos CSV ya presentes en esta carpeta llegaron al repositorio sin metadatos de procedencia ni licencia.
No se debe asumir que pueden redistribuirse fuera de este proyecto. Antes de compartirlos, confirma su origen y
las condiciones de uso con quien los proporcionó.

| Archivo | Uso sugerido | Procedencia/licencia |
|---|---|---|
| `diabetes.csv` | Clasificación (`Outcome`) | No registrada en el repositorio; verificar antes de redistribuir. |
| `drug200.csv` | Clasificación (`Drug`) | No registrada en el repositorio; verificar antes de redistribuir. |
| `Mall_Customers.csv` | Agrupamiento | No registrada en el repositorio; verificar antes de redistribuir. |
| `water_potability.csv` | Clasificación (`Potability`) | No registrada en el repositorio; verificar antes de redistribuir. |
| `regression_sample.csv` | Regresión (`ingreso_anual`) | Sintético, generado por `scripts/generate_regression_sample.py`; no procede de personas ni de una fuente externa. |

Para regenerar el conjunto sintético de regresión:

```powershell
python scripts/generate_regression_sample.py
```

El generador produce las mismas 300 filas con la semilla predeterminada. Se pueden cambiar el tamaño y la semilla:

```powershell
python scripts/generate_regression_sample.py --rows 500 --seed 7
```

Para cargar un archivo propio, utiliza **Datos** en el dashboard. Los CSV y archivos Excel se leen localmente.
