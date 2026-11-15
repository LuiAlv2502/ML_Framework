# Laboratorio IA

Aplicación local para explorar datos, prepararlos y probar análisis de agrupamiento, clasificación y regresión.
La interfaz está en español y no requiere una clave de IA para cargar datos ni ejecutar los análisis principales.

## Requisitos

- Python 3.11 o posterior. El entorno validado para esta entrega fue Python 3.13.7 en Windows.
- Una instalación de las dependencias indicadas en `requirements.txt`.
- Para compilar el informe PDF: MiKTeX u otra distribución con `pdflatex` y `bibtex`.

## Instalación y ejecución en Windows

En PowerShell, desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La aplicación se abrirá en el navegador; **Inicio** es la página inicial. Para detenerla, presiona `Ctrl+C` en la
terminal. Para ejecutar las pruebas, instala también las herramientas de desarrollo:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Recorrido por la aplicación

Primero carga un archivo CSV o Excel en la barra lateral, o selecciona uno de los archivos de ejemplo. El original
se conserva durante la sesión y puedes restaurarlo después de hacer cambios.

| Página | Para qué sirve |
|---|---|
| **Inicio** | Ver el estado de la aplicación, los datos cargados y la actividad reciente. |
| **Datos** | Revisar dimensiones, columnas, tipos y una vista previa del conjunto. |
| **EDA** | Explorar distribuciones, relaciones y valores ausentes; obtener un resumen automático. |
| **Preprocesamiento** | Aplicar y ordenar cambios como completar valores, escalar o transformar columnas. |
| **Agrupamiento** | Reducir dimensiones o buscar grupos con K-Means y HAC; comparar métricas. |
| **Clasificación** | Elegir la respuesta categórica, entrenar modelos y consultar métricas y errores por clase. |
| **Regresión** | Elegir una respuesta numérica, probar modelos y comparar sus predicciones. |
| **MAS** | Consultar agentes disponibles, sus mensajes y decisiones; crear/probar agentes y ejecutar el Estudio de Caso 04. |
| **Experimentos** | Consultar, comparar, descargar o eliminar ejecuciones guardadas. |

Los valores ausentes de los predictores supervisados se completan dentro del proceso de entrenamiento, después de
separar los datos para prueba. Las filas sin respuesta conocida no se utilizan para entrenar.

## Configurar la API key de Gemini (opcional)

Para usar Gemini con los agentes:

1. Entra en [Google AI Studio](https://aistudio.google.com/apikey), inicia sesión y crea una API key.
2. Abre PowerShell en la carpeta del proyecto y activa el entorno virtual.
3. Escribe el siguiente comando, reemplazando `PEGA_TU_CLAVE_AQUI` por tu clave, y luego inicia la aplicación
   **en esa misma terminal**:

```powershell
.\.venv\Scripts\Activate.ps1
$env:GEMINI_API_KEY = "PEGA_TU_CLAVE_AQUI"
python -m streamlit run app.py
```

Si la aplicación ya estaba abierta, detenla con `Ctrl+C` y vuelve a iniciarla después de definir la clave. Esta
configuración dura mientras esa terminal esté abierta; al abrir otra, debes volver a definirla.

La clave es opcional: sin ella, o si Gemini no está disponible, los análisis compatibles usan reglas locales.
No compartas tu clave ni la incluyas en archivos del proyecto, capturas o commits. Gemini requiere conexión a
internet y puede tener límites de uso.

## Datos de ejemplo

`data/` incluye archivos para clasificación, agrupamiento y pruebas, además de `regression_sample.csv`, que se
genera de forma sintética. Las fuentes y licencias de los CSV heredados no están documentadas; confirma sus
condiciones antes de redistribuirlos. Consulta [la guía de datos](data/README.md).

## Experimentos e informe

Para repetir los benchmarks y generar las tablas, el manifiesto y las cuatro figuras:

```powershell
python scripts/run_experiments.py
```

Para ejecutar el Estudio de Caso 04 en tres conjuntos de datos y generar sus resultados textuales:

```powershell
python scripts/run_case04_experiments.py
```

El protocolo, las métricas y los archivos generados se explican en
[la guía del Caso 04](docs/CASO04_INTEGRADO.md).

Para compilar el [informe técnico en PDF](docs/technical_report.pdf), con arquitectura MAS, diagramas, resultados y
bibliografía:

```powershell
.\scripts\build_report.ps1
```

El script genera también los diagramas del informe. La metodología y los archivos de resultados se explican en
[docs/resultados/README.md](docs/resultados/README.md) y las fuentes de los diagramas en [docs/DIAGRAMAS.md](docs/DIAGRAMAS.md).
La compilación del PDF requiere la herramienta TeX indicada en Requisitos.

## Pruebas y evidencias

```powershell
python -m pytest -q
.\scripts\run_tests_report.ps1
```

El segundo comando guarda la salida de pruebas y cobertura en [docs/evidencias/](docs/evidencias/README.md).
La cobertura se ejecuta con `pytest-cov`. Las capturas visuales de la interfaz son manuales y sus pasos están en
[docs/evidencias/CAPTURAS.md](docs/evidencias/CAPTURAS.md).

## Paquete de entrega

El generador excluye entornos virtuales, archivos de ejecución y secretos. Cuando el equipo confirme las
condiciones de redistribución de los datos y el nombre solicitado por el profesor, puede crear el archivo con:

```powershell
python scripts/package_project.py --output .\dist\nombre-de-entrega.zip
```

No entregues el ZIP hasta comprobar el nombre requerido y la licencia/procedencia de los conjuntos de datos.

## Estructura principal

```text
app.py                  Entrada de Streamlit
frontend/               Páginas y componentes de la interfaz
framework_ia/            Controlador, análisis, agentes y servicios
data/                    Datos de ejemplo
scripts/                 Pruebas reproducibles, informe y empaquetado
tests/                   Pruebas automatizadas
docs/                    Informe, resultados y evidencias
```
