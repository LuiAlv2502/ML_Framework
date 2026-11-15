# Evidencias de validación

Entorno: Windows, Python 3.13.7. Validación de entrega: 15 de noviembre de 2026.

## Suite automatizada

Comando:

```powershell
python -m pytest -q
```

Resultado: **209 pruebas aprobadas**, una advertencia conocida de UMAP relacionada con `random_state` y `n_jobs`.

Las pruebas nuevas abren las páginas Inicio, Datos, Preprocesamiento, Agentes y Experimentos mediante Streamlit
AppTest, incluyendo un registro existente en el historial; verifican la creación y eliminación de un agente de
plugin a través del controlador; comprueban que el generador de regresión es reproducible; y prueban que el
empaquetador omite archivos locales y secretos.

## Comprobaciones adicionales

- `python -m compileall -q framework_ia frontend scripts tests`: correcto.
- `git diff --check`: correcto.
- `python -m pip check`: no encontró dependencias incompatibles.
- La lista fijada de dependencias coincide con las versiones instaladas del entorno validado.
- La compilación PDF de `docs/technical_report.tex` no se ejecutó: no hay `pdflatex` ni `tectonic` instalados en el entorno.

## Creación dinámica de agentes

`tests/test_dashboard_pages.py::test_controlador_permite_crear_y_quitar_agente_dinamico` usa el tipo plugin
`knn_auto`, lo registra mediante `FrameworkController.create_agent`, verifica su presencia y configuración, y lo
retira mediante `FrameworkController.remove_agent`. La prueba no necesita credenciales de IA ni cambia el estado
persistente del usuario.

## Paquete de entrega

El empaquetador se prueba en un proyecto temporal con un entorno virtual, ejecuciones guardadas, un archivo de
secretos y el estado de agentes. La prueba confirma que el ZIP solo contiene archivos fuente permitidos. El archivo
final se genera con:

```powershell
python scripts/package_project.py
```

Las fuentes y licencias de los cuatro CSV preexistentes no figuran en el repositorio. `data/README.md` señala esa
limitación y no atribuye licencias no verificadas.
