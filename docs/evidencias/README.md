# Evidencias del proyecto

Este directorio conserva salidas reproducibles, instrucciones de capturas y resultados de pruebas.

## Artefactos

- [Registro de creación dinámica de agente](dynamic_agent_log.md), generado por `python scripts/demo_dynamic_agent.py`.
- [Salida de la suite de pruebas](pytest_output.txt), generada por `scripts/run_tests_report.ps1`.
- [Reporte de cobertura](coverage.txt), generado por `scripts/run_tests_report.ps1`.
- [Guía de capturas manuales](CAPTURAS.md).
- [Informe técnico en PDF](../technical_report.pdf), con [fuente LaTeX](../technical_report.tex) y bibliografía.
- [Fuentes de los diagramas](../DIAGRAMAS.md) y [resultados experimentales](../resultados/README.md).

## Validación automatizada actual

La suite pasó **225 pruebas** en Windows con Python 3.13.7; queda una advertencia conocida de UMAP relacionada con
`random_state` y `n_jobs`. La cobertura agregada medida para `framework_ia` fue **83 %**, y las carpetas solicitadas
superaron el 80 %: agentes **89.2 %**, servicios **91.0 %**, persistencia **92.4 %**. Consulte los archivos de
salida generados para el detalle línea por línea.

La fuente del informe se actualizó con el diseño MAS y tres diagramas nuevos. La compilación del PDF actualizado
sigue pendiente: este entorno no tiene `pdflatex` ni `bibtex`, por lo que el PDF existente puede corresponder a una
versión anterior de la fuente. Las capturas visuales manuales siguen pendientes; sus pasos están descritos en
[CAPTURAS.md](CAPTURAS.md). La procedencia/licencia de los CSV heredados aún requiere confirmación del equipo.

## Regenerar las evidencias

```powershell
python scripts/demo_dynamic_agent.py
.\scripts\run_tests_report.ps1
```

El script de pruebas escribe los resultados aquí. Para ejecutar cobertura manualmente:

```powershell
python -m pytest --cov=framework_ia --cov-report=term-missing -q
```
