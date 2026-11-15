# Material heredado (legacy)

Esta carpeta conserva el material de los laboratorios anteriores que **no forma parte del framework**:

| Archivo | Descripción |
|---------|-------------|
| `controller.py` | `DataController` del patrón MVC original (para notebooks). |
| `view.py` | `DataView` basada en `ipywidgets` (solo funciona dentro de Jupyter). |
| `notebooks/` | Experimentos de los laboratorios 1–3 (MVC, no supervisado, clasificación). |

Notas:

- Ningún módulo de `framework_ia/` ni de `frontend/` importa estos archivos; `view.py` depende de `ipywidgets`,
  que no es una dependencia del framework.
- El modelo `DataModel` se trasladó a `framework_ia/ml/data_model.py`; los algoritmos están en `framework_ia/ml/`.
- Los notebooks se escribieron para la estructura anterior (`src/` agregado a `sys.path`). Para ejecutarlos con el
  framework actual, instale el paquete (`pip install -e .`) y reemplace los imports, por ejemplo:

  ```python
  from framework_ia.ml import DataModel, ClasificacionModelos
  import sys; sys.path.insert(0, "..")  # solo para DataView/DataController de esta carpeta
  from view import DataView
  from controller import DataController
  ```
