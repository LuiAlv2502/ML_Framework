import pytest

from framework_ia.controller import FrameworkController
from framework_ia.utils.exceptions import ValidationError


def test_controller_muestra_errores_validos_sin_traceback(tmp_path):
    controlador = FrameworkController(persist=False, runs_dir=tmp_path)

    with pytest.raises(ValidationError) as dataset_error:
        controlador.load_dataset(b"invalid", "datos.exe")
    assert "Traceback" not in str(dataset_error.value)
    assert "no soportado" in str(dataset_error.value)

    controlador.load_sample("drug200.csv")
    with pytest.raises(ValidationError) as target_error:
        controlador.run_classification(target="objetivo_inexistente", algorithm="knn")
    assert "Traceback" not in str(target_error.value)
    assert "no existe" in str(target_error.value).lower()

    with pytest.raises(ValidationError) as task_error:
        controlador.run_task("tarea_desconocida")
    assert "Traceback" not in str(task_error.value)
    assert "desconocida" in str(task_error.value).lower()
