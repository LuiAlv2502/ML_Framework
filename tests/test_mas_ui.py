from pathlib import Path
from tempfile import TemporaryDirectory

from streamlit.testing.v1 import AppTest

from framework_ia.controller import FrameworkController

ROOT = Path(__file__).resolve().parents[1]


def test_ui_crea_agente_lo_prueba_y_guarda_mensajes_y_decision():
    with TemporaryDirectory() as directorio:
        controlador = FrameworkController(persist=False, runs_dir=Path(directorio))
        controlador.load_sample("drug200.csv")
        prueba = AppTest.from_file(str(ROOT / "frontend" / "pages" / "agents.py"), default_timeout=30)
        prueba.session_state["controller"] = controlador
        prueba.run()

        selector_tipo = next(selector for selector in prueba.selectbox if selector.label == "Tipo de agente")
        opcion_plugin = next(opcion for opcion in selector_tipo.options if opcion.startswith("KNNAutoAgent"))
        selector_tipo.select(opcion_plugin).run()
        next(campo for campo in prueba.text_input if campo.label == "Nombre").set_value("KNN UI")
        next(boton for boton in prueba.button if boton.label == "Crear agente").click().run()

        agente = next(agente for agente in controlador.list_agents() if agente["name"] == "KNN UI")
        assert not prueba.exception
        assert any(
            "ID" in frame.value.columns and agente["agent_id"] in frame.value["ID"].tolist()
            for frame in prueba.dataframe
        )

        next(boton for boton in prueba.button if boton.label == "Probar con una tarea").click().run()

        assert not prueba.exception
        resultado = controlador.last_task
        assert resultado and resultado["ok"]
        run = controlador.get_experiment(resultado["run_id"])["run"]
        assert run["coordination"]["selected_agent"] == agente["agent_id"]
        assert any(
            mensaje["sender"] == agente["agent_id"]
            for mensaje in controlador.list_messages()
        )
        assert any(
            mensaje["sender"] == agente["agent_id"]
            for mensaje in controlador.list_messages(sender=agente["agent_id"])
        )
        assert run["decisions"]
        assert any(
            agente["agent_id"] in str(row)
            for frame in prueba.dataframe
            for row in frame.value.to_dict("records")
        )


def test_ui_ejecuta_el_estudio_caso04_desde_evaluacion():
    with TemporaryDirectory() as directorio:
        controlador = FrameworkController(persist=False, runs_dir=Path(directorio))
        controlador.load_sample("drug200.csv")
        prueba = AppTest.from_file(str(ROOT / "frontend" / "pages" / "agents.py"), default_timeout=60)
        prueba.session_state["controller"] = controlador
        prueba.run()

        semillas = next(
            selector for selector in prueba.multiselect
            if selector.label == "Semillas para repetir las pruebas"
        )
        semillas.set_value([42])
        next(
            boton for boton in prueba.button
            if boton.label == "Ejecutar los tres escenarios"
        ).click().run()

        assert not prueba.exception
        assert len(controlador.list_experiments()) == 5
        assert any("F1 macro medio" in frame.value.columns for frame in prueba.dataframe)
