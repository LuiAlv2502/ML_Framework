import pytest

from framework_ia.agents import Agency, CoordinatorAgent, Performative, build_default_agency
from framework_ia.persistence import ConfigStore

from conftest import EchoAgent, FlakyAgent


@pytest.fixture
def agencia():
    agencia = Agency()
    agencia.factory.register_agent_type("echo_test", EchoAgent, overwrite=True)
    agencia.factory.register_agent_type("flaky_test", FlakyAgent, overwrite=True)
    agencia.coordinator = agencia.factory.create("coordinator", "Coordinador", agent_id="coordinador")
    agencia.coordinator.routing = {**agencia.coordinator.routing, "eco": {"capability": "test.echo"},
                                   "eco_todos": {"capability": "test.echo", "policy": "fanout"}}
    return agencia


def test_enrutamiento_por_capacidad_desde_la_tabla(agencia):
    agencia.create_agent({"agent_type": "echo_test", "name": "Eco"}, persist=False)
    respuesta = agencia.request("eco", {"value": 5})
    assert respuesta.performative is Performative.INFORM
    resultado = respuesta.payload
    assert resultado["capability"] == "test.echo" and resultado["result"]["echo"] == 5
    assert resultado["policy"] == "priority"
    decisiones = [d["decision"] for d in resultado["decisions"]]
    assert decisiones[0] == "Solicitud recibida" and decisiones[-1] == "Consolidación"
    assert any(d.startswith("Descubrimiento") for d in decisiones)
    assert set(resultado["timings"]) == {"analysis_s", "delegation_s", "consolidation_s", "total_s"}


def test_capacidad_explicita_y_tarea_desconocida(agencia):
    agencia.create_agent({"agent_type": "echo_test", "name": "Eco"}, persist=False)
    assert agencia.request("test.echo", {"value": 1}).payload["capability"] == "test.echo"
    assert agencia.request("otra", {"capability": "test.echo"}).ok
    desconocida = agencia.request("tarea_inventada", {})
    assert not desconocida.ok and desconocida.payload["user_error"]
    assert "desconocida" in desconocida.error


def test_politica_priority_usa_al_de_mayor_prioridad(agencia):
    agencia.create_agent({"agent_type": "echo_test", "name": "Bajo", "priority": 1}, persist=False)
    alto = agencia.create_agent({"agent_type": "echo_test", "name": "Alto", "priority": 60}, persist=False)
    resultado = agencia.request("eco", {"value": 1}).payload
    assert resultado["selected_agent"] == alto.agent_id and resultado["agents_used"] == [alto.agent_id]


def test_politica_fanout_consolida_por_metrica_principal(agencia):
    agencia.create_agent({"agent_type": "echo_test", "name": "Peor", "config": {"score": 0.3}}, persist=False)
    mejor = agencia.create_agent({"agent_type": "echo_test", "name": "Mejor", "config": {"score": 0.8}}, persist=False)
    resultado = agencia.request("eco_todos", {"value": 1}).payload
    assert resultado["policy"] == "fanout" and len(resultado["agents_used"]) == 2
    assert resultado["selected_agent"] == mejor.agent_id
    assert {a["value"] for a in resultado["alternatives"]} == {0.3, 0.8}
    assert "mayor score" in resultado["decisions"][-1]["justification"]


def test_politica_por_payload_y_politica_invalida(agencia):
    agencia.create_agent({"agent_type": "echo_test", "name": "Eco"}, persist=False)
    assert agencia.request("eco", {"policy": "fanout"}).payload["policy"] == "fanout"
    assert not agencia.request("eco", {"policy": "azar"}).ok


def test_error_si_no_hay_agente_con_la_capacidad(agencia):
    respuesta = agencia.request("eco", {})
    assert respuesta.performative is Performative.FAILURE
    assert "test.echo" in respuesta.error
    assert respuesta.payload["error_type"] == "CapabilityNotFoundError"
    assert any(d["decision"] == "Sin candidatos" for d in respuesta.payload["decisions"])
    assert agencia.coordinator.status.value == "idle"  # no es un fallo propio del coordinador


def test_fallback_queda_en_las_decisiones(agencia):
    agencia.create_agent({"agent_type": "flaky_test", "name": "Roto", "priority": 90,
                          "config": {"fail_times": 99}}, persist=False)
    agencia.create_agent({"agent_type": "echo_test", "name": "Sano"}, persist=False)
    resultado = agencia.request("eco", {"value": 2}).payload
    assert any(d["decision"] == "Fallback" for d in resultado["decisions"])
    mensajes = agencia.bus.history(correlation_id=resultado["correlation_id"])
    assert {m.performative.value for m in mensajes} >= {"request", "failure", "inform"}


def test_coordinador_no_se_lista_como_candidato_de_capacidades_de_dominio(agencia):
    assert isinstance(agencia.coordinator, CoordinatorAgent)
    assert agencia.candidates("classification.train") == []
    assert agencia.coordinator.can_handle("classification.train")


def test_tabla_de_enrutamiento_viene_de_la_configuracion(tmp_path):
    import json

    archivo = tmp_path / "config.json"
    archivo.write_text(json.dumps({"coordinator": {"routing": {"saludo": {"capability": "eda.summary"}}}}),
                       encoding="utf-8")
    agencia = build_default_agency(ConfigStore(archivo, root=tmp_path), state_store=False, model_store=False)
    assert agencia.coordinator.routing["saludo"] == {"capability": "eda.summary"}
    assert agencia.coordinator.routing["classification"]["capability"] == "classification.train"


def test_flujo_de_punta_a_punta_con_drug200(drug):
    agencia = build_default_agency(state_store=False, model_store=False, rehydrate=False)
    respuesta = agencia.request("classification", {"data": drug, "target": "Drug", "algorithm": "knn",
                                                   "use_llm": False})
    assert respuesta.ok, respuesta.error
    resultado = respuesta.payload
    assert resultado["selected_agent"] == "agente-clasificacion"
    assert 0 <= resultado["result"]["metrics"]["accuracy"] <= 1
    assert len(resultado["result"]["confusion_matrix"]["labels"]) == 5
