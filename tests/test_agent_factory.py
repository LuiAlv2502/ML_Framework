import pytest

from framework_ia.agents import (
    AgentFactory,
    AgentRegistry,
    BaseAgent,
    MessageBus,
    Performative,
    agent_type,
    registered_agent_types,
)
from framework_ia.persistence import AgentStateStore
from framework_ia.utils.exceptions import AgentError, ValidationError

from conftest import EchoAgent


@pytest.fixture
def fabrica(tmp_path):
    fabrica = AgentFactory(AgentRegistry(), MessageBus(), state_store=AgentStateStore(tmp_path / "agents.json"))
    fabrica.register_agent_type("echo_test", EchoAgent)
    return fabrica


def test_creacion_dinamica_registra_suscribe_y_publica_evento(fabrica):
    agente = fabrica.create("echo_test", "Eco dinámico", config={"score": 0.7}, priority=30)
    assert agente.agent_id in fabrica.registry
    assert fabrica.bus.is_subscribed(agente.agent_id)
    evento = fabrica.bus.history(task="agent.created")[0]
    assert evento.performative is Performative.CONFIRM and evento.receiver == agente.agent_id
    assert agente.config == {"score": 0.7} and agente.priority == 30


@pytest.mark.parametrize(
    ("argumentos", "mensaje"),
    [
        ({"agent_type": "no_existe", "name": "X1"}, "desconocido"),
        ({"agent_type": "echo_test", "name": "X1", "capabilities": ["Mal Formato"]}, "Capacidad inválida"),
        ({"agent_type": "echo_test", "name": "X1", "config": {"score": 5}}, "entre 0.0 y 1.0"),
        ({"agent_type": "echo_test", "name": "!"}, "nombre"),
    ],
)
def test_creacion_valida_definiciones(fabrica, argumentos, mensaje):
    with pytest.raises(ValidationError, match=mensaje):
        fabrica.create(**argumentos)


def test_tipos_soportados_restringen_capacidades(fabrica):
    from framework_ia.agents import ClassificationAgent

    fabrica.register_agent_type("clasif_test", ClassificationAgent)
    with pytest.raises(ValidationError, match="no sabe atender"):
        fabrica.create("clasif_test", "Clasificador", capabilities=["regression.train"])
    with pytest.raises(ValidationError, match="Algoritmos desconocidos"):
        fabrica.create("clasif_test", "Clasificador", config={"allowed_algorithms": ["magia"]})
    agente = fabrica.create("clasif_test", "Solo KNN", capabilities=["classification.train"],
                            config={"allowed_algorithms": ["knn"], "default_params": {"knn": {"n_neighbors": 3}}})
    assert agente.allowed_algorithms == ["knn"]


def test_nombres_duplicados_y_destruccion(fabrica):
    agente = fabrica.create("echo_test", "Único")
    with pytest.raises(ValidationError, match="Ya existe"):
        fabrica.create("echo_test", "único")
    fabrica.destroy(agente.agent_id)
    assert agente.agent_id not in fabrica.registry and not fabrica.bus.is_subscribed(agente.agent_id)
    with pytest.raises(AgentError):
        fabrica.destroy(agente.agent_id)


def test_registro_de_tipos_valida_clases(fabrica):
    with pytest.raises(ValidationError, match="BaseAgent"):
        fabrica.register_agent_type("cosa", dict)
    with pytest.raises(ValidationError, match="abstracta"):
        fabrica.register_agent_type("abstracto", BaseAgent)
    with pytest.raises(ValidationError, match="ya existe"):
        fabrica.register_agent_type("echo_test", EchoAgent)
    with pytest.raises(ValidationError, match="inválido"):
        fabrica.register_agent_type("Con Espacios", EchoAgent)


def test_decorador_registra_en_catalogo_global():
    @agent_type("decorado_test", description="Tipo creado en la prueba")
    class Decorado(EchoAgent):
        pass

    assert registered_agent_types()["decorado_test"] is Decorado
    assert Decorado.agent_type == "decorado_test"
    fabrica = AgentFactory(AgentRegistry())
    assert any(t["agent_type"] == "decorado_test" for t in fabrica.available_types())


def test_descubrimiento_de_plugins_registra_knn_auto():
    fabrica = AgentFactory(AgentRegistry())
    modulos = fabrica.discover_plugins()
    assert "framework_ia.agents.plugins.knn_auto" in modulos
    assert "knn_auto" in fabrica.types
    assert fabrica.discover_plugins("paquete.que.no.existe") == []
    assert "paquete.que.no.existe" in fabrica.plugin_errors


def test_persistencia_y_rehidratacion(fabrica, tmp_path):
    fabrica.create("echo_test", "Persistente", config={"score": 0.9}, priority=42, persist=True)
    almacen = AgentStateStore(tmp_path / "agents.json")
    assert almacen.load()[0]["name"] == "Persistente"

    nueva = AgentFactory(AgentRegistry(), MessageBus(), state_store=almacen)
    nueva.register_agent_type("echo_test", EchoAgent)
    almacen.upsert({"name": "Roto", "agent_type": "no_existe"})
    creados, errores = nueva.rehydrate()
    assert [a.name for a in creados] == ["Persistente"]
    assert creados[0].priority == 42 and creados[0].config == {"score": 0.9}
    assert "Roto" in errores
    assert nueva.rehydrate()[0] == []  # no duplica agentes ya presentes


def test_persistir_sin_almacen_falla():
    fabrica = AgentFactory(AgentRegistry())
    fabrica.register_agent_type("echo_test", EchoAgent)
    with pytest.raises(AgentError):
        fabrica.create("echo_test", "Sin almacén", persist=True)


def test_create_from_definition_valida_campos(fabrica):
    with pytest.raises(ValidationError, match="Faltan"):
        fabrica.create_from_definition({"name": "X1"})
    with pytest.raises(ValidationError):
        fabrica.create_from_definition("no es dict")
