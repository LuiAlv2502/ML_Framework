import pytest

from framework_ia.agents import AgentRegistry, AgentStatus, Message, Performative
from framework_ia.utils.exceptions import AgentError, ValidationError

from conftest import EchoAgent, FlakyAgent


def test_registro_alta_baja_y_consulta():
    registro = AgentRegistry()
    eventos = []
    registro.subscribe(eventos.append)
    agente = registro.register(EchoAgent("Eco uno", agent_id="eco-1"))
    assert "eco-1" in registro and len(registro) == 1
    assert registro.get("eco-1") is agente and registro.find_by_name("ECO UNO") is agente
    assert registro.capabilities() == {"test.echo": ["eco-1"]}
    assert registro.snapshot()[0]["name"] == "Eco uno"
    registro.unregister("eco-1")
    assert len(registro) == 0 and agente.status is AgentStatus.STOPPED
    assert [e["event"] for e in eventos] == ["registered", "unregistered"]
    with pytest.raises(AgentError):
        registro.get("eco-1")
    with pytest.raises(AgentError):
        registro.unregister("eco-1")


def test_registro_rechaza_duplicados_y_objetos_invalidos():
    registro = AgentRegistry()
    registro.register(EchoAgent("Eco", agent_id="eco"))
    with pytest.raises(AgentError, match="id"):
        registro.register(EchoAgent("Otro", agent_id="eco"))
    with pytest.raises(AgentError, match="llamado"):
        registro.register(EchoAgent("eco", agent_id="eco-2"))
    with pytest.raises(AgentError):
        registro.register(object())


def test_busqueda_por_capacidad_solo_activos_y_ordenada():
    registro = AgentRegistry()
    bajo = registro.register(EchoAgent("Bajo", agent_id="bajo", priority=1))
    alto = registro.register(EchoAgent("Alto", agent_id="alto", priority=50))
    pausado = registro.register(EchoAgent("Pausado", agent_id="pausado", priority=99))
    registro.update_status("pausado", AgentStatus.PAUSED)
    degradado = registro.register(EchoAgent("Degradado", agent_id="degradado", priority=80))
    degradado.set_status(AgentStatus.DEGRADED)

    encontrados = registro.find_by_capability("test.echo")
    assert [a.agent_id for a in encontrados] == ["alto", "bajo", "degradado"]
    assert pausado not in encontrados
    assert registro.find_by_capability("test.echo", exclude=["alto"])[0] is bajo
    assert registro.find_by_capability("otra.cosa") == []
    assert len(registro.list(include_inactive=False)) == 3
    assert alto.describe()["status"] == "idle"


def test_busqueda_prefiere_mayor_tasa_de_exito_a_igual_prioridad():
    registro = AgentRegistry()
    bueno = registro.register(EchoAgent("Bueno", agent_id="bueno"))
    malo = registro.register(FlakyAgent("Malo", agent_id="malo", config={"fail_times": 1}))
    malo.handle(Message("t", "malo", Performative.REQUEST, "test.echo"))  # falla -> DEGRADED
    malo.set_status(AgentStatus.IDLE)
    assert malo.success_rate == 0.0
    assert registro.find_by_capability("test.echo")[0] is bueno


def test_agente_valida_su_definicion():
    with pytest.raises(ValidationError, match="Capacidad inválida"):
        EchoAgent("Eco", capabilities=["MAL"])
    with pytest.raises(ValidationError, match="prioridad"):
        EchoAgent("Eco", priority=500)
    with pytest.raises(ValidationError, match="no permitidos"):
        EchoAgent("Eco", config={"zzz": 1})
    with pytest.raises(ValidationError, match="nombre"):
        EchoAgent("")
