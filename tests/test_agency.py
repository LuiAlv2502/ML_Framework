import pytest

from framework_ia.agents import Agency, AgentStatus, Message, Performative
from framework_ia.utils.exceptions import CapabilityNotFoundError, TaskExecutionError, ValidationError

from conftest import EchoAgent, FlakyAgent, PickyAgent


@pytest.fixture
def agencia():
    agencia = Agency(max_retries=1)
    for cls in (EchoAgent, FlakyAgent, PickyAgent):
        agencia.factory.register_agent_type(cls.agent_type, cls, overwrite=True)
    return agencia


def test_ciclo_de_vida(agencia):
    agente = agencia.create_agent({"agent_type": "echo_test", "name": "Eco"}, persist=False)
    agencia.pause_agent(agente.agent_id, "mantenimiento")
    assert agente.status is AgentStatus.PAUSED
    assert agencia.candidates("test.echo") == []
    agencia.resume_agent(agente.agent_id)
    assert agente.status is AgentStatus.IDLE
    agencia.stop_agent(agente.agent_id)
    with pytest.raises(ValidationError):
        agencia.resume_agent(agente.agent_id)
    agencia.start_agent(agente.agent_id)
    assert agente.status is AgentStatus.IDLE
    eventos = [m.task for m in agencia.bus.history(performative="confirm") if m.task.startswith("lifecycle")]
    assert eventos == ["lifecycle.pause", "lifecycle.resume", "lifecycle.stop", "lifecycle.start"]
    agencia.remove_agent(agente.agent_id)
    assert len(agencia.registry) == 0


def test_reintentos_recuperan_un_fallo_transitorio(agencia):
    agente = agencia.create_agent({"agent_type": "flaky_test", "name": "Inestable",
                                   "config": {"fail_times": 1}}, persist=False)
    resultado = agencia.dispatch("test.echo", {"value": 3}, sender="tester")
    assert resultado.responses[0].payload["echo"] == 3
    assert [a.ok for a in resultado.attempts] == [False, True]
    assert agente.status is AgentStatus.IDLE and agente.tasks_failed == 1


def test_fallback_a_otro_agente_con_la_misma_capacidad(agencia):
    roto = agencia.create_agent({"agent_type": "flaky_test", "name": "Roto", "priority": 90,
                                 "config": {"fail_times": 99}}, persist=False)
    sano = agencia.create_agent({"agent_type": "echo_test", "name": "Sano", "priority": 10}, persist=False)
    resultado = agencia.dispatch("test.echo", {"value": 7}, sender="tester", correlation_id="corr-1")
    assert resultado.agents_used == [sano.agent_id]
    assert resultado.fallbacks[0]["from"] == roto.agent_id
    assert roto.status is AgentStatus.DEGRADED
    assert agencia.bus.history(task="agency.fallback", correlation_id="corr-1")


def test_fallos_consecutivos_llevan_a_failed_y_restart_recupera(agencia):
    roto = agencia.create_agent({"agent_type": "flaky_test", "name": "Roto",
                                 "config": {"fail_times": 99}}, persist=False)
    agencia.max_retries = 2
    with pytest.raises(TaskExecutionError) as error:
        agencia.dispatch("test.echo", {}, sender="tester")
    assert len(error.value.details["attempts"]) == 3
    assert roto.status is AgentStatus.FAILED
    with pytest.raises(CapabilityNotFoundError):
        agencia.dispatch("test.echo", {}, sender="tester")
    agencia.restart_agent(roto.agent_id)
    assert roto.status is AgentStatus.IDLE and roto.consecutive_failures == 0


def test_error_de_usuario_no_se_reintenta_ni_degrada(agencia):
    exigente = agencia.create_agent({"agent_type": "picky_test", "name": "Exigente", "priority": 90}, persist=False)
    agencia.create_agent({"agent_type": "echo_test", "name": "Eco"}, persist=False)
    with pytest.raises(ValidationError, match="dato inválido"):
        agencia.dispatch("test.echo", {}, sender="tester")
    assert exigente.status is AgentStatus.IDLE and exigente.tasks_rejected == 1
    assert exigente.tasks_failed == 0


def test_fanout_reune_todas_las_respuestas(agencia):
    for nombre, score in (("Agente A", 0.2), ("Agente B", 0.9)):
        agencia.create_agent({"agent_type": "echo_test", "name": nombre, "config": {"score": score}}, persist=False)
    resultado = agencia.dispatch("test.echo", {"value": 1}, sender="tester", policy="fanout")
    assert len(resultado.responses) == 2
    with pytest.raises(ValidationError):
        agencia.dispatch("test.echo", {}, sender="tester", policy="aleatoria")


def test_sin_candidatos(agencia):
    with pytest.raises(CapabilityNotFoundError):
        agencia.dispatch("test.nada", {}, sender="tester")
    assert agencia.bus.history(performative="failure")[0].task == "test.nada"


def test_agente_no_disponible_y_performativas(agencia):
    agente = agencia.create_agent({"agent_type": "echo_test", "name": "Eco"}, persist=False)
    consulta = agente.handle(Message("t", agente.agent_id, Performative.QUERY, "describe"))
    assert consulta.payload["name"] == "Eco"
    assert not agente.handle(Message("t", agente.agent_id, Performative.INFORM, "test.echo")).ok
    assert not agente.handle(Message("t", agente.agent_id, Performative.REQUEST, "otra.cosa")).ok
    agencia.stop_agent(agente.agent_id)
    assert "no está disponible" in agente.handle(Message("t", agente.agent_id, Performative.REQUEST, "test.echo")).error


def test_status_summary(agencia):
    agencia.create_agent({"agent_type": "echo_test", "name": "Eco"}, persist=False)
    resumen = agencia.status_summary()
    assert resumen["agents"] == 1 and resumen["active"] == 1 and resumen["messages"] >= 1
    with pytest.raises(ValidationError):
        Agency(max_retries=9)
    with pytest.raises(ValidationError, match="coordinador"):
        agencia.request("classification", {})
