import pandas as pd
import pytest

from framework_ia.agents import Message, MessageBus, MessageStatus, Performative
from framework_ia.utils.exceptions import ValidationError


def _request(**extra):
    datos = {"sender": "a", "receiver": "b", "performative": "request", "task": "test.echo", "payload": {"value": 1}}
    datos.update(extra)
    return Message(**datos)


def test_message_valores_por_defecto_y_correlacion():
    mensaje = _request()
    assert mensaje.performative is Performative.REQUEST
    assert mensaje.correlation_id == mensaje.id
    assert mensaje.status == MessageStatus.CREATED.value


@pytest.mark.parametrize(
    ("extra", "texto"),
    [
        ({"sender": ""}, "emisor"),
        ({"task": " "}, "tarea"),
        ({"performative": "gritar"}, "Performativa"),
        ({"payload": [1, 2]}, "diccionario"),
    ],
)
def test_message_valida_campos(extra, texto):
    with pytest.raises(ValidationError, match=texto):
        _request(**extra)


def test_reply_conserva_correlacion_e_invierte_participantes():
    mensaje = _request()
    respuesta = mensaje.reply(Performative.INFORM, {"ok": True})
    assert (respuesta.sender, respuesta.receiver) == ("b", "a")
    assert respuesta.correlation_id == mensaje.correlation_id
    assert respuesta.in_reply_to == mensaje.id
    assert respuesta.ok and not mensaje.reply("failure").ok


def test_to_dict_resume_dataframes():
    mensaje = _request(payload={"data": pd.DataFrame({"x": range(500)})})
    datos = mensaje.to_dict()
    assert datos["payload"]["data"]["tabla"] == "500 filas × 1 columnas"
    assert datos["performative"] == "request"


def test_bus_entrega_registra_y_filtra():
    bus = MessageBus()
    bus.subscribe("b", lambda m: m.reply(Performative.INFORM, {"eco": m.payload["value"]}))
    respuesta = bus.send(_request())
    assert respuesta.payload == {"eco": 1}
    historial = bus.history()
    assert len(historial) == 2 and historial[0].status == MessageStatus.PROCESSED.value
    assert [m.performative for m in bus.history(performative="inform")] == [Performative.INFORM]
    assert len(bus.history(correlation_id=historial[0].correlation_id)) == 2
    assert len(bus.history(agent="b")) == 2 and bus.history(sender="zzz") == []
    assert bus.history(limit=1)[0] is respuesta
    assert bus.records()[0]["sender"] == "a"


def test_bus_receptor_inexistente_devuelve_failure():
    bus = MessageBus()
    respuesta = bus.send(_request(receiver="fantasma"))
    assert respuesta.performative is Performative.FAILURE and "fantasma" in respuesta.error
    assert bus.history()[0].status == MessageStatus.UNDELIVERABLE.value


def test_bus_publish_y_observadores():
    bus = MessageBus(max_history=3)
    vistos = []
    bus.add_listener(vistos.append)
    bus.add_listener(lambda m: 1 / 0)  # un observador defectuoso no rompe el bus
    for _ in range(5):
        bus.publish(_request(performative="confirm"))
    assert len(bus) == 3 and len(vistos) == 5
    assert bus.history()[0].status == MessageStatus.LOGGED.value
    bus.unsubscribe("b")
    assert not bus.is_subscribed("b")
    bus.clear()
    assert len(bus) == 0
