from pathlib import Path
import json
import sys
from types import SimpleNamespace as NS

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import AgenteClasificacion, AgenteRegresion, AgenteSupervisado


def _tool_call(nombre, argumentos, call_id):
    return NS(id=call_id, function=NS(name=nombre, arguments=json.dumps(argumentos)))


def _respuesta(*llamadas):
    return NS(choices=[NS(message=NS(content=None, tool_calls=list(llamadas)))])


class _ClienteFalso:
    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.llamadas.append(kwargs)
        return self.respuestas.pop(0)


def _drug():
    return pd.read_csv(ROOT / "data" / "drug200.csv")


def _mall():
    return pd.read_csv(ROOT / "data" / "Mall_Customers.csv")


def _validar(algoritmo, call_id, **parametros):
    args = {"algoritmo": algoritmo, "justificacion": "probar"}
    if parametros:
        args["parametros"] = parametros
    return _tool_call("validar_modelo", args, call_id)


def test_agente_llm_elige_objetivo_valida_y_finaliza():
    cliente = _ClienteFalso([
        _respuesta(
            _tool_call("elegir_variable_objetivo", {"columna": "Drug", "justificacion": "última columna"}, "1"),
            _validar("knn", "2", n_neighbors=5),
            _validar("decision_tree", "3", max_depth=4),
            _validar("random_forest", "4", n_estimators=50),
        ),
        _respuesta(_tool_call(
            "finalizar", {"mejor_modelo": "decision_tree(max_depth=4)", "analisis": "El árbol funciona bien."}, "5"
        )),
    ])
    df = _drug()
    copia = df.copy()

    agente = AgenteClasificacion(df, repeticiones=2, cliente=cliente).ejecutar()

    assert agente.modo == "llm"
    assert agente.objetivo == "Drug"
    assert agente.eleccion_llm == "decision_tree(max_depth=4)"
    assert agente.mejor in agente.validaciones
    assert agente.analisis_llm == "El árbol funciona bien."
    assert agente.autoevaluacion and agente.supera_linea_base is not None
    assert {i["clave"] for i in agente.plan if i["ejecutar"]} == {"knn", "decision_tree", "random_forest"}
    assert agente.uso_llm["iteraciones"] == 2
    assert any(fila["etapa"] == "autoevaluación" for fila in agente.bitacora)
    pd.testing.assert_frame_equal(df, copia)


def test_finalizar_exige_familias_distintas():
    cliente = _ClienteFalso([
        _respuesta(
            _tool_call("elegir_variable_objetivo", {"columna": "Drug", "justificacion": "objetivo"}, "1"),
            _validar("knn", "2"),
            _validar("knn", "3", n_neighbors=3),
        ),
        _respuesta(_tool_call("finalizar", {"mejor_modelo": "knn", "analisis": "x"}, "4")),
        _respuesta(_validar("decision_tree", "5"), _validar("random_forest", "6", n_estimators=30)),
        _respuesta(_tool_call("finalizar", {"mejor_modelo": "knn", "analisis": "ok"}, "7")),
    ])

    agente = AgenteClasificacion(_drug(), repeticiones=2, cliente=cliente).ejecutar()

    assert agente.modo == "llm"
    assert agente.uso_llm["errores_herramientas"] == 1
    assert len(agente.validaciones) == 4


def test_herramientas_rechazan_objetivo_y_parametros_invalidos():
    agente = AgenteRegresion(_mall(), repeticiones=2)
    agente.percibir()
    agente.planificar()

    id_rechazado = agente._despachar(
        "elegir_variable_objetivo", {"columna": "CustomerID", "justificacion": "x"}
    )
    categorica_rechazada = agente._despachar(
        "elegir_variable_objetivo", {"columna": "Gender", "justificacion": "x"}
    )
    agente._despachar(
        "elegir_variable_objetivo", {"columna": "Spending Score (1-100)", "justificacion": "x"}
    )
    fuera_de_rango = agente._despachar(
        "validar_modelo", {"algoritmo": "ridge", "parametros": {"alpha": -1}, "justificacion": "x"}
    )
    desconocido = agente._despachar(
        "validar_modelo", {"algoritmo": "ridge", "parametros": {"gamma": 1}, "justificacion": "x"}
    )

    assert "identificador" in id_rechazado["error"]
    assert "no es numérica" in categorica_rechazada["error"]
    assert "entre" in fuera_de_rango["error"]
    assert "no permitidos" in desconocido["error"]
    assert not agente.validaciones


def test_sin_clave_usa_reglas_y_excluye_identificadores(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    agente = AgenteRegresion(_mall(), repeticiones=2).ejecutar()

    assert agente.modo == "reglas"
    assert agente.motivo_respaldo
    assert agente.objetivo == "Spending Score (1-100)"
    assert "CustomerID" in agente.excluidas
    assert agente.mejor == agente.ranking().iloc[0]["Modelo"]
    assert agente.autoevaluacion
    informe = agente.informe_html()
    assert "Autoevaluación del agente" in informe
    assert "Bitácora de razonamiento" in informe


def test_usa_reglas_si_el_llm_no_finaliza():
    cliente = _ClienteFalso([_respuesta(_tool_call("finalizar", {"mejor_modelo": "x", "analisis": "x"}, "1"))])

    agente = AgenteClasificacion(
        _drug(), repeticiones=2, cliente=cliente, max_iteraciones=1
    ).ejecutar()

    assert agente.modo == "reglas"
    assert "no finalizó" in agente.motivo_respaldo
    assert agente.objetivo == "Drug"
    assert not agente.analisis_llm


def test_objetivo_del_usuario_no_puede_cambiarse():
    cliente = _ClienteFalso([
        _respuesta(
            _tool_call("elegir_variable_objetivo", {"columna": "Annual Income (k$)", "justificacion": "x"}, "1"),
            _validar("regresion_lineal_multiple", "2"),
            _validar("decision_tree", "3"),
            _validar("random_forest", "4", n_estimators=30),
        ),
        _respuesta(_tool_call("finalizar", {"mejor_modelo": "decision_tree", "analisis": "ok"}, "5")),
    ])

    agente = AgenteRegresion(
        _mall(), repeticiones=2, cliente=cliente, objetivo="Age"
    ).ejecutar()

    assert agente.modo == "llm"
    assert agente.objetivo == "Age"
    assert "CustomerID" in agente.excluidas
    assert agente.uso_llm["errores_herramientas"] == 1


def test_constructor_valida_el_objetivo_del_usuario():
    with pytest.raises(KeyError):
        AgenteClasificacion(_drug(), objetivo="no_existe")
    with pytest.raises(ValueError, match="no es numérica"):
        AgenteRegresion(_drug(), objetivo="Drug")
    with pytest.raises(ValueError, match="parece continua"):
        AgenteClasificacion(_mall(), objetivo="Annual Income (k$)")


def test_la_base_supervisada_no_se_instancia_directamente():
    with pytest.raises(TypeError):
        AgenteSupervisado(_drug())
    assert AgenteClasificacion.TIPO == "clasificacion"
    assert AgenteRegresion.TIPO == "regresion"


def test_al_llm_solo_se_envia_el_perfil():
    df = _drug()
    df.loc[0, "Sex"] = "SENSITIVE_ROW_VALUE"
    cliente = _ClienteFalso([_respuesta(_tool_call("finalizar", {"mejor_modelo": "x", "analisis": "x"}, "1"))])

    AgenteClasificacion(df, repeticiones=2, cliente=cliente, max_iteraciones=1).ejecutar()

    contenido = cliente.llamadas[0]["messages"][1]["content"]
    assert "perfil_de_datos" in contenido and "plan_sugerido_por_reglas" in contenido
    assert "SENSITIVE_ROW_VALUE" not in contenido
