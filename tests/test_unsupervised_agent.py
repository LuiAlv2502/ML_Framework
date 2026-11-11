from pathlib import Path
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import AgenteNoSupervisado


def _tool_call(nombre, argumentos, call_id):
    return NS(
        id=call_id,
        function=NS(name=nombre, arguments=json.dumps(argumentos)),
    )


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


def _datos():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x": np.r_[rng.normal(-3, 0.3, 20), rng.normal(3, 0.3, 20)],
            "y": np.r_[rng.normal(-2, 0.3, 20), rng.normal(2, 0.3, 20)],
            "grupo": ["A"] * 20 + ["B"] * 20,
        }
    )


def test_agente_ejecuta_pca_codo_y_kmeans_con_cliente_falso():
    cliente = _ClienteFalso(
        [
            _respuesta(
                _tool_call("ejecutar_paso", {"clave": "pca", "justificacion": "resumir"}, "1"),
                _tool_call("ejecutar_paso", {"clave": "codo", "justificacion": "estimar k"}, "2"),
                _tool_call(
                    "ejecutar_paso",
                    {"clave": "kmeans", "justificacion": "agrupar", "n_clusters": 2},
                    "3",
                ),
                _tool_call("omitir_paso", {"clave": "hac", "justificacion": "no necesario"}, "4"),
            ),
            _respuesta(_tool_call("finalizar", {"analisis": "Se observan dos grupos."}, "5")),
        ]
    )
    df = _datos()
    copia = df.copy()

    agente = AgenteNoSupervisado(df, cliente=cliente).ejecutar()

    assert agente.modo == "llm"
    assert agente.analisis_llm == "Se observan dos grupos."
    assert set(agente.resultados) == {"pca", "codo", "kmeans"}
    assert agente.resultados["kmeans"]["grupos_obtenidos"] == 2
    assert set(agente.resultados["kmeans"]["perfil_categorico"]) == {0, 1}
    pd.testing.assert_frame_equal(df, copia)
    mensaje_inicial = cliente.llamadas[0]["messages"][1]["content"]
    assert "perfil_de_datos" in mensaje_inicial
    assert "SENSITIVE_ROW_VALUE" not in mensaje_inicial


def test_agente_usa_reglas_si_el_llm_no_finaliza():
    cliente = _ClienteFalso(
        [_respuesta(_tool_call("finalizar", {"analisis": "prematuro"}, "1"))]
    )

    agente = AgenteNoSupervisado(
        _datos(), cliente=cliente, max_iteraciones=1
    ).ejecutar()

    assert agente.modo == "reglas"
    assert "no finalizó" in agente.motivo_respaldo
    assert "pca" in agente.resultados
    assert "kmeans" in agente.resultados
    assert not agente.analisis_llm


def test_agente_sin_clave_usa_reglas_y_omite_hac(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    agente = AgenteNoSupervisado(_datos()).ejecutar()

    assert agente.modo == "reglas"
    assert agente.motivo_respaldo
    assert {"pca", "codo", "kmeans"} <= set(agente.resultados)
    assert "hac" not in agente.resultados
    assert not next(item for item in agente.plan if item["clave"] == "hac")["ejecutar"]


def test_agente_rechaza_numero_de_clusters_invalido():
    agente = AgenteNoSupervisado(_datos())
    agente.percibir()

    try:
        agente._validar_k(1)
    except ValueError as error:
        assert "n_clusters debe estar" in str(error)
    else:
        raise AssertionError("El agente aceptó un número de clusters inválido.")


def test_perfil_enviado_al_llm_no_contiene_filas_crudas():
    df = _datos()
    df.loc[0, "grupo"] = "SENSITIVE_ROW_VALUE"
    cliente = _ClienteFalso(
        [
            _respuesta(
                *[
                    _tool_call(
                        "omitir_paso",
                        {"clave": clave, "justificacion": "prueba de privacidad"},
                        str(indice),
                    )
                    for indice, clave in enumerate(AgenteNoSupervisado.CLAVES)
                ]
            ),
            _respuesta(_tool_call("finalizar", {"analisis": "Sin análisis."}, "final")),
        ]
    )

    agente = AgenteNoSupervisado(df, cliente=cliente).ejecutar()

    assert agente.modo == "llm"
    contenido = cliente.llamadas[0]["messages"][1]["content"]
    assert "SENSITIVE_ROW_VALUE" not in contenido


def test_informe_html_incluye_resultados_y_escapa_texto_del_dataset():
    df = _datos()
    df["grupo"] = ["<script>alert(1)</script>"] * 20 + ["B"] * 20
    agente = AgenteNoSupervisado(df).ejecutar()

    informe = agente.informe_html()

    assert "<h1>Informe del agente no supervisado</h1>" in informe
    assert "Plan y decisiones" in informe
    assert "Método del codo" in informe
    assert "K-Means" in informe
    assert "Bitácora de razonamiento" in informe
    assert "<script>" not in informe
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in informe
