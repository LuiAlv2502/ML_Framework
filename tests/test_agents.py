from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import AgenteEDA


def _ejecutar(df, objetivo=None):
    agente = AgenteEDA(df, objetivo=objetivo, cliente=_ClienteQueSigueElPlan()).ejecutar()
    agente.cerrar_figuras()
    return agente


def _respuesta(nombre, args):
    from types import SimpleNamespace as NS
    llamada = NS(id="c", function=NS(name=nombre, arguments=json.dumps(args)))
    return NS(choices=[NS(message=NS(content=None, tool_calls=[llamada]))])


class _ClienteFalso:
    """Cliente que imita chat.completions.create con un guion de llamadas a herramientas."""

    def __init__(self, guion):
        self.guion = list(guion)
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return _respuesta(*self.guion.pop(0))


class _ClienteQueSigueElPlan(_ClienteFalso):
    """Imita a un LLM que acepta el plan sugerido en el primer mensaje."""

    def __init__(self):
        super().__init__([])

    def create(self, messages, **kwargs):
        if not self.guion:
            plan = json.loads(messages[1]["content"])["plan_sugerido_por_reglas"]
            self.guion = [
                ("ejecutar_paso" if p["sugerido"] else "omitir_paso", {"clave": p["clave"], "justificacion": p["razon"]})
                for p in plan
            ] + [("finalizar", {"analisis": "ok"})]
        return super().create()


def _guion_completo(omitir_duplicados=True):
    claves = ["estructura", "irrelevantes", "renombrar", "duplicados", "faltantes",
              "atipicos", "distribuciones", "relaciones", "objetivo"]
    guion = []
    for c in claves:
        if c == "renombrar" or (c == "duplicados" and omitir_duplicados):
            guion.append(("omitir_paso", {"clave": c, "justificacion": "no aplica"}))
        else:
            guion.append(("ejecutar_paso", {"clave": c, "justificacion": "relevante"}))
    guion.append(("finalizar", {"analisis": "Datos listos con cautela."}))
    return guion


def test_agente_llm_ejecuta_el_plan_que_decide_el_modelo():
    df = pd.read_csv(ROOT / "data" / "diabetes.csv")
    agente = AgenteEDA(df, objetivo="Outcome", cliente=_ClienteFalso(_guion_completo())).ejecutar()
    agente.cerrar_figuras()

    assert agente.sintesis["analisis_llm"] == "Datos listos con cautela."
    assert len(agente.pasos) == 7
    assert {i["clave"]: i["ejecutar"] for i in agente.plan}["duplicados"] is False


def test_agente_cae_a_reglas_si_el_llm_no_termina():
    df = pd.read_csv(ROOT / "data" / "diabetes.csv")
    cliente = _ClienteFalso([("finalizar", {"analisis": "x"})])

    agente = AgenteEDA(df, objetivo="Outcome", cliente=cliente, max_iteraciones=1).ejecutar()
    agente.cerrar_figuras()

    assert agente.modo == "reglas"
    assert agente.motivo_respaldo
    assert len(agente.pasos) > 0
    assert agente.sintesis["analisis_llm"] == ""


def test_agente_sin_api_key_usa_reglas_y_explica_el_motivo(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    df = pd.read_csv(ROOT / "data" / "diabetes.csv")

    agente = AgenteEDA(df).ejecutar()
    agente.cerrar_figuras()

    assert agente.modo == "reglas"
    assert "GEMINI_API_KEY" in agente.motivo_respaldo or "openai" in agente.motivo_respaldo
    assert len(agente.pasos) > 0


def test_agente_no_modifica_los_datos_originales():
    df = pd.read_csv(ROOT / "data" / "diabetes.csv")
    copia = df.copy()

    _ejecutar(df, objetivo="Outcome")

    pd.testing.assert_frame_equal(df, copia)


def test_agente_detecta_ceros_que_son_datos_faltantes():
    df = pd.read_csv(ROOT / "data" / "diabetes.csv")

    agente = _ejecutar(df, objetivo="Outcome")

    sospechosos = agente.percepcion["ceros_sospechosos"]
    assert {"Glucose", "BloodPressure", "Insulin"} <= set(sospechosos)
    assert "Pregnancies" not in sospechosos  # 0 embarazos es un valor válido


def test_datos_para_modelos_conservan_nulos_para_imputarlos_despues():
    df = pd.read_csv(ROOT / "data" / "diabetes.csv")
    df.loc[:10, "Glucose"] = np.nan
    agente = _ejecutar(df, objetivo="Outcome")

    preparados = agente.datos_para_modelos

    assert preparados["Glucose"].isna().any()
    assert df["Glucose"].isna().any()
    assert preparados["Outcome"].notna().all()
    assert df["Outcome"].notna().all()
    assert "Outcome" in preparados.columns


def test_datos_para_modelos_marca_ceros_sospechosos_tras_renombrar_columnas():
    df = pd.read_csv(ROOT / "data" / "diabetes.csv").rename(
        columns={"Glucose": "Glucose level"}
    )
    agente = _ejecutar(df, objetivo="Outcome")

    preparados = agente.datos_para_modelos

    assert "Glucose_level" in preparados.columns
    assert preparados["Glucose_level"].isna().sum() == int((df["Glucose level"] == 0).sum())


def test_agente_omite_pasos_innecesarios_con_su_razon():
    df = pd.read_csv(ROOT / "data" / "drug200.csv")

    agente = _ejecutar(df)

    omitidos = {p["paso"]: p["razon"] for p in agente.plan if not p["ejecutar"]}
    assert "Tratar valores faltantes" in omitidos
    assert "Analizar la variable objetivo" in omitidos
    assert all(razon for razon in omitidos.values())


def test_agente_elimina_identificadores_en_la_copia():
    df = pd.DataFrame({"id": range(40), "x": np.arange(40) % 7, "y": np.arange(40) % 5 * 1.5})

    agente = _ejecutar(df)

    assert "id" not in agente.datos_limpios.columns  # identificador eliminado
    assert "id" in df.columns  # el original no cambia


def test_agente_elimina_duplicados_en_la_copia():
    df = pd.DataFrame({"x": np.arange(40) % 7, "y": np.arange(40) * 1.5})
    df = pd.concat([df, df.iloc[[1, 2]]], ignore_index=True)  # 2 filas duplicadas

    agente = _ejecutar(df)

    assert len(agente.datos_limpios) == 40
    assert len(df) == 42  # el original no cambia


def test_agente_identifica_tipo_de_problema():
    df = pd.read_csv(ROOT / "data" / "drug200.csv")

    agente = _ejecutar(df, objetivo="Drug")

    paso_objetivo = agente.pasos[-1]
    assert "clasificación" in paso_objetivo.observacion


@pytest.mark.parametrize("archivo", ["drug200.csv", "Mall_Customers.csv", "water_potability.csv"])
def test_agente_genera_informe_html(archivo, tmp_path):
    df = pd.read_csv(ROOT / "data" / archivo)
    agente = AgenteEDA(df, cliente=_ClienteQueSigueElPlan()).ejecutar()

    ruta = agente.guardar_informe(tmp_path / "informe.html")
    contenido = Path(ruta).read_text(encoding="utf-8")
    agente.cerrar_figuras()

    assert "Bitácora de razonamiento" in contenido
    assert "data:image/png;base64" in contenido
    assert len(agente.pasos) >= 4


def test_agente_rechaza_objetivo_inexistente():
    df = pd.read_csv(ROOT / "data" / "drug200.csv")

    with pytest.raises(KeyError):
        AgenteEDA(df, objetivo="no_existe")
