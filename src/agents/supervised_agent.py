"""Base común de los agentes supervisados (clasificación y regresión).

El LLM solo decide (qué columna predecir, qué columnas excluir, qué algoritmos e hiperparámetros
probar y cuál es el mejor); Python valida cada decisión y entrena con las clases del framework
usando varias particiones (validación por repetición de hold-out). Al LLM solo se le envían
perfiles agregados y métricas, nunca filas. Si el LLM no está disponible, se usa un respaldo por
reglas. Las partes propias de cada problema están en AgenteClasificacion y AgenteRegresion.
"""
from __future__ import annotations

import html
import importlib.util
import json
import re
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from . import llm

MAX_CLASES = 30
MIN_FILAS = 10
MAX_VALIDACIONES = 12
MIN_FAMILIAS = 3
MAX_FILAS_SVR = 5000

FAMILIAS = {
    "knn": "vecinos",
    "decision_tree": "árbol",
    "random_forest": "ensamble bagging",
    "gradient_boosting": "ensamble boosting",
    "adaboost": "ensamble boosting",
    "xgboost": "ensamble boosting",
    "regresion_lineal_multiple": "lineal",
    "lasso": "lineal",
    "lasso_cv": "lineal",
    "ridge": "lineal",
    "ridge_cv": "lineal",
    "svr": "kernel",
}

PROFUNDIDAD = ("entero_o_nulo", 1, 30)
DIVISION = ("entero", 2, 20)
ARBOLES = ("entero", 10, 500)


class AgenteSupervisado(ABC):
    """Ciclo agéntico común; las subclases definen la configuración y las reglas del problema."""

    TIPO: str = ""
    PROBLEMA: str = ""
    # clase del framework, métrica principal, métricas, algoritmos con sus hiperparámetros y nombres
    CONFIGURACION: dict[str, Any] = {}

    def __init__(
        self,
        datos: pd.DataFrame,
        repeticiones: int = 5,
        test_size: float = 0.25,
        semilla: int = 42,
        modelo: str = llm.MODELO_POR_DEFECTO,
        cliente: Any = None,
        max_iteraciones: int = 24,
        objetivo: Any = None,
    ) -> None:
        if not isinstance(datos, pd.DataFrame):
            raise TypeError("El agente recibe un pandas.DataFrame.")
        if datos.empty or datos.shape[1] < 2:
            raise ValueError("Se necesitan datos con al menos dos columnas.")
        if repeticiones < 2:
            raise ValueError("Se necesitan al menos dos repeticiones para validar.")
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        if max_iteraciones < 1:
            raise ValueError("max_iteraciones debe ser mayor que cero.")
        if objetivo is not None and objetivo not in datos.columns:
            raise KeyError(f"La variable objetivo '{objetivo}' no existe.")

        self.original = datos.copy()
        self.tipo = self.TIPO
        self.repeticiones = int(repeticiones)
        self.test_size = float(test_size)
        self.semillas = [int(semilla) + i for i in range(self.repeticiones)]
        self.modelo = modelo
        self.cliente = cliente
        self.max_iteraciones = max_iteraciones
        self._config = self.CONFIGURACION
        if objetivo is not None:
            motivo = self._motivo_objetivo_invalido(objetivo)
            if motivo:
                raise ValueError(motivo)
        self.objetivo_usuario = objetivo
        self._reiniciar()

    # ==================================================================
    # Estado
    # ==================================================================
    def _reiniciar(self) -> None:
        self.modo = "llm"
        self.motivo_respaldo = ""
        self.percepcion: dict[str, Any] = {}
        self.plan: list[dict[str, Any]] = []
        self.decisiones: list[dict[str, str]] = []
        self.bitacora: list[dict[str, str]] = []
        self.objetivo: str | None = None
        self.justificacion_objetivo = ""
        self.excluidas: list[str] = []
        self.predictores: list[str] = []
        self.datos: pd.DataFrame | None = None
        self.filas_sin_objetivo = 0
        self.validaciones: dict[str, dict[str, Any]] = {}
        self.mejor: str | None = None
        self.eleccion_llm: str | None = None
        self.analisis_llm = ""
        self.autoevaluacion: list[str] = []
        self.supera_linea_base: bool | None = None
        self.uso_llm: dict[str, int] = {}

    @property
    def metrica(self) -> str:
        return self._config["metrica"]

    @property
    def maximizar(self) -> bool:
        return self._config["maximizar"]

    def nombre_algoritmo(self, algoritmo: str) -> str:
        return self._config["nombres"].get(algoritmo, algoritmo)

    def _registrar(self, etapa: str, observacion: str, razonamiento: str, accion: str) -> None:
        self.bitacora.append(
            {"etapa": etapa, "observacion": observacion, "razonamiento": razonamiento, "accion": accion}
        )

    def _decidir(self, decision: str, justificacion: str) -> None:
        self.decisiones.append({"decision": decision, "justificacion": justificacion})

    # ==================================================================
    # Percepción
    # ==================================================================
    @staticmethod
    def _parece_id(serie: pd.Series, nombre: Any) -> bool:
        n = len(serie)
        if n <= 20 or serie.nunique(dropna=True) != n:
            return bool(re.search(r"^id$|_id$|^id_|^unnamed", str(nombre), re.I))
        nombre_id = re.search(r"id$|^id|codigo|^unnamed", str(nombre), re.I)
        consecutivo = pd.api.types.is_integer_dtype(serie) and serie.is_monotonic_increasing
        texto_unico = not pd.api.types.is_numeric_dtype(serie)
        return bool(nombre_id or consecutivo or texto_unico)

    def _motivo_objetivo_invalido(self, columna: Any) -> str | None:
        if columna not in self.original.columns:
            return f"La columna '{columna}' no existe."
        serie = self.original[columna]
        if self._parece_id(serie, columna):
            return f"'{columna}' parece un identificador; no tiene sentido predecirla."
        filas = int(serie.notna().sum())
        if filas < MIN_FILAS:
            return f"'{columna}' solo tiene {filas} valores no nulos."
        unicos = int(serie.nunique(dropna=True))
        return self._motivo_por_tipo(serie, columna, unicos)

    # ==================================================================
    # Partes propias de cada problema
    # ==================================================================
    @abstractmethod
    def _motivo_por_tipo(self, serie: pd.Series, columna: Any, unicos: int) -> str | None:
        """Motivo por el que la columna no sirve como objetivo de este problema (None si sirve)."""

    @abstractmethod
    def _describir_objetivo(self, y: pd.Series) -> dict[str, Any]:
        """Resumen del objetivo fijado que se devuelve al LLM."""

    @abstractmethod
    def _comparar_linea_base(self, resumen: dict[str, dict[str, float]], y: pd.Series) -> tuple[bool, str]:
        """Indica si el mejor modelo supera una línea base trivial y lo explica."""

    def _argumentos_particion(self) -> dict[str, Any]:
        return {}

    # ==================================================================
    # Percepción
    # ==================================================================
    def percibir(self) -> dict[str, Any]:
        df = self.original
        columnas = []
        for posicion, columna in enumerate(df.columns):
            serie = df[columna]
            unicos = int(serie.nunique(dropna=True))
            info: dict[str, Any] = {
                "columna": str(columna),
                "posicion": posicion,
                "tipo": str(serie.dtype),
                "numerica": bool(pd.api.types.is_numeric_dtype(serie)),
                "valores_unicos": unicos,
                "nulos_pct": round(float(serie.isna().mean()) * 100, 2),
                "parece_identificador": self._parece_id(serie, columna),
            }
            if 2 <= unicos <= MAX_CLASES:
                proporciones = serie.value_counts(normalize=True)
                info["proporcion_clase_mayoritaria"] = round(float(proporciones.iloc[0]), 4)
                info["proporcion_clase_minoritaria"] = round(float(proporciones.iloc[-1]), 4)
            columnas.append(info)
        candidatas = [str(c) for c in df.columns if self._motivo_objetivo_invalido(c) is None]
        self.percepcion = {
            "tipo_problema": self.tipo,
            "filas": int(len(df)),
            "columnas": columnas,
            "candidatas_validas_como_objetivo": candidatas,
            "objetivo_indicado_por_usuario": (
                None if self.objetivo_usuario is None else str(self.objetivo_usuario)
            ),
            "metrica_principal": self.metrica,
            "repeticiones_de_validacion": self.repeticiones,
            "proporcion_prueba": self.test_size,
        }
        self._registrar(
            "percepción",
            f"{len(df)} filas, {df.shape[1]} columnas; {len(candidatas)} candidatas a objetivo.",
            "El tipo, la cardinalidad y la posición de cada columna orientan la elección del objetivo.",
            "Construir un perfil agregado sin enviar filas al LLM.",
        )
        return self.percepcion

    # ==================================================================
    # Planificación
    # ==================================================================
    def planificar(self) -> list[dict[str, Any]]:
        p = self.percepcion or self.percibir()
        plan = []
        for algoritmo in self._config["algoritmos"]:
            sugerido, razon = True, f"Algoritmo de la familia '{FAMILIAS[algoritmo]}' disponible."
            if algoritmo == "xgboost" and importlib.util.find_spec("xgboost") is None:
                sugerido, razon = False, "El paquete 'xgboost' no está instalado."
            elif algoritmo == "svr" and p["filas"] > MAX_FILAS_SVR:
                sugerido, razon = False, f"SVR escala mal con más de {MAX_FILAS_SVR} filas."
            plan.append({
                "clave": algoritmo,
                "paso": self.nombre_algoritmo(algoritmo),
                "familia": FAMILIAS[algoritmo],
                "sugerido": sugerido,
                "razon": razon,
            })
        self.plan = plan
        self._registrar(
            "planificación",
            f"{sum(item['sugerido'] for item in plan)} de {len(plan)} algoritmos son factibles.",
            "La propuesta orienta al LLM; Python rechaza los algoritmos no factibles.",
            "Preparar las opciones de validación.",
        )
        return plan

    def _familias_requeridas(self) -> int:
        factibles = {item["familia"] for item in self.plan if item["sugerido"]}
        return min(MIN_FAMILIAS, len(factibles))

    def _cerrar_plan(self) -> None:
        for item in self.plan:
            justificaciones = [
                v["justificacion"] for v in self.validaciones.values() if v["algoritmo"] == item["clave"]
            ]
            item["ejecutar"] = bool(justificaciones)
            if justificaciones:
                item["razon"] = justificaciones[0]
            elif item["sugerido"]:
                item["razon"] = "El agente no lo consideró necesario."

    def _excluir_identificadores(self, objetivo: Any) -> list[Any]:
        return [
            c for c in self.original.columns
            if c != objetivo and self._parece_id(self.original[c], c)
        ]

    # ==================================================================
    # Herramientas (validadas por Python)
    # ==================================================================
    def _fijar_objetivo(self, columna: Any, excluir: list[Any] | None, justificacion: str) -> dict[str, Any]:
        motivo = self._motivo_objetivo_invalido(columna)
        if motivo:
            raise ValueError(motivo)
        excluir = list(dict.fromkeys(excluir or []))
        invalidas = [c for c in excluir if c not in self.original.columns or c == columna]
        if invalidas:
            raise ValueError(f"Columnas a excluir inválidas: {invalidas}")
        predictores = [c for c in self.original.columns if c != columna and c not in excluir]
        if not predictores:
            raise ValueError("No quedan variables predictoras tras las exclusiones.")

        datos = self.original.dropna(subset=[columna]).reset_index(drop=True)
        self.objetivo = columna
        self.justificacion_objetivo = justificacion
        self.excluidas = excluir
        self.predictores = predictores
        self.datos = datos
        self.filas_sin_objetivo = len(self.original) - len(datos)

        resumen: dict[str, Any] = {
            "objetivo": str(columna),
            "predictores": [str(c) for c in predictores],
            "excluidas": [str(c) for c in excluir],
            "filas_utilizables": len(datos),
            "filas_descartadas_por_objetivo_nulo": self.filas_sin_objetivo,
        }
        y = datos[columna]
        resumen.update(self._describir_objetivo(y))
        self._decidir(f"Variable objetivo: {columna}", justificacion)
        if excluir:
            self._decidir(f"Columnas excluidas: {', '.join(map(str, excluir))}", justificacion)
        self._registrar(
            "elección del objetivo",
            json.dumps(resumen, ensure_ascii=False, default=str),
            justificacion,
            "Fijar el objetivo y los predictores.",
        )
        return resumen

    def _validar_parametros(self, algoritmo: str, parametros: Any) -> dict[str, Any]:
        especificacion = self._config["algoritmos"][algoritmo]
        if parametros is None:
            return {}
        if not isinstance(parametros, dict):
            raise ValueError("parametros debe ser un objeto.")
        desconocidos = sorted(set(parametros) - set(especificacion))
        if desconocidos:
            raise ValueError(
                f"Parámetros no permitidos para {algoritmo}: {desconocidos}. "
                f"Permitidos: {list(especificacion)}"
            )
        limpios = {}
        for nombre, valor in parametros.items():
            tipo, *rango = especificacion[nombre]
            if tipo == "opcion":
                if valor not in rango[0]:
                    raise ValueError(f"{nombre} debe ser uno de {rango[0]}.")
            elif tipo == "entero_o_nulo" and valor in (None, 0):
                valor = None
            else:
                if isinstance(valor, bool) or not isinstance(valor, (int, float)):
                    raise ValueError(f"{nombre} debe ser numérico.")
                if tipo != "real":
                    if float(valor) != int(valor):
                        raise ValueError(f"{nombre} debe ser entero.")
                    valor = int(valor)
                else:
                    valor = float(valor)
                minimo, maximo = rango
                if not minimo <= valor <= maximo:
                    raise ValueError(f"{nombre} debe estar entre {minimo} y {maximo}.")
            limpios[nombre] = valor
        return limpios

    def _etiqueta(self, algoritmo: str, parametros: dict[str, Any]) -> str:
        if not parametros:
            return algoritmo
        partes = ", ".join(f"{k}={v}" for k, v in sorted(parametros.items()))
        return f"{algoritmo}({partes})"

    def _validar_modelo(self, algoritmo: str, parametros: dict[str, Any], justificacion: str) -> dict[str, Any]:
        if self.objetivo is None:
            raise ValueError("Primero elige la variable objetivo.")
        if algoritmo not in self._config["algoritmos"]:
            raise ValueError(f"Algoritmo inválido. Opciones: {list(self._config['algoritmos'])}")
        propuesta = next((item for item in self.plan if item["clave"] == algoritmo), None)
        if propuesta is not None and not propuesta["sugerido"]:
            raise ValueError(f"'{algoritmo}' no es factible: {propuesta['razon']}")
        parametros = self._validar_parametros(algoritmo, parametros)
        etiqueta = self._etiqueta(algoritmo, parametros)
        if etiqueta in self.validaciones:
            raise ValueError(f"'{etiqueta}' ya fue validado; prueba otra configuración.")
        if len(self.validaciones) >= MAX_VALIDACIONES:
            raise ValueError(f"Se alcanzó el máximo de {MAX_VALIDACIONES} validaciones; finaliza.")

        clase = self._config["clase"]
        resultados = []
        for semilla in self.semillas:
            argumentos = dict(
                target=self.objetivo,
                test_size=self.test_size,
                random_state=semilla,
                scale=True,
                feature_columns=self.predictores,
                **self._argumentos_particion(),
            )
            resultados.append(getattr(clase(self.datos, **argumentos), algoritmo)(**parametros))

        metricas = self._config["metricas"]
        por_semilla = pd.DataFrame([{m: r["metricas"][m] for m in metricas} for r in resultados])
        por_semilla.insert(0, "semilla", self.semillas)
        resumen = {
            m: {"media": float(por_semilla[m].mean()), "desv": float(por_semilla[m].std(ddof=0))}
            for m in metricas
        }
        self.validaciones[etiqueta] = {
            "algoritmo": algoritmo,
            "nombre": self.nombre_algoritmo(algoritmo),
            "parametros": parametros,
            "justificacion": justificacion,
            "resumen": resumen,
            "por_semilla": por_semilla,
            "detalle": resultados[0],
        }
        self._decidir(f"Validar {etiqueta}", justificacion)
        self._registrar(
            f"validación: {etiqueta}",
            json.dumps(self._resumen_metricas(resumen), ensure_ascii=False),
            justificacion,
            f"Entrenar y evaluar en {len(self.semillas)} particiones distintas.",
        )
        return {"modelo": etiqueta, "metricas": self._resumen_metricas(resumen)}

    @staticmethod
    def _resumen_metricas(resumen: dict[str, dict[str, float]]) -> dict[str, str]:
        return {m: f"{v['media']:.4f} ± {v['desv']:.4f}" for m, v in resumen.items()}

    def ranking(self) -> pd.DataFrame:
        """Tabla de modelos validados ordenada por la métrica principal."""
        filas = []
        for etiqueta, validacion in self.validaciones.items():
            fila = {"Modelo": etiqueta, "Algoritmo": validacion["nombre"]}
            for metrica, valores in validacion["resumen"].items():
                fila[metrica] = valores["media"]
            fila[f"Desv. {self.metrica}"] = validacion["resumen"][self.metrica]["desv"]
            filas.append(fila)
        if not filas:
            return pd.DataFrame()
        return pd.DataFrame(filas).sort_values(
            self.metrica, ascending=not self.maximizar, ignore_index=True
        )

    def _mejor_por_metrica(self) -> str | None:
        tabla = self.ranking()
        return None if tabla.empty else str(tabla.iloc[0]["Modelo"])

    # ==================================================================
    # Bucle con LLM
    # ==================================================================
    def _herramientas(self) -> list[dict[str, Any]]:
        tipos_json = {"entero": "integer", "entero_o_nulo": "integer", "real": "number"}
        propiedades: dict[str, Any] = {}
        for especificacion in self._config["algoritmos"].values():
            for nombre, (tipo, *rango) in especificacion.items():
                if tipo == "opcion":
                    propiedades[nombre] = {"type": "string", "enum": rango[0]}
                else:
                    descripcion = f"Entre {rango[0]} y {rango[1]}."
                    if tipo == "entero_o_nulo":
                        descripcion += " Usa 0 para 'sin límite'."
                    propiedades[nombre] = {"type": tipos_json[tipo], "description": descripcion}
        columnas = [str(c) for c in self.original.columns]
        justificacion = {"type": "string", "description": "Motivo breve de la decisión."}
        permitidos = {
            algoritmo: list(especificacion)
            for algoritmo, especificacion in self._config["algoritmos"].items()
        }
        return [
            {"type": "function", "function": {
                "name": "elegir_variable_objetivo",
                "description": (
                    "Fija la variable a predecir y, opcionalmente, columnas a excluir de los predictores "
                    "(identificadores, fugas de información). Debe llamarse antes de validar modelos."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "columna": {"type": "string", "enum": columnas},
                        "excluir": {"type": "array", "items": {"type": "string", "enum": columnas}},
                        "justificacion": justificacion,
                    },
                    "required": ["columna", "justificacion"],
                },
            }},
            {"type": "function", "function": {
                "name": "validar_modelo",
                "description": (
                    "Entrena y evalúa un algoritmo en varias particiones y devuelve media ± desviación "
                    f"de las métricas. Parámetros permitidos por algoritmo: {json.dumps(permitidos)}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "algoritmo": {"type": "string", "enum": list(self._config["algoritmos"])},
                        "parametros": {"type": "object", "properties": propiedades},
                        "justificacion": justificacion,
                    },
                    "required": ["algoritmo", "justificacion"],
                },
            }},
            {"type": "function", "function": {
                "name": "finalizar",
                "description": "Declara el mejor modelo validado y entrega la conclusión.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mejor_modelo": {
                            "type": "string",
                            "description": "Etiqueta exacta de un modelo validado.",
                        },
                        "analisis": {"type": "string", "description": "Conclusión en español."},
                    },
                    "required": ["mejor_modelo", "analisis"],
                },
            }},
        ]

    def _columna_real(self, nombre: Any) -> Any:
        # El LLM envía nombres como texto; las columnas del CSV pueden no serlo.
        return {str(c): c for c in self.original.columns}.get(str(nombre), nombre)

    def _despachar(self, nombre: str, args: dict[str, Any]) -> dict[str, Any]:
        justificacion = str(args.get("justificacion", "")).strip() or "Decisión del agente."
        try:
            if nombre == "elegir_variable_objetivo":
                if self.validaciones:
                    return {"error": "El objetivo ya no puede cambiarse: hay modelos validados."}
                columna = self._columna_real(args.get("columna"))
                if self.objetivo_usuario is not None and columna != self.objetivo_usuario:
                    return {"error": (
                        f"El usuario fijó el objetivo '{self.objetivo_usuario}'; "
                        "solo puedes ajustar las columnas a excluir."
                    )}
                excluir = [self._columna_real(c) for c in args.get("excluir") or []]
                return {"ok": True, **self._fijar_objetivo(columna, excluir, justificacion)}
            if nombre == "validar_modelo":
                return {"ok": True, **self._validar_modelo(
                    args.get("algoritmo"), args.get("parametros"), justificacion
                )}
            if nombre == "finalizar":
                return self._finalizar(args)
        except Exception as error:  # el LLM recibe el error y puede corregir su decisión
            return {"error": f"{type(error).__name__}: {error}"}
        return {"error": f"Herramienta desconocida: {nombre}"}

    def _finalizar(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.objetivo is None:
            return {"error": "Primero elige la variable objetivo."}
        if len(self.validaciones) < 2:
            return {"error": "Valida al menos dos modelos antes de finalizar."}
        familias = {FAMILIAS[v["algoritmo"]] for v in self.validaciones.values()}
        requeridas = self._familias_requeridas()
        if len(familias) < requeridas and len(self.validaciones) < MAX_VALIDACIONES:
            return {"error": (
                f"Valida al menos {requeridas} familias de algoritmos distintas; "
                f"llevas {sorted(familias)}."
            )}
        elegido = str(args.get("mejor_modelo", "")).strip()
        if elegido not in self.validaciones:
            return {"error": f"'{elegido}' no fue validado. Opciones: {list(self.validaciones)}"}
        self.eleccion_llm = elegido
        self.mejor = self._mejor_por_metrica()
        self.analisis_llm = str(args.get("analisis", "")).strip()
        self._decidir(f"Mejor modelo según el LLM: {elegido}", "Conclusión del agente.")
        return {"ok": True}

    def _ejecutar_con_llm(self) -> None:
        cliente = self.cliente or llm.crear_cliente()
        self.percibir()
        self.planificar()
        if self.objetivo_usuario is not None:
            self._fijar_objetivo(
                self.objetivo_usuario,
                self._excluir_identificadores(self.objetivo_usuario),
                "Indicado por el usuario; se excluyen identificadores.",
            )
        sentido = "mayor" if self.maximizar else "menor"
        mensajes: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"Eres un agente de aprendizaje supervisado para un problema de {self.PROBLEMA}. "
                    "No escribas ni ejecutes código: usa solo las herramientas; Python entrena y mide. "
                    "1) Si 'objetivo_indicado_por_usuario' tiene valor, ese objetivo ya está fijado y no "
                    "puede cambiarse; solo puedes ajustar exclusiones con elegir_variable_objetivo. Si no, "
                    "con el perfil de columnas (sin filas) elige con elegir_variable_objetivo la "
                    "variable que tiene más sentido predecir: considera el nombre, el tipo, la "
                    "cardinalidad y que suele ser la última columna; excluye identificadores y "
                    "columnas que filtren la respuesta. Si la herramienta rechaza la columna, elige otra "
                    f"de las candidatas válidas. 2) Valida algoritmos de al menos {MIN_FAMILIAS} familias "
                    "distintas (ver plan_sugerido_por_reglas; los no sugeridos no son factibles) y ajusta "
                    "hiperparámetros de los más prometedores. La métrica principal es "
                    f"{self.metrica} ({sentido} es mejor); considera también la desviación entre "
                    "particiones (estabilidad). 3) Llama a finalizar con el mejor modelo y una "
                    "conclusión en español basada solo en las métricas devueltas. No inventes cifras. "
                    "Puedes llamar varias herramientas en un mismo turno si no dependen entre sí."
                ),
            },
            {"role": "user", "content": json.dumps({
                "perfil_de_datos": self.percepcion,
                "plan_sugerido_por_reglas": self.plan,
            }, ensure_ascii=False, default=str)},
        ]
        self.uso_llm = llm.bucle_herramientas(
            cliente, self.modelo, mensajes, self._herramientas(), self._despachar, self.max_iteraciones
        )
        self._cerrar_plan()
        self._autoevaluar()

    # ==================================================================
    # Respaldo por reglas
    # ==================================================================
    def _ejecutar_reglas(self) -> None:
        self.percibir()
        self.planificar()
        if self.objetivo_usuario is not None:
            objetivo = self.objetivo_usuario
            justificacion = "Indicado por el usuario; se excluyen identificadores."
        else:
            candidatas = self.percepcion["candidatas_validas_como_objetivo"]
            if not candidatas:
                raise ValueError(f"Ninguna columna es válida como objetivo de {self.PROBLEMA}.")
            objetivo = self._columna_real(candidatas[-1])
            justificacion = (
                "Regla: se toma la última columna válida como objetivo (convención habitual) "
                "y se excluyen identificadores."
            )
        self._fijar_objetivo(objetivo, self._excluir_identificadores(objetivo), justificacion)
        for item in self.plan:
            if not item["sugerido"]:
                self._registrar(f"omisión: {item['clave']}", item["razon"], "No es factible.", "Omitido.")
                continue
            try:
                self._validar_modelo(
                    item["clave"], {}, "Regla: validar cada algoritmo con su configuración estándar."
                )
            except Exception as error:
                self._registrar(
                    f"error: {item['clave']}", str(error), "El algoritmo no pudo validarse.", "Omitido."
                )
        if not self.validaciones:
            raise ValueError("Ningún algoritmo pudo validarse con estos datos.")
        self.mejor = self._mejor_por_metrica()
        self._decidir(f"Mejor modelo: {self.mejor}", f"Mejor {self.metrica} medio entre particiones.")
        self._cerrar_plan()
        self._autoevaluar()

    # ==================================================================
    # Autoevaluación
    # ==================================================================
    def _autoevaluar(self) -> None:
        """Contrasta el mejor modelo con una línea base y revisa su estabilidad."""
        resumen = self.validaciones[self.mejor]["resumen"]
        y = self.datos[self.objetivo]
        self.supera_linea_base, referencia = self._comparar_linea_base(resumen, y)
        hallazgos = [
            f"Mejor modelo '{self.mejor}': {referencia}"
            + ("." if self.supera_linea_base else "; NO supera la línea base, el modelo no aporta valor.")
        ]
        principal = resumen[self.metrica]
        if principal["media"] and principal["desv"] / abs(principal["media"]) > 0.10:
            hallazgos.append(
                f"{self.metrica} varía más de un 10 % entre particiones: el resultado es inestable; "
                "conviene más datos o más particiones."
            )
        else:
            hallazgos.append(f"{self.metrica} es estable entre particiones.")
        tabla = self.ranking()
        if len(tabla) > 1:
            segundo = tabla.iloc[1]
            diferencia = abs(float(tabla.iloc[0][self.metrica]) - float(segundo[self.metrica]))
            if diferencia < principal["desv"]:
                hallazgos.append(
                    f"La diferencia con '{segundo['Modelo']}' ({diferencia:.4f}) es menor que la desviación "
                    "entre particiones: ambos modelos son prácticamente equivalentes."
                )
        if self.eleccion_llm and self.eleccion_llm != self.mejor:
            hallazgos.append(
                f"El LLM eligió '{self.eleccion_llm}', pero '{self.mejor}' tiene mejor {self.metrica} medio."
            )
        self.autoevaluacion = hallazgos
        self._registrar(
            "autoevaluación",
            f"{len(self.validaciones)} configuraciones validadas.",
            " ".join(hallazgos),
            "Verificar la calidad del resultado antes de concluir.",
        )

    # ==================================================================
    # Punto de entrada
    # ==================================================================
    def ejecutar(self) -> AgenteSupervisado:
        self._reiniciar()
        if self.cliente is None:
            self.motivo_respaldo = llm.motivo_no_disponible() or ""
        if not self.motivo_respaldo:
            try:
                self._ejecutar_con_llm()
                return self
            except Exception as error:  # red, cuota, respuesta inesperada...
                motivo = llm.describir_motivo(error)
                self._reiniciar()
                self.motivo_respaldo = motivo
        self.modo = "reglas"
        self._ejecutar_reglas()
        return self

    # ==================================================================
    # Informe HTML
    # ==================================================================
    def informe_html(self) -> str:
        if self.mejor is None:
            raise RuntimeError("Ejecute el agente antes de generar el informe.")
        e = html.escape
        problema = self.PROBLEMA
        partes = [
            "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
            f"<title>Informe del agente de {problema}</title><style>"
            "body{font-family:Arial,sans-serif;max-width:1000px;margin:32px auto;padding:0 18px;"
            "line-height:1.55;color:#20242b}h1,h2{color:#174a72}"
            "table{border-collapse:collapse;font-size:14px;margin:10px 0}"
            "td,th{border:1px solid #cbd3dc;padding:6px 9px;text-align:left}th{background:#edf2f7}"
            "</style></head><body>",
            f"<h1>Informe del agente de {problema}</h1>",
            f"<p><b>Modo:</b> {e(self.modo)}" + (f" — {e(self.modelo)}" if self.modo == "llm" else "") + "</p>",
        ]
        if self.motivo_respaldo:
            partes.append(f"<p><b>Motivo del respaldo:</b> {e(self.motivo_respaldo)}</p>")
        if self.uso_llm:
            partes.append(f"<p><b>Uso del LLM:</b> {e(llm.resumen_uso(self.uso_llm))}</p>")
        partes.append(
            f"<p><b>Variable objetivo:</b> {e(str(self.objetivo))} · <b>Predictores:</b> "
            f"{len(self.predictores)} · <b>Excluidas:</b> {e(', '.join(map(str, self.excluidas)) or 'ninguna')} · "
            f"<b>Validación:</b> {self.repeticiones} particiones con {self.test_size:.0%} de prueba</p>"
        )
        partes.append("<h2>Plan y algoritmos</h2><ul>")
        for item in self.plan:
            estado = "Validado" if item.get("ejecutar") else "No validado"
            partes.append(
                f"<li><b>{e(item['paso'])}</b> ({e(item['familia'])}) — {estado}: {e(str(item['razon']))}</li>"
            )
        partes.append("</ul><h2>Decisiones del agente</h2><ul>")
        partes.extend(
            f"<li><b>{e(d['decision'])}</b> — {e(d['justificacion'])}</li>" for d in self.decisiones
        )
        partes.append("</ul>")
        partes.append(f"<h2>Ranking de modelos (métrica principal: {e(self.metrica)})</h2>")
        partes.append(self.ranking().to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}"))
        if self.analisis_llm:
            partes.append("<h2>Interpretación del LLM</h2>")
            partes.extend(f"<p>{e(p)}</p>" for p in self.analisis_llm.splitlines() if p.strip())
        partes.append("<h2>Autoevaluación del agente</h2><ul>")
        partes.extend(f"<li>{e(texto)}</li>" for texto in self.autoevaluacion)
        partes.append("</ul><h2>Bitácora de razonamiento</h2>")
        partes.append(pd.DataFrame(self.bitacora).to_html(index=False, border=0, escape=True))
        partes.append(
            "<p><i>Las métricas provienen de particiones hold-out repetidas; validar con conocimiento "
            "del dominio antes de usar el modelo en decisiones reales.</i></p></body></html>"
        )
        return "\n".join(partes)

    def guardar_informe(self, ruta: str = "informe_supervisado.html") -> str:
        with open(ruta, "w", encoding="utf-8") as archivo:
            archivo.write(self.informe_html())
        return ruta

    def resultado_mejor(self) -> dict[str, Any] | None:
        """Resultado detallado (primera partición) del mejor modelo."""
        if self.mejor is None:
            return None
        return self.validaciones[self.mejor]["detalle"]

    def mejores_por_semilla(self) -> pd.DataFrame:
        """Métrica principal por partición de cada modelo, en formato largo."""
        filas = [
            {"Modelo": etiqueta, "Semilla": fila["semilla"], self.metrica: fila[self.metrica]}
            for etiqueta, validacion in self.validaciones.items()
            for _, fila in validacion["por_semilla"].iterrows()
        ]
        return pd.DataFrame(filas)
