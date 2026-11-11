"""Agente que coordina análisis no supervisados con herramientas controladas."""
from __future__ import annotations

import html
import json
from typing import Any

import numpy as np
import pandas as pd

from unsupervised.clustering import ClusteringAnalysis
from unsupervised.pca import PCAAnalysis

from . import llm


class AgenteNoSupervisado:
    """Orquesta PCA y clustering; el código ejecuta los algoritmos, no el LLM."""

    CLAVES = ("pca", "codo", "kmeans", "hac")

    def __init__(
        self,
        datos: pd.DataFrame,
        modelo: str = llm.MODELO_POR_DEFECTO,
        cliente: Any = None,
        max_iteraciones: int = 16,
    ) -> None:
        if not isinstance(datos, pd.DataFrame):
            raise TypeError("El agente recibe un pandas.DataFrame.")
        if datos.empty:
            raise ValueError("No hay datos para analizar.")
        if max_iteraciones < 1:
            raise ValueError("max_iteraciones debe ser mayor que cero.")

        self.original = datos.copy()
        self.modelo = modelo
        self.cliente = cliente
        self.max_iteraciones = max_iteraciones
        self._clustering = ClusteringAnalysis(self.original)
        self._pca = PCAAnalysis(self.original)
        self._reiniciar()

    def _reiniciar(self) -> None:
        self.modo = "llm"
        self.motivo_respaldo = ""
        self.percepcion: dict[str, Any] = {}
        self.plan: list[dict[str, Any]] = []
        self.resultados: dict[str, dict[str, Any]] = {}
        self.bitacora: list[dict[str, str]] = []
        self.analisis_llm = ""
        self.autoevaluacion: list[str] = []
        self.uso_llm: dict[str, int] = {}
        self._decisiones: dict[str, tuple[bool, str]] = {}

    def _registrar(self, etapa: str, observacion: str, razonamiento: str, accion: str) -> None:
        self.bitacora.append(
            {
                "etapa": etapa,
                "observacion": observacion,
                "razonamiento": razonamiento,
                "accion": accion,
            }
        )

    def percibir(self) -> dict[str, Any]:
        df = self.original
        numericas = df.select_dtypes(include=np.number).columns.tolist()
        categoricas = [columna for columna in df.columns if columna not in numericas]
        self.percepcion = {
            "filas": int(len(df)),
            "columnas": int(df.shape[1]),
            "numericas": numericas,
            "categoricas": categoricas,
            "columnas_con_nulos": df.columns[df.isna().any()].tolist(),
            "columnas_constantes": [
                columna for columna in df.columns if df[columna].nunique(dropna=False) <= 1
            ],
            "max_clusters": max(0, min(10, len(df) - 1)),
        }
        self._registrar(
            "percepción",
            f"{len(df)} filas, {df.shape[1]} columnas; "
            f"{len(numericas)} numéricas y {len(categoricas)} categóricas.",
            "El tipo, cantidad y variación de las columnas determinan qué análisis son posibles.",
            "Construir un perfil agregado sin enviar filas al LLM.",
        )
        return self.percepcion

    def planificar(self) -> list[dict[str, Any]]:
        p = self.percepcion or self.percibir()
        columnas_utilizables = len(p["numericas"]) + len(p["categoricas"])
        puede_reducir = len(self.original) >= 2 and columnas_utilizables >= 2
        puede_agrupar = len(self.original) >= 3 and columnas_utilizables >= 1
        propuestas = [
            ("pca", "Reducir dimensiones con PCA", puede_reducir,
             "PCA ayuda a resumir variables y observar cuánta varianza conservan los componentes."),
            ("codo", "Calcular inercias para sugerir k", puede_agrupar,
             "El método del codo aporta una referencia para elegir el número de grupos."),
            ("kmeans", "Agrupar con K-Means", puede_agrupar,
             "K-Means permite comparar grupos compactos usando los datos preparados."),
            ("hac", "Agrupar con HAC", puede_agrupar and len(self.original) <= 2000,
             "HAC permite comparar una agrupación jerárquica; se limita a 2,000 filas por su costo."),
        ]
        self.plan = [
            {"clave": clave, "paso": paso, "sugerido": sugerido, "razon": razon}
            for clave, paso, sugerido, razon in propuestas
        ]
        self._registrar(
            "planificación",
            f"{sum(item['sugerido'] for item in self.plan)} de {len(self.plan)} análisis son factibles.",
            "La propuesta es una guía; el LLM puede escoger entre las herramientas disponibles.",
            "Preparar las opciones y sus límites antes de ejecutar algoritmos.",
        )
        return self.plan

    def _acciones(self) -> dict[str, Any]:
        return {
            "pca": self._ejecutar_pca,
            "codo": self._ejecutar_codo,
            "kmeans": self._ejecutar_kmeans,
            "hac": self._ejecutar_hac,
        }

    def _ejecutar_pca(self) -> dict[str, Any]:
        matriz, _ = self._pca.preparar_datos_numericos()
        n_componentes = min(2, matriz.shape[0], matriz.shape[1])
        if n_componentes < 2:
            raise ValueError("PCA requiere al menos dos filas y dos columnas utilizables.")
        resultado = self._pca.pca(n_components=n_componentes)
        varianza = [float(valor) for valor in resultado["varianza_explicada"]]
        cargas = resultado["cargas"]
        principales = {
            componente: {
                str(nombre): round(float(valor), 4)
                for nombre, valor in cargas[componente].abs().nlargest(5).items()
            }
            for componente in cargas.columns[:2]
        }
        resumen = {
            "varianza_por_componente": varianza,
            "varianza_acumulada": float(resultado["metrica"]),
            "variables_mas_influyentes": principales,
        }
        return {**resumen, "componentes": resultado["componentes"], "cargas": cargas}

    def _ejecutar_codo(self) -> dict[str, Any]:
        k_max = self.percepcion["max_clusters"]
        if k_max < 2:
            raise ValueError("Se requieren al menos tres filas para calcular el método del codo.")
        inercias = self._clustering.kmeans_codo(k_max=k_max)
        valores = {int(k): float(valor) for k, valor in inercias.items()}
        resumen = {"inercias": valores, "k_evaluados": list(valores)}
        return {**resumen, "serie": inercias}

    def _resumen_clusters(self, resultado: dict[str, Any], metodo: str, k: int) -> dict[str, Any]:
        df = resultado["resultado"]
        etiquetas = df["cluster"]
        tamanos = etiquetas.value_counts().sort_index()
        numericas = self.original.select_dtypes(include=np.number).columns.tolist()[:8]
        categoricas = self.original.select_dtypes(exclude=np.number).columns.tolist()[:5]
        perfil = (
            df.assign(cluster=etiquetas)
            .groupby("cluster")[numericas]
            .mean()
            .round(3)
            .to_dict(orient="index")
            if numericas else {}
        )
        perfil_categorico = {}
        for grupo, datos_grupo in df.groupby("cluster"):
            perfil_categorico[int(grupo)] = {}
            for columna in categoricas:
                frecuencias = datos_grupo[columna].value_counts(dropna=False, normalize=True)
                if not frecuencias.empty:
                    perfil_categorico[int(grupo)][columna] = {
                        "categoria_mas_frecuente": str(frecuencias.index[0]),
                        "proporcion": round(float(frecuencias.iloc[0]), 3),
                    }
        metricas: dict[str, Any] = {
            "algoritmo": metodo,
            "k_solicitado": k,
            "grupos_obtenidos": int(etiquetas.nunique()),
            "tamanos": {int(grupo): int(cantidad) for grupo, cantidad in tamanos.items()},
            "perfil_numerico": perfil,
            "perfil_categorico": perfil_categorico,
            "silhouette": float(resultado["silhouette"]),
        }
        if "inercia" in resultado:
            metricas["inercia"] = float(resultado["inercia"])
        if "correlacion_cofenetica" in resultado:
            metricas["correlacion_cofenetica"] = float(resultado["correlacion_cofenetica"])
        return {**metricas, "resultado": df}

    def _ejecutar_kmeans(self, n_clusters: int) -> dict[str, Any]:
        self._validar_k(n_clusters)
        return self._resumen_clusters(
            self._clustering.kmeans(n_clusters=n_clusters), "K-Means", n_clusters
        )

    def _ejecutar_hac(self, n_clusters: int) -> dict[str, Any]:
        self._validar_k(n_clusters)
        if len(self.original) > 2000:
            raise ValueError("HAC está limitado a conjuntos de hasta 2,000 filas.")
        return self._resumen_clusters(
            self._clustering.hac(n_clusters=n_clusters), "HAC", n_clusters
        )

    def _validar_k(self, n_clusters: int) -> None:
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
            raise ValueError("n_clusters debe ser un entero.")
        if not 2 <= n_clusters <= self.percepcion["max_clusters"]:
            raise ValueError(
                f"n_clusters debe estar entre 2 y {self.percepcion['max_clusters']} "
                "y ser menor que la cantidad de filas."
            )

    def _herramientas(self) -> list[dict[str, Any]]:
        claves = {"type": "string", "enum": list(self.CLAVES)}
        justificacion = {"type": "string", "description": "Motivo breve de la decisión."}
        entero = {"type": "integer", "minimum": 2, "maximum": self.percepcion["max_clusters"]}
        return [
            {"type": "function", "function": {
                "name": "ejecutar_paso",
                "description": (
                    "Ejecuta un análisis permitido. Para kmeans o hac debes indicar n_clusters; "
                    "usa los resultados de codo como referencia."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clave": claves,
                        "justificacion": justificacion,
                        "n_clusters": entero,
                    },
                    "required": ["clave", "justificacion"],
                },
            }},
            {"type": "function", "function": {
                "name": "omitir_paso",
                "description": "Omite un análisis y explica por qué.",
                "parameters": {
                    "type": "object",
                    "properties": {"clave": claves, "justificacion": justificacion},
                    "required": ["clave", "justificacion"],
                },
            }},
            {"type": "function", "function": {
                "name": "finalizar",
                "description": "Finaliza con una interpretación basada solo en resultados observados.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analisis": {"type": "string", "description": "Conclusión en español."}
                    },
                    "required": ["analisis"],
                },
            }},
        ]

    def _despachar(
        self,
        nombre: str,
        args: dict[str, Any],
        decisiones: dict[str, tuple[bool, str]],
    ) -> dict[str, Any]:
        if nombre == "finalizar":
            pendientes = [clave for clave in self.CLAVES if clave not in decisiones]
            if pendientes:
                return {"error": f"Faltan decisiones para: {pendientes}"}
            self.analisis_llm = str(args.get("analisis", "")).strip()
            return {"ok": True}
        clave = args.get("clave")
        if clave not in self.CLAVES:
            return {"error": f"Análisis inválido. Opciones: {list(self.CLAVES)}"}
        if clave in decisiones:
            return {"error": f"El análisis '{clave}' ya fue decidido."}
        justificacion = str(args.get("justificacion", "")).strip()
        if nombre == "omitir_paso":
            decisiones[clave] = (False, justificacion or "Omitido por el agente.")
            self._registrar(
                "omisión (LLM)", clave, decisiones[clave][1], "Análisis omitido."
            )
            return {"ok": True, "omitido": clave}
        if nombre != "ejecutar_paso":
            return {"error": f"Herramienta desconocida: {nombre}"}
        sugerido = next(item for item in self.plan if item["clave"] == clave)
        if not sugerido["sugerido"]:
            return {"error": f"No es factible ejecutar '{clave}': {sugerido['razon']}"}
        codo_factible = next(item for item in self.plan if item["clave"] == "codo")["sugerido"]
        if clave in ("kmeans", "hac") and codo_factible and "codo" not in decisiones:
            return {"error": "Decide primero 'codo' (ejecutar u omitir): orienta la elección de n_clusters."}
        try:
            if clave in ("kmeans", "hac"):
                if "n_clusters" not in args:
                    return {"error": "Debes indicar n_clusters para ejecutar clustering."}
                resultado = self._acciones()[clave](args["n_clusters"])
            else:
                resultado = self._acciones()[clave]()
        except Exception as error:
            return {"error": f"No se pudo ejecutar '{clave}': {type(error).__name__}: {error}"}
        self.resultados[clave] = resultado
        decisiones[clave] = (True, justificacion or "Ejecutado por el agente.")
        self._registrar(
            f"{clave} (LLM: {decisiones[clave][1]})",
            json.dumps(self._resumen_para_llm(clave, resultado), ensure_ascii=False),
            "El algoritmo calculó las métricas; el LLM solo seleccionó la herramienta.",
            "Guardar métricas calculadas por Python.",
        )
        return {"ok": True, "resultado": self._resumen_para_llm(clave, resultado)}

    @staticmethod
    def _resumen_para_llm(clave: str, resultado: dict[str, Any]) -> dict[str, Any]:
        excluir = {"componentes", "cargas", "serie", "resultado"}
        return {llave: valor for llave, valor in resultado.items() if llave not in excluir}

    def _ejecutar_con_llm(self) -> None:
        cliente = self.cliente or llm.crear_cliente()
        self.percibir()
        self.planificar()
        perfil = {
            "perfil_de_datos": self.percepcion,
            "plan_sugerido_por_reglas": self.plan,
        }
        mensajes: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Eres un agente de análisis no supervisado. No escribas ni ejecutes código. "
                    "Usa exclusivamente las herramientas disponibles; Python calcula los resultados. "
                    "Solo recibes perfiles y métricas agregadas, nunca filas del dataset. "
                    "Considera ejecutar codo antes de K-Means o HAC, y elige n_clusters válido. "
                    "No inventes métricas; interpreta solo resultados devueltos por las herramientas. "
                    "Decide ejecutar u omitir cada análisis factible y finaliza con una conclusión "
                    "en español, mencionando límites y que los grupos no implican causalidad."
                ),
            },
            {"role": "user", "content": json.dumps(perfil, ensure_ascii=False, default=str)},
        ]
        decisiones: dict[str, tuple[bool, str]] = {}
        self.uso_llm = llm.bucle_herramientas(
            cliente,
            self.modelo,
            mensajes,
            self._herramientas(),
            lambda nombre, args: self._despachar(nombre, args, decisiones),
            self.max_iteraciones,
        )

        self._decisiones = decisiones
        self.plan = [
            {
                **item,
                "ejecutar": decisiones[item["clave"]][0],
                "razon": decisiones[item["clave"]][1],
            }
            for item in self.plan
        ]
        self._autoevaluar()

    def _autoevaluar(self) -> None:
        """Revisa la calidad de los resultados antes de entregarlos."""
        hallazgos = []
        pca = self.resultados.get("pca")
        if pca:
            acumulada = pca["varianza_acumulada"]
            hallazgos.append(
                f"Los dos primeros componentes conservan el {acumulada:.1%} de la varianza"
                + ("; la proyección 2D es representativa." if acumulada >= 0.6
                   else "; la proyección 2D pierde mucha información y debe interpretarse con cautela.")
            )
        silhouettes = {}
        for clave in ("kmeans", "hac"):
            resultado = self.resultados.get(clave)
            if not resultado:
                continue
            silhouette = resultado["silhouette"]
            silhouettes[resultado["algoritmo"]] = silhouette
            calidad = "fuerte" if silhouette >= 0.5 else "moderada" if silhouette >= 0.25 else "débil"
            hallazgos.append(
                f"{resultado['algoritmo']} con {resultado['grupos_obtenidos']} grupos: silhouette "
                f"{silhouette:.3f} (estructura {calidad})."
            )
            tamanos = resultado["tamanos"]
            if min(tamanos.values()) / sum(tamanos.values()) < 0.05:
                hallazgos.append(
                    f"{resultado['algoritmo']} tiene un grupo con menos del 5 % de las observaciones: "
                    "puede tratarse de atípicos más que de un segmento real."
                )
        if len(silhouettes) == 2:
            mejor = max(silhouettes, key=silhouettes.get)
            hallazgos.append(f"{mejor} obtiene la mejor separación entre los algoritmos de agrupamiento.")
        if not hallazgos:
            hallazgos.append("No se ejecutaron análisis: no hay resultados que evaluar.")
        self.autoevaluacion = hallazgos
        self._registrar(
            "autoevaluación",
            f"{len(self.resultados)} análisis ejecutados.",
            " ".join(hallazgos),
            "Verificar la calidad de los resultados antes de concluir.",
        )

    def _ejecutar_reglas(self) -> None:
        self.percibir()
        self.planificar()
        reglas = {"pca": True, "codo": True, "kmeans": True, "hac": False}
        for item in self.plan:
            clave = item["clave"]
            ejecutar = reglas[clave] and item["sugerido"]
            if not ejecutar:
                razon = item["razon"] if not item["sugerido"] else "No se ejecuta en el respaldo por reglas."
                item.update({"ejecutar": False, "razon": razon})
                self._registrar("omisión (reglas)", item["paso"], razon, "Análisis omitido.")
                continue
            try:
                if clave in ("kmeans", "hac"):
                    n_clusters = min(3, self.percepcion["max_clusters"])
                    resultado = self._acciones()[clave](n_clusters)
                else:
                    resultado = self._acciones()[clave]()
                self.resultados[clave] = resultado
                item.update({"ejecutar": True, "razon": "Ejecutado por el respaldo basado en reglas."})
                self._registrar(
                    f"{clave} (reglas)",
                    json.dumps(self._resumen_para_llm(clave, resultado), ensure_ascii=False),
                    "El algoritmo calculó las métricas localmente.",
                    "Guardar resultados del análisis.",
                )
            except Exception as error:
                item.update({"ejecutar": False, "razon": f"No se pudo ejecutar: {error}"})
                self._registrar("error (reglas)", clave, str(error), "Análisis no ejecutado.")
        self._autoevaluar()

    def ejecutar(self) -> AgenteNoSupervisado:
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

    def guardar_informe(self, ruta: str = "informe_no_supervisado.html") -> str:
        with open(ruta, "w", encoding="utf-8") as archivo:
            archivo.write(self.informe_html())
        return ruta

    def informe_html(self) -> str:
        """Genera un informe HTML autocontenido con el resumen de los análisis."""
        if not self.plan:
            raise RuntimeError("Ejecute el agente antes de generar el informe.")

        escapar = html.escape
        partes = [
            "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Informe del agente no supervisado</title><style>"
            "body{font-family:Arial,sans-serif;max-width:1000px;margin:32px auto;padding:0 18px;"
            "line-height:1.55;color:#20242b}h1,h2{color:#174a72}"
            "section{border-bottom:1px solid #d5dce3;padding:12px 0}"
            "table{border-collapse:collapse;font-size:14px;margin:10px 0;width:100%}"
            "td,th{border:1px solid #cbd3dc;padding:6px 9px;text-align:left}"
            "th{background:#edf2f7}.status{padding:10px;background:#eef5fa;border-radius:6px}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body>",
            "<h1>Informe del agente no supervisado</h1>",
            f"<p class='status'><b>Modo:</b> {escapar(self.modo)}"
            + (f" — {escapar(self.modelo)}" if self.modo == "llm" else "")
            + "</p>",
        ]
        if self.motivo_respaldo:
            partes.append(
                f"<p><b>Motivo del respaldo:</b> {escapar(self.motivo_respaldo)}</p>"
            )
        if self.uso_llm:
            partes.append(f"<p><b>Uso del LLM:</b> {escapar(llm.resumen_uso(self.uso_llm))}</p>")

        if self.percepcion:
            perfil = pd.DataFrame(
                [
                    ("Filas", self.percepcion["filas"]),
                    ("Columnas", self.percepcion["columnas"]),
                    ("Variables numéricas", len(self.percepcion["numericas"])),
                    ("Variables categóricas", len(self.percepcion["categoricas"])),
                    ("Columnas con nulos", ", ".join(self.percepcion["columnas_con_nulos"]) or "Ninguna"),
                    ("Columnas constantes", ", ".join(self.percepcion["columnas_constantes"]) or "Ninguna"),
                ],
                columns=["Medida", "Valor"],
            )
            partes.extend(["<h2>Perfil de los datos</h2>", perfil.to_html(index=False, border=0)])

        partes.append("<h2>Plan y decisiones</h2><ul>")
        for item in self.plan:
            estado = "Ejecutado" if item.get("ejecutar") else "Omitido"
            razon = item.get("razon", "")
            partes.append(
                f"<li><b>{escapar(item['paso'])}</b> — {estado}: {escapar(str(razon))}</li>"
            )
        partes.append("</ul>")

        if self.analisis_llm:
            partes.extend(["<h2>Interpretación del LLM</h2>"])
            partes.extend(
                f"<p>{escapar(parrafo)}</p>"
                for parrafo in self.analisis_llm.splitlines()
                if parrafo.strip()
            )

        if self.autoevaluacion:
            partes.append("<h2>Autoevaluación del agente</h2><ul>")
            partes.extend(f"<li>{escapar(texto)}</li>" for texto in self.autoevaluacion)
            partes.append("</ul>")

        resultado_pca = self.resultados.get("pca")
        if resultado_pca:
            partes.append("<section><h2>PCA</h2>")
            varianza = pd.DataFrame(
                [
                    {"Componente": f"PC{indice}", "Varianza explicada": valor}
                    for indice, valor in enumerate(
                        resultado_pca["varianza_por_componente"], start=1
                    )
                ]
            )
            partes.append(
                f"<p><b>Varianza acumulada:</b> "
                f"{resultado_pca['varianza_acumulada']:.2%}</p>"
            )
            partes.append(varianza.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}"))
            cargas = pd.DataFrame(resultado_pca["variables_mas_influyentes"]).fillna("")
            partes.append("<h3>Variables más influyentes por componente</h3>")
            partes.append(cargas.to_html(border=0, float_format=lambda x: f"{x:.4f}"))
            partes.append("</section>")

        resultado_codo = self.resultados.get("codo")
        if resultado_codo:
            inercias = pd.DataFrame(
                [
                    {"Número de grupos (k)": k, "Inercia": valor}
                    for k, valor in resultado_codo["inercias"].items()
                ]
            )
            partes.extend(
                [
                    "<section><h2>Método del codo</h2>",
                    inercias.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}"),
                    "</section>",
                ]
            )

        for clave in ("kmeans", "hac"):
            resultado = self.resultados.get(clave)
            if not resultado:
                continue
            partes.extend(
                [
                    f"<section><h2>{escapar(resultado['algoritmo'])}</h2>",
                    f"<p><b>Grupos:</b> {resultado['grupos_obtenidos']} · "
                    f"<b>Silhouette:</b> {resultado['silhouette']:.4f}</p>",
                ]
            )
            if "inercia" in resultado:
                partes.append(f"<p><b>Inercia:</b> {resultado['inercia']:.4f}</p>")
            if "correlacion_cofenetica" in resultado:
                partes.append(
                    "<p><b>Correlación cofenética:</b> "
                    f"{resultado['correlacion_cofenetica']:.4f}</p>"
                )
            tamanos = pd.DataFrame(
                [
                    {"Grupo": grupo, "Observaciones": cantidad}
                    for grupo, cantidad in resultado["tamanos"].items()
                ]
            )
            partes.append("<h3>Tamaño de los grupos</h3>")
            partes.append(tamanos.to_html(index=False, border=0))
            if resultado["perfil_numerico"]:
                partes.append("<h3>Promedios numéricos por grupo</h3>")
                perfil_numerico = pd.DataFrame.from_dict(
                    resultado["perfil_numerico"], orient="index"
                )
                perfil_numerico.index.name = "Grupo"
                partes.append(
                    perfil_numerico.to_html(
                        border=0, float_format=lambda x: f"{x:.4f}"
                    )
                )
            if any(resultado["perfil_categorico"].values()):
                filas_categoricas = [
                    {
                        "Grupo": grupo,
                        "Variable": variable,
                        "Categoría más frecuente": valores["categoria_mas_frecuente"],
                        "Proporción": valores["proporcion"],
                    }
                    for grupo, variables in resultado["perfil_categorico"].items()
                    for variable, valores in variables.items()
                ]
                partes.append("<h3>Categorías más frecuentes por grupo</h3>")
                partes.append(
                    pd.DataFrame(filas_categoricas).to_html(
                        index=False, border=0, escape=True,
                        float_format=lambda x: f"{x:.1%}",
                    )
                )
            partes.append("</section>")

        partes.extend(
            [
                "<h2>Bitácora de razonamiento</h2>",
                pd.DataFrame(self.bitacora).to_html(index=False, border=0, escape=True),
                "<p><i>Los grupos describen similitudes estadísticas; no prueban causalidad "
                "ni sustituyen la interpretación del dominio.</i></p>",
                "</body></html>",
            ]
        )
        return "\n".join(partes)
