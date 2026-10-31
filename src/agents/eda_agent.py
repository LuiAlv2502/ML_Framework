"""Agente inteligente de Análisis Exploratorio de Datos (EDA).

El agente sigue el ciclo PERCIBIR -> PLANIFICAR -> ACTUAR -> INTERPRETAR -> RECOMENDAR.
Los pasos siguen el tutorial "Intro to Exploratory Data Analysis (EDA) in Python"
(Kaggle): tipos de datos, columnas irrelevantes, renombrar columnas, duplicados,
valores faltantes, valores atípicos y relaciones entre variables.

El razonamiento es basado en reglas: cada decisión queda registrada con la
observación que la provocó y la razón por la que se tomó, de modo que sea
verificable. El agente NUNCA modifica los datos originales: aplica la limpieza
sobre una copia y la entrega como sugerencia.
"""
from __future__ import annotations

import base64
import html
import io
import re
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from EDA.dataframe import DataFrame  # noqa: E402

# Umbrales que usa el agente para razonar (se documentan en el informe).
UMBRAL_NULOS_COLUMNA = 0.60     # > 60 % de nulos: la columna aporta poca información
UMBRAL_NULOS_FILAS = 0.05       # < 5 % de filas con nulos: se pueden eliminar esas filas
UMBRAL_ASIMETRIA = 1.0          # |asimetría| > 1: distribución muy sesgada
UMBRAL_ATIPICOS = 0.05          # > 5 % de atípicos en una columna: revisar
UMBRAL_CORR_FUERTE = 0.70       # |r| >= 0.7: relación fuerte
UMBRAL_MULTICOLINEALIDAD = 0.90  # |r| >= 0.9: variables casi redundantes
UMBRAL_CATEGORIA_RARA = 0.01    # categoría con < 1 % de los registros


@dataclass
class Paso:
    """Un paso del EDA con su razonamiento completo."""

    numero: int
    titulo: str
    observacion: str = ""
    razonamiento: str = ""
    accion: str = ""
    interpretacion: str = ""
    recomendaciones: list[str] = field(default_factory=list)
    tablas: dict[str, pd.DataFrame] = field(default_factory=dict)
    figuras: list[tuple[str, Any]] = field(default_factory=list)  # (título, figura matplotlib)
    severidad: str = "informativa"  # informativa | media | alta


class AgenteEDA:
    """Agente que realiza un EDA completo, interpreta cada paso y recomienda."""

    def __init__(self, datos, objetivo: str | None = None, max_graficos: int = 12):
        if isinstance(datos, DataFrame):
            datos = datos.dataframe
        if not isinstance(datos, pd.DataFrame):
            raise TypeError("El agente recibe un pandas.DataFrame o un DataFrame del framework.")
        if datos.empty:
            raise ValueError("No hay datos para analizar.")
        if objetivo is not None and objetivo not in datos.columns:
            raise KeyError(f"La variable objetivo '{objetivo}' no existe.")

        self.original = datos.copy()
        # Copia de trabajo: se usa la clase DataFrame del framework.
        self.trabajo = DataFrame(datos)
        self.objetivo = objetivo
        self.max_graficos = max_graficos
        self.percepcion: dict[str, Any] = {}
        self.plan: list[dict[str, str]] = []
        self.pasos: list[Paso] = []
        self.bitacora: list[dict[str, str]] = []
        self.sintesis: dict[str, Any] = {}
        self.imputadas: dict[str, float] = {}  # columna -> % de valores imputados

    # ==================================================================
    # Utilidades
    # ==================================================================
    @property
    def datos_limpios(self) -> pd.DataFrame:
        """Datos después de las limpiezas que el agente aplicó sobre la copia."""
        return self.trabajo.dataframe

    def _registrar(self, etapa: str, observacion: str, razonamiento: str, accion: str) -> None:
        self.bitacora.append(
            {"etapa": etapa, "observacion": observacion, "razonamiento": razonamiento, "accion": accion}
        )

    @staticmethod
    def _numericas(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=np.number).columns.tolist()

    @staticmethod
    def _categoricas(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(exclude=np.number).columns.tolist()

    @staticmethod
    def _pct(valor: float) -> str:
        return f"{valor:.1%}"

    # ==================================================================
    # 1. PERCIBIR: observar los datos antes de decidir qué hacer
    # ==================================================================
    def percibir(self) -> dict[str, Any]:
        df = self.original
        n = len(df)
        nulos = df.isna().sum()
        numericas = self._numericas(df)
        self.percepcion = {
            "filas": n,
            "columnas": df.shape[1],
            "numericas": numericas,
            "categoricas": self._categoricas(df),
            "columnas_con_nulos": nulos[nulos > 0].index.tolist(),
            "duplicados": int(df.duplicated().sum()),
            "nombres_irregulares": [c for c in df.columns if re.search(r"[^0-9a-zA-Z_]", str(c))],
            "ceros_sospechosos": self._detectar_ceros_sospechosos(df),
            "objetivo": self.objetivo,
        }
        self._registrar(
            "percepción",
            f"{n} filas, {df.shape[1]} columnas ({len(numericas)} numéricas).",
            "Antes de analizar hay que conocer el tamaño, los tipos y los problemas evidentes.",
            "Construir un perfil inicial de los datos.",
        )
        return self.percepcion

    @staticmethod
    def _detectar_ceros_sospechosos(df: pd.DataFrame) -> dict[str, int]:
        """Columnas donde 0 parece usarse como 'dato faltante' (ej. presión = 0).

        Regla: columna numérica no binaria, sin negativos, con ceros entre el 0.5 %
        y el 50 % de las filas, y donde los valores distintos de 0 empiezan lejos
        de 0 (hay un "salto" entre 0 y el mínimo real). Se excluyen los conteos
        (enteros que empiezan en 1, ej. número de embarazos), donde 0 es válido.
        """
        sospechosos = {}
        for columna in df.select_dtypes(include=np.number).columns:
            serie = df[columna].dropna()
            if serie.empty or serie.nunique() <= 2 or (serie < 0).any():
                continue
            ceros = int((serie == 0).sum())
            proporcion = ceros / len(serie)
            if not 0.005 <= proporcion <= 0.5:
                continue
            sin_ceros = serie[serie != 0]
            es_conteo = bool((sin_ceros % 1 == 0).all()) and sin_ceros.min() <= 1
            if not es_conteo and sin_ceros.min() > 0.1 * sin_ceros.median():
                sospechosos[columna] = ceros
        return sospechosos

    # ==================================================================
    # 2. PLANIFICAR: decidir qué pasos ejecutar y cuáles omitir (y por qué)
    # ==================================================================
    def planificar(self) -> list[dict[str, str]]:
        p = self.percepcion or self.percibir()
        plan = [
            ("estructura", "Revisar estructura y tipos de datos", True,
             "Siempre es necesario saber qué tipo de dato tiene cada columna."),
            ("irrelevantes", "Detectar columnas irrelevantes", p["columnas"] > 1,
             "Columnas tipo identificador o constantes no aportan al análisis."),
            ("renombrar", "Estandarizar nombres de columnas", bool(p["nombres_irregulares"]),
             f"Hay {len(p['nombres_irregulares'])} nombres con espacios o símbolos."
             if p["nombres_irregulares"] else "Los nombres ya son válidos: no hace falta renombrar."),
            ("duplicados", "Eliminar filas duplicadas", p["duplicados"] > 0,
             f"Se detectaron {p['duplicados']} filas duplicadas."
             if p["duplicados"] else "No hay filas duplicadas: el paso se omite."),
            ("faltantes", "Tratar valores faltantes",
             bool(p["columnas_con_nulos"] or p["ceros_sospechosos"]),
             "Hay nulos o ceros que parecen datos faltantes."
             if (p["columnas_con_nulos"] or p["ceros_sospechosos"])
             else "No hay valores faltantes: el paso se omite."),
            ("atipicos", "Detectar valores atípicos", bool(p["numericas"]),
             "Los atípicos afectan medias, correlaciones y modelos."
             if p["numericas"] else "No hay variables numéricas: el paso se omite."),
            ("distribuciones", "Analizar distribuciones", True,
             "La forma de cada variable define qué transformaciones y modelos convienen."),
            ("relaciones", "Analizar relaciones entre variables", len(p["numericas"]) >= 2,
             "Se requieren al menos dos variables numéricas para correlaciones."
             if len(p["numericas"]) < 2 else "Las correlaciones revelan redundancias y relaciones útiles."),
            ("objetivo", "Analizar la variable objetivo", self.objetivo is not None,
             f"Se indicó '{self.objetivo}' como objetivo." if self.objetivo
             else "No se indicó variable objetivo: el paso se omite."),
        ]
        self.plan = [
            {"clave": clave, "paso": titulo, "ejecutar": ejecutar, "razon": razon}
            for clave, titulo, ejecutar, razon in plan
        ]
        incluidos = sum(1 for paso in self.plan if paso["ejecutar"])
        self._registrar(
            "planificación",
            f"{incluidos} de {len(self.plan)} pasos son pertinentes.",
            "Solo se ejecutan los pasos que los datos justifican; los demás se omiten con su razón.",
            "Definir el plan de análisis.",
        )
        return self.plan

    # ==================================================================
    # 3. ACTUAR + INTERPRETAR: cada paso del EDA
    # ==================================================================
    def _paso_estructura(self, numero: int) -> Paso:
        df = self.datos_limpios
        tipos = pd.DataFrame(
            {
                "Tipo": df.dtypes.astype(str),
                "Valores únicos": df.nunique(),
                "Nulos": df.isna().sum(),
                "Ejemplo": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
            }
        )
        # Razonamiento: ¿hay números guardados como texto o categorías codificadas como número?
        texto_numerico = []
        for columna in self._categoricas(df):
            convertida = pd.to_numeric(df[columna], errors="coerce")
            if df[columna].notna().sum() and convertida.notna().sum() / df[columna].notna().sum() > 0.9:
                texto_numerico.append(columna)
        numero_categorico = [
            c for c in self._numericas(df)
            if df[c].nunique() <= 10 and c != self.objetivo and (df[c].dropna() % 1 == 0).all()
        ]
        recomendaciones = []
        if texto_numerico:
            recomendaciones.append(
                f"Convertir a número las columnas {texto_numerico}: más del 90 % de sus valores son numéricos guardados como texto."
            )
        if numero_categorico:
            recomendaciones.append(
                f"Las columnas {numero_categorico} son numéricas pero tienen pocos valores enteros: "
                "podrían ser categorías codificadas; revisar su significado antes de usarlas como números."
            )
        return Paso(
            numero,
            "Estructura y tipos de datos",
            observacion=f"{df.shape[0]} filas y {df.shape[1]} columnas: "
                        f"{len(self._numericas(df))} numéricas y {len(self._categoricas(df))} de texto/categóricas.",
            razonamiento="Cada tipo de variable requiere análisis distintos: histogramas y correlaciones para "
                         "números, frecuencias para categorías. Un tipo mal asignado lleva a conclusiones erróneas.",
            accion="Revisar tipo, cantidad de valores únicos y nulos de cada columna.",
            interpretacion="Los tipos son coherentes con los valores."
            if not recomendaciones else "Algunas columnas podrían tener un tipo que no corresponde a su significado.",
            recomendaciones=recomendaciones or ["Los tipos de datos son adecuados para continuar."],
            tablas={"Tipos de datos": tipos},
            severidad="media" if texto_numerico else "informativa",
        )

    def _paso_irrelevantes(self, numero: int) -> Paso:
        df = self.datos_limpios
        n, total_columnas = df.shape
        motivos = {}
        for columna in df.columns:
            if columna == self.objetivo:
                continue
            unicos = df[columna].nunique(dropna=True)
            nulos = df[columna].isna().mean()
            if unicos <= 1:
                motivos[columna] = "valor constante (no distingue entre registros)"
            elif nulos > UMBRAL_NULOS_COLUMNA:
                motivos[columna] = f"{self._pct(nulos)} de valores faltantes"
            elif unicos == n and n > 20:
                nombre_id = re.search(r"id$|^id|codigo|^unnamed", str(columna), re.I)
                consecutivo = (
                        pd.api.types.is_integer_dtype(df[columna]) and df[columna].is_monotonic_increasing
                )
                texto_unico = not pd.api.types.is_numeric_dtype(df[columna])
                if nombre_id or consecutivo or texto_unico:
                    motivos[columna] = "identificador único por fila (no aporta patrones)"
        for columna in motivos:
            self.trabajo.eliminar_columna(columna)
        tabla = pd.DataFrame({"Columna": list(motivos), "Motivo": list(motivos.values())})
        return Paso(
            numero,
            "Columnas irrelevantes",
            observacion=f"Se revisaron {total_columnas} columnas buscando identificadores, constantes y columnas casi vacías.",
            razonamiento="Una columna que es distinta en cada fila (un ID) o igual en todas no ayuda a encontrar "
                         "patrones, y en un modelo puede causar sobreajuste o ruido.",
            accion=f"Eliminar en la copia de trabajo: {list(motivos)}." if motivos else "No se eliminó ninguna columna.",
            interpretacion=f"{len(motivos)} columnas no aportan información útil." if motivos
            else "Todas las columnas pueden aportar información.",
            recomendaciones=[f"Excluir '{c}' del modelado: {m}." for c, m in motivos.items()]
                            or ["Conservar todas las columnas."],
            tablas={"Columnas eliminadas": tabla} if motivos else {},
            severidad="media" if motivos else "informativa",
        )

    def _paso_renombrar(self, numero: int) -> Paso:
        df = self.datos_limpios
        nuevos = {}
        for columna in df.columns:
            limpio = re.sub(r"[^0-9a-zA-Z_]+", "_", str(columna).strip()).strip("_")
            if limpio and limpio != columna:
                nuevos[columna] = limpio
        self.trabajo.dataframe = df.rename(columns=nuevos)
        if self.objetivo in nuevos:
            self.objetivo = nuevos[self.objetivo]
        return Paso(
            numero,
            "Estandarizar nombres de columnas",
            observacion=f"{len(nuevos)} columnas tienen espacios o símbolos en el nombre.",
            razonamiento="Los nombres con espacios o símbolos dificultan escribir código y pueden causar errores "
                         "en librerías; nombres simples hacen el análisis más legible.",
            accion="Renombrar en la copia de trabajo reemplazando espacios y símbolos por '_'.",
            interpretacion="Los nombres quedaron listos para usarse en código.",
            recomendaciones=["Usar los nombres estandarizados en el resto del proyecto."],
            tablas={"Nombres": pd.DataFrame({"Antes": list(nuevos), "Después": list(nuevos.values())})},
        )

    def _paso_duplicados(self, numero: int) -> Paso:
        antes = len(self.datos_limpios)
        duplicados = int(self.datos_limpios.duplicated().sum())
        self.trabajo.eliminar_duplicados()
        proporcion = duplicados / antes
        return Paso(
            numero,
            "Filas duplicadas",
            observacion=f"{duplicados} filas duplicadas ({self._pct(proporcion)} de los datos).",
            razonamiento="Las filas repetidas cuentan dos veces la misma observación: inflan frecuencias y "
                         "pueden hacer que un registro aparezca en entrenamiento y en prueba a la vez.",
            accion=f"Eliminar duplicados en la copia de trabajo ({antes} → {len(self.datos_limpios)} filas).",
            interpretacion="Los duplicados son pocos y su eliminación no cambia la estructura de los datos."
            if proporcion < 0.05 else "Hay muchos duplicados: podría tratarse de un problema en la captura de datos.",
            recomendaciones=["Confirmar que no existan registros que legítimamente se repitan antes de eliminarlos."],
            severidad="alta" if proporcion >= 0.05 else "media",
        )

    def _paso_faltantes(self, numero: int) -> Paso:
        df = self.datos_limpios.copy()
        recomendaciones = []
        # Ceros que representan datos faltantes (ej. glucosa = 0).
        ceros = {c: v for c, v in self.percepcion["ceros_sospechosos"].items() if c in df.columns}
        for columna in ceros:
            df[columna] = df[columna].replace(0, np.nan)
        if ceros:
            recomendaciones.append(
                f"Las columnas {list(ceros)} usan 0 donde probablemente falta el dato; se trataron como nulos."
            )

        nulos = df.isna().sum()
        nulos = nulos[nulos > 0].sort_values(ascending=False)
        tabla = pd.DataFrame(
            {"Nulos": nulos, "% nulos": (nulos / len(df) * 100).round(2)}
        )
        filas_con_nulos = df.isna().any(axis=1).mean()

        # Decisión: eliminar filas (como el tutorial) solo si son pocas; si no, imputar.
        if filas_con_nulos <= UMBRAL_NULOS_FILAS:
            antes = len(df)
            df = df.dropna().reset_index(drop=True)
            accion = f"Eliminar las filas con nulos ({antes} → {len(df)} filas)."
            decision = f"Solo el {self._pct(filas_con_nulos)} de las filas tiene nulos: eliminarlas casi no pierde información."
        else:
            self.imputadas = {c: float(v) for c, v in (nulos / len(df)).items()}
            self.trabajo.dataframe = df
            self.trabajo.imputar_nulos()
            df = self.trabajo.dataframe
            accion = "Imputar con la mediana (numéricas) y la moda (categóricas)."
            decision = (
                f"El {self._pct(filas_con_nulos)} de las filas tiene algún nulo: eliminarlas perdería demasiados "
                "datos, por eso se imputan."
            )
            recomendaciones.append(
                "La imputación se hizo sobre todos los datos para explorar; al modelar, calcular la mediana solo "
                "con el conjunto de entrenamiento (como hace el framework en su Pipeline)."
            )
        self.trabajo.dataframe = df

        figuras = []
        if not nulos.empty:
            fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(nulos))))
            tabla["% nulos"].iloc[::-1].plot.barh(ax=ax, color="#d9534f")
            ax.set_xlabel("% de valores faltantes")
            ax.set_title("Valores faltantes por columna")
            figuras.append(("Valores faltantes por columna", fig))

        peor = nulos.index[0] if not nulos.empty else None
        return Paso(
            numero,
            "Valores faltantes",
            observacion=f"{len(nulos)} columnas con datos faltantes; el {self._pct(filas_con_nulos)} de las filas "
                        "tiene al menos uno." + (f" La más afectada es '{peor}' ({tabla.loc[peor, '% nulos']}%)." if peor else ""),
            razonamiento=decision,
            accion=accion,
            interpretacion="Un dato faltante no siempre es aleatorio: si falta más en ciertos grupos, la imputación "
                           "puede introducir sesgo.",
            recomendaciones=recomendaciones + [
                "Investigar por qué faltan los datos de las columnas más afectadas antes de modelar."
            ],
            tablas={"Valores faltantes": tabla},
            figuras=figuras,
            severidad="alta" if (not tabla.empty and tabla["% nulos"].max() > 20) else "media",
        )

    def _paso_atipicos(self, numero: int) -> Paso:
        df = self.datos_limpios
        numericas = [c for c in self._numericas(df) if df[c].nunique() > 2]
        if not numericas:
            return Paso(numero, "Valores atípicos", observacion="No hay variables numéricas continuas.")
        # Se reutiliza el método del framework.
        atipicos = DataFrame(df[numericas]).detectar_outliers()
        conteo = pd.Series({c: len(filas) for c, filas in atipicos.items()}, name="Atípicos")
        tabla = pd.DataFrame({"Atípicos": conteo, "% atípicos": (conteo / len(df) * 100).round(2)})
        tabla = tabla.sort_values("Atípicos", ascending=False)
        problematicas = tabla[tabla["% atípicos"] > UMBRAL_ATIPICOS * 100].index.tolist()
        # Razonamiento: si una columna se imputó mucho, sus atípicos están inflados artificialmente.
        infladas = [c for c in problematicas if self.imputadas.get(c, 0) > 0.20]
        notas_imputacion = [
            f"El {tabla.loc[c, '% atípicos']}% de atípicos en '{c}' está inflado: al imputar el "
            f"{self._pct(self.imputadas[c])} de sus valores con la mediana, muchos datos quedaron en un solo "
            "valor, la caja se achica y valores normales parecen atípicos. Repetir este análisis sin imputar."
            for c in infladas
        ]

        mostrar = numericas[: self.max_graficos]
        estandarizados = (df[mostrar] - df[mostrar].mean()) / df[mostrar].std(ddof=0).replace(0, 1)
        fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(mostrar)), 4.5))
        ax.boxplot([estandarizados[c].dropna() for c in mostrar])
        ax.set_xticks(range(1, len(mostrar) + 1), mostrar)
        ax.set_ylabel("Valor estandarizado (z)")
        ax.set_title("Diagramas de caja (escala estandarizada)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        return Paso(
            numero,
            "Valores atípicos",
            observacion=f"Total de valores atípicos por columna (regla IQR). Columnas con más del "
                        f"{UMBRAL_ATIPICOS:.0%}: {problematicas or 'ninguna'}.",
            razonamiento="Un valor atípico está fuera de [Q1 − 1.5·IQR, Q3 + 1.5·IQR]. Puede ser un error de "
                         "captura o un caso real extremo; ambos afectan medias, correlaciones y modelos lineales.",
            accion="Contar atípicos por columna con el método detectar_outliers del framework y graficar diagramas "
                   "de caja estandarizados para comparar variables con escalas distintas.",
            interpretacion="Los puntos fuera de los 'bigotes' de cada caja son los atípicos. Cuantos más hay, más "
                           "asimétrica o con colas largas es la variable.",
            recomendaciones=notas_imputacion + [
                f"Revisar '{c}' ({tabla.loc[c, '% atípicos']}% atípicos): verificar si son errores o casos reales."
                for c in problematicas if c not in infladas
            ] + ["No eliminar atípicos automáticamente; preferir modelos robustos (árboles) o transformaciones."],
            tablas={"Atípicos por columna": tabla},
            figuras=[("Diagramas de caja", fig)],
            severidad="media" if problematicas else "informativa",
        )

    def _paso_distribuciones(self, numero: int) -> Paso:
        df = self.datos_limpios
        numericas = [c for c in self._numericas(df) if df[c].nunique() > 2][: self.max_graficos]
        categoricas = [c for c in self._categoricas(df) if df[c].nunique() <= 30]
        figuras, recomendaciones, filas = [], [], []

        for columna in numericas:
            serie = df[columna].dropna()
            asimetria = float(serie.skew()) if len(serie) > 2 else 0.0
            forma = (
                "sesgada a la derecha (cola larga de valores altos)" if asimetria > UMBRAL_ASIMETRIA
                else "sesgada a la izquierda" if asimetria < -UMBRAL_ASIMETRIA
                else "aproximadamente simétrica"
            )
            filas.append({"Variable": columna, "Media": serie.mean(), "Mediana": serie.median(),
                          "Asimetría": round(asimetria, 2), "Forma": forma})
            if asimetria > UMBRAL_ASIMETRIA and (serie > 0).all():
                recomendaciones.append(f"'{columna}' está muy sesgada (asimetría {asimetria:.2f}): considerar log(x).")

        if numericas:
            columnas_fig = 3
            filas_fig = int(np.ceil(len(numericas) / columnas_fig))
            fig, ejes = plt.subplots(filas_fig, columnas_fig, figsize=(12, 3.2 * filas_fig))
            for eje, columna in zip(np.ravel(ejes), numericas):
                eje.hist(df[columna].dropna(), bins=30, color="#337ab7", edgecolor="white")
                eje.axvline(df[columna].median(), color="#d9534f", linestyle="--", label="mediana")
                eje.set_title(columna)
            for eje in np.ravel(ejes)[len(numericas):]:
                eje.axis("off")
            fig.suptitle("Histogramas de variables numéricas")
            figuras.append(("Histogramas", fig))

        todas_frecuencias = self.trabajo.frecuencias()  # método del framework
        for columna in categoricas[:4]:
            frecuencias = todas_frecuencias[columna]
            proporciones = frecuencias / frecuencias.sum()
            raras = proporciones[proporciones < UMBRAL_CATEGORIA_RARA].index.tolist()
            if proporciones.iloc[0] > 0.8:
                recomendaciones.append(
                    f"En '{columna}' la categoría '{proporciones.index[0]}' domina con {self._pct(proporciones.iloc[0])}: "
                    "la variable aporta poca variación."
                )
            if raras:
                recomendaciones.append(f"'{columna}' tiene {len(raras)} categorías raras (<1 %): considerar agruparlas en 'Otros'.")
            fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * min(len(frecuencias), 15))))
            frecuencias.head(15).iloc[::-1].plot.barh(ax=ax, color="#5cb85c")
            ax.set_title(f"Frecuencias: {columna}")
            figuras.append((f"Frecuencias de {columna}", fig))

        sesgadas = [f["Variable"] for f in filas if abs(f["Asimetría"]) > UMBRAL_ASIMETRIA]
        return Paso(
            numero,
            "Distribuciones",
            observacion=f"{len(numericas)} variables numéricas y {len(categoricas)} categóricas analizadas. "
                        f"Variables muy sesgadas: {sesgadas or 'ninguna'}.",
            razonamiento="Si la media y la mediana difieren mucho, la variable es asimétrica: la media deja de ser "
                         "representativa y los modelos lineales pueden verse afectados.",
            accion="Graficar histogramas (con la mediana en rojo), calcular la asimetría y revisar frecuencias de categorías.",
            interpretacion="En un histograma simétrico la mediana queda en el centro; si queda a un lado y hay una "
                           "cola larga, la variable está sesgada hacia ese lado.",
            recomendaciones=recomendaciones or ["Las distribuciones no requieren transformaciones especiales."],
            tablas={"Forma de las distribuciones": pd.DataFrame(filas).round(3)} if filas else {},
            figuras=figuras,
            severidad="media" if sesgadas else "informativa",
        )

    def _paso_relaciones(self, numero: int) -> Paso:
        df = self.datos_limpios
        numericas = [c for c in self._numericas(df) if df[c].nunique() > 1]
        pearson = DataFrame(df[numericas]).correlacion()
        spearman = df[numericas].corr(method="spearman")

        pares = []
        for i, a in enumerate(numericas):
            for b in numericas[i + 1:]:
                pares.append({"Variable 1": a, "Variable 2": b,
                              "Pearson": pearson.loc[a, b], "Spearman": spearman.loc[a, b]})
        tabla = pd.DataFrame(pares)
        tabla["|r|"] = tabla["Pearson"].abs()
        tabla = tabla.sort_values("|r|", ascending=False).drop(columns="|r|").round(3).reset_index(drop=True)

        fuertes = tabla[tabla["Pearson"].abs() >= UMBRAL_CORR_FUERTE]
        redundantes = tabla[tabla["Pearson"].abs() >= UMBRAL_MULTICOLINEALIDAD]
        no_lineales = tabla[(tabla["Spearman"].abs() - tabla["Pearson"].abs()) > 0.15]

        recomendaciones = []
        for _, fila in redundantes.iterrows():
            recomendaciones.append(
                f"'{fila['Variable 1']}' y '{fila['Variable 2']}' son casi redundantes (r={fila['Pearson']}): "
                "conservar solo una en modelos lineales (multicolinealidad)."
            )
        for _, fila in no_lineales.head(3).iterrows():
            recomendaciones.append(
                f"'{fila['Variable 1']}'–'{fila['Variable 2']}': Spearman ({fila['Spearman']}) supera a Pearson "
                f"({fila['Pearson']}); la relación es monótona pero no lineal."
            )

        mostrar = numericas[:15]
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(mostrar)), max(5, 0.55 * len(mostrar))))
        imagen = ax.imshow(pearson.loc[mostrar, mostrar], cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(mostrar)), mostrar, rotation=45, ha="right")
        ax.set_yticks(range(len(mostrar)), mostrar)
        if len(mostrar) <= 12:
            for i in range(len(mostrar)):
                for j in range(len(mostrar)):
                    ax.text(j, i, f"{pearson.loc[mostrar[i], mostrar[j]]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(imagen, ax=ax, shrink=0.8)
        ax.set_title("Mapa de calor de correlaciones (Pearson)")
        figuras = [("Mapa de calor de correlaciones", fig)]

        if not tabla.empty:
            mejor = tabla.iloc[0]
            muestra = df[[mejor["Variable 1"], mejor["Variable 2"]]].dropna()
            if len(muestra) > 3000:
                muestra = muestra.sample(3000, random_state=42)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(muestra.iloc[:, 0], muestra.iloc[:, 1], s=12, alpha=0.5)
            ax.set_xlabel(mejor["Variable 1"])
            ax.set_ylabel(mejor["Variable 2"])
            ax.set_title(f"Par más correlacionado (r = {mejor['Pearson']})")
            figuras.append(("Dispersión del par más correlacionado", fig))

        texto_top = "; ".join(
            f"{f['Variable 1']}–{f['Variable 2']} (r={f['Pearson']})" for _, f in tabla.head(3).iterrows()
        )
        return Paso(
            numero,
            "Relaciones entre variables",
            observacion=f"Pares con mayor correlación: {texto_top}. "
                        f"{len(fuertes)} pares con relación fuerte (|r| ≥ {UMBRAL_CORR_FUERTE}).",
            razonamiento="Pearson mide relación lineal; Spearman mide relación monótona (aunque sea curva). "
                         "Comparar ambas permite detectar relaciones no lineales. Correlación no implica causalidad.",
            accion="Calcular correlaciones de Pearson (método del framework) y Spearman, graficar el mapa de calor "
                   "y el diagrama de dispersión del par más relacionado.",
            interpretacion="En el mapa de calor, rojo indica relación positiva (si una sube, la otra también), azul "
                           "negativa y blanco ausencia de relación lineal."
                           + (" No se encontraron relaciones fuertes: las variables aportan información distinta entre sí."
                              if fuertes.empty else ""),
            recomendaciones=recomendaciones or ["No hay variables redundantes; se pueden usar todas en los modelos."],
            tablas={"Pares de variables": tabla.head(15)},
            figuras=figuras,
            severidad="media" if not redundantes.empty else "informativa",
        )

    def _paso_objetivo(self, numero: int) -> Paso:
        df = self.datos_limpios
        objetivo = self.objetivo
        y = df[objetivo]
        es_regresion = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
        recomendaciones, tablas, figuras = [], {}, []

        if es_regresion:
            tipo = "regresión (variable numérica continua)"
            otras = [c for c in self._numericas(df) if c != objetivo and df[c].nunique() > 1]
            relacion = df[otras].corrwith(y).sort_values(key=np.abs, ascending=False).round(3)
            tablas["Correlación con el objetivo"] = relacion.rename("r").to_frame()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(y.dropna(), bins=30, color="#337ab7", edgecolor="white")
            ax.set_title(f"Distribución del objetivo: {objetivo}")
            figuras.append(("Distribución del objetivo", fig))
            interpretacion = (
                f"Las variables más relacionadas con '{objetivo}' son: "
                + ", ".join(f"{k} (r={v})" for k, v in relacion.head(3).items()) + "."
                if not relacion.empty else "No hay predictores numéricos para comparar."
            )
            recomendaciones.append("Problema de regresión: usar RegresionModelos del framework y evaluar con RMSE, MAE y R².")
            if abs(float(y.skew())) > UMBRAL_ASIMETRIA:
                recomendaciones.append(f"El objetivo está sesgado (asimetría {y.skew():.2f}): considerar predecir log({objetivo}).")
        else:
            tipo = "clasificación (variable categórica)"
            proporciones = y.value_counts(normalize=True)
            tablas["Balance de clases"] = (proporciones * 100).round(2).rename("%").to_frame()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            proporciones.plot.bar(ax=ax, color="#f0ad4e")
            ax.set_ylabel("Proporción")
            ax.set_title(f"Balance de clases: {objetivo}")
            figuras.append(("Balance de clases", fig))
            mayoritaria = proporciones.iloc[0]
            interpretacion = (
                f"La clase '{proporciones.index[0]}' representa el {self._pct(mayoritaria)}. Un modelo que siempre "
                f"la predice tendría {self._pct(mayoritaria)} de accuracy (línea base)."
            )
            recomendaciones.append("Problema de clasificación: usar ClasificacionModelos del framework con partición estratificada.")
            if mayoritaria > 0.6:
                recomendaciones.append("Las clases están desbalanceadas: usar F1 macro en vez de accuracy como métrica principal.")
            numericas = [c for c in self._numericas(df) if c != objetivo]
            if numericas:
                medias = df.groupby(objetivo)[numericas].median().T.round(3)
                tablas["Mediana de cada variable por clase"] = medias
                diferencia = (medias.max(axis=1) - medias.min(axis=1)) / df[numericas].std().replace(0, 1)
                top = diferencia.sort_values(ascending=False).head(3).index.tolist()
                interpretacion += f" Las variables que más cambian entre clases son: {top}."

        return Paso(
            numero,
            f"Variable objetivo: {objetivo}",
            observacion=f"'{objetivo}' tiene {y.nunique()} valores distintos: es un problema de {tipo}.",
            razonamiento="El tipo de objetivo define la familia de modelos y las métricas. Revisar su balance o su "
                         "distribución evita conclusiones engañosas.",
            accion="Inferir el tipo de problema y analizar la relación del objetivo con las demás variables.",
            interpretacion=interpretacion,
            recomendaciones=recomendaciones,
            tablas=tablas,
            figuras=figuras,
            severidad="media" if len(recomendaciones) > 1 else "informativa",
        )

    # ==================================================================
    # 4. EJECUTAR TODO Y SINTETIZAR
    # ==================================================================
    def ejecutar(self) -> AgenteEDA:
        self.percibir()
        self.planificar()
        acciones = {
            "estructura": self._paso_estructura,
            "irrelevantes": self._paso_irrelevantes,
            "renombrar": self._paso_renombrar,
            "duplicados": self._paso_duplicados,
            "faltantes": self._paso_faltantes,
            "atipicos": self._paso_atipicos,
            "distribuciones": self._paso_distribuciones,
            "relaciones": self._paso_relaciones,
            "objetivo": self._paso_objetivo,
        }
        numero = 1
        for item in self.plan:
            if not item["ejecutar"]:
                self._registrar("omisión", item["paso"], item["razon"], "Paso omitido.")
                continue
            paso = acciones[item["clave"]](numero)
            self.pasos.append(paso)
            self._registrar(paso.titulo, paso.observacion, paso.razonamiento, paso.accion)
            numero += 1
        self._sintetizar()
        return self

    def _sintetizar(self) -> None:
        orden = {"alta": 0, "media": 1, "informativa": 2}
        prioritarias = []
        for paso in sorted(self.pasos, key=lambda p: orden[p.severidad]):
            if paso.severidad == "informativa":
                continue
            prioritarias.extend(f"[{paso.severidad.upper()}] {r}" for r in paso.recomendaciones)
        original, limpio = self.original.shape, self.datos_limpios.shape
        self.sintesis = {
            "resumen": (
                f"Se analizaron {original[0]} filas y {original[1]} columnas en {len(self.pasos)} pasos. "
                f"Tras la limpieza sugerida quedan {limpio[0]} filas y {limpio[1]} columnas."
            ),
            "recomendaciones": prioritarias or ["Los datos están en buen estado para modelar."],
            "advertencia": "Las interpretaciones son automáticas y basadas en reglas estadísticas: deben validarse "
                           "con conocimiento del dominio. Correlación no implica causalidad.",
        }
        self._registrar(
            "autoevaluación",
            f"{len(self.pasos)} pasos ejecutados, {len(prioritarias)} recomendaciones prioritarias.",
            "Las conclusiones deben ser verificables y declarar su incertidumbre.",
            "Generar la síntesis final.",
        )

    # ==================================================================
    # Informe HTML (autocontenido: los gráficos van incrustados)
    # ==================================================================
    @staticmethod
    def _figura_a_base64(figura) -> str:
        buffer = io.BytesIO()
        figura.tight_layout()
        figura.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        return base64.b64encode(buffer.getvalue()).decode()

    def informe_html(self) -> str:
        if not self.sintesis:
            self.ejecutar()
        e = html.escape
        partes = [
            "<html><head><meta charset='utf-8'><title>Informe del Agente EDA</title><style>"
            "body{font-family:Arial,sans-serif;max-width:1000px;margin:30px auto;line-height:1.5;color:#222}"
            "h1,h2{color:#1f4e79}section{border-bottom:1px solid #ddd;padding:12px 0}"
            "table{border-collapse:collapse;font-size:13px;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 7px}"
            "img{max-width:100%}.alta{color:#c9302c}.media{color:#ec971f}</style></head><body>",
            "<h1>Informe del Agente Inteligente de EDA</h1>",
            f"<p>{e(self.sintesis['resumen'])}</p>",
            "<h2>Plan del agente</h2><ul>",
        ]
        for item in self.plan:
            estado = "✔ ejecutado" if item["ejecutar"] else "✘ omitido"
            partes.append(f"<li><b>{e(item['paso'])}</b> — {estado}: {e(item['razon'])}</li>")
        partes.append("</ul>")
        for paso in self.pasos:
            partes.append(f"<section><h2>Paso {paso.numero}. {e(paso.titulo)} "
                          f"<span class='{paso.severidad}'>({e(paso.severidad)})</span></h2>")
            for etiqueta, texto in (("Observación", paso.observacion), ("Razonamiento", paso.razonamiento),
                                    ("Acción", paso.accion), ("Interpretación", paso.interpretacion)):
                partes.append(f"<p><b>{etiqueta}:</b> {e(texto)}</p>")
            for nombre, tabla in paso.tablas.items():
                partes.append(f"<h4>{e(nombre)}</h4>{tabla.to_html(border=0)}")
            for titulo, figura in paso.figuras:
                partes.append(f"<h4>{e(titulo)}</h4><img src='data:image/png;base64,{self._figura_a_base64(figura)}'>")
            partes.append("<p><b>Recomendaciones:</b></p><ul>")
            partes.extend(f"<li>{e(r)}</li>" for r in paso.recomendaciones)
            partes.append("</ul></section>")
        partes.append("<h2>Síntesis y recomendaciones prioritarias</h2><ul>")
        partes.extend(f"<li>{e(r)}</li>" for r in self.sintesis["recomendaciones"])
        partes.append(f"</ul><p><i>{e(self.sintesis['advertencia'])}</i></p>")
        partes.append("<h2>Bitácora de razonamiento</h2><table><tr><th>Etapa</th><th>Observación</th>"
                      "<th>Razonamiento</th><th>Acción</th></tr>")
        for fila in self.bitacora:
            partes.append("<tr>" + "".join(f"<td>{e(str(fila[k]))}</td>"
                                           for k in ("etapa", "observacion", "razonamiento", "accion")) + "</tr>")
        partes.append("</table></body></html>")
        return "\n".join(partes)

    def guardar_informe(self, ruta: str = "informe_eda.html") -> str:
        with open(ruta, "w", encoding="utf-8") as archivo:
            archivo.write(self.informe_html())
        return ruta

    def cerrar_figuras(self) -> None:
        """Libera la memoria de las figuras de matplotlib."""
        for paso in self.pasos:
            for _, figura in paso.figuras:
                plt.close(figura)