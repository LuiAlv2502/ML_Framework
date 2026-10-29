import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pages._shared import obtener_modelo
from pages._theme import AMARILLO, CIAN, MAGENTA, ROJO as ROJO_NEON, VERDE as VERDE_NEON, brillo, grafico
from supervised import ClasificacionModelos


ALGORITMOS = {
    "K Vecinos Más Cercanos": "knn",
    "Árbol de decisión": "decision_tree",
    "Bosques aleatorios": "random_forest",
    "Potenciación (Gradient Boosting)": "gradient_boosting",
    "AdaBoost": "adaboost",
    "XGBoost": "xgboost",
}
ADABOOST_OPT = "AdaBoost optimizado (búsqueda de cuadrícula)"
OPCIONES_ALGORITMO = [*ALGORITMOS, ADABOOST_OPT]

ABREVIATURAS = {
    "K Vecinos Más Cercanos": "knn",
    "Árbol de decisión": "arbol",
    "Bosques aleatorios": "rf",
    "Potenciación (Gradient Boosting)": "gb",
    "AdaBoost": "ada",
    "XGBoost": "xgb",
    ADABOOST_OPT: "ada-opt",
}

MAX_CLASES = 30
COLOR_APRENDIZAJE = "rgba(0, 240, 255, 0.25)"
COLOR_PRUEBA = "rgba(255, 42, 109, 0.30)"
VERDE = VERDE_NEON
ROJO = ROJO_NEON
FONDO_ACIERTO = "rgba(0, 255, 159, 0.22)"
FONDO_ERROR = "rgba(255, 0, 60, 0.22)"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _crear_clasificador(df, objetivo, predictores, test_size, semilla, estratificar, escalar):
    return ClasificacionModelos(
        df,
        target=objetivo,
        feature_columns=predictores,
        test_size=test_size,
        random_state=int(semilla),
        stratify=estratificar,
        scale=escalar,
    )


def _tabla_predicciones(split, modelo):
    predicciones = modelo.predict(split.X_test)
    resultado = split.X_test.copy()
    resultado.insert(0, "Real", split.y_test.to_numpy())
    resultado.insert(1, "Predicción", predicciones)
    resultado.insert(
        2,
        "Correcto",
        resultado["Real"].to_numpy() == resultado["Predicción"].to_numpy(),
    )
    return resultado


def _colorear_particion(valor):
    if valor == "Aprendizaje":
        return f"background-color: {COLOR_APRENDIZAJE}; color: {CIAN}"
    if valor == "Prueba":
        return f"background-color: {COLOR_PRUEBA}; color: {MAGENTA}"
    return ""


def _parametros_modelo(algoritmo, n_filas, profundidad_opts):
    """Dibuja las opciones del algoritmo elegido y devuelve sus parámetros."""
    p = {}
    k = f"param_{algoritmo}"
    if algoritmo == "K Vecinos Más Cercanos":
        max_vecinos = max(1, min(50, n_filas))
        p["n_neighbors"] = st.slider(
            "K (número de vecinos)", 1, max_vecinos, min(5, max_vecinos), key=f"{k}_k"
        )
        p["algorithm"] = st.selectbox(
            "Método de búsqueda",
            ["auto", "ball_tree", "kd_tree", "brute"],
            key=f"{k}_alg",
        )
    elif algoritmo == "Árbol de decisión":
        p["max_depth"] = st.selectbox(
            "Profundidad máxima",
            profundidad_opts,
            format_func=lambda v: "Sin límite" if v is None else str(v),
            key=f"{k}_depth",
        )
        p["min_samples_split"] = st.slider(
            "Mínimo de muestras para dividir", 2, 20, 2, key=f"{k}_split"
        )
    elif algoritmo == "Bosques aleatorios":
        p["n_estimators"] = st.slider("Cantidad de árboles", 10, 500, 200, 10, key=f"{k}_n")
        p["max_depth"] = st.selectbox(
            "Profundidad máxima",
            profundidad_opts,
            format_func=lambda v: "Sin límite" if v is None else str(v),
            key=f"{k}_depth",
        )
        p["min_samples_split"] = st.slider(
            "Mínimo de muestras para dividir", 2, 20, 2, key=f"{k}_split"
        )
    elif algoritmo == "Potenciación (Gradient Boosting)":
        p["n_estimators"] = st.slider("Cantidad de estimadores", 10, 500, 120, 10, key=f"{k}_n")
        p["max_depth"] = st.slider("Profundidad máxima", 1, 10, 2, key=f"{k}_depth")
    elif algoritmo == "AdaBoost":
        p["n_estimators"] = st.slider("Cantidad de estimadores", 10, 300, 80, 10, key=f"{k}_n")
    elif algoritmo == "XGBoost":
        p["n_estimators"] = st.slider("Cantidad de estimadores", 10, 500, 120, 10, key=f"{k}_n")
        p["max_depth"] = st.slider("Profundidad máxima", 1, 15, 3, key=f"{k}_depth")
        p["learning_rate"] = st.slider("Learning rate", 0.01, 1.0, 0.1, 0.01, key=f"{k}_lr")
    elif algoritmo == ADABOOST_OPT:
        p["n_estimators_lista"] = st.text_input(
            "n_estimators a probar", value="40, 80, 120", key=f"{k}_ne"
        )
        p["learning_rate_lista"] = st.text_input(
            "learning_rate a probar", value="0.5, 1.0", key=f"{k}_lr"
        )
        p["cv"] = st.slider("Pliegues de validación cruzada", 2, 10, 5, key=f"{k}_cv")
    return p


def _leer_lista_numeros(valor, tipo, nombre):
    try:
        valores = [tipo(e.strip()) for e in valor.split(",")]
    except ValueError as error:
        raise ValueError(
            f"Revise los valores de {nombre}; deben ser números separados por comas."
        ) from error
    if len(set(valores)) != len(valores):
        raise ValueError(f"Los valores de {nombre} no deben repetirse.")
    if any(elemento <= 0 for elemento in valores):
        raise ValueError(f"Todos los valores de {nombre} deben ser mayores que cero.")
    return valores


def _validar_objetivo_clasificacion(df, objetivo, test_size, estratificar):
    y = df[objetivo]
    if y.isna().any():
        return "La variable objetivo contiene valores vacíos. Corríjalos antes de generar la partición."

    clases = y.nunique()
    observaciones = len(y)
    if clases > MAX_CLASES:
        return (
            f"'{objetivo}' tiene {clases} valores distintos en {observaciones} "
            "filas; parece una variable continua (por ejemplo, un precio), no "
            "una variable de clase. Seleccione una columna categórica para "
            f"clasificación (como máximo {MAX_CLASES} clases)."
        )

    if estratificar:
        conteos = y.value_counts()
        clases_insuficientes = int((conteos < 2).sum())
        if clases_insuficientes:
            return (
                f"No se puede estratificar '{objetivo}': "
                f"{clases_insuficientes} clases tienen menos de 2 filas. "
                "Elija una variable de clase con suficientes ejemplos por "
                "categoría o desactive la estratificación si esas clases "
                "individuales son intencionales."
            )

        filas_prueba = int(np.ceil(observaciones * test_size))
        filas_aprendizaje = observaciones - filas_prueba
        if min(filas_prueba, filas_aprendizaje) < clases:
            return (
                f"La partición no tiene suficientes filas para representar "
                f"las {clases} clases en entrenamiento y prueba. Aumente el "
                "tamaño de ambos grupos o seleccione una variable con menos clases."
            )

    return None


# ----- Índices de evaluación (como en discoveR) ----------------------------
def _etiquetas(metricas):
    etiquetas = metricas.get("confusion_matrix_labels")
    if etiquetas is not None:
        return [str(e) for e in etiquetas]
    resumen = {"accuracy", "macro avg", "micro avg", "weighted avg", "samples avg"}
    return [
        str(etiqueta)
        for etiqueta, valores in metricas["classification_report"].items()
        if isinstance(valores, dict) and etiqueta not in resumen
    ]


def _indices(resultado, nombres_clase=None):
    """Devuelve matriz de confusión, etiquetas, precisión global y por clase (en %)."""
    metricas = resultado["metricas"]
    cm = np.asarray(metricas["confusion_matrix"])
    etiquetas = resultado.get("etiquetas") or _etiquetas(metricas)
    if nombres_clase:
        etiquetas = [
            nombres_clase.get(str(etiqueta), str(etiqueta))
            for etiqueta in etiquetas
        ]
    totales = cm.sum(axis=1).astype(float)
    aciertos = np.diag(cm).astype(float)
    por_clase = np.divide(aciertos, totales, out=np.zeros_like(aciertos), where=totales > 0) * 100
    global_ = cm.trace() / cm.sum() * 100 if cm.sum() else 0.0
    return cm, etiquetas, float(global_), dict(zip(etiquetas, por_clase.tolist()))


def _etiqueta_modelo(algoritmo, parametros, escalar):
    partes = [ABREVIATURAS[algoritmo]]
    partes += [str(v).replace(" ", "") for v in parametros.values() if v is not None]
    if not escalar:
        partes.append("sin_escala")
    return "-".join(partes)


def _entrenar(algoritmo, parametros, escalar, clasificador):
    split = clasificador.prepare_data()
    if algoritmo == ADABOOST_OPT:
        param_grid = {
            "n_estimators": _leer_lista_numeros(
                parametros["n_estimators_lista"], int, "n_estimators"
            ),
            "learning_rate": _leer_lista_numeros(
                parametros["learning_rate_lista"], float, "learning_rate"
            ),
        }
        resultado = clasificador.adaboost_grid(param_grid, cv=parametros["cv"])
    else:
        resultado = getattr(clasificador, ALGORITMOS[algoritmo])(**parametros)
    resultado["predicciones"] = _tabla_predicciones(split, resultado["model"])
    resultado["etiquetas"] = _etiquetas(resultado["metricas"])
    return resultado


def _mostrar_generacion(label, algoritmo, parametros, escalar, resultado, exp_idx, semilla):
    _, _, global_, _ = _indices(resultado)
    m1, m2 = st.columns(2)
    m1.metric("Precisión global", f"{global_:.2f}%")
    m2.metric("Error global", f"{100 - global_:.2f}%")

    lineas = [
        f"Modelo: {label}",
        f"Algoritmo: {algoritmo}",
        f"Experimento: {exp_idx + 1} (semilla {semilla})",
        f"Datos escalados: {'Sí' if escalar else 'No'}",
        "",
        "Parámetros:",
    ]
    lineas += [f"  {k}: {v}" for k, v in parametros.items()]
    if "best_params" in resultado:
        lineas += [
            "",
            f"Mejores parámetros: {resultado['best_params']}",
            f"Mejor accuracy en validación cruzada: {resultado['best_cv_score']:.4f}",
        ]
    st.code("\n".join(lineas), language=None)


def _mostrar_indices(label, resultado, exp_idx, nombres_clase):
    cm, etiquetas, global_, por_clase = _indices(resultado, nombres_clase)

    # --- Matriz de confusión (verde = acierto, rojo = error) ---
    totales = cm.sum(axis=1)
    totales_seguros = np.where(totales == 0, 1, totales)
    texto = [
        [f"{cm[i, j]} ({cm[i, j] / totales_seguros[i] * 100:.0f}%)" for j in range(len(etiquetas))]
        for i in range(len(etiquetas))
    ]
    matriz = pd.DataFrame(texto, index=etiquetas, columns=etiquetas)
    matriz.index.name = "Real"
    matriz.columns.name = "Predicción"
    diagonal = np.eye(len(etiquetas), dtype=bool)

    def _estilos(_):
        arreglo = np.where(
            diagonal,
            f"background-color: {FONDO_ACIERTO}; color: {VERDE}; font-weight: 700; text-align: center",
            f"background-color: {FONDO_ERROR}; color: #FF6B8B; text-align: center",
        )
        return pd.DataFrame(arreglo, index=matriz.index, columns=matriz.columns)

    st.markdown("**Matriz de confusión**")
    st.table(matriz.style.apply(_estilos, axis=None))

    st.divider()
    col_graf, col_tabla = st.columns(2)

    # --- Gráfico de precisión / error por clase ---
    with col_graf:
        categorias = ["Global", *etiquetas]
        precision = [global_, *[por_clase[e] for e in etiquetas]]
        datos = pd.DataFrame(
            {
                "Categoría": categorias * 2,
                "Tipo": ["Precisión"] * len(categorias) + ["Error"] * len(categorias),
                "Valor": precision + [100 - p for p in precision],
            }
        )
        figura = px.bar(
            datos,
            x="Valor",
            y="Categoría",
            color="Tipo",
            orientation="h",
            text=datos["Valor"].map(lambda v: f"{v:.2f}%"),
            color_discrete_map={"Precisión": VERDE, "Error": ROJO},
            category_orders={"Categoría": categorias[::-1]},
        )
        figura.update_traces(textfont=dict(color="#000"), selector=dict(name="Precisión"))
        figura.update_xaxes(range=[0, 100], title="%")
        figura.update_layout(barmode="stack", legend_title_text="")
        grafico(figura, key=f"barras_{label}_{exp_idx}", height=max(300, 80 * len(categorias)))

    # --- Tabla de datos reales vs. predicción ---
    with col_tabla:
        pred = resultado["predicciones"]
        if pd.api.types.is_integer_dtype(pred.index):
            ids = pred.index.to_numpy() + 1
        else:
            ids = np.arange(1, len(pred) + 1)
        tabla = pd.DataFrame(
            {
                "ID": ids,
                "Datos Reales": pred["Real"]
                .map(lambda valor: nombres_clase.get(str(valor), str(valor)))
                .to_numpy(),
                "Predicción": pred["Predicción"]
                .map(lambda valor: nombres_clase.get(str(valor), str(valor)))
                .to_numpy(),
                "Correcto": pred["Correcto"].to_numpy(),
            }
        ).sort_values("ID").reset_index(drop=True)
        correcto = tabla.pop("Correcto").to_numpy()

        def _estilos_filas(_):
            arreglo = np.where(
                correcto[:, None],
                f"background-color: {FONDO_ACIERTO}; color: {VERDE}",
                f"background-color: {FONDO_ERROR}; color: #FF6B8B",
            )
            arreglo = np.tile(arreglo[:, :1], (1, tabla.shape[1]))
            return pd.DataFrame(arreglo, index=tabla.index, columns=tabla.columns)

        st.dataframe(
            tabla.style.apply(_estilos_filas, axis=None),
            width="stretch",
            hide_index=True,
            height=380,
        )
        st.download_button(
            "Descargar predicciones CSV",
            data=pred.to_csv(index=False).encode("utf-8"),
            file_name=f"predicciones_{label}_exp{exp_idx + 1}.csv",
            mime="text/csv",
            key=f"descargar_{label}_{exp_idx}",
        )


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
st.title("Supervisado - Clasificación")

model = obtener_modelo()
if model is None:
    st.stop()

df = model.dataframe.dataframe
columnas = df.columns.tolist()
if len(columnas) < 2:
    st.warning("El archivo debe tener una columna objetivo y al menos un predictor.")
    st.stop()

# ---------------------------------------------------------------------------
# PASO 1: Generar datos (partición aprendizaje / prueba)
# ---------------------------------------------------------------------------
st.header("1. Generar datos")

with st.container(border=True):
    cabecera, boton_col = st.columns([4, 1])
    cabecera.markdown("**Opciones de partición**")

    objetivo_inicial = next(
        (columnas.index(c) for c in ("Drug", "FuelType") if c in columnas),
        next(
            (i for i, c in enumerate(columnas) if df[c].nunique() <= MAX_CLASES),
            0,
        ),
    )
    objetivo = st.selectbox(
        "Seleccione la variable a predecir",
        columnas,
        index=objetivo_inicial,
    )
    objetivo_clases = df[objetivo].nunique()
    if objetivo_clases > MAX_CLASES:
        st.warning(
            f"'{objetivo}' tiene {objetivo_clases} valores distintos y parece "
            "una variable continua. Para clasificación, seleccione una columna "
            "que contenga categorías (por ejemplo, FuelType)."
        )

    clases_originales = sorted(df[objetivo].dropna().unique(), key=lambda valor: str(valor))
    with st.expander("Nombres de las clases (opcional)", expanded=False):
        st.caption(
            "Si el CSV usa códigos como 0, 1, 2, escriba aquí sus nombres. "
            "Esto solo cambia las etiquetas mostradas; no modifica el entrenamiento."
        )
        tabla_nombres = pd.DataFrame(
            {
                "Código original": [str(clase) for clase in clases_originales],
                "Nombre para mostrar": [str(clase) for clase in clases_originales],
            }
        )
        tabla_nombres = st.data_editor(
            tabla_nombres,
            key=f"nombres_clases_{objetivo}",
            hide_index=True,
            width="stretch",
            disabled=["Código original"],
            column_config={
                "Código original": st.column_config.TextColumn(),
                "Nombre para mostrar": st.column_config.TextColumn(
                    help="Nombre legible que se verá en índices y comparaciones."
                ),
            },
        )

    valores_nombres = tabla_nombres["Nombre para mostrar"].astype(str).str.strip()
    if valores_nombres.eq("").any() or valores_nombres.duplicated().any():
        st.warning(
            "Los nombres para mostrar deben estar completos y ser únicos. "
            "Se conservarán temporalmente los códigos originales."
        )
        nombres_clase = {
            str(clase): str(clase)
            for clase in clases_originales
        }
    else:
        nombres_clase = dict(
            zip(tabla_nombres["Código original"], valores_nombres)
        )

    predictores_disponibles = [c for c in columnas if c != objetivo]
    with st.expander("Variables predictoras", expanded=False):
        predictores = st.multiselect(
            "Variables predictoras",
            predictores_disponibles,
            default=predictores_disponibles,
            label_visibility="collapsed",
            help="La variable objetivo se excluye automáticamente.",
        )

    s1, s2 = st.columns([2, 1])
    modo_semilla = s1.radio(
        "Semilla aleatoria",
        ["Habilitada", "Deshabilitada"],
        horizontal=True,
        help="Habilitada: resultados reproducibles. Deshabilitada: cada partición usa una semilla al azar.",
    )
    semilla = s2.number_input(
        "Semilla",
        min_value=0,
        max_value=2_147_483_647,
        value=5,
        step=1,
        disabled=(modo_semilla == "Deshabilitada"),
    )

    porc_aprendizaje = st.slider(
        "Aprendizaje (%)  ←→  Prueba (%)",
        min_value=5,
        max_value=95,
        value=80,
        step=5,
        help="Porcentaje de filas para aprendizaje; el resto es prueba.",
    )
    st.caption(f"Aprendizaje: {porc_aprendizaje}%  ·  Prueba: {100 - porc_aprendizaje}%")

    n_experimentos = st.number_input(
        "Cantidad de experimentos", min_value=1, max_value=50, value=1, step=1
    )
    estratificar = False  # la partición es aleatoria simple, sin estratificar

    ejecutar_particion = boton_col.button("▶ Ejecutar", type="primary", key="btn_particion")

test_size = 1 - porc_aprendizaje / 100
configuracion_actual = {
    "objetivo": objetivo,
    "predictores": predictores,
    "test_size": test_size,
    "estratificar": estratificar,
    "n_experimentos": int(n_experimentos),
    "modo_semilla": modo_semilla,
    "semilla_base": int(semilla) if modo_semilla == "Habilitada" else None,
    "dataset_hash": st.session_state.get("archivo_hash"),
}

if ejecutar_particion:
    st.session_state.pop("clf_resultados", None)
    st.session_state.pop("clf_particiones", None)

    if not predictores:
        st.error("Seleccione al menos una variable predictora.")
    else:
        try:
            problema_objetivo = _validar_objetivo_clasificacion(
                df,
                objetivo,
                test_size,
                estratificar,
            )
            if problema_objetivo:
                raise ValueError(problema_objetivo)

            if modo_semilla == "Habilitada":
                semillas = [
                    (int(semilla) + i) % (2**32 - 1)
                    for i in range(int(n_experimentos))
                ]
            else:
                semillas = np.random.default_rng().integers(0, 10**6, int(n_experimentos)).tolist()

            experimentos = []
            for s in semillas:
                clf = _crear_clasificador(df, objetivo, predictores, test_size, s, estratificar, False)
                split = clf.prepare_data()
                particion = pd.Series("Aprendizaje", index=df.index, name="Partición")
                particion.loc[split.X_test.index] = "Prueba"
                experimentos.append(
                    {
                        "semilla": int(s),
                        "particion": particion,
                        "n_train": len(split.X_train),
                        "n_test": len(split.X_test),
                    }
                )

            st.session_state["clf_particiones"] = {
                "objetivo": objetivo,
                "predictores": predictores,
                "test_size": test_size,
                "estratificar": estratificar,
                "n_experimentos": int(n_experimentos),
                "modo_semilla": modo_semilla,
                "semilla_base": configuracion_actual["semilla_base"],
                "dataset_hash": configuracion_actual["dataset_hash"],
                "experimentos": experimentos,
            }
        except (ImportError, KeyError, ValueError) as error:
            st.error(f"No se pudo generar la partición: {error}")

config = st.session_state.get("clf_particiones")
if config is None:
    st.info("Configure la partición y presione **Ejecutar** para continuar.")
    st.stop()

campos_configuracion = (
    "objetivo",
    "predictores",
    "test_size",
    "estratificar",
    "n_experimentos",
    "modo_semilla",
    "semilla_base",
    "dataset_hash",
)
if any(config.get(campo) != configuracion_actual[campo] for campo in campos_configuracion):
    st.warning(
        "Cambió la configuración de los datos. Presione **Ejecutar** en el paso 1 "
        "para generar una nueva partición antes de entrenar o ver resultados."
    )
    st.stop()

# ---------------------------------------------------------------------------
# PASO 2: Datos con la columna Partición
# ---------------------------------------------------------------------------
st.header("2. Datos")

n_exp = len(config["experimentos"])
exp_datos = st.selectbox(
    "Experimento",
    list(range(1, n_exp + 1)),
    key="exp_seleccionado",
) - 1
exp = config["experimentos"][exp_datos]

d1, d2, d3 = st.columns(3)
d1.metric("Filas de aprendizaje", exp["n_train"])
d2.metric("Filas de prueba", exp["n_test"])
d3.metric("Semilla usada", exp["semilla"])

tabla_datos = df.copy()
tabla_datos.insert(0, "Partición", exp["particion"])
st.dataframe(
    tabla_datos.style.map(_colorear_particion, subset=["Partición"]),
    width="stretch",
    height=320,
)
st.download_button(
    "Descargar datos con partición (CSV)",
    data=tabla_datos.to_csv(index=False).encode("utf-8"),
    file_name=f"datos_particion_exp{exp_datos + 1}.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# PASO 3: Modelos
# ---------------------------------------------------------------------------
st.header("3. Modelos")

if "clf_resultados" not in st.session_state:
    st.session_state["clf_resultados"] = {}
# cache[etiqueta_modelo] = {"algoritmo": str, "experimentos": {idx: resultado}}
cache = st.session_state["clf_resultados"]

profundidad_opts = [None, *range(1, 31)]
min_train = min(e["n_train"] for e in config["experimentos"])
clases_objetivo = [
    nombres_clase.get(str(clase), str(clase))
    for clase in clases_originales
]


def _clasificador_experimento(idx, escalar):
    return _crear_clasificador(
        df,
        config["objetivo"],
        config["predictores"],
        config["test_size"],
        config["experimentos"][idx]["semilla"],
        config["estratificar"],
        escalar,
    )


fila_a, fila_b = st.columns([3, 1])
algoritmo = fila_a.selectbox("Algoritmo", OPCIONES_ALGORITMO, key="algoritmo_elegido")
fila_b.caption(f"Experimentos generados: {n_exp}")

tab_gen, tab_ind, tab_var, tab_comp = st.tabs(
    [
        "Generación del Modelo",
        "Índices Generales",
        "Variación del Error",
        "Comparación de Modelos",
    ]
)

# ----- Generación del modelo -----
with tab_gen:
    exp_gen_idx = st.selectbox(
        "Experimento",
        list(range(1, n_exp + 1)),
        key="exp_generacion",
    ) - 1
    col_opts, col_out = st.columns([1, 2])

    with col_opts.container(border=True):
        cab, btn = st.columns([2, 1])
        cab.markdown("**Opciones**")
        ejecutar_modelo = btn.button("▶ Ejecutar", type="primary", key="btn_modelo")

        parametros = _parametros_modelo(algoritmo, min_train, profundidad_opts)
        escalar = st.radio(
            "Escalar datos", ["SI", "NO"], horizontal=True, key="escalar_modelo"
        ) == "SI"

    etiqueta = _etiqueta_modelo(algoritmo, parametros, escalar)

    if ejecutar_modelo:
        try:
            with st.spinner(f"Entrenando {etiqueta} en {n_exp} experimento(s)..."):
                resultados_exp = {}
                for i in range(n_exp):
                    clf = _clasificador_experimento(i, escalar)
                    resultados_exp[i] = _entrenar(
                        algoritmo,
                        parametros,
                        escalar,
                        clf,
                    )
            cache[etiqueta] = {"algoritmo": algoritmo, "experimentos": resultados_exp}
        except (ImportError, KeyError, ValueError) as error:
            st.error(f"No se pudo generar el modelo: {error}")

    resultado_actual = cache.get(etiqueta, {}).get("experimentos", {}).get(exp_gen_idx)

    with col_out:
        if resultado_actual is None:
            st.code("No se ha generado el modelo.", language=None)
        else:
            _mostrar_generacion(
                etiqueta,
                algoritmo,
                parametros,
                escalar,
                resultado_actual,
                exp_gen_idx,
                config["experimentos"][exp_gen_idx]["semilla"],
            )

# ----- Índices generales -----
with tab_ind:
    control_exp, _ = st.columns([1, 4])
    exp_indices_idx = control_exp.selectbox(
        "Experimento",
        list(range(1, n_exp + 1)),
        key="exp_indices_generales",
    ) - 1
    resultado_indices = cache.get(etiqueta, {}).get("experimentos", {}).get(
        exp_indices_idx
    )
    if resultado_indices is None:
        st.info("Genere el modelo en la pestaña **Generación del Modelo** para ver sus índices.")
    else:
        _mostrar_indices(
            etiqueta,
            resultado_indices,
            exp_indices_idx,
            nombres_clase,
        )

# ----- Variación del error -----
with tab_var:
    v1, v2 = st.columns(2)
    indice = v1.selectbox("Seleccione el índice", ["Error", "Precisión"], key="var_indice")
    categoria = v2.selectbox("Categoría", ["Global", *clases_objetivo], key="var_categoria")

    filas = []
    for nombre, entrada in cache.items():
        for i, r in entrada["experimentos"].items():
            _, _, global_, por_clase = _indices(r, nombres_clase)
            valor = global_ if categoria == "Global" else por_clase.get(categoria)
            if valor is None:
                continue
            filas.append(
                {
                    "Modelo": nombre,
                    "Número de experimento": i + 1,
                    indice: valor if indice == "Precisión" else 100 - valor,
                }
            )

    if not filas:
        st.info("Ejecute al menos un modelo para ver cómo varía entre experimentos.")
    else:
        datos_var = pd.DataFrame(filas)
        figura = px.line(
            datos_var,
            x="Número de experimento",
            y=indice,
            color="Modelo",
            markers=True,
        )
        figura.update_traces(line=dict(width=3), marker=dict(size=10, symbol="diamond"))
        figura.update_xaxes(dtick=1)
        figura.update_yaxes(title=f"{indice} {categoria} (%)", range=[0, 100])
        grafico(brillo(figura), key="grafico_variacion", height=450)

# ----- Comparación de modelos -----
with tab_comp:
    if not cache:
        st.info(
            "Todavía no hay modelos para comparar. Ejecute los que quiera en la pestaña "
            "**Generación del Modelo** (cada uno con las variantes que prefiera) y aparecerán aquí."
        )
    else:
        cc1, cc2 = st.columns([3, 1])
        exp_comp = cc2.selectbox(
            "Experimento",
            list(range(1, n_exp + 1)),
            key="exp_comparacion",
        ) - 1
        etiquetas_disponibles = list(cache)
        elegidos = cc1.multiselect(
            "Modelos a comparar",
            etiquetas_disponibles,
            default=etiquetas_disponibles,
        )

        filas = []
        for nombre in elegidos:
            r = cache[nombre]["experimentos"].get(exp_comp)
            if r is None:
                continue
            _, _, global_, por_clase = _indices(r, nombres_clase)
            fila = {"Modelo": nombre, "Precisión Global": global_}
            for clase, valor in por_clase.items():
                fila[f"Precisión {clase}"] = valor
            filas.append(fila)

        tab_tabla, tab_grafico = st.tabs(["Tabla Comparativa", "Gráfico Comparativo"])
        if not filas:
            tab_tabla.info("Seleccione al menos un modelo.")
        else:
            comparacion = pd.DataFrame(filas)
            columnas_num = [c for c in comparacion.columns if c != "Modelo"]
            with tab_tabla:
                st.dataframe(
                    comparacion.style.format({c: "{:.2f}" for c in columnas_num}, na_rep="-"),
                    width="stretch",
                    hide_index=True,
                )
            with tab_grafico:
                largo = comparacion.melt(
                    id_vars="Modelo", var_name="Serie", value_name="Precisión"
                )
                largo["Serie"] = largo["Serie"].str.replace("Precisión ", "", regex=False)
                figura = px.bar(
                    largo,
                    x="Modelo",
                    y="Precisión",
                    color="Serie",
                    barmode="group",
                    text=largo["Precisión"].map(lambda v: f"{v:.1f}%"),
                    color_discrete_map={"Global": AMARILLO},
                )
                figura.update_yaxes(range=[0, 100], title="Precisión (%)")
                # Como en discoveR: solo "Global" visible; las clases se activan en la leyenda
                figura.for_each_trace(
                    lambda t: t.update(visible="legendonly") if t.name != "Global" else None
                )
                grafico(figura, key="grafico_comparativo", height=450)

        if st.button("Quitar todos los modelos ejecutados", key="btn_limpiar_modelos"):
            cache.clear()
            st.rerun()