import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components.shared import ejecutar, obtener_datos, pie_de_ejecucion
from frontend.components.supervised_agent import seccion_agente
from frontend.components.theme import AMARILLO, CIAN, MAGENTA, ROJO as ROJO_NEON, VERDE as VERDE_NEON, brillo, grafico

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
# Utilidades (solo presentación y validación ligera; el cálculo lo hace el backend)
# ---------------------------------------------------------------------------
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
        p["n_neighbors"] = st.slider("K (número de vecinos)", 1, max_vecinos, min(5, max_vecinos), key=f"{k}_k")
        p["algorithm"] = st.selectbox("Método de búsqueda", ["auto", "ball_tree", "kd_tree", "brute"], key=f"{k}_alg")
    elif algoritmo == "Árbol de decisión":
        p["max_depth"] = st.selectbox("Profundidad máxima", profundidad_opts,
                                      format_func=lambda v: "Sin límite" if v is None else str(v), key=f"{k}_depth")
        p["min_samples_split"] = st.slider("Mínimo de muestras para dividir", 2, 20, 2, key=f"{k}_split")
    elif algoritmo == "Bosques aleatorios":
        p["n_estimators"] = st.slider("Cantidad de árboles", 10, 500, 200, 10, key=f"{k}_n")
        p["max_depth"] = st.selectbox("Profundidad máxima", profundidad_opts,
                                      format_func=lambda v: "Sin límite" if v is None else str(v), key=f"{k}_depth")
        p["min_samples_split"] = st.slider("Mínimo de muestras para dividir", 2, 20, 2, key=f"{k}_split")
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
        p["n_estimators_lista"] = st.text_input("n_estimators a probar", value="40, 80, 120", key=f"{k}_ne")
        p["learning_rate_lista"] = st.text_input("learning_rate a probar", value="0.5, 1.0", key=f"{k}_lr")
        p["cv"] = st.slider("Pliegues de validación cruzada", 2, 10, 5, key=f"{k}_cv")
    return p


def _leer_lista_numeros(valor, tipo, nombre):
    try:
        valores = [tipo(e.strip()) for e in valor.split(",")]
    except ValueError as error:
        raise ValueError(f"Revise los valores de {nombre}; deben ser números separados por comas.") from error
    if len(set(valores)) != len(valores):
        raise ValueError(f"Los valores de {nombre} no deben repetirse.")
    if any(elemento <= 0 for elemento in valores):
        raise ValueError(f"Todos los valores de {nombre} deben ser mayores que cero.")
    return valores


def _parametros_backend(algoritmo, parametros):
    """Traduce los controles de la UI a los parámetros que valida el agente."""
    if algoritmo != ADABOOST_OPT:
        return ALGORITMOS[algoritmo], parametros
    return "adaboost_grid", {
        "n_estimators": _leer_lista_numeros(parametros["n_estimators_lista"], int, "n_estimators"),
        "learning_rate": _leer_lista_numeros(parametros["learning_rate_lista"], float, "learning_rate"),
        "cv": parametros["cv"],
    }


def _validar_objetivo_clasificacion(df, objetivo, test_size, estratificar):
    y = df[objetivo]
    if y.isna().any():
        return "La variable objetivo contiene valores vacíos. Corríjalos antes de generar la partición."
    clases = y.nunique()
    observaciones = len(y)
    if clases > MAX_CLASES:
        return (f"'{objetivo}' tiene {clases} valores distintos en {observaciones} filas; parece una variable "
                "continua (por ejemplo, un precio), no una variable de clase. Seleccione una columna categórica "
                f"para clasificación (como máximo {MAX_CLASES} clases).")
    if estratificar:
        conteos = y.value_counts()
        insuficientes = int((conteos < 2).sum())
        if insuficientes:
            return (f"No se puede estratificar '{objetivo}': {insuficientes} clases tienen menos de 2 filas.")
        filas_prueba = int(np.ceil(observaciones * test_size))
        if min(filas_prueba, observaciones - filas_prueba) < clases:
            return (f"La partición no tiene suficientes filas para representar las {clases} clases en "
                    "entrenamiento y prueba.")
    return None


def _indices(resultado, nombres_clase=None):
    """Devuelve matriz de confusión, etiquetas, precisión global y por clase (en %)."""
    cm = np.asarray(resultado["confusion_matrix"]["matrix"])
    etiquetas = [str(e) for e in resultado["confusion_matrix"]["labels"]]
    if nombres_clase:
        etiquetas = [nombres_clase.get(e, e) for e in etiquetas]
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


def _mostrar_generacion(label, algoritmo, parametros, escalar, resultado, exp_idx, semilla, agente):
    _, _, global_, _ = _indices(resultado)
    m1, m2 = st.columns(2)
    m1.metric("Precisión global", f"{global_:.2f}%")
    m2.metric("Error global", f"{100 - global_:.2f}%")
    lineas = [f"Modelo: {label}", f"Algoritmo: {algoritmo}", f"Agente: {agente}",
              f"Experimento: {exp_idx + 1} (semilla {semilla})", f"Datos escalados: {'Sí' if escalar else 'No'}",
              "", "Parámetros:"]
    lineas += [f"  {k}: {v}" for k, v in parametros.items()]
    if "best_params" in resultado:
        lineas += ["", f"Mejores parámetros: {resultado['best_params']}",
                   f"Mejor accuracy en validación cruzada: {resultado['best_cv_score']:.4f}"]
    st.code("\n".join(lineas), language=None)


def _matriz_confusion(cm, etiquetas):
    totales = np.where(cm.sum(axis=1) == 0, 1, cm.sum(axis=1))
    texto = [[f"{cm[i, j]} ({cm[i, j] / totales[i] * 100:.0f}%)" for j in range(len(etiquetas))]
             for i in range(len(etiquetas))]
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

    st.table(matriz.style.apply(_estilos, axis=None))


def _mostrar_indices(label, resultado, exp_idx, nombres_clase):
    cm, etiquetas, global_, por_clase = _indices(resultado, nombres_clase)
    st.markdown("**Matriz de confusión**")
    _matriz_confusion(cm, etiquetas)
    st.divider()
    col_graf, col_tabla = st.columns(2)
    with col_graf:
        categorias = ["Global", *etiquetas]
        precision = [global_, *[por_clase[e] for e in etiquetas]]
        datos = pd.DataFrame({
            "Categoría": categorias * 2,
            "Tipo": ["Precisión"] * len(categorias) + ["Error"] * len(categorias),
            "Valor": precision + [100 - p for p in precision],
        })
        figura = px.bar(datos, x="Valor", y="Categoría", color="Tipo", orientation="h",
                        text=datos["Valor"].map(lambda v: f"{v:.2f}%"),
                        color_discrete_map={"Precisión": VERDE, "Error": ROJO},
                        category_orders={"Categoría": categorias[::-1]})
        figura.update_traces(textfont=dict(color="#000"), selector=dict(name="Precisión"))
        figura.update_xaxes(range=[0, 100], title="%")
        figura.update_layout(barmode="stack", legend_title_text="")
        grafico(figura, key=f"barras_{label}_{exp_idx}", height=max(300, 80 * len(categorias)))
    with col_tabla:
        pred = resultado["predictions"]
        ids = pred.index.to_numpy() + 1 if pd.api.types.is_integer_dtype(pred.index) else np.arange(1, len(pred) + 1)
        tabla = pd.DataFrame({
            "ID": ids,
            "Datos Reales": pred["Real"].map(lambda v: nombres_clase.get(str(v), str(v))).to_numpy(),
            "Predicción": pred["Predicción"].map(lambda v: nombres_clase.get(str(v), str(v))).to_numpy(),
            "Correcto": pred["Correcto"].to_numpy(),
        }).sort_values("ID").reset_index(drop=True)
        correcto = tabla.pop("Correcto").to_numpy()

        def _estilos_filas(_):
            arreglo = np.where(correcto[:, None], f"background-color: {FONDO_ACIERTO}; color: {VERDE}",
                               f"background-color: {FONDO_ERROR}; color: #FF6B8B")
            arreglo = np.tile(arreglo[:, :1], (1, tabla.shape[1]))
            return pd.DataFrame(arreglo, index=tabla.index, columns=tabla.columns)

        st.dataframe(tabla.style.apply(_estilos_filas, axis=None), width="stretch", hide_index=True, height=380)
        st.download_button("Descargar predicciones CSV", data=pred.to_csv(index=False).encode("utf-8"),
                           file_name=f"predicciones_{label}_exp{exp_idx + 1}.csv", mime="text/csv",
                           key=f"descargar_{label}_{exp_idx}")


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
st.title("Supervisado - Clasificación")

ctrl = obtener_datos(recomendar_eda=True)
if ctrl is None:
    st.stop()

df = ctrl.active_data()
columnas = df.columns.tolist()
if len(columnas) < 2:
    st.warning("El archivo debe tener una columna objetivo y al menos un predictor.")
    st.stop()

seccion_agente("classification", df)

# ---------------------------------------------------------------------------
# PASO 1: Generar datos (partición aprendizaje / prueba)
# ---------------------------------------------------------------------------
st.header("1. Generar datos")

with st.container(border=True):
    cabecera, boton_col = st.columns([4, 1])
    cabecera.markdown("**Opciones de partición**")
    objetivo_inicial = next(
        (columnas.index(c) for c in ("Drug", "FuelType") if c in columnas),
        next((i for i, c in enumerate(columnas) if df[c].nunique() <= MAX_CLASES), 0),
    )
    objetivo = st.selectbox("Seleccione la variable a predecir", columnas, index=objetivo_inicial)
    etiquetas_faltantes = int(df[objetivo].isna().sum())
    if etiquetas_faltantes:
        st.warning(f"Se excluirán {etiquetas_faltantes} filas sin etiqueta en '{objetivo}'; "
                   "no se inventan valores para la variable que se va a predecir.")
    if df[objetivo].nunique() > MAX_CLASES:
        st.warning(f"'{objetivo}' tiene {df[objetivo].nunique()} valores distintos y parece una variable continua.")

    clases_originales = sorted(df[objetivo].dropna().unique(), key=lambda valor: str(valor))
    with st.expander("Nombres de las clases (opcional)", expanded=False):
        st.caption("Si el archivo usa códigos como 0, 1, 2, escriba aquí sus nombres. "
                   "Esto solo cambia las etiquetas mostradas; no modifica el entrenamiento.")
        tabla_nombres = st.data_editor(
            pd.DataFrame({"Código original": [str(c) for c in clases_originales],
                          "Nombre para mostrar": [str(c) for c in clases_originales]}),
            key=f"nombres_clases_{objetivo}", hide_index=True, width="stretch", disabled=["Código original"],
        )
    valores_nombres = tabla_nombres["Nombre para mostrar"].astype(str).str.strip()
    if valores_nombres.eq("").any() or valores_nombres.duplicated().any():
        st.warning("Los nombres para mostrar deben estar completos y ser únicos. "
                   "Se conservarán temporalmente los códigos originales.")
        nombres_clase = {str(c): str(c) for c in clases_originales}
    else:
        nombres_clase = dict(zip(tabla_nombres["Código original"], valores_nombres))

    predictores_disponibles = [c for c in columnas if c != objetivo]
    with st.expander("Variables predictoras", expanded=False):
        predictores = st.multiselect("Variables predictoras", predictores_disponibles,
                                     default=predictores_disponibles, label_visibility="collapsed")

    s1, s2 = st.columns([2, 1])
    modo_semilla = s1.radio("Semilla aleatoria", ["Habilitada", "Deshabilitada"], horizontal=True)
    semilla = s2.number_input("Semilla", min_value=0, max_value=2_147_483_647, value=5, step=1,
                              disabled=(modo_semilla == "Deshabilitada"))
    porc_aprendizaje = st.slider("Aprendizaje (%)  ←→  Prueba (%)", min_value=5, max_value=95, value=80, step=5)
    st.caption(f"Aprendizaje: {porc_aprendizaje}%  ·  Prueba: {100 - porc_aprendizaje}%")
    n_experimentos = st.number_input("Cantidad de experimentos", min_value=1, max_value=50, value=1, step=1)
    estratificar = False  # la partición es aleatoria simple, sin estratificar
    ejecutar_particion = boton_col.button("▶ Ejecutar", type="primary", key="btn_particion")

info = ctrl.dataset_info()
test_size = 1 - porc_aprendizaje / 100
configuracion_actual = {
    "objetivo": objetivo,
    "predictores": predictores,
    "test_size": test_size,
    "estratificar": estratificar,
    "n_experimentos": int(n_experimentos),
    "modo_semilla": modo_semilla,
    "semilla_base": int(semilla) if modo_semilla == "Habilitada" else None,
    "dataset_hash": info["hash"] + info["source"],
}

if ejecutar_particion:
    st.session_state.pop("clf_resultados", None)
    st.session_state.pop("clf_particiones", None)
    if not predictores:
        st.error("Seleccione al menos una variable predictora.")
    else:
        problema = _validar_objetivo_clasificacion(df, objetivo, test_size, estratificar)
        if problema:
            st.error(f"No se pudo generar la partición: {problema}")
        else:
            if modo_semilla == "Habilitada":
                semillas = [(int(semilla) + i) % (2**32 - 1) for i in range(int(n_experimentos))]
            else:
                semillas = np.random.default_rng().integers(0, 10**6, int(n_experimentos)).tolist()
            particiones = ejecutar(lambda: ctrl.partitions(task="classification", target=objetivo,
                                                           features=predictores, test_size=test_size,
                                                           seeds=semillas, stratify=estratificar),
                                   "Generando particiones...", "No se pudo generar la partición: ")
            if particiones is not None:
                experimentos = []
                for p in particiones:
                    particion = pd.Series("Aprendizaje", index=df.index, name="Partición")
                    particion.loc[p["test_index"]] = "Prueba"
                    experimentos.append({"semilla": p["seed"], "particion": particion,
                                         "n_train": p["n_train"], "n_test": p["n_test"]})
                st.session_state["clf_particiones"] = {**configuracion_actual, "experimentos": experimentos}

config = st.session_state.get("clf_particiones")
if config is None:
    st.info("Configure la partición y presione **Ejecutar** para continuar.")
    st.stop()
if any(config.get(campo) != valor for campo, valor in configuracion_actual.items()):
    st.warning("Cambió la configuración de los datos. Presione **Ejecutar** en el paso 1 "
               "para generar una nueva partición antes de entrenar o ver resultados.")
    st.stop()

# ---------------------------------------------------------------------------
# PASO 2: Datos con la columna Partición
# ---------------------------------------------------------------------------
st.header("2. Datos")
n_exp = len(config["experimentos"])
exp_datos = st.selectbox("Experimento", list(range(1, n_exp + 1)), key="exp_seleccionado") - 1
exp = config["experimentos"][exp_datos]
d1, d2, d3 = st.columns(3)
d1.metric("Filas de aprendizaje", exp["n_train"])
d2.metric("Filas de prueba", exp["n_test"])
d3.metric("Semilla usada", exp["semilla"])
tabla_datos = df.copy()
tabla_datos.insert(0, "Partición", exp["particion"])
st.dataframe(tabla_datos.style.map(_colorear_particion, subset=["Partición"]), width="stretch", height=320)
st.download_button("Descargar datos con partición (CSV)", data=tabla_datos.to_csv(index=False).encode("utf-8"),
                   file_name=f"datos_particion_exp{exp_datos + 1}.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# PASO 3: Modelos
# ---------------------------------------------------------------------------
st.header("3. Modelos")
cache = st.session_state.setdefault("clf_resultados", {})
profundidad_opts = [None, *range(1, 31)]
min_train = min(e["n_train"] for e in config["experimentos"])
clases_objetivo = [nombres_clase.get(str(c), str(c)) for c in clases_originales]
semillas_config = [e["semilla"] for e in config["experimentos"]]

fila_a, fila_b = st.columns([3, 1])
algoritmo = fila_a.selectbox("Algoritmo", OPCIONES_ALGORITMO, key="algoritmo_elegido")
fila_b.caption(f"Experimentos generados: {n_exp}")

tab_gen, tab_ind, tab_var, tab_comp, tab_bench = st.tabs(
    ["Generación del Modelo", "Índices Generales", "Variación del Error", "Comparación de Modelos", "Benchmarking"]
)

with tab_gen:
    exp_gen_idx = st.selectbox("Experimento", list(range(1, n_exp + 1)), key="exp_generacion") - 1
    col_opts, col_out = st.columns([1, 2])
    with col_opts.container(border=True):
        cab, btn = st.columns([2, 1])
        cab.markdown("**Opciones**")
        ejecutar_modelo = btn.button("▶ Ejecutar", type="primary", key="btn_modelo")
        parametros = _parametros_modelo(algoritmo, min_train, profundidad_opts)
        escalar = st.radio("Escalar datos", ["SI", "NO"], horizontal=True, key="escalar_modelo") == "SI"
    etiqueta = _etiqueta_modelo(algoritmo, parametros, escalar)

    if ejecutar_modelo:
        def _entrenar():
            clave, params = _parametros_backend(algoritmo, parametros)
            return ctrl.run_classification(target=config["objetivo"], algorithm=clave, params=params,
                                           features=config["predictores"], test_size=config["test_size"],
                                           seeds=semillas_config, stratify=config["estratificar"], scale=escalar,
                                           label=etiqueta)

        resultado = ejecutar(_entrenar, f"Entrenando {etiqueta} en {n_exp} experimento(s)...",
                             "No se pudo generar el modelo: ")
        if resultado is not None:
            cache[etiqueta] = {"algoritmo": algoritmo, "agente": resultado["selected_agent"],
                               "ejecucion": resultado,
                               "experimentos": dict(enumerate(resultado["result"]["experiments"]))}

    entrada = cache.get(etiqueta, {})
    resultado_actual = entrada.get("experimentos", {}).get(exp_gen_idx)
    with col_out:
        if resultado_actual is None:
            st.code("No se ha generado el modelo.", language=None)
        else:
            pie_de_ejecucion(entrada["ejecucion"])
            _mostrar_generacion(etiqueta, algoritmo, parametros, escalar, resultado_actual, exp_gen_idx,
                                config["experimentos"][exp_gen_idx]["semilla"], entrada["agente"])

with tab_ind:
    control_exp, _ = st.columns([1, 4])
    exp_indices_idx = control_exp.selectbox("Experimento", list(range(1, n_exp + 1)), key="exp_indices_generales") - 1
    resultado_indices = cache.get(etiqueta, {}).get("experimentos", {}).get(exp_indices_idx)
    if resultado_indices is None:
        st.info("Genere el modelo en la pestaña **Generación del Modelo** para ver sus índices.")
    else:
        _mostrar_indices(etiqueta, resultado_indices, exp_indices_idx, nombres_clase)

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
            filas.append({"Modelo": nombre, "Número de experimento": i + 1,
                          indice: valor if indice == "Precisión" else 100 - valor})
    if not filas:
        st.info("Ejecute al menos un modelo para ver cómo varía entre experimentos.")
    else:
        figura = px.line(pd.DataFrame(filas), x="Número de experimento", y=indice, color="Modelo", markers=True)
        figura.update_traces(line=dict(width=3), marker=dict(size=10, symbol="diamond"))
        figura.update_xaxes(dtick=1)
        figura.update_yaxes(title=f"{indice} {categoria} (%)", range=[0, 100])
        grafico(brillo(figura), key="grafico_variacion", height=450)

with tab_comp:
    if not cache:
        st.info("Todavía no hay modelos para comparar. Ejecute los que quiera en la pestaña "
                "**Generación del Modelo** y aparecerán aquí.")
    else:
        cc1, cc2 = st.columns([3, 1])
        exp_comp = cc2.selectbox("Experimento", list(range(1, n_exp + 1)), key="exp_comparacion") - 1
        disponibles = list(cache)
        elegidos = cc1.multiselect("Modelos a comparar", disponibles, default=disponibles)
        filas = []
        for nombre in elegidos:
            r = cache[nombre]["experimentos"].get(exp_comp)
            if r is None:
                continue
            _, _, global_, por_clase = _indices(r, nombres_clase)
            fila = {"Modelo": nombre, "Agente": cache[nombre]["agente"], "Precisión Global": global_}
            fila.update({f"Precisión {c}": v for c, v in por_clase.items()})
            filas.append(fila)
        tab_tabla, tab_grafico = st.tabs(["Tabla Comparativa", "Gráfico Comparativo"])
        if not filas:
            tab_tabla.info("Seleccione al menos un modelo.")
        else:
            comparacion = pd.DataFrame(filas)
            columnas_num = [c for c in comparacion.columns if c not in ("Modelo", "Agente")]
            with tab_tabla:
                st.dataframe(comparacion.style.format({c: "{:.2f}" for c in columnas_num}, na_rep="-"),
                             width="stretch", hide_index=True)
            with tab_grafico:
                largo = comparacion.drop(columns="Agente").melt(id_vars="Modelo", var_name="Serie",
                                                                value_name="Precisión")
                largo["Serie"] = largo["Serie"].str.replace("Precisión ", "", regex=False)
                figura = px.bar(largo, x="Modelo", y="Precisión", color="Serie", barmode="group",
                                text=largo["Precisión"].map(lambda v: f"{v:.1f}%"),
                                color_discrete_map={"Global": AMARILLO})
                figura.update_yaxes(range=[0, 100], title="Precisión (%)")
                figura.for_each_trace(lambda t: t.update(visible="legendonly") if t.name != "Global" else None)
                grafico(figura, key="grafico_comparativo", height=450)
        if st.button("Quitar todos los modelos ejecutados", key="btn_limpiar_modelos"):
            cache.clear()
            st.rerun()

with tab_bench:
    st.caption("Entrena el modelo base de cada algoritmo y variaciones de hiperparámetros con la misma "
               "partición y semilla del experimento elegido, y los ordena por F1 macro.")
    b1, b2, b3 = st.columns(3)
    exp_bench = b1.selectbox("Experimento", list(range(1, n_exp + 1)), key="clf_exp_bench") - 1
    escalar_bench = b2.radio("Escalar datos", ["SI", "NO"], horizontal=True, key="clf_escalar_bench") == "SI"
    if b3.button("▶ Ejecutar benchmarking", type="primary", key="clf_btn_bench"):
        resultado = ejecutar(lambda: ctrl.run_benchmark("classification", {
            "target": config["objetivo"], "features": config["predictores"], "test_size": config["test_size"],
            "random_state": config["experimentos"][exp_bench]["semilla"], "stratify": config["estratificar"],
            "scale": escalar_bench}), "Entrenando todas las configuraciones...", "No se pudo ejecutar el benchmarking: ")
        if resultado is not None:
            st.session_state["clf_benchmark"] = resultado
    bench = st.session_state.get("clf_benchmark")
    if bench is None:
        st.info("Presione **Ejecutar benchmarking** para comparar los algoritmos.")
    else:
        datos_b = bench["result"]
        pie_de_ejecucion(bench)
        tabla = datos_b["table"]
        st.success(f"Mejor modelo: **{datos_b['best']}** · F1 macro {datos_b['primary_metric']['value']:.4f}")
        metrica = st.radio("Índice", ["f1_macro", "accuracy", "precision_macro", "recall_macro"], horizontal=True,
                           key="clf_metrica_bench")
        ordenada = tabla.dropna(subset=[metrica]).sort_values(metrica)
        figura = px.bar(ordenada, x=metrica, y="Modelo", color="Familia", orientation="h",
                        text=ordenada[metrica].map(lambda v: f"{v:.4f}"))
        grafico(figura, key="clf_grafico_bench", height=max(360, 40 * len(ordenada)))
        st.dataframe(tabla.style.format({c: "{:.4f}" for c in ("accuracy", "precision_macro", "recall_macro",
                                                                "f1_macro") if c in tabla}),
                     width="stretch", hide_index=True)
        elegido = st.selectbox("Matriz de confusión de", list(datos_b["confusion_matrices"]), key="clf_cm_bench")
        matriz = datos_b["confusion_matrices"][elegido]
        etiquetas_cm = [nombres_clase.get(str(e), str(e)) for e in matriz["labels"]]
        _matriz_confusion(np.asarray(matriz["matrix"]), etiquetas_cm)
        st.download_button("Descargar benchmarking (CSV)", data=tabla.to_csv(index=False).encode("utf-8"),
                           file_name="benchmark_clasificacion.csv", mime="text/csv")
