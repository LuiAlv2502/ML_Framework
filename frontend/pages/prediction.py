import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.components.shared import ejecutar, obtener_datos, pie_de_ejecucion
from frontend.components.supervised_agent import seccion_agente
from frontend.components.theme import AZUL_SUAVE, CIAN, ESCALA_CONTINUA, MAGENTA, ROJO, VERDE, brillo, grafico

ALGORITMOS = {
    "Regresión lineal simple": "regresion_lineal_simple",
    "Regresión lineal múltiple": "regresion_lineal_multiple",
    "Lasso": "lasso",
    "LassoCV (alpha óptimo)": "lasso_cv",
    "Ridge": "ridge",
    "RidgeCV (alpha óptimo)": "ridge_cv",
    "Máquina de soporte vectorial (SVR)": "svr",
    "Árbol de decisión": "decision_tree",
    "Bosques aleatorios": "random_forest",
}
ABREVIATURAS = {
    "Regresión lineal simple": "rls",
    "Regresión lineal múltiple": "rlm",
    "Lasso": "lasso",
    "LassoCV (alpha óptimo)": "lassocv",
    "Ridge": "ridge",
    "RidgeCV (alpha óptimo)": "ridgecv",
    "Máquina de soporte vectorial (SVR)": "svr",
    "Árbol de decisión": "arbol",
    "Bosques aleatorios": "rf",
}
METRICAS = {
    "RMSE": "Raíz del error cuadrático medio (menor es mejor)",
    "MAE": "Error absoluto medio (menor es mejor)",
    "ER": "Error relativo (menor es mejor)",
    "R2": "Coeficiente de determinación (mayor es mejor)",
}
COLOR_APRENDIZAJE = "rgba(0, 240, 255, 0.25)"
COLOR_PRUEBA = "rgba(255, 42, 109, 0.30)"


def _colorear_particion(valor):
    if valor == "Aprendizaje":
        return f"background-color: {COLOR_APRENDIZAJE}; color: {CIAN}"
    if valor == "Prueba":
        return f"background-color: {COLOR_PRUEBA}; color: {MAGENTA}"
    return ""


def _parametros_modelo(algoritmo, predictores_numericos, profundidad_opts):
    p = {}
    k = f"reg_param_{algoritmo}"
    if algoritmo == "Regresión lineal simple":
        p["columna"] = st.selectbox("Variable predictora (x)", predictores_numericos, key=f"{k}_col")
    elif algoritmo in ("Lasso", "Ridge"):
        p["alpha"] = st.number_input("Alpha (penalización)", min_value=0.0001, max_value=10_000.0,
                                     value=0.1 if algoritmo == "Lasso" else 1.0, step=0.1, format="%.4f",
                                     key=f"{k}_alpha")
    elif algoritmo == "LassoCV (alpha óptimo)":
        p["cv"] = st.slider("Pliegues de validación cruzada", 2, 10, 5, key=f"{k}_cv")
    elif algoritmo == "RidgeCV (alpha óptimo)":
        p["alphas"] = st.multiselect(
            "Valores de penalización a comparar",
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            default=[0.01, 0.1, 1.0, 10.0, 100.0],
            key=f"{k}_alphas",
            help="RidgeCV elegirá el valor con menor error en la validación cruzada.",
        )
    elif algoritmo == "Máquina de soporte vectorial (SVR)":
        p["kernel"] = st.selectbox("Núcleo (kernel)", ["rbf", "linear", "poly", "sigmoid"], key=f"{k}_kernel")
        p["C"] = st.number_input("C", min_value=0.01, max_value=10_000.0, value=100.0, step=10.0, key=f"{k}_c")
        p["epsilon"] = st.number_input("Epsilon", min_value=0.0, max_value=100.0, value=0.1, step=0.05, key=f"{k}_eps")
        if p["kernel"] == "poly":
            p["degree"] = st.slider("Grado del polinomio", 2, 6, 3, key=f"{k}_deg")
    elif algoritmo == "Árbol de decisión":
        p["max_depth"] = st.selectbox("Profundidad máxima", profundidad_opts, index=3,
                                      format_func=lambda v: "Sin límite" if v is None else str(v), key=f"{k}_depth")
        p["min_samples_split"] = st.slider("Mínimo de muestras para dividir", 2, 20, 2, key=f"{k}_split")
    elif algoritmo == "Bosques aleatorios":
        p["n_estimators"] = st.slider("Cantidad de árboles", 10, 500, 100, 10, key=f"{k}_n")
        p["max_depth"] = st.selectbox("Profundidad máxima", profundidad_opts,
                                      format_func=lambda v: "Sin límite" if v is None else str(v), key=f"{k}_depth")
        p["min_samples_split"] = st.slider("Mínimo de muestras para dividir", 2, 20, 2, key=f"{k}_split")
    return p


def _etiqueta_modelo(algoritmo, parametros, escalar):
    partes = [ABREVIATURAS[algoritmo]] + [str(v).replace(" ", "") for v in parametros.values() if v is not None]
    if not escalar:
        partes.append("sin_escala")
    return "-".join(partes)


def _validar_objetivo(df, objetivo):
    y = df[objetivo]
    if not pd.api.types.is_numeric_dtype(y):
        return f"'{objetivo}' no es numérica; la regresión requiere una variable continua."
    if y.isna().any():
        return "La variable objetivo contiene valores vacíos. Corríjalos en la página EDA."
    return None


def _mostrar_metricas(metricas):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE", f"{metricas['RMSE']:.4f}", help=METRICAS["RMSE"])
    m2.metric("MAE", f"{metricas['MAE']:.4f}", help=METRICAS["MAE"])
    m3.metric("ER", f"{metricas['ER']:.2%}", help=METRICAS["ER"])
    m4.metric("R²", f"{metricas['R2']:.4f}", help=METRICAS["R2"])


def _mostrar_generacion(label, algoritmo, parametros, escalar, resultado, exp_idx, semilla, objetivo, agente):
    _mostrar_metricas(resultado["metrics"])
    lineas = [f"Modelo: {label}", f"Algoritmo: {algoritmo}", f"Agente: {agente}",
              f"Experimento: {exp_idx + 1} (semilla {semilla})", f"Datos escalados: {'Sí' if escalar else 'No'}",
              "", "Parámetros:"]
    lineas += [f"  {k}: {v}" for k, v in parametros.items()] or ["  (por defecto)"]
    if "alpha_optimo" in resultado:
        lineas += ["", f"Alpha óptimo: {resultado['alpha_optimo']:.6g}"]
    if "w0" in resultado:
        x = parametros["columna"] + (" (estandarizada)" if escalar else "")
        lineas += ["", f"Ecuación: {objetivo} = {resultado['w0']:.4f} + {resultado['w1']:.4f} · {x}"]
    st.code("\n".join(lineas), language=None)

    coeficientes = resultado.get("coefficients")
    if coeficientes is not None:
        coeficientes = coeficientes.copy()
        coeficientes["Signo"] = np.where(coeficientes["coef"] >= 0, "Positivo", "Negativo")
        en_cero = int((coeficientes["coef"].abs() < 1e-10).sum())
        if en_cero:
            st.caption(f"{en_cero} coeficiente(s) quedaron en cero (variables descartadas).")
        fig = px.bar(coeficientes.iloc[::-1], x="coef", y="predictor", orientation="h", color="Signo",
                     color_discrete_map={"Positivo": CIAN, "Negativo": MAGENTA}, title="Coeficientes del modelo")
        fig.add_vline(x=0, line=dict(color=AZUL_SUAVE, width=1))
        grafico(fig, key=f"coef_{label}_{exp_idx}", height=max(320, 26 * len(coeficientes)))


def _grafico_simple(df, objetivo, columna, resultado, label, exp_idx):
    fig = px.scatter(df, x=columna, y=objetivo, opacity=0.7, title="Recta de regresión")
    fig.update_traces(marker=dict(color=CIAN, size=8))
    fig.add_scatter(x=resultado["line"]["x"], y=resultado["line"]["y"], mode="lines",
                    line=dict(color=AZUL_SUAVE, width=3), name="Ajuste")
    grafico(brillo(fig), key=f"simple_{label}_{exp_idx}", height=450)


def _mostrar_indices(label, resultado, exp_idx, objetivo):
    _mostrar_metricas(resultado["metrics"])
    tabla = resultado["predictions"]
    g1, g2 = st.columns(2)
    with g1:
        minimo = float(min(tabla["Real"].min(), tabla["Predicción"].min()))
        maximo = float(max(tabla["Real"].max(), tabla["Predicción"].max()))
        fig = px.scatter(tabla, x="Real", y="Predicción", color="Error absoluto",
                         color_continuous_scale=ESCALA_CONTINUA, hover_data=["ID", "Residuo"],
                         title="Real vs. predicción")
        fig.update_traces(marker=dict(size=9))
        fig.add_scatter(x=[minimo, maximo], y=[minimo, maximo], mode="lines",
                        line=dict(color=CIAN, dash="dash", width=2), name="Predicción perfecta")
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        grafico(fig, key=f"real_pred_{label}_{exp_idx}", height=460)
    with g2:
        fig_res = px.scatter(tabla, x="Predicción", y="Residuo",
                             color=np.where(tabla["Residuo"] >= 0, "Subestima", "Sobreestima"),
                             color_discrete_map={"Subestima": VERDE, "Sobreestima": ROJO},
                             hover_data=["ID", "Real"], title="Residuos")
        fig_res.update_traces(marker=dict(size=9))
        fig_res.add_hline(y=0, line=dict(color=AZUL_SUAVE, width=2))
        fig_res.update_layout(legend_title_text="", legend=dict(orientation="h", y=-0.2))
        grafico(fig_res, key=f"residuos_{label}_{exp_idx}", height=460)

    fig_lineas = go.Figure()
    fig_lineas.add_scatter(x=tabla["ID"], y=tabla["Real"], mode="lines+markers", name="Real",
                           line=dict(color=CIAN, width=2), marker=dict(size=5))
    fig_lineas.add_scatter(x=tabla["ID"], y=tabla["Predicción"], mode="lines+markers", name="Predicción",
                           line=dict(color=MAGENTA, width=2, dash="dot"), marker=dict(size=5))
    fig_lineas.update_layout(title=f"{objetivo}: real vs. predicción por observación", xaxis_title="ID",
                             yaxis_title=objetivo)
    grafico(brillo(fig_lineas), key=f"lineas_{label}_{exp_idx}", height=420)

    fig_hist = px.histogram(tabla, x="Residuo", nbins=30, marginal="box", title="Distribución de los residuos")
    fig_hist.update_traces(marker_color=MAGENTA, selector=dict(type="histogram"))
    fig_hist.update_traces(marker_color=AZUL_SUAVE, selector=dict(type="box"))
    t1, t2 = st.columns([1, 1])
    with t1:
        grafico(fig_hist, key=f"hist_res_{label}_{exp_idx}", height=400)
    with t2:
        st.dataframe(tabla.style.format({"Real": "{:.4f}", "Predicción": "{:.4f}", "Residuo": "{:.4f}",
                                         "Error absoluto": "{:.4f}"})
                     .background_gradient(subset=["Error absoluto"], cmap="magma"),
                     width="stretch", hide_index=True, height=340)
        st.download_button("Descargar predicciones CSV", data=tabla.to_csv(index=False).encode("utf-8"),
                           file_name=f"predicciones_{label}_exp{exp_idx + 1}.csv", mime="text/csv",
                           key=f"reg_descargar_{label}_{exp_idx}")


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
st.title("Supervisado - Predicción")

ctrl = obtener_datos(recomendar_eda=True)
if ctrl is None:
    st.stop()

df = ctrl.active_data()
columnas = df.columns.tolist()
numericas = df.select_dtypes(include=np.number).columns.tolist()
if not numericas or len(columnas) < 2:
    st.warning("El archivo debe tener una variable objetivo numérica y al menos un predictor.")
    st.stop()

seccion_agente("regression", df)

# ---------------------------------------------------------------------------
# PASO 1: Generar datos
# ---------------------------------------------------------------------------
st.header("1. Generar datos")
with st.container(border=True):
    cabecera, boton_col = st.columns([4, 1])
    cabecera.markdown("**Opciones de partición**")
    objetivo = st.selectbox("Seleccione la variable a predecir (numérica)", numericas, index=len(numericas) - 1,
                            key="reg_objetivo")
    faltantes = int(df[objetivo].isna().sum())
    if faltantes:
        st.warning(f"Se excluirán {faltantes} filas sin valor en '{objetivo}'; "
                   "no se imputan los resultados que el modelo debe predecir.")
    if df[objetivo].nunique() <= 10:
        st.warning(f"'{objetivo}' tiene pocos valores distintos; podría ser más adecuada para la página de "
                   "clasificación.")
    predictores_disponibles = [c for c in columnas if c != objetivo]
    with st.expander("Variables predictoras", expanded=False):
        predictores = st.multiselect("Variables predictoras", predictores_disponibles, default=predictores_disponibles,
                                     label_visibility="collapsed", key=f"reg_predictores_{objetivo}")
    s1, s2 = st.columns([2, 1])
    modo_semilla = s1.radio("Semilla aleatoria", ["Habilitada", "Deshabilitada"], horizontal=True,
                            key="reg_modo_semilla")
    semilla = s2.number_input("Semilla", min_value=0, max_value=2_147_483_647, value=5, step=1,
                              disabled=(modo_semilla == "Deshabilitada"), key="reg_semilla")
    porc_aprendizaje = st.slider("Aprendizaje (%)  ←→  Prueba (%)", min_value=5, max_value=95, value=75, step=5,
                                 key="reg_porcentaje")
    st.caption(f"Aprendizaje: {porc_aprendizaje}%  ·  Prueba: {100 - porc_aprendizaje}%")
    n_experimentos = st.number_input("Cantidad de experimentos", min_value=1, max_value=50, value=1, step=1,
                                     key="reg_n_exp")
    ejecutar_particion = boton_col.button("▶ Ejecutar", type="primary", key="reg_btn_particion")

info = ctrl.dataset_info()
test_size = 1 - porc_aprendizaje / 100
configuracion_actual = {
    "objetivo": objetivo,
    "predictores": predictores,
    "test_size": test_size,
    "n_experimentos": int(n_experimentos),
    "modo_semilla": modo_semilla,
    "semilla_base": int(semilla) if modo_semilla == "Habilitada" else None,
    "dataset_hash": info["hash"] + info["source"],
}

if ejecutar_particion:
    for clave in ("reg_resultados", "reg_particiones", "reg_benchmark"):
        st.session_state.pop(clave, None)
    if not predictores:
        st.error("Seleccione al menos una variable predictora.")
    elif (problema := _validar_objetivo(df, objetivo)):
        st.error(f"No se pudo generar la partición: {problema}")
    else:
        if modo_semilla == "Habilitada":
            semillas = [(int(semilla) + i) % (2**32 - 1) for i in range(int(n_experimentos))]
        else:
            semillas = np.random.default_rng().integers(0, 10**6, int(n_experimentos)).tolist()
        particiones = ejecutar(lambda: ctrl.partitions(task="regression", target=objetivo, features=predictores,
                                                       test_size=test_size, seeds=semillas),
                               "Generando particiones...", "No se pudo generar la partición: ")
        if particiones is not None:
            experimentos = []
            for p in particiones:
                particion = pd.Series("Aprendizaje", index=df.index, name="Partición")
                particion.loc[p["test_index"]] = "Prueba"
                experimentos.append({"semilla": p["seed"], "particion": particion, "n_train": p["n_train"],
                                     "n_test": p["n_test"]})
            st.session_state["reg_particiones"] = {**configuracion_actual, "experimentos": experimentos}

config = st.session_state.get("reg_particiones")
if config is None:
    st.info("Configure la partición y presione **Ejecutar** para continuar.")
    st.stop()
if any(config.get(campo) != valor for campo, valor in configuracion_actual.items()):
    st.warning("Cambió la configuración de los datos. Presione **Ejecutar** en el paso 1 "
               "para generar una nueva partición antes de entrenar o ver resultados.")
    st.stop()

# ---------------------------------------------------------------------------
# PASO 2: Datos
# ---------------------------------------------------------------------------
st.header("2. Datos")
n_exp = len(config["experimentos"])
exp_datos = st.selectbox("Experimento", list(range(1, n_exp + 1)), key="reg_exp_seleccionado") - 1
exp = config["experimentos"][exp_datos]
d1, d2, d3 = st.columns(3)
d1.metric("Filas de aprendizaje", exp["n_train"])
d2.metric("Filas de prueba", exp["n_test"])
d3.metric("Semilla usada", exp["semilla"])
tabla_datos = df.copy()
tabla_datos.insert(0, "Partición", exp["particion"])
st.dataframe(tabla_datos.style.map(_colorear_particion, subset=["Partición"]), width="stretch", height=300)
st.download_button("Descargar datos con partición (CSV)", data=tabla_datos.to_csv(index=False).encode("utf-8"),
                   file_name=f"datos_particion_regresion_exp{exp_datos + 1}.csv", mime="text/csv")

st.subheader(f"Resumen de '{config['objetivo']}'")
st.caption("Sirve como referencia para interpretar si un RMSE o MAE es grande o pequeño.")
resumen = ctrl.target_summary(config["objetivo"])
q1, q2, q3 = resumen["Cuartiles"]
r1, r2, r3, r4, r5, r6 = st.columns(6)
r1.metric("Mínimo", f"{resumen['Mínimo']:.3f}")
r2.metric("Q1", f"{q1:.3f}")
r3.metric("Mediana", f"{q2:.3f}")
r4.metric("Q3", f"{q3:.3f}")
r5.metric("Máximo", f"{resumen['Máximo']:.3f}")
r6.metric("Media", f"{resumen['Media']:.3f}")
fig_obj = px.histogram(tabla_datos, x=config["objetivo"], color="Partición", nbins=40, marginal="box",
                       barmode="overlay", opacity=0.75,
                       color_discrete_map={"Aprendizaje": CIAN, "Prueba": MAGENTA})
grafico(fig_obj, key="reg_hist_objetivo", height=380)

# ---------------------------------------------------------------------------
# PASO 3: Modelos
# ---------------------------------------------------------------------------
st.header("3. Modelos")
cache = st.session_state.setdefault("reg_resultados", {})
profundidad_opts = [None, *range(1, 31)]
predictores_numericos = [c for c in config["predictores"] if pd.api.types.is_numeric_dtype(df[c])]
opciones_algoritmo = [a for a in ALGORITMOS if a != "Regresión lineal simple" or predictores_numericos]
semillas_config = [e["semilla"] for e in config["experimentos"]]

fila_a, fila_b = st.columns([3, 1])
algoritmo = fila_a.selectbox("Algoritmo", opciones_algoritmo, key="reg_algoritmo")
fila_b.caption(f"Experimentos generados: {n_exp}")

tab_gen, tab_ind, tab_var, tab_comp, tab_bench = st.tabs(
    ["Generación del Modelo", "Índices Generales", "Variación del Error", "Comparación de Modelos",
     "Benchmarking por familias"]
)

with tab_gen:
    exp_gen_idx = st.selectbox("Experimento", list(range(1, n_exp + 1)), key="reg_exp_generacion") - 1
    col_opts, col_out = st.columns([1, 2])
    with col_opts.container(border=True):
        cab, btn = st.columns([2, 1])
        cab.markdown("**Opciones**")
        ejecutar_modelo = btn.button("▶ Ejecutar", type="primary", key="reg_btn_modelo")
        parametros = _parametros_modelo(algoritmo, predictores_numericos, profundidad_opts)
        escalar = st.radio("Escalar datos", ["SI", "NO"], horizontal=True, key="reg_escalar") == "SI"
    etiqueta = _etiqueta_modelo(algoritmo, parametros, escalar)

    if ejecutar_modelo:
        resultado = ejecutar(
            lambda: ctrl.run_regression(target=config["objetivo"], algorithm=ALGORITMOS[algoritmo],
                                        params=parametros, features=config["predictores"],
                                        test_size=config["test_size"], seeds=semillas_config, scale=escalar,
                                        label=etiqueta),
            f"Entrenando {etiqueta} en {n_exp} experimento(s)...", "No se pudo generar el modelo: ")
        if resultado is not None:
            cache[etiqueta] = {"algoritmo": algoritmo, "parametros": parametros, "agente": resultado["selected_agent"],
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
                                config["experimentos"][exp_gen_idx]["semilla"], config["objetivo"], entrada["agente"])
    if resultado_actual is not None and "line" in resultado_actual:
        _grafico_simple(df, config["objetivo"], parametros["columna"], resultado_actual, etiqueta, exp_gen_idx)

with tab_ind:
    control_exp, _ = st.columns([1, 4])
    exp_indices_idx = control_exp.selectbox("Experimento", list(range(1, n_exp + 1)), key="reg_exp_indices") - 1
    resultado_indices = cache.get(etiqueta, {}).get("experimentos", {}).get(exp_indices_idx)
    if resultado_indices is None:
        st.info("Genere el modelo en la pestaña **Generación del Modelo** para ver sus índices.")
    else:
        _mostrar_indices(etiqueta, resultado_indices, exp_indices_idx, config["objetivo"])

with tab_var:
    indice = st.selectbox("Seleccione el índice", list(METRICAS), format_func=lambda m: f"{m} · {METRICAS[m]}",
                          key="reg_var_indice")
    filas = [{"Modelo": nombre, "Número de experimento": i + 1, indice: r["metrics"][indice]}
             for nombre, entrada in cache.items() for i, r in entrada["experimentos"].items()]
    if not filas:
        st.info("Ejecute al menos un modelo para ver cómo varía entre experimentos.")
    else:
        datos_var = pd.DataFrame(filas)
        figura = px.line(datos_var, x="Número de experimento", y=indice, color="Modelo", markers=True)
        figura.update_traces(line=dict(width=3), marker=dict(size=10, symbol="diamond"))
        figura.update_xaxes(dtick=1)
        grafico(brillo(figura), key="reg_grafico_variacion", height=450)
        resumen_var = datos_var.groupby("Modelo")[indice].agg(["mean", "std", "min", "max"])
        resumen_var.columns = ["Media", "Desv. estándar", "Mínimo", "Máximo"]
        st.dataframe(resumen_var.style.format("{:.4f}", na_rep="-"), width="stretch")

with tab_comp:
    if not cache:
        st.info("Todavía no hay modelos para comparar. Ejecute los que quiera en la pestaña "
                "**Generación del Modelo** y aparecerán aquí.")
    else:
        cc1, cc2 = st.columns([3, 1])
        exp_comp = cc2.selectbox("Experimento", list(range(1, n_exp + 1)), key="reg_exp_comparacion") - 1
        disponibles = list(cache)
        elegidos = cc1.multiselect("Modelos a comparar", disponibles, default=disponibles)
        filas = [{"Modelo": n, "Agente": cache[n]["agente"], **cache[n]["experimentos"][exp_comp]["metrics"]}
                 for n in elegidos if exp_comp in cache[n]["experimentos"]]
        tab_tabla, tab_grafico = st.tabs(["Tabla Comparativa", "Gráfico Comparativo"])
        if not filas:
            tab_tabla.info("Seleccione al menos un modelo.")
        else:
            comparacion = pd.DataFrame(filas).sort_values("RMSE", ignore_index=True)
            mejor = f"color: #000; background-color: {VERDE}; font-weight: 700"
            with tab_tabla:
                st.caption("En verde, el mejor valor de cada índice.")
                st.dataframe(comparacion.style.format({m: "{:.4f}" for m in METRICAS})
                             .highlight_min(subset=["RMSE", "MAE", "ER"], props=mejor)
                             .highlight_max(subset=["R2"], props=mejor), width="stretch", hide_index=True)
            with tab_grafico:
                metrica_graf = st.radio("Índice", list(METRICAS), horizontal=True, key="reg_metrica_comp")
                ascendente = metrica_graf != "R2"
                ordenada = comparacion.sort_values(metrica_graf, ascending=ascendente)
                figura = px.bar(ordenada, x="Modelo", y=metrica_graf, color=metrica_graf,
                                color_continuous_scale=([[0, VERDE], [0.5, AZUL_SUAVE], [1, ROJO]] if ascendente
                                                        else [[0, ROJO], [0.5, AZUL_SUAVE], [1, VERDE]]),
                                text=ordenada[metrica_graf].map(lambda v: f"{v:.4f}"))
                figura.update_layout(coloraxis_showscale=False)
                grafico(figura, key="reg_grafico_comparativo", height=450)
        if st.button("Quitar todos los modelos ejecutados", key="reg_btn_limpiar"):
            cache.clear()
            st.rerun()

with tab_bench:
    st.caption("Entrena el modelo base de cada algoritmo (lineales, SVR y árboles) y variaciones de "
               "hiperparámetros sobre el experimento elegido, y compara las familias.")
    b1, b2, b3 = st.columns(3)
    exp_bench = b1.selectbox("Experimento", list(range(1, n_exp + 1)), key="reg_exp_bench") - 1
    escalar_bench = b2.radio("Escalar datos", ["SI", "NO"], horizontal=True, key="reg_escalar_bench") == "SI"
    clave_bench = (exp_bench, escalar_bench)
    if b3.button("▶ Ejecutar benchmarking", type="primary", key="reg_btn_bench"):
        resultado = ejecutar(lambda: ctrl.run_benchmark("regression", {
            "target": config["objetivo"], "features": config["predictores"], "test_size": config["test_size"],
            "random_state": config["experimentos"][exp_bench]["semilla"], "scale": escalar_bench}),
            "Entrenando todas las familias...", "No se pudo ejecutar el benchmarking: ")
        if resultado is not None:
            st.session_state["reg_benchmark"] = {"clave": clave_bench, "ejecucion": resultado}

    bench = st.session_state.get("reg_benchmark")
    if bench is None or bench["clave"] != clave_bench:
        st.info("Presione **Ejecutar benchmarking** para comparar las familias de algoritmos.")
    else:
        datos_b = bench["ejecucion"]["result"]
        pie_de_ejecucion(bench["ejecucion"])
        tabla_familias, mejores = datos_b["table"], datos_b["best_by_family"]
        ganador = mejores.iloc[0]
        st.success(f"Mejor modelo: **{ganador['Modelo']}** ({ganador['Familia']}) · "
                   f"RMSE {ganador['RMSE']:.4f} · R² {ganador['R2']:.4f}")
        colores_familia = dict(zip(tabla_familias["Familia"].unique(), [CIAN, MAGENTA, AZUL_SUAVE]))
        metrica_bench = st.radio("Índice", list(METRICAS), horizontal=True, key="reg_metrica_bench")
        ordenada = tabla_familias.sort_values(metrica_bench, ascending=metrica_bench != "R2")
        figura = px.bar(ordenada, x=metrica_bench, y="Modelo", color="Familia", orientation="h",
                        color_discrete_map=colores_familia, text=ordenada[metrica_bench].map(lambda v: f"{v:.4f}"),
                        category_orders={"Modelo": ordenada["Modelo"].tolist()})
        grafico(figura, key="reg_grafico_bench", height=max(360, 45 * len(ordenada)))
        t1, t2 = st.columns([3, 2])
        with t1:
            st.markdown("**Todos los algoritmos**")
            st.dataframe(tabla_familias.style.format({m: "{:.4f}" for m in METRICAS}), width="stretch",
                         hide_index=True)
        with t2:
            st.markdown("**Mejor de cada familia**")
            st.dataframe(mejores.style.format({m: "{:.4f}" for m in METRICAS}), width="stretch", hide_index=True)
