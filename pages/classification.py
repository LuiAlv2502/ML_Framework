import pandas as pd
import plotly.express as px
import streamlit as st

from pages._shared import obtener_modelo
from supervised import ClasificacionModelos

st.title("Supervisado - Clasificación")

model = obtener_modelo()
if model is None:
    st.stop()

df = model.dataframe.dataframe
columnas = df.columns.tolist()

col1, col2 = st.columns(2)
objetivo = col1.selectbox("Variable objetivo", columnas)
test_size = col2.slider("Proporción test", 0.1, 0.5, 0.25)

algoritmo = st.selectbox(
    "Algoritmo",
    [
        "KNN",
        "Árbol de decisión",
        "Random Forest",
        "Gradient Boosting",
        "AdaBoost",
        "XGBoost",
    ],
)

params = {}
if algoritmo == "KNN":
    params["n_neighbors"] = st.slider("Vecinos (k)", 1, 20, 5)
elif algoritmo == "Árbol de decisión":
    params["max_depth"] = st.slider("Profundidad máxima", 1, 20, 5)
elif algoritmo == "Random Forest":
    params["n_estimators"] = st.slider("Árboles", 10, 500, 200)
    params["max_depth"] = st.slider("Profundidad", 1, 20, 5)
elif algoritmo == "Gradient Boosting":
    params["n_estimators"] = st.slider("Estimadores", 10, 500, 120)
    params["max_depth"] = st.slider("Profundidad", 1, 10, 2)
elif algoritmo == "AdaBoost":
    params["n_estimators"] = st.slider("Estimadores", 10, 300, 80)
elif algoritmo == "XGBoost":
    params["n_estimators"] = st.slider("Estimadores", 10, 500, 120)
    params["max_depth"] = st.slider("Profundidad", 1, 15, 3)
    params["learning_rate"] = st.slider("Learning rate", 0.01, 1.0, 0.1)

if st.button("Entrenar", type="primary"):
    with st.spinner("Entrenando..."):
        try:
            clf = ClasificacionModelos(df, target=objetivo, test_size=test_size)
            metodo = {
                "KNN": clf.knn,
                "Árbol de decisión": clf.decision_tree,
                "Random Forest": clf.random_forest,
                "Gradient Boosting": clf.gradient_boosting,
                "AdaBoost": clf.adaboost,
                "XGBoost": clf.xgboost,
            }[algoritmo]
            resultado = metodo(**params)
            metricas = resultado["metricas"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metricas['accuracy']:.3f}")
            m2.metric("Precision", f"{metricas['precision_macro']:.3f}")
            m3.metric("Recall", f"{metricas['recall_macro']:.3f}")
            m4.metric("F1", f"{metricas['f1_macro']:.3f}")

            st.subheader("Matriz de confusión")
            etiquetas = [str(c) for c in sorted(df[objetivo].unique())]
            fig = px.imshow(
                metricas["confusion_matrix"],
                text_auto=True,
                x=etiquetas,
                y=etiquetas,
                labels=dict(x="Predicho", y="Real", color="Cuenta"),
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Reporte por clase")
            st.dataframe(
                pd.DataFrame(metricas["classification_report"]).T,
                use_container_width=True,
            )
        except Exception as error:
            st.error(f"Error: {error}")
