"""Panel de agentes, mensajes, decisiones y creación."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from frontend.components.shared import ejecutar, obtener_controlador

st.title("MAS")
st.write("Consulta los agentes, sus comunicaciones y las decisiones de asignación de tareas.")
controlador = obtener_controlador()
tab_agentes, tab_mensajes, tab_decisiones, tab_crear = st.tabs(
    ["Agentes", "Mensajes", "Decisiones", "Crear agente"]
)


def _etiqueta_capacidad(capacidad: str) -> str:
    etiquetas = {
        "eda.profile": "Explorar y describir datos",
        "eda.summary": "Resumir datos",
        "eda.clean": "Limpiar datos",
        "eda.preprocess": "Preparar datos",
        "eda.correlations": "Analizar relaciones entre variables",
        "clustering.auto": "Analizar grupos automáticamente",
        "clustering.benchmark": "Comparar métodos de agrupamiento",
        "clustering.dendrogram": "Visualizar grupos jerárquicos",
        "clustering.elbow": "Estimar cantidad de grupos",
        "clustering.hac": "Formar grupos jerárquicos",
        "clustering.kmeans": "Formar grupos",
        "clustering.pca": "Resumir variables",
        "clustering.tsne": "Visualizar datos",
        "clustering.umap": "Visualizar datos",
        "classification.auto": "Clasificar automáticamente",
        "classification.benchmark": "Comparar clasificadores",
        "classification.train": "Entrenar un clasificador",
        "regression.auto": "Predecir valores automáticamente",
        "regression.benchmark": "Comparar métodos de predicción",
        "regression.train": "Entrenar un modelo de predicción",
        "coordination.route": "Coordinar solicitudes",
    }
    return etiquetas.get(capacidad, capacidad)


def _tarea_demo(capacidad: str, datos: pd.DataFrame) -> dict[str, Any]:
    if capacidad == "classification.train":
        objetivo = controlador.suggest_target()
        if objetivo not in datos.columns or pd.api.types.is_numeric_dtype(datos[objetivo]):
            candidatas = [c for c in datos if not pd.api.types.is_numeric_dtype(datos[c])]
            if not candidatas:
                raise ValueError("No hay una columna categórica que se pueda clasificar.")
            objetivo = candidatas[0]
        return {"target": objetivo, "algorithm": "knn", "test_size": 0.25, "stratify": True}
    if capacidad == "regression.train":
        candidatas = [c for c in datos if pd.api.types.is_numeric_dtype(datos[c])]
        if not candidatas:
            raise ValueError("No hay una columna numérica para probar la predicción.")
        return {"target": candidatas[-1], "algorithm": "ridge", "test_size": 0.25}
    if capacidad == "clustering.kmeans":
        return {"params": {"n_clusters": 3, "init": "k-means++", "n_init": 10}, "projection_dims": 2}
    if capacidad == "clustering.hac":
        return {"params": {"n_clusters": 3, "linkage": "ward"}, "projection_dims": 2}
    if capacidad == "clustering.pca":
        return {"params": {"n_components": min(2, len(datos.columns)), "whiten": False, "svd_solver": "auto"}}
    if capacidad.startswith("eda."):
        return {"use_llm": False, "target": None}
    raise ValueError("La prueba guiada está disponible para clasificación, regresión, PCA y agrupamiento.")


with tab_agentes:
    agentes = controlador.list_agents()
    if agentes:
        st.dataframe(
            [
                {
                    "ID": agente["agent_id"],
                    "Tipo": agente["agent_type"],
                    "Capacidades": ", ".join(agente["capabilities"]),
                    "Estado": agente["status"],
                    "Prioridad": agente["priority"],
                    "Tareas atendidas": agente["tasks_received"],
                    "Éxitos": agente["tasks_succeeded"],
                    "Fallos": agente["tasks_failed"],
                }
                for agente in agentes
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Aún no hay agentes registrados.")

    administrables = [agente for agente in agentes if agente["agent_type"] != "coordinator"]
    if administrables:
        st.subheader("Administrar un agente")
        por_etiqueta = {f"{a['name']} ({a['agent_id']})": a for a in administrables}
        elegido = st.selectbox("Agente", list(por_etiqueta))
        agente = por_etiqueta[elegido]
        accion = st.selectbox(
            "Acción",
            ["start", "pause", "resume", "restart", "stop"],
            format_func=lambda valor: {
                "start": "Activar",
                "pause": "Pausar",
                "resume": "Reanudar",
                "restart": "Reiniciar",
                "stop": "Detener",
            }[valor],
        )
        if st.button("Aplicar acción"):
            resultado = ejecutar(
                lambda: controlador.agent_lifecycle(agente["agent_id"], accion),
                "Actualizando agente...",
            )
            if resultado:
                st.success("Estado actualizado.")
                st.rerun()
        confirmar = st.checkbox("Confirmo que quiero eliminar este agente")
        if st.button("Eliminar agente", disabled=not confirmar):
            def _eliminar() -> bool:
                controlador.remove_agent(agente["agent_id"])
                return True

            if ejecutar(_eliminar, "Eliminando agente..."):
                st.success("Agente eliminado.")
                st.rerun()

with tab_mensajes:
    st.subheader("Mensajes entre agentes")
    todos = controlador.list_messages()
    emisores = ["Todos", *sorted({mensaje["sender"] for mensaje in todos})]
    receptores = ["Todos", *sorted({mensaje["receiver"] for mensaje in todos})]
    performativas = ["Todas", "request", "inform", "failure", "query", "confirm"]
    tareas = ["Todas", *sorted({mensaje["task"] for mensaje in todos})]
    with st.form("filtros_mensajes"):
        columnas = st.columns(4)
        emisor = columnas[0].selectbox("Emisor", emisores)
        receptor = columnas[1].selectbox("Receptor", receptores)
        performativa = columnas[2].selectbox("Tipo de mensaje", performativas)
        tarea = columnas[3].selectbox("Tarea", tareas)
        correlacion = st.text_input("Identificador de conversación")
        actualizar = st.form_submit_button("Actualizar mensajes")
    filtros = {
        "sender": None if emisor == "Todos" else emisor,
        "receiver": None if receptor == "Todos" else receptor,
        "performative": None if performativa == "Todas" else performativa,
        "task": None if tarea == "Todas" else tarea,
        "correlation_id": correlacion.strip() or None,
    }
    mensajes = controlador.list_messages(**filtros)
    if mensajes:
        mensajes.sort(key=lambda mensaje: mensaje["timestamp"], reverse=True)
        st.dataframe(
            [
                {
                    "Hora": mensaje["timestamp"],
                    "Emisor": mensaje["sender"],
                    "Receptor": mensaje["receiver"],
                    "Tipo": mensaje["performative"],
                    "Tarea": mensaje["task"],
                    "Conversación": mensaje["correlation_id"],
                    "Estado": mensaje["status"],
                    "Contenido resumido": json.dumps(mensaje["payload"], ensure_ascii=False, default=str),
                }
                for mensaje in mensajes[:500]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No hay mensajes que coincidan con esos filtros.")
    if actualizar:
        st.rerun()

with tab_decisiones:
    st.subheader("Decisiones del coordinador")
    historial = controlador.list_experiments()
    if historial.empty:
        st.info("Las decisiones aparecerán aquí después de ejecutar análisis.")
    else:
        for experimento in historial.head(10).to_dict("records"):
            detalle = ejecutar(
                lambda run_id=experimento["id"]: controlador.get_experiment(str(run_id)),
                "Consultando decisiones...",
            )
            if not detalle:
                continue
            run = detalle["run"]
            coordinacion = run.get("coordination") or {}
            with st.expander(f"{run.get('task')} · {run.get('created_at')} · {run.get('id')}"):
                st.write(f"**Política:** {coordinacion.get('policy', 'No registrada')}")
                st.write(f"**Candidatos:** {', '.join(coordinacion.get('candidates') or []) or 'Ninguno'}")
                st.write(f"**Agente elegido:** {coordinacion.get('selected_agent', 'No registrado')}")
                decisiones = run.get("decisions") or []
                if decisiones:
                    st.dataframe(
                        [{"Decisión": decision.get("decision"), "Justificación": decision.get("justification")}
                         for decision in decisiones],
                        width="stretch",
                        hide_index=True,
                    )

with tab_crear:
    tipos = [item for item in controlador.agent_types() if item.get("creatable")]
    if not tipos:
        st.info("No hay tipos de agente disponibles para crear.")
    else:
        mapa_tipos = {item["agent_type"]: item for item in tipos}
        tipo_nombre = st.selectbox(
            "Tipo de agente",
            list(mapa_tipos),
            format_func=lambda tipo: f"{mapa_tipos[tipo]['class']} — {mapa_tipos[tipo]['description']}",
        )
        tipo = mapa_tipos[tipo_nombre]
        capacidades_disponibles = tipo.get("supported_capabilities") or tipo["default_capabilities"]
        with st.form("crear_agente"):
            nombre = st.text_input("Nombre")
            capacidades = st.multiselect(
                "Tareas que puede realizar",
                capacidades_disponibles,
                default=tipo["default_capabilities"],
                format_func=_etiqueta_capacidad,
            )
            prioridad = st.slider("Prioridad", 0, 100, 10)
            configuracion: dict[str, Any] = {}
            esquema = tipo.get("config_schema", {})
            personalizar = st.checkbox("Configurar opciones avanzadas") if esquema else False
            if personalizar:
                for clave, especificacion in esquema.items():
                    clase, *rango = especificacion
                    if clase == "booleano":
                        configuracion[clave] = st.checkbox(clave.replace("_", " ").capitalize())
                    elif clase == "opcion":
                        configuracion[clave] = st.selectbox(clave.replace("_", " ").capitalize(), rango[0])
                    elif clase in ("entero", "real"):
                        configuracion[clave] = st.number_input(
                            clave.replace("_", " ").capitalize(),
                            min_value=rango[0],
                            max_value=rango[1],
                            value=rango[0],
                            step=1 if clase == "entero" else 0.1,
                        )
            crear = st.form_submit_button("Crear agente")
        if crear:
            if not nombre.strip():
                st.error("Escribe un nombre para el agente.")
            elif not capacidades:
                st.error("Selecciona al menos una tarea.")
            else:
                agente_creado = ejecutar(
                    lambda: controlador.create_agent(
                        {
                            "agent_type": tipo_nombre,
                            "name": nombre.strip(),
                            "capabilities": capacidades,
                            "priority": prioridad,
                            "config": configuracion,
                        }
                    ),
                    "Creando agente...",
                )
                if agente_creado:
                    st.session_state["mas_agente_para_probar"] = agente_creado["agent_id"]
                    st.success(f"Agente «{agente_creado['name']}» creado.")
                    st.rerun()

    agente_id = st.session_state.get("mas_agente_para_probar")
    if agente_id:
        try:
            agente_nuevo = next(a for a in controlador.list_agents() if a["agent_id"] == agente_id)
        except StopIteration:
            st.session_state.pop("mas_agente_para_probar", None)
        else:
            st.subheader("Demostrar que el coordinador lo encuentra")
            capacidades_prueba = [
                capacidad for capacidad in agente_nuevo["capabilities"]
                if capacidad in {
                    "classification.train", "regression.train", "clustering.kmeans", "clustering.hac",
                    "clustering.pca", "eda.profile", "eda.summary",
                }
            ]
            if not controlador.has_data:
                st.info("Carga datos antes de probar el agente.")
            elif not capacidades_prueba:
                st.info("Este agente no tiene una tarea compatible con la prueba guiada.")
            else:
                capacidad = st.selectbox(
                    "Tarea de prueba",
                    capacidades_prueba,
                    format_func=_etiqueta_capacidad,
                )
                if st.button("Probar con una tarea", type="primary"):
                    try:
                        payload = _tarea_demo(capacidad, controlador.active_data())
                        prueba = ejecutar(
                            lambda: controlador.probe_agent(agente_id, capacidad, payload),
                            "Probando el agente a través del coordinador...",
                        )
                    except ValueError as error:
                        st.error(str(error))
                        prueba = None
                    if prueba:
                        st.session_state["mas_ultima_prueba"] = prueba
                prueba = st.session_state.get("mas_ultima_prueba")
                if prueba:
                    st.success(f"Agente elegido por el coordinador: {prueba['selected_agent']}")
                    st.write(f"**Candidatos descubiertos:** {', '.join(prueba['candidates'])}")
                    if prueba.get("run_id"):
                        st.caption(f"Decisión guardada en el historial: {prueba['run_id']}")
