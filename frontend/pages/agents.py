"""Consulta y administración de agentes."""
from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.components.shared import ejecutar, obtener_controlador

st.title("Agentes")
st.write("Consulta qué tareas puede realizar cada agente y administra los que están disponibles.")
controlador = obtener_controlador()

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

def _resumen_tareas(capacidades: list[str]) -> str:
    grupos = {
        "eda": "exploración de datos",
        "clustering": "agrupamiento",
        "classification": "clasificación",
        "regression": "predicción",
        "coordination": "coordinación",
    }
    tareas = {grupos.get(capacidad.split(".", 1)[0], capacidad) for capacidad in capacidades}
    return ", ".join(sorted(tareas))

def _estado_legible(estado: str) -> str:
    return {
        "idle": "Disponible",
        "busy": "Trabajando",
        "degraded": "Disponible con incidencias",
        "paused": "Pausado",
        "stopped": "Detenido",
        "failed": "Requiere atención",
    }.get(estado, estado)

def _agentes() -> list[dict[str, Any]]:
    return controlador.list_agents()

agentes = _agentes()
if agentes:
    st.dataframe(
        [{"Nombre": a["name"], "Estado": _estado_legible(a["status"]),
          "Tareas": _resumen_tareas(a["capabilities"]), "Prioridad": a["priority"],
          "Éxitos": a["tasks_succeeded"], "Errores": a["tasks_failed"]}
         for a in agentes],
        width="stretch",
        hide_index=True,
    )

tipos = [item for item in controlador.agent_types() if item.get("creatable")]
with st.expander("Agregar un agente"):
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
                help="Las tareas elegidas deben ser compatibles con el tipo de agente.",
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
                        paso = 1 if clase == "entero" else 0.1
                        configuracion[clave] = st.number_input(
                            clave.replace("_", " ").capitalize(),
                            min_value=rango[0],
                            max_value=rango[1],
                            value=rango[0],
                            step=paso,
                        )
            crear = st.form_submit_button("Crear agente")
        if crear:
            if not nombre.strip():
                st.error("Escribe un nombre para el agente.")
            elif not capacidades:
                st.error("Selecciona al menos una tarea.")
            else:
                definicion = {
                    "agent_type": tipo_nombre,
                    "name": nombre.strip(),
                    "capabilities": capacidades,
                    "priority": prioridad,
                    "config": configuracion,
                }
                resultado = ejecutar(lambda: controlador.create_agent(definicion), "Creando agente...")
                if resultado:
                    st.success(f"Agente «{resultado['name']}» creado.")
                    st.rerun()

if agentes:
    st.subheader("Administrar un agente")
    nombres = {f"{a['name']} ({a['agent_id']})": a for a in agentes if a["agent_type"] != "coordinator"}
    if nombres:
        elegido = st.selectbox("Agente", list(nombres))
        agente = nombres[elegido]
        accion = st.selectbox(
            "Acción",
            ["start", "pause", "resume", "restart", "stop"],
            format_func=lambda accion: {
                "start": "Activar", "pause": "Pausar", "resume": "Reanudar",
                "restart": "Reiniciar", "stop": "Detener",
            }[accion],
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

with st.expander("Actividad reciente"):
    eventos = controlador.registry_events()
    if eventos:
        st.dataframe(eventos[::-1][:50], width="stretch", hide_index=True)
    else:
        st.caption("Todavía no hay cambios registrados en los agentes.")
