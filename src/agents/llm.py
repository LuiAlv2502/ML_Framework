"""Integración con el LLM (Google Gemini) que gobierna el bucle agéntico de los agentes.

Gemini expone un endpoint compatible con OpenAI, por lo que solo hace falta el paquete 'openai'.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

MODELO_POR_DEFECTO = os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash"
URL_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
VARIABLES_CLAVE = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

SISTEMA = (
    "Eres un agente experto en Análisis Exploratorio de Datos (EDA). Controlas el análisis usando "
    "SOLO las herramientas disponibles. Recibirás un perfil de los datos (sin valores crudos) y un plan "
    "sugerido por reglas. Para cada paso de la lista decide si ejecutarlo (ejecutar_paso) u omitirlo "
    "(omitir_paso), siempre con una justificación breve. Puedes elegir el orden (p. ej. duplicados antes "
    "de faltantes) y debes replanificar según lo que cada resultado revele. Cuando todos los pasos estén "
    "decididos, llama a finalizar con un análisis crítico en español: hallazgos clave, riesgos y siguientes "
    "pasos para modelar. No inventes cifras: usa solo lo que devolvieron las herramientas. "
    "Para ahorrar peticiones, llama varias herramientas en un mismo turno cuando no dependan entre sí."
)


class ErrorLLM(RuntimeError):
    """El agente no pudo usar el LLM (falta la clave, el paquete, red, saldo...)."""


def _codigo_estado(exc: Exception) -> int | None:
    codigo = getattr(exc, "status_code", None)
    if codigo is None:
        respuesta = getattr(exc, "response", None)
        codigo = getattr(respuesta, "status_code", None)
    try:
        return int(codigo) if codigo is not None else None
    except (TypeError, ValueError):
        return None


def _espera_reintento(exc: Exception, intento: int) -> float:
    respuesta = getattr(exc, "response", None)
    cabeceras = getattr(respuesta, "headers", {}) or {}
    cabeceras = {str(nombre).lower(): valor for nombre, valor in cabeceras.items()}

    milisegundos = cabeceras.get("retry-after-ms")
    if milisegundos is not None:
        try:
            return max(0.0, float(milisegundos) / 1000)
        except (TypeError, ValueError):
            pass

    reintentar_despues = cabeceras.get("retry-after")
    if reintentar_despues is not None:
        try:
            return max(0.0, float(reintentar_despues))
        except (TypeError, ValueError):
            try:
                fecha = parsedate_to_datetime(str(reintentar_despues))
                if fecha.tzinfo is None:
                    fecha = fecha.replace(tzinfo=timezone.utc)
                return max(0.0, (fecha - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass

    return min(5 * (2 ** intento), 60)


def _clave() -> str | None:
    for nombre in VARIABLES_CLAVE:
        if os.environ.get(nombre):
            return os.environ[nombre]
    return None


def motivo_no_disponible() -> str | None:
    """Explica por qué no se puede usar el LLM (None si está disponible)."""
    if importlib.util.find_spec("openai") is None:
        return "El paquete 'openai' no está instalado en este entorno (pip install openai)."
    if not _clave():
        return "La variable de entorno GEMINI_API_KEY no está definida en el proceso que ejecuta la app."
    return None


def disponible() -> bool:
    return motivo_no_disponible() is None


def crear_cliente():
    from openai import OpenAI

    return OpenAI(api_key=_clave(), base_url=URL_BASE)


def completar(cliente, reintentos: int = 4, **kwargs):
    """Reintenta límites temporales usando Retry-After cuando el servidor lo proporciona."""
    for intento in range(reintentos + 1):
        try:
            return cliente.chat.completions.create(**kwargs)
        except Exception as exc:
            codigo = _codigo_estado(exc)
            rate_limit = codigo == 429 or type(exc).__name__ == "RateLimitError"
            error_temporal = rate_limit or codigo is not None and codigo >= 500 or (
                type(exc).__name__ == "InternalServerError"
            )
            if not error_temporal or intento == reintentos:
                if rate_limit:
                    raise ErrorLLM(
                        "Gemini mantuvo el límite de solicitudes (HTTP 429) después de "
                        f"{reintentos + 1} intentos. Puede ser un límite temporal o una cuota "
                        "agotada; revise la cuota de Gemini AI Studio y vuelva a intentarlo "
                        "cuando se restablezca."
                    ) from exc
                raise
            time.sleep(_espera_reintento(exc, intento))


def llamada_como_dict(llamada: Any) -> dict[str, Any]:
    # Gemini 3 exige devolver la llamada completa (incluye thought_signature en extra_content).
    if hasattr(llamada, "model_dump"):
        resultado = llamada.model_dump(exclude_none=True)
        resultado["type"] = "function"
        return resultado
    return {
        "id": llamada.id,
        "type": "function",
        "function": {"name": llamada.function.name, "arguments": llamada.function.arguments},
    }


def bucle_herramientas(
    cliente: Any,
    modelo: str,
    mensajes: list[dict[str, Any]],
    herramientas: list[dict[str, Any]],
    despachar: Callable[[str, dict[str, Any]], dict[str, Any]],
    max_iteraciones: int,
) -> dict[str, int]:
    """Bucle de tool calling común a los agentes.

    Termina cuando 'finalizar' se ejecuta sin error y devuelve estadísticas de uso del LLM.
    Lanza RuntimeError si el LLM no finaliza dentro de max_iteraciones.
    """
    uso = {
        "iteraciones": 0,
        "llamadas_herramientas": 0,
        "errores_herramientas": 0,
        "tokens_entrada": 0,
        "tokens_salida": 0,
    }
    for _ in range(max_iteraciones):
        uso["iteraciones"] += 1
        respuesta = completar(
            cliente, model=modelo, messages=mensajes, tools=herramientas, tool_choice="auto", temperature=0
        )
        consumo = getattr(respuesta, "usage", None)
        if consumo is not None:
            uso["tokens_entrada"] += int(getattr(consumo, "prompt_tokens", 0) or 0)
            uso["tokens_salida"] += int(getattr(consumo, "completion_tokens", 0) or 0)
        mensaje = respuesta.choices[0].message
        llamadas = mensaje.tool_calls or []
        mensajes.append({
            "role": "assistant",
            "content": mensaje.content,
            **({"tool_calls": [llamada_como_dict(c) for c in llamadas]} if llamadas else {}),
        })
        if not llamadas:
            mensajes.append({"role": "user", "content": "Continúa usando las herramientas hasta finalizar."})
            continue
        terminado = False
        for llamada in llamadas:
            uso["llamadas_herramientas"] += 1
            try:
                args = json.loads(llamada.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            salida = despachar(llamada.function.name, args)
            if "error" in salida:
                uso["errores_herramientas"] += 1
            elif llamada.function.name == "finalizar":
                terminado = True
            mensajes.append({
                "role": "tool",
                "tool_call_id": llamada.id,
                "content": json.dumps(salida, ensure_ascii=False, default=str),
            })
        if terminado:
            return uso
    raise RuntimeError("El LLM no finalizó dentro del máximo de iteraciones.")


def describir_motivo(error: Exception) -> str:
    """Texto del motivo por el que un agente pasó al respaldo por reglas."""
    return str(error) if isinstance(error, ErrorLLM) else f"{type(error).__name__}: {error}"


def resumen_uso(uso: dict[str, int]) -> str:
    return (
        f"{uso.get('iteraciones', 0)} turnos, {uso.get('llamadas_herramientas', 0)} llamadas a herramientas "
        f"({uso.get('errores_herramientas', 0)} rechazadas), {uso.get('tokens_entrada', 0)} tokens de entrada "
        f"y {uso.get('tokens_salida', 0)} de salida."
    )


def recortar(texto: Any, limite: int = 500) -> Any:
    """Acorta textos largos para no gastar tokens por minuto reenviando todo el historial."""
    if isinstance(texto, str) and len(texto) > limite:
        return texto[:limite] + "…"
    if isinstance(texto, list):
        return [recortar(t, limite) for t in texto[:6]]
    return texto


def herramientas(claves: list[str]) -> list[dict]:
    justificacion = {"type": "string", "description": "Razón breve de la decisión."}
    clave = {"type": "string", "enum": claves}
    return [
        {"type": "function", "function": {
            "name": "ejecutar_paso",
            "description": "Ejecuta un paso del EDA y devuelve su observación, interpretación y recomendaciones.",
            "parameters": {"type": "object", "properties": {"clave": clave, "justificacion": justificacion},
                           "required": ["clave", "justificacion"]}}},
        {"type": "function", "function": {
            "name": "omitir_paso",
            "description": "Omite un paso que los datos no justifican.",
            "parameters": {"type": "object", "properties": {"clave": clave, "justificacion": justificacion},
                           "required": ["clave", "justificacion"]}}},
        {"type": "function", "function": {
            "name": "finalizar",
            "description": "Termina el análisis. Solo válido cuando todos los pasos fueron ejecutados u omitidos.",
            "parameters": {"type": "object", "properties": {"analisis": {
                "type": "string", "description": "Análisis crítico final en español."}},
                "required": ["analisis"]}}},
    ]
