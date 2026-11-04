"""Integración con el LLM (Google Gemini) que gobierna el bucle agéntico del EDA.

Gemini expone un endpoint compatible con OpenAI, por lo que solo hace falta el paquete 'openai'.
"""
from __future__ import annotations

import importlib.util
import os
import time
from typing import Any

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
    """chat.completions.create con espera y reintento ante límites de tasa (429) del plan gratuito."""
    for intento in range(reintentos + 1):
        try:
            return cliente.chat.completions.create(**kwargs)
        except Exception as exc:
            if type(exc).__name__ not in ("RateLimitError", "InternalServerError") or intento == reintentos:
                raise
            time.sleep(15 * (intento + 1))


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
