"""Fábrica de agentes: catálogo de tipos, creación dinámica, descubrimiento de plugins y rehidratación.

Un tipo nuevo se incorpora **sin modificar el núcleo**: basta con decorar la clase con ``@agent_type`` en
cualquier módulo de ``framework_ia/agents/plugins/`` (o llamar a :meth:`AgentFactory.register_agent_type`).

.. code-block:: python

    from framework_ia.agents import BaseAgent, agent_type

    @agent_type("mi_tipo", description="...")
    class MiAgente(BaseAgent):
        default_capabilities = frozenset({"classification.train"})

        def _execute(self, message):
            return {...}
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from typing import Any, Callable

from framework_ia.persistence.agent_state_store import AgentStateStore
from framework_ia.utils.exceptions import AgentError, FrameworkError, ValidationError
from framework_ia.utils.logging_config import get_logger
from framework_ia.utils.validation import validate_name

from .base_agent import BaseAgent
from .messaging import Message, MessageBus, Performative
from .registry import AgentRegistry

logger = get_logger("agents.factory")

_AGENT_TYPES: dict[str, type[BaseAgent]] = {}
TYPE_NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
PLUGINS_PACKAGE = "framework_ia.agents.plugins"


def _validar_tipo(name: str, cls: Any) -> None:
    if not TYPE_NAME.match(name or ""):
        raise ValidationError(f"Nombre de tipo inválido '{name}' (minúsculas, números y '_').")
    if not inspect.isclass(cls) or not issubclass(cls, BaseAgent):
        raise ValidationError(f"'{getattr(cls, '__name__', cls)}' no hereda de BaseAgent.")
    if inspect.isabstract(cls):
        raise ValidationError(f"'{cls.__name__}' es abstracta: implemente _execute.")


def agent_type(name: str, *, description: str | None = None) -> Callable[[type[BaseAgent]], type[BaseAgent]]:
    """Decorador que registra una clase de agente en el catálogo global de tipos."""

    def decorador(cls: type[BaseAgent]) -> type[BaseAgent]:
        _validar_tipo(name, cls)
        cls.agent_type = name
        if description:
            cls.description = description
        _AGENT_TYPES[name] = cls
        return cls

    return decorador


def registered_agent_types() -> dict[str, type[BaseAgent]]:
    """Copia del catálogo global de tipos."""
    return dict(_AGENT_TYPES)


class AgentFactory:
    """Crea, valida y registra agentes en el :class:`AgentRegistry` (y los suscribe al bus)."""

    def __init__(
        self,
        registry: AgentRegistry,
        bus: MessageBus | None = None,
        *,
        context: dict[str, Any] | None = None,
        state_store: AgentStateStore | None = None,
    ) -> None:
        self.registry = registry
        self.bus = bus
        self.context: dict[str, Any] = context if context is not None else {}
        self.state_store = state_store
        self._types: dict[str, type[BaseAgent]] = {}
        self.plugin_errors: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Catálogo de tipos
    # ------------------------------------------------------------------
    def register_agent_type(
        self, name: str, cls: type[BaseAgent], *, description: str | None = None, overwrite: bool = False
    ) -> None:
        """Registra un tipo solo en esta fábrica (útil en pruebas o extensiones en caliente)."""
        _validar_tipo(name, cls)
        if not overwrite and name in self.types:
            raise ValidationError(f"El tipo '{name}' ya existe.")
        cls.agent_type = name
        if description:
            cls.description = description
        self._types[name] = cls

    @property
    def types(self) -> dict[str, type[BaseAgent]]:
        return {**_AGENT_TYPES, **self._types}

    def available_types(self) -> list[dict[str, Any]]:
        """Plantillas disponibles para el formulario de creación."""
        filas = []
        for nombre, cls in sorted(self.types.items()):
            filas.append({
                "agent_type": nombre,
                "class": cls.__name__,
                "description": cls.description,
                "default_capabilities": sorted(cls.default_capabilities),
                "supported_capabilities": None if cls.supported_capabilities is None
                else sorted(cls.supported_capabilities),
                "config_schema": {k: list(v) for k, v in cls.config_schema.items()},
                "creatable": nombre != "coordinator",
            })
        return filas

    def discover_plugins(self, package: str = PLUGINS_PACKAGE) -> list[str]:
        """Importa todos los módulos del paquete de plugins (sus decoradores registran los tipos)."""
        try:
            paquete = importlib.import_module(package)
        except ImportError as error:
            self.plugin_errors[package] = str(error)
            return []
        importados = []
        for modulo in pkgutil.iter_modules(getattr(paquete, "__path__", []), prefix=f"{package}."):
            try:
                importlib.import_module(modulo.name)
                importados.append(modulo.name)
            except Exception as error:  # un plugin defectuoso no debe impedir el arranque
                self.plugin_errors[modulo.name] = f"{type(error).__name__}: {error}"
                logger.warning("No se pudo cargar el plugin %s: %s", modulo.name, error)
        return importados

    # ------------------------------------------------------------------
    # Creación
    # ------------------------------------------------------------------
    def create(
        self,
        agent_type: str,
        name: str,
        capabilities: list[str] | None = None,
        config: dict[str, Any] | None = None,
        *,
        priority: int = 10,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        persist: bool = False,
    ) -> BaseAgent:
        """Valida la definición, instancia el agente, lo registra, lo suscribe al bus y (opcional) lo persiste."""
        cls = self.types.get(agent_type)
        if cls is None:
            raise ValidationError(
                f"Tipo de agente desconocido '{agent_type}'. Disponibles: {', '.join(sorted(self.types))}."
            )
        nombre = validate_name(name, what="nombre del agente")
        if self.registry.find_by_name(nombre) is not None:
            raise ValidationError(f"Ya existe un agente llamado '{nombre}'.")
        if agent_id and agent_id in self.registry:
            raise ValidationError(f"Ya existe un agente con id '{agent_id}'.")
        agente = cls(
            name=nombre,
            capabilities=list(capabilities) if capabilities else None,
            agent_id=agent_id,
            priority=priority,
            config=config or {},
            context=self.context,
            metadata=metadata,
        )
        self.registry.register(agente)
        if self.bus is not None:
            self.bus.subscribe(agente.agent_id, agente.handle)
            self.bus.publish(Message(
                sender="factory", receiver=agente.agent_id, performative=Performative.CONFIRM,
                task="agent.created", payload={"agent_type": agent_type, "capabilities": sorted(agente.capabilities),
                                               "priority": agente.priority},
            ))
        if persist:
            if self.state_store is None:
                raise AgentError("No hay almacenamiento configurado para persistir agentes.")
            self.state_store.upsert(agente.to_definition())
        logger.info("Agente creado: %s", agente)
        return agente

    def create_from_definition(self, definition: dict[str, Any], *, persist: bool = False) -> BaseAgent:
        """Crea un agente a partir de un diccionario (formulario del dashboard o archivo persistido)."""
        if not isinstance(definition, dict):
            raise ValidationError("La definición del agente debe ser un diccionario.")
        faltan = [c for c in ("agent_type", "name") if not definition.get(c)]
        if faltan:
            raise ValidationError(f"Faltan campos en la definición del agente: {faltan}.")
        return self.create(
            definition["agent_type"],
            definition["name"],
            definition.get("capabilities"),
            definition.get("config"),
            priority=int(definition.get("priority", 10)),
            agent_id=definition.get("agent_id"),
            metadata=definition.get("metadata"),
            persist=persist,
        )

    def rehydrate(self, state_store: AgentStateStore | None = None) -> tuple[list[BaseAgent], dict[str, str]]:
        """Recrea los agentes persistidos; devuelve los creados y los errores por nombre."""
        almacen = state_store or self.state_store
        if almacen is None:
            return [], {}
        creados, errores = [], {}
        for definicion in almacen.load():
            nombre = str(definicion.get("name"))
            if self.registry.find_by_name(nombre) is not None:
                continue
            try:
                creados.append(self.create_from_definition(definicion))
            except FrameworkError as error:
                errores[nombre] = error.user_message
            except Exception as error:
                errores[nombre] = f"{type(error).__name__}: {error}"
        return creados, errores

    def destroy(self, agent_id: str, *, forget: bool = False) -> BaseAgent:
        """Da de baja un agente (y opcionalmente borra su definición persistida)."""
        agente = self.registry.unregister(agent_id)
        if self.bus is not None:
            self.bus.unsubscribe(agent_id)
            self.bus.publish(Message(sender="factory", receiver=agent_id, performative=Performative.CONFIRM,
                                     task="agent.removed", payload={"name": agente.name}))
        if forget and self.state_store is not None:
            self.state_store.remove(agente.name)
        return agente


__all__ = ["AgentFactory", "agent_type", "registered_agent_types"]
