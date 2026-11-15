"""Sistema multiagente del framework.

- Infraestructura: :class:`Message`/:class:`MessageBus` (mensajería), :class:`BaseAgent` (clase base),
  :class:`AgentRegistry` (comunidad), :class:`AgentFactory` y ``@agent_type`` (creación dinámica),
  :class:`Agency` (ciclo de vida, asignación, cooperación y fallos) y :class:`CoordinatorAgent`.
- Agentes especializados (adaptadores del MAS): :class:`EDAAgent`, :class:`ClusteringAgent`,
  :class:`ClassificationAgent`, :class:`RegressionAgent`.
- Agentes de los laboratorios (se conservan con sus nombres en español): ``AgenteEDA``,
  ``AgenteNoSupervisado``, ``AgenteClasificacion``, ``AgenteRegresion``, ``AgenteSupervisado``.
"""

from .agency import POLICIES, Agency, DispatchAttempt, DispatchResult
from .agent_factory import AgentFactory, agent_type, registered_agent_types
from .base_agent import AVAILABLE_STATUSES, AgentStatus, BaseAgent
from .bootstrap import COORDINATOR_ID, DEFAULT_AGENTS, build_default_agency
from .classification_agent import AgenteClasificacion, ClassificationAgent
from .clustering_agent import AgenteNoSupervisado, ClusteringAgent
from .coordinator_agent import CoordinatorAgent, CoordinatorResult
from .eda_agent import AgenteEDA, EDAAgent, Paso
from .llm import ErrorLLM
from .messaging import Message, MessageBus, MessageStatus, Performative
from .registry import AgentRegistry
from .regression_agent import AgenteRegresion, RegressionAgent
from .supervised_agent import AgenteSupervisado, SupervisedAgentAdapter

__all__ = [
    "AVAILABLE_STATUSES",
    "COORDINATOR_ID",
    "DEFAULT_AGENTS",
    "POLICIES",
    "AgenteClasificacion",
    "AgenteEDA",
    "AgenteNoSupervisado",
    "AgenteRegresion",
    "AgenteSupervisado",
    "Agency",
    "AgentFactory",
    "AgentRegistry",
    "AgentStatus",
    "BaseAgent",
    "ClassificationAgent",
    "ClusteringAgent",
    "CoordinatorAgent",
    "CoordinatorResult",
    "DispatchAttempt",
    "DispatchResult",
    "EDAAgent",
    "ErrorLLM",
    "Message",
    "MessageBus",
    "MessageStatus",
    "Paso",
    "Performative",
    "RegressionAgent",
    "SupervisedAgentAdapter",
    "agent_type",
    "build_default_agency",
    "registered_agent_types",
]
