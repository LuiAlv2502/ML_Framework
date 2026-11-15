"""Capa de aprendizaje autom?tico del framework (backend).

Los algoritmos viven aqu? y nunca en el frontend. Se exponen los nombres p?blicos hist?ricos
(en espa?ol) para mantener la compatibilidad con el c?digo de los laboratorios.
"""

from .classification import ClasificacionModelos, SplitData, SupervisadoBase
from .clustering import ClusteringAnalysis, PCAAnalysis, UnsupervisedBase
from .data_model import DataModel
from .eda import DataFrame
from .regression import RegresionModelos

__all__ = [
    "ClasificacionModelos",
    "ClusteringAnalysis",
    "DataFrame",
    "DataModel",
    "PCAAnalysis",
    "RegresionModelos",
    "SplitData",
    "SupervisadoBase",
    "UnsupervisedBase",
]
