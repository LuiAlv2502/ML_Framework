"""Capa de aprendizaje automático del framework (backend).

Los algoritmos viven aquí y nunca en el frontend. Se exponen los nombres públicos históricos
(en español) para mantener la compatibilidad con el código de los laboratorios.
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
