"""Algoritmos de aprendizaje supervisado."""

from ._base import SplitData, SupervisadoBase
from .Classification import ClasificacionModelos
from .Regression import RegresionModelos

__all__ = [
	"ClasificacionModelos",
	"RegresionModelos",
	"SplitData",
	"SupervisadoBase",
]