"""Algoritmos de aprendizaje supervisado."""

from .supervised_base import SplitData, SupervisadoBase
from .Classification import ClasificacionModelos
from .Regression import RegresionModelos

__all__ = [
	"ClasificacionModelos",
	"RegresionModelos",
	"SplitData",
	"SupervisadoBase",
]