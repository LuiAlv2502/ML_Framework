"""Preprocesamiento reutilizable (imputación, codificación, escalado, selección de atributos)."""

from .transforms import DESCRIPTIONS, DTYPES, FILTER_OPERATORS, OPERATIONS, apply_operation, detect_id_columns

__all__ = ["DESCRIPTIONS", "DTYPES", "FILTER_OPERATORS", "OPERATIONS", "apply_operation", "detect_id_columns"]
