"""ML / DL 门面（可选 torch；无 torch 时 sklearn 部分仍可用）。

::

    from wanglaoshi.ml import DataPreprocessor, MLModel, TORCH_AVAILABLE
"""

from __future__ import annotations

from .core import (
    TORCH_AVAILABLE,
    DataPreprocessor,
    DLModel,
    FeatureEngineer,
    MLModel,
    ModelEvaluator,
    NeuralNetwork,
    example_usage,
)

__all__ = [
    "TORCH_AVAILABLE",
    "DataPreprocessor",
    "FeatureEngineer",
    "MLModel",
    "ModelEvaluator",
    "NeuralNetwork",
    "DLModel",
    "example_usage",
]
