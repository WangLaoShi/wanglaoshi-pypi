"""竞赛工具门面。

推荐::

    from wanglaoshi.compete import SubmissionGenerator, QuickBaseline
"""

from __future__ import annotations

from .baseline import QuickBaseline
from .cv import CrossValidator
from .ensemble import ModelEnsemble
from .hyperopt import HyperparameterOptimizer
from .importance import FeatureImportanceAnalyzer
from .leakage import LeakageDetector
from .submission import SubmissionGenerator
from .tracker import ExperimentTracker

__all__ = [
    "SubmissionGenerator",
    "QuickBaseline",
    "LeakageDetector",
    "ModelEnsemble",
    "FeatureImportanceAnalyzer",
    "CrossValidator",
    "HyperparameterOptimizer",
    "ExperimentTracker",
]
