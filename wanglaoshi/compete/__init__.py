"""竞赛工具门面（零破坏：实现仍在 ``CompetitionTools``）。

推荐::

    from wanglaoshi.compete import SubmissionGenerator, QuickBaseline

过渡期仍可用::

    from wanglaoshi import CompetitionTools as CT
    from wanglaoshi.CompetitionTools import SubmissionGenerator
"""

from __future__ import annotations

from ..CompetitionTools import (
    CrossValidator,
    ExperimentTracker,
    FeatureImportanceAnalyzer,
    HyperparameterOptimizer,
    LeakageDetector,
    ModelEnsemble,
    QuickBaseline,
    SubmissionGenerator,
)

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
