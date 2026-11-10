"""
比赛工具模块 (CompetitionTools.py)

Phase 3：实现已拆到 ``wanglaoshi.compete.*``；本模块为兼容 shim，符号不变。

推荐新代码::

    from wanglaoshi.compete import SubmissionGenerator, QuickBaseline

仍支持::

    from wanglaoshi import CompetitionTools as CT
    from wanglaoshi.CompetitionTools import SubmissionGenerator
"""

from __future__ import annotations

from .compete import (
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
