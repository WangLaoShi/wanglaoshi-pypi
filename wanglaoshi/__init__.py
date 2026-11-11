"""
wanglaoshi 顶层导出 (API)

自 1.15.1 起，公开写法以稳定公开接口为准（见 docs/ARCHITECTURE.md）::

    from wanglaoshi.runtime import ready
    from wanglaoshi.analyze import DataAnalyzer
    from wanglaoshi.compete import QuickBaseline, SubmissionGenerator
    from wanglaoshi.features import AdvancedFeatureEngineer
    from wanglaoshi.analytics import StatsToolkit, TimeSeriesTools
    from wanglaoshi.ml import DataPreprocessor, MLModel
    from wanglaoshi.runtime import JupyterEnv, VERSIONS
    from wanglaoshi._testing import assert_close

旧顶层捷径（``CompetitionTools`` / ``FeatureEngineering`` / 顶层 ``DataAnalyzer`` 等）
已放弃为推荐路径；根上 shim 仅过渡保留，新代码请勿再依赖。

导入策略：稳定公开接口与常用捷径从子包 re-export；兼容 shim
（Analyzer / CompetitionTools / MLDL 等）改为按需懒加载，减轻无关依赖拖累。
"""

from __future__ import annotations

import importlib
from typing import Any

# ---------------------------------------------------------------------------
# 稳定公开接口（实现所在）
# ---------------------------------------------------------------------------
from . import _testing  # noqa: F401
from . import analytics
from . import analyze
from . import business
from . import compete
from . import features
from . import ml
from . import runtime

# Notebook / EDA 捷径（走公开接口，不经根 shim）
from .analyze import DataAnalyzer, analyze_data, analyze_multiple_files, analyze_notebook
from .runtime import ready

# 业务流水线捷径
from .business import (
    BusinessPipeline,
    RunRecord,
    StageResult,
    add_value_tiers,
    archive_charts,
    archive_delivery,
    build_analysis_plan,
    build_metric_dict,
    calculate_metrics,
    check_join_inflation,
    complete_period_index,
    convert_units,
    generate_insights,
    intake_file,
    reconcile_to_source,
    run_eda_profile,
    yoy_with_rules,
)

# analytics / _testing 捷径
from .analytics import (
    AnomalyDetector,
    CalibrationReport,
    ExperimentDesign,
    GroupMetricsReport,
    RankingMetrics,
    StatsToolkit,
    SurvivalAnalyzer,
    TimeSeriesTools,
    expected_calibration_error,
)
from ._testing import (
    SCORE_FLOORS,
    ChartMeta,
    DataRegistry,
    ExperimentContext,
    RegressionSummary,
    assert_above_floor,
    assert_close,
    assert_frame_contract,
    assert_metrics_close,
    assert_report_ok,
    file_sha256,
    from_junit_xml,
    write_chart_sidecar,
)

# runtime 兼容名（模块对象，与历史 `from wanglaoshi import VERSIONS` 一致）
from .runtime import JupyterEnv, JupyterFont, Useful, VERSIONS, WebGetter

# ML：仅取可用性标志，避免经根 MLDL shim
from .ml import TORCH_AVAILABLE as _MLDL_TORCH_AVAILABLE

_MLDL_AVAILABLE = bool(_MLDL_TORCH_AVAILABLE)

# ---------------------------------------------------------------------------
# Phase 2：公开面分层
# ---------------------------------------------------------------------------
_STABLE_PUBLIC = [
    "compete",
    "analyze",
    "features",
    "analytics",
    "runtime",
    "ml",
    "_testing",
    "business",
    "ready",
    "DataAnalyzer",
    "analyze_data",
    "analyze_multiple_files",
    "analyze_notebook",
    "BusinessPipeline",
    "StageResult",
    "RunRecord",
    "intake_file",
    "build_analysis_plan",
    "build_metric_dict",
    "calculate_metrics",
    "generate_insights",
    "archive_delivery",
    "archive_charts",
    "run_eda_profile",
    "check_join_inflation",
    "reconcile_to_source",
    "complete_period_index",
    "yoy_with_rules",
    "convert_units",
    "add_value_tiers",
    "stats",
    "StatsToolkit",
    "ExperimentDesign",
    "TimeSeriesTools",
    "CalibrationReport",
    "expected_calibration_error",
    "GroupMetricsReport",
    "AnomalyDetector",
    "RankingMetrics",
    "SurvivalAnalyzer",
    "assert_close",
    "assert_metrics_close",
    "assert_frame_contract",
    "DataRegistry",
    "file_sha256",
    "ExperimentContext",
    "SCORE_FLOORS",
    "assert_above_floor",
    "ChartMeta",
    "assert_report_ok",
    "write_chart_sidecar",
    "RegressionSummary",
    "from_junit_xml",
    "require_data_path",
    "optional_data_available",
]

_COMPAT_EXPORTS = [
    "JupyterEnv",
    "VERSIONS",
    "Analyzer",
    "Analyzer_Plain",
    "CompetitionTools",
    "FeatureEngineering",
    "MLDL",
    "JupyterFont",
    "Useful",
    "WebGetter",
]

__all__ = _STABLE_PUBLIC + _COMPAT_EXPORTS

# 兼容 shim / pytest 工具：首次访问时再加载
_LAZY_MODULES = {
    "Analyzer": ".Analyzer",
    "Analyzer_Plain": ".Analyzer_Plain",
    "CompetitionTools": ".CompetitionTools",
    "FeatureEngineering": ".FeatureEngineering",
    "MLDL": ".MLDL",
    "stats": ".stats",
}


def __getattr__(name: str) -> Any:
    """懒加载：兼容 shim、legacy 与 pytest 测试工具。"""
    if name in {"require_data_path", "optional_data_available"}:
        from . import testing_utils

        return getattr(testing_utils, name)
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name], __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()) | set(_LAZY_MODULES))
