"""
wanglaoshi 顶层导出 (API)

推荐新代码走架构门面（见 docs/ARCHITECTURE.md）::

    from wanglaoshi import ready
    from wanglaoshi.analyze import DataAnalyzer
    from wanglaoshi.compete import QuickBaseline, SubmissionGenerator
    from wanglaoshi.features import AdvancedFeatureEngineer
    from wanglaoshi.analytics import StatsToolkit, TimeSeriesTools
    from wanglaoshi.ml import DataPreprocessor, MLModel
    from wanglaoshi.runtime import JupyterEnv, VERSIONS
    from wanglaoshi._testing import assert_close

课堂兼容仍可用::

    from wanglaoshi import CompetitionTools as CT
    from wanglaoshi import FeatureEngineering as FE
    from wanglaoshi import DataAnalyzer, analyze_data
"""

# 模块导出（用于 from wanglaoshi import VERSIONS as V 这类用法）
from . import JupyterEnv
from . import VERSIONS
from . import Analyzer
from . import CompetitionTools
from . import FeatureEngineering
from . import JupyterFont
from . import Useful
from . import WebGetter
# Analyzer_Plain 为 legacy：改懒加载，避免 import wanglaoshi 就刷 DeprecationWarning

# MLDL：实现在 ml.core；根 MLDL 为 shim。torch 可选（TORCH_AVAILABLE），模块本身始终可导入。
from . import MLDL  # noqa: F401
from .ml.core import TORCH_AVAILABLE as _MLDL_TORCH_AVAILABLE

_MLDL_AVAILABLE = bool(_MLDL_TORCH_AVAILABLE)

# 直接导出常用类和函数（方便用户少写一层模块前缀）
from .Analyzer import DataAnalyzer, analyze_data, analyze_multiple_files, analyze_notebook
from .ready import ready

# 业务流水线（Skill 固化）；不依赖可选重型包
from . import business
from .business import (
    BusinessPipeline,
    StageResult,
    RunRecord,
    intake_file,
    build_analysis_plan,
    calculate_metrics,
    generate_insights,
    archive_delivery,
    archive_charts,
    run_eda_profile,
    check_join_inflation,
    reconcile_to_source,
    complete_period_index,
    yoy_with_rules,
    convert_units,
    add_value_tiers,
)
from . import stats
from .stats import StatsToolkit, ExperimentDesign
from .timeseries import TimeSeriesTools
from .calibration import CalibrationReport, expected_calibration_error
from .fairness import GroupMetricsReport
from .anomaly import AnomalyDetector
from .ranking import RankingMetrics
from .survival import SurvivalAnalyzer
from .asserting import assert_close, assert_metrics_close, assert_frame_contract
from .data_registry import DataRegistry, file_sha256
from .experiment_context import ExperimentContext
from .score_floors import SCORE_FLOORS, assert_above_floor
from .report_meta import ChartMeta, assert_report_ok, write_chart_sidecar
from .regression_summary import RegressionSummary, from_junit_xml

# 架构门面（Phase 1：re-export；旧路径仍可用）
from . import compete, analyze, features, analytics, runtime, ml
from . import _testing  # noqa: F401

# testing_utils 依赖 pytest，仅开发/测试需要；勿在 import wanglaoshi 时强拉
# 仍支持：from wanglaoshi import require_data_path（首次访问时再加载）

# ---------------------------------------------------------------------------
# Phase 2：公开面分层
# - _STABLE_PUBLIC：推荐对外稳定入口（门面 + 常用捷径）
# - _COMPAT_EXPORTS：课堂/历史兼容，仍可 import，新代码请优先门面
# ---------------------------------------------------------------------------
_STABLE_PUBLIC = [
    # 架构门面
    "compete",
    "analyze",
    "features",
    "analytics",
    "runtime",
    "ml",
    "_testing",
    "business",
    # Notebook / EDA 捷径
    "ready",
    "DataAnalyzer",
    "analyze_data",
    "analyze_multiple_files",
    "analyze_notebook",
    # business 捷径
    "BusinessPipeline",
    "StageResult",
    "RunRecord",
    "intake_file",
    "build_analysis_plan",
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
    # analytics 捷径
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
    # 测试 / 实验治理捷径
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
    # 历史模块级导入（等价于门面 / 子模块）
    "JupyterEnv",
    "VERSIONS",
    "Analyzer",
    "Analyzer_Plain",  # legacy：请优先 wanglaoshi.analyze
    "CompetitionTools",
    "FeatureEngineering",
    "MLDL",  # 兼容模块路径；推荐 wanglaoshi.ml（torch 可选）
    "JupyterFont",
    "Useful",
    "WebGetter",
]

__all__ = _STABLE_PUBLIC + _COMPAT_EXPORTS


def __getattr__(name: str):
    """懒加载：测试工具（避免强依赖 pytest）与 legacy 模块（触发 soft deprecate）。"""
    if name in {"require_data_path", "optional_data_available"}:
        from . import testing_utils

        return getattr(testing_utils, name)
    if name == "Analyzer_Plain":
        import importlib

        return importlib.import_module(".Analyzer_Plain", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
