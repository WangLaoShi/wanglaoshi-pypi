"""
wanglaoshi 顶层导出 (API)

为了让 README 中的用法与实际安装后的体验一致，这里整理了推荐的对外 API。

常见用法示例：

    from wanglaoshi import ready
    ready()  # 一键：关警告 + 打印环境（推荐，替代 JE.no_warning + V.show_env_info 拼盘）

    from wanglaoshi import Analyzer
    from wanglaoshi import CompetitionTools as CT
    from wanglaoshi import FeatureEngineering as FE
    from wanglaoshi import DataAnalyzer, analyze_data, analyze_multiple_files, analyze_notebook

模块级别导出：
- JupyterEnv
- VERSIONS
- Analyzer
- Analyzer_Plain
- CompetitionTools
- FeatureEngineering
- MLDL
- JupyterFont
- Useful
- WebGetter

函数 / 类级别导出：
- DataAnalyzer
- analyze_data
- analyze_multiple_files
- analyze_notebook
"""

# 模块导出（用于 from wanglaoshi import VERSIONS as V 这类用法）
from . import JupyterEnv
from . import VERSIONS
from . import Analyzer
from . import Analyzer_Plain
from . import CompetitionTools
from . import FeatureEngineering
from . import JupyterFont
from . import Useful
from . import WebGetter

# MLDL 依赖 torch，为了不影响普通用户导入，本模块中将其视为「可选模块」。
# - 如果用户未安装 torch，则 from wanglaoshi import MLDL 不会自动失败；
# - 需要使用 MLDL 时，推荐：在安装 torch 后，单独 from wanglaoshi import MLDL。
try:
    from . import MLDL  # type: ignore  # noqa: F401
    _MLDL_AVAILABLE = True
except ImportError:
    MLDL = None  # type: ignore
    _MLDL_AVAILABLE = False

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
)
from . import stats
from .stats import StatsToolkit, ExperimentDesign
from .timeseries import TimeSeriesTools
from .calibration import CalibrationReport, expected_calibration_error
from .fairness import GroupMetricsReport
from .anomaly import AnomalyDetector
from .ranking import RankingMetrics
from .survival import SurvivalAnalyzer

__all__ = [
    # 模块
    "JupyterEnv",
    "VERSIONS",
    "Analyzer",
    "Analyzer_Plain",
    "CompetitionTools",
    "FeatureEngineering",
    # MLDL 是可选模块，这里继续导出名称，但需要用户自己确保安装了 torch
    "MLDL",
    "JupyterFont",
    "Useful",
    "WebGetter",
    # 类 / 函数
    "ready",
    "DataAnalyzer",
    "analyze_data",
    "analyze_multiple_files",
    "analyze_notebook",
    # business
    "business",
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
]
