"""通用分析能力门面：时序 / 生存 / 排序 / 公平 / 校准 / 异常 / 统计。"""

from __future__ import annotations

from .anomaly import AnomalyDetector
from .calibration import CalibrationReport, expected_calibration_error
from .fairness import GroupMetricsReport
from .ranking import RankingMetrics
from .stats import ExperimentDesign, StatsToolkit
from .survival import SurvivalAnalyzer
from .timeseries import TimeSeriesTools

__all__ = [
    "TimeSeriesTools",
    "SurvivalAnalyzer",
    "RankingMetrics",
    "GroupMetricsReport",
    "CalibrationReport",
    "expected_calibration_error",
    "AnomalyDetector",
    "StatsToolkit",
    "ExperimentDesign",
]
