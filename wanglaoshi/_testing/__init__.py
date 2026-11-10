"""测试与回归门禁辅助（开发用途；生产代码勿强依赖）。"""

from __future__ import annotations

from .asserting import assert_close, assert_frame_contract, assert_metrics_close
from .data_registry import DataRegistry, file_sha256
from .experiment_context import ExperimentContext
from .regression_summary import RegressionSummary, from_junit_xml
from .report_meta import ChartMeta, assert_report_ok, write_chart_sidecar
from .score_floors import SCORE_FLOORS, assert_above_floor

try:
    from .testing_utils import optional_data_available, require_data_path
except ImportError:  # pragma: no cover
    require_data_path = None  # type: ignore
    optional_data_available = None  # type: ignore

__all__ = [
    "assert_close",
    "assert_metrics_close",
    "assert_frame_contract",
    "SCORE_FLOORS",
    "assert_above_floor",
    "ChartMeta",
    "assert_report_ok",
    "write_chart_sidecar",
    "RegressionSummary",
    "from_junit_xml",
    "DataRegistry",
    "file_sha256",
    "ExperimentContext",
    "require_data_path",
    "optional_data_available",
]
