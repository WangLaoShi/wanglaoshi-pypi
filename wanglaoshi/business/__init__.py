"""业务分析流水线（Skill 固化入口）。

已覆盖：intake → structuring → cleaning → derived → planning → metrics
→ analysis → insights → archive，以及总控 pipeline。

设计原则见仓库根目录 DEVELOPMENT_ROADMAP.md（工作流 D）。
"""

from .contracts import StageResult, RunRecord
from .intake import intake_file
from .pipeline import BusinessPipeline
from .structuring import structure_dataframe, structure_file
from .cleaning import clean_dataframe
from .derived import derive_metrics
from .planning import build_analysis_plan
from .metrics import calculate_metrics, DEFAULT_METRIC_DICT
from .analysis import run_multidimensional_analysis
from .insights import generate_insights
from .archive import archive_delivery

__all__ = [
    "StageResult",
    "RunRecord",
    "intake_file",
    "BusinessPipeline",
    "structure_dataframe",
    "structure_file",
    "clean_dataframe",
    "derive_metrics",
    "build_analysis_plan",
    "calculate_metrics",
    "DEFAULT_METRIC_DICT",
    "run_multidimensional_analysis",
    "generate_insights",
    "archive_delivery",
]
