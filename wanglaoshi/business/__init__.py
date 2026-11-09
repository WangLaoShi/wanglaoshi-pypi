"""业务分析流水线（Skill 固化入口）。

主链：intake → structuring → cleaning → EDA → derived → planning → metrics
→ analysis → visualization → insights → archive，以及总控 pipeline。
"""

from .contracts import StageResult, RunRecord
from .intake import intake_file
from .pipeline import BusinessPipeline
from .structuring import structure_dataframe, structure_file
from .cleaning import clean_dataframe
from .eda import run_eda_profile, STAT_METHOD_CATALOG
from .derived import derive_metrics
from .planning import build_analysis_plan
from .metrics import calculate_metrics, DEFAULT_METRIC_DICT
from .analysis import run_multidimensional_analysis
from .visualization import archive_charts
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
    "run_eda_profile",
    "STAT_METHOD_CATALOG",
    "derive_metrics",
    "build_analysis_plan",
    "calculate_metrics",
    "DEFAULT_METRIC_DICT",
    "run_multidimensional_analysis",
    "archive_charts",
    "generate_insights",
    "archive_delivery",
]
