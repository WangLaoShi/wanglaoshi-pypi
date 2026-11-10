"""业务分析流水线（Skill 固化入口）。

主链：
  intake → structuring → cleaning → EDA → derived → planning → metrics
  → analysis → visualization → insights → archive
  （总控：BusinessPipeline）

怎么选入口？
- 只想「收一份 CSV 入库」→ intake_file
- 想跑完整销售分析主链 → BusinessPipeline(...).run(path)
- 想单独算指标 / 出结论 → calculate_metrics / generate_insights

状态约定见 contracts.StageResult（passed/warning/blocked/skipped）。
"""

from .contracts import StageResult, RunRecord
from .intake import intake_file
from .pipeline import BusinessPipeline
from .structuring import structure_dataframe, structure_file
from .cleaning import clean_dataframe
from .eda import run_eda_profile, STAT_METHOD_CATALOG
from .derived import derive_metrics, add_value_tiers
from .planning import build_analysis_plan
from .metrics import calculate_metrics, DEFAULT_METRIC_DICT
from .analysis import run_multidimensional_analysis
from .visualization import archive_charts
from .insights import generate_insights
from .archive import archive_delivery
from .reconciliation import check_join_inflation, reconcile_to_source
from .calendar import complete_period_index, yoy_with_rules, yoy_stage
from .units import convert_units, DEFAULT_FACTORS

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
    "add_value_tiers",
    "build_analysis_plan",
    "calculate_metrics",
    "DEFAULT_METRIC_DICT",
    "run_multidimensional_analysis",
    "archive_charts",
    "generate_insights",
    "archive_delivery",
    "check_join_inflation",
    "reconcile_to_source",
    "complete_period_index",
    "yoy_with_rules",
    "yoy_stage",
    "convert_units",
    "DEFAULT_FACTORS",
]
