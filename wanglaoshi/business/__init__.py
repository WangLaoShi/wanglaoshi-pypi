"""业务分析流水线（Skill 固化入口）。

首版提供：
- 统一阶段结果 StageResult
- 数据接收 intake
- 总控 pipeline（可按阶段开关运行）

设计原则见仓库根目录 DEVELOPMENT_ROADMAP.md（工作流 D）。
"""

from .contracts import StageResult, RunRecord
from .intake import intake_file
from .pipeline import BusinessPipeline
from .structuring import structure_dataframe, structure_file
from .cleaning import clean_dataframe
from .derived import derive_metrics

__all__ = [
    "StageResult",
    "RunRecord",
    "intake_file",
    "BusinessPipeline",
    "structure_dataframe",
    "structure_file",
    "clean_dataframe",
    "derive_metrics",
]
