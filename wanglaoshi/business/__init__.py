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

__all__ = [
    "StageResult",
    "RunRecord",
    "intake_file",
    "BusinessPipeline",
]
