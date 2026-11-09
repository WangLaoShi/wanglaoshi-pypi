"""统一阶段结果与运行记录契约。

为什么需要这个文件？
- 业务流水线有很多阶段（接收、清洗、指标、报告…）
- 如果每个阶段返回格式都不一样，后面很难统一校验 / 归档
- 所以约定：每个阶段都返回 StageResult；整次运行记在 RunRecord 里

状态含义（记住这 4 个就够用）：
- passed  : 成功
- warning : 能继续，但有需要关注的问题（如缺列）
- blocked : 阻断，正式下游不应继续当作可交付
- skipped : 因上游阻断或尚未实现而跳过（显式跳过，不是偷偷成功）
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json
from pathlib import Path


# 允许的状态集合；写错拼写会在 StageResult 初始化时直接报错
ALLOWED_STATUS = {"passed", "warning", "blocked", "skipped"}


@dataclass
class StageResult:
    """单个业务阶段的结构化结果。

    字段怎么理解：
    - stage_id     : 阶段编号，如 D01_intake / D06_metrics
    - status       : 上面 4 种状态之一
    - data         : 本阶段算出的业务数据（指标、计划、图表清单等）
    - rules        : 用了哪些规则编号（方便追溯「按什么标准判断的」）
    - evidence_ids : 证据编号（报告/结论必须能指回这里）
    - warnings     : 警告文案（人不看 data 也能知道发生了什么）
    - block_reasons: 阻断原因（只有 blocked 时通常非空）
    - input_paths / output_paths : 输入输出文件路径，方便归档核对
    """

    stage_id: str
    status: str  # passed | warning | blocked | skipped
    data: Dict[str, Any] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    block_reasons: List[str] = field(default_factory=list)
    input_paths: List[str] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 初始化后立刻校验，避免错误状态悄悄传到下游
        if self.status not in ALLOWED_STATUS:
            raise ValueError(f"invalid status {self.status!r}; expected one of {sorted(ALLOWED_STATUS)}")

    @property
    def ok(self) -> bool:
        """「还能不能往下走」的宽松判断。

        注意：skipped 也算 ok（表示明确跳过，不是崩溃）；
        真正不能交付时看 RunRecord.blocked() 或 status=='blocked'。
        """
        return self.status in {"passed", "warning", "skipped"}

    def to_dict(self) -> Dict[str, Any]:
        """转成普通 dict，便于 JSON 序列化或写日志。"""
        return asdict(self)

    def to_json(self, path: Optional[str | Path] = None, indent: int = 2) -> str:
        """导出 JSON 字符串；若给了 path 则同时写到文件。"""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


@dataclass
class RunRecord:
    """一次流水线运行的可追溯记录。

    可以把 RunRecord 想成「这次分析的病历本」：
    - 用了什么配置、随机种子
    - 输入文件哈希（证明原始数据没被偷偷改）
    - 每个阶段的 StageResult 按顺序排列
    """

    run_id: str
    input_hashes: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    dependency_versions: Dict[str, str] = field(default_factory=dict)
    random_seed: Optional[int] = None
    stages: List[StageResult] = field(default_factory=list)

    def add_stage(self, stage: StageResult) -> None:
        """按执行顺序追加一个阶段结果。"""
        self.stages.append(stage)

    def blocked(self) -> bool:
        """整次运行是否出现过阻断阶段。

        规则：任一阶段 blocked，则正式交付应失败或标记不可交付。
        """
        return any(s.status == "blocked" for s in self.stages)

    def summary(self) -> Dict[str, Any]:
        """给人看的短摘要（不含大表 data），适合 print / 日志。"""
        return {
            "run_id": self.run_id,
            "blocked": self.blocked(),
            "stages": [
                {
                    "stage_id": s.stage_id,
                    "status": s.status,
                    "warnings": s.warnings,
                    "block_reasons": s.block_reasons,
                }
                for s in self.stages
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """完整结构化结果（含各阶段 data），用于归档。"""
        return asdict(self)
