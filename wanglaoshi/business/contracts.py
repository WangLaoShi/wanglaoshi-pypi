"""统一阶段结果与运行记录契约。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json
from pathlib import Path


ALLOWED_STATUS = {"passed", "warning", "blocked", "skipped"}


@dataclass
class StageResult:
    """单个业务阶段的结构化结果。"""

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
        if self.status not in ALLOWED_STATUS:
            raise ValueError(f"invalid status {self.status!r}; expected one of {sorted(ALLOWED_STATUS)}")

    @property
    def ok(self) -> bool:
        return self.status in {"passed", "warning", "skipped"}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Optional[str | Path] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


@dataclass
class RunRecord:
    """一次流水线运行的可追溯记录。"""

    run_id: str
    input_hashes: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    dependency_versions: Dict[str, str] = field(default_factory=dict)
    random_seed: Optional[int] = None
    stages: List[StageResult] = field(default_factory=list)

    def add_stage(self, stage: StageResult) -> None:
        self.stages.append(stage)

    def blocked(self) -> bool:
        return any(s.status == "blocked" for s in self.stages)

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "blocked": self.blocked(),
            "stages": [
                {"stage_id": s.stage_id, "status": s.status, "warnings": s.warnings, "block_reasons": s.block_reasons}
                for s in self.stages
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
