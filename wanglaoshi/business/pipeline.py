"""业务分析总控：按阶段串联，阻断后不生成正式下游结果。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .contracts import RunRecord, StageResult
from .intake import intake_file


class BusinessPipeline:
    """最小可跑总控（D11 雏形）。

    当前已接线：
    - D01 intake

    后续阶段（structuring/cleaning/...）以 skipped 记录占位，直到实现。
    """

    def __init__(
        self,
        archive_dir: str | Path,
        *,
        required_columns: Optional[Sequence[str]] = None,
        period: Optional[str] = None,
        random_seed: int = 42,
        stop_on_block: bool = True,
    ) -> None:
        self.archive_dir = Path(archive_dir)
        self.required_columns = list(required_columns) if required_columns else None
        self.period = period
        self.random_seed = random_seed
        self.stop_on_block = stop_on_block

    def run(self, source_path: str | Path, run_id: Optional[str] = None) -> RunRecord:
        rid = run_id or uuid.uuid4().hex[:12]
        record = RunRecord(
            run_id=rid,
            config={
                "archive_dir": str(self.archive_dir),
                "required_columns": self.required_columns,
                "period": self.period,
                "stop_on_block": self.stop_on_block,
            },
            random_seed=self.random_seed,
        )

        intake = intake_file(
            source_path,
            self.archive_dir,
            required_columns=self.required_columns,
            period=self.period,
        )
        record.add_stage(intake)
        if intake.data.get("sha256"):
            record.input_hashes[str(source_path)] = intake.data["sha256"]

        if intake.status == "blocked" and self.stop_on_block:
            record.add_stage(
                StageResult(
                    stage_id="D02_structuring",
                    status="skipped",
                    warnings=["skipped because upstream blocked"],
                )
            )
            return record

        # 占位：后续 Skill 未实现时显式 skipped，避免静默当作完成
        for stage_id in [
            "D02_structuring",
            "D03_cleaning",
            "D04_derived",
            "D06_metrics",
            "D07_analysis",
            "D09_insights",
            "D10_report_archive",
        ]:
            record.add_stage(
                StageResult(
                    stage_id=stage_id,
                    status="skipped",
                    warnings=["not implemented in MVP; placeholder"],
                )
            )

        return record

    def run_to_dict(self, source_path: str | Path, **kwargs: Any) -> Dict[str, Any]:
        return self.run(source_path, **kwargs).summary()
