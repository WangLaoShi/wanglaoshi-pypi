"""业务分析总控：按阶段串联，阻断后不生成正式下游结果。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .contracts import RunRecord, StageResult
from .intake import intake_file
from .structuring import structure_file
from .cleaning import clean_dataframe
from .derived import derive_metrics
from .planning import build_analysis_plan
from .metrics import calculate_metrics
from .analysis import run_multidimensional_analysis


_DOWNSTREAM_AFTER = {
    "D01": [
        "D02_structuring",
        "D03_cleaning",
        "D04_derived",
        "D05_planning",
        "D06_metrics",
        "D07_analysis",
        "D09_insights",
        "D10_report_archive",
    ],
    "D02": [
        "D03_cleaning",
        "D04_derived",
        "D05_planning",
        "D06_metrics",
        "D07_analysis",
        "D09_insights",
        "D10_report_archive",
    ],
}
# NOTE: keep D07 in skip lists so blocked upstream does not pretend analysis ran.


def _skip_remaining(record: RunRecord, stage_ids: Sequence[str], reason: str) -> None:
    for stage_id in stage_ids:
        record.add_stage(
            StageResult(
                stage_id=stage_id,
                status="skipped",
                warnings=[reason],
            )
        )


class BusinessPipeline:
    """最小可跑总控（D11）。

    已接线：D01 → D02 → D03 → D04 → D05 → D06 → D07；其后显式 skipped。
    """

    def __init__(
        self,
        archive_dir: str | Path,
        *,
        required_columns: Optional[Sequence[str]] = None,
        period: Optional[str] = None,
        random_seed: int = 42,
        stop_on_block: bool = True,
        date_col: Optional[str] = None,
        value_col: Optional[str] = None,
        dim_cols: Optional[Sequence[str]] = None,
        segment_cols: Optional[Sequence[str]] = None,
        metric_group_cols: Optional[Sequence[str]] = None,
    ) -> None:
        self.archive_dir = Path(archive_dir)
        self.required_columns = list(required_columns) if required_columns else None
        self.period = period
        self.random_seed = random_seed
        self.stop_on_block = stop_on_block
        self.date_col = date_col
        self.value_col = value_col
        self.dim_cols = list(dim_cols) if dim_cols else None
        self.segment_cols = list(segment_cols) if segment_cols else None
        self.metric_group_cols = list(metric_group_cols) if metric_group_cols else None

    def run(self, source_path: str | Path, run_id: Optional[str] = None) -> RunRecord:
        rid = run_id or uuid.uuid4().hex[:12]
        record = RunRecord(
            run_id=rid,
            config={
                "archive_dir": str(self.archive_dir),
                "required_columns": self.required_columns,
                "period": self.period,
                "stop_on_block": self.stop_on_block,
                "date_col": self.date_col,
                "value_col": self.value_col,
                "dim_cols": self.dim_cols,
                "segment_cols": self.segment_cols,
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
            _skip_remaining(record, _DOWNSTREAM_AFTER["D01"], "skipped because upstream blocked")
            return record

        structured_path = self.archive_dir / f"{rid}__structured.csv"
        structuring = structure_file(source_path, structured_path)
        record.add_stage(structuring)
        if structuring.status == "blocked" and self.stop_on_block:
            _skip_remaining(record, _DOWNSTREAM_AFTER["D02"], "skipped because upstream blocked")
            return record

        structured_df = getattr(structuring, "frame", None)
        if structured_df is None and structured_path.exists():
            structured_df = pd.read_csv(structured_path)

        cleaned_path = self.archive_dir / f"{rid}__cleaned.csv"
        log_path = self.archive_dir / f"{rid}__cleaning_log.json"
        cleaning = clean_dataframe(
            structured_df if structured_df is not None else pd.DataFrame(),
            output_path=cleaned_path,
            log_path=log_path,
        )
        record.add_stage(cleaning)

        cleaned_df = getattr(cleaning, "frame", None)
        if cleaned_df is None and cleaned_path.exists():
            cleaned_df = pd.read_csv(cleaned_path)
        if cleaned_df is None:
            cleaned_df = pd.DataFrame()

        date_col = self.date_col or next(
            (
                c
                for c in cleaned_df.columns
                if "date" in c or c.endswith("_dt") or c in {"dt", "day", "timestamp"}
            ),
            None,
        )
        value_col = self.value_col or next(
            (c for c in cleaned_df.columns if c in {"amount", "gmv", "sales", "revenue", "cnt", "value"}),
            None,
        )
        dim_cols = self.dim_cols or [c for c in ["channel", "region", "category"] if c in cleaned_df.columns]
        segment_cols = self.segment_cols or [c for c in ["user_tier", "segment", "vip"] if c in cleaned_df.columns]

        derived = derive_metrics(
            cleaned_df,
            date_col=date_col,
            value_col=value_col,
            share_group_cols=dim_cols[:1] if dim_cols else None,
            add_wow=bool(date_col and value_col),
        )
        record.add_stage(derived)

        derived_df = getattr(derived, "frame", cleaned_df)
        planning = build_analysis_plan(
            derived_df,
            date_col=date_col,
            value_col=value_col,
            dim_cols=dim_cols,
            segment_cols=segment_cols,
            goal="sales_overview",
        )
        record.add_stage(planning)

        group_cols = self.metric_group_cols or dim_cols[:1] or None
        metrics = calculate_metrics(
            derived_df,
            group_cols=group_cols,
            share_of="gmv" if value_col else None,
        )
        record.add_stage(metrics)

        analysis = run_multidimensional_analysis(
            derived_df,
            date_col=date_col,
            value_col=value_col,
            dim_col=dim_cols[0] if dim_cols else None,
            segment_col=segment_cols[0] if segment_cols else None,
        )
        record.add_stage(analysis)

        for stage_id in ["D09_insights", "D10_report_archive"]:
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
