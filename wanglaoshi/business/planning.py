"""D05 analysis-decomposition：四大分析方向计划与不适用原因。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .contracts import StageResult

FOUR_DIRECTIONS = ("overall", "structure", "trend", "segment")


def build_analysis_plan(
    df: pd.DataFrame,
    *,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    dim_cols: Optional[Sequence[str]] = None,
    segment_cols: Optional[Sequence[str]] = None,
    goal: str = "sales_overview",
) -> StageResult:
    """根据字段可用性生成四方向任务清单。

    - overall：需要 value_col
    - structure：需要 value_col + 至少一个 dim
    - trend：需要 date_col + value_col
    - segment：需要 value_col + 至少一个 segment
    """
    rules = ["D05.four_directions"]
    warnings: List[str] = []
    dims = [c for c in (dim_cols or []) if c in df.columns]
    segs = [c for c in (segment_cols or []) if c in df.columns]
    if dim_cols:
        missing = [c for c in dim_cols if c not in df.columns]
        if missing:
            warnings.append(f"dim_cols missing: {missing}")
    if segment_cols:
        missing = [c for c in segment_cols if c not in df.columns]
        if missing:
            warnings.append(f"segment_cols missing: {missing}")

    tasks: List[Dict[str, Any]] = []

    def _task(direction: str, applicable: bool, reason: str, requires: List[str]) -> None:
        tasks.append(
            {
                "direction": direction,
                "status": "applicable" if applicable else "not_applicable",
                "reason": reason if not applicable else "fields available",
                "requires": requires,
            }
        )

    has_value = bool(value_col and value_col in df.columns)
    has_date = bool(date_col and date_col in df.columns)

    _task("overall", has_value, "missing value_col", ["value_col"])
    _task("structure", has_value and len(dims) > 0, "need value_col and >=1 dim", ["value_col", "dim_cols"])
    _task("trend", has_value and has_date, "need date_col and value_col", ["date_col", "value_col"])
    _task("segment", has_value and len(segs) > 0, "need value_col and >=1 segment", ["value_col", "segment_cols"])

    applicable = [t for t in tasks if t["status"] == "applicable"]
    not_applicable = [t for t in tasks if t["status"] == "not_applicable"]
    if not applicable:
        warnings.append("no analysis direction applicable; check field roles")
    for t in not_applicable:
        warnings.append(f"{t['direction']} not_applicable: {t['reason']}")

    status = "warning" if warnings else "passed"
    return StageResult(
        stage_id="D05_planning",
        status=status,
        data={
            "goal": goal,
            "tasks": tasks,
            "applicable_count": len(applicable),
            "field_roles": {
                "date_col": date_col,
                "value_col": value_col,
                "dim_cols": dims,
                "segment_cols": segs,
            },
        },
        rules=rules,
        warnings=warnings,
        evidence_ids=[f"plan:{goal}:{len(applicable)}"],
    )
