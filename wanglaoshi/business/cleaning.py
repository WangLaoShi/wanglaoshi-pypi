"""D03 data-cleaning：可配置清洗，输出新文件与逐规则影响日志。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .contracts import StageResult


def clean_dataframe(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    subset: Optional[Sequence[str]] = None,
    fillna: Optional[Dict[str, Any]] = None,
    dropna_columns: Optional[Sequence[str]] = None,
    clip_bounds: Optional[Dict[str, Sequence[float]]] = None,
    output_path: Optional[str | Path] = None,
    log_path: Optional[str | Path] = None,
) -> StageResult:
    """按规则清洗；默认不因统计异常直接删行（clip 只截断）。"""
    rules: List[str] = []
    warnings: List[str] = []
    impacts: List[Dict[str, Any]] = []
    work = df.copy()
    n0 = len(work)

    if drop_duplicates:
        rules.append("D03.drop_duplicates")
        before = len(work)
        work = work.drop_duplicates(subset=list(subset) if subset else None)
        impacts.append({"rule": "drop_duplicates", "rows_before": before, "rows_after": len(work), "rows_removed": before - len(work)})

    if dropna_columns:
        rules.append("D03.dropna_columns")
        cols = [c for c in dropna_columns if c in work.columns]
        before = len(work)
        work = work.dropna(subset=cols)
        impacts.append({"rule": "dropna_columns", "columns": cols, "rows_before": before, "rows_after": len(work), "rows_removed": before - len(work)})

    if fillna:
        rules.append("D03.fillna")
        applied = {}
        for col, val in fillna.items():
            if col not in work.columns:
                warnings.append(f"fillna skip missing column: {col}")
                continue
            na_before = int(work[col].isna().sum())
            work[col] = work[col].fillna(val)
            applied[col] = {"value": val, "filled": na_before}
        impacts.append({"rule": "fillna", "applied": applied})

    if clip_bounds:
        rules.append("D03.clip_bounds")
        clipped = {}
        for col, bounds in clip_bounds.items():
            if col not in work.columns:
                warnings.append(f"clip skip missing column: {col}")
                continue
            lo, hi = bounds[0], bounds[1]
            s = pd.to_numeric(work[col], errors="coerce")
            before_out = int(((s < lo) | (s > hi)).sum())
            work[col] = s.clip(lo, hi)
            clipped[col] = {"low": lo, "high": hi, "n_clipped": before_out}
        impacts.append({"rule": "clip_bounds", "clipped": clipped})

    outputs: List[str] = []
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        work.to_csv(out, index=False)
        outputs.append(str(out))

    if log_path is not None:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(impacts, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(str(lp))

    status = "warning" if warnings else "passed"
    result = StageResult(
        stage_id="D03_cleaning",
        status=status,
        data={
            "rows_in": n0,
            "rows_out": int(len(work)),
            "impacts": impacts,
            "columns": list(work.columns),
        },
        rules=rules,
        warnings=warnings,
        output_paths=outputs,
    )
    result.frame = work  # type: ignore[attr-defined]
    return result
