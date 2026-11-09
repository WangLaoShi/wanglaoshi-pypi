"""D04 derived-metrics：时间维度、环比同比、占比（MVP）。

衍生字段是「从已有列算出来的新列」，例如：
- 从 order_date 拆 year/month/quarter
- 按日算金额占比 share
- 简单 WoW（周环比/期环比，缺期处理仍属后续增强）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .contracts import StageResult


def add_time_parts(df: pd.DataFrame, date_col: str, prefix: Optional[str] = None) -> pd.DataFrame:
    work = df.copy()
    p = prefix or date_col
    dt = pd.to_datetime(work[date_col], errors="coerce")
    work[f"{p}_year"] = dt.dt.year
    work[f"{p}_month"] = dt.dt.month
    work[f"{p}_quarter"] = dt.dt.quarter
    work[f"{p}_week"] = dt.dt.isocalendar().week.astype("Int64")
    work[f"{p}_dow"] = dt.dt.dayofweek
    return work


def add_share(df: pd.DataFrame, value_col: str, group_cols: Sequence[str], out_col: str = "share") -> pd.DataFrame:
    work = df.copy()
    totals = work.groupby(list(group_cols), dropna=False)[value_col].transform("sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        work[out_col] = np.where(totals == 0, np.nan, work[value_col] / totals)
    return work


def add_period_change(
    df: pd.DataFrame,
    value_col: str,
    *,
    date_col: str,
    periods: int = 1,
    out_col: str = "wow",
) -> pd.DataFrame:
    """按日期排序后做简单 shift 环比（单序列）；分组场景请先聚合。"""
    work = df.copy()
    work = work.sort_values(date_col)
    prev = work[value_col].shift(periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        work[out_col] = np.where(prev == 0, np.nan, (work[value_col] - prev) / prev)
    return work


def derive_metrics(
    df: pd.DataFrame,
    *,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    share_group_cols: Optional[Sequence[str]] = None,
    add_wow: bool = False,
) -> StageResult:
    warnings: List[str] = []
    rules: List[str] = []
    work = df.copy()
    created: List[str] = []

    if date_col:
        if date_col not in work.columns:
            return StageResult(
                stage_id="D04_derived",
                status="blocked",
                block_reasons=[f"date_col not found: {date_col}"],
            )
        rules.append("D04.time_parts")
        before = set(work.columns)
        work = add_time_parts(work, date_col)
        created.extend(sorted(set(work.columns) - before))

    if value_col and share_group_cols:
        if value_col not in work.columns:
            return StageResult(
                stage_id="D04_derived",
                status="blocked",
                block_reasons=[f"value_col not found: {value_col}"],
            )
        missing = [c for c in share_group_cols if c not in work.columns]
        if missing:
            warnings.append(f"share group cols missing: {missing}")
        else:
            rules.append("D04.share")
            work = add_share(work, value_col, share_group_cols, out_col="share")
            created.append("share")

    if add_wow and date_col and value_col and value_col in work.columns:
        rules.append("D04.wow")
        work = add_period_change(work, value_col, date_col=date_col, periods=1, out_col="wow")
        created.append("wow")

    if not rules:
        warnings.append("no derive rules applied; pass date_col/value_col/share_group_cols")

    status = "warning" if warnings else "passed"
    result = StageResult(
        stage_id="D04_derived",
        status=status,
        data={"created_columns": created, "n_rows": int(len(work)), "columns": list(work.columns)},
        rules=rules,
        warnings=warnings,
    )
    result.frame = work  # type: ignore[attr-defined]
    return result
