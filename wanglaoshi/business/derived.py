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
    """对**已按周期聚合**的序列做 shift 环比。

    警告：直接对订单明细行调用会得到「相邻订单变化」，不是日/周环比。
    流水线应先按日聚合再调用；行级变化请用 ``row_pct_change`` 命名。
    """
    work = df.copy()
    work = work.sort_values(date_col)
    prev = work[value_col].shift(periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        work[out_col] = np.where(prev == 0, np.nan, (work[value_col] - prev) / prev)
    return work


def add_value_tiers(
    df: pd.DataFrame,
    value_col: str,
    *,
    method: str = "pareto",
    head: float = 0.8,
    waist: float = 0.95,
    out_col: str = "value_tier",
    abs_value: bool = True,
) -> pd.DataFrame:
    """价值分层：头 / 腰 / 尾。

    - ``pareto``：按金额从大到小累计占比，前 ``head`` 为头，到 ``waist`` 为腰，其余为尾
    - ``quantile``：三分位（高/中/低），忽略 head/waist
    """
    if value_col not in df.columns:
        raise KeyError(f"value_col not found: {value_col}")
    if method not in {"pareto", "quantile"}:
        raise ValueError("method must be 'pareto' or 'quantile'")
    if not (0 < head < waist <= 1):
        raise ValueError("require 0 < head < waist <= 1")

    work = df.copy()
    vals = pd.to_numeric(work[value_col], errors="coerce")
    if abs_value:
        vals = vals.abs()
    work["_tier_val"] = vals.fillna(0.0)

    if method == "quantile":
        # 高→头，中→腰，低→尾
        try:
            cats = pd.qcut(work["_tier_val"], q=3, labels=["尾", "腰", "头"], duplicates="drop")
        except ValueError:
            # 取值过少时退化为常数尾
            cats = pd.Series(["尾"] * len(work), index=work.index)
        work[out_col] = cats.astype(str)
    else:
        order = work["_tier_val"].sort_values(ascending=False)
        total = float(order.sum())
        if total <= 0:
            work[out_col] = "尾"
        else:
            cum = order.cumsum() / total
            tier = pd.Series(index=order.index, dtype=object)
            tier[cum <= head] = "头"
            tier[(cum > head) & (cum <= waist)] = "腰"
            tier[cum > waist] = "尾"
            # 边界：第一笔必为头
            if len(tier):
                tier.iloc[0] = "头"
            work[out_col] = tier.reindex(work.index)

    work = work.drop(columns=["_tier_val"])
    return work


def derive_metrics(
    df: pd.DataFrame,
    *,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    share_group_cols: Optional[Sequence[str]] = None,
    add_wow: bool = False,
    add_tiers: bool = False,
    tier_method: str = "pareto",
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
        # 先按自然日聚合，再算环比；禁止把订单行 shift 冒充 wow
        rules.append("D04.wow_daily")
        period = pd.to_datetime(work[date_col], errors="coerce").dt.floor("D")
        daily = (
            pd.DataFrame({"_period": period, value_col: pd.to_numeric(work[value_col], errors="coerce")})
            .dropna(subset=["_period"])
            .groupby("_period", dropna=False)[value_col]
            .sum()
            .sort_index()
            .reset_index()
        )
        daily = add_period_change(daily, value_col, date_col="_period", periods=1, out_col="wow")
        work = work.copy()
        work["_period"] = period
        work = work.merge(daily[["_period", "wow"]], on="_period", how="left")
        work = work.drop(columns=["_period"])
        created.append("wow")

    if add_tiers and value_col:
        if value_col not in work.columns:
            return StageResult(
                stage_id="D04_derived",
                status="blocked",
                block_reasons=[f"value_col not found: {value_col}"],
            )
        rules.append("D04.value_tiers")
        work = add_value_tiers(work, value_col, method=tier_method)
        created.append("value_tier")
        tier_counts = work["value_tier"].value_counts(dropna=False).to_dict()
    else:
        tier_counts = {}

    if not rules:
        warnings.append("no derive rules applied; pass date_col/value_col/share_group_cols/add_tiers")

    status = "warning" if warnings else "passed"
    result = StageResult(
        stage_id="D04_derived",
        status=status,
        data={
            "created_columns": created,
            "n_rows": int(len(work)),
            "columns": list(work.columns),
            "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        },
        rules=rules,
        warnings=warnings,
    )
    result.frame = work  # type: ignore[attr-defined]
    return result
