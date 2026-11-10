"""缺期对齐与同比规则（P0）。

对齐进阶课 E15 增补：先补日历骨架，再算 YoY；
区分缺月、真实 0、零基数、未完月。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .contracts import StageResult


def complete_period_index(
    df: pd.DataFrame,
    time_col: str,
    *,
    value_col: Optional[str] = None,
    freq: str = "MS",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """补齐周期骨架，标记 ``is_missing``（源表无该期记录）。

    - 缺期行：``is_missing=True``，``value_col`` 为 NaN（若提供）
    - 真实 0：源表有记录且值为 0，``is_missing=False``
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col not found: {time_col}")

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col])
    # 归一到 period 起点，避免日级噪音
    work[time_col] = work[time_col].dt.to_period(freq.replace("S", "") if freq.endswith("S") else freq).dt.to_timestamp()

    t0 = start if start is not None else work[time_col].min()
    t1 = end if end is not None else work[time_col].max()
    idx = pd.date_range(t0, t1, freq=freq)
    cal = pd.DataFrame({time_col: idx})

    if value_col and value_col in work.columns:
        agg = work.groupby(time_col, as_index=False)[value_col].sum()
        out = cal.merge(agg, on=time_col, how="left")
        out["is_missing"] = out[value_col].isna()
    else:
        present = set(work[time_col])
        out = cal.copy()
        out["is_missing"] = ~out[time_col].isin(present)
        if value_col:
            out[value_col] = np.nan
    return out


def yoy_with_rules(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    *,
    as_of: Optional[Union[str, pd.Timestamp]] = None,
    freq: str = "MS",
    complete: bool = True,
) -> pd.DataFrame:
    """计算同比，并给出 reason。

    reason 取值：
    - ``ok``
    - ``no_prior_period``
    - ``current_missing_month``
    - ``prior_missing_month``
    - ``zero_base``
    - ``incomplete_month``
    """
    if complete:
        base = complete_period_index(df, time_col, value_col=value_col, freq=freq)
    else:
        base = df.copy()
        base[time_col] = pd.to_datetime(base[time_col])
        if "is_missing" not in base.columns:
            base["is_missing"] = False

    as_of_ts = pd.Timestamp(as_of) if as_of is not None else None
    if as_of_ts is not None:
        # 未完月：as_of 所在月且尚未到月末
        days_in_month = as_of_ts.days_in_month
        base["incomplete"] = (base[time_col].dt.to_period("M") == as_of_ts.to_period("M")) & (
            as_of_ts.day < days_in_month
        )
    else:
        base["incomplete"] = False

    by_time = {row[time_col]: row for _, row in base.iterrows()}
    rows: List[Dict[str, Any]] = []
    for _, row in base.iterrows():
        cur_t = row[time_col]
        cur_v = row[value_col]
        incomplete = bool(row["incomplete"])
        is_missing = bool(row["is_missing"])

        if incomplete:
            rows.append({**row.to_dict(), "yoy": np.nan, "reason": "incomplete_month"})
            continue
        if is_missing or pd.isna(cur_v):
            rows.append({**row.to_dict(), "yoy": np.nan, "reason": "current_missing_month"})
            continue

        prev_t = cur_t - pd.DateOffset(years=1)
        # 对齐到同 freq 起点
        prev_t = pd.Timestamp(prev_t).to_period("M").to_timestamp() if freq.upper().startswith("M") else prev_t
        prev = by_time.get(prev_t)
        if prev is None:
            rows.append({**row.to_dict(), "yoy": np.nan, "reason": "no_prior_period"})
            continue
        if bool(prev["is_missing"]) or pd.isna(prev[value_col]):
            rows.append({**row.to_dict(), "yoy": np.nan, "reason": "prior_missing_month"})
            continue
        prev_v = float(prev[value_col])
        if prev_v == 0:
            rows.append({**row.to_dict(), "yoy": np.nan, "reason": "zero_base"})
            continue
        yoy = (float(cur_v) - prev_v) / prev_v
        rows.append({**row.to_dict(), "yoy": yoy, "reason": "ok"})

    return pd.DataFrame(rows)


def yoy_stage(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    *,
    as_of: Optional[Union[str, pd.Timestamp]] = None,
    freq: str = "MS",
) -> StageResult:
    """包装为 StageResult，便于流水线挂接。"""
    try:
        out = yoy_with_rules(df, time_col, value_col, as_of=as_of, freq=freq)
    except Exception as exc:  # noqa: BLE001
        return StageResult(
            stage_id="D04_yoy",
            status="blocked",
            block_reasons=[f"{type(exc).__name__}: {exc}"],
        )
    reason_counts = out["reason"].value_counts(dropna=False).to_dict()
    warnings: List[str] = []
    for key in ("zero_base", "incomplete_month", "current_missing_month", "prior_missing_month"):
        n = int(reason_counts.get(key, 0))
        if n:
            warnings.append(f"{key}: {n} period(s)")
    status = "passed" if not warnings else "warning"
    return StageResult(
        stage_id="D04_yoy",
        status=status,
        data={"frame": out, "reason_counts": {str(k): int(v) for k, v in reason_counts.items()}},
        rules=["D04.yoy_with_rules"],
        warnings=warnings,
        evidence_ids=["E15"],
    )
