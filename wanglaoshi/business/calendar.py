"""缺期对齐与同比规则（P0）。

对齐进阶课 E15 增补：先补日历骨架，再算 YoY；
区分缺月、金额缺失、真实 0、零基数、未完月。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .contracts import StageResult


def _period_code(freq: str) -> str:
    """将 date_range 风格 freq 归一为 Period 频率码（聚合与补期共用）。"""
    f = str(freq).upper()
    # 起点锚点：MS/QS/YS → 对应周期码 M/Q/Y
    if f.endswith("S") and len(f) >= 2:
        f = f[:-1]
    aliases = {
        "ME": "M",
        "QE": "Q",
        "YE": "Y",
        "A": "Y",
        "Y": "Y",
        "M": "M",
        "Q": "Q",
        "D": "D",
        "W": "W",
        "H": "H",
    }
    return aliases.get(f, f)


def _to_period(series: pd.Series, freq: str) -> pd.Series:
    return pd.to_datetime(series).dt.to_period(_period_code(freq))


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
    - 有记录但金额全空：``is_missing=False``，``value_col`` 为 NaN（与真实 0 区分）
    - 真实 0：源表有记录且汇总值 0，``is_missing=False``

    聚合与补期共用同一 Period 键，避免 ``freq="M"`` 时月初/月末错位。
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col not found: {time_col}")

    work = df.copy()
    pcode = _period_code(freq)
    work["_period"] = _to_period(work[time_col], freq)

    if start is not None:
        p0 = pd.Timestamp(start).to_period(pcode)
    else:
        p0 = work["_period"].min()
    if end is not None:
        p1 = pd.Timestamp(end).to_period(pcode)
    else:
        p1 = work["_period"].max()

    full = pd.period_range(p0, p1, freq=pcode)
    cal = pd.DataFrame({"_period": full})
    present = set(work["_period"].dropna().tolist())

    if value_col and value_col in work.columns:

        def _sum_keep_all_nan(s: pd.Series) -> float:
            num = pd.to_numeric(s, errors="coerce")
            if num.isna().all():
                return np.nan
            return float(num.sum(skipna=True))

        agg = (
            work.groupby("_period", as_index=False)[value_col]
            .agg(_sum_keep_all_nan)
        )
        out = cal.merge(agg, on="_period", how="left")
        out["is_missing"] = ~out["_period"].isin(present)
        # 缺期强制 NaN（防止异常合并污染）
        out.loc[out["is_missing"], value_col] = np.nan
    else:
        out = cal.copy()
        out["is_missing"] = ~out["_period"].isin(present)
        if value_col:
            out[value_col] = np.nan

    # 对外时间列统一为周期起点时间戳（与 Period 键一一对应）
    out[time_col] = out["_period"].dt.to_timestamp(how="start")
    return out.drop(columns=["_period"])


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
    - ``current_missing_month``  （源表无该期记录）
    - ``current_amount_missing`` （有记录但金额全空）
    - ``prior_missing_month``
    - ``prior_amount_missing``
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
        days_in_month = as_of_ts.days_in_month
        base["incomplete"] = (base[time_col].dt.to_period("M") == as_of_ts.to_period("M")) & (
            as_of_ts.day < days_in_month
        )
    else:
        base["incomplete"] = False

    by_time = {row[time_col]: row for _, row in base.iterrows()}
    rows: List[Dict[str, Any]] = []
    pcode = _period_code(freq)
    for _, row in base.iterrows():
        cur_t = row[time_col]
        cur_v = row[value_col]
        incomplete = bool(row["incomplete"])
        is_missing = bool(row["is_missing"])

        if incomplete:
            rows.append({**row.to_dict(), "yoy": np.nan, "reason": "incomplete_month"})
            continue
        if is_missing:
            rows.append({**row.to_dict(), "yoy": np.nan, "reason": "current_missing_month"})
            continue
        if pd.isna(cur_v):
            rows.append({**row.to_dict(), "yoy": np.nan, "reason": "current_amount_missing"})
            continue

        # 上年同期：按同一 Period 码对齐
        cur_p = pd.Timestamp(cur_t).to_period(pcode)
        if pcode == "M":
            prev_period = cur_p - 12
        elif pcode == "Q":
            prev_period = cur_p - 4
        elif pcode in {"Y", "A"}:
            prev_period = cur_p - 1
        else:
            prev_period = (pd.Timestamp(cur_t) - pd.DateOffset(years=1)).to_period(pcode)
        prev_t = prev_period.to_timestamp(how="start")
        prev = by_time.get(prev_t)
        if prev is None:
            rows.append({**row.to_dict(), "yoy": np.nan, "reason": "no_prior_period"})
            continue
        if bool(prev["is_missing"]):
            rows.append({**row.to_dict(), "yoy": np.nan, "reason": "prior_missing_month"})
            continue
        if pd.isna(prev[value_col]):
            rows.append({**row.to_dict(), "yoy": np.nan, "reason": "prior_amount_missing"})
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
    for key in (
        "zero_base",
        "incomplete_month",
        "current_missing_month",
        "current_amount_missing",
        "prior_missing_month",
        "prior_amount_missing",
    ):
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
