"""多表连接膨胀检查与源表对账（P0）。

对齐进阶课 E21：一对多连接后对头表度量求和会翻倍；
正确做法是在正确粒度汇总，再与源表勾稽。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .contracts import StageResult

OnSpec = Union[str, Sequence[str]]


def _as_list(on: OnSpec) -> List[str]:
    if isinstance(on, str):
        return [on]
    return list(on)


def check_join_inflation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: OnSpec,
    measure_col: str,
    *,
    how: str = "left",
    stage_id: str = "D02_join_inflation",
) -> StageResult:
    """检查一对多连接后，对 ``measure_col``（来自 left）求和是否膨胀。

    Parameters
    ----------
    left, right :
        待连接表；``measure_col`` 必须在 left 中。
    on :
        连接键（str 或列名列表）。
    measure_col :
        头表度量列（连接后若被复制多行再 sum 会翻倍）。
    how :
        pandas merge how。

    Returns
    -------
    StageResult
        ``data`` 含 true_sum / joined_sum / inflation_ratio / n_left / n_joined 等。
        若膨胀倍数 > 1 + 1e-9 → ``warning``；键缺失等 → ``blocked``。
    """
    keys = _as_list(on)
    warnings: List[str] = []
    for k in keys:
        if k not in left.columns or k not in right.columns:
            return StageResult(
                stage_id=stage_id,
                status="blocked",
                block_reasons=[f"join key missing: {k}"],
            )
    if measure_col not in left.columns:
        return StageResult(
            stage_id=stage_id,
            status="blocked",
            block_reasons=[f"measure_col not in left: {measure_col}"],
        )

    true_sum = float(pd.to_numeric(left[measure_col], errors="coerce").sum(skipna=True))
    joined = left.merge(right, on=keys, how=how, suffixes=("", "_r"))
    # 连接后若 measure 被复制，直接 sum 即错误汇总
    joined_sum = float(pd.to_numeric(joined[measure_col], errors="coerce").sum(skipna=True))
    ratio = float(joined_sum / true_sum) if true_sum != 0 else (np.nan if joined_sum == 0 else np.inf)

    inflated = bool(np.isfinite(ratio) and ratio > 1.0 + 1e-9) or (true_sum == 0 and abs(joined_sum) > 1e-9)
    if inflated:
        warnings.append(
            f"join inflation detected: sum({measure_col}) {true_sum} -> {joined_sum} "
            f"(ratio={ratio:.4g}); aggregate at grain of left before summing"
        )

    # 去重后应回到真值
    dedup = joined.drop_duplicates(keys)
    dedup_sum = float(pd.to_numeric(dedup[measure_col], errors="coerce").sum(skipna=True))

    status = "warning" if inflated else "passed"
    return StageResult(
        stage_id=stage_id,
        status=status,
        data={
            "n_left": int(len(left)),
            "n_right": int(len(right)),
            "n_joined": int(len(joined)),
            "true_sum": true_sum,
            "joined_sum": joined_sum,
            "dedup_sum": dedup_sum,
            "inflation_ratio": ratio,
            "inflated": inflated,
            "on": keys,
            "measure_col": measure_col,
        },
        rules=["D02.join_inflation"],
        warnings=warnings,
        evidence_ids=["E21"],
    )


def reconcile_to_source(
    computed: pd.DataFrame,
    source: pd.DataFrame,
    key: OnSpec,
    value_col: str,
    *,
    atol: float = 1e-9,
    stage_id: str = "D02_reconcile",
) -> StageResult:
    """将计算结果与源表按键勾稽 ``value_col``。

    ``computed`` / ``source`` 均需含 key 与 value_col；按 key 聚合 sum 后比对。
    """
    keys = _as_list(key)
    for frame_name, frame in ("computed", computed), ("source", source):
        for k in keys:
            if k not in frame.columns:
                return StageResult(
                    stage_id=stage_id,
                    status="blocked",
                    block_reasons=[f"{frame_name} missing key: {k}"],
                )
        if value_col not in frame.columns:
            return StageResult(
                stage_id=stage_id,
                status="blocked",
                block_reasons=[f"{frame_name} missing value_col: {value_col}"],
            )

    c = (
        computed.groupby(keys, dropna=False)[value_col]
        .sum()
        .rename("computed")
        .reset_index()
    )
    s = (
        source.groupby(keys, dropna=False)[value_col]
        .sum()
        .rename("source")
        .reset_index()
    )
    merged = c.merge(s, on=keys, how="outer")
    merged["computed"] = merged["computed"].fillna(0.0)
    merged["source"] = merged["source"].fillna(0.0)
    merged["diff"] = merged["computed"] - merged["source"]
    merged["balanced"] = merged["diff"].abs() <= atol

    n_bad = int((~merged["balanced"]).sum())
    total_diff = float(merged["diff"].sum())
    warnings: List[str] = []
    if n_bad:
        warnings.append(f"{n_bad} key(s) out of balance; total_diff={total_diff}")

    status = "passed" if n_bad == 0 else "warning"
    return StageResult(
        stage_id=stage_id,
        status=status,
        data={
            "n_keys": int(len(merged)),
            "n_unbalanced": n_bad,
            "total_diff": total_diff,
            "atol": atol,
            "detail": merged,
            "all_balanced": n_bad == 0,
        },
        rules=["D02.reconcile_to_source"],
        warnings=warnings,
        evidence_ids=["E21"],
    )
