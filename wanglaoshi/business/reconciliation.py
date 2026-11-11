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

    除净额比对外，还会统计每条左表记录被匹配次数；正负金额抵消时
    仍能检出行级膨胀。
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
    left_tmp = left.copy()
    left_tmp["_left_row_id"] = np.arange(len(left_tmp))
    joined = left_tmp.merge(right, on=keys, how=how, suffixes=("", "_r"))
    joined_sum = float(pd.to_numeric(joined[measure_col], errors="coerce").sum(skipna=True))
    ratio = float(joined_sum / true_sum) if true_sum != 0 else (np.nan if joined_sum == 0 else np.inf)

    match_counts = joined.groupby("_left_row_id", sort=False).size()
    # left join 无匹配时仍有 1 行（右表空）；一对多时 size>1
    # inner join 无匹配则该左行消失 → 用 reindex 对齐
    if how == "left":
        counts = match_counts.reindex(left_tmp["_left_row_id"], fill_value=0)
        # pandas left merge 无匹配也会保留左行，size 至少为 1；若右键全空仍为 1
        n_multi_matched = int((counts > 1).sum())
        max_match_count = int(counts.max()) if len(counts) else 0
    else:
        n_multi_matched = int((match_counts > 1).sum())
        max_match_count = int(match_counts.max()) if len(match_counts) else 0

    sum_inflated = bool(np.isfinite(ratio) and ratio > 1.0 + 1e-9) or (
        true_sum == 0 and abs(joined_sum) > 1e-9
    )
    row_inflated = n_multi_matched > 0
    inflated = bool(sum_inflated or row_inflated)

    if row_inflated:
        warnings.append(
            f"join row inflation: {n_multi_matched} left row(s) matched more than once "
            f"(max_match_count={max_match_count}); n_left={len(left)} -> n_joined={len(joined)}"
        )
    if sum_inflated:
        warnings.append(
            f"join sum inflation: sum({measure_col}) {true_sum} -> {joined_sum} "
            f"(ratio={ratio:.4g}); aggregate at grain of left before summing"
        )

    dedup = joined.drop_duplicates("_left_row_id")
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
            "net_sum_diff": float(joined_sum - true_sum),
            "inflation_ratio": ratio,
            "inflated": inflated,
            "row_inflated": row_inflated,
            "n_multi_matched_left_rows": n_multi_matched,
            "max_match_count": max_match_count,
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
    strict: bool = True,
    stage_id: str = "D02_reconcile",
) -> StageResult:
    """将计算结果与源表按键勾稽 ``value_col``。

    Parameters
    ----------
    strict :
        ``True``（默认）：缺键 / 多键一律视为不平衡，不允许用 0 填缺失金额后「假平衡」。
        ``False``：仅比较双方都存在的键上的金额差；缺键单独计入 ``n_missing_keys`` 警告。
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
        .sum(min_count=1)
        .rename("computed")
        .reset_index()
    )
    s = (
        source.groupby(keys, dropna=False)[value_col]
        .sum(min_count=1)
        .rename("source")
        .reset_index()
    )
    merged = c.merge(s, on=keys, how="outer", indicator=True)
    status_map = {
        "both": "matched",
        "left_only": "extra_in_computed",
        "right_only": "missing_in_computed",
    }
    merged["key_status"] = merged["_merge"].map(status_map)
    both = merged["_merge"] == "both"
    merged["diff"] = np.where(
        both,
        pd.to_numeric(merged["computed"], errors="coerce")
        - pd.to_numeric(merged["source"], errors="coerce"),
        np.nan,
    )
    amount_ok = both & merged["diff"].notna() & (merged["diff"].abs() <= atol)
    if strict:
        # 缺键/多键不能靠补零假装平衡
        merged["balanced"] = False
        merged.loc[amount_ok, "balanced"] = True
    else:
        merged["balanced"] = False
        merged.loc[amount_ok, "balanced"] = True
        # 非严格：缺键不计入 unbalanced 金额，但单独报告
        # balanced 列对缺键仍为 False；all_balanced 只看 matched

    n_extra = int((merged["key_status"] == "extra_in_computed").sum())
    n_missing = int((merged["key_status"] == "missing_in_computed").sum())
    n_amount_bad = int((both & ~amount_ok).sum())
    if strict:
        # 缺键/多键 + 金额差均计入不平衡
        n_unbalanced = int((~merged["balanced"]).sum())
        all_balanced = n_unbalanced == 0
        status = "passed" if all_balanced else "warning"
    else:
        # 非严格：all_balanced 只看已匹配键的金额；缺键单独警告
        n_unbalanced = n_amount_bad
        all_balanced = n_amount_bad == 0
        status = "warning" if (n_amount_bad or n_missing or n_extra) else "passed"

    total_diff = float(merged.loc[both, "diff"].fillna(0.0).sum()) if both.any() else 0.0
    warnings: List[str] = []
    if n_missing:
        warnings.append(f"{n_missing} key(s) missing in computed")
    if n_extra:
        warnings.append(f"{n_extra} key(s) extra in computed (not in source)")
    if n_amount_bad:
        warnings.append(f"{n_amount_bad} matched key(s) amount mismatch; total_diff={total_diff}")
    if strict and (n_missing or n_extra):
        warnings.append("strict reconcile: missing/extra keys are not allowed")
    return StageResult(
        stage_id=stage_id,
        status=status,
        data={
            "n_keys": int(len(merged)),
            "n_unbalanced": n_unbalanced,
            "n_missing_in_computed": n_missing,
            "n_extra_in_computed": n_extra,
            "n_amount_mismatch": n_amount_bad,
            "total_diff": total_diff,
            "atol": atol,
            "strict": strict,
            "detail": merged.drop(columns=["_merge"]),
            "all_balanced": bool(all_balanced),
        },
        rules=["D02.reconcile_to_source"],
        warnings=warnings,
        evidence_ids=["E21"],
    )
