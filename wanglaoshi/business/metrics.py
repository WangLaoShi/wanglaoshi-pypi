"""D06 metrics-calculation：指标字典、汇总底稿、占比一致性校验。

指标不是「随便 sum 一下」：
- 先有 metric_dict（公式 / 聚合方式 / 方向）
- 再产出可核对的 workbook（分组明细）
- 分组占比之和应接近 1（share_check），否则给 warning
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .contracts import StageResult


DEFAULT_METRIC_DICT: Dict[str, Dict[str, Any]] = {
    "gmv": {
        "formula": "sum(amount)",
        "value_col": "amount",
        "agg": "sum",
        "direction": "higher_better",
    },
    "orders": {
        "formula": "count(order_id)",
        "value_col": "order_id",
        "agg": "count",
        "direction": "higher_better",
    },
    "aov": {
        "formula": "gmv / orders",
        "depends_on": ["gmv", "orders"],
        "direction": "higher_better",
    },
}


def _agg_value(df: pd.DataFrame, col: str, how: str) -> float:
    if col not in df.columns:
        return float("nan")
    s = df[col]
    if how == "sum":
        return float(pd.to_numeric(s, errors="coerce").sum())
    if how == "count":
        return float(s.notna().sum())
    if how == "mean":
        return float(pd.to_numeric(s, errors="coerce").mean())
    raise ValueError(f"unsupported agg: {how}")


def calculate_metrics(
    df: pd.DataFrame,
    *,
    metric_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    group_cols: Optional[Sequence[str]] = None,
    share_of: Optional[str] = None,
    tol: float = 1e-6,
) -> StageResult:
    """计算指标并做占比一致性校验（各组 share 之和 ≈ 1）。"""
    rules = ["D06.metric_dict", "D06.workbook"]
    warnings: List[str] = []
    block_reasons: List[str] = []
    mdict = metric_dict or DEFAULT_METRIC_DICT

    base_metrics: Dict[str, float] = {}
    for name, spec in mdict.items():
        if "depends_on" in spec:
            continue
        col = spec.get("value_col")
        how = spec.get("agg", "sum")
        if not col or col not in df.columns:
            warnings.append(f"metric {name}: column {col!r} missing")
            base_metrics[name] = float("nan")
            continue
        base_metrics[name] = _agg_value(df, col, how)

    for name, spec in mdict.items():
        deps = spec.get("depends_on")
        if not deps:
            continue
        if deps == ["gmv", "orders"] and name == "aov":
            gmv, orders = base_metrics.get("gmv"), base_metrics.get("orders")
            if orders in (0, None) or (isinstance(orders, float) and (orders == 0 or np.isnan(orders))):
                base_metrics[name] = float("nan")
                warnings.append("aov undefined because orders=0 or missing")
            else:
                base_metrics[name] = float(gmv) / float(orders)
        else:
            warnings.append(f"metric {name}: unsupported depends_on {deps}")

    workbook_rows: List[Dict[str, Any]] = []
    share_check: Dict[str, Any] = {}
    groups = [c for c in (group_cols or []) if c in df.columns]
    if group_cols and len(groups) != len(list(group_cols)):
        warnings.append(f"some group_cols missing: {set(group_cols) - set(groups)}")

    if groups and share_of and share_of in base_metrics:
        rules.append("D06.share_consistency")
        value_col = None
        for spec in mdict.values():
            if spec.get("value_col") and share_of in ("gmv",) and spec.get("agg") == "sum":
                value_col = spec["value_col"]
                break
        # Prefer explicit amount-like column from gmv spec
        value_col = mdict.get("gmv", {}).get("value_col") or value_col
        if value_col and value_col in df.columns:
            grouped = (
                df.groupby(list(groups), dropna=False)[value_col]
                .sum(min_count=1)
                .reset_index(name="value")
            )
            total = float(grouped["value"].sum())
            if total == 0:
                grouped["share"] = np.nan
                warnings.append("share total is 0; cannot validate")
            else:
                grouped["share"] = grouped["value"] / total
                share_sum = float(grouped["share"].sum())
                ok = abs(share_sum - 1.0) <= tol
                share_check = {"metric": share_of, "share_sum": share_sum, "ok": ok, "tol": tol}
                if not ok:
                    warnings.append(f"share sum {share_sum} != 1 within tol={tol}")
            workbook_rows = grouped.to_dict(orient="records")
        else:
            warnings.append("cannot build grouped workbook: value column missing")

    status = "blocked" if block_reasons else ("warning" if warnings else "passed")
    return StageResult(
        stage_id="D06_metrics",
        status=status,
        data={
            "metrics": base_metrics,
            "metric_dict": mdict,
            "workbook": workbook_rows,
            "share_check": share_check,
            "group_cols": groups,
        },
        rules=rules,
        warnings=warnings,
        block_reasons=block_reasons,
        evidence_ids=[f"metric:{k}" for k in base_metrics],
    )
