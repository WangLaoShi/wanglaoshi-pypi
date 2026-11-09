"""D07 multidimensional-analysis：趋势/结构/对比/相关/二八/异动（MVP）。

六个视角帮助「从不同角度看同一张表」。
特别提醒：二八/贡献分解给出的是数值贡献，**不证明业务因果**。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .contracts import StageResult


def _trend(df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
    work = df[[date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()
    if work.empty:
        return {"status": "skipped", "reason": "no valid date/value rows"}
    daily = work.groupby(work[date_col].dt.date)[value_col].sum().sort_index()
    return {
        "status": "ok",
        "n_periods": int(len(daily)),
        "first": float(daily.iloc[0]),
        "last": float(daily.iloc[-1]),
        "change_pct": float((daily.iloc[-1] - daily.iloc[0]) / daily.iloc[0]) if daily.iloc[0] else None,
        "series_tail": {str(k): float(v) for k, v in daily.tail(7).items()},
    }


def _structure(df: pd.DataFrame, dim_col: str, value_col: str, top_n: int = 10) -> Dict[str, Any]:
    work = df[[dim_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    g = work.groupby(dim_col, dropna=False)[value_col].sum().sort_values(ascending=False)
    total = float(g.sum())
    shares = (g / total) if total else g * np.nan
    top = [
        {"key": str(k), "value": float(v), "share": float(shares.loc[k]) if total else None}
        for k, v in g.head(top_n).items()
    ]
    return {"status": "ok", "dim": dim_col, "total": total, "top": top}


def _contrast(df: pd.DataFrame, seg_col: str, value_col: str) -> Dict[str, Any]:
    work = df[[seg_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    g = work.groupby(seg_col, dropna=False)[value_col].agg(["sum", "mean", "count"])
    rows = [
        {"segment": str(idx), "sum": float(r["sum"]), "mean": float(r["mean"]), "count": int(r["count"])}
        for idx, r in g.iterrows()
    ]
    return {"status": "ok", "segment_col": seg_col, "rows": rows}


def _correlation(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Any]:
    use = [c for c in cols if c in df.columns]
    if len(use) < 2:
        return {"status": "skipped", "reason": "need >=2 numeric columns"}
    num = df[use].apply(pd.to_numeric, errors="coerce")
    corr = num.corr()
    return {
        "status": "ok",
        "columns": use,
        "matrix": corr.round(4).fillna(0).to_dict(),
    }


def _pareto(df: pd.DataFrame, dim_col: str, value_col: str, threshold: float = 0.8) -> Dict[str, Any]:
    work = df[[dim_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    g = work.groupby(dim_col, dropna=False)[value_col].sum().sort_values(ascending=False)
    total = float(g.sum())
    if total == 0:
        return {"status": "skipped", "reason": "total is 0"}
    cum = g.cumsum() / total
    head = cum[cum <= threshold]
    # include first item that crosses threshold
    if len(head) < len(cum):
        head = cum.iloc[: len(head) + 1]
    return {
        "status": "ok",
        "threshold": threshold,
        "n_head": int(len(head)),
        "n_total": int(len(g)),
        "head_share": float(head.iloc[-1]) if len(head) else None,
        "head_keys": [str(k) for k in head.index.tolist()],
        "note": "数值贡献分解，不证明业务因果",
    }


def _anomaly(df: pd.DataFrame, date_col: str, value_col: str, z_thresh: float = 2.5) -> Dict[str, Any]:
    work = df[[date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()
    if len(work) < 3:
        return {"status": "skipped", "reason": "need >=3 points"}
    daily = work.groupby(work[date_col].dt.date)[value_col].sum().sort_index()
    mu, sigma = float(daily.mean()), float(daily.std(ddof=0))
    if sigma == 0:
        return {"status": "skipped", "reason": "zero variance"}
    z = (daily - mu) / sigma
    flags = z[abs(z) >= z_thresh]
    return {
        "status": "ok",
        "z_thresh": z_thresh,
        "mean": mu,
        "std": sigma,
        "anomalies": [{"date": str(k), "value": float(daily.loc[k]), "z": float(z.loc[k])} for k in flags.index],
    }


def run_multidimensional_analysis(
    df: pd.DataFrame,
    *,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    dim_col: Optional[str] = None,
    segment_col: Optional[str] = None,
    corr_cols: Optional[Sequence[str]] = None,
    top_n: int = 10,
) -> StageResult:
    """六视角：趋势、结构、对比、相关、二八、异动。"""
    rules = ["D07.six_views"]
    warnings: List[str] = []
    views: Dict[str, Any] = {}

    if date_col and value_col and date_col in df.columns and value_col in df.columns:
        views["trend"] = _trend(df, date_col, value_col)
        views["anomaly"] = _anomaly(df, date_col, value_col)
    else:
        views["trend"] = {"status": "skipped", "reason": "need date_col and value_col"}
        views["anomaly"] = {"status": "skipped", "reason": "need date_col and value_col"}

    if dim_col and value_col and dim_col in df.columns and value_col in df.columns:
        views["structure"] = _structure(df, dim_col, value_col, top_n=top_n)
        views["pareto"] = _pareto(df, dim_col, value_col)
    else:
        views["structure"] = {"status": "skipped", "reason": "need dim_col and value_col"}
        views["pareto"] = {"status": "skipped", "reason": "need dim_col and value_col"}

    if segment_col and value_col and segment_col in df.columns and value_col in df.columns:
        views["contrast"] = _contrast(df, segment_col, value_col)
    else:
        views["contrast"] = {"status": "skipped", "reason": "need segment_col and value_col"}

    if corr_cols:
        views["correlation"] = _correlation(df, corr_cols)
    elif value_col:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        use = [value_col] + [c for c in numeric if c != value_col][:3]
        views["correlation"] = _correlation(df, use)
    else:
        views["correlation"] = {"status": "skipped", "reason": "no columns for correlation"}

    for name, payload in views.items():
        if payload.get("status") == "skipped":
            warnings.append(f"{name}: {payload.get('reason')}")

    ok_count = sum(1 for v in views.values() if v.get("status") == "ok")
    if ok_count == 0:
        status = "warning"
        warnings.append("no analysis view produced results")
    else:
        status = "warning" if warnings else "passed"

    return StageResult(
        stage_id="D07_analysis",
        status=status,
        data={"views": views, "ok_count": ok_count},
        rules=rules,
        warnings=warnings,
        evidence_ids=[f"view:{k}" for k, v in views.items() if v.get("status") == "ok"],
    )
