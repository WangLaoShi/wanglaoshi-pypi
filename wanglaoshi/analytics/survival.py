"""生存分析最小工具（P1 extras 级，无 lifelines 硬依赖）。对齐 E16。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _validate_km_inputs(times: Sequence[float], events: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    times_a = np.asarray(times, dtype=float).reshape(-1)
    events_raw = np.asarray(events).reshape(-1)
    if times_a.size == 0 or events_raw.size == 0:
        raise ValueError("times and events must be non-empty")
    if times_a.shape[0] != events_raw.shape[0]:
        raise ValueError("times and events length mismatch")
    if not np.isfinite(times_a).all():
        raise ValueError("times must be finite")
    if np.any(times_a < 0):
        raise ValueError("times must be non-negative")
    # 事件标记必须在类型收窄前校验为 {0,1}
    events_f = pd.to_numeric(pd.Series(events_raw), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(events_f).all():
        raise ValueError("events must be finite 0/1 labels")
    uniq = set(np.unique(events_f).tolist())
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError("events must be binary labels in {0, 1}")
    return times_a, events_f.astype(int)


def kaplan_meier(
    times: Sequence[float],
    events: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """手写 KM：返回 (t_axis, survival)。"""
    times_a, events_a = _validate_km_inputs(times, events)
    order = np.argsort(times_a)
    times_a, events_a = times_a[order], events_a[order]
    uniq = np.unique(times_a)
    surv, t_axis = [1.0], [0.0]
    at_risk = len(times_a)
    s = 1.0
    for t in uniq:
        died = int(events_a[times_a == t].sum())
        if at_risk > 0 and died > 0:
            s *= 1.0 - died / at_risk
        surv.append(s)
        t_axis.append(float(t))
        at_risk -= int(np.sum(times_a == t))
    return np.asarray(t_axis), np.asarray(surv)


class SurvivalAnalyzer:
    """Kaplan–Meier 曲线与分组对照。"""

    @staticmethod
    def km(
        times: Sequence[float],
        events: Sequence[int],
        *,
        name: str = "all",
    ) -> Dict[str, Any]:
        times_a, events_a = _validate_km_inputs(times, events)
        t_axis, surv = kaplan_meier(times_a, events_a)
        return {
            "name": name,
            "time": t_axis,
            "survival": surv,
            "n": int(len(times_a)),
            "event_rate": float(np.mean(events_a)),
            "median_survival": SurvivalAnalyzer._median_survival(t_axis, surv),
        }

    @staticmethod
    def _median_survival(t_axis: np.ndarray, surv: np.ndarray) -> Optional[float]:
        below = np.where(surv <= 0.5)[0]
        if below.size == 0:
            return None
        return float(t_axis[below[0]])

    @staticmethod
    def km_by_group(
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        group_col: str,
    ) -> Dict[str, Any]:
        curves = {}
        for name, g in df.groupby(group_col):
            curves[str(name)] = SurvivalAnalyzer.km(
                g[time_col].to_numpy(),
                g[event_col].to_numpy(),
                name=str(name),
            )
        return {"groups": curves, "group_col": group_col}
