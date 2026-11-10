"""生存分析最小工具（P1 extras 级，无 lifelines 硬依赖）。对齐 E16。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def kaplan_meier(
    times: Sequence[float],
    events: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """手写 KM：返回 (t_axis, survival)。"""
    times_a = np.asarray(times, dtype=float)
    events_a = np.asarray(events, dtype=int)
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
        t_axis, surv = kaplan_meier(times, events)
        return {
            "name": name,
            "time": t_axis,
            "survival": surv,
            "n": int(len(times)),
            "event_rate": float(np.mean(np.asarray(events))),
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
