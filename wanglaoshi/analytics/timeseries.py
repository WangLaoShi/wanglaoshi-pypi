"""时间序列最小工具（P1）。对齐 E08 / E15。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


class TimeSeriesTools:
    """分解摘要、ADF 提示、安全滞后工厂。"""

    @staticmethod
    def decompose_summary(
        series: Union[pd.Series, Sequence[float]],
        *,
        period: int = 7,
        model: str = "additive",
    ) -> Dict[str, Any]:
        """季节分解摘要（需 statsmodels）。"""
        from statsmodels.tsa.seasonal import seasonal_decompose

        s = pd.Series(series, dtype=float).dropna()
        if len(s) < period * 2:
            raise ValueError(f"need >= {period * 2} points for period={period}")
        decomp = seasonal_decompose(s, model=model, period=period)
        return {
            "period": period,
            "model": model,
            "n": int(len(s)),
            "trend_nan_ratio": float(decomp.trend.isna().mean()),
            "seasonal_std": float(np.nanstd(decomp.seasonal)),
            "resid_std": float(np.nanstd(decomp.resid)),
            "trend": decomp.trend,
            "seasonal": decomp.seasonal,
            "resid": decomp.resid,
        }

    @staticmethod
    def adf_report(series: Union[pd.Series, Sequence[float]], name: str = "series") -> Dict[str, Any]:
        """ADF 平稳性提示。"""
        from statsmodels.tsa.stattools import adfuller

        s = pd.Series(series, dtype=float).dropna()
        r = adfuller(s, autolag="AIC")
        return {
            "name": name,
            "adf_stat": float(r[0]),
            "pvalue": float(r[1]),
            "usedlag": int(r[2]),
            "nobs": int(r[3]),
            "critical": {str(k): float(v) for k, v in r[4].items()},
            "stationary_hint": bool(r[1] < 0.05),
        }

    @staticmethod
    def _order_for_time(
        df: pd.DataFrame,
        *,
        time_col: Optional[str],
        group_col: Optional[str],
        assume_row_order: bool,
    ) -> pd.DataFrame:
        """按时间排序；无时间列时必须显式承认行序即时间序。"""
        work = df.copy()
        if time_col is not None:
            if time_col not in work.columns:
                raise ValueError(f"time_col not found: {time_col}")
            work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
            sort_cols: List[str] = [group_col, time_col] if group_col else [time_col]
            if group_col and group_col not in work.columns:
                raise ValueError(f"group_col not found: {group_col}")
            work = work.sort_values(sort_cols, kind="mergesort")
            # 组内时间必须非降序（排序后应满足）；重复时间戳给出明确错误
            if group_col:
                dup = work.duplicated(subset=[group_col, time_col], keep=False)
            else:
                dup = work.duplicated(subset=[time_col], keep=False)
            if bool(dup.any()):
                raise ValueError(
                    "duplicate timestamps within group; aggregate before lag/rolling "
                    "or pass unique time keys"
                )
            return work
        if not assume_row_order:
            raise ValueError(
                "safe_lag/safe_rolling require time_col=... "
                "or assume_row_order=True when rows are already chronological"
            )
        return work

    @staticmethod
    def safe_lag(
        df: pd.DataFrame,
        col: str,
        lags: Sequence[int] = (1,),
        *,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
        prefix: Optional[str] = None,
        assume_row_order: bool = False,
    ) -> pd.DataFrame:
        """安全滞后：只 ``shift(+k)``，禁止负滞后（未来）。

        - 推荐传 ``time_col``：会按组+时间排序后再 lag，再恢复原行序
        - 若确定行已按时间排好，可 ``assume_row_order=True``
        """
        for lag in lags:
            if int(lag) <= 0:
                raise ValueError(f"lag must be positive (got {lag}); negative would look ahead")
        if col not in df.columns:
            raise ValueError(f"col not found: {col}")

        ordered = TimeSeriesTools._order_for_time(
            df, time_col=time_col, group_col=group_col, assume_row_order=assume_row_order
        )
        pfx = prefix or col
        if group_col:
            g = ordered.groupby(group_col, sort=False)[col]
            for lag in lags:
                ordered[f"{pfx}_lag{int(lag)}"] = g.shift(int(lag))
        else:
            for lag in lags:
                ordered[f"{pfx}_lag{int(lag)}"] = ordered[col].shift(int(lag))

        # 恢复调用方行序，避免打乱下游对齐
        return ordered.reindex(df.index)

    @staticmethod
    def safe_rolling_mean(
        df: pd.DataFrame,
        col: str,
        window: int,
        *,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
        out_col: Optional[str] = None,
        assume_row_order: bool = False,
    ) -> pd.DataFrame:
        """滚动均值前先 ``shift(1)``，避免当期泄漏。

        同样需要 ``time_col`` 或 ``assume_row_order=True``。
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        if col not in df.columns:
            raise ValueError(f"col not found: {col}")

        ordered = TimeSeriesTools._order_for_time(
            df, time_col=time_col, group_col=group_col, assume_row_order=assume_row_order
        )
        name = out_col or f"{col}_roll{window}"
        if group_col:
            shifted = ordered.groupby(group_col, sort=False)[col].shift(1)
            ordered[name] = (
                shifted.groupby(ordered[group_col], sort=False)
                .rolling(window)
                .mean()
                .reset_index(level=0, drop=True)
            )
        else:
            ordered[name] = ordered[col].shift(1).rolling(window).mean()
        return ordered.reindex(df.index)
