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
    def safe_lag(
        df: pd.DataFrame,
        col: str,
        lags: Sequence[int] = (1,),
        *,
        group_col: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        """安全滞后：只 ``shift(+k)``，禁止负滞后（未来）。"""
        for lag in lags:
            if int(lag) <= 0:
                raise ValueError(f"lag must be positive (got {lag}); negative would look ahead")
        work = df.copy()
        pfx = prefix or col
        if group_col:
            g = work.groupby(group_col, sort=False)[col]
            for lag in lags:
                work[f"{pfx}_lag{int(lag)}"] = g.shift(int(lag))
        else:
            for lag in lags:
                work[f"{pfx}_lag{int(lag)}"] = work[col].shift(int(lag))
        return work

    @staticmethod
    def safe_rolling_mean(
        df: pd.DataFrame,
        col: str,
        window: int,
        *,
        group_col: Optional[str] = None,
        out_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """滚动均值前先 ``shift(1)``，避免当期泄漏。"""
        if window < 1:
            raise ValueError("window must be >= 1")
        work = df.copy()
        name = out_col or f"{col}_roll{window}"
        if group_col:
            shifted = work.groupby(group_col, sort=False)[col].shift(1)
            work[name] = shifted.groupby(work[group_col], sort=False).rolling(window).mean().reset_index(level=0, drop=True)
        else:
            work[name] = work[col].shift(1).rolling(window).mean()
        return work
