"""P1：时序 / 校准 / 公平性 / 异常 / NDCG。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wanglaoshi import (
    AnomalyDetector,
    CalibrationReport,
    GroupMetricsReport,
    RankingMetrics,
    TimeSeriesTools,
)


@pytest.mark.quick
def test_timeseries_safe_lag_and_adf() -> None:
    df = pd.DataFrame({"y": np.arange(30, dtype=float)})
    out = TimeSeriesTools.safe_lag(df, "y", lags=(1, 2), assume_row_order=True)
    assert "y_lag1" in out.columns and out["y_lag1"].iloc[1] == 0
    with pytest.raises(ValueError):
        TimeSeriesTools.safe_lag(df, "y", lags=(-1,), assume_row_order=True)
    with pytest.raises(ValueError, match="time_col"):
        TimeSeriesTools.safe_lag(df, "y", lags=(1,))
    rolled = TimeSeriesTools.safe_rolling_mean(df, "y", window=3, assume_row_order=True)
    assert "y_roll3" in rolled.columns
    adf = TimeSeriesTools.adf_report(df["y"])
    assert "pvalue" in adf
    # seasonal series for decompose
    t = np.arange(56)
    s = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 7)
    dec = TimeSeriesTools.decompose_summary(s, period=7)
    assert dec["period"] == 7 and "resid_std" in dec


@pytest.mark.quick
def test_timeseries_lag_sorts_by_time_col() -> None:
    # 故意乱序：若按行序 shift，day1 会吃到 day2（未来）
    df = pd.DataFrame(
        {
            "day": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"]),
            "y": [20.0, 10.0, 30.0],
        },
        index=[10, 11, 12],
    )
    out = TimeSeriesTools.safe_lag(df, "y", lags=(1,), time_col="day")
    # 恢复原行序后：索引 11 = day1，lag 应为 NaN；索引 10 = day2，lag=10
    assert pd.isna(out.loc[11, "y_lag1"])
    assert out.loc[10, "y_lag1"] == pytest.approx(10.0)
    assert out.loc[12, "y_lag1"] == pytest.approx(20.0)


@pytest.mark.quick
def test_group_metrics_noncontiguous_index() -> None:
    y = pd.Series([0, 1], index=[10, 11])
    pred = pd.Series([0, 1], index=[10, 11])
    g = pd.Series(["A", "B"], index=[10, 11])
    audit = GroupMetricsReport.audit(y, pred, g)
    assert len(audit) == 2
    assert set(audit["group"]) == {"A", "B"}


@pytest.mark.quick
def test_calibration_and_groups() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p_raw = np.clip(y * 0.7 + rng.random(200) * 0.3, 0, 1)
    p_flat = np.full(200, 0.5)
    cmp = CalibrationReport.compare(y, {"raw": p_raw, "flat": p_flat}, n_bins=10)
    assert set(cmp["name"]) == {"raw", "flat"}
    assert "ece" in cmp.columns

    pred = (p_raw >= 0.5).astype(int)
    sex = rng.choice(["F", "M"], 200)
    race = rng.choice(["A", "B"], 200)
    audit = GroupMetricsReport.audit(y, pred, {"sex": sex, "race": race})
    assert len(audit) >= 1
    assert {"n", "tpr", "fpr"}.issubset(audit.columns)


@pytest.mark.quick
def test_anomaly_and_ndcg() -> None:
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (80, 3)), rng.normal(5, 1, (10, 3))])
    y = np.array([0] * 80 + [1] * 10)
    det = AnomalyDetector(contamination=10 / 90, random_state=1)
    summary = det.fit_predict_summary(X, y_injected=y)
    assert summary["n_flagged"] >= 1
    assert "f1" in summary

    assert RankingMetrics.ndcg_at_k([1, 2, 3], {1}, k=3) == 1.0
    assert RankingMetrics.ndcg_at_k([9, 1, 2], {1}, k=3) < 1.0
    assert RankingMetrics.hit_rate_at_k([9, 1], {1}, k=2) == 1.0
