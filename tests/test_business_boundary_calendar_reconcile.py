"""业务口径边界：空金额同比、连接膨胀、严格对账、月末频率、异常列序、KM 事件校验。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wanglaoshi.analytics.anomaly import AnomalyDetector
from wanglaoshi.analytics.survival import SurvivalAnalyzer
from wanglaoshi.business.calendar import complete_period_index, yoy_with_rules
from wanglaoshi.business.reconciliation import check_join_inflation, reconcile_to_source


@pytest.mark.quick
def test_yoy_null_amount_not_treated_as_zero() -> None:
    """有记录但金额全空 ≠ 真实 0，更不能算出 yoy=-100% / ok。"""
    sales = pd.DataFrame(
        {
            "month": pd.to_datetime(["2023-01-01", "2024-01-01"]),
            "sales": [100.0, np.nan],
        }
    )
    out = yoy_with_rules(sales, "month", "sales", freq="MS")
    row = out.loc[out["month"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert bool(row["is_missing"]) is False
    assert pd.isna(row["sales"])
    assert pd.isna(row["yoy"])
    assert row["reason"] == "current_amount_missing"


@pytest.mark.quick
def test_join_inflation_detects_canceling_duplicates() -> None:
    """+100/-100 净额抵消时，仍应检出左表行被一对多复制。"""
    left = pd.DataFrame({"id": [1, 2], "amt": [100.0, -100.0]})
    right = pd.DataFrame({"id": [1, 1, 2, 2], "x": [1, 2, 3, 4]})
    r = check_join_inflation(left, right, on="id", measure_col="amt")
    assert r.data["n_left"] == 2
    assert r.data["n_joined"] == 4
    assert r.data["true_sum"] == 0.0
    assert r.data["joined_sum"] == 0.0  # 净额仍为 0，不能只靠净额
    assert r.data["row_inflated"] is True
    assert r.data["inflated"] is True
    assert r.status == "warning"
    assert r.data["max_match_count"] == 2


@pytest.mark.quick
def test_reconcile_strict_rejects_disjoint_keys() -> None:
    """无共同业务键时，严格对账不得因补零而通过。"""
    computed = pd.DataFrame({"id": [1], "amount": [0.0]})
    source = pd.DataFrame({"id": [2], "amount": [0.0]})
    r = reconcile_to_source(computed, source, key="id", value_col="amount", strict=True)
    assert r.status == "warning"
    assert r.data["all_balanced"] is False
    detail = r.data["detail"]
    assert set(detail["key_status"]) == {"extra_in_computed", "missing_in_computed"}
    assert r.data["n_missing_in_computed"] == 1
    assert r.data["n_extra_in_computed"] == 1


@pytest.mark.quick
def test_month_end_freq_keeps_existing_period() -> None:
    """freq='M'/'ME' 与聚合共用 Period 键，已有月份不应被标成缺期。"""
    df = pd.DataFrame(
        {"month": pd.to_datetime(["2024-01-15", "2024-02-10"]), "sales": [10.0, 20.0]}
    )
    for freq in ("M", "ME", "MS"):
        cal = complete_period_index(df, "month", value_col="sales", freq=freq)
        assert len(cal) >= 2
        jan = cal.loc[cal["month"] == pd.Timestamp("2024-01-01")].iloc[0]
        assert bool(jan["is_missing"]) is False
        assert jan["sales"] == pytest.approx(10.0)


@pytest.mark.quick
def test_anomaly_detector_aligns_columns_by_name() -> None:
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0, 40.0]})
    det = AnomalyDetector(random_state=0).fit(X)
    s_same = det.score_samples(X)
    # 列顺序调换后按列名对齐，分数应与训练布局一致
    s_reordered = det.score_samples(X[["b", "a"]])
    np.testing.assert_allclose(s_same, s_reordered)
    # 训练用 DataFrame 后，禁止再静默按位置吃 ndarray
    with pytest.raises(ValueError, match="DataFrame"):
        det.score_samples(X[["b", "a"]].to_numpy())
    with pytest.raises(ValueError, match="missing columns"):
        det.score_samples(X[["a"]])


@pytest.mark.quick
def test_survival_rejects_invalid_events() -> None:
    with pytest.raises(ValueError, match="0, 1"):
        SurvivalAnalyzer.km([1], [2])
    with pytest.raises(ValueError, match="length"):
        SurvivalAnalyzer.km([1, 2], [1])
    with pytest.raises(ValueError, match="non-negative"):
        SurvivalAnalyzer.km([-1], [1])
    ok = SurvivalAnalyzer.km([1, 2, 3], [1, 0, 1])
    assert ok["survival"].min() >= 0.0
    assert 0.0 <= ok["event_rate"] <= 1.0
