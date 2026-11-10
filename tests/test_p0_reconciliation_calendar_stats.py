"""P0：对账 / 同比 / StatsToolkit / ExperimentDesign。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wanglaoshi import (
    ExperimentDesign,
    StatsToolkit,
    check_join_inflation,
    complete_period_index,
    reconcile_to_source,
    yoy_with_rules,
)
from wanglaoshi.business import yoy_stage


@pytest.mark.quick
def test_join_inflation_detected() -> None:
    orders = pd.DataFrame({"order_id": [1, 2], "order_amount": [100.0, 200.0]})
    items = pd.DataFrame(
        {"order_id": [1, 1, 2, 2, 2], "line_amount": [40, 60, 50, 70, 80]}
    )
    r = check_join_inflation(orders, items, on="order_id", measure_col="order_amount")
    assert r.status == "warning"
    assert r.data["inflated"] is True
    assert r.data["true_sum"] == 300.0
    assert r.data["joined_sum"] > 300.0
    assert abs(r.data["dedup_sum"] - 300.0) < 1e-9


@pytest.mark.quick
def test_reconcile_to_source_balanced() -> None:
    source = pd.DataFrame({"order_id": [1, 2], "amount": [100.0, 200.0]})
    computed = pd.DataFrame({"order_id": [1, 2], "amount": [100.0, 200.0]})
    r = reconcile_to_source(computed, source, key="order_id", value_col="amount")
    assert r.status == "passed"
    assert r.data["all_balanced"] is True


@pytest.mark.quick
def test_reconcile_unbalanced_warning() -> None:
    source = pd.DataFrame({"order_id": [1], "amount": [100.0]})
    computed = pd.DataFrame({"order_id": [1], "amount": [110.0]})
    r = reconcile_to_source(computed, source, key="order_id", value_col="amount")
    assert r.status == "warning"
    assert r.data["n_unbalanced"] == 1


@pytest.mark.quick
def test_yoy_edge_reasons() -> None:
    sales = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-03-01",  # 故意缺 2023-02；3 月为真 0
                    "2023-04-01",
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-11-01",
                ]
            ),
            "sales": [100.0, 0.0, 120.0, 110.0, 130.0, 0.0, 140.0, 50.0],
        }
    )
    out = yoy_with_rules(sales, "month", "sales", as_of="2024-11-15", freq="MS")
    reasons = set(out["reason"])
    assert "zero_base" in reasons  # 2024-03 vs 2023-03=0
    assert "incomplete_month" in reasons  # 2024-11
    assert "ok" in reasons
    # 缺月在 complete 后应出现 current_missing 或 prior_missing
    assert "current_missing_month" in reasons or "prior_missing_month" in reasons

    stage = yoy_stage(sales, "month", "sales", as_of="2024-11-15")
    assert stage.status in {"passed", "warning"}
    assert "reason_counts" in stage.data


@pytest.mark.quick
def test_complete_period_marks_missing() -> None:
    df = pd.DataFrame(
        {"month": pd.to_datetime(["2024-01-01", "2024-03-01"]), "sales": [1.0, 2.0]}
    )
    cal = complete_period_index(df, "month", value_col="sales", freq="MS")
    assert cal["is_missing"].sum() >= 1
    miss = cal.loc[cal["is_missing"], "month"]
    assert any(ts.month == 2 for ts in miss)


@pytest.mark.quick
def test_stats_toolkit_basic() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 40)
    b = rng.normal(0.5, 1, 40)
    t = StatsToolkit.ttest_ind(a, b)
    assert "pvalue" in t and "cohens_d" in t
    tab = [[10, 5], [3, 12]]
    c = StatsToolkit.chi2_independence(tab)
    assert c["pvalue"] < 0.05
    assert "min_expected" in c
    an = StatsToolkit.anova_oneway(a, b, rng.normal(1.0, 1, 40))
    assert "eta_squared" in an


@pytest.mark.quick
def test_experiment_design() -> None:
    r = ExperimentDesign.two_proportion_test(80, 800, 100, 800)
    assert "lift_pp" in r and "ci95" in r
    n = ExperimentDesign.sample_size_two_proportion(0.10, 0.01)
    assert n["n_per_arm"] > 1000
    b = ExperimentDesign.bayes_prob_beat(80, 800, 100, 800, delta=0.005)
    assert 0.0 <= b["prob_b_beats_a"] <= 1.0
    s = ExperimentDesign.prior_sensitivity(80, 800, 100, 800, delta=0.005)
    assert len(s["rows"]) == 3
