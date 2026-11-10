"""S4：头腰尾价值分层。"""

from __future__ import annotations

import pandas as pd
import pytest

from wanglaoshi.business import add_value_tiers, derive_metrics


@pytest.mark.quick
def test_pareto_tiers_head_gets_top_mass() -> None:
    # 一个大客户贡献绝大部分
    df = pd.DataFrame({"amount": [1000.0, 50.0, 30.0, 20.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0]})
    out = add_value_tiers(df, "amount", method="pareto", head=0.8, waist=0.95)
    assert set(out["value_tier"]).issubset({"头", "腰", "尾"})
    assert out.loc[out["amount"] == 1000.0, "value_tier"].iloc[0] == "头"
    # 头层累计应覆盖主要金额
    head_sum = out.loc[out["value_tier"] == "头", "amount"].sum()
    assert head_sum / out["amount"].sum() >= 0.8 - 1e-9


@pytest.mark.quick
def test_quantile_tiers_and_derive_metrics() -> None:
    df = pd.DataFrame({"amount": list(range(1, 31))})
    out = add_value_tiers(df, "amount", method="quantile")
    assert out["value_tier"].nunique() >= 2
    stage = derive_metrics(df, value_col="amount", add_tiers=True, tier_method="pareto")
    assert stage.status == "passed"
    assert "value_tier" in stage.data["created_columns"]
    assert stage.frame["value_tier"].notna().all()  # type: ignore[attr-defined]
