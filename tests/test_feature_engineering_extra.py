"""FeatureEngineering 编码与数值变换覆盖。"""

from __future__ import annotations

import pandas as pd
import pytest

from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer


@pytest.mark.quick
def test_one_hot_and_log_transform() -> None:
    df = pd.DataFrame(
        {
            "channel": ["web", "app", "web", "offline"],
            "amount": [0.0, 10.0, 20.0, 5.0],
        }
    )
    fe = AdvancedFeatureEngineer()
    out = fe.one_hot_encode(df, ["channel"], drop_first=True)
    assert any(c.startswith("channel_") for c in out.columns)

    logged = fe.log_transform(out, ["amount"], add_one=True)
    assert "amount_log" in logged.columns
    assert abs(float(logged.loc[0, "amount_log"]) - 0.0) < 1e-12


@pytest.mark.quick
def test_select_features_by_correlation() -> None:
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 1.0, 0.0, 4.0, 3.0],
            "c": [0.1, 0.2, 0.15, 0.05, 0.0],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # 与 a 完全相关
    fe = AdvancedFeatureEngineer()
    selected = fe.select_features_by_correlation(X, y, threshold=0.5)
    assert isinstance(selected, list)
    assert "a" in selected
