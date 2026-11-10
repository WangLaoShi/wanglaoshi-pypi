"""C04：数值容差与结构精确断言。"""

from __future__ import annotations

import pandas as pd
import pytest

from wanglaoshi.asserting import assert_close, assert_frame_contract, assert_metrics_close
from wanglaoshi.business import calculate_metrics


@pytest.mark.quick
def test_c04_assert_close_and_metrics() -> None:
    assert_close(1.00000001, 1.0, atol=1e-7, name="x")
    with pytest.raises(AssertionError):
        assert_close(1.1, 1.0, atol=1e-3, name="x")

    df = pd.DataFrame({"order_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    result = calculate_metrics(df, group_cols=["order_id"], share_of="gmv")
    # 手工：gmv=60, orders=3, aov=20
    assert_metrics_close(
        result.data["metrics"],
        {"gmv": 60.0, "orders": 3.0, "aov": 20.0},
        atol=1e-9,
    )


@pytest.mark.quick
def test_c04_frame_contract_exact() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert_frame_contract(df, columns=["a", "b"], n_rows=1, ordered_columns=True)
    with pytest.raises(AssertionError, match="n_rows"):
        assert_frame_contract(df, n_rows=2)
    with pytest.raises(AssertionError, match="columns"):
        assert_frame_contract(df, columns=["b", "a"], ordered_columns=True)
