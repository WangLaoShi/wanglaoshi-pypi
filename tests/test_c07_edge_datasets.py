"""C07：全空/常数列、单类别、小样本、高维、未见类别、中文、错误日期、重复键。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wanglaoshi import DataAnalyzer
from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer
from wanglaoshi.business import clean_dataframe, structure_dataframe


@pytest.mark.quick
def test_c07_all_null_constant_single_class() -> None:
    df = pd.DataFrame(
        {
            "empty": [np.nan, np.nan, np.nan],
            "const": [1, 1, 1],
            "y": [0, 0, 0],
            "x": [1.0, 2.0, 3.0],
        }
    )
    info = DataAnalyzer(df).get_basic_info()
    assert info["行数"] == 3
    cleaned = clean_dataframe(df, fillna={"empty": 0}, drop_duplicates=False)
    assert cleaned.status in {"passed", "warning"}
    assert cleaned.frame["empty"].tolist() == [0, 0, 0]  # type: ignore[attr-defined]
    assert df["y"].nunique() == 1


@pytest.mark.quick
def test_c07_chinese_bad_dates_duplicate_keys() -> None:
    df = pd.DataFrame(
        {
            "订单号": [1, 1, 2],
            "日期": ["2024-01-01", "not-a-date", "2024-02-01"],
            "金额": [10.0, 20.0, 30.0],
        }
    )
    assert df["订单号"].duplicated().any()
    parsed = pd.to_datetime(df["日期"], errors="coerce")
    assert int(parsed.isna().sum()) == 1

    st = structure_dataframe(df)
    assert st.status in {"passed", "warning"}
    assert hasattr(st, "frame")
    # 中文列名应被规整为可机器处理的 snake 风格（非空）
    assert all(isinstance(c, str) and len(c) > 0 for c in st.frame.columns)  # type: ignore[attr-defined]


@pytest.mark.quick
def test_c07_unseen_category_and_small_high_dim() -> None:
    train = pd.DataFrame({"channel": ["web", "app", "web", "app"], "y": [0, 1, 0, 1]})
    test = pd.DataFrame({"channel": ["offline", "web"], "y": [0, 1]})
    fe = AdvancedFeatureEngineer()
    enc = fe.target_encode(train, "channel", "y", smoothing=1.0)
    mapping = enc.groupby("channel")["channel_target_encoded"].first().to_dict()
    gmean = float(train["y"].mean())
    test_enc = test["channel"].map(mapping).fillna(gmean)
    assert np.isfinite(test_enc).all()
    assert float(test_enc.iloc[0]) == pytest.approx(gmean)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(8, 40)), columns=[f"f{i}" for i in range(40)])
    X["y"] = rng.integers(0, 2, 8)
    info = DataAnalyzer(X).get_basic_info()
    assert info["行数"] == 8 and info["列数"] == 41
