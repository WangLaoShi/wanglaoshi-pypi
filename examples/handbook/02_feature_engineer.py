"""Handbook 02 smoke: AdvancedFeatureEngineer on synthetic orders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wanglaoshi.features import AdvancedFeatureEngineer
from wanglaoshi.runtime import ready


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "order_time": pd.date_range("2024-01-01", periods=n, freq="D"),
            "note": rng.choice(["快", "普通备注 hello", "差评 物流慢", "好评"], size=n),
            "city": rng.choice([f"c{i}" for i in range(25)], size=n),
            "amount": rng.lognormal(mean=3.0, sigma=0.5, size=n),
        }
    )
    df["y"] = (df["amount"] > df["amount"].median()).astype(int)

    eng = AdvancedFeatureEngineer()
    df = eng.extract_datetime_features(df, "order_time")
    df = eng.extract_text_features(df, "note", method="basic")
    df = eng.frequency_encode(df, "city")
    df_small = eng.one_hot_encode(df.head(30).copy(), ["city"])
    assert any(c.startswith("city") for c in df_small.columns)

    train = df.iloc[:150].copy()
    valid = df.iloc[150:].copy()
    train = eng.target_encode(train, "city", "y", smoothing=5.0)
    te_col = "city_target_encoded"
    te_map = train.groupby("city")[te_col].first()
    valid[te_col] = valid["city"].map(te_map)
    assert te_col in train.columns and te_col in valid.columns

    df_num = eng.log_transform(df, ["amount"])
    df_num = eng.boxcox_transform(df_num, ["amount"])
    df_num = eng.bin_features(df_num, ["amount"], n_bins=4)

    feat_cols = [
        c
        for c in train.columns
        if c not in {"y", "note", "order_time", "city"}
        and pd.api.types.is_numeric_dtype(train[c])
    ]
    X = train[feat_cols].fillna(0)
    y = train["y"]
    sel_imp = eng.select_features_by_importance(X, y, n_features=8, model_type="random_forest")
    sel_corr = eng.select_features_by_correlation(X, y, threshold=0.05)
    sel_mi = eng.select_features_by_mutual_info(X, y, n_features=8)
    assert len(sel_imp) <= 8 and len(sel_mi) <= 8 and len(sel_corr) >= 1
    print("02_feature_engineer OK", sel_imp[:3], len(sel_corr))


if __name__ == "__main__":
    main()
