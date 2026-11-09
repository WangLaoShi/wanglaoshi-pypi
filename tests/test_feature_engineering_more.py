"""FeatureEngineering：目标编码、分箱、文本、重要性选择。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer


@pytest.mark.quick
def test_target_encode_and_bin() -> None:
    df = pd.DataFrame(
        {
            "channel": ["web", "app", "web", "app", "web", "offline"],
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )
    fe = AdvancedFeatureEngineer()
    enc = fe.target_encode(df, "channel", "y", smoothing=1.0)
    assert "channel_target_encoded" in enc.columns
    assert enc["channel_target_encoded"].notna().all()

    binned = fe.bin_features(enc, ["amount"], n_bins=3, strategy="quantile")
    assert "amount_binned" in binned.columns


@pytest.mark.quick
def test_text_features_basic_and_tfidf() -> None:
    df = pd.DataFrame({"text": ["Hello world!", "foo 123", "bar", "Hello again"]})
    fe = AdvancedFeatureEngineer()
    basic = fe.extract_text_features(df, "text", method="basic")
    assert "text_length" in basic.columns
    assert "text_word_count" in basic.columns
    assert int(basic.loc[0, "text_word_count"]) == 2

    tfidf = fe.extract_text_features(df, "text", method="tfidf", max_features=5)
    assert any(c.startswith("text_tfidf_") for c in tfidf.columns)


@pytest.mark.quick
def test_select_features_by_importance() -> None:
    rng = np.random.default_rng(3)
    X = pd.DataFrame({f"f{i}": rng.normal(size=40) for i in range(6)})
    y = pd.Series((X["f0"] + 0.2 * X["f1"] > 0).astype(int))
    fe = AdvancedFeatureEngineer()
    selected = fe.select_features_by_importance(X, y, n_features=3, model_type="random_forest")
    assert isinstance(selected, list)
    assert len(selected) == 3
    assert all(c in X.columns for c in selected)


@pytest.mark.quick
def test_text_invalid_method_raises() -> None:
    fe = AdvancedFeatureEngineer()
    with pytest.raises(ValueError, match="不支持"):
        fe.extract_text_features(pd.DataFrame({"t": ["a"]}), "t", method="magic")
