"""C08：编码/缩放只在训练折拟合；目标编码全表 fit 会泄漏。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer


@pytest.mark.quick
def test_c08_target_encode_full_fit_leaks_vs_oof() -> None:
    """构造：稀有类别 R 只在「验证段」带正样本。

    全表目标编码会把验证标签信息编进特征；OOF/折内编码不应如此。
    """
    rng = np.random.default_rng(0)
    n = 60
    # 前 40 训练区：只有 A/B；后 20 验证区：混入 R，且 R 全是正例
    channel = ["A"] * 20 + ["B"] * 20 + ["R"] * 10 + ["A"] * 10
    y = (
        list(rng.integers(0, 2, 40))
        + [1] * 10  # R → 全正
        + list(rng.integers(0, 2, 10))
    )
    df = pd.DataFrame({"channel": channel, "y": y})
    train = df.iloc[:40].copy()
    valid = df.iloc[40:].copy()

    fe = AdvancedFeatureEngineer()
    # ❌ 全表 fit（含验证）再切片——泄漏
    leaky = fe.target_encode(df, "channel", "y", smoothing=1.0)
    leaky_valid_R = leaky.loc[leaky["channel"] == "R", "channel_target_encoded"]
    assert (leaky_valid_R > 0.8).all()  # 几乎等于 1，来自验证标签

    # ✅ 只在 train fit，再 map 到 valid
    train_enc = fe.target_encode(train, "channel", "y", smoothing=1.0)
    mapping = train_enc.groupby("channel")["channel_target_encoded"].first().to_dict()
    global_mean = float(train["y"].mean())
    safe_valid = valid.copy()
    safe_valid["channel_target_encoded"] = safe_valid["channel"].map(mapping).fillna(global_mean)
    # R 在训练中未见 → 全局均值，绝不应接近 1
    r_vals = safe_valid.loc[safe_valid["channel"] == "R", "channel_target_encoded"]
    assert (r_vals <= global_mean + 1e-9).all()
    assert float(r_vals.mean()) < float(leaky_valid_R.mean())


@pytest.mark.quick
def test_c08_oof_target_encode_no_self_label() -> None:
    """K 折 OOF：每个样本的编码不得使用自身所在折的标签统计。"""
    df = pd.DataFrame(
        {
            "channel": ["A", "A", "B", "B", "C", "C"] * 5,
            "y": [0, 1, 0, 1, 0, 1] * 5,
        }
    )
    fe = AdvancedFeatureEngineer()
    oof = np.full(len(df), np.nan)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(df):
        tr = df.iloc[tr_idx]
        enc = fe.target_encode(tr, "channel", "y", smoothing=1.0)
        mapping = enc.groupby("channel")["channel_target_encoded"].first().to_dict()
        gmean = float(tr["y"].mean())
        oof[va_idx] = df.iloc[va_idx]["channel"].map(mapping).fillna(gmean).to_numpy()
    assert np.isfinite(oof).all()
    # 与全表编码对比：两者不应完全相同（全表更「乐观」地用了全部标签）
    full = fe.target_encode(df, "channel", "y", smoothing=1.0)["channel_target_encoded"].to_numpy()
    assert not np.allclose(oof, full)


@pytest.mark.quick
def test_c08_scaler_must_fit_train_only() -> None:
    X = pd.DataFrame({"x": np.linspace(0, 100, 50)})
    y = (X["x"] > 50).astype(int)
    Xtr, Xte, _, _ = train_test_split(X, y, test_size=0.3, random_state=0)

    # ✅ 正确
    sc = StandardScaler().fit(Xtr)
    Xte_ok = sc.transform(Xte)

    # ❌ 错误：先全表 fit
    sc_bad = StandardScaler().fit(X)
    Xte_bad = sc_bad.transform(Xte)

    # 两种变换的测试集均值应不同（全表 fit 混入了测试分布）
    assert not np.allclose(Xte_ok.mean(axis=0), Xte_bad.mean(axis=0))
