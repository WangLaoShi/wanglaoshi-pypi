"""回归：多分类 Blending、无效 ECE、NDCG 重复项、零转化 A/B。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from wanglaoshi.analytics.calibration import expected_calibration_error
from wanglaoshi.analytics.ranking import RankingMetrics
from wanglaoshi.analytics.stats.experiment import ExperimentDesign
from wanglaoshi.compete.ensemble import ModelEnsemble


@pytest.mark.quick
def test_blending_multiclass_aligns_proba_columns() -> None:
    """不同模型各自置信的类别可能不同；必须按 classes_ 对齐后按列融合。"""
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    X_val, X_hold, y_val, _y_hold = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=1, stratify=y_tmp
    )
    assert y_train.nunique() == 3

    models = [
        LogisticRegression(max_iter=500, random_state=0),
        DecisionTreeClassifier(max_depth=3, random_state=1),
    ]
    blend = ModelEnsemble().blending(
        models, X_train, y_train, X_val, y_val, meta_model=None, weights=[0.5, 0.5]
    )
    assert "predict_proba" in blend and "classes_" in blend
    proba = blend["predict_proba"](X_hold)
    labels = blend["predict"](X_hold)
    assert proba.shape == (len(X_hold), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(labels)).issubset(set(blend["classes_"].tolist()))
    # 手工核对：对齐后逐列平均应等于 predict_proba
    aligned = []
    classes = list(blend["classes_"])
    for m in models:
        raw = m.predict_proba(X_hold)
        mat = np.zeros_like(proba)
        for j, c in enumerate(m.classes_):
            mat[:, classes.index(c)] = raw[:, j]
        aligned.append(mat)
    expected = 0.5 * aligned[0] + 0.5 * aligned[1]
    np.testing.assert_allclose(proba, expected, atol=1e-8)
    # 不得再退化成「每样本一个最大类概率」的一维分数
    assert proba.ndim == 2 and proba.shape[1] == 3


@pytest.mark.quick
def test_ece_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="finite|\\[0, 1\\]"):
        expected_calibration_error([0, 1], [np.nan, 2.0])
    with pytest.raises(ValueError, match="length"):
        expected_calibration_error([0, 1], [0.1])
    with pytest.raises(ValueError, match="binary"):
        expected_calibration_error([0, 2], [0.1, 0.2])
    # 合法输入仍给出非「假完美」分数
    ece = expected_calibration_error([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], n_bins=2)
    assert 0.0 <= ece <= 1.0


@pytest.mark.quick
def test_ndcg_duplicate_recommendations_do_not_inflate() -> None:
    score = RankingMetrics.ndcg_at_k(["a", "a"], {"a"}, k=2)
    assert 0.0 <= score <= 1.0
    # 重复命中不得高于单次理想排序
    assert score == pytest.approx(RankingMetrics.ndcg_at_k(["a", "b"], {"a"}, k=2))


@pytest.mark.quick
def test_two_proportion_zero_conversions_not_applicable() -> None:
    r = ExperimentDesign.two_proportion_test(0, 100, 0, 100)
    assert r["p1"] == 0.0 and r["p2"] == 0.0
    assert r["ci95_applicable"] is False
    assert r["chi2_applicable"] is False
    assert r["applicable"] is False
    assert r["ci95"] is None
    assert r["chi2"] is None
    assert r["notes"]
    # 正常对照仍适用
    ok = ExperimentDesign.two_proportion_test(80, 800, 100, 800)
    assert ok["applicable"] is True
    assert ok["ci95"] is not None and ok["chi2"] is not None
