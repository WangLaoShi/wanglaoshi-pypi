"""CompetitionTools 堆叠/混合/泄露/实验追踪/调参边界。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from wanglaoshi.CompetitionTools import (
    CrossValidator,
    ExperimentTracker,
    HyperparameterOptimizer,
    LeakageDetector,
    ModelEnsemble,
)


@pytest.fixture
def clf_xy():
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"f1": rng.normal(size=48), "f2": rng.normal(size=48)})
    y = pd.Series((X["f1"] > 0).astype(int))
    return X, y


@pytest.mark.quick
def test_stacking_returns_meta_bundle(clf_xy) -> None:
    X, y = clf_xy
    models = [
        LogisticRegression(max_iter=300, random_state=0),
        DecisionTreeClassifier(max_depth=2, random_state=0),
    ]
    meta = LogisticRegression(max_iter=300, random_state=1)
    result = ModelEnsemble().stacking(models, meta, X, y, cv=2)
    assert isinstance(result, dict)
    assert "meta_model" in result
    assert len(result["first_level_models"]) == 2
    # 省略 X_test 时返回的第一层模型必须已在完整训练集上拟合，可直接预测
    for m in result["first_level_models"]:
        pred = m.predict(X.iloc[:3])
        assert len(pred) == 3


@pytest.mark.quick
def test_stacking_with_test_returns_predictions(clf_xy) -> None:
    X, y = clf_xy
    X_train, X_test = X.iloc[:36], X.iloc[36:]
    y_train = y.iloc[:36]
    models = [
        LogisticRegression(max_iter=300, random_state=0),
        DecisionTreeClassifier(max_depth=2, random_state=0),
    ]
    preds = ModelEnsemble().stacking(
        models,
        LogisticRegression(max_iter=300, random_state=1),
        X_train,
        y_train,
        X_test=X_test,
        cv=2,
    )
    assert len(preds) == len(X_test)


@pytest.mark.quick
def test_blending_weighted_average_predict(clf_xy) -> None:
    X, y = clf_xy
    X_train, X_val = X.iloc[:30], X.iloc[30:40]
    y_train, y_val = y.iloc[:30], y.iloc[30:40]
    X_hold = X.iloc[40:]
    models = [
        LogisticRegression(max_iter=300, random_state=0),
        DecisionTreeClassifier(max_depth=2, random_state=0),
    ]
    blend = ModelEnsemble().blending(models, X_train, y_train, X_val, y_val, meta_model=None)
    assert "predict" in blend
    out = blend["predict"](X_hold)
    assert len(out) == len(X_hold)


@pytest.mark.quick
def test_blending_with_meta_model(clf_xy) -> None:
    X, y = clf_xy
    X_train, X_val = X.iloc[:30], X.iloc[30:]
    y_train, y_val = y.iloc[:30], y.iloc[30:]
    models = [
        LogisticRegression(max_iter=300, random_state=0),
        DecisionTreeClassifier(max_depth=2, random_state=0),
    ]
    blend = ModelEnsemble().blending(
        models,
        X_train,
        y_train,
        X_val,
        y_val,
        meta_model=LogisticRegression(max_iter=300, random_state=2),
    )
    preds = blend["predict"](X_val)
    assert len(preds) == len(X_val)


@pytest.mark.quick
def test_time_leakage_future_named_feature() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "target": np.arange(20),
            "future_sales": np.arange(20) * 2,
        }
    )
    result = LeakageDetector().detect_time_leakage(df, "date", "target")
    assert result["has_leakage"] is True
    assert any("future_sales" in w for w in result["warnings"])
    assert "date_range" in result["info"]


@pytest.mark.quick
def test_distribution_leakage_ks() -> None:
    train = pd.DataFrame({"x": np.random.default_rng(0).normal(0, 1, 80)})
    test = pd.DataFrame({"x": np.random.default_rng(1).normal(10, 1, 80)})
    result = LeakageDetector().detect_distribution_leakage(train, test, threshold=0.05)
    assert result["has_leakage"] is True
    assert any(item["feature"] == "x" for item in result["leakage_features"])


@pytest.mark.quick
def test_group_cv(clf_xy) -> None:
    X, y = clf_xy
    groups = pd.Series(np.repeat(np.arange(8), 6)[: len(X)])
    result = CrossValidator().group_cv(
        LogisticRegression(max_iter=300, random_state=0),
        X,
        y,
        groups,
        n_splits=4,
    )
    assert result["cv_type"] == "group"
    assert len(result["scores"]) == 4


@pytest.mark.quick
def test_experiment_tracker_roundtrip(tmp_path: Path) -> None:
    log = tmp_path / "experiments.json"
    tracker = ExperimentTracker(str(log))
    tracker.log_experiment(
        "exp_a",
        "lr",
        features=["f1", "f2"],
        params={"C": 1.0},
        score=0.81,
        metrics={"f1": 0.8},
        notes="unit",
    )
    tracker.log_experiment(
        "exp_b",
        "tree",
        features=["f1"],
        params={"max_depth": 3},
        score=0.77,
    )
    assert log.exists()
    hist = tracker.get_experiment_history(sort_by="score", top_n=1)
    assert len(hist) == 1
    assert hist.iloc[0]["experiment_name"] == "exp_a"

    reloaded = ExperimentTracker(str(log))
    assert len(reloaded.experiments) == 2
    cmp = reloaded.compare_experiments(["exp_a", "exp_b"])
    assert set(cmp["实验名称"]) == {"exp_a", "exp_b"}


@pytest.mark.quick
def test_hyperparameter_optimizer_requires_optuna(clf_xy) -> None:
    X, y = clf_xy
    opt = HyperparameterOptimizer()
    optuna = pytest.importorskip("optuna", reason="optuna optional")
    # 若环境有 optuna，跑极小试验；否则上一行已 skip
    from sklearn.tree import DecisionTreeClassifier as DTC

    result = opt.bayesian_optimize(
        DTC,
        {"max_depth": {"type": "int", "low": 1, "high": 3}},
        X,
        y,
        n_trials=2,
        cv=2,
    )
    assert "best_params" in result
    assert result["n_trials"] == 2
    assert optuna is not None


@pytest.mark.quick
def test_hyperparameter_optimizer_import_error_message(clf_xy, monkeypatch) -> None:
    X, y = clf_xy

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "optuna" or name.startswith("optuna."):
            raise ImportError("no optuna")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="Optuna|optuna"):
        HyperparameterOptimizer().bayesian_optimize(
            DecisionTreeClassifier,
            {"max_depth": {"type": "int", "low": 1, "high": 2}},
            X,
            y,
            n_trials=1,
            cv=2,
        )
