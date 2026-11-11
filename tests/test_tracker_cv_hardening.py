"""实验追踪原子写入 / 多实例合并，以及 CV 评分与分层入口边界。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from wanglaoshi.compete.cv import CrossValidator, resolve_cv_scoring
from wanglaoshi.compete.tracker import ExperimentTracker


@pytest.mark.quick
def test_tracker_failed_save_keeps_history(tmp_path: Path) -> None:
    log = tmp_path / "experiments.json"
    t = ExperimentTracker(str(log))
    t.log_experiment("ok", "m", ["a"], {"C": 1.0}, 0.9)

    with pytest.raises(TypeError, match="JSON-serializable"):
        t.log_experiment("bad", "m", ["a"], {"fn": lambda x: x}, 0.1)

    text = log.read_text(encoding="utf-8")
    assert "ok" in text
    assert "bad" not in text
    reloaded = ExperimentTracker(str(log))
    assert len(reloaded.experiments) == 1
    assert reloaded.experiments[0]["experiment_name"] == "ok"
    # 失败写入不应污染内存中的有效历史
    assert len(t.experiments) == 1


@pytest.mark.quick
def test_tracker_two_instances_merge_appends(tmp_path: Path) -> None:
    log = tmp_path / "shared.json"
    seed = ExperimentTracker(str(log))
    seed.log_experiment("seed", "m", ["a"], {}, 0.5)

    a = ExperimentTracker(str(log))
    b = ExperimentTracker(str(log))
    a.log_experiment("new", "m", ["a"], {}, 0.6)
    b.log_experiment("other", "m", ["a"], {}, 0.7)

    names = [e["experiment_name"] for e in ExperimentTracker(str(log)).experiments]
    assert names == ["seed", "new", "other"]


@pytest.mark.quick
def test_resolve_scoring_small_continuous_is_regression() -> None:
    y = pd.Series(np.linspace(0, 1, 12))
    scoring, metric, higher = resolve_cv_scoring(y, task_type="auto")
    assert scoring == "neg_mean_squared_error"
    assert metric == "mse"
    assert higher is False


@pytest.mark.quick
def test_time_series_cv_small_continuous_uses_mse() -> None:
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(np.linspace(0, 1, 12))
    result = CrossValidator().time_series_cv(LinearRegression(), X, y, n_splits=2)
    assert result["scoring"] == "neg_mean_squared_error"
    assert result["metric"] == "mse"
    assert result["higher_is_better"] is False
    assert len(result["scores"]) == 2
    assert np.isfinite(result["scores"]).all()
    assert np.isfinite(result["mean"])


@pytest.mark.quick
def test_time_series_cv_all_folds_failed_raises() -> None:
    from sklearn.base import BaseEstimator, RegressorMixin

    class _Boom(BaseEstimator, RegressorMixin):
        def fit(self, X, y):  # noqa: ANN001, N803
            raise RuntimeError("cannot fit")

        def predict(self, X):  # noqa: ANN001, N803
            return np.zeros(len(X))

    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(np.linspace(0, 1, 12))
    with pytest.raises(RuntimeError, match="all CV folds failed"):
        CrossValidator().time_series_cv(_Boom(), X, y, n_splits=2, task_type="regression")


@pytest.mark.quick
def test_stratified_cv_rejects_continuous_target() -> None:
    X = pd.DataFrame({"x": np.arange(30, dtype=float)})
    y = pd.Series(np.linspace(0, 1, 30))
    with pytest.raises(ValueError, match="classification"):
        CrossValidator().stratified_cv(LinearRegression(), X, y, n_splits=3)


@pytest.mark.quick
def test_stratified_cv_classification_still_works() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=40), "f2": rng.normal(size=40)})
    y = pd.Series((X["f1"] > 0).astype(int))
    result = CrossValidator().stratified_cv(
        LogisticRegression(max_iter=300, random_state=0), X, y, n_splits=3
    )
    assert result["cv_type"] == "stratified"
    assert result["metric"] == "accuracy"
    assert len(result["scores"]) == 3
