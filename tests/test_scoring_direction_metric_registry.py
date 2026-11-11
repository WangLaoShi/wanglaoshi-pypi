"""评分方向与指标注册：MSE 选优、时间趋势≠泄露、agg/version 校验。"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from wanglaoshi.business.metric_registry import validate_metric_dict
from wanglaoshi.business.metrics import SUPPORTED_AGGS, calculate_metrics
from wanglaoshi.compete.baseline import QuickBaseline
from wanglaoshi.compete.hyperopt import HyperparameterOptimizer
from wanglaoshi.compete.leakage import LeakageDetector


@pytest.mark.quick
def test_baseline_compare_models_mse_ascending() -> None:
    """CV MSE=10 不得排在 MSE=1 前面。"""
    results = {
        "worse": {
            "model": None,
            "cv_mean": 10.0,
            "cv_std": 0.0,
            "metric": "mse",
            "higher_is_better": False,
            "task_type": "regression",
        },
        "better": {
            "model": None,
            "cv_mean": 1.0,
            "cv_std": 0.0,
            "metric": "mse",
            "higher_is_better": False,
            "task_type": "regression",
        },
    }
    df = QuickBaseline().compare_models(results, top_n=5)
    assert list(df["Model"]) == ["better", "worse"]
    assert df.attrs["sort_column"] == "CV Mean"
    assert df.attrs["sort_ascending"] is True
    assert df.attrs["higher_is_better"] is False


@pytest.mark.quick
def test_baseline_compare_models_accuracy_descending() -> None:
    results = {
        "low": {
            "model": None,
            "cv_mean": 0.5,
            "cv_std": 0.0,
            "metric": "accuracy",
            "higher_is_better": True,
            "task_type": "classification",
        },
        "high": {
            "model": None,
            "cv_mean": 0.9,
            "cv_std": 0.0,
            "metric": "accuracy",
            "higher_is_better": True,
            "task_type": "classification",
        },
    }
    df = QuickBaseline().compare_models(results)
    assert list(df["Model"]) == ["high", "low"]
    assert df.attrs["sort_ascending"] is False


@pytest.mark.quick
def test_hyperopt_regression_selects_lower_mse(monkeypatch: pytest.MonkeyPatch) -> None:
    """两个候选 MSE=1 / 10 时，必须选中误差较小的参数（不依赖本机是否安装 Optuna）。"""
    import sys
    from types import ModuleType, SimpleNamespace

    class _ToyReg(BaseEstimator, RegressorMixin):
        def __init__(self, level: int = 1):
            self.level = level

        def fit(self, X, y):  # noqa: ANN001, N803
            return self

        def predict(self, X):  # noqa: ANN001, N803
            return np.zeros(len(X))

    def fake_cv(model, X, y, cv=5, scoring=None, n_jobs=-1):  # noqa: ANN001
        # 目标函数应直接 maximize neg_mse：level=1 → -1；level=10 → -10
        assert scoring == "neg_mean_squared_error"
        neg = -float(model.level)
        n = cv if isinstance(cv, int) else 2
        return np.full(n, neg)

    class _Study:
        def __init__(self, direction: str):
            assert direction == "maximize"
            self.best_params = None
            self.best_value = None

        def optimize(self, objective, n_trials, show_progress_bar=True):  # noqa: ANN001
            values = {}
            for level in (1, 10):
                trial = SimpleNamespace(
                    suggest_categorical=lambda name, choices, _level=level: _level,
                    suggest_int=lambda *a, **k: 1,
                    suggest_float=lambda *a, **k: 1.0,
                )
                values[level] = objective(trial)
            best_level = max(values, key=values.get)
            self.best_params = {"level": best_level}
            self.best_value = values[best_level]

    fake_optuna = ModuleType("optuna")
    fake_optuna.create_study = lambda direction="maximize": _Study(direction)
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    X = pd.DataFrame({"x": np.arange(30, dtype=float)})
    y = pd.Series(np.linspace(0.0, 1.0, 30))  # 唯一值>=20 → 回归
    with patch("sklearn.model_selection.cross_val_score", side_effect=fake_cv):
        result = HyperparameterOptimizer().bayesian_optimize(
            _ToyReg,
            {"level": {"type": "categorical", "choices": [1, 10]}},
            X,
            y,
            n_trials=2,
            cv=2,
        )
    assert result["best_params"]["level"] == 1
    assert result["metric"] == "mse"
    assert result["higher_is_better"] is False
    assert result["best_score"] == pytest.approx(1.0)
    assert result["best_objective"] == pytest.approx(-1.0)


@pytest.mark.quick
def test_time_trend_alone_is_not_leakage() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "target": np.arange(30, dtype=float),
        }
    )
    result = LeakageDetector().detect_time_leakage(df, "date", "target")
    assert result["has_leakage"] is False
    assert result["info"].get("serial_correlation_flag") is True
    assert result["info"]["n_feature_cols"] == 0


@pytest.mark.quick
def test_metric_registry_rejects_bad_agg_and_keeps_version() -> None:
    bad = {
        "__version__": "v2",
        "gmv": {"value_col": "amount", "agg": "typo", "direction": "higher_better"},
    }
    r = validate_metric_dict(bad)
    assert r.status == "blocked"
    assert r.data["version"] == "v2"
    assert "__version__" not in r.data["metric_dict"]
    assert any("unsupported agg" in b for b in r.block_reasons)

    ok = validate_metric_dict(
        {
            "__version__": "v2",
            "gmv": {"value_col": "amount", "agg": "sum", "direction": "higher_better"},
        }
    )
    assert ok.status == "passed"
    assert ok.data["version"] == "v2"
    assert set(SUPPORTED_AGGS) == {"sum", "count", "mean"}

    # 计算器与注册器规则一致
    with pytest.raises(ValueError, match="unsupported agg"):
        calculate_metrics(
            pd.DataFrame({"amount": [1.0]}),
            metric_dict={"gmv": {"value_col": "amount", "agg": "typo"}},
        )
