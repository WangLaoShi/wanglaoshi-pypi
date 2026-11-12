"""Optuna 已安装时的真实调参方向验收（可选依赖）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from wanglaoshi.compete.hyperopt import HyperparameterOptimizer

optuna = pytest.importorskip("optuna", reason="optuna optional")


class _BiasRegressor(BaseEstimator, RegressorMixin):
    """predict 恒为 bias；bias 越接近 y 均值则 MSE 越小。"""

    def __init__(self, bias: float = 0.0):
        self.bias = bias

    def fit(self, X, y):  # noqa: ANN001, N803
        return self

    def predict(self, X):  # noqa: ANN001, N803
        return np.full(len(X), float(self.bias))


@pytest.mark.quick
def test_optuna_live_selects_lower_mse_bias(monkeypatch: pytest.MonkeyPatch) -> None:
    """两个可区分偏置：应选中更接近目标的 bias（真实 Optuna，非 mock）。"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _orig_create = optuna.create_study

    # 固定网格：保证两个候选都被评估，避免 TPE 随机漏掉更优值
    def _create_study(direction: str = "maximize", **kwargs):
        return _orig_create(
            direction=direction,
            sampler=optuna.samplers.GridSampler({"bias": [0.0, 5.0]}),
        )

    monkeypatch.setattr(optuna, "create_study", _create_study)

    # 串行 CV：避免 n_jobs=-1 在受限环境里不稳定
    import sklearn.model_selection as sms

    _cv = sms.cross_val_score

    def _cv_serial(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return _cv(*args, **kwargs)

    monkeypatch.setattr(sms, "cross_val_score", _cv_serial)

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=40)})
    # 唯一值 >= 20 → 走回归 / neg_mean_squared_error
    y = pd.Series(np.linspace(4.5, 5.5, 40))

    result = HyperparameterOptimizer().bayesian_optimize(
        _BiasRegressor,
        {"bias": {"type": "categorical", "choices": [0.0, 5.0]}},
        X,
        y,
        n_trials=2,
        cv=2,
    )
    assert result["scoring"] == "neg_mean_squared_error"
    assert result["higher_is_better"] is False
    assert result["best_params"]["bias"] == pytest.approx(5.0)
    assert result["best_score"] < 1.0  # bias=5 接近 y；bias=0 的 MSE≈25


@pytest.mark.quick
def test_optuna_live_small_continuous_uses_mse(monkeypatch: pytest.MonkeyPatch) -> None:
    """12 个连续目标不得因 unique<20 误走 accuracy；回归估计器默认 MSE。"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _orig_create = optuna.create_study

    def _create_study(direction: str = "maximize", **kwargs):
        return _orig_create(
            direction=direction,
            sampler=optuna.samplers.GridSampler({"bias": [0.0, 1.0]}),
        )

    monkeypatch.setattr(optuna, "create_study", _create_study)

    import sklearn.model_selection as sms

    _cv = sms.cross_val_score

    def _cv_serial(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return _cv(*args, **kwargs)

    monkeypatch.setattr(sms, "cross_val_score", _cv_serial)

    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(np.linspace(0.0, 1.0, 12))
    assert y.nunique() < 20

    result = HyperparameterOptimizer().bayesian_optimize(
        _BiasRegressor,
        {"bias": {"type": "categorical", "choices": [0.0, 1.0]}},
        X,
        y,
        n_trials=2,
        cv=2,
        n_jobs=1,
    )
    assert result["scoring"] == "neg_mean_squared_error"
    assert result["metric"] == "mse"
    assert result["higher_is_better"] is False
    assert result["best_params"]["bias"] == pytest.approx(1.0)
    assert result["n_completed"] >= 1


@pytest.mark.quick
def test_param_space_rejects_range_tuple() -> None:
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(np.linspace(0.0, 1.0, 12))
    with pytest.raises(ValueError, match="搜索空间"):
        HyperparameterOptimizer().bayesian_optimize(
            _BiasRegressor,
            {"bias": (0.0, 1.0)},
            X,
            y,
            n_trials=1,
            cv=2,
            n_jobs=1,
            show_progress_bar=False,
        )


@pytest.mark.quick
def test_all_fits_failed_prunes_instead_of_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    """全部折失败应剪枝该 trial，合法候选仍能完成优化。"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _orig_create = optuna.create_study

    def _create_study(direction: str = "maximize", **kwargs):
        return _orig_create(
            direction=direction,
            sampler=optuna.samplers.GridSampler({"bias": [0.0, 1.0]}),
        )

    monkeypatch.setattr(optuna, "create_study", _create_study)

    import sklearn.model_selection as sms

    _cv = sms.cross_val_score
    calls = {"n": 0}

    def _cv_flaky(*args, **kwargs):
        kwargs["n_jobs"] = 1
        calls["n"] += 1
        model = args[0]
        if float(getattr(model, "bias", 0.0)) == 0.0:
            raise ValueError("All the 2 fits failed. It is very likely that your model is misconfigured.")
        return _cv(*args, **kwargs)

    monkeypatch.setattr(sms, "cross_val_score", _cv_flaky)

    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(np.linspace(0.0, 1.0, 12))
    result = HyperparameterOptimizer().bayesian_optimize(
        _BiasRegressor,
        {"bias": {"type": "categorical", "choices": [0.0, 1.0]}},
        X,
        y,
        n_trials=2,
        cv=2,
        n_jobs=1,
        show_progress_bar=False,
    )
    assert result["best_params"]["bias"] == pytest.approx(1.0)
    assert result["n_completed"] >= 1
    assert calls["n"] >= 2


@pytest.mark.quick
def test_bayesian_optimize_explicit_scoring_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 scoring 优先于估计器/目标推断。"""
    from types import ModuleType, SimpleNamespace
    import sys
    from unittest.mock import patch

    class _Toy(BaseEstimator, RegressorMixin):
        def __init__(self, k: int = 1):
            self.k = k

        def fit(self, X, y):  # noqa: ANN001, N803
            return self

        def predict(self, X):  # noqa: ANN001, N803
            return np.zeros(len(X))

    def fake_cv(model, X, y, cv=5, scoring=None, n_jobs=-1, error_score=None):  # noqa: ANN001
        assert scoring == "neg_mean_absolute_error"
        return np.full(2, -float(model.k))

    class _Study:
        def __init__(self, direction: str):
            self.best_params = {"k": 1}
            self.best_value = -1.0
            self.trials = [
                SimpleNamespace(state=SimpleNamespace(name="COMPLETE"), value=-1.0)
            ]

        def optimize(self, objective, n_trials, show_progress_bar=True):  # noqa: ANN001
            trial = SimpleNamespace(
                suggest_categorical=lambda name, choices: choices[0],
                suggest_int=lambda *a, **k: 1,
                suggest_float=lambda *a, **k: 1.0,
            )
            objective(trial)

    fake_optuna = ModuleType("optuna")
    fake_optuna.create_study = lambda direction="maximize": _Study(direction)
    fake_optuna.TrialPruned = type("TrialPruned", (Exception,), {})
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(np.linspace(0.0, 1.0, 12))
    with patch("sklearn.model_selection.cross_val_score", side_effect=fake_cv):
        result = HyperparameterOptimizer().bayesian_optimize(
            _Toy,
            {"k": {"type": "categorical", "choices": [1, 2]}},
            X,
            y,
            n_trials=1,
            cv=2,
            scoring="neg_mean_absolute_error",
        )
    assert result["scoring"] == "neg_mean_absolute_error"
    assert result["best_score"] == pytest.approx(1.0)
