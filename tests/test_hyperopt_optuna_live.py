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
