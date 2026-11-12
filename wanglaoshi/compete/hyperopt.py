"""Compete: 超参数优化（从 CompetitionTools 拆出）。"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from wanglaoshi.compete.cv import resolve_cv_scoring

logger = logging.getLogger(__name__)

_SEARCH_TYPES = {"categorical", "int", "float"}


def _task_type_from_estimator(model_class: Any) -> Optional[str]:
    """按估计器类型推断任务；无法判断时返回 None（再交给 y 的 auto）。"""
    et = getattr(model_class, "_estimator_type", None)
    if et == "regressor":
        return "regression"
    if et == "classifier":
        return "classification"
    try:
        from sklearn.base import ClassifierMixin, RegressorMixin

        if isinstance(model_class, type):
            is_reg = issubclass(model_class, RegressorMixin)
            is_clf = issubclass(model_class, ClassifierMixin)
            if is_reg and not is_clf:
                return "regression"
            if is_clf and not is_reg:
                return "classification"
    except Exception:  # noqa: BLE001
        pass
    return None


def _validate_param_space(param_space: Dict[str, Any]) -> None:
    """拒绝易误用的非标量固定值（如 (low, high) 元组被当成 C=）。"""
    for name, cfg in param_space.items():
        if isinstance(cfg, dict):
            t = cfg.get("type")
            if t not in _SEARCH_TYPES:
                raise ValueError(
                    f"param_space[{name!r}] dict 需含 type in {_SEARCH_TYPES}, got {t!r}"
                )
            continue
        if isinstance(cfg, (list, tuple, set, dict)):
            raise ValueError(
                f"param_space[{name!r}]={cfg!r} 不能直接作为模型参数；"
                f"请用 {{'type': 'float'|'int'|'categorical', ...}} 描述搜索空间"
            )


def _sample_params(trial: Any, param_space: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for param_name, param_config in param_space.items():
        if isinstance(param_config, dict):
            if param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            elif param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
        else:
            params[param_name] = param_config
    return params


class HyperparameterOptimizer:
    """超参数优化器（贝叶斯 / 自动调参）。"""

    def __init__(self):
        pass

    def bayesian_optimize(
        self,
        model_class: Any,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        cv: int = 5,
        scoring: Optional[str] = None,
        task_type: str = "auto",
        n_jobs: int = -1,
        show_progress_bar: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        贝叶斯优化超参数（Optuna）。

        默认评分优先看估计器类型（回归器用 MSE，分类器用 accuracy），
        再回退到目标列 dtype；可用 ``scoring`` / ``task_type`` 显式覆盖。
        回归默认 maximize ``neg_mean_squared_error``（展示为正 MSE）。
        """
        try:
            import optuna
        except ImportError:
            logger.warning("Optuna未安装，无法使用贝叶斯优化")
            raise ImportError("需要安装Optuna: pip install optuna") from None

        from sklearn.model_selection import cross_val_score

        _validate_param_space(param_space)

        y_s = y if isinstance(y, pd.Series) else pd.Series(y)
        resolved_task = task_type
        if scoring is None and resolved_task == "auto":
            inferred = _task_type_from_estimator(model_class)
            if inferred is not None:
                resolved_task = inferred

        sklearn_scoring, display_metric, higher_is_better = resolve_cv_scoring(
            y_s, scoring=scoring, task_type=resolved_task
        )

        if show_progress_bar is None:
            show_progress_bar = bool(getattr(sys.stderr, "isatty", lambda: False)())

        def objective(trial):
            params = _sample_params(trial, param_space)
            model = model_class(**params)
            try:
                scores = cross_val_score(
                    model,
                    X,
                    y_s,
                    cv=cv,
                    scoring=sklearn_scoring,
                    n_jobs=n_jobs,
                    error_score=np.nan,
                )
            except ValueError as exc:
                # sklearn：全部折失败时即使 error_score=nan 仍可能抛 ValueError
                msg = str(exc).lower()
                if "fits failed" in msg or "all the" in msg:
                    raise optuna.TrialPruned() from exc
                raise
            mean = float(np.nanmean(np.asarray(scores, dtype=float)))
            if not np.isfinite(mean):
                raise optuna.TrialPruned()
            return mean

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=show_progress_bar,
        )

        completed = [
            t
            for t in study.trials
            if t.state.name == "COMPLETE" and t.value is not None and np.isfinite(t.value)
        ]
        if not completed:
            raise RuntimeError(
                f"No trials are completed yet under scoring={sklearn_scoring!r} "
                f"(task_type={resolved_task!r}). "
                "Pass scoring=... or task_type='regression'/'classification' explicitly."
            )

        best_objective = float(study.best_value)
        if sklearn_scoring.startswith("neg_"):
            display_score = float(-best_objective)
        else:
            display_score = best_objective

        result = {
            "best_params": study.best_params,
            "best_score": display_score,
            "best_objective": best_objective,
            "scoring": sklearn_scoring,
            "metric": display_metric,
            "higher_is_better": higher_is_better,
            "task_type": resolved_task,
            "n_trials": n_trials,
            "n_completed": len(completed),
        }

        logger.info(
            f"贝叶斯优化完成，最佳{display_metric}: {result['best_score']:.4f} "
            f"(objective={best_objective:.4f})"
        )
        logger.info(f"最佳参数: {result['best_params']}")
        return result

    def auto_tune(
        self,
        model_class: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "auto",
        scoring: Optional[str] = None,
        task_type: str = "auto",
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """基于模型类型选择默认参数空间并优化。

        ``model_type`` 为 ``regression`` / ``classification`` 时同时作为默认
        ``task_type``（若调用方未显式传入 task_type）。
        """
        resolved_task = task_type
        if resolved_task == "auto" and model_type in {"regression", "classification"}:
            resolved_task = model_type

        if "RandomForest" in str(model_class):
            param_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int", "low": 3, "high": 20},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
            }
        elif "XGB" in str(model_class) or "XGBoost" in str(model_class):
            param_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log": True,
                },
            }
        else:
            param_space = {
                "C": {"type": "float", "low": 0.1, "high": 100, "log": True},
                "gamma": {"type": "float", "low": 0.001, "high": 1, "log": True},
            }

        return self.bayesian_optimize(
            model_class,
            param_space,
            X,
            y,
            n_trials=n_trials,
            scoring=scoring,
            task_type=resolved_task,
        )
