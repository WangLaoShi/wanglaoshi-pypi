"""Compete: 交叉验证（从 CompetitionTools 拆出）。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _is_classification_target(y: pd.Series) -> bool:
    """分类判定：类别/对象，或少量整型标签；连续浮点即使唯一值 <20 也视为回归。"""
    if y.dtype == "object" or getattr(y.dtype, "name", "") == "category":
        return True
    if pd.api.types.is_bool_dtype(y):
        return True
    if pd.api.types.is_integer_dtype(y) and y.nunique(dropna=True) < 20:
        return True
    return False


def resolve_cv_scoring(
    y: pd.Series,
    *,
    scoring: Optional[str] = None,
    task_type: str = "auto",
) -> Tuple[str, str, bool]:
    """返回 (sklearn_scoring, display_metric, higher_is_better)。"""
    if scoring is not None:
        name = str(scoring)
        if name.startswith("neg_"):
            display = name[4:] if name.startswith("neg_") else name
            # neg_mean_squared_error → mse 更友好
            if display == "mean_squared_error":
                display = "mse"
            return name, display, False
        return name, name, True

    tt = task_type
    if tt == "auto":
        tt = "classification" if _is_classification_target(y) else "regression"
    if tt == "classification":
        return "accuracy", "accuracy", True
    if tt == "regression":
        return "neg_mean_squared_error", "mse", False
    raise ValueError("task_type must be 'auto', 'classification', or 'regression'")


def _cross_val_score_or_raise(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: Any,
    scoring: str,
    groups: Any = None,
) -> np.ndarray:
    from sklearn.model_selection import cross_val_score

    try:
        kwargs = dict(cv=cv, scoring=scoring, n_jobs=-1, error_score=np.nan)
        if groups is not None:
            kwargs["groups"] = groups
        return np.asarray(cross_val_score(model, X, y, **kwargs), dtype=float)
    except ValueError as exc:
        msg = str(exc)
        if "fits failed" in msg.lower() or "all the" in msg.lower():
            raise RuntimeError(
                f"all CV folds failed under scoring={scoring!r}: {exc}"
            ) from exc
        raise


def _finalize_cv_scores(
    scores: np.ndarray,
    *,
    sklearn_scoring: str,
    display_metric: str,
    higher_is_better: bool,
    cv_type: str,
) -> Dict[str, Any]:
    arr = np.asarray(scores, dtype=float)
    if sklearn_scoring.startswith("neg_"):
        display = -arr
    else:
        display = arr

    finite = np.isfinite(display)
    if display.size == 0 or not bool(finite.any()):
        raise RuntimeError(
            f"all CV folds failed under scoring={sklearn_scoring!r} (cv_type={cv_type})"
        )
    if not bool(finite.all()):
        logger.warning(
            "部分折失败为 NaN：%s/%s folds",
            int((~finite).sum()),
            int(display.size),
        )

    mean = float(np.nanmean(display))
    std = float(np.nanstd(display))
    result = {
        "scores": display,
        "mean": mean,
        "std": std,
        "cv_type": cv_type,
        "scoring": sklearn_scoring,
        "metric": display_metric,
        "higher_is_better": higher_is_better,
        "n_failed_folds": int((~finite).sum()),
    }
    logger.info(
        f"{cv_type} 交叉验证完成，{display_metric}: {mean:.4f} ± {std:.4f} "
        f"(scoring={sklearn_scoring})"
    )
    return result


class CrossValidator:
    """增强的交叉验证工具：时间序列、分组、分层。"""

    def __init__(self) -> None:
        pass

    def time_series_cv(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        *,
        scoring: Optional[str] = None,
        task_type: str = "auto",
    ) -> Dict[str, Any]:
        """时间序列交叉验证。可显式传 ``scoring`` / ``task_type``。"""
        from sklearn.model_selection import TimeSeriesSplit

        sklearn_scoring, metric, higher = resolve_cv_scoring(
            y, scoring=scoring, task_type=task_type
        )
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = _cross_val_score_or_raise(
            model, X, y, cv=tscv, scoring=sklearn_scoring
        )
        return _finalize_cv_scores(
            scores,
            sklearn_scoring=sklearn_scoring,
            display_metric=metric,
            higher_is_better=higher,
            cv_type="time_series",
        )

    def group_cv(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        n_splits: int = 5,
        *,
        scoring: Optional[str] = None,
        task_type: str = "auto",
    ) -> Dict[str, Any]:
        """分组交叉验证。可显式传 ``scoring`` / ``task_type``。"""
        from sklearn.model_selection import GroupKFold

        sklearn_scoring, metric, higher = resolve_cv_scoring(
            y, scoring=scoring, task_type=task_type
        )
        gkf = GroupKFold(n_splits=n_splits)
        scores = _cross_val_score_or_raise(
            model, X, y, cv=gkf, scoring=sklearn_scoring, groups=groups
        )
        return _finalize_cv_scores(
            scores,
            sklearn_scoring=sklearn_scoring,
            display_metric=metric,
            higher_is_better=higher,
            cv_type="group",
        )

    def stratified_cv(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        *,
        scoring: Optional[str] = None,
        task_type: str = "auto",
    ) -> Dict[str, Any]:
        """分层交叉验证（**仅分类**）。

        连续回归目标请使用 ``time_series_cv`` / ``group_cv``，
        或先离散化标签再调用本方法。
        """
        from sklearn.model_selection import StratifiedKFold

        tt = task_type
        if tt == "auto":
            tt = "classification" if _is_classification_target(y) else "regression"
        if tt == "regression":
            raise ValueError(
                "stratified_cv only supports classification targets; "
                "for regression use time_series_cv/group_cv, or provide discrete labels"
            )
        if not _is_classification_target(y) and scoring is None:
            raise ValueError(
                "stratified_cv requires classification labels (got continuous-like target); "
                "pass discrete y, or use time_series_cv/group_cv for regression"
            )

        sklearn_scoring, metric, higher = resolve_cv_scoring(
            y, scoring=scoring or "accuracy", task_type="classification"
        )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = _cross_val_score_or_raise(
            model, X, y, cv=skf, scoring=sklearn_scoring
        )
        return _finalize_cv_scores(
            scores,
            sklearn_scoring=sklearn_scoring,
            display_metric=metric,
            higher_is_better=higher,
            cv_type="stratified",
        )
