"""无监督异常检测薄封装（P1）。对齐 E18。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """IsolationForest 封装：fit → score / predict。"""

    def __init__(
        self,
        *,
        contamination: Union[str, float] = "auto",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_: Optional[IsolationForest] = None
        self.feature_names_: Optional[List[str]] = None
        self.n_features_: Optional[int] = None

    def _to_matrix(self, X: Union[pd.DataFrame, np.ndarray], *, fitting: bool) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if fitting:
                self.feature_names_ = list(X.columns)
                self.n_features_ = int(X.shape[1])
                return X.to_numpy(dtype=float, copy=True)
            if self.feature_names_ is None:
                # 训练时用的是 ndarray
                if self.n_features_ is not None and X.shape[1] != self.n_features_:
                    raise ValueError(
                        f"feature width mismatch: expected {self.n_features_}, got {X.shape[1]}"
                    )
                return X.to_numpy(dtype=float, copy=True)
            missing = [c for c in self.feature_names_ if c not in X.columns]
            if missing:
                raise ValueError(f"missing columns for scoring: {missing}")
            # 按训练列名对齐，拒绝静默列顺序错位
            return X.loc[:, self.feature_names_].to_numpy(dtype=float, copy=True)

        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be 2-d")
        if fitting:
            self.feature_names_ = None
            self.n_features_ = int(arr.shape[1])
            return arr
        if self.feature_names_ is not None:
            raise ValueError(
                "model was fit on a DataFrame; pass a DataFrame with the same columns"
            )
        if self.n_features_ is not None and arr.shape[1] != self.n_features_:
            raise ValueError(
                f"feature width mismatch: expected {self.n_features_}, got {arr.shape[1]}"
            )
        return arr

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "AnomalyDetector":
        mat = self._to_matrix(X, fitting=True)
        self.model_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.model_.fit(mat)
        return self

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        # sklearn: higher score = more normal; flip for "anomaly score"
        return -self.model_.score_samples(self._to_matrix(X, fitting=False))

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """返回 1=异常, 0=正常。"""
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        raw = self.model_.predict(self._to_matrix(X, fitting=False))  # -1 anomaly
        return (raw == -1).astype(int)

    def fit_predict_summary(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_injected: Optional[Sequence] = None,
    ) -> Dict[str, Any]:
        self.fit(X)
        yhat = self.predict(X)
        out: Dict[str, Any] = {
            "n": int(len(yhat)),
            "n_flagged": int(yhat.sum()),
            "flag_rate": float(yhat.mean()),
            "predictions": yhat,
            "anomaly_scores": self.score_samples(X),
        }
        if y_injected is not None:
            from sklearn.metrics import precision_recall_fscore_support

            yt = np.asarray(y_injected).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(yt, yhat, average="binary", zero_division=0)
            out.update({"precision": float(p), "recall": float(r), "f1": float(f1)})
        return out
