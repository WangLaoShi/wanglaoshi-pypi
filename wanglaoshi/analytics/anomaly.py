"""无监督异常检测薄封装（P1）。对齐 E18。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

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

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "AnomalyDetector":
        self.model_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.model_.fit(np.asarray(X))
        return self

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        # sklearn: higher score = more normal; flip for "anomaly score"
        return -self.model_.score_samples(np.asarray(X))

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """返回 1=异常, 0=正常。"""
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        raw = self.model_.predict(np.asarray(X))  # -1 anomaly
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
