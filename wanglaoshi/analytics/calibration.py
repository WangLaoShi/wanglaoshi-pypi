"""概率校准报表（P1）。对齐 E04 增补。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


def expected_calibration_error(
    y_true: Sequence,
    y_prob: Sequence,
    *,
    n_bins: int = 10,
) -> float:
    """固定等宽分箱 ECE。

    要求：非空、等长、``y_true ∈ {0,1}``、``y_prob`` 有限且落在 ``[0, 1]``。
    无效概率不得被静默忽略成「完美校准」(ECE=0)。
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if y_true.size == 0 or y_prob.size == 0:
        raise ValueError("y_true and y_prob must be non-empty")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob length mismatch")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob must be finite")
    if np.any(y_prob < 0.0) or np.any(y_prob > 1.0):
        raise ValueError("y_prob must be in [0, 1]")
    if not np.isfinite(y_true).all():
        raise ValueError("y_true must be finite")
    uniq = set(np.unique(y_true).tolist())
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError("y_true must be binary labels in {0, 1}")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


class CalibrationReport:
    """Brier / ECE / 可靠性曲线数据。"""

    @staticmethod
    def summarize(
        y_true: Sequence,
        y_prob: Sequence,
        *,
        n_bins: int = 10,
        name: str = "model",
    ) -> Dict[str, Any]:
        from sklearn.metrics import brier_score_loss
        from sklearn.calibration import calibration_curve

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob, dtype=float)
        frac, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        return {
            "name": name,
            "brier": float(brier_score_loss(y_true, y_prob)),
            "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
            "n_bins": n_bins,
            "reliability": pd.DataFrame({"mean_predicted": mean_pred, "frac_positive": frac}),
        }

    @staticmethod
    def compare(
        y_true: Sequence,
        prob_map: Dict[str, Sequence],
        *,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        rows = []
        for name, p in prob_map.items():
            s = CalibrationReport.summarize(y_true, p, n_bins=n_bins, name=name)
            rows.append({"name": name, "brier": s["brier"], "ece": s["ece"], "n_bins": n_bins})
        return pd.DataFrame(rows)
