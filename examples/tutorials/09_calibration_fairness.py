"""Tutorial 09 smoke: CalibrationReport + GroupMetricsReport."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from wanglaoshi.runtime import ready
from wanglaoshi.analytics import CalibrationReport, GroupMetricsReport


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    summary = CalibrationReport.summarize(yte, proba, name="logistic")
    assert "ece" in summary and "brier" in summary
    flat = np.full(len(yte), 0.5)
    cmp = CalibrationReport.compare(yte, {"logistic": proba, "flat": flat})
    assert set(cmp["name"]) == {"logistic", "flat"}

    rng = np.random.default_rng(0)
    sex = rng.choice(["F", "M"], size=len(yte))
    audit = GroupMetricsReport.audit(yte, pred, {"sex": sex})
    assert len(audit) >= 1
    assert {"n", "tpr", "fpr"}.issubset(audit.columns)
    print("09_calibration_fairness OK", summary["ece"], len(audit))


if __name__ == "__main__":
    main()
