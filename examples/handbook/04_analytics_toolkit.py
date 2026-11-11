"""Handbook 04 smoke: analytics toolkit collage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from wanglaoshi.analytics import (
    AnomalyDetector,
    CalibrationReport,
    ExperimentDesign,
    GroupMetricsReport,
    RankingMetrics,
    StatsToolkit,
    SurvivalAnalyzer,
    TimeSeriesTools,
    expected_calibration_error,
)
from wanglaoshi.runtime import ready


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    rng = np.random.default_rng(0)

    a, b = rng.normal(0, 1, 60), rng.normal(0.3, 1, 60)
    assert "pvalue" in StatsToolkit.ttest_ind(a, b)
    assert "pvalue" in StatsToolkit.chi2_independence([[10, 5], [6, 12]])
    assert "pvalue" in StatsToolkit.anova_oneway(a, b, rng.normal(0.1, 1, 60))
    assert ExperimentDesign.sample_size_two_proportion(0.1, 0.03)
    assert "pvalue" in ExperimentDesign.two_proportion_test(20, 200, 30, 200)

    t = np.arange(56)
    s = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 7)
    df = pd.DataFrame({"y": s})
    assert "y_lag1" in TimeSeriesTools.safe_lag(df, "y", lags=(1, 2), assume_row_order=True).columns
    assert TimeSeriesTools.safe_rolling_mean(df, "y", window=3, assume_row_order=True) is not None
    assert "pvalue" in TimeSeriesTools.adf_report(df["y"])
    assert "resid_std" in TimeSeriesTools.decompose_summary(df["y"], period=7)

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000, random_state=42).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    assert CalibrationReport.summarize(yte, proba)
    assert expected_calibration_error(yte, proba) >= 0
    pred = (proba >= 0.5).astype(int)
    sex = rng.choice(["F", "M"], size=len(yte))
    audit = GroupMetricsReport.audit(yte, pred, {"sex": sex})
    assert len(audit) >= 1

    Xab = np.vstack([rng.normal(0, 1, (80, 3)), rng.normal(6, 1, (8, 3))])
    summary = AnomalyDetector(contamination=0.1, random_state=1).fit_predict_summary(Xab)
    assert summary["n_flagged"] >= 1
    assert RankingMetrics.ndcg_at_k([1, 9, 2], {1, 2}, k=3) > 0

    surv = pd.DataFrame(
        {"t": np.arange(1, 9), "e": [1, 0, 1, 1, 0, 1, 0, 1], "g": list("AAAABBBB")}
    )
    km = SurvivalAnalyzer.km(surv["t"], surv["e"])
    assert "median_survival" in km
    by = SurvivalAnalyzer.km_by_group(surv, "t", "e", "g")
    assert list(by["groups"])
    print("04_analytics_toolkit OK", summary["n_flagged"])


if __name__ == "__main__":
    main()
