"""Tutorial 14 smoke: anomaly / ranking / survival."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wanglaoshi.analytics import AnomalyDetector, RankingMetrics, SurvivalAnalyzer
from wanglaoshi.runtime import ready


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (80, 3)), rng.normal(5, 1, (10, 3))])
    y_inj = np.array([0] * 80 + [1] * 10)
    det = AnomalyDetector(contamination=10 / 90, random_state=1)
    summary = det.fit_predict_summary(X, y_injected=y_inj)
    assert summary["n_flagged"] >= 1
    assert det.score_samples(X[:5]) is not None
    assert len(det.predict(X[:5])) == 5

    assert RankingMetrics.ndcg_at_k([1, 2, 3], {1}, k=3) == 1.0
    assert RankingMetrics.hit_rate_at_k([9, 1], {1}, k=2) == 1.0
    assert RankingMetrics.dcg_at_k([1.0, 1.0, 0.0], k=3) > 0

    df_surv = pd.DataFrame(
        {
            "t": [1, 2, 3, 4, 5, 6, 7, 8],
            "e": [1, 0, 1, 1, 0, 1, 0, 1],
            "g": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )
    km = SurvivalAnalyzer.km(df_surv["t"], df_surv["e"], name="all")
    assert "survival" in km
    by = SurvivalAnalyzer.km_by_group(df_surv, "t", "e", "g")
    assert "groups" in by and len(by["groups"]) == 2
    print("14_analytics_extra OK", summary["n_flagged"])


if __name__ == "__main__":
    main()
