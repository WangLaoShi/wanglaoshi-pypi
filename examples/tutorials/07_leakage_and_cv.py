"""Tutorial 07 smoke: LeakageDetector + CrossValidator on Titanic."""

from __future__ import annotations

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from wanglaoshi.runtime import ready
from wanglaoshi.compete import LeakageDetector, CrossValidator


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    raw = sns.load_dataset("titanic").copy()
    y = raw["survived"]

    leaky = pd.get_dummies(
        raw.drop(
            columns=["survived", "deck", "embark_town", "class", "who", "adult_male"],
            errors="ignore",
        ),
        drop_first=True,
    ).fillna(0)
    hits = LeakageDetector().detect_target_leakage(
        leaky, y, threshold=0.85, method="correlation"
    )
    assert hits, "expected to detect alive-like leakage"

    clean = pd.get_dummies(
        raw.drop(
            columns=[
                "survived",
                "alive",
                "deck",
                "embark_town",
                "class",
                "who",
                "adult_male",
            ],
            errors="ignore",
        ),
        drop_first=True,
    ).fillna(0)
    model = LogisticRegression(max_iter=500, random_state=42)
    result = CrossValidator().stratified_cv(model, clean, y, n_splits=5)
    assert "mean" in result and result["mean"] > 0.5
    print("07_leakage_and_cv OK", len(hits), round(float(result["mean"]), 4))


if __name__ == "__main__":
    main()
