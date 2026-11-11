"""Handbook 03 smoke: compete toolkit on Titanic."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from wanglaoshi.compete import (
    CrossValidator,
    ExperimentTracker,
    FeatureImportanceAnalyzer,
    LeakageDetector,
    ModelEnsemble,
    QuickBaseline,
    SubmissionGenerator,
)
from wanglaoshi.runtime import ready


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    raw = sns.load_dataset("titanic").copy()
    y = raw["survived"]

    leaky_X = pd.get_dummies(
        raw.drop(
            columns=["survived", "deck", "embark_town", "class", "who", "adult_male"],
            errors="ignore",
        ),
        drop_first=True,
    ).fillna(0)
    hits = LeakageDetector().detect_target_leakage(leaky_X, y, threshold=0.85)
    assert hits is not None

    X = pd.get_dummies(
        raw.drop(
            columns=["survived", "alive", "deck", "embark_town", "class", "who", "adult_male"],
            errors="ignore",
        ),
        drop_first=True,
    ).fillna(0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    qb = QuickBaseline()
    results = qb.run_all_models(Xtr, ytr, Xte, yte, task_type="classification")
    cmp = qb.compare_models(results, top_n=3)
    assert len(cmp) >= 1

    cv = CrossValidator().stratified_cv(
        LogisticRegression(max_iter=500, random_state=42), X, y, n_splits=5
    )
    assert "mean" in cv and "std" in cv

    voter = ModelEnsemble().voting(
        [
            LogisticRegression(max_iter=400, random_state=0),
            DecisionTreeClassifier(max_depth=3, random_state=0),
        ],
        Xtr,
        ytr,
        voting="hard",
    )
    assert len(voter.predict(Xte)) == len(yte)

    rf = RandomForestClassifier(n_estimators=40, random_state=42).fit(Xtr, ytr)
    imp = FeatureImportanceAnalyzer().calculate_importance(rf, Xtr, ytr, method="builtin")
    assert len(imp) == Xtr.shape[1]

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        sample = pd.DataFrame({"Id": np.arange(len(yte)), "Survived": yte.to_numpy()})
        sample_path = td_path / "sample.csv"
        sample.to_csv(sample_path, index=False)
        gen = SubmissionGenerator()
        pred = results[cmp.iloc[0]["Model"]]["predictions"]
        p1 = td_path / "s1.csv"
        p2 = td_path / "s2.csv"
        gen.create_submission(pred, str(sample_path), str(p1), prediction_type="label")
        gen.create_submission(voter.predict(Xte), str(sample_path), str(p2), prediction_type="label")
        val = gen.validate_submission(str(p1), str(sample_path))
        assert val["is_valid"] is True
        ens = gen.ensemble_submissions(
            [str(p1), str(p2)], method="simple_average", output_path=str(td_path / "ens.csv")
        )
        assert len(ens) == len(yte)

        tracker = ExperimentTracker(log_file=str(td_path / "exp.json"))
        tracker.log_experiment(
            "titanic-hb",
            "rf",
            list(X.columns),
            {"n": 40},
            float(rf.score(Xte, yte)),
        )
        hist = tracker.get_experiment_history()
        assert len(hist) >= 1 and float(hist.iloc[0]["score"]) >= 0

    print("03_compete_toolkit OK", cmp.iloc[0]["Model"], cv["mean"])


if __name__ == "__main__":
    main()
