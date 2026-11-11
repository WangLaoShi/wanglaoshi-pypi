"""Tutorial 13 smoke: ensemble / importance / tracker / hyperopt."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from wanglaoshi.compete import (
    ExperimentTracker,
    FeatureImportanceAnalyzer,
    HyperparameterOptimizer,
    ModelEnsemble,
)
from wanglaoshi.runtime import ready


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=80),
            "f2": rng.normal(size=80),
            "f3": rng.integers(0, 3, 80),
        }
    )
    y = pd.Series((X["f1"] + 0.4 * X["f2"] > 0).astype(int))

    ens = ModelEnsemble()
    voter = ens.voting(
        [
            LogisticRegression(max_iter=400, random_state=42),
            DecisionTreeClassifier(max_depth=3, random_state=42),
        ],
        X,
        y,
        voting="hard",
    )
    assert len(voter.predict(X.iloc[:3])) == 3

    stack_bundle = ens.stacking(
        [
            LogisticRegression(max_iter=400, random_state=1),
            DecisionTreeClassifier(max_depth=2, random_state=1),
        ],
        LogisticRegression(max_iter=400, random_state=2),
        X,
        y,
        cv=3,
    )
    assert "meta_model" in stack_bundle

    blender = ens.blending(
        [
            LogisticRegression(max_iter=400, random_state=3),
            DecisionTreeClassifier(max_depth=2, random_state=3),
        ],
        X.iloc[:60],
        y.iloc[:60],
        X.iloc[60:],
        y.iloc[60:],
    )
    assert len(blender["predict"](X.iloc[:3])) == 3

    rf = RandomForestClassifier(n_estimators=30, random_state=42).fit(X, y)
    analyzer = FeatureImportanceAnalyzer()
    imp = analyzer.calculate_importance(rf, X, y, method="builtin")
    assert len(imp) == X.shape[1]
    analyzer.plot_importance(imp, top_n=3)

    with tempfile.TemporaryDirectory() as td:
        tracker = ExperimentTracker(log_file=str(Path(td) / "experiments.json"))
        tracker.log_experiment(
            experiment_name="tut13",
            model_name="rf",
            features=list(X.columns),
            params={"n_estimators": 30},
            score=float(rf.score(X, y)),
            metrics={"acc": float(rf.score(X, y))},
        )
        assert len(tracker.get_experiment_history()) >= 1
        assert tracker.compare_experiments() is not None

        opt = HyperparameterOptimizer()
        try:
            best = opt.bayesian_optimize(
                LogisticRegression,
                {"C": {"type": "float", "low": 0.1, "high": 10.0, "log": True}},
                X,
                y,
                n_trials=2,
                cv=2,
            )
            assert "best_params" in best
            assert callable(opt.auto_tune)
        except ImportError:
            assert hasattr(opt, "bayesian_optimize") and hasattr(opt, "auto_tune")
            print("13 skip optuna run")
    print("13_compete_advanced OK")


if __name__ == "__main__":
    main()
