"""CompetitionTools：基线对比、CV、集成、特征重要性烟测。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from wanglaoshi.CompetitionTools import (
    CrossValidator,
    FeatureImportanceAnalyzer,
    ModelEnsemble,
    QuickBaseline,
)


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=60),
            "f2": rng.normal(size=60),
            "f3": rng.integers(0, 3, size=60),
        }
    )
    y = pd.Series((X["f1"] + 0.5 * X["f2"] > 0).astype(int), name="y")
    return X, y


@pytest.mark.quick
def test_quick_baseline_compare_models_table() -> None:
    baseline = QuickBaseline()
    assert baseline._sklearn_available is True
    results = {
        "Logistic": {"test_accuracy": 0.91, "test_f1": 0.88},
        "Tree": {"test_accuracy": 0.80, "test_f1": 0.77},
        "RF": {"cv_mean": 0.85, "cv_std": 0.02},
    }
    table = baseline.compare_models(results, top_n=2)
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 3
    assert "Model" in table.columns
    assert table.iloc[0]["Model"] == "Logistic"


@pytest.mark.full
def test_quick_baseline_run_all_models_classification(clf_data) -> None:
    X, y = clf_data
    X_train, X_test = X.iloc[:45], X.iloc[45:]
    y_train, y_test = y.iloc[:45], y.iloc[45:]
    baseline = QuickBaseline()
    results = baseline.run_all_models(
        X_train,
        y_train,
        X_test,
        y_test,
        task_type="classification",
    )
    assert len(results) >= 2
    assert any("test_accuracy" in r for r in results.values())
    table = baseline.compare_models(results)
    assert not table.empty


@pytest.mark.quick
def test_cross_validator_stratified(clf_data) -> None:
    X, y = clf_data
    model = LogisticRegression(max_iter=500, random_state=42)
    result = CrossValidator().stratified_cv(model, X, y, n_splits=3)
    assert result["cv_type"] == "stratified"
    assert len(result["scores"]) == 3
    assert 0.0 <= float(result["mean"]) <= 1.0


@pytest.mark.quick
def test_cross_validator_time_series(clf_data) -> None:
    X, y = clf_data
    model = DecisionTreeClassifier(random_state=42)
    result = CrossValidator().time_series_cv(model, X, y, n_splits=3)
    assert result["cv_type"] == "time_series"
    assert len(result["scores"]) == 3


@pytest.mark.quick
def test_model_ensemble_voting_hard(clf_data) -> None:
    X, y = clf_data
    models = [
        LogisticRegression(max_iter=400, random_state=42),
        DecisionTreeClassifier(max_depth=3, random_state=42),
    ]
    ensemble = ModelEnsemble().voting(models, X, y, voting="hard")
    preds = ensemble.predict(X.iloc[:5])
    assert len(preds) == 5


@pytest.mark.quick
def test_feature_importance_builtin(clf_data) -> None:
    X, y = clf_data
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    importance = FeatureImportanceAnalyzer().calculate_importance(model, X, y, method="builtin")
    assert isinstance(importance, pd.DataFrame)
    assert len(importance) == X.shape[1]
    assert set(importance["feature"]) == set(X.columns)


@pytest.mark.quick
def test_feature_importance_permutation_small(clf_data) -> None:
    X, y = clf_data
    model = DecisionTreeClassifier(random_state=42).fit(X, y)
    importance = FeatureImportanceAnalyzer().calculate_importance(
        model, X, y, method="permutation", n_repeats=2
    )
    assert "importance_mean" in importance.columns
    assert len(importance) == X.shape[1]
