"""Handbook 05 smoke: ml facade on breast cancer."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
from sklearn.datasets import load_breast_cancer

from wanglaoshi.ml import (
    DLModel,
    DataPreprocessor,
    FeatureEngineer,
    MLModel,
    ModelEvaluator,
    NeuralNetwork,
    TORCH_AVAILABLE,
    example_usage,
)
from wanglaoshi.runtime import ready


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    df = load_breast_cancer(as_frame=True).frame.copy()
    rng = np.random.default_rng(0)
    df.loc[rng.random(len(df)) < 0.02, df.columns[0]] = np.nan
    y = df.pop("target")

    pre = DataPreprocessor()
    df = pre.handle_missing_values(df, method="mean")
    df = pre.scale_features(df, list(df.columns))

    fe = FeatureEngineer()
    df2 = fe.create_polynomial_features(df.iloc[:, :2].copy(), list(df.columns[:2]), degree=2)
    df2 = fe.create_interaction_features(df2, list(df.columns[:2]))
    assert df2.shape[1] >= 2

    model = MLModel("logistic")
    model.train(df, y, test_size=0.25)
    metrics = model.evaluate()
    assert metrics
    pred = model.predict(model.X_test)

    ModelEvaluator().plot_confusion_matrix(model.y_test, pred)
    ModelEvaluator().plot_learning_curve(
        np.array([50, 100, 150]),
        np.array([[0.9, 0.91], [0.93, 0.94], [0.95, 0.96]]),
        np.array([[0.88, 0.89], [0.91, 0.92], [0.93, 0.94]]),
    )

    assert DLModel is not None and NeuralNetwork is not None
    example_usage()
    print("05_ml_guide OK", TORCH_AVAILABLE, metrics)


if __name__ == "__main__":
    main()
