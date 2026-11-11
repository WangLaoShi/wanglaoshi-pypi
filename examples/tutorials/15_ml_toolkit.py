"""Tutorial 15 smoke: FeatureEngineer / ModelEvaluator / DL stubs / example_usage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

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
    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0, 5.0], "B": ["a", "b", "a", "c", "b"]})
    pre = DataPreprocessor()
    df_num = pre.handle_missing_values(df[["A"]], method="mean")
    df = df.copy()
    df["A"] = df_num["A"]
    df = pre.encode_categorical(df, ["B"])

    fe = FeatureEngineer()
    df = fe.create_polynomial_features(df, ["A"], degree=2)
    df = fe.create_interaction_features(df, ["A", "A_power_2"])

    X = df[["A", "A_power_2"]]
    y = df["B"]
    model = MLModel("logistic")
    model.train(X, y, test_size=0.4)
    pred = model.predict(model.X_test)

    ev = ModelEvaluator()
    ev.plot_confusion_matrix(model.y_test, pred)
    ev.plot_learning_curve(
        np.array([2, 3, 4]),
        np.array([[0.7, 0.72], [0.75, 0.76], [0.8, 0.81]]),
        np.array([[0.65, 0.66], [0.7, 0.71], [0.74, 0.75]]),
    )

    assert DLModel is not None and NeuralNetwork is not None
    if TORCH_AVAILABLE:
        _ = NeuralNetwork(2, 4, 2)
        _ = DLModel(2, 4, 2)

    example_usage()
    print("15_ml_toolkit OK", TORCH_AVAILABLE)


if __name__ == "__main__":
    main()
