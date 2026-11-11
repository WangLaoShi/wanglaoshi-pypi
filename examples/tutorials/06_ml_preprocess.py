"""Tutorial 06 smoke: ml.DataPreprocessor + MLModel on breast cancer."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer

from wanglaoshi.runtime import ready
from wanglaoshi.ml import DataPreprocessor, MLModel, TORCH_AVAILABLE


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    assert isinstance(TORCH_AVAILABLE, bool)

    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    rng = np.random.default_rng(0)
    mask = rng.random(len(df)) < 0.02
    df.loc[mask, df.columns[0]] = np.nan

    y = df.pop("target")
    pre = DataPreprocessor()
    df2 = pre.handle_missing_values(df, method="mean")
    df2 = pre.scale_features(df2, list(df2.columns))

    model = MLModel("logistic")
    model.train(df2, y, test_size=0.25)
    metrics = model.evaluate()
    assert metrics
    print("06_ml_preprocess OK", metrics, "torch=", TORCH_AVAILABLE)


if __name__ == "__main__":
    main()
