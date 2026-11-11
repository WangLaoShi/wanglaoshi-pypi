"""Tutorial 10 smoke: _testing registry / context / floors."""

from __future__ import annotations

import tempfile
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from wanglaoshi.runtime import ready
from wanglaoshi._testing import (
    SCORE_FLOORS,
    DataRegistry,
    ExperimentContext,
    assert_above_floor,
    assert_close,
)


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    ctx = ExperimentContext(random_seed=42, test_size=0.3, notes="tutorial-10")
    Xtr, Xte, ytr, yte = ctx.split(X, y, stratify=True)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(Xtr, ytr)
    acc = float(model.score(Xte, yte))
    assert_above_floor("iris_accuracy", acc)
    assert_close(acc, acc, atol=0, rtol=0)
    assert "iris_accuracy" in SCORE_FLOORS
    assert "python" in ctx.environment()

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "iris.csv"
        iris.frame.to_csv(path, index=False)
        reg = DataRegistry(Path(td) / "ledger.json")
        asset = reg.register(
            "iris",
            path,
            source="sklearn",
            license_or_terms="BSD",
            version="1.0",
        )
        assert reg.verify("iris") is True
        assert asset.sha256
    print("10_testing_repro OK", round(acc, 4), asset.sha256[:12])


if __name__ == "__main__":
    main()
