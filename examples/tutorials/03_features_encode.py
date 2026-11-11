"""Tutorial 03 smoke: features.AdvancedFeatureEngineer encodings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wanglaoshi.runtime import ready
from wanglaoshi.features import AdvancedFeatureEngineer


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    rng = np.random.default_rng(42)
    n = 400
    cats = np.array([f"job_{i}" for i in range(40)])
    job = rng.choice(cats, size=n)
    base = pd.Series(job).str.extract(r"(\d+)")[0].astype(int) / 40.0
    y = (rng.random(n) < (0.2 + 0.6 * base)).astype(int)

    train = pd.DataFrame({"job": job[:300], "y": y[:300]})
    valid = pd.DataFrame({"job": job[300:]})

    eng = AdvancedFeatureEngineer()
    train_te = eng.target_encode(train, "job", "y", smoothing=5.0)
    assert "job_target_encoded" in train_te.columns

    train_freq = eng.frequency_encode(train[["job"]].copy(), "job")
    assert any("frequency" in c for c in train_freq.columns)

    te_map = train_te.groupby("job")["job_target_encoded"].first()
    valid = valid.copy()
    valid["job_te"] = valid["job"].map(te_map)
    assert valid["job_te"].notna().any()
    print("03_features_encode OK", train_te.shape, valid.shape)


if __name__ == "__main__":
    main()
