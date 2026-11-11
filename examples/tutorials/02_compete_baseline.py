"""Tutorial 02 smoke: compete QuickBaseline + SubmissionGenerator on Titanic."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from wanglaoshi.runtime import ready
from wanglaoshi.compete import QuickBaseline, SubmissionGenerator


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)

    raw = sns.load_dataset("titanic").copy()
    drop_cols = [
        c
        for c in ("alive", "deck", "embark_town", "class", "who", "adult_male")
        if c in raw.columns
    ]
    df = raw.drop(columns=drop_cols)
    y = df.pop("survived")
    X = pd.get_dummies(df, drop_first=True).fillna(0)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    qb = QuickBaseline()
    results = qb.run_all_models(Xtr, ytr, Xte, yte, task_type="classification")
    cmp = qb.compare_models(results, top_n=3)
    assert len(cmp) >= 1
    pred = results[cmp.iloc[0]["Model"]]["predictions"]

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        sample = pd.DataFrame({"Id": np.arange(len(pred)), "Survived": yte.to_numpy()})
        sample_path = td_path / "sample_submission.csv"
        sample.to_csv(sample_path, index=False)
        out = td_path / "submission.csv"
        gen = SubmissionGenerator()
        gen.create_submission(
            pred, str(sample_path), str(out), prediction_type="label"
        )
        val = gen.validate_submission(str(out), str(sample_path))
        assert val["is_valid"] is True
    print("02_compete_baseline OK", cmp.iloc[0]["Model"])


if __name__ == "__main__":
    main()
