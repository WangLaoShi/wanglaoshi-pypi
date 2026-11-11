"""提交 ID 前导零、非法概率、文本词表复用与缺失文档。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wanglaoshi.CompetitionTools import SubmissionGenerator
from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer


@pytest.mark.quick
def test_submission_preserves_leading_zero_ids(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    sample_path.write_text("Id,target\n001,0\n002,0\n", encoding="utf-8")
    out = tmp_path / "sub.csv"
    gen = SubmissionGenerator()
    df = gen.create_submission([0, 1], str(sample_path), str(out))
    assert df["Id"].tolist() == ["001", "002"]
    assert out.read_text(encoding="utf-8").splitlines()[1].startswith("001,")
    ok = gen.validate_submission(str(out), str(sample_path))
    assert ok["is_valid"] is True

    # 丢失前导零的文件应校验失败
    bad = tmp_path / "bad.csv"
    bad.write_text("Id,target\n1,0\n2,1\n", encoding="utf-8")
    result = gen.validate_submission(str(bad), str(sample_path))
    assert result["is_valid"] is False
    assert any("ID" in e for e in result["errors"])


@pytest.mark.quick
def test_ensemble_preserves_leading_zero_ids(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("Id,target\n001,0.1\n002,0.2\n", encoding="utf-8")
    b.write_text("Id,target\n001,0.3\n002,0.4\n", encoding="utf-8")
    out = tmp_path / "ens.csv"
    ens = SubmissionGenerator().ensemble_submissions(
        [str(a), str(b)], weights=[0.5, 0.5], output_path=str(out)
    )
    assert ens["Id"].tolist() == ["001", "002"]
    assert "001," in out.read_text(encoding="utf-8")


@pytest.mark.quick
def test_proba_rejects_nan_and_out_of_range(tmp_path: Path) -> None:
    sample = pd.DataFrame({"Id": ["1", "2"], "target": ["a", "a"]})
    sample_path = tmp_path / "sample.csv"
    sample.to_csv(sample_path, index=False)
    gen = SubmissionGenerator()
    with pytest.raises(ValueError, match="finite"):
        gen.create_submission(
            np.array([[np.nan, 0.9], [0.1, 0.9]]),
            str(sample_path),
            str(tmp_path / "p1.csv"),
            prediction_type="proba",
            class_labels=["a", "b"],
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]|sum"):
        gen.create_submission(
            np.array([[-2.0, 3.0], [0.2, 0.8]]),
            str(sample_path),
            str(tmp_path / "p2.csv"),
            prediction_type="proba",
            class_labels=["a", "b"],
        )


@pytest.mark.quick
def test_count_vectorizer_reuses_vocabulary_and_handles_missing() -> None:
    fe = AdvancedFeatureEngineer()
    train = pd.DataFrame({"text": ["apple banana", "apple"]})
    key = fe.fit_text_vectorizer(train, "text", method="count", max_features=20)
    out_train = fe.transform_text_features(train, key=key)
    apple_cols = [c for c in out_train.columns if c.startswith("text_count_")]
    # 列名应含词项，而非仅编号 text_count_0
    assert any(c.endswith("_apple") for c in apple_cols)

    zebra = pd.DataFrame({"text": ["zebra", "zebra"]})
    out_zebra = fe.transform_text_features(zebra, key=key)
    # 复用词表：列集合不变；未见词 zebra 不新增列，计数全零
    zebra_feat = [c for c in out_zebra.columns if c.startswith("text_count_")]
    assert zebra_feat == apple_cols
    assert (out_zebra[zebra_feat].to_numpy() == 0).all()

    missing = pd.DataFrame({"text": [None, "apple"]})
    out_miss = fe.transform_text_features(missing, key=key)
    miss_cols = [c for c in out_miss.columns if c.startswith("text_count_")]
    # 缺失文档全零；不应出现 "none" 词列
    assert not any("none" in c.lower() for c in miss_cols)
    assert (out_miss.loc[out_miss.index[0], miss_cols] == 0).all()
    apple_col = [c for c in miss_cols if c.endswith("_apple")][0]
    assert out_miss.loc[out_miss.index[1], apple_col] >= 1
