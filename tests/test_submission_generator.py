"""SubmissionGenerator 回归测试（对应路线图 B01–B06）。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wanglaoshi.CompetitionTools import SubmissionGenerator


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def generator() -> SubmissionGenerator:
    return SubmissionGenerator()


def _write_sample(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


@pytest.mark.quick
@pytest.mark.submission
def test_infer_target_column_when_not_named_target(generator, tmp_dir):
    """B01: 模板目标列不是 'target' 时也能推断（如 Survived）。"""
    sample = pd.DataFrame({"PassengerId": [1, 2, 3], "Survived": [0, 1, 0]})
    sample_path = _write_sample(tmp_dir / "sample.csv", sample)
    out = tmp_dir / "sub.csv"

    pred = np.array([0, 1, 1])
    df = generator.create_submission(pred, sample_path, str(out))

    assert list(df.columns) == ["PassengerId", "Survived"]
    assert df["Survived"].tolist() == [0, 1, 1]

    result = generator.validate_submission(str(out), sample_path)
    assert result["is_valid"] is True


@pytest.mark.quick
@pytest.mark.submission
def test_string_labels_validation_uses_categorical_stats(generator, tmp_dir):
    """B02/B04: 字符串标签不应因数值统计失败。"""
    sample = pd.DataFrame({"Id": [10, 11, 12], "Species": ["setosa", "setosa", "virginica"]})
    sample_path = _write_sample(tmp_dir / "sample.csv", sample)
    out = tmp_dir / "sub.csv"

    pred = np.array(["setosa", "versicolor", "virginica"], dtype=object)
    generator.create_submission(pred, sample_path, str(out), prediction_type="label")

    result = generator.validate_submission(str(out), sample_path)
    assert result["is_valid"] is True
    assert result["info"]["Species_stats"]["dtype"] == "categorical"
    assert result["info"]["Species_stats"]["nunique"] == 3


@pytest.mark.quick
@pytest.mark.submission
def test_explicit_wrong_target_column_raises(generator, tmp_dir):
    """B03: 显式错误列名应报错，不静默推断错误。"""
    sample = pd.DataFrame({"Id": [1, 2], "label": [0, 1]})
    sample_path = _write_sample(tmp_dir / "sample.csv", sample)

    with pytest.raises(ValueError, match="target_column"):
        generator.create_submission(
            np.array([0, 1]),
            sample_path,
            str(tmp_dir / "out.csv"),
            target_column="target",
        )


@pytest.mark.quick
@pytest.mark.submission
def test_proba_without_class_labels_does_not_silent_argmax(generator, tmp_dir):
    """B05: 单目标模板 + 概率矩阵时禁止静默 argmax。"""
    sample = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
    sample_path = _write_sample(tmp_dir / "sample.csv", sample)
    proba = np.array([[0.2, 0.8], [0.7, 0.3]])

    with pytest.raises(ValueError, match="禁止静默 argmax|class_labels"):
        generator.create_submission(
            proba,
            sample_path,
            str(tmp_dir / "out.csv"),
            prediction_type="proba",
        )


@pytest.mark.quick
@pytest.mark.submission
def test_proba_with_class_labels_maps_argmax(generator, tmp_dir):
    """B05: 提供 class_labels 时可安全映射。"""
    sample = pd.DataFrame({"id": [1, 2], "species": ["a", "b"]})
    sample_path = _write_sample(tmp_dir / "sample.csv", sample)
    out = tmp_dir / "out.csv"
    proba = np.array([[0.2, 0.8], [0.7, 0.3]])

    df = generator.create_submission(
        proba,
        sample_path,
        str(out),
        prediction_type="proba",
        class_labels=["a", "b"],
    )
    assert df["species"].tolist() == ["b", "a"]


@pytest.mark.quick
@pytest.mark.submission
def test_multioutput_column_count_must_match(generator, tmp_dir):
    """B05: 多输出列数必须与模板一致。"""
    sample = pd.DataFrame({"id": [1, 2], "y1": [0.1, 0.2], "y2": [0.3, 0.4]})
    sample_path = _write_sample(tmp_dir / "sample.csv", sample)

    with pytest.raises(ValueError, match="multioutput|不一致"):
        generator.create_submission(
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            sample_path,
            str(tmp_dir / "out.csv"),
            prediction_type="multioutput",
        )


@pytest.mark.quick
@pytest.mark.submission
def test_length_mismatch_and_empty(generator, tmp_dir):
    """B06: 长度不匹配与空输入。"""
    sample = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})
    sample_path = _write_sample(tmp_dir / "sample.csv", sample)

    with pytest.raises(ValueError, match="不匹配"):
        generator.create_submission(np.array([0, 1]), sample_path, str(tmp_dir / "a.csv"))

    with pytest.raises(ValueError, match="空"):
        generator.create_submission(np.array([]), sample_path, str(tmp_dir / "b.csv"))


@pytest.mark.quick
@pytest.mark.submission
def test_id_mismatch_fails_validation(generator, tmp_dir):
    """B06: ID 不匹配应校验失败。"""
    sample = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
    sample_path = _write_sample(tmp_dir / "sample.csv", sample)
    bad = pd.DataFrame({"id": [9, 8], "target": [0, 1]})
    bad_path = tmp_dir / "bad.csv"
    bad.to_csv(bad_path, index=False)

    result = generator.validate_submission(str(bad_path), sample_path)
    assert result["is_valid"] is False
    assert any("ID" in e for e in result["errors"])
