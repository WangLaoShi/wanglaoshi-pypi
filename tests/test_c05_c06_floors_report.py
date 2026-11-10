"""C05 分数下限 + C06 报告章节 / 图表元信息。"""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wanglaoshi.report_meta import (
    ChartMeta,
    assert_report_ok,
    check_report_chapters,
    load_chart_sidecar,
    write_chart_sidecar,
)
from wanglaoshi.score_floors import SCORE_FLOORS, assert_above_floor


@pytest.mark.quick
def test_c05_iris_accuracy_floor() -> None:
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    clf.fit(Xtr, ytr)
    acc = float(accuracy_score(yte, clf.predict(Xte)))
    assert_above_floor("iris_accuracy", acc)
    assert SCORE_FLOORS["iris_accuracy"]["floor"] == 0.85


@pytest.mark.quick
def test_c05_breast_cancer_auc_floor() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba))
    assert_above_floor("breast_cancer_roc_auc", auc)


@pytest.mark.quick
def test_c05_diabetes_r2_floor() -> None:
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg.fit(Xtr, ytr)
    r2 = float(r2_score(yte, reg.predict(Xte)))
    assert_above_floor("diabetes_r2", r2)


@pytest.mark.quick
def test_c05_below_floor_raises() -> None:
    with pytest.raises(AssertionError, match="iris_accuracy"):
        assert_above_floor("iris_accuracy", 0.1)


@pytest.mark.quick
def test_c05_unknown_task_raises() -> None:
    with pytest.raises(KeyError, match="unknown score floor"):
        assert_above_floor("not_a_task", 1.0)


@pytest.mark.quick
def test_c05_exact_floor_passes() -> None:
    """等于下限应通过（不要求必胜，只防明显退化）。"""
    floor = SCORE_FLOORS["iris_accuracy"]["floor"]
    assert_above_floor("iris_accuracy", floor)


@pytest.mark.quick
def test_c06_report_chapters_and_chart_meta(tmp_path: Path) -> None:
    good = "## 摘要\n...\n## 数据说明\n...\n## 核心指标\nGMV=1\n## 结论\nok\n"
    assert check_report_chapters(good) == []
    assert_report_ok(good)
    bad = "## 摘要\nonly"
    assert "数据说明" in check_report_chapters(bad)
    with pytest.raises(AssertionError, match="missing chapters"):
        assert_report_ok(bad)

    img = tmp_path / "gmv_trend.png"
    img.write_bytes(b"fake-png")
    meta = ChartMeta(title="GMV 趋势", unit="元", period="2024-Q1", source="orders.csv", chart_type="line")
    side = write_chart_sidecar(img, meta)
    assert side.exists()
    loaded = load_chart_sidecar(img)
    assert loaded.validate() == []
    assert loaded.title == "GMV 趋势"
    empty = ChartMeta(title="", unit="", period="", source="")
    assert set(empty.validate()) == {"missing title", "missing unit", "missing period", "missing source"}


@pytest.mark.quick
def test_c06_custom_required_chapters() -> None:
    text = "## 方法\n## 结果\n"
    assert check_report_chapters(text, required=["方法", "结果"]) == []
    assert_report_ok(text, required=["方法", "结果"])
    with pytest.raises(AssertionError, match="局限"):
        assert_report_ok(text, required=["方法", "结果", "局限"])


@pytest.mark.quick
def test_c06_whitespace_meta_invalid_and_sidecar_roundtrip(tmp_path: Path) -> None:
    meta = ChartMeta(title="  ", unit="\t", period="", source="src")
    errs = set(meta.validate())
    assert "missing title" in errs
    assert "missing unit" in errs
    assert "missing period" in errs
    assert "missing source" not in errs

    img = tmp_path / "plot"
    img.write_bytes(b"x")
    good = ChartMeta(title="占比", unit="%", period="日", source="eda", notes="n1")
    write_chart_sidecar(img, good)
    loaded = load_chart_sidecar(img)
    assert loaded.notes == "n1"
    assert loaded.chart_type == ""
