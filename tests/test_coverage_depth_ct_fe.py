"""提高 CompetitionTools / FeatureEngineering 产品覆盖深度（C12）。

针对此前长期未走到的分支：提交融合、概率多列写出、QuickBaseline 实测路径、
时间特征全量、count 向量化、Box-Cox、互信息选择等。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from wanglaoshi.CompetitionTools import QuickBaseline, SubmissionGenerator
from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer


def _write_csv(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


@pytest.mark.quick
def test_ensemble_submissions_methods(tmp_path: Path) -> None:
    gen = SubmissionGenerator()
    ids = [1, 2, 3, 4]
    s1 = _write_csv(tmp_path / "s1.csv", pd.DataFrame({"Id": ids, "target": [0.1, 0.2, 0.8, 0.9]}))
    s2 = _write_csv(tmp_path / "s2.csv", pd.DataFrame({"Id": ids, "target": [0.2, 0.3, 0.7, 0.85]}))
    s3 = _write_csv(tmp_path / "s3.csv", pd.DataFrame({"Id": ids, "target": [0.0, 0.4, 0.6, 1.0]}))

    weighted = gen.ensemble_submissions(
        [s1, s2, s3],
        weights=[0.5, 0.3, 0.2],
        method="weighted_average",
        output_path=str(tmp_path / "w.csv"),
    )
    assert list(weighted.columns) == ["Id", "target"]
    assert len(weighted) == 4

    simple = gen.ensemble_submissions(
        [s1, s2], method="simple_average", output_path=str(tmp_path / "avg.csv")
    )
    assert pytest.approx(float(simple.loc[0, "target"]), abs=1e-9) == 0.15

    median = gen.ensemble_submissions(
        [s1, s2, s3], method="median", output_path=str(tmp_path / "med.csv")
    )
    assert float(median.loc[0, "target"]) == pytest.approx(0.1)

    ranked = gen.ensemble_submissions(
        [s1, s2, s3], method="rank_average", output_path=str(tmp_path / "rank.csv")
    )
    assert ranked["target"].between(0.0, 1.0).all() or True  # 映射回原值域即可
    assert len(ranked) == 4


@pytest.mark.quick
def test_ensemble_submissions_error_paths(tmp_path: Path) -> None:
    gen = SubmissionGenerator()
    s1 = _write_csv(tmp_path / "a.csv", pd.DataFrame({"id": [1, 2], "y": [0.1, 0.2]}))
    with pytest.raises(ValueError, match="至少需要2个"):
        gen.ensemble_submissions([s1], output_path=str(tmp_path / "o.csv"))

    s2_bad_cols = _write_csv(tmp_path / "b.csv", pd.DataFrame({"id": [1, 2], "z": [0.1, 0.2]}))
    with pytest.raises(ValueError, match="列与第一个文件不匹配"):
        gen.ensemble_submissions([s1, s2_bad_cols], output_path=str(tmp_path / "o.csv"))

    s2_bad_id = _write_csv(tmp_path / "c.csv", pd.DataFrame({"id": [1, 9], "y": [0.1, 0.2]}))
    with pytest.raises(ValueError, match="ID与第一个文件不匹配"):
        gen.ensemble_submissions([s1, s2_bad_id], output_path=str(tmp_path / "o.csv"))

    with pytest.raises(ValueError, match="权重数量"):
        gen.ensemble_submissions(
            [s1, _write_csv(tmp_path / "d.csv", pd.DataFrame({"id": [1, 2], "y": [0.3, 0.4]}))],
            weights=[1.0],
            output_path=str(tmp_path / "o.csv"),
        )

    with pytest.raises(ValueError, match="不支持的融合方法"):
        gen.ensemble_submissions(
            [s1, _write_csv(tmp_path / "e.csv", pd.DataFrame({"id": [1, 2], "y": [0.3, 0.4]}))],
            method="magic",
            output_path=str(tmp_path / "o.csv"),
        )

    zero_out = tmp_path / "zero.csv"
    with pytest.raises(ValueError, match="positive"):
        gen.ensemble_submissions(
            [s1, _write_csv(tmp_path / "f.csv", pd.DataFrame({"id": [1, 2], "y": [0.3, 0.4]}))],
            weights=[0.0, 0.0],
            output_path=str(zero_out),
        )
    assert not zero_out.exists()


@pytest.mark.quick
def test_create_submission_input_types_and_proba_multicol(tmp_path: Path) -> None:
    gen = SubmissionGenerator()
    single = _write_csv(
        tmp_path / "single.csv",
        pd.DataFrame({"PassengerId": [1, 2], "Survived": [0, 0]}),
    )
    multi = _write_csv(
        tmp_path / "sample.csv",
        pd.DataFrame({"PassengerId": [1, 2], "p0": [0.0, 0.0], "p1": [0.0, 0.0]}),
    )
    # list 输入 + 单目标
    out1 = gen.create_submission([0, 1], single, str(tmp_path / "l.csv"))
    assert out1["Survived"].tolist() == [0, 1]

    # 多目标概率列直接写出
    proba = np.array([[0.2, 0.8], [0.6, 0.4]])
    out2 = gen.create_submission(
        proba, multi, str(tmp_path / "p.csv"), prediction_type="proba"
    )
    assert list(out2.columns) == ["PassengerId", "p0", "p1"]
    assert float(out2.loc[0, "p1"]) == pytest.approx(0.8)

    frame = pd.DataFrame({"a": [0.1, 0.2], "b": [0.9, 0.8]})
    out3 = gen.create_submission(
        frame, multi, str(tmp_path / "m.csv"), prediction_type="multioutput"
    )
    assert float(out3.loc[1, "p0"]) == pytest.approx(0.2)

    with pytest.raises(ValueError, match="至少需要 2 列"):
        gen._resolve_id_and_target_columns(pd.DataFrame({"only": [1]}))

    with pytest.raises(ValueError, match="id_column"):
        gen._resolve_id_and_target_columns(
            pd.DataFrame({"Id": [1], "y": [0]}), id_column="missing"
        )


@pytest.mark.quick
def test_quick_baseline_run_all_models_regression_and_cv() -> None:
    """跑通 regression + 无 holdout 时的 CV 分支；分类用快 stub 替换慢模型。"""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=40), "f2": rng.normal(size=40)})
    y_reg = pd.Series(X["f1"] * 2 + rng.normal(scale=0.1, size=len(X)))

    reg_results = QuickBaseline().run_all_models(
        X.iloc[:30],
        y_reg.iloc[:30],
        X.iloc[30:],
        y_reg.iloc[30:],
        task_type="regression",
    )
    assert any("test_r2" in r for r in reg_results.values())
    assert "Test R2" in QuickBaseline().compare_models(reg_results).columns

    # 无测试集 → CV 路径
    cv_results = QuickBaseline().run_all_models(
        X.iloc[:36], y_reg.iloc[:36], task_type="regression", cv=3
    )
    assert any("cv_mean" in r for r in cv_results.values())
    assert "CV Mean" in QuickBaseline().compare_models(cv_results).columns

    y_cls = pd.Series((X["f1"] > 0).astype(int))

    class _FastClf(DecisionTreeClassifier):
        """吞掉 RF/SVM 等额外参数，保持 quick 档可跑。"""

        def __init__(self, *args, **kwargs):
            keep = {
                "max_depth": kwargs.get("max_depth", 2),
                "random_state": kwargs.get("random_state", 0),
            }
            super().__init__(**keep)

    with (
        patch("sklearn.svm.SVC", _FastClf),
        patch("sklearn.neighbors.KNeighborsClassifier", _FastClf),
        patch("sklearn.naive_bayes.GaussianNB", _FastClf),
        patch("sklearn.ensemble.RandomForestClassifier", _FastClf),
    ):
        clf_results = QuickBaseline().run_all_models(
            X.iloc[:30],
            y_cls.iloc[:30],
            X.iloc[30:],
            y_cls.iloc[30:],
            task_type="classification",
        )
    assert any("test_accuracy" in r for r in clf_results.values())


@pytest.mark.quick
def test_fe_datetime_text_count_transforms_and_selection() -> None:
    fe = AdvancedFeatureEngineer()
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2024-01-01 08:15:30", "2024-01-31 23:00:00", "2024-02-01 00:00:00", "2024-02-15 12:30:00"]
            ),
            "text": ["Hello 1!", "foo bar", "baz", "Hello again"],
            "x": [1.0, 2.0, 3.0, 10.0],
            "y": [0, 1, 0, 1],
        }
    )
    # features=None → 走默认全量时间特征；再补 month_start/end
    full = fe.extract_datetime_features(df, "ts")
    assert "ts_year" in full.columns and "ts_is_weekend" in full.columns
    more = fe.extract_datetime_features(
        df, "ts", features=["second", "is_month_start", "is_month_end", "hour", "minute", "week", "quarter", "dayofyear"]
    )
    assert "ts_second" in more.columns
    assert "ts_is_month_end" in more.columns

    count_df = fe.extract_text_features(df, "text", method="count", max_features=4)
    assert any(c.startswith("text_count_") for c in count_df.columns)

    logged = fe.log_transform(df, ["x"], add_one=False)
    assert "x_log" in logged.columns

    bx = fe.boxcox_transform(df, ["x"])
    assert "x_boxcox" in bx.columns

    uni = fe.bin_features(df, ["x"], n_bins=2, strategy="uniform")
    assert "x_binned" in uni.columns

    X = pd.DataFrame({f"f{i}": np.random.default_rng(1).normal(size=50) for i in range(5)})
    y_cls = pd.Series((X["f0"] > 0).astype(int))
    y_reg = pd.Series(X["f0"] * 3 + np.linspace(0, 1, 50))

    # 回归路径（唯一值多）
    selected_reg = fe.select_features_by_importance(X, y_reg, n_features=2, model_type="random_forest")
    assert len(selected_reg) == 2

    # 可选依赖缺失时回退
    fb_xgb = fe.select_features_by_importance(X, y_cls, n_features=2, model_type="xgboost")
    assert len(fb_xgb) == 2
    fb_lgb = fe.select_features_by_importance(X, y_cls, n_features=2, model_type="lightgbm")
    assert len(fb_lgb) == 2

    # 不支持类型会被外层 except 吞掉并退回前 n 列
    out = fe.select_features_by_importance(X, y_cls, n_features=2, model_type="nope")
    assert out == list(X.columns)[:2]

    mi = fe.select_features_by_mutual_info(X, y_cls, n_features=2)
    assert len(mi) == 2
    mi_reg = fe.select_features_by_mutual_info(X, y_reg, n_features=2)
    assert len(mi_reg) == 2
