"""DataAnalyzer 质量分析与报告生成覆盖。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wanglaoshi.Analyzer import DataAnalyzer, analyze_data


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(5, 2, n),
            "z": rng.normal(10, 3, n),
            "cat": rng.choice(["a", "b", "c"], size=n),
            "dup_key": [1, 1] + list(range(2, n)),
        }
    )


@pytest.mark.quick
def test_analyzer_quality_apis(sample_df: pd.DataFrame) -> None:
    df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)  # force a duplicate
    analyzer = DataAnalyzer(df)

    outliers = analyzer.outlier_analysis()
    assert not outliers.empty

    dup = analyzer.duplicate_analysis()
    assert dup["重复行数量"] >= 1

    corr = analyzer.correlation_analysis()
    assert corr.shape[0] >= 2

    pca = analyzer.pca_analysis()
    assert isinstance(pca, dict)
    assert len(pca) > 0

    interp = analyzer.interpret_results()
    assert isinstance(interp, dict)


@pytest.mark.quick
def test_analyzer_generate_report_html(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    # 报告生成较重：缩小列数，避免为过多列画图
    df = sample_df[["x", "y", "cat"]].copy()
    out = tmp_path / "tiny_report.html"
    DataAnalyzer(df).generate_report(str(out))
    assert out.exists()
    text = out.read_text(encoding="utf-8", errors="ignore")
    assert "<html" in text.lower() or "<!doctype" in text.lower() or len(text) > 100


@pytest.mark.quick
def test_analyze_data_entrypoint(tmp_path: Path) -> None:
    # normality_test / scipy.normaltest 需要足够样本量（>=8）
    rng = np.random.default_rng(1)
    src = tmp_path / "data.csv"
    pd.DataFrame({"a": rng.normal(size=16), "b": rng.normal(size=16)}).to_csv(src, index=False)
    out = tmp_path / "from_path.html"
    analyze_data(str(src), output_html=str(out))
    assert out.exists() and out.stat().st_size > 0
