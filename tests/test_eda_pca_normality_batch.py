"""EDA：小样本 PCA、全空列、正态性方法适用性、批量报告文件名。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wanglaoshi.analyze.core import (
    DataAnalyzer,
    _batch_report_html_path,
    analyze_multiple_files,
)


@pytest.mark.quick
def test_pca_chinese_labels_no_glyph_warnings() -> None:
    """PCA 碎石图必须用包内中文字体，不得回退 DejaVu 缺字。"""
    import logging
    import warnings

    import matplotlib

    matplotlib.use("Agg")

    logs: list[str] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if "Glyph" in msg or "DejaVu" in msg:
                logs.append(msg)

    handler = _H()
    logging.getLogger("matplotlib.font_manager").addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rng = np.random.default_rng(0)
            df = pd.DataFrame(
                {
                    "销售额": rng.normal(size=40),
                    "成本": rng.normal(size=40),
                    "利润": rng.normal(size=40),
                }
            )
            out = DataAnalyzer(df).pca_analysis()
            assert out["applicable"] is True
        text = [str(w.message) for w in caught if "Glyph" in str(w.message) or "DejaVu" in str(w.message)]
    finally:
        logging.getLogger("matplotlib.font_manager").removeHandler(handler)

    assert not logs and not text, f"glyph warnings: {logs + text}"


@pytest.mark.quick
def test_pca_handles_n_less_than_p_and_all_null_column() -> None:
    rng = np.random.default_rng(0)
    wide = pd.DataFrame({f"f{i}": rng.normal(size=5) for i in range(8)})
    pca = DataAnalyzer(wide).pca_analysis()
    assert pca["applicable"] is True
    n_pc = len(pca["variance_ratio"])
    assert n_pc == min(5, 8)
    assert list(pca["components"].index) == [f"PC{i+1}" for i in range(n_pc)]
    assert list(pca["scores"].columns) == [f"PC{i+1}" for i in range(n_pc)]

    mixed = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": [np.nan] * 8,
        }
    )
    pca2 = DataAnalyzer(mixed).pca_analysis()
    assert pca2["applicable"] is True
    assert "b" in pca2["excluded_columns"]
    assert "a" in pca2["feature_names"]


@pytest.mark.quick
def test_normality_five_samples_does_not_raise() -> None:
    out = DataAnalyzer(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})).normality_test()
    assert len(out) == 1
    row = out.iloc[0]
    assert pd.notna(row["Shapiro-Wilk p值"])
    assert pd.isna(row["D-Agostino-Pearson p值"])
    assert row["结论"] in {"未拒绝正态假设", "拒绝正态假设"}
    assert "8" in str(row["备注"])


@pytest.mark.quick
def test_eda_report_survives_wide_and_null_columns(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f"f{i}": rng.normal(size=6) for i in range(7)})
    df["empty"] = np.nan
    df["cat"] = ["a", "b"] * 3
    out = tmp_path / "edge_report.html"
    DataAnalyzer(df).generate_report(str(out))
    assert out.exists() and out.stat().st_size > 100
    text = out.read_text(encoding="utf-8", errors="ignore").lower()
    assert "<html" in text or "rawdata" in text
    # 基础统计仍应嵌入
    assert "basic_stats" in text or "f0" in text


@pytest.mark.quick
def test_batch_report_paths_keep_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _batch_report_html_path(str(tmp_path), "sales.csv").endswith("sales_csv_report.html")
    assert _batch_report_html_path(str(tmp_path), "sales.json").endswith("sales_json_report.html")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sales.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    (data_dir / "sales.json").write_text('[{"a":1,"b":2},{"a":3,"b":4}]', encoding="utf-8")
    out_dir = tmp_path / "reports"

    def _fake_analyze(file_path: str, output_html: str = "analysis_report.html") -> None:
        Path(output_html).write_text(f"from:{Path(file_path).name}", encoding="utf-8")

    import wanglaoshi.analyze.core as core_mod

    monkeypatch.setattr(core_mod, "analyze_data", _fake_analyze)
    analyze_multiple_files(str(data_dir), str(out_dir))
    names = sorted(p.name for p in out_dir.glob("*.html"))
    assert names == ["sales_csv_report.html", "sales_json_report.html"]
    assert (out_dir / "sales_csv_report.html").read_text(encoding="utf-8") == "from:sales.csv"
    assert (out_dir / "sales_json_report.html").read_text(encoding="utf-8") == "from:sales.json"
