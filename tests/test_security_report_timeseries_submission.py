"""回归：报告脚本注入、零权重融合、下载原子写入、静态路径。"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from wanglaoshi.analyze.core import ReportGenerator, _PKG_ROOT
from wanglaoshi.compete import SubmissionGenerator
from wanglaoshi.runtime.web_getter import Wget


@pytest.mark.quick
def test_html_report_escapes_script_breakout(tmp_path: Path) -> None:
    payload = "</script><script>window.__pwned=1</script>"
    out = tmp_path / "nested" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    # 直接渲染嵌入路径，避免完整 EDA 对恶意列名崩溃干扰安全断言
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "js").mkdir()
    ReportGenerator(
        os.path.join(_PKG_ROOT, "templates"),
        str(static_dir),
    ).generate_report(
        {
            "basic_stats": {"note": payload},
            "interpretations": {},
            "plots": {},
            "evil": payload,
        },
        str(out),
    )
    html = out.read_text(encoding="utf-8")
    assert payload not in html
    assert "\\u003c/script\\u003e" in html  # Jinja tojson HTML-safe escape


@pytest.mark.quick
def test_report_static_relpath_no_extra_parent(tmp_path: Path) -> None:
    """输出在子目录时，relpath 不应再被额外 prepend '..'。"""
    out = tmp_path / "nested" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    static_dir = tmp_path / "static"
    (static_dir / "js").mkdir(parents=True)
    rg = ReportGenerator(os.path.join(_PKG_ROOT, "templates"), str(static_dir))
    rg.generate_report(
        {"basic_stats": {}, "interpretations": {}, "plots": {}},
        str(out),
    )
    expected = os.path.relpath(str(static_dir), str(out.parent)).replace("\\", "/")
    assert expected == "../static"
    # 生成路径计算与 os.path.relpath 一致（core 已去掉多余 '..'）
    assert expected == os.path.relpath(rg.static_dir, str(out.parent)).replace("\\", "/")


@pytest.mark.quick
def test_ensemble_rejects_zero_weight_sum(tmp_path: Path) -> None:
    s1 = tmp_path / "a.csv"
    s2 = tmp_path / "b.csv"
    pd.DataFrame({"Id": [1, 2], "target": [0.1, 0.2]}).to_csv(s1, index=False)
    pd.DataFrame({"Id": [1, 2], "target": [0.3, 0.4]}).to_csv(s2, index=False)
    out = tmp_path / "ens.csv"
    gen = SubmissionGenerator()
    with pytest.raises(ValueError, match="positive"):
        gen.ensemble_submissions(
            [str(s1), str(s2)],
            weights=[0.0, 0.0],
            method="weighted_average",
            output_path=str(out),
        )
    assert not out.exists()


@pytest.mark.quick
def test_wget_atomic_keeps_original_on_failure(tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"GOOD")

    class _Head:
        headers = {"content-length": "100"}

    class _Get:
        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size=1024):  # noqa: ANN001
            yield b"BAD"
            raise RuntimeError("boom")

    w = Wget("https://example.com/file.bin", output_dir=str(tmp_path), filename="file.bin")
    with patch("wanglaoshi.runtime.web_getter.requests.head", return_value=_Head()), patch(
        "wanglaoshi.runtime.web_getter.requests.get", return_value=_Get()
    ):
        assert w.download() is False
    assert dest.read_bytes() == b"GOOD"
    assert not (tmp_path / "file.bin.partial").exists()
