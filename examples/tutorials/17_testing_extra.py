"""Tutorial 17 smoke: remaining _testing public APIs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from wanglaoshi._testing import (
    ChartMeta,
    RegressionSummary,
    assert_frame_contract,
    assert_metrics_close,
    assert_report_ok,
    from_junit_xml,
    optional_data_available,
    require_data_path,
    write_chart_sidecar,
)
from wanglaoshi._testing.regression_summary import CaseStatus
from wanglaoshi.runtime import ready


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert_frame_contract(df, columns=["a", "b"], n_rows=2)
    assert_metrics_close({"acc": 0.9}, {"acc": 0.90000001}, atol=1e-5)
    assert_report_ok("## 摘要\nx\n## 数据说明\nx\n## 核心指标\nx\n## 结论\nx\n")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        img = td_path / "chart.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        meta = ChartMeta(
            title="GMV", unit="yuan", period="2024-Q1", source="demo", chart_type="line"
        )
        assert meta.validate() == []
        side = write_chart_sidecar(img, meta)
        assert side.exists() and meta.to_dict()["title"] == "GMV"

        junit = td_path / "junit.xml"
        junit.write_text(
            """<?xml version='1.0'?>
<testsuite name='pytest' tests='2' failures='0' skipped='1'>
  <testcase classname='t' name='ok' time='0.1'/>
  <testcase classname='t' name='skip' time='0.0'><skipped message='no data'/></testcase>
</testsuite>
""",
            encoding="utf-8",
        )
        summary = from_junit_xml(junit)
        assert summary.counts()["passed"] >= 1
        summary2 = RegressionSummary(
            cases=[CaseStatus(name="x", outcome="passed", duration_s=0.01)]
        )
        out = summary2.write(td_path / "regression_summary.json")
        assert out.exists()
        assert optional_data_available(junit) is True
        require_data_path(junit)
    print("17_testing_extra OK")


if __name__ == "__main__":
    main()
