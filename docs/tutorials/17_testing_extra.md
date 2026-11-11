# 17 · _testing 其余（契约 / 图表元数据 / 回归汇总）

## 目标

补齐 `assert_metrics_close`、`assert_frame_contract`、`ChartMeta`、`assert_report_ok`、`write_chart_sidecar`、`RegressionSummary`、`from_junit_xml`、`optional_data_available`、`require_data_path`。

## 本节公开接口

上述符号。

## 数据集

临时 DataFrame + 迷你 junit.xml。

## 完整示例

```python
from pathlib import Path
import tempfile
import pandas as pd

from wanglaoshi.runtime import ready
from wanglaoshi._testing import (
    assert_metrics_close,
    assert_frame_contract,
    ChartMeta,
    assert_report_ok,
    write_chart_sidecar,
    RegressionSummary,
    from_junit_xml,
    optional_data_available,
    require_data_path,
)
from wanglaoshi._testing.regression_summary import CaseStatus

ready(seed=42, quiet=True, show_env=False)

df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
assert_frame_contract(df, columns=["a", "b"], n_rows=2)
assert_metrics_close({"acc": 0.9}, {"acc": 0.90000001}, atol=1e-5)

report = "## 摘要\n...\n## 数据说明\n...\n## 核心指标\n...\n## 结论\n..."
assert_report_ok(report)

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    img = td / "chart.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # 占位
    meta = ChartMeta(title="GMV", unit="yuan", period="2024-Q1", source="demo", chart_type="line")
    assert meta.validate() == []
    side = write_chart_sidecar(img, meta)
    assert side.exists() and meta.to_dict()["title"] == "GMV"

    junit = td / "junit.xml"
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
    out = summary2.write(td / "regression_summary.json")
    assert out.exists()

    assert optional_data_available(junit) is True
    require_data_path(junit)
```

配套脚本：[`examples/tutorials/17_testing_extra.py`](../../examples/tutorials/17_testing_extra.py)

## 常见坑

- `require_data_path` 在缺文件时会 `pytest.skip`（仅测试场景）
- 报告章节名需与 `assert_report_ok` 默认清单一致（摘要/数据说明/核心指标/结论）

## 下一步

- 回到索引 → [README.md](README.md)
- 深案例 → `examples/advanced/`
