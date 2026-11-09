"""intake / metrics / pipeline 补充边界。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from wanglaoshi.business import (
    BusinessPipeline,
    calculate_metrics,
    intake_file,
)
from wanglaoshi.business.metrics import DEFAULT_METRIC_DICT


@pytest.mark.quick
@pytest.mark.business
def test_intake_period_written_to_ledger(tmp_path: Path) -> None:
    src = tmp_path / "orders.csv"
    pd.DataFrame({"order_id": [1], "amount": [9.0]}).to_csv(src, index=False)
    archive = tmp_path / "archive"
    result = intake_file(src, archive, required_columns=["order_id"], period="2024-H1")
    assert result.status == "passed"
    ledger = (archive / "intake_ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(ledger[-1])
    assert row.get("period") == "2024-H1" or result.data.get("period") == "2024-H1"


@pytest.mark.quick
@pytest.mark.business
def test_custom_metric_dict_mean() -> None:
    df = pd.DataFrame({"score": [1.0, 3.0, 5.0], "order_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    mdict = {
        **DEFAULT_METRIC_DICT,
        "avg_score": {"formula": "mean(score)", "value_col": "score", "agg": "mean", "direction": "higher_better"},
    }
    result = calculate_metrics(df, metric_dict=mdict)
    assert result.status == "passed"
    assert result.data["metrics"]["avg_score"] == 3.0


@pytest.mark.quick
@pytest.mark.business
def test_pipeline_auto_detects_common_columns(tmp_path: Path) -> None:
    src = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "gmv": [10.0, 20.0, 15.0],
            "region": ["east", "west", "east"],
            "segment": ["A", "B", "A"],
        }
    ).to_csv(src, index=False)
    # 不显式传 date/value/dim，走自动探测
    record = BusinessPipeline(tmp_path / "archive", period="auto").run(src, run_id="auto01")
    assert not record.blocked()
    ids = [s.stage_id for s in record.stages]
    assert "D07_analysis" in ids
    assert "D08_visualization" in ids
