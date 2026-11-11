"""业务流水线边界 / 契约回归（C04/C07/C09 子集）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wanglaoshi.business import (
    BusinessPipeline,
    RunRecord,
    StageResult,
    archive_delivery,
    calculate_metrics,
    clean_dataframe,
    derive_metrics,
    structure_dataframe,
    structure_file,
)


@pytest.mark.quick
@pytest.mark.business
def test_stage_result_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="invalid status"):
        StageResult(stage_id="D01_intake", status="success")


@pytest.mark.quick
@pytest.mark.business
def test_structure_file_missing_blocked(tmp_path: Path) -> None:
    result = structure_file(tmp_path / "missing.csv", tmp_path / "out.csv")
    assert result.status == "blocked"
    assert result.block_reasons


@pytest.mark.quick
@pytest.mark.business
def test_structure_file_and_duplicate_normalize(tmp_path: Path) -> None:
    src = tmp_path / "raw.csv"
    pd.DataFrame({"Order ID": [1, 2], "order id": [10, 20], "GMV ($)": ["1", "x"]}).to_csv(src, index=False)
    out = tmp_path / "structured.csv"
    result = structure_file(src, out, dtype_map={"gmv": "float64"})
    assert result.status == "warning"  # dtype convert fail on 'x'
    assert out.exists()
    cols = result.data["columns"]
    assert "order_id" in cols
    assert any(c.startswith("order_id_") for c in cols)
    assert "gmv" in result.data["failed_dtypes"]


@pytest.mark.quick
@pytest.mark.business
def test_structure_column_map_missing_key_warns() -> None:
    df = pd.DataFrame({"a": [1]})
    result = structure_dataframe(df, column_map={"nope": "a2"})
    assert result.status == "warning"
    assert any("column_map" in w for w in result.warnings)


@pytest.mark.quick
@pytest.mark.business
def test_clean_fillna_dropna_and_missing_col_warn(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 3],
            "amount": [10.0, np.nan, 30.0, 30.0],
            "channel": ["web", None, "app", "app"],
        }
    )
    result = clean_dataframe(
        df,
        drop_duplicates=True,
        dropna_columns=["channel"],
        fillna={"amount": 0.0, "missing_col": -1},
        clip_bounds={"ghost": (0, 1)},
        output_path=tmp_path / "cleaned.csv",
        log_path=tmp_path / "clean_log.json",
    )
    assert result.status == "warning"
    assert any("fillna skip" in w for w in result.warnings)
    assert any("clip skip" in w for w in result.warnings)
    assert result.data["rows_out"] < result.data["rows_in"]
    assert result.frame["amount"].isna().sum() == 0  # type: ignore[attr-defined]
    log = json.loads((tmp_path / "clean_log.json").read_text(encoding="utf-8"))
    rules = {item["rule"] for item in log}
    assert {"drop_duplicates", "dropna_columns", "fillna", "clip_bounds"} <= rules


@pytest.mark.quick
@pytest.mark.business
def test_clean_constant_and_all_null_column() -> None:
    df = pd.DataFrame({"const": [1, 1, 1], "empty": [np.nan, np.nan, np.nan], "id": [1, 2, 3]})
    result = clean_dataframe(df, fillna={"empty": 0}, drop_duplicates=False)
    assert result.status == "passed"
    assert result.frame["empty"].tolist() == [0, 0, 0]  # type: ignore[attr-defined]
    assert result.frame["const"].nunique() == 1  # type: ignore[attr-defined]


@pytest.mark.quick
@pytest.mark.business
def test_derive_missing_date_blocked() -> None:
    df = pd.DataFrame({"amount": [1.0]})
    result = derive_metrics(df, date_col="order_date", value_col="amount")
    assert result.status == "blocked"
    assert "order_date" in result.block_reasons[0]


@pytest.mark.quick
@pytest.mark.business
def test_derive_bad_dates_and_zero_base_wow() -> None:
    df = pd.DataFrame(
        {
            "order_date": ["2024-01-01", "not-a-date", "2024-01-03"],
            "amount": [0.0, 10.0, 20.0],
        }
    )
    result = derive_metrics(df, date_col="order_date", value_col="amount", add_wow=True)
    assert result.status == "passed"
    frame = result.frame
    assert "order_date_year" in frame.columns  # type: ignore[attr-defined]
    assert pd.isna(frame.loc[1, "order_date_year"])  # type: ignore[attr-defined]
    # 前一期为 0 → wow 应为 NaN（零基期）
    assert "wow" in frame.columns  # type: ignore[attr-defined]
    wow_vals = frame["wow"].tolist()  # type: ignore[attr-defined]
    assert any(pd.isna(v) for v in wow_vals)


@pytest.mark.quick
@pytest.mark.business
def test_derive_share_group_missing_warns() -> None:
    df = pd.DataFrame({"amount": [1.0, 2.0]})
    result = derive_metrics(df, value_col="amount", share_group_cols=["channel"])
    assert result.status == "warning"
    assert "share" not in result.frame.columns  # type: ignore[attr-defined]


@pytest.mark.quick
@pytest.mark.business
def test_metrics_missing_col_and_zero_share_total() -> None:
    df = pd.DataFrame({"order_id": [1, 2], "amount": [0.0, 0.0], "channel": ["web", "app"]})
    result = calculate_metrics(df, group_cols=["channel", "ghost"], share_of="gmv")
    assert result.status == "warning"
    assert any("group_cols missing" in w for w in result.warnings)
    assert any("share total is 0" in w for w in result.warnings)
    assert result.data["metrics"]["gmv"] == 0.0


@pytest.mark.quick
@pytest.mark.business
def test_metrics_aov_undefined_when_no_orders() -> None:
    df = pd.DataFrame({"amount": [10.0, 20.0]})  # no order_id
    result = calculate_metrics(df)
    assert result.status == "warning"
    assert np.isnan(result.data["metrics"]["orders"])
    assert any("aov" in w for w in result.warnings)


@pytest.mark.quick
@pytest.mark.business
def test_archive_redacts_sensitive_fields(tmp_path: Path) -> None:
    record = RunRecord(
        run_id="priv01",
        config={"phone": "13800138000", "period": "2024-Q1"},
        random_seed=7,
    )
    record.add_stage(
        StageResult(
            stage_id="D01_intake",
            status="passed",
            data={"customer_email": "a@b.com", "sha256": "abc"},
        )
    )
    record.add_stage(
        StageResult(
            stage_id="D06_metrics",
            status="passed",
            data={"metrics": {"gmv": 1.0}},
        )
    )
    result = archive_delivery(
        record,
        tmp_path / "delivery",
        redact_fields=["phone", "customer_email"],
        metrics_data={"metrics": {"gmv": 1.0}},
        mode="formal",
    )
    assert result.status == "passed"
    full = json.loads((tmp_path / "delivery" / "run_record.json").read_text(encoding="utf-8"))
    assert full["config"]["phone"] == "***REDACTED***"
    assert full["config"]["period"] == "2024-Q1"
    assert full["stages"][0]["data"]["customer_email"] == "***REDACTED***"
    assert full["stages"][0]["data"]["sha256"] == "abc"


@pytest.mark.quick
@pytest.mark.business
def test_archive_missing_required_blocks(tmp_path: Path) -> None:
    record = RunRecord(run_id="req1")
    record.add_stage(StageResult(stage_id="D01_intake", status="passed"))
    record.add_stage(
        StageResult(stage_id="D06_metrics", status="passed", data={"metrics": {"gmv": 1.0}})
    )
    result = archive_delivery(
        record,
        tmp_path / "delivery",
        required=["report.md", "delivery_index.json", "extra_must_exist.txt"],
        metrics_data={"metrics": {"gmv": 1.0}},
        mode="formal",
    )
    assert result.status == "blocked"
    assert result.data["delivery_index"]["deliverable"] is False
    assert any("extra_must_exist.txt" in r for r in result.block_reasons)


@pytest.mark.quick
@pytest.mark.business
def test_pipeline_stop_on_block_false_continues(tmp_path: Path) -> None:
    pipe = BusinessPipeline(tmp_path / "archive", stop_on_block=False)
    record = pipe.run(tmp_path / "missing.csv", run_id="cont01")
    assert record.blocked()
    statuses = {s.stage_id: s.status for s in record.stages}
    assert statuses["D01_intake"] == "blocked"
    assert statuses["D02_structuring"] == "blocked"
    # 未提前 return，应跑完主链各阶段（无 skipped）
    assert "skipped" not in statuses.values()
    assert statuses["D10_report_archive"] == "blocked"
    assert any(s.stage_id == "D10_report_archive" and s.data["delivery_index"]["deliverable"] is False for s in record.stages)


@pytest.mark.quick
@pytest.mark.business
def test_pipeline_redact_fields_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "order_id": [1, 2],
            "order_date": ["2024-01-01", "2024-01-02"],
            "amount": [10.0, 20.0],
            "channel": ["web", "app"],
            "phone": ["111", "222"],
        }
    ).to_csv(src, index=False)
    pipe = BusinessPipeline(
        tmp_path / "archive",
        required_columns=["order_id", "amount"],
        date_col="order_date",
        value_col="amount",
        dim_cols=["channel"],
        redact_fields=["phone"],
        period="2024-Q1",
    )
    record = pipe.run(src, run_id="redact01")
    assert not record.blocked()
    delivery = tmp_path / "archive" / "redact01__delivery" / "run_record.json"
    assert delivery.exists()
    # phone 在原始 csv 列里不一定进入 record.data；config 侧脱敏列表应被记录
    d10 = record.stages[-1]
    assert d10.data["redact_fields"] == ["phone"]
