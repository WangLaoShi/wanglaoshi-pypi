"""业务正确性：正式交付门槛、格式加载、日环比、字段唯一名、目标编码 fit/transform。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wanglaoshi.business import (
    BusinessPipeline,
    RunRecord,
    StageResult,
    archive_delivery,
    build_metric_dict,
    calculate_metrics,
    derive_metrics,
    structure_dataframe,
    structure_file,
)
from wanglaoshi.features import AdvancedFeatureEngineer


@pytest.mark.quick
@pytest.mark.business
def test_empty_run_not_formally_deliverable(tmp_path: Path) -> None:
    out = archive_delivery(RunRecord(run_id="empty"), tmp_path / "out", mode="formal")
    assert out.status == "blocked"
    assert out.data["delivery_index"]["deliverable"] is False
    assert any("missing stages" in r for r in out.block_reasons)


@pytest.mark.quick
@pytest.mark.business
def test_required_metric_nan_blocks_and_not_deliverable(tmp_path: Path) -> None:
    df = pd.DataFrame({"order_id": [1, 2, 3], "revenue": [10.0, 20.0, 30.0]})
    # 故意用默认 amount 口径 → GMV NaN → blocked
    met = calculate_metrics(df, required_metrics=["gmv"])
    assert met.status == "blocked"
    assert np.isnan(met.data["metrics"]["gmv"])

    src = tmp_path / "o.csv"
    pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "order_date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "revenue": [10.0, 20.0, 30.0],
            "channel": ["web", "app", "web"],
        }
    ).to_csv(src, index=False)
    # 正确配置应可交付
    ok = BusinessPipeline(
        tmp_path / "arch_ok",
        required_columns=["order_id", "revenue"],
        date_col="order_date",
        value_col="revenue",
        dim_cols=["channel"],
        required_metrics=["gmv"],
        delivery_mode="formal",
    ).run(src, run_id="ok")
    d10 = next(s for s in ok.stages if s.stage_id == "D10_report_archive")
    assert d10.data["delivery_index"]["deliverable"] is True
    assert next(s for s in ok.stages if s.stage_id == "D06_metrics").data["metrics"]["gmv"] == 60.0

    # 错误 value_col：指标 blocked → 不可交付
    bad = BusinessPipeline(
        tmp_path / "arch_bad",
        required_columns=["order_id", "revenue"],
        date_col="order_date",
        value_col="amount",
        dim_cols=["channel"],
        required_metrics=["gmv"],
        delivery_mode="formal",
    ).run(src, run_id="bad")
    assert bad.blocked()
    d10b = next(s for s in bad.stages if s.stage_id == "D10_report_archive")
    assert d10b.data["delivery_index"]["deliverable"] is False


@pytest.mark.quick
@pytest.mark.business
def test_structure_file_format_gates(tmp_path: Path) -> None:
    csv = tmp_path / "a.csv"
    pd.DataFrame({"订单号": [1], "金额": [2.0]}).to_csv(csv, index=False)
    ok = structure_file(csv, tmp_path / "out.csv")
    assert ok.status == "passed"
    assert "订单号" in ok.data["columns"] or any("订单" in c for c in ok.data["columns"])

    parquet = tmp_path / "x.parquet"
    parquet.write_bytes(b"not-parquet")
    bad = structure_file(parquet, tmp_path / "o.csv")
    assert bad.status == "blocked"
    assert any("unsupported" in r for r in bad.block_reasons)

    broken = tmp_path / "b.xlsx"
    broken.write_bytes(b"not-an-excel")
    blocked = structure_file(broken, tmp_path / "o2.csv")
    assert blocked.status == "blocked"
    assert any("failed to read" in r for r in blocked.block_reasons)


@pytest.mark.quick
@pytest.mark.business
def test_normalize_unique_and_unicode() -> None:
    df = pd.DataFrame({"a": [1], "a!": [2], "a_1": [3], "中文列": [4]})
    result = structure_dataframe(df)
    cols = result.data["columns"]
    assert len(cols) == len(set(cols))
    assert any("中文" in c for c in cols)


@pytest.mark.quick
@pytest.mark.business
def test_wow_is_daily_not_row_shift() -> None:
    # 同一天两单：行级 shift 会得到订单间变化；日聚合后同日 wow 应相同
    df = pd.DataFrame(
        {
            "order_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"],
            "amount": [10.0, 30.0, 40.0, 20.0],
        }
    )
    result = derive_metrics(df, date_col="order_date", value_col="amount", add_wow=True)
    frame = result.frame  # type: ignore[attr-defined]
    day1 = frame.loc[frame["order_date"].astype(str).str.startswith("2024-01-01"), "wow"]
    assert day1.nunique(dropna=False) == 1
    # 1/1 合计 40 → 1/2 合计 40 → wow=0；1/2→1/3: (20-40)/40=-0.5
    day2 = frame.loc[frame["order_date"].astype(str).str.startswith("2024-01-02"), "wow"].iloc[0]
    day3 = frame.loc[frame["order_date"].astype(str).str.startswith("2024-01-03"), "wow"].iloc[0]
    assert day2 == pytest.approx(0.0)
    assert day3 == pytest.approx(-0.5)


@pytest.mark.quick
@pytest.mark.business
def test_target_encoder_fit_transform_unseen() -> None:
    eng = AdvancedFeatureEngineer()
    train = pd.DataFrame({"city": ["a", "a", "b"], "y": [1, 0, 1]})
    test = pd.DataFrame({"city": ["a", "c"]})
    enc = eng.fit_target_encoder(train, "city", "y", smoothing=1.0)
    out = eng.transform_target_encoder(test, enc)
    assert enc["col_name"] in out.columns
    # unseen c → global mean
    assert out.loc[1, enc["col_name"]] == pytest.approx(enc["global_mean"])
