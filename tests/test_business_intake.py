"""D01–D06 / pipeline 最小回归。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wanglaoshi.business import (
    BusinessPipeline,
    RunRecord,
    StageResult,
    archive_charts,
    archive_delivery,
    build_analysis_plan,
    calculate_metrics,
    clean_dataframe,
    derive_metrics,
    generate_insights,
    intake_file,
    run_eda_profile,
    run_multidimensional_analysis,
    structure_dataframe,
)


@pytest.mark.quick
def test_intake_archives_and_hashes(tmp_path: Path) -> None:
    src = tmp_path / "orders.csv"
    pd.DataFrame({"order_id": [1, 2], "amount": [10.0, 20.0]}).to_csv(src, index=False)
    archive = tmp_path / "archive"

    result = intake_file(src, archive, required_columns=["order_id", "amount"], period="2024-Q1")
    assert result.status == "passed"
    assert result.data["sha256"]
    assert result.data["missing_columns"] == []
    assert any(Path(p).name.startswith("orders__") for p in result.output_paths)
    ledger = archive / "intake_ledger.jsonl"
    assert ledger.exists()
    assert "sha256" in ledger.read_text(encoding="utf-8")


@pytest.mark.quick
def test_intake_missing_columns_warning(tmp_path: Path) -> None:
    src = tmp_path / "orders.csv"
    pd.DataFrame({"order_id": [1]}).to_csv(src, index=False)
    result = intake_file(src, tmp_path / "archive", required_columns=["order_id", "amount"])
    assert result.status == "warning"
    assert "amount" in result.data["missing_columns"]


@pytest.mark.quick
def test_intake_missing_file_blocked(tmp_path: Path) -> None:
    result = intake_file(tmp_path / "nope.csv", tmp_path / "archive")
    assert result.status == "blocked"
    assert result.block_reasons


@pytest.mark.quick
def test_pipeline_stops_on_block(tmp_path: Path) -> None:
    pipe = BusinessPipeline(tmp_path / "archive", stop_on_block=True)
    record = pipe.run(tmp_path / "missing.csv")
    assert record.blocked()
    assert record.stages[0].stage_id == "D01_intake"
    assert record.stages[1].status == "skipped"
    assert any(s.stage_id == "D05_planning" and s.status == "skipped" for s in record.stages)


@pytest.mark.quick
def test_pipeline_happy_path_placeholders(tmp_path: Path) -> None:
    src = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "order_date": ["2024-01-01", "2024-01-02", "2024-01-02"],
            "amount": [10.0, 20.0, 30.0],
            "channel": ["web", "app", "web"],
            "user_tier": ["A", "B", "A"],
        }
    ).to_csv(src, index=False)
    pipe = BusinessPipeline(
        tmp_path / "archive",
        required_columns=["order_id", "amount"],
        date_col="order_date",
        value_col="amount",
        dim_cols=["channel"],
        segment_cols=["user_tier"],
    )
    record = pipe.run(src)
    assert not record.blocked()
    ids = [s.stage_id for s in record.stages]
    assert ids == [
        "D01_intake",
        "D02_structuring",
        "D03_cleaning",
        "D12_eda",
        "D04_derived",
        "D05_planning",
        "D06_metrics",
        "D07_analysis",
        "D08_visualization",
        "D09_insights",
        "D10_report_archive",
    ]
    assert all(s.status in {"passed", "warning"} for s in record.stages)
    d10 = record.stages[-1]
    assert d10.data["delivery_index"]["deliverable"] is True
    assert any(p.endswith("report.md") for p in d10.output_paths)
    d08 = next(s for s in record.stages if s.stage_id == "D08_visualization")
    assert len(d08.data.get("charts", [])) >= 1


@pytest.mark.quick
def test_structure_normalizes_columns() -> None:
    df = pd.DataFrame({"Order ID": [1], "GMV ($)": ["3.5"]})
    result = structure_dataframe(df, dtype_map={"gmv": "float64"})
    assert result.status in {"passed", "warning"}
    assert "order_id" in result.data["columns"]
    assert "gmv" in result.data["columns"]
    assert result.frame["gmv"].dtype == "float64"  # type: ignore[attr-defined]


@pytest.mark.quick
def test_clean_drop_duplicates_and_clip(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1, 1, 2], "amount": [10.0, 10.0, 999.0]})
    result = clean_dataframe(
        df,
        drop_duplicates=True,
        clip_bounds={"amount": (0.0, 100.0)},
        output_path=tmp_path / "cleaned.csv",
        log_path=tmp_path / "log.json",
    )
    assert result.status == "passed"
    assert result.data["rows_out"] == 2
    assert result.frame["amount"].max() <= 100.0  # type: ignore[attr-defined]
    assert (tmp_path / "log.json").exists()


@pytest.mark.quick
def test_derive_time_and_share() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "channel": ["web", "web", "app"],
            "amount": [10.0, 20.0, 10.0],
        }
    )
    result = derive_metrics(df, date_col="date", value_col="amount", share_group_cols=["date"], add_wow=True)
    assert result.status == "passed"
    assert "date_year" in result.data["created_columns"]
    assert "share" in result.frame.columns  # type: ignore[attr-defined]


@pytest.mark.quick
def test_planning_four_directions() -> None:
    df = pd.DataFrame(
        {
            "order_date": ["2024-01-01"],
            "amount": [10.0],
            "channel": ["web"],
            "user_tier": ["A"],
        }
    )
    result = build_analysis_plan(
        df,
        date_col="order_date",
        value_col="amount",
        dim_cols=["channel"],
        segment_cols=["user_tier"],
    )
    assert result.status == "passed"
    assert result.data["applicable_count"] == 4
    statuses = {t["direction"]: t["status"] for t in result.data["tasks"]}
    assert statuses["trend"] == "applicable"
    assert statuses["segment"] == "applicable"


@pytest.mark.quick
def test_planning_missing_fields_not_applicable() -> None:
    df = pd.DataFrame({"amount": [1.0]})
    result = build_analysis_plan(df, value_col="amount")
    assert result.status == "warning"
    by_dir = {t["direction"]: t for t in result.data["tasks"]}
    assert by_dir["overall"]["status"] == "applicable"
    assert by_dir["trend"]["status"] == "not_applicable"


@pytest.mark.quick
def test_metrics_gmv_aov_and_share() -> None:
    df = pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "amount": [10.0, 20.0, 30.0],
            "channel": ["web", "app", "web"],
        }
    )
    result = calculate_metrics(df, group_cols=["channel"], share_of="gmv")
    assert result.status == "passed"
    assert result.data["metrics"]["gmv"] == 60.0
    assert result.data["metrics"]["orders"] == 3.0
    assert result.data["metrics"]["aov"] == 20.0
    assert result.data["share_check"]["ok"] is True
    assert abs(sum(r["share"] for r in result.data["workbook"]) - 1.0) < 1e-9


@pytest.mark.quick
def test_analysis_six_views() -> None:
    df = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=10, freq="D").astype(str).tolist() * 2,
            "amount": [10, 12, 11, 40, 13, 12, 11, 10, 12, 11] * 2,
            "channel": (["web"] * 10) + (["app"] * 10),
            "user_tier": (["A"] * 10) + (["B"] * 10),
        }
    )
    result = run_multidimensional_analysis(
        df,
        date_col="order_date",
        value_col="amount",
        dim_col="channel",
        segment_col="user_tier",
    )
    assert result.status in {"passed", "warning"}
    assert result.data["ok_count"] >= 4
    assert result.data["views"]["pareto"]["status"] == "ok"
    assert "不证明业务因果" in result.data["views"]["pareto"]["note"]


@pytest.mark.quick
def test_insights_and_recommendations() -> None:
    metrics = calculate_metrics(
        pd.DataFrame({"order_id": [1, 2], "amount": [10.0, 30.0], "channel": ["web", "app"]}),
        group_cols=["channel"],
        share_of="gmv",
    )
    analysis = run_multidimensional_analysis(
        pd.DataFrame(
            {
                "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                "amount": [10, 12, 100, 11, 10],
                "channel": ["web", "web", "app", "web", "app"],
            }
        ),
        date_col="order_date",
        value_col="amount",
        dim_col="channel",
    )
    result = generate_insights(metrics_data=metrics.data, analysis_data=analysis.data)
    assert result.status in {"passed", "warning"}
    types = {f["type"] for f in result.data["findings"]}
    assert "status" in types
    assert result.data["counts"]["recommendations"] >= 1
    assert all(r.get("expected_uplift") == "pending" for r in result.data["recommendations"])


@pytest.mark.quick
def test_archive_delivery_and_block_on_upstream(tmp_path: Path) -> None:
    record = RunRecord(run_id="demo123", random_seed=42)
    record.add_stage(StageResult(stage_id="D01_intake", status="passed"))
    insights = generate_insights(
        metrics_data={
            "metrics": {"gmv": 10.0, "orders": 1.0, "aov": 10.0},
            "share_check": {"ok": True, "share_sum": 1.0, "tol": 1e-6},
        }
    )
    result = archive_delivery(
        record,
        tmp_path / "delivery",
        insights_data=insights.data,
        metrics_data={"metrics": {"gmv": 10.0}},
    )
    assert result.status == "passed"
    assert (tmp_path / "delivery" / "report.md").exists()
    assert result.data["delivery_index"]["deliverable"] is True

    blocked = RunRecord(run_id="blocked1")
    blocked.add_stage(
        StageResult(stage_id="D01_intake", status="blocked", block_reasons=["missing file"])
    )
    bad = archive_delivery(blocked, tmp_path / "delivery_bad")
    assert bad.status == "blocked"
    assert bad.data["delivery_index"]["deliverable"] is False


@pytest.mark.quick
def test_archive_charts_metadata(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "amount": [10.0, 20.0, 15.0],
            "channel": ["web", "app", "web"],
        }
    )
    result = archive_charts(
        df,
        tmp_path / "charts",
        date_col="order_date",
        value_col="amount",
        dim_col="channel",
        period="2024-Q1",
        source="unit_test",
        unit="CNY",
    )
    assert result.status == "passed"
    assert (tmp_path / "charts" / "trend.png").exists()
    assert (tmp_path / "charts" / "structure.json").exists()
    meta = result.data["charts"][0]
    assert meta["title"] and meta["unit"] == "CNY" and meta["period"] == "2024-Q1" and meta["source"]


@pytest.mark.quick
def test_eda_profile_and_stat_catalog() -> None:
    df = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0],
            "channel": ["web", "app", "web"],
            "user_tier": ["A", "B", "A"],
        }
    )
    result = run_eda_profile(
        df,
        field_roles={"value_col": "amount", "segment_cols": ["user_tier"], "dim_cols": ["channel"]},
        use_analyzer=True,
    )
    assert result.status in {"passed", "warning"}
    assert result.data["profile"]["n_rows"] == 3
    methods = {m["id"]: m for m in result.data["stat_methods"]}
    assert methods["pearson_corr"]["applicable"] is False  # only 1 numeric col
    assert methods["describe"]["applicable"] is True
