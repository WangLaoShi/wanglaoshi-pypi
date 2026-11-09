"""业务模块剩余缺口：洞察降级、分析跳过、图表空输入、EDA 抽样、契约序列化。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wanglaoshi.business import (
    BusinessPipeline,
    StageResult,
    archive_charts,
    generate_insights,
    run_eda_profile,
    run_multidimensional_analysis,
)
from wanglaoshi.business.contracts import RunRecord


@pytest.mark.quick
@pytest.mark.business
def test_insights_downgrade_without_gmv() -> None:
    result = generate_insights(
        metrics_data={"metrics": {}, "share_check": {"ok": False, "share_sum": 1.5, "tol": 1e-6}},
        analysis_data={"views": {"pareto": {"status": "skipped"}, "anomaly": {"status": "skipped"}}},
        planning_data={"tasks": [{"direction": "trend", "status": "not_applicable", "reason": "no date"}]},
    )
    assert result.status == "warning"
    types = {f["type"] for f in result.data["findings"]}
    assert "observation" in types
    assert "problem" in types  # share_check not ok
    assert any(f["title"].startswith("方向不可用") for f in result.data["findings"])
    assert any(r["id"] == "rec_fix_share" for r in result.data["recommendations"])


@pytest.mark.quick
@pytest.mark.business
def test_insights_highlight_and_anomaly_paths() -> None:
    result = generate_insights(
        metrics_data={"metrics": {"gmv": 100.0, "orders": 10.0, "aov": 10.0}},
        analysis_data={
            "views": {
                "pareto": {
                    "status": "ok",
                    "head_keys": ["web"],
                    "head_share": 0.82,
                    "n_head": 1,
                    "threshold": 0.8,
                    "note": "不证明业务因果",
                },
                "anomaly": {
                    "status": "ok",
                    "z_thresh": 2.5,
                    "anomalies": [{"date": "2024-01-03", "z": 3.1}],
                },
            }
        },
    )
    types = {f["type"] for f in result.data["findings"]}
    assert {"status", "highlight", "problem"} <= types
    rec_ids = {r["id"] for r in result.data["recommendations"]}
    assert "rec_focus_head" in rec_ids
    assert "rec_investigate_anomaly" in rec_ids


@pytest.mark.quick
@pytest.mark.business
def test_analysis_all_views_skipped_warns() -> None:
    result = run_multidimensional_analysis(pd.DataFrame({"a": [1, 2]}))
    assert result.status == "warning"
    assert result.data["ok_count"] == 0
    assert all(v.get("status") == "skipped" for k, v in result.data["views"].items() if k != "correlation")


@pytest.mark.quick
@pytest.mark.business
def test_archive_charts_missing_roles_warns(tmp_path: Path) -> None:
    result = archive_charts(pd.DataFrame({"x": [1, 2]}), tmp_path / "charts")
    assert result.status == "warning"
    assert result.data["charts"] == []
    assert (tmp_path / "charts" / "charts_manifest.json").exists()
    assert any("no charts produced" in w for w in result.warnings)


@pytest.mark.quick
@pytest.mark.business
def test_archive_charts_invalid_dates_skip_trend(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "order_date": ["bad", "also-bad"],
            "amount": [1.0, 2.0],
            "channel": ["web", "app"],
        }
    )
    result = archive_charts(
        df,
        tmp_path / "charts",
        date_col="order_date",
        value_col="amount",
        dim_col="channel",
    )
    assert result.status == "warning"
    types = {c["chart_type"] for c in result.data["charts"]}
    assert "structure" in types
    assert "trend" not in types
    assert any("trend chart skipped" in w for w in result.warnings)


@pytest.mark.quick
@pytest.mark.business
def test_eda_sampling_warning_and_lightweight() -> None:
    df = pd.DataFrame({"amount": list(range(20)), "channel": ["web"] * 10 + ["app"] * 10})
    sampled = run_eda_profile(df, sample_n=5, random_seed=0, use_analyzer=True)
    assert sampled.status == "warning"
    assert sampled.data["profile"]["n_rows"] == 5
    assert any("sampled" in w for w in sampled.warnings)

    light = run_eda_profile(df, use_analyzer=False)
    assert light.status == "passed"
    assert light.data["analyzer"] == {}
    methods = {m["id"]: m for m in light.data["stat_methods"]}
    assert methods["pearson_corr"]["applicable"] is False
    assert methods["chi_square"]["applicable"] is False


@pytest.mark.quick
@pytest.mark.business
def test_stage_result_ok_and_json(tmp_path: Path) -> None:
    stage = StageResult(stage_id="D01_intake", status="warning", warnings=["x"])
    assert stage.ok is True
    text = stage.to_json(tmp_path / "stage.json")
    assert '"status": "warning"' in text
    assert (tmp_path / "stage.json").exists()

    blocked = StageResult(stage_id="D01_intake", status="blocked", block_reasons=["nope"])
    assert blocked.ok is False
    record = RunRecord(run_id="r", random_seed=1)
    record.add_stage(blocked)
    assert record.blocked() is True
    assert record.to_dict()["run_id"] == "r"


@pytest.mark.quick
@pytest.mark.business
def test_pipeline_run_to_dict(tmp_path: Path) -> None:
    src = tmp_path / "s.csv"
    pd.DataFrame(
        {
            "order_id": [1, 2],
            "order_date": ["2024-01-01", "2024-01-02"],
            "amount": [5.0, 7.0],
            "channel": ["web", "app"],
        }
    ).to_csv(src, index=False)
    summary = BusinessPipeline(
        tmp_path / "archive",
        date_col="order_date",
        value_col="amount",
        dim_cols=["channel"],
    ).run_to_dict(src, run_id="sum01")
    assert summary["run_id"] == "sum01"
    assert summary["blocked"] is False
    assert len(summary["stages"]) >= 8
