"""业务口径回归：value_col 与指标字典衔接、字段改名不变性、缺字段失败行为。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wanglaoshi.business import (
    BusinessPipeline,
    DEFAULT_METRIC_DICT,
    build_metric_dict,
    calculate_metrics,
    intake_file,
)


def _orders_frame(*, value_name: str = "amount") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "order_date": pd.date_range("2024-01-01", periods=6, freq="MS"),
            value_name: [10.0, 20.0, 7.0, 80.0, 15.0, 12.0],
            "channel": ["web", "app", "web", "app", "web", "app"],
            "user_tier": ["A", "B", "A", "A", "B", "B"],
        }
    )


def _stage(record, stage_id: str):
    return next(s for s in record.stages if s.stage_id == stage_id)


@pytest.mark.quick
@pytest.mark.business
def test_build_metric_dict_rewrites_value_col() -> None:
    mdict = build_metric_dict(value_col="revenue", order_id_col="oid")
    assert mdict["gmv"]["value_col"] == "revenue"
    assert mdict["gmv"]["formula"] == "sum(revenue)"
    assert mdict["orders"]["value_col"] == "oid"
    # 默认字典不被原地修改
    assert DEFAULT_METRIC_DICT["gmv"]["value_col"] == "amount"


@pytest.mark.quick
@pytest.mark.business
def test_calculate_metrics_respects_value_col_rename() -> None:
    base = _orders_frame(value_name="amount")
    renamed = base.rename(columns={"amount": "revenue"})
    m_amount = calculate_metrics(base, metric_dict=build_metric_dict(value_col="amount"))
    m_revenue = calculate_metrics(renamed, metric_dict=build_metric_dict(value_col="revenue"))
    assert m_amount.status == "passed"
    assert m_revenue.status == "passed"
    assert m_amount.data["metrics"]["gmv"] == pytest.approx(m_revenue.data["metrics"]["gmv"])
    assert m_amount.data["metrics"]["aov"] == pytest.approx(m_revenue.data["metrics"]["aov"])


@pytest.mark.quick
@pytest.mark.business
def test_pipeline_amount_vs_revenue_invariance(tmp_path: Path) -> None:
    """同一数据仅把 amount 改名为 revenue 并更新配置，业务指标应一致。"""
    amount_df = _orders_frame(value_name="amount")
    revenue_df = amount_df.rename(columns={"amount": "revenue"})

    src_a = tmp_path / "orders_amount.csv"
    src_r = tmp_path / "orders_revenue.csv"
    amount_df.to_csv(src_a, index=False)
    revenue_df.to_csv(src_r, index=False)

    rec_a = BusinessPipeline(
        tmp_path / "arch_a",
        required_columns=["order_id", "amount", "channel"],
        date_col="order_date",
        value_col="amount",
        dim_cols=["channel"],
        segment_cols=["user_tier"],
        period="2024",
    ).run(src_a, run_id="amount")

    rec_r = BusinessPipeline(
        tmp_path / "arch_r",
        required_columns=["order_id", "revenue", "channel"],
        date_col="order_date",
        value_col="revenue",
        dim_cols=["channel"],
        segment_cols=["user_tier"],
        period="2024",
    ).run(src_r, run_id="revenue")

    assert not rec_a.blocked() and not rec_r.blocked()
    met_a = _stage(rec_a, "D06_metrics").data["metrics"]
    met_r = _stage(rec_r, "D06_metrics").data["metrics"]
    assert met_a["gmv"] == pytest.approx(met_r["gmv"])
    assert met_a["orders"] == pytest.approx(met_r["orders"])
    assert met_a["aov"] == pytest.approx(met_r["aov"])
    assert not np.isnan(met_r["gmv"])

    # 指标字典侧车应记录实际 value_col
    assert _stage(rec_r, "D06_metrics").data["metric_dict"]["gmv"]["value_col"] == "revenue"

    # 多维分析应能用 revenue 出趋势，而不是因缺 amount 被跳过
    trend = _stage(rec_r, "D07_analysis").data["views"]["trend"]
    assert trend.get("status") == "ok"
    assert int(trend.get("n_periods", 0)) >= 1


@pytest.mark.quick
@pytest.mark.business
def test_pipeline_missing_value_col_metrics_warn(tmp_path: Path) -> None:
    """配置指向不存在的金额列时，指标不得静默用 amount 成功。"""
    df = _orders_frame(value_name="revenue")
    src = tmp_path / "orders.csv"
    df.to_csv(src, index=False)
    rec = BusinessPipeline(
        tmp_path / "arch",
        required_columns=["order_id", "revenue"],
        date_col="order_date",
        value_col="amount",  # 故意配错
        dim_cols=["channel"],
    ).run(src, run_id="badcfg")
    met = _stage(rec, "D06_metrics")
    assert met.status in {"warning", "blocked"}
    assert np.isnan(met.data["metrics"]["gmv"])
    assert any("amount" in w for w in met.warnings)


@pytest.mark.quick
@pytest.mark.business
def test_intake_missing_required_warns(tmp_path: Path) -> None:
    """缺必需列：intake 应 warning（仍归档），不得伪装成无问题的 passed。"""
    src = tmp_path / "orders.csv"
    pd.DataFrame({"order_id": [1], "channel": ["web"]}).to_csv(src, index=False)
    result = intake_file(
        src,
        tmp_path / "arch",
        required_columns=["order_id", "amount"],
    )
    assert result.status == "warning"
    assert result.warnings
