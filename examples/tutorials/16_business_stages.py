"""Tutorial 16 smoke: business stage-level public APIs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from wanglaoshi.business import (
    DEFAULT_FACTORS,
    DEFAULT_METRIC_DICT,
    STAT_METHOD_CATALOG,
    RunRecord,
    StageResult,
    add_value_tiers,
    archive_charts,
    archive_delivery,
    build_analysis_plan,
    calculate_metrics,
    check_join_inflation,
    clean_dataframe,
    complete_period_index,
    convert_units,
    default_metric_registry,
    derive_metrics,
    generate_insights,
    intake_file,
    reconcile_to_source,
    run_eda_profile,
    run_multidimensional_analysis,
    structure_dataframe,
    structure_file,
    validate_metric_dict,
    yoy_stage,
    yoy_with_rules,
)
from wanglaoshi.runtime import ready


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "order_date": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "amount": [10.0, 20.0, 7.0, 80.0, 15.0, 12.0],
            "channel": ["web", "app", "web", "app", "web", "app"],
            "user_tier": ["A", "B", "A", "A", "B", "B"],
        }
    )
    refunds = pd.DataFrame({"order_id": [2, 2, 4], "refund": [5.0, 3.0, 10.0]})

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / "orders.csv"
        orders.to_csv(src, index=False)

        r_in = intake_file(
            src, td_path / "archive", required_columns=["order_id", "amount"], period="2024"
        )
        assert isinstance(r_in, StageResult)
        r_st = structure_dataframe(orders, column_map={"amount": "amount"})
        r_sf = structure_file(src, td_path / "structured.csv")
        r_cl = clean_dataframe(orders, drop_duplicates=True, subset=["order_id"])
        r_eda = run_eda_profile(orders, field_roles={"segment_cols": ["channel"]})
        assert STAT_METHOD_CATALOG
        r_der = derive_metrics(
            orders, date_col="order_date", value_col="amount", add_tiers=True
        )
        tiered = add_value_tiers(orders, "amount", method="pareto")
        r_plan = build_analysis_plan(
            orders, date_col="order_date", value_col="amount", dim_cols=["channel"]
        )
        r_met = calculate_metrics(
            orders,
            metric_dict=DEFAULT_METRIC_DICT,
            group_cols=["channel"],
            share_of="gmv",
        )
        r_an = run_multidimensional_analysis(
            orders,
            date_col="order_date",
            value_col="amount",
            dim_col="channel",
            segment_col="user_tier",
        )
        r_charts = archive_charts(
            orders,
            td_path / "charts",
            date_col="order_date",
            value_col="amount",
            dim_col="channel",
            period="2024",
        )
        r_ins = generate_insights(
            metrics_data=r_met.data,
            analysis_data=r_an.data,
            planning_data=r_plan.data,
        )
        record = RunRecord(run_id="tut16")
        for st in (r_in, r_st, r_sf, r_cl, r_eda, r_der, r_plan, r_met, r_an, r_charts, r_ins):
            record.add_stage(st)
        r_arc = archive_delivery(
            record,
            td_path / "delivery",
            insights_data=r_ins.data,
            metrics_data=r_met.data,
        )
        assert r_arc.status in {"passed", "warning", "blocked", "skipped"}

        assert check_join_inflation(
            orders, refunds, on="order_id", measure_col="amount"
        ).status
        assert reconcile_to_source(
            orders[["order_id", "amount"]],
            orders[["order_id", "amount"]],
            key="order_id",
            value_col="amount",
        ).status
        cal = complete_period_index(orders, "order_date", value_col="amount", freq="MS")
        yoy = yoy_with_rules(orders, "order_date", "amount", freq="MS")
        assert yoy_stage(orders, "order_date", "amount").status
        assert convert_units(
            orders, "amount", from_unit="yuan", to_unit="wan", factors=DEFAULT_FACTORS
        ).status
        assert validate_metric_dict(DEFAULT_METRIC_DICT, version="v1").status
        assert default_metric_registry(version="v1").status
        assert len(cal) >= 1 and len(yoy) >= 1 and "value_tier" in tiered.columns
    print("16_business_stages OK", len(record.stages))


if __name__ == "__main__":
    main()
