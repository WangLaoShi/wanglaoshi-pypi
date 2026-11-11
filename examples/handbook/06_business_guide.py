"""Handbook 06 smoke: business pipeline + stage APIs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from wanglaoshi.business import (
    DEFAULT_FACTORS,
    DEFAULT_METRIC_DICT,
    STAT_METHOD_CATALOG,
    BusinessPipeline,
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

        pipe = BusinessPipeline(
            td_path / "archive_pipe",
            required_columns=["order_id", "amount", "channel"],
            date_col="order_date",
            value_col="amount",
            dim_cols=["channel"],
            segment_cols=["user_tier"],
            period="2024",
        )
        record = pipe.run(src)
        assert record.summary()
        assert not record.blocked()

        assert isinstance(
            intake_file(src, td_path / "arch", required_columns=["order_id", "amount"]),
            StageResult,
        )
        structure_file(src, td_path / "structured.csv")
        structure_dataframe(orders)
        clean_dataframe(orders, drop_duplicates=True, subset=["order_id"])
        run_eda_profile(orders, field_roles={"segment_cols": ["channel"]})
        assert STAT_METHOD_CATALOG
        derive_metrics(orders, date_col="order_date", value_col="amount", add_tiers=True)
        add_value_tiers(orders, "amount")
        plan = build_analysis_plan(
            orders, date_col="order_date", value_col="amount", dim_cols=["channel"]
        )
        met = calculate_metrics(
            orders, metric_dict=DEFAULT_METRIC_DICT, group_cols=["channel"], share_of="gmv"
        )
        an = run_multidimensional_analysis(
            orders,
            date_col="order_date",
            value_col="amount",
            dim_col="channel",
            segment_col="user_tier",
        )
        archive_charts(
            orders,
            td_path / "charts",
            date_col="order_date",
            value_col="amount",
            dim_col="channel",
        )
        ins = generate_insights(
            metrics_data=met.data, analysis_data=an.data, planning_data=plan.data
        )
        rr = RunRecord(run_id="hb06")
        rr.add_stage(met)
        rr.add_stage(ins)
        archive_delivery(
            rr, td_path / "delivery", insights_data=ins.data, metrics_data=met.data
        )

        assert check_join_inflation(
            orders, refunds, on="order_id", measure_col="amount"
        ).status
        assert reconcile_to_source(
            orders[["order_id", "amount"]],
            orders[["order_id", "amount"]],
            key="order_id",
            value_col="amount",
        ).status
        assert len(complete_period_index(orders, "order_date", value_col="amount")) >= 6
        assert yoy_with_rules(orders, "order_date", "amount") is not None
        assert yoy_stage(orders, "order_date", "amount").status
        assert convert_units(
            orders, "amount", from_unit="yuan", to_unit="wan", factors=DEFAULT_FACTORS
        ).status
        assert validate_metric_dict(DEFAULT_METRIC_DICT, version="v1").status
        assert default_metric_registry().status

    print("06_business_guide OK")


if __name__ == "__main__":
    main()
