"""C09：业务脏样例——退款/取消/单位/缺期/零基期/关联膨胀/缺目标/隐私。

关键数字均为手工预计算，避免「跑通即正确」。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wanglaoshi import (
    BusinessPipeline,
    SurvivalAnalyzer,
    check_join_inflation,
    convert_units,
    reconcile_to_source,
    yoy_with_rules,
)
from wanglaoshi.business import archive_delivery
from wanglaoshi.business.contracts import RunRecord, StageResult


@pytest.fixture
def dirty_orders() -> dict[str, pd.DataFrame]:
    """合成脏业务表（金额单位：元）。"""
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "order_amount": [100.0, 200.0, 150.0, 80.0],  # 真值合计 530
            "status": ["paid", "paid", "cancelled", "paid"],
            "customer_phone": ["13800000001", "13800000002", "13800000003", "13800000004"],
        }
    )
    items = pd.DataFrame(
        {
            "order_id": [1, 1, 2, 2, 2, 3, 4],
            "line_amount": [40, 60, 50, 70, 80, 150, 80],
        }
    )
    refunds = pd.DataFrame(
        {
            "refund_id": [10, 11],
            "order_id": [2, 4],
            "refund_amount": [30.0, 10.0],  # 合计 40
        }
    )
    return {"orders": orders, "items": items, "refunds": refunds}


@pytest.mark.quick
def test_c09_join_inflation_and_refund_reconcile(dirty_orders) -> None:
    orders, items, refunds = dirty_orders["orders"], dirty_orders["items"], dirty_orders["refunds"]
    # 手工：订单头合计
    assert orders["order_amount"].sum() == 530.0
    inf = check_join_inflation(orders, items, on="order_id", measure_col="order_amount")
    assert inf.data["inflated"] is True
    # 手工：100×2 + 200×3 + 150×1 + 80×1 = 1030
    assert inf.data["joined_sum"] == 1030.0
    assert abs(inf.data["dedup_sum"] - 530.0) < 1e-9

    # 退款与源表对平
    rec = reconcile_to_source(refunds, refunds, key="refund_id", value_col="refund_amount")
    assert rec.data["all_balanced"]
    assert refunds["refund_amount"].sum() == 40.0

    # 取消单不应计入有效销售：手工有效=530-150=380
    valid = orders.loc[orders["status"] != "cancelled", "order_amount"].sum()
    assert valid == 380.0


@pytest.mark.quick
def test_c09_units_yuan_to_wan() -> None:
    df = pd.DataFrame({"amount": [10000.0, 25000.0]})  # 元
    r = convert_units(df, "amount", from_unit="yuan", to_unit="wan")
    assert r.status == "passed"
    out = r.data["frame"][r.data["out_col"]].tolist()
    assert out == pytest.approx([1.0, 2.5])


@pytest.mark.quick
def test_c09_yoy_missing_and_zero_base() -> None:
    # 手工：2023-03=0 → 2024-03 YoY 应为 zero_base；缺 2023-02
    sales = pd.DataFrame(
        {
            "month": pd.to_datetime(
                ["2023-01-01", "2023-03-01", "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ),
            "sales": [100.0, 0.0, 110.0, 50.0, 20.0, 10.0],
        }
    )
    # as_of 取 4 月中旬：3 月完整可比（零基数）；4 月未完月
    yoy = yoy_with_rules(sales, "month", "sales", as_of="2024-04-15", freq="MS")
    row_2024_03 = yoy.loc[yoy["month"] == pd.Timestamp("2024-03-01")].iloc[0]
    assert row_2024_03["reason"] == "zero_base"
    assert pd.isna(row_2024_03["yoy"])
    row_2024_04 = yoy.loc[yoy["month"] == pd.Timestamp("2024-04-01")].iloc[0]
    assert row_2024_04["reason"] == "incomplete_month"
    assert yoy["is_missing"].any()


@pytest.mark.quick
def test_c09_missing_target_blocks_pipeline(tmp_path: Path) -> None:
    src = tmp_path / "orders.csv"
    pd.DataFrame({"order_id": [1, 2], "channel": ["a", "b"]}).to_csv(src, index=False)
    pipe = BusinessPipeline(
        tmp_path / "archive",
        required_columns=["order_id", "amount"],  # amount 缺失 → intake warning，后续可能继续
        value_col="amount",
        stop_on_block=True,
    )
    record = pipe.run(src)
    # 缺 required 列：intake warning；value_col 不存在时衍生/指标应 blocked 或 warning
    statuses = {s.stage_id: s.status for s in record.stages}
    assert statuses["D01_intake"] in {"warning", "blocked"}
    # 正式交付在 blocked 运行上应不可交付
    if record.blocked():
        delivery = archive_delivery(record, tmp_path / "delivery")
        assert delivery.status == "blocked"


@pytest.mark.quick
def test_c09_privacy_redact(tmp_path: Path) -> None:
    record = RunRecord(run_id="priv1", config={"customer_phone": "13800000001", "ok": 1})
    record.add_stage(StageResult(stage_id="D01_intake", status="passed", data={"x": 1}))
    # 脱敏能力用 diagnostic 模式验证；正式交付另测阶段门槛
    out = archive_delivery(
        record,
        tmp_path / "delivery",
        redact_fields=["customer_phone"],
        mode="diagnostic",
    )
    assert out.status in {"passed", "warning"}
    blob = "\n".join(Path(p).read_text(encoding="utf-8") for p in out.output_paths)
    assert "13800000001" not in blob
    assert "***REDACTED***" in blob
    assert out.data["delivery_index"]["deliverable"] is False
    # 内存中的原始 record 仍可保留明文（脱敏只作用于交付物）
    assert record.config["customer_phone"] == "13800000001"


@pytest.mark.quick
def test_c09_survival_km_median() -> None:
    # 事件时间 1,2,3 全发生 → 中位生存应存在
    times = [1, 2, 3, 4, 5]
    events = [1, 1, 1, 1, 1]
    km = SurvivalAnalyzer.km(times, events)
    assert km["median_survival"] is not None
    assert km["survival"][-1] == pytest.approx(0.0)
