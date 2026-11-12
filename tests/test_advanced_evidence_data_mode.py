"""发布案例 data_mode / release-data 判定。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "run_advanced_evidence.py"


@pytest.fixture(scope="module")
def evidence_mod():
    spec = importlib.util.spec_from_file_location("run_advanced_evidence", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.quick
def test_require_real_includes_fraud_bike_mall(evidence_mod) -> None:
    assert "05_credit_card_fraud" in evidence_mod.REQUIRE_REAL_DATA
    assert "08_bike_sharing" in evidence_mod.REQUIRE_REAL_DATA
    assert "09_mall_customers" in evidence_mod.REQUIRE_REAL_DATA


@pytest.mark.quick
def test_kernel_output_parses_demo_and_data_mode(evidence_mod) -> None:
    blob = "loaded ok\nDEMO mode = True\n{'data_mode': 'demo', 'n_rows': 100}\n"
    meta = evidence_mod._data_mode_from_output_text(blob)
    assert meta["data_mode"] == "demo"
    assert meta["n_rows"] == 100

    blob2 = "OpenML 17379\nDEMO = False\n"
    meta2 = evidence_mod._data_mode_from_output_text(blob2)
    assert meta2["data_mode"] == "real_or_local"


@pytest.mark.quick
def test_release_verdict_demo_is_data_fail(evidence_mod) -> None:
    assert (
        evidence_mod._release_data_verdict(
            "05_credit_card_fraud",
            {"status": "passed", "data_mode": "demo"},
        )
        == "data_fail"
    )
    assert (
        evidence_mod._release_data_verdict(
            "01_iris",
            {"status": "passed", "data_mode": "demo"},
        )
        == "n/a"
    )


@pytest.mark.quick
def test_release_verdict_claims_real_but_missing_evidence_fails(evidence_mod) -> None:
    """声称 real 但缺 source/n_rows/sha256 → data_fail（对齐 DATA_ACCEPTANCE）。"""
    assert (
        evidence_mod._release_data_verdict(
            "08_bike_sharing",
            {"status": "passed", "data_mode": "real"},
        )
        == "data_fail"
    )
    assert evidence_mod._release_evidence_gaps(
        {"status": "passed", "data_mode": "real"}
    ) == ["data_source", "n_rows", "data_sha256"]

    # 截断哈希也不够
    assert (
        evidence_mod._release_data_verdict(
            "08_bike_sharing",
            {
                "status": "passed",
                "data_mode": "real_or_local",
                "data_source": "data/bike_sharing_hour.csv",
                "n_rows": 17379,
                "data_sha256": "375d870c6a8adb4c",  # 仅 16 位
            },
        )
        == "data_fail"
    )


@pytest.mark.quick
def test_release_verdict_real_with_full_evidence_ok(evidence_mod) -> None:
    assert (
        evidence_mod._release_data_verdict(
            "08_bike_sharing",
            {
                "status": "passed",
                "data_mode": "real",
                "data_source": "data/bike_sharing_hour.csv",
                "n_rows": 17379,
                "data_sha256": "a" * 64,
            },
        )
        == "ok"
    )
