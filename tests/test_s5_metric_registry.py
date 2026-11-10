"""S5：指标口径版本登记最小版。"""

from __future__ import annotations

import pytest

from wanglaoshi.business.metric_registry import default_metric_registry, validate_metric_dict


@pytest.mark.quick
def test_s5_default_registry_versioned() -> None:
    r = default_metric_registry(version="v1")
    assert r.status in {"passed", "warning"}
    assert r.data["version"] == "v1"
    assert r.data["n_metrics"] >= 3


@pytest.mark.quick
def test_s5_missing_dependency_blocked() -> None:
    bad = {
        "x": {"value_col": "amount", "agg": "sum"},
        "y": {"depends_on": ["missing_metric"]},
    }
    r = validate_metric_dict(bad, version="v0")
    assert r.status == "blocked"
    assert any("missing_metric" in b for b in r.block_reasons)
