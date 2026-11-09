"""ready() 一键就绪回归。"""

from __future__ import annotations

import warnings

import pytest

from wanglaoshi import ready


@pytest.mark.quick
def test_ready_sets_ignore_filter_and_seed_config() -> None:
    warnings.resetwarnings()
    cfg = ready(quiet=True, show_env=False, seed=7, check_core=True)
    assert cfg["quiet"] is True
    assert cfg["seed"] == 7
    assert cfg["core"]["numpy"] == "ok"
    assert cfg["core"]["pandas"] == "ok"
    assert cfg["core"]["sklearn"] == "ok"
    assert any(f[0] == "ignore" for f in warnings.filters)


@pytest.mark.quick
def test_ready_without_quiet_keeps_warnings_visible() -> None:
    warnings.resetwarnings()
    ready(quiet=False, show_env=False, seed=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("visible", UserWarning)
    assert any("visible" in str(w.message) for w in caught)
