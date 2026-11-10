"""架构门面：Phase 1 双轨兼容 + Phase 2 公开面分层。"""

from __future__ import annotations

import pytest


@pytest.mark.quick
def test_facade_imports_match_legacy_symbols() -> None:
    from wanglaoshi import CompetitionTools as CT
    from wanglaoshi import DataAnalyzer as DA_top
    from wanglaoshi.analyze import DataAnalyzer as DA_facade
    from wanglaoshi.compete import QuickBaseline, SubmissionGenerator
    from wanglaoshi.features import AdvancedFeatureEngineer
    from wanglaoshi.analytics import StatsToolkit, TimeSeriesTools
    from wanglaoshi.runtime import ready
    from wanglaoshi._testing import assert_close, ExperimentContext

    assert SubmissionGenerator is CT.SubmissionGenerator
    assert QuickBaseline is CT.QuickBaseline
    assert DA_facade is DA_top
    assert AdvancedFeatureEngineer is not None
    assert StatsToolkit is not None and TimeSeriesTools is not None
    assert callable(ready)
    assert callable(assert_close)
    assert ExperimentContext is not None


@pytest.mark.quick
def test_legacy_imports_still_work() -> None:
    from wanglaoshi import Analyzer, CompetitionTools, FeatureEngineering, ready
    from wanglaoshi.CompetitionTools import SubmissionGenerator
    from wanglaoshi.Analyzer import DataAnalyzer

    assert Analyzer is not None
    assert CompetitionTools is not None
    assert FeatureEngineering is not None
    assert callable(ready)
    assert SubmissionGenerator is not None
    assert DataAnalyzer is not None


@pytest.mark.quick
def test_top_level_exports_facade_modules() -> None:
    import wanglaoshi as wl

    for name in ("compete", "analyze", "features", "analytics", "runtime", "_testing", "business"):
        assert hasattr(wl, name), name


@pytest.mark.quick
def test_public_surface_split_stable_vs_compat() -> None:
    import wanglaoshi as wl

    stable = set(wl._STABLE_PUBLIC)
    compat = set(wl._COMPAT_EXPORTS)
    assert stable.isdisjoint(compat)
    assert set(wl.__all__) == stable | compat
    # 门面必须在稳定面
    for name in ("compete", "analyze", "features", "analytics", "runtime", "_testing", "business"):
        assert name in stable
    # legacy 模块留在兼容面
    for name in ("Analyzer_Plain", "CompetitionTools", "FeatureEngineering", "Analyzer"):
        assert name in compat
    # 兼容面符号仍可从顶层取到
    assert wl.Analyzer_Plain is not None


@pytest.mark.quick
def test_legacy_modules_document_status() -> None:
    import wanglaoshi.Analyzer_Plain as plain
    import wanglaoshi.wqe as wqe

    assert "Legacy" in (plain.__doc__ or "")
    assert "Legacy" in (wqe.__doc__ or "")
