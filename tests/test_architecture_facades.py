"""架构 Phase 1：门面导入与旧路径双轨兼容。"""

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
