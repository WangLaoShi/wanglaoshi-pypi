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


@pytest.mark.quick
def test_legacy_import_emits_deprecation(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import warnings

    import wanglaoshi.Analyzer_Plain as plain_mod
    import wanglaoshi.wqe as wqe_mod

    monkeypatch.delenv("WANGLAOSHI_SILENCE_LEGACY", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.reload(plain_mod)
        importlib.reload(wqe_mod)
    msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("Analyzer_Plain" in m for m in msgs)
    assert any("wqe" in m for m in msgs)


@pytest.mark.quick
def test_legacy_warning_can_be_silenced(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import warnings

    import wanglaoshi.Analyzer_Plain as plain_mod

    monkeypatch.setenv("WANGLAOSHI_SILENCE_LEGACY", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.reload(plain_mod)
    assert not [
        w for w in caught if issubclass(w.category, DeprecationWarning) and "Analyzer_Plain" in str(w.message)
    ]


@pytest.mark.quick
def test_competition_tools_split_reexports_same_objects() -> None:
    import wanglaoshi.CompetitionTools as CT
    from wanglaoshi.compete import (
        CrossValidator,
        ExperimentTracker,
        FeatureImportanceAnalyzer,
        HyperparameterOptimizer,
        LeakageDetector,
        ModelEnsemble,
        QuickBaseline,
        SubmissionGenerator,
    )
    from wanglaoshi.compete.submission import SubmissionGenerator as SG_mod

    assert CT.SubmissionGenerator is SubmissionGenerator is SG_mod
    assert CT.QuickBaseline is QuickBaseline
    assert CT.LeakageDetector is LeakageDetector
    assert CT.ModelEnsemble is ModelEnsemble
    assert CT.FeatureImportanceAnalyzer is FeatureImportanceAnalyzer
    assert CT.CrossValidator is CrossValidator
    assert CT.HyperparameterOptimizer is HyperparameterOptimizer
    assert CT.ExperimentTracker is ExperimentTracker


@pytest.mark.quick
def test_phase4_implementations_live_in_subpackages() -> None:
    from wanglaoshi.analyze import DataAnalyzer
    from wanglaoshi.features import AdvancedFeatureEngineer
    from wanglaoshi.analytics import TimeSeriesTools
    from wanglaoshi.runtime import ready
    from wanglaoshi._testing import assert_close
    from wanglaoshi.Analyzer import DataAnalyzer as DA_shim
    from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer as FE_shim
    from wanglaoshi.timeseries import TimeSeriesTools as TS_shim
    from wanglaoshi.asserting import assert_close as ac_shim

    assert DataAnalyzer.__module__.startswith("wanglaoshi.analyze.")
    assert AdvancedFeatureEngineer.__module__.startswith("wanglaoshi.features.")
    assert TimeSeriesTools.__module__.startswith("wanglaoshi.analytics.")
    assert ready.__module__.startswith("wanglaoshi.runtime.")
    assert assert_close.__module__.startswith("wanglaoshi._testing.")
    assert DA_shim is DataAnalyzer
    assert FE_shim is AdvancedFeatureEngineer
    assert TS_shim is TimeSeriesTools
    assert ac_shim is assert_close


@pytest.mark.quick
def test_setup_extras_include_compete_business() -> None:
    from pathlib import Path
    import re

    setup = Path(__file__).resolve().parents[1] / "setup.py"
    text = setup.read_text(encoding="utf-8")
    assert re.search(r"['\"]compete['\"]\s*:", text)
    assert re.search(r"['\"]business['\"]\s*:", text)
    assert re.search(r"['\"]dev['\"]\s*:", text)


@pytest.mark.quick
def test_check_architecture_script_passes() -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_architecture.py")],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASSED" in proc.stdout
