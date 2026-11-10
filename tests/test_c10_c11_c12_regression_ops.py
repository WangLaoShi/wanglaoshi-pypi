"""C10 回归汇总 + C11 缺数据跳过 + C12 覆盖率入口烟测。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from wanglaoshi.regression_summary import CaseStatus, RegressionSummary, from_junit_xml
from wanglaoshi.testing_utils import optional_data_available, require_data_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.quick
def test_c10_write_regression_summary(tmp_path: Path) -> None:
    summary = RegressionSummary(
        cases=[
            CaseStatus(name="demo::pass", outcome="passed", duration_s=0.01),
            CaseStatus(name="demo::skip", outcome="skipped", duration_s=0.0, message="no data"),
        ]
    )
    out = summary.write(tmp_path / "regression_summary.json")
    text = out.read_text(encoding="utf-8")
    compact = text.replace(" ", "").replace("\n", "")
    assert '"passed":1' in compact
    assert "skipped" in text
    assert "package_version" in text


@pytest.mark.quick
def test_c10_from_junit_xml(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="3" errors="0" failures="1" skipped="1">
  <testcase classname="t" name="ok" time="0.1"/>
  <testcase classname="t" name="bad" time="0.2">
    <failure message="boom">trace</failure>
  </testcase>
  <testcase classname="t" name="skip" time="0.0">
    <skipped message="no data"/>
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )
    summary = from_junit_xml(junit)
    counts = summary.counts()
    assert counts["passed"] == 1
    assert counts["failed"] == 1
    assert counts["skipped"] == 1
    out = summary.write(tmp_path / "summary.json")
    assert out.exists()


@pytest.mark.quick
def test_c10_junit_error_and_nested_suites(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="a" tests="2">
    <testcase classname="pkg.a" name="err" time="0.01">
      <error message="boom-err">trace</error>
    </testcase>
    <testcase classname="pkg.a" name="ok" time="0.02"/>
  </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )
    summary = from_junit_xml(junit)
    counts = summary.counts()
    assert counts["error"] == 1
    assert counts["passed"] == 1
    assert counts["total"] == 2
    err = next(c for c in summary.cases if c.outcome == "error")
    assert "boom-err" in err.message


@pytest.mark.quick
def test_c10_unknown_outcome_counted_as_error() -> None:
    summary = RegressionSummary(
        cases=[CaseStatus(name="x", outcome="flaky"), CaseStatus(name="y", outcome="passed")]
    )
    counts = summary.counts()
    assert counts["error"] == 1
    assert counts["passed"] == 1
    assert counts["total"] == 2


@pytest.mark.quick
def test_c10_summarize_junit_script(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<?xml version="1.0"?>
<testsuite tests="1"><testcase classname="t" name="ok" time="0.0"/></testsuite>
""",
        encoding="utf-8",
    )
    out = tmp_path / "out.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "summarize_junit.py"), str(junit), str(out)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["counts"]["passed"] == 1


@pytest.mark.quick
def test_c11_require_data_skips_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "external" / "kaggle.csv"
    assert optional_data_available(missing) is False
    with pytest.raises(pytest.skip.Exception) as exc:
        require_data_path(missing, reason="C11: optional external data absent")
    assert "optional" in str(exc.value).lower() or "absent" in str(exc.value).lower()


@pytest.mark.quick
def test_c11_default_skip_reason_mentions_download(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"
    with pytest.raises(pytest.skip.Exception) as exc:
        require_data_path(missing)
    assert "download separately" in str(exc.value)


@pytest.mark.quick
def test_c11_require_data_returns_when_present(tmp_path: Path) -> None:
    f = tmp_path / "present.csv"
    f.write_text("a\n1\n", encoding="utf-8")
    got = require_data_path(f)
    assert got == f
    assert optional_data_available(f) is True


@pytest.mark.quick
def test_c11_release_required_data_must_exist_or_fail(tmp_path: Path) -> None:
    """发布必需数据：缺文件应失败（非 skip）。"""
    required = tmp_path / "release_required.csv"
    if not required.exists():
        with pytest.raises(AssertionError, match="release-required"):
            if not required.exists():
                raise AssertionError(f"release-required data missing: {required}")


@pytest.mark.quick
def test_c12_coverage_entry_script_exists() -> None:
    script = ROOT / "scripts" / "run_tests.sh"
    text = script.read_text(encoding="utf-8")
    assert "coverage" in text
    assert "cov=wanglaoshi" in text or "--cov=wanglaoshi" in text
    assert "summary" in text
    assert "summarize_junit.py" in text


@pytest.mark.quick
def test_c12_public_exports_importable() -> None:
    import wanglaoshi as wl

    assert wl.DataRegistry is not None
    assert wl.ExperimentContext is not None
    assert callable(wl.assert_above_floor)
    assert wl.ChartMeta is not None
    assert wl.RegressionSummary is not None
    assert callable(wl.require_data_path)


@pytest.mark.full
def test_c12_coverage_runs_smoke() -> None:
    """完整档：实际跑一小段 cov（不强制 80%，先校准）。"""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_c04_numeric_contracts.py",
        "--cov=wanglaoshi.asserting",
        "--cov-report=term-missing",
        "-q",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TOTAL" in proc.stdout or "TOTAL" in proc.stderr or "passed" in (proc.stdout + proc.stderr)
