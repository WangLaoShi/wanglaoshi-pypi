"""C10：回归汇总产物（案例×状态、跳过/失败原因、耗时、版本）。"""

from __future__ import annotations

import json
import platform
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CaseStatus:
    name: str
    outcome: str  # passed | failed | skipped | error
    duration_s: float = 0.0
    message: str = ""
    metric: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegressionSummary:
    cases: List[CaseStatus] = field(default_factory=list)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    package_version: str = ""

    def counts(self) -> Dict[str, int]:
        out = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "total": len(self.cases)}
        for c in self.cases:
            key = c.outcome if c.outcome in out else "error"
            out[key] = out.get(key, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at_utc": self.created_at_utc,
            "python": self.python,
            "platform": self.platform,
            "package_version": self.package_version,
            "counts": self.counts(),
            "cases": [c.to_dict() for c in self.cases],
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p


def _package_version() -> str:
    try:
        from wanglaoshi import __version__  # type: ignore

        return str(__version__)
    except Exception:
        try:
            from importlib.metadata import version

            return version("wanglaoshi")
        except Exception:
            return "unknown"


def from_junit_xml(junit_path: str | Path) -> RegressionSummary:
    """从 pytest --junitxml 产物汇总。"""
    root = ET.parse(Path(junit_path)).getroot()
    cases: List[CaseStatus] = []
    # junit 可能是 testsuites / testsuite
    suites = root.findall(".//testcase")
    for tc in suites:
        name = f"{tc.get('classname', '')}::{tc.get('name', '')}".strip(":")
        duration = float(tc.get("time") or 0.0)
        fail = tc.find("failure")
        err = tc.find("error")
        skip = tc.find("skipped")
        if fail is not None:
            outcome, msg = "failed", (fail.get("message") or "")[:500]
        elif err is not None:
            outcome, msg = "error", (err.get("message") or "")[:500]
        elif skip is not None:
            outcome, msg = "skipped", (skip.get("message") or skip.text or "")[:500]
        else:
            outcome, msg = "passed", ""
        cases.append(CaseStatus(name=name, outcome=outcome, duration_s=duration, message=msg))
    return RegressionSummary(cases=cases, package_version=_package_version())
