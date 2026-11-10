#!/usr/bin/env python3
"""C10：从 pytest --junitxml 生成回归汇总 JSON。

用法:
  python -m pytest tests/ -m quick --junitxml=artifacts/junit.xml
  python scripts/summarize_junit.py artifacts/junit.xml artifacts/regression_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wanglaoshi.regression_summary import from_junit_xml


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize pytest junitxml into regression JSON")
    ap.add_argument("junit_xml")
    ap.add_argument("out_json")
    args = ap.parse_args()
    summary = from_junit_xml(args.junit_xml)
    path = summary.write(args.out_json)
    counts = summary.counts()
    print(f"wrote {path} counts={counts}")
    return 0 if counts.get("failed", 0) == 0 and counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
