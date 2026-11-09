#!/usr/bin/env bash
# C01：分层测试入口
#   ./scripts/run_tests.sh quick
#   ./scripts/run_tests.sh full
#   ./scripts/run_tests.sh release

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"

case "$MODE" in
  quick)
    python -m pytest -m "quick or submission" tests/
    ;;
  full)
    python -m pytest tests/
    ;;
  release)
    python -m pytest tests/
    python -c "import wanglaoshi; from wanglaoshi import BusinessPipeline, CompetitionTools, DataAnalyzer; print('import_ok', wanglaoshi.__file__)"
    ;;
  *)
    echo "usage: $0 {quick|full|release}" >&2
    exit 2
    ;;
esac
