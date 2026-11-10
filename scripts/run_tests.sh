#!/usr/bin/env bash
# C01 / C10 / C12：分层测试入口（只跑 pytest）
#   ./scripts/run_tests.sh quick
#   ./scripts/run_tests.sh full
#   ./scripts/run_tests.sh coverage   # C12：产品口径覆盖率（见 .coveragerc）
#   ./scripts/run_tests.sh summary    # C10：junit + 回归汇总 JSON
#
# 发版请用总入口（测试 + API 说明书 + 烟测 + …）：
#   ./scripts/release_check.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"
ART="${ROOT}/artifacts"
mkdir -p "$ART"

case "$MODE" in
  quick)
    python -m pytest -m "quick or submission" tests/
    ;;
  full)
    python -m pytest tests/
    ;;
  coverage)
    # C12：产品模块行覆盖率（排除 Jupyter/可选 DL 等，见 .coveragerc）
    # 默认门槛 78；可用 COV_FAIL_UNDER=80 抬高
    FAIL_UNDER="${COV_FAIL_UNDER:-78}"
    python -m pytest -m "quick or submission" tests/ \
      --cov=wanglaoshi \
      --cov-config="${ROOT}/.coveragerc" \
      --cov-report=term-missing \
      --cov-report="xml:${ART}/coverage.xml" \
      --cov-report="json:${ART}/coverage.json" \
      --cov-fail-under="${FAIL_UNDER}"
    ;;
  summary)
    # C10：跑 quick 并写出回归汇总
    python -m pytest -m "quick or submission" tests/ --junitxml="${ART}/junit.xml"
    python "${ROOT}/scripts/summarize_junit.py" "${ART}/junit.xml" "${ART}/regression_summary.json"
    ;;
  release)
    # 兼容旧习惯：转到总入口
    echo "提示: run_tests.sh release 已合并进 ./scripts/release_check.sh" >&2
    exec bash "$ROOT/scripts/release_check.sh" release
    ;;
  *)
    echo "usage: $0 {quick|full|coverage|summary|release}" >&2
    echo "发版总入口: ./scripts/release_check.sh" >&2
    exit 2
    ;;
esac
