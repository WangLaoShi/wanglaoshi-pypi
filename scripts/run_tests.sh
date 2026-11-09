#!/usr/bin/env bash
# C01：分层测试入口（只跑 pytest）
#   ./scripts/run_tests.sh quick
#   ./scripts/run_tests.sh full
#
# 发版请用总入口（测试 + API 说明书 + 烟测 + …）：
#   ./scripts/release_check.sh

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
    # 兼容旧习惯：转到总入口
    echo "提示: run_tests.sh release 已合并进 ./scripts/release_check.sh" >&2
    exec bash "$ROOT/scripts/release_check.sh" release
    ;;
  *)
    echo "usage: $0 {quick|full|release}" >&2
    echo "发版总入口: ./scripts/release_check.sh" >&2
    exit 2
    ;;
esac
