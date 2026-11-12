#!/usr/bin/env bash
# 发布 / 日常验收：一条命令跑完该跑的。
#
#   ./scripts/release_check.sh          # 默认 = release（发版前）
#   ./scripts/release_check.sh quick    # 日常改代码后
#   ./scripts/release_check.sh full     # 最全：含案例证据 + 教材构建
#
# 不用背一长串脚本名。失败即停，最后打印清单。

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

MODE="${1:-release}"
START_TS=$(date +%s)
STEPS=()

step() {
  local title="$1"
  shift
  echo ""
  echo "════════════════════════════════════════"
  echo "▶ $title"
  echo "════════════════════════════════════════"
  "$@"
  STEPS+=("OK  $title")
}

on_err() {
  echo ""
  echo "失败。已完成步骤："
  printf '  %s\n' "${STEPS[@]:-（尚无）}"
  exit 1
}
trap 'on_err' ERR

case "$MODE" in
  quick|release|full) ;;
  -h|--help|help)
    sed -n '2,12p' "$0"
    exit 0
    ;;
  *)
    echo "usage: $0 {quick|release|full}" >&2
    exit 2
    ;;
esac

echo "wanglaoshi release_check  mode=$MODE  root=$ROOT"
echo "python: $(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"

# 0) full：先装 Notebook/教材依赖，避免证据步骤缺 nbformat
if [[ "$MODE" == "full" ]]; then
  step "安装 Notebook/教材依赖 [book]" python -m pip install -q -e ".[book]"
fi

# 1) 测试
if [[ "$MODE" == "quick" ]]; then
  step "测试 quick（quick|submission）" bash scripts/run_tests.sh quick
else
  step "测试 full（全部 pytest）" bash scripts/run_tests.sh full
fi

# 2) API 说明书（发版必跑，日常也跑——避免文档再漂）
step "API 说明书【可运行】示例" python scripts/verify_api_reference.py

# 3) 文档示例烟测
step "文档/示例烟测" python scripts/smoke_docs_examples.py

# 4) 发版档：导入 + 包内资源（字体/模板）
if [[ "$MODE" != "quick" ]]; then
  step "公开接口导入烟测" python -c "
import wanglaoshi
from wanglaoshi.runtime import ready, WebGetter
from wanglaoshi.analyze import DataAnalyzer
from wanglaoshi.compete import SubmissionGenerator
from wanglaoshi.business import BusinessPipeline
from wanglaoshi.WebGetter import Wget
print('ok', wanglaoshi.__file__, ready, DataAnalyzer, SubmissionGenerator, BusinessPipeline, Wget)
"
  step "包内中文字体与报告资源" python scripts/verify_wheel_assets.py
  step "中文图表渲染烟测" python scripts/verify_chinese_plot.py
  step "中文完整报告（含 PCA）缺字检查" python scripts/verify_chinese_report.py
  step "wheel 外装中文完整报告" python scripts/verify_wheel_chinese_report.py
  step "Optuna 调参方向专项" bash -c '
    python -m pip install -q "optuna>=3"
    python -m pytest -q tests/test_scoring_direction_metric_registry.py::test_hyperopt_regression_selects_lower_mse tests/test_hyperopt_optuna_live.py -q
  '
fi

# 5) 大版本：证据 + 教材（依赖已在步骤 0 安装）
if [[ "$MODE" == "full" ]]; then
  # 尽量准备 Bike/Mall；Fraud 需人工 CSV。台账失败不阻断，由 --release-data 最终判定。
  step "发布案例数据台账（Fraud/Bike/Mall）" bash -c '
    python scripts/fetch_release_case_data.py || true
  '
  step "22 进阶 Notebook + 发布数据验收" python scripts/run_advanced_evidence.py --release-data
  step "构建 Jupyter Book 教材" python scripts/build_book.py
fi

ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "════════════════════════════════════════"
echo "✅ release_check 通过  mode=$MODE  ${ELAPSED}s"
echo "════════════════════════════════════════"
printf '  %s\n' "${STEPS[@]}"
echo ""
echo "还需要你手动做的（脚本故意不替你做；详见 docs/RELEASE.md）："
echo "  1. 干净环境基线（建议）：./scripts/verify_clean_env.sh"
echo "  2. 改版本号：setup.py / README 版本摘要"
echo "  3. git commit / push（先于 tag）"
echo "  4. 构建：rm -rf build dist *.egg-info && python setup.py sdist bdist_wheel"
echo "  5. 检查并上传：twine check dist/* && twine upload dist/*"
echo "  6. 打 tag：git tag vX.Y.Z && git push origin vX.Y.Z"
echo "  7. wheel 安装后再跑：python scripts/verify_wheel_assets.py"
echo ""
echo "日常只记一句：  ./scripts/release_check.sh quick"
echo "发版前记一句：  ./scripts/release_check.sh"
echo "大版本记一句：  ./scripts/release_check.sh full"
echo "干净环境：      ./scripts/verify_clean_env.sh"
