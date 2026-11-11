#!/usr/bin/env bash
# 干净环境基线：临时 venv 安装可编辑包并跑 pytest，写出证据文件。
#
#   ./scripts/verify_clean_env.sh           # 全量 pytest
#   ./scripts/verify_clean_env.sh quick     # 仅 quick 标记
#
# 失败即停。证据写入 docs/evidence/CLEAN_ENV_BASELINE.md

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-full}"
# 优先非 conda 解释器，避免「venv 仍挂在 Anaconda 标准库上」造成基线歧义
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x /usr/bin/python3 ]]; then
  PY=/usr/bin/python3
else
  PY=python3
fi
VENV_DIR="${CLEAN_ENV_VENV:-$ROOT/.venv-clean-baseline}"
EVIDENCE="$ROOT/docs/evidence/CLEAN_ENV_BASELINE.md"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "clean-env baseline  mode=$MODE  python=$PY  venv=$VENV_DIR"

rm -rf "$VENV_DIR"
"$PY" -m venv "$VENV_DIR" --clear
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev]"

python - <<'PY'
import importlib.metadata as md
import platform
import sys
print("python", sys.version.replace("\n", " "))
print("platform", platform.platform())
for pkg in ("wanglaoshi", "numpy", "pandas", "scipy", "scikit-learn", "pytest"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg}==MISSING")
PY

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

START=$(date +%s)
set +e
if [[ "$MODE" == "quick" ]]; then
  python -m pytest -q -m "quick" --tb=line
else
  python -m pytest -q --tb=line
fi
CODE=$?
set -e
ELAPSED=$(( $(date +%s) - START ))

PY_VER="$(python -c 'import sys; print(sys.version.split()[0])')"
PLAT="$(python -c 'import platform; print(platform.platform())')"
DEPS="$(python - <<'PY'
import importlib.metadata as md
pkgs = ("wanglaoshi", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "statsmodels", "pytest")
for p in pkgs:
    try:
        print(f"- `{p}=={md.version(p)}`")
    except Exception:
        print(f"- `{p}==MISSING`")
PY
)"

STATUS="failed"
[[ "$CODE" -eq 0 ]] && STATUS="passed"

mkdir -p "$(dirname "$EVIDENCE")"
cat > "$EVIDENCE" <<EOF
# 干净环境测试基线

| 字段 | 值 |
| --- | --- |
| 状态 | **${STATUS}** |
| UTC 时间 | ${STAMP} |
| 耗时 | ${ELAPSED}s |
| 模式 | \`${MODE}\` |
| Python | ${PY_VER} |
| 平台 | ${PLAT} |
| 命令 | \`python -m venv .venv-clean-baseline && pip install -e ".[dev]" && pytest\` |
| 退出码 | ${CODE} |

## 已安装关键包

${DEPS}

## 说明

- 本文件由 \`scripts/verify_clean_env.sh\` 生成；**通过才可作为 F01 验证证据**。
- 本机 Anaconda / 污染环境上的 \`pytest\` 收集失败，**不能**代替本基线。
- venv 目录 \`.venv-clean-baseline/\` 默认不提交；可删可重建。
EOF

echo ""
echo "evidence -> $EVIDENCE  status=$STATUS  exit=$CODE"
deactivate || true
exit "$CODE"
