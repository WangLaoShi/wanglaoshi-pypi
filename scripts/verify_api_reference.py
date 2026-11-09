#!/usr/bin/env python3
"""执行 docs/API_REFERENCE.md 中所有标注为【可运行】的 python 代码块。

用法（在仓库根目录）:
  PYTHONPATH=. python scripts/verify_api_reference.py
  # 或已 pip install -e . 后直接:
  python scripts/verify_api_reference.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "API_REFERENCE.md"


def extract_runnable_blocks(text: str) -> list[tuple[int, str]]:
    """提取以「# 【可运行】」开头的 python 围栏。"""
    blocks: list[tuple[int, str]] = []
    fence = re.compile(r"^```python\n(.*?)(?=^```)", re.M | re.S)
    for m in fence.finditer(text):
        body = m.group(1)
        start_line = text[: m.start()].count("\n") + 1
        if body.lstrip().startswith("# 【可运行】"):
            blocks.append((start_line, body))
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--doc",
        type=Path,
        default=DOC,
        help="API 说明书路径",
    )
    args = parser.parse_args()
    if not args.doc.exists():
        print(f"找不到文档: {args.doc}", file=sys.stderr)
        return 2

    # 保证可导入本地包
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    text = args.doc.read_text(encoding="utf-8")
    blocks = extract_runnable_blocks(text)
    if not blocks:
        print("未找到【可运行】代码块", file=sys.stderr)
        return 2

    print(f"共 {len(blocks)} 个【可运行】块，开始校验…")
    failed = 0
    with tempfile.TemporaryDirectory(prefix="api_ref_") as tmp:
        os.chdir(tmp)
        # 无界面绘图
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib

        matplotlib.use("Agg")

        for i, (line, body) in enumerate(blocks):
            code = body
            # 去掉首行标记注释，保留其余
            lines = code.splitlines(True)
            if lines and lines[0].startswith("# 【可运行】"):
                code = "".join(lines[1:])
            label = f"block#{i} @ {args.doc.name}:{line}"
            try:
                exec(compile(code, label, "exec"), {"__name__": "__main__"})
                print(f"  PASS {label}")
            except Exception as exc:  # noqa: BLE001 — 汇总报告
                failed += 1
                print(f"  FAIL {label}: {type(exc).__name__}: {exc}")
                traceback.print_exc()

    if failed:
        print(f"\n失败 {failed}/{len(blocks)}")
        return 1
    print(f"\n全部通过 {len(blocks)}/{len(blocks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
