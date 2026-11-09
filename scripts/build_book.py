#!/usr/bin/env python3
"""准备内容并构建 Jupyter Book。

用法：
  python scripts/build_book.py
  python scripts/build_book.py --serve   # 构建后提示本地打开路径
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build wanglaoshi Jupyter Book")
    parser.add_argument("--serve", action="store_true", help="print local html path")
    args = parser.parse_args()

    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "prepare_book.py")], cwd=ROOT)

    jb = shutil.which("jupyter-book") or shutil.which("jb")
    if jb is None:
        print("需要安装经典 Jupyter Book 1.x：pip install -e '.[book]'", file=sys.stderr)
        return 2

    cmd = [jb, "build", str(BOOK)]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)

    index = BOOK / "_build" / "html" / "index.html"
    if not index.exists():
        print(f"build failed: missing {index}", file=sys.stderr)
        print("请确认已安装经典 Jupyter Book 1.x：pip install 'jupyter-book>=1.0,<2'", file=sys.stderr)
        return 1
    print(f"build ok: {index}")
    if args.serve:
        print(f"open: file://{index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
