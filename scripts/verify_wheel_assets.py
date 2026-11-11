#!/usr/bin/env python3
"""验收：当前可导入的 wanglaoshi 包内是否含中文字体与报告模板。

用法（源码可编辑安装或 wheel 安装后均可）：
  python scripts/verify_wheel_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    import wanglaoshi

    root = Path(wanglaoshi.__file__).resolve().parent
    font = root / "SimHei.ttf"
    template = root / "templates" / "analyzer.html"
    js = root / "static" / "js" / "analyzer.js"

    errors = []
    if not font.exists() or font.stat().st_size < 1000:
        errors.append(f"missing or tiny font: {font}")
    else:
        print(f"OK font {font} bytes={font.stat().st_size}")
    if not template.exists():
        errors.append(f"missing template: {template}")
    else:
        print(f"OK template {template}")
    if not js.exists():
        errors.append(f"missing analyzer.js: {js}")
    else:
        print(f"OK js {js}")

    if errors:
        print("FAIL:")
        for e in errors:
            print(" -", e)
        return 1
    print("wheel/package assets OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
