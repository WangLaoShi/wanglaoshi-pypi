#!/usr/bin/env python3
"""验收：当前可导入的 wanglaoshi 能用包内 SimHei 渲染中文标题（无缺字警告）。

用法（建议对 wheel 安装后的干净环境）：
  python scripts/verify_chinese_plot.py
  python scripts/verify_chinese_plot.py --out /tmp/zh_plot.png
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/evidence/artifacts/chinese_plot_smoke.png"),
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    import wanglaoshi  # noqa: F401 — 触发 analyze.core 中文字体注册
    from wanglaoshi.analyze import core as analyze_core

    font_path = Path(analyze_core.font_path)
    if not font_path.exists():
        print(f"FAIL: font missing at {font_path}")
        return 1
    print(f"OK font {font_path} bytes={font_path.stat().st_size}")

    prop = analyze_core.font_prop
    args.out.parent.mkdir(parents=True, exist_ok=True)

    glyph_warnings: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
            msg = record.getMessage()
            if "Glyph" in msg or "missing from font" in msg.lower():
                glyph_warnings.append(msg)

    mpl_logger = logging.getLogger("matplotlib.font_manager")
    handler = _Handler()
    mpl_logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.plot([0, 1, 2], [1, 3, 2])
            ax.set_title("中文标题验收：销售额趋势", fontproperties=prop)
            ax.set_xlabel("月份", fontproperties=prop)
            ax.set_ylabel("金额", fontproperties=prop)
            fig.tight_layout()
            fig.savefig(args.out, dpi=120)
            plt.close(fig)

        text_warns = [
            str(w.message)
            for w in caught
            if "Glyph" in str(w.message) or "missing from font" in str(w.message).lower()
        ]
    finally:
        mpl_logger.removeHandler(handler)

    if glyph_warnings or text_warns:
        print("FAIL: Chinese glyph warnings:")
        for w in glyph_warnings + text_warns:
            print(" -", w)
        return 1

    # 二次确认：常用汉字在 SimHei 中有字形
    from matplotlib import ft2font

    font = ft2font.FT2Font(str(font_path))
    for ch in "中文标题验收销售额趋势月份金额":
        if font.get_char_index(ord(ch)) == 0:
            print(f"FAIL: glyph missing for {ch!r} in {font_path.name}")
            return 1

    print(f"OK chinese plot {args.out.resolve()}")
    print("chinese plot assets OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
