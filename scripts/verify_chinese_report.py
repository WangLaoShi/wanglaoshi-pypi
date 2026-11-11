#!/usr/bin/env python3
"""验收：用当前可导入的 wanglaoshi 生成含中文列名的完整 HTML 报告，捕获缺字警告。

覆盖分布图、相关热图、PCA 碎石图等路径。建议在 wheel 安装后的干净环境跑：

  python scripts/verify_chinese_report.py
  python scripts/verify_chinese_report.py --out /tmp/zh_report.html
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/evidence/artifacts/chinese_report_smoke.html"),
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    import numpy as np
    import pandas as pd

    from wanglaoshi.analyze import DataAnalyzer
    from wanglaoshi.analyze import core as analyze_core

    font_path = Path(analyze_core.font_path)
    if not font_path.exists():
        print(f"FAIL: font missing at {font_path}")
        return 1
    print(f"OK font {font_path} bytes={font_path.stat().st_size}")
    print(f"OK font family={analyze_core.font_prop.get_name()}")

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "销售额": rng.normal(100, 20, 80),
            "成本": rng.normal(60, 10, 80),
            "利润": rng.normal(40, 8, 80),
            "区域": rng.choice(["华东", "华南", "华北"], size=80),
        }
    )

    glyph_logs: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if "Glyph" in msg or "missing from font" in msg.lower() or "DejaVu" in msg:
                glyph_logs.append(msg)

    # matplotlib 缺字既可能走 logging，也可能走 warnings
    for name in ("matplotlib", "matplotlib.font_manager", "matplotlib.backends"):
        logging.getLogger(name).addHandler(_Handler())
        logging.getLogger(name).setLevel(logging.WARNING)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer = DataAnalyzer(df)
        # 显式跑会出图的章节（PCA 此前未传 fontproperties）
        analyzer.pca_analysis()
        analyzer.visualizer.plot_distribution(df["销售额"], "销售额")
        analyzer.visualizer.plot_distribution(df["区域"], "区域")
        corr = df[["销售额", "成本", "利润"]].corr()
        analyzer.visualizer.plot_correlation_heatmap(corr)
        analyzer.generate_report(str(args.out))

        text_warns = [
            str(w.message)
            for w in caught
            if any(
                k in str(w.message)
                for k in ("Glyph", "missing from font", "DejaVu Sans")
            )
        ]

    out_file = args.out
    if not out_file.exists():
        print(f"FAIL: report not written ({out_file})")
        return 1

    html = out_file.read_text(encoding="utf-8", errors="replace")
    # 列名在模板 JSON 里常以 \\uXXXX 出现；图为 PNG base64
    has_zh = ("销售额" in html) or ("\\u9500\\u552e\\u989d" in html)
    has_img = ("iVBORw0KGgo" in html) or ("data:image" in html)
    if not has_zh or not has_img:
        print(
            f"FAIL: report missing Chinese content / embedded images "
            f"(zh={has_zh}, img={has_img})"
        )
        return 1

    problems = glyph_logs + text_warns
    # 去重保序
    uniq: list[str] = []
    for p in problems:
        if p not in uniq:
            uniq.append(p)

    if uniq:
        print(f"FAIL: {len(uniq)} chinese glyph / font fallback warnings")
        for w in uniq[:20]:
            print(" -", w)
        if len(uniq) > 20:
            print(f" - ... and {len(uniq) - 20} more")
        return 1

    print(f"OK chinese report {out_file.resolve()}")
    print("chinese report OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
