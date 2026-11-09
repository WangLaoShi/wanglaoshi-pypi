#!/usr/bin/env python3
"""把 Notebook 与参考文档同步进 book/，供 Jupyter Book 构建。

Notebook 真相仍在 examples/advanced/；docs 真相仍在 docs/。
book/cases/*.ipynb 与部分 reference/*.md 为生成物。
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"
CASES = BOOK / "cases"
REF = BOOK / "reference"

NOTEBOOKS = [
    "01_iris",
    "02_titanic",
    "04_breast_cancer",
    "12_digit_recognizer",
    "06_california_housing",
    "03_wine_quality",
    "07_abalone",
    "08_bike_sharing",
    "09_mall_customers",
    "10_adult_census",
    "05_credit_card_fraud",
    "11_house_prices",
]

# (源相对仓库根, 同步到 book/cases/ 的文件名不含扩展名)
EXTRA_NOTEBOOKS = [
    ("examples/business_pipeline_course.ipynb", "business_pipeline"),
]

# api.md 由手册手写维护（链到 API_REFERENCE）；短盘点同步为 api-inventory.md
DOC_COPIES = {
    "API_INVENTORY.md": "api-inventory.md",
    "SKILL_API_MAPPING.md": "skill-mapping.md",
    "DEPENDENCIES.md": "dependencies.md",
    "evidence/ADVANCED_EVIDENCE.md": "evidence.md",
}


def main() -> None:
    CASES.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)

    src_dir = ROOT / "examples" / "advanced"
    for name in NOTEBOOKS:
        src = src_dir / f"{name}.ipynb"
        if not src.exists():
            raise FileNotFoundError(f"missing notebook: {src}")
        dst = CASES / f"{name}.ipynb"
        shutil.copy2(src, dst)
        print(f"synced case {dst.relative_to(ROOT)}")

    for src_rel, name in EXTRA_NOTEBOOKS:
        src = ROOT / src_rel
        if not src.exists():
            raise FileNotFoundError(f"missing notebook: {src}")
        dst = CASES / f"{name}.ipynb"
        shutil.copy2(src, dst)
        print(f"synced case {dst.relative_to(ROOT)} <- {src_rel}")

    for src_rel, dst_name in DOC_COPIES.items():
        src = ROOT / "docs" / src_rel
        if not src.exists():
            raise FileNotFoundError(f"missing doc: {src}")
        dst = REF / dst_name
        text = src.read_text(encoding="utf-8")
        # 手册内相对链接：docs 互链 → book/reference 文件名
        text = text.replace("](SKILL_API_MAPPING.md)", "](skill-mapping.md)")
        text = text.replace("](API_INVENTORY.md)", "](api-inventory.md)")
        text = text.replace("](API_REFERENCE.md)", "](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/docs/API_REFERENCE.md)")
        text = text.replace("](DEPENDENCIES.md)", "](dependencies.md)")
        banner = (
            "```{note}\n"
            f"本页从仓库 `docs/{src_rel}` 同步生成；请以仓库文件为编辑源。\n"
            "```\n\n"
        )
        if not text.lstrip().startswith("```{note}"):
            text = banner + text
        dst.write_text(text, encoding="utf-8")
        print(f"synced ref  {dst.relative_to(ROOT)}")

    print("prepare_book: done")


if __name__ == "__main__":
    main()
