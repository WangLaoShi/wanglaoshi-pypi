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
TUTORIALS = BOOK / "tutorials"
HANDBOOK = BOOK / "handbook"

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
    "13_ab_testing",
    "14_hypothesis_testing",
    "15_timeseries_forecast",
    "16_survival_churn",
    "17_text_eda",
    "18_anomaly_detection",
    "19_association_rules",
    "20_causal_did",
    "21_order_refund_reconciliation",
    "22_movielens_ndcg",
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
    "ANALYSIS_COVERAGE.md": "analysis-coverage.md",
}

# 门面教程：docs/tutorials → book/tutorials
TUTORIAL_COPIES = [
    "README.md",
    "01_ready_and_eda.md",
    "02_compete_baseline.md",
    "03_features_encode.md",
    "04_analytics_stats.md",
    "05_business_pipeline.md",
    "06_ml_preprocess.md",
    "07_leakage_and_cv.md",
    "08_timeseries_tools.md",
    "09_calibration_fairness.md",
    "10_testing_repro.md",
    "11_runtime_toolkit.md",
    "12_analyze_helpers.md",
    "13_compete_advanced.md",
    "14_analytics_extra.md",
    "15_ml_toolkit.md",
    "16_business_stages.md",
    "17_testing_extra.md",
]

# 使用手册：docs/handbook → book/handbook
HANDBOOK_COPIES = [
    "README.md",
    "01_data_analyzer.md",
    "02_feature_engineer.md",
    "03_compete_toolkit.md",
    "04_analytics_toolkit.md",
    "05_ml_guide.md",
    "06_business_guide.md",
]


def main() -> None:
    CASES.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)
    TUTORIALS.mkdir(parents=True, exist_ok=True)
    HANDBOOK.mkdir(parents=True, exist_ok=True)

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

    tut_src = ROOT / "docs" / "tutorials"
    for name in TUTORIAL_COPIES:
        src = tut_src / name
        if not src.exists():
            raise FileNotFoundError(f"missing tutorial: {src}")
        dst = TUTORIALS / name
        text = src.read_text(encoding="utf-8")
        # book 内相对链接：指向仓库路径的 scripts/examples 保留；docs 互链改相对
        text = text.replace("](../API_REFERENCE.md)", "](../reference/api)")
        text = text.replace("](../ARCHITECTURE.md)", "](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/docs/ARCHITECTURE.md)")
        text = text.replace("](../../examples/tutorials/", "](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/tutorials/")
        text = text.replace("](../../examples/advanced/", "](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/advanced/")
        text = text.replace("](../handbook/", "](../handbook/")
        banner = (
            "```{note}\n"
            f"本页从仓库 `docs/tutorials/{name}` 同步生成；请以仓库文件为编辑源。\n"
            "```\n\n"
        )
        if not text.lstrip().startswith("```{note}"):
            text = banner + text
        dst.write_text(text, encoding="utf-8")
        print(f"synced tut  {dst.relative_to(ROOT)}")

    hb_src = ROOT / "docs" / "handbook"
    for name in HANDBOOK_COPIES:
        src = hb_src / name
        if not src.exists():
            raise FileNotFoundError(f"missing handbook: {src}")
        dst = HANDBOOK / name
        text = src.read_text(encoding="utf-8")
        text = text.replace("](../API_REFERENCE.md)", "](../reference/api)")
        text = text.replace("](../tutorials/", "](../tutorials/")
        text = text.replace(
            "](../../examples/handbook/",
            "](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/handbook/",
        )
        text = text.replace(
            "](../../examples/advanced/",
            "](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/advanced/",
        )
        banner = (
            "```{note}\n"
            f"本页从仓库 `docs/handbook/{name}` 同步生成；请以仓库文件为编辑源。\n"
            "```\n\n"
        )
        if not text.lstrip().startswith("```{note}"):
            text = banner + text
        dst.write_text(text, encoding="utf-8")
        print(f"synced hb   {dst.relative_to(ROOT)}")

    print("prepare_book: done")


if __name__ == "__main__":
    main()
