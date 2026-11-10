"""C05：公开/内置集上的模型分数下限（防回归）。

阈值刻意保守，覆盖 sklearn 内置玩具数据；外部竞赛集由 C11 缺数据策略跳过。
"""

from __future__ import annotations

from typing import Dict

# 任务 -> 主指标下限（越高越好）
SCORE_FLOORS: Dict[str, Dict[str, float]] = {
    "iris_accuracy": {"metric": "accuracy", "floor": 0.85},
    "breast_cancer_roc_auc": {"metric": "roc_auc", "floor": 0.90},
    "diabetes_r2": {"metric": "r2", "floor": 0.30},
}


def assert_above_floor(task: str, score: float) -> None:
    if task not in SCORE_FLOORS:
        raise KeyError(f"unknown score floor task: {task}")
    floor = SCORE_FLOORS[task]["floor"]
    if score < floor:
        raise AssertionError(f"{task}: score {score:.4f} < floor {floor:.4f}")
