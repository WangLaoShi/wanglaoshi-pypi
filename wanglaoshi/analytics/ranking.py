"""排序指标（P1）。对齐 E22。"""

from __future__ import annotations

from typing import Iterable, Sequence, Set, Union

import numpy as np


class RankingMetrics:
    """NDCG / 命中类 Top-K 指标。"""

    @staticmethod
    def dcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
        rels = np.asarray(list(relevances)[:k], dtype=float)
        if rels.size == 0:
            return 0.0
        discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
        return float((rels * discounts).sum())

    @staticmethod
    def ndcg_at_k(
        recommended: Sequence,
        relevant: Union[Set, Iterable],
        k: int = 10,
    ) -> float:
        """NDCG@k。

        同一相关项多次出现时**只计首次命中**，避免重复推荐刷高分数。
        有效输入下返回值落在 ``[0, 1]``。
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        rel_set = set(relevant)
        seen: Set = set()
        rels = []
        for item in list(recommended)[:k]:
            if item in rel_set and item not in seen:
                rels.append(1.0)
                seen.add(item)
            else:
                rels.append(0.0)
        dcg = RankingMetrics.dcg_at_k(rels, k)
        ideal = RankingMetrics.dcg_at_k([1.0] * min(k, len(rel_set)), k)
        if ideal == 0:
            return 0.0
        score = float(dcg / ideal)
        return float(min(1.0, max(0.0, score)))

    @staticmethod
    def hit_rate_at_k(recommended: Sequence, relevant: Union[Set, Iterable], k: int = 10) -> float:
        rel_set = set(relevant)
        return 1.0 if any(i in rel_set for i in list(recommended)[:k]) else 0.0
