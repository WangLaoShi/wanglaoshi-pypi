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
        rel_set = set(relevant)
        rels = [1.0 if i in rel_set else 0.0 for i in list(recommended)[:k]]
        dcg = RankingMetrics.dcg_at_k(rels, k)
        ideal = RankingMetrics.dcg_at_k([1.0] * min(k, len(rel_set)), k)
        return 0.0 if ideal == 0 else float(dcg / ideal)

    @staticmethod
    def hit_rate_at_k(recommended: Sequence, relevant: Union[Set, Iterable], k: int = 10) -> float:
        rel_set = set(relevant)
        return 1.0 if any(i in rel_set for i in list(recommended)[:k]) else 0.0
