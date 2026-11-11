"""分组表现 / 公平性最小报表（P1）。对齐 E10 增补。"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class GroupMetricsReport:
    """按敏感属性交叉分组输出 n / TPR / FPR 等。"""

    @staticmethod
    def audit(
        y_true: Any,
        y_pred: Any,
        groups: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    ) -> pd.DataFrame:
        """按组评估。

        y / groups 一律按**位置**对齐（先 ``reset_index``），避免把 pandas
        标签索引误当成 NumPy 下标。
        """
        y_true_a = np.asarray(y_true)
        y_pred_a = np.asarray(y_pred)
        if len(y_true_a) != len(y_pred_a):
            raise ValueError("y_true and y_pred length mismatch")

        if isinstance(groups, pd.Series):
            gdf = groups.astype(str).to_frame("group")
        elif isinstance(groups, dict):
            gdf = pd.DataFrame({k: pd.Series(v).astype(str).to_numpy() for k, v in groups.items()})
        else:
            gdf = groups.astype(str).copy()

        if len(gdf) != len(y_true_a):
            raise ValueError("groups length mismatch")

        # 强制位置对齐：丢弃调用方索引标签
        gdf = gdf.reset_index(drop=True)
        cols = list(gdf.columns)
        rows: List[Dict[str, Any]] = []
        for key, idx in gdf.groupby(cols, sort=False).groups.items():
            positions = np.asarray(list(idx), dtype=int)
            yt = y_true_a[positions]
            yp = y_pred_a[positions]
            if len(yt) == 0:
                continue
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else np.nan
            fpr = fp / (fp + tn) if (fp + tn) else np.nan
            label = key if not isinstance(key, tuple) else "|".join(map(str, key))
            rows.append(
                {
                    "group": label,
                    "n": int(len(yt)),
                    "pos_rate": float(np.mean(yt)),
                    "pred_pos_rate": float(np.mean(yp)),
                    "tpr": float(tpr) if tpr == tpr else np.nan,
                    "fpr": float(fpr) if fpr == fpr else np.nan,
                    "fn": int(fn),
                    "fp": int(fp),
                }
            )
        return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
