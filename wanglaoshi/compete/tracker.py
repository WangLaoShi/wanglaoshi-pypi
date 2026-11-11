"""Compete: 实验追踪（从 CompetitionTools 拆出）。

写入路径：校验可序列化 → 文件锁内重新加载 → 临时文件原子替换。
保存失败向调用者抛错，不截断已有历史。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@contextmanager
def _exclusive_file_lock(lock_path: str) -> Iterator[None]:
    """跨进程文件锁（fcntl；Windows 无 fcntl 时退化为同机 best-effort）。"""
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)) or ".", exist_ok=True)
    lock_f = open(lock_path, "a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        lock_f.close()


class ExperimentTracker:
    """记录和管理机器学习实验，包括参数、结果、模型版本等。"""

    def __init__(self, log_file: str = "experiments.json"):
        self.log_file = log_file
        self.experiments: List[Dict[str, Any]] = []
        self._lock_path = f"{log_file}.lock"
        if os.path.exists(log_file):
            try:
                self.experiments = self._read_experiments_unlocked()
                logger.info(f"加载了 {len(self.experiments)} 个历史实验记录")
            except Exception as e:
                logger.warning(f"加载实验记录失败: {str(e)}")
                self.experiments = []

    def _read_experiments_unlocked(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_file):
            return []
        with open(self.log_file, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return []
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"experiment log must be a JSON list, got {type(data).__name__}")
        return data

    @staticmethod
    def _serialize(experiments: List[Dict[str, Any]]) -> str:
        """先完整序列化；不可 JSON 化时抛错，避免半写。"""
        return json.dumps(experiments, ensure_ascii=False, indent=2)

    def _atomic_write_unlocked(self, text: str) -> None:
        directory = os.path.dirname(os.path.abspath(self.log_file)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".experiments_", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.log_file)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def log_experiment(
        self,
        experiment_name: str,
        model_name: str,
        features: List[str],
        params: Dict[str, Any],
        score: float,
        metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """记录实验；持久化失败时抛错且不损坏已有文件。"""
        experiment = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": experiment_name,
            "model_name": model_name,
            "features": features,
            "n_features": len(features),
            "params": params,
            "score": score,
            "metrics": metrics or {},
            "notes": notes,
        }

        with _exclusive_file_lock(self._lock_path):
            # 写入前重新读取，避免多实例/多 Notebook 互相覆盖
            on_disk = self._read_experiments_unlocked()
            tentative = list(on_disk) + [experiment]
            try:
                text = self._serialize(tentative)
            except TypeError as exc:
                raise TypeError(
                    f"experiment payload is not JSON-serializable: {exc}; "
                    "history file left unchanged"
                ) from exc
            self._atomic_write_unlocked(text)
            self.experiments = tentative

        logger.info(f"记录实验: {experiment_name}, 分数: {score:.4f}")
        return experiment

    def get_experiment_history(
        self,
        sort_by: str = "score",
        ascending: bool = False,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        if not self.experiments:
            logger.warning("没有实验记录")
            return pd.DataFrame()

        df = pd.DataFrame(self.experiments)
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        if top_n:
            df = df.head(top_n)
        return df

    def compare_experiments(
        self,
        experiment_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if experiment_names:
            experiments = [
                exp for exp in self.experiments if exp["experiment_name"] in experiment_names
            ]
        else:
            experiments = self.experiments

        if not experiments:
            logger.warning("没有找到要对比的实验")
            return pd.DataFrame()

        comparison = []
        for exp in experiments:
            comparison.append(
                {
                    "实验名称": exp["experiment_name"],
                    "模型": exp["model_name"],
                    "特征数": exp["n_features"],
                    "分数": exp["score"],
                    "时间": exp["timestamp"],
                }
            )

        df = pd.DataFrame(comparison).sort_values("分数", ascending=False)
        print("\n" + "=" * 60)
        print("实验对比")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60 + "\n")
        return df
