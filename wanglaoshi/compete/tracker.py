"""Compete: 实验追踪（从 CompetitionTools 拆出，行为不变）。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    实验追踪器
    
    记录和管理机器学习实验，包括参数、结果、模型版本等。
    """
    
    def __init__(self, log_file: str = 'experiments.json'):
        """
        初始化实验追踪器
        
        参数:
            log_file: 实验记录文件路径
        """
        self.log_file = log_file
        self.experiments = []
        
        # 如果文件存在，加载已有记录
        if os.path.exists(log_file):
            try:
                import json
                with open(log_file, 'r', encoding='utf-8') as f:
                    self.experiments = json.load(f)
                logger.info(f"加载了 {len(self.experiments)} 个历史实验记录")
            except Exception as e:
                logger.warning(f"加载实验记录失败: {str(e)}")
                self.experiments = []
    
    def log_experiment(self,
                      experiment_name: str,
                      model_name: str,
                      features: List[str],
                      params: Dict[str, Any],
                      score: float,
                      metrics: Optional[Dict[str, float]] = None,
                      notes: Optional[str] = None) -> None:
        """
        记录实验
        
        参数:
            experiment_name: 实验名称
            model_name: 模型名称
            features: 使用的特征列表
            params: 模型参数
            score: 主要分数
            metrics: 其他指标（可选）
            notes: 备注（可选）
        """
        from datetime import datetime
        
        experiment = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_name': experiment_name,
            'model_name': model_name,
            'features': features,
            'n_features': len(features),
            'params': params,
            'score': score,
            'metrics': metrics or {},
            'notes': notes
        }
        
        self.experiments.append(experiment)
        
        # 保存到文件
        self._save_experiments()
        
        logger.info(f"记录实验: {experiment_name}, 分数: {score:.4f}")
    
    def _save_experiments(self) -> None:
        """保存实验记录到文件"""
        try:
            import json
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.experiments, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存实验记录失败: {str(e)}")
    
    def get_experiment_history(self,
                              sort_by: str = 'score',
                              ascending: bool = False,
                              top_n: Optional[int] = None) -> pd.DataFrame:
        """
        获取实验历史
        
        参数:
            sort_by: 排序字段（'score', 'timestamp', 'n_features'等）
            ascending: 是否升序
            top_n: 返回前N个实验
            
        返回:
            实验历史DataFrame
        """
        if not self.experiments:
            logger.warning("没有实验记录")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.experiments)
        
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        
        if top_n:
            df = df.head(top_n)
        
        return df
    
    def compare_experiments(self,
                           experiment_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        对比实验
        
        参数:
            experiment_names: 要对比的实验名称列表（如果为None则对比所有）
            
        返回:
            对比结果DataFrame
        """
        if experiment_names:
            experiments = [exp for exp in self.experiments if exp['experiment_name'] in experiment_names]
        else:
            experiments = self.experiments
        
        if not experiments:
            logger.warning("没有找到要对比的实验")
            return pd.DataFrame()
        
        comparison = []
        for exp in experiments:
            comparison.append({
                '实验名称': exp['experiment_name'],
                '模型': exp['model_name'],
                '特征数': exp['n_features'],
                '分数': exp['score'],
                '时间': exp['timestamp']
            })
        
        df = pd.DataFrame(comparison).sort_values('分数', ascending=False)
        
        print("\n" + "="*60)
        print("实验对比")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60 + "\n")
        
        return df
