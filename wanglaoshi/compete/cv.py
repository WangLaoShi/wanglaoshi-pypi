"""Compete: 交叉验证（从 CompetitionTools 拆出，行为不变）。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    增强的交叉验证工具
    
    提供多种交叉验证方法，包括时间序列、分组、分层等。
    """
    
    def __init__(self):
        """初始化交叉验证工具"""
        pass
    
    def time_series_cv(self,
                      model: Any,
                      X: pd.DataFrame,
                      y: pd.Series,
                      n_splits: int = 5) -> Dict[str, Any]:
        """
        时间序列交叉验证
        
        适合时间序列数据，确保训练集的时间早于测试集。
        
        参数:
            model: 模型
            X: 特征数据
            y: 目标变量
            n_splits: 折数
            
        返回:
            交叉验证结果字典
        """
        try:
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # 判断是分类还是回归
            if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
            
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=-1)
            
            if scoring == 'neg_mean_squared_error':
                scores = -scores  # 转换为MSE
            
            result = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'cv_type': 'time_series'
            }
            
            logger.info(f"时间序列交叉验证完成，平均分数: {result['mean']:.4f} ± {result['std']:.4f}")
            
            return result
            
        except ImportError:
            logger.error("sklearn未安装，无法进行时间序列交叉验证")
            raise
    
    def group_cv(self,
                model: Any,
                X: pd.DataFrame,
                y: pd.Series,
                groups: pd.Series,
                n_splits: int = 5) -> Dict[str, Any]:
        """
        分组交叉验证
        
        确保同一组的数据不会同时出现在训练集和测试集中。
        
        参数:
            model: 模型
            X: 特征数据
            y: 目标变量
            groups: 分组变量
            n_splits: 折数
            
        返回:
            交叉验证结果字典
        """
        try:
            from sklearn.model_selection import GroupKFold, cross_val_score
            
            gkf = GroupKFold(n_splits=n_splits)
            
            # 判断是分类还是回归
            if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
            
            scores = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring=scoring, n_jobs=-1)
            
            if scoring == 'neg_mean_squared_error':
                scores = -scores
            
            result = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'cv_type': 'group'
            }
            
            logger.info(f"分组交叉验证完成，平均分数: {result['mean']:.4f} ± {result['std']:.4f}")
            
            return result
            
        except ImportError:
            logger.error("sklearn未安装，无法进行分组交叉验证")
            raise
    
    def stratified_cv(self,
                     model: Any,
                     X: pd.DataFrame,
                     y: pd.Series,
                     n_splits: int = 5) -> Dict[str, Any]:
        """
        分层交叉验证
        
        确保每折中各类别的比例与整体一致（适合不平衡数据）。
        
        参数:
            model: 模型
            X: 特征数据
            y: 目标变量
            n_splits: 折数
            
        返回:
            交叉验证结果字典
        """
        try:
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # 判断是分类还是回归
            if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                scoring = 'accuracy'
                scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
            else:
                scoring = 'neg_mean_squared_error'
                scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
                scores = -scores
            
            result = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'cv_type': 'stratified'
            }
            
            logger.info(f"分层交叉验证完成，平均分数: {result['mean']:.4f} ± {result['std']:.4f}")
            
            return result
            
        except ImportError:
            logger.error("sklearn未安装，无法进行分层交叉验证")
            raise
