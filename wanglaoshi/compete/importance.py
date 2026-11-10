"""Compete: 特征重要性（从 CompetitionTools 拆出，行为不变）。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    特征重要性分析器
    
    提供多种方法分析特征重要性，包括：
    - Permutation Importance
    - SHAP值（如果可用）
    - 模型内置重要性（树模型）
    - 相关性分析
    
    使用示例:
        analyzer = FeatureImportanceAnalyzer()
        importance = analyzer.calculate_importance(model, X, y, method='shap')
        analyzer.plot_importance(importance, top_n=20)
    """
    
    def __init__(self):
        """初始化特征重要性分析器"""
        pass
    
    def calculate_importance(self,
                            model: Any,
                            X: pd.DataFrame,
                            y: pd.Series,
                            method: str = 'permutation',
                            n_repeats: int = 10) -> pd.DataFrame:
        """
        计算特征重要性
        
        参数:
            model: 训练好的模型
            X: 特征数据
            y: 目标变量
            method: 计算方法
                   - 'permutation': 排列重要性（最可靠）
                   - 'shap': SHAP值（如果可用）
                   - 'builtin': 模型内置重要性（仅树模型）
            n_repeats: 排列重要性的重复次数
            
        返回:
            特征重要性DataFrame
        """
        if method == 'permutation':
            return self._permutation_importance(model, X, y, n_repeats)
        elif method == 'shap':
            return self._shap_importance(model, X, y)
        elif method == 'builtin':
            return self._builtin_importance(model, X)
        else:
            raise ValueError(f"不支持的重要性计算方法: {method}")
    
    def _permutation_importance(self,
                               model: Any,
                               X: pd.DataFrame,
                               y: pd.Series,
                               n_repeats: int = 10) -> pd.DataFrame:
        """计算排列重要性"""
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            # 判断是分类还是回归
            if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
            
            result = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=42,
                scoring=scoring,
                n_jobs=-1
            )
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            logger.info(f"计算了排列重要性（重复{n_repeats}次）")
            
            return importance_df
            
        except ImportError:
            logger.error("sklearn未安装，无法计算排列重要性")
            raise
    
    def _shap_importance(self,
                        model: Any,
                        X: pd.DataFrame,
                        y: pd.Series) -> pd.DataFrame:
        """计算SHAP值"""
        try:
            import shap
            
            # 创建SHAP解释器
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.Explainer(model)
            else:
                explainer = shap.Explainer(model)
            
            # 计算SHAP值（使用样本以减少计算时间）
            sample_size = min(100, len(X))
            X_sample = X.sample(n=sample_size, random_state=42)
            shap_values = explainer.shap_values(X_sample)
            
            # 处理多分类情况
            if isinstance(shap_values, list):
                # 多分类，取平均
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values = np.abs(shap_values)
            
            # 计算平均重要性
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': np.mean(shap_values, axis=0)
            }).sort_values('importance_mean', ascending=False)
            
            logger.info("计算了SHAP重要性")
            
            return importance_df
            
        except ImportError:
            logger.warning("SHAP未安装，改用排列重要性")
            return self._permutation_importance(model, X, y)
    
    def _builtin_importance(self,
                           model: Any,
                           X: pd.DataFrame) -> pd.DataFrame:
        """使用模型内置的重要性"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': model.feature_importances_
            }).sort_values('importance_mean', ascending=False)
            
            logger.info("使用了模型内置的重要性")
            return importance_df
        else:
            logger.warning("模型不支持内置重要性，改用排列重要性")
            raise ValueError("模型不支持内置重要性")
    
    def plot_importance(self,
                       importance_df: pd.DataFrame,
                       top_n: int = 20,
                       figsize: tuple = (10, 8)) -> None:
        """
        绘制特征重要性图
        
        参数:
            importance_df: 特征重要性DataFrame
            top_n: 显示前N个特征
            figsize: 图表大小
        """
        try:
            import matplotlib.pyplot as plt
            
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=figsize)
            plt.barh(range(len(top_features)), top_features['importance_mean'].values)
            plt.yticks(range(len(top_features)), top_features['feature'].values)
            plt.xlabel('重要性')
            plt.title(f'特征重要性 Top {top_n}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            logger.info(f"绘制了前 {top_n} 个重要特征")
            
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制图表")
