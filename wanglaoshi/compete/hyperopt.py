"""Compete: 超参数优化（从 CompetitionTools 拆出，行为不变）。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    超参数优化器
    
    提供智能的超参数优化方法，包括贝叶斯优化和自动调参。
    """
    
    def __init__(self):
        """初始化超参数优化器"""
        pass
    
    def bayesian_optimize(self,
                         model_class: Any,
                         param_space: Dict[str, Any],
                         X: pd.DataFrame,
                         y: pd.Series,
                         n_trials: int = 100,
                         cv: int = 5) -> Dict[str, Any]:
        """
        贝叶斯优化超参数
        
        使用Optuna进行高效的超参数搜索。
        
        参数:
            model_class: 模型类（未实例化）
            param_space: 参数空间字典
            X: 特征数据
            y: 目标变量
            n_trials: 试验次数
            cv: 交叉验证折数
            
        返回:
            最佳参数和最佳分数
        """
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            
            def objective(trial):
                # 从参数空间采样
                params = {}
                for param_name, param_config in param_space.items():
                    if isinstance(param_config, dict):
                        if param_config['type'] == 'categorical':
                            params[param_name] = trial.suggest_categorical(
                                param_name, param_config['choices']
                            )
                        elif param_config['type'] == 'int':
                            params[param_name] = trial.suggest_int(
                                param_name,
                                param_config['low'],
                                param_config['high'],
                                log=param_config.get('log', False)
                            )
                        elif param_config['type'] == 'float':
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config['low'],
                                param_config['high'],
                                log=param_config.get('log', False)
                            )
                    else:
                        params[param_name] = param_config
                
                # 创建模型
                model = model_class(**params)
                
                # 交叉验证：Optuna 始终 maximize「越大越好」的分数
                # 回归保留 neg_mean_squared_error，禁止取反成正 MSE 再 maximize
                if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                    scoring = 'accuracy'
                else:
                    scoring = 'neg_mean_squared_error'

                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                return float(scores.mean())

            is_clf = y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20
            scoring = 'accuracy' if is_clf else 'neg_mean_squared_error'

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            # 展示：回归换算回正 MSE；优化目标仍是 neg_mse
            best_objective = float(study.best_value)
            if scoring == 'neg_mean_squared_error':
                display_score = float(-best_objective)
                metric = 'mse'
                higher_is_better = False
            else:
                display_score = best_objective
                metric = 'accuracy'
                higher_is_better = True

            result = {
                'best_params': study.best_params,
                'best_score': display_score,
                'best_objective': best_objective,
                'scoring': scoring,
                'metric': metric,
                'higher_is_better': higher_is_better,
                'n_trials': n_trials,
            }

            logger.info(
                f"贝叶斯优化完成，最佳{metric}: {result['best_score']:.4f} "
                f"(objective={best_objective:.4f})"
            )
            logger.info(f"最佳参数: {result['best_params']}")

            return result
            
        except ImportError:
            logger.warning("Optuna未安装，无法使用贝叶斯优化")
            raise ImportError("需要安装Optuna: pip install optuna")
    
    def auto_tune(self,
                 model_class: Any,
                 X: pd.DataFrame,
                 y: pd.Series,
                 model_type: str = 'auto') -> Dict[str, Any]:
        """
        自动调参
        
        基于模型类型和数据特征，自动选择参数范围进行优化。
        
        参数:
            model_class: 模型类
            X: 特征数据
            y: 目标变量
            model_type: 模型类型（用于选择参数范围）
            
        返回:
            最佳参数和最佳分数
        """
        # 根据模型类型设置默认参数空间
        if 'RandomForest' in str(model_class):
            param_space = {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20}
            }
        elif 'XGB' in str(model_class) or 'XGBoost' in str(model_class):
            param_space = {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True}
            }
        else:
            # 默认参数空间
            param_space = {
                'C': {'type': 'float', 'low': 0.1, 'high': 100, 'log': True},
                'gamma': {'type': 'float', 'low': 0.001, 'high': 1, 'log': True}
            }
        
        return self.bayesian_optimize(model_class, param_space, X, y, n_trials=50)
