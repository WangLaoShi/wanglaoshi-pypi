"""Compete: 模型集成（从 CompetitionTools 拆出，行为不变）。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """
    模型集成工具
    
    提供多种模型集成方法，包括投票、堆叠、Blending等。
    模型集成是数据科学比赛中提高成绩的关键技术。
    
    使用示例:
        ensemble = ModelEnsemble()
        # 堆叠集成
        stacked_model = ensemble.stacking(models=[model1, model2, model3], 
                                         meta_model=meta_model,
                                         X_train=X_train, y_train=y_train)
    """
    
    def __init__(self):
        """初始化模型集成工具"""
        pass
    
    def voting(self,
              models: List[Any],
              X: pd.DataFrame,
              y: pd.Series,
              voting: str = 'soft',
              weights: Optional[List[float]] = None) -> Any:
        """
        投票集成（Voting Ensemble）
        
        多个模型投票决定最终预测结果。
        
        参数:
            models: 模型列表
            X: 特征数据
            y: 目标变量
            voting: 投票方式
                   - 'hard': 硬投票（分类问题，直接投票）
                   - 'soft': 软投票（分类问题，使用概率）
            weights: 模型权重列表（可选）
            
        返回:
            训练好的投票集成模型
        """
        try:
            from sklearn.ensemble import VotingClassifier, VotingRegressor
            
            # 判断是分类还是回归
            if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                # 分类问题
                if voting == 'soft':
                    # 确保所有模型都支持predict_proba
                    ensemble = VotingClassifier(
                        estimators=[(f'model_{i}', model) for i, model in enumerate(models)],
                        voting='soft',
                        weights=weights
                    )
                else:
                    ensemble = VotingClassifier(
                        estimators=[(f'model_{i}', model) for i, model in enumerate(models)],
                        voting='hard',
                        weights=weights
                    )
            else:
                # 回归问题
                ensemble = VotingRegressor(
                    estimators=[(f'model_{i}', model) for i, model in enumerate(models)],
                    weights=weights
                )
            
            ensemble.fit(X, y)
            logger.info(f"训练了投票集成模型（{len(models)}个模型，voting={voting}）")
            
            return ensemble
            
        except ImportError:
            logger.error("sklearn未安装，无法使用投票集成")
            raise
    
    def stacking(self,
                models: List[Any],
                meta_model: Any,
                X_train: pd.DataFrame,
                y_train: pd.Series,
                X_test: Optional[pd.DataFrame] = None,
                cv: int = 5,
                use_features_in_secondary: bool = False) -> Any:
        """
        堆叠集成（Stacking Ensemble）
        
        使用第一层模型的预测结果作为第二层模型的输入。
        
        参数:
            models: 第一层模型列表
            meta_model: 第二层（元）模型
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征（可选，用于生成最终预测）
            cv: 交叉验证折数
            use_features_in_secondary: 第二层模型是否也使用原始特征
            
        返回:
            训练好的堆叠模型（或预测结果）
        """
        try:
            from sklearn.model_selection import cross_val_predict, KFold
            
            logger.info(f"开始堆叠集成，第一层有 {len(models)} 个模型")
            
            # 第一层：使用交叉验证生成预测
            first_level_predictions = []
            
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            for i, model in enumerate(models):
                logger.info(f"训练第一层模型 {i+1}/{len(models)}")
                
                # 使用交叉验证生成预测
                if hasattr(model, 'predict_proba') and len(y_train.unique()) < 20:
                    # 分类问题，使用概率
                    pred = cross_val_predict(model, X_train, y_train, cv=kf, method='predict_proba')
                    if pred.ndim > 1:
                        # 多分类，使用所有类别的概率
                        first_level_predictions.append(pred)
                    else:
                        first_level_predictions.append(pred.reshape(-1, 1))
                else:
                    # 回归问题或二分类
                    pred = cross_val_predict(model, X_train, y_train, cv=kf)
                    first_level_predictions.append(pred.reshape(-1, 1))
            
            # 合并第一层预测
            first_level_train = np.hstack(first_level_predictions)
            
            # 如果使用原始特征，合并特征
            if use_features_in_secondary:
                first_level_train = np.hstack([X_train.values, first_level_train])
            
            # 第二层：训练元模型
            logger.info("训练第二层（元）模型")
            meta_model.fit(first_level_train, y_train)

            # 两条路径共用：在完整训练集上拟合第一层，保证返回的模型可直接预测新数据
            for model in models:
                model.fit(X_train, y_train)

            # 如果提供了测试集，生成最终预测
            if X_test is not None:
                first_level_test = []
                for model in models:
                    if hasattr(model, 'predict_proba') and len(y_train.unique()) < 20:
                        pred = model.predict_proba(X_test)
                        if pred.ndim > 1:
                            first_level_test.append(pred)
                        else:
                            first_level_test.append(pred.reshape(-1, 1))
                    else:
                        pred = model.predict(X_test)
                        first_level_test.append(pred.reshape(-1, 1))

                first_level_test = np.hstack(first_level_test)

                if use_features_in_secondary:
                    first_level_test = np.hstack([X_test.values, first_level_test])

                final_predictions = meta_model.predict(first_level_test)

                logger.info("堆叠集成完成，返回预测结果")
                return final_predictions

            logger.info("堆叠集成完成")
            return {
                'meta_model': meta_model,
                'first_level_models': models,
                'use_features_in_secondary': use_features_in_secondary
            }
            
        except ImportError:
            logger.error("sklearn未安装，无法使用堆叠集成")
            raise
    
    def blending(self,
                models: List[Any],
                X_train: pd.DataFrame,
                y_train: pd.Series,
                X_val: pd.DataFrame,
                y_val: pd.Series,
                meta_model: Optional[Any] = None,
                weights: Optional[List[float]] = None) -> Any:
        """
        Blending集成
        
        Blending是Stacking的简化版本，使用一个固定的验证集而不是交叉验证。
        
        参数:
            models: 第一层模型列表
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            meta_model: 第二层模型（如果为None，使用简单平均）
            weights: 模型权重（仅当meta_model为None时使用）
            
        返回:
            训练好的集成模型或预测结果
        """
        logger.info(f"开始Blending集成，第一层有 {len(models)} 个模型")
        
        # 第一层：在训练集上训练，在验证集上预测
        first_level_predictions = []
        
        for i, model in enumerate(models):
            logger.info(f"训练第一层模型 {i+1}/{len(models)}")
            model.fit(X_train, y_train)
            
            if hasattr(model, 'predict_proba') and len(y_train.unique()) < 20:
                pred = model.predict_proba(X_val)
                if pred.ndim > 1:
                    first_level_predictions.append(pred)
                else:
                    first_level_predictions.append(pred.reshape(-1, 1))
            else:
                pred = model.predict(X_val)
                first_level_predictions.append(pred.reshape(-1, 1))
        
        first_level_val = np.hstack(first_level_predictions)

        # 第二层
        if meta_model is None:
            if weights is None:
                weights = [1.0 / len(models)] * len(models)
            if len(weights) != len(models):
                raise ValueError(
                    f"权重数量 ({len(weights)}) 与模型数量 ({len(models)}) 不匹配"
                )
            w = np.asarray(weights, dtype=float)
            if not np.isfinite(w).all() or np.any(w < 0) or float(w.sum()) <= 0:
                raise ValueError("weights must be finite, non-negative, and sum to a positive value")

            logger.info("使用加权平均作为第二层")

            # 分类：按 classes_ 对齐概率列后按列融合；回归：一维分数加权平均
            is_clf = all(hasattr(m, "predict_proba") for m in models) and len(y_train.unique()) < 20
            classes_: Optional[np.ndarray] = None
            if is_clf:
                # 以训练集标签顺序为权威列序，再按各模型 classes_ 对齐
                classes_ = np.asarray(sorted(pd.unique(y_train), key=lambda x: str(x)))

                def _align_proba(model: Any, X_test: pd.DataFrame) -> np.ndarray:
                    assert classes_ is not None
                    raw = np.asarray(model.predict_proba(X_test), dtype=float)
                    model_classes = np.asarray(getattr(model, "classes_"))
                    aligned = np.zeros((raw.shape[0], len(classes_)), dtype=float)
                    index = {c: i for i, c in enumerate(classes_)}
                    for j, c in enumerate(model_classes):
                        if c not in index:
                            raise ValueError(
                                f"model class {c!r} not in training labels {classes_.tolist()}"
                            )
                        aligned[:, index[c]] = raw[:, j]
                    return aligned

                def predict_proba(X_test):
                    mats = [_align_proba(m, X_test) for m in models]
                    stacked = np.stack(mats, axis=0)  # (n_models, n_samples, n_classes)
                    return np.average(stacked, axis=0, weights=w)

                def predict(X_test):
                    assert classes_ is not None
                    proba = predict_proba(X_test)
                    return classes_[np.argmax(proba, axis=1)]

                return {
                    "models": models,
                    "weights": weights,
                    "classes_": classes_,
                    "predict": predict,
                    "predict_proba": predict_proba,
                }

            def predict(X_test):
                model_scores = []
                for model in models:
                    model_scores.append(np.asarray(model.predict(X_test), dtype=float).reshape(-1))
                stacked = np.vstack(model_scores)
                return np.average(stacked, axis=0, weights=w)

            return {
                "models": models,
                "weights": weights,
                "predict": predict,
            }
        else:
            # 使用元模型
            logger.info("训练第二层（元）模型")
            meta_model.fit(first_level_val, y_val)
            
            def predict(X_test):
                test_predictions = []
                for model in models:
                    if hasattr(model, 'predict_proba') and len(y_train.unique()) < 20:
                        pred = model.predict_proba(X_test)
                        if pred.ndim > 1:
                            test_predictions.append(pred)
                        else:
                            test_predictions.append(pred.reshape(-1, 1))
                    else:
                        pred = model.predict(X_test)
                        test_predictions.append(pred.reshape(-1, 1))
                
                test_predictions = np.hstack(test_predictions)
                return meta_model.predict(test_predictions)
            
            return {
                'models': models,
                'meta_model': meta_model,
                'predict': predict
            }
