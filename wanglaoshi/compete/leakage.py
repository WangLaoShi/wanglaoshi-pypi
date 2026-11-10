"""Compete: 数据泄露检测（从 CompetitionTools 拆出，行为不变）。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LeakageDetector:
    """
    数据泄露检测器
    
    检测常见的数据泄露问题，这是数据科学比赛中最容易犯的错误之一。
    
    使用示例:
        detector = LeakageDetector()
        # 检测目标泄露
        leakage_features = detector.detect_target_leakage(X, y, threshold=0.9)
        # 检测时间泄露
        time_leakage = detector.detect_time_leakage(df, date_col='date', target_col='target')
    """
    
    def __init__(self):
        """初始化数据泄露检测器"""
        pass
    
    def detect_target_leakage(self,
                              X: pd.DataFrame,
                              y: pd.Series,
                              threshold: float = 0.9,
                              method: str = 'correlation') -> List[str]:
        """
        检测目标泄露
        
        检测特征中是否包含目标变量的信息（数据泄露）。
        
        参数:
            X: 特征DataFrame
            y: 目标变量
            threshold: 相关性阈值，超过此值认为可能存在泄露
            method: 检测方法
                   - 'correlation': 使用相关性（适合连续目标）
                   - 'mutual_info': 使用互信息（适合分类目标）
                   
        返回:
            可能存在泄露的特征列表
        """
        leakage_features = []
        
        try:
            if method == 'correlation':
                # 计算相关性
                for col in X.columns:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        corr = abs(X[col].corr(y))
                        if corr > threshold:
                            leakage_features.append({
                                'feature': col,
                                'correlation': corr,
                                'type': 'high_correlation'
                            })
                            logger.warning(f"特征 {col} 与目标变量的相关性为 {corr:.4f}，可能存在泄露")
            
            elif method == 'mutual_info':
                try:
                    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                    
                    # 判断是分类还是回归
                    if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                        mi_scores = mutual_info_classif(X, y, random_state=42)
                    else:
                        mi_scores = mutual_info_regression(X, y, random_state=42)
                    
                    for col, mi in zip(X.columns, mi_scores):
                        if mi > threshold:
                            leakage_features.append({
                                'feature': col,
                                'mutual_info': mi,
                                'type': 'high_mutual_info'
                            })
                            logger.warning(f"特征 {col} 与目标变量的互信息为 {mi:.4f}，可能存在泄露")
                            
                except ImportError:
                    logger.warning("scikit-learn未安装，无法使用互信息方法，改用相关性方法")
                    return self.detect_target_leakage(X, y, threshold, method='correlation')
        
        except Exception as e:
            logger.error(f"检测目标泄露时出错: {str(e)}")
        
        return leakage_features
    
    def detect_time_leakage(self,
                            df: pd.DataFrame,
                            date_col: str,
                            target_col: str,
                            check_future: bool = True) -> Dict[str, Any]:
        """
        检测时间泄露
        
        检测是否使用了未来信息（时间泄露）。
        
        参数:
            df: 包含日期和目标的数据框
            date_col: 日期列名
            target_col: 目标列名
            check_future: 是否检查未来信息
            
        返回:
            检测结果字典
        """
        result = {
            'has_leakage': False,
            'warnings': [],
            'info': {}
        }
        
        try:
            # 确保日期列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
            
            # 按日期排序
            df_sorted = df.sort_values(date_col)
            
            # 检查目标变量是否与未来日期相关
            if check_future:
                # 计算目标变量的移动平均（使用未来值）
                future_window = 7  # 检查未来7天
                df_sorted['target_future_mean'] = df_sorted[target_col].shift(-future_window).rolling(
                    window=future_window
                ).mean()
                
                # 计算相关性
                corr = df_sorted[target_col].corr(df_sorted['target_future_mean'])
                
                if abs(corr) > 0.5:
                    result['has_leakage'] = True
                    result['warnings'].append(
                        f"检测到时间泄露：目标变量与未来信息的相关性为 {corr:.4f}"
                    )
            
            # 检查特征中是否包含未来日期
            for col in df.columns:
                if col in [date_col, target_col]:
                    continue
                
                if 'future' in col.lower() or 'next' in col.lower() or 'tomorrow' in col.lower():
                    result['has_leakage'] = True
                    result['warnings'].append(f"特征 {col} 可能包含未来信息")
            
            result['info']['date_range'] = {
                'start': str(df_sorted[date_col].min()),
                'end': str(df_sorted[date_col].max())
            }
            
            if result['has_leakage']:
                logger.warning("检测到时间泄露！")
                for warning in result['warnings']:
                    logger.warning(warning)
            else:
                logger.info("未检测到时间泄露")
        
        except Exception as e:
            logger.error(f"检测时间泄露时出错: {str(e)}")
            result['warnings'].append(f"检测过程出错: {str(e)}")
        
        return result
    
    def detect_distribution_leakage(self,
                                   train_df: pd.DataFrame,
                                   test_df: pd.DataFrame,
                                   threshold: float = 0.1) -> Dict[str, Any]:
        """
        检测分布泄露
        
        检测训练集和测试集的分布是否一致，分布差异过大可能导致模型泛化能力差。
        
        参数:
            train_df: 训练集
            test_df: 测试集
            threshold: 分布差异阈值
            
        返回:
            检测结果字典
        """
        result = {
            'has_leakage': False,
            'leakage_features': [],
            'warnings': []
        }
        
        try:
            from scipy import stats
            
            for col in train_df.columns:
                if not pd.api.types.is_numeric_dtype(train_df[col]):
                    continue
                
                # 使用KS检验比较分布
                try:
                    statistic, p_value = stats.ks_2samp(
                        train_df[col].dropna(),
                        test_df[col].dropna()
                    )
                    
                    if p_value < 0.05:  # 分布显著不同
                        result['leakage_features'].append({
                            'feature': col,
                            'ks_statistic': statistic,
                            'p_value': p_value
                        })
                        
                        if statistic > threshold:
                            result['has_leakage'] = True
                            result['warnings'].append(
                                f"特征 {col} 在训练集和测试集中的分布差异较大 "
                                f"(KS统计量: {statistic:.4f}, p值: {p_value:.4f})"
                            )
                except Exception:
                    continue
            
            if result['has_leakage']:
                logger.warning("检测到分布泄露！训练集和测试集分布不一致")
            else:
                logger.info("未检测到分布泄露，训练集和测试集分布基本一致")
        
        except ImportError:
            logger.warning("scipy未安装，无法进行分布检验")
        
        return result
