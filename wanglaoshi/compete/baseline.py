"""Compete: 快速基线模型（从 CompetitionTools 拆出，行为不变）。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QuickBaseline:
    """
    快速基线模型工具
    
    一键运行多个基础模型，快速获得baseline分数，适合比赛初期快速评估。
    
    使用示例:
        baseline = QuickBaseline()
        # 自动运行多个模型
        results = baseline.run_all_models(X_train, y_train, X_test, y_test, task_type='classification')
        # 显示对比结果
        baseline.compare_models(results)
    """
    
    def __init__(self):
        """初始化快速基线模型工具"""
        try:
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            from sklearn.svm import SVC, SVR
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            from sklearn.naive_bayes import GaussianNB
            from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
            
            self._sklearn_available = True
        except ImportError:
            self._sklearn_available = False
            logger.warning("scikit-learn未安装，快速基线模型功能不可用")
    
    def run_all_models(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_test: Optional[pd.DataFrame] = None,
                      y_test: Optional[pd.Series] = None,
                      task_type: str = 'auto',
                      cv: int = 5) -> Dict[str, Any]:
        """
        运行所有基础模型
        
        参数:
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征（可选）
            y_test: 测试目标（可选）
            task_type: 任务类型
                      - 'auto': 自动检测（根据目标变量类型）
                      - 'classification': 分类任务
                      - 'regression': 回归任务
            cv: 交叉验证折数（当没有测试集时使用）
            
        返回:
            包含所有模型结果的字典
        """
        if not self._sklearn_available:
            raise ImportError("需要安装scikit-learn: pip install scikit-learn")
        
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
        
        # 自动检测任务类型
        if task_type == 'auto':
            if y_train.dtype == 'object' or y_train.dtype.name == 'category':
                task_type = 'classification'
            elif len(y_train.unique()) < 20 and y_train.dtype in ['int64', 'int32']:
                # 可能是分类问题（类别数较少）
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        logger.info(f"任务类型: {task_type}")
        
        # 根据任务类型选择模型
        if task_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'SVM': SVC(random_state=42, probability=True),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Naive Bayes': GaussianNB()
            }
            scoring = 'accuracy'
        else:  # regression
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'SVM': SVR(),
                'KNN': KNeighborsRegressor(n_neighbors=5)
            }
            scoring = 'neg_mean_squared_error'
        
        results = {}
        
        for model_name, model in models.items():
            try:
                logger.info(f"训练模型: {model_name}")
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 评估
                if X_test is not None and y_test is not None:
                    # 使用测试集评估
                    y_pred = model.predict(X_test)
                    
                    if task_type == 'classification':
                        score = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        results[model_name] = {
                            'model': model,
                            'test_accuracy': score,
                            'test_f1': f1,
                            'predictions': y_pred,
                            'metric': 'accuracy',
                            'higher_is_better': True,
                            'task_type': task_type,
                        }
                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results[model_name] = {
                            'model': model,
                            'test_mse': mse,
                            'test_r2': r2,
                            'predictions': y_pred,
                            'metric': 'mse',
                            'higher_is_better': False,
                            'task_type': task_type,
                        }
                else:
                    # 使用交叉验证
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                    if task_type == 'regression':
                        # 展示用正 MSE；排序方向由 higher_is_better=False 决定
                        cv_scores = -cv_scores

                    results[model_name] = {
                        'model': model,
                        'cv_mean': float(cv_scores.mean()),
                        'cv_std': float(cv_scores.std()),
                        'cv_scores': cv_scores,
                        'metric': 'mse' if task_type == 'regression' else 'accuracy',
                        'higher_is_better': task_type != 'regression',
                        'task_type': task_type,
                    }
                
                logger.info(f"{model_name} 完成")
                
            except Exception as e:
                logger.warning(f"模型 {model_name} 训练失败: {str(e)}")
                continue
        
        return results
    
    def compare_models(self, results: Dict[str, Any], top_n: int = 5) -> pd.DataFrame:
        """
        对比模型性能
        
        参数:
            results: run_all_models返回的结果字典
            top_n: 显示前N个模型
            
        返回:
            模型对比DataFrame
        """
        comparison = []
        
        for model_name, result in results.items():
            row = {'Model': model_name}
            
            if 'test_accuracy' in result:
                row['Test Accuracy'] = result['test_accuracy']
                row['Test F1'] = result['test_f1']
            elif 'test_mse' in result:
                row['Test MSE'] = result['test_mse']
                row['Test R2'] = result['test_r2']
            elif 'cv_mean' in result:
                row['CV Mean'] = result['cv_mean']
                row['CV Std'] = result['cv_std']
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)

        # 按结果元数据决定排序方向（回归 MSE 升序，分类准确率降序）
        higher_flags = [
            bool(result.get("higher_is_better", True))
            for result in results.values()
            if any(k in result for k in ("test_accuracy", "test_mse", "cv_mean"))
        ]
        higher_is_better = True if not higher_flags else all(higher_flags)
        metrics = sorted(
            {
                str(result.get("metric"))
                for result in results.values()
                if result.get("metric") is not None
            }
        )

        if "Test Accuracy" in df.columns:
            sort_col, ascending = "Test Accuracy", False
        elif "Test MSE" in df.columns and not higher_is_better:
            # 回归持有测试集时优先按 MSE 升序（更差的误差不应排第一）
            sort_col, ascending = "Test MSE", True
        elif "Test R2" in df.columns:
            sort_col, ascending = "Test R2", False
        elif "CV Mean" in df.columns:
            sort_col, ascending = "CV Mean", (not higher_is_better)
        else:
            sort_col, ascending = None, False

        if sort_col is not None:
            df = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

        df.attrs["sort_column"] = sort_col
        df.attrs["sort_ascending"] = ascending
        df.attrs["higher_is_better"] = higher_is_better
        df.attrs["metrics"] = metrics

        print("\n" + "=" * 60)
        print("模型性能对比")
        if sort_col is not None:
            direction = "升序(越小越好)" if ascending else "降序(越大越好)"
            print(f"排序: {sort_col} {direction}; metrics={metrics}")
        print("=" * 60)
        print(df.head(top_n).to_string(index=False))
        print("=" * 60 + "\n")

        return df
