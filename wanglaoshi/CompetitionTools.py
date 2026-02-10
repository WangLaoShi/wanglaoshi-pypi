"""
比赛工具模块 (CompetitionTools.py)

这个模块提供了专门用于数据科学比赛（如Kaggle）的工具，包括：
1. 提交文件生成器：自动生成符合格式的提交文件
2. 快速基线模型：一键运行多个基础模型获得baseline
3. 数据泄露检测：检测常见的数据泄露问题

使用场景：
- Kaggle等数据科学比赛
- 快速原型开发
- 模型评估和对比
"""

import os
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubmissionGenerator:
    """
    提交文件生成器
    
    用于数据科学比赛（如Kaggle）中生成符合格式要求的提交文件。
    支持单模型提交和多模型融合提交。
    
    使用示例:
        generator = SubmissionGenerator()
        # 生成单模型提交文件
        generator.create_submission(y_pred, 'sample_submission.csv', 'my_submission.csv')
        # 多模型融合
        generator.ensemble_submissions(['sub1.csv', 'sub2.csv', 'sub3.csv'], 
                                     weights=[0.4, 0.3, 0.3], 
                                     output_path='ensemble_submission.csv')
    """
    
    def __init__(self):
        """初始化提交文件生成器"""
        pass
    
    def create_submission(self, 
                         predictions: Union[np.ndarray, pd.Series, List],
                         sample_submission_path: str,
                         output_path: str = 'submission.csv',
                         id_column: str = 'id',
                         target_column: str = 'target') -> pd.DataFrame:
        """
        生成提交文件
        
        这个方法会根据样本提交文件的格式，生成符合要求的提交文件。
        
        参数:
            predictions: 预测结果
                        - numpy数组
                        - pandas Series
                        - Python列表
            sample_submission_path: 样本提交文件路径（用于获取ID列和格式）
            output_path: 输出文件路径，默认为 'submission.csv'
            id_column: ID列名，默认为 'id'
            target_column: 目标列名，默认为 'target'
            
        返回:
            生成的提交文件DataFrame
            
        示例:
            # 假设你已经有了预测结果
            y_pred = model.predict(X_test)
            
            # 生成提交文件
            submission = generator.create_submission(
                y_pred, 
                'sample_submission.csv',
                'my_submission.csv'
            )
        """
        try:
            # 读取样本提交文件，获取格式和ID
            sample_submission = pd.read_csv(sample_submission_path)
            logger.info(f"读取样本提交文件: {sample_submission_path}")
            logger.info(f"样本文件形状: {sample_submission.shape}")
            logger.info(f"样本文件列名: {sample_submission.columns.tolist()}")
            
            # 检查ID列是否存在
            if id_column not in sample_submission.columns:
                # 尝试常见的ID列名
                possible_id_cols = ['Id', 'ID', 'id', 'index', 'Index']
                id_column = None
                for col in possible_id_cols:
                    if col in sample_submission.columns:
                        id_column = col
                        logger.info(f"自动检测到ID列: {id_column}")
                        break
                
                if id_column is None:
                    raise ValueError(f"未找到ID列。可用列: {sample_submission.columns.tolist()}")
            
            # 获取ID列
            submission_ids = sample_submission[id_column].values
            
            # 将预测结果转换为数组
            if isinstance(predictions, pd.Series):
                predictions = predictions.values
            elif isinstance(predictions, list):
                predictions = np.array(predictions)
            elif isinstance(predictions, np.ndarray):
                pass
            else:
                raise TypeError(f"不支持的预测结果类型: {type(predictions)}")
            
            # 检查预测结果长度是否匹配
            if len(predictions) != len(submission_ids):
                raise ValueError(
                    f"预测结果长度 ({len(predictions)}) 与ID数量 ({len(submission_ids)}) 不匹配"
                )
            
            # 处理多列目标（多分类或多输出）
            if predictions.ndim > 1:
                # 多列预测（例如多分类的概率）
                if predictions.shape[1] > 1:
                    # 如果是多分类，可能需要转换为类别标签
                    if target_column in sample_submission.columns:
                        # 检查目标列是否期望类别标签
                        predictions = np.argmax(predictions, axis=1)
                    else:
                        # 多输出回归，需要创建多个列
                        target_columns = [f"{target_column}_{i}" for i in range(predictions.shape[1])]
                        submission_df = pd.DataFrame({
                            id_column: submission_ids
                        })
                        for i, col in enumerate(target_columns):
                            submission_df[col] = predictions[:, i]
                        submission_df.to_csv(output_path, index=False)
                        logger.info(f"提交文件已生成: {output_path}")
                        return submission_df
            
            # 创建提交文件DataFrame
            submission_df = pd.DataFrame({
                id_column: submission_ids,
                target_column: predictions
            })
            
            # 保存提交文件
            submission_df.to_csv(output_path, index=False)
            logger.info(f"提交文件已生成: {output_path}")
            logger.info(f"提交文件形状: {submission_df.shape}")
            logger.info(f"预测值统计: min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}")
            
            return submission_df
            
        except FileNotFoundError:
            logger.error(f"未找到样本提交文件: {sample_submission_path}")
            raise
        except Exception as e:
            logger.error(f"生成提交文件时出错: {str(e)}")
            raise
    
    def ensemble_submissions(self,
                            submission_files: List[str],
                            weights: Optional[List[float]] = None,
                            method: str = 'weighted_average',
                            output_path: str = 'ensemble_submission.csv',
                            sample_submission_path: Optional[str] = None) -> pd.DataFrame:
        """
        融合多个提交文件
        
        支持多种融合方法：
        - weighted_average: 加权平均（默认）
        - simple_average: 简单平均
        - rank_average: 排名平均（适合分类问题）
        - median: 中位数融合
        
        参数:
            submission_files: 要融合的提交文件路径列表
            weights: 权重列表（用于加权平均），如果为None则使用简单平均
            method: 融合方法
                   - 'weighted_average': 加权平均
                   - 'simple_average': 简单平均
                   - 'rank_average': 排名平均
                   - 'median': 中位数
            output_path: 输出文件路径
            sample_submission_path: 样本提交文件路径（用于获取格式），如果为None则使用第一个文件
            
        返回:
            融合后的提交文件DataFrame
            
        示例:
            # 融合3个提交文件，使用加权平均
            ensemble = generator.ensemble_submissions(
                ['sub1.csv', 'sub2.csv', 'sub3.csv'],
                weights=[0.4, 0.3, 0.3],
                output_path='ensemble_submission.csv'
            )
        """
        try:
            if len(submission_files) < 2:
                raise ValueError("至少需要2个提交文件才能进行融合")
            
            # 读取所有提交文件
            submissions = []
            for file_path in submission_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"未找到提交文件: {file_path}")
                df = pd.read_csv(file_path)
                submissions.append(df)
                logger.info(f"读取提交文件: {file_path}, 形状: {df.shape}")
            
            # 检查所有文件是否有相同的列
            first_cols = set(submissions[0].columns)
            for i, df in enumerate(submissions[1:], 1):
                if set(df.columns) != first_cols:
                    raise ValueError(f"提交文件 {submission_files[i]} 的列与第一个文件不匹配")
            
            # 确定ID列和目标列
            id_column = None
            target_column = None
            
            # 尝试识别ID列（通常是第一列或名为'id'/'Id'/'ID'的列）
            possible_id_cols = ['id', 'Id', 'ID', 'index', 'Index']
            for col in possible_id_cols:
                if col in submissions[0].columns:
                    id_column = col
                    break
            
            if id_column is None:
                # 使用第一列作为ID列
                id_column = submissions[0].columns[0]
            
            # 目标列是除了ID列之外的其他列
            target_columns = [col for col in submissions[0].columns if col != id_column]
            
            if len(target_columns) == 0:
                raise ValueError("未找到目标列")
            
            # 获取ID
            ids = submissions[0][id_column].values
            
            # 检查所有文件的ID是否一致
            for i, df in enumerate(submissions[1:], 1):
                if not np.array_equal(df[id_column].values, ids):
                    raise ValueError(f"提交文件 {submission_files[i]} 的ID与第一个文件不匹配")
            
            # 融合预测结果
            ensemble_df = pd.DataFrame({id_column: ids})
            
            for target_col in target_columns:
                # 收集所有文件的该列预测值
                predictions = np.array([df[target_col].values for df in submissions])
                
                if method == 'weighted_average':
                    # 加权平均
                    if weights is None:
                        weights = [1.0 / len(submissions)] * len(submissions)
                    else:
                        if len(weights) != len(submissions):
                            raise ValueError(f"权重数量 ({len(weights)}) 与提交文件数量 ({len(submissions)}) 不匹配")
                        # 归一化权重
                        weights = np.array(weights)
                        weights = weights / weights.sum()
                    
                    ensemble_pred = np.average(predictions, axis=0, weights=weights)
                    logger.info(f"使用加权平均融合 {target_col}，权重: {weights}")
                    
                elif method == 'simple_average':
                    # 简单平均
                    ensemble_pred = np.mean(predictions, axis=0)
                    logger.info(f"使用简单平均融合 {target_col}")
                    
                elif method == 'rank_average':
                    # 排名平均（适合分类问题）
                    # 将预测值转换为排名，然后平均排名，再转换回预测值
                    ranks = np.array([np.argsort(np.argsort(pred)) for pred in predictions])
                    avg_ranks = np.mean(ranks, axis=0)
                    # 将平均排名转换回预测值范围
                    min_val = np.min([np.min(pred) for pred in predictions])
                    max_val = np.max([np.max(pred) for pred in predictions])
                    ensemble_pred = min_val + (avg_ranks / len(ensemble_pred)) * (max_val - min_val)
                    logger.info(f"使用排名平均融合 {target_col}")
                    
                elif method == 'median':
                    # 中位数融合
                    ensemble_pred = np.median(predictions, axis=0)
                    logger.info(f"使用中位数融合 {target_col}")
                    
                else:
                    raise ValueError(f"不支持的融合方法: {method}")
                
                ensemble_df[target_col] = ensemble_pred
            
            # 保存融合后的提交文件
            ensemble_df.to_csv(output_path, index=False)
            logger.info(f"融合提交文件已生成: {output_path}")
            logger.info(f"融合了 {len(submissions)} 个提交文件")
            
            return ensemble_df
            
        except Exception as e:
            logger.error(f"融合提交文件时出错: {str(e)}")
            raise
    
    def validate_submission(self,
                           submission_path: str,
                           sample_submission_path: str,
                           check_values: bool = True) -> Dict[str, Any]:
        """
        验证提交文件格式是否正确
        
        参数:
            submission_path: 要验证的提交文件路径
            sample_submission_path: 样本提交文件路径
            check_values: 是否检查预测值的合理性（范围、NaN等）
            
        返回:
            验证结果字典，包含：
            - is_valid: 是否有效
            - errors: 错误列表
            - warnings: 警告列表
            - info: 信息统计
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # 读取文件
            submission = pd.read_csv(submission_path)
            sample = pd.read_csv(sample_submission_path)
            
            result['info']['submission_shape'] = submission.shape
            result['info']['sample_shape'] = sample.shape
            
            # 检查行数
            if len(submission) != len(sample):
                result['is_valid'] = False
                result['errors'].append(
                    f"行数不匹配: 提交文件有 {len(submission)} 行，样本文件有 {len(sample)} 行"
                )
            
            # 检查列名
            if set(submission.columns) != set(sample.columns):
                result['is_valid'] = False
                result['errors'].append(
                    f"列名不匹配: 提交文件列 {submission.columns.tolist()}, "
                    f"样本文件列 {sample.columns.tolist()}"
                )
            
            # 检查ID列
            id_column = sample.columns[0]  # 假设第一列是ID
            if id_column in submission.columns:
                # 检查ID是否匹配
                if not submission[id_column].equals(sample[id_column]):
                    result['is_valid'] = False
                    result['errors'].append("ID列不匹配")
            else:
                result['is_valid'] = False
                result['errors'].append(f"未找到ID列: {id_column}")
            
            # 检查预测值
            if check_values:
                target_columns = [col for col in submission.columns if col != id_column]
                for col in target_columns:
                    values = submission[col].values
                    
                    # 检查NaN
                    nan_count = pd.isna(values).sum()
                    if nan_count > 0:
                        result['is_valid'] = False
                        result['errors'].append(f"列 {col} 包含 {nan_count} 个NaN值")
                    
                    # 检查无穷大
                    inf_count = np.isinf(values).sum()
                    if inf_count > 0:
                        result['is_valid'] = False
                        result['errors'].append(f"列 {col} 包含 {inf_count} 个无穷大值")
                    
                    # 统计信息
                    result['info'][f'{col}_stats'] = {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    }
                    
                    # 警告：值范围异常
                    if values.min() < -1e10 or values.max() > 1e10:
                        result['warnings'].append(
                            f"列 {col} 的值范围异常: [{values.min()}, {values.max()}]"
                        )
            
            logger.info(f"提交文件验证完成: {'通过' if result['is_valid'] else '失败'}")
            if result['errors']:
                logger.error(f"错误: {result['errors']}")
            if result['warnings']:
                logger.warning(f"警告: {result['warnings']}")
            
            return result
            
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"验证过程出错: {str(e)}")
            logger.error(f"验证提交文件时出错: {str(e)}")
            return result


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
                            'predictions': y_pred
                        }
                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results[model_name] = {
                            'model': model,
                            'test_mse': mse,
                            'test_r2': r2,
                            'predictions': y_pred
                        }
                else:
                    # 使用交叉验证
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                    if task_type == 'regression':
                        cv_scores = -cv_scores  # neg_mean_squared_error需要取负
                    
                    results[model_name] = {
                        'model': model,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'cv_scores': cv_scores
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
        
        # 排序
        if 'Test Accuracy' in df.columns:
            df = df.sort_values('Test Accuracy', ascending=False)
        elif 'Test R2' in df.columns:
            df = df.sort_values('Test R2', ascending=False)
        elif 'CV Mean' in df.columns:
            df = df.sort_values('CV Mean', ascending=False)
        
        # 显示前N个
        print("\n" + "="*60)
        print("模型性能对比")
        print("="*60)
        print(df.head(top_n).to_string(index=False))
        print("="*60 + "\n")
        
        return df


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
            
            # 如果提供了测试集，生成最终预测
            if X_test is not None:
                # 训练所有第一层模型
                for model in models:
                    model.fit(X_train, y_train)
                
                # 生成测试集的第一层预测
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
                
                # 第二层预测
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
            # 简单平均或加权平均
            if weights is None:
                weights = [1.0 / len(models)] * len(models)
            
            logger.info("使用加权平均作为第二层")
            
            # 返回一个可以预测的函数
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
                return np.average(test_predictions, axis=1, weights=weights)
            
            return {
                'models': models,
                'weights': weights,
                'predict': predict
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
                
                # 交叉验证
                if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                    scoring = 'accuracy'
                else:
                    scoring = 'neg_mean_squared_error'
                
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                
                if scoring == 'neg_mean_squared_error':
                    return -scores.mean()
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            result = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': n_trials
            }
            
            logger.info(f"贝叶斯优化完成，最佳分数: {result['best_score']:.4f}")
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

