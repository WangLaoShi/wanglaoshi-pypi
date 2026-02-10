"""
增强特征工程模块 (FeatureEngineering.py)

这个模块提供了丰富的特征工程功能，专门为数据科学比赛和学习设计。
包括：
1. 时间特征提取：从日期时间列提取各种时间特征
2. 文本特征提取：TF-IDF、词向量、文本统计特征
3. 类别特征编码：Target Encoding、Frequency Encoding、One-Hot Encoding
4. 数值特征变换：对数变换、Box-Cox变换、分箱
5. 特征选择：基于重要性、相关性、互信息的特征选择

使用场景：
- 数据科学比赛中的特征工程
- 学习特征工程最佳实践
- 快速构建高质量特征
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from scipy import stats
from scipy.special import boxcox

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    增强特征工程类
    
    提供丰富的特征工程方法，适合数据科学比赛和学习使用。
    """
    
    def __init__(self):
        """初始化特征工程器"""
        self._target_encoders = {}  # 存储目标编码器
        self._fitted_transformers = {}  # 存储已拟合的转换器
    
    # ==================== 时间特征提取 ====================
    
    def extract_datetime_features(self,
                                  df: pd.DataFrame,
                                  datetime_col: str,
                                  features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        从日期时间列提取时间特征
        
        提取的特征包括：
        - 年、月、日、小时、分钟、秒
        - 星期几、一年中的第几天、一年中的第几周
        - 季度、是否周末、是否月初/月末
        - 时间差特征（距离某个参考时间）
        
        参数:
            df: 输入数据框
            datetime_col: 日期时间列名
            features: 要提取的特征列表，如果为None则提取所有特征
                    可选: 'year', 'month', 'day', 'hour', 'minute', 'second',
                          'weekday', 'dayofyear', 'week', 'quarter',
                          'is_weekend', 'is_month_start', 'is_month_end'
            
        返回:
            添加了时间特征的数据框
            
        示例:
            df['date'] = pd.to_datetime(df['date'])
            engineer = AdvancedFeatureEngineer()
            df = engineer.extract_datetime_features(df, 'date')
        """
        df_result = df.copy()
        
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df_result[datetime_col]):
            df_result[datetime_col] = pd.to_datetime(df_result[datetime_col])
        
        # 默认提取所有特征
        if features is None:
            features = ['year', 'month', 'day', 'hour', 'minute', 'weekday', 
                       'dayofyear', 'week', 'quarter', 'is_weekend']
        
        date_series = df_result[datetime_col]
        
        # 提取各种时间特征
        if 'year' in features:
            df_result[f'{datetime_col}_year'] = date_series.dt.year
        if 'month' in features:
            df_result[f'{datetime_col}_month'] = date_series.dt.month
        if 'day' in features:
            df_result[f'{datetime_col}_day'] = date_series.dt.day
        if 'hour' in features:
            df_result[f'{datetime_col}_hour'] = date_series.dt.hour
        if 'minute' in features:
            df_result[f'{datetime_col}_minute'] = date_series.dt.minute
        if 'second' in features:
            df_result[f'{datetime_col}_second'] = date_series.dt.second
        if 'weekday' in features:
            df_result[f'{datetime_col}_weekday'] = date_series.dt.dayofweek  # 0=Monday
        if 'dayofyear' in features:
            df_result[f'{datetime_col}_dayofyear'] = date_series.dt.dayofyear
        if 'week' in features:
            df_result[f'{datetime_col}_week'] = date_series.dt.isocalendar().week
        if 'quarter' in features:
            df_result[f'{datetime_col}_quarter'] = date_series.dt.quarter
        if 'is_weekend' in features:
            df_result[f'{datetime_col}_is_weekend'] = (date_series.dt.dayofweek >= 5).astype(int)
        if 'is_month_start' in features:
            df_result[f'{datetime_col}_is_month_start'] = date_series.dt.is_month_start.astype(int)
        if 'is_month_end' in features:
            df_result[f'{datetime_col}_is_month_end'] = date_series.dt.is_month_end.astype(int)
        
        logger.info(f"从 {datetime_col} 提取了 {len([f for f in features if f in ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekday', 'dayofyear', 'week', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end']])} 个时间特征")
        
        return df_result
    
    # ==================== 文本特征提取 ====================
    
    def extract_text_features(self,
                              df: pd.DataFrame,
                              text_col: str,
                              method: str = 'basic',
                              max_features: int = 100) -> pd.DataFrame:
        """
        从文本列提取特征
        
        支持的方法：
        - 'basic': 基础统计特征（长度、词数、字符数等）
        - 'tfidf': TF-IDF特征（需要sklearn）
        - 'count': 词频特征（需要sklearn）
        
        参数:
            df: 输入数据框
            text_col: 文本列名
            method: 提取方法
            max_features: TF-IDF/Count的最大特征数（仅用于tfidf和count方法）
            
        返回:
            添加了文本特征的数据框
            
        示例:
            engineer = AdvancedFeatureEngineer()
            df = engineer.extract_text_features(df, 'text_col', method='basic')
        """
        df_result = df.copy()
        
        if method == 'basic':
            # 基础统计特征
            text_series = df_result[text_col].astype(str)
            
            # 文本长度
            df_result[f'{text_col}_length'] = text_series.str.len()
            
            # 词数（按空格分割）
            df_result[f'{text_col}_word_count'] = text_series.str.split().str.len()
            
            # 字符数（不含空格）
            df_result[f'{text_col}_char_count'] = text_series.str.replace(' ', '').str.len()
            
            # 大写字母数
            df_result[f'{text_col}_upper_count'] = text_series.str.findall(r'[A-Z]').str.len()
            
            # 数字数
            df_result[f'{text_col}_digit_count'] = text_series.str.findall(r'\d').str.len()
            
            # 特殊字符数
            df_result[f'{text_col}_special_count'] = text_series.str.findall(r'[^A-Za-z0-9\s]').str.len()
            
            logger.info(f"从 {text_col} 提取了基础文本特征")
            
        elif method in ['tfidf', 'count']:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
                
                text_series = df_result[text_col].astype(str).fillna('')
                
                if method == 'tfidf':
                    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
                    features = vectorizer.fit_transform(text_series)
                    feature_names = [f'{text_col}_tfidf_{i}' for i in range(features.shape[1])]
                else:  # count
                    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
                    features = vectorizer.fit_transform(text_series)
                    feature_names = [f'{text_col}_count_{i}' for i in range(features.shape[1])]
                
                # 转换为DataFrame
                text_features_df = pd.DataFrame(
                    features.toarray(),
                    columns=feature_names,
                    index=df_result.index
                )
                
                # 合并到原数据框
                df_result = pd.concat([df_result, text_features_df], axis=1)
                
                logger.info(f"从 {text_col} 提取了 {len(feature_names)} 个 {method.upper()} 特征")
                
            except ImportError:
                logger.warning("sklearn未安装，无法使用TF-IDF/Count方法，改用基础方法")
                return self.extract_text_features(df, text_col, method='basic')
        
        else:
            raise ValueError(f"不支持的文本特征提取方法: {method}")
        
        return df_result
    
    # ==================== 类别特征编码 ====================
    
    def target_encode(self,
                     df: pd.DataFrame,
                     categorical_col: str,
                     target_col: str,
                     smoothing: float = 1.0,
                     min_samples_leaf: int = 1) -> pd.DataFrame:
        """
        目标编码（Target Encoding）
        
        目标编码是一种强大的类别编码方法，用目标变量的统计量（通常是均值）来编码类别。
        为了避免过拟合，通常需要平滑处理。
        
        参数:
            df: 输入数据框
            categorical_col: 要编码的类别列名
            target_col: 目标列名
            smoothing: 平滑参数，越大越平滑（避免过拟合）
            min_samples_leaf: 最小样本数，类别样本数少于此值时不使用该类别
            
        返回:
            添加了目标编码特征的数据框
            
        示例:
            engineer = AdvancedFeatureEngineer()
            df = engineer.target_encode(df, 'category_col', 'target', smoothing=1.0)
        """
        df_result = df.copy()
        
        # 计算全局均值
        global_mean = df_result[target_col].mean()
        
        # 计算每个类别的目标均值
        category_stats = df_result.groupby(categorical_col)[target_col].agg(['mean', 'count'])
        
        # 平滑处理：weighted average of category mean and global mean
        # formula: (n * category_mean + smoothing * global_mean) / (n + smoothing)
        category_stats['target_encoded'] = (
            category_stats['count'] * category_stats['mean'] + 
            smoothing * global_mean
        ) / (category_stats['count'] + smoothing)
        
        # 对于样本数太少的类别，使用全局均值
        category_stats.loc[category_stats['count'] < min_samples_leaf, 'target_encoded'] = global_mean
        
        # 应用编码
        df_result[f'{categorical_col}_target_encoded'] = df_result[categorical_col].map(
            category_stats['target_encoded']
        )
        
        # 填充缺失值（新类别）
        df_result[f'{categorical_col}_target_encoded'].fillna(global_mean, inplace=True)
        
        logger.info(f"对 {categorical_col} 进行了目标编码")
        
        return df_result
    
    def frequency_encode(self,
                        df: pd.DataFrame,
                        categorical_col: str) -> pd.DataFrame:
        """
        频率编码（Frequency Encoding）
        
        用类别出现的频率来编码类别特征。
        
        参数:
            df: 输入数据框
            categorical_col: 要编码的类别列名
            
        返回:
            添加了频率编码特征的数据框
        """
        df_result = df.copy()
        
        # 计算每个类别的频率
        frequency_map = df_result[categorical_col].value_counts(normalize=True).to_dict()
        
        # 应用编码
        df_result[f'{categorical_col}_frequency'] = df_result[categorical_col].map(frequency_map)
        
        # 填充缺失值（新类别）
        df_result[f'{categorical_col}_frequency'].fillna(0, inplace=True)
        
        logger.info(f"对 {categorical_col} 进行了频率编码")
        
        return df_result
    
    def one_hot_encode(self,
                      df: pd.DataFrame,
                      categorical_cols: List[str],
                      drop_first: bool = False) -> pd.DataFrame:
        """
        One-Hot编码（独热编码）
        
        将类别变量转换为二进制特征。
        
        参数:
            df: 输入数据框
            categorical_cols: 要编码的类别列名列表
            drop_first: 是否删除第一个类别（避免多重共线性）
            
        返回:
            添加了One-Hot编码特征的数据框
        """
        df_result = df.copy()
        
        for col in categorical_cols:
            # 使用pandas的get_dummies
            dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=drop_first)
            df_result = pd.concat([df_result, dummies], axis=1)
            logger.info(f"对 {col} 进行了One-Hot编码，生成了 {len(dummies.columns)} 个特征")
        
        return df_result
    
    # ==================== 数值特征变换 ====================
    
    def log_transform(self,
                     df: pd.DataFrame,
                     numeric_cols: List[str],
                     add_one: bool = True) -> pd.DataFrame:
        """
        对数变换
        
        对数变换可以处理右偏分布，使数据更接近正态分布。
        对于包含0或负数的数据，可以先加1再取对数。
        
        参数:
            df: 输入数据框
            numeric_cols: 要变换的数值列名列表
            add_one: 是否先加1（处理0值）
            
        返回:
            添加了对数变换特征的数据框
        """
        df_result = df.copy()
        
        for col in numeric_cols:
            if add_one:
                # 先加1，避免log(0)
                df_result[f'{col}_log'] = np.log1p(df_result[col])
            else:
                # 只对正值取对数
                df_result[f'{col}_log'] = np.log(df_result[col].clip(lower=1e-10))
            
            logger.info(f"对 {col} 进行了对数变换")
        
        return df_result
    
    def boxcox_transform(self,
                        df: pd.DataFrame,
                        numeric_cols: List[str]) -> pd.DataFrame:
        """
        Box-Cox变换
        
        Box-Cox变换是一种幂变换，可以稳定方差并使数据更接近正态分布。
        只适用于正值数据。
        
        参数:
            df: 输入数据框
            numeric_cols: 要变换的数值列名列表
            
        返回:
            添加了Box-Cox变换特征的数据框
        """
        df_result = df.copy()
        
        for col in numeric_cols:
            # 只处理正值
            positive_data = df_result[col].clip(lower=1e-10)
            
            try:
                # 计算最优lambda
                transformed_data, lambda_param = boxcox(positive_data)
                df_result[f'{col}_boxcox'] = transformed_data
                logger.info(f"对 {col} 进行了Box-Cox变换，lambda={lambda_param:.4f}")
            except Exception as e:
                logger.warning(f"对 {col} 进行Box-Cox变换失败: {str(e)}，跳过该列")
        
        return df_result
    
    def bin_features(self,
                    df: pd.DataFrame,
                    numeric_cols: List[str],
                    n_bins: int = 5,
                    strategy: str = 'quantile') -> pd.DataFrame:
        """
        特征分箱（Binning）
        
        将连续数值特征转换为离散的类别特征。
        
        参数:
            df: 输入数据框
            numeric_cols: 要分箱的数值列名列表
            n_bins: 分箱数量
            strategy: 分箱策略
                     - 'quantile': 等频分箱（每个箱的样本数相同）
                     - 'uniform': 等距分箱（每个箱的宽度相同）
                     
        返回:
            添加了分箱特征的数据框
        """
        df_result = df.copy()
        
        try:
            from sklearn.preprocessing import KBinsDiscretizer
            
            for col in numeric_cols:
                # 使用KBinsDiscretizer
                encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                binned = encoder.fit_transform(df_result[[col]])
                df_result[f'{col}_binned'] = binned.ravel()
                logger.info(f"对 {col} 进行了分箱（{strategy}，{n_bins}个箱）")
        
        except ImportError:
            logger.warning("sklearn未安装，使用pandas的cut/qcut进行分箱")
            for col in numeric_cols:
                if strategy == 'quantile':
                    df_result[f'{col}_binned'] = pd.qcut(
                        df_result[col], 
                        q=n_bins, 
                        labels=False, 
                        duplicates='drop'
                    )
                else:  # uniform
                    df_result[f'{col}_binned'] = pd.cut(
                        df_result[col], 
                        bins=n_bins, 
                        labels=False, 
                        duplicates='drop'
                    )
                logger.info(f"对 {col} 进行了分箱（{strategy}，{n_bins}个箱）")
        
        return df_result
    
    # ==================== 特征选择 ====================
    
    def select_features_by_importance(self,
                                      X: pd.DataFrame,
                                      y: pd.Series,
                                      n_features: int = 10,
                                      model_type: str = 'random_forest') -> List[str]:
        """
        基于重要性的特征选择
        
        使用树模型（随机森林、XGBoost等）的特征重要性来选择特征。
        
        参数:
            X: 特征数据框
            y: 目标变量
            n_features: 要选择的特征数量
            model_type: 模型类型
                       - 'random_forest': 随机森林
                       - 'xgboost': XGBoost（如果可用）
                       - 'lightgbm': LightGBM（如果可用）
                       
        返回:
            选中的特征列表
        """
        try:
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                # 判断是分类还是回归
                if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                
                model.fit(X, y)
                importances = model.feature_importances_
                
            elif model_type == 'xgboost':
                try:
                    import xgboost as xgb
                    
                    if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                        model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
                    else:
                        model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
                    
                    model.fit(X, y)
                    importances = model.feature_importances_
                except ImportError:
                    logger.warning("XGBoost未安装，改用随机森林")
                    return self.select_features_by_importance(X, y, n_features, 'random_forest')
            
            elif model_type == 'lightgbm':
                try:
                    import lightgbm as lgb
                    
                    if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                        model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
                    else:
                        model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
                    
                    model.fit(X, y)
                    importances = model.feature_importances_
                except ImportError:
                    logger.warning("LightGBM未安装，改用随机森林")
                    return self.select_features_by_importance(X, y, n_features, 'random_forest')
            
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 获取重要性排序
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            selected_features = feature_importance.head(n_features)['feature'].tolist()
            
            logger.info(f"基于 {model_type} 选择了 {len(selected_features)} 个重要特征")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"特征选择时出错: {str(e)}")
            return X.columns.tolist()[:n_features]
    
    def select_features_by_correlation(self,
                                      X: pd.DataFrame,
                                      y: pd.Series,
                                      threshold: float = 0.1) -> List[str]:
        """
        基于相关性的特征选择
        
        选择与目标变量相关性较高的特征。
        
        参数:
            X: 特征数据框
            y: 目标变量
            threshold: 相关性阈值（绝对值）
            
        返回:
            选中的特征列表
        """
        correlations = {}
        
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                corr = abs(X[col].corr(y))
                if corr > threshold:
                    correlations[col] = corr
        
        selected_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in selected_features]
        
        logger.info(f"基于相关性选择了 {len(selected_features)} 个特征（阈值={threshold}）")
        
        return selected_features
    
    def select_features_by_mutual_info(self,
                                      X: pd.DataFrame,
                                      y: pd.Series,
                                      n_features: int = 10) -> List[str]:
        """
        基于互信息的特征选择
        
        使用互信息来选择与目标变量最相关的特征。
        
        参数:
            X: 特征数据框
            y: 目标变量
            n_features: 要选择的特征数量
            
        返回:
            选中的特征列表
        """
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest
            
            # 判断是分类还是回归
            if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20:
                selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            else:
                selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            logger.info(f"基于互信息选择了 {len(selected_features)} 个特征")
            
            return selected_features
            
        except ImportError:
            logger.warning("sklearn未安装，无法使用互信息特征选择")
            return X.columns.tolist()[:n_features]

