import os  # 确保os模块在文件开头导入
import base64
import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
from scipy.stats import skew, kurtosis, zscore, normaltest, shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(current_dir, 'SimHei.ttf')

# 添加字体文件到 matplotlib 的字体管理器
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['SimHei'] + rcParams['font.sans-serif']
    rcParams['axes.unicode_minus'] = False
    logger.info(f"已加载字体文件: {font_path}")
else:
    logger.warning(f"警告: 未找到字体文件: {font_path}")
    # 使用系统默认字体
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
    rcParams['axes.unicode_minus'] = False

class DataProcessor:
    """数据处理工具类"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> None:
        """验证DataFrame的有效性"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame不能为空")
    
    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """获取列类型信息"""
        return {
            'numeric_cols': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_cols': df.select_dtypes(include=['datetime']).columns.tolist()
        }
    
    @staticmethod
    def process_value(v: Any) -> Any:
        """处理单个值，转换为可序列化格式"""
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.int64, np.int32)):
            return int(v)
        if isinstance(v, (np.float64, np.float32)):
            if np.isnan(v) or np.isinf(v):
                return None
            return float(v)
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, pd.Series):
            return v.to_dict()
        if isinstance(v, pd.DataFrame):
            return v.to_dict('records')
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return v
    
    @staticmethod
    def process_data(obj: Any) -> Any:
        """递归处理数据，转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: DataProcessor.process_data(v) for k, v in obj.items()}
        elif isinstance(obj, (list, np.ndarray)):
            return [DataProcessor.process_data(item) for item in obj]
        else:
            return DataProcessor.process_value(obj)
    
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
        """将DataFrame分块处理"""
        return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

class DataAnalyzer:
    """数据分析器主类"""
    
    def __init__(self, df: pd.DataFrame):
        # 验证输入数据
        DataProcessor.validate_dataframe(df)
        self.df = df.copy()
        
        # 获取列类型信息
        column_types = DataProcessor.get_column_types(self.df)
        self.numeric_cols = column_types['numeric_cols']
        self.categorical_cols = column_types['categorical_cols']
        self.datetime_cols = column_types['datetime_cols']
        
        # 初始化可视化工具
        self.visualizer = Visualizer(font_prop)
        
        # 初始化缓存
        self._correlation_cache = None
        self._multicollinearity_cache = None
        self._memory_usage = None
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'visualizer'):
            self.visualizer.clear_cache()
    
    def correlation_analysis(self) -> pd.DataFrame:
        """相关性分析"""
        if self._correlation_cache is None:
            self._correlation_cache = self.df[self.numeric_cols].corr()
        return self._correlation_cache
    
    def multicollinearity_analysis(self) -> pd.DataFrame:
        """多重共线性分析"""
        if self._multicollinearity_cache is None:
            vif_data = pd.DataFrame()
            vif_data["列名"] = self.numeric_cols
            vif_data["VIF"] = [variance_inflation_factor(self.df[self.numeric_cols].values, i) 
                              for i in range(len(self.numeric_cols))]
            self._multicollinearity_cache = vif_data
        return self._multicollinearity_cache
    
    def plot_distribution(self, column: str) -> str:
        """绘制分布图"""
        if column not in self.df.columns:
            raise ValueError(f"列名不存在: {column}")
        return self.visualizer.plot_distribution(self.df[column], column)
    
    def plot_correlation_heatmap(self) -> str:
        """绘制相关性热图"""
        return self.visualizer.plot_correlation_heatmap(self.correlation_analysis())
    
    def generate_report(self, output_html: str = "analysis_report.html") -> None:
        """生成分析报告"""
        try:
            # 获取数据集名称
            try:
                base_name = os.path.basename(output_html)
                name_without_ext = os.path.splitext(base_name)[0]
                suffixes_to_remove = ['_analysis', '_report', '_data']
                dataset_name = name_without_ext
                for suffix in suffixes_to_remove:
                    if dataset_name.endswith(suffix):
                        dataset_name = dataset_name[:-len(suffix)]
                dataset_name = dataset_name.replace('_', ' ').title()
            except Exception as e:
                logger.error(f"处理数据集名称时出错: {str(e)}")
                dataset_name = "未命名数据集"
            
            # 收集所有分析结果
            logger.info("收集分析结果...")
            analysis_results = {
                "dataset_name": dataset_name,
                "basic_stats": self.basic_statistics().to_dict('records'),
                "normality_test": self.normality_test().to_dict('records'),
                "missing_analysis": self.missing_value_analysis().to_dict('records'),
                "outlier_analysis": self.outlier_analysis().to_dict('records'),
                "duplicate_analysis": self.duplicate_analysis(),
                "correlation_matrix": self.correlation_analysis().to_dict(),
                "multicollinearity": self.multicollinearity_analysis().to_dict('records'),
                "pca_analysis": self.pca_analysis(),
                "plots": {
                    "distribution": {col: self.plot_distribution(col) for col in self.df.columns},
                    "correlation": self.plot_correlation_heatmap()
                },
                "interpretations": self.interpret_results()
            }
            
            # 获取包内模板目录的绝对路径
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
            static_dir = os.path.join(os.path.dirname(__file__), 'static')
            
            # 创建报告生成器并生成报告
            report_generator = ReportGenerator(template_dir, static_dir)
            report_generator.generate_report(analysis_results, output_html)
            
        except Exception as e:
            logger.error(f"生成报告时出错: {str(e)}")
            raise

    # ==================== 基础统计分析 ====================
    def basic_statistics(self) -> pd.DataFrame:
        """计算基本统计量"""
        # 获取基本统计信息
        stats_df = self.df.describe(include='all').transpose()
        
        # 添加缺失值相关信息
        stats_df['缺失值数量'] = self.df.isnull().sum()
        stats_df['缺失率 (%)'] = (self.df.isnull().mean() * 100).round(2)
        stats_df['唯一值数量'] = self.df.nunique()
        
        # 处理空值，将其转换为更易读的格式
        stats_df = stats_df.apply(lambda x: x.apply(lambda v: '空值' if pd.isna(v) else v))
        
        # 对数值列的统计量进行格式化
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        for col in stats_df.index:
            if col in numeric_cols:
                for stat in ['mean', 'std', '25%', '50%', '75%', 'min', 'max']:
                    if stat in stats_df.columns and not isinstance(stats_df.at[col, stat], str):
                        stats_df.at[col, stat] = f"{stats_df.at[col, stat]:.2f}"
        
        return stats_df.reset_index().rename(columns={'index': '列名'})

    def normality_test(self) -> pd.DataFrame:
        """正态性检验"""
        results = []
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 3:  # 至少需要3个样本才能进行正态性检验
                # Shapiro-Wilk检验
                shapiro_stat, shapiro_p = shapiro(data)
                # D'Agostino-Pearson检验
                norm_stat, norm_p = normaltest(data)
                
                results.append({
                    '列名': col,
                    'Shapiro-Wilk统计量': shapiro_stat,
                    'Shapiro-Wilk p值': shapiro_p,
                    'D-Agostino-Pearson统计量': norm_stat,
                    'D-Agostino-Pearson p值': norm_p,
                    '是否正态分布': '是' if shapiro_p > 0.05 and norm_p > 0.05 else '否'
                })
        return pd.DataFrame(results)

    # ==================== 数据质量分析 ====================
    def missing_value_analysis(self) -> pd.DataFrame:
        """缺失值分析"""
        missing_counts = self.df.isnull().sum()
        missing_ratios = self.df.isnull().mean() * 100
        suggestions = []
        
        for col, ratio in missing_ratios.items():
            if ratio == 0:
                suggestion = "无缺失，无需处理"
            elif ratio < 5:
                suggestion = "填充缺失值，例如均值/众数"
            elif ratio < 50:
                suggestion = "视情况填充或丢弃列"
            else:
                suggestion = "考虑丢弃列"
            suggestions.append(suggestion)
        
        return pd.DataFrame({
            "列名": self.df.columns,
            "缺失值数量": missing_counts,
            "缺失率 (%)": missing_ratios,
            "建议处理方案": suggestions
        })

    def outlier_analysis(self) -> pd.DataFrame:
        """异常值分析"""
        results = []
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            
            # Z-score方法
            z_scores = zscore(data)
            outliers_zscore = np.sum(np.abs(z_scores) > 3)
            zscore_explanation = self._get_zscore_interpretation(z_scores)
            
            # IQR方法
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr = np.sum((data < lower_bound) | (data > upper_bound))
            iqr_explanation = self._get_iqr_interpretation(data, Q1, Q3, IQR)
            
            # 箱线图方法
            outliers_box = np.sum((data < Q1 - 3 * IQR) | (data > Q3 + 3 * IQR))
            
            results.append({
                "列名": col,
                "Z-score异常值数量": outliers_zscore,
                "Z-score解释": zscore_explanation,
                "IQR异常值数量": outliers_iqr,
                "IQR解释": iqr_explanation,
                "箱线图异常值数量": outliers_box,
                "异常值比例 (%)": (max(outliers_zscore, outliers_iqr, outliers_box) / len(data) * 100).round(2)
            })
        
        return pd.DataFrame(results)

    def _get_zscore_interpretation(self, z_scores: np.ndarray) -> str:
        """生成Z-score方法的解释"""
        total_points = len(z_scores)
        extreme_outliers = np.sum(np.abs(z_scores) > 3)  # 极端异常值
        moderate_outliers = np.sum((np.abs(z_scores) > 2) & (np.abs(z_scores) <= 3))  # 中度异常值
        
        interpretation = []
        
        # 总体情况
        if extreme_outliers == 0 and moderate_outliers == 0:
            interpretation.append("数据分布较为集中，未检测到异常值。")
        else:
            interpretation.append(f"共检测到 {extreme_outliers + moderate_outliers} 个异常值，占总数据的 {((extreme_outliers + moderate_outliers) / total_points * 100):.2f}%。")
        
        # 详细解释
        if extreme_outliers > 0:
            interpretation.append(f"其中 {extreme_outliers} 个为极端异常值（|Z-score| > 3），占总数据的 {extreme_outliers / total_points * 100:.2f}%。")
        if moderate_outliers > 0:
            interpretation.append(f"另有 {moderate_outliers} 个为中度异常值（2 < |Z-score| ≤ 3），占总数据的 {moderate_outliers / total_points * 100:.2f}%。")
        
        # 建议
        if extreme_outliers > 0:
            interpretation.append("建议检查这些极端异常值的合理性，必要时进行处理。")
        elif moderate_outliers > 0:
            interpretation.append("建议关注这些中度异常值，确认其是否合理。")
        
        return " ".join(interpretation)

    def _get_iqr_interpretation(self, data: pd.Series, Q1: float, Q3: float, IQR: float) -> str:
        """生成IQR方法的解释"""
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mild_outliers = np.sum((data < lower_bound) | (data > upper_bound))
        
        # 计算四分位距的分布
        q1_percentile = np.percentile(data, 25)
        q3_percentile = np.percentile(data, 75)
        median = np.median(data)
        
        interpretation = []
        
        # 总体情况
        if mild_outliers == 0:
            interpretation.append("数据分布较为均匀，未检测到异常值。")
        else:
            interpretation.append(f"检测到 {mild_outliers} 个异常值，占总数据的 {mild_outliers / len(data) * 100:.2f}%。")
        
        # 数据分布特征
        interpretation.append(f"数据的中位数为 {median:.2f}，")
        if Q3 - Q1 < np.std(data):
            interpretation.append("四分位距较小，说明数据较为集中；")
        else:
            interpretation.append("四分位距较大，说明数据较为分散；")
        
        # 异常值范围
        interpretation.append(f"异常值范围为：小于 {lower_bound:.2f} 或大于 {upper_bound:.2f}。")
        
        # 建议
        if mild_outliers > 0:
            if mild_outliers / len(data) > 0.1:  # 异常值比例超过10%
                interpretation.append("异常值比例较高，建议进行详细检查并考虑是否需要处理。")
            else:
                interpretation.append("异常值比例较低，建议检查这些值的合理性。")
        
        return " ".join(interpretation)

    def duplicate_analysis(self) -> Dict[str, Any]:
        """重复值分析"""
        duplicate_rows = self.df.duplicated().sum()
        duplicate_ratio = (duplicate_rows / len(self.df) * 100).round(2)
        
        return {
            "重复行数量": duplicate_rows,
            "重复率 (%)": duplicate_ratio,
            "建议": "建议删除重复行" if duplicate_ratio > 5 else "重复率较低，可保留"
        }

    # ==================== 高级统计分析 ====================
    def pca_analysis(self) -> Dict[str, Any]:
        """主成分分析"""
        if len(self.numeric_cols) < 2:
            return {"error": "需要至少两个数值型变量进行PCA分析"}
            
        # 只对数值型变量进行PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[self.numeric_cols])
        
        # 执行PCA
        pca = PCA()
        pca.fit(scaled_data)
        
        # 计算累计方差贡献率
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        return {
            "主成分数量": len(self.numeric_cols),
            "各主成分方差贡献率": pca.explained_variance_ratio_,
            "累计方差贡献率": cumulative_variance,
            "建议保留主成分数量": np.argmax(cumulative_variance >= 0.95) + 1
        }

    # ==================== 结果解读 ====================
    def interpret_results(self) -> Dict[str, Any]:
        """解读所有分析结果"""
        interpretations = {
            "basic_info": self._interpret_basic_info(),
            "data_quality": self._interpret_data_quality(),
            "statistical_analysis": self._interpret_statistical_analysis(),
            "recommendations": self._generate_recommendations()
        }
        return interpretations

    def _interpret_basic_info(self) -> Dict[str, Any]:
        """解读基本信息"""
        total_rows, total_cols = self.df.shape
        numeric_count = len(self.numeric_cols)
        categorical_count = len(self.categorical_cols)
        datetime_count = len(self.datetime_cols)

        # 获取各类型变量的详细信息
        numeric_vars = []
        for col in self.numeric_cols:
            sample = self.df[col].iloc[0]
            if pd.notna(sample):  # 确保样本值不是NaN
                numeric_vars.append({
                    "name": col,
                    "sample": f"{sample:.4f}" if isinstance(sample, (int, float)) else str(sample)
                })

        categorical_vars = []
        for col in self.categorical_cols:
            sample = self.df[col].iloc[0]
            if pd.notna(sample):  # 确保样本值不是NaN
                categorical_vars.append({
                    "name": col,
                    "sample": str(sample)
                })

        datetime_vars = []
        for col in self.datetime_cols:
            sample = self.df[col].iloc[0]
            if pd.notna(sample):  # 确保样本值不是NaN
                datetime_vars.append({
                    "name": col,
                    "sample": str(sample)
                })

        print("变量详情：")
        print(f"数值型变量: {numeric_vars}")
        print(f"分类型变量: {categorical_vars}")
        print(f"时间型变量: {datetime_vars}")

        return {
            "数据集规模": f"数据集包含 {total_rows} 行和 {total_cols} 列。",
            "变量类型": f"其中包含 {numeric_count} 个数值型变量、{categorical_count} 个分类型变量和 {datetime_count} 个时间型变量。",
            "说明": "这个规模的数据集适合进行统计分析，但需要注意数据质量和变量之间的关系。",
            "变量详情": {
                "数值型变量": numeric_vars,
                "分类型变量": categorical_vars,
                "时间型变量": datetime_vars
            }
        }

    def _interpret_data_quality(self) -> Dict[str, Any]:
        """解读数据质量分析结果"""
        # 缺失值分析
        missing_stats = self.missing_value_analysis()
        missing_interpretation = {
            "总体情况": f"数据集中共有 {len(missing_stats)} 个变量，其中 {sum(missing_stats['缺失值数量'] > 0)} 个变量存在缺失值。",
            "缺失值分布": "缺失值分布情况如下：",
            "details": []
        }

        for _, row in missing_stats.iterrows():
            if row['缺失值数量'] > 0:
                missing_interpretation["details"].append({
                    "变量": row['列名'],
                    "缺失情况": f"缺失 {row['缺失值数量']} 个值，缺失率为 {row['缺失率 (%)']:.2f}%",
                    "建议": row['建议处理方案']
                })

        # 异常值分析
        outlier_stats = self.outlier_analysis()
        outlier_interpretation = {
            "总体情况": f"数据集中共有 {len(outlier_stats)} 个数值型变量，其中 {sum(outlier_stats['异常值比例 (%)'] > 0)} 个变量存在异常值。",
            "异常值分布": "异常值分布情况如下：",
            "details": []
        }

        for _, row in outlier_stats.iterrows():
            if row['异常值比例 (%)'] > 0:
                outlier_interpretation["details"].append({
                    "变量": row['列名'],
                    "Z-score分析": row['Z-score解释'],
                    "IQR分析": row['IQR解释'],
                    "异常值比例": f"{row['异常值比例 (%)']:.2f}%",
                    "建议": "建议根据Z-score和IQR的分析结果，综合考虑是否需要处理异常值。"
                })

        # 重复值分析
        duplicate_stats = self.duplicate_analysis()
        duplicate_interpretation = {
            "总体情况": f"数据集中存在 {duplicate_stats['重复行数量']} 行重复数据，重复率为 {duplicate_stats['重复率 (%)']:.2f}%。",
            "建议": duplicate_stats['建议']
        }

        return {
            "缺失值分析": missing_interpretation,
            "异常值分析": outlier_interpretation,
            "重复值分析": duplicate_interpretation
        }

    def _interpret_statistical_analysis(self) -> Dict[str, Any]:
        """解读统计分析结果"""
        # 正态性检验解读
        normality_stats = self.normality_test()
        normality_interpretation = {
            "总体情况": f"对 {len(normality_stats)} 个数值型变量进行了正态性检验。",
            "检验结果": "检验结果如下：",
            "details": []
        }

        for _, row in normality_stats.iterrows():
            normality_interpretation["details"].append({
                "变量": row['列名'],
                "是否正态分布": row['是否正态分布'],
                "解释": "该变量服从正态分布，可以使用参数检验方法。" if row['是否正态分布'] == '是' else "该变量不服从正态分布，建议使用非参数检验方法。"
            })

        # 峰度和偏度解读
        skew_kurt_stats = self._analyze_skew_kurtosis()
        skew_kurt_interpretation = {
            "总体情况": "对数值型变量进行了峰度和偏度分析，结果如下：",
            "details": []
        }

        for _, row in skew_kurt_stats.iterrows():
            # 偏度解释
            skewness = row['偏度']
            if abs(skewness) < 0.5:
                skew_explanation = "数据分布接近对称"
            elif skewness > 0:
                skew_explanation = "数据右偏，右尾较长，可能存在较大的异常值"
            else:
                skew_explanation = "数据左偏，左尾较长，可能存在较小的异常值"

            # 峰度解释
            kurtosis_val = row['峰度']
            if abs(kurtosis_val) < 0.5:
                kurtosis_explanation = "数据分布接近正态分布"
            elif kurtosis_val > 0:
                kurtosis_explanation = "数据分布尖峰，尾部较厚，存在较多极端值"
            else:
                kurtosis_explanation = "数据分布平坦，尾部较薄，极端值较少"

            skew_kurt_interpretation["details"].append({
                "变量": row['列名'],
                "偏度": f"{skewness:.3f}",
                "偏度解释": skew_explanation,
                "峰度": f"{kurtosis_val:.3f}",
                "峰度解释": kurtosis_explanation,
                "建议": self._get_skew_kurtosis_recommendation(skewness, kurtosis_val)
            })

        # 相关性分析解读
        corr_matrix = self.correlation_analysis()
        corr_interpretation = {
            "总体情况": "变量间的相关性分析结果如下：",
            "强相关变量": [],
            "中等相关变量": [],
            "弱相关变量": []
        }

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= 0.7:
                    corr_interpretation["强相关变量"].append({
                        "变量对": f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                        "相关系数": f"{corr:.2f}",
                        "解释": "这两个变量存在强相关性，可能存在多重共线性问题。"
                    })
                elif abs(corr) >= 0.3:
                    corr_interpretation["中等相关变量"].append({
                        "变量对": f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                        "相关系数": f"{corr:.2f}",
                        "解释": "这两个变量存在中等程度的相关性。"
                    })
                elif abs(corr) >= 0.1:
                    corr_interpretation["弱相关变量"].append({
                        "变量对": f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                        "相关系数": f"{corr:.2f}",
                        "解释": "这两个变量存在弱相关性。"
                    })

        # PCA分析解读
        pca_stats = self.pca_analysis()
        pca_interpretation = {
            "总体情况": f"主成分分析结果显示，数据集中有 {pca_stats['主成分数量']} 个主成分。",
            "建议保留主成分数": f"建议保留 {pca_stats['建议保留主成分数量']} 个主成分，可以解释 {pca_stats['累计方差贡献率'][pca_stats['建议保留主成分数量']-1]*100:.2f}% 的方差。",
            "解释": "主成分分析可以帮助降维，减少变量间的相关性。"
        }

        return {
            "正态性检验": normality_interpretation,
            "峰度偏度分析": skew_kurt_interpretation,
            "相关性分析": corr_interpretation,
            "主成分分析": pca_interpretation
        }

    def _analyze_skew_kurtosis(self) -> pd.DataFrame:
        """分析数值型变量的峰度和偏度"""
        results = []
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                skewness = skew(data)
                kurtosis_val = kurtosis(data)
                results.append({
                    "列名": col,
                    "偏度": skewness,
                    "峰度": kurtosis_val
                })
        return pd.DataFrame(results)

    def _get_skew_kurtosis_recommendation(self, skewness: float, kurtosis: float) -> str:
        """根据峰度和偏度生成建议"""
        recommendations = []
        
        # 偏度建议
        if abs(skewness) >= 1:
            recommendations.append("数据严重偏斜，建议进行数据转换（如对数转换）")
        elif abs(skewness) >= 0.5:
            recommendations.append("数据存在一定偏斜，可以考虑进行数据转换")
        
        # 峰度建议
        if abs(kurtosis) >= 2:
            recommendations.append("数据分布与正态分布差异较大，建议使用稳健统计方法")
        elif abs(kurtosis) >= 1:
            recommendations.append("数据分布与正态分布有一定差异，建议检查异常值")
        
        if not recommendations:
            recommendations.append("数据分布接近正态，可以使用常规统计方法")
        
        return "；".join(recommendations)

    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """生成数据分析和处理建议"""
        recommendations = []

        # 数据质量建议
        missing_stats = self.missing_value_analysis()
        for _, row in missing_stats.iterrows():
            if row['缺失率 (%)'] > 0:
                recommendations.append({
                    "类型": "数据质量",
                    "问题": f"变量 {row['列名']} 存在缺失值",
                    "建议": row['建议处理方案']
                })

        outlier_stats = self.outlier_analysis()
        for _, row in outlier_stats.iterrows():
            if row['异常值比例 (%)'] > 5:
                recommendations.append({
                    "类型": "数据质量",
                    "问题": f"变量 {row['列名']} 存在较多异常值",
                    "建议": "建议检查异常值的合理性，必要时进行处理。"
                })

        # 统计分析建议
        normality_stats = self.normality_test()
        for _, row in normality_stats.iterrows():
            if row['是否正态分布'] == '否':
                recommendations.append({
                    "类型": "统计分析",
                    "问题": f"变量 {row['列名']} 不服从正态分布",
                    "建议": "建议使用非参数检验方法进行分析。"
                })

        # 峰度偏度建议
        skew_kurt_stats = self._analyze_skew_kurtosis()
        for _, row in skew_kurt_stats.iterrows():
            if abs(row['偏度']) >= 1 or abs(row['峰度']) >= 2:
                recommendations.append({
                    "类型": "统计分析",
                    "问题": f"变量 {row['列名']} 的分布严重偏离正态分布",
                    "建议": self._get_skew_kurtosis_recommendation(row['偏度'], row['峰度'])
                })

        corr_matrix = self.correlation_analysis()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= 0.7:
                    recommendations.append({
                        "类型": "统计分析",
                        "问题": f"变量 {corr_matrix.columns[i]} 和 {corr_matrix.columns[j]} 存在强相关性",
                        "建议": "建议考虑删除其中一个变量或使用主成分分析降维。"
                    })

        return recommendations

    def get_memory_usage(self) -> float:
        """获取DataFrame的内存使用情况（MB）"""
        if self._memory_usage is None:
            self._memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
        return self._memory_usage

    def get_basic_info(self) -> Dict[str, Any]:
        """获取数据集的基本信息"""
        return {
            "行数": self.df.shape[0],
            "列数": self.df.shape[1],
            "内存使用(MB)": self.get_memory_usage(),
            "列名列表": self.df.columns.tolist(),
            "数据类型": self.df.dtypes.to_dict()
        }

    def get_categorical_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取分类变量的统计信息"""
        stats = {}
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            stats[col] = {
                "唯一值数量": value_counts.nunique(),
                "前10个值的分布": value_counts.head(10).to_dict()
            }
        return stats

    def get_time_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取时间变量的统计信息"""
        stats = {}
        for col in self.datetime_cols:
            stats[col] = {
                "时间范围": f"{self.df[col].min()} 到 {self.df[col].max()}",
                "时间跨度": str(self.df[col].max() - self.df[col].min())
            }
        return stats

    def explore_dataframe(self, name: str = "DataFrame", show_plots: bool = True) -> Dict[str, Any]:
        """
        对DataFrame进行全面的探索性分析，整合所有分析方法
        
        参数:
            name: DataFrame的名称,用于输出显示
            show_plots: 是否显示可视化图表
            
        返回:
            包含所有分析结果的字典
        """
        logger.info(f"开始探索性分析: {name}")
        
        # 收集所有分析结果
        analysis_results = {
            "基本信息": self.get_basic_info(),
            "缺失值分析": self.missing_value_analysis().to_dict('records'),
            "数值统计": self.basic_statistics().to_dict('records'),
            "分类统计": self.get_categorical_stats(),
            "相关性分析": self.correlation_analysis().to_dict(),
            "重复值分析": self.duplicate_analysis(),
            "异常值分析": self.outlier_analysis().to_dict('records'),
            "正态性检验": self.normality_test().to_dict('records'),
            "时间分析": self.get_time_stats()
        }
        
        # 如果show_plots为True，添加可视化结果
        if show_plots:
            analysis_results["可视化"] = {
                "相关性热图": self.plot_correlation_heatmap(),
                "分布图": {col: self.plot_distribution(col) for col in self.df.columns}
            }
            
            # 为时间列添加时间序列图
            for col in self.datetime_cols:
                if "可视化" not in analysis_results:
                    analysis_results["可视化"] = {}
                analysis_results["可视化"][f"{col}_时间序列"] = self.plot_time_series(self.df[col], col)
        
        return analysis_results

    def plot_time_series(self, data: pd.Series, column: str) -> str:
        """绘制时间序列图"""
        try:
            fig = self.visualizer._create_figure((12, 6))
            data.value_counts().sort_index().plot()
            plt.title(f"{column} 时间分布", fontproperties=self.font_prop)
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            self.visualizer._close_figure(fig)
            return img_base64
        except Exception as e:
            logger.error(f"绘制时间序列图时出错: {str(e)}")
            self.visualizer._close_figure(fig)
            raise

class Visualizer:
    """数据可视化工具类"""
    
    def __init__(self, font_prop: fm.FontProperties):
        self.font_prop = font_prop
        self._figure_cache = {}
    
    def _create_figure(self, figsize: tuple = (10, 6)) -> plt.Figure:
        """创建新的图表"""
        return plt.figure(figsize=figsize)
    
    def _close_figure(self, fig: plt.Figure) -> None:
        """关闭图表"""
        plt.close(fig)
    
    def plot_distribution(self, data: pd.Series, column: str) -> str:
        """绘制分布图"""
        try:
            fig = self._create_figure()
            if pd.api.types.is_numeric_dtype(data):
                sns.histplot(data, kde=True)
                plt.title(f"{column} 的分布", fontproperties=self.font_prop)
                plt.xlabel(column, fontproperties=self.font_prop)
                plt.ylabel('频数', fontproperties=self.font_prop)
            else:
                sns.countplot(x=data)
                plt.title(f"{column} 的频数分布", fontproperties=self.font_prop)
                plt.xlabel(column, fontproperties=self.font_prop)
                plt.ylabel('频数', fontproperties=self.font_prop)
                plt.xticks(rotation=45, fontproperties=self.font_prop)
                plt.yticks(fontproperties=self.font_prop)
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            self._close_figure(fig)
            return img_base64
        except Exception as e:
            logger.error(f"绘制分布图时出错: {str(e)}")
            self._close_figure(fig)
            raise
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> str:
        """绘制相关性热图"""
        try:
            fig = self._create_figure((12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('相关性热图', fontproperties=self.font_prop)
            plt.xticks(fontproperties=self.font_prop)
            plt.yticks(fontproperties=self.font_prop)
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            self._close_figure(fig)
            return img_base64
        except Exception as e:
            logger.error(f"绘制相关性热图时出错: {str(e)}")
            self._close_figure(fig)
            raise
    
    def clear_cache(self) -> None:
        """清除图表缓存"""
        for fig in self._figure_cache.values():
            self._close_figure(fig)
        self._figure_cache.clear()

# 工具函数
def load_data(file_path: str) -> pd.DataFrame:
    """加载数据文件"""
    # 根据文件类型加载数据
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("不支持的文件格式")
    
    return df

def analyze_data(file_path: str, output_html: str = "analysis_report.html") -> None:
    """分析数据并生成报告"""
    df = load_data(file_path)
    analyzer = DataAnalyzer(df)
    analyzer.generate_report(output_html)

def analyze_multiple_files(folder_path: str, output_dir: str = "reports") -> None:
    """分析文件夹中的所有数据文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for file in os.listdir(folder_path):
        if file.endswith(('.csv', '.xls', '.xlsx', '.json')):
            file_path = os.path.join(folder_path, file)
            output_html = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_report.html")
            try:
                analyze_data(file_path, output_html)
                print(f"已分析文件: {file}")
            except Exception as e:
                print(f"分析文件 {file} 时出错: {str(e)}")

def analyze_notebook(df: pd.DataFrame) -> None:
    """在Jupyter Notebook中分析数据并显示报告
    
    参数:
        df: pandas DataFrame，要分析的数据集
    """
    try:
        analyzer = DataAnalyzer(df)
        analyzer.analyze_notebook()
    except Exception as e:
        print(f"分析数据时出错: {str(e)}")

class ReportGenerator:
    """报告生成工具类"""
    
    def __init__(self, template_dir: str, static_dir: str):
        self.template_dir = template_dir
        self.static_dir = static_dir
        self.env = Environment(loader=FileSystemLoader(template_dir))
    
    def _copy_static_files(self) -> None:
        """复制静态文件到输出目录"""
        import shutil
        
        # 源文件路径
        source_js = os.path.join(os.path.dirname(__file__), 'static', 'js', 'analyzer.js')
        target_js = os.path.join(self.static_dir, 'js', 'analyzer.js')
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(target_js), exist_ok=True)
        
        # 复制文件
        try:
            if os.path.exists(source_js):
                if os.path.exists(target_js):
                    # 如果目标文件存在且与源文件不同，则更新
                    if not self._files_are_identical(source_js, target_js):
                        shutil.copy2(source_js, target_js)
                        logger.info(f"更新了JavaScript文件: {target_js}")
                else:
                    shutil.copy2(source_js, target_js)
                    logger.info(f"复制了JavaScript文件: {target_js}")
            else:
                logger.warning(f"警告: 源文件不存在: {source_js}")
        except Exception as e:
            logger.error(f"复制静态文件时出错: {str(e)}")
    
    @staticmethod
    def _files_are_identical(file1: str, file2: str) -> bool:
        """比较两个文件是否相同"""
        try:
            with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
                return f1.read() == f2.read()
        except Exception:
            return False
    
    def generate_report(self, analysis_results: Dict[str, Any], output_html: str) -> None:
        """生成分析报告"""
        try:
            logger.info("开始生成报告...")
            
            # 确保静态文件目录存在并复制静态文件
            os.makedirs(self.static_dir, exist_ok=True)
            os.makedirs(os.path.join(self.static_dir, 'js'), exist_ok=True)
            self._copy_static_files()
            
            # 加载模板
            template = self.env.get_template('analyzer.html')
            logger.info("模板加载成功")
            
            # 获取输出文件的目录
            output_dir = os.path.dirname(os.path.abspath(output_html))
            if not output_dir:
                output_dir = os.getcwd()
            logger.info(f"输出目录: {output_dir}")
            
            # 计算静态文件的相对路径
            static_url = os.path.relpath(self.static_dir, output_dir)
            if static_url.startswith('..'):
                static_url = os.path.join('..', static_url)
            logger.info(f"静态文件URL: {static_url}")
            
            # 处理数据并转换为JSON
            processed_data = DataProcessor.process_data(analysis_results)
            json_data = json.dumps(processed_data, ensure_ascii=False)
            
            # 渲染模板
            logger.info("开始渲染模板...")
            rendered_html = template.render(
                data=json_data,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                static_url=static_url
            )
            logger.info("模板渲染完成")
            
            # 保存报告
            logger.info(f"保存报告到: {output_html}")
            with open(output_html, 'w', encoding='utf-8') as f:
                f.write(rendered_html)
            logger.info(f"分析报告已保存至: {output_html}")
            
        except Exception as e:
            logger.error(f"生成报告时出错: {str(e)}")
            raise
