import base64
import io
import json
from datetime import datetime
from io import BytesIO
from typing import Union, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
from matplotlib import rcParams
from scipy import stats
from scipy.stats import skew, kurtosis, zscore, chi2_contingency, normaltest, shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    """数据分析器主类"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.datetime_cols = self.df.select_dtypes(include=['datetime']).columns

    # ==================== 基础统计分析 ====================
    def basic_statistics(self) -> pd.DataFrame:
        """计算基本统计量"""
        stats_df = self.df.describe(include='all', datetime_is_numeric=True).transpose()
        stats_df['缺失值数量'] = self.df.isnull().sum()
        stats_df['缺失率 (%)'] = (self.df.isnull().mean() * 100).round(2)
        stats_df['唯一值数量'] = self.df.nunique()
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
                    'D\'Agostino-Pearson统计量': norm_stat,
                    'D\'Agostino-Pearson p值': norm_p,
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
    def correlation_analysis(self) -> pd.DataFrame:
        """相关性分析"""
        corr_matrix = self.df.corr()
        return corr_matrix

    def multicollinearity_analysis(self) -> pd.DataFrame:
        """多重共线性分析"""
        vif_data = pd.DataFrame()
        vif_data["列名"] = self.numeric_cols
        vif_data["VIF"] = [variance_inflation_factor(self.df[self.numeric_cols].values, i) 
                          for i in range(len(self.numeric_cols))]
        return vif_data

    def pca_analysis(self) -> Dict[str, Any]:
        """主成分分析"""
        if len(self.numeric_cols) < 2:
            return {"error": "需要至少两个数值型变量进行PCA分析"}
            
        # 标准化数据
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

    # ==================== 数据可视化 ====================
    def plot_distribution(self, column: str) -> str:
        """绘制分布图"""
        if column not in self.df.columns:
            return "列名不存在"
            
        plt.figure(figsize=(10, 6))
        if column in self.numeric_cols:
            sns.histplot(self.df[column], kde=True)
            plt.title(f"{column} 的分布")
        else:
            sns.countplot(x=column, data=self.df)
            plt.title(f"{column} 的频数分布")
            plt.xticks(rotation=45)
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64

    def plot_correlation_heatmap(self) -> str:
        """绘制相关性热图"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.correlation_analysis(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('相关性热图')
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64

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

    def _interpret_basic_info(self) -> Dict[str, str]:
        """解读基本信息"""
        total_rows, total_cols = self.df.shape
        numeric_count = len(self.numeric_cols)
        categorical_count = len(self.categorical_cols)
        datetime_count = len(self.datetime_cols)

        return {
            "数据集规模": f"数据集包含 {total_rows} 行和 {total_cols} 列。",
            "变量类型": f"其中包含 {numeric_count} 个数值型变量、{categorical_count} 个分类型变量和 {datetime_count} 个时间型变量。",
            "说明": "这个规模的数据集适合进行统计分析，但需要注意数据质量和变量之间的关系。"
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
                    "问题": f"变量 '{row['列名']}' 存在缺失值",
                    "建议": row['建议处理方案']
                })

        outlier_stats = self.outlier_analysis()
        for _, row in outlier_stats.iterrows():
            if row['异常值比例 (%)'] > 5:
                recommendations.append({
                    "类型": "数据质量",
                    "问题": f"变量 '{row['列名']}' 存在较多异常值",
                    "建议": "建议检查异常值的合理性，必要时进行处理。"
                })

        # 统计分析建议
        normality_stats = self.normality_test()
        for _, row in normality_stats.iterrows():
            if row['是否正态分布'] == '否':
                recommendations.append({
                    "类型": "统计分析",
                    "问题": f"变量 '{row['列名']}' 不服从正态分布",
                    "建议": "建议使用非参数检验方法进行分析。"
                })

        # 峰度偏度建议
        skew_kurt_stats = self._analyze_skew_kurtosis()
        for _, row in skew_kurt_stats.iterrows():
            if abs(row['偏度']) >= 1 or abs(row['峰度']) >= 2:
                recommendations.append({
                    "类型": "统计分析",
                    "问题": f"变量 '{row['列名']}' 的分布严重偏离正态分布",
                    "建议": self._get_skew_kurtosis_recommendation(row['偏度'], row['峰度'])
                })

        corr_matrix = self.correlation_analysis()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= 0.7:
                    recommendations.append({
                        "类型": "统计分析",
                        "问题": f"变量 '{corr_matrix.columns[i]}' 和 '{corr_matrix.columns[j]}' 存在强相关性",
                        "建议": "建议考虑删除其中一个变量或使用主成分分析降维。"
                    })

        return recommendations

    # ==================== 报告生成 ====================
    def generate_report(self, output_html: str = "analysis_report.html") -> None:
        """生成分析报告"""
        # 收集所有分析结果
        analysis_results = {
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
            "interpretations": self.interpret_results()  # 添加结果解读
        }
        
        # 加载模板
        env = Environment(loader=FileSystemLoader('./templates/'))
        template = env.get_template('analyzer.html')
        
        # 渲染模板
        rendered_html = template.render({
            "data": json.dumps(analysis_results, default=self._convert_to_serializable),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 保存报告
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
        print(f"分析报告已保存至: {output_html}")

    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """将对象转换为可序列化的格式"""
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        raise TypeError(f"Type {type(obj)} not serializable")

# 工具函数
def load_data(file_path: str) -> pd.DataFrame:
    """加载数据文件"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("不支持的文件格式")

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
