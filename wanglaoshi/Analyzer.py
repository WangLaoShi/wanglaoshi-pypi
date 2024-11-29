import base64
import io
import json
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
from matplotlib import rcParams
from scipy.stats import skew, kurtosis, zscore

# 设置字体为 SimHei
rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def generate_skew_kurtosis_table(df):
    results = []

    # 遍历所有数值列
    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        col_data = df[column].dropna()
        skewness = skew(col_data)
        kurtosis_val = kurtosis(col_data)

        # 偏度解释
        if skewness > 0:
            skew_explanation = "偏度为正值，表示数据右偏，数据分布的右尾较长。"
        elif skewness < 0:
            skew_explanation = "偏度为负值，表示数据左偏，数据分布的左尾较长。"
        else:
            skew_explanation = "偏度接近零，数据呈对称分布。"

        # 峰度解释
        if kurtosis_val > 0:
            kurtosis_explanation = "峰度为正值，表示数据分布尖峰较高，尾部更厚（可能存在更多极端值）。"
        elif kurtosis_val < 0:
            kurtosis_explanation = "峰度为负值，表示数据分布较平坦，尾部较薄。"
        else:
            kurtosis_explanation = "峰度接近零，数据呈正态分布形态。"

        # 添加结果
        results.append({
            "列名": column,
            "偏度": skewness,
            "偏度解释": skew_explanation,
            "峰度": kurtosis_val,
            "峰度解释": kurtosis_explanation
        })

    # 转换为 DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def calculate_missing_values(df):
    # 缺失值统计
    missing_counts = df.isnull().sum()  # 缺失值数量
    missing_ratios = df.isnull().mean() * 100  # 缺失率
    suggestions = []

    # 生成建议处理方案
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

    # 汇总结果为 DataFrame
    missing_summary = pd.DataFrame({
        "列名": df.columns,
        "缺失值数量": missing_counts,
        "缺失率 (%)": missing_ratios,
        "建议处理方案": suggestions
    })
    return missing_summary


def extended_describe(df):
    # 筛选数值列
    numeric_cols = df.select_dtypes(include=['number']).columns

    # 初步描述性统计
    desc = df.describe(include='all', datetime_is_numeric=True).transpose()

    # 添加额外统计信息
    desc['缺失值数量'] = df.isnull().sum()
    desc['缺失率 (%)'] = (df.isnull().mean() * 100).round(2)
    desc['唯一值数量'] = df.nunique()

    # 仅对数值列计算偏度和峰度
    desc['偏度 (Skewness)'] = None
    desc['峰度 (Kurtosis)'] = None
    for col in numeric_cols:
        desc.at[col, '偏度 (Skewness)'] = skew(df[col].dropna())
        desc.at[col, '峰度 (Kurtosis)'] = kurtosis(df[col].dropna())
    desc.fillna(value='', inplace=True)
    # 检测并将 datetime 列转换为字符串
    for col in desc.select_dtypes(include=['datetime']):
        desc[col] = desc[col].astype(str)
    # desc.to_csv("desc.csv")
    # 返回结果
    return desc.reset_index().rename(columns={'index': '列名'})


def generate_frequency_table(df):
    frequency_tables = []

    # 遍历 object 和 category 类型的列
    for column in df.select_dtypes(include=['object', 'category']).columns:
        # 计算频率分布
        freq = df[column].value_counts().reset_index()
        freq.columns = ['类别', '频数']

        # 添加列名信息
        freq['列名'] = column

        # 重排列顺序
        freq = freq[['列名', '类别', '频数']]
        frequency_tables.append(freq)

    # 合并所有列的结果
    final_table = pd.concat(frequency_tables, ignore_index=True)
    return final_table


# 自定义序列化函数
def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif pd.isna(obj):  # 处理 NaN 值
        return None
    elif isinstance(obj, datetime):
        return obj.isoformat()  # 转换为 ISO 8601 格式字符串
    raise TypeError("Type not serializable")

    return obj


def detect_outliers(df):
    results = []

    # 遍历所有数值列
    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        col_data = df[column].dropna()

        # Z-score 异常值检测
        z_scores = zscore(col_data)
        outliers_zscore_count = np.sum(np.abs(z_scores) > 3)
        zscore_explanation = (
            "使用 Z-score 检测到异常值，通常 Z-score > 3 的值可能是异常值。"
            if outliers_zscore_count > 0
            else "无异常值。"
        )

        # IQR 异常值检测
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_iqr_count = np.sum((col_data < lower_bound) | (col_data > upper_bound))
        iqr_explanation = (
            "使用 IQR 检测到异常值，低于 Q1 - 1.5 * IQR 或高于 Q3 + 1.5 * IQR 的值可能是异常值。"
            if outliers_iqr_count > 0
            else "无异常值。"
        )

        # 添加结果
        results.append({
            "列名": column,
            "异常值数量": outliers_zscore_count,
            "解释": zscore_explanation
        })
        results.append({
            "列名": column,
            "异常值数量": outliers_iqr_count,
            "解释": iqr_explanation
        })

    # 转换为 DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def generate_correlation_table(df):
    # 计算相关系数矩阵
    correlation_matrix = df.corr()

    # 格式化为字符串表格形式
    table_str = correlation_matrix.to_string(
        formatters={col: '{:.2f}'.format for col in correlation_matrix.columns}
    )
    return table_str


def analyze_data_to_html(file_path, output_html="analysis_report.html"):
    ## 如果 file_path 是一个字符串的话，读取文件，否则直接使用 DataFrame
    if isinstance(file_path, str):
        # Load Data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")
    else:
        df = file_path

    # 使用 io.StringIO 获取 DataFrame 的 info
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    data_structure = {
        "menus": [
            {"label": "基本信息", "key": "basic_info"},
            {"label": "描述性统计", "key": "desc_stats"},
            {"label": "缺失值统计", "key": "missing_stats"},
            {"label": "唯一性统计", "key": "unique_stats"},
            {"label": "频度分布", "key": "value_counts"},
            {"label": "偏度和峰度分析", "key": "skew_kurt"},
            {"label": "异常值检测", "key": "outliers"},
            # {"label": "相关系数矩阵", "key": "correlation_matrix"},
            {"label": "图表展示", "key": "plots"}
        ],
        "tables": {
            "basic_info": {"type": "String", "columns": []},
            "desc_stats": {"type": "Table",
                           "columns": ['列名', 'count', 'unique', 'top', 'freq', 'mean', 'min', '25%', '50%', '75%',
                                       'max', 'std', '缺失值数量', '缺失率 (%)', '唯一值数量', '偏度 (Skewness)',
                                       '峰度 (Kurtosis)']},
            "missing_stats": {"type": "Table", "columns": ["列名", "缺失值数量", "缺失率 (%)", "建议处理方案"]},
            "unique_stats": {"type": "Table", "columns": ['列名', '唯一值数量']},
            "value_counts": {"type": "Table", "columns": ['列名', '类别', '频度分布']},
            "skew_kurt": {"type": "Table", "columns": ['列名', '偏度', '偏度解释', '峰度', '峰度解释']},
            "outliers": {"type": "Table", "columns": ['列名', '异常值数量', '解释']},
            # "correlation_matrix":{"type":"String","columns":[]},
            "plots": {"type": "Plot", "columns": []}
        }
    }
    # Data Analysis
    data = {
        "basic_info": info_str,
        "missing_stats": calculate_missing_values(df).values.tolist(),
        "unique_stats": df.nunique().reset_index().values.tolist(),
        "desc_stats": extended_describe(df).values.tolist(),
        "value_counts": generate_frequency_table(df).values.tolist(),
        "skew_kurt": generate_skew_kurtosis_table(df).values.tolist(),
        "outliers": detect_outliers(df).values.tolist(),
        # "correlation_matrix": generate_correlation_table(df),
        "plots": []
    }

    numeric_cols = df.select_dtypes(include=np.number).columns

    # Plots
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        data["plots"].append(img_base64)
        buf.close()
        plt.close(fig)
    # 绘制相关性矩阵热图
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("数值特征的相关性矩阵")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    data["plots"].append(img_base64)
    buf.close()
    plt.close(fig)
    # Load External Template
    env = Environment(loader=FileSystemLoader('./templates/'))
    template = env.get_template('analyzer.html')
    data_json = json.dumps(data, default=convert_to_serializable)
    # 保存 data_json
    # with open("data.json","w",encoding="utf-8") as f:
    #     f.write(data_json)
    # Render Template
    rendered_html = template.render({
        "menus": data_structure['menus'],
        "data": data_json,
        "tables": data_structure['tables']
    })

    # 得到当前时间，追加到文件上
    file_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_html = f"analysis_report_{file_time}.html"
    # Save Rendered HTML
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(rendered_html)
        print(f"Analysis report saved to {output_html}")


# Example usage
# if __name__ == "__main__":
#     analyze_data_to_html("科研数据汇总（2024-10-27清洁版）.xlsx")
