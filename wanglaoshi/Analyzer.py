import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, zscore
import missingno as msno
import os


def analyze_data(file_path):
    # 获取文件名和扩展名
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_extension = os.path.splitext(file_path)[1].lower()

    # 判断文件格式并选择读取方法
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    elif file_extension == '.json':
        df = pd.read_json(file_path)
    else:
        raise ValueError("不支持的文件格式。请使用 CSV、Excel 或 JSON 格式的文件。")

    # 设置输出文件名称
    output_text_file = f"{file_name}_analysis_output.txt"

    # 开始分析并输出结果
    with open(output_text_file, "w") as f:
        def write_output(text):
            # 在文件和控制台同时输出
            print(text)
            f.write(text + "\n")

        # 1. 数据基本信息
        write_output("=== DataFrame 基本信息 ===")
        df_info = df.info(buf=f)

        # 缺失值数量和比例
        write_output("\n=== DataFrame 缺失值统计 ===")
        missing_values = df.isnull().sum()
        missing_percentage = df.isnull().mean() * 100
        missing_df = pd.DataFrame({'缺失值数量': missing_values, '缺失值比例': missing_percentage})
        write_output(str(missing_df))

        # 唯一值数量
        write_output("\n=== DataFrame 唯一值数量 ===")
        unique_values = df.nunique()
        write_output(str(unique_values))

        # 2. 数值型特征的描述性统计分析
        write_output("\n=== 数值型特征的描述性统计 ===")
        desc_stats = df.describe()
        write_output(str(desc_stats))

        # 偏度和峰度分析
        write_output("\n=== 偏度和峰度分析 ===")
        for column in df.select_dtypes(include=['int64', 'float64']).columns:
            skewness = skew(df[column].dropna())
            kurtosis_val = kurtosis(df[column].dropna())

            # 偏度
            write_output(f"\n{column} 的偏度 (Skewness): {skewness}")
            if skewness > 0:
                write_output("解释: 偏度为正值，表示数据右偏，数据分布的右尾较长。")
            elif skewness < 0:
                write_output("解释: 偏度为负值，表示数据左偏，数据分布的左尾较长。")
            else:
                write_output("解释: 偏度接近零，数据呈对称分布。")

            # 峰度
            write_output(f"{column} 的峰度 (Kurtosis): {kurtosis_val}")
            if kurtosis_val > 0:
                write_output("解释: 峰度为正值，表示数据分布尖峰较高，尾部更厚（可能存在更多极端值）。")
            elif kurtosis_val < 0:
                write_output("解释: 峰度为负值，表示数据分布较平坦，尾部较薄。")
            else:
                write_output("解释: 峰度接近零，数据呈正态分布形态。")

        # 3. 分类特征的频率分布
        for column in df.select_dtypes(include=['object', 'category']).columns:
            write_output(f"\n=== {column} 的频率分布 ===")
            write_output(str(df[column].value_counts()))
            write_output(f"解释: {column} 各类别的出现次数，便于了解该列的类别是否均衡。")

        # 6. 异常值检测
        write_output("=== 异常值检测 ===")
        for column in df.select_dtypes(include=['int64', 'float64']).columns:
            df['zscore_' + column] = zscore(df[column].dropna())
            outliers_zscore = df[np.abs(df['zscore_' + column]) > 3]
            write_output(f"\n{column} 的 Z-score 异常值数量：{len(outliers_zscore)}")
            if len(outliers_zscore) > 0:
                write_output("解释: 使用 Z-score 检测到异常值，通常 Z-score > 3 的值可能是异常值。")

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers_iqr = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
            write_output(f"{column} 的 IQR 异常值数量：{len(outliers_iqr)}")
            if len(outliers_iqr) > 0:
                write_output("解释: 使用 IQR 检测到异常值，低于 Q1 - 1.5 * IQR 或高于 Q3 + 1.5 * IQR 的值可能是异常值。")

        # 7. 数值特征的相关性分析
        write_output("\n=== 数值特征的相关系数矩阵 ===")
        correlation_matrix = df.corr()
        write_output(str(correlation_matrix))
        write_output(
            "解释: 相关性矩阵用于显示数值特征之间的相关程度。高相关性可能表明特征间的共线性，可在模型训练中考虑消除冗余特征。")

    # 生成图表并直接显示
    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[column], kde=True)
        plt.title(f"{column}的分布图 - 直方图和密度图")
        plt.xlabel(column)
        plt.ylabel("频率")
        plt.show()  # 显示直方图

        sns.boxplot(x=df[column])
        plt.title(f"{column}的箱线图 - 显示中位数、四分位数和异常值")
        plt.xlabel(column)
        plt.show()  # 显示箱线图

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("数值特征的相关性矩阵")
    plt.show()  # 显示相关性热图

    msno.heatmap(df)
    plt.title("缺失值热图")
    plt.show()  # 显示缺失值热图

    write_output("分析已完成，结果已显示在控制台。")

# 使用示例
# analyze_data("your_data_file.csv")