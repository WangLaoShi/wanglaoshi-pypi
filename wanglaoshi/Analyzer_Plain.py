"""
简化版数据分析模块 (Analyzer_Plain.py)

这个模块提供了一个简化版的数据分析函数，功能包括：
1. 数据基本信息统计
2. 缺失值分析
3. 描述性统计分析
4. 偏度和峰度分析
5. 分类特征频率分布
6. 异常值检测（Z-score和IQR方法）
7. 相关性分析
8. 数据可视化（分布图、箱线图、热图）

与Analyzer.py的区别：
- 这个模块更简单，只有一个函数
- 输出结果到文本文件和控制台
- 适合快速分析，不需要生成HTML报告

使用场景：
- 快速数据探索
- 简单的数据分析任务
- 不需要复杂报告的场景
"""

import pandas as pd  # 数据处理库
import numpy as np   # 数值计算库
import seaborn as sns  # 数据可视化库
from scipy.stats import skew, kurtosis, zscore  # 统计函数
import missingno as msno  # 缺失值可视化库
import os  # 操作系统接口
import matplotlib.pyplot as plt  # 绘图库
from matplotlib import rcParams  # matplotlib配置

# ==================== 配置中文字体 ====================
# 设置字体为 SimHei（黑体），用于中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题（避免显示为方块）

def analyze_data(file_path):
    """
    对数据文件进行全面的探索性数据分析
    
    这个函数会执行完整的数据分析流程，包括：
    1. 加载数据（支持CSV、Excel、JSON格式）
    2. 基本统计信息
    3. 缺失值分析
    4. 描述性统计
    5. 偏度和峰度分析
    6. 分类特征频率分布
    7. 异常值检测
    8. 相关性分析
    9. 数据可视化
    
    参数:
        file_path: 数据文件的路径
                  支持格式：.csv, .xls, .xlsx, .json
        
    输出:
        - 控制台输出：分析结果
        - 文本文件：{文件名}_analysis_output.txt（保存所有分析结果）
        - 图表：直接显示（分布图、箱线图、热图等）
        
    示例:
        analyze_data("data.csv")  # 分析CSV文件
        analyze_data("data.xlsx")  # 分析Excel文件
    """
    # ========== 步骤1：解析文件路径 ==========
    # 获取文件名（不含扩展名）
    # os.path.basename() 获取文件名，例如: "data.csv" -> "data.csv"
    # os.path.splitext() 分离文件名和扩展名，例如: "data.csv" -> ("data", ".csv")
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # 获取文件扩展名（转换为小写，便于比较）
    file_extension = os.path.splitext(file_path)[1].lower()

    # ========== 步骤2：根据文件格式加载数据 ==========
    if file_extension == '.csv':
        # 读取CSV文件
        df = pd.read_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        # 读取Excel文件（支持.xls和.xlsx格式）
        df = pd.read_excel(file_path)
    elif file_extension == '.json':
        # 读取JSON文件
        df = pd.read_json(file_path)
    else:
        # 如果不支持的文件格式，抛出错误
        raise ValueError("不支持的文件格式。请使用 CSV、Excel 或 JSON 格式的文件。")

    # ========== 步骤3：设置输出文件 ==========
    # 输出文本文件名：原文件名 + "_analysis_output.txt"
    output_text_file = f"{file_name}_analysis_output.txt"

    # ========== 步骤4：开始分析并输出结果 ==========
    # 打开输出文件（写入模式）
    with open(output_text_file, "w", encoding='utf-8') as f:
        def write_output(text):
            """
            辅助函数：同时输出到控制台和文件
            
            这样可以：
            - 在控制台实时查看分析结果
            - 将结果保存到文件，方便后续查看
            """
            # 在控制台打印
            print(text)
            # 写入文件（添加换行符）
            f.write(text + "\n")

        # ========== 1. 数据基本信息 ==========
        write_output("=== DataFrame 基本信息 ===")
        # df.info() 显示DataFrame的详细信息（列数、数据类型、非空值数量等）
        # buf=f 将输出重定向到文件对象f
        df_info = df.info(buf=f)

        # ========== 缺失值分析 ==========
        write_output("\n=== DataFrame 缺失值统计 ===")
        # 计算每列的缺失值数量
        missing_values = df.isnull().sum()
        # 计算每列的缺失值比例（百分比）
        missing_percentage = df.isnull().mean() * 100
        # 创建DataFrame显示缺失值统计
        missing_df = pd.DataFrame({'缺失值数量': missing_values, '缺失值比例': missing_percentage})
        write_output(str(missing_df))

        # ========== 唯一值统计 ==========
        write_output("\n=== DataFrame 唯一值数量 ===")
        # nunique() 计算每列的唯一值数量
        unique_values = df.nunique()
        write_output(str(unique_values))

        # ========== 2. 数值型特征的描述性统计分析 ==========
        write_output("\n=== 数值型特征的描述性统计 ===")
        # describe() 计算基本统计量：计数、均值、标准差、最小值、25%分位数、中位数、75%分位数、最大值
        desc_stats = df.describe()
        write_output(str(desc_stats))

        # ========== 偏度和峰度分析 ==========
        write_output("\n=== 偏度和峰度分析 ===")
        # 只对数值型列进行分析
        for column in df.select_dtypes(include=['int64', 'float64']).columns:
            # 删除缺失值后计算偏度和峰度
            skewness = skew(df[column].dropna())  # 偏度：衡量数据分布的对称性
            kurtosis_val = kurtosis(df[column].dropna())  # 峰度：衡量数据分布的尖锐程度

            # 偏度解释
            write_output(f"\n{column} 的偏度 (Skewness): {skewness}")
            if skewness > 0:
                write_output("解释: 偏度为正值，表示数据右偏，数据分布的右尾较长。")
            elif skewness < 0:
                write_output("解释: 偏度为负值，表示数据左偏，数据分布的左尾较长。")
            else:
                write_output("解释: 偏度接近零，数据呈对称分布。")

            # 峰度解释
            write_output(f"{column} 的峰度 (Kurtosis): {kurtosis_val}")
            if kurtosis_val > 0:
                write_output("解释: 峰度为正值，表示数据分布尖峰较高，尾部更厚（可能存在更多极端值）。")
            elif kurtosis_val < 0:
                write_output("解释: 峰度为负值，表示数据分布较平坦，尾部较薄。")
            else:
                write_output("解释: 峰度接近零，数据呈正态分布形态。")

        # ========== 3. 分类特征的频率分布 ==========
        # 只对分类型列进行分析
        for column in df.select_dtypes(include=['object', 'category']).columns:
            write_output(f"\n=== {column} 的频率分布 ===")
            # value_counts() 统计每个类别出现的次数
            write_output(str(df[column].value_counts()))
            write_output(f"解释: {column} 各类别的出现次数，便于了解该列的类别是否均衡。")

        # ========== 6. 异常值检测 ==========
        write_output("=== 异常值检测 ===")
        # 只对数值型列进行异常值检测
        for column in df.select_dtypes(include=['int64', 'float64']).columns:
            # 方法1：Z-score方法
            # 计算Z-score（标准分数），表示数据点距离均值有多少个标准差
            df['zscore_' + column] = zscore(df[column].dropna())
            # 找出Z-score绝对值大于3的异常值（通常认为|Z| > 3是异常值）
            outliers_zscore = df[np.abs(df['zscore_' + column]) > 3]
            write_output(f"\n{column} 的 Z-score 异常值数量：{len(outliers_zscore)}")
            if len(outliers_zscore) > 0:
                write_output("解释: 使用 Z-score 检测到异常值，通常 Z-score > 3 的值可能是异常值。")

            # 方法2：IQR方法（四分位距方法）
            Q1 = df[column].quantile(0.25)  # 第一四分位数（25%分位数）
            Q3 = df[column].quantile(0.75)  # 第三四分位数（75%分位数）
            IQR = Q3 - Q1  # 四分位距
            # 找出超出Q1-1.5*IQR或Q3+1.5*IQR范围的异常值
            outliers_iqr = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
            write_output(f"{column} 的 IQR 异常值数量：{len(outliers_iqr)}")
            if len(outliers_iqr) > 0:
                write_output("解释: 使用 IQR 检测到异常值，低于 Q1 - 1.5 * IQR 或高于 Q3 + 1.5 * IQR 的值可能是异常值。")

        # ========== 7. 数值特征的相关性分析 ==========
        write_output("\n=== 数值特征的相关系数矩阵 ===")
        # corr() 计算所有数值型列之间的相关系数矩阵
        correlation_matrix = df.corr()
        write_output(str(correlation_matrix))
        write_output(
            "解释: 相关性矩阵用于显示数值特征之间的相关程度。高相关性可能表明特征间的共线性，可在模型训练中考虑消除冗余特征。")

    # ========== 8. 数据可视化 ==========
    # 为每个数值型列生成分布图和箱线图
    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        # 绘制直方图和密度曲线
        plt.figure(figsize=(10, 4))
        # histplot() 绘制直方图
        # kde=True 同时绘制核密度估计曲线（密度曲线）
        sns.histplot(df[column], kde=True)
        plt.title(f"{column}的分布图 - 直方图和密度图")
        plt.xlabel(column)
        plt.ylabel("频率")
        plt.show()  # 显示直方图

        # 绘制箱线图
        # boxplot() 绘制箱线图，显示中位数、四分位数和异常值
        sns.boxplot(x=df[column])
        plt.title(f"{column}的箱线图 - 显示中位数、四分位数和异常值")
        plt.xlabel(column)
        plt.show()  # 显示箱线图

    # 绘制相关性热图
    plt.figure(figsize=(12, 8))
    # heatmap() 绘制热图
    # annot=True 在单元格中显示数值
    # cmap='coolwarm' 使用冷暖色配色方案（蓝-白-红）
    # fmt=".2f" 数值格式：保留2位小数
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("数值特征的相关性矩阵")
    plt.show()  # 显示相关性热图

    # 绘制缺失值热图
    # missingno库提供的缺失值可视化功能
    # 用颜色表示缺失值的位置和模式
    msno.heatmap(df)
    plt.title("缺失值热图")
    plt.show()  # 显示缺失值热图

    # 分析完成提示
    write_output("分析已完成，结果已显示在控制台。")

# ==================== 使用示例 ====================
# analyze_data("your_data_file.csv")  # 分析CSV文件
# analyze_data("your_data_file.xlsx")  # 分析Excel文件
# analyze_data("your_data_file.json")  # 分析JSON文件