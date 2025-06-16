"""
DataAnalyzer 使用示例
这个示例展示了如何使用 DataAnalyzer 进行数据分析并生成报告
"""

import pandas as pd
import numpy as np
from wanglaoshi import DataAnalyzer
import os

def create_sample_data():
    """创建示例数据"""
    # 创建随机数据
    np.random.seed(42)
    n_samples = 1000
    
    # 数值型数据
    data = {
        'age': np.random.normal(35, 10, n_samples),  # 年龄
        'income': np.random.lognormal(10, 1, n_samples),  # 收入
        'height': np.random.normal(170, 10, n_samples),  # 身高
        'weight': np.random.normal(65, 15, n_samples),  # 体重
        'satisfaction': np.random.randint(1, 6, n_samples),  # 满意度评分
    }
    
    # 添加一些缺失值
    for col in ['age', 'income', 'height', 'weight']:
        mask = np.random.random(n_samples) < 0.05  # 5%的缺失值
        data[col][mask] = np.nan
    
    # 添加一些异常值
    data['income'][np.random.choice(n_samples, 5)] = data['income'].max() * 2
    
    # 创建分类数据
    data['gender'] = np.random.choice(['男', '女'], n_samples)
    data['education'] = np.random.choice(['高中', '本科', '硕士', '博士'], n_samples)
    data['occupation'] = np.random.choice(['工程师', '教师', '医生', '销售', '其他'], n_samples)
    
    # 创建时间数据
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    data['date'] = dates
    
    return pd.DataFrame(data)

def basic_analysis_demo():
    """基础分析示例"""
    print("=== 基础分析示例 ===")
    
    # 创建示例数据
    df = create_sample_data()
    print("\n数据预览:")
    print(df.head())
    
    # 创建分析器实例
    analyzer = DataAnalyzer(df)
    
    # 基本统计分析
    print("\n基本统计信息:")
    basic_stats = analyzer.basic_statistics()
    print(basic_stats)
    
    # 正态性检验
    print("\n正态性检验结果:")
    normality_test = analyzer.normality_test()
    print(normality_test)
    
    # 缺失值分析
    print("\n缺失值分析:")
    missing_analysis = analyzer.missing_value_analysis()
    print(missing_analysis)
    
    # 异常值分析
    print("\n异常值分析:")
    outlier_analysis = analyzer.outlier_analysis()
    print(outlier_analysis)
    
    # 重复值分析
    print("\n重复值分析:")
    duplicate_analysis = analyzer.duplicate_analysis()
    print(duplicate_analysis)

def advanced_analysis_demo():
    """高级分析示例"""
    print("\n=== 高级分析示例 ===")
    
    # 创建示例数据
    df = create_sample_data()
    analyzer = DataAnalyzer(df)
    
    # 相关性分析
    print("\n相关性分析:")
    correlation_matrix = analyzer.correlation_analysis()
    print(correlation_matrix)
    
    # 多重共线性分析
    print("\n多重共线性分析:")
    multicollinearity = analyzer.multicollinearity_analysis()
    print(multicollinearity)
    
    # 主成分分析
    print("\n主成分分析:")
    pca_analysis = analyzer.pca_analysis()
    print(pca_analysis)

def visualization_demo():
    """可视化示例"""
    print("\n=== 可视化示例 ===")
    
    # 创建示例数据
    df = create_sample_data()
    analyzer = DataAnalyzer(df)
    
    # 分布图
    print("\n生成分布图...")
    for column in ['age', 'income', 'height', 'weight']:
        print(f"\n{column} 的分布图:")
        img_base64 = analyzer.plot_distribution(column)
        print(f"图片已生成，base64长度: {len(img_base64)}")
    
    # 相关性热图
    print("\n生成相关性热图...")
    heatmap_base64 = analyzer.plot_correlation_heatmap()
    print(f"热图已生成，base64长度: {len(heatmap_base64)}")

def report_generation_demo():
    """报告生成示例"""
    print("\n=== 报告生成示例 ===")
    
    # 创建示例数据
    df = create_sample_data()
    analyzer = DataAnalyzer(df)
    
    # 生成HTML报告
    print("\n生成分析报告...")
    analyzer.generate_report("analysis_report.html")
    print("报告已生成: analysis_report.html")

def file_analysis_demo():
    """文件分析示例"""
    print("\n=== 文件分析示例 ===")
    
    # 创建示例数据文件
    print("\n创建示例数据文件...")
    data_dir = "example_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建CSV文件
    df1 = create_sample_data()
    df1.to_csv(os.path.join(data_dir, "sample_data1.csv"), index=False)
    
    # 创建另一个CSV文件，使用不同的数据
    df2 = create_sample_data()  # 使用不同的随机种子
    df2.to_csv(os.path.join(data_dir, "sample_data2.csv"), index=False)
    
    # 创建Excel文件
    df3 = create_sample_data()  # 使用不同的随机种子
    df3.to_excel(os.path.join(data_dir, "sample_data3.xlsx"), index=False)
    
    print(f"示例数据文件已创建在 {data_dir} 目录下")
    
    # 分析单个文件
    print("\n分析单个文件示例:")
    print("分析 sample_data1.csv...")
    from wanglaoshi import analyze_data
    analyze_data(
        os.path.join(data_dir, "sample_data1.csv"),
        "analysis_report_single.html"
    )
    print("单个文件分析报告已生成: analysis_report_single.html")
    
    # 分析多个文件
    print("\n分析多个文件示例:")
    print("分析目录下的所有数据文件...")
    from wanglaoshi import analyze_multiple_files
    analyze_multiple_files(data_dir, "reports")
    print("多个文件的分析报告已生成在 reports 目录下")
    
    # 清理示例文件
    print("\n清理示例文件...")
    import shutil
    shutil.rmtree(data_dir)
    print("示例数据文件已清理")

def notebook_demo():
    """Jupyter Notebook示例"""
    print("\n=== Jupyter Notebook示例 ===")
    print("""
    在Jupyter Notebook中使用:
    
    ```python
    import pandas as pd
    from wanglaoshi import DataAnalyzer
    
    # 创建或加载数据
    df = pd.DataFrame(...)
    
    # 创建分析器实例
    analyzer = DataAnalyzer(df)
    
    # 在notebook中显示分析报告
    analyzer.analyze_notebook()
    ```
    """)

def main():
    """主函数"""
    print("DataAnalyzer 使用示例\n")
    
    # 运行各个示例
    basic_analysis_demo()
    advanced_analysis_demo()
    visualization_demo()
    report_generation_demo()
    file_analysis_demo()
    notebook_demo()

if __name__ == "__main__":
    main() 