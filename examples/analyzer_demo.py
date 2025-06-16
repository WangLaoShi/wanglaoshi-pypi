"""
DataAnalyzer 使用示例
这个示例展示了如何使用 DataAnalyzer 进行数据分析并生成报告
"""

import pandas as pd
import numpy as np
from wanglaoshi import DataAnalyzer
import os
import json
from pprint import pprint

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
    print("\n=== 基础分析示例 ===")
    
    # 创建示例数据
    df = create_sample_data()
    print("\n数据预览:")
    print(df.head())
    
    # 创建分析器实例
    analyzer = DataAnalyzer(df)
    
    # 使用explore_dataframe进行基础分析
    print("\n进行基础分析...")
    results = analyzer.explore_dataframe(name="示例数据集", show_plots=True)
    
    # 打印基本信息
    print("\n基本信息:")
    pprint(results["基本信息"])
    
    # 打印缺失值分析
    print("\n缺失值分析:")
    pprint(results["缺失值分析"])
    
    # 打印异常值分析
    print("\n异常值分析:")
    pprint(results["异常值分析"])
    
    return results

def advanced_analysis_demo():
    """高级分析示例"""
    print("\n=== 高级分析示例 ===")
    
    # 创建示例数据
    df = create_sample_data()
    analyzer = DataAnalyzer(df)
    
    # 使用explore_dataframe进行高级分析
    print("\n进行高级分析...")
    results = analyzer.explore_dataframe(name="示例数据集", show_plots=True)
    
    # 打印相关性分析
    print("\n相关性分析:")
    pprint(results["相关性分析"])
    
    # 打印正态性检验
    print("\n正态性检验:")
    pprint(results["正态性检验"])
    
    # 打印时间分析
    print("\n时间分析:")
    pprint(results["时间分析"])
    
    return results

def visualization_demo():
    """可视化示例"""
    print("\n=== 可视化示例 ===")
    
    # 创建示例数据
    df = create_sample_data()
    analyzer = DataAnalyzer(df)
    
    # 使用explore_dataframe生成可视化
    print("\n生成可视化...")
    results = analyzer.explore_dataframe(name="示例数据集", show_plots=True)
    
    # 检查可视化结果
    if "可视化" in results:
        print("\n已生成以下可视化:")
        for plot_name in results["可视化"].keys():
            print(f"- {plot_name}")
    
    return results

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
    
    # 保存分析结果为JSON
    print("\n保存分析结果为JSON...")
    results = analyzer.explore_dataframe(name="示例数据集", show_plots=True)
    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("分析结果已保存: analysis_results.json")

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
    
    # 进行探索性分析
    results = analyzer.explore_dataframe(name="我的数据集", show_plots=True)
    
    # 查看分析结果
    print("基本信息:", results["基本信息"])
    print("缺失值分析:", results["缺失值分析"])
    print("异常值分析:", results["异常值分析"])
    
    # 生成HTML报告
    analyzer.generate_report("analysis_report.html")
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