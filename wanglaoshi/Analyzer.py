"""
数据分析工具模块 (Analyzer.py)

这个文件是一个功能强大的数据分析工具，主要作用包括：

1. 【数据质量分析】
   - 检测和分析数据中的缺失值（空值）
   - 识别异常值（离群值）
   - 查找重复的数据行
   - 提供数据质量改进建议

2. 【统计分析】
   - 计算基本统计量（平均值、标准差、最大值、最小值等）
   - 进行正态性检验（判断数据是否符合正态分布）
   - 分析变量之间的相关性
   - 检测多重共线性问题
   - 进行主成分分析（PCA）降维

3. 【数据可视化】
   - 绘制数据分布图（直方图、条形图）
   - 生成相关性热图
   - 创建时间序列图

4. 【报告生成】
   - 自动生成美观的HTML格式分析报告
   - 包含所有分析结果和可视化图表
   - 提供数据解读和处理建议

使用场景：
- 数据探索性分析（EDA）
- 数据质量评估
- 统计分析前的数据检查
- 生成数据分析报告

主要类：
- DataProcessor: 数据处理工具类，负责数据验证和格式转换
- DataAnalyzer: 数据分析器主类，提供各种分析方法
- Visualizer: 数据可视化工具类，负责生成图表
- ReportGenerator: 报告生成工具类，负责生成HTML报告
"""

# ==================== 导入必要的库 ====================
# os: 用于文件路径操作
import os
# base64: 用于将图片转换为Base64编码（用于在HTML中嵌入图片）
import base64
# json: 用于将数据转换为JSON格式（用于在HTML报告中传递数据）
import json
# logging: 用于记录程序运行日志（帮助调试和追踪问题）
import logging
# datetime: 用于处理日期和时间
from datetime import datetime
# BytesIO: 用于在内存中处理二进制数据（如图片）
from io import BytesIO
# typing: 用于类型提示（让代码更易读，帮助IDE提供代码补全）
from typing import Dict, List, Any

# matplotlib: Python最流行的绘图库，用于创建各种图表
import matplotlib.pyplot as plt
# matplotlib.font_manager: 用于管理字体（支持中文显示）
import matplotlib.font_manager as fm
# rcParams: 用于配置matplotlib的全局参数
from matplotlib import rcParams
# numpy: 数值计算库，提供数组和数学函数
import numpy as np
# pandas: 数据分析库，提供DataFrame数据结构（类似Excel表格）
import pandas as pd
# seaborn: 基于matplotlib的统计绘图库，提供更美观的图表
import seaborn as sns
# jinja2: 模板引擎，用于生成HTML报告
from jinja2 import Environment, FileSystemLoader
# scipy.stats: 提供统计检验函数
# - skew: 计算偏度（衡量数据分布的对称性）
# - kurtosis: 计算峰度（衡量数据分布的尖锐程度）
# - zscore: 计算Z分数（用于异常值检测）
# - normaltest: 正态性检验
# - shapiro: Shapiro-Wilk正态性检验
from scipy.stats import skew, kurtosis, zscore, normaltest, shapiro
# sklearn: 机器学习库
# - StandardScaler: 数据标准化（将数据转换为均值为0、标准差为1的分布）
from sklearn.preprocessing import StandardScaler
# - PCA: 主成分分析（用于降维）
from sklearn.decomposition import PCA
# statsmodels: 统计模型库
# - variance_inflation_factor: 计算方差膨胀因子（VIF），用于检测多重共线性
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==================== 配置日志系统 ====================
# 配置日志记录器，用于记录程序运行过程中的信息、警告和错误
# level=logging.INFO: 只记录INFO级别及以上的日志（INFO, WARNING, ERROR）
# format: 定义日志的格式，包括时间、模块名、级别和消息内容
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# 创建当前模块的日志记录器
logger = logging.getLogger(__name__)

# ==================== 配置中文字体 ====================
# 获取当前Python文件所在的目录路径
# __file__ 是当前Python文件的路径
# os.path.abspath() 获取绝对路径
# os.path.dirname() 获取目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建字体文件的完整路径（SimHei.ttf是黑体字体，用于显示中文）
font_path = os.path.join(current_dir, 'SimHei.ttf')

# 检查字体文件是否存在，如果存在则加载，否则使用系统默认字体
if os.path.exists(font_path):
    # 创建字体属性对象，指定字体文件路径
    font_prop = fm.FontProperties(fname=font_path)
    # 设置matplotlib的字体族为无衬线字体
    rcParams['font.family'] = 'sans-serif'
    # 将SimHei添加到字体列表的最前面（优先使用）
    rcParams['font.sans-serif'] = ['SimHei'] + rcParams['font.sans-serif']
    # 解决负号显示问题（默认情况下负号可能显示为方块）
    rcParams['axes.unicode_minus'] = False
    logger.info(f"已加载字体文件: {font_path}")
else:
    # 如果字体文件不存在，记录警告并使用系统默认字体
    logger.warning(f"警告: 未找到字体文件: {font_path}")
    # 使用系统默认字体属性
    font_prop = fm.FontProperties()
    # 设置字体列表，按优先级排序（SimHei > Microsoft YaHei > WenQuanYi Micro Hei > 系统默认）
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
    # 解决负号显示问题
    rcParams['axes.unicode_minus'] = False

class DataProcessor:
    """
    数据处理工具类
    
    这个类提供了一些静态方法（不需要创建对象就能使用的方法），
    用于处理数据的基本操作，比如验证数据、分类列类型、转换数据格式等。
    
    为什么需要这个类？
    - 确保输入的数据格式正确
    - 将复杂的数据类型转换为简单的格式（比如转换为JSON）
    - 对数据进行预处理，为后续分析做准备
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> None:
        """
        验证DataFrame的有效性
        
        这个方法会检查：
        1. 输入是否真的是pandas DataFrame类型
        2. DataFrame是否为空（没有数据）
        
        参数:
            df: 要验证的pandas DataFrame对象
            
        如果验证失败，会抛出异常（程序会报错并停止）
        """
        # isinstance() 用于检查对象的类型
        # 如果不是DataFrame类型，抛出类型错误
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        # empty 属性检查DataFrame是否为空（没有行或列）
        # 如果为空，抛出值错误
        if df.empty:
            raise ValueError("DataFrame不能为空")
    
    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        获取列类型信息
        
        这个方法会将DataFrame中的所有列按照数据类型分类：
        - 数值型列：可以计算平均值、标准差等的数字列（如年龄、价格）
        - 分类型列：文本或类别数据（如性别、城市名称）
        - 时间型列：日期时间数据（如出生日期、订单时间）
        
        参数:
            df: 要分析的pandas DataFrame
            
        返回:
            一个字典，包含三类列的列表
        """
        return {
            # select_dtypes(include=['number']) 选择所有数值类型的列
            # columns.tolist() 将列名转换为列表
            'numeric_cols': df.select_dtypes(include=['number']).columns.tolist(),
            # 选择文本和类别类型的列
            'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            # 选择日期时间类型的列
            'datetime_cols': df.select_dtypes(include=['datetime']).columns.tolist()
        }
    
    @staticmethod
    def process_value(v: Any) -> Any:
        """
        处理单个值，转换为可序列化格式
        
        什么是"可序列化"？
        - 序列化就是将数据转换为可以存储或传输的格式（如JSON）
        - 有些Python对象（如numpy数组）不能直接转换为JSON
        - 这个方法会将它们转换为JSON可以接受的格式
        
        参数:
            v: 要处理的单个值（可能是各种类型）
            
        返回:
            转换后的值（Python基本类型，可以转换为JSON）
        """
        # 如果是numpy数组，转换为Python列表
        if isinstance(v, np.ndarray):
            return v.tolist()
        # 如果是numpy的整数类型，转换为Python的int类型
        if isinstance(v, (np.int64, np.int32)):
            return int(v)
        # 如果是numpy的浮点数类型
        if isinstance(v, (np.float64, np.float32)):
            # 检查是否为NaN（不是数字）或Inf（无穷大）
            # 这些值在JSON中无法表示，所以转换为None
            if np.isnan(v) or np.isinf(v):
                return None
            # 否则转换为Python的float类型
            return float(v)
        # 如果是日期时间对象，转换为ISO格式字符串（如 "2023-01-01T12:00:00"）
        if isinstance(v, datetime):
            return v.isoformat()
        # 如果是pandas Series（一维数据），转换为字典
        if isinstance(v, pd.Series):
            return v.to_dict()
        # 如果是pandas DataFrame（二维数据），转换为字典列表
        if isinstance(v, pd.DataFrame):
            return v.to_dict('records')
        # 尝试检查是否为NaN值
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            # 如果检查失败（比如v不是可以检查NaN的类型），就忽略
            pass
        # 如果都不匹配，直接返回原值
        return v
    
    @staticmethod
    def process_data(obj: Any) -> Any:
        """
        递归处理数据，转换为可序列化格式
        
        这个方法会递归地处理复杂的数据结构（字典、列表等），
        将其中的所有值都转换为可序列化的格式。
        
        什么是递归？
        - 递归就是函数调用自己
        - 这里用于处理嵌套的数据结构（比如字典里包含列表，列表里包含字典）
        
        参数:
            obj: 要处理的数据（可能是字典、列表、数组或其他类型）
            
        返回:
            转换后的数据
        """
        # 如果是字典，递归处理每个值
        if isinstance(obj, dict):
            # 字典推导式：遍历字典的每个键值对，递归处理值
            return {k: DataProcessor.process_data(v) for k, v in obj.items()}
        # 如果是列表或numpy数组，递归处理每个元素
        elif isinstance(obj, (list, np.ndarray)):
            # 列表推导式：遍历列表的每个元素，递归处理
            return [DataProcessor.process_data(item) for item in obj]
        else:
            # 如果是基本类型，直接处理
            return DataProcessor.process_value(obj)
    
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
        """
        将DataFrame分块处理
        
        为什么要分块？
        - 当数据量很大时，一次性处理可能会导致内存不足
        - 分块处理可以分批处理数据，减少内存占用
        
        参数:
            df: 要分块的DataFrame
            chunk_size: 每块的大小（行数），默认1000行
            
        返回:
            一个列表，包含多个小的DataFrame
        """
        # 列表推导式：从第0行开始，每隔chunk_size行取一块
        # range(0, len(df), chunk_size) 生成起始索引：0, 1000, 2000, ...
        # df[i:i + chunk_size] 取从第i行到第i+chunk_size行的数据
        return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

class DataAnalyzer:
    """
    数据分析器主类
    
    这是整个模块的核心类，提供了各种数据分析功能。
    
    主要功能包括：
    1. 基础统计分析：计算均值、标准差、最大值、最小值等
    2. 数据质量分析：检测缺失值、异常值、重复值
    3. 高级统计分析：正态性检验、相关性分析、主成分分析
    4. 数据可视化：生成各种图表
    5. 报告生成：生成HTML格式的分析报告
    
    使用方法：
        # 创建分析器对象
        analyzer = DataAnalyzer(df)
        # 生成报告
        analyzer.generate_report("report.html")
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化数据分析器
        
        参数:
            df: 要分析的pandas DataFrame对象
            
        初始化过程：
        1. 验证数据有效性
        2. 复制数据（避免修改原始数据）
        3. 识别列类型（数值型、分类型、时间型）
        4. 初始化可视化工具
        5. 初始化缓存（用于存储计算结果，提高效率）
        """
        # 验证输入数据是否有效（必须是DataFrame且不为空）
        DataProcessor.validate_dataframe(df)
        # 复制数据，避免修改原始数据
        # copy() 创建数据的副本，这样对self.df的修改不会影响原始的df
        self.df = df.copy()
        
        # 获取列类型信息，将列分为三类
        column_types = DataProcessor.get_column_types(self.df)
        # 保存数值型列的列表（可以计算统计量的列）
        self.numeric_cols = column_types['numeric_cols']
        # 保存分类型列的列表（文本、类别数据）
        self.categorical_cols = column_types['categorical_cols']
        # 保存时间型列的列表（日期时间数据）
        self.datetime_cols = column_types['datetime_cols']
        
        # 初始化可视化工具，传入字体属性（用于中文显示）
        self.visualizer = Visualizer(font_prop)
        # 保存字体属性，供其他方法使用
        self.font_prop = font_prop
        
        # 初始化缓存变量（用于存储计算结果，避免重复计算）
        # 以下划线开头表示这是内部使用的变量（私有变量）
        self._correlation_cache = None  # 相关性矩阵缓存
        self._multicollinearity_cache = None  # 多重共线性分析缓存
        self._memory_usage = None  # 内存使用量缓存
    
    def __del__(self):
        """
        析构函数：当对象被销毁时自动调用
        
        用于清理资源，比如关闭图表、释放内存等。
        这样可以避免内存泄漏（内存没有被正确释放）。
        """
        # hasattr() 检查对象是否有某个属性
        # 如果visualizer存在，清除其缓存
        if hasattr(self, 'visualizer'):
            self.visualizer.clear_cache()
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        相关性分析
        
        什么是相关性？
        - 相关性衡量两个变量之间的线性关系强度
        - 相关系数范围：-1 到 1
        - 接近1：强正相关（一个增加，另一个也增加）
        - 接近-1：强负相关（一个增加，另一个减少）
        - 接近0：无相关性（两个变量独立）
        
        返回:
            一个DataFrame，包含所有数值型变量之间的相关系数矩阵
        """
        # 如果缓存为空，计算相关性矩阵
        # 使用缓存可以避免重复计算，提高效率
        if self._correlation_cache is None:
            # corr() 计算相关系数矩阵
            # 只对数值型列进行计算（分类型列无法计算相关性）
            self._correlation_cache = self.df[self.numeric_cols].corr()
        # 返回缓存的结果
        return self._correlation_cache
    
    def multicollinearity_analysis(self) -> pd.DataFrame:
        """
        多重共线性分析
        
        什么是多重共线性？
        - 多重共线性是指多个自变量（特征）之间存在高度相关性
        - 这会导致回归分析时系数不稳定、难以解释
        - 例如：如果"身高"和"体重"高度相关，在预测"健康评分"时，很难区分它们各自的影响
        
        什么是VIF（方差膨胀因子）？
        - VIF用于衡量多重共线性的严重程度
        - VIF < 2：无显著多重共线性（很好）
        - 2 ≤ VIF < 5：轻微多重共线性（可接受）
        - 5 ≤ VIF < 10：中等多重共线性（需要注意）
        - VIF ≥ 10：严重多重共线性（需要处理，比如删除某些变量）
        
        返回:
            一个DataFrame，包含每列的VIF值和解释
        """
        # 如果缓存为空，进行计算
        if self._multicollinearity_cache is None:
            # 只使用数值型列（VIF只能计算数值型变量）
            numeric_df = self.df[self.numeric_cols].copy()
            
            # 处理无穷大和NaN值
            # 无穷大值（inf）会导致计算错误，需要先处理
            # replace() 将无穷大值替换为NaN
            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
            # fillna() 用均值填充NaN值（缺失值）
            # mean() 计算每列的平均值
            numeric_df = numeric_df.fillna(numeric_df.mean())
            
            # 标准化数据（将数据转换为均值为0、标准差为1的分布）
            # 为什么要标准化？
            # - VIF计算需要标准化后的数据
            # - 标准化可以消除不同量纲的影响（比如身高用米，体重用公斤）
            scaler = StandardScaler()
            # fit_transform() 先学习数据的均值和标准差，然后进行转换
            scaled_data = scaler.fit_transform(numeric_df)
            
            # 计算VIF（方差膨胀因子）
            # 创建一个空的DataFrame来存储结果
            vif_data = pd.DataFrame()
            # 添加列名
            vif_data["列名"] = self.numeric_cols
            # 计算每列的VIF值
            # variance_inflation_factor() 计算VIF
            # 参数：scaled_data（标准化后的数据），i（列的索引）
            # 列表推导式：对每列都计算一次VIF
            vif_data["VIF"] = [variance_inflation_factor(scaled_data, i) 
                              for i in range(len(self.numeric_cols))]
            
            # 添加解释（根据VIF值判断多重共线性的严重程度）
            # apply() 对每行的VIF值应用一个函数
            # lambda 是匿名函数（没有名字的函数），用于简单的转换
            vif_data["解释"] = vif_data["VIF"].apply(
                lambda x: "严重多重共线性" if x > 10 else  # 如果VIF > 10
                         "中等多重共线性" if x > 5 else      # 如果VIF > 5
                         "轻微多重共线性" if x > 2 else      # 如果VIF > 2
                         "无显著多重共线性"                  # 否则
            )
            
            # 保存到缓存
            self._multicollinearity_cache = vif_data
        # 返回缓存的结果
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
        """
        生成分析报告
        
        这个方法会：
        1. 执行所有分析（统计、质量、可视化等）
        2. 收集所有分析结果
        3. 使用HTML模板生成美观的报告
        4. 保存为HTML文件，可以在浏览器中打开查看
        
        参数:
            output_html: 输出HTML文件的路径，默认为 "analysis_report.html"
            
        生成的报告包含：
        - 数据集基本信息
        - 基本统计量
        - 数据质量分析（缺失值、异常值、重复值）
        - 统计分析（正态性检验、相关性分析、PCA等）
        - 可视化图表（分布图、热图等）
        - 结果解读和处理建议
        """
        try:
            # ========== 步骤1：获取数据集名称 ==========
            # 从输出文件名中提取数据集名称（用于报告标题）
            try:
                # os.path.basename() 获取文件名（不含路径）
                # 例如："/path/to/report.html" -> "report.html"
                base_name = os.path.basename(output_html)
                # os.path.splitext() 分离文件名和扩展名
                # 例如："report.html" -> ("report", ".html")
                name_without_ext = os.path.splitext(base_name)[0]
                # 要移除的后缀（如果文件名包含这些，就删除）
                suffixes_to_remove = ['_analysis', '_report', '_data']
                dataset_name = name_without_ext
                # 遍历后缀列表，如果文件名以某个后缀结尾，就删除它
                for suffix in suffixes_to_remove:
                    if dataset_name.endswith(suffix):
                        # 删除后缀（从末尾删除指定长度的字符）
                        dataset_name = dataset_name[:-len(suffix)]
                # 将下划线替换为空格，并转换为标题格式（首字母大写）
                # 例如："iris_data" -> "Iris Data"
                dataset_name = dataset_name.replace('_', ' ').title()
            except Exception as e:
                # 如果处理失败，使用默认名称
                logger.error(f"处理数据集名称时出错: {str(e)}")
                dataset_name = "未命名数据集"
            
            # ========== 步骤2：收集所有分析结果 ==========
            logger.info("收集分析结果...")
            # 创建一个字典，包含所有分析结果
            analysis_results = {
                "dataset_name": dataset_name,  # 数据集名称
                # to_dict('records') 将DataFrame转换为字典列表格式
                # 例如：[{"列名": "age", "mean": 30}, {"列名": "height", "mean": 170}]
                "basic_stats": self.basic_statistics().to_dict('records'),  # 基本统计量
                "normality_test": self.normality_test().to_dict('records'),  # 正态性检验
                "missing_analysis": self.missing_value_analysis().to_dict('records'),  # 缺失值分析
                "outlier_analysis": self.outlier_analysis().to_dict('records'),  # 异常值分析
                "duplicate_analysis": self.duplicate_analysis(),  # 重复值分析（已经是字典）
                "correlation_matrix": self.correlation_analysis().to_dict(),  # 相关性矩阵
                "multicollinearity": self.multicollinearity_analysis().to_dict('records'),  # 多重共线性
                "pca_analysis": self.pca_analysis(),  # PCA分析（已经是字典）
                # 可视化图表（Base64编码的图片）
                "plots": {
                    # 为每列生成分布图
                    # 字典推导式：遍历所有列，为每列生成分布图
                    "distribution": {col: self.plot_distribution(col) for col in self.df.columns},
                    # 生成相关性热图
                    "correlation": self.plot_correlation_heatmap()
                },
                # 结果解读（自动生成的文字说明和建议）
                "interpretations": self.interpret_results()
            }
            
            # ========== 步骤3：获取模板和静态文件目录 ==========
            # 获取包内模板目录的绝对路径
            # __file__ 是当前Python文件的路径
            # os.path.dirname(__file__) 获取包目录
            # os.path.join() 拼接路径
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
            static_dir = os.path.join(os.path.dirname(__file__), 'static')
            
            # ========== 步骤4：生成报告 ==========
            # 创建报告生成器对象
            report_generator = ReportGenerator(template_dir, static_dir)
            # 调用生成方法，传入分析结果和输出路径
            report_generator.generate_report(analysis_results, output_html)
            
        except Exception as e:
            # 如果出错，记录错误并重新抛出异常
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
        """
        正态性检验
        
        什么是正态分布？
        - 正态分布（也叫高斯分布）是一种常见的概率分布
        - 它的图形是钟形曲线，左右对称
        - 很多统计方法（如t检验、方差分析）都要求数据符合正态分布
        
        为什么要进行正态性检验？
        - 如果数据不符合正态分布，需要使用非参数检验方法
        - 参数检验（如t检验）要求数据正态分布，否则结果可能不准确
        
        检验方法：
        1. Shapiro-Wilk检验：适合小样本（n < 50）
        2. D'Agostino-Pearson检验：适合大样本
        
        如何判断？
        - p值 > 0.05：数据符合正态分布（接受原假设）
        - p值 ≤ 0.05：数据不符合正态分布（拒绝原假设）
        
        返回:
            一个DataFrame，包含每列的正态性检验结果
        """
        # 存储所有列的检验结果
        results = []
        # 遍历每个数值型列
        for col in self.numeric_cols:
            # 删除缺失值（NaN值会影响检验结果）
            data = self.df[col].dropna()
            # 至少需要3个样本才能进行正态性检验
            if len(data) > 3:
                # Shapiro-Wilk检验
                # 返回两个值：统计量（statistic）和p值（p-value）
                # p值：如果p > 0.05，说明数据符合正态分布
                shapiro_stat, shapiro_p = shapiro(data)
                # D'Agostino-Pearson检验（另一种正态性检验方法）
                # 使用两种方法可以互相验证，提高准确性
                norm_stat, norm_p = normaltest(data)
                
                # 将结果添加到列表中
                results.append({
                    '列名': col,  # 列名
                    'Shapiro-Wilk统计量': shapiro_stat,  # 统计量（用于计算p值）
                    'Shapiro-Wilk p值': shapiro_p,  # p值（判断依据）
                    'D-Agostino-Pearson统计量': norm_stat,  # 另一种方法的统计量
                    'D-Agostino-Pearson p值': norm_p,  # 另一种方法的p值
                    # 如果两种方法的p值都 > 0.05，则认为数据符合正态分布
                    '是否正态分布': '是' if shapiro_p > 0.05 and norm_p > 0.05 else '否'
                })
        # 将结果列表转换为DataFrame并返回
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
        """
        异常值分析
        
        什么是异常值？
        - 异常值（也叫离群值）是指明显偏离其他数据点的值
        - 例如：在年龄数据中，如果大部分人是20-60岁，但有一个200岁，那就是异常值
        - 异常值可能是数据录入错误、测量错误，也可能是真实但罕见的情况
        
        为什么要检测异常值？
        - 异常值会影响统计分析的结果（如平均值、标准差）
        - 异常值可能导致模型预测不准确
        - 需要判断异常值是错误还是真实情况
        
        检测方法：
        1. Z-score方法：基于标准差，适合数据接近正态分布的情况
        2. IQR方法：基于四分位距，适合任何分布的数据
        3. 箱线图方法：基于IQR的扩展版本，更严格
        
        返回:
            一个DataFrame，包含每列的异常值检测结果
        """
        # 存储所有列的检测结果
        results = []
        # 遍历每个数值型列
        for col in self.numeric_cols:
            # 删除缺失值
            data = self.df[col].dropna()
            
            # ========== 方法1：Z-score方法 ==========
            # Z-score（标准分数）表示一个值距离平均值有多少个标准差
            # 公式：Z = (x - 均值) / 标准差
            # 如果 |Z| > 3，通常认为是异常值（距离均值超过3个标准差）
            z_scores = zscore(data)
            # 统计异常值数量：|Z| > 3 的数量
            # np.abs() 取绝对值，np.sum() 统计True的数量
            outliers_zscore = np.sum(np.abs(z_scores) > 3)
            # 获取详细的解释说明
            zscore_explanation = self._get_zscore_interpretation(z_scores)
            
            # ========== 方法2：IQR方法（四分位距方法）==========
            # IQR（四分位距）是Q3和Q1的差值，用于衡量数据的离散程度
            # Q1（第一四分位数）：25%的数据小于它
            # Q3（第三四分位数）：75%的数据小于它
            # IQR = Q3 - Q1
            Q1 = data.quantile(0.25)  # 25%分位数
            Q3 = data.quantile(0.75)  # 75%分位数
            IQR = Q3 - Q1  # 四分位距
            # 异常值边界：超出Q1-1.5*IQR或Q3+1.5*IQR的值被认为是异常值
            # 这是箱线图的标准定义
            lower_bound = Q1 - 1.5 * IQR  # 下边界
            upper_bound = Q3 + 1.5 * IQR  # 上边界
            # 统计异常值数量：小于下边界或大于上边界的数量
            # | 表示"或"（逻辑或运算）
            outliers_iqr = np.sum((data < lower_bound) | (data > upper_bound))
            # 获取详细的解释说明
            iqr_explanation = self._get_iqr_interpretation(data, Q1, Q3, IQR)
            
            # ========== 方法3：箱线图方法（更严格）==========
            # 使用3倍IQR作为边界（比标准箱线图更严格）
            # 这样可以检测更极端的异常值
            outliers_box = np.sum((data < Q1 - 3 * IQR) | (data > Q3 + 3 * IQR))
            
            # 将结果添加到列表
            results.append({
                "列名": col,
                "Z-score异常值数量": outliers_zscore,
                "Z-score解释": zscore_explanation,  # 详细的文字说明
                "IQR异常值数量": outliers_iqr,
                "IQR解释": iqr_explanation,  # 详细的文字说明
                "箱线图异常值数量": outliers_box,
                # 异常值比例：取三种方法中检测到的最大数量，计算比例
                # max() 取最大值，len(data) 是总数据量
                "异常值比例 (%)": (max(outliers_zscore, outliers_iqr, outliers_box) / len(data) * 100).round(2)
            })
        
        # 将结果列表转换为DataFrame并返回
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
        
        # 计算中位数（用于解释数据分布特征）
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
        """
        主成分分析（PCA - Principal Component Analysis）
        
        什么是PCA？
        - PCA是一种降维技术，可以将多个相关变量转换为少数几个不相关的主成分
        - 主成分是原始变量的线性组合，能够保留大部分原始信息
        - 例如：如果有10个高度相关的变量，PCA可能只需要2-3个主成分就能解释大部分信息
        
        为什么要使用PCA？
        1. 降维：减少变量数量，简化模型
        2. 消除多重共线性：主成分之间不相关
        3. 可视化：高维数据难以可视化，降维后可以画图
        4. 特征提取：主成分可能比原始变量更有意义
        
        如何理解结果？
        - 方差贡献率：每个主成分能解释多少原始数据的方差
        - 累计方差贡献率：前N个主成分总共能解释多少方差
        - 通常选择能解释95%方差的主成分数量
        
        返回:
            一个字典，包含PCA分析的所有结果
        """
        # 只使用数值型列（PCA只能处理数值数据）
        numeric_df = self.df[self.numeric_cols].copy()
        
        # 处理无穷大值（会导致计算错误）
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        
        # 检查是否有缺失值
        # isna() 检查是否为NaN，any() 检查是否有任何True值
        if numeric_df.isna().any().any():
            logger.warning("数据中存在缺失值，将使用均值填充")
            # 用每列的均值填充缺失值
            numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # 标准化数据（PCA要求数据标准化）
        # 标准化可以消除不同量纲的影响
        scaler = StandardScaler()
        # fit_transform() 先学习参数，然后转换数据
        scaled_data = scaler.fit_transform(numeric_df)
        
        # 执行PCA
        # PCA() 创建PCA对象，不指定n_components会保留所有主成分
        pca = PCA()
        # fit() 学习主成分（计算主成分的方向）
        pca.fit(scaled_data)
        
        # 计算累计方差贡献率
        # explained_variance_ratio_ 是每个主成分的方差贡献率
        # np.cumsum() 计算累计和（第1个、前2个、前3个...的累计贡献率）
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # 计算每个主成分的方差贡献率
        # 例如：[0.5, 0.3, 0.2] 表示PC1解释50%方差，PC2解释30%，PC3解释20%
        variance_ratio = pca.explained_variance_ratio_
        
        # 计算特征向量（主成分的系数）
        # components_ 是主成分的系数矩阵，表示每个原始变量对主成分的贡献
        # 例如：PC1 = 0.5*变量1 + 0.3*变量2 + 0.2*变量3
        components = pd.DataFrame(
            pca.components_,  # 主成分系数矩阵
            columns=self.numeric_cols,  # 列名是原始变量名
            index=[f'PC{i+1}' for i in range(len(self.numeric_cols))]  # 行名是主成分名（PC1, PC2, ...）
        )
        
        # 计算主成分得分
        # transform() 将原始数据转换为主成分空间
        # 得到每个样本在主成分上的得分
        scores = pd.DataFrame(
            pca.transform(scaled_data),  # 转换后的数据
            columns=[f'PC{i+1}' for i in range(len(self.numeric_cols))]  # 列名是主成分名
        )
        
        # 确定主成分数量（解释95%方差所需的主成分数）
        # np.argmax() 返回第一个满足条件的索引
        # cumulative_variance >= 0.95 返回布尔数组，True表示累计贡献率>=95%
        # +1 因为索引从0开始，但主成分编号从1开始
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
        
        # ========== 可视化结果 ==========
        # 创建图表，大小12x6英寸
        plt.figure(figsize=(12, 6))
        
        # 绘制碎石图（Scree Plot）
        # 碎石图显示累计方差贡献率随主成分数量的变化
        plt.subplot(1, 2, 1)  # 1行2列的第1个子图
        # 绘制累计方差贡献率曲线
        plt.plot(range(1, len(variance_ratio) + 1), 
                cumulative_variance, 'bo-')  # 'bo-' 表示蓝色圆点连线
        # 画一条水平参考线（95%阈值）
        plt.axhline(y=0.95, color='r', linestyle='--')  # 红色虚线
        plt.xlabel('主成分数量')
        plt.ylabel('累计方差贡献率')
        plt.title('碎石图')
        
        # 绘制主成分方差贡献率柱状图
        plt.subplot(1, 2, 2)  # 1行2列的第2个子图
        # 绘制每个主成分的方差贡献率
        plt.bar(range(1, len(variance_ratio) + 1), 
                variance_ratio)
        plt.xlabel('主成分')
        plt.ylabel('方差贡献率')
        plt.title('主成分方差贡献率')
        
        # 调整子图间距，避免重叠
        plt.tight_layout()
        # 显示图表
        plt.show()
        
        # 返回所有结果
        return {
            'n_components': n_components,  # 建议保留的主成分数量
            'variance_ratio': variance_ratio,  # 每个主成分的方差贡献率
            'cumulative_variance': cumulative_variance,  # 累计方差贡献率
            'components': components,  # 主成分系数矩阵
            'scores': scores,  # 主成分得分
            'feature_names': self.numeric_cols  # 原始变量名
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
            "总体情况": f"主成分分析结果显示，数据集中有 {pca_stats['n_components']} 个主成分。",
            "建议保留主成分数": f"建议保留 {pca_stats['n_components']} 个主成分，可以解释 {pca_stats['cumulative_variance'][pca_stats['n_components']-1]*100:.2f}% 的方差。",
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
    """
    数据可视化工具类
    
    这个类负责生成各种数据可视化图表。
    
    主要功能：
    1. 绘制数据分布图（直方图、条形图）
    2. 绘制相关性热图
    3. 将图表转换为Base64编码（用于在HTML中嵌入）
    
    为什么需要Base64编码？
    - HTML报告需要包含图片
    - Base64编码可以将图片转换为文本格式
    - 这样就不需要单独保存图片文件，HTML文件可以独立使用
    """
    
    def __init__(self, font_prop: fm.FontProperties):
        """
        初始化可视化工具
        
        参数:
            font_prop: 字体属性对象（用于中文显示）
        """
        # 保存字体属性，用于图表中的中文显示
        self.font_prop = font_prop
        # 图表缓存（用于管理创建的图表对象）
        self._figure_cache = {}
    
    def _create_figure(self, figsize: tuple = (10, 6)) -> plt.Figure:
        """
        创建新的图表
        
        参数:
            figsize: 图表大小（宽度, 高度），单位是英寸，默认(10, 6)
            
        返回:
            matplotlib的Figure对象
        """
        return plt.figure(figsize=figsize)
    
    def _close_figure(self, fig: plt.Figure) -> None:
        """
        关闭图表
        
        为什么要关闭图表？
        - 释放内存，避免内存泄漏
        - 图表已经保存为图片，不再需要Figure对象
        
        参数:
            fig: 要关闭的图表对象
        """
        plt.close(fig)
    
    def plot_distribution(self, data: pd.Series, column: str) -> str:
        """
        绘制分布图
        
        根据数据类型选择不同的图表：
        - 数值型数据：绘制直方图（histogram）+ 密度曲线（KDE）
        - 分类型数据：绘制条形图（count plot）
        
        参数:
            data: 要绘制的数据（pandas Series）
            column: 列名（用于图表标题）
            
        返回:
            Base64编码的图片字符串（可以直接嵌入HTML）
        """
        try:
            # 创建新的图表
            fig = self._create_figure()
            # 判断数据类型
            if pd.api.types.is_numeric_dtype(data):
                # ========== 数值型数据：绘制直方图 ==========
                # histplot() 绘制直方图
                # kde=True 同时绘制密度曲线（Kernel Density Estimation）
                sns.histplot(data, kde=True)
                plt.title(f"{column} 的分布", fontproperties=self.font_prop)
                plt.xlabel(column, fontproperties=self.font_prop)
                plt.ylabel('频数', fontproperties=self.font_prop)
            else:
                # ========== 分类型数据：绘制条形图 ==========
                # countplot() 绘制计数条形图（统计每个类别的数量）
                sns.countplot(x=data)
                plt.title(f"{column} 的频数分布", fontproperties=self.font_prop)
                plt.xlabel(column, fontproperties=self.font_prop)
                plt.ylabel('频数', fontproperties=self.font_prop)
                # rotation=45 将x轴标签旋转45度（避免重叠）
                plt.xticks(rotation=45, fontproperties=self.font_prop)
                plt.yticks(fontproperties=self.font_prop)
            
            # ========== 将图表转换为Base64编码 ==========
            # BytesIO() 创建一个内存中的二进制流（类似文件对象）
            buf = BytesIO()
            # savefig() 将图表保存到流中
            # format='png' 保存为PNG格式
            # bbox_inches='tight' 自动调整边界，去除多余空白
            # dpi=100 分辨率（每英寸100点）
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            # seek(0) 将流的位置重置到开头（准备读取）
            buf.seek(0)
            # 读取二进制数据，转换为Base64编码，再解码为字符串
            # b64encode() 编码，decode('utf-8') 转换为字符串
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            # 关闭图表，释放内存
            self._close_figure(fig)
            # 返回Base64编码的字符串
            return img_base64
        except Exception as e:
            # 如果出错，记录错误并关闭图表
            logger.error(f"绘制分布图时出错: {str(e)}")
            self._close_figure(fig)
            raise
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> str:
        """
        绘制相关性热图
        
        热图（Heatmap）用颜色深浅表示数值大小：
        - 红色：正相关（相关系数接近1）
        - 蓝色：负相关（相关系数接近-1）
        - 白色：无相关（相关系数接近0）
        
        参数:
            corr_matrix: 相关系数矩阵（DataFrame）
            
        返回:
            Base64编码的图片字符串
        """
        try:
            # 创建较大的图表（12x8英寸），因为热图需要更多空间
            fig = self._create_figure((12, 8))
            # heatmap() 绘制热图
            # annot=True 在单元格中显示数值
            # cmap='coolwarm' 使用冷暖色配色方案（蓝-白-红）
            # fmt='.2f' 数值格式：保留2位小数
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('相关性热图', fontproperties=self.font_prop)
            plt.xticks(fontproperties=self.font_prop)
            plt.yticks(fontproperties=self.font_prop)
            
            # 转换为Base64编码（与plot_distribution()相同）
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
        """
        清除图表缓存
        
        用于释放内存，通常在对象销毁时调用。
        """
        # 遍历缓存中的所有图表，逐个关闭
        for fig in self._figure_cache.values():
            self._close_figure(fig)
        # 清空缓存字典
        self._figure_cache.clear()

# ==================== 工具函数 ====================
# 这些函数提供了便捷的使用方式，可以直接分析文件而不需要手动创建对象

def load_data(file_path: str) -> pd.DataFrame:
    """
    加载数据文件
    
    支持的文件格式：
    - CSV文件（.csv）：逗号分隔值文件，最常见的数据格式
    - Excel文件（.xls, .xlsx）：Microsoft Excel文件
    - JSON文件（.json）：JavaScript对象表示法格式
    
    参数:
        file_path: 数据文件的路径
        
    返回:
        加载后的pandas DataFrame对象
        
    示例:
        df = load_data("data.csv")
    """
    # 根据文件扩展名判断文件类型，使用相应的pandas函数加载
    if file_path.endswith('.csv'):
        # 读取CSV文件
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        # 读取Excel文件
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        # 读取JSON文件
        df = pd.read_json(file_path)
    else:
        # 如果不支持的文件格式，抛出错误
        raise ValueError("不支持的文件格式")
    
    return df

def analyze_data(file_path: str, output_html: str = "analysis_report.html") -> None:
    """
    分析数据并生成报告（一键式函数）
    
    这个函数封装了完整的分析流程：
    1. 加载数据文件
    2. 创建分析器
    3. 生成HTML报告
    
    参数:
        file_path: 要分析的数据文件路径
        output_html: 输出HTML报告的路径，默认为 "analysis_report.html"
        
    示例:
        analyze_data("data.csv", "report.html")
    """
    # 步骤1：加载数据
    df = load_data(file_path)
    # 步骤2：创建分析器对象
    analyzer = DataAnalyzer(df)
    # 步骤3：生成报告
    analyzer.generate_report(output_html)

def analyze_multiple_files(folder_path: str, output_dir: str = "reports") -> None:
    """
    批量分析文件夹中的所有数据文件
    
    这个函数会：
    1. 遍历指定文件夹中的所有文件
    2. 对每个支持的数据文件进行分析
    3. 将报告保存到指定目录
    
    参数:
        folder_path: 包含数据文件的文件夹路径
        output_dir: 输出报告的目录，默认为 "reports"
        
    示例:
        analyze_multiple_files("data_folder", "reports")
    """
    # 创建输出目录（如果不存在）
    # exist_ok=True 表示如果目录已存在也不报错
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        # 检查文件扩展名，只处理支持的数据文件
        if file.endswith(('.csv', '.xls', '.xlsx', '.json')):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, file)
            # 构建输出HTML文件路径
            # os.path.splitext(file)[0] 获取文件名（不含扩展名）
            # 例如："data.csv" -> "data"
            output_html = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_report.html")
            try:
                # 分析文件并生成报告
                analyze_data(file_path, output_html)
                print(f"已分析文件: {file}")
            except Exception as e:
                # 如果分析失败，打印错误信息但继续处理其他文件
                print(f"分析文件 {file} 时出错: {str(e)}")

def analyze_notebook(df: pd.DataFrame) -> None:
    """
    在Jupyter Notebook中分析数据并显示报告
    
    这个函数专门用于Jupyter Notebook环境，可以直接显示分析结果。
    
    参数:
        df: pandas DataFrame，要分析的数据集
        
    示例（在Jupyter Notebook中）:
        import pandas as pd
        df = pd.read_csv("data.csv")
        analyze_notebook(df)
    """
    try:
        # 创建分析器
        analyzer = DataAnalyzer(df)
        # 调用notebook专用的分析方法（如果存在）
        # 注意：这个方法在代码中可能没有实现，这里只是示例
        analyzer.analyze_notebook()
    except Exception as e:
        print(f"分析数据时出错: {str(e)}")

class ReportGenerator:
    """
    报告生成工具类
    
    这个类负责将分析结果转换为HTML格式的报告。
    
    工作流程：
    1. 加载HTML模板（使用Jinja2模板引擎）
    2. 将分析结果转换为JSON格式
    3. 将数据填充到模板中（渲染模板）
    4. 复制静态文件（JavaScript、CSS等）
    5. 保存最终的HTML文件
    
    什么是模板引擎？
    - 模板是一个HTML文件，但包含一些占位符（如 {{变量名}}）
    - 模板引擎会用实际数据替换这些占位符
    - 这样可以生成动态的HTML内容
    """
    
    def __init__(self, template_dir: str, static_dir: str):
        """
        初始化报告生成器
        
        参数:
            template_dir: HTML模板文件所在的目录
            static_dir: 静态文件（JS、CSS）所在的目录
        """
        # 保存目录路径
        self.template_dir = template_dir
        self.static_dir = static_dir
        # 创建Jinja2环境，用于加载和渲染模板
        # FileSystemLoader 从文件系统加载模板
        self.env = Environment(loader=FileSystemLoader(template_dir))
    
    def _copy_static_files(self) -> None:
        """
        复制静态文件到输出目录
        
        静态文件包括：
        - JavaScript文件（.js）：用于报告的交互功能
        - CSS文件（.css）：用于样式（如果有）
        
        为什么要复制？
        - HTML报告需要引用这些文件
        - 复制到输出目录可以确保报告可以独立使用
        """
        import shutil  # 用于文件复制操作
        
        # 构建源文件路径（包内的JavaScript文件）
        source_js = os.path.join(os.path.dirname(__file__), 'static', 'js', 'analyzer.js')
        # 构建目标文件路径（输出目录中的JavaScript文件）
        target_js = os.path.join(self.static_dir, 'js', 'analyzer.js')
        
        # 确保目标目录存在（如果不存在则创建）
        os.makedirs(os.path.dirname(target_js), exist_ok=True)
        
        # 复制文件
        try:
            if os.path.exists(source_js):
                # 如果目标文件已存在
                if os.path.exists(target_js):
                    # 检查文件是否相同（避免不必要的复制）
                    if not self._files_are_identical(source_js, target_js):
                        # 如果不同，则更新
                        shutil.copy2(source_js, target_js)
                        logger.info(f"更新了JavaScript文件: {target_js}")
                else:
                    # 如果目标文件不存在，直接复制
                    shutil.copy2(source_js, target_js)
                    logger.info(f"复制了JavaScript文件: {target_js}")
            else:
                # 如果源文件不存在，记录警告
                logger.warning(f"警告: 源文件不存在: {source_js}")
        except Exception as e:
            logger.error(f"复制静态文件时出错: {str(e)}")
    
    @staticmethod
    def _files_are_identical(file1: str, file2: str) -> bool:
        """
        比较两个文件是否相同
        
        通过比较文件内容来判断文件是否相同。
        这样可以避免重复复制相同的文件。
        
        参数:
            file1: 第一个文件路径
            file2: 第二个文件路径
            
        返回:
            True表示文件相同，False表示不同
        """
        try:
            # 以二进制模式打开两个文件
            # 'rb' 表示只读二进制模式
            with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
                # 读取文件内容并比较
                return f1.read() == f2.read()
        except Exception:
            # 如果读取失败（文件不存在等），返回False
            return False
    
    def generate_report(self, analysis_results: Dict[str, Any], output_html: str) -> None:
        """
        生成分析报告
        
        这是报告生成的主要方法，执行完整的报告生成流程。
        
        参数:
            analysis_results: 包含所有分析结果的字典
            output_html: 输出HTML文件的路径
            
        生成的报告特点：
        - 包含所有分析结果和可视化图表
        - 美观的HTML格式，可以在浏览器中打开
        - 包含交互功能（如果模板支持）
        - 可以独立使用（不依赖服务器）
        """
        try:
            logger.info("开始生成报告...")
            
            # ========== 步骤1：准备静态文件 ==========
            # 确保静态文件目录存在
            os.makedirs(self.static_dir, exist_ok=True)
            os.makedirs(os.path.join(self.static_dir, 'js'), exist_ok=True)
            # 复制静态文件（JavaScript等）
            self._copy_static_files()
            
            # ========== 步骤2：加载HTML模板 ==========
            # 从模板目录加载模板文件
            template = self.env.get_template('analyzer.html')
            logger.info("模板加载成功")
            
            # ========== 步骤3：计算路径 ==========
            # 获取输出文件的目录（用于计算静态文件的相对路径）
            output_dir = os.path.dirname(os.path.abspath(output_html))
            # 如果没有目录（只有文件名），使用当前工作目录
            if not output_dir:
                output_dir = os.getcwd()
            logger.info(f"输出目录: {output_dir}")
            
            # 计算静态文件的相对路径（用于HTML中的引用）
            # os.path.relpath() 计算相对路径
            # 例如：如果静态文件在 "../static"，HTML中引用时使用 "../static/js/analyzer.js"
            static_url = os.path.relpath(self.static_dir, output_dir)
            if static_url.startswith('..'):
                static_url = os.path.join('..', static_url)
            logger.info(f"静态文件URL: {static_url}")
            
            # ========== 步骤4：处理数据 ==========
            # 将分析结果转换为可序列化的格式（处理numpy数组、日期等）
            processed_data = DataProcessor.process_data(analysis_results)
            # 转换为JSON字符串（用于在HTML中嵌入数据）
            # ensure_ascii=False 允许使用中文字符
            json_data = json.dumps(processed_data, ensure_ascii=False)
            
            # ========== 步骤5：渲染模板 ==========
            logger.info("开始渲染模板...")
            # render() 将数据填充到模板中，生成最终的HTML
            rendered_html = template.render(
                data=json_data,  # 分析结果数据（JSON格式）
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 当前时间戳
                static_url=static_url  # 静态文件的路径
            )
            logger.info("模板渲染完成")
            
            # ========== 步骤6：保存报告 ==========
            logger.info(f"保存报告到: {output_html}")
            # 以UTF-8编码写入HTML文件（支持中文）
            with open(output_html, 'w', encoding='utf-8') as f:
                f.write(rendered_html)
            logger.info(f"分析报告已保存至: {output_html}")
            
        except Exception as e:
            # 如果出错，记录错误并重新抛出异常
            logger.error(f"生成报告时出错: {str(e)}")
            raise
