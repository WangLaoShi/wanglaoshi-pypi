"""
机器学习和深度学习基础方法模块 (MLDL.py)

这个模块提供了完整的机器学习和深度学习工作流程的工具类，包括：

1. 【数据预处理】
   - 缺失值处理（均值、中位数、众数填充或删除）
   - 分类变量编码（标签编码）
   - 特征标准化（StandardScaler、MinMaxScaler）

2. 【特征工程】
   - 特征选择
   - 特征创建
   - 特征转换

3. 【基础机器学习模型】
   - 线性回归、逻辑回归
   - 决策树（分类和回归）
   - 随机森林（分类和回归）
   - 支持向量机（SVM）
   - K近邻（KNN）

4. 【深度学习模型】
   - 使用PyTorch构建神经网络
   - 自定义数据集类
   - 模型训练和评估

5. 【模型评估】
   - 分类指标：准确率、精确率、召回率、F1分数
   - 回归指标：均方误差、R²分数
   - 混淆矩阵

6. 【可视化工具】
   - 训练过程可视化
   - 结果可视化

使用场景：
- 快速构建ML/DL模型
- 学习ML/DL工作流程
- 原型开发和实验

主要类：
- DataPreprocessor: 数据预处理类
- FeatureEngineer: 特征工程类
- MLModel: 机器学习模型类
- DeepLearningModel: 深度学习模型类
"""

# ==================== 导入必要的库 ====================
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 绘图库
import seaborn as sns  # 数据可视化库

# sklearn: 机器学习库
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder  # 数据预处理
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # 模型选择和评估
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 分类指标
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix  # 回归指标和混淆矩阵

# sklearn模型
from sklearn.linear_model import LinearRegression, LogisticRegression  # 线性模型
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 决策树
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # 随机森林
from sklearn.svm import SVC, SVR  # 支持向量机
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # K近邻

# PyTorch: 深度学习框架
import torch  # PyTorch主模块
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器

class DataPreprocessor:
    """
    数据预处理类
    
    这个类提供了数据预处理的基础功能，包括：
    - 缺失值处理（多种填充方法）
    - 分类变量编码（标签编码）
    - 特征标准化（Z-score标准化）
    
    使用场景：
    - 在训练模型前对数据进行预处理
    - 统一数据格式和范围
    - 处理缺失值和分类变量
    
    示例:
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.handle_missing_values(df, method='mean')
        df_encoded = preprocessor.encode_categorical(df_processed, ['category_col'])
        df_scaled = preprocessor.scale_features(df_encoded, ['numeric_col1', 'numeric_col2'])
    """
    
    def __init__(self):
        """
        初始化预处理器
        
        创建标准化器和标签编码器对象，供后续使用。
        """
        # StandardScaler: Z-score标准化（均值为0，标准差为1）
        self.scaler = StandardScaler()
        # LabelEncoder: 将分类变量转换为数值标签（0, 1, 2, ...）
        self.label_encoder = LabelEncoder()
        
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """
        处理缺失值
        
        这个函数提供了多种缺失值处理方法：
        - 'mean': 用均值填充（适合数值型数据）
        - 'median': 用中位数填充（适合有异常值的数据）
        - 'mode': 用众数填充（适合分类型数据）
        - 'drop': 删除包含缺失值的行
        
        参数:
            df: 输入数据框（可能包含缺失值）
            method: 填充方法，可选 'mean', 'median', 'mode', 'drop'
                   默认 'mean'
            
        返回:
            处理后的数据框（无缺失值或已删除缺失值行）
            
        注意：
        - 使用copy()避免修改原始数据
        - 不同列可能适合不同的填充方法
        """
        # 复制数据，避免修改原始DataFrame
        df_processed = df.copy()
        
        if method == 'mean':
            # 用每列的均值填充缺失值
            df_processed = df_processed.fillna(df_processed.mean())
        elif method == 'median':
            # 用每列的中位数填充缺失值（对异常值更稳健）
            df_processed = df_processed.fillna(df_processed.median())
        elif method == 'mode':
            # 用每列的众数填充缺失值（mode()返回Series，取第一个值）
            df_processed = df_processed.fillna(df_processed.mode().iloc[0])
        elif method == 'drop':
            # 删除包含缺失值的行
            df_processed = df_processed.dropna()
            
        return df_processed
    
    def encode_categorical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        对分类变量进行编码（标签编码）
        
        标签编码将分类变量转换为数值标签：
        - 例如: ['A', 'B', 'C'] -> [0, 1, 2]
        
        参数:
            df: 输入数据框
            columns: 需要编码的列名列表
                    例如: ['category1', 'category2']
            
        返回:
            编码后的数据框（分类变量已转换为数值）
            
        注意：
        - 标签编码是有序的（0 < 1 < 2），可能不适合无序分类变量
        - 对于无序分类变量，建议使用独热编码（One-Hot Encoding）
        """
        # 复制数据，避免修改原始DataFrame
        df_encoded = df.copy()
        # 遍历需要编码的列
        for col in columns:
            # fit_transform() 先学习标签映射，然后转换
            # 例如: ['A', 'B', 'A'] -> [0, 1, 0]
            df_encoded[col] = self.label_encoder.fit_transform(df_encoded[col])
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        特征标准化（Z-score标准化）
        
        标准化将数据转换为均值为0、标准差为1的分布：
        - 公式: z = (x - mean) / std
        - 这样可以消除不同量纲的影响
        
        参数:
            df: 输入数据框
            columns: 需要标准化的列名列表
                    例如: ['feature1', 'feature2']
            
        返回:
            标准化后的数据框（指定列已标准化）
            
        使用场景：
        - 不同特征的单位和范围差异很大时
        - 使用基于距离的算法（如SVM、KNN）时
        - 使用梯度下降优化时
        """
        # 复制数据，避免修改原始DataFrame
        df_scaled = df.copy()
        # fit_transform() 先学习均值和标准差，然后转换
        # 只对指定的列进行标准化
        df_scaled[columns] = self.scaler.fit_transform(df_scaled[columns])
        return df_scaled

class FeatureEngineer:
    """特征工程类
    
    提供特征选择、特征创建、特征转换等功能
    """
    
    def __init__(self):
        """初始化特征工程器"""
        pass
        
    def create_polynomial_features(self, df: pd.DataFrame, columns: list, degree: int = 2) -> pd.DataFrame:
        """创建多项式特征
        
        参数:
            df: 输入数据框
            columns: 需要创建多项式特征的列名列表
            degree: 多项式次数
            
        返回:
            添加了多项式特征的数据框
        """
        df_poly = df.copy()
        for col in columns:
            for d in range(2, degree + 1):
                df_poly[f'{col}_power_{d}'] = df_poly[col] ** d
        return df_poly
    
    def create_interaction_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """创建交互特征
        
        参数:
            df: 输入数据框
            columns: 需要创建交互特征的列名列表
            
        返回:
            添加了交互特征的数据框
        """
        df_inter = df.copy()
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                df_inter[f'{col1}_{col2}_interaction'] = df_inter[col1] * df_inter[col2]
        return df_inter

class MLModel:
    """机器学习模型类
    
    提供常用机器学习模型的训练、预测和评估功能
    """
    
    def __init__(self, model_type: str):
        """初始化模型
        
        参数:
            model_type: 模型类型，可选 'linear', 'logistic', 'tree', 'forest', 'svm', 'knn'
        """
        self.model_type = model_type
        self.model = self._create_model()
        
    def _create_model(self):
        """创建模型实例"""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'logistic':
            return LogisticRegression()
        elif self.model_type == 'tree':
            return DecisionTreeClassifier()
        elif self.model_type == 'forest':
            return RandomForestClassifier()
        elif self.model_type == 'svm':
            return SVC()
        elif self.model_type == 'knn':
            return KNeighborsClassifier()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """训练模型
        
        参数:
            X: 特征数据
            y: 目标变量
            test_size: 测试集比例
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测
        
        参数:
            X: 特征数据
            
        返回:
            预测结果
        """
        return self.model.predict(X)
    
    def evaluate(self) -> dict:
        """评估模型
        
        返回:
            包含各项评估指标的字典
        """
        y_pred = self.predict(self.X_test)
        
        if self.model_type in ['linear']:
            metrics = {
                'MSE': mean_squared_error(self.y_test, y_pred),
                'R2': r2_score(self.y_test, y_pred)
            }
        else:
            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred, average='weighted'),
                'Recall': recall_score(self.y_test, y_pred, average='weighted'),
                'F1': f1_score(self.y_test, y_pred, average='weighted')
            }
            
        return metrics

class NeuralNetwork(nn.Module):
    """神经网络模型类
    
    提供基础神经网络模型的实现
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """初始化神经网络
        
        参数:
            input_size: 输入层大小
            hidden_size: 隐藏层大小
            output_size: 输出层大小
        """
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入数据
            
        返回:
            模型输出
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class DLModel:
    """深度学习模型类
    
    提供深度学习模型的训练、预测和评估功能
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """初始化深度学习模型
        
        参数:
            input_size: 输入层大小
            hidden_size: 隐藏层大小
            output_size: 输出层大小
        """
        self.model = NeuralNetwork(input_size, hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100):
        """训练模型
        
        参数:
            X: 特征数据
            y: 目标变量
            epochs: 训练轮数
        """
        for epoch in range(epochs):
            # 前向传播
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """预测
        
        参数:
            X: 特征数据
            
        返回:
            预测结果
        """
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

class ModelEvaluator:
    """模型评估类
    
    提供模型评估和可视化功能
    """
    
    def __init__(self):
        """初始化评估器"""
        pass
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """绘制混淆矩阵
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.show()
        
    def plot_learning_curve(self, train_sizes: np.ndarray, train_scores: np.ndarray, 
                          test_scores: np.ndarray):
        """绘制学习曲线
        
        参数:
            train_sizes: 训练集大小
            train_scores: 训练集得分
            test_scores: 测试集得分
        """
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='训练集得分')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='测试集得分')
        plt.xlabel('训练样本数')
        plt.ylabel('得分')
        plt.title('学习曲线')
        plt.legend()
        plt.show()

# 使用示例
def example_usage():
    """使用示例"""
    # 1. 数据预处理
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': ['a', 'b', 'a', 'c', 'b']
    })
    df_processed = preprocessor.handle_missing_values(df, method='mean')
    df_encoded = preprocessor.encode_categorical(df_processed, ['B'])
    
    # 2. 特征工程
    engineer = FeatureEngineer()
    df_features = engineer.create_polynomial_features(df_encoded, ['A'], degree=2)
    
    # 3. 机器学习模型
    ml_model = MLModel('logistic')
    X = df_features[['A', 'A_power_2']]
    y = df_features['B']
    ml_model.train(X, y)
    metrics = ml_model.evaluate()
    print("ML模型评估结果:", metrics)
    
    # 4. 深度学习模型
    dl_model = DLModel(input_size=2, hidden_size=4, output_size=3)
    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.LongTensor(y.values)
    dl_model.train(X_tensor, y_tensor, epochs=100)
    
    # 5. 模型评估
    evaluator = ModelEvaluator()
    y_pred = ml_model.predict(X)
    evaluator.plot_confusion_matrix(y, y_pred)

if __name__ == '__main__':
    example_usage()
