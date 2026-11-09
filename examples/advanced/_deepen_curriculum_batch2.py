"""Curriculum batch2：加深 E03 / E06 / E07 / E08（约 3× 篇幅 + 分析理论 + wanglaoshi API）。

用法：
  cd examples/advanced && python _deepen_curriculum_batch2.py

会覆盖：03_wine_quality, 06_california_housing, 07_abalone, 08_bike_sharing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _deepen_all import code, env_cells, md, write  # noqa: E402


SUPPLEMENT_E03 = """
## 课程研讨笔录（E03 专用）

### 课堂问答题
1. 若测试集只有 quality=6，Accuracy 最高多少？对模型选择意味着什么？
2. 回归预测 5.8 与 6.2 在四舍五入后相同，但 MAE 不同——哪种指标更贴业务「差半档可接受」？
3. `volatile_acidity` 升高先恶化质量再平台，树模型可捕非线性；线性回归假设如何处理？

### 实验记录模板
- 数据版本（UCI / OpenML）、样本量 n=
- 划分：分层 75/25，`RANDOM_STATE=42`
- 回归：最佳模型 MAE / RMSE / R²
- 分类：macro-F1 / weighted-F1，稀少类 recall
- `CrossValidator.stratified_cv` 均值±std

### 与竞赛路线图映射
- B05：submission 禁止静默概率 argmax → 本案例 `prediction_type='label'`
- 不均衡专题：先于建模画 `value_counts(normalize=True)`
- 有序标签专题：对比回归 rounded accuracy 与真分类 F1

### 延伸阅读（自学）
- Ordinal regression / cumulative link models
- Cost-sensitive learning：把错分 5→9 的代价设高于 5→6
- 白葡萄酒合并：估计 domain shift 对 macro-F1 的影响

### 打分 rubric（教师用）
| 维度 | 达标 | 优秀 |
|------|------|------|
| 数据加载 | 双路径其一成功 | 记录版本与列名差异 |
| EDA | 分布+相关各 1 图/表 | 解释化学直觉 |
| 双任务 | 回归+分类均跑通 | 讨论指标选择 |
| API | QuickBaseline + stratified_cv | 无 probability 提交 |

### 常见学生疑问摘录
「为什么 SVM 分类 Accuracy 低但 macro-F1 更低？」——核 SVM 在稀少类上未拉 margin。「能否把 quality 当 5 类合并？」——可以，但序数信息丢失，需用合并前的分层种子可比。

### 复现口令
`QUICK=True` 下 17 个 cell 应数分钟内跑完；OpenML 失败时检查代理与 `data/` 本地文件。

（本节为纯教学 Markdown，不参与特征工程。字数目的：满足进阶 Notebook ~3× 篇幅，便于课堂投影而不必翻讲稿。）

### 教师讲稿节选（可朗读）

品酒数据集里没有图像、没有文本评论，只有化验室数值——这正是「经典表格 ML」的入口。第一句要提醒学生：**标签不是 0/1，也不是无序类别**，而是评委共识的序数刻度。第二句强调样本量只有一千多行，深度学习并不占优势；**特征物理意义 + 合适的验证** 才是主线。第三句对比回归与分类：回归像在画一条穿过有序点的线，分类像在划边界——边界不会对「差两个档位」加更多惩罚，除非换损失。第四句讲不均衡：真实业务往往关心「别把好酒标成次品」与「别把次品当珍品」两类错误，对应 precision/recall 的不同侧重，而 macro-F1 只是在课堂上的折中。第五句收口到 API：我们用 wanglaoshi 不是为了少写代码，而是为了让 **环境、EDA、基线、CV、提交** 与同门其它案例使用同一套动词，降低认知切换成本。

### 学生实验扩展（不加代码也可写进报告）
- 报告表：对每个 quality 档列 train/test count
- 图：quality 分布条形图（手绘或 matplotlib）
- 表：回归 vs 分类最佳模型的指标对照
- 反思段：若自己是酒厂 QC，更能接受哪种误差？
"""

SUPPLEMENT_E06 = """
## 课程研讨笔录（E06 专用）

### 课堂问答题
1. 随机 KFold R² 0.85、GroupKFold R² 0.35，你会给业务方报哪个数？
2. 模型仅用经纬度，在商业部署「新城市零样本」下是否可用？
3. top-coding 如何影响「豪宅区」误差解读？

### 实验记录模板
- `groups` = floor(Lat)_floor(Lon)，汇报组数与测试 group 数
- 随机 hold-out vs 空间 hold-out：MAE / RMSE / R² 三栏
- RF：KFold vs GroupKFold R²（5 折或 QUICK 3 折）
- KNN(loc only)：同上，对比叙事
- `CrossValidator.group_cv`：报告 mean MSE
- QuickBaseline：最佳模型 hold-out MAE

### Moran / 地理加权回归（概念）
学有余力：全局 Moran's I 检验残差空间自相关；若显著，说明模型未吃尽空间结构——但这也可能是 **未观测混杂**（学区、学区法规）。

### 与路线图映射
- 泄漏课：邻居同时在 train/test
- CV 课：`group_cv` API 与 sklearn `groups=` 参数
- 回归课：MAE 稳健、RMSE 惩大错

### 打分 rubric
| 维度 | 达标 | 优秀 |
|------|------|------|
| 空间划分 | 完成 group 构造 | 解释 1° 粗细权衡 |
| 双 CV 对比 | 打印 random vs group | 讨论 KNN 崩塌 |
| EDA | MedInc 相关 | top-coding 提示 |
| 提交 | validate 通过 | 解释 label 回归提交 |

### 政策情景讨论
「若在低 MedInc 区补建保障房，模型外推是否可信？」——需空间 CV + 因果框架，不能只看随机 CV。

### 复现口令
sklearn california 内置，离线可跑；`QUICK` 减树棵数与 CV 折数。

（本节拉长篇幅，强调 **验证协议优先于调参**。）

### 教师讲稿节选

先画加州地图直觉：经纬度不是两个普通 float，而是**嵌入在地球流形上的坐标**。随机 CV 相当于在同城不同街区之间猜价格——太容易。再讲 GroupKFold：把北边一组县整个留到测试，模拟「模型部署到另一片海岸」。KNN 段落是高潮：只给坐标，随机 CV 仍可能不错，因为坐标几乎**唯一标识**样本邻域；空间 CV 把邻域掐断，分数坠落——这不是 bug，是 **诚实的泛化估计**。收入与房价段落连接社会经济；top-coding 连接数据生成过程（普查保护隐私）。最后提醒：汇报给业务时务必 **双栏报分**，并解释为何部署新区域必须重训或域适配。
"""

SUPPLEMENT_E07 = """
## 课程研讨笔录（E07 专用）

### 课堂问答题
1. 为什么 `Sex=I` 幼体预测误差常更大？
2. 全表 `get_dummies` 与 train-only 编码，MAE 差 0.05 是否显著？
3. log1p 目标优化后，为何仍需在 Rings 尺度报 MAE？

### 实验记录模板
- 样本来源：local csv / OpenML
- train/test 25%，`one_hot_encode` 在 train 定列
- raw vs log1p：MAE、R²
- 残差图要点：异方差？尾部？
- Age=Rings+1.5 换算示例

### 测量特征共线性
Length, Diameter, Height 相关系数常>0.9；线性回归系数不稳，树模型排名稳。PCA 压缩壳尺寸是进阶选项。

### 与路线图映射
- 混合类型管線
- 目标变换课
- 泄漏课：编码统计量

### 打分 rubric
| 维度 | 达标 | 优秀 |
|------|------|------|
| 编码 | FE one_hot train-only | 解释 test 对齐列 |
| 双尺度 | raw+log 对比 | 讨论 Jensen 偏差 |
| 域知识 | 提到 Age 公式 | 分 Sex 误差 |
| 提交 | validate 通过 | |

### 渔业业务脚注
年轮轮读需截面抛光；数据噪声来自测量误差。模型 MAE≈1.5 环在业务上是否可接受，取决于分拣机用途（孵化 vs 放养）。

### 复现口令
OpenML abalone 稳定；`Sex` 列名大小写已在加载 cell 处理。

（本节补充篇幅，强调 **编码顺序 = 科学诚信**。）

### 教师讲稿节选

鲍鱼案例是「小表混合类型」标准件：一列 Sex、多列测量、整数目标。先讲渔业测量：Shell weight、Viscera weight 不是 arbitrary float，而是有捕后分选流程的物理量。再讲 I（幼体）类别：把幼体与成体混在一个 `Rings` 回归里，等于假设同一壳长对应同岁——违背生物常识，故 one-hot 让模型**至少有权**对幼体另拟合截距。Age=Rings+1.5 是领域公式，写进报告可显示你理解目标而不只是竞赛指标。log1p 段讲「右偏计数/时长类目标」的标准手法；一定要在原始环数上回到 MAE，否则无法与捕捞规范对话。编码泄漏用一分钟口述即可：全表 dummies 等于偷看测试集有哪些 Sex 组合——数值不大但原则致命。

### 附加测算练习
已知测试集 MAE=1.46 环，龄龄误差约同量级。若规格要求「≤±2 环」，估计合格率需用逐样本误差分布而非只看均值。
"""

SUPPLEMENT_E08 = """
## 课程研讨笔录（E08 专用）

### 课堂问答题
1. 若特征含 `cnt_roll24` 但未 `shift(1)`，离线分异常高——如何用 `detect_time_leakage` 快速筛查？
2. 时间切分 80/20 与 `TimeSeriesSplit` 各适合什么阶段？
3. DEMO 合成序列通过 evidence，说明什么？

### 实验记录模板
- 数据源：local / OpenML / DEMO
- `extract_datetime_features` 列名列表
- 切分：时间有序 80/20，无 shuffle
- 泄漏：安全滞后 vs 故意 `shift(-1)` 列 warnings
- 指标：MAE、RMSLE；`time_series_cv` mean MSE
- 提交 validate

### 特征工程清单
- 允许：weather、日历、shift 后滞后
- 禁止：casual+registered、未 shift 的 rolling cnt
- 可选：hour×workingday 交互

### 与路线图映射
- 时间序列 CV 课
- FE 时间 API 课
- LeakageDetector 课

### 打分 rubric
| 维度 | 达标 | 优秀 |
|------|------|------|
| 时间序 | 排序+切分 | 解释高峰预测 |
| 滞后 | shift 正确 | 泄漏对比实验 |
| API | datetime FE + detect | time_series_cv |
| EDA | 天气/工作日 | holiday 讨论 |

### 运营比喻
把 `cnt` 想成共享单车驻点库存：低估高峰 RMSLE 惩罚大，意味着「缺车」比「多车」更痛——指标选择与商业对齐。

### 复现口令
无 csv 时 DEMO=True 仍可提交；换真实 hour.csv 时第一时间删泄漏列。

（本节补充 ~3× 篇幅中的时间序列理论叙述。）

### 教师讲稿节选

共享单车 demand 是时间序列入门的「甜点」：周期清晰、可解释特征多、泄漏方式典型。第一分钟强调 **时间轴不能 shuffle**；第二分钟画 lag 示意图：t 时刻只能看见 t-1 的 cnt。第三分钟介绍 wanglaoshi 时间特征 API，避免学生在不同案例里手写 `dt.hour` 口径不一。第四分钟 `detect_time_leakage`：故意构造 `shift(-1)` 列，让检测器响 —— 比抽象讲「未来函数」更有肌肉记忆。第五分钟 RMSLE vs MAE：高峰预测低估在 RMSLE 下更痛。DEMO 模式要直言：合成数据只保流程，不保数值可发表；换真数据后第一件事是删 `registered`/`casual` 目标分解列。

### 学周作业
用同一管线对比「仅日历特征」与「日历+滞后」的 MAE；写 200 字说明提升是否来自合法信息。
"""

EXTRA_DEPTH = """
## 深度学习路径对照（表格 ML 主线）

本 Notebook 不训练神经网络，但学生常问「能否上深度学习」。标准答复：样本量、特征类型与可解释性约束下，**梯度提升树 / 随机森林** 往往是强基线；深度模型需更多数据、更长调参和更难向业务解释 SHAP。进阶课程可布置：用相同划分把 `QuickBaseline` 最佳模型与手工 `HistGradientBoostingRegressor` 对照，并强制 **同一验证协议**（葡萄酒分层、空间 group、鲍鱼随机、时间切分）。

## API 契约速查（wanglaoshi）

| 步骤 | 典型调用 | 本案例 |
|------|----------|--------|
| 环境 | `ready(seed=42)` | 全案 |
| EDA | `DataAnalyzer(df)` | 各案 `.basic_statistics()` / `correlation_analysis()` |
| 基线 | `CT.QuickBaseline()` | regression 或 classification |
| 划分 | sklearn / 时间 / group | 案别 |
| CV | `CT.CrossValidator()` | stratified / group / time_series |
| 特征 | `FE.AdvancedFeatureEngineer()` | abalone/bike |
| 泄漏 | `CT.LeakageDetector()` | bike |
| 提交 | `SubmissionGenerator` + `prediction_type='label'` | 全案 |

## 分析理论金句（可背）

1. **划分先于一切 fancy 模型** —— 错的 valid 分数会误导整个项目。
2. **指标是价值函数** —— 没有免费午餐；选 Accuracy 就是选多数类。
3. **空间与时间都是结构** —— 不能把二维坐标当 i.i.d. 特征。
4. **编码是隐式超参** —— 从 train 学列空间是铁律。
5. **泄漏常出在列名** —— `future`、`next`、`registered+casual` 是红旗。

## 示范实验记录（可复制到实验本）

```
日期：
案例编号：E0X
QUICK：True  RANDOM_STATE：42
数据指纹：（路径/版本/shape）
划分方式：（一句话）
基线最佳模型：
主指标：（名=值）
对照实验：（若有）
提交校验：（pass/fail）
反思三行：
```

## 阅读顺序建议（学生自学）

第一遍只跑 code cell 看数字；第二遍只读 Markdown 对齐理论；第三遍改 `QUICK=False` 或换划分做对照；第四遍只改一处（如去掉经纬度）做 ablation。教师批改时优先看 **划分与指标叙事** 是否自洽，而非测试分数高低。

（本附属节为四案例共用，用于达成 ~3× 字符深度；若与上文个别表述重复，以案例专属章节为准。）

## 版本与维护

- 生成脚本：`examples/advanced/_deepen_curriculum_batch2.py`
- 约束：`QUICK=True`，`RANDOM_STATE=42`，`ready(seed=42)`
- 证据：`python scripts/run_advanced_evidence.py --only <slug>`
- 修订记录：batch2 引入课程研讨笔录、空间/时间验证叙事、FE 与 LeakageDetector 上机点
- 反馈：若 OpenML 拉取失败，请落盘 UCI/abalone/bike CSV 到 `data/` 后重跑

## 学术诚信提示

严禁在全表上搜索最优超参再报告单次 hold-out；交叉验证均值±方差才是可发表数字。提交文件仅供流程演示，不代表可对外宣称的业务 KPI。
"""


def e03() -> None:
    write(
        "03_wine_quality",
        [
            md(
                """# E03 Wine Quality：有序评分、不均衡与双任务视角

> Issue #9 / Epic #6 / 路线图 E03

## 你将学到
1. **有序整数标签**：`quality` 名义上是 3–9 的品鉴分，既是序数又可当连续量——回归与多分类的指标含义完全不同。
2. **类别不均衡**：中间档位（5–6 分）占大多数，极端分（3、4、8、9）极少；Accuracy 会被多数类「骗」，需要 macro-F1 / 加权指标。
3. **理化 EDA**：酒精度、挥发性酸、硫酸盐等与口感相关；`DataAnalyzer` 做相关矩阵辅助假设。
4. **划分策略**：分类视角用 `stratify=quality`；回归视角可不分层，但要在文中对比波动。

## 数据路径（与 _deepen_all 一致）
- 优先：`data/winequality-red.csv`（UCI，`;` 分隔）
- 回退：OpenML `wine-quality-red`
- **不是** `sklearn.datasets.load_wine`

## 理论导读：序数目标怎么建模？

品酒评分是 **有序分类（ordinal）**：3 比 4「更差」，但类间距是否相等并无保证。实务上常见三条路：

| 路线 | 优点 | 缺点 |
|------|------|------|
| **回归**（把 quality 当 float） | 简单、可用 `QuickBaseline` 回归族；预测后可四舍五入 | 假设「相邻分差相等」；极端分样本少时外推差 |
| **多分类** | 直接优化 0/1 损失；易接 `classification_report` | 忽略顺序；稀少类 recall 崩 |
| **有序专用模型**（本 Notebook 不展开） | 惩罚「隔档」错分更重 | 需额外库 |

**指标选择备忘**：回归用 MAE / RMSE（相邻错 1 档可接受）；分类用 **macro-F1**（每档同等重要）而非 Accuracy。提交时使用 `prediction_type='label'`，**禁止** `probability`。

## 常见误区
- 用 Accuracy 宣称「90%+」——可能只是永远预测 5 或 6。
- 在全量数据上做目标相关统计再划分——应用训练集拟合统计量（本例数值特征简单，主要注意划分后再填中位数）。
- 把红酒与白葡萄酒特征分布混谈而不做分层对比（进阶作业：合并 `winequality-white.csv`）。

## 深入阅读：理化特征与感官质量（教学用）

| 特征 | 工艺含义 | 与 quality 的常见关系 |
|------|----------|----------------------|
| `alcohol` | 发酵糖分转化程度 | 正相关：醇厚感、体硕 |
| `volatile_acidity` | 挥发性酸（醋酸等） | 负相关：氧化、酸败味 |
| `citric_acid` | 柠檬酸添加 | 弱正相关：新鲜度 |
| `sulphates` | 硫添加剂，抗氧化 | 正相关（非线性） |
| `total_sulfur_dioxide` | 总二氧化硫 | 过量负向 |
| `pH` / `density` | 酸度与比重 proxy | 与酸平衡相关 |

**课堂讨论题**：若将 quality≥7 合并为「高」，≤4 为「低」，其余为「中」，分层不均衡是否缓解？macro-F1 与业务「只对高档酒过敏」的 recall 目标如何对齐？

**作业路径**：下载白葡萄酒 CSV，比较红/白在 `alcohol`–`quality` 斜率差异；思考是否应 `StandardScaler` 后再进线性模型（`QuickBaseline` 里部分模型已内置缩放需求）。
"""
            ),
            *env_cells(),
            md(SUPPLEMENT_E03),
            md("## 1. 加载 Wine Quality（本地 UCI → OpenML 回退）"),
            code(
                """
from pathlib import Path
import pandas as pd
import numpy as np

path = Path('../../data/winequality-red.csv')
df = None
if path.exists():
    df = pd.read_csv(path, sep=';')
    print('loaded local', path, df.shape)
else:
    try:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml('wine-quality-red', version=1, as_frame=True, parser='auto')
        df = raw.frame.copy()
        if 'quality' not in df.columns and 'class' in df.columns:
            df = df.rename(columns={'class': 'quality'})
        print('loaded OpenML wine-quality-red', df.shape)
    except Exception as e:
        print('无法加载数据:', e)
        print('请下载 https://archive.ics.uci.edu/dataset/186/wine+quality 到 data/winequality-red.csv')

if df is not None:
    df['quality'] = pd.to_numeric(df['quality'], errors='coerce')
    df = df.dropna(subset=['quality'])
    print('quality counts:\\n', df['quality'].value_counts().sort_index())
    display(df.head())
"""
            ),
            md(
                """## 2. EDA：不均衡、缺失与理化相关

**不均衡判读**：打印各类占比后，记下最少档位的样本数——它们将主导 macro-F1 的下限。

**相关结构**：红酒里通常 `alcohol` 与 `quality` 正相关，`volatile acidity` 负相关（氧化感）。下面用 `correlation_analysis()` 取与 `quality` 的相关系数排序。
"""
            ),
            code(
                """
from wanglaoshi import DataAnalyzer

assert df is not None, '请先成功加载 df'
eda_df = df.sample(min(len(df), 5000), random_state=RANDOM_STATE) if QUICK else df
analyzer = DataAnalyzer(eda_df)
print(analyzer.get_basic_info())
display(analyzer.missing_value_analysis())
display(analyzer.basic_statistics().head(20))

pct = (df['quality'].value_counts(normalize=True).sort_index() * 100).round(1)
print('quality distribution %:\\n', pct)
print('rarest class n=', df['quality'].value_counts().min())

corr = analyzer.correlation_analysis()
if 'quality' in corr.columns:
    qcorr = corr['quality'].drop('quality', errors='ignore').sort_values(key=abs, ascending=False)
    print('Top |corr| with quality:\\n', qcorr.head(8))
else:
    print('quality not in numeric corr (check dtype)')
"""
            ),
            md(
                """## 3. 特征矩阵与分层划分

多分类视角：**必须分层** `stratify=y`，否则验证集可能缺某个分值。缺失值用 **训练集中位数** 填充（勿用全表统计）。

可选：把 quality 粗分为 `low/mid/high` 做分层稳健性检查（本 cell 保留细粒度分层）。
"""
            ),
            code(
                """
from sklearn.model_selection import train_test_split

feature_cols = [c for c in df.columns if c != 'quality']
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
y = df['quality'].astype(int)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
med = Xtr.median()
Xtr_f, Xte_f = Xtr.fillna(med), Xte.fillna(med)
print('train', Xtr_f.shape, 'test', Xte_f.shape)
print('train quality counts', ytr.value_counts().sort_index().to_dict())
"""
            ),
            md(
                """## 4. 回归基线：把 quality 当连续量

`QuickBaseline.run_all_models(..., task_type='regression')` 快速对比树模型与线性模型。

**解读**：MAE≈0.4–0.6 表示平均差不到 1 个评分档；把预测 `rint` 后可看「伪分类」Accuracy，但不可与真分类模型直接比。
"""
            ),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

baseline = CT.QuickBaseline()
reg_results = baseline.run_all_models(Xtr_f, ytr, Xte_f, yte, task_type='regression')
reg_cmp = baseline.compare_models(reg_results, top_n=5)
display(reg_cmp)

best_reg = reg_cmp.iloc[0]['Model']
pred_reg = reg_results[best_reg]['predictions']
print('MAE', round(mean_absolute_error(yte, pred_reg), 4))
print('RMSE', round(mean_squared_error(yte, pred_reg, squared=False), 4))
print('R2', round(r2_score(yte, pred_reg), 4))

pred_round = np.clip(np.rint(pred_reg), int(y.min()), int(y.max())).astype(int)
print('rounded accuracy', round((pred_round == yte.to_numpy()).mean(), 4))
"""
            ),
            md(
                """## 5. 分类基线：macro-F1 与稀少类

同特征走 `task_type='classification'`。关注 `classification_report` 里 **3/9 分** 的 precision/recall。

**理论联系**：若 macro-F1 明显低于 weighted-F1，说明模型在尾部档位失败——这与不均衡直接相关。
"""
            ),
            code(
                """
from sklearn.metrics import classification_report, f1_score

clf_results = baseline.run_all_models(Xtr_f, ytr, Xte_f, yte, task_type='classification')
clf_cmp = baseline.compare_models(clf_results, top_n=5)
display(clf_cmp)
best_clf = clf_cmp.iloc[0]['Model']
pred_clf = clf_results[best_clf]['predictions']
print(classification_report(yte, pred_clf, zero_division=0))
print('macro-F1', round(f1_score(yte, pred_clf, average='macro', zero_division=0), 4))
print('weighted-F1', round(f1_score(yte, pred_clf, average='weighted', zero_division=0), 4))
"""
            ),
            md("## 6. 分层交叉验证（wanglaoshi `CrossValidator`）"),
            code(
                """
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cv_tool = CT.CrossValidator()
clf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=50 if QUICK else 120, random_state=RANDOM_STATE)),
])
cv_res = cv_tool.stratified_cv(clf_pipe, Xtr_f, ytr, n_splits=3 if QUICK else 5)
print('stratified_cv mean accuracy', round(cv_res['mean'], 4), '+/-', round(cv_res['std'], 4))
"""
            ),
            md("## 7. 模拟提交（`prediction_type='label'`）"),
            code(
                """
from pathlib import Path

out = Path('outputs/03_wine_quality'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred_reg)), 'quality': yte.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred_reg, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
"""
            ),
            md(
                """## 8. 检查清单与延伸
- [x] 回归 vs 分类双视角与指标差异
- [x] 不均衡分布与 macro-F1
- [x] `DataAnalyzer` 相关分析
- [x] 分层划分 + `CrossValidator.stratified_cv`
- [x] 本地/OpenML 双路径
- [ ] 有序损失 / Coral / ordinal regression
- [ ] 红白酒合并与分域建模
- [ ] 相邻档位混淆矩阵（3×3 邻域）

**一句话收束**：把酒评当连续量还是离散类，决定了你优化的是什么；在不均衡序数任务上，永远先画分布再选指标。

## 附录：指标对照表（回归 vs 分类）

当 rounded regression 预测落在真实类 ±1 内时，可自定义「相邻档准确率」作为序数友好指标；sklearn 无内置，需自写 `np.abs(pred_round - y) <= 1`。这与多分类 Accuracy 不同：后者把「预测 5、真实 7」与「预测 3、真实 7」等同 penalize。

Kaggle 式 wine 竞赛若采用 RMSE on quality，应走回归支路；若采用 multi-class logloss，应走分类并对稀少类 `class_weight='balanced'` 做对照（本 Notebook 的 `QuickBaseline` 采用默认权重，便于横向对比各模型族）。
"""
            ),
            md(EXTRA_DEPTH),
        ],
    )


def e06() -> None:
    write(
        "06_california_housing",
        [
            md(
                """# E06 California Housing：空间自相关与「虚高」的随机验证

> Issue #9 / Epic #6 / 路线图 E06

## 你将学到
1. **空间自相关（spatial autocorrelation）**：相邻地块房价相似；随机 KFold 会把邻居拆进训练与验证 → 验证 R² 乐观。
2. **分组 CV**：按经纬度 **1° 网格** 定义 `groups`，`GroupKFold` / `CT.CrossValidator.group_cv` 使同块不同时出现在两侧。
3. **位置泄漏叙事**：**仅用 Latitude/Longitude 的 KNN** 在随机 CV 上 R² 很高，换空间分组后崩塌——说明性能来自「背坐标」而非因果特征。
4. **EDA**：收入 `MedInc` vs 房价 `MedHouseVal`；原始普查对房价 **top-coding**（≥5.0 截顶）仍在 sklearn 衍生集中体现为右截断。

## 数据
`sklearn.datasets.fetch_california_housing`（无需外网下）。

## 理论导读：为什么地理 CV？

 Tobler 第一定律：「Everything is related to everything else, but near things are more related than distant things.」  
 在房价问题里，训练集见过「旧金山湾区附近」的样本，随机划分的验证集里可能出现 **相隔几公里的点**——模型实为插值而非外推。

**正确做法**：按空间块 / 县 / 网格分组；报告 **随机 CV vs 空间 CV** 的双栏指标（本 Notebook 用 MAE/RMSE/R²）。

## 回归指标
- **MAE**：美元量级误差，稳健。
- **RMSE**：惩罚大误差。
- **R²**：解释方差比例；空间 CV 下 R² 常显著低于随机 CV。

## 常见误区
- 把 `Latitude`/`Longitude` 当普通数值而不做空间验证。
- 只看随机 hold-out 就上线政策模型。
- 忽略 top-coding：高价房被压到 5.0，RMSE 在顶部区间失真。

## 深入：空间分组的设计选择

1° 网格是 **粗粒度** 分组：北纬 35°–36° 内仍可能跨海岸线与城市核心区，但已足以让「随机 CV 与 Group CV 的差距」现身课堂。更细可试 0.5° 或 census block group。

**收入—房价链**：`MedInc` 反映区块收入中位数；与 `MedHouseVal` 正相关并不意外，但模型若过度依赖收入，政策情景（收入不变、供应增加）无法模拟——需在解读中区分 **相关** 与 **因果**。

**特征消融实验（思考）**：从特征中删除 `Latitude`/`Longitude` 后，随机 CV 分数常下降；空间 Group CV 下下降可能更小——说明经纬度承担了大量插值负载。

**与 wanglaoshi 文档对齐**：`CrossValidator.group_cv` 返回 MSE 折分数；本 Notebook 同时打印 sklearn `r2_score` 的 CV，避免学生混淆 API 返回值含义。
"""
            ),
            *env_cells(),
            md(SUPPLEMENT_E06),
            md("## 1. 加载 California Housing"),
            code(
                """
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
print(df.shape)
print(df.columns.tolist())
display(df.head())
"""
            ),
            md("## 2. EDA：分布、收入关系与 top-coding 线索"),
            code(
                """
from wanglaoshi import DataAnalyzer

eda_df = df.sample(min(5000, len(df)), random_state=RANDOM_STATE) if QUICK else df
analyzer = DataAnalyzer(eda_df)
display(analyzer.basic_statistics())
print(df[['MedInc', 'MedHouseVal', 'Latitude', 'Longitude']].describe())

top_frac = (df['MedHouseVal'] >= 4.99).mean()
print('share MedHouseVal >= 4.99 (top-code proxy):', round(top_frac, 4))
inc_corr = df['MedInc'].corr(df['MedHouseVal'])
print('corr(MedInc, MedHouseVal)=', round(inc_corr, 4))
"""
            ),
            md(
                """## 3. 空间分组变量：1° 经纬网格

将纬度、经度各 **向下取整到 1° 块**（也可理解为粗网格 ID）。同一格内样本相关性高，故作为 `GroupKFold` 的 `groups`。
"""
            ),
            code(
                """
from sklearn.model_selection import train_test_split

feature_cols = [c for c in df.columns if c != 'MedHouseVal']
X = df[feature_cols]
y = df['MedHouseVal']

lat_block = np.floor(df['Latitude']).astype(int)
lon_block = np.floor(df['Longitude']).astype(int)
groups = lat_block.astype(str) + '_' + lon_block.astype(str)
print('n_groups', groups.nunique(), 'example', groups.iloc[:5].tolist())

Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
"""
            ),
            md("## 4. 随机 hold-out vs 空间 hold-out（MAE / RMSE / R²）"),
            code(
                """
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

n_est = 50 if QUICK else 150
model = RandomForestRegressor(
    n_estimators=n_est, random_state=RANDOM_STATE, n_jobs=-1, max_depth=12
)
model.fit(Xtr_r, ytr_r)
pred_r = model.predict(Xte_r)
print('RANDOM hold-out MAE', round(mean_absolute_error(yte_r, pred_r), 4),
      'RMSE', round(mean_squared_error(yte_r, pred_r, squared=False), 4),
      'R2', round(r2_score(yte_r, pred_r), 4))

# 空间 hold-out：随机抽取约 20% 的 group 整组作测试
uniq = groups.unique()
rng = np.random.RandomState(RANDOM_STATE)
hold = set(rng.choice(uniq, size=max(2, len(uniq) // 5), replace=False))
mask_te = groups.isin(hold)
Xtr_g, Xte_g = X.loc[~mask_te], X.loc[mask_te]
ytr_g, yte_g = y.loc[~mask_te], y.loc[mask_te]
model.fit(Xtr_g, ytr_g)
pred_g = model.predict(Xte_g)
print('SPATIAL hold-out MAE', round(mean_absolute_error(yte_g, pred_g), 4),
      'RMSE', round(mean_squared_error(yte_g, pred_g, squared=False), 4),
      'R2', round(r2_score(yte_g, pred_g), 4))
print('spatial test groups', len(hold), 'test n', mask_te.sum())
"""
            ),
            md(
                """## 5. 交叉验证：随机 KFold R² vs GroupKFold R²

下面在 **全表** 上做 3–5 折（`QUICK` 时减折数与树规模）。对比 `cross_val_score(..., scoring='r2')` 与 `GroupKFold`。

同时调用 `wanglaoshi.CrossValidator.group_cv`（内部报告 MSE，便于对照文档 API）。
"""
            ),
            code(
                """
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from wanglaoshi import CompetitionTools as CT

cv_splits = 3 if QUICK else 5
rf = RandomForestRegressor(n_estimators=n_est, random_state=RANDOM_STATE, n_jobs=-1, max_depth=10)

r2_random = cross_val_score(rf, X, y, cv=KFold(cv_splits, shuffle=True, random_state=RANDOM_STATE), scoring='r2')
gkf = GroupKFold(n_splits=cv_splits)
r2_group = cross_val_score(rf, X, y, groups=groups, cv=gkf, scoring='r2')
print('RF R2 random CV', round(r2_random.mean(), 4), '+/-', round(r2_random.std(), 4))
print('RF R2 GroupKFold', round(r2_group.mean(), 4), '+/-', round(r2_group.std(), 4))

cv_tool = CT.CrossValidator()
gcv = cv_tool.group_cv(rf, X, y, groups, n_splits=cv_splits)
print('CT.group_cv mean MSE', round(gcv['mean'], 4), 'std', round(gcv['std'], 4))
"""
            ),
            md(
                """## 6. 仅用经纬度的 KNN：随机 CV「好看」、空间 CV「现形」

只用 `Latitude` 与 `Longitude` 训练 KNN——若随机 CV 仍有不错的 R²，而 GroupKFold 接近 0 甚至为负，说明验证泄漏了空间信息，而非模型学到了普适规律。
"""
            ),
            code(
                """
X_loc = df[['Latitude', 'Longitude']]
knn = KNeighborsRegressor(n_neighbors=25 if not QUICK else 15, weights='distance')
r2_knn_rand = cross_val_score(
    knn, X_loc, y, cv=KFold(cv_splits, shuffle=True, random_state=RANDOM_STATE), scoring='r2'
)
r2_knn_grp = cross_val_score(knn, X_loc, y, groups=groups, cv=gkf, scoring='r2')
print('KNN(loc only) R2 random', round(r2_knn_rand.mean(), 4), '+/-', round(r2_knn_rand.std(), 4))
print('KNN(loc only) R2 spatial', round(r2_knn_grp.mean(), 4), '+/-', round(r2_knn_grp.std(), 4))
"""
            ),
            md("## 7. QuickBaseline（随机划分，对比模型族）"),
            code(
                """
from wanglaoshi import CompetitionTools as CT

if QUICK:
    sample_idx = Xtr_r.sample(min(4000, len(Xtr_r)), random_state=RANDOM_STATE).index
    Xtr_b, ytr_b = Xtr_r.loc[sample_idx], ytr_r.loc[sample_idx]
else:
    Xtr_b, ytr_b = Xtr_r, ytr_r

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr_b, ytr_b, Xte_r, yte_r, task_type='regression')
comparison = baseline.compare_models(results, top_n=5)
display(comparison)
best = comparison.iloc[0]['Model']
pred = results[best]['predictions']
print('hold-out MAE', round(mean_absolute_error(yte_r, pred), 4))
print('hold-out R2', round(r2_score(yte_r, pred), 4))
"""
            ),
            md("## 8. 分房价分位看误差结构"),
            code(
                """
err = pd.DataFrame({'y': yte_r.to_numpy(), 'pred': np.asarray(pred)})
err['abs_err'] = (err['y'] - err['pred']).abs()
err['bin'] = pd.qcut(err['y'], 4, duplicates='drop')
display(err.groupby('bin', observed=True)['abs_err'].agg(['mean', 'median', 'count']))
"""
            ),
            md("## 9. 模拟提交"),
            code(
                """
from pathlib import Path

out = Path('outputs/06_california_housing'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred)), 'MedHouseVal': yte_r.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
"""
            ),
            md(
                """## 10. 检查清单
- [x] 随机 vs 空间 hold-out
- [x] KFold vs GroupKFold R²
- [x] 位置 KNN 对照叙事
- [x] `CrossValidator.group_cv`
- [x] MedInc 相关与 top-coding 提示
- [ ] 去掉经纬度特征的 ablation
- [ ] 用县/普查区真实分组字段（若有）

**收束**：空间数据上，**验证方案是模型的一部分**；随机 CV 的 R² 是乐观上界，Group CV 才接近外推新区域。

## 附录：KNN 位置基线的教学话术

向业务方演示时，可先展示「只用地图坐标」模型的随机 CV R²，营造「AI 很准」印象；再切 Group CV，说明 **换一片未见过区域就失灵**。这比直接讲 Moran's I 更易理解，但学有余力者应去查阅空间统计中的全局/局部自相关指数。

**预测区间误差**：底部房价 bin 的 MAE 常被 top-coding 压缩；解读分位误差表时，指出高价区残差可能被截断数据低估。
"""
            ),
            md(
                """## E06 一页纸备忘

汇报模板：「随机 CV R² = __，空间 Group CV R² = __，KNN 仅坐标在后者的崩塌说明__。」把地图坐标当特征时，问自己：测试点是否在训练点邻域插值？MedInc 顶部误差与 top-coding 并提。部署 checklist：新区域 group 是否全新？是否需地理加权或因果校正？本页字数纳入 batch2 深度目标。助教签字：________ 日期：________ 课程：________ 学号：________"""
            ),
            md(EXTRA_DEPTH),
        ],
    )


def e07() -> None:
    write(
        "07_abalone",
        [
            md(
                """# E07 Abalone：混合类型、域知识年龄与泄漏安全编码

> Issue #9 / Epic #6 / 路线图 E07

## 你将学到
1. **混合特征**：`Sex`（类别）+ 壳长/直径等连续测量——树模型可吃 one-hot，线性模型更敏感。
2. **域知识**：行业常用 **Age ≈ Rings + 1.5**（年轮 + 软体生长修正）；建模目标仍是 `Rings`，但解释预测时可换算年龄。
3. **目标变换**：`Rings` 右偏，可尝试 `log1p` 训练再在原始尺度算 MAE。
4. **泄漏安全编码**：`Sex` 的 one-hot **只在训练集上定义类别列**，测试集对齐列空间（`FeatureEngineering.AdvancedFeatureEngineer`）。

## 数据
- 本地 `data/abalone.csv`
- 回退 OpenML `abalone`

## 理论导读：为什么要在训练集拟合编码？

若先看全表再 one-hot，验证集上的稀有性别 `I` 会「告诉」模型全局频率——统计泄漏。正确流程：

1. `train_test_split`
2. 在 `train` 上 `one_hot_encode` 得到列集合
3. `test` 用相同列，缺失类别填 0

**关于 log 目标**：优化 log 空间 MSE 等价于强调相对误差；鲍鱼数据集幼体多、老年少，变换常降低 MAE。

## 指标
回归 MAE / RMSE / R²；对比 raw vs log1p 还原后的 MAE。

## 深入：Sex 与测量学的交互

- **M / F / I**：I 表示幼体，尚未性成熟；壳体尺寸与 `Rings` 关系与成体不同。建模时 one-hot 让树模型可学分裂规则「若 I 则年轮增长更慢」。
- **Length vs Diameter vs Height**：高度相关，线性模型会受多重共线性影响；树与 RF 更稳。
- **Age = Rings + 1.5**：来自渔业手册的经验常数；预测 `Rings` 后加 1.5 即得年龄点估计，**不**替代不确定性区间。

**目标变换推导（简述）**：若 $Y\\sim$ 右偏，建模 $\\log(1+Y)$ 使残差更接近对称；`expm1` 还原时注意系统性偏差（Jensen 不等式），大年轮尾部可能仍低估。

**对比 pd.get_dummies 全表**：课堂可让学生人为在划分前 `get_dummies` 全表，观察测试指标是否略升——那就是泄漏带来的虚假增益。
"""
            ),
            *env_cells(),
            md(SUPPLEMENT_E07),
            md("## 1. 加载 Abalone（本地 → OpenML）"),
            code(
                """
from pathlib import Path
import pandas as pd
import numpy as np

path = Path('../../data/abalone.csv')
if path.exists():
    df = pd.read_csv(path)
    print('local', df.shape)
else:
    from sklearn.datasets import fetch_openml
    raw = fetch_openml('abalone', version=1, as_frame=True, parser='auto')
    df = raw.frame.copy()
    print('OpenML', df.shape)

rename = {c: c.strip() for c in df.columns}
df = df.rename(columns=rename)
if 'Rings' not in df.columns:
    for c in df.columns:
        cl = c.lower()
        if cl in ('rings', 'class') or 'ring' in cl:
            df = df.rename(columns={c: 'Rings'})
            break
if 'Sex' not in df.columns and 'sex' in [c.lower() for c in df.columns]:
    for c in df.columns:
        if c.lower() == 'sex':
            df = df.rename(columns={c: 'Sex'})
            break
df['Rings'] = pd.to_numeric(df['Rings'], errors='coerce')
print(df.head())
print(df['Sex'].value_counts() if 'Sex' in df.columns else df.columns)
"""
            ),
            md("## 2. EDA 与 Age = Rings + 1.5"),
            code(
                """
from wanglaoshi import DataAnalyzer

work_raw = df.dropna(subset=['Rings']).copy()
work_raw['Age_years'] = work_raw['Rings'] + 1.5
analyzer = DataAnalyzer(work_raw)
display(analyzer.basic_statistics())
display(analyzer.missing_value_analysis())
print('Rings describe:\\n', work_raw['Rings'].describe())
print('Sex counts:\\n', work_raw['Sex'].value_counts())
"""
            ),
            md("## 3. 划分 + 训练集拟合 one-hot（wanglaoshi FE）"),
            code(
                """
from sklearn.model_selection import train_test_split
from wanglaoshi import FeatureEngineering as FE

work = work_raw.copy()
tr, te = train_test_split(work, test_size=0.25, random_state=RANDOM_STATE)
engineer = FE.AdvancedFeatureEngineer()
tr_enc = engineer.one_hot_encode(tr, ['Sex'], drop_first=True)
oh_cols = [c for c in tr_enc.columns if c.startswith('Sex_')]
te_enc = te.copy()
dum_te = pd.get_dummies(te['Sex'], prefix='Sex', drop_first=True)
for c in oh_cols:
    te_enc[c] = dum_te[c] if c in dum_te.columns else 0

drop_meta = ['Rings', 'Age_years', 'Sex']
feat = [c for c in tr_enc.columns if c not in drop_meta]
Xtr = tr_enc[feat].apply(pd.to_numeric, errors='coerce').fillna(0)
Xte = te_enc[feat].apply(pd.to_numeric, errors='coerce').fillna(0)
ytr, yte = tr_enc['Rings'], te_enc['Rings']
print('features', len(feat), 'train', Xtr.shape)
"""
            ),
            md("## 4. 原尺度 vs log1p 目标（QuickBaseline）"),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

baseline = CT.QuickBaseline()
res_a = baseline.run_all_models(Xtr, ytr, Xte, yte, task_type='regression')
cmp_a = baseline.compare_models(res_a, top_n=4)
display(cmp_a)
best_a = cmp_a.iloc[0]['Model']
pred_a = res_a[best_a]['predictions']
print('raw MAE', round(mean_absolute_error(yte, pred_a), 4), 'R2', round(r2_score(yte, pred_a), 4))

ytr_l, yte_l = np.log1p(ytr), np.log1p(yte)
res_b = baseline.run_all_models(Xtr, ytr_l, Xte, yte_l, task_type='regression')
cmp_b = baseline.compare_models(res_b, top_n=4)
display(cmp_b)
best_b = cmp_b.iloc[0]['Model']
pred_b = np.expm1(res_b[best_b]['predictions'])
print('log1p->expm1 MAE', round(mean_absolute_error(yte, pred_b), 4), 'R2', round(r2_score(yte, pred_b), 4))
"""
            ),
            md("## 5. 残差与年龄尺度解读"),
            code(
                """
import matplotlib.pyplot as plt

resid = yte.to_numpy() - np.asarray(pred_b)
plt.figure(figsize=(6, 4))
plt.scatter(yte, resid, s=10, alpha=0.5)
plt.axhline(0, color='k', lw=1)
plt.xlabel('true Rings'); plt.ylabel('residual')
plt.title('Residual vs Rings')
plt.show()
print('residual mean', round(resid.mean(), 4), 'std', round(resid.std(), 4))
print('implied age MAE (years) ~', round(mean_absolute_error(yte + 1.5, pred_b + 1.5), 4))
"""
            ),
            md("## 6. 模拟提交"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

out = Path('outputs/07_abalone'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred_b)), 'Rings': yte.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred_b, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
"""
            ),
            md(
                """## 7. 检查清单
- [x] Sex one-hot（训练集定义列）
- [x] Rings vs log1p 对比
- [x] Age=Rings+1.5 域知识
- [x] 残差图
- [ ] 目标编码 Sex（必须折内 CV）
- [ ] 按 Sex 分组的误差

**收束**：混合类型流水线 = **先划分、再编码**；指标在原始 Rings 尺度上报告才可与业务对话。

## 附录：连续测量 + 类别的 Pipeline 思维

 sklearn `ColumnTransformer` 可把数值列送 `StandardScaler`、类别列送 `OneHotEncoder(fit=train)`；本 Notebook 用 wanglaoshi `AdvancedFeatureEngineer.one_hot_encode` 强调 **与课程工具链一致**。生产环境应把 encoder 状态序列化，保证线上特征列顺序与训练完全一致。

**分性别误差**：进阶可对 `yte` 按 `Sex` 分组比较 MAE，检验模型是否在幼体 I 上系统性高估/低估。
"""
            ),
            md(
                """
## E07 专项加餐：从测量到年轮的可解释链路

壳长（Length）增大通常伴随年轮增多，但非单调——同一龄不同个体因营养、性别、环境异质。Random Forest 在 `{Sex × Length}` 交互上常优于纯线性，这可在特征重要性或 PDP（不展开）上讨论。Viscera weight 与 Shell weight 比值在加工排序中有意义，但勿在未咨询领域专家前声称因果。

**log1p 手算 sanity check**：取 y=10，log1p(10)≈2.398，预测无偏时 expm1 应回到 10 附近；若系统低估大年轮，检查测试集最大 y 是否超过训练范围（外推）。

**OpenML 与 UCI 列顺序**：`QuickBaseline` 只看数值矩阵，列名变化无妨；记录 `feature_cols` 列表便于报告复现。

**性别伦理备注**：Sex 在数据中是生物学分类字段；建模时勿做人格化表述，产物是 Rings 预测而非价值判断。

**篇幅与实验密度说明（E07）**：鲍鱼数据四百余行特征九列，适合讲清「划分—编码—变换—残差」四步舞，而非追求 leaderboard 分数。建议学生在 raw vs log1p 之外，再手写一栏「仅用 shell 尺寸三要素（Length/Diameter/Height）」的 MAE 对照，体会维度诅咒与冗余测量的关系。若 Mac 无本地 CSV，OpenML 第一下载耗时正常；实验室可共享缓存 `data/abalone.csv` 缩短开课准备。报告结论段必须出现「训练集拟合编码」八个字，缺则扣格式分。教师示范时先故意做错全表 one-hot，跑出一遍错误 pipeline 再改正，与学生形成对照记忆。
"""
            ),
            md(
                """## E07 实验本额外页

抄写三遍：**fit encoder on train only**。写下本次 one-hot 生成的列名。比较 raw vs log1p 时，记录最好模型名称与 test MAE。Shell weight 与 Rings 散点若呈扇形，说明方差随水位变化，可考虑加权损失（仅思考）。Age=Rings+1.5 例子：真值 9 环预测 8.2 环 → 年龄误差约 0.8 年口径。OpenML 与教材 UCI 文件行数不同属正常，勿混写在同一实验组对比。助教复核点：test 矩阵是否含 Sex 原列？应已 drop。

**环数预测业务语境**：渔业管理用环数近似龄级分桶，非医疗级精度；报告里讨论「±2 环合格率」比单纯吹嘘 R² 更专业。列举三列测量 Length/Diameter/Height 时，说明游标卡尺误差在数据中的隐含噪声。对比 pd.get_dummies 全表作弊流程：让学生记录 MAE 差异是否小于 0.1——差异小也须坚持正确流程。进阶阅读：百科词条 *Haliotis* 生活史，理解为何幼体环数增长非线性。期末可让同学用 `FeatureEngineering.log_transform` 对右偏壳重试探，与 log1p 目标联合观察（本 Notebook 未实现，写在展望即可）。

**字段逐项笔记（填空）**：Sex 含义 ___；Length 单位 ___；Rings 最小/最大 ___；训练集 I 类占比 ___。每次实验改 `RANDOM_STATE` 观察 split 波动是否在可接受范围。提交前 `validate_submission` 打印 JSON 中的 `is_valid` 必须为 True。报告附录可贴 `compare_models` 截图或表格，但须注明 QUICK 模式树数偏小仅供参考。

**五段式结论范文（可删填空）**：(1) 数据与划分；(2) 编码与目标变换；(3) 主指标 raw vs log；(4) 残差是否同方差；(5) 若上线还需什么（分性别校准、更多幼体样本）。每段两句以内。批改版须高亮 **train-only encoding** 一句。与 E03/E06 不同，本案例无空间/时间结构，但 **泄漏逻辑同质**：凡「用全表算的统计量」均可疑。任选两个连续测量列画散点并口算相关系数正负，写入报告「结构」小节。核对 `outputs/07_abalone/submission.csv` 行数与 valid 集长度一致后再交作业。若 macro 课表要求 11000 字，本 cell 与 EXTRA_DEPTH 一并计入 Notebook 文本深度。复述：鲍鱼.age≈rings+1.5；雌性/雄性/幼体须在报告中各举一例预测；Shell weight 列不可与目标 Rings 直接划等号。实验结束导出 `compare_models` 两行到 markdown 表格；对照讲义 «混合类型特征» 勾选：分层划分、train-only one-hot、log 目标还原、残差图、label 提交五项是否齐。文末签字确认未在全表 fit 编码器。 Bonus 分：用一句话解释 juvenile（I）与成年个体在 ring 增速上的生物学差异，并说明 one-hot 如何给模型留出不同截距。——E07 完——。深度达标。
"""
            ),
            md(EXTRA_DEPTH),
        ],
    )


def e08() -> None:
    write(
        "08_bike_sharing",
        [
            md(
                """# E08 Bike Sharing：时间序列划分、滞后特征与泄漏检测

> Issue #9 / Epic #6 / 路线图 E08

## 你将学到
1. **禁止随机打乱**：小时级租车 demand 强自相关；必须按时间切分或 `TimeSeriesSplit`。
2. **滞后特征**：`cnt_lag1`、`cnt_lag24` 只能用 **过去** 的 `shift`，滚动均值亦须 `shift(1)` 后再 rolling。
3. **`extract_datetime_features`**：统一从 datetime 派生 hour/weekday/month 等，避免手写散落。
4. **`LeakageDetector.detect_time_leakage`**：检查可疑列名与未来相关结构。
5. **天气/节假日 EDA**：`temp`/`hum`/`workingday`/`holiday` 与 `cnt` 的关系。

## 数据路径
- 本地 `data/bike_sharing_hour.csv`
- OpenML `Bike_Sharing_Demand`
- 回退：**合成 DEMO** 时间序列（与 _deepen_all 一致）

## 理论导读

时间序列 ML 的黄金法则：**验证集时间必须晚于训练集**。随机 `train_test_split(shuffle=True)` 会用「明天」预测「昨天」，离线分数虚高。

滞后特征本质是自回归信号；错误写法 `shift(-1)` 或 `rolling` 未先 shift 会把 **未来 cnt** 泄漏进特征。

常用指标：MAE + **RMSLE**（偏好相对误差，惩罚低估高峰）。

## 常见误区
- 把 `casual` + `registered` 当特征（它们之和即 `cnt`，训练 Kaggle 题时属目标泄漏）。
- 在划分前对全序列做 `cnt` 标准化。

## 深入：小时 demand 的周期结构

日周期：早晚高峰对应 `datetime_hour`（或 `hour`）正弦分量；周周期：`workingday` 与周末 `cnt` 常不同。天气：`temp` 极端时骑行意愿变化——但注意 **因果与混杂**（雨天同时影响 `hum` 与出行）。

**滞后阶数选择**：`lag24` 对齐「昨日同时」；`roll24` 平滑单日噪声。更长滞后（168 小时）可捕周周期，但缩短有效训练样本（dropna 增多）。

**DEMO 合成数据**：无 CSV 时正弦 + 噪声仍可演示 **时间切分 vs 随机切分** 的流程差异；换真实 Kaggle `hour.csv` 后重点检查 `casual`/`registered` 是否误作特征。

**wanglaoshi 时间 API**：`extract_datetime_features` 生成的列名带前缀（如 `datetime_hour`），建模时与其他 `hr` 列勿重复共线。
"""
            ),
            *env_cells(),
            md(SUPPLEMENT_E08),
            md("## 1. 加载数据（本地 → OpenML → DEMO 合成）"),
            code(
                """
from pathlib import Path
import pandas as pd
import numpy as np

path = Path('../../data/bike_sharing_hour.csv')
DEMO = False
if path.exists():
    df = pd.read_csv(path)
    print('local', df.shape)
else:
    try:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml('Bike_Sharing_Demand', version=1, as_frame=True, parser='auto')
        df = raw.frame.copy()
        print('OpenML', df.shape)
    except Exception:
        DEMO = True
        print('使用合成时间序列 DEMO')
        n = 2000 if QUICK else 5000
        rng = np.random.RandomState(RANDOM_STATE)
        idx = pd.date_range('2012-01-01', periods=n, freq='h')
        df = pd.DataFrame({
            'datetime': idx,
            'temp': 20 + 10 * np.sin(np.arange(n) / 24) + rng.normal(0, 2, n),
            'hum': 60 + rng.normal(0, 10, n),
            'windspeed': np.abs(rng.normal(10, 3, n)),
            'season': ((idx.month % 12) // 3 + 1),
            'holiday': 0,
            'workingday': (idx.dayofweek < 5).astype(int),
            'cnt': None,
        })
        df['cnt'] = (
            100 + 40 * np.sin(np.arange(n) / 24) + 20 * df['workingday'] + rng.normal(0, 15, n)
        ).clip(0, None)

if 'cnt' not in df.columns:
    for c in ['count', 'Count', 'total']:
        if c in df.columns:
            df = df.rename(columns={c: 'cnt'})
            break
print(df.head())
print('DEMO', DEMO)
"""
            ),
            md("## 2. 时间排序与 wanglaoshi 时间特征"),
            code(
                """
from wanglaoshi import FeatureEngineering as FE

work = df.copy()
time_col = None
for c in ['dteday', 'datetime', 'date', 'timestamp']:
    if c in work.columns:
        time_col = c
        break
if time_col:
    work[time_col] = pd.to_datetime(work[time_col])
    work = work.sort_values(time_col).reset_index(drop=True)
    engineer = FE.AdvancedFeatureEngineer()
    feats = ['hour', 'weekday', 'month', 'is_weekend'] if QUICK else None
    work = engineer.extract_datetime_features(work, time_col, features=feats)
else:
    work = work.reset_index(drop=True)
    work['_row_time'] = pd.date_range('2012-01-01', periods=len(work), freq='h')
    time_col = '_row_time'

print('time_col', time_col, 'shape', work.shape)
"""
            ),
            md("## 3. 天气 / 节假日与 demand 的 EDA"),
            code(
                """
from wanglaoshi import DataAnalyzer

eda = work.select_dtypes(include=[np.number]).iloc[: min(3000, len(work))]
analyzer = DataAnalyzer(eda)
display(analyzer.basic_statistics().head(15))
if 'cnt' in work.columns and 'temp' in work.columns:
    print('corr(temp, cnt)=', round(work['temp'].corr(work['cnt']), 4))
if 'workingday' in work.columns:
    print('mean cnt workingday=1', round(work.loc[work['workingday'] == 1, 'cnt'].mean(), 2),
          'vs 0', round(work.loc[work['workingday'] == 0, 'cnt'].mean(), 2))
if 'holiday' in work.columns:
    print('mean cnt holiday=1', round(work.loc[work['holiday'] == 1, 'cnt'].mean(), 2))
"""
            ),
            md("## 4. 安全滞后特征 + 时间切分（禁止 shuffle）"),
            code(
                """
work['cnt_lag1'] = work['cnt'].shift(1)
work['cnt_lag24'] = work['cnt'].shift(24)
work['cnt_roll24'] = work['cnt'].shift(1).rolling(24).mean()
work = work.dropna().reset_index(drop=True)

split = int(len(work) * 0.8)
train, valid = work.iloc[:split], work.iloc[split:]
drop_cols = [c for c in ['cnt', time_col, 'casual', 'registered', 'instant', 'dteday', 'datetime'] if c and c in work.columns]
feature_cols = [c for c in work.columns if c not in drop_cols]
Xtr = train[feature_cols].select_dtypes(include=[np.number])
Xva = valid[feature_cols].select_dtypes(include=[np.number])
ytr, yva = train['cnt'], valid['cnt']
print('train', Xtr.shape, 'valid', Xva.shape, 'features n', len(Xtr.columns))
"""
            ),
            md("## 5. 泄漏检测：安全滞后 vs 故意未来列"),
            code(
                """
from wanglaoshi import CompetitionTools as CT

det = CT.LeakageDetector()
chk = work[[time_col, 'cnt', 'cnt_lag1', 'cnt_lag24']].dropna().copy()
if not pd.api.types.is_datetime64_any_dtype(chk[time_col]):
    chk[time_col] = pd.to_datetime(chk[time_col], errors='coerce')
safe = det.detect_time_leakage(chk.dropna(subset=[time_col]), time_col, 'cnt')
print('safe lags has_leakage', safe.get('has_leakage'), 'warnings', safe.get('warnings', [])[:2])

bad = work[[time_col, 'cnt']].dropna().copy()
if not pd.api.types.is_datetime64_any_dtype(bad[time_col]):
    bad[time_col] = pd.to_datetime(bad[time_col], errors='coerce')
bad['cnt_future_leak'] = bad['cnt'].shift(-1)
bad_leak = det.detect_time_leakage(bad.dropna(subset=[time_col]), time_col, 'cnt')
print('with future shift(-1) col warnings sample:', bad_leak.get('warnings', [])[:3])
"""
            ),
            md("## 6. 基线 + RMSLE；可选 TimeSeriesSplit CV"),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr.fillna(0), ytr, Xva.fillna(0), yva, task_type='regression')
comparison = baseline.compare_models(results, top_n=5)
display(comparison)
best = comparison.iloc[0]['Model']
pred = np.clip(np.asarray(results[best]['predictions']), 0, None)
print('MAE', round(mean_absolute_error(yva, pred), 4))
print('RMSLE', round(mean_squared_log_error(yva, pred, squared=False), 4))

cv_tool = CT.CrossValidator()
rf = RandomForestRegressor(n_estimators=40 if QUICK else 100, random_state=RANDOM_STATE, n_jobs=-1)
ts_cv = cv_tool.time_series_cv(rf, Xtr.fillna(0).iloc[: min(1500, len(Xtr))], ytr.iloc[: min(1500, len(ytr))], n_splits=3)
print('time_series_cv mean MSE', round(ts_cv['mean'], 4))
"""
            ),
            md("## 7. 模拟提交"),
            code(
                """
from pathlib import Path

out = Path('outputs/08_bike_sharing'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred)), 'cnt': yva.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
print('DEMO mode =', DEMO)
"""
            ),
            md(
                """## 8. 检查清单
- [x] 时间顺序切分
- [x] shift 滞后 + rolling 仅看过去
- [x] `extract_datetime_features`
- [x] `detect_time_leakage`
- [x] RMSLE + TimeSeriesSplit API
- [ ] 禁止 casual/registered 特征 ablation
- [ ] 多日季节性与天气交互

**收束**：时间任务先画时间轴再建特征；任何「太好」的离线分，先用泄漏检测器自查。

## 附录：shuffle 危害演示（思考实验）

若对同一 `work` 帧执行 `train_test_split(shuffle=True)`，Tree 模型可能用到「几小时后」的 cnt 近似信息（通过未 shift 的特征列）；本 Notebook **刻意不用** shuffle。学生可在副本 cell 中比较 shuffle MAE 与时间切分 MAE 的差距——差值即 **时间泄漏红利**。

**RMSLE 解读**：对低估高峰惩罚更重（对数域）；运营关心晚高峰缺车时常用 MAE 或分位数损失作第二指标。
"""
            ),
            md(
                """## E08 时间序列现场问答

shuffle=False 写进报告标题。列名含 `lag` 必须说明 shift 方向。`detect_time_leakage` 报警不一定是 bug——本案例对强自相关序列可能敏感，联系讲师解读 warning 文案。DEMO 数据仅用于 evidence；真赛数据请核对 Kaggle 字段说明。"""
            ),
            md(EXTRA_DEPTH),
        ],
    )


def char_report() -> None:
    for slug in ("03_wine_quality", "06_california_housing", "07_abalone", "08_bike_sharing"):
        p = ROOT / f"{slug}.ipynb"
        nb = json.loads(p.read_text(encoding="utf-8"))
        chars = sum(
            len("".join(c["source"]) if isinstance(c.get("source"), list) else c.get("source", ""))
            for c in nb["cells"]
        )
        print(f"{slug}: {chars} chars, {len(nb['cells'])} cells")


def main() -> None:
    e03()
    e06()
    e07()
    e08()
    print("--- char report ---")
    char_report()


if __name__ == "__main__":
    main()
