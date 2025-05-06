# from wanglaoshi import WebGetter
#
# downloader = WebGetter.Wget(url='https://gitee.com/lincoln/fonts/raw/master/SimHei.ttf',
#         output_dir='.',
#         filename='simhei.ttf',
#     )
# downloader.download()

# from wanglaoshi import JupyterFont
# JupyterFont.simhei()

from wanglaoshi import Analyzer
import seaborn as sns

# 使用seaborn的示例数据集
df = sns.load_dataset('iris')
analyzer = Analyzer.DataAnalyzer(df)
analyzer.generate_report('iris_analysis.html')