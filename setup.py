from setuptools import setup, find_packages, Command
import os
import sys

about = {}
# 与 README 中的版本历史保持一致
about['__version__'] = '1.15.1'
about['__project_name__'] = 'wanglaoshi'

class UploadCommand(Command):
    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # 故意不再一键清理+上传+commit+push：失败难定位，也易误推。
        # 分步流程见 docs/RELEASE.md
        self.status('UploadCommand 已停用（请分步发布）')
        print('按 docs/RELEASE.md：')
        print('  1) ./scripts/release_check.sh')
        print('  2) rm -rf build dist *.egg-info && python setup.py sdist bdist_wheel')
        print('  3) twine check dist/* && twine upload dist/*')
        print('  4) git tag v{0} && git push origin v{0} && git push'.format(about['__version__']))
        sys.exit(2)

setup(
    name=about['__project_name__'],  # 包的名称
    version=about['__version__'],  # 版本号
    packages=find_packages(),  # 自动找到所有模块
    include_package_data=True,
    install_requires=[         # 默认课堂核（与 requirements/core.txt 对齐）
        'rich',
        'requests',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'missingno',
        'jinja2',
        'tqdm',
        'statsmodels',
        'scikit-learn',
    ],
    extras_require={
        # 说明见 docs/DEPENDENCIES.md；保持少量分组，避免维护成本过高
        'excel': ['openpyxl'],
        'jupyter': ['jupyter', 'ipykernel'],
        'optuna': ['optuna'],
        # Phase 4 场景 extras（代码默认仍随核心安装；此处是可选依赖门禁）
        'compete': ['optuna'],  # HyperparameterOptimizer.bayesian_optimize
        'business': ['openpyxl'],  # 业务流水线读 xlsx / Excel intake
        'dev': ['pytest', 'pytest-cov', 'flake8'],
        'book': [
            # 1.x = Sphinx 经典 Jupyter Book；2.x 换成 MystMD，配置不兼容
            'jupyter-book>=1.0.0,<2',
            'jupyter',
            'ipykernel',
        ],
        'all': ['openpyxl', 'jupyter', 'ipykernel', 'optuna'],
    },
    author='WangLaoShi',  # 作者
    author_email='ginger547@gmail.com',  # 邮箱
    description='A utility module for DA and ML and DL tasks',  # 简短描述
    long_description=open('README.md').read(),  # 从 README.md 文件读取详细描述
    long_description_content_type='text/markdown',  # README.md 文件格式
    url='https://github.com/wanglaoshi/wanglaoshi-pypi',  # 项目的 GitHub 链接
    classifiers=[   # 分类信息
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Python 版本要求
    cmdclass={
        'upload': UploadCommand,
    }
)