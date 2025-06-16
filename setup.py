from setuptools import setup, find_packages, Command
import os
import sys

about = {}
about['__version__'] = '0.10.11'
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
        self.status('Clean build,dist,wanglaoshi.egg-info directory …')
        os.system('rm -rf ./build ./dist ./*.egg-info')
        self.status('Building Source and Wheel distribution…')
        os.system('{0} setup.py sdist bdist_wheel'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git add .')
        os.system('git commit -m v{0} .'.format(about['__version__']))
        os.system('git push --tags')
        os.system('git push')
        sys.exit()

setup(
    name=about['__project_name__'],  # 包的名称
    version=about['__version__'],  # 版本号
    packages=find_packages(),  # 自动找到所有模块
    install_requires=[         # 依赖的库
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
        # 在这里列出其他依赖的库
    ],
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