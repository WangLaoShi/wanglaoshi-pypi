from setuptools import setup, find_packages

setup(
    name='wanglaoshi',  # 包的名称
    version='0.0.1',  # 版本号
    packages=find_packages(),  # 自动找到所有模块
    install_requires=[         # 依赖的库
        'numpy',
        'pandas',
        'scikit-learn',
        # 在这里列出其他依赖的库
    ],
    author='WangLaoShi',  # 作者
    author_email='ginger547@gmail.com',  # 邮箱
    description='A utility module for ML and DL tasks',  # 简短描述
    long_description=open('README.md.md').read(),  # 从 README.md 文件读取详细描述
    long_description_content_type='text/markdown',  # README.md 文件格式
    url='https://github.com/wanglaoshi/wanglaoshi',  # 项目的 GitHub 链接
    classifiers=[   # 分类信息
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Python 版本要求
)