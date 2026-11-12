# 一键云端运行（Binder）

不想本地装环境时，可用 MyBinder 打开本仓库的 JupyterLab。

## 启动

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/WangLaoShi/wanglaoshi-pypi/master)

或打开：

`https://mybinder.org/v2/gh/WangLaoShi/wanglaoshi-pypi/master`

首次构建可能较慢（要装依赖）；成功后进入 JupyterLab。

## 建议打开的文件

| 路径 | 内容 |
| --- | --- |
| `examples/advanced/01_iris.ipynb` | 第一课 |
| `examples/business_pipeline_course.ipynb` | 业务流水线专章 |
| `examples/business_intake_demo.ipynb` | 业务短 demo |

## 手册页面上的火箭按钮

本 Jupyter Book 已启用 Binder 启动按钮：在 **Notebook 课页** 顶部可尝试一键打开对应 notebook（需 Pages 部署完成且路径正确）。

## 说明

- Binder 是公共免费资源，不适合跑超大数据或长时间训练  
- 环境定义在仓库 `binder/`（`environment.yml` + `postBuild`）  
- 若徽章失败，多半是依赖构建问题，可改回本地：`pip install -e ".[jupyter]"`  
