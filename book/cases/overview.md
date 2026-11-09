# 进阶案例总览

本部分的每一课都是 **真实 `.ipynb` 渲染进学习手册** 的结果。源文件在仓库：

`examples/advanced/*.ipynb`

## 数据可得性

- **可直接跑（sklearn / seaborn）**：Iris、Titanic、Breast Cancer、California Housing、Digits  
- **可能需本地下载到 `data/`**：Wine Quality、Fraud、Abalone、Bike、Mall、Adult、House Prices（Notebook 内通常有 DEMO 回退）

请遵守各数据集许可与竞赛条款。

## 本地打开（可编辑复现）

```bash
python -m pip install -e ".[jupyter]"
jupyter lab examples/advanced/
```

## 批量证据

{doc}`../reference/evidence`
