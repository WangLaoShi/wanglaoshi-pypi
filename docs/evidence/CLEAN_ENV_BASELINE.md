# 干净环境测试基线

| 字段 | 值 |
| --- | --- |
| 状态 | **passed** |
| UTC 时间 | 2026-11-11T02:01:22Z |
| 耗时 | 96s |
| 模式 | `full` |
| Python | 3.9.6 |
| 平台 | macOS-27.0.1-arm64-arm-64bit |
| 命令 | `python -m venv .venv-clean-baseline && pip install -e ".[dev]" && pytest` |
| 退出码 | 0 |

## 已安装关键包

- `wanglaoshi==1.15.1`
- `numpy==2.0.2`
- `pandas==2.3.3`
- `scipy==1.13.1`
- `scikit-learn==1.6.1`
- `matplotlib==3.9.4`
- `statsmodels==0.14.6`
- `pytest==8.4.2`

## 说明

- 本文件由 `scripts/verify_clean_env.sh` 生成；**通过才可作为 F01 验证证据**。
- 本机 Anaconda / 污染环境上的 `pytest` 收集失败，**不能**代替本基线。
- venv 目录 `.venv-clean-baseline/` 默认不提交；可删可重建。
