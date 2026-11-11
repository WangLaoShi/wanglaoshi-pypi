# 案例数据验收口径

## 两种证据，不要混用

| 证据 | 证明什么 | 不证明什么 |
| --- | --- | --- |
| `ADVANCED_EVIDENCE.md`（inprocess） | Notebook 能跑完、主路径无未捕获异常 | 是否使用真实外部数据；指标是否手算一致 |
| 业务手算 / 契约测试 | 口径与阻断行为正确 | 22 课全部真实数据 |

## 发布验证（真实数据）最低字段

每个声称「真实数据通过」的案例应记录：

1. **data_mode**：`real` | `demo` | `synthetic`
2. **source**：URL / 本地路径 / 包内资源
3. **sha256**（或等价指纹）与 **n_rows**
4. **业务断言**：至少一条与案例目标相关的数值/形状检查
5. **命令与日期**

若运行中发生 DEMO/合成回退：`data_mode` 必须记为 `demo`/`synthetic`，且**不得**计入「真实数据验收通过」。

## 日常演示

允许合成与 DEMO；教学烟测继续用 `scripts/run_advanced_evidence.py`。

## 脚本如何记录

`run_advanced_evidence.py` 在每个案例结果里写入：

- `data_mode`：若 Notebook 命名空间有 `DEMO=True` → `demo`；`DEMO=False` → `real_or_local`（或 Notebook 自设的 `data_mode`）；否则 `unspecified`
- 可选透传：`data_source` / `n_rows` / `data_sha256`（需 Notebook 自行赋值）

汇总字段 `demo_or_synthetic` / `real_or_local_claimed` 仅作分类，**不得**把 `demo` 记进「真实数据通过」。发布级真实验收仍须按上表手填或单独流水线产出。
