# 发布案例真实数据验收

- 生成时间(UTC): `2026-11-11T04:28:29.279054+00:00`
- 口径见 [`DATA_ACCEPTANCE.md`](DATA_ACCEPTANCE.md)

| Case | status | data_mode | n_rows | source |
| --- | --- | --- | --- | --- |
| `05_credit_card_fraud` | **blocked** | `missing` | — | expected:data/creditcard.csv (Kaggle creditcardfraud) |
| | note | 需人工放置授权 CSV；禁止无授权自动下载 | | |
| `08_bike_sharing` | **ready** | `real` | 17379 | OpenML:Bike_Sharing_Demand/v2 → data/bike_sharing_hour.csv |
| | sha256 | `375d870c6a8adb4c…` | | |

复现：
```bash
python scripts/fetch_release_case_data.py
# 欺诈检测：将 Kaggle creditcard.csv 放到 data/creditcard.csv 后再跑
```
