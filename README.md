# ATR Stop SSV2 4H 审计库

本仓库当前是 ATR Stop SSV2 4 小时策略的审计输入归档，不包含可直接运行的回测入口或交易执行器。

文件职责索引见 [SCRIPT_CATALOG.md](SCRIPT_CATALOG.md)。

## 文件说明

| 文件 | 作用 |
|---|---|
| `BTCUSDT_4h_real_binance.csv` | BTCUSDT 4 小时 K 线审计数据集，供外部回测或参数复核使用。 |
| `config/optimized_params.yaml` | 参数建议种子；包含多头信号、空头信号和 shooting-star 过滤条件。 |

## 使用边界

参数文件只是审计输入，不代表已验证为当前最优，也不会自动下单。正式回测时应记录数据截止时间、手续费、滑点、未收盘 K 线处理方式和样本外区间。
