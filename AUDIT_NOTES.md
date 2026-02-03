# ATRSTOP+SSV2 4H 审计说明

**仓库**：https://github.com/ccnzdd1-gif/atrstop-ssv2-4h-audit

## 目标
针对“收盘价触发 + 滑点/延迟”模式下，原本盈利但现变为负贡献的指标进行参数优化与规律统计，判断是否能恢复为正期望。

## 数据
- 文件：`BTCUSDT_4h_real_binance.csv`
- 频率：4H
- 时间范围：约 2021-02-05 ~ 2026-02-04

## 基线脚本
- 文件：`backtest_milestone_atrstop_ssv2_coexist__20260203_20_close_slip_latency.py`
- 说明：ATRSTOP Milestone + Shooting Star V2（多空可共存），收盘价触发 + 滑点/延迟。

## 关注的负贡献指标
（按当前基线统计，收盘价触发后转负）
- **sig_sma50_b**（多）
- **sig_short**（空）
- **sig_shooting_star_v2**（空）

平均单笔净收益（%）参考：
- sig_sma50_b：-0.26%
- sig_short：-0.18%
- sig_shooting_star_v2：-0.57%

## 审计建议方向（可选）
1) 调整触发条件阈值（RSI/ADX/EMA斜率/BOLL宽度）
2) 加入确认/过滤（2~3 根确认、量能过滤、趋势过滤）
3) 调整出场（SL/TP/追踪止损）
4) 细分波动率 regime（如 vol_ratio 分层参数）

## 反馈格式建议
- 给出每个指标的优化参数与回测结果（Total / MaxDD / Sharpe）
- 说明是否具备统计显著性 & 失败原因

谢谢！
