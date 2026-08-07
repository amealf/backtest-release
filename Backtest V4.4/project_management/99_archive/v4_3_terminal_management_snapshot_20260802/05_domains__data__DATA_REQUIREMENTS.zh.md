# 数据要求

## 范围

该可选模块存在，是因为 **Backtest V4.3 Research** 包含需要维护的数据来源、结构、迁移或管线。

## 来源与契约登记

| 数据集或契约 | 负责人／来源 | 结构或格式 | 有效范围 | 状态 |
| --- | --- | --- | --- | --- |
| K200 15 秒 bar | `runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | CSV；有序 OHLCV、成交笔数、synthetic 标记 | 哈希绑定来源范围 | 当前 |
| 即时恢复清洗审计 | `runtime_inputs\market_data\data_preparation_audit.json` | JSON | 必须绑定来源哈希 | 当前 |
| V4.3 准备契约 | `runtime_inputs\data_preparation\data_preparation_manifest.json` | JSON schema v4；相对产物路径与受支持策略声明 | 当前来源哈希与准备代码 | 当前 |
| Baseline 标记原子／事件 | 准备 manifest 产物 | CSV／JSON；策略中性 `baseline_excluded` 与 `eligible_if_excluding_marked` | 与来源 datetime 一对一对齐 | 当前 |

## 质量与血缘

- 15 秒来源必须包含 datetime、OHLC、volume、trade count 与 synthetic-empty-bar 状态，并匹配绑定哈希。
- 准备产物必须按 datetime 一对一对齐，并从自身 manifest 相对解析。
- 准备层不选择 baseline 采样策略。`all_window` 在合格判断中忽略标记；`exclude_marked` 使用中性标记。两种策略对信号、成交、平仓与图表保留相同来源行。
- 来源每个 datetime 只能有一行。重复时间戳会使引擎停止；当前没有 keep-first／keep-last 合并政策。
- Synthetic 状态与 `baseline_excluded` 相互独立；当前两种策略都不会移除全部 synthetic bar。
- 历史绝对路径可以保留为来源记录，但活跃运行代码不能访问它们。
- 用户要求 ZIP 时，包含哈希绑定的 15 秒 OHLC 来源；排除 campaign 根目录、原始 batch、阶段 summary／trades、chunks、snapshots 与其他运算结果载荷，同时包含打包契约明确要求的分析报告文件。

## 验证登记

| 检查 | 范围 | 失败动作 | 证据 |
| --- | --- | --- | --- |
| 来源 SHA-256 | 完整行情来源 | 执行前停止 | 运行／来源 manifest |
| 准备身份与产物 SHA-256 | manifest 与过滤文件 | 执行前停止 | 引擎与准备验证 |
| 时间戳对齐 | 来源与过滤原子 | 执行前停止 | 引擎验证 |
| 来源 datetime 唯一 | 完整行情来源 | 执行前停止 | 准备层与引擎验证 |
| 受支持／默认 baseline 策略 | 准备契约与引擎／计划 | 执行前停止 | Runner 验证 |
| 真实成交执行 bar | 开仓与信号驱动平仓 | 不匹配时停止交付 | 引擎／逐笔回归测试 |

## 维护规则

长期来源、结构、契约、质量阈值、血缘规则、访问政策、保留期限、迁移、备份或恢复流程变化后更新。
