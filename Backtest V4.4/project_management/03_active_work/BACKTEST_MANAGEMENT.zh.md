# 回测管理

## 必做操作

每次启动回测前，在本文档追加一行，记录：

- 品种；
- 本次回测实际读取的数据文件；
- 回测区间的开始时间；
- 回测区间的结束时间。

起止时间以本次回测实际使用的区间为准。数据文件中仅用于预热或尚未使用的后续数据不计入区间。回测中途停止时仍保留该行，确保每次启动都有记录。生成 HTML 与这项记录无关。

原始计算闭合后，在本次结果包的 `EXPERIMENT.md` 和 `evaluation_manifest.json` 中登记结果。目录由品种和精确评价区间确定。记录候选集来源、计划、完成状态、参数汇总、不可变逐笔记录、逐笔入口，以及使用该结果包的比较方案。实验角色只写入这些记录，不改结果包目录名。

新运行完成后，填写 `runtime_inputs\templates\EVALUATION_PACKAGE_SPEC.template.json`，再调用 `tools\register_v4_4_evaluation_package.py`。声明文件把运行专用汇总列映射为共享参数／指标字段。已有结果包清单保持不可变；同一品种和区间的另一组候选，需要通过经过确认的发布改动，在同一结果包内增加候选集产物。

这套规则适用于所有品种和评价区间。已就绪品种配置、完整数据契约、精确起止时间和已授权实验计划共同构成执行边界；管理规则不再依赖 K200 专用命名。

## 回测记录

| 品种 | 数据文件 | 回测开始时间 | 回测结束时间 |
| --- | --- | --- | --- |
| K200 时间迁移 R1 | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-08 23:52:15` | `2026-07-17 05:59:45` |
| K200 时间迁移 R2 | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-20 08:45:00` | `2026-07-25 05:59:45` |
| K200 时间迁移 R3 | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-27 08:45:00` | `2026-08-01 05:59:45` |
| K200 时间迁移 R4 最终留出 | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-08-03 08:45:00` | `2026-08-07 03:21:45` |
| K200 时间迁移全测试期描述性重放 | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-08 23:52:15` | `2026-08-07 03:21:45` |
| SI 精确迁移新增 100 组时间迁移候选 | `D:\Code\data\ibkr\SImain\SImain_15s_20260128_20260223_session_filled.csv` | `2026-01-29 00:00:00` | `2026-02-23 23:59:45` |
| K200 测试集重放合并后的 350 组候选 | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-08 23:52:15` | `2026-08-07 03:21:45` |
| K200 当前优参数 100 组初步后续重放 | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-08 23:52:15` | `2026-08-07 03:21:45` |
| K200 当前优参数训练期成本后为正 100 组修正重放 | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-08 23:52:15` | `2026-08-07 03:21:45` |
