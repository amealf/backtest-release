# 代码架构

## 范围

本文档记录 **Backtest V4.3 Research** 当前代码结构、执行流程、组件职责与系统边界。只记录经过代码、配置、生成产物或用户明确确认支持的架构事实。

## 架构状态

V4.3 是一个离线 Windows／Python 回测管线，使用仓库本地运行输入并生成 HTML 审阅产物。

## 当前系统流程

```text
runtime_inputs + 中性 baseline 标记 + 情景定义 + 已审阅参数计划
  -> v4_3_engine.py
  -> run_v4_3_resumable_campaign.py（原始不可变 batch／stage）
  -> analyze_v4_3_scenario_3_stage.py
       -> build_v4_3_review_delivery.py（固定逐笔模板，四线程）
       -> build_v4_3_combined_union_analysis.py（稳定累计路径）
```

```text
共享场景流程：
D:\Code\backtest-release\shared_tools\scenario_manager\index.html
  -> 浏览器本地跨版本场景库
  -> 已审阅的版本专属情景定义／计划输入
```

## 组件职责登记

| 路径或组件 | 职责 | 输入 | 输出 | 状态与证据 |
| --- | --- | --- | --- | --- |
| `runtime_inputs\RUNTIME_INPUTS.json` | 绑定可移动运行资产。 | 仓库相对文件 | 哈希／大小闭合 | 当前 |
| `data_preparation\prepare_dataset.py` | 生成策略中性的 baseline 标记与相对路径准备 manifest。 | K200 15 秒 bar 与清洗审计 | 标记原子／事件／报告／manifest | 当前 |
| `code\v4_3_engine.py` | 按坐标策略选择 baseline 合格原子，再计算信号、开仓、pending exit、成交与交易。 | 已准备 bar 与单个坐标 | 带策略身份和审计的逐笔记录 | 当前 |
| `code\run_v4_3_resumable_campaign.py` | 验证计划身份并执行可恢复 batch。 | 已审阅计划 | 原始阶段产物与 manifest | 当前 |
| `code\analyze_v4_3_scenario_3_stage.py` | 验证闭合阶段证据并计算审阅表。 | 已闭合原始阶段 | 分析 CSV／manifest／HTML | 当前 |
| `code\build_v4_3_review_delivery.py` | 使用指定历史模板生成逐笔审阅。 | summary、trades、本地模板资产 | 懒加载逐笔 chunk 与 HTML | 当前 |
| `code\build_v4_3_combined_union_analysis.py` | 反连接兼容已完成阶段并发布稳定路径。 | 已完成阶段分析 | 累计总入口／逐笔交付 | 当前 |
| 跨版本场景管理器 | 复用指定市场选定器界面，以自动连续的 `情景N` 名称保存多区间场景，通过两行按钮选取，并支持清空、归档和恢复。 | 内置 K200 15 秒选定器数据或用户 CSV／TSV | 浏览器本地共享场景库 | 当前共享工具 |

## 依赖与调用关系

- 活跃运行依赖为 Python、NumPy、pandas、仓库本地数据／模板和 Windows `msvcrt` 锁。
- Node／Playwright 是 Dashboard 与浏览器 QA 的验证依赖。
- `runtime_inputs\provenance` 内的路径不参与活跃调用。

## 接口与边界

- 原始计算、阶段分析和累计发布分别拥有独立 writer lock 与输出所有权。
- 开仓／平仓审计列是分析与逐笔 HTML 的长期接口。
- 每个原始阶段使用一种 baseline 采样策略；fingerprint 与 stage／batch／completion manifest 绑定解析后的策略。
- 累计生成器可以纳入任一受支持策略的兼容阶段，并发布策略到 strategy／result identity 的映射；排名不能跨策略分区。
- 不同阶段使用不同原始与派生输出根目录，因此一个阶段交付时，另一个阶段可以计算。
- 文档或验证命令不会启动回测。
- 共享场景管理器承担人工编辑与查询；启动回测前仍需形成已审阅、不可变、版本专属的情景定义与身份。

## 架构变更日志

| 日期 | 变化 | 证据 | 状态 |
| --- | --- | --- | --- |
| 2026-08-01 | 建立中性代码架构记录。 | 项目管理初始化 | 检查项目代码前保持待确认 |
| 2026-08-01 | 将 V4.3 运行输入绑定到当前文件夹，并增加 synthetic-exit pending 成交。 | 当前代码、manifest 与测试 | 当前 |
| 2026-08-02 | 将准备标记改为策略中性，并把所选 baseline 策略传播到计算、身份、manifest、分析与固定模板交付。 | 当前代码、重建后的 manifest 与测试 | 当前 |
| 2026-08-02 | 建立版本中立场景管理入口，随后优化为自动连续编号、两行按钮选择与默认 15 秒显示，同时保留历史结果快照。 | 共享 HTML 与桌面／手机浏览器 QA | 当前 |

## 维护规则

代码入口、组件、依赖或调用关系、数据流、接口边界、生成产物或组件职责被新增、改名、退役或发生重要变化时，更新本文档。权威路径变化时更新 `SOURCE_OF_TRUTH.en.md`，行为变化时更新 `CURRENT_VERSION.en.md`，架构变化属于长期选择时更新 `04_decisions\DECISIONS.en.md`。
