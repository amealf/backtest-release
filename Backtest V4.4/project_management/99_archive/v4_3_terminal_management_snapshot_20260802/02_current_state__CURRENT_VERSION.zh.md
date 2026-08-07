# 当前版本说明

## 当前概览

| 字段 | 当前值 |
| --- | --- |
| 项目 | Backtest V4.3 Research |
| 管理上下文建立时间 | 2026-08-01 |
| 初始化模式 | 现有项目 |
| 已确认可选模块 | data、research |
| 项目实现状态 | 五轮 max-W campaign 已按累计 684 坐标、232,225 笔交易完成终局闭合；Round 5 已交付并解释；无 Round 6 或参数接受 |

## 当前行为

- 开仓方法保持 calculated-threshold 与 `wait_next_real_trade`。
- pending entry 在既有等待／连续性边界内按第一根真实成交 bar 的 open 成交。根据明确决定，不包含价格重触发、创新高取消或结构反转取消。
- rebound 和 speed exit 若在无真实成交的 bar 上触发，会冻结第一次触发原因、时间、理论价格与证据。持仓继续存在，并在下一根真实成交 bar 的 open 成交；成交后才执行 flat reset。
- rebound 基准为持仓期间已完成 bar 上正数、有限的因果 W 窗口 `open[start]-low[end]` 候选的单调最大值。每根 bar 使用截至上一根已完成 bar 生效的基准检查；每次真实入场重置该笔交易的最大值。
- `rolling_tr_sum` 是唯一聚合方法。开仓基准采样是独立策略轴：`all_window` 为默认，标记只作审计／图表证据；`exclude_marked` 忽略 `baseline_excluded` 原子，并在同一连续段内向前补足更早的合格原子。
- 两种策略都会保留有限 synthetic TR15 原子；只有 synthetic 原子同时被标记且选择 `exclude_marked` 时才会被忽略。当前没有 real-only baseline 策略。
- 来源存在重复时间戳时，执行前报错。`baseline_pending_atom_count` 按物理 baseline 跨度内真实 `low_activity_state == pending_low_activity_buffer` 原子统计。
- 活跃运行输入与固定模板从 `runtime_inputs` 下仓库相对、哈希绑定文件解析。
- 用户要求的 ZIP 包含当前版本代码、项目管理文件、分析报告与 15 秒 OHLC 数据；运算结果载荷不纳入。
- 场景新建与查看使用版本中立的共享 HTML：`D:\Code\backtest-release\shared_tools\scenario_manager\index.html`。保存时自动采用下一个连续的 `情景N` 名称，不显示名称输入框；编号同时统计回收站，删除后不会复用。当前场景在图表上方以两行按钮展示，点击按钮即可载入对应区间；删除场景仍可恢复。显示默认使用 15 秒 bar，也支持 1、5、15、30、60 与 120 分钟；CSV／TSV 上传位于下方操作行。

## 当前形态与接口

引擎与 runner 生成不可变原始阶段证据。分析器和四线程逐笔生成器使用固定历史模板创建逐笔 HTML。累计生成器只纳入已完成且身份兼容的阶段，发布稳定总入口／逐笔路径。阶段与累计 `main_data` 提供四个独立 `highReturnViews`，并按 baseline 采样策略分区。五个 max-W 阶段与 684 坐标／232,225 笔交易累计交付均已基于不可变原始证据闭合。

每个原始阶段只能使用一种 baseline 采样策略。累计交付可以包含两种策略，但排名与身份映射必须按策略和方法分区。

## 已知缺口

- 终局领导者属于相互依赖的样本内候选，只能进入外部验证／审阅。当前没有 holdout、新日期、跨工具、成本／滑点或生产接受证据。
- 情景一资格依赖特定交互；最终无限制总收益领导者位于 E/W/M 下边界并包含 3,743 笔交易；笔均收益领导者保留 Round 3 的 M10。这些属于限制，不能作为 gap 审计拒绝规则。
- V4.2 既有结果不能改称 V4.3 结果，因为执行语义已经变化。

## 最近变化

| 日期 | 变化 | 证据 |
| --- | --- | --- |
| 2026-08-01 | 建立中性的项目管理结构。 | 生成文件与初始化报告 |
| 2026-08-01 | 建立 V4.3 synthetic-exit pending 成交与仓库相对运行输入。 | 来源 manifest、运行 manifest 与回归测试 |
| 2026-08-02 | 建立并优化跨版本场景管理器，同时保留原始选定器与历史快照。当前采用自动连续的 `情景N` 名称、两行场景按钮、下方上传操作，以及默认 15 秒与新增 30／60／120 分钟周期。 | 共享 HTML、本地场景持久化与桌面／390px 浏览器 QA |
| 2026-08-02 | 增加可选择的双 baseline 采样、中性准备标记、按策略分区的身份／排名、真实 pending 原子审计与重复时间戳拒绝。 | 当前代码、重建后的准备／运行 manifest 与测试 |
| 2026-08-02 | 闭合 max-completed-W 身份，并冻结包含四个独立高收益视图的 12 坐标 `all_window` Round 1 计划；尚未执行坐标。 | 来源 manifest SHA-256 `78fcea5452769fe14f531c4e3036fe7ba3b2513092a0ad1e43a5b7d495c949ec`、计划 SHA-256 `36f9f47fd1c49ac1be68abf7a0f146b81d667c115123b385dde31239daccdc80`、来源闭合 artifact 与完整测试 |
| 2026-08-02 | 将 Round 1 闭合为 12 坐标实现／smoke probe：1,196 笔不可变交易、纠正后的阶段／累计交付，以及四个完整收益视图。Gap-excluded return 只作显示用审计。 | 不可变证据 SHA-256 `935fd5e562b54deb8d8334251753a5b04e2ed3b71148d8ad00a9f5137dcdc216`；纠正后交付证据 SHA-256 `d0f88bc6a07c0ec0b5f8a298998aeee900a5a8a47b3dc9c581202bde14703b7c` |
| 2026-08-02 | 冻结确定性的 504 坐标 `all_window` Round 2 笛卡尔计划，共 42 个 12 坐标 batch；S480 固定，E/BH/TRW/K/W/M 使用宽跨度跳点。 | 计划 SHA-256 `5735c12e439c8686f1f8e2e5e5d4ffcdb956902aa21825619f5c6d9ab4eebe85`；来源 manifest SHA-256 `306bf83ca50dab913fc7d8f081ec249dc05e03529ced7ce1c6dc58addab3f946` |
| 2026-08-02 | Round 2 按 504 个坐标与 73,306 笔交易闭合；独立领导值为情景一总收益 10.8547%、无限制总收益 51.3786%、笔均收益 0.554842%。 | Completion SHA-256 `863eb92646e5ae78d6d2bd72f6c9e5d24acfb311b069e17c83c4fd94894eb2cc`；交付证据 SHA-256 `af17ff903b66877bcebae41d1bc7e1e6f041847b091945a573f17c8df2059160` |
| 2026-08-02 | 72 坐标 Round 3 交互细化与 588 坐标累计交付完成终局闭合。终局样本内领导值为情景一总收益 18.2861%、无限制总收益 115.8216%，以及两个笔数视图的笔均收益 0.578419%。无 Round 4 或参数接受。 | Completion SHA-256 `d42d806e0e9c4d651b18e03846f300c8f059e70dff987bab9dc544f281e474a7`；终局交付证据 SHA-256 `7b8f1450c9b53b84ad5d5d2e3ddd51fa6d84147624b41b4135cbe9d29262bf6f` |
| 2026-08-02 | 用户替换原有「无 Round 4」决定，并批准 72 坐标有界边界续轮；计划已冻结，计算仍受 B 的独立反连接与 validate-only 门禁约束。 | 计划 SHA-256 `f991e8c46b42ccc38677eef3be7f2e67d9e822592bb53d3578a4fbc1cdb77670`；已批准的预运行设计证据 |
| 2026-08-02 | Round 4 完成不可变原始证据与稳定固定模板交付闭合，累计达到 660 坐标／185,493 笔交易；随后批准并冻结终局 24 坐标 Round 5 计划，包含两个 12 点块且无笔均收益块。 | Round 4 交付证据 SHA-256 `0d5d3187de8a2a1297bbb35f1de220930d34ed30c117906b936506cd404b7fb8`；Round 5 计划 SHA-256 `bb7f323e077ebb919d122764d7fac20b97b0790959d71761b77bcf69e6a23ed8` |
| 2026-08-02 | Round 5 与五阶段 campaign 完成终局闭合，累计 684 坐标／232,225 笔交易。最终视图领导者为 Round 4 情景一总收益 37.082743%、Round 5 无限制总收益 515.916078%，以及两个笔数视图中 Round 3 的 0.578419% 笔均收益领导者。 | Round 5 不可变证据 SHA-256 `ae53e778944aa3fd4ed41627655fbaeb07aa863632110a0073728bba11b784c3`；交付证据 SHA-256 `6d3cc38cfe98d16d2b095990f2f6a8c85d69c230b9b873bae61d5c414c6751a0`；快照 `4bed7f828a7068ee0ee70001a245133c481e785358cfb3f2b043d08713ca7259` |

## 维护规则

交付物、当前行为、项目结构、接口、依赖、命令或运行流程变化后更新。长期理由写入 `04_decisions\DECISIONS.en.md`，当前权威路径写入 `SOURCE_OF_TRUTH.en.md`。
