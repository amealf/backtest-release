# 工作进展

## 用途

为 Backtest V4.3 Research 保存简洁的当前工作日志。记录已完成或发生重要变化的工作、结果、证据、限制与有用的交接路径，不保存聊天原文或每个中间动作。

`CURRENT_TASKS.en.md` 描述计划中或正在进行的工作；本文档记录已经完成，或变化程度足以影响项目理解的工作。

## 当前交付版本或工作方法

- 当前主版本或工作方法：V4.3 `rolling_tr_sum`，baseline 采样可选择（`all_window` 默认、`exclude_marked` 可选），计算价开仓，第一根真实成交 open 的 pending entry，以及下一根真实成交 open 的 pending exit。
- 有效边界：Round 1–5 已保持不可变并完整交付，累计 684 坐标／232,225 笔交易。Round 5 为终止轮；不授权 Round 6 或参数接受。

## 2026-08-02 — 终局 Round 5 与五阶段 campaign 闭合

- Round 5 按已批准的两个 12 点块执行 24 个新反连接坐标、两个 batch 与 46,732 笔交易，compute-only 资源为 3／12／4096。不可变证据 SHA-256 为 `ae53e778944aa3fd4ed41627655fbaeb07aa863632110a0073728bba11b784c3`。
- 唯一一次四 worker 固定模板交付闭合于累计快照 `4bed7f828a7068ee0ee70001a245133c481e785358cfb3f2b043d08713ca7259`：五个阶段、684 个坐标、232,225 笔交易。交付 QA 证据 SHA-256 为 `6d3cc38cfe98d16d2b095990f2f6a8c85d69c230b9b873bae61d5c414c6751a0`；775／775 制品哈希、13／13 来源／原始／计划检查、80+80 浏览器状态、12／12 截图与 3／3 稳定路由均通过。
- 情景一总收益保留 Round 4 的 E30/BH720/TRW6/K1.5/W1/M2 领导者，37.082743%。无限制总收益更新为 Round 5 的 E120/BH360/TRW6/K0.75/W1/M0.25，515.916078%。两个笔均收益视图保留 Round 3 的 E200/BH720/TRW24/K0.75/W192/M10，0.578419%。
- 四个视图继续彼此独立；gap-excluded return 继续只作显示。最终领导者只属于相互依赖的样本内候选，需要另行授权外部验证／审阅。
- 决定：max-W campaign 在 Round 5 后闭合。无 Round 6，也没有参数接受、验证、推广或生产声明。
- 交接：canonical 终局报告为 `D:\Code\backtest-release\Backtest V4.3 max-W rebound\.omo\teams\team-a31b9876\artifacts\A_v4_3_max_w_round_05_20260802.md`；负责人声明终局后，由交付成员负责干净打包 staging。

## 2026-08-02 — Round 4 闭合；终局 Round 5 计划已冻结

- Round 4 按 72 坐标／73,373 笔交易闭合，四阶段累计交付达到 660 坐标／185,493 笔交易。
- 新累计领导者为情景一总收益 E30/BH720/TRW6/K1.5/W1/M2，37.082743%；无限制总收益 E180/BH360/TRW6/K0.75/W1/M0.5，340.137206%。两个笔均收益视图继续保留 Round 3 的 M10 领导者，因为 Round 4 的 M12 领导值更低。
- 稳定交付证据 SHA-256 为 `0d5d3187de8a2a1297bbb35f1de220930d34ed30c117906b936506cd404b7fb8`；803／803 资产哈希、80+80 浏览器状态与 12 张截图人工检查通过。
- leader 批准两个终局 12 点 Round 5 块，共 24 个坐标，无笔均收益块。冻结计划 SHA-256 为 `bb7f323e077ebb919d122764d7fac20b97b0790959d71761b77bcf69e6a23ed8`。
- 本段保留为启用记录。Round 5 随后通过 B 的全部权威门禁，并按上方终局记录闭合。

## 2026-08-02 — 用户授权的 Round 4 边界计划已冻结

- 目标：只沿三个独立 Round 3 领导者尚未解决的边界继续检查过的样本内搜索。
- 变化：用户明确授权后反转原有「无 Round 4」决定，并在 all-window／rolling-only／S480 与资源 3／12／4096 下冻结三个彼此独立的 24 点块。
- 证据：计划 `research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_max_w_multiround_20260802_round_04_all_window.json`，13,221 bytes，SHA-256 `f991e8c46b42ccc38677eef3be7f2e67d9e822592bb53d3578a4fbc1cdb77670`；预运行备忘录位于当前团队 artifacts。
- 门禁：权威 runner 展开、已完成 ID 加活动／待执行计划的精确反连接、fingerprint、资源／锁检查与 validate-only 由 B 负责。尚未计算或交付；不可在不可变闭合前解释结果。
- 限制：四个独立视图与仅显示用 gap 审计保持不变。条件 Round 5 默认不存在，需要新的批准，最多 36 个新坐标，并且是终止轮。没有参数被接受。
- 有效边界：三轮 max-W campaign 已完成不可变原始证据、四视图解释、固定模板交付、浏览器／视觉／哈希 QA 与无接受结论的终局闭合。

## 2026-08-02 — Round 3 终局交互细化与 campaign 闭合

- 目标：围绕三个去重 Round 2 领导者测试有意义的 E/W/K 或 M 交互，再按四个独立视图作出终局证据决定。
- 变化：Round 3 完成 72 个经过反连接的 `all_window` 坐标、六个 batch 与 37,618 笔交易。累计快照现在包含三个完成阶段、588 个坐标与 112,120 笔交易。
- 结果：情景一合格总收益由 E60/BH720/TRW6/K2/W2/M2 领导：总收益 18.2861%、336 笔、最大回撤 -6.6971%。无限制总收益由 E160/BH360/TRW6/K0.75/W2/M1 领导：总收益 115.8216%、2,541 笔、最大回撤 -3.5619%。两个笔均收益视图均由 E200/BH720/TRW24/K0.75/W192/M10 领导：笔均收益 0.578419%、77 笔、总收益 51.8332%、最大回撤 -13.8733%。三者的样本内主指标均高于对应 Round 2 领导值。
- 证据：Completion SHA-256 `d42d806e0e9c4d651b18e03846f300c8f059e70dff987bab9dc544f281e474a7`；不可变证据 SHA-256 `0652f87f01f0b617b074279b80a0379197fdbff9871a51008477c60abfede3f8`；终局交付证据 SHA-256 `7b8f1450c9b53b84ad5d5d2e3ddd51fa6d84147624b41b4135cbe9d29262bf6f`；当前 snapshot manifest SHA-256 `510c914b4d76f1fb5dd09af72e4b92a86bd9915e195b924989a1cf75a5d76eee`。
- 限制：领导者只属于相互依赖的样本内候选。情景资格依赖特定交互；无限制总收益强烈依赖 W2/M1；笔均收益保留 M10 已测试边界。Gap-excluded return 只作显示，不能接受或拒绝坐标。当前没有 holdout、新日期、跨工具、成本／滑点或生产证据。
- 决定：Campaign 在 Round 3 闭合。三个视图专属领导者只保留给外部验证／审阅；不自动建立 Round 4，也不接受参数。
- 交接路径：`results\campaigns\v4_3_max_w_multiround_20260802\round_03_all_window`、`results\all_completed_union_analysis` 与 `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\A_v4_3_max_w_round_03_20260802.md`。

## 2026-08-02 — Round 2 宽网格探索闭合，Round 3 交互细化已提出

- 目标：通过四个批准的完整收益视图解释 504 坐标宽网格结果，再定义不会混合目标的有意义有界细化。
- 变化：Round 2 完成 504 个 `all_window` 坐标、42 个 batch 与 73,306 笔交易。四个独立决定产生三个不同领导者：情景一总收益 E80/BH720/TRW6/K2.5/W3/M2；无限制总收益 E120/BH360/TRW6/K0.75/W3/M2；两个笔均收益视图均为 E200/BH720/TRW24/K0.75/W192/M8。
- 结果：主指标分别为情景一总收益 10.8547%、无限制总收益 51.3786%，以及 `>=10`／`>=20` 两个视图的笔均收益 0.554842%。已冻结 Round 3 计划使用三个彼此独立的 24 点 E×W×K/M 局部块，共 72 个坐标；runner 加载显示 72 个唯一坐标，与 516 个已完成坐标的精确 combo-id 重叠为零。计划 SHA-256 为 `78f695c76e508e0c7210f60ea01717b03c677581f589c3b69bb277f187e7c47b`。
- 证据：Completion SHA-256 `863eb92646e5ae78d6d2bd72f6c9e5d24acfb311b069e17c83c4fd94894eb2cc`；stage-summary SHA-256 `00f21ab09e52be9a53ce54c34f5b5d2a8f6c695b3829e4b95029b8db7546def0`；不可变证据 SHA-256 `d3d1110bb9755865028ede4305171cda9e48a89ab95abb193acb6107e0e3ac47`；交付证据 SHA-256 `af17ff903b66877bcebae41d1bc7e1e6f041847b091945a573f17c8df2059160`；团队 artifacts 下的 Round 2 与 Round 3 记录。
- 限制：结果属于样本内，并依赖同一工具、区间、情景与已检查网格。Gap-excluded return 继续只作显示，不能影响排名或细化设计。已获原则批准的 72 坐标设计等待 B 的交付后身份复哈希。
- 交接路径：`results\campaigns\v4_3_max_w_multiround_20260802\round_02_all_window`、`.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\A_v4_3_max_w_round_02_20260802.md`，以及同一 artifact 目录下的 `A_v4_3_max_w_round_03_20260802.md`。

## 2026-08-02 — max-completed-W Round 1 smoke 结果与宽网格 Round 2 决定

- 目标：用四个独立当前视图评价宽跨度 W 跃迁，只有声明的证据门槛支持时才继续下一轮。
- 变化：Round 1 在原始 fingerprint `da15e3965696637b55ac582d58d8fdd0eea54e2ae0c814bf2e444edd3f0adfec` 下完成 12 个 `all_window` 坐标与 1,196 笔交易。窄范围运行后分析器修复接受精确的已审计计划状态，并按原因处理同信号 bar 的 max-W 来源边界；计划、引擎、runner、原始文件、fingerprint 与模板均未改变。
- 结果：一个坐标满足情景一。无限制总收益最高为 48.0851%；由于所有行都超过 20 笔，两种笔均收益视图共享 0.489439% 的最高值。这些是样本内 smoke-probe 领导者，不代表参数已接受。已批准的 Round 2 设计扩展为确定性的 504 坐标 E/BH/TRW/K/W/M 笛卡尔网格；可执行计划已冻结并等待审计。
- 证据：不可变闭合 artifact SHA-256 `935fd5e562b54deb8d8334251753a5b04e2ed3b71148d8ad00a9f5137dcdc216`；completion manifest SHA-256 `3a3428b5d5f8b8f24db07b0d8f8ed733a03a48c6950e7a3e63196c7df4e273ca`；纠正后交付 QA SHA-256 `d0f88bc6a07c0ec0b5f8a298998aeee900a5a8a47b3dc9c581202bde14703b7c`；Round 2 计划 SHA-256 `5735c12e439c8686f1f8e2e5e5d4ffcdb956902aa21825619f5c6d9ab4eebe85`；当前来源 manifest SHA-256 `306bf83ca50dab913fc7d8f081ec249dc05e03529ced7ce1c6dc58addab3f946`。
- 限制：结果属于样本内，只覆盖一个工具／区间与 `all_window`，并包含 424 笔跨缺口交易。Gap-excluded return 只是显示用 gap 依赖审计，不能改变资格、排名、续轮或候选。Round 1 固定模板阶段／累计交付与响应式浏览器 QA 已闭合。
- 交接路径：`results\campaigns\v4_3_max_w_multiround_20260802\round_01_all_window`、`.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\A_v4_3_max_w_round_01_20260802.md` 与 `research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_max_w_multiround_20260802_round_02_all_window.json`。

## 2026-08-02 — 物化 max-completed-W Round 1 计划

- 目标：在不合并排名目标的前提下，用有界、可审计 campaign 隔离新的因果 max-completed-W rebound 基准。
- 变化：引擎／runner 身份现已绑定原始 schema 6、fingerprint schema 7、max-completed-W rebound policy、审计 schema v2 与新的策略／结果语义。Round 1 包含 12 个 `all_window` 坐标：三个来源几何分别测试 `W={3,12,48,192}`。四个排名视图彼此独立：情景一合格坐标的总收益、无限制总收益、`>=10` 笔的无限制笔均收益、`>=20` 笔的无限制笔均收益；每个视图都在主指标后使用 `combo_id` 排序。
- 结果：最终产品代码／来源 manifest 闭合后，Round 1 可执行计划冻结为 12 个唯一 combo ID。后续执行与交付结果记录在上方更新的 smoke-result 条目。两个 runner 可移植性缺陷已经修复；C 冻结交付代码后，B 独立复现 61 项通过、2 项结果夹具跳过。
- 证据：最终计划 SHA-256 `36f9f47fd1c49ac1be68abf7a0f146b81d667c115123b385dde31239daccdc80`；最终来源 manifest SHA-256 `78fcea5452769fe14f531c4e3036fe7ba3b2513092a0ad1e43a5b7d495c949ec`；来源闭合 artifact SHA-256 `aaf52d1e445a176c540fc818bb162ea49f6128f21f6b42682e2979dd9b38a5bb`；团队 artifacts 下的设计与 Round 1 记录。
- 限制：本条记录运行前边界；结果状态以上方 smoke-result 条目为准。所有结果仍标记为样本内，不能自动接受参数。
- 交接路径：`research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_max_w_multiround_20260802_round_01_all_window.json` 与 `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\A_v4_3_max_w_round_01_20260802.md`。

## 2026-08-02 — 跨版本场景管理器交付

- 目标：把绑定单一版本的场景查询改为共享编辑入口，并复用用户指定的市场选定器。
- 变化：共享 HTML 可把一段或多段已选行情自动保存为下一个 `情景N`。回收站中的编号继续保留，因此删除 `情景2` 后，下一个未使用的连续名称为 `情景3`。当前场景在图表上方以两行按钮展示，点击即可载入；归档／恢复继续有效。标题／主题首行已经移除，CSV／TSV 上传移到下方操作行；显示默认改为 15 秒，并增加 30／60／120 分钟周期。
- 结果：稳定跨版本入口为 `D:\Code\backtest-release\shared_tools\scenario_manager\index.html`。原始选定器和现有结果快照保持不变。
- 证据：自动保存生成 `情景1` 至 `情景5`；归档 `情景2` 后，下一保存键显示 `情景6`，确认不会复用编号。两行场景按钮成功载入两段行情。默认图表绘制 199,200 根 15 秒 bar；桌面与 390px 检查均无控制台／资源请求错误，也没有重叠、乱码、异常留白或横向溢出。
- 限制：可编辑场景库属于该固定 HTML 路径的浏览器本地数据。回测仍需把已审阅、不可变的场景定义绑定到计划。
- 交接路径：`D:\Code\backtest-release\shared_tools\scenario_manager\index.html` 及其 `README.md`。

## 2026-08-01 — V4.3 成交与可移植性修复闭合

- 目标：消除 synthetic exit 成交，保留已确认的 pending-entry 行为，并让活跃运行输入随 V4.3 文件夹移动。
- 变化：rebound／speed 信号在无真实成交的 bar 上会冻结 pending exit，在下一根真实成交 bar open 成交。pending entry 继续按第一根真实成交 bar open 成交，不要求重触发，也不增加结构取消。数据、准备产物、固定模板、Plotly 与市场选定器均位于仓库内并受哈希绑定。
- 结果：新的 V4.3 策略／结果／组合／审计身份已经生效；V4.2 保持不变；没有运行回测。
- 证据：`SOURCE_MANIFEST.json`、`runtime_inputs\RUNTIME_INPUTS.json`、包测试与生成的数据准备 manifest。
- 限制：V4.2 既有结果不属于 V4.3 证据。参数探索需要新的明确指令与已审阅计划。
- 交接路径：`research_variants\short_momentum_net_drop_rebound_v4_3`、`runtime_inputs` 与 `RUNTIME.md`。

## 2026-08-02 — 双 baseline 采样与审计修复闭合

- 目标：保留两种已确认 baseline 采样选择，同时不恢复 `tr_average`，也不混合两种策略的证据。
- 变化：准备层发布中性 `baseline_excluded` 标记与 `eligible_if_excluding_marked`。`all_window` 保持默认；`exclude_marked` 忽略被标记的有限 TR15 原子，并在同一连续段内向前补足更早的合格原子。策略已进入 combo、strategy、result、plan、fingerprint、stage、batch、completion、analysis、catalog 与累计身份。排名按策略和 `rolling_tr_sum` 分区。
- 结果：当前 UI 源移除 `tr_average`，兼容页面增加 `>=10` 筛选，并通过既定模板展示两种采样选择。重复 datetime 会报错；pending 低活跃审计按真实状态统计；准备／运行／来源 manifest 已重新绑定。没有运行参数回测。
- 证据：当前来源与运行 manifest、重建的准备产物、包测试，以及下述 synthetic baseline 诊断。
- 限制：两种策略都不会排除全部 synthetic bar。已复核的 real-only 对照只属于诊断，不定义第三种可执行策略。
- 交接路径：`research_variants\short_momentum_net_drop_rebound_v4_3`、`runtime_inputs\data_preparation` 与 `project_management\index.html`。

## 2026-08-01 — 建立项目管理记录

- 目标：建立可复用的双语项目记忆。
- 结果：建立中性管理结构与 Dashboard 输入。
- 证据：初始化报告与生成文件。
- 限制：实现状态以后续 V4.3 闭合记录为准。

## 记录格式

有实际用途时，将新记录放在顶部，并使用下列字段：

- 目标：已完成或发生重要变化的工作要解决什么问题。
- 变化：哪些内容变得不同。
- 结果：已经确认的结果或当前状态。
- 证据：测试、文件、命令、产物或用户明确确认。
- 限制：现有证据不能说明什么。
- 交接路径：继续工作或审阅时需要使用的当前文件或产物。

## 维护规则

- 工作完成，或发生后续工作需要了解的重要变化时追加记录。
- 保持记录简洁，链接保留证据，不复制大型输出。
- 前瞻状态更新到 `CURRENT_TASKS.en.md`，当前行为更新到 `CURRENT_VERSION.en.md`。
- 主交付版本或工作方法变化，并且旧细节会误导日常工作时，将被替代的详细正文移入 `99_archive` 下带日期的目录，在本文保留简洁恢复指针，并完整保留存档材料。
