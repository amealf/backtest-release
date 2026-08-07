# 当前有效文件

## 当前发布身份——2026-08-08

`RELEASE.json` 是当前界面发布版本（`V4.41`）及兼容边界的权威来源。策略与结果身份继续由 `research_variants\short_momentum_net_drop_rebound_v4_4\SOURCE_MANIFEST.json` 以 V4.4 身份约束。累计发布程序仍按原有主版本与排序谱系检查，纳入全部已完成且兼容 V4.4 的阶段。

## 当前 K200 训练／测试逐笔切换与布局——2026-08-08

- K200 训练逐笔入口仍为 `results\all_completed_union_analysis\trade_review\index.html`，它跳转到当前快照并显示「显示测试集」。K200 测试逐笔入口仍为 `results\cross_instrument_comparison\runs\k200_train_test_si__combined_350_v56_20260807\trade_review_k200_test\index.html`，并显示「显示训练集」。
- 两个按钮都保留当前精确 `combo_id`，同时绑定目标研究合同。目标区间缺少该精确坐标时，不会改用邻近参数。
- 交易选择器的重复标题／计数和下方摘要行已经移除。drop、ratio、开仓基准、开仓阈值、W 基准和 active low 改为理由标题下方的紧凑彩色框，不再作为 Plotly 标注遮挡价格。Plotly 图例移到横轴下方。
- 浏览器标签标题为「组合平仓逐笔查看」，图标为蓝色 `Z`。共享来源：`runtime_inputs\templates\historical_v4_trade.html`；专项检查：`tools\qa_v4_4_trade_review_layout.mjs`；保留证据：`project_management\screenshots\trade_review_layout_20260808`。

## 当前按日期区间保存的结果包框架——2026-08-08

- 稳定比较入口：`results\cross_instrument_comparison\index.html`。该入口现在跳转到 `results\evaluation_comparison\index.html`；保留的 350 组运行页面和全部现有总入口／逐笔页面仍保持字节不变。
- 当前证据适配器：`tools\build_v4_4_evaluation_framework.py`。通用结果包登记程序：`tools\register_v4_4_evaluation_package.py`，使用 `runtime_inputs\templates\EVALUATION_PACKAGE_SPEC.template.json`。浏览器一致性检查：`tools\qa_v4_4_evaluation_framework.mjs`。机器审计：`results\evaluation_comparison\compatibility_audit.json`。
- 结果包目录：`results\evaluation_packages\catalog.json`。当前三个结果包为 `K200\20260526T000000__20260708T235200`、`K200\20260708T235215__20260807T032145` 和 `SImain\20260129T000000__20260223T235945`。
- 每个结果包包含清单、准确日期实验记录、参数汇总、当前候选集的浏览器精简数据、硬链接的不可变逐笔记录和逐笔兼容入口。目录名不含训练、测试或迁移角色。
- 当前比较方案：`results\evaluation_comparison\comparisons\K200_20260526T000000__20260708T235200__K200_20260708T235215__20260807T032145__SImain_20260129T000000__20260223T235945\comparison_plan.json`。
- 兼容性结果：350 行及全部字段值与保留比较页面完全一致；比较页面 HTML 只在数据脚本来源处不同。新旧比较页面的标题、正文、首行数据和渲染行数相同，浏览器错误均为零。后来确认的 K200 训练／测试逐笔界面更新只改变小型 HTML 外壳与清单；逐笔数据、区块、指标、总入口、保留比较页和 SI 逐笔证据均未改变。
- 这套结构现在是全项目结果存储规则。任何已就绪品种配置都可以选取精确区间创建独立结果包，并通过比较方案与其他结果包并列，不改变已有包身份。

## 当前 K200 优参数初步后续重放——2026-08-07

- 当前结果目录：`results\temporal_migration\v4_4_k200_current_optimal_forward_initial_v2_20260807`；解释报告：`INITIAL_REPORT.md`；完整比较：`comparison.csv`；机器摘要：`summary.json`。
- 冻结计划：`research_variants\short_momentum_net_drop_rebound_v4_4\plans\k200_current_optimal_forward_initial_v2_20260807.json`；候选冻结文件为相邻的 `_candidate_freeze.csv`。
- 修正版包含 100 组训练期成本后为正的唯一候选。其中 6 组是保留的已评价主视图对照，94 组是此前没有在后续一个月运行过的精确坐标。候选选择没有读取目标期指标。
- 结果：`2026-07-08 23:52:15` 至 `2026-08-07 03:21:45` 共有 29/100 组成本后为正，8/100 组非 gap 收益为正；收益中位数为 -0.671%，训练／后续 Spearman 为 -0.346。训练期总收益冠军在后续为 -1.567%。`parameter_acceptance=none`。
- 不含 `_v2_` 的较早目录属于历史记录，不能用于当前决策。其第一次候选定义带入 12 组训练期成本后为负的坐标，目录内 `STATUS.md` 指向修正版。

## 当前 K200 训练／测试／SI 迁移——2026-08-07

- 三组收益稳定入口：`results\cross_instrument_comparison\index.html`；当前运行：`results\cross_instrument_comparison\runs\k200_train_test_si__combined_350_v56_20260807`。
- 页面同时显示 350 组候选的 K200（训）、K200（测）与 SI 总收益。排名列表头统一为「排名」，固定宽度蓝色按钮只显示「#N」，并在新标签页打开 SI 逐笔证据。三列总收益单元格使用相同蓝色按钮样式，分别打开对应样本的逐笔分析。完整比较为 `migration_comparison.csv`；K200（训）复用累计 K200 逐笔入口，K200（测）使用 `trade_review_k200_test\index.html`，SI 使用 `trade_review\index.html`；解释报告为 `FINAL_TRAIN_TEST_SI_REPORT.md`。
- 本次保留既有 250 组 SI 结果，并新增 100 组根据已闭合 K200 时间迁移证据冻结的候选。新增候选在读取 SI 结果前冻结，两批候选重合为零。全部 350 组统一重放 K200 测试期。
- 证据：K200 测试期 199/350 组为正，SI 275/350 组为正，三列同时为正 149/350 组。新增 100 组中，K200 测试期 72 组为正，SI 59 组为正，三列同时为正 43 组。发布复用 250 个 SI 逐笔区块，生成 100 个新区块。
- 当前较有希望的新增邻域为 E320／BH240／TRW12／K1.25／W6、M4.25–4.75、S340–370。三列收益中位数为 +50.123%／+8.827%／+14.350%，但 K200 测试期非 gap 收益中位数为 -12.213%。该 gap 依赖使参数暂不具备接纳条件。`parameter_acceptance=none`。

## 当前 K200 时间迁移——2026-08-07

- 时间迁移排序：`results\temporal_migration\v4_4_k200_temporal_migration_20260807\index.html`。
- 解释报告：`results\temporal_migration\v4_4_k200_temporal_migration_20260807\TEMPORAL_MIGRATION_REPORT.md`；完整对比为 `temporal_comparison.csv`，机器摘要为 `final_summary.json`。
- 后续行情逐笔入口：`results\temporal_migration\v4_4_k200_temporal_migration_20260807\full_replay\analysis\trade_review\index.html`。
- 计划与候选冻结文件：`research_variants\short_momentum_net_drop_rebound_v4_4\plans\k200_temporal_migration_20260807`。R1–R4 和描述性全区间重放都已在时间迁移结果目录下完整闭合。
- 证据：每段 400 组；R1/R2/R3/R4 的正收益数为 296/26/383/25。218 组覆盖全部四段，只有两组四段全正；两组全测试期分别只有 11、13 笔，分期 Top2 正收益占比中位数都是 100%。训练期与全测试期收益排名 Spearman 为 -0.26169。`parameter_acceptance=none`。

## 当前 K200 行情来源——2026-08-07

- 活动来源：`runtime_inputs\market_data\k200_clean_15s_session_filled.csv`。
- 当前身份：233,368 根补齐交易时段的 15 秒数据，范围为 `2026-05-23T00:00:00+09:00` 至 `2026-08-07T03:21:45+09:00`；SHA-256 为 `9760d367a109777c4789ce45d982a6c0708bacddad8f549450ed94f81ad5c405`。
- 最新保留的 Tick 下载目录：`F:\Backtest test 6.11\02_DATA_AND_AUDITS\market_data\k200_historical_ticks_supplements\k200_postroll_supplement_20260728T161430_to_20260807T032200_20260807_022303`。其中包含 1,065,932 笔 IBKR `TRADES` Tick、34,168 根清理并补齐交易时段的 15 秒数据、检查点、审计文件和目录内 `README.md`。
- 准备数据身份：`runtime_inputs\data_preparation` 下的 `v4_4_confirmed_low_activity_gate_9760d367a109777c_76f2695bc1f4_9e27394dbe49`。
- 既有 V4.4 结果阶段与累计排名没有重新计算，继续作为各自已记录评价区间的证据；扩展后的来源供后续明确启动的回测使用。

## 当前逐笔区间统计——2026-08-07

- 稳定逐笔入口：`results\all_completed_union_analysis\trade_review\index.html`。
- 已更新快照入口：`results\all_completed_union_analysis\snapshots\eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5\trade_review\index.html`。
- 展示源码：`runtime_inputs\templates\historical_v4_trade.html`；生成器：`research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_review_delivery.py`。
- 「区间统计」会把现有 Plotly 图表切换为横向圈选。圈选完成后，页面从已载入的可见 OHLC 数组计算一次；启动阶段不预计算，不建立区间缓存，也不增加请求。页面随后清除原生选择状态，再用普通 Plotly 图形绘制浅色矩形，使选区外行情保持正常明暗。
- 紧凑三列结果框位于右侧详情栏的交易理由上方。启用任一检测会收起参数抽屉；关闭结果框会移除自定义选区、退出检测模式并恢复缩放拖动。
- 「持仓检测」与区间统计互斥，并保留 Plotly 拖动缩放。移动超过五像素视为拖动；图表层的 pointer-down 与文档层的 pointer-up 将短按映射到最近的可见 K 线，右键被排除。随后按 `entry_index <= selected_index < exit_index` 判断持仓。逐根 K 线使用自身来源 bar；合并 K 线的状态计算使用与显示 close 对应的末端来源 bar。所选显示 K 线的聚合最高价与中心位置使用蓝色 Plotly 标记，像素大小与缩放表现和绿色开仓点一致。更深的无描边蓝色着色从有效基准起点延伸到所选时点的 active low。带标注的浅红密虚线横线显示 `active low + 回撤阈值`；带标注的竖线显示 `activeLowIndex + S`，即当前 active low 后没有新低时的理论速度平仓位置，并显示剩余 bar 数与时间，即使 S 已形成也会绘制。面板继续列出真实 max-W 候选来源区间。
- K 线 `whiskerwidth` 为 0，高低点末端不再显示横线。
- 页面已移除视图切换说明句，以及「紫色」「橙色」标注后缀。上述改动只影响展示，坐标数据、交易、指标、排名语义与参数接受状态均未改变。

## 当前 K200 累计展示——2026-08-06

- 稳定入口为 `results\all_completed_union_analysis\index.html`；它会跳转到 `main\index.html`，不再保留重复导航栏与 iframe。
- 当前页面标题为「V4.4 K200回测结果排序」。主结果表使用全宽布局，占用接近一个视口高度；表头固定，每页渲染 500 行排序结果。
- 交易数筛选标题为「交易数」。预设保留「不限」「至少 10 笔」「至少 20 笔」，并提供可同时使用的严格「大于 x 笔」与「小于 x 笔」数字条件；原有至少 100 笔和至少 150 笔预设已移除。主表列标题统一居中。
- 页面已移除 gap 依赖收益审计、`segment_end_exit_count`、`waited_entry_count` 与 `maximum_entry_wait_bars`。「往返成本 bps」和「跨 gap 笔数」保留在最后两列；底层结果字段和不可变快照均未改变。
- 累计页面不再显示「研究合同」。

## 当前统一更严格开仓迁移交付——2026-08-06

- 稳定入口：`results\cross_instrument_comparison\index.html`。
- 当前运行：`results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__combined_250_stricter_entry_v54_20260806`。
- 目标品种逐笔入口：`results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__combined_250_stricter_entry_v54_20260806\trade_review\index.html`。
- 迁移方案：`research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_migration_k200_to_simain_stricter_entry_round2_20260806.json`。
- 来源阶段：`results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_16_migration_stricter_entry_k_expansion_all_window`。
- 闭合：250 个兼容候选、24,194 笔 SImain 交易、复用 247 个目标逐笔区块、生成三个新区块。K1.4 仍是固定坐标下的目标局部峰值；分类为 `not_improved_target_peak_at_k1p4`，参数接受为 `none`。本节取代下方 247 候选迁移交付说明。

## 当前 AI 主导 K200 跳跃／网格累计交付——2026-08-06

- 稳定总入口：`results\all_completed_union_analysis\index.html`。
- 稳定共享逐笔入口：`results\all_completed_union_analysis\trade_review\index.html`。
- 当前不可变快照：`results\all_completed_union_analysis\snapshots\eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5`。
- 当前规模：campaign `v4_4_positive_entry_signal_repair_20260805` 下共 37,058 个唯一坐标、11,749,606 笔交易和 109 个兼容阶段。
- 最新阶段：`results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_108_ai_generated_leap_all_window`；闭合为 512 个坐标和 151,372 笔交易。
- 探索会话：`results\ai_exploration\k200_leap_grid_cycle_20260806`；共完成 89 个自动轮次，其中 60 个跳跃轮次、29 个自适应网格轮次，使用精确反连接去重，参数接受为 `none`。
- 增量发布复用 5,320 个未变化逐笔区块，生成 31,738 个新区块。稳定路由指向本快照。K200 尚未形成全局闭合，因此未启动 SI 迁移。本节取代下方较早的 K200 累计交付说明。

## 已替代的 K200 累计交付——2026-08-05

- 稳定总入口：`results\all_completed_union_analysis\index.html`。
- 稳定共享逐笔入口：`results\all_completed_union_analysis\trade_review\index.html`。
- 已替代的不可变快照：`results\all_completed_union_analysis\snapshots\db85efb36f3de1c1f8255c6108fb365ad9f3d337f77a8d37a0e0ae41982e5699`。
- 已替代规模：campaign `v4_4_positive_entry_signal_repair_20260805` 下共 5,044 个坐标、742,706 笔交易和十五个兼容阶段。
- 最新阶段：`results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_14_large_multiblock_exploration_all_window`；闭合为 294 个坐标、28,577 笔交易和 37 个批次。
- 最新解释报告：`results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_14_large_multiblock_exploration_all_window\interpretation\round_14_report.json`；分类为 `improved`，参数接受仍为 `none`。
- 累计增量交付复用 4,747 个未变化逐笔区块，并生成 297 个缺失区块。稳定路由已经指向新快照。本节取代下方较早的累计交付说明。

## 当前增量更严格开仓迁移交付——2026-08-05

- 原品种本轮总入口：`results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_13_stricter_entry_k_expansion_all_window\analysis\index.html`。
- 原品种本轮逐笔分析：`results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_13_stricter_entry_k_expansion_all_window\analysis\trade_review\index.html`。
- 稳定入口：`results\cross_instrument_comparison\index.html`。
- 当前运行：`results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__combined_247_stricter_entry_v52_20260805`。
- 目标品种逐笔入口：`results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__combined_247_stricter_entry_v52_20260805\trade_review\index.html`。
- 报告：`results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__combined_247_stricter_entry_v52_20260805\MIGRATION_REPORT.zh.md` 及英文镜像。
- 固定增量发布程序：`research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_cross_instrument_comparison.py build --run-id <run_id>`。
- 闭合：247 个候选、24,024 笔 SImain 交易；复用 244 个逐笔区块，只生成三个新区块。本节取代下方 244 候选当前交付说明。
- 原品种本轮 HTML 仅包含三个新增 K200 坐标与 243 笔交易。它通过固定阶段分析程序生成，并关闭累计刷新，因此没有重建约 71.4 万笔交易的原品种稳定快照。

## 当前跨品种交付——2026-08-05

- 迁移方案：`research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_migration_k200_to_simain_20260805.json`。
- 稳定合并排序：`results\cross_instrument_comparison\index.html`。
- 当前运行目录：`results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__original_180_plus_repaired_64_v50_20260805`。
- 目标品种逐笔分析：`results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__original_180_plus_repaired_64_v50_20260805\trade_review\index.html`。
- 迁移报告：`results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__original_180_plus_repaired_64_v50_20260805\MIGRATION_REPORT.zh.md` 及英文镜像。
- 展示并集恰好包含 244 个无重合候选：旧结果 180 个、修复后来源结果 64 个；原品种逐笔为 25,129 笔，目标品种逐笔为 23,799 笔。本节取代下方较早的「当前展示」陈述；旧内容仅保留为过渡历史细节。

## 当前 K200 累计交付——2026-08-05

- 稳定总入口：`results\all_completed_union_analysis\index.html`。
- 稳定共享逐笔入口：`results\all_completed_union_analysis\trade_review\index.html`。
- 当前不可变快照：`results\all_completed_union_analysis\snapshots\0126cd77b436aef1434e7072bac0d6dfa15b3d2ad4dc2cf1b2fafe936ee1e626`。
- 当前总体：campaign `v4_4_positive_entry_signal_repair_20260805` 下 4,747 个坐标、713,886 笔交易和十三个兼容阶段。
- 当前交付包含十三个固定其他参数的 E 窗口敏感性坐标，以及十轮单轴探索的三十个坐标。

## 与品种无关的契约权威——2026-08-05

- 策略核心：`research_variants\short_momentum_net_drop_rebound_v4_4\contracts\STRATEGY_CONTRACT_V4_4.json`。
- 契约加载器：`research_variants\short_momentum_net_drop_rebound_v4_4\code\instrument_contracts.py`。
- 已就绪 K200 配置：`research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\k200m.json`。
- 未完成模板：`research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\simain.template.json` 与 `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\nq.template.json`。
- campaign 起点：`research_variants\short_momentum_net_drop_rebound_v4_4\campaign_contracts\CAMPAIGN_MANIFEST.template.json`。
- 工作流权威：`project_management\00_core\STRATEGY_CONTRACT_V4_4.en.md`、`project_management\10_instruments\INSTRUMENT_PROFILE_CONTRACT.en.md` 与 `project_management\20_campaigns\CAMPAIGN_WORKFLOW.en.md`。
- 修复前 K200 历史权威为快照 `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7`，包含 4,704 个坐标、706,470 笔交易和十九个阶段。当前修复后结果权威是上方列出的正值开仓快照。

## 跨品种对比权威——2026-08-05

- 源码清单 V48：`6952a4d30c1ad5fb9276de1c2d3248a1a70a0157a1260118b4086be3e04da1e0`，40,814 字节。
- 当前精确迁移展示独立绑定修复后的正值开仓 K200 快照。修复前历史迁移运行继续保持不可变并与本批分离。
- 稳定入口：`results\cross_instrument_comparison\index.html`。
- 当前修复来源验证运行：`results\cross_instrument_comparison\runs\k200_repaired_v48_20260526_20260708__simain_20260129_20260223__promising_exact_transfer_v49_20260805`。
- 共享 SImain 逐笔入口：`results\cross_instrument_comparison\runs\k200_repaired_v48_20260526_20260708__simain_20260129_20260223__promising_exact_transfer_v49_20260805\trade_review\index.html`；64 个候选均有自己的交易目录、K 线图、开仓理由、平仓理由与参数说明。
- 历史 266 候选合并结果继续保留在 `results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__all_exact_transfers_v46_20260805`，不与修复来源指标合并。
- 对比页提供以已完成文件为依据的原品种、原样本区间、被比较品种与被比较区间选择器。全文搜索与可叠加字段筛选覆盖每个展示列；表格继续隐藏候选来源与 `combo_id`。
- 收益口径控件在手续费／滑点后与无手续费／滑点之间同步切换排名，以及 K200／SImain 的总收益、中位单笔、最大回撤和胜率；默认采用成本后口径。
- 第一列复用 K200 的 `rank-link` 样式，显示为「查看 #N」；表头随口径显示「成本后排名」或「成本前排名」，每个按钮都在新标签页打开对应的 SImain 逐笔分析。
- 列分组依次为参数、SImain 指标、K200 指标与迁移诊断。MFE／MAE 同时保留 bps 和原始点数列；K200 总收益悬停说明明确指出其 2026-05-26 至 2026-07-08 区间长于 SImain 的 2026-01-29 至 2026-02-23 区间。
- 稳定累计入口是纯展示导航壳：它嵌入字节保留的当前累计快照，并链接独立跨品种页面；快照 HTML、`analysis_data.js` 与 `union_trades.csv` 保持不变。
- 修复来源批次在 SImain 评价前冻结。它从 4,747 个 K200 坐标出发，排除 266 个旧迁移坐标与三个冠军，在 225 个来源合格点中保留 11 个 W／M／S 参数族的 64 个族内 Pareto 候选。重复、旧迁移重叠、冠军重叠与目标驱动修改均为零。
- SImain 使用明确 SIH6 的 15 秒 OHLC，并加载一个测试日前的预热数据。只统计 2026-01-29 至 2026-02-23 测试区间内产生的开仓。SIH6 在整个测试区间均为主力合约，因此换月次数为零。
- 当前对比包含 64 个冻结候选和 6,755 笔 SImain 交易。58 个候选成本后为正，占 90.625%；29 个位于目标正收益稳定区域，26 个属于孤立正收益点。
- K200 与 SImain 成本后收益排名 Spearman 为 -0.45728。目标收益第一名 E320／BH240／TRW21／K1.05／W6／M4.5／S340 为 19.6892%；K200 第一名 E480／BH171／TRW12／K1.26／W6／M4.5／S388 在 SImain 只有 1.1311%。来源排名不能作为目标排名。
- 对比不建立综合分数，也不接受参数。SImain 结果不能增加、删除或修改冻结候选。以后如获授权运行 SImain 全网格，它必须作为单独标注的事后诊断。
- 跨品种发布以后，当前 K200 累计快照 HTML、`analysis_data.js` 与 `union_trades.csv` 字节保持不变。稳定累计导航壳链接当前独立跨品种入口。

## V4.4 当前权威入口

| 角色 | 路径 | 状态 |
| --- | --- | --- |
| 项目根目录 | `D:\Code\backtest-release\Backtest V4.4` | 当前临时 V4.4 工作区 |
| 结果物理根目录 | `F:\Backtest\Backtest V4.4\results` | 当前及后续 V4.4 结果存储；43,817 个文件已完成逐字节与成对 SHA-256 迁移核验 |
| 结果逻辑根目录 | `D:\Code\backtest-release\Backtest V4.4\results` | 指向 F 盘结果物理根目录的 Windows 目录联接；保留历史路径和入口 |
| 结果恢复副本 | `F:\Backtest\migration_recovery\Backtest V4.4 results before junction 20260804` | 可恢复的迁移源字节一致副本 |
| 旧结果隔离区 | `F:\Backtest\D_cleanup_quarantine_20260804` | 从 D 盘移出的四个已批准非 V4.4 结果目录的可恢复目标 |
| 引擎 | `research_variants\short_momentum_net_drop_rebound_v4_4\code\v4_4_engine.py` | 当前 |
| 运行器 | `research_variants\short_momentum_net_drop_rebound_v4_4\code\run_v4_4_resumable_campaign.py` | 当前 |
| 分析器 | `research_variants\short_momentum_net_drop_rebound_v4_4\code\analyze_v4_4_scenario_3_stage.py` | 当前 |
| 逐笔交付 | `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_review_delivery.py` | 当前 |
| 累计构建器 | `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_combined_union_analysis.py` | 当前 |
| 身份清单 | `research_variants\short_momentum_net_drop_rebound_v4_4\SOURCE_MANIFEST.json` | 当前 V48 正值开仓且与品种无关的源码身份；正式源码审阅 ZIP 必须在独立解压副本中通过包根目录测试入口；V4.4 结果通过兼容 D 盘目录联接物理存放于 F 盘 |
| 运行输入清单 | `runtime_inputs\RUNTIME_INPUTS.json` | 当前 |
| 数据准备清单 | `runtime_inputs\data_preparation\data_preparation_manifest.json` | schema 5 因果可用时间 |
| V4.4 交接 ZIP | `D:\Code\backtest-release\Backtest_V4.4_with_trade_records_20260803_final.zip` | 已审计的历史 `SOURCE_FINAL_V4` 包：九份报告、五个已完成阶段和 120 个原始／衍生逐笔交易 CSV；不属于当前 V5 源码包 |
| ZIP 审计 | `D:\Code\backtest-release\Backtest_V4.4_with_trade_records_20260803_final.zip.audit.json` | 压缩包流和解压副本核验通过 |
| 最近一次源码完整审阅 ZIP | `D:\Code\backtest-release\Backtest_V4.4_current_UI_review_20260805_source_complete.zip` | 保留的 V37 包，绑定快照 `20464535...`；它早于 V38 品种契约源码层，属于历史包证据 |
| 当前审阅 ZIP 审计 | `D:\Code\backtest-release\Backtest_V4.4_current_UI_review_20260805_source_complete.zip.audit.json` | 时间点、源码布局、压缩包流、解压副本与独立测试核验通过 |
| 临时计划 | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_temporary_single_combo_close_fill_20260802.json` | 已执行隔离计划 |
| 第一轮计划 | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_cost_adjusted_multiround_20260803_round_01_broad_all_window.json` | 已冻结 372 坐标计划；validate-only、反连接、资源与阶段建立门槛均在运算前完成 |
| 第二轮计划 | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_cost_adjusted_multiround_20260803_round_02_broad_local_all_window.json` | 已冻结 247 坐标大范围＋局部计划；原始运算已不可变闭合 |
| 第二轮阶段 | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_02_broad_local_all_window` | 已不可变闭合并交付：247 个坐标、20,629 笔交易，指纹 `7ad95dbd7ba9ebc1faffd8cbc1723211273453af0471ab950e5b3d798ee6c4e8` |
| 第二轮阶段总入口 | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_02_broad_local_all_window\analysis\index.html` | 已完成固定模板阶段入口 |
| 第三轮计划 | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_cost_adjusted_multiround_20260803_round_03_terminal_local_all_window.json` | 已冻结 212 坐标终结型局部计划；原始运算已不可变闭合 |
| 第三轮阶段 | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_03_terminal_local_all_window` | 已不可变闭合并交付：212 个坐标、16,847 笔交易，指纹 `6d40e4a35562bbbb16347a63b48442f132025bbc4695f0ac87bc4312eae0955e` |
| 第三轮阶段总入口 | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_03_terminal_local_all_window\analysis\index.html` | 已完成固定模板终结阶段入口 |
| 多轮设计 | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\v4_4_cost_adjusted_multiround_design_20260803.md` | 非执行型大范围／细化契约 |
| 第一轮解释 | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\round_01_interpretation_and_round_02_design_20260803.md` | 最终闭合轮次解释与第二轮理由 |
| 第二轮解释 | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\round_02_interpretation_and_round_03_terminal_design_20260803.md` | 最终闭合轮次解释与终结型第三轮理由 |
| 终局解释 | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\round_03_terminal_interpretation_and_campaign_closure_20260803.md` | 规范终局结果与 campaign 闭合；禁止第四轮 |
| 最终总审计 | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\final_read_only_total_audit.json` | 独立只读汇总审计：全部原始／交付身份、数量、哈希、唯一性、锁与退出进程均通过 |
| Continuation 设计 | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\v4_4_continuation_subseries_design_20260803.md` | 非执行型证据驱动大范围后细化设计及每轮强制交付规则 |
| Continuation 第一轮计划 | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_cost_adjusted_multiround_20260803_continuation_round_01_broad_span_all_window.json` | 已执行的 528 坐标冻结大范围计划；原始闭合保持不可变 |
| Continuation 第一轮阶段 | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\continuation_round_01_broad_span_all_window` | 不可变历史证据：528 个坐标、54,842 笔交易 |
| 已替代的 V4 绑定 Continuation 第二轮计划 | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_cost_adjusted_multiround_20260803_continuation_round_02_dual_objective_local_all_window.json` | 保留的运算前证据；不得编辑、移动、删除、验证或运算；替代 V5 计划待建立 |
| 已替代的 V4 绑定第二轮根目录 | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\continuation_round_02_dual_objective_local_all_window` | 只有四个确定性 validate-only 元数据文件；不存在 progress、批次、逐笔、completion 或 analysis |
| 第一轮阶段 | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_01_broad_all_window` | 不可变：372 坐标、316,398 笔交易 |
| 第一轮阶段总入口 | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_01_broad_all_window\analysis\index.html` | 已完成固定模板阶段入口 |
| 当前累计快照 | `results\all_completed_union_analysis\snapshots\eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5` | 当前 109 阶段 AI 主导探索快照：37,058 个唯一坐标和 11,749,606 笔交易；累计逐笔发布复用 5,320 个区块，生成 31,738 个新区块 |
| 临时阶段 | `results\campaigns\v4_4_temporary_close_fill_validation_20260802\single_combo_all_window` | 已完成：1 坐标、3,882 笔交易 |
| 阶段总入口 | `results\campaigns\v4_4_temporary_close_fill_validation_20260802\single_combo_all_window\analysis\index.html` | 当前临时查看入口 |
| 阶段逐笔 | `results\campaigns\v4_4_temporary_close_fill_validation_20260802\single_combo_all_window\analysis\trade_review\index.html` | 当前临时逐笔入口 |
| V4.4 稳定总入口 | `results\all_completed_union_analysis\index.html` | 当前共享累计总入口；整组探索结束时的最终发布会在此刷新全部兼容已完成结果 |
| V4.4 稳定逐笔 | `results\all_completed_union_analysis\trade_review\index.html` | 当前共享累计逐笔入口；整组探索结束时的最终发布会刷新这一份大型页面 |

## 后续交付规则

所有已有阶段总入口与逐笔页面均作为历史证据保留。Continuation 第八至第十五轮均在同一 campaign 根目录中闭合。多轮探索期间，中间轮次只闭合不可变原始证据与精简摘要，不重建共享 HTML。整组探索结束后统一发布上面的两个稳定累计入口一次，其中包含全部新旧兼容坐标。只有用户明确要求时，才允许提前发布。当前未授权下一轮。

## 来源边界

## 成本参考切换

后续衍生成本后排名正在由保留的固定 3.56 bps 旧版视图切换到 `runtime_inputs\cost_models\k200m_current_notional_cost_reference_20260803.json`。该参考冻结 KRX 乘数、最后真实 15 秒 K200M 价格、带日期 KRW/USD 汇率、2 bps 往返滑点、6 USD 往返手续费，以及每项推导的 KRW／bps 数值。已完成原始阶段保持不可变；本次切换不重跑或覆盖原始结果。

用户最新决定覆盖固定未来成本规则。现有 K200 衍生结果保留历史成本参考。每个新 campaign 绑定冻结的参考价乘以 point value 成本模型，并在运算以前由经审阅计划绑定该配置。

`SOURCE_FINAL_V7` 为 `54a0a272b2b2215e60ff0649796bef3d3babfd69893b4cd93c0da4d69dbeb4cb`；其闭合 memo 位于 `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\source_final_v7_dynamic_k200m_cost_20260803.md`，SHA-256 为 `4d83aba6ed71fbee24186652fbc477726ba9b5e84f551165961523e0323baae1`。

复制的 V4.3 计划和源码清单位于 `runtime_inputs\provenance`。此前的 V4.3 活跃管理记录位于 `project_management\99_archive\v4_3_terminal_management_snapshot_20260802`。这些内容只作为记录，不参与 V4.4 运行。

当前累计 lineage 只包含 campaign `v4_4_positive_entry_signal_repair_20260805`。已经退出权威的修复前 campaign、旧临时验证 campaign 与失败局部快照 `1ba05465b49a1de45c407bfd9b4456eeb83c92b9dadea8d1c5d060a40ae22d98` 分别作为历史或恢复证据保留，不是稳定路由目标。

当前 Continuation 阶段继续使用 campaign ID `v4_4_positive_entry_signal_repair_20260805`，采用 `continuation_round_*` 阶段命名，并作为该当前根目录的直接子目录。累计构建器只有在严格结果身份筛选下才能扫描 `results\campaigns`；不兼容历史兄弟 campaign 会被排除并记录。

## 闭合哈希

- 第十五轮累计指针：`6720ce8ba979dfd641e2db90cc78c6fcf958237c0c8fe34433cb96f7c58bb1bb`；快照 `45b9a08396493a53ece45bd62af91070fb6b443539cd2e12ae5aeac5c756faad`；分析 `ab7141a0cf210061b1dc4ef94981c39b56f29507f21e2f51bc6e8bff65e7fe80`；完成 `d9bc1f8024eda348a8fc77c599c3957ced063af1b0af98b55f3f14b508ef2374`；逐笔 `f1fb5b5ce407a6bbd8513dd09d2375c806374266ffc54915aa89f9307ab39b07`。它包含 4,499 个坐标、683,835 笔交易和 18 个兼容阶段；`parameter_acceptance=none`。
- 合并跨品种闭合后的当前源码身份：累计构建器 `4bdb447200e740f02a65ed2bb52b2c5d3b4130a0c541edfc50702efb580e93a3`；阶段分析器 `fc0142d046c90d432e017808647e52400647e0f7ad8606da6b70fac7d1a88aba`；逐笔生成器 `be0d0e08dc5c6b131d39dc0c1c0bea3ec3462c6465cffffd5203836c1162fdd1`。第十五轮的发布内存与索引化逐笔修改均已保留；引擎、原始证据、排名公式与 HTML 语义保持不变。
- 历史 `SOURCE_FINAL_V17` 清单：`d707f560080a4c3f589ec24e1161e2126211fbbd09366f2279f4845554c73a26`；28,810 bytes。当前 V20 清单闭合第十一至十三轮及十六阶段累计快照。引擎、运行器、模板、原始结果语义与共享交付架构均未改变。
- 当前累计指针：`cada016193dd551dd4a0930b739dc04bd5907732ad84cfa71aef457ef7ee2983`；快照 `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7`；分析 `0cd061ca54d17acc1673f42c3f4d273b66dbdaa554473f6f2699a6695c3435b5`；完成 `22aa69610483145e946d9458bbf09271a1ad006c988f1b3f407f1049daa26db5`；逐笔 `1ead44e9bd49f0cfc5d95f073f24e68925879481dd8fad122c4653e928c899a6`。
- Continuation 第十六轮计划：`1802a9d15118069be1aba19f0918e7a8bed4f6cf94010dd3942936cdc12b113f`；完成清单 `0e569a9458af3f2146c6d5874c123e14ad70993d104ca7f4d30dc7ed80946060`；阶段分析 `4911577da5baa01836204b33bb42b282922a7f454cd3039ac791240e0a34c46c`；交付审计核验 4,728 个唯一声明产物，零不一致。
- 当前 `SOURCE_FINAL_V12` 清单：`40bb10ff06f9ca662e1052edd8655303cab2d475da3512d916768c84c8ee4763`；22,556 bytes。V12 保留累计总入口修复，并将逐笔页面的全部行情区间矩形改为语义半透明填色与零宽透明边框。快照 `2da2a0dff4c1890627f78c0556a2d8504ff0f384f77db147da54572367635a52` 仅更新修正后的共享逐笔 HTML 与相应展示元数据；原始逐笔交易、收益、排名及累计数据均未改变。此前 V2 至 V11 的源码权威继续作为历史证据保留。当前运行输入清单为 `df67b04e0f3e6c516f8456d222b49841b417dc793779ac71403db0e86da530a8`／2,683 bytes，并递归绑定当前无边框区间逐笔模板。
- 逐笔交易记录打包工具保持 `tools\\package_v4_4_with_trade_records.ps1`，`1e4a12a7c2e138ca253b9d2be9f78973af54f7f7220e0c961966e76bba2cd191`，17,698 bytes。现有最终 ZIP 与相邻证据继续作为历史 V4 发布权威，不属于当前 V5 包字节。
- 当前历史逐笔模板为 `0b30ac0e0ed189d83a6cf962cf1bf826a82d312bfd07ca294dc9952ece3d6fc9`；阶段分析器 `a6c33456e11a0cb82cbd7e248ea4bf15e3c74292891452de835564e3564355d3`；逐笔生成器 `c9055734ec66b4f1aec53af90194f9003e44044fdab2e0cfe7bc55a34a24b4f2`；阶段 QA `6f572c4b4cff22002e2ea8cdf605e461085b85597152f5c00c18895327db3214`；逐笔测试 `7ca723d4e6d5f8d60694912984e5458988648ac69f21d710507e9cd94c14b624`。
- 已冻结第一轮计划：`1424dc17862a2bfe0b8f0439fef061e64efc487c5057b7cff64498ed40a78046`。
- 第一轮完成清单：`c9532f77b626f647dcfe7b1fdc09ee76b2b895e851da0b98f6206b26ff1e6539`。
- 第一轮阶段分析清单：`3c55f1222db2586f8e5fb4dee5800e4534ee695c9ce7e101fd9a7e5f7c56d03f`。
- 当前修正源码累计指针：`725caf1515554a67ad6c2cccec43da14261d9441b2b1e38b688cd4950749e79e`；快照 `a55ee98105958c699a29a1e32a9ccd0f3afc60cd82b29b5d88d74068fa59219a`；分析 `2f8021494eef8da99a8a62866eb9976da1cc45b1cac3d8f81f47d3fa88eb9ea3`；完成 `6db44fb04515ea5fdd918c5d2023f44229667de30786c1c7b071ab900cae9363`；逐笔 `b5859cb46825542eafdcd75f98cf71caf9699948ec3819443755b00d047bfe5a`。保留的旧快照 `ce1e20f7...` 继续作为历史证据。
- 已冻结第二轮计划：`f90dbf5563ae9128304d5b48b902db440d9b73a4af3c0a7654079ef73628f7fd`。
- 第二轮完成清单：`b97a4811ebf3520fb6086ec26d5dfa149b2154d989ca6d94a1149f8d7a28350c`。
- 第二轮阶段分析清单：`be731db4bc274b85271ea5bb26421aba0e8f8b8f538825e3925e5d459aa47164`。
- 已冻结第三轮计划：`46c95b24feab49b6f260a0e8f1e1125fd74c34c6a0e268b89e0e1fb83a6d9b8c`。
- 第三轮完成清单：`edec2c43ecf4c4035a690f763b6a4d68d8be8f97a9da40ec9b3f3aac20ff25ea`。
- 第三轮阶段分析清单：`d163c36e471c732181f93d85d3e9752f8938b3cb14d92c025648e53815044d2f`。
- 终局解释 memo：`d740561e13e7fe5da6b0dc96e826c4b155f99a56b34b899c9b2b77653d373837`。
- 最终只读总审计：`58009d4eb357e3022de423d63310c9821fd167e518b43c739dc8efea6a694c0e`；可复用脚本 `86aa89a37b4780ca1b7719ced215d9566a77e5cc438356b8d05f0b7563b2f8eb`。
- Continuation 设计 memo：`94966f8096bb5317e09d2eb178a10cc9b1f31d3871c2bb8a3efa61218f2b9412`。
- 已冻结 Continuation 第一轮计划：`481fd28365757f739cb0e260d3cc36a4390db9cde9b1f1ccf3063aefdb8c9bf5`。
- Continuation 第一轮完成清单：`0990507be75526618663b4e08a3d628fd7af856dd692c9d9c3313de2cd0fdf6d`；旧源码阶段分析 `2d87ad79920740980141aaef6f7e5c4b650ebcf99dd5e15c485c1c05986a0f70`；旧源码交付证据 `f6409f53da7f33d6de2b6c3074b8cf11ef41e60e314c2f00de81950d84044909`。
- 修正源码 Continuation 第一轮阶段分析：`e643b934f32e1f84db963c33ea8e4c24276da462c5e6a2199b5aa4b369f99b2f`。
- 已替代的运算前 V4 绑定 Continuation 第二轮计划：`05d0d2edb604ce80cea391e13f62c0caa56cb4fc23caec98bb25cd590cb21dcb`；局部 validate-only 指纹 `976fac8d1e6ce5b280127ad6d7000116e2280a81242e4d2f754cccac7b139e35`。它继续只作为不可变的四文件元数据证据保留。已交付的 V6 计划为 `plans\\v4_4_cost_adjusted_multiround_20260803_continuation_round_02_dual_objective_local_v6_all_window.json`，SHA-256 `d982267710abab0355a37271c18a25df40decc3d9f846f82030a3ecbeab82a07`；其完成根目录为 `results\\campaigns\\v4_4_cost_adjusted_multiround_20260803\\continuation_round_02_dual_objective_local_v6_all_window`。不可变 completion 为 `2c0364fda3fc17cd09419d0a6003e6a3e6d7f1da035b8228c201fd21a6570d6e`；当前累计快照为 `2da2a0dff4c1890627f78c0556a2d8504ff0f384f77db147da54572367635a52`。任何后续轮次均等待本轮交付证据与新的经审阅哈希绑定计划。
- Continuation 准备清单：`ff7e0b7790e52b1bf86ac185d79c148552c0b14240453dd0ed512ca476d58a27`。
- 第二轮解释 memo：`42d6a69150ca939d0cce13fa424a80a2fc79d785097c4e916efb14545a3fe262`。
- 第一轮解释 memo：`7e500cd13ccb9c7dc928b5a8cb1bc1f3f9335b2765b577e58372ccbec1f26411`。
- 多轮设计 memo：`1286acf656babb9ad885ef8e75ac6cffb8afbde7a9f6a51dc5601d8cfb17bf24`。
- 临时计划：`b44a2d75722dd582e588b146760ce78841cec85a2ea5f2bca65a5db6826227c6`。
- 完成清单：`eddaf0d6335b2e718e4e78d1d1e5fc06aa16a3e4bb559ef062a7c357de36770e`。
- 历史临时累计指针：`995a4e625051318eae35fbcfe1bab3f1157ef80938be3e24ddbfd612ae60226a`；快照 `cae521e3066e80d1e3d1f5b7bc9c68ba20737cde67493723bfb482d5a0e80181`。

## P0-8 修复权威

- 受影响坐标清单：`D:\Code\backtest-release\Backtest V4.4\.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\p0_8_zero_signal_affected_combos.csv`；4,383 个坐标，SHA-256 `3469a630fc47e2614594536d3301746d8e68811e1171fd19f027933accf88c44`。
- 审计摘要：`D:\Code\backtest-release\Backtest V4.4\.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\p0_8_zero_signal_affected_summary.json`。
- 已退出活动状态的 campaign：`F:\Backtest\Backtest V4.4\results\staging_recoverable\p0_8_zero_signal_retired_20260805\v4_4_cost_adjusted_multiround_20260803`。
- 历史审阅快照 `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7` 在修正替换发布以前只具有参考权威。
