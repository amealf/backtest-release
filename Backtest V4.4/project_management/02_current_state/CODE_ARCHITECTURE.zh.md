# 代码架构

## 外部审阅修复边界——2026-08-09

`build_v4_4_review_delivery.py::_native_trade_frame_fast` 遇到合法的零交易坐标时返回空表。多进程 worker 仍写入该坐标的正常 `NATIVE_COMBO` 记录，同时写入 `NATIVE_TRADES=[]`、`waited_count=0` 和 `maximum_wait=0`。线程与进程发布路径由此保持一致，也不会制造交易。

`analyze_v4_4_scenario_3_stage.py` 继续在 `report_data.js` 中保留完整阶段 `instrumentProfile`。精简的 `analysis_data.js` 只加入 `instrumentSummary`，包含品种 ID、显示名称、排序谱系、成本模型 ID、实验模式和情景策略。本地配置路径与策略实现路径不会进入精简浏览器载荷。

`build_v4_4_cross_instrument_comparison.py::_comparison_html` 允许单元测试传入基础 CSS；正常发布仍从当前累计页面读取 CSS。跨品种测试可以验证按角色生成的动态名称，无需依赖 F 盘快照。展示测试改为检查当前语义属性和「开仓理由」中的现行信息位置，不再依赖过期的 CSS 顺序或图内标签文字。

## 行情场景流程

`runtime_inputs\scenarios\market_catalog.json` 把选择器选项映射到一个品种、一个评价结果包、一个行情文件、一个时区和一个精确区间。`tools\build_v4_41_scenario_manager.py` 为每个区间生成压缩浏览器资源并发布选择器，界面代码不包含品种专用文件名。

选择器每次只加载一项已登记行情，选区保存在浏览器内存中，再通过 Chrome 的文件保存流程写出完整 `scenario_catalog.json`。载入控件导入整份多场景目录；新草稿使用下一个未占用的 `scenario_N` ID 和「新场景N」默认名称，仅名称可以修改。框选结果会规范为铺满图表高度的时间带。每个场景重复记录数据文件、评价身份、显示区间和全部选区，单独查看文件也能还原来源。

`tools\apply_v4_41_scenario.py` 解析绑定的评价结果包，根据 `trade_records\trades.csv` 判断每段选区，再用 AND 合并分段资格，筛选结果包中的参数总体，并生成 `analysis_data.js` 和当前总入口页面外壳。迁入的 V4.4 场景可以复用兼容的预计算资格字段；新增或编辑后的场景会流式读取不可变逐笔记录。逐笔链接通过结果包自己的 `trade_review\index.html` 打开。

## V4.41 交付打包

- `tools\package_v4_4_with_trade_records.ps1` 发布 `Backtest_V4.41_current_with_trade_records_20260808.zip`。脚本名保留 V4.4 策略主版本命名空间；交付包身份与根发布记录采用 V4.41。
- 交付包复制 `RELEASE.json`、根运行／配置文档、当前研究变体、`runtime_inputs`、`project_management`、工具、来自 `research_variants\short_momentum_net_drop_rebound_v4_4\handoffs\v4_4_cost_adjusted_multiround_20260803\analysis_reports` 的九份保留规范报告，以及已完成 campaign 阶段的不可变原始逐笔记录与已有衍生逐笔记录。`.omo` 不再是打包依赖。
- 完整结果载荷、HTML 快照、`.git`、`.omo`、依赖、缓存、浏览器配置与编译字节码不进入 ZIP。原始 batch 逐笔账本为强制项；衍生阶段账本只在该已完成阶段实际生成时纳入。可恢复终结流程检查清单身份、ZIP 名称与大小和整包 SHA-256，不再建立重复解压目录。

## 发布身份边界

- `RELEASE.json` 管理当前界面发布版本（`V4.41`），并声明策略／排序主版本继续使用 V4.4。引擎清单、参数身份与累计纳入规则继续采用现有 V4.4 合同和排序谱系。
- 页面生成程序在当前标题中显示 V4.41。`v4_4_engine.py::VERSION_LABEL`、`SOURCE_MANIFEST.json::version_label` 与 `build_v4_4_combined_union_analysis.py::RANKING_MAJOR_VERSION` 保持 V4.4，使已完成结果继续兼容。

## 按日期区间保存的结果包与通用比较入口

- `tools\build_v4_4_evaluation_framework.py` 负责发布结果框架。结果包身份由 `instrument_id`、精确评价开始时间和结束时间组成。程序写入清单、实验说明、统一参数汇总、按候选集生成的浏览器数据、不可变逐笔记录链接和逐笔兼容入口，再生成比较方案与结果包读取页面。
- `tools\register_v4_4_evaluation_package.py` 根据声明式模板 `runtime_inputs\templates\EVALUATION_PACKAGE_SPEC.template.json` 登记一个新完成的品种／区间。说明文件把来源汇总列映射为统一参数和指标字段；登记过程创建日期结果包、硬链接不可变逐笔记录并更新共享目录，同时拒绝改写已有结果包。
- `results\evaluation_packages\catalog.json` 是机器可读目录。每个品种按精确区间建立目录。结果包不保存固定实验角色，因此同一份已完成证据可以被后续多份方案引用。
- `results\evaluation_comparison` 保存比较目录和方案。浏览器加载器读取每个已选结果包的数据，按 `combo_id` 建立查找表，按原字段顺序恢复保留页面需要的角色指标，随后运行未改动的比较界面。
- 历史大体积逐笔 CSV 在兼容结果包内使用同卷硬链接。结果包逐笔入口保留 query 和 hash，再跳转到字节不变的历史页面。以后生成的原生结果包可以在自己的目录内保存区块，并增量复用未变化区块。
- `tools\qa_v4_4_evaluation_framework.mjs` 用 Chromium 比较保留页面与结果包页面，检查标题、全部可见正文、首行数据、可见行数和控制台／页面错误，并保存两张截图。
- `run_v4_4_resumable_campaign.py` 已经通过品种配置解析行情与准备数据，并从计划读取 `train_start`／`train_end`。新的存储层只移除目录身份对 K200 和训练／测试角色的依赖，不改变引擎或 runner 语义。

## 当前 K200 优参数后续一个月重放

- `tools\run_v4_4_k200_current_optimal_forward_initial.py` 从训练期多指标冻结候选，对非对照坐标执行早期时间迁移结果反连接，写入 schema-4 精确迁移计划，调用既有四工作线程可续跑执行器，并生成精简比较与 Markdown，不生成 HTML。
- 候选必须满足训练期成本后收益为正。程序在反连接新增坐标以前选取八个训练视图对照；冻结文件记录每个精确坐标是否曾有后续评价。
- 程序复用 `run_v4_4_k200_temporal_migration.py` 的计划执行与成本／依赖分析，因此引擎、成交、成本、准备数据和 gap 语义均未改变。

## K200 训练／测试／SI 三组迁移流程

- `tools\run_v4_4_train_test_si_triple_migration.py` 对保留的 250 组 SI 候选执行精确反连接，冻结 100 组只依赖 K200 时间迁移证据的新增候选；四个工作线程完成 SI 评价，再把合并后的 350 组重放到 K200 测试期，最终写入三组收益比较与解释报告。
- `build_v4_4_cross_instrument_comparison.py` 读取可选的 `source_test` 合同，使用相对角色显示 K200（训）、K200（测）和 SI 收益列，把每列总收益按钮路由到对应角色的逐笔证据，并根据保留交易记录生成专用 K200（测）逐笔入口，无需重跑策略。K200（训）复用累计 K200 逐笔入口；SI 保持增量逐笔入口。
- SI 最终逐笔发布将保留的 250 组运行作为 `incremental_parent_run`，复用 250 个区块，只生成新增 100 组区块。中间迁移阶段不发布 HTML。

## K200 时间迁移工作流

- `tools\run_v4_4_k200_temporal_migration.py` 从训练期多指标证据冻结 R1 候选，用四个 worker 运行可续跑阶段并写入成本与依赖摘要。后续候选由已闭合的更早时间片、训练期邻近组合和结构性对照组成。最后一段保持未见状态；完整区间重放明确标为事后描述。
- 计划与候选 CSV 位于 `research_variants\short_momentum_net_drop_rebound_v4_4\plans\k200_temporal_migration_20260807`；原始与派生阶段位于 `results\temporal_migration\v4_4_k200_temporal_migration_20260807`。内存门槛暂停时保留已完成批次，恢复运行不会改动候选冻结文件。
- 最终阶段写入一份时间迁移 HTML/CSV、一份 Markdown 报告和一份阶段逐笔分析。后续只刷新报告时复用已完成分析，不重新生成回测或逐笔区块。
- `build_v4_4_review_delivery.py::_stage_filter_bundle` 同时接受保留的 policy-neutral 准备身份与当前确认后低活动门禁身份，并继续核对来源和产物哈希。`analyze_v4_4_scenario_3_stage.py::_atomic_json` 把清单中的路径字段序列化为字符串。

## 逐笔区间统计展示

- `runtime_inputs\templates\historical_v4_trade.html` 负责配对逐笔按钮、无版本号标签标题、蓝色 `Z` 图标、理由面板指标框、精简交易选择器和底部图例。`build_v4_4_review_delivery.py` 注入可选的配对路由与研究合同身份，共享模板继续保持品种无关。
- `results\all_completed_union_analysis\trade_review_peer.json` 把当前累计 K200 逐笔页面绑定到后续测试逐笔页面。`build_v4_4_combined_union_analysis.py` 在当前快照刷新和后续累计外壳中读取这份小型发布合同；`build_v4_4_cross_instrument_comparison.py` 发布 K200 测试逐笔页面时提供反向链接。
- `runtime_inputs\templates\historical_v4_trade.html` 负责「区间统计」、Plotly 横向圈选、右栏紧凑结果框、基于现有 OHLC 的圈选后计算、清除原生选择状态、持续显示的浅色普通图形，以及关闭时恢复缩放模式。
- 同一模板负责互斥的「持仓检测」，并保留 Plotly 拖动缩放。图表记录 pointer-down，`document` 接收 pointer-up；移动超过五像素视为拖动，短按通过当前坐标轴映射到最近的可见 K 线，右键被排除。随后解析逐根来源 bar 或合并 K 线的末端来源 bar，查找所在交易，按引擎规则重建完成 W 候选从下一根生效的 max-W 回撤状态和 S 窗口累计低点状态，在所选 close 绘制蓝色点，并绘制更深的无描边蓝色「基准起点至 active low」填充。启用任一模式会收起参数抽屉。
- K 线 trace 使用 `whiskerwidth=0`，只移除高低点末端横线，不改变 OHLC 数值。
- `build_v4_4_review_delivery.py::refresh_trade_review_shell` 只重新生成 `trade_review\index.html`，并更新资源审计与逐笔清单；process payload 和各坐标逐笔区块保持不变。
- `build_v4_4_combined_union_analysis.py::_refresh_completed_snapshot_trade_shell` 更新已完成快照外层的分析与完成产物引用，不运行累计交易或逐笔区块生成。

## K200 增量 Tick 获取与启用

- `tools\download_k200_incremental_ticks.py` 从保留的 IBKR K200 `TRADES` Tick 检查点续传，每个下载目录独立保留，使用当前一秒即时恢复规则清理 Tick，生成补齐交易时段的 15 秒衍生数据和有序扩展候选。请求间隔保持 10.25 秒；IBKR 缺少完成回调时，用一页 500 笔沿同一游标前进，随后恢复每页 1,000 笔。
- `tools\write_k200_download_readme.py` 在每个保留的 K200 下载目录内写入并刷新 `README.md`。它从血缘清单生成更新时间记录，并写明已审计的主力合约规则、016M／016U 准确边界、未复权合并政策、请求区间、状态和主要文件。
- `tools\activate_k200_incremental_15s.py` 按严格时间顺序追加原活动 K200 来源和各段独立清理后的补充数据，替换仓库内的活动 15 秒文件，重新生成准备数据，并刷新品种配置、证明、运行输入和源码清单身份。启用数据不会重新计算历史结果阶段。

## K200 累计展示路由

- `build_v4_4_cross_instrument_comparison.py::publish_current_main_standalone_view` 将稳定根入口发布为保留 query／hash 的 `main\index.html` 跳转页，不再生成重复导航外壳与 iframe。
- `analyze_v4_4_scenario_3_stage.py::_legacy_v4_main_html` 负责当前 K200 标题、颜色、可见列顺序与名称、固定宽度排名按钮、每页 500 行的客户端分页，以及全宽表头固定滚动区域；页面不再生成「研究合同」。`publish_stable_main_assets` 只根据当前不可变快照载荷刷新轻量主展示。

## 双目的探索工具

- `tools\build_v4_4_k200_dual_purpose_round_15_plan.py` 生成确定性大范围覆盖与精确单参数细化区块，执行已完成坐标反连接，并写入经审阅的执行计划和审计。
- `tools\analyze_v4_4_k200_dual_purpose_round_15.py` 把计划区块身份附加到已完成阶段结果，分开比较大范围与细化分支，生成主视图与逐笔诊断，并写入经过反连接的下一轮交接。这两个工具不改变引擎、成交、成本或结果语义。

## 累计逐笔增量发布

- `build_v4_4_combined_union_analysis.py` 递归解析嵌套生成计划，在不保留全部交易表的情况下读取阶段汇总，并逐阶段处理保留交易。累计交易 CSV 流式写入临时文件，全部阶段闭合后再原子改名。
- 确定性命名且未变化的坐标区块通过同卷硬链接复用。缺失区块使用向量化逐笔转换和四个工作进程生成；排名、目录、process payload、汇总、清单与稳定路由只在完整兼容阶段集合上更新一次。
- 闭合阶段清单构成发布审计边界，累计发布不再重复执行完整交易文件哈希和逐笔成交审计；策略、结果、成本、情景与计划身份仍保持约束。
- 当前交付复用 5,320 个区块，生成 31,738 个新区块，以低于 2 GB 的发布器内存完成 37,058 个坐标和 11,749,606 笔交易。

## 与品种无关的契约

- `research_variants\short_momentum_net_drop_rebound_v4_4\contracts\STRATEGY_CONTRACT_V4_4.json` 只保存共享开仓、平仓、时序与状态机语义。
- `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\*.json` 绑定品种数据、准备证据、成本、gap、低活跃、可选情景和排名谱系。K200 已就绪；SImain 与 NQ 在用户提供成本和 gap 决策以前保持模板状态。
- `research_variants\short_momentum_net_drop_rebound_v4_4\campaign_contracts\CAMPAIGN_MANIFEST.template.json` 是 `transfer_exact`、`target_local_refinement` 或 `fresh_search` 的配置起点。
- `code\instrument_contracts.py` 验证这些文件并统一成本字段。名义价值按参考价乘以 point value 计算，同时保留 K200 旧字段。
- schema-v5 runner 计划绑定 campaign manifest、品种配置、模式、排名谱系与情景策略。旧计划不含这些字段，继续保留历史行为。

## 跨品种对比

- 目标品种逐笔增量发布由 `build_v4_4_review_delivery.py::build_stage_trade_review` 实现。运行配置可以声明 `incremental_parent_run`；`build_v4_4_cross_instrument_comparison.py` 解析已完成父逐笔目录，把确定性命名且未变化的组合区块硬链接到新交付，只生成缺少的组合区块，然后更新小型 shell、目录、process payload、汇总与清单。固定命令为 `build_v4_4_cross_instrument_comparison.py build --run-id <run_id>`。清单记录父运行、复用数量、新增数量与复用方式。
- `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_repaired_source_transfer.py` 可以把已经完成的精确迁移结果组成展示并集，无需重跑任一品种。它从用户表格移除批次来源字段，绑定经审阅的迁移方案，生成与品种无关的相对名称，发布一份合并目标逐笔与迁移报告，并让修复后独立结果转向合并展示。
- 对比构建器从运行配置绑定的 `migration_plan` 读取原品种／目标品种名称与样本区间。表格把原品种总收益放在目标品种总收益前一列，并采用三态排序。累计导航壳与跨品种页头提供采用新标签页的双向导航。
- `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_cross_instrument_comparison.py` 负责候选冻结、隔离的 SImain 评价、迁移诊断、CSV／JSON 发布、支持范围选择与全字段过滤的 K200 风格排名表、SImain 专用逐笔生成、已完成运行目录路由，以及不修改不可变快照的累计导航壳。构建阶段根据保留的原品种／目标品种逐笔 CSV 推导用于展示切换的毛中位单笔、回撤、胜率与迁移诊断字段；它不重跑目标评价，也不修改冻结候选。
- `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_strict_entry_transfer.py` 冻结只使用来源数据的 K200 前 20% 严格开仓 Pareto 批次，调用不变的 SImain 评价器，在各自独立冻结的迁移批次之间重算合并诊断，并发布一个共享对比入口和一个共享 SImain 逐笔入口。它记录阈值四分位及五维跨品种 Pareto 集合，不建立综合分数。
- `research_variants\short_momentum_net_drop_rebound_v4_4\code\qa_v4_4_cross_instrument_comparison.mjs` 负责对比页和 SImain 逐笔页的本地文件浏览器交互检查与桌面／移动截图。
- 对比读取当前完整累计快照及其逐笔 CSV，只向 `results\cross_instrument_comparison` 写入新结果。目标品种评价调用现有不可变引擎，不修改 campaign 阶段、计划、累计数据或冻结候选。
- 当前快照保持不可变。`build_v4_4_combined_union_analysis.py::publish_stable_main_assets` 在 `results\all_completed_union_analysis\main` 生成小型稳定展示：每个主表行只保留控制与表格实际使用的字段，精确链接继续进入当前快照，不复制或重新生成逐笔区块。稳定累计 `index.html` 是纯展示导航壳，通过 iframe 嵌入这个轻量主入口，并提供累计结果与跨品种页面链接；以后生成的累计快照继续独立绑定哈希。

## 结果存储间接层

全部 runner、analyzer 与 delivery 组件继续使用项目相对逻辑 `results` 根目录。在 Windows 上，`D:\Code\backtest-release\Backtest V4.4\results` 通过目录联接解析到物理根目录 `F:\Backtest\Backtest V4.4\results`。这样可以保留历史绝对路径，同时让后续每个结果字节存放于 F 盘。源码与运行输入继续位于 D 盘。

## 当前流程

1. `data_preparation/prepare_dataset.py` 生成仓库内准备数据、确认时排除时间戳与低活跃开仓门禁状态。
2. `code/v4_4_engine.py` 计算开仓、平仓、W 基准与逐笔审计。
3. `code/run_v4_4_resumable_campaign.py` 验证冻结计划并写入不可变批次结果。
4. `code/analyze_v4_4_scenario_3_stage.py` 验证完结阶段，读取阶段绑定的品种成本模型；旧阶段使用 K200 默认参考。它在不修改原始收益的前提下推导每笔成本视图，并生成独立毛收益／成本后排名；默认使用成本后口径。品种配置可以不含情景，此时阶段只展示全坐标视图。
5. `code/build_v4_4_review_delivery.py` 使用四个工作线程生成固定历史模板的逐笔页面。
6. `code/build_v4_4_combined_union_analysis.py` 递归发现指定 `campaigns_root` 下的已完成阶段，纳入已接受的 V4.4 K200 大版本排名谱系，把每个阶段的实现与数据准备身份保留为来源证据，并发布累计快照与稳定入口。读取闭合阶段所需的结构字段继续验证。

累计构建器按每个来源阶段绑定的成本模型计算。V4.4 内引擎、数据准备、策略或结果身份的小幅差异不会拆分已接受的 K200 大版本排名谱系。升级到 V4.5 或迁移到其他品种时，使用另外的累计根目录与排名谱系。

## 边界

原始运算、衍生成本分析与累计发布保持独立边界。中间轮闭合不可变原始输出与小型汇总；运行器只有收到 `--publish-html` 才发布 HTML。整组探索结束后，一次明确发布从全部兼容已完成阶段生成共享累计总入口与共享逐笔入口。已有阶段页面作为历史证据保留。当前运行输入采用仓库相对路径；`runtime_inputs/provenance` 中的历史绝对路径只用于来源记录。

`analyze_v4_4_scenario_3_stage.py::_legacy_v4_main_html()` 将固定主入口模板转换为 V4.4 累计入口。它隐藏摘要卡条，由 `DATA.coordinateCount` 填入 `#all-strategy-count`，并为 BH、E、W、S 渲染独占整行的分钟区间多选控件。每个轴对已选区间取并集，排名前再对四个轴条件取交集。`main_summary_payload()` 定义主入口资料边界；生成的 JavaScript 按方法与基准策略预建索引，筛选和排序只排列整数行索引，不克隆行对象。
