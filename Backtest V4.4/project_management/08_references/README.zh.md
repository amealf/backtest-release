# 参考材料

- `D:\Code\backtest-release\Backtest_V4.41_source_release_20260809.zip`：方案 A 的 Windows 正式源码发布包；相邻 `.sha256` 与 `.audit.json` 文件负责整包完整性和打包事实。
- Git 标签 `V4.41`：已发布 `v4.41` 分支上的正式源码发布身份。
- `project_management\screenshots\trade_review_inline_peer_20260809`：Chrome Beta 检查证据，覆盖配对按钮居中、同页载入训练结果、隐藏内嵌参数卡片以及关闭／卸载行为。

- `F:\Backtest test 6.11\02_DATA_AND_AUDITS\market_data\k200_historical_ticks\k200_ticks_20260523_to_20260723_20260723_014534\README.md`：K200 原始 Tick 下载、已审计 016M／016U 边界与合并契约。
- `F:\Backtest test 6.11\02_DATA_AND_AUDITS\market_data\k200_historical_ticks_supplements\k200_postroll_supplement_20260723T024400_to_20260728T161430_20260728_151540\README.md`：第一批保留的换月后补充数据与更新血缘。
- `F:\Backtest test 6.11\02_DATA_AND_AUDITS\market_data\k200_historical_ticks_supplements\k200_postroll_supplement_20260728T161430_to_20260807T032200_20260807_022303\README.md`：当前最新 K200 Tick 补充数据、合约规则、更新时间与主要原始／衍生／审计文件。

- `research_variants/short_momentum_net_drop_rebound_v4_4/SOURCE_MANIFEST.json`：当前源码与行为身份。
- `runtime_inputs/RUNTIME_INPUTS.json`：仓库相对运行输入。
- `runtime_inputs/data_preparation/data_preparation_manifest.json`：schema 5 准备数据身份。
- `research_variants/short_momentum_net_drop_rebound_v4_4/plans/v4_4_temporary_single_combo_close_fill_20260802.json`：临时重跑计划。
- `research_variants/short_momentum_net_drop_rebound_v4_4/plans/v4_4_cost_adjusted_multiround_20260803_round_01_broad_all_window.json`：已冻结 372 坐标第一轮计划。
- `research_variants/short_momentum_net_drop_rebound_v4_4/plans/v4_4_cost_adjusted_multiround_20260803_round_02_broad_local_all_window.json`：已冻结 247 坐标第二轮大范围＋局部计划。
- `research_variants/short_momentum_net_drop_rebound_v4_4/plans/v4_4_cost_adjusted_multiround_20260803_round_03_terminal_local_all_window.json`：已冻结 212 坐标终结型第三轮局部计划。
- `research_variants/short_momentum_net_drop_rebound_v4_4/plans/v4_4_cost_adjusted_multiround_20260803_continuation_round_01_broad_span_all_window.json`：已冻结 528 坐标 Continuation 第一轮大范围计划。
- `runtime_inputs/templates`：哈希受控的历史总入口／逐笔模板、Plotly 与市场选择器。
- `runtime_inputs/provenance`：保留的 V4.3 身份与计划，只用于记录。
- `research_variants/short_momentum_net_drop_rebound_v4_4/handoffs/v4_4_cost_adjusted_multiround_20260803/analysis_reports`：九份已保留 campaign 设计、解释、闭合与交付报告的稳定目录；这些报告原先位于临时 `.omo` 团队工作区。
- `research_variants/short_momentum_net_drop_rebound_v4_4/handoffs/v4_4_cost_adjusted_multiround_20260803/analysis_reports/README.md`：上述报告的来源与迁移说明。
- `results/all_completed_union_analysis/index.html` 与 `results/all_completed_union_analysis/trade_review/index.html`：未来刷新使用的当前共享累计总入口与逐笔入口。
- `runtime_inputs/templates/historical_v4_trade.html`：当前逐笔展示权威文件，包含开仓理由第 2 项空心椭圆及桌面端「参数」标签上移 6 像素。

KRX Mini KOSPI 200 合约规模规则是当前成本输入：期货价格 × 50,000 KRW。哈希绑定的当前名义价值参考还记录了活动 15 秒输入的最后真实价格，以及将 6 USD 手续费换算成 bps 所用的带日期 KRW/USD 收盘参考。最小变动单位资料继续只作为来源背景保留。

当前结果与交付路径见 `SOURCE_OF_TRUTH.zh.md`。

临时 `.omo` 团队工作区及其中非规范的 JSON／Python 审计辅助文件已于 2026-08-08 移入回收站。已接受的结论继续保留在当前状态与历史文档中；上述九份 Markdown 报告是当前稳定参考。
