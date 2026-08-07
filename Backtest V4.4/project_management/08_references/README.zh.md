# 参考材料

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
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/v4_4_cost_adjusted_multiround_design_20260803.md`：已授权 campaign 的非执行型大范围／细化设计与适应规则。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/round_01_interpretation_and_round_02_design_20260803.md`：已闭合第一轮解释及由证据推动的第二轮设计。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/round_01_delivery_final_C.md`：独立的第一轮交付、哈希、浏览器 QA 与视觉 QA 证据。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/round_02_interpretation_and_round_03_terminal_design_20260803.md`：已闭合第二轮解释及终结型第三轮设计。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/round_02_delivery_final_C.md`：独立的第二轮交付、累计、哈希、浏览器 QA 与视觉 QA 证据。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/round_03_terminal_interpretation_and_campaign_closure_20260803.md`：规范终局解释与 campaign 闭合。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/round_03_terminal_delivery_final_C.md`：独立的终结轮交付、最终累计、哈希、浏览器 QA 与视觉 QA 证据。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/final_read_only_total_audit.json`：独立最终汇总身份、数量、唯一性、产物、锁及进程审计。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/final_read_only_total_audit.py`：可复用的只读总审计脚本。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/v4_4_continuation_subseries_design_20260803.md`：非执行型 continuation 设计、累计边界及每轮强制交付契约。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/continuation_phase_readiness_inventory.json`：独立源码、完成／active／pending、锁、进程与临时内存准备清单。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/continuation_round_01_immutable_closure.json`：Continuation 第一轮不可变原始闭合与一致性证据。
- `.omo/teams/019fbd76-9439-7b10-a7e1-6b34003778c8/artifacts/continuation_round_01_delivery_final_C.md`：保留的历史四入口交付、哈希、浏览器、桌面／移动与人工视觉证据。已有页面继续保留；后续交付使用共享累计入口。
- `results/all_completed_union_analysis/index.html` 与 `results/all_completed_union_analysis/trade_review/index.html`：未来刷新使用的当前共享累计总入口与逐笔入口。
- `runtime_inputs/templates/historical_v4_trade.html`：当前逐笔展示权威文件，包含开仓理由第 2 项空心椭圆及桌面端「参数」标签上移 6 像素。

KRX Mini KOSPI 200 合约规模规则是当前成本输入：期货价格 × 50,000 KRW。哈希绑定的当前名义价值参考还记录了活动 15 秒输入的最后真实价格，以及将 6 USD 手续费换算成 bps 所用的带日期 KRW/USD 收盘参考。最小变动单位资料继续只作为来源背景保留。

当前结果与交付路径见 `SOURCE_OF_TRUTH.zh.md`。
