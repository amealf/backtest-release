# Current Valid Files

## Formal V4.41 source release — 2026-08-09

- ZIP: `D:\Code\backtest-release\Backtest_V4.41_source_release_20260809.zip`.
- SHA-256 sidecar: `D:\Code\backtest-release\Backtest_V4.41_source_release_20260809.zip.sha256`.
- Audit: `D:\Code\backtest-release\Backtest_V4.41_source_release_20260809.zip.audit.json`.
- Package source: `tools\package_v4_41_source_review.ps1`; the retained filename is historical, while its default output and embedded state now describe the formal Windows source release.
- Git release identity: tag `V4.41` on the published `v4.41` branch. Acceptance requires an independent extracted-package result of 112 passed, 2 skipped, and 0 failed.
- Scheme A publishes GitHub source plus the compact ZIP, sidecar, and audit. Multi-gigabyte historical ledgers and the 2026-08-08 complete local handoff remain outside the public release assets.
- `Backtest_V4.41_source_review_20260808.zip` remains immutable historical review evidence and is not a formal release artifact.

## V4.41 market-scenario system — 2026-08-08

- Market registry: `runtime_inputs\scenarios\market_catalog.json`.
- Saved-scenario registry: `runtime_inputs\scenarios\scenario_catalog.json`.
- Scenario contract and operation note: `runtime_inputs\scenarios\README.md`.
- Selector source and builder: `runtime_inputs\templates\market-intuition-selector.html` and `tools\build_v4_41_scenario_manager.py`.
- Selector entry: `results\market_scenario_manager\index.html`.
- The selector treats `scenario_catalog.json` as one multi-scenario directory: the import button loads that complete file, the ID is generated from the next unused `scenario_N`, and the editable name defaults to the next unused `新场景N`. Selection bands are normalized to the full chart height, so only the time interval is selected.
- Scenario application: `tools\apply_v4_41_scenario.py`; outputs live under `results\scenario_analysis\<scenario_id>`.
- The three migrated scenarios remain bound to K200 2026-05-26 through 2026-07-08. Scenario 1 qualifies 4,406 coordinates, Scenario 2 qualifies 6,292, and Scenario 3 is an explicit zero-coordinate population in the current 37,058-coordinate cumulative evidence.

## Historical compact external-review package — 2026-08-08

- ZIP: `D:\Code\backtest-release\Backtest_V4.41_source_review_20260808.zip`
- SHA-256 sidecar: `D:\Code\backtest-release\Backtest_V4.41_source_review_20260808.zip.sha256`
- Audit: `D:\Code\backtest-release\Backtest_V4.41_source_review_20260808.zip.audit.json`
- This immutable archive is historical review evidence. It does not replace the 2026-08-09 formal source release or the complete local handoff below.

## Current portable handoff — 2026-08-08

- ZIP: `D:\Code\backtest-release\Backtest_V4.41_current_complete_20260808.zip`
- SHA-256 sidecar: `D:\Code\backtest-release\Backtest_V4.41_current_complete_20260808.zip.sha256`
- Audit: `D:\Code\backtest-release\Backtest_V4.41_current_complete_20260808.zip.audit.json`
- Package source: `tools\package_v4_4_with_trade_records.ps1`; V4.41 is the release identity and V4.4 remains the strategy/ranking major lineage.

## Active release identity — 2026-08-08

`RELEASE.json` is authoritative for the active presentation release (`V4.41`) and its compatibility boundary. Strategy and result identity remain authoritative in `research_variants\short_momentum_net_drop_rebound_v4_4\SOURCE_MANIFEST.json` as V4.4. The cumulative publisher continues to admit completed V4.4-compatible stages by the existing major-version and ranking-lineage checks.

## Current K200 train/test per-trade navigation and layout — 2026-08-08

- The K200 training per-trade entry remains `results\all_completed_union_analysis\trade_review\index.html`; it forwards to the current snapshot and shows `显示测试集`. The K200 test entry remains `results\cross_instrument_comparison\runs\k200_train_test_si__combined_350_v56_20260807\trade_review_k200_test\index.html` and shows `显示训练集`.
- Both controls preserve the exact selected `combo_id` and bind the destination research contract. They now open the paired review inside the current page. `隐藏训练集` / `隐藏测试集` and Escape close the overlay and unload its iframe. A missing exact coordinate is not replaced by a nearby parameter set.
- The paired-review button uses the same centered button vocabulary as the surrounding toolbar and never shows a hover underline. The embedded page omits its `当前参数` card; the outer page is hidden while the overlay is open and restored on close.
- The redundant trade-picker heading/count and secondary summary line are removed. Drop, ratio, entry baseline, entry threshold, W baseline, and active-low values now render as compact colored boxes immediately below the reason heading; they are no longer Plotly annotations over price data. The Plotly legend is below the x-axis.
- The browser tab title is `组合平仓逐笔查看` and the favicon is a blue `Z`. Shared source: `runtime_inputs\templates\historical_v4_trade.html`; focused QA: `tools\qa_v4_4_trade_review_layout.mjs`; retained inline-review evidence: `project_management\screenshots\trade_review_inline_peer_20260809`.

## Current date-based evaluation-package framework — 2026-08-08

- Stable comparison entry: `results\cross_instrument_comparison\index.html`. It now redirects to the package-backed reader at `results\evaluation_comparison\index.html`; the retained 350-coordinate run page and all existing main/per-trade pages remain byte-preserved.
- Current-evidence adapter: `tools\build_v4_4_evaluation_framework.py`. Generic package registrar: `tools\register_v4_4_evaluation_package.py`, using `runtime_inputs\templates\EVALUATION_PACKAGE_SPEC.template.json`. Browser parity check: `tools\qa_v4_4_evaluation_framework.mjs`. Machine audit: `results\evaluation_comparison\compatibility_audit.json`.
- Evaluation catalog: `results\evaluation_packages\catalog.json`. Current packages are `K200\20260526T000000__20260708T235200`, `K200\20260708T235215__20260807T032145`, and `SImain\20260129T000000__20260223T235945`.
- Each package contains a manifest, exact-date experiment record, parameter summary, current candidate-set browser projection, hard-linked immutable trade records, and a per-trade compatibility entry. The package names contain no train/test/transfer role.
- Current comparison plan: `results\evaluation_comparison\comparisons\K200_20260526T000000__20260708T235200__K200_20260708T235215__20260807T032145__SImain_20260129T000000__20260223T235945\comparison_plan.json`.
- Compatibility closure: all 350 rows and field values match the retained comparison exactly; the comparison-page HTML differs only at the data-source script boundary. Old and new comparison pages have identical title, body text, first-row data, and rendered row count, with zero browser errors. The later approved K200 training/test per-trade presentation update changed only their small HTML shells and manifests; trade data, chunks, metrics, main entry, retained comparison, and SI per-trade evidence remain unchanged.
- This is now the project-wide result-storage contract. Any executable instrument profile and exact interval may create another independent package and may be compared through a plan without changing existing package identities.

## Current K200 optimal-parameter initial forward replay — 2026-08-07

- Current result root: `results\temporal_migration\v4_4_k200_current_optimal_forward_initial_v2_20260807`; interpretation: `INITIAL_REPORT.md`; complete comparison: `comparison.csv`; machine summary: `summary.json`.
- Frozen plan: `research_variants\short_momentum_net_drop_rebound_v4_4\plans\k200_current_optimal_forward_initial_v2_20260807.json`; candidate freeze: the adjacent `_candidate_freeze.csv`.
- The corrected freeze contains 100 unique training-period cost-positive candidates. Six are retained previously evaluated headline controls and 94 are exact coordinates not previously run over the later month. Selection reads no target metrics.
- Result: 29/100 are cost-positive over `2026-07-08 23:52:15` through `2026-08-07 03:21:45`; 8/100 have positive non-gap return; median return is -0.671%; train/test Spearman is -0.346. The training total-return leader returns -1.567% later. `parameter_acceptance=none`.
- The earlier root without `_v2_` is historical and invalid for decisions because its first candidate definition admitted 12 cost-negative training rows. Its local `STATUS.md` points to the replacement.

## Current K200 train/test/SI migration — 2026-08-07

- Stable three-return entry: `results\cross_instrument_comparison\index.html`; current run: `results\cross_instrument_comparison\runs\k200_train_test_si__combined_350_v56_20260807`.
- The page displays K200（训）, K200（测）, and SI total return together for 350 candidates. The rank column is labeled `排名` and its fixed-width blue button displays only `#N`, opening SI per-trade evidence in a new tab. Each of the three total-return cells uses the same blue-button treatment and opens its own sample's per-trade analysis. The complete comparison is `migration_comparison.csv`; K200（训） reuses the cumulative K200 per-trade entry, K200（测） uses `trade_review_k200_test\index.html`, SI uses `trade_review\index.html`, and interpretation is `FINAL_TRAIN_TEST_SI_REPORT.md`.
- The run retains the completed 250-candidate SI population and adds 100 candidates frozen from closed K200 temporal evidence before their SI evaluation. The two populations have zero overlap. K200 test replay covers all 350 candidates.
- Evidence: K200 test is positive for 199/350, SI is positive for 275/350, and 149/350 are positive in all three columns. The new 100 contain 72 K200-test positives, 59 SI positives, and 43 three-column positives. Publication reused 250 SI trade chunks and generated 100.
- The most promising new neighborhood is E320/BH240/TRW12/K1.25/W6, M4.25–4.75, S340–370. Its median returns are +50.123%/+8.827%/+14.350%, but median K200-test non-gap return is -12.213%. This gap dependence prevents parameter acceptance. `parameter_acceptance=none`.

## Current K200 temporal migration — 2026-08-07

- Stable temporal ranking: `results\temporal_migration\v4_4_k200_temporal_migration_20260807\index.html`.
- Interpretation: `results\temporal_migration\v4_4_k200_temporal_migration_20260807\TEMPORAL_MIGRATION_REPORT.md`; complete comparison: `temporal_comparison.csv`; machine summary: `final_summary.json`.
- Subsequent-market per-trade entry: `results\temporal_migration\v4_4_k200_temporal_migration_20260807\full_replay\analysis\trade_review\index.html`.
- Reviewed plan and candidate freezes: `research_variants\short_momentum_net_drop_rebound_v4_4\plans\k200_temporal_migration_20260807`. R1–R4 and the descriptive full replay are immutable complete stages below the temporal result root.
- Evidence: 400 candidates per slice; positive counts R1/R2/R3/R4 are 296/26/383/25. Of 218 candidates observed in all four slices, two are positive in all four; both have only 11 or 13 full-test trades and 100% median Top-2 positive-return concentration. Training/full-test return-rank Spearman is -0.26169. `parameter_acceptance=none`.

## Current K200 market-data source — 2026-08-07

- Active source: `runtime_inputs\market_data\k200_clean_15s_session_filled.csv`.
- Current identity: 233,368 session-filled 15-second rows from `2026-05-23T00:00:00+09:00` through `2026-08-07T03:21:45+09:00`; SHA-256 `9760d367a109777c4789ce45d982a6c0708bacddad8f549450ed94f81ad5c405`.
- Latest retained Tick acquisition: `F:\Backtest test 6.11\02_DATA_AND_AUDITS\market_data\k200_historical_ticks_supplements\k200_postroll_supplement_20260728T161430_to_20260807T032200_20260807_022303`. It contains 1,065,932 IBKR `TRADES` ticks, 34,168 cleaned session-filled 15-second rows, its checkpoint/audits, and a local `README.md`.
- Prepared identity: `v4_4_confirmed_low_activity_gate_9760d367a109777c_76f2695bc1f4_9e27394dbe49` under `runtime_inputs\data_preparation`.
- Existing V4.4 result stages and cumulative rankings were not recomputed. They remain evidence for their recorded evaluation intervals; the extended source is available to later explicitly launched backtests.

## Current on-demand per-trade interval statistics — 2026-08-07

- Stable per-trade entry: `results\all_completed_union_analysis\trade_review\index.html`.
- Updated snapshot entry: `results\all_completed_union_analysis\snapshots\eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5\trade_review\index.html`.
- Presentation source: `runtime_inputs\templates\historical_v4_trade.html`; generator: `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_review_delivery.py`.
- `区间统计` switches the existing Plotly chart to horizontal selection. Statistics are calculated once after selection from the already loaded visible OHLC arrays; startup precomputation, interval caches, and additional requests are absent. Native selection state is cleared after completion, then a pale ordinary Plotly shape is drawn over the interval so the rest of the chart keeps normal opacity.
- The compact flat three-column panel is rendered in the right detail column above trade reasons. Activating either inspection closes the parameter drawer. Closing the panel removes the custom interval shape, disables interval mode, and restores zoom dragging.
- `持仓检测` is mutually exclusive with interval statistics. Plotly zoom dragging remains active: movement beyond five pixels is treated as a drag, while a short left press is mapped to the nearest visible candle through chart-level pointer-down and document-level pointer-up; right clicks are excluded. Then `entry_index <= selected_index < exit_index` determines the holding state. A raw candle uses its exact source bar; an aggregated candle uses its final source bar, matching its displayed close for state calculation. The displayed candle's aggregate high and center receive a blue Plotly marker with the same pixel size and zoom behavior as the green entry marker. The darker borderless blue holding highlight spans the effective baseline start through the selected-time active low. A labeled pale-red dense dashed horizontal guide shows `active low + rebound threshold`; a labeled vertical guide shows `activeLowIndex + S`, the no-new-low theoretical speed-exit position, with remaining bars/time even when S is already formed. The exact max-W candidate source interval remains listed in the panel.
- Candlestick `whiskerwidth` is zero, removing the horizontal caps at the high and low endpoints.
- The removed view-mode sentence, compact interval overlay, and removal of the `紫色` and `橙色` annotation suffixes change presentation only. Coordinate data, trades, metrics, ranking semantics, and parameter acceptance remain unchanged.

## Current K200 cumulative presentation — 2026-08-06

- Stable entry: `results\all_completed_union_analysis\index.html`; it redirects to `main\index.html` and no longer keeps a duplicate navigation shell or iframe alive.
- The current page title is `V4.4 K200回测结果排序`. The full-width main result table occupies nearly one viewport, keeps a sticky header, and renders 500 sorted rows per page.
- The trade-count fieldset is labeled `交易数`. Its presets are `不限`, `至少 10 笔`, and `至少 20 笔`; editable strict `大于 x 笔` and `小于 x 笔` bounds can be applied together. The former 100- and 150-trade presets are absent. Main-table column headings are centered.
- The visible gap-dependence return audit, `segment_end_exit_count`, `waited_entry_count`, and `maximum_entry_wait_bars` are removed. `round_trip_cost_bps` and `gap_spanning_trade_count` remain available as the last two visible columns; underlying result fields and immutable snapshots remain unchanged.
- The cumulative page no longer renders the research-contract section.

## Current unified stricter-entry migration delivery — 2026-08-06

- Stable entry: `results\cross_instrument_comparison\index.html`.
- Current run: `results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__combined_250_stricter_entry_v54_20260806`.
- Target per-trade entry: `results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__combined_250_stricter_entry_v54_20260806\trade_review\index.html`.
- Migration plan: `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_migration_k200_to_simain_stricter_entry_round2_20260806.json`.
- Source stage: `results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_16_migration_stricter_entry_k_expansion_all_window`.
- Closure: 250 compatible candidates, 24,194 SImain trades, 247 reused target trade chunks, and three generated chunks. K1.4 remains the fixed-coordinate local target peak; classification is `not_improved_target_peak_at_k1p4`, and parameter acceptance is `none`. This section supersedes the 247-candidate migration delivery below.

## Current AI-led K200 leap/grid cumulative delivery — 2026-08-06

- Stable main entry: `results\all_completed_union_analysis\index.html`.
- Stable shared per-trade entry: `results\all_completed_union_analysis\trade_review\index.html`.
- Current immutable snapshot: `results\all_completed_union_analysis\snapshots\eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5`.
- Current population: 37,058 unique coordinates, 11,749,606 trades, and 109 compatible stages under campaign `v4_4_positive_entry_signal_repair_20260805`.
- Latest stage: `results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_108_ai_generated_leap_all_window`; closure is 512 coordinates and 151,372 trades.
- Exploration session: `results\ai_exploration\k200_leap_grid_cycle_20260806`; it completed 89 automated rounds comprising 60 leap rounds and 29 adaptive-grid rounds, with exact anti-join deduplication and no parameter acceptance.
- Incremental publication reused 5,320 unchanged per-trade chunks and generated 31,738. The stable route targets this snapshot. K200 global closure was not established, so SI migration was not started. This section supersedes the earlier K200 cumulative-delivery sections below.

## Superseded K200 cumulative delivery — 2026-08-05

- Stable main entry: `results\all_completed_union_analysis\index.html`.
- Stable shared per-trade entry: `results\all_completed_union_analysis\trade_review\index.html`.
- Superseded immutable snapshot: `results\all_completed_union_analysis\snapshots\db85efb36f3de1c1f8255c6108fb365ad9f3d337f77a8d37a0e0ae41982e5699`.
- Superseded population: 5,044 coordinates, 742,706 trades, and fifteen compatible stages under campaign `v4_4_positive_entry_signal_repair_20260805`.
- Latest stage: `results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_14_large_multiblock_exploration_all_window`; closure is 294 coordinates, 28,577 trades, and 37 batches.
- Latest interpretation: `results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_14_large_multiblock_exploration_all_window\interpretation\round_14_report.json`; classification is `improved`, and parameter acceptance remains `none`.
- Incremental cumulative delivery reused 4,747 unchanged per-trade chunks and generated 297 missing chunks. The stable route now targets the new snapshot. This section supersedes the older cumulative-delivery sections below.

## Current incremental stricter-entry migration delivery — 2026-08-05

- Source-stage main: `results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_13_stricter_entry_k_expansion_all_window\analysis\index.html`.
- Source-stage per-trade analysis: `results\campaigns\v4_4_positive_entry_signal_repair_20260805\continuation_round_13_stricter_entry_k_expansion_all_window\analysis\trade_review\index.html`.
- Stable entry: `results\cross_instrument_comparison\index.html`.
- Current run: `results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__combined_247_stricter_entry_v52_20260805`.
- Target per-trade entry: `results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__combined_247_stricter_entry_v52_20260805\trade_review\index.html`.
- Report: `results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__combined_247_stricter_entry_v52_20260805\MIGRATION_REPORT.en.md` and Chinese mirror.
- Fixed incremental publisher: `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_cross_instrument_comparison.py build --run-id <run_id>`.
- Closure: 247 candidates and 24,024 SImain trades; 244 trade chunks reused and three generated. This section supersedes the 244-candidate current-delivery section below.
- The source-stage HTML contains only the three new K200 coordinates and 243 trades. It was generated through the fixed stage analyzer with cumulative refresh disabled, so the approximately 714,000-trade stable source snapshot was not rebuilt.

## Current cross-instrument delivery — 2026-08-05

- Migration plan: `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_migration_k200_to_simain_20260805.json`.
- Stable combined ranking: `results\cross_instrument_comparison\index.html`.
- Current run: `results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__original_180_plus_repaired_64_v50_20260805`.
- Target per-trade analysis: `results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__original_180_plus_repaired_64_v50_20260805\trade_review\index.html`.
- Migration report: `results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__original_180_plus_repaired_64_v50_20260805\MIGRATION_REPORT.en.md` and its Chinese mirror.
- The presentation union contains exactly 244 nonoverlapping candidates: 180 from the earlier result and 64 from the repaired-source result. It has 25,129 source trades and 23,799 target trades. This heading supersedes the older current-presentation statements below while retaining them as historical transition detail.

## Current K200 cumulative delivery — 2026-08-05

- Stable main entry: `results\all_completed_union_analysis\index.html`.
- Stable shared per-trade entry: `results\all_completed_union_analysis\trade_review\index.html`.
- Superseded immutable snapshot: `results\all_completed_union_analysis\snapshots\0126cd77b436aef1434e7072bac0d6dfa15b3d2ad4dc2cf1b2fafe936ee1e626`.
- Superseded population: 4,747 coordinates, 713,886 trades, and thirteen compatible stages under campaign `v4_4_positive_entry_signal_repair_20260805`.
- That historical delivery includes the thirteen fixed-parameter E-window sensitivity coordinates plus thirty coordinates from the ten-round one-axis exploration.

## Instrument-neutral contract authorities — 2026-08-05

- Strategy core: `research_variants\short_momentum_net_drop_rebound_v4_4\contracts\STRATEGY_CONTRACT_V4_4.json`.
- Contract loader: `research_variants\short_momentum_net_drop_rebound_v4_4\code\instrument_contracts.py`.
- Ready K200 profile: `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\k200m.json`.
- Incomplete templates: `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\simain.template.json` and `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\nq.template.json`.
- Campaign starting point: `research_variants\short_momentum_net_drop_rebound_v4_4\campaign_contracts\CAMPAIGN_MANIFEST.template.json`.
- Workflow authorities: `project_management\00_core\STRATEGY_CONTRACT_V4_4.en.md`, `project_management\10_instruments\INSTRUMENT_PROFILE_CONTRACT.en.md`, and `project_management\20_campaigns\CAMPAIGN_WORKFLOW.en.md`.
- Historical pre-repair K200 authority is snapshot `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7`, with 4,704 coordinates, 706,470 trades, and nineteen stages. The current repaired result authority is the positive-entry snapshot listed above.

## Cross-instrument comparison authority — 2026-08-05

- Source manifest V48: `6952a4d30c1ad5fb9276de1c2d3248a1a70a0157a1260118b4086be3e04da1e0`, 40,814 bytes.
- The current exact-transfer presentation is independently bound to the repaired positive-entry K200 snapshot. Historical pre-repair transfer runs remain immutable and separate.
- Stable entry: `results\cross_instrument_comparison\index.html`.
- Current repaired-source validation run: `results\cross_instrument_comparison\runs\k200_repaired_v48_20260526_20260708__simain_20260129_20260223__promising_exact_transfer_v49_20260805`.
- Shared SImain trade-review entry: `results\cross_instrument_comparison\runs\k200_repaired_v48_20260526_20260708__simain_20260129_20260223__promising_exact_transfer_v49_20260805\trade_review\index.html`; all 64 candidates have their own trade catalog, chart, entry reasoning, exit reasoning, and parameter explanation.
- The historical 266-candidate aggregate remains preserved at `results\cross_instrument_comparison\runs\k200_20260526_20260708__simain_20260129_20260223__all_exact_transfers_v46_20260805` and is not merged with repaired-source metrics.
- The comparison page exposes file-backed source instrument, source interval, target instrument, and target interval selectors. Global search and composable field filters cover every displayed column; candidate-source and `combo_id` fields remain hidden from the table.
- The return-view control switches ranking and displayed K200/SImain total return, median trade, maximum drawdown, and win rate between fee/slippage-adjusted and gross modes; adjusted mode is default.
- Its first column reuses the K200 `rank-link` treatment as `查看 #N`; the header follows the selected mode as `成本后排名` or `成本前排名`, and every button opens the exact SImain trade review in a new tab.
- Columns are grouped as parameters, SImain metrics, K200 metrics, and transfer diagnostics. MFE and MAE retain both bps and raw-point columns; K200 total-return hover help states that its 2026-05-26 through 2026-07-08 interval is longer than SImain's 2026-01-29 through 2026-02-23 interval.
- The stable cumulative entry is a presentation-only navigation shell. It embeds the byte-preserved current cumulative snapshot and links to the standalone cross-instrument page; snapshot HTML, `analysis_data.js`, and `union_trades.csv` remain unchanged.
- The repaired-source batch was frozen before SImain evaluation. It starts from 4,747 K200 coordinates, excludes 266 previous transfers and three champions, retains 225 eligible source points, and freezes 64 within-family Pareto candidates across eleven W/M/S families. Duplicate, previous-transfer overlap, champion overlap, and target-driven edits are all zero.
- SImain uses explicit SIH6 15-second OHLC with one prior warm-up day. Only entries created from 2026-01-29 through 2026-02-23 are counted. SIH6 remains the main contract throughout the test interval, so the roll count is zero.
- The current comparison contains 64 frozen candidates and 6,755 SImain trades. SImain is cost-positive for 58 candidates (90.625%); 29 are in target-positive stable regions and 26 are isolated positives.
- K200-versus-SImain cost-return rank Spearman is -0.45728. The target-return leader is E320/BH240/TRW21/K1.05/W6/M4.5/S340 at 19.6892%, while the K200 leader E480/BH171/TRW12/K1.26/W6/M4.5/S388 reaches only 1.1311% on SImain. Source rank therefore cannot be reused as target rank.
- The comparison creates no aggregate score and accepts no parameter. SImain results cannot add, remove, or edit frozen candidates. A later SImain full-grid run, if authorized, is a separately labeled post-hoc diagnostic.
- The current K200 cumulative snapshot HTML, `analysis_data.js`, and `union_trades.csv` remain byte-identical after cross-instrument publication. The stable cumulative navigation shell links to the standalone current comparison entry.

## Active V4.4 authorities

| Role | Path | Status |
| --- | --- | --- |
| Project root | `D:\Code\backtest-release\Backtest V4.4` | Current temporary V4.4 workspace |
| Physical result root | `F:\Backtest\Backtest V4.4\results` | Current and future V4.4 result storage; full 43,817-file migration verified byte-for-byte and SHA-256 pairwise |
| Logical result root | `D:\Code\backtest-release\Backtest V4.4\results` | Windows directory junction to the physical F-drive result root; preserves historical paths and entry points |
| Result recovery copy | `F:\Backtest\migration_recovery\Backtest V4.4 results before junction 20260804` | Recoverable byte-identical migration source copy |
| Old-result quarantine | `F:\Backtest\D_cleanup_quarantine_20260804` | Recoverable destination for the four approved non-V4.4 result directories removed from D |
| Engine | `research_variants\short_momentum_net_drop_rebound_v4_4\code\v4_4_engine.py` | Current |
| Runner | `research_variants\short_momentum_net_drop_rebound_v4_4\code\run_v4_4_resumable_campaign.py` | Current |
| Analyzer | `research_variants\short_momentum_net_drop_rebound_v4_4\code\analyze_v4_4_scenario_3_stage.py` | Current |
| Trade delivery | `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_review_delivery.py` | Current |
| Cumulative builder | `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_combined_union_analysis.py` | Current |
| Identity manifest | `research_variants\short_momentum_net_drop_rebound_v4_4\SOURCE_MANIFEST.json` | Current V48 positive-entry instrument-neutral source identity; formal source-review ZIPs must pass their package-root test runner from an independent extraction; V4.4 results are physically on F through the compatible D-drive junction |
| Runtime input manifest | `runtime_inputs\RUNTIME_INPUTS.json` | Current |
| Data-preparation manifest | `runtime_inputs\data_preparation\data_preparation_manifest.json` | Schema 5 causal availability |
| V4.4 handoff ZIP | `D:\Code\backtest-release\Backtest_V4.4_with_trade_records_20260803_final.zip` | Audited historical `SOURCE_FINAL_V4` package: nine reports, five completed stages, and 120 raw/derived transaction CSVs; not a current-V5 source package |
| ZIP audit | `D:\Code\backtest-release\Backtest_V4.4_with_trade_records_20260803_final.zip.audit.json` | Package stream and extraction verification pass |
| Most recent source-complete review ZIP | `D:\Code\backtest-release\Backtest_V4.4_current_UI_review_20260805_source_complete.zip` | Preserved V37 package bound to snapshot `20464535...`; it predates the V38 instrument-contract source layer and is historical package evidence |
| Current review ZIP audit | `D:\Code\backtest-release\Backtest_V4.4_current_UI_review_20260805_source_complete.zip.audit.json` | Timepoint, source-layout, archive-stream, extraction, and independent-test verification pass |
| Temporary plan | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_temporary_single_combo_close_fill_20260802.json` | Executed isolated plan |
| Round-1 plan | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_cost_adjusted_multiround_20260803_round_01_broad_all_window.json` | Frozen 372-coordinate plan; validate-only, anti-join, resource, and materialization gates completed before compute |
| Round-2 plan | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_cost_adjusted_multiround_20260803_round_02_broad_local_all_window.json` | Frozen 247-coordinate broad-plus-local plan; raw compute immutably closed |
| Round-2 stage | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_02_broad_local_all_window` | Immutable and delivered: 247 coordinates, 20,629 trades, fingerprint `7ad95dbd7ba9ebc1faffd8cbc1723211273453af0471ab950e5b3d798ee6c4e8` |
| Round-2 stage main | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_02_broad_local_all_window\analysis\index.html` | Complete fixed-template stage entry |
| Round-3 plan | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_cost_adjusted_multiround_20260803_round_03_terminal_local_all_window.json` | Frozen 212-coordinate terminal local plan; raw compute immutably closed |
| Round-3 stage | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_03_terminal_local_all_window` | Immutable and delivered: 212 coordinates, 16,847 trades, fingerprint `6d40e4a35562bbbb16347a63b48442f132025bbc4695f0ac87bc4312eae0955e` |
| Round-3 stage main | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_03_terminal_local_all_window\analysis\index.html` | Complete fixed-template terminal stage entry |
| Multi-round design | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\v4_4_cost_adjusted_multiround_design_20260803.md` | Non-executable broad/refinement contract |
| Round-1 interpretation | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\round_01_interpretation_and_round_02_design_20260803.md` | Final closed-round interpretation and Round-2 rationale |
| Round-2 interpretation | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\round_02_interpretation_and_round_03_terminal_design_20260803.md` | Final closed-round interpretation and terminal Round-3 rationale |
| Terminal interpretation | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\round_03_terminal_interpretation_and_campaign_closure_20260803.md` | Canonical final result and campaign closure; Round 4 prohibited |
| Final total audit | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\final_read_only_total_audit.json` | Independent read-only aggregate audit: all raw/delivery identities, counts, hashes, uniqueness, locks, and exited processes passed |
| Continuation design | `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\v4_4_continuation_subseries_design_20260803.md` | Non-executable evidence-led broad-then-refine design and mandatory per-round delivery rule |
| Continuation Round-1 plan | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_cost_adjusted_multiround_20260803_continuation_round_01_broad_span_all_window.json` | Executed frozen 528-coordinate broad-span plan; raw closure remains immutable |
| Continuation Round-1 stage | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\continuation_round_01_broad_span_all_window` | Immutable historical evidence: 528 coordinates and 54,842 trades |
| Superseded V4-bound Continuation Round-2 plan | `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_cost_adjusted_multiround_20260803_continuation_round_02_dual_objective_local_all_window.json` | Preserved pre-compute evidence; do not edit, move, delete, validate, or compute; replacement V5 plan pending |
| Superseded V4-bound Round-2 root | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\continuation_round_02_dual_objective_local_all_window` | Four deterministic validate-only metadata files only; no progress, batches, trades, completion, or analysis |
| Round-1 stage | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_01_broad_all_window` | Immutable: 372 coordinates, 316,398 trades |
| Round-1 stage main | `results\campaigns\v4_4_cost_adjusted_multiround_20260803\round_01_broad_all_window\analysis\index.html` | Complete fixed-template stage entry |
| Active cumulative snapshot | `results\all_completed_union_analysis\snapshots\eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5` | Current 109-stage AI-led exploration snapshot: 37,058 unique coordinates and 11,749,606 trades; incremental trade-review publication reused 5,320 chunks and generated 31,738 |
| Temporary stage | `results\campaigns\v4_4_temporary_close_fill_validation_20260802\single_combo_all_window` | Complete: 1 coordinate, 3,882 trades |
| Stage main | `results\campaigns\v4_4_temporary_close_fill_validation_20260802\single_combo_all_window\analysis\index.html` | Current temporary review entry |
| Stage trade | `results\campaigns\v4_4_temporary_close_fill_validation_20260802\single_combo_all_window\analysis\trade_review\index.html` | Current temporary trade entry |
| Stable V4.4 main | `results\all_completed_union_analysis\index.html` | Current shared cumulative main entry; the final exploration-series publication refreshes it with all compatible completed results |
| Stable V4.4 trade | `results\all_completed_union_analysis\trade_review\index.html` | Current shared cumulative per-trade entry; the final exploration-series publication refreshes this one large page with all compatible completed results |

## Future delivery rule

Preserve all existing stage main and per-trade pages as historical evidence. Continuation Rounds 8–15 closed under the same campaign root. During a multi-round exploration, intermediate rounds close immutable raw evidence and compact summaries without rebuilding the shared HTML. Publish the two stable cumulative entries once after the exploration series ends, including every compatible old and new coordinate. An earlier publication is allowed only when the user explicitly requests it. No next round is authorized.

## Provenance boundary

## Cost-reference transition

Future derived cost-adjusted ranking is transitioning from the preserved fixed 3.56-bps legacy view to `runtime_inputs\cost_models\k200m_current_notional_cost_reference_20260803.json`. The reference freezes the KRX multiplier, the latest real 15-second K200M price, the dated KRW/USD rate, 2 bps round-trip slippage, USD 6 round-trip commission, and each derived KRW/bps amount. Completed raw stages remain immutable; no raw result is rerun or overwritten during this transition.

The latest user decision supersedes the fixed-future-cost rule. Existing K200 derived results retain their historical cost reference. Every new campaign binds a frozen price-times-point-value cost model, and a reviewed plan must bind that profile before compute.

`SOURCE_FINAL_V7` is `54a0a272b2b2215e60ff0649796bef3d3babfd69893b4cd93c0da4d69dbeb4cb`; its closure memo is `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\source_final_v7_dynamic_k200m_cost_20260803.md` with SHA-256 `4d83aba6ed71fbee24186652fbc477726ba9b5e84f551165961523e0323baae1`.

Copied V4.3 plans and source manifests are under `runtime_inputs\provenance`. The former active V4.3 management records are under `project_management\99_archive\v4_3_terminal_management_snapshot_20260802`. They are records only and are never active V4.4 inputs.

The active cumulative lineage is scoped to campaign `v4_4_positive_entry_signal_repair_20260805`. The retired pre-repair campaign, the older temporary validation campaign, and failed partial snapshot `1ba05465b49a1de45c407bfd9b4456eeb83c92b9dadea8d1c5d060a40ae22d98` remain separate historical or recovery evidence and are not stable-route targets.

Current continuation stages keep campaign ID `v4_4_positive_entry_signal_repair_20260805`, use the `continuation_round_*` stage namespace, and remain direct children of this active root. The cumulative builder may scan `results\campaigns` only through its strict result-identity filter; incompatible historical siblings remain excluded and are reported.

## Closure hashes

- Round-15 cumulative pointer: `6720ce8ba979dfd641e2db90cc78c6fcf958237c0c8fe34433cb96f7c58bb1bb`; snapshot `45b9a08396493a53ece45bd62af91070fb6b443539cd2e12ae5aeac5c756faad`; analysis `ab7141a0cf210061b1dc4ef94981c39b56f29507f21e2f51bc6e8bff65e7fe80`; completion `d9bc1f8024eda348a8fc77c599c3957ced063af1b0af98b55f3f14b508ef2374`; trade review `f1fb5b5ce407a6bbd8513dd09d2375c806374266ffc54915aa89f9307ab39b07`. It contains 4,499 coordinates, 683,835 trades, and 18 compatible stages; `parameter_acceptance=none`.
- Current source identities after the merged cross-instrument closure: cumulative builder `4bdb447200e740f02a65ed2bb52b2c5d3b4130a0c541edfc50702efb580e93a3`; stage analyzer `fc0142d046c90d432e017808647e52400647e0f7ad8606da6b70fac7d1a88aba`; review generator `be0d0e08dc5c6b131d39dc0c1c0bea3ec3462c6465cffffd5203836c1162fdd1`. The Round-15 publisher-memory and indexed-review changes are retained; the engine, raw evidence, rank formulas, and HTML semantics are unchanged.
- Historical `SOURCE_FINAL_V17` manifest: `d707f560080a4c3f589ec24e1161e2126211fbbd09366f2279f4845554c73a26`; 28,810 bytes. The current V20 manifest closes Rounds 11–13 and the sixteen-stage cumulative snapshot. Engine, runner, templates, raw-result semantics, and the shared delivery architecture remain unchanged.
- Current cumulative pointer: `cada016193dd551dd4a0930b739dc04bd5907732ad84cfa71aef457ef7ee2983`; snapshot `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7`; analysis `0cd061ca54d17acc1673f42c3f4d273b66dbdaa554473f6f2699a6695c3435b5`; completion `22aa69610483145e946d9458bbf09271a1ad006c988f1b3f407f1049daa26db5`; trade review `1ead44e9bd49f0cfc5d95f073f24e68925879481dd8fad122c4653e928c899a6`.
- Continuation Round-16 plan: `1802a9d15118069be1aba19f0918e7a8bed4f6cf94010dd3942936cdc12b113f`; completion `0e569a9458af3f2146c6d5874c123e14ad70993d104ca7f4d30dc7ed80946060`; stage analysis `4911577da5baa01836204b33bb42b282922a7f454cd3039ac791240e0a34c46c`; delivery audit records 4,728 unique declared artifacts with zero mismatches.
- Current `SOURCE_FINAL_V12` manifest: `40bb10ff06f9ca662e1052edd8655303cab2d475da3512d916768c84c8ee4763`; 22,556 bytes. V12 keeps the cumulative-main repair and changes all per-trade market-range rectangles to semantic translucent fill with zero-width transparent borders. Snapshot `2da2a0dff4c1890627f78c0556a2d8504ff0f384f77db147da54572367635a52` receives only the corrected shared per-trade HTML and reconciled presentation metadata; raw trades, returns, rankings, and cumulative data remain unchanged. Earlier V2 through V11 source authorities remain historical evidence. The active runtime-input manifest is `df67b04e0f3e6c516f8456d222b49841b417dc793779ac71403db0e86da530a8` / 2,683 bytes and recursively binds the current borderless-range trade template.
- Trade-record package tool remains `tools\\package_v4_4_with_trade_records.ps1`, `1e4a12a7c2e138ca253b9d2be9f78973af54f7f7220e0c961966e76bba2cd191`, 17,698 bytes. The existing final ZIP and adjacent evidence remain historical V4 release authority and are not current-V5 package bytes.
- Current historical trade template is `0b30ac0e0ed189d83a6cf962cf1bf826a82d312bfd07ca294dc9952ece3d6fc9`; stage analyzer `a6c33456e11a0cb82cbd7e248ea4bf15e3c74292891452de835564e3564355d3`; review generator `c9055734ec66b4f1aec53af90194f9003e44044fdab2e0cfe7bc55a34a24b4f2`; stage QA `6f572c4b4cff22002e2ea8cdf605e461085b85597152f5c00c18895327db3214`; review tests `7ca723d4e6d5f8d60694912984e5458988648ac69f21d710507e9cd94c14b624`.
- Frozen Round-1 plan: `1424dc17862a2bfe0b8f0439fef061e64efc487c5057b7cff64498ed40a78046`.
- Round-1 completion manifest: `c9532f77b626f647dcfe7b1fdc09ee76b2b895e851da0b98f6206b26ff1e6539`.
- Round-1 stage analysis manifest: `3c55f1222db2586f8e5fb4dee5800e4534ee695c9ce7e101fd9a7e5f7c56d03f`.
- Current corrected-source cumulative pointer: `725caf1515554a67ad6c2cccec43da14261d9441b2b1e38b688cd4950749e79e`; snapshot `a55ee98105958c699a29a1e32a9ccd0f3afc60cd82b29b5d88d74068fa59219a`; analysis `2f8021494eef8da99a8a62866eb9976da1cc45b1cac3d8f81f47d3fa88eb9ea3`; completion `6db44fb04515ea5fdd918c5d2023f44229667de30786c1c7b071ab900cae9363`; trade review `b5859cb46825542eafdcd75f98cf71caf9699948ec3819443755b00d047bfe5a`. Preserved old snapshot `ce1e20f7...` remains historical evidence.
- Frozen Round-2 plan: `f90dbf5563ae9128304d5b48b902db440d9b73a4af3c0a7654079ef73628f7fd`.
- Round-2 completion manifest: `b97a4811ebf3520fb6086ec26d5dfa149b2154d989ca6d94a1149f8d7a28350c`.
- Round-2 stage analysis manifest: `be731db4bc274b85271ea5bb26421aba0e8f8b8f538825e3925e5d459aa47164`.
- Frozen Round-3 plan: `46c95b24feab49b6f260a0e8f1e1125fd74c34c6a0e268b89e0e1fb83a6d9b8c`.
- Round-3 completion manifest: `edec2c43ecf4c4035a690f763b6a4d68d8be8f97a9da40ec9b3f3aac20ff25ea`.
- Round-3 stage analysis manifest: `d163c36e471c732181f93d85d3e9752f8938b3cb14d92c025648e53815044d2f`.
- Terminal interpretation memo: `d740561e13e7fe5da6b0dc96e826c4b155f99a56b34b899c9b2b77653d373837`.
- Final read-only total audit: `58009d4eb357e3022de423d63310c9821fd167e518b43c739dc8efea6a694c0e`; reusable script `86aa89a37b4780ca1b7719ced215d9566a77e5cc438356b8d05f0b7563b2f8eb`.
- Continuation design memo: `94966f8096bb5317e09d2eb178a10cc9b1f31d3871c2bb8a3efa61218f2b9412`.
- Frozen Continuation Round-1 plan: `481fd28365757f739cb0e260d3cc36a4390db9cde9b1f1ccf3063aefdb8c9bf5`.
- Continuation Round-1 completion: `0990507be75526618663b4e08a3d628fd7af856dd692c9d9c3313de2cd0fdf6d`; old-source stage analysis `2d87ad79920740980141aaef6f7e5c4b650ebcf99dd5e15c485c1c05986a0f70`; old-source delivery evidence `f6409f53da7f33d6de2b6c3074b8cf11ef41e60e314c2f00de81950d84044909`.
- Corrected-source Continuation Round-1 stage analysis: `e643b934f32e1f84db963c33ea8e4c24276da462c5e6a2199b5aa4b369f99b2f`.
- Superseded pre-compute V4-bound Continuation Round-2 plan: `05d0d2edb604ce80cea391e13f62c0caa56cb4fc23caec98bb25cd590cb21dcb`; partial validate-only fingerprint `976fac8d1e6ce5b280127ad6d7000116e2280a81242e4d2f754cccac7b139e35`. It remains immutable four-file metadata evidence only. The delivered V6 plan is `plans\\v4_4_cost_adjusted_multiround_20260803_continuation_round_02_dual_objective_local_v6_all_window.json`, SHA-256 `d982267710abab0355a37271c18a25df40decc3d9f846f82030a3ecbeab82a07`, with completed root `results\\campaigns\\v4_4_cost_adjusted_multiround_20260803\\continuation_round_02_dual_objective_local_v6_all_window`. Its immutable completion is `2c0364fda3fc17cd09419d0a6003e6a3e6d7f1da035b8228c201fd21a6570d6e`; the current cumulative snapshot is `2da2a0dff4c1890627f78c0556a2d8504ff0f384f77db147da54572367635a52`. Any later round awaits this delivered evidence and a new reviewed, hash-bound plan.
- Continuation readiness inventory: `ff7e0b7790e52b1bf86ac185d79c148552c0b14240453dd0ed512ca476d58a27`.
- Round-2 interpretation memo: `42d6a69150ca939d0cce13fa424a80a2fc79d785097c4e916efb14545a3fe262`.
- Round-1 interpretation memo: `7e500cd13ccb9c7dc928b5a8cb1bc1f3f9335b2765b577e58372ccbec1f26411`.
- Multi-round design memo: `1286acf656babb9ad885ef8e75ac6cffb8afbde7a9f6a51dc5601d8cfb17bf24`.
- Temporary plan: `b44a2d75722dd582e588b146760ce78841cec85a2ea5f2bca65a5db6826227c6`.
- Completion manifest: `eddaf0d6335b2e718e4e78d1d1e5fc06aa16a3e4bb559ef062a7c357de36770e`.
- Historical temporary cumulative pointer: `995a4e625051318eae35fbcfe1bab3f1157ef80938be3e24ddbfd612ae60226a`, snapshot `cae521e3066e80d1e3d1f5b7bc9c68ba20737cde67493723bfb482d5a0e80181`.

## P0-8 repair authority

- Affected-coordinate inventory: `D:\Code\backtest-release\Backtest V4.4\.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\p0_8_zero_signal_affected_combos.csv`; 4,383 coordinates, SHA-256 `3469a630fc47e2614594536d3301746d8e68811e1171fd19f027933accf88c44`.
- Audit summary: `D:\Code\backtest-release\Backtest V4.4\.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\p0_8_zero_signal_affected_summary.json`.
- Retired active campaign: `F:\Backtest\Backtest V4.4\results\staging_recoverable\p0_8_zero_signal_retired_20260805\v4_4_cost_adjusted_multiround_20260803`.
- Historical review snapshot `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7` has reference authority only until corrected replacement publication.
