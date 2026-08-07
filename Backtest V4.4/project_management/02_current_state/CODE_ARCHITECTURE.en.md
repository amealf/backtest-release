# Code Architecture

## Release identity boundary

- `RELEASE.json` owns the active presentation release (`V4.41`) and declares the unchanged V4.4 strategy/ranking major version. Engine manifests, parameter identities, and cumulative admission continue to use the existing V4.4 contracts and ranking lineage.
- Page builders render V4.41 in active headings. `v4_4_engine.py::VERSION_LABEL`, `SOURCE_MANIFEST.json::version_label`, and `build_v4_4_combined_union_analysis.py::RANKING_MAJOR_VERSION` remain V4.4 so completed results stay compatible.

## Date-based evaluation packages and generic comparison

- `tools\build_v4_4_evaluation_framework.py` publishes the result framework. It derives package identity from `instrument_id` and exact evaluation start/end; writes manifests, experiment notes, neutral summaries, candidate-set browser projections, immutable trade-record links, and per-trade compatibility routes; then builds a comparison plan and package-backed page.
- `tools\register_v4_4_evaluation_package.py` registers one newly completed instrument/interval from the declarative `runtime_inputs\templates\EVALUATION_PACKAGE_SPEC.template.json`. The spec maps source summary columns into neutral parameter and metric fields; registration creates the date package, hard-links immutable trade records, and updates the shared catalog without changing an existing package.
- `results\evaluation_packages\catalog.json` is the machine-readable registry. Each instrument owns one directory per exact interval. Result packages do not encode an experiment role, so completed evidence can participate in several later plans.
- `results\evaluation_comparison` stores comparison catalogs and plans. Its browser loader reads each selected package projection, builds a `combo_id` lookup, restores role metrics to the retained page field contract in the original field order, and then runs the unchanged comparison UI.
- Large retained trade CSVs use same-volume hard links inside compatibility packages. Package per-trade routes preserve query and hash values while forwarding to byte-preserved historical pages. Future native packages may own generated chunks locally and reuse unchanged chunks incrementally.
- `tools\qa_v4_4_evaluation_framework.mjs` compares the retained and package-backed pages in Chromium. It checks title, complete visible body text, first-row data, visible row count, and console/page errors, and captures both pages.
- `run_v4_4_resumable_campaign.py` already resolves market data and preparation through an instrument profile and reads `train_start`/`train_end` from the plan. The storage layer removes K200 and train/test roles from directory identity without changing engine or runner semantics.

## Current-optimal K200 one-month replay

- `tools\run_v4_4_k200_current_optimal_forward_initial.py` freezes a training-only multi-metric candidate population, anti-joins non-control rows against earlier temporal evaluations, writes a schema-4 exact-transfer plan, invokes the existing four-worker resumable campaign runner, and produces compact comparison and Markdown outputs without HTML.
- Candidate eligibility requires positive cost-adjusted training return. Eight training-view controls are selected before the anti-joined additions; the freeze records whether each exact coordinate had any earlier test evaluation.
- The script reuses `run_v4_4_k200_temporal_migration.py` for plan-safe execution and cost/dependency analysis, so engine, fill, cost, preparation, and gap semantics remain unchanged.

## K200 train/test/SI triple-migration workflow

- `tools\run_v4_4_train_test_si_triple_migration.py` freezes 100 source-only temporal candidates after an exact anti-join against the retained 250 SI candidates, evaluates the new freeze on SI with four workers, replays the combined 350 on the K200 test interval, and writes the three-return comparison and final interpretation.
- `build_v4_4_cross_instrument_comparison.py` reads the optional `source_test` contract, renders role-relative K200（训）, K200（测）, and SI return columns, routes each total-return button to role-specific per-trade evidence, and builds the dedicated K200（测） review from retained trade records without rerunning the strategy. K200（训） reuses the cumulative K200 review; SI retains its incremental review.
- Final SI per-trade publication passes the retained 250-candidate run as `incremental_parent_run`; 250 chunks are reused and only the 100 new candidate chunks are generated. Intermediate migration stages do not publish HTML.

## K200 temporal-migration workflow

- `tools\run_v4_4_k200_temporal_migration.py` freezes source-only multi-metric R1 candidates, runs four-worker resumable stages, writes compact cost/dependency summaries, and creates later candidate freezes from closed earlier slices plus source neighbors and structural controls. It preserves a final unseen slice and labels the full-interval replay as post-hoc descriptive evidence.
- Plans and candidate CSV freezes live under `research_variants\short_momentum_net_drop_rebound_v4_4\plans\k200_temporal_migration_20260807`; raw and derived stages live below `results\temporal_migration\v4_4_k200_temporal_migration_20260807`. Completed batches survive a memory-floor stop and resume without changing a freeze.
- Finalization writes one temporal comparison HTML/CSV, one Markdown report, and one stage-native per-trade analysis. Existing analysis is retained on later report refreshes, so no completed compute or trade chunks are regenerated.
- `build_v4_4_review_delivery.py::_stage_filter_bundle` accepts both the retained policy-neutral preparation identity and the current confirmed-low-activity-gate identity while preserving bound source and artifact hashes. `analyze_v4_4_scenario_3_stage.py::_atomic_json` serializes path-valued manifest fields as strings.

## On-demand per-trade range-statistics presentation

- `runtime_inputs\templates\historical_v4_trade.html` owns the paired-review control, version-free tab title, blue `Z` favicon, reason-panel metric boxes, simplified trade picker, and bottom legend. `build_v4_4_review_delivery.py` injects an optional peer route and research-contract identity while keeping the shared template instrument-neutral.
- `results\all_completed_union_analysis\trade_review_peer.json` binds the current cumulative K200 review to its later-period test review. `build_v4_4_combined_union_analysis.py` reads that small publication contract for current-snapshot refreshes and future cumulative shells. `build_v4_4_cross_instrument_comparison.py` supplies the reverse link when it publishes the K200 test review.
- `runtime_inputs\templates\historical_v4_trade.html` owns `区间统计`, Plotly horizontal selection, the right-column compact statistics panel, selection-complete calculation over existing OHLC, clearing of native selection state, a persistent pale ordinary shape over the interval, and close-time restoration of zoom mode.
- The same template owns mutually exclusive `持仓检测`. It preserves Plotly zoom dragging, stores pointer-down on the chart, receives pointer-up on `document`, treats movement beyond five pixels as a drag, rejects non-left buttons, and maps a short press through the current axes to the nearest visible candle. It then resolves the exact raw source bar or an aggregate's final source bar, finds the enclosing trade, reconstructs the engine's max-completed-W next-bar-effective rebound state and S-window running-low state, adds a blue point at the selected close, and adds a darker borderless blue fill over the baseline-start-to-active-low span. Activating either mode collapses the parameter drawer.
- The candlestick trace uses `whiskerwidth=0`, removing high/low endpoint caps without changing OHLC values.
- `build_v4_4_review_delivery.py::refresh_trade_review_shell` regenerates only `trade_review\index.html`, updates its resource audit and trade-review manifest, and leaves process payloads and per-coordinate trade chunks untouched.
- `build_v4_4_combined_union_analysis.py::_refresh_completed_snapshot_trade_shell` updates the outer analysis and completion artifact references for a completed snapshot without invoking union-trade or chunk generation.

## K200 incremental Tick acquisition and activation

- `tools\download_k200_incremental_ticks.py` resumes the retained IBKR K200 `TRADES` Tick checkpoint, preserves each acquisition directory, applies the current one-second immediate-recovery cleaner, builds session-filled 15-second derivatives, and creates an ordered extension candidate. It keeps the 10.25-second request pace; when IBKR omits a completion callback, one 500-tick page advances the unchanged cursor before 1,000-tick pages resume.
- `tools\write_k200_download_readme.py` writes and refreshes the `README.md` inside every retained K200 acquisition directory. It derives update history from the lineage manifests and records the audited main-contract rule, exact 016M/016U boundary, unadjusted merge policy, request range, status, and principal files.
- `tools\activate_k200_incremental_15s.py` appends the original active K200 source and independently cleaned supplement segments in strict timestamp order, replaces the repository-local active 15-second file, rebuilds prepared data, and refreshes the profile, attestation, runtime-input, and source-manifest identities. Historical result stages are not recomputed by activation.

## K200 cumulative presentation route

- `build_v4_4_cross_instrument_comparison.py::publish_current_main_standalone_view` publishes the stable root as a query/hash-preserving redirect to `main\index.html`; the duplicate navigation shell and iframe are no longer generated.
- `analyze_v4_4_scenario_3_stage.py::_legacy_v4_main_html` owns the current K200 title, colors, visible column order and labels, fixed-width rank links, 500-row client-side pagination, and the full-width sticky-header table viewport. It omits the research-contract section. `publish_stable_main_assets` refreshes only the lean main presentation from the current immutable snapshot payload.

## Dual-purpose exploration utilities

- `tools\build_v4_4_k200_dual_purpose_round_15_plan.py` freezes deterministic broad coverage and exact one-parameter refinement blocks, performs the completed-coordinate anti-join, and writes the reviewed execution plan plus audit.
- `tools\analyze_v4_4_k200_dual_purpose_round_15.py` attaches plan block identities to completed stage results, compares broad and refinement branches separately, derives primary-view and trade-level diagnostics, and writes the anti-joined next-round handoff. These utilities do not change engine, fill, cost, or result semantics.

## Incremental cumulative per-trade publication

- `build_v4_4_combined_union_analysis.py` recursively resolves nested generated plans, loads stage summaries without retaining every trade frame, and processes retained trades one stage at a time. The cumulative trade CSV is streamed to a temporary file and atomically renamed after all stages close.
- Deterministically named unchanged coordinate chunks are reused through same-volume hard links. Missing chunks use vectorized native-record construction and four worker processes; the ranking, catalog, process payload, summaries, manifests, and stable routes are refreshed once from the complete compatible stage set.
- Closed stage manifests provide the publication audit boundary, avoiding repeated full trade-file hashing and per-trade execution audits during cumulative publication. Strategy, result, cost, scenario, and plan identities remain enforced.
- The current delivery reused 5,320 chunks, generated 31,738, and published 37,058 coordinates/11,749,606 trades with publisher memory below 2 GB.

## Instrument-neutral contracts

- `research_variants\short_momentum_net_drop_rebound_v4_4\contracts\STRATEGY_CONTRACT_V4_4.json` contains only shared entry, exit, timing, and state-machine semantics.
- `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\*.json` bind instrument data, preparation evidence, cost, gap, low-activity, optional scenarios, and ranking lineage. K200 is ready; SImain and NQ remain templates until the user supplies cost and gap decisions.
- `research_variants\short_momentum_net_drop_rebound_v4_4\campaign_contracts\CAMPAIGN_MANIFEST.template.json` is the starting point for `transfer_exact`, `target_local_refinement`, or `fresh_search`.
- `code\instrument_contracts.py` validates those files and normalizes cost models. It calculates notional as reference price times point value and retains legacy K200 aliases.
- Schema-v5 runner plans bind a campaign manifest, instrument profile, mode, ranking lineage, and scenario policy. Older plans omit these fields and retain their historical behavior.

## Cross-instrument comparison

- Incremental target trade-review publication is implemented in `build_v4_4_review_delivery.py::build_stage_trade_review`. A run config may name `incremental_parent_run`; `build_v4_4_cross_instrument_comparison.py` resolves its completed parent review, hard-links deterministic unchanged combo chunks into the new delivery, generates only missing combo chunks, then refreshes the small shell, catalog, process payload, summaries, and manifests. The fixed CLI is `build_v4_4_cross_instrument_comparison.py build --run-id <run_id>`. Manifests record parent, reused count, generated count, and reuse mechanism.
- `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_repaired_source_transfer.py` can create a presentation union from completed exact-transfer result sets without rerunning either instrument. It removes batch provenance from the user table, binds a reviewed migration plan, writes instrument-neutral labels, publishes one combined target trade review and report, and redirects the repaired standalone result to the combined presentation.
- The comparison builder reads relative source/target names and sample ranges from the run's `migration_plan`. Its table places source total return immediately before target total return and uses a three-state sort cycle. The cumulative navigation shell and comparison header provide reciprocal new-tab navigation.
- `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_cross_instrument_comparison.py` owns candidate freezing, isolated SImain evaluation, transfer diagnostics, CSV/JSON publication, the selectable/filterable K200-style rank table, dedicated SImain per-trade generation, completed-run catalog routing, and the cumulative navigation shell without changing immutable snapshots. Its build phase derives gross median-trade, drawdown, win-rate, and transfer-diagnostic fields from retained source/target trade CSVs for the presentation switch; it does not rerun target evaluation or alter frozen candidates.
- `research_variants\short_momentum_net_drop_rebound_v4_4\code\build_v4_4_strict_entry_transfer.py` freezes the source-only top-20% strict-entry K200 Pareto batch, calls the unchanged SImain evaluator, recomputes aggregate diagnostics across independently frozen transfer batches, and publishes one shared comparison plus one shared SImain per-trade entry. It records threshold quartiles and the five-dimensional cross-instrument Pareto set without a combined score.
- `research_variants\short_momentum_net_drop_rebound_v4_4\code\qa_v4_4_cross_instrument_comparison.mjs` owns file-based browser interaction checks plus desktop/mobile screenshots for both the comparison page and the SImain trade review.
- The comparison reads the current complete cumulative snapshot and its trade CSV, then writes only below `results\cross_instrument_comparison`. Target evaluation calls the existing immutable engine and does not alter campaign stages, plans, cumulative data, or frozen candidates.
- The current snapshot remains immutable. `build_v4_4_combined_union_analysis.py::publish_stable_main_assets` derives a small stable presentation under `results\all_completed_union_analysis\main`: it strips each main row to the fields consumed by the controls/table, keeps exact links into the current snapshot, and does not copy or regenerate trade chunks. The stable cumulative `index.html` is a presentation-only shell that embeds this lean main in an iframe and exposes cumulative/cross-instrument navigation; future cumulative snapshots remain independently hash-bound.

## Result-storage indirection

All runner, analyzer, and delivery components continue to use the logical project-relative `results` root. On Windows this resolves from `D:\Code\backtest-release\Backtest V4.4\results` through a directory junction to the physical root `F:\Backtest\Backtest V4.4\results`. This keeps historical absolute paths valid while placing every future result byte on F. Source code and runtime inputs remain on D.

## Active flow

1. `data_preparation/prepare_dataset.py` builds repository-local prepared inputs, confirmation-time exclusion timestamps, and the active low-activity entry-gate flag.
2. `code/v4_4_engine.py` calculates entries, exits, W baselines, and trade audits.
3. `code/run_v4_4_resumable_campaign.py` validates a frozen plan and writes immutable batch results.
4. `code/analyze_v4_4_scenario_3_stage.py` validates a completed stage, loads its instrument-bound cost model or the legacy K200 default, derives per-trade cost without mutating raw returns, and creates separate gross/cost-adjusted rankings with cost-adjusted default. A profile may omit scenarios; such a stage exposes unrestricted views only.
5. `code/build_v4_4_review_delivery.py` generates the fixed historical-template trade review with four workers.
6. `code/build_v4_4_combined_union_analysis.py` recursively discovers completed stages below the supplied `campaigns_root`, admits the accepted V4.4 K200 major ranking lineage, preserves each stage's implementation and preparation identities as provenance, and publishes the cumulative snapshot and stable entries. Structural fields required to parse a closed stage remain validated.

The cumulative builder applies the cost model bound to each source stage. Minor V4.4 engine, preparation, strategy, or result-identity differences do not split the accepted K200 major ranking lineage. A V4.5 update or a different instrument uses another cumulative root and ranking lineage.

## Boundaries

Raw compute, derived cost analysis, and cumulative publication remain separate boundaries. Intermediate rounds close immutable raw outputs and compact summaries; the runner does not publish HTML unless `--publish-html` is supplied. After the exploration series ends, one explicit publication builds the shared cumulative main and shared per-trade entries from every compatible completed stage. Existing stage pages remain historical evidence. Active runtime inputs are repository-relative. Historical absolute paths under `runtime_inputs/provenance` are records only.

`analyze_v4_4_scenario_3_stage.py::_legacy_v4_main_html()` transforms the fixed main template for the V4.4 cumulative entry. It hides the summary-card strip, fills `#all-strategy-count` from `DATA.coordinateCount`, and renders full-row multi-select minute ranges for BH, E, W, and S. Each axis unions selected intervals; the four axis predicates are intersected before ranking. `main_summary_payload()` defines the main-page data boundary, while the generated JavaScript pre-indexes rows by method and baseline policy and sorts/filters integer row indexes rather than cloning row objects.
