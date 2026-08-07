# Work Progress

## 2026-08-08 — Visual delivery evidence rule

- Future work with a visible result includes a representative screenshot when the environment allows it, with a brief assessment of overlap, garbling, asymmetry, and obvious layout errors. Local Chrome file pages may be opened through Computer Use.

## 2026-08-08 — V4.41 minor release identity

- Added `RELEASE.json` with V4.41 as the active minor release and V4.4 as the unchanged strategy/ranking major version.
- Updated active presentation labels and package metadata to V4.41 while retaining all V4.4 engine, result, parameter, and ranking-lineage identities.
- Kept every compatible completed V4.4 result eligible for the existing cumulative ranking. No backtest output was recomputed or excluded.

## 2026-08-08 — K200 train/test paired per-trade review

- Added destination-named toolbar controls to the current K200 training and test reviews. They transfer the exact selected `combo_id` and the destination research-contract identity in both directions.
- Removed the trade-picker heading/count and secondary summary line while retaining the transaction dropdown with an accessible label.
- Moved drop, ratio, entry baseline, entry threshold, W baseline, and active-low boxes from Plotly annotations to the reason panel. Their borders and semantic colors are retained in compact wrapping controls.
- Moved the Plotly legend below the x-axis, changed the browser title to `组合平仓逐笔查看`, and installed a blue `Z` favicon.
- Refreshed only the K200 training/test HTML shells and their manifests. Existing OHLC, process payloads, catalogs, deterministic trade chunks, fills, metrics, and ranking data were not rebuilt. Desktop dark/light and 390-pixel browser checks passed with exact-coordinate round-trip navigation and zero errors.

## 2026-08-08 — Date-based evaluation packages and compatibility comparison

- Added `tools\build_v4_4_evaluation_framework.py` and registered three current date packages: K200 `2026-05-26 00:00:00`–`2026-07-08 23:52:00`, K200 `2026-07-08 23:52:15`–`2026-08-07 03:21:45`, and SImain `2026-01-29 00:00:00`–`2026-02-23 23:59:45`.
- Added `tools\register_v4_4_evaluation_package.py` and a declarative package-spec template so a newly completed executable instrument and exact interval can enter the same catalog without adding instrument-specific directory logic.
- Each package contains `evaluation_manifest.json`, `EXPERIMENT.md`, a neutral `parameter_summary.csv`, a 350-candidate browser projection, an immutable trade-record hard link, and a query/hash-preserving per-trade compatibility entry. The first K200 package retains all 37,058 completed training-summary rows.
- Added `results\evaluation_comparison` with a comparison plan and loader that reads all three packages, joins exact `combo_id` values, restores the retained field order, and runs the unchanged 350-coordinate page shell. The stable cross-instrument URL now redirects to this entry; the prior entry is retained under the compatibility directory.
- Compatibility audit passed exact row/value reconstruction for all 350 candidates. Existing K200 stable main/per-trade and retained comparison/test/SI per-trade page hashes remain unchanged. Chromium found equal title, visible body text, first-row data, and visible row count with zero errors; paired screenshots and the QA report are retained under `project_management\screenshots\evaluation_framework_20260808`.
- Updated bilingual operating, goal, constraint, current-state, architecture, data, backtest-management, campaign, decision, reason, task, and progress records. No backtest, metric, parameter, engine rule, or per-trade result was recomputed.

## 2026-08-08 — Triple-return per-trade routing published

- Renamed the cross-instrument rank heading to `排名` and reduced the fixed-width blue rank button to `#N`.
- Applied the same blue-button component to K200（训）, K200（测）, and SI total-return cells. Each opens the corresponding per-trade evidence in a new tab; K200（测） received a dedicated 350-candidate, 34,248-trade review generated from retained records, with no strategy rerun or metric change.
- Reworked the comparison-range panel into a full-width heading row, one aligned row of four selectors plus the load action, and a full-width status row. The comparison table header and data cells are centered without changing sorting or frozen-column behavior.

## 2026-08-07 — Current K200 optimal-parameter one-month replay closed

- The first 100-candidate run completed but exposed a selection-definition error: the gross non-gap queue admitted twelve candidates with negative cost-adjusted training return. Its raw evidence is retained under `v4_4_k200_current_optimal_forward_initial_20260807`, classified historical and invalid for decisions, and points to the corrected replacement.
- The corrected `_v2_` freeze contains 100 cost-positive training candidates, six previously evaluated controls, and 94 exact coordinates not previously run over the later month. Four workers completed the stage while available memory remained above 8.8 GB.
- The later month has 29 cost-positive candidates, 28 cost-positive candidates with at least ten trades, and eight positive non-gap candidates. Median return is -0.671%, median trade count is 84.5, and train/later Spearman is -0.346.
- The total-return leader E432/BH240/TRW22/K1/W6/M4.5/S330 earns +9.175% with -8.243% non-gap return. E112/BH612/TRW30/K1.3/W10/M2.5/S308 earns +4.693% total and +4.573% non-gap with ten trades, but Top-2 concentration is 72.2%. The current conclusion is weak portability with sparse non-gap exceptions; no parameter is accepted and no HTML was published.

## 2026-08-07 — K200 train/test/SI migration closed

- Froze 100 temporal candidates before SI evaluation after excluding all 250 retained SI candidates; overlap is zero. Four workers produced 23,828 new SI trades, with 59 positive candidates.
- Replayed all 350 candidates over the K200 test interval with four workers, producing 34,248 trades and 199 positive candidates. The final SI union contains 48,022 trades and 275 positive candidates; 149 candidates are positive in all three columns.
- The new E320/BH240/TRW12/K1.25/W6, M4.25–4.75, S340–370 neighborhood is continuous across seven points, but median K200-test non-gap return is -12.213%. No parameter is accepted.
- Published one final three-return entry and one shared SI per-trade entry. The incremental build reused 250 retained chunks and generated 100. The directory-local SImain README now records its SIH6 contract choice, exact intervals, unadjusted policy, merge rules, and update history.

## 2026-08-07 — K200 temporal migration closed

- Froze and evaluated five 400-coordinate stages: four sequential walk-forward slices plus one post-hoc full-test replay. Four workers completed 2,000 coordinate evaluations; an R1 memory-floor stop at 344/400 resumed from retained batches without changing the candidate freeze.
- Positive counts by unseen slice are 296, 26, 383, and 25. Of 218 candidates observed in all four slices, only two remain positive throughout; both have 11–13 full-test trades and 100% median Top-2 positive-return concentration.
- The training/full-test return-rank Spearman is -0.26169. The training leader falls from +82.466% in training to -1.567% over the full test. The evidence rejects a static general parameter claim and retains short-window re-estimation plus regime gating as a future hypothesis only.
- Delivered the temporal ranking, 400-candidate full-test per-trade analysis, complete comparison CSV, machine summary, and Markdown report. Intermediate rounds generated no HTML. `parameter_acceptance=none`.

## 2026-08-07 — K200 Tick data extended and activated

- Resumed one retained IBKR K200 `TRADES` acquisition from `2026-07-28T16:14:30+09:00` to the half-open end `2026-08-07T03:22:00+09:00`. It closed 1,071 pages and 1,065,932 ticks; two invalid rows and two immediate-recovery outlier ticks were excluded by the current cleaner.
- Produced 34,168 cleaned session-filled 15-second rows. Appended the original 183,056-row source, the prior 16,144-row supplement, and the new segment without overlap or timestamp reversal.
- Activated 233,368 rows from `2026-05-23T00:00:00+09:00` through `2026-08-07T03:21:45+09:00`; active SHA-256 is `9760d367a109777c4789ce45d982a6c0708bacddad8f549450ed94f81ad5c405` and prepared identity is `v4_4_confirmed_low_activity_gate_9760d367a109777c_76f2695bc1f4_9e27394dbe49`.
- Added and backfilled per-download K200 READMEs. The common data rule now requires every instrument acquisition directory to record update history, contract selection/splits, lineage, merge behavior, and principal files.
- Added bilingual `BACKTEST_MANAGEMENT` documents and made their instrument/file/evaluated-interval row mandatory before every future backtest. No backtest or HTML generation occurred in this task.

## 2026-08-07 — On-demand interval statistics added to per-trade review

- Removed the requested view-mode sentence and added the `区间统计` control immediately before the dark-theme button.
- Added horizontal Plotly selection and a theme-aware temporary panel containing the requested ten interval fields. The calculation runs once after selection over the selected visible OHLC slice.
- Reworked the panel into a compact 510-pixel flat three-column layout. Closing it now clears the selection rectangle and selected points, exits interval mode, and restores zoom dragging.
- Cleared trace-level selected points after each interval selection so unselected market data keeps normal opacity; retained a pale white overlay inside the selected range in dark mode.
- Removed the `紫色` and `橙色` text suffixes from the baseline and threshold chart annotations.
- Added mutually exclusive `持仓检测`. A raw-candle click derives open-position status, active low, contemporaneous rebound check, max-W baseline range/value, M threshold, and S-window speed distance. It uses recorded pending-exit boundaries and highlights the effective baseline interval. Aggregated candles request more zoom instead of returning an approximate point.
- Moved both inspection panels from the chart overlay to the right detail column above trade reasons and made activation collapse the parameter drawer.
- Replaced retained native selection styling with an ordinary Plotly shape after clearing native selected points and selections. This keeps the range visible without dimming surrounding market data.
- Made holding check respond to every visible candle: a raw candle uses its exact bar, while an aggregate uses its final source bar. Set candlestick `whiskerwidth` to zero to remove high/low endpoint caps.
- Replaced chart-local pointer-up with document-level pointer-up because Plotly's left-button drag layer could retain the release and leave stale state that a later right click consumed. Holding mode now keeps Plotly zoom dragging: movement beyond five pixels is treated as a drag, a short left press selects the candle, and right clicks are excluded. The selected close receives a blue point. A darker borderless blue baseline fill extends from the effective baseline start to the selected-time active low; the exact max-W source interval remains in the panel.
- Holding check now draws a labeled pale-red dense dashed horizontal guide at the contemporaneous theoretical rebound cover (`active low + threshold`). While the S window is incomplete, it also draws a labeled vertical guide at the earliest eligible speed-check bar with remaining bars and elapsed time; this line denotes eligibility, not guaranteed execution.
- Corrected the speed guide to `activeLowIndex + S`, so it remains visible after S is formed and represents the no-new-low theoretical speed-exit position. The blue selection marker now uses the displayed aggregate candle's center and high instead of the final source bar's x/close.
- Refreshed snapshot `eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5` through the presentation-only path. No backtest, result metric, process payload, or per-coordinate trade chunk was regenerated.
- Browser interaction capture was unavailable because the browser security policy blocks local `file://` navigation. The generated inline scripts passed the focused source-level interaction check.

## 2026-08-06 — K200 trade-count filters and column headings updated

- Renamed the cumulative-page fieldset from `最低交易数` to `交易数`, retained the all/10/20 presets, and removed the 100/150 presets.
- Added strict editable `greater than` and `less than` trade-count bounds that can operate together and remain independent of the retained preset threshold.
- Centered main-table column headings without changing body-cell alignment, result values, sorting, or immutable evidence.

## 2026-08-06 — Six-hour K200 leap/grid cycle closed and published

- Closed 89 automated rounds with four workers: 60 leap rounds and 29 adaptive-grid rounds, adding 31,375 coordinates and 10,843,398 trades. Exact anti-join prevented repeated coordinates.
- The final cumulative population is 37,058 unique coordinates and 11,749,606 trades across 109 stages. Cost-positive coordinates rose from 4,721 to 19,974 and Scenario-1-qualified coordinates from 1,471 to 4,406, while all four headline leaders remained unchanged.
- Published snapshot `eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5` once at session end. Incremental trade review reused 5,320 chunks and generated 31,738; peak publisher memory stayed below 2 GB after the streaming, vectorized, four-process rewrite.
- K200 global convergence was not established because later leap rounds continued producing promising nonadjacent anchors. SI migration was therefore not started. No parameter was accepted.

## 2026-08-06 — Leap/grid exploration governance updated

- Removed the fixed point-count limit from future one-parameter searches. Each grid now freezes finite bounds, values or steps, anchor, and expected coordinate count without a fixed cap.
- Replaced the requirement that every round mix broad and refinement blocks with an AI-led cycle: multi-round leap search, grids around promising nonadjacent anchors, renewed leap search, and user observation or correction between cycles.
- Registered the approximately six-hour, four-worker unattended K200 search as active. Intermediate HTML remains deferred; no new compute evidence is claimed in this governance entry.

## 2026-08-06 — K200 cumulative presentation simplified

- Replaced the persistent stable-entry navigation shell and iframe with a query/hash-preserving redirect to the lean main page.
- Updated the title and blue accents, simplified return labels, removed the visible gap-dependence return audit, moved cost and cross-gap count to the last two columns, and changed rank buttons to fixed-width `#N` labels.
- Replaced full-table DOM generation with 500-row client-side pagination. Filtering and sorting still process the complete eligible set; page or sort changes render only the current slice. The table is full-width, occupies nearly one viewport, and no research-contract section follows it.

## 2026-08-06 — Unified migration ranking updated through K1.75

- Completed the four-worker K200 source stage and unchanged SImain evaluation for K1.5/K1.6/K1.75 with E320/BH240/TRW20/W6/M4.5/S332 fixed.
- Published the compatible 247-candidate parent plus three new candidates as one 250-candidate ranking with 24,194 target trades.
- Closed the stricter-K direction at the K1.4 local target peak. No parameter was accepted and no follow-up migration run is active.

## 2026-08-05 — Dual-purpose K200 continuation round closed

- Recorded the durable exploration rule: ordinary rounds combine broad legal-space coverage with evidence-led one-parameter refinement, and interpret the two branches separately.
- Froze 276 zero-overlap coordinates: 192 deterministic stratified broad points plus three new values for each of seven parameters around four strong anchors. Four workers closed all 35 batches and 54,314 trades while available memory stayed above the 4,096-MiB gate.
- Broad coverage produced 103 cost-positive rows and one new return/drawdown frontier point. That remote point has 55 trades, 8.0376% return, and 1.6933% drawdown, but non-gap cost-adjusted return is negative, 36 signals are synthetic, and the best two positive trades contribute 50.3% of positive return. The broad branch is `mixed` and does not support local refinement of that point.
- One-parameter refinement improved the average-E branch. E96 becomes the minimum-10 and minimum-20 average-trade leader with 25 trades, 35.2290% total return, 1.23914% average trade, and 3.0190% drawdown. E128 is the stronger balanced point: 29 trades, 38.9271% return, 1.16460% average trade, and 4.21795% drawdown, improving all three metrics against E112.
- Published one final snapshot with 5,320 coordinates and 797,020 trades. Incremental trade-review generation reused 5,044 chunks and generated 276. The sixty-coordinate handoff has zero completed, active, or pending overlap and remains unauthorized for compute.

## 2026-08-05 — Large K200 multiblock round closed and published incrementally

- Froze a 294-coordinate plan after removing twelve internal duplicates and forty-nine completed-coordinate overlaps from 355 requested candidates. Four workers completed all 37 batches and 28,577 trades; available memory stayed above the 4,096-MiB execution gate.
- Published one final cumulative snapshot with 5,044 coordinates, 742,706 trades, and fifteen compatible stages. The cumulative publisher reused 4,747 unchanged per-trade chunks from the prior snapshot and generated 297 missing chunks, including the 294 Round-14 coordinates and three previously unpublished stricter-K coordinates.
- The unrestricted total-return leader remains E480/BH171/TRW12/K1.26/W7/M4.5/S388 at 82.4664%. The Scenario-1 leader also remains unchanged. Both minimum-10 and minimum-20 average-return views now lead with E112/BH612/TRW24/K1.6/W16/M2/S308: 29 trades, 38.4486% total return, 1.15253% average trade, and 4.23380% drawdown.
- Relative to the prior average-return leader, total return improved by 4.34393 percentage points, average trade by 0.14630 points, and drawdown fell by 0.40942 points with one fewer trade. The retained entry population changed only by one removed entry, while ten retained trades changed exit time; the improvement is therefore primarily exit-behavior evidence.
- The result remains sensitive to synthetic-signal and gap trades. No parameter is accepted. Fourteen anti-joined coordinates were written as a proposed next-round handoff; no further compute was launched.

## 2026-08-05 — Human project guide added to the management Dashboard

- Added a default human-only Dashboard landing page with two responsive workflows: the four official campaign modes plus their shared execution path, and the causal lifecycle of one trade.
- Added a generated directory grouped by the existing management manifest. Directory links open the current formal documents, while the project icon returns to the guide.
- Promoted `Project guide` to the first left-navigation row and rendered it through the same document header, metadata, content width, typography, outline, and responsive shell as every managed document.
- Kept the guide outside the document manifest and AI priority-reading matrix. Desktop and 390-pixel mobile checks passed with both languages, themes, navigation, and no horizontal overflow or visible layout defect.

## 2026-08-05 — Stricter-entry transfer and incremental migration publisher closed

- Closed one K200 confirmation batch at three coordinates and 243 trades using four workers without intermediate HTML. K1.2/K1.3/K1.4 reduce K200 trades from 89 at K1.1 to 83/83/77, while K200 cost-adjusted return falls to 46.9394%/29.3673%/19.7248%.
- Froze those three coordinates and evaluated them on SImain with four workers. Target trades are 85/73/67 and cost-adjusted returns are 16.4678%/33.9182%/35.6004%. K1.3/K1.4 also improve target drawdown and non-gap return against the K1.1 parent.
- Published the 247-candidate combined cross-instrument entry once. The fixed migration builder reused 244 deterministic target trade chunks from the parent through same-volume hard links and generated only the three new chunks. The trade-review manifest records reused=244 and generated=3.
- Added `--run-id` to the fixed comparison builder and bound future incremental publication through `run_config.json::incremental_parent_run`.
- Generated the source-side three-coordinate stage main and per-trade analysis through the fixed stage analyzer with cumulative refresh disabled. The 4,747-coordinate stable K200 snapshot and its approximately 714,000 historical trades were not rebuilt.

## 2026-08-05 — Plan-driven 244-candidate migration redelivery published

- Added a reviewed migration-plan file with relative source and target identity fields and deferred candidate-filter status.
- Combined the user-linked 180-candidate result and repaired-source 64-candidate result into one 244-candidate ranking with zero coordinate overlap; retained both completed source runs as evidence.
- Removed the migration-batch field, placed source total return immediately before target total return, added three-state sorting, and made all instrument labels plan-driven.
- Added reciprocal new-tab navigation between the source main entry and cross-instrument entry. Published one combined SImain per-trade analysis and bilingual migration report.
- Recorded the approval-before-extra-change rule and five-item final migration delivery contract in bilingual management documents. Focused browser evidence covered the affected interactions and showed no runtime or visible layout error.

## 2026-08-05 — Five-round K200 one-axis exploration published once

- Ran five immutable three-coordinate rounds with four workers while returning to the same E480/BH171/TRW12/K1.26/W6/M4.5/S388 anchor after every unsupported direction. Raw compute added fifteen coordinates and 1,707 trades.
- BH205/257/480, TRW13/14/15, K1.4/1.5/1.6, and K1.15/1.05/0.95 all failed to improve the anchor. W7 reached 82.4664% cost-adjusted return versus W6 at 82.4352%, but W8/W9 fell to 75.2452%/75.4916%; W7 also has lower average trade, slightly higher drawdown, and lower gap-excluded return.
- Every intermediate round retained raw evidence, an interpretation JSON, and an anti-joined next-round CSV without HTML. The final header-only CSV closes the series with `parameter_acceptance=none`.
- Published cumulative HTML once at the end. Snapshot `7528265de87be7e855f8e2c80585c52de95c248f0747ca60c1f3a7bcc5ae81b2` contains 4,732 coordinates, 712,108 trades, eight compatible stages, and zero duplicate coordinates. The stable main and shared per-trade entries point to it.
- Compute memory remained above the 4,096-MiB gate; observed final-publication free memory stayed above approximately 4.4 GiB. The final publisher exited normally.

## 2026-08-05 — Exploration batching and final-only HTML publication adopted

- This historical update moved from isolated single points to bounded scale-aware one-axis directional sets plus the anchor, with every other parameter fixed. Its fixed point-count limit is superseded by the 2026-08-06 leap/grid cycle.
- Updated the resumable campaign runner so intermediate rounds defer cumulative HTML by default. A final or user-requested publication now requires `--publish-html` and still supports the existing background or synchronous delivery modes.
- Updated bilingual goals, constraints, research guidance, campaign workflow, architecture, source-of-truth, decisions, and active records. Existing raw evidence and published snapshots remain unchanged.

## 2026-08-05 — E-window reduction experiment published

- Ran thirteen unique fixed-parameter E coordinates in two four-worker stages: E=304, 256, 192, 160, 136, 112, 96, 80, 64, 48, 32, 24, and 16. Raw closure contains 1,388 new trades.
- Published cumulative snapshot `b077548e654277738e1d953ce7bea01eb184a0e223f064c7728dbc2de4d1a561` with 4,717 coordinates and 710,401 trades. Stable main and shared per-trade entries now include every new coordinate.
- The first attempted stage used an unrecognized plan-status label. Its complete raw bytes were moved intact to `results\staging_recoverable\entry_window_reduction_plan_status_failed_20260805`; no file was deleted. The approved replacement produced identical eight-coordinate raw summaries.
- Added durable project rules favoring minimal implementation, limited defensive code, and risk-proportional verification. This experiment used only immutable raw closure, delivery completion, coordinate/trade presence, and entry-file existence as its final evidence.
- Full cumulative publication remains a performance bottleneck: each small stage rebuilds the approximately 710,000-trade immutable snapshot. Incremental publishing remains a proposed performance improvement and was not implemented in this task.

## 2026-08-05 — Rolling-sum-mean label clarified and entry causality audited

- Renamed the cumulative method display from `滚动 TR 总和` to `滚动 TR 总和均值` in the current generator, user-selected snapshot, alias, status summary, and stage-page source. The value remains `rolling_tr_sum`; filtering, ranking, coordinates, trades, and returns are unchanged.
- Confirmed the actual formula: BH=171 and TRW=11 produce 161 overlapping 11-atom sums, and the baseline is their mean. The selected 2026-06-26 trade records baseline 4.6642857, threshold 6.53, drop 7.0, and ratio 1.5007657.
- Audited source indexing and all 114 trades of the leading coordinate. Every trade satisfies `baseline_start <= baseline_end == H < signal <= entry`; the engine chooses H only from bars observed through the current signal calculation, rejects same-bar H/signal ordering, and retains the existing batch-versus-stepwise-prefix equality test.
- The broad red E band is a high-search window, not the measured net-drop interval. For the selected trade E spans 19:15:15–20:35:00, BH spans 19:30:00–20:12:30, and the measured drop spans H at 20:12:30 through the signal at 20:35:00. Current logic has no future-data read, but BH includes the complete H atom. A strict H-minus-one baseline would be a new method requiring a new source identity and fresh results.
- Focused browser QA confirmed the new label, unchanged `rolling_tr_sum` value, one method option, populated ranking and trade route, zero runtime errors, zero external requests, no replacement characters, and no page overflow. Visual review found no truncation, overlap, garbling, or asymmetry.

## 2026-08-05 — Low-latency icon-only cumulative filter panel published

- Replaced the text-and-triangle toggle with a fixed-size, icon-only SVG chevron disclosure button. It shows no visible label, points up while expanded and down while collapsed, and retains an English accessible label plus `aria-expanded` state.
- Isolated the large ranking table with `contain: layout paint`. Measured button-handler time is at most 0.3 milliseconds and the 12-toggle next-frame p95 is 14.9 milliseconds, replacing the prior observed 150–450-millisecond repaint delay.
- Published the UI-only change to the user-selected snapshot `e4a20d1d5bcb8974f4341a3647e2e246c3c1ab855d66ce4b3d7d4998d7fb3d44` and its `analysis_report.html` alias, then reconciled their analysis and completion manifest hashes. Analysis data, raw trades, rankings, and all other snapshot artifacts remain unchanged.
- Focused browser checks covered collapse, re-expansion, icon direction, fixed button dimensions, ARIA state, desktop and exact 390-pixel mobile layout, runtime errors, external requests, replacement characters, and page overflow. Visual review found no overlap, garbling, asymmetry, or abnormal whitespace.

## 2026-08-05 — Separated strategy, instrument, and campaign contracts

- Added an instrument-neutral strategy contract, a ready K200 profile, incomplete SImain/NQ templates, and a campaign-manifest template for `transfer_exact`, `target_local_refinement`, and `fresh_search`.
- Added a generic cost loader that calculates notional from reference price times point value and preserves the historical K200 numeric contract and aliases.
- Extended schema-v5 plans with instrument profile, campaign mode, optional scenario policy, and ranking-lineage identity while preserving old plan behavior.
- Updated cumulative cost processing so each source stage retains its bound model. Existing K200 results remain untouched; future compatible K200 rows can rank in the same lineage with their actual cost disclosed.
- Updated bilingual project management and prompt forms. No backtest, raw write, cumulative rebuild, or parameter acceptance occurred.

## 2026-08-05 — Added gross/cost comparison modes and paired total returns

- Added the K200-style two-option return-view control with fee/slippage-adjusted default and gross alternative. The selected mode changes ranking and displayed K200/SImain total return, median trade, maximum drawdown, and win rate together.
- Kept paired K200/SImain return views while grouping the final table as parameters, SImain metrics, K200 metrics, and transfer diagnostics. Hover help on the K200 total explains its longer evaluation interval.
- Retained the completed-run selectors, global/field filters, and presentation-only cumulative navigation shell while adding the new return-view axis.
- Derived gross-view presentation metrics from the retained source/target trade CSVs, rebuilt only derived HTML/data resources, and preserved frozen candidates, target trades, migration CSV/JSON, and immutable cumulative snapshot files.

## 2026-08-05 — Restored selectable and fully filterable cross-instrument review

- Added source instrument, source interval, target instrument, and target interval selectors backed by the completed-run catalog. The current default remains K200 2026-05-26–2026-07-08 versus SImain 2026-01-29–2026-02-23.
- Added global search and composable field filters over every displayed parameter, target metric, source metric, and transfer diagnostic while retaining interactive sorting on every column and the three-key default target ranking.
- Replaced the stable cumulative redirect with a presentation-only navigation shell that embeds the byte-preserved current snapshot and links to the cross-instrument entry. The snapshot HTML, analysis data, union trades, frozen candidates, SImain trades, migration CSV, report JSON, and run config remain unchanged.
- Rebuilt derived comparison and SImain trade-review presentation only. Focused Python tests and desktop/mobile browser QA passed; no external requests, runtime errors, body overflow, aggregate score, target-driven candidate mutation, or parameter acceptance was introduced.

## 2026-08-04 — Cross-instrument ranking aligned with the K200 interface

- Removed the four dataset selectors and every table filter input from the standalone K200-to-SImain page.
- Reused the K200 `rank-link` component: the first header is `成本后排名 ▲`, and all 180 rows expose blue `查看 #N` buttons that open dedicated SImain trade review in a new tab.
- Removed the old K200/cross-instrument switch hub from the stable cumulative entry while preserving the immutable cumulative snapshot, analysis data, and union trades.
- Rebuilt the derived comparison and SImain trade-review presentation from the existing 17,044 target trades; no candidate, raw trade, migration metric, or parameter-acceptance state changed.

## 2026-08-04 — Parameter-scale search resolution added to the exploration guide

- Added parameter-specific broad, local, and stability resolutions for E/BH/S, TRW/W, K/M, and A/floor.
- Replaced routine single-digit refinement of large time parameters with multiplicative grids and classified continued single-digit densification as overfitting.
- Required at least one exploratory block in every search round. A justified round without exploratory coverage is confirmation-only and cannot establish global convergence.
- This is a documentation and future-plan-design change only; no plan, compute, result, ranking, or HTML was changed.

## 2026-08-04 — V4.4 results migrated physically from D to F

- Froze 43,817 V4.4 result files totaling 115,416,958,661 bytes, with zero matching writers and all 30 result locks acquirable.
- Copied the complete tree to `F:\Backtest\Backtest V4.4\results` with eight threads. Robocopy reported 43,817 copied files, zero missing, mismatched, failed, or extra files.
- Performed source-versus-target SHA-256 comparison for all 43,817 files and the full 115,416,958,661 bytes; mismatch count is zero.
- Replaced the historical D result directory with a Windows directory junction to F. Pointer, main entry, shared per-trade entry, and desktop/mobile main interaction checks pass through the D logical path.
- Moved the byte-identical D source copy to `F:\Backtest\migration_recovery\Backtest V4.4 results before junction 20260804`. Moved the four approved old result directories to `F:\Backtest\D_cleanup_quarantine_20260804` without permanent deletion.
- D free space increased to approximately 165.04 GiB. Source code, runtime inputs, plans, snapshots, manifests, rankings, and HTML bytes were not regenerated.

## 2026-08-04 — Four cumulative-main window filters published

- Added multi-select minute ranges for entry baseline BH, entry market E, exit baseline W, and exit market S to the existing cumulative main entry.
- Multiple ranges on one axis are unioned; the four axis conditions are intersected. `All` leaves one axis unrestricted.
- Published the presentation change in place on snapshot `e4a20d1d5bcb8974f4341a3647e2e246c3c1ab855d66ce4b3d7d4998d7fb3d44`. All 4,348 coordinates, 669,694 trades, ranking data, and the shared per-trade HTML remain unchanged.
- Used focused desktop/mobile interaction and visual checks because this is frontend filtering behavior, not a version, engine, or result-semantic change.

## 2026-08-04 — Continuation Rounds 11–13 completed in one cumulative lineage

- Ran bounded broad and local refinement in Round 11, surface refinement in Round 12, and one-bar speed-peak confirmation in Round 13 without creating a new campaign or result branch.
- Added 681 compatible coordinates across the three rounds. The final shared snapshot contains 4,348 coordinates, 669,694 trades, and 16 stages; the same cumulative main and one large shared per-trade HTML were refreshed after every round.
- The average-return leader improved from +0.8458% before reopening to +0.9464% at S308. The S308–S312 plateau is supported, while the remaining gain is too small to justify further in-sample speed refinement.
- All raw closures, browser suites, stable routes, and full delivery manifests passed. Exploration stops with `parameter_acceptance=none`.

## 2026-08-04 — Continuation Round 10 delivered; exploration paused

- Closed and delivered 83 average-return stability coordinates in the same campaign and shared cumulative lineage.
- Published 3,667-coordinate snapshot `73ea40e633f2ea4d4c70ab906c5b4119fa7fb1ecc41113b8e949a089a8a4fdb3`; browser, artifact, route, and mobile visual checks passed.
- Recorded the improved average-return leader and its remaining concentration/gap dependence. No next plan was created.

## 2026-08-04 — Continuation Round 9 closed in the same cumulative lineage

- Closed and delivered 196 new coordinates without creating another campaign or result branch.
- Published 3,584-coordinate snapshot `3f893a17657375bbdf665b238ee737fb2e7709d98d259dfb46ab3655d74c19fa`; all browser and 3,588-output integrity checks passed.
- Stopped both non-improving total-return branches and prepared an 83-coordinate Round-10 handoff around the improved average-return S320 region.

## 2026-08-04 — Continuation Round 8 closed and published in the existing cumulative lineage

- Closed 637 new coordinates, 63,388 trades, and 54 batches under the unchanged campaign ID; all raw contracts and hashes reconciled.
- Published cumulative snapshot `0569e8b7859ba4f6c896a870dcb20f092e4c55747496dafb64a49552edf56ebe`, which ranks the prior 2,751 and the new 637 coordinates together: 3,388 compatible coordinates, 624,033 trades, and 11 stages.
- Stage, immutable-snapshot, and stable-route browser QA each passed 280 interaction states; the full 3,392-output hash/size audit passed with zero mismatches.
- Recorded block-level continuation reasons. Round 9 retains 196 anti-joined coordinates for the two total-return micro-stability surfaces, strict-entry average-return checks, and medium-speed controls; no new result branch is created.

## 2026-08-04 — Continuation Round 8 designed in the existing lineage

- Reconciled the real ten-stage cumulative state at 2,751 compatible coordinates and 560,645 trades; older five-stage project summaries were retired from active state.
- Recomputed exact fixed-3.57-bps compounded rankings and anchor diagnostics from the current cumulative summary, union trades, and 15-second OHLC.
- Built a 637-row expanded Round-8 handoff across eight experiment blocks. All 637 coordinates are unique and have zero overlap after protecting 2,752 completed coordinates; active=0 and pending=0.
- Kept campaign ID/root unchanged so new results will rank and render with the existing 2,751 coordinates in the shared main and per-trade HTML.

## 2026-08-04 — Mandatory Parameter Exploration Guide added

- Added a bilingual guide under the active research domain and registered it in the management manifest.
- Root agent rules and the management entry now require the guide for every parameter-exploration design, run, interpretation, and next-round handoff.
- The guide records random early trade inspection, later emphasis on profitable candidates with remaining improvement room, trade-type diagnosis, flexible multi-block rounds, continuing broad jumps, and model-explained multi-metric judgment without permanent weights or a fixed score.
- Future exploration cost is fixed at `3.57 bps`. Every round ends with expanded next-round coordinates in a standalone `next_round_parameters.csv`, followed by uniqueness and completed+active+pending anti-join before plan binding.
- This task changed management documentation only. Backtest source, raw results, result HTML, and cumulative snapshots were not changed.

## 2026-08-04 — Validation tiers documented

- Added a durable three-tier validation rule to the project operating entry, constraints, decisions, and agent instructions.
- Complete regression is reserved for version/core/result-semantic changes; frontend behavior uses focused interaction checks; presentation-only changes use regenerated HTML, a simple functional check, and desktop/mobile screenshots.
- The cumulative-main speed-window grouping change is classified as frontend interaction plus presentation and therefore does not require another complete regression run.
- Published cumulative snapshot `8ddc0d2d0a32f3e5a6ec4710a3a8a64774029ce5a6170ccd724bf07282229fb8` from the 2,751 completed coordinates and 560,645 trades without raw recompute. The shared main now uses seven minute ranges: all, under 5, 5–under 15, 15–under 30, 30–under 60, 60–under 120, and 120 or more.
- Targeted browser checks passed for all range boundaries, one-row desktop layout, two-column mobile layout, stable-entry routing, zero runtime errors, and zero external requests. Desktop and mobile screenshots showed no overlap, garbling, abnormal whitespace, or horizontal overflow.

## 2026-08-02 — V4.4 temporary repair validation closed

- Copied the relevant V4.3 project files into an independent V4.4 folder; prior results were not copied into the active result tree.
- Added the four requested causal execution/data repairs and focused regression coverage.
- Regenerated schema-5 prepared data with `baseline_available_from`: 122,843 atoms immediately available, 55,221 available after recovery, and 4,992 never available.
- Closed one immutable coordinate: 3,882 trades, one batch, fixed-template delivery generated with four review workers.
- The reported trade now fills at 1514.850 close after a 1514.825 theoretical rebound line. The source window begins after H.
- Desktop, mobile, route, and target-trade visual checks passed.

No parameter acceptance or broad parameter exploration was performed.

## 2026-08-03 — Cost/method source and Round-1 plan closed

- Closed dual gross/cost-adjusted ranking with cost-adjusted default. The selected shadow cost is 3.56 bps per completed trade and does not change raw fills or raw returns.
- Corrected the gross-mode rank header and extended browser QA to require the mode-specific header.
- Recorded the exact available-prefix `w_open_to_end_low_drop` and retained pending-entry contracts while preserving legacy raw audit field compatibility.
- The current source manifest is `6fa3d0c8eb0277066ef5f70fca4a9fbab1d31fbb30e023cd8fd83d233192ae16`; full source tests closed with 70 passes and 2 result-dependent skips.
- Frozen the non-executable multi-round design and 372-coordinate Round-1 plan. Validate-only, anti-join/resource evidence, and new-stage writes remained outside this source/design work; no raw compute or delivery was written by it.

## 2026-08-03 — Round-1 delivered and Round-2 launched

- Closed Round 1 immutably at 372 coordinates and 316,398 trades, then published the fixed-template active-campaign snapshot `2020ad7b12d57889f1c1d0cf69f981bcf2b5e3ec5b8a4808c196dbb6cdd51d47`.
- The stage, snapshot, and stable routes each passed 200 browser states with no runtime, external-request, or layout failures. Cost/gross ordering, displayed returns, and rank headers were verified.
- Kept the older temporary validation campaign outside the active multi-round cumulative lineage because it has a different engine identity. The older campaign and the failed partial recovery snapshot remain preserved as historical evidence.
- Interpreted only the closed Round-1 result: 41 coordinates qualified for Scenario 1 and 127 had positive cost-adjusted total return. The Scenario-1 leader pressed the `M=8` and `S=320` boundaries; the unrestricted leaders concentrated at `E=40`, `BH=720`, `TRW=6`, and `K=2`, with competing W/M/S neighborhoods.
- Frozen the evidence-driven 247-coordinate Round-2 broad-plus-local plan. The gates passed and compute-only closed immutably at 20,629 trades and 21 batches under fingerprint `7ad95dbd7ba9ebc1faffd8cbc1723211273453af0471ab950e5b3d798ee6c4e8`; all 210 indexed artifacts reconciled and delivery was cleared.

## 2026-08-03 — Round-2 delivered; terminal Round-3 frozen

- Published two compatible active-campaign stages as snapshot `dde99537b4584f0d5d98a70e388cacffd226736a455963a2f54acd47b4bfd847`: 619 coordinates and 337,027 trades. All 631 artifact checks and stage/snapshot/stable browser and visual QA passed.
- The Scenario-1 leader improved from 17.0157% to 25.5775% cost-adjusted at E320/BH720/TRW24/K1.25/W2/M6/S400. Its lower-M and upper-S boundaries remain specifically unresolved.
- The unrestricted leader improved from 22.6187% to 36.0556% cost-adjusted at E40/BH720/TRW6/K2/W48/M4/S400. W/M are interior; S=400 is the lower local boundary.
- The 144-coordinate broad block did not improve either objective and is stopped. The terminal plan contains local resolution only.
- Frozen 212 unique, non-overlapping terminal Round-3 coordinates across eight blocks; plan SHA-256 `46c95b24feab49b6f260a0e8f1e1125fd74c34c6a0e268b89e0e1fb83a6d9b8c`. Round 4 is prohibited and no parameter is accepted.

## 2026-08-03 — Terminal campaign closed

- Closed Round 3 immutably at 212 coordinates and 16,847 trades, then published final snapshot `0fb3e1e5e8ef890f3b225db46288fa4b3957bcb88c7ca2dff72d750679db6922`: 831 coordinates, 353,874 trades, and three active stages.
- All 843 artifact checks passed. Browser QA passed 360 terminal-stage, 520 snapshot, and 520 stable-route states with no runtime, request, or layout failures; manual visual review was clean.
- The final Scenario-1 cost-adjusted leader is E320/BH720/TRW24/K1.25/W4/M4/S400 at +30.6696%, improving Round 2 by 5.0921 percentage points.
- The terminal unrestricted branch did not improve: its best new row reached +32.8804%, so the Round-2 E40/BH720/TRW6/K2/W48/M4/S400 leader remains final at +36.0556%.
- Frozen canonical interpretation `round_03_terminal_interpretation_and_campaign_closure_20260803.md`. The campaign is complete; Round 4 is prohibited and no parameter is accepted.
- Independent final read-only audit rehashed all three rounds and the final delivery: 831 unique coordinates, zero cross-round duplicates, 353,874 trades, 70 batches, 712 raw artifact records and 843 delivery checks with zero mismatches. All locks are released and no matching process remains.

## 2026-08-03 — Continuation subseries authorized and first plan frozen

- Preserved the original three-round closure, historical Round-4 prohibition, canonical terminal memo, final snapshot, and no-parameter-acceptance conclusion.
- Added a separately named `continuation_round_*` subseries under the same compatible campaign ID and active cumulative root.
- At that time, the subseries required HTML refresh and broad QA after every round. The 2026-08-05 user correction supersedes that workflow: intermediate rounds now close raw evidence and compact summaries, and cumulative main/per-trade HTML is published once after the exploration series.
- Designed a 528-coordinate broad-span first round: 288 shared entry-geometry coordinates, 180 Scenario-1 broad exit coordinates, and 60 unrestricted broad exit coordinates.
- Exact full-tuple audit found 528 unique planned coordinates, zero internal duplicates, zero overlap with the 831 compatible completed coordinates, and zero overlap with the complete 832-ID current-V4.4 protected set. Active and pending counts were zero.
- Frozen plan SHA-256 `481fd28365757f739cb0e260d3cc36a4390db9cde9b1f1ccf3063aefdb8c9bf5`; validate-only and stage creation occurred only after plan handoff.

## 2026-08-03 — Continuation Round 1 delivered; corrected-source redelivery required

- Closed raw compute immutably at 528 coordinates, 54,842 trades, and 44 batches. The plan fingerprint is `5b893814832b88a6a4e8db66ccc204065f5719060cea1f011644ff9dec237f84`; raw completion SHA-256 is `0990507be75526618663b4e08a3d628fd7af856dd692c9d9c3313de2cd0fdf6d`.
- Exactly one four-worker old-source delivery published snapshot `ce1e20f7366135cb92c098dc3db4c3245bdc2374630a89f9adafaa54d715d714`: 1,359 coordinates, 408,716 trades, and four stages. All 1,921 hash/size checks and 400/720/720 browser states passed; desktop/mobile/manual visual review was clean.
- The user then required hollow Entry Reason step-2 hover ellipses, a six-pixel upward move for the collapsed per-trade Parameters tab, solid strokes for every candlestick interval/guide while retaining semantic colors, removal of the long theoretical-line/actual-fill chart annotation, and an exact `L=<value>` green frozen-low chart label. Detailed side-panel reasoning remains. The raw stage and old-source delivery remain preserved; raw compute will not rerun.
- One executor or multiple executors may perform evidence analysis, backtest compute, and HTML delivery. After immutable raw closure, delivery and evidence analysis may run concurrently; next-round work may overlap only with closed source identity, exact anti-join, separate process/output/root/lock boundaries, one union writer, and a resource gate that includes the live delivery process.
- Every round still requires eventual four-entry `DELIVERY_FINAL`, but it no longer blocks next-round analysis or compute. A result-affecting inconsistency found during delivery re-pauses compute until a new source identity closes.
- Provisional source manifest `82f3a81edb30cbc61a62f0175806025f0c4eb4a1b63622859f41c8db547d9d15` passed an earlier focused/full gate and 28 bindings, then was withdrawn because it predated the final governance and chart-presentation corrections. It is not launch authority; `SOURCE_FINAL_V2` must rebind the corrected source, tests, and EN/ZH operating documents.
- Independent audit rejected V2 `27c99100...` because its bound active runtime manifest still declared an older trade-template hash/size. The active runtime manifest was narrowly repaired to `493880758c3dfe62d51402d92605424b74bd307d8ba401a281b79e42cb436d78` / 2,424 bytes, recursively binding trade template `eeb9689e...` / 193,787 bytes. Replacement V3 source tests remain 16 passed / 2 expected skips focused and 70 passed / 2 expected skips full; no raw or derived result was regenerated.

## 2026-08-03 — V4.4 transaction-record ZIP closed

- The user’s durable ZIP contract now includes each completed stage’s raw batch `trades.csv` and derived `analysis\\stage_trades.csv`, with source-stage provenance in `trade_records\\TRADE_RECORDS_MANIFEST.json`; every other results payload remains excluded.
- The final handoff target is `D:\\Code\\backtest-release\\Backtest_V4.4_with_trade_records_20260803_final.zip`; its adjacent sidecar and audit are the authoritative release identity after closure.
- The archive has 297 entries: 296 manifest records, nine canonical reports, five completed stages, and 120 transaction-record CSVs. Package-stream and extracted-copy verification both passed; duplicate and forbidden-entry counts are zero.
- The bound package script is `tools\\package_v4_4_with_trade_records.ps1`, SHA-256 `1e4a12a7c2e138ca253b9d2be9f78973af54f7f7220e0c961966e76bba2cd191`, 17,698 bytes. No raw result, existing HTML, or source execution logic was changed by packaging.

## 2026-08-03 — Corrected-source redelivery published; V5 QA-only repair tested

- One corrected-source direct delivery reused immutable Continuation Round-1 raw evidence and atomically published snapshot `a55ee98105958c699a29a1e32a9ccd0f3afc60cd82b29b5d88d74068fa59219a`: 1,359 coordinates, 408,716 trades, and four stages. The new stage analysis is `e643b934f32e1f84db963c33ea8e4c24276da462c5e6a2199b5aa4b369f99b2f`.
- Stage browser QA observed the contract-correct two hollow circles, transparent browser-normalized fills, semantic outlines, and 19 solid shapes, then failed because the bound assertion required an unspaced RGBA spelling. No HTML, raw result, or snapshot defect was found.
- The narrow V5 patch normalizes CSS whitespace before requiring black RGB with alpha zero and adds a regression that prohibits the old exact-string comparison. The isolated authorized runtime is Python 3.12.13 with pytest 9.1.1, NumPy 2.3.5, and pandas 3.0.1; focused tests passed 17 with 2 skips, and the explicit code-plus-data-preparation suite passed 71 with 2 skips.
- Round-2 validate-only reached the runner memory floor after materializing only four deterministic metadata files. Fingerprint `976fac8d1e6ce5b280127ad6d7000116e2280a81242e4d2f754cccac7b139e35` covers all 416 IDs; no progress, batch, trade, completion, or analysis exists. The leader superseded this V4-bound plan/root before compute. Preserve it byte-for-byte; after V5 confirmation create a new V5-bound plan/root with the same coordinates.
- The later stop-after-R2 boundary is superseded by the user's renewed instruction to continue parameter exploration. Complete the current R2 through delivery, interpretation, bilingual records, and read-only total audit; any following round needs new delivered evidence and a new reviewed, hash-bound plan.

## 2026-08-03 — V6 source closure and current Round-2 freeze

- `SOURCE_FINAL_V6` is closed at manifest SHA-256 `0aee46e6edf23eb60e5a2843e4abc5ff33ebfd0fd32e5acd945d66576104b123`. All 47 bound source/runtime/template checks match; no mandatory subagent wording remains in active project-management documents.
- The prior V4-bound R2 root remains preserved as four non-compute metadata files only. A distinct V6-bound R2 plan is frozen at SHA-256 `d982267710abab0355a37271c18a25df40decc3d9f846f82030a3ecbeab82a07`: 416 unique coordinates, 35 expected batches, zero overlap with 1,360 completed IDs, and an absent new output root.
- The only remaining pre-compute work is QA-only recovery on existing R1 HTML and fresh validate-only/pre-launch gates for the new root. Raw R2 compute has not started.

## 2026-08-03 — V6 Continuation Round 2 delivered

- Immutable raw closure passed: 416 unique coordinates, 41,134 trades, 35 batches, and 350 indexed raw artifacts. Completion SHA-256 `2c0364fda3fc17cd09419d0a6003e6a3e6d7f1da035b8228c201fd21a6570d6e`.
- The one V6 delivery published stage main/per-trade pages and cumulative snapshot `2da2a0dff4c1890627f78c0556a2d8504ff0f384f77db147da54572367635a52`: 1,775 coordinates, 449,850 trades, and five active compatible stages. Stage, snapshot, and stable-route QA passed at 320, 920, and 720 states respectively; manual desktop/mobile inspection is clean.
- Both independent cost-adjusted total-return leaders moved to the delivered local region: Scenario 1 is +42.9652% and unrestricted is +64.1373%. The current work ends after the total audit and bilingual closure; no next round is prepared.
- Final read-only audit passed: five completed compatible stages total 1,775 unique coordinates, 449,850 trades, and 149 batches. It verified 745 raw batch-artifact records, 90 stage-analysis artifacts, and 21 snapshot artifacts with zero mismatches; no relevant Python process remains and the stable routes point to the current snapshot.

## 2026-08-03 — Dynamic K200M-notional cost contract opened

- User superseded the fixed HKD 300,000 / 3.56-bps shadow-cost assumption for future analysis while preserving existing raw and delivered results.
- The frozen reference derives notional from the K200M multiplier of KRW 50,000 per point and the last real 15-second close of 1,106.70: KRW 55,335,000.
- At the recorded KRW/USD closing reference of 1,446.7, USD 6 round-trip commission is KRW 8,680.2 or 1.568663594470046 bps. With the confirmed 2 bps round-trip slippage, future derived cost is 3.568663594470046 bps.
- Source, tests, bilingual management records, and cumulative derived ranking require rebinding before the next evidence-led plan; raw compute has not been rerun.

## 2026-08-03 — Preserve current delivery; use one shared per-trade entry going forward

- Preserved the current five-stage snapshot `2da2a0dff4c1890627f78c0556a2d8504ff0f384f77db147da54572367635a52`, all raw results, and every historical stage page. No cumulative rebuild, raw rerun, or HTML regeneration was requested or performed.
- Retired the future requirement for a dedicated per-round per-trade HTML page. Each future delivery refreshes only the current cumulative main entry and the one shared cumulative per-trade entry, which both include all completed compatible results.
- Updated the role-neutral operating records: no subagent, separate conversation, or separate compute-versus-HTML thread is mandatory. The existing source/anti-join/root-lock/live-resource/single-cumulative-writer safeguards remain.
- The next backtest remains paused until the user explicitly authorizes it.

## 2026-08-03 — Simplify the next cumulative-main control surface

- Updated the cumulative-main generator source to hide the four summary cards. The only retained summary is the dynamic `全部策略 <coordinateCount>` label near the title.
- Made the speed-window filter span the whole grid row, so its many values no longer share space with the ranking-metric group.
- Preserved the current hash-sealed snapshot and all existing results. A display-only publication needs explicit authorization because replacing its generated HTML would change delivery artifact hashes.

## 2026-08-03 — Standardize project-management bold markup

- Converted all Markdown strong emphasis in the 36 active Dashboard documents to HTML `<strong>...</strong>`. Inline code, fenced code, and glob paths such as `batches/**/trades.csv` retain literal asterisks.
- Added the same rule to the root operating instructions, bilingual entry documents, the project-management skill, its generated templates, and its validator. Future validation reports Markdown `**...**` bold syntax in managed documents.
- Rebuilt the offline Dashboard from the converted sources. This documentation-only change does not alter backtest source, raw results, analysis output, trading HTML, or the current cumulative snapshot.

## 2026-08-03 — User-authorized cumulative-main presentation publication

- Published the updated `index.html` inside snapshot `2da2a0dff4c1890627f78c0556a2d8504ff0f384f77db147da54572367635a52` after the user explicitly authorized a UI-only overwrite.
- The publication uses the existing `analysis_data.js`; raw trades, derived returns, rankings, and every non-main-entry artifact remain unchanged. The affected snapshot artifact hashes were reconciled in its analysis and completion manifests.
- The initial presentation page exposed a JavaScript scope defect: `renderContract()` referenced the cost-model variable while it was local to `render()`, so page initialization stopped before controls and tables populated. V11 moves the variable to shared page scope and adds a regression assertion; the corrected page now initializes the all-strategy count, every control group, the ranking table, and the contract table from the preserved data file.

## 2026-08-03 — Borderless per-trade market-range rectangles

- Updated the shared per-trade template so every x-range market rectangle uses semantic translucent fill and a zero-width transparent border. This covers filtered-market, quiet-activity, signal, baseline, rebound-window, frozen, and hover-selected ranges.
- Replaced outline-only hover rectangles with translucent semantic fill, retaining the intended color-state change without a visible frame. Point circles, trade links, price guides, and non-range marks are unchanged.
- Published only the shared cumulative per-trade `index.html` and reconciled its resource, trade-review, analysis, and completion manifests. Existing trade chunks, raw trades, return metrics, rankings, and cumulative data remain byte-identical.

## 2026-08-03 — Project HTML documentation classification and readability repair

- Updated the current milestone against the live V4.4 source identity, five-stage cumulative snapshot, and delivered interpretation. The original three rounds and both continuation rounds are complete; no backtest is running and no next-round plan is authorized.
- Moved execution, delivery, anti-join, and preservation rules out of the future-directions page while retaining them in their existing constraint and task authorities. No operating or delivery contract changed.
- Replaced the former rule list with testable research directions: parameter-ridge stability around the leaders, time-slice and out-of-sample validation, gap dependence, cost and trading-frequency robustness, cross-instrument transfer, and alternative method definitions.
- Rewrote the affected Chinese pages in a technical project-document register, then synchronized the English sources. Parameters, metrics, paths, hashes, evidence boundaries, and historical provenance remain intact; V4.3 appears only in clearly historical or provenance contexts.

## 2026-08-04 — Superseded compact current-UI review package

- Created `D:\Code\backtest-release\Backtest_V4.4_current_UI_review_20260804.zip` without recomputing raw trades or returns. Later inspection found that it mixed the older 1,775-coordinate / 449,850-trade / five-stage cumulative payload with project-management records from a later sixteen-stage timepoint.
- Retain the original ZIP, staging directory, and extraction as recoverable mismatch evidence only. It is superseded and must not be used as a timepoint-consistent review package.

## 2026-08-05 — Corrected timepoint-consistent current-UI review package

- Created `D:\Code\backtest-release\Backtest_V4.4_current_UI_review_20260805_corrected.zip`, bound throughout to snapshot `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7`: 4,704 coordinates, 706,470 trades, and 19 completed compatible stages.
- `CURRENT_REVIEW_STATE.json`, cumulative `analysis_data.js`, both snapshot manifests, and the copied bilingual source-of-truth records all carry that same current identity. Older counts retained in `WORK_PROGRESS` are explicitly dated historical progress.
- The archive contains 29 entries. All 28 non-self records passed ZIP-stream and independent-extraction path, size, and SHA-256 verification with zero mismatches, duplicates, unexpected entries, or forbidden large payloads. ZIP SHA-256 is `7d5ebf34c779c0a4053aa24ca92640d97bafcfed3041d00b31ee18fa0ac3ef82`.
- The compact scope includes the complete current cumulative analysis and the requested 130-trade combination only; the large cumulative trade CSV and the complete 4,704-combination trade-chunk collection remain excluded. Raw results and retained snapshots were not changed.

## 2026-08-04 — Relative-step exploration Rounds 14–15 closed

- Round 14 tested 103 new coordinates with relative timing scales, broad interaction points, module pairs, and stability checks. It closed at 8,490 trades and improved unrestricted cost-adjusted total return from 78.4595% to 79.1437%; Scenario 1 and average return did not improve.
- Round 15 tested 48 new coordinates across broad E/S interaction, the TRW/K ridge, and K stability checks. It closed at 5,651 trades. The new unrestricted leader is E480/BH171/TRW12/K1.26/W6/M4.5/S388: 112 trades, 90.0092% gross return, 82.6033% cost-adjusted total return, 0.5627% cost-adjusted average return, and 15.1770% cost-adjusted maximum drawdown.
- Nearby E336, E576, and E720 values remain strong, so the improvement is a broad in-sample plateau rather than an isolated single point. No parameter is accepted.
- Shared cumulative snapshot `45b9a08396493a53ece45bd62af91070fb6b443539cd2e12ae5aeac5c756faad` now contains 4,499 coordinates, 683,835 trades, and 18 stages. All 4,523 unique declared artifacts match; targeted main/trade desktop/mobile QA passed with zero runtime errors, external requests, or page overflow.
- Cumulative publication was made memory-bounded and faster by releasing stage frames after concat, narrowing a read-only audit subset, indexing trades by `combo_id`, and removing duplicate entry validation. Engine, raw results, ranking formulas, and HTML semantics are unchanged.
## 2026-08-04 — Continuation Round 16 multimetric broad coverage closed

- Built 231 requested candidates across unrestricted, Scenario-1, average-return, low-drawdown, and remote-control objectives. Deduplication plus the exact completed/active/pending anti-join retained 205 new coordinates: 170 broad-jump and 35 module-pair points.
- Raw compute closed at 205 coordinates, 22,635 trades, and 18 batches. The immutable closure reconciled every coordinate, batch manifest, indexed artifact, trade count, and released raw lock.
- Published snapshot `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7` into the existing shared entries: 4,704 coordinates, 706,470 trades, and 19 stages. Artifact audit checked 4,728 unique declared records with zero mismatches; desktop/mobile browser QA passed with zero runtime errors, external requests, replacement characters, or layout overflow.
- The unrestricted and Scenario-1 total-return leaders remain unchanged. Both average-return views now select E112/BH612/TRW24/K1.6/W10/M2.5/S308 at 0.999935% cost-adjusted average return. E150/BH504/TRW24/K1.6/W10/M2.5/S310 adds a nondominated 30.817684% return / 3.089401% drawdown point.
- `parameter_acceptance=none`. No combined score was formed and gap-excluded return remains display-only.

## 2026-08-05 — Timepoint-consistent, source-complete review package

- Confirmed that the first review ZIP mixed a 1,775-coordinate / 449,850-trade / five-stage cumulative payload with later project-management records. The intermediate timepoint-only correction aligned to the current snapshot but still lacked test dependencies. Both packages remain as recoverable superseded evidence.
- Updated two stale generated-HTML assertions to verify the current shared window-filter structure and field-based minute conversion. Engine, runner, templates, raw trades, returns, rankings, and delivered HTML remain unchanged. The complete project suite passes 80 tests with 2 skips.
- Created `D:\Code\backtest-release\Backtest_V4.4_current_UI_review_20260805_source_complete.zip`, bound throughout to snapshot `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7`: 4,704 coordinates, 706,470 trades, and 19 stages.
- The archive preserves the complete V4.4 code, data-preparation, plans, review templates, runtime inputs, dependency declaration, management tree, and minimal local cumulative fixture required by its tests. Its package-root `RUN_TESTS.ps1` passes 80 tests with 2 skips from an independent extraction.
- ZIP SHA-256 is `82ec55e75d25e05eb15cdbc5a79e211f69ed5376d8376c33b9139d5b4789fee8`. All 204 non-self records match in the ZIP stream and independent extraction; duplicate, unexpected, forbidden, cache, and mismatch counts are zero. No raw trade or return was recomputed.

## 2026-08-05 — P0-8 zero-signal inventory and result-authority transition

- Added the positive entry gate: finite `baseline`, `drop`, and `threshold` must each be greater than zero before inclusive `drop >= threshold` is evaluated. Positive equality remains valid.
- The closed read-only scan of snapshot `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7` found 4,404 invalid trades across 4,383 of 4,704 coordinates and all 19 completed active stages. Only 321 coordinates had no such trade.
- Retired the complete 9.17 GB active campaign to `F:\Backtest\Backtest V4.4\results\staging_recoverable\p0_8_zero_signal_retired_20260805\v4_4_cost_adjusted_multiround_20260803`. The move preserved 6,742 files; no matching writer was running and every result lock was acquirable.
- The replacement boundary is a complete 4,704-coordinate rerun. This avoids fabricating partial batch manifests for the 321 unaffected coordinates and ensures that the corrected cumulative ranking has one engine and result semantics.
## 2026-08-05 — Lean cumulative-main presentation published

- Added a stable `main` presentation layer derived from the current immutable cumulative snapshot. It preserves all 4,747 coordinates and routes exact per-trade/scenario links back to that snapshot without rebuilding 713,886 trades or any trade chunk.
- Reduced the loaded main payload from 14,337,950 to 5,289,966 bytes by retaining only 32 row fields consumed by filters, ranking, display, and routing. Changed filter/rank/column-sort work to integer row indexes and removed per-render row-object copies.
- Kept the existing table DOM, visual template, controls, sorting semantics, and columns. Virtualization and sort caching were not added. Chrome file-page QA was unavailable because browser security policy blocked local `file://` navigation; the focused generated-code contract check passed.
## 2026-08-06 — Confirmed low-activity gate implemented

- Added the `confirmed_low_activity_gate` policy. Pending runs remain baseline-eligible and entry-neutral; confirmation retroactively excludes the run from later baselines, cancels unfilled entry orders, and blocks new entries until normal volume returns.
- Regenerated the active K200 preparation manifest and filter artifacts. The source market data and all existing backtest results remain unchanged.
- Changed cumulative compatibility to the accepted V4.4 K200 major ranking lineage. Minor implementation and preparation hashes remain per-stage provenance and no longer split the cumulative page.
- No parameter backtest or cumulative publication was launched. The next authorized V4.4 run will use the new policy and can be published beside existing V4.4 rows.
