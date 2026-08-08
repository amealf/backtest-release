# Current Version — V4.41 Minor Release on the V4.4 Ranking Lineage

## V4.41 formal source release — 2026-08-09

V4.41 is formally published through Scheme A: the GitHub source, a compact Windows source ZIP, its SHA-256 sidecar, its machine audit, and tag `V4.41`. The package includes source, tests, project documents, runtime contracts, scenario catalogs and tools, the current cumulative browser payload, the stable main shell, and one representative per-trade chunk. It excludes complete historical trade ledgers, dependencies, caches, and the remaining per-coordinate chunks.

The final package gate is an independent extraction with 112 passed, 2 skipped, and 0 failed. The two skips explicitly require closed historical local artifacts that are absent from the compact release. A preflight extraction exposed a stale market-selector source hash; the bound hash was corrected before the final package. Strategy execution, completed trades, returns, rankings, and retained result snapshots were not recomputed.

The 2026-08-08 compact review archive remains immutable historical evidence. It is not the formal release package. The 2026-08-08 multi-gigabyte complete handoff remains local and is not published under Scheme A.

## V4.41 market-scenario extension — 2026-08-08

V4.41 now includes a catalog-driven market selector and saved-scenario application layer. This is a research-delivery extension inside the V4.4 major lineage: it does not change strategy semantics, completed trades, or cumulative ranking compatibility. Three current market intervals and the existing three K200 scenarios are registered. Scenario application writes separate ranking pages and keeps the existing stable main and per-trade pages unchanged.

The selector now explains the catalog workflow in the interface, generates the next unused scenario ID and default name, keeps only the name editable, and normalizes drawn selections to full-height time bands. Saving still writes the complete multi-scenario catalog; no strategy or qualification rule changed.

## V4.41 compact external-review package — 2026-08-08

`D:\Code\backtest-release\Backtest_V4.41_source_review_20260808.zip` is retained as the compact artifact reviewed by another GPT. It contains the source and tests captured on 2026-08-08, project management, runtime inputs, the 37,058-coordinate cumulative browser payload, and one representative per-trade chunk. It excludes the 3.5 GB complete ledger handoff and the remaining per-coordinate chunks. That immutable archive still records its observed 4 failed, 105 passed, and 2 skipped result.

The current source resolves those four review findings and now completes the same suite with 109 passed, 2 skipped, and 0 failed. The package script records that current result for its next run. No replacement compact ZIP has been generated, and the 2026-08-08 archive remains historical review evidence.

## V4.41 current handoff package and local cleanup — 2026-08-08

The current portable handoff is `D:\Code\backtest-release\Backtest_V4.41_current_complete_20260808.zip`; its adjacent `.sha256` sidecar is the whole-archive hash authority. It contains source, runtime inputs, current project documents, retained reports, and 4,467 immutable raw or available derived transaction ledgers from 110 completed V4.4-compatible stages. The package retains V4.41 release identity and V4.4 strategy/ranking compatibility.

The final package passed manifest identity, ZIP name/size, duplicate/forbidden-path, and whole-archive SHA-256 checks. Repeated per-entry ZIP hashing and a duplicate extraction tree are no longer part of normal verification. Nine canonical 2026-08-03 reports now live in the project handoff tree, so `.omo` is no longer required.

After the final package passed, 22 obsolete package/sidecar files and two local directories were moved to the Windows Recycle Bin, totaling 39.972 GiB. Earlier in the same cleanup, `.omo` (68.72 MiB), old caches, Crashpad material, dashboard cache, and four obsolete QA images were also moved to the Recycle Bin. Current results, V4.2 evidence, the two V2 download dependencies, shared tools, provenance, and management archives remain retained.

## V4.41 release identity — 2026-08-08

V4.41 is the active release and presentation label. It is a minor release: the strategy major version, result contract, parameter identities, and cumulative ranking boundary remain V4.4. Completed V4.4 results remain active and authoritative within their original evidence boundaries and continue to enter the same cumulative ranking when their ranking lineage and result contracts match. `RELEASE.json` records this split identity; no historical result is archived or excluded by this release.

## K200 paired per-trade review — 2026-08-08

K200 training and later-period test reviews now form a bidirectional exact-coordinate pair. The toolbar names the destination (`显示测试集` or `显示训练集`) and carries the current `combo_id` plus the destination research-contract identity. The paired result opens as an overlay inside the current page and can be closed with its `隐藏…` button or Escape. Closing unloads the iframe so the paired chart does not retain another renderer. The established main and per-trade URLs remain valid.

The paired button is centered and uses the same hover state as adjacent toolbar buttons. The embedded page omits its `当前参数` card, while the outer page is hidden until the overlay closes. The trade selector keeps only the transaction dropdown. Entry/drop audit values moved from the plot into compact color-preserving boxes below the reason heading, and the legend moved below the candlestick chart. The browser tab omits the version number and uses a blue `Z` favicon. This is a presentation and routing update; trade chunks, OHLC, fills, metrics, ranking, and strategy behavior are unchanged.

## Instrument- and interval-neutral evaluation packages — 2026-08-08

V4.4 now stores published backtest evidence by instrument and exact evaluated interval under `results\evaluation_packages`. Training, test, transfer, holdout, and descriptive meanings are assigned by `EXPERIMENT.md` and `comparison_plan.json`; they do not alter a package directory name. The current K200/K200/SI evidence is registered as three completed date packages, each with a manifest, neutral parameter summary, browser projection, immutable trade record, and per-trade entry.

`results\evaluation_comparison` loads the three package-owned browser projections and reconstructs the retained 350-coordinate comparison by exact `combo_id`. The reconstructed rows and values are exact, and the retained page shell is unchanged outside its script-source boundary. Browser inspection found identical visible text, title, first row, and row count with no errors. The stable cross-instrument URL now targets this reader. The cumulative main and retained comparison remain unchanged; later approved per-trade shell changes do not alter package data. The same framework accepts any ready instrument profile and any exact evaluation interval.

## Current K200 optimal-parameter one-month replay — 2026-08-07

Four workers evaluated a corrected frozen set of 100 training-period cost-positive candidates over the later month. The candidate set contains six retained headline controls and 94 exact coordinates not previously evaluated on that interval. The interval itself was already used by the earlier temporal campaign, so this run is parameter-level new evidence and post-hoc at the period level.

Only 29 candidates remain cost-positive, 28 are positive with at least ten trades, and eight have positive non-gap return. Median later return is -0.671%, median trade count is 84.5, and train/later rank Spearman is -0.346. The best total-return coordinate earns +9.175% but has -8.243% non-gap return. The strongest at-least-ten-trade non-gap coordinate is E112/BH612/TRW30/K1.3/W10/M2.5/S308 at +4.693% total and +4.573% non-gap return with ten trades; its Top-2 positive-return share is 72.2%.

The first completed attempt remains retained as historical, invalid-for-decisions evidence because its candidate queue admitted twelve cost-negative training rows. The `_v2_` result replaces it. No HTML was published and no parameter is accepted.

## K200 train/test/SI unified migration — 2026-08-07

The stable cross-instrument entry now ranks 350 candidates with K200（训）, K200（测）, and SI return columns. It retains 250 completed SI results, freezes 100 new candidates from closed K200 temporal evidence before reading their SI outcomes, evaluates those candidates on SI with four workers, and replays all 350 on the K200 test interval. The final SI population contains 48,022 trades; the K200 test replay contains 34,248.

The visible rank heading is now `排名`; rank buttons display only `#N`. Rank opens SI per-trade evidence, while the K200（训）, K200（测）, and SI total-return cells use the same blue-button component and open their corresponding per-trade analyses in new tabs. K200（测） has a dedicated 350-candidate, 34,248-trade review; K200（训） routes to the existing cumulative K200 review.

K200 test is positive for 199 candidates, SI for 275, and all three returns for 149. Within the new 100, the corresponding counts are 72, 59, and 43. The E320/BH240/TRW12/K1.25/W6, M4.25–4.75, S340–370 neighborhood is continuous and positive across all three samples, but its K200-test result remains gap-dependent. Static universal parameters remain unsupported; short-window re-estimation followed by a frozen forward test remains the more credible direction. Publication reused 250 existing SI per-trade chunks and generated 100. No parameter is accepted.

## K200 training-to-subsequent-market temporal migration — 2026-08-07

Four workers evaluated 400 frozen candidates in each of four sequential market slices after the completed training interval. Later freezes used only training evidence and already closed earlier slices; the fourth slice remained unseen until its candidate set was frozen. The same 400 final candidates were replayed over the complete subsequent interval for descriptive reporting only. Intermediate slices generated no HTML.

R1/R2/R3/R4 contain 296/26/383/25 cost-positive coordinates. Only two of 218 candidates observed in every slice remain positive in all four, and both are low-frequency concentrated results. The training/full-test return-rank Spearman is -0.262; the training leader returns -1.567% over the full test with 14.902% drawdown. The final interpretation is `no_static_general_parameter_found`; no parameter is accepted.

## Incremental K200 data and operational records — 2026-08-07

The active K200 source now contains 233,368 session-filled 15-second rows through `2026-08-07T03:21:45+09:00`. The latest increment came from 1,065,932 retained IBKR `TRADES` ticks and produced 34,168 cleaned 15-second rows; the repository source SHA-256 is `9760d367a109777c4789ce45d982a6c0708bacddad8f549450ed94f81ad5c405`. Prepared inputs, the K200 profile, policy attestation, runtime manifest, and source manifest now bind the extended source.

Every retained K200 acquisition directory now has a local `README.md` describing update history, the volume-based main-contract rule, the exact unadjusted 016M/016U split, lineage, and merge behavior. The retained SI directory now has `D:\Code\data\ibkr\SImain\README.md`, recording SIH6 selection, exact intervals, unadjusted policy, merge rules, and update history. The same README contract is mandatory for future instruments and downloads. `project_management\03_active_work\BACKTEST_MANAGEMENT.en.md` and its Chinese mirror are the separate mandatory pre-run log for instrument, exact data file, evaluated start, and evaluated end. No historical backtest or HTML was regenerated by this data update.

## On-demand candlestick interval statistics — 2026-08-07

The cumulative per-trade page adds `区间统计` immediately before the dark-theme control and removes the former view-mode explanation sentence. When enabled, a horizontal drag selects an inclusive candlestick interval and opens a compact flat three-column panel in the right detail column above trade reasons. It contains start/end time, first open, highest high, lowest low, final close, candlestick count, price change, amplitude, and maximum high-to-later-low drawdown.

The feature adds no startup computation or data request. It performs one linear scan over the selected interval after `plotly_selected`; native selection state is then cleared and the pale interval is redrawn as an ordinary Plotly shape. This keeps normal opacity outside the range and makes the selected rectangle persistent. Closing the panel clears the shape and returns the chart to zoom dragging. Changing parameters or redrawing the chart also clears the temporary panel. The `紫色` and `橙色` suffixes are removed from baseline and threshold annotations.

The page also adds mutually exclusive `持仓检测`. A short visible-candle press identifies whether the position remains open after that bar and derives the contemporaneous rebound and speed checks, while movement beyond five pixels remains available for Plotly zoom dragging. A raw candle uses its exact source bar; an aggregated candle uses the final source bar represented by its close. The panel reports active low, effective max-W baseline interval/value, multiplier, threshold, current rebound relation, and distance to the S-window exit; the selected close receives a blue point and the baseline interval uses a darker borderless blue highlight. Both inspection panels appear in the right detail column and collapse the parameter drawer when activated. Candlestick endpoint caps are removed with `whiskerwidth=0`. The completed snapshot shell can be refreshed independently from its immutable trade chunks and payloads.

## Six-hour K200 leap/grid cycle and streamed cumulative delivery — 2026-08-06

The AI-led cycle closed 89 automated rounds with four workers: 60 leap rounds and 29 adaptive-grid rounds. The current cumulative snapshot is `eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5`, containing 37,058 unique coordinates, 11,749,606 trades, and 109 compatible stages. It reused 5,320 unchanged trade chunks and generated 31,738. All four headline leaders remain unchanged from the prior 5,320-coordinate snapshot; `parameter_acceptance=none`.

The cumulative publisher now discovers nested generated plans recursively, loads and processes one stage at a time, streams the 22.13-GB union trade CSV, vectorizes cost metrics and native trade conversion, and uses four worker processes for missing trade chunks. Final publisher memory stayed below 2 GB instead of the former 14.4-GB peak. SI migration was not started because later leap rounds continued finding promising nonadjacent anchors, so K200 global closure was not established.

## K200 cumulative-table presentation — 2026-08-06

The stable K200 entry now redirects to its main document, removing the repeated top navigation and persistent iframe. The page title is `V4.4 K200回测结果排序`, common blue accents are one step darker, and the full-width main table occupies nearly one viewport with a sticky header. Pagination renders 500 rows at a time while filtering and sorting continue over the complete eligible set.

Return columns are labeled `总收益`, `笔均`, and `回撤` in both return modes; the rank column is labeled `排名`. Rank links display only `#N` while retaining a 90-pixel blue button. The visible gap-dependence return audit, `segment_end`, waited-entry count, maximum wait, and research-contract section are absent. `往返成本 bps` and `跨 gap 笔数` are the final two columns. Result values, filters, rank rules, raw fields, and immutable snapshots are unchanged.

The trade-count control is labeled `交易数`. It retains `不限`, `至少 10 笔`, and `至少 20 笔`, removes the 100- and 150-trade presets, and adds strict editable lower and upper bounds labeled `大于 x 笔` and `小于 x 笔`. Both custom bounds may be active together. Main-table column headings are centered; body-cell alignment remains unchanged.

## Unified 250-candidate migration ranking — 2026-08-06

The stable cross-instrument entry now ranks 250 compatible K200-to-SImain candidates together. Compatibility is defined by the same target instrument, target sample, cost model, strategy semantics, and result schema; experiment or migration identity is retained as provenance rather than used as a ranking partition. The publication contains 24,194 SImain trades, reuses 247 completed trade chunks, and adds three.

The new fixed-coordinate continuation tested K1.5/K1.6/K1.75. Target trades decline to 62/57/51 and cost-adjusted return declines to 23.9157%/16.7475%/20.1076%. K1.4 remains the local target peak at 35.6004% with 67 trades. Classification is `not_improved_target_peak_at_k1p4`; `parameter_acceptance=none`.

## Dual-purpose K200 exploration delivery — 2026-08-05

Continuation Round 15 formalizes the two continuing exploration purposes. The reviewed plan contains 192 deterministic broad-coverage coordinates and 84 one-parameter refinements around the unrestricted, Scenario-1, average-return, and low-drawdown anchors. Four workers closed 276 coordinates, 54,314 trades, and 35 batches.

The average-return branch improved. E96 is the new minimum-10 and minimum-20 average-trade leader with 25 trades, 35.2290% cost-adjusted total return, 1.23914% average trade, and 3.0190% drawdown. E128 is a balanced neighboring improvement with 29 trades, 38.9271% return, 1.16460% average trade, and 4.21795% drawdown. The unrestricted and Scenario-1 leaders remain unchanged. Broad coverage found no robust new primary leader.

Final incremental publication created snapshot `5b4e11b4c137028dc0a33d792a47800c8d792f6125e2cc8d2f5796ec6ef4fa94` with 5,320 coordinates, 797,020 trades, and sixteen stages. It reused 5,044 unchanged trade chunks and generated 276. `parameter_acceptance=none`.

## Large K200 multiblock exploration and incremental cumulative delivery — 2026-08-05

Continuation Round 14 added 294 exact new coordinates and 28,577 trades through a reviewed 37-batch, four-worker plan. The final cumulative snapshot is `db85efb36f3de1c1f8255c6108fb365ad9f3d337f77a8d37a0e0ae41982e5699`, with 5,044 coordinates, 742,706 trades, and fifteen compatible positive-entry stages. The cumulative publisher now reuses deterministic unchanged per-trade chunks from the current snapshot; this delivery reused 4,747 and generated 297.

The unrestricted total-return and Scenario-1 leaders are unchanged. Both minimum-trade average-return views now select E112/BH612/TRW24/K1.6/W16/M2/S308 at 29 trades, 38.4486% cost-adjusted total return, 1.15253% average trade, and 4.23380% drawdown. The change mainly reflects exit-time differences among retained entries and remains exposed to synthetic-signal and gap trades. Round classification is `improved`; `parameter_acceptance=none`.

## Human project guide landing page — 2026-08-05

The project-management Dashboard now opens on a human-facing guide. `Project guide` is the first item in the left navigation and uses the same document header, metadata, content width, typography, outline, language control, and responsive page shell as every managed document. Its body contains two workflows: the four campaign modes with their shared backtest lifecycle, and the causal lifecycle of one trade from completed 15-second data through entry, exit, and transaction evidence. A generated directory below them links every managed document. This virtual guide is not a manifest document, does not change the AI reading matrix, and carries no strategy or evidence authority.

## Incremental 247-candidate stricter-entry migration — 2026-08-05

The current stable cross-instrument entry contains 247 candidates: the completed 244-candidate parent plus three frozen K200 K-expansion coordinates. The three SImain results are all cost-positive. K1.2 has 85 trades and 16.4678% cost-adjusted return; K1.3 has 73 trades, 33.9182%, and 11.6094% drawdown; K1.4 has 67 trades, 35.6004%, and 11.9132% drawdown. The K1.1 parent anchor has 86 trades, 22.0575% return, and 14.6209% drawdown. Thus K1.3 and K1.4 reduce entries while improving target return and drawdown, while the same K direction reduces K200 return. Classification is `improved_on_target_mixed_across_instruments`; `parameter_acceptance=none`.

Migration HTML now uses the fixed `build_v4_4_cross_instrument_comparison.py build --run-id <run_id>` path. The 247-candidate publication reused 244 completed target trade chunks through same-volume hard links and generated only three new chunks. The ranking index, catalog, summaries, and manifests were refreshed without regenerating unchanged per-candidate trade evidence.

## Plan-driven combined K200 to SImain delivery — 2026-08-05

The current cross-instrument entry combines the user-linked 180-candidate result with the repaired-source 64-candidate result. The sets have zero overlapping coordinates, so the current comparison contains 244 candidates, 25,129 K200 trades, and 23,799 SImain trades. It contains 210 SImain cost-positive candidates, 148 stable candidates, and 57 isolated positive candidates; source/target cost-return rank Spearman is 0.060149.

The page reads instrument display names and sample ranges from `research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_migration_k200_to_simain_20260805.json`. The migration-batch column is absent. K200 total return is immediately before SImain total return, and clicking one sort header three times cycles ascending, descending, and restored default target ranking. The source main entry opens the cross-instrument page in a new tab; the comparison page exposes a new-tab source main entry. Final migration delivery includes source main, source per-trade, combined ranking, target per-trade, and the migration report.

## New-rules K200 one-axis series — 2026-08-05

Ten four-worker rounds added thirty coordinates around the repaired unrestricted anchor E480/BH171/TRW12/K1.26/W6/M4.5/S388. Each round changed one parameter and tested three new points. Intermediate rounds closed raw evidence and compact handoffs without HTML; one final publication produced snapshot `0126cd77b436aef1434e7072bac0d6dfa15b3d2ad4dc2cf1b2fafe936ee1e626` with 4,747 coordinates, 713,886 trades, and thirteen compatible stages.

BH expansion/contraction, TRW expansion/contraction, K expansion/contraction, both sides of M4.5, and S expansion are `not_improved`. W7 raises cost-adjusted total return from 82.4352% to 82.4664%, but the 0.0312-percentage-point gain is isolated: W8 and W9 are weaker, while W7 has lower average trade, slightly higher drawdown, and lower gap-excluded return than W6. The W direction is `mixed`; `parameter_acceptance` remains `none`.

## Fixed-parameter E-window sensitivity — 2026-08-05

Thirteen new coordinates hold BH=171, TRW=11, K=1.4, W=6, M=4.5, and S=388 fixed while reducing E from the E=320 anchor through 304, 256, 192, 160, 136, 112, 96, 80, 64, 48, 32, 24, and 16. They are published in cumulative snapshot `b077548e654277738e1d953ce7bea01eb184a0e223f064c7728dbc2de4d1a561` and in the shared per-trade entry.

The evidence does not support the claim that E=320 is too large. Cost-adjusted total return falls from 78.2952% at E=320 to 75.7622% at E=304, 74.2774% at E=256, 62.0632% at E=192, 60.6744% at E=80, and 7.8332% at E=16. Smaller E can reduce trade count and maximum drawdown—for example E=64 records 108 trades, 62.3971% return, and 12.4072% drawdown versus E=320 at 114 trades, 78.2952% return, and 15.1499% drawdown—but the return sacrifice is material. `parameter_acceptance` remains `none`.

## Instrument-neutral research interface — 2026-08-05

The V4.4 strategy semantics are separated from instrument profiles and campaign intent. K200 remains the ready default and its current repaired cumulative evidence contains 4,747 coordinates. SImain and NQ profiles remain incomplete templates for fresh campaigns; the retained SIH6 evaluation contract remains valid for the completed exact-transfer validation. Official campaign modes are `transfer_exact`, `target_local_refinement`, `continuation_search`, and `fresh_search`.

Schema-v5 plans bind a campaign manifest, instrument profile, ranking lineage, and optional scenario policy. The accepted K200 V4.4 ranking lineage now spans minor implementation and preparation corrections; their hashes remain stage provenance. A V4.5 update creates a new cumulative partition. Other instruments use separate rank lineages and cross-instrument comparison.

The active low-activity policy is `confirmed_low_activity_gate`. Pending low-volume runs have no effect. Confirmation at 120 consecutive 15-second atoms retroactively excludes the run from later BH/TRW baselines, cancels unfilled entry orders, and blocks new entries until the first normal-volume atom. Existing positions retain normal exits. Prepared fields are `baseline_excluded_from` and `confirmed_low_activity_active`.

## Cross-instrument comparison — 2026-08-05

A separate current cross-instrument page validates 64 candidates frozen from the repaired 4,747-coordinate K200 snapshot on SImain SIH6 for 2026-01-29 through 2026-02-23. Source selection excludes the 266 historical transfers and current champions, then applies the source top-20%, minimum-10-trade, W/M/S-family, and within-family strictness-Pareto rules. It uses the fixed 3.57-bps research cost, exposes target/source/transfer metrics without a combined score, and keeps any SImain full-grid diagnostic separate.

The page retains four file-backed scope selectors plus global and composable field filters. A two-option return-view control switches ranking and displayed K200/SImain total return, median trade, maximum drawdown, and win rate between fee/slippage-adjusted and gross modes; adjusted mode is default. Its first column mirrors the K200 rank component with blue `查看 #N` buttons, and every frozen candidate opens its dedicated SImain per-trade review in a new tab. The 64 candidates create 6,755 SImain trades; 58 are cost-positive, 29 occupy stable positive regions, and source-versus-target return-rank Spearman is -0.45728. The historical 266-candidate comparison remains preserved under its run directory.

## Status

V4.4 is an independent research version with repaired execution semantics and a cost-aware derived ranking layer. It is not a final strategy release and no parameter has been accepted.

## Current behavior

- A prior rebound trigger exits at the real bar open when `open >= trigger`; otherwise it exits at the trigger when `high >= trigger`. Equality exits.
- A strict-new-low bar uses only rebound baselines formed by earlier completed bars. When that same real bar confirms the rebound with its close, the fill is its close.
- A W source window starts no earlier than H, the continuous-segment start, or `end-W+1`.
- W accepts the available 1..W prefix. A shorter early prefix may remain the monotonic maximum for the position; no full-W or minimum-ratio gate applies.
- Each candidate is `w_open_to_end_low_drop = open[start] - low[end]`; it is not an internal maximum ordered decline.
- An open position remaining at the declared sample end exits at the sample-end bar close; no later price is read.
- `all_window` remains the default. Under `exclude_marked`, a TR atom enters the baseline only when its causal `baseline_available_from` time has arrived.
- A pending entry retains its signal and fills at the first real-trade open within the 120-bar continuity boundary without recross or structural cancellation.
- The cumulative method label is `滚动 TR 总和均值`, matching the implemented baseline: sum every overlapping TRW group inside BH, then average those window sums. E is the fully observed high-search window, while the measured net drop runs only from strict H to the later signal low. BH ends at and includes the complete 15-second TR atom containing H. The implementation is causal—baseline atoms are at or before H, H must predate the signal bar, and batch results match stepwise-prefix delivery—but it shares the H boundary bar rather than implementing a strict H-minus-one separation.
- Derived analysis offers gross and cost-adjusted modes; switching changes both rank order and return display. Delivered results preserve their historical cost contracts. Future campaigns bind a price-times-point-value notional model plus commission, slippage, and any FX conversion. The cumulative builder retains each stage's cost model; raw fills and raw returns remain unchanged.
- The cumulative main-page generator exposes a keyboard-accessible, icon-only disclosure toggle on the `筛选与排序` panel. A fixed 40-by-36-pixel button contains a 20-pixel SVG chevron: up while expanded and down while collapsed. Collapsing preserves every selected filter and sort state while reducing the panel to one row. Layout and paint containment on the ranking table prevents the large table from being repainted during the height change. The user-selected snapshot `e4a20d1d5bcb8974f4341a3647e2e246c3c1ab855d66ce4b3d7d4998d7fb3d44` carries this UI-only publication; the current stable snapshot remains byte-preserved.
- The stable cumulative main loads a lean summary payload under `results\all_completed_union_analysis\main` while its exact per-trade and scenario links continue routing into the immutable current snapshot. Each main-table row contains only the 32 fields used by controls, ranking, display, and routing; the current 5,044-row payload is 5,621,151 bytes. Filtering, metric ranking, and column sorting operate on integer row indexes without copying row objects. Table virtualization and sort-result caching remain intentionally absent.
- Entry Reason step 2 uses hollow colored-outline ellipses for its high/entry point hover highlight. The collapsed per-trade Parameters tab sits six pixels above its prior desktop position; mobile drawer behavior is unchanged.
- Candlestick line guides use solid strokes. Every market-range rectangle—including filtered-market, signal, baseline, rebound, frozen, and quiet-activity ranges—uses semantic translucent fill with no border. The chart omits the long theoretical-line/actual-fill annotation and the drop, ratio, entry-baseline, entry-threshold, W-baseline, and active-low value boxes. Those six bordered, semantically colored values now appear under Entry Reason; detailed side-panel exit reasoning remains.
- ZIP handoffs contain the current code, project-management tree, canonical analysis reports, hash-bound 15-second OHLC, and per-stage raw/derived transaction CSVs. Package `D:\\Code\\backtest-release\\Backtest_V4.4_with_trade_records_20260803_final.zip` remains audited historical `SOURCE_FINAL_V4` evidence after the QA-only V5 source change; it is not a current-V5 source package. Other result payloads remain excluded.
- A formal source-review ZIP preserves the project-relative code, data-preparation, plan, review-template, runtime-input, dependency, and local cumulative-fixture layout. It includes one package-root test runner and must pass it from an independent extraction. Compact UI-only archives may omit these dependencies only when explicitly labeled non-executable.
- All current and future V4.4 result bytes are stored under `F:\Backtest\Backtest V4.4\results`. The historical logical root `D:\Code\backtest-release\Backtest V4.4\results` remains available as a directory junction, so existing plans, manifests, main HTML, and shared per-trade HTML retain their paths without regeneration.

## Temporary rerun

The user-selected coordinate E120/BH360/TRW6/K0.75/W1/M0.25/S480 completed with 3,882 trades. The reported 2026-06-19 11:07 trade now records a theoretical rebound line of 1514.825 and an actual same-bar close fill of 1514.850.

## Historical pre-repair exploration context

The retired pre-repair lineage contained nineteen completed stages, 4,704 unique coordinates, and 706,470 trades in snapshot `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7`. It remains historical review evidence; the current positive-entry lineage is described at the top of this document.

At fixed 3.57-bps research cost, the unrestricted leader is Round-8 E320/BH171/TRW11/K1.4/W6/M4.5/S388 with 114 trades and +78.4568% compounded return. Its median trade is -0.1789%, 39.3% of total return disappears without the best two trades, and non-gap compounded return is only +0.4662%, so the improvement remains gap-sensitive. The Scenario-1 leader is Round-8 E320/BH240/TRW22/K1.0/W6/M4.5/S330 with 89 trades and +55.9743%; the gain over the prior leader is only 0.0504 percentage points and remains gap-sensitive. Both average-return views now select Round-10 E160/BH720/TRW24/K1.6/W8/M3/S320 with 33 trades and +0.8458% average fixed-cost return.

Rounds 11–13 reopened the average-return branch with bounded broad controls, local surface refinement, and one-bar speed confirmation. The average-return leader remains E160/BH720/TRW24/K1.6/W10/M2.5/S308: 33 trades, +0.9464% cost-adjusted average return, +35.3717% cost-adjusted total return, and 4.54% maximum drawdown.

Rounds 14–15 applied the relative-resolution contract. Round 14 improved unrestricted cost-adjusted total return to 79.1437%. Round 15 then tested broad E/S interaction, the TRW/K ridge, and K stability, producing the current unrestricted leader E480/BH171/TRW12/K1.26/W6/M4.5/S388: 112 trades, 90.0092% gross return, 82.6033% cost-adjusted total return, 0.5627% cost-adjusted average return, and 15.1770% cost-adjusted maximum drawdown. Strong nearby E336/E576/E720 results indicate a broad plateau. Scenario 1 and the average-return leader did not improve in these two rounds.

Round 16 used 205 sparse multimetric coordinates after exact anti-join: 170 large-span points and 35 paired-module points. It did not exceed the unrestricted or Scenario-1 total-return incumbents. Both average-return views now select E112/BH612/TRW24/K1.6/W10/M2.5/S308: 30 trades, 0.999935% cost-adjusted average return, 33.850987% cost-adjusted total return, and 4.823612% maximum drawdown. It also added the nondominated low-drawdown point E150/BH504/TRW24/K1.6/W10/M2.5/S310: 36 trades, 30.817684% cost-adjusted total return, and 3.089401% maximum drawdown.

Existing stage pages remain historical evidence. Intermediate rounds close immutable raw evidence and compact summaries without rebuilding the shared cumulative HTML. After the exploration series ends, one explicit final publication refreshes the shared cumulative main and per-trade entries with every compatible old and new coordinate; earlier publication requires an explicit user request. Evidence analysis may continue after immutable raw closure under the closed-source, exact anti-join, separate-root/lock, and live-resource gates. `parameter_acceptance` remains `none`; no later round is prepared automatically.

## Current P0-8 repair identity

Entry qualification rejects zero or negative baseline, drop, or threshold before the inclusive threshold comparison. The prior 19-stage K200 campaign and snapshot `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7` remain recoverable historical evidence. The current fifteen-stage positive-entry lineage is snapshot `db85efb36f3de1c1f8255c6108fb365ad9f3d337f77a8d37a0e0ae41982e5699`, containing 5,044 coordinates and 742,706 trades.
