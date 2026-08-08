# Change Reasons

## 2026-08-09 — Close external-review reporting and identity gaps

<strong>Reason:</strong> Valid zero-trade coordinates could complete numerical evaluation but fail during four-process HTML publication. Compact browser data also dropped the target instrument identity, and two presentation tests depended on a local snapshot path or obsolete literal UI text.

<strong>Prior behavior:</strong> The process worker assumed trade columns existed even for a completely empty DataFrame. `main_summary_payload()` omitted the stage instrument profile entirely. Cross-instrument HTML tests read current snapshot CSS from the result drive and asserted fixed K200/SImain labels; presentation assertions depended on CSS property order and chart annotations removed by later accepted layout changes.

<strong>Updated behavior:</strong> A zero-trade coordinate writes an empty chunk with zero wait statistics. Compact browser data retains only a minimal `instrumentSummary`. Cross-instrument tests inject CSS and check dynamic labels, while presentation tests check current semantic behavior. The package script records the current full-suite result.

<strong>Evidence impact:</strong> No strategy state, fill, cost, trade, return, ranking, completed result, or existing published HTML changed. The source manifest binds the repaired report generators and tests; compatible V4.4 results remain eligible for the same cumulative ranking.

<strong>Validity boundary:</strong> These changes affect future report generation, compact analysis metadata, tests, and future source-review packages. The retained 2026-08-08 compact ZIP remains immutable historical review evidence.

## 2026-08-09 — Make scenario drafts self-managing

- Prior behavior: the selector exposed manual scenario ID and name fields and labeled the full-catalog import as opening a scenario file, leaving the multi-scenario storage model implicit. Drawn ranges also looked like two-axis rectangles.
- Updated behavior: the selector imports the complete `scenario_catalog.json`, generates the next unused ID and default name, keeps only the name editable, advances to the next draft after saving, and renders selection as a full-height time band.
- Reason: scenario creation should not require bookkeeping, while the stored catalog and time endpoints remain explicit and reusable.
- Evidence boundary: this is a selector-only presentation and authoring change; strategy, qualification, immutable trade records, and result packages are unchanged.

## 2026-08-08 — Separate market identity, saved intuition, and result application

- Prior behavior: the selector had one bundled K200 source and could only copy or export ranges. The retained scenarios lived in one strategy plan, and producing another scenario ranking required code-directed publication work.
- Updated behavior: market intervals live in a reusable JSON catalog; named scenarios live in a separate JSON catalog; the selector can save its current one-or-many-range selection; one script applies a scenario to the matching completed evaluation package and generates the accepted main interface.
- Reason: the researcher needs to test market intuitions repeatedly across instruments and dates without changing strategy code or rebuilding the storage model.
- Validity boundary: one scenario belongs to one evaluation package. Cross-evaluation comparisons remain the responsibility of the generic evaluation comparison layer.

## 2026-08-08 — Replace exhaustive package revalidation with bounded integrity checks

The retained transaction population makes the current package several gigabytes. Re-extracting the archive and hashing every file again duplicated already completed source-to-staging hashes, created a second multi-gigabyte tree, and exceeded the execution window. The package now preserves the same evidence boundary while checking the manifest record set, archive index, forbidden/duplicate paths, and one whole-archive SHA-256. Missing optional derived ledgers are recorded instead of recomputed. This changes packaging cost only; strategy results and retained trade bytes are unchanged.

## 2026-08-08 — V4.41 release identity

The current accumulated work needs a visible release name without splitting comparable backtest evidence. Separating the V4.41 presentation release from the V4.4 strategy/ranking identity gives the release a precise name while preserving all compatible completed results in one ranking.

## 2026-08-08 — Separate trade explanation metadata from the price plot

<strong>Reason:</strong> The training and test charts lacked a one-action way to inspect the same coordinate across time intervals. Repeated picker metadata and in-chart audit boxes also consumed space needed for price evidence.

<strong>Prior behavior:</strong> Users returned to the comparison table to change intervals. The picker showed a heading/count plus a repeated summary line, audit values covered the upper-left price area, the legend occupied the chart top, and the browser tab used a versioned title with an empty icon.

<strong>Updated behavior:</strong> Training and test reviews link bidirectionally by exact `combo_id`. The dropdown stands alone, audit values wrap below the reason heading, the legend sits below the x-axis, and the tab uses a version-free title with a blue `Z`.

<strong>Evidence impact:</strong> Only HTML shells and their presentation manifests changed. Existing market data, process payloads, catalogs, trade chunks, fills, returns, metrics, and rankings remain authoritative and unchanged.

<strong>Validity boundary:</strong> The pair is available for the current K200 training and later-period test reviews. Coordinates absent from the destination catalog are reported as absent; no substitute coordinate is loaded. SI has no paired K200-time role and therefore receives no switch.

## 2026-08-08 — Replace role-named result storage with date-based evaluation packages

<strong>Reason:</strong> Training, test, migration, holdout, and later research roles can change while the evaluated bytes and interval remain the same. Result locations therefore need stable factual identities that also work for any instrument and time range.

<strong>Prior behavior:</strong> Current evidence was distributed across the K200 union directory, a K200 later-period campaign, an SI migration run, and a comparison run whose data were embedded in one generated JavaScript file. The comparison entry depended on those role-specific locations.

<strong>Updated behavior:</strong> Each current interval has an independent package identified by instrument plus exact start/end timestamps. Each package records data lineage, experiment meaning, parameter metrics, browser summary, immutable trade records, and a per-trade compatibility route. A generic comparison plan lists package roles and loads their browser summaries by exact `combo_id`.

<strong>Evidence impact:</strong> This is a storage and reading-boundary change only. Existing main and per-trade artifacts remain byte-identical; the new comparison reconstructs all 350 rows and the retained browser-visible content exactly. No backtest, fill, cost, metric, or candidate selection was recomputed.

<strong>Validity boundary:</strong> The current adapter registers retained K200 and SImain evidence. A future instrument still requires an executable profile and a completed interval result before it can be packaged. The proposed median-selection workflow remains unimplemented.

## 2026-08-07 — Require cost-positive training eligibility for the current-optimal replay

<strong>Reason:</strong> The requested population is current K200 optimal parameters. A gross non-gap ranking can favor extremely high-frequency coordinates whose cost-adjusted training return is negative.

<strong>Prior behavior:</strong> The first frozen queue admitted twelve cost-negative training coordinates because gross non-gap return had no cost-positive eligibility gate. The completed output therefore mixed deliberately profitable training parameters with unsuitable stress cases.

<strong>Updated behavior:</strong> The corrected `_v2_` freeze restricts every queue to positive cost-adjusted training return, retains six previously evaluated headline controls, and adds 94 exact coordinates not previously run over the later month.

<strong>Evidence impact:</strong> The first run remains historical and invalid for decisions. Its raw bytes are retained. Only the corrected run supports the current interpretation; engine, fills, cost model, data, target interval, and four-worker execution remain unchanged.

<strong>Validity boundary:</strong> The target month was already used by earlier temporal work. The corrected result is new at the exact-coordinate level and post-hoc at the period level. It cannot support parameter acceptance.

## 2026-08-07 — Compute interval statistics only after chart selection

<strong>Reason:</strong> Researchers need quick measurements for an arbitrary visible candlestick interval, while startup and chart-drag latency must remain unchanged.

<strong>Prior behavior:</strong> The per-trade chart supported zoom and trade inspection but had no interval measurement. The header also carried a view-mode explanation sentence the user no longer wanted.

<strong>Updated behavior:</strong> The sentence is removed. An explicit `区间统计` mode changes Plotly drag behavior to horizontal selection; `plotly_selected` then scans the selected OHLC slice once and opens a temporary statistics panel.

<strong>Evidence impact:</strong> This is presentation-only. It adds no backtest field, payload, request, startup precomputation, interval cache, ranking change, or parameter acceptance.

<strong>Validity boundary:</strong> Price change and amplitude use the first selected open as denominator. Maximum drawdown uses the largest running-high to current-or-later-low decline inside the inclusive selected interval.

## 2026-08-06 — Replace per-round mixed blocks and point caps with an AI-led cycle

<strong>Reason:</strong> K200 backtests are fast enough to evaluate finite grids with more evidence than a fixed two-or-three-point batch. Requiring leap coverage and local refinement inside every individual round also makes each plan less focused.

<strong>Prior behavior:</strong> Every ordinary round mixed broad and refinement blocks, and each one-parameter direction used a fixed small point count.

<strong>Updated behavior:</strong> AI alternates multi-round leap search, finite one-parameter grids around promising nonadjacent anchors, and renewed leap search. The user observes and corrects between cycles. Grids have no fixed point-count cap, but every plan freezes finite bounds, values or steps, expected coordinates, and anti-join evidence.

<strong>Evidence impact:</strong> This changes future exploration design and handoff rules only. Existing plans, raw results, rankings, migration evidence, and immutable snapshots remain historical facts.

<strong>Validity boundary:</strong> Fast compute does not authorize unbounded densification, repeated coordinates, method changes, new data identities, or parameter acceptance. Unattended work remains inside the explicitly authorized duration and resources.

## 2026-08-06 — Simplify and stabilize the K200 cumulative table

<strong>Reason:</strong> The stable entry repeated navigation already available in the main page, several display-only diagnostics and the research-contract section were not needed, and rendering the complete ranking table caused avoidable browser work while long-table review lost column context.

<strong>Prior behavior:</strong> The root kept an iframe and duplicate top bar. Cost-adjusted return labels repeated the cost wording, rank buttons included `查看`, several display-only diagnostic columns and the research-contract section were visible, and the browser generated every eligible row in one DOM update.

<strong>Updated behavior:</strong> The root redirects to the lean main page. The full-width main page uses the confirmed title and darker blues, simplified return and rank labels, fixed-width `#N` buttons, fewer display-only diagnostics, cost and cross-gap count as the final columns, no research-contract section, and 500-row pagination within a sticky-header viewport.

<strong>Evidence impact:</strong> Only generated presentation source, the stable presentation, focused UI assertions, and project records changed. Underlying fields, raw trades, metrics, filters, ranking semantics, and immutable snapshots remain unchanged.

<strong>Validity boundary:</strong> Pagination changes only rendered row count. It does not change the complete filtered member set, ranking order, result data, or immutable snapshots; virtualization and sort-result caching remain absent.

## 2026-08-05 — Give ordinary parameter exploration two explicit purposes

<strong>Reason:</strong> Local refinement alone can miss distant promising regions, while broad search alone cannot attribute an improvement to one parameter around a strong combination.

<strong>Updated behavior:</strong> This 2026-08-05 change established both broad coverage and single-parameter attribution. Its per-round coexistence requirement and fixed point-count limit are superseded by the 2026-08-06 AI-led leap/grid cycle; separately labeled leap and grid evidence remain current.

<strong>Evidence impact:</strong> Continuation Round 15 used 192 broad coordinates and 84 one-parameter coordinates. The broad branch found no robust primary leader; the refinement branch improved the average-E region. Existing strategy semantics, historical results, ranking rules, and parameter acceptance remain unchanged.

## 2026-08-05 — Make cumulative K200 per-trade publication incremental

<strong>Reason:</strong> Final cumulative publication previously regenerated every historical K200 per-trade chunk even when a round added only a small coordinate set. The repeated work dominated exploration delivery time and rewrote unchanged derived presentation artifacts.

<strong>Updated behavior:</strong> When a compatible current snapshot exists, the cumulative builder supplies its trade-review directory as a reuse source. Deterministic unchanged chunks are reused through same-volume hard links; only missing coordinate chunks are generated. The ranking payload, catalog, summaries, manifests, and stable routes are refreshed from the complete compatible population.

<strong>Evidence impact:</strong> The 5,044-coordinate delivery reused 4,747 chunks and generated 297. Raw trades, metrics, strategy semantics, qualification rules, and historical snapshot bytes are unchanged by reuse. The new snapshot and manifest make the new complete population explicit.

## 2026-08-05 — Make migration HTML publication incremental

<strong>Reason:</strong> A small migration append previously rebuilt every historical target per-trade chunk. That repeated unchanged work and left the migration workflow without a fixed, documented publication command.

<strong>Updated behavior:</strong> Completed combined runs declare `incremental_parent_run` and publish through `build_v4_4_cross_instrument_comparison.py build --run-id <run_id>`. Deterministic unchanged combo chunks are reused through same-volume hard links; only missing new-candidate chunks are generated. The small ranking index, catalog, process payload, summaries, and manifests are refreshed.

<strong>Evidence impact:</strong> The 247-candidate delivery reused 244 parent chunks and generated three. Target trades, metrics, frozen candidates, and strategy semantics are unchanged by chunk reuse. Each trade-review manifest records its incremental parent and reused/generated counts.

## 2026-08-05 — Restore one combined, plan-driven migration delivery

<strong>Reason:</strong> The repaired transfer was published as a separate current page and exposed a migration-batch column without user approval. That split hid the earlier positive result from the current comparison and made instrument-specific column names require future code edits.

<strong>Updated behavior:</strong> The current page is the 244-candidate union of the user-linked 180 and repaired 64 results. It removes batch display, reads relative instrument names from the approved migration plan, places source total return before target total return, restores default ranking on the third same-header click, and provides reciprocal new-tab navigation.

<strong>Governance:</strong> Any modification outside an explicit request or accepted proposal requires prior approval. A migration starts with required-information intake and a reviewed plan, and finishes with source main, source per-trade, combined ranking, target per-trade, and migration report.

<strong>Evidence impact:</strong> Completed raw transfer evaluations remain unchanged. The new artifacts are a presentation union, combined target trade review, relative-label plan, navigation shell update, and bilingual report. No combined score or parameter acceptance is introduced.

## 2026-08-05 — Batch nearby points and defer cumulative HTML to the final round

<strong>Reason:</strong> Raw parameter compute completes quickly, while each cumulative publication rebuilds roughly 710,000 historical trades and takes about twenty minutes. Publishing after every small round spends most of the exploration time rewriting unchanged historical output. A single new point also gives weak directional evidence.

<strong>Updated behavior:</strong> Each one-axis grid holds all other parameters fixed. The former fixed point-count limit is superseded by finite declared grids without a fixed cap. Intermediate campaign runs close raw evidence and compact summaries without HTML. The runner publishes cumulative main and per-trade HTML only when invoked with `--publish-html`, normally once after the complete exploration series.

<strong>Evidence impact:</strong> Existing snapshots and historical pages remain unchanged. Intermediate raw results remain authoritative but do not appear in the stable cumulative HTML until the final publication. An explicit user request can authorize an earlier publication.

## 2026-08-05 — Name the rolling-sum mean precisely without changing entry semantics

<strong>Reason:</strong> The short label `滚动 TR 总和` can be read as one total over BH, while the implemented method averages all overlapping TRW window sums. The chart also shades the full E high-search window, which visually overlaps BH and can be mistaken for an overlap between the baseline and the measured H-to-low drop.

<strong>Updated behavior:</strong> The cumulative and stage-page method display is `滚动 TR 总和均值`. The method value, formula, engine, and result selection remain unchanged. E continues to search strict H through the observed signal bar; BH ends at and includes the complete H atom; the net drop is `high[H] - low[signal]` with `H < signal`.

<strong>Evidence impact:</strong> Only presentation HTML, the generated-page label contract, declared HTML hashes, QA evidence, and bilingual records change. Source inspection, the stepwise-prefix equality test, and a 114-trade invariant audit show no future-data read. Raw trades, ranks, returns, and parameter acceptance remain unchanged.

<strong>Validity boundary:</strong> This audit validates causal availability, not the user's proposed strict no-shared-bar method. Ending BH at H-minus-one would change every affected baseline and threshold, require a new strategy/source identity, and require fresh compute before comparing performance.

## 2026-08-05 — Collapse the cumulative filter panel after selection

<strong>Reason:</strong> The full filter-and-sort panel occupies more than one desktop viewport after the four minute-range selectors were added. Researchers need to recover table space after finishing their selections without losing the chosen population or rank order.

<strong>Updated behavior:</strong> The panel starts expanded and exposes a fixed-size, icon-only SVG chevron disclosure button. The chevron points up while expanded and down while collapsed; no visible button text changes. Collapsing hides the control grid and current-summary block, preserves all filter and sort state, updates the English accessibility label and `aria-expanded`, and leaves one row. Ranking-table layout and paint containment prevents the large table from being repainted during the height change.

<strong>Evidence impact:</strong> Only the main-page generator, focused QA contract, the user-selected snapshot main HTML and alias, their declared hashes, and QA screenshots change. A 12-toggle browser measurement records 0.3 milliseconds maximum handler time and 14.9 milliseconds next-frame p95. `analysis_data.js`, ranking inputs, raw trades, result metrics, and parameter acceptance remain unchanged.

<strong>Validity boundary:</strong> This interaction does not auto-collapse, persist across reloads, alter any selector, change ranking, or modify the current stable cumulative snapshot. Future pages generated from the updated source receive the same toggle.

## 2026-08-05 — Add one dual-return-view axis and paired source/target totals

<strong>Reason:</strong> Readers need to inspect the same candidates before and after the 3.57-bps research cost. The source-instrument total also needs to be visible beside the target total because their evaluation intervals differ materially.

<strong>Updated behavior:</strong> A two-option control switches ranking and displayed total return, median trade, maximum drawdown, and win rate between fee/slippage-adjusted and gross modes. Existing file-backed selectors, global/field filters, and the cumulative navigation shell remain available. The final table groups parameters, SImain metrics, K200 metrics, and transfer diagnostics in that order; K200 total-return hover help explains its longer interval.

<strong>Evidence impact:</strong> Gross-view metrics are derived from the retained source/target transaction CSVs during presentation build. Frozen candidates, target execution, migration CSV/JSON, strategy results, and immutable cumulative snapshot files remain unchanged. Focused tests and browser QA cover both return modes, ordering, headers, hover help, new-tab trade routing, and responsive layout.

<strong>Validity boundary:</strong> This is a derived presentation change. It does not rerun either instrument, modify candidate selection, create an aggregate score, or accept a parameter. The fixed-cost migration report below the table remains explicitly labeled as cost-adjusted audit evidence.

## 2026-08-05 — Restore comparison selectors, all-field filters, and cumulative navigation

<strong>Reason:</strong> The accepted cross-instrument contract requires explicit source/target scope selection, interactive filtering across every result field, and a route from the existing cumulative entry. The prior presentation simplification removed those required controls.

<strong>Updated behavior:</strong> Four selectors are populated from completed comparison-run configs. Global search and composable field rules cover all displayed columns. The stable cumulative entry is a navigation shell that embeds the immutable snapshot and links to the comparison page.

<strong>Evidence impact:</strong> Candidate content SHA, source and target trade CSVs, migration metrics, report JSON, and immutable cumulative snapshot remain unchanged. Only presentation HTML, derived trade-review resources, QA evidence, source bindings, and the stable navigation shell change.

<strong>Validity boundary:</strong> Selectors route only to completed, file-backed comparison runs; they do not launch a browser-side backtest. Target results still cannot modify candidates, no aggregate score exists, and parameter acceptance remains `none`.

## 2026-08-04 — Simplify the cross-instrument page and add target per-trade review

<strong>Reason:</strong> The page-wide and field filters were unnecessary for the fixed 180-candidate validation, long candidate-source identifiers dominated the table, and its ranking interaction differed from the established K200 evidence interface.

<strong>Updated behavior:</strong> All selectors and filter inputs plus visible candidate-source/`combo_id` fields are removed. The first column reuses the K200 cost-rank button as `查看 #N`; every button opens the dedicated SImain trade review in a new tab. The stable cumulative entry no longer provides a K200/cross-instrument switch hub.

<strong>Evidence impact:</strong> This rebuilds derived comparison and review HTML from the already completed SImain candidate trade CSV. Frozen candidates, raw target trades, migration metrics, and source cumulative results are unchanged.

<strong>Validity boundary:</strong> The change improves presentation, routing, and evidence inspection only. It does not rerun the strategy, alter parameters, mutate immutable cumulative snapshots, create a combined score, or accept a candidate.

## 2026-08-04 — Relocate result storage from D to F

<strong>Reason:</strong> D was capacity-constrained while F had more than one TiB free. The 107.49-GiB V4.4 result tree and future rounds needed a durable physical storage root outside D.

<strong>Prior behavior:</strong> Source code, plans, runtime inputs, raw campaign output, cumulative snapshots, and HTML all lived below the D-drive project root.

<strong>Updated behavior:</strong> Source, plans, runtime inputs, and project management remain on D. The logical `results` path is a Windows directory junction to `F:\Backtest\Backtest V4.4\results`, so current and future result bytes live on F while historical paths remain valid.

<strong>Evidence impact:</strong> All 43,817 files and 115,416,958,661 bytes passed pairwise SHA-256 verification. Current pointer and HTML hashes are unchanged. No backtest, ranking, snapshot, or HTML regeneration occurred.

<strong>Validity boundary:</strong> The migration changes physical storage only. The recovery copy and four approved old-result directories remain recoverable on F. Git history and non-result project files are outside this migration.

## 2026-08-04 — Add four composable timing filters to the cumulative main entry

<strong>Reason:</strong> Exact timing values made the cumulative filter surface difficult to scan, while comparing several broad timing regions is a common research need.

<strong>Updated behavior:</strong> BH, E, W, and S each use multi-select minute intervals. One-axis selections form a union and cross-axis selections form an intersection.

<strong>Evidence impact:</strong> Only the cumulative main presentation, analyzer identity, and related analysis/completion manifest records change. Existing coordinates, trades, ranking data, and shared per-trade HTML remain byte-identical.

<strong>Validity boundary:</strong> This is frontend filtering behavior. It does not authorize a new parameter round, change return calculations, or accept a parameter.

## 2026-08-04 — Reopen continuation after Round-7 confirmation

<strong>Reason:</strong> The user explicitly authorized additional multi-round exploration and required every new result to remain comparable with the existing 2,751-coordinate cumulative set.

<strong>Prior behavior:</strong> Round 7 carried a plan-local stop-after-confirmation condition, and active documents still described only the earlier five-stage snapshot.

<strong>Updated behavior:</strong> Round 8 continues under the same campaign lineage with 637 anti-joined coordinates. Local blocks test the observed unrestricted and Scenario-1 ridges; broad blocks test strict entry and previously unobserved sub-50-minute speed windows. Every CSV row records its evidence and selection reason.

<strong>Round-8 outcome and next change:</strong> Round 8 added all 637 coordinates to the same cumulative set. Small total-return gains and a stronger average-return candidate justify only three focused continuation branches; five non-improving branches stop. Round 9 therefore narrows to 196 anti-joined coordinates while retaining one broad timing control.

<strong>Round-9 outcome and next change:</strong> Round 9 added all 196 coordinates to the same cumulative set. Neither total-return micro surface improved its Round-8 incumbent, so both stop. The average-return views improved from +0.6986% to +0.7758% per trade at S320 with lower drawdown; Round 10 therefore keeps 83 anti-joined coordinates around that timing and its E/BH, TRW/K, and W/M neighborhoods, plus three distant speed controls.

<strong>Round-10 outcome and stop reason:</strong> Round 10 added all 83 coordinates to the same cumulative set. M3 at S320 improved average fixed-cost return to +0.8458%, but the median trade stayed negative, the best two trades still contributed 41.85%, and gap-spanning trades supplied most of the compounded gain. The local improvement is recorded as an in-sample candidate; the current exploration stops without parameter acceptance or another plan.

<strong>Round-11 reopen reason:</strong> The user explicitly reopened multi-round exploration after the prior stop. The preserved Round-10 anchor has an unresolved lower-M boundary plus material MFE giveback and post-exit continuation. Round 11 therefore combines four local refinement modules with five small broad-jump controls inside historically used parameter bounds, while keeping the same cumulative lineage and shared HTML.

<strong>Round-11 outcome and Round-12 reason:</strong> W10/M2.5 improved average return and drawdown, and several neighboring W/M coordinates remained competitive. Best-two concentration also fell, although the median trade and gap dependence remain concerns. Round 12 narrows W/M/S resolution, rechecks E/BH and TRW/K at the improved exit anchor, and retains three distant exit-axis controls.

<strong>Evidence impact:</strong> Existing raw stages and historical snapshots remain immutable. New stages append to the shared cumulative ranking and per-trade review instead of creating a separate branch.

<strong>Validity boundary:</strong> This authorization does not accept a parameter or alter method semantics. Any later round still requires delivered evidence and another reviewed plan.

## 2026-08-04 — Replace fixed round roles with a mandatory evidence-led exploration guide

<strong>Reason:</strong> A fixed diagnostic/local/refinement sequence can trap exploration near the current leader and encourages agents to optimize isolated trades or one favored metric. The user requires broad jumps throughout exploration, trade-type reasoning, multi-metric model judgment, and a concrete next-round parameter handoff.

<strong>Prior behavior:</strong> Research constraints described broad and local branches but did not provide one mandatory operating guide. Cost rules still pointed to a dynamic K200M reference, and no standard CSV carried the next round's expanded coordinates and reasons.

<strong>Updated behavior:</strong> Every parameter-exploration task reads the bilingual guide. Rounds may mix independent experiment blocks, random trade inspection transitions toward profitable candidates with remaining improvement room, cost is fixed at `3.57 bps`, and the model explains improvement across several metrics without permanent weights or a fixed score. Each round hands off `next_round_parameters.csv` in a standalone directory.

<strong>Evidence impact:</strong> The guide changes future research design and documentation only. Existing source, raw results, rankings, HTML, and snapshots remain unchanged. A future compute needs a reviewed plan bound to the fixed-cost contract and the CSV's exact anti-join.

<strong>Validity boundary:</strong> The guide governs future V4.4 parameter exploration. Method changes still require separate user authorization and a new source identity; in-sample evidence still cannot accept a parameter.

## 2026-08-04 — Scale search resolution and retain exploration in every round

<strong>Reason:</strong> Fixed single-digit steps have very different meaning at different parameter magnitudes. Repeatedly moving E, BH, or S by a few bars around a large anchor adds in-sample selection freedom and is treated as overfitting. A high-dimensional, wide-bounded parameter space also cannot be assessed by local refinement alone.

<strong>Prior behavior:</strong> The guide required broad-jump coverage and neighborhood checks but did not define parameter-specific resolution or require an exploratory block in every search round.

<strong>Updated behavior:</strong> E, BH, and S use multiplicative broad, local, and stability grids. TRW/W, K/M, and A/floor use the resolution table in the guide. The current AI-led cycle returns to leap search after supported finite local-grid phases; phase-specific rounds remain allowed, and local grids cannot establish global convergence.

<strong>Evidence impact:</strong> This changes future plan design and documentation only. Existing plans, raw results, rankings, HTML, snapshots, and accepted method semantics remain unchanged.

<strong>Validity boundary:</strong> The finest grid checks local stability only. It does not authorize continuous peak-seeking with smaller steps, change method semantics, or accept an in-sample parameter.

## 2026-08-04 — Match validation effort to change risk

<strong>Reason:</strong> Running the complete suite for every frontend presentation edit consumes time and compute without adding proportional evidence.

<strong>Prior behavior:</strong> Small copy, spacing, color, and layout changes could trigger the same broad validation used for engine or strategy changes.

<strong>Updated behavior:</strong> Validation uses three tiers: complete regression for version/core/result-semantic changes, focused interaction checks for frontend behavior, and simple functional plus desktop/mobile visual checks for presentation-only changes.

<strong>Evidence impact:</strong> The handoff records the selected tier and its evidence. Existing full-suite evidence remains valid for the source identity it tested; a later presentation-only edit does not require repeating it.

<strong>Validity boundary:</strong> Any change that affects behavior, data, or result semantics escalates to the applicable higher tier. Uncertain scope also escalates.

## 2026-08-03 — Use HTML strong markup in management Markdown

<strong>Prior behavior:</strong> Managed Markdown used `**...**` for bold labels. CommonMark does not treat a closing `**` as right-flanking when it is preceded by Chinese punctuation and followed immediately by CJK text, so the dashboard displayed literal asterisks for forms such as `**长期目标：**修复`.

<strong>Updated behavior:</strong> Active management documents use HTML `<strong>...</strong>` for bold text. The project rules and project-management skill templates require the same form, and validation rejects Markdown strong delimiters outside inline code, fenced code, and literal glob examples.

<strong>Reason:</strong> HTML strong markup preserves the intended typography without adding an unnatural Chinese space or depending on punctuation-sensitive Markdown delimiter rules.

<strong>Evidence impact:</strong> Rebuilding the management dashboard changes only its documentation HTML. Backtest source, raw results, analysis results, trading HTML, and cumulative snapshots remain unchanged.

<strong>Validity boundary:</strong> This rule applies to active managed Markdown and future documents generated or maintained by the project-management skill. Archived historical snapshots remain byte-preserved; literal asterisks inside code and glob paths remain unchanged.

## 2026-08-03 — Normalize browser-rendered transparent RGBA in QA

<strong>Prior behavior:</strong> The stage browser QA required the exact source spelling `rgba(0,0,0,0)`. Plotly and Playwright rendered the contract-valid hollow fill as `rgba(0, 0, 0, 0)`, so the corrected HTML failed only because of whitespace normalization.

<strong>Updated behavior:</strong> The hollow-fill assertion removes CSS whitespace and lowercases the value before requiring `rgba(0,0,0,0)`. RGB must remain 0/0/0 and alpha must remain 0; any nonzero alpha still fails.

<strong>Reason:</strong> Browser serialization may normalize harmless CSS whitespace. QA must test transparency semantics while remaining strict about the actual channels.

<strong>Evidence impact:</strong> Existing corrected-source HTML and snapshot `a55ee981...` are preserved. QA-only recovery reuses those outputs after a new source identity closes; no HTML or raw compute rerun is required. The earlier trade-record ZIP remains valid historical V4 evidence but is not a current-V5 source package.

<strong>Validity boundary:</strong> This changes only QA comparison precision. It does not change templates, rendering, engine, runner, fills, returns, coordinates, raw results, or delivered HTML bytes.

## 2026-08-03 — Package V4.4 transaction-record CSVs

<strong>Prior behavior:</strong> ZIP handoffs excluded every compute-result payload, including the authoritative per-batch trade ledgers.

<strong>Updated behavior:</strong> Include immutable `batches/**/trades.csv` and derived `analysis/stage_trades.csv` from every completed V4.4 stage under `trade_records/`, with a manifest binding their source stage, role, row count, size, SHA-256, and source completion/stage-manifest identities.

<strong>Reason:</strong> A package must expose the actual buy/sell records for every parameter combination without carrying the entire result tree.

<strong>Evidence impact:</strong> Raw ledgers are copied byte-for-byte. The archive adds no result mutation and excludes all other result payloads.

<strong>Validity boundary:</strong> Applies to V4.4 user-requested ZIPs; it does not add partial, failed, or active stages to a package.

## 2026-08-02 — Remove same-bar look-ahead and impossible theoretical fills

Previous V4.3 behavior could construct a rebound basis with information available only after the current bar ended and then fill inside that same bar at a theoretical line. A 15-second OHLC bar does not reveal this intrabar ordering. V4.4 therefore uses only earlier completed bars for that check and fills at the confirming real close.

## 2026-08-02 — Prevent prior-regime W history

A large W could reach before the trade's H and import unrelated history. H now limits the source start while retaining the signal-generating decline between H and entry.

## 2026-08-02 — Keep data handling causal

The final low-activity label reveals a future recovery or confirmation. `baseline_available_from` expresses when that conclusion was actually knowable, so `exclude_marked` no longer uses future state prematurely.

## 2026-08-03 — Remove W-method ambiguity without changing raw behavior

<strong>Reason:</strong> Generic max-W wording was easy to misread as a full-W requirement or an internal-high-to-later-low maximum ordered decline.

<strong>Prior behavior:</strong> The engine already used every available 1..W prefix and calculated `open[start] - low[end]`, but project records did not state that an early prefix maximum could govern the trade or distinguish this value from maximum ordered decline.

<strong>Updated behavior:</strong> Trading logic and legacy raw audit fields remain unchanged. The exact method contract now names `w_open_to_end_low_drop`, states the available-prefix rule, and records the absence of full-W and minimum-ratio gates.

<strong>Evidence impact:</strong> New regression assertions and source-manifest fields make the existing semantics auditable. Historical raw rows remain schema-compatible.

<strong>Validity boundary:</strong> This does not authorize `full_W_only`, `min_rebound_window_ratio`, or an internal-peak decline alternative.

## 2026-08-03 — Add a turnover-sensitive derived ranking layer

<strong>Reason:</strong> Gross ranking can favor high-turnover coordinates even when a user-selected fee/slippage model materially changes their economic ordering.

<strong>Prior behavior:</strong> Primary views ranked and displayed gross returns only.

<strong>Updated behavior:</strong> Each primary view can rank and display either gross returns or returns after 3.56 bps per completed trade; cost-adjusted is default. The UI rank header follows the selected mode.

<strong>Evidence impact:</strong> Stage and cumulative analysis gain derived cost fields, ranks, model identity, and QA. Raw fills, raw `return`, and compute identity remain unchanged.

<strong>Validity boundary:</strong> The HKD-notional model is a selected shadow-cost assumption, not an exchange-tick execution model. Mini KOSPI 200 tick facts remain provenance only.

## 2026-08-03 — Replace fixed notional with a K200M-derived reference

<strong>Reason:</strong> A fixed HKD 300,000 notional obscures the actual one-contract size. The user requested a cost model derived from the K200M multiplier and current reference price.

<strong>Updated behavior:</strong> Future derived cost-adjusted rankings use a hash-bound snapshot: K200M price × KRW 50,000, then USD 6 commission converted through a dated KRW/USD reference, plus 2 bps round-trip slippage. The reference exposes its notional, each cost component, and computed bps in the stage and cumulative data.

<strong>Validity boundary:</strong> This changes only derived costs and rankings after the new source identity closes. Completed raw fills, raw returns, batch manifests, historical deliveries, and the historical ZIP remain preserved.

## 2026-08-03 — Scope cumulative publication to one compatible campaign

<strong>Reason:</strong> The first Round-1 delivery completed stage analysis, then the cumulative union correctly rejected an older temporary stage because its `engine_sha256` differed from the current multi-round stage.

<strong>Prior behavior:</strong> A full-root cumulative discovery mixed independently purposed campaigns and required every completed stage to share one union identity.

<strong>Updated behavior:</strong> The authorized cumulative-only recovery discovers completed stages from the active multi-round campaign root. The older temporary campaign is preserved outside that lineage, and its engine-identity mismatch is recorded as the exclusion reason.

<strong>Evidence impact:</strong> Round-1 stage analysis was reused without recomputation. A new one-stage snapshot was atomically published after full artifact and browser QA; the failed partial snapshot and prior stable snapshot remain preserved.

<strong>Validity boundary:</strong> This is a campaign inclusion boundary, not relaxed identity validation. Identity disagreement within the active campaign must still fail closed.

## 2026-08-03 — Simplify per-trade chart presentation and align the collapsed control tab

<strong>Reason:</strong> The Entry Reason step-2 hover highlight covered the exact high and entry prices with solid ellipses; the collapsed Parameters tab sat slightly lower than the intended control alignment; dashed interval guides and verbose chart annotations added visual encoding that the user did not need.

<strong>Prior behavior:</strong> `pointShape()` filled the two drop-point ellipses with their outline color. The desktop open tab used `top: calc(84px + 14px)`. Candlestick interval and guide shapes could use dashed or dotted strokes. The chart included a long theoretical-line/actual-fill/exit-basis annotation, and the frozen-low annotation appended freeze and duration details after `L=<value>`.

<strong>Updated behavior:</strong> Only `pointShape()` changes to transparent fill with the existing colored outline; other circle helpers remain unchanged. The desktop tab uses `top: calc(84px + 8px)`, a six-pixel upward move, while its mobile override remains `top: 16px`. All candlestick interval and guide strokes are solid and keep their semantic colors; text identifies meanings by color rather than line style. The long theoretical-line/actual-fill chart annotation is absent, while detailed side-panel reasoning remains. This delivery initially reduced the frozen-low label to `L=<formatted value>`. A later V4.41 layout change moved that value, together with the other five colored evidence boxes, from the chart to Entry Reason.

<strong>Evidence impact:</strong> The historical trade-template hash, analyzer/generator identity, QA source, review tests, and `SOURCE_MANIFEST` must close under a new source identity. Continuation Round-1 raw remains valid and immutable, while its derived stage/cumulative HTML requires one corrected-source redelivery and replacement QA. Source and browser assertions reject non-solid chart guides, misleading dashed-style copy, the removed chart annotation, and any non-exact frozen-low label.

<strong>Validity boundary:</strong> This changes review presentation only. It does not change fills, returns, coordinates, raw manifests, cost calculations, or research qualification.

## 2026-08-03 — Consolidate future per-trade HTML delivery without rebuilding results

<strong>Reason:</strong> Separate per-round per-trade pages duplicate the same review surface and make the delivery set larger than the user needs.

<strong>Prior behavior:</strong> The operating contract required stage and cumulative main/per-trade entries for every round.

<strong>Updated behavior:</strong> Preserve every existing page. A future authorized round refreshes the current cumulative main entry and the one shared cumulative per-trade entry only; the shared page includes all completed compatible results. No dedicated new per-round per-trade HTML is required.

<strong>Evidence impact:</strong> Current raw stages, snapshots, stable HTML, stage pages, and their evidence remain unchanged. The next run must apply this delivery contract before publishing new derived HTML.

<strong>Validity boundary:</strong> This is a future delivery-contract change. It does not alter raw compute, historical results, returns, costs, or existing HTML bytes.

## 2026-08-03 — Remove nonessential cumulative-main cards and isolate speed controls

<strong>Reason:</strong> The four cards repeat transient filter state and ranking output. The all-strategy count is the sole stable summary the user needs. The speed control has many choices and needs a full row for readable scanning.

<strong>Prior behavior:</strong> The cumulative main page displayed four top summary cards, and its speed-window fieldset shared the control grid with the ranking metric.

<strong>Updated behavior:</strong> Future cumulative-main output hides the card strip, shows `全部策略 <coordinateCount>` near the title, and gives the speed-window selector a full row.

<strong>Evidence impact:</strong> Existing snapshot HTML and delivery manifests remain unchanged. The source change is covered by a generated-HTML regression; a future presentation publication must produce new artifact hashes.

<strong>Validity boundary:</strong> No raw result, ranking, filter semantics, or existing delivered snapshot is changed.

## 2026-08-05 — Make formal source-review ZIPs independently testable

<strong>Reason:</strong> A compact UI review archive included selected tests without their sibling modules, templates, plan definition, or local cumulative fixture. Adding its partial code directory to `PYTHONPATH` could not resolve `run_v4_4_resumable_campaign` and related imports.

<strong>Prior behavior:</strong> The archive looked source-bearing but only supported HTML review. Its copied tests could not run as a self-contained suite, and two current project assertions still expected superseded generated-HTML strings.

<strong>Updated behavior:</strong> Formal source-review archives preserve the project-relative code, data-preparation, plan, review-template, runtime-input, and current cumulative-fixture layout; include `requirements-v4_4.txt`; expose one package-root PowerShell test runner; and prove that runner from an independent extraction. Assertions verify current structural behavior rather than obsolete selector concatenation details.

<strong>Evidence impact:</strong> The test-only source update changes no engine, runner, template, raw trade, return, ranking, or HTML result. The complete project suite and independently extracted package suite must both pass before release.

<strong>Validity boundary:</strong> Compact UI-only archives may remain intentionally non-executable when labeled as such. Any archive presented as a formal source release must satisfy the independent-test contract.

## 2026-08-05 — Require a positive entry signal

<strong>Reason:</strong> Session-filled flat synthetic intervals can produce zero TR, zero baseline, and zero drop. The prior inclusive comparison accepted `0 >= K × 0`, retained a pending signal, and opened at a later real-trade bar.

<strong>Prior behavior:</strong> Finite zero values passed the threshold comparison. Snapshot `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7` contains 4,404 such trades across 4,383 coordinates.

<strong>Updated behavior:</strong> Baseline, drop, and threshold must each be greater than zero. Inclusive equality is evaluated only after those gates, so a positive boundary equality remains valid.

<strong>Evidence impact:</strong> This is a result-semantics change. The old active campaign is retired recoverably and the complete 4,704-coordinate set is rerun. Existing SI evidence contains zero matching trades and remains historical cross-instrument evidence.

<strong>Validity boundary:</strong> The old K200 snapshot is reference-only after replacement publication. Corrected K200 rankings must use only the positive-entry result semantics.
## 2026-08-06 — Low-activity semantics and cumulative compatibility correction

<strong>Reason:</strong> The prior lifecycle delayed pending atoms until recovery and used exact implementation identities as cumulative compatibility gates. The confirmed strategy requires pending behavior to remain unchanged and minor V4.4 corrections to remain comparable with existing V4.4 results.

<strong>Updated behavior:</strong> Data preparation publishes `baseline_excluded_from` and `confirmed_low_activity_active`. The engine applies retroactive baseline exclusion and entry gating only at confirmation. The cumulative builder accepts the established V4.4 K200 major ranking lineage across minor engine and preparation hashes while retaining each stage's identities as provenance.

<strong>Evidence impact:</strong> Existing raw trades, cumulative snapshots, and rankings remain unchanged. The active preparation artifacts were regenerated for future runs. Any future V4.4 run uses the new policy and joins the existing V4.4 cumulative ranking when the final cumulative publication is requested.
