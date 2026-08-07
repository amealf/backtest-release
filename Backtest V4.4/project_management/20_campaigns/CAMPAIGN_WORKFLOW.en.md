# Campaign Workflow

## Four official modes

### `transfer_exact`

Freeze a nonempty, unique source-instrument candidate set before reading target results. The plan must contain exactly the same recomputed coordinate IDs: no missing, extra, duplicate, or altered coordinate is allowed. Source and target must use the strategy's same 15-second data granularity. This mode must precede any local target refinement.

### `target_local_refinement`

Start from a completed `transfer_exact` parent for the same target instrument and test a declared bounded neighborhood. Bind the parent's candidate freeze, select anchors that exist in that freeze, declare fixed fields and numeric per-parameter bounds, and validate every new coordinate against at least one anchor. A distant grid cannot be labeled local merely by naming its scope `bounded_neighborhood`.

### `continuation_search`

Continue evidence-led exploration for the same instrument and ranking lineage from a completed parent stage. The parent instrument and lineage must match the new campaign. Follow the Parameter Exploration Guide for broad, local, diagnostic, stability, anti-join, interpretation, and handoff rules. This mode cannot import a parent from another instrument.

### `fresh_search`

Explore a specified instrument from scratch. Do not import source candidates or a parent stage. Bind the current Parameter Exploration Guide by path, size, and SHA-256; declare the legal parameter space, coordinate budget, and experiment-block/search-mode labels. Begin with instrument-appropriate broad coverage.

## Ranking and result lineages

The cumulative rank table merges only compatible results from the same instrument and `ranking_lineage_id`. Current K200 exploration remains under the repaired campaign/result root, so later compatible K200 rounds continue ranking with the existing 4,732 coordinates. A new instrument receives a separate lineage and rank table. Cross-instrument comparison displays source and target evidence together without mixing their ranks.

Every completed run also publishes or updates one date-based evaluation package for its instrument and exact evaluated interval. Package identity is independent from campaign mode and experiment role. The same package may later be described as training evidence, a holdout, a transfer source, or a comparison reference without moving or renaming it.

Completed migration batches are ranked together when target instrument, target sample, cost model, strategy semantics, and result schema match. Migration mode, run ID, and source stage remain provenance fields and do not create separate rank partitions. Deduplicate exact coordinates before publication; conflicting metrics for the same coordinate invalidate the union.

Historical K200 rows are not recomputed. New K200 rows use their stage-bound cost model, and the cumulative table sorts the union by each row's stored cost-adjusted metric. Every row exposes its actual round-trip cost.

## Same-instrument temporal migration

Use sequential walk-forward slices when completed training evidence is evaluated on subsequently observed data. Freeze the first candidate set from training only. Freeze each later set from training plus completed earlier slices before reading the next slice. Preserve the latest declared slice as the final holdout. The complete subsequent interval may be replayed once for descriptive ranking and per-trade delivery; label it post-hoc and keep it outside independent validation claims. Intermediate slices close raw evidence and compact summaries without HTML; publish the temporal ranking, target-period per-trade analysis, and report once after the series.

## Minimum workflow

1. Ask the user for any missing migration facts and write a reviewed migration plan containing source and target instrument IDs/display names, mode, data/sample, commission/slippage/FX, gap policy, low-activity policy, and candidate-filter status. Filtering remains deferred when it has not been approved.
2. Confirm that source and target use the same 15-second bar granularity. Stop and tell the user when profile, preparation manifest, or observed data differs.
3. Freeze the instrument profile and campaign manifest.
4. Freeze coordinates before compute and perform completed+active+pending anti-join inside the target lineage.
5. Apply the mode-specific contract: exact-set equality, bounded-neighborhood validation, same-lineage continuation, or from-scratch search-space validation.
6. Before compute starts, append the instrument, exact market-data file, evaluated start time, and evaluated end time to `project_management\03_active_work\BACKTEST_MANAGEMENT.en.md` and its Chinese mirror.
7. Run immutable raw compute only after explicit user authorization.
8. For intermediate rounds, keep immutable raw results and compact summaries without HTML publication. Continue only from complete evidence.
9. After a run or compatible series closes, write its date-based evaluation package: manifest, experiment record, neutral parameter summary, immutable trade records, browser projection when requested, and per-trade entry. Append only immutable batches and missing per-coordinate chunks.
10. After the exploration series ends, publish the shared cumulative main and shared per-trade HTML once for the lineage. Publish earlier only on explicit user request. No parameter is accepted automatically.
11. A multi-interval or multi-instrument report creates `comparison_plan.json`, assigns experiment roles there, and reads the selected completed packages by `combo_id`. The package directories remain role-neutral.
12. For a migration experiment, deliver the source main entry, source per-trade analysis, combined cross-instrument ranking, target per-trade analysis, and migration report. Read relative source/target labels from the plan. Link the source and cross-instrument entries to each other in new tabs.
13. Historical migration HTML may still be produced through `build_v4_4_cross_instrument_comparison.py build --run-id <run_id>`. Package-backed comparison publication uses `tools\build_v4_4_evaluation_framework.py`; unchanged per-trade chunks and immutable trade records are reused, and reused/generated counts remain recorded.

## Required pre-transfer smoke test

Before a new generic transfer path is used on full target data, run a small 15-second target fixture with an independent cost model, no scenario set, an independent ranking lineage, and two or three frozen K200 candidates. It must prove exact freeze/plan equality, freeze-before-target identity, target cost use, absence of K200 scenarios, target-instrument page naming, target-only lineage placement, plan-identity invalidation after any freeze change, target-tuning rejection, and granularity mismatch rejection. The fixture is validation evidence and never joins a research ranking.

## Short prompt forms

`Run transfer_exact from the current frozen K200 candidates to SImain. Data=...; sample=...; commission/slippage/FX=...; gap policy=...; low-activity policy=.... Do not tune on SImain.`

`Run target_local_refinement from the completed SImain transfer. Neighborhood=...; keep the same SImain profile and lineage.`

`Run continuation_search on K200 from the latest completed K200 stage. Follow the current exploration guide; keep the same profile and lineage; deliver a completed+active+pending anti-joined plan.`

`Run fresh_search on NQ. Data=...; sample=...; commission/slippage/FX=...; gap policy=...; low-activity policy=.... Start with broad coverage.`

If cost or gap information is missing, ask for it and do not launch compute.

The JSON starting point is `research_variants\short_momentum_net_drop_rebound_v4_4\campaign_contracts\CAMPAIGN_MANIFEST.template.json`.
