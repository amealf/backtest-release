# Project Goal

## Scenario research surface

Let the researcher choose one registered instrument-and-date market interval, mark one or more exact ranges, save that selection as a named scenario, and apply it to completed evaluation packages. The resulting qualified-coordinate page must preserve the current main ranking workflow and route every coordinate to its package-owned per-trade evidence without rerunning the strategy.

## Durable objective

Maintain one reproducible instrument-neutral V4.4 strategy protocol. Any executable instrument profile and exact time interval can be backtested independently, stored as its own date-based evaluation package, and compared through an explicit experiment plan. Preserve the established K200 evidence lineage while supporting controlled exact transfer, target-local refinement, continuation, and fresh exploration on other futures instruments.

## Current milestone

The four-slice K200 temporal migration is complete. Among 218 candidates observed in all four subsequent-market slices, two are positive in every slice and both are low-frequency, concentrated results. The training/full-test return-rank Spearman is -0.262, and only 25 of 400 frozen candidates are positive in the final holdout. Current evidence does not support one static broadly applicable parameter set. The active research direction is short-window re-estimation plus a separately authorized market-regime gate, followed by frozen forward evaluation. No parameter is accepted.

The current K200 cumulative snapshot `5b4e11b4c137028dc0a33d792a47800c8d792f6125e2cc8d2f5796ec6ef4fa94` contains 5,320 coordinates, 797,020 trades, and sixteen compatible positive-entry stages. Continuation Round 15 combined 192 broad-coverage coordinates with 84 one-parameter refinement coordinates. The average-return branch improved; the unrestricted total-return and Scenario-1 leaders remain unchanged. Another instrument receives a separate ranking lineage and can be compared through an explicit cross-instrument view. No parameter is accepted.

## Success criteria

- Same-bar rebound confirmation never assumes an intrabar threshold fill that cannot be ordered from 15-second OHLC.
- Prior-trigger exits use `open >= trigger` then `high >= trigger`, including equality.
- Every W source window begins at or after the trade's H.
- No exit reads a price after the declared sample end.
- `exclude_marked` baseline atoms become available only according to their causal lifecycle.
- W uses available prefixes from 1 through W and measures `w_open_to_end_low_drop = open[start] - low[end]`.
- Pending entry remains a retained signal that fills at the first real-trade open within 120 continuous candidate bars.
- Gross and cost-adjusted modes change both ranking and displayed returns; cost-adjusted is the default.
- Instrument profiles bind data, cost, gap, low-activity, optional scenario, and ranking-lineage identities without changing the shared strategy semantics.
- Evaluation-package directories are identified by instrument and exact evaluated start/end timestamps. Training, test, transfer, holdout, and descriptive roles remain experiment metadata.
- A generic comparison entry reads completed evaluation packages, joins exact `combo_id` values, and routes each metric to that package's own per-trade entry without rewriting retained trade evidence.
- Framework changes preserve the visible content and behavior of existing stable main and per-trade HTML; compatibility is proven before a stable redirect changes.
- `transfer_exact`, `target_local_refinement`, `continuation_search`, and `fresh_search` have separate evidence rules and cannot be silently substituted for one another.
- Cross-instrument transfer preserves the strategy's 15-second data granularity. A different target granularity fails closed before compute.
- Future entry signals require positive baseline, drop, and threshold values; equality at a positive threshold remains valid. Historical results remain immutable and are not recomputed by this contract repair.
- Existing K200 results remain unchanged. Later compatible K200 results rank with them; other-instrument ranks remain separate.
- Intermediate continuation rounds close immutable raw evidence and compact summaries without HTML publication. After the authorized multi-round exploration series ends, publish the shared cumulative main and shared per-trade entries once and apply the risk-proportional final delivery checks. An intermediate publication occurs only on explicit user request.
- Ordinary parameter exploration serves two purposes in the same round: broad legal-space coverage to reduce missed promising regions, and evidence-led one-parameter refinement around strong combinations to seek cost-adjusted return or maximum-drawdown improvement. Report the two branches separately.

## Non-goals

- In-sample leaders do not accept or promote a parameter.
- Derived cost analysis does not alter raw fills or raw returns.
- Cross-instrument comparison does not treat target refinement as evidence that an exact transfer succeeded.
