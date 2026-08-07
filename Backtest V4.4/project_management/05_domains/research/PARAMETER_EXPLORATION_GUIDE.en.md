# Parameter Exploration Guide

## Mandatory reading

Before an agent designs, runs, or interprets a parameter exploration, or prepares the next-round coordinates, it must read and follow this guide together with the current project constraints, research constraints, experiment progress, source identity, and latest complete evidence.

This guide turns trade-level observations into testable parameter hypotheses and defines how each round assesses improvement, records evidence, and hands off the next coordinates. It does not change the accepted V4.4 execution and signal contracts.

## 1. Fixed research boundary

- V4.4 remains in-sample research; no leader is an accepted parameter.
- Research cost comes from the campaign's frozen instrument cost model. It applies only to derived analysis; raw compute keeps `entry_slippage=0`, raw fills, and raw returns unchanged. Historical K200 rows retain their existing cost reference.
- Do not use a fixed aggregate score or assign any metric a permanent high weight.
- Gap-excluded return remains a display and dependency-audit field. It cannot independently rank, filter, qualify, or authorize continuation.
- Interpret only complete `IMMUTABLE_CLOSED` evidence, never partial batches.
- Ordinary parameter exploration preserves the current method contract. Direction efficiency, micro-new-low reset, trigger fill, gap policy, synthetic fill, W definition, and pending-entry policy changes are method variants that require separate user authorization and a new closed source identity.

### Choose the campaign mode

- `transfer_exact`: freeze source candidates before target execution and prohibit target-driven tuning.
- `target_local_refinement`: require a completed exact-transfer parent and a frozen bounded neighborhood.
- `continuation_search`: continue from a completed same-instrument, same-lineage parent while retaining broad, local, diagnostic, and stability evidence rules.
- `fresh_search`: specify an instrument, import no source candidates or parent stage, bind this guide, and start with instrument-appropriate broad coverage.

All transfer modes preserve the source strategy's 15-second data granularity. A target using another granularity is outside the current strategy contract and must stop for user review.

Read `project_management\20_campaigns\CAMPAIGN_WORKFLOW.en.md` for the full workflow and short prompt forms. A mode cannot change after target evidence is read.

## 2. How trade observation begins

Early exploration may sample trades randomly to learn how the strategy enters, holds, and exits across market states. Include profitable and losing trades, varied holding periods and exit reasons, and trades involving gaps, synthetic data, or low activity.

After some parameter combinations earn meaningful cost-adjusted returns, increasingly inspect those combinations for remaining improvement opportunities. Keep a random control sample so unfamiliar failure modes can still surface.

For a high-return trade, ask whether entry was late, exit was early or late, open profit was surrendered, results depended on gaps/synthetic data/idealized fills, the trade survives entry-parameter changes, or a few trades dominate the result.

For a losing trade, ask whether the decline was too weak, the baseline too low, the decline too slow, a strong opposite move preceded entry, rebound exit was too wide, speed exit was repeatedly reset by tiny new lows, or the loss came from signal quality, timing, or fill assumptions.

A single trade produces a hypothesis only. Parameter changes require type-level and portfolio-level evidence.

## 3. Classify trades into diagnostic types

Assign each important trade to one or more diagnostic types, then locate all similar trades.

| Diagnostic type | Typical evidence | Existing parameters to study | Separate method-version ideas |
| --- | --- | --- | --- |
| Weak-decline false entry | Low decline magnitude and speed, no continuation | K, E, A/floor | Direction-efficiency filter |
| Baseline too low | Low-activity period, small decline triggers | BH, TRW, A/floor | New activity filter |
| Entry too late | Most decline completes before the signal | E, K | New trigger fill |
| Rebound exit too early | New low follows soon after exit | W, M | New rebound baseline |
| Speed exit too early | Sideways pause precedes another decline | S | New speed-exit definition |
| Speed exit too late | Little progress followed by reversal | S | Micro-new-low reset rule |
| Excess profit giveback | Large MFE, small realized return | W, M, S | Structural exit |
| Gap-dependent | Most profit comes from a market closure jump | Robustness audit | Gap policy |
| Synthetic-related | Signal or fill depends on filled data | Baseline-sampling audit | Fill policy |
| Concentrated return | A few trades dominate total return | Trade-type and neighborhood checks | Usually no method change |

For each type, compare count, aggregate and median cost-adjusted return, win rate, drawdown contribution, MFE, MAE, MFE retention, and retained/disappeared/new trades after a parameter change.

## 4. Maintain anchor-trade diagnostics

Maintain a complete trade catalog for representative anchor combinations. Anchors may include profitable candidates, centers of stable regions, or informative failed combinations; they are not limited to the highest-return point.

At minimum calculate gross and instrument-cost-adjusted trade return, MFE, MAE, MFE retention, time to a new low after entry, new lows 30/60/120 minutes after exit, actual signal-window length, baseline/drop/ratio, gap/synthetic/low-activity participation, contribution to total return and maximum drawdown, and portfolio results after removing the best one and two trades.

List at least the top 10 return contributors, 10 largest losses, 10 largest profit givebacks, 10 greatest post-exit continuations, major drawdown contributors, and all relevant gap/synthetic/low-activity trades.

Diagnostics may run in the same round as parameter compute; they do not require a dedicated diagnostic-only round.

## 5. Run exploration as a repeating AI-led cycle

Parameter exploration alternates two purposes. Leap search samples nonadjacent legal regions away from current leaders so promising combinations are less likely to be missed. Evidence-led grid search starts from promising, mutually nonadjacent anchors, changes exactly one parameter per grid, and tests whether return quality or maximum drawdown improves.

The operating cycle is: AI performs one or more leap-search rounds, selects promising nonadjacent anchors from closed multi-metric evidence, performs one or more finite one-parameter grid rounds around those anchors, then returns to leap search. Individual rounds may belong to only one phase. Every report records the phase and keeps leap and grid evidence separate. A local-grid phase cannot support a claim of global convergence.

The user observes summaries and may correct the objective, anchors, ranges, metrics needing emphasis, or search direction between cycles. When the user is temporarily unavailable, AI may continue multiple rounds only within the already authorized duration, instrument, method, data, and resource boundary. A requested method change, new data identity, new instrument authority, or materially wider scope pauses for user confirmation.

Leap jumps, single-axis grids, supported module pairs, trade-type experiments, and stability checks may coexist when each block has its own question and evidence. Module pairs do not replace a declared one-parameter grid phase.

### Broad-jump block

Leap-search rounds evaluate and normally retain broad-jump coordinates outside the current leading neighborhood. Report them separately from local grids. Failure of one leap block stops that block only.

### One-axis directional batch

Choose one of E, BH, TRW, K, A/floor, W, M, or S and hold every other parameter fixed. Test a declared finite scale-aware grid in one parameter direction and keep the current anchor as the comparison row. There is no fixed point-count cap. This applies equally to time windows and threshold multipliers.

- If the closed grid shows a useful and reasonably consistent direction, the next grid may continue farther or refine a supported interval.
- If results deteriorate, behave abnormally, or fail to support the hypothesis, stop that direction, return to the anchor, and test the opposite direction or another parameter.
- Do not combine parameters or densify around one value until a closed directional grid supports doing so.
- User-supplied legal values take precedence; otherwise use Section 5A's scale-aware steps.

This rule creates an experiment block but does not authorize compute by itself. A run still requires user authorization and a reviewed plan.

### Same-module pair block

Combine only parameter pairs supported by diagnostic evidence, such as E × K, BH × K, TRW × K, A × K, W × M, M × S, or W × S. Do not build a full joint grid across every entry and exit dimension.

### Trade-type block

Change parameters for a named diagnostic type and list filtered, retained, new, and disappeared trades. Assess both the target type and the global trade set.

### Stable-region block

Use smaller steps around candidates to find neighboring parameter regions with similar behavior. A stable region has more research value than one isolated maximum.

## 5A. Scale search resolution to parameter magnitude

Do not routinely refine large time parameters with fixed single-digit absolute steps. In this project, continued single-digit densification of E, BH, or S is treated as overfitting: moving a parameter by a few bars around a large anchor adds in-sample selection freedom without representing a comparable economic change. Scale the search step to the current anchor instead.

| Parameter | Broad search | Local search | Finest check |
| --- | --- | --- | --- |
| E, BH, S | 20%–30% of the anchor | About 10% | About 5%; stability check only |
| TRW, W | 1–2 bars, or about 10%–25% | 1 bar | Usually no finer search |
| K, M | 10%–20% | 5%–10% | 2.5%–5%; stability check only |
| A/floor | Set from tick size, a baseline ratio, or historical quantiles | Change on an economically meaningful scale | Do not densify with arbitrary decimals |

Each one-axis directional batch contains the unchanged anchor and a finite set of new scale-aware points. There is no fixed point-count cap. Before compute, record the finite lower and upper bounds, step schedule or exact values, and expected coordinate count. Fast compute supports broader grid evidence; it does not justify one-point chasing, unbounded densification, or repeated reuse of completed coordinates.

Use a multiplicative grid for large time parameters. Do not switch step rules abruptly at an arbitrary absolute boundary such as 100. Around anchor `x`, use:

- Broad search: `0.70x`, `0.85x`, `1.00x`, `1.20x`, `1.50x`.
- Local search: `0.90x`, `1.00x`, `1.10x`.
- Stability check: `0.95x`, `1.00x`, `1.05x`.

An explicit legal step schedule supplied by the user overrides these defaults. Record that schedule in the frozen block definition and preserve every other coordinate field when the task is a one-axis contraction challenge.

When a proportional value must map back to discrete bars, round it to the parameter's legal unit, remove duplicates, and run the exact completed+active+pending anti-join. The finest check only tests whether the neighborhood is continuous and the ordering is stable. A leader that depends on one exact single-digit bar location, or disappears within the 5% neighborhood, must be marked as high overfitting risk and must not justify further step reduction.

### Return to leap search after local grids

The parameter space is high-dimensional and its legal bounds are broad. After a finite local-grid phase closes, return to leap search across under-sampled directions, legal regions away from incumbents, or alternative hypotheses that compete with the current explanation.

There is no permanent quota for leap-search coordinates. Each plan states their count, coverage, selection reason, and distinction from grid refinement. Several consecutive grid rounds are allowed when closed evidence supports them, but the campaign must label the phase and cannot treat local success or failure as global convergence.

## 6. Attribute entry and exit changes separately

For entry problems, prefer E, BH, TRW, K, and A/floor. For exit problems, prefer W, M, and S.

This is an attribution rule, not a fixed round schedule. Entry, exit, and broad blocks may coexist, but each block needs an independent hypothesis. Limit large cross-module searches until a direction is supported; use small interaction sets when evidence exists.

## 7. Define each block before compute

Record the question, diagnostic type, range, step, coordinate source, expected behavior change, supporting and falsifying outcomes, metrics to inspect, minimum trade count, evidence boundary, neighborhood check, gap/synthetic/idealized-fill disclosure, and conditions for widening, refining, observing, or stopping the block.

Freeze these statements before viewing new results.

## 8. Judge improvement across multiple metrics

The model must not assign a permanent high weight to one metric or replace research judgment with a fixed score.

Each decision reviews cost-adjusted total return, median cost-adjusted trade, maximum drawdown, win rate, trade count, MFE/MAE/MFE retention, return concentration, neighboring parameters, the target diagnostic type, retained/disappeared/new trades, gap/synthetic/low-activity dependency, and whether improvement exists only at an isolated point.

The round question determines which evidence is most informative for that hypothesis. Total return is always reported but has no automatic priority.

Classify the result as `improved`, `mixed`, `not_improved`, or `uncertain`, and state which metrics improved, which worsened, whether a few trades caused the gain, the trade-offs, why the classification is justified, and what supports further exploration.

Pareto comparison may identify non-dominated candidates, but it does not mechanically replace model judgment.

## 9. Validation-data boundary

Current work uses training data. If validation or holdout data is introduced later, freeze candidates, ranges, and decision rules before viewing it. Validation evaluates frozen candidates; it cannot be reused to generate new parameters repeatedly.

## 10. Required round interpretation

Record experiment mode, instrument profile, experiment blocks and hypotheses, coordinates and evidence sources, gross and stage-bound cost-adjusted results, retained/disappeared/new trades for important types, MFE/MAE/profit retention/post-exit continuation, return and drawdown concentration, neighborhood stability, gap/synthetic/low-activity dependency, the multi-metric improvement classification, each block's widen/refine/observe/stop decision, and `parameter_acceptance=none`.

## 11. Deliver next-round parameters as CSV

At the end of every exploration round, create a separate handoff directory whose core file is `next_round_parameters.csv`. Each row is one complete coordinate intended for the next run; do not store only a Cartesian-product definition.

Required columns:

| Column | Meaning |
| --- | --- |
| `candidate_id` | Unique ID inside the proposal |
| `experiment_block` | Owning experiment block |
| `search_mode` | `broad_jump`, `single_axis`, `module_pair`, `trade_type`, or `stability` |
| `anchor_combo_id` | Source anchor; blank for independent broad jumps |
| `diagnostic_type` | Target trade type |
| `hypothesis` | Question being tested |
| `method` | Fixed method |
| `baseline_sampling_policy` | Baseline sampling policy |
| `e`, `bh`, `trw`, `k`, `abs_floor_value`, `w`, `m`, `speed_window_bars` | Complete coordinate |
| `cost_bps` | Exact round-trip bps from the frozen instrument cost model |
| `expected_behavior_change` | Expected trade-behavior change |
| `evidence_summary` | Current-round evidence |
| `selection_reason` | Model's reason for selecting the coordinate |
| `source_round` | Evidence round |
| `status` | `proposed_for_next_round` |

The CSV must contain expanded unique coordinates, distinguish broad and local blocks, trace every row to a hypothesis, and pass an exact anti-join against completed, active, and pending coordinates. It is a research handoff; compute still requires a reviewed, hash-bound execution plan. If no next round is supported, deliver a header-only CSV and record the stop reason in the interpretation.

## 12. Agent checklist for every round

1. Read this guide and the current authorities.
2. Confirm source identity, instrument profile, experiment mode, ranking lineage, completed coordinates, and frozen cost model.
3. Use random trade observation early; as profitable candidates emerge, inspect their remaining improvement opportunities while retaining a random control sample.
4. Classify important trades instead of tuning around one trade.
5. Consider several improvement paths without assigning a permanent high weight to any metric.
6. Let the round question determine which metrics need more explanation, and record trade-offs.
7. Label every round as leap search or grid refinement, follow the cycle sequence, and return to nonadjacent leap coverage after a supported finite grid phase; do not infer global convergence from local grids.
8. For one parameter direction, run a declared finite scale-aware grid with every other parameter fixed; continue, refine, widen, stop, or change direction only after the complete closed grid pattern is interpreted.
9. Prefer entry parameters for entry problems and exit parameters for exit problems; require evidence for cross-module coordinates.
10. Freeze hypotheses, coordinates, evidence boundaries, and falsification conditions before compute.
11. Use a reviewed hash-bound plan and pass uniqueness plus completed/active/pending anti-join checks.
12. Pass source, output-root, process, lock, memory, and validate-only gates.
13. Interpret only `IMMUTABLE_CLOSED` evidence. Invalid closure, nonfinite evidence, or unattributable results do not advance.
14. Compare new, retained, and disappeared trades; inspect return concentration and neighborhood stability.
15. Issue an `improved`, `mixed`, `not_improved`, or `uncertain` classification with reasons.
16. Stop a block without eligible improvement; other blocks and broad exploration may continue.
17. Create the standalone handoff directory, write expanded `next_round_parameters.csv`, and complete its anti-join.
18. Defer cumulative main and per-trade HTML during intermediate rounds. After the exploration series ends, publish them once with `DELIVERY_FINAL`; publish earlier only on explicit user request.
19. Pause new compute if delivery finds a result-semantic source issue; wait for another closed source identity.
20. Keep `parameter_acceptance=none`.
