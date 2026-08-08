# V4.4 Continuation Subseries Design

Status: final non-executable design. This document records the newly authorized continuation phase and authorizes freezing `continuation_round_01_broad_span_all_window` only after its exact tuple and cumulative-inclusion audits. It does not authorize A to run validate-only, create a stage, run raw compute, or generate delivery outputs.

## Historical boundary and new authority

The original three-round campaign remains closed exactly as recorded. Its canonical evidence still states that Round 4 was prohibited and no parameter was accepted. The new user authorization does not rename, reopen, or reinterpret that campaign. It creates a separately named continuation subseries under the same compatible cumulative campaign lineage so future total-entry publication can include both the closed stages and new compatible stages.

- Existing campaign ID and compatible cumulative discovery root: `v4_4_cost_adjusted_multiround_20260803` / `results\campaigns\v4_4_cost_adjusted_multiround_20260803`.
- Closed historical stages: `round_01_broad_all_window`, `round_02_broad_local_all_window`, `round_03_terminal_local_all_window`.
- Continuation stages retain the same campaign ID and sit directly below the existing campaign root. The first planned output is `results\campaigns\v4_4_cost_adjusted_multiround_20260803\continuation_round_01_broad_span_all_window`.
- New subseries naming: `continuation_round_01_*`, `continuation_round_02_*`, and so on. No new stage may be called Round 4.
- Historical Round-4 prohibition remains true for the closed original series.
- Parameter acceptance remains `none` across both the closed series and continuation.

## Closed evidence identity

- Source manifest SHA-256: `6fa3d0c8eb0277066ef5f70fca4a9fbab1d31fbb30e023cd8fd83d233192ae16`.
- Final closed snapshot: `0fb3e1e5e8ef890f3b225db46288fa4b3957bcb88c7ca2dff72d750679db6922`.
- Final cumulative pointer/analysis/completion/trade-review SHA-256: `f91f5eaee14f4b92d91f2b3b28150a702731c4968c557c7302fa30f22a6478fc` / `d8dca67c5631fa3da3cb0dc7630b1f60fd19ec0b1e6368ac84a3242ea5aa13e7` / `32adc2941a99f2c4733dc2ed7da84d44823f903cc3e5958b9ba5443ca3a7f17c` / `06b71481a7348c437807eb3664232948a8c91d81141ccd87cb5fe5606aaf0780`.
- Canonical terminal interpretation SHA-256: `d740561e13e7fe5da6b0dc96e826c4b155f99a56b34b899c9b2b77653d373837`.
- Final read-only total-audit SHA-256: `58009d4eb357e3022de423d63310c9821fd167e518b43c739dc8efea6a694c0e`.
- Closed active-campaign scope: 831 unique coordinates, 353,874 trades, 70 batches, three compatible stages, zero cross-round duplicates.

## Unchanged objective and method contracts

- Primary objective A: Scenario-1-qualified cost-adjusted total return.
- Primary objective B: unrestricted cost-adjusted total return.
- Objectives remain independent; no combined score is allowed.
- Cost-adjusted is the default ranking and display mode. Gross remains an optional alternate display.
- Selected shadow cost remains 3.56 bps per completed trade: 2 bps round-trip slippage plus 1.56 bps commission. It is derived analysis only; raw fills and raw returns remain unchanged.
- Gap-excluded return remains a display-only dependence audit and cannot rank, qualify, or authorize a continuation.
- W uses available prefixes 1..W with no full-W or minimum-ratio gate; an early prefix maximum may govern a trade.
- The exact candidate remains `w_open_to_end_low_drop = open[start] - low[end]`, not an internal maximum ordered decline.
- Pending entry remains retained-signal `fill_first_real_open` for at most 120 continuous candidate bars, without recross, higher-high, or structural-reversal cancellation.
- Baseline policy remains `all_window`; raw `entry_slippage=0`; exit mode remains combined.

## Mandatory per-round closure and delivery rule

Every continuation round must complete the following sequence. A later step is prohibited until every earlier step is closed:

1. A freezes one reviewed, hash-bound plan after exact completed/active/pending anti-join construction.
2. B independently verifies source, plan, predecessor, anti-join, lock, process, and memory identities; runs validate-only; then launches compute-only with three workers, batch size 12, a 4,096 MiB minimum-free-memory floor, and HTML disabled.
3. B closes every post-anti-join coordinate immutably and reconciles all raw manifests, counts, trades, IDs, hashes, locks, and processes.
4. C runs exactly one four-worker fixed-template delivery for the new stage and active-campaign cumulative publication.
5. The stage must publish its main HTML, per-trade HTML, and scenario page. The cumulative output must refresh its main HTML and cumulative per-trade HTML through an atomic snapshot/pointer publication.
6. Hash/size QA, browser interaction QA, desktop screenshots, mobile screenshots, and manual visual review must pass for stage, snapshot, and stable routes. Cost/gross sorting, displayed returns, and rank headers must change together.
7. Only after `DELIVERY_FINAL` may A interpret the round, update the objective leaders, stop branches, or design the next continuation round.

Partial raw batches, stage analysis without delivery, a cumulative snapshot without QA, or a stable route that has not passed QA cannot be interpreted and cannot motivate another plan.

## Cumulative discovery and compatibility boundary

The cumulative builder recursively discovers every `stage_manifest.json` below its supplied `campaigns_root`, then validates exact retained plan/raw artifacts and shared source, data, engine, execution, and schema identities. The minimal continuation topology retains the existing campaign ID and distinguishes the new subseries through its `continuation_round_*` stage namespace. Snapshot provenance includes each stage's campaign ID, stage ID, fingerprint, and completion hash.

Every continuation delivery must therefore use:

- `campaigns_root = D:\Code\backtest-release\Backtest V4.4\results\campaigns\v4_4_cost_adjusted_multiround_20260803`
- `union_output = D:\Code\backtest-release\Backtest V4.4\results\all_completed_union_analysis`

This root includes preserved R1–R3 and direct `continuation_round_*` stages while excluding the incompatible temporary sibling. The full `results\campaigns` root is prohibited because it also discovers the historical temporary stage with incompatible engine identity.

## Evidence-led continuation approach

The closed campaign leaves two distinct cost-adjusted regions:

- Scenario-1 final leader: E320/BH720/TRW24/K1.25/W4/M4/S400, 58 trades, +30.6696% cost-adjusted, 7.8125% cost-adjusted maximum drawdown, and +0.6431% gross return after excluding gap-spanning trades. W=4 and M=4 pressed the terminal local boundaries.
- Unrestricted final leader: E40/BH720/TRW6/K2/W48/M4/S400, 94 trades, +36.0556% cost-adjusted, 16.9845% cost-adjusted maximum drawdown, and +0.3523% gross return after excluding gap-spanning trades. The terminal unrestricted refinement did not improve this seed.
- The previous 144-coordinate shared broad block used one W/M/S anchor and did not improve either objective. The continuation therefore uses a different anchor and a materially wider entry map, while retaining separate broad exit maps for the two objectives.

The first continuation round is broad-span rather than local refinement. Later continuation rounds may refine only regions motivated by a fully delivered continuation round.

## Continuation Round 1 — broad-span design

Planned new coordinates before B's external anti-join: 528 across three non-overlapping blocks. Expected batches at size 12: 44.

### C1-BROAD-ENTRY — shared entry-geometry map, 288 coordinates

- E: 20, 40, 80, 160, 320, 640
- BH: 120, 240, 480, 720
- TRW: 3, 6, 12, 24
- K: 1.0, 1.25, 2.0
- W: 8
- M: 4.0
- S: 400

Purpose: map both objectives across a larger entry-geometry span at a new W8/M4/S400 exit anchor. This anchor differs from the closed broad anchors W3/M2/S480 and W48/M8/S600, so the block contributes new evidence rather than repeating prior broad surfaces.

### C1-S1-BROAD-EXIT — Scenario-1 exit span, 180 coordinates

Entry geometry fixed at E320/BH720/TRW24/K1.25.

- W: 1, 2, 4, 8, 16, 32
- M: 2.0, 3.0, 4.0, 5.0, 6.0, 8.0
- S: 200, 300, 420, 640, 800

Purpose: test the terminal W/M boundary pressure across a materially wider exit surface and new speed-window values, while keeping Scenario-1 qualification mandatory for this objective.

### C1-UR-BROAD-EXIT — unrestricted exit span, 60 coordinates

Entry geometry fixed at E40/BH720/TRW6/K2.

- W: 16, 32, 64, 128, 256
- M: 2.0, 3.0, 4.0, 6.0
- S: 240, 480, 720

Purpose: remap the unrestricted W/M/S surface on both sides of the retained W48/M4/S400 leader using broad logarithmic W coverage and speed windows outside the terminal local slice.

## Exact tuple and anti-join construction

The pre-freeze audit used the complete execution tuple:

`method | baseline_sampling_policy | entry_fill_mode | entry_execution_policy | entry_slippage | exit_mode | E | BH | TRW | K | W | M | S`

It expanded the proposed blocks in memory and compared them with every row in the final 831-coordinate cumulative `analysis_summary.csv`.

- Planned tuples: 528.
- Planned unique tuples: 528.
- Internal duplicates: 0.
- Completed active-campaign tuples: 831.
- Completed active-campaign unique tuples: 831.
- Planned/completed overlaps: 0.
- Block counts: 288 / 180 / 60.

B must still independently repeat the exact anti-join across all completed plus active/pending current V4.4 stages before validate-only and launch. The historical temporary single-coordinate stage remains outside the cumulative lineage but inside B's broader current-V4.4 collision audit.

## Continuation adaptation rules

- After a fully delivered continuation round, retain each objective's leader independently and compare strict cost-adjusted primary-metric improvement against the closed baseline leaders and the previous continuation round.
- A later refinement block must name the delivered broad region, boundary, ridge, or stability pattern that motivates it.
- Stop an objective branch when a delivered round has no eligible cost-adjusted improvement and exposes no unresolved broad region that the user has explicitly authorized exploring.
- Gross leaders that are cost-negative cannot seed cost-adjusted continuation.
- Gap dependence, drawdown, trade count, average trade, lifecycle, wait behavior, pending exits, and exit reasons remain required interpretation diagnostics but do not replace either primary objective.
- No continuation result accepts a parameter. Any production selection or out-of-sample validation remains separately scoped.
