# V4.4 Cost-Adjusted Multi-Round Design

Status: non-executable design memo. It does not authorize compute and must not be converted to an executable JSON until the corrected V4.4 source identity is closed.

## Fixed question and evidence boundary

Map two objectives independently under unchanged V4.4 trading rules and `all_window`:

1. Scenario-1-qualified cost-adjusted total return.
2. Unrestricted cost-adjusted total return.

Gross results remain an alternate display/ranking mode and an interpretation comparison. Cost-adjusted is the default decision view. The selected derived model subtracts 3.56 bps from each completed trade: 2 bps round-trip slippage plus USD 6 round-trip commission at 7.8 HKD/USD on HKD 300,000 notional. Raw compute retains `entry_slippage=0`; raw fills and raw `return` are immutable.

All results are in-sample. No round accepts a parameter. Gap-excluded return, drawdown, average trade, trade count, exit reason, pending/wait behavior, lifecycle, and cost drag are diagnostics; they do not form a combined score.

## Round 1 — multiple broad blocks

The first executable plan should contain four non-overlapping Cartesian blocks, 372 coordinates before the historical anti-join. Each block has a distinct question.

### B1 — entry geometry map (84 coordinates)

- E: 40, 80, 120, 200, 320, 480, 720
- BH: 120, 360, 720
- TRW: 6, 24
- K: 0.75, 2.5
- W: 3
- M: 2.0
- S: 480

Question: which broad entry-frequency and threshold regimes survive the 3.56-bps turnover penalty before exit geometry is expanded?

### B2 — Scenario 1 exit geometry map (96 coordinates)

- Seed geometry: E40/BH720/TRW6/K2.0
- W: 1, 3, 12, 48, 192, 384
- M: 0.25, 0.75, 2.0, 8.0
- S: 320, 480, 720, 960

Question: across very short through long available-prefix W and weak through strong rebound budgets, where can Scenario 1 remain qualified while maximizing cost-adjusted total return?

### B3 — unrestricted exit geometry map (96 coordinates)

- Seed geometry: E200/BH360/TRW6/K0.75
- W: 1, 3, 12, 48, 192, 384
- M: 0.25, 0.75, 4.0, 8.0
- S: 320, 480, 720, 960

Question: how do rebound and speed exits trade turnover against unrestricted cost-adjusted total return around the prior unrestricted region?

### B4 — cross-regime interaction map (96 coordinates)

- E: 20, 80, 320
- BH: 120, 720
- TRW: 3, 24
- K: 1.25
- W: 1, 192
- M: 0.25, 8.0
- S: 320, 960

Question: do conclusions from the anchored exit blocks persist across extreme entry, W, M, and speed regimes, including a lower E value absent from the earlier broad Cartesian design?

## Later rounds — concurrent broad coverage and local refinement

Later executable plans are written only after immutable closure, fixed-template delivery, and a closed-round interpretation memo.

### Broad continuation branch

- Expand a boundary only when the objective's top five distinct coordinates materially concentrate on that tested boundary.
- Add one new scale step beyond the pressed boundary and retain one interior control value.
- If Scenario 1 has no eligible row, allow one additional broad qualification-recovery block; do not create a local Scenario-1 branch.
- Do not repeat coordinates present in any completed, active, or pending stage.
- Cap a broad continuation block at 144 new coordinates.

### Local refinement branch

- Select at most one distinct seed per objective after exact cost-adjusted ranking and `combo_id` tie-break.
- Build a small orthogonal neighborhood around the seed using the nearest tested values and one midpoint or log-midpoint where the parameter scale warrants it.
- Vary at most three interacting dimensions per local block; keep other dimensions at the seed value.
- Keep Scenario-1 and unrestricted seeds independent even when they share a coordinate.
- Cap each objective's local block at 96 new coordinates.

Broad and local branches may run in the same later round because they answer different questions. Their block labels, seed provenance, and continuation decisions remain separate.

## Adaptation and stop rules

1. Source gate: stop before plan execution if the source-manifest hash, implementation hashes, scenario hash, runtime-input hash, or full tests differ from the frozen handoff.
2. Resource gate: use three workers, batches of 12, and a 4,096 MiB minimum-free-memory floor. Pause and resume on the same identity if the floor would be breached.
3. Closure gate: interpret no partial batch. A round requires exact plan fingerprint, batch inventory, completion manifest, immutable raw hashes, and exactly one fixed-template delivery.
4. Objective independence: one objective cannot rescue, overwrite, or authorize a branch for the other.
5. Eligibility stop: a Scenario-1 round with no qualified row produces no local Scenario-1 seed. One broad qualification-recovery block is permitted; a second consecutive empty closed round stops that branch.
6. Improvement stop: a local branch stops when its closed leader does not strictly improve the predecessor's cost-adjusted primary metric after exact anti-join, or when the same interior leader is retained across two consecutive refinements.
7. Boundary adaptation: one outward expansion is permitted for a pressed boundary. If the new leader remains on the new outer boundary, report unresolved boundary dependence and stop that dimension rather than extending indefinitely.
8. Evidence stop: nonfinite primary metrics, trade/summary reconciliation failure, lifecycle mismatch, template hash failure, route failure, or browser QA failure blocks interpretation and continuation.
9. Time boundary: do not start a new round when less than 45 minutes remain in the authorized work window. If compute is active at the window boundary, preserve resumable state and report progress without interpreting partial output.
10. Terminal boundary: stop after three closed exploration rounds or earlier when both objective branches have stopped. Any further round needs a new explicit decision.

## Per-round interpretation memo

For each closed round record:

- exact source-manifest hash, plan hash/fingerprint, stage path, completion-manifest hash, and delivery-manifest hash;
- coordinate counts before/after anti-join and trade count;
- objective leaders in both cost-adjusted and gross modes, with rank reversals caused by modeled costs;
- Scenario 1 qualification, cost drag, trade count, maximum drawdown, gap dependence, exit mix, pending/wait behavior, and holding lifecycle;
- boundary pressure and whether each broad/local branch continues, adapts, or stops, with the exact rule invoked;
- the in-sample and shadow-cost validity boundary and the statement that no parameter is accepted.
