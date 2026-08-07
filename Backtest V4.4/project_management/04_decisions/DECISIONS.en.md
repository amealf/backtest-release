# Decisions

## 2026-08-08 — Include representative screenshots in visual deliveries

- Whenever completed work has a visible result and the environment allows it, include one representative screenshot and judge basic display errors.
- Use Computer Use to open local Chrome `file:///` pages when ordinary browser automation cannot access the path. This does not expand small changes into broad browser matrices.

## 2026-08-08 — Separate the V4.41 release label from the V4.4 result lineage

- Use V4.41 for the active release, page headings, and package metadata.
- Keep V4.4 as the strategy major version, result identity boundary, parameter namespace, and cumulative ranking major version.
- Treat this release as minor: earlier compatible V4.4 results remain active in the same cumulative ranking and are not archived or excluded.

## 2026-08-08 — Pair K200 training and test reviews by exact coordinate

- Training/test navigation names the destination and carries the exact `combo_id`; it never translates or approximates parameter values.
- Entry audit numbers belong beside their textual explanation rather than over the price chart. The chart keeps the evidence paths and moves its legend below the x-axis.
- Presentation refreshes remain shell-only. Result data and deterministic trade chunks keep their existing identities.

## 2026-08-08 — Identify result packages by instrument and exact dates

- The project is no longer organized around one instrument or fixed train/test directory names. Any executable instrument profile and exact evaluated interval can produce an independent result package.
- Directory identity is `<instrument_id>\<start_YYYYMMDDTHHMMSS>__<end_YYYYMMDDTHHMMSS>`. Experiment roles such as training, test, transfer, holdout, and descriptive replay are declared in `EXPERIMENT.md` and `comparison_plan.json`.
- The generic comparison entry reads package-owned browser summaries and joins exact `combo_id` values. Candidate-selection methods remain independent and are not part of this framework change.
- Existing stable and completed main/per-trade pages remain compatibility contracts. A stable redirect changes only after exact data parity and browser-visible parity pass; retained pages and the prior entry remain available for rollback.

## 2026-08-07 — Present K200 training, K200 test, and SI in one migration entry

- Retain the completed 250-candidate SI population, freeze 100 additional K200-temporal candidates before reading their SI outcomes, and anti-join the two populations exactly.
- Replay all 350 candidates on K200 test data and present the three total-return columns together with role-relative labels. Evidence roles remain explicit: old-candidate SI and new-candidate K200 test are descriptive rather than pristine validation.
- Publish HTML once after compute. Reuse the retained 250 SI per-trade chunks and generate only the 100 additions. A promising neighborhood remains unaccepted when its K200 test return depends on gap trades.

## 2026-08-07 — Evaluate live subsequent data with sequential unseen slices

- Freeze each round before evaluating its next time slice. Later rounds may use closed earlier slices, but they cannot read or tune within the slice they are about to evaluate.
- Keep the latest declared slice as the final holdout. A replay over the complete subsequent interval is post-hoc descriptive evidence and cannot become a second validation claim.
- Compare total return, average/median trade, drawdown, trade count, non-gap return, concentration, worst-slice return, and neighborhood evidence without a combined score. Sparse all-positive candidates remain observations, not accepted parameters.

## 2026-08-07 — Keep a contract and merge README in every data-download directory

- Every instrument acquisition directory contains its own `README.md`, created with the directory and refreshed on every resume and completion.
- The README records update times, source interval, main-contract calculation, exact contract split, adjustment policy, lineage, merge rules, and principal files.
- A supplement explicitly distinguishes a freshly recomputed main-contract decision from an inherited audited decision. The chronological update table remains with the downloaded data so future multi-instrument management does not depend on conversation history.

## 2026-08-07 — Record every backtest in one management document

- Before compute starts, append one row to `03_active_work\BACKTEST_MANAGEMENT.en.md` and its Chinese mirror.
- Each row records the instrument, exact market-data file, evaluated start time, and evaluated end time. File-only warm-up and later unused rows stay outside the recorded interval.
- Keep the row when a launched run stops early. This record is independent from raw-result closure and HTML publication.

## 2026-08-06 — Use a repeating AI-led leap/grid exploration cycle

- AI performs one or more leap-search rounds across nonadjacent legal regions, selects promising mutually nonadjacent anchors from closed multi-metric evidence, runs finite one-parameter grids around them, then returns to leap search.
- Individual rounds may be phase-specific. The complete cycle preserves separately labeled leap and grid evidence; local grids cannot establish global convergence.
- One-parameter grids have no fixed point-count cap. Each plan freezes finite bounds, exact values or steps, anchor, expected coordinates, resource limits, and completed+active+pending anti-join evidence before compute.
- The user reviews summaries and may correct objectives, anchors, ranges, or direction between cycles. During an explicitly authorized unattended period, AI may continue rounds inside the frozen instrument, method, data, duration, and resource boundary.
- This decision supersedes the fixed two-or-three-point rule and the requirement that broad and refinement blocks coexist in every individual round. Deferred intermediate HTML and one final cumulative publication remain unchanged.

## 2026-08-05 — Historical directional-grid and deferred-HTML decision

- Change one parameter per exploration batch and hold every other parameter at the current anchor.
- Use a bounded scale-aware directional set together with the anchor. The fixed point-count portion of this decision is superseded by the 2026-08-06 leap/grid cycle; single-parameter attribution remains current.
- Intermediate rounds close immutable raw evidence and compact summaries only. Publish the cumulative main and shared per-trade HTML once after the exploration series ends, or earlier only when the user explicitly requests it.
- The resumable campaign runner defers HTML by default. The final round opts in with `--publish-html`. This decision supersedes earlier requirements to publish cumulative HTML after every round.

## 2026-08-05 — Use strict-entry source Pareto selection for the second exact SImain transfer

- Freeze at most 100 source-only candidates from completed K200 evidence. Use the unrestricted fixed-cost total-return top 20% with at least 10 trades, exclude the original 180 transfers and current primary-view champions, and retain only W/M/S families already represented in the original transfer.
- Within each W/M/S family, vary only E/BH/TRW/K and retain the nondominated set on higher K200 cost-adjusted return, higher `median(entry_baseline_value × K)`, and lower K200 trade count. Do not use a combined score.
- Reuse the exact prior SIH6 target contract and append the completed batch to one shared SImain migration main entry and one shared per-trade entry. Do not run a SImain local grid.
- The aggregate cross-instrument Pareto set maximizes K200 cost-adjusted total return, SImain cost-adjusted total return, and SImain median cost-adjusted trade while minimizing SImain trade count and maximum drawdown.

## 2026-08-05 — Separate strategy, instrument, and campaign authorities

- Strategy semantics are instrument-neutral. Instrument data, cost, gap, low-activity, scenarios, and ranking lineage live in an instrument profile.
- Campaign intent is explicit: `transfer_exact`, `target_local_refinement`, or `fresh_search`.
- Cost and gap rules must be confirmed by the user for a new instrument. Missing values block execution.
- Compatible K200 additions stay in the current rank lineage. Other instruments use separate ranks; cross-instrument views compare evidence without merging ranks.
- Existing K200 results are not rebuilt. Future cumulative construction retains the cost model bound to each source stage.

## 2026-08-05 — Cross-instrument scope selection is file-backed and result-neutral

- The four scope selectors enumerate completed comparison runs and navigate to their retained pages. They do not create candidates or start target compute from the browser.
- Every displayed column remains sortable and filterable. Filtering changes only the visible row population; it does not alter metrics, frozen candidates, reports, or parameter acceptance.
- The stable cumulative entry provides navigation while the current cumulative snapshot remains byte-preserved and authoritative for its own results.

## 2026-08-04 — Store all V4.4 results physically on F while preserving the D logical path

The authoritative physical result root is `F:\Backtest\Backtest V4.4\results`. The established path `D:\Code\backtest-release\Backtest V4.4\results` remains as a Windows directory junction so historical absolute paths, manifests, plans, and HTML continue to work. Future result writers use the same logical root and therefore write physically to F. A verified recovery copy and recoverable old-result quarantine remain on F; no permanent deletion is part of the migration.

## 2026-08-04 — Use composable minute ranges for all four main timing windows

The cumulative main entry exposes entry baseline BH, entry market E, exit baseline W, and exit market S as independent minute-range filters. Each control supports any number of selected intervals. Selections are ORed within one control and ANDed across controls; `All` means unrestricted. This is a derived presentation filter and cannot alter ranking inputs, raw trades, or parameter acceptance.

## 2026-08-04 — Continue parameter exploration in the existing cumulative lineage

- Keep campaign ID/root `v4_4_cost_adjusted_multiround_20260803`; do not create a separate result branch.
- Compare every new coordinate against all compatible completed coordinates in one cumulative ranking and one shared per-trade HTML. Round 10 expanded that set from 3,584 to 3,667.
- Continuation Round 10 closed at 83 coordinates after exact anti-join against 3,585 completed current-V4.4 coordinates. Its average-return leader improved, while negative median trade and concentration remained. The resulting stop record remains historical evidence; the user's 2026-08-04 instruction reopens the same lineage with bounded broad controls and evidence-led local refinement. `parameter_acceptance` remains `none`.
- Continuation Round 11 improved the average-return candidate to W10/M2.5 while reducing drawdown and best-two concentration. Round 12 therefore refines the supported W/M/S surface and retests entry modules at the new exit anchor, with small distant W/M/S controls.
- Preserve independent Scenario-1 and unrestricted objectives, multi-metric judgment, broad-jump coverage, per-row selection reasons, and `parameter_acceptance=none`.

## 2026-08-04 — Make the Parameter Exploration Guide mandatory

- Every parameter-exploration agent reads and follows the bilingual guide before design, compute, interpretation, or next-round handoff.
- Each round may mix broad jumps, local axes, same-module pairs, trade-type experiments, and stability checks. Broad-jump coverage remains under consideration throughout exploration.
- E, BH, and S use relative multiplicative grids instead of continued single-digit absolute-step refinement. TRW/W, K/M, and A/floor use the parameter-specific resolutions recorded in the guide; the finest grid is a stability check, not a new peak-seeking phase.
- The current cycle returns to leap search after supported finite grid phases. Individual rounds may be phase-specific, and local grids cannot establish global convergence.
- Improvement is a model-explained multi-metric judgment. No metric receives a permanent high weight and no fixed aggregate score selects candidates.
- Future exploration cost is fixed at `3.57 bps` per completed round trip.
- Every round hands off expanded next-round coordinates in a standalone `next_round_parameters.csv`, including hypotheses and selection reasons, before plan binding and anti-join gates.

## 2026-08-04 — Use risk-tiered validation instead of routine full-suite runs

- Version upgrades and changes affecting engine behavior, fills, returns/costs, data preparation, schemas, execution contracts, or result semantics require the complete regression suite.
- Frontend filters, sorting, selectors, buttons, navigation, and state synchronization use focused interaction checks.
- Presentation-only copy, style, spacing, color, typography, alignment, visibility, and responsive layout use regenerated HTML, a simple functional check, and desktop/mobile screenshots.
- A presentation change escalates when it also alters behavior or data, or when its semantic impact is uncertain.

## 2026-08-02 — Create V4.4 as an independent temporary version

V4.4 is a copied, renamed workspace. V4.3 code and results remain unchanged. V4.4 is temporary and may receive further user-directed changes.

## 2026-08-02 — Causal rebound execution

- A previously established trigger uses `open >= trigger` for an open fill, then `high >= trigger` for a trigger fill.
- A strict-new-low bar cannot use its own completed W candidate. If its close confirms the rebound against the prior completed basis, the fill is that close.
- Equality triggers an exit.

## 2026-08-02 — H-bounded W source

The W source start is `max(H, continuous_start, end-W+1)`. The H-to-entry decline remains eligible; pre-H history is excluded.

## 2026-08-02 — Sample and low-activity causality

Open positions close at the declared sample-end bar close. `exclude_marked` uses causal `baseline_available_from`; `all_window` remains unchanged.

## 2026-08-03 — Exact W candidate and available-prefix contract

- Generate candidates from every available prefix length 1..W; do not wait for a full W or apply a minimum-window ratio.
- Define the candidate as `w_open_to_end_low_drop = open[start] - low[end]`, not the maximum ordered decline from any internal high to a later low.
- Retain the monotonic maximum inside one actual position, so an early shorter-prefix maximum may govern later exits. Keep legacy raw audit field names for compatibility.

## 2026-08-03 — Retained pending-entry contract

Use `retained_signal_fill_first_real_open`: preserve the signal for at most 120 continuous candidate bars and fill at the first real-trade bar open. Do not require a trigger recross or cancel on a higher high or structural reversal. A continuity break still cancels the wait.

## 2026-08-03 — Dual gross/cost-adjusted ranking

Every primary view has gross and cost-adjusted ordering/display modes. The chosen mode changes both sorting and displayed returns; cost-adjusted is default. The selected derived model is 2 bps round-trip slippage plus USD 6 round-trip commission at 7.8 HKD/USD on HKD 300,000 notional: HKD 106.8, or 3.56 bps, per completed trade. Raw fills and raw returns remain unchanged. Mini KOSPI 200 tick facts are provenance only and are excluded from this calculation.

## 2026-08-03 — Derived K200M current-notional cost reference

For future derived rankings, calculate one-contract notional from Mini KOSPI 200 price × KRW 50,000 rather than a fixed HKD amount. Freeze the latest real 15-second reference price and a dated KRW/USD reference in a hash-bound JSON record; derive the USD 6 commission in bps from that notional and add the already-confirmed 2 bps round-trip slippage. The initial frozen reference yields 3.568663594470046 bps. Existing raw results remain immutable and usable; legacy 3.56-bps outputs remain historical evidence.

## 2026-08-03 — Multi-round exploration governance

Explore Scenario-1-qualified and unrestricted cost-adjusted total return with multiple broad blocks before using closed evidence for concurrent broad coverage and local refinement. Each round is separately planned, hash-bound, closed, delivered, and interpreted; continuation is evidence-driven and never implies parameter acceptance.

## 2026-08-03 — Active-campaign cumulative boundary

The current multi-round cumulative lineage is built only from completed stages under campaign `v4_4_cost_adjusted_multiround_20260803`. A completed stage must agree on the union identity fields within that campaign. The older temporary validation campaign has a different engine identity, remains preserved as historical evidence, and is excluded from this cumulative lineage with that reason recorded. Failed partial snapshots are retained and never promoted to stable routes.

## 2026-08-03 — Terminal Round-3 boundary resolution

Round 2 gave strict cost-adjusted improvement to both objective-specific local branches and no improvement to the broad branch. Stop broad exploration. Run one terminal 212-coordinate local refinement around the new Scenario-1 lower-M/upper-S boundaries and unrestricted lower-S boundary. Round 3 closes the bounded in-sample campaign regardless of its result; Round 4 and parameter acceptance are prohibited.

## 2026-08-03 — Close the bounded campaign

Round 3 improved the Scenario-1 objective to 30.6696% cost-adjusted but did not improve the unrestricted objective, whose Round-2 leader remains at 36.0556%. End the campaign at 831 unique coordinates and 353,874 trades. Preserve both leaders as in-sample descriptive evidence only. Do not create Round 4 or accept a parameter.

## 2026-08-03 — Start a separately named continuation subseries

New user authorization permits further V4.4 exploration without reopening the closed original series. Preserve its Round-4 prohibition as historical truth; name new stages `continuation_round_*` under the same campaign ID and compatible active root. Begin with a 528-coordinate broad-span round, then refine only fully delivered promising regions.

Every continuation round must eventually complete immutable raw closure, stage main/per-trade delivery, cumulative main/per-trade refresh, and hash/browser/desktop/mobile/manual visual QA. Objectives remain independent, cost-adjusted remains default at 3.56 bps, and no parameter is accepted.

## 2026-08-03 — Role-neutral pipelined execution governance

One executor or multiple executors may perform analysis, backtest compute, and HTML delivery. After immutable closure, HTML delivery and evidence analysis may run concurrently; overlapping work is permitted only under closed-source, exact anti-join, separate process/output/root/lock, single-union-writer, and live-delivery-aware resource gates.

Every round still requires eventual stage main, stage per-trade, cumulative main, cumulative per-trade, full QA, and `DELIVERY_FINAL`. Delivery no longer blocks next-round analysis or compute. A result-affecting source inconsistency discovered by delivery pauses new compute until a new source identity closes.

## 2026-08-03 — Simplify per-trade chart presentation without rerunning raw compute

Entry Reason step 2 uses hollow colored-outline high/entry ellipses instead of solid fills. The collapsed desktop Parameters tab moves upward by six pixels; mobile drawer positioning remains unchanged. Every candlestick interval and guide stroke is solid while its semantic color remains. The long theoretical-line/actual-fill chart annotation is removed while detailed side-panel exit reasoning remains. The green frozen-low chart label is exactly `L=<formatted value>` without freeze, duration, or bar-count suffixes. Preserve the immutable Continuation Round-1 raw stage and old-source delivery, then close a new source identity before any new compute. After that closure, one corrected-source derived redelivery and Continuation Round-1 interpretation/Round-2 work may pipeline concurrently under the governance gates.

## 2026-08-03 — Include transaction records in V4.4 ZIP handoffs

User-requested V4.4 ZIPs now include every completed stage's raw per-batch `trades.csv` and derived `stage_trades.csv`. They are the auditable per-parameter buy/sell records, copied under `trade_records/` with a hash-bound manifest. Retain the narrow result boundary: all other `results/` payloads remain outside the ZIP.

## 2026-08-03 — Superseded stop-after-R2 boundary

The earlier stop-after-R2 boundary is superseded by the user's renewed instruction to continue parameter exploration. Complete immutable raw closure, stage/cumulative main/per-trade delivery, objective-specific interpretation, bilingual records, and read-only total audit for the current Round 2. Its delivered evidence alone may authorize a later plan; no later plan or compute is automatic. No parameter is accepted automatically.

## 2026-08-03 — Supersede the V4-bound Round-2 plan before compute

Do not bypass the V4-to-V5 source-identity mismatch. Preserve frozen plan `05d0d2ed...` and partial root fingerprint `976fac8d...` byte-for-byte as superseded pre-compute evidence; they contain only deterministic metadata and no progress, batch, trade, completion, or analysis. After the current source identity is confirmed, create a new bound Round-2 plan with the same audited 416 coordinates, a new filename/stage ID, and an absent output root. Exclude only the explicitly superseded V4 plan/root from the new target's active/pending anti-join. Any later round remains unapproved until current-Round-2 delivery and interpretation provide fresh evidence.

## 2026-08-03 — Preserve results and consolidate future per-trade delivery

Preserve all completed raw results, the current five-stage cumulative snapshot, and every existing stage page. Do not rebuild or replace them. For each future user-authorized round, update only the current cumulative main entry and the one shared cumulative per-trade entry so they include every completed compatible result. A dedicated per-round per-trade HTML page is retired as a required deliverable. Execution remains role-neutral: no subagent, separate conversation, or separate compute-versus-HTML thread is mandatory. Future compute still waits for an explicit user instruction and the established identity, anti-join, root/lock, resource, and single-writer gates.

## 2026-08-03 — Reduce cumulative-main summaries and separate the speed selector

Retire the four-card cumulative-main summary strip. The only summary retained in the top header is the dynamic all-strategy coordinate count. Give the S speed-window selector a full row because its option set is materially larger than the ranking-metric selector. This affects the next generated cumulative main page only; it does not authorize replacement of the sealed current snapshot.

## 2026-08-05 — Replace the zero-signal K200 lineage

The user authorized removal and rerun of every coordinate that produced `baseline<=0`, `drop<=0`, and `K×baseline<=0`. Because 4,383 of 4,704 coordinates are affected and batch manifests mix affected and unaffected coordinates, retire the entire old campaign recoverably and rerun all 4,704 coordinates under the positive-entry gate. The corrected cumulative snapshot replaces the old snapshot as current authority; the old bytes remain historical reference evidence.
## 2026-08-06 — Confirmation-time low-activity gate and V4.4 ranking boundary

Pending low-volume runs have no strategy effect. At the 120th consecutive low-volume 15-second atom, the complete run from its first atom becomes excluded from every later baseline calculation; unfilled entry orders are cancelled and new entries are blocked until the first normal-volume atom. Existing positions keep their normal exit rules.

K200 cumulative ranking is partitioned by accepted major version. Minor V4.4 corrections remain in the same V4.4 ranking page. Stage hashes and semantic identities remain integrity and provenance evidence. A V4.5 adoption creates a new ranking partition.
