# Work Progress

## Purpose

Keep a concise active log of completed or materially changed work for Backtest V4.3 Research. Record outcomes, evidence, limits, and useful handoff paths rather than raw transcripts or every intermediate action.

`CURRENT_TASKS.en.md` describes work that is planned or underway. This document records work that has completed or changed enough to affect how the project should be understood.

## Active delivery or method identity

- Current major version or working method: V4.3 `rolling_tr_sum` with selectable baseline sampling (`all_window` default; `exclude_marked` optional), calculated entry, first-real-open pending entry, and next-real-open pending exits.
- Validity boundary: Rounds 1–5 are immutable and fully delivered at 684 cumulative coordinates/232,225 trades. Round 5 is terminal; no Round 6 or parameter acceptance is authorized.

## 2026-08-02 — Terminal Round 5 and five-stage campaign closed

- Round 5 executed the approved two 12-point blocks with 24 fresh anti-joined coordinates, two batches, 46,732 trades, and compute-only resources 3/12/4096. Immutable evidence SHA-256 is `ae53e778944aa3fd4ed41627655fbaeb07aa863632110a0073728bba11b784c3`.
- Exactly one four-worker fixed-template delivery closed at cumulative snapshot `4bed7f828a7068ee0ee70001a245133c481e785358cfb3f2b043d08713ca7259`: five stages, 684 coordinates, and 232,225 trades. Delivery QA evidence SHA-256 is `6d3cc38cfe98d16d2b095990f2f6a8c85d69c230b9b873bae61d5c414c6751a0`; 775/775 artifact hashes, 13/13 source/raw/plan checks, 80+80 browser states, 12/12 screenshots, and 3/3 stable routes passed.
- Scenario-1 total retains the Round-4 E30/BH720/TRW6/K1.5/W1/M2 leader at 37.082743%. Unrestricted total moves to Round-5 E120/BH360/TRW6/K0.75/W1/M0.25 at 515.916078%. Both average views retain Round-3 E200/BH720/TRW24/K0.75/W192/M10 at 0.578419%.
- All four views remain separate. Gap-excluded return remains display-only. The final leaders are dependent in-sample candidates for externally authorized validation/review only.
- Decision: close the max-W campaign after Round 5. No Round 6 and no parameter acceptance, validation, promotion, or production claim.
- Handoff: canonical terminal report `D:\Code\backtest-release\Backtest V4.3 max-W rebound\.omo\teams\team-a31b9876\artifacts\A_v4_3_max_w_round_05_20260802.md`; clean-package staging remains owned by the delivery member after the leader's terminal declaration.

## 2026-08-02 — Round 4 closed; terminal Round-5 plan frozen

- Round 4 closed at 72 coordinates/73,373 trades, with cumulative delivery at 660 coordinates/185,493 trades across four stages.
- New cumulative leaders are E30/BH720/TRW6/K1.5/W1/M2 for Scenario-1 total at 37.082743% and E180/BH360/TRW6/K0.75/W1/M0.5 for unrestricted total at 340.137206%. Both average views retain the Round-3 M10 leader because the Round-4 M12 leader is lower.
- Stable delivery evidence SHA-256 is `0d5d3187de8a2a1297bbb35f1de220930d34ed30c117906b936506cd404b7fb8`; 803/803 asset hashes, 80+80 browser states, and 12 screenshot inspections passed.
- The leader approved exactly two terminal 12-point Round-5 blocks, 24 coordinates total, no average block. Frozen plan SHA-256 is `bb7f323e077ebb919d122764d7fac20b97b0790959d71761b77bcf69e6a23ed8`.
- This was the activation record. Round 5 subsequently passed B's authoritative gates and closed under the terminal record above.

## 2026-08-02 — User-authorized Round-4 boundary plan frozen

- Objective: Continue the inspected in-sample search only across the unresolved boundaries of the three independent Round-3 leaders.
- Change: Reversed the prior no-Round-4 decision after explicit user authorization and froze three independent 24-point blocks under all-window/rolling-only/S480 and resources 3/12/4096.
- Evidence: Plan `research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_max_w_multiround_20260802_round_04_all_window.json`, 13,221 bytes, SHA-256 `f991e8c46b42ccc38677eef3be7f2e67d9e822592bb53d3578a4fbc1cdb77670`; pre-run memo in the active team artifacts.
- Gate: Member B owns authoritative runner expansion, exact completed plus active/pending anti-join, fingerprint, resource/lock checks, and validate-only. No compute or delivery has started; do not interpret results before immutable closure.
- Limit: Four independent views and display-only gap audit remain unchanged. Conditional Round 5 is absent by default, requires new approval, is capped at 36 fresh coordinates, and is terminal. No parameter is accepted.

## 2026-08-02 — Terminal Round-3 interaction refinement and campaign closure

- Objective: Test meaningful E/W/K-or-M interactions around the three deduplicated Round-2 leaders, then make a terminal evidence decision under the four independent views.
- Change: Round 3 completed 72 anti-joined `all_window` coordinates in six batches and 37,618 trades. The cumulative snapshot now contains 588 coordinates and 112,120 trades across three completed stages.
- Outcome: Scenario-1-qualified total leads at E60/BH720/TRW6/K2/W2/M2 with 18.2861% total, 336 trades, and -6.6971% max drawdown. Unrestricted total leads at E160/BH360/TRW6/K0.75/W2/M1 with 115.8216% total, 2,541 trades, and -3.5619% max drawdown. Both average views lead at E200/BH720/TRW24/K0.75/W192/M10 with 0.578419% average, 77 trades, 51.8332% total, and -13.8733% max drawdown. All improve the corresponding Round-2 primary leader in-sample.
- Evidence: Completion SHA-256 `d42d806e0e9c4d651b18e03846f300c8f059e70dff987bab9dc544f281e474a7`; immutable evidence SHA-256 `0652f87f01f0b617b074279b80a0379197fdbff9871a51008477c60abfede3f8`; terminal delivery evidence SHA-256 `7b8f1450c9b53b84ad5d5d2e3ddd51fa6d84147624b41b4135cbe9d29262bf6f`; current snapshot manifest SHA-256 `510c914b4d76f1fb5dd09af72e4b92a86bd9915e195b924989a1cf75a5d76eee`.
- Limits: The leaders are dependent in-sample candidates only. Scenario qualification is interaction-specific, unrestricted total depends strongly on W2/M1, and average return retains an M10 tested boundary. Gap-excluded return is display-only and cannot accept or reject a row. No holdout, new-date, cross-instrument, cost/slippage, or production evidence exists.
- Decision: Close the campaign at Round 3. Retain the three view-specific leaders for external validation/review only; create no automatic Round 4 and accept no parameter.
- Handoff paths: `results\campaigns\v4_3_max_w_multiround_20260802\round_03_all_window`, `results\all_completed_union_analysis`, and `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\A_v4_3_max_w_round_03_20260802.md`.

## 2026-08-02 — Round-2 broad exploration closed and Round-3 interaction refinement proposed

- Objective: Interpret the broad 504-coordinate result through the four approved full-return views, then define a meaningful bounded refinement without blending objectives.
- Change: Round 2 completed 504 `all_window` coordinates in 42 batches and 73,306 trades. Four independent decisions yield three distinct leaders: Scenario-1 total E80/BH720/TRW6/K2.5/W3/M2; unrestricted total E120/BH360/TRW6/K0.75/W3/M2; and both average views E200/BH720/TRW24/K0.75/W192/M8.
- Outcome: Primary metrics are 10.8547% Scenario-1 total, 51.3786% unrestricted total, and 0.554842% average trade for both `>=10` and `>=20`. The frozen 72-coordinate Round-3 plan uses three separate 24-point E×W×K/M local blocks; runner loading reports 72 unique and zero exact combo-id overlap against 516 completed coordinates. Plan SHA-256 is `78f695c76e508e0c7210f60ea01717b03c677581f589c3b69bb277f187e7c47b`.
- Evidence: Completion SHA-256 `863eb92646e5ae78d6d2bd72f6c9e5d24acfb311b069e17c83c4fd94894eb2cc`; stage-summary SHA-256 `00f21ab09e52be9a53ce54c34f5b5d2a8f6c695b3829e4b95029b8db7546def0`; immutable-evidence SHA-256 `d3d1110bb9755865028ede4305171cda9e48a89ab95abb193acb6107e0e3ac47`; delivery-evidence SHA-256 `af17ff903b66877bcebae41d1bc7e1e6f041847b091945a573f17c8df2059160`; Round-2 and Round-3 records under team artifacts.
- Limits: Results are in-sample and dependent on the same instrument, period, scenario, and inspected grid. Gap-excluded return remains display-only and has no role in rankings or the refinement design. The approved-in-principle 72-coordinate design awaits B's post-delivery identity rehash.
- Handoff paths: `results\campaigns\v4_3_max_w_multiround_20260802\round_02_all_window`, `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\A_v4_3_max_w_round_02_20260802.md`, and `A_v4_3_max_w_round_03_20260802.md` in the same artifact directory.

## 2026-08-02 — Max-completed-W Round-1 smoke result and broad Round-2 decision

- Objective: Evaluate the broad W jump under four independent current views and continue only when the declared evidence guards support another round.
- Change: Round 1 completed 12 `all_window` coordinates and 1,196 trades under raw fingerprint `da15e3965696637b55ac582d58d8fdd0eea54e2ae0c814bf2e444edd3f0adfec`. A narrow post-run analyzer repair accepted the exact audited plan status and reason-specific same-signal max-W source boundary without changing plan, engine, runner, raw files, fingerprint, or templates.
- Outcome: One coordinate qualifies Scenario 1. The unrestricted total-return leader is 48.0851%; the two average-return views share a 0.489439% leader because all rows exceed 20 trades. These are in-sample smoke-probe leaders, not accepted parameters. The approved Round-2 design expands to a deterministic 504-coordinate E/BH/TRW/K/W/M Cartesian grid; its executable plan is frozen pending audit.
- Evidence: Immutable closure artifact SHA-256 `935fd5e562b54deb8d8334251753a5b04e2ed3b71148d8ad00a9f5137dcdc216`; completion manifest SHA-256 `3a3428b5d5f8b8f24db07b0d8f8ed733a03a48c6950e7a3e63196c7df4e273ca`; corrected delivery QA SHA-256 `d0f88bc6a07c0ec0b5f8a298998aeee900a5a8a47b3dc9c581202bde14703b7c`; Round-2 plan SHA-256 `5735c12e439c8686f1f8e2e5e5d4ffcdb956902aa21825619f5c6d9ab4eebe85`; current source manifest SHA-256 `306bf83ca50dab913fc7d8f081ec249dc05e03529ced7ce1c6dc58addab3f946`.
- Limits: Results are in-sample, use one instrument/period and `all_window`, and contain 424 gap-spanning trades. Gap-excluded return is a display-only gap-dependence audit and cannot change qualification, ranking, continuation, or candidates. Round-1 fixed-template stage/cumulative delivery and responsive browser QA are closed.
- Handoff paths: `results\campaigns\v4_3_max_w_multiround_20260802\round_01_all_window`, `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\A_v4_3_max_w_round_01_20260802.md`, and `research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_max_w_multiround_20260802_round_02_all_window.json`.

## 2026-08-02 — Max-completed-W Round-1 plan materialized

- Objective: Isolate the new causal max-completed-W rebound basis in a bounded, auditable campaign without merging ranking objectives.
- Change: The engine/runner identity now binds raw schema 6, fingerprint schema 7, max-completed-W rebound policy, audit schema v2, and new strategy/result semantics. Round 1 contains 12 `all_window` coordinates: three provenance geometries crossed with `W={3,12,48,192}`. Four ranking views remain independent: Scenario-1-qualified total return, unrestricted total return, unrestricted average return at `>=10` trades, and unrestricted average return at `>=20` trades; each uses `combo_id` after its primary metric.
- Outcome: The Round-1 executable plan froze at 12 unique combo IDs after final product-code/source-manifest closure. Its later execution and delivery outcomes are recorded in the newer smoke-result entry above. Two runner portability defects were repaired; B independently reproduced 61 passed and 2 result-fixture skips after C froze delivery code.
- Evidence: Final plan SHA-256 `36f9f47fd1c49ac1be68abf7a0f146b81d667c115123b385dde31239daccdc80`; final source-manifest SHA-256 `78fcea5452769fe14f531c4e3036fe7ba3b2513092a0ad1e43a5b7d495c949ec`; source-closure artifact SHA-256 `aaf52d1e445a176c540fc818bb162ea49f6128f21f6b42682e2979dd9b38a5bb`; design and Round-1 records under the team artifacts directory.
- Limits: This entry records the pre-run boundary and is superseded for result status by the smoke-result entry above. Every result remains in-sample and cannot accept a parameter automatically.
- Handoff paths: `research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_max_w_multiround_20260802_round_01_all_window.json` and `.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts\A_v4_3_max_w_round_01_20260802.md`.

## 2026-08-02 — Cross-version scenario manager delivered

- Objective: Replace version-bound scenario lookup with one shared editor based on the exact market selector named by the user.
- Change: A shared HTML saves one or more selected intervals under the next automatic `情景N` name. Archived numbers remain reserved, so deleting `情景2` makes the next unused sequential name `情景3`. Active scenarios appear as a two-row button strip above the chart and reload on click; archive/restore remains available. The title/theme header was removed, CSV/TSV upload moved to the lower action row, and display now defaults to 15-second bars with added 30/60/120-minute choices.
- Outcome: The stable cross-version entry is `D:\Code\backtest-release\shared_tools\scenario_manager\index.html`. The original selector and existing result snapshots remain unchanged.
- Evidence: Automatic save produced `情景1` through `情景5`; after archiving `情景2`, the next label became `情景6`, proving no number reuse. Two-row scenario buttons reloaded two intervals. The default chart rendered 199,200 15-second bars; desktop and 390px checks had zero console/request errors and no overlap, mojibake, abnormal whitespace, or horizontal overflow.
- Limits: The editable library is browser-local to this stable HTML path. A backtest still requires a reviewed immutable scenario definition bound into its plan.
- Handoff paths: `D:\Code\backtest-release\shared_tools\scenario_manager\index.html` and its `README.md`.

## 2026-08-01 — V4.3 execution and portability repair closed

- Objective: Remove synthetic exit fills, preserve the confirmed pending-entry behavior, and make active runtime inputs movable with the V4.3 folder.
- Change: Rebound/speed signals on non-real bars now freeze a pending exit and fill at the next real-trade open. Pending entry continues to fill at the first real-trade open without a retrigger or structural cancellation. Data, preparation artifacts, fixed templates, Plotly, and the market selector are repository-local and hash-bound.
- Outcome: New V4.3 strategy/result/combo/audit identities are active; V4.2 remains untouched; no backtest was run.
- Evidence: `SOURCE_MANIFEST.json`, `runtime_inputs\RUNTIME_INPUTS.json`, package tests, and the generated preparation manifest.
- Limits: Existing V4.2 results are not V4.3 evidence. Parameter exploration requires a new explicit instruction and reviewed plan.
- Handoff paths: `research_variants\short_momentum_net_drop_rebound_v4_3`, `runtime_inputs`, and `RUNTIME.md`.

## 2026-08-02 — Dual baseline sampling and audit repair closed

- Objective: Preserve both confirmed baseline-sampling choices without reviving `tr_average` or mixing their evidence.
- Change: Preparation now publishes a neutral `baseline_excluded` marker plus `eligible_if_excluding_marked`. `all_window` remains the default; `exclude_marked` omits marked finite TR15 atoms and backfills older eligible atoms inside one continuity segment. Policy now enters combo, strategy, result, plan, fingerprint, stage, batch, completion, analysis, catalog, and cumulative identities. Rankings are partitioned by policy and `rolling_tr_sum`.
- Outcome: Current UI sources remove `tr_average`, include the `>=10` compatibility filter, and expose the two sampling choices through the established templates. Duplicate datetimes fail. Pending low-activity audit counts use the actual pending state. Preparation/runtime/source manifests are rebound. No parameter backtest was run.
- Evidence: Current source and runtime manifests, regenerated preparation artifacts, package tests, and the synthetic-baseline diagnostic described below.
- Limits: Neither policy removes every synthetic bar. The verified real-only comparison is diagnostic only and does not define a third executable policy.
- Handoff paths: `research_variants\short_momentum_net_drop_rebound_v4_3`, `runtime_inputs\data_preparation`, and `project_management\index.html`.

## 2026-08-01 — Project-management record initialized

- Objective: Establish reusable bilingual project memory.
- Outcome: Created the neutral management structure and Dashboard inputs.
- Evidence: Initialization report and generated files.
- Limits: This initialization entry is superseded for implementation status by the V4.3 closure above.

## Entry format

Add new entries newest first with these fields when they are useful:

- Objective: What the completed or materially changed work was meant to accomplish.
- Change: What became different.
- Outcome: The confirmed result or current state.
- Evidence: Tests, files, commands, artifacts, or explicit user confirmation.
- Limits: What the evidence does not establish.
- Handoff paths: Current files or artifacts needed to continue or review the work.

## Maintenance

- Append after completed work or a material change that future work needs to understand.
- Keep entries concise and link to retained evidence instead of copying large output.
- Update `CURRENT_TASKS.en.md` for forward-looking status and `CURRENT_VERSION.en.md` for current behavior.
- When a major delivery version or working method changes and older detail would mislead normal work, move the superseded detailed body into a dated folder under `99_archive`, leave a concise restoration pointer here, and keep the archived material intact.
