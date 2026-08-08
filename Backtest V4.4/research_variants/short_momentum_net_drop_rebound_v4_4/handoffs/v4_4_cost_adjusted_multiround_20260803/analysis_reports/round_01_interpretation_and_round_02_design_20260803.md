# Round 1 Interpretation and Round 2 Design

Status: final. Immutable raw closure, stage delivery, cumulative recovery, and browser/hash QA are complete. This memo authorizes freezing the bounded Round-2 JSON; B still owns validate-only, anti-join, resource gates, and materialization.

## Evidence identity

- Plan SHA-256: `1424dc17862a2bfe0b8f0439fef061e64efc487c5057b7cff64498ed40a78046`
- Plan fingerprint: `1fd03c4a6fa54d97ed455ba90dde5ccb6f65c4b08a7397795c832483120ccea1`
- Completion manifest SHA-256: `c9532f77b626f647dcfe7b1fdc09ee76b2b895e851da0b98f6206b26ff1e6539`
- Stage analysis manifest SHA-256: `3c55f1222db2586f8e5fb4dee5800e4534ee695c9ce7e101fd9a7e5f7c56d03f`
- Published cumulative snapshot: `2020ad7b12d57889f1c1d0cf69f981bcf2b5e3ec5b8a4808c196dbb6cdd51d47`
- Cumulative current-pointer SHA-256: `24ae69b6d57becb6931d79ee630e7b9d0a91e81821a32d38fe8ee76834a784a6`
- Cumulative analysis/completion/trade-review manifest SHA-256: `dbaccce3bc61ea3979c2340ddd9616e82c01a0261bea25a6998ffdd2ac4534db` / `46f70611a95f57a7f44c22310d1d4432cc63a9e47b089090f879f760643d4a0a` / `a51dff3aa3855073712d90b9cec39c1aaf4c52be4bacb6c71b2aff17ca8293c2`
- Closed scope: 372 coordinates, 316,398 trades, 31 batches, `all_window`.
- Stage audit: 1,823 waited entries, maximum wait 110 bars; 305,767 rebound exits, 10,487 speed exits, 144 segment-end exits; 9,691 gap-spanning trades; median holding distance 2 bars and P95 300 bars.
- Qualification: 41 coordinates qualify Scenario 1; zero qualify Scenario 3. Scenario 3 is diagnostic and is not a target in this campaign.
- Cost-positive coordinates: 127 of 372 under the selected 3.56-bps per-trade shadow model.

## Objective 1 — Scenario-1-qualified cost-adjusted total return

Leader: E320/BH720/TRW24/K1.25/W1/M8/S320.

- Gross total return: 19.6206%.
- Cost-adjusted total return: 17.0157%.
- Gross/cost-adjusted average trade: 0.3099% / 0.2743%.
- Gross/cost-adjusted maximum drawdown: 9.7432% / 10.6737%.
- Trades: 62; modeled cost drag: 2.6049 percentage points.
- Gap audit: 15 gap-spanning trades; gross return excluding those trades is -5.0089%. This is strong gap dependence and remains display-only.
- Execution/lifecycle: 2 waited entries, maximum wait 41 bars; 6 pending exits; median/P95 holding distance 376.5 / 1,047.9 bars; 32 rebound and 30 speed exits.
- Scenario evidence: one entry during Market 1 at 2026-06-23 14:32, held past the segment end and exited by speed at 19:20. Markets 2 and 3 have no entry for this coordinate.

Interpretation: this is the current Scenario-1 objective seed, not an accepted parameter. The top five Scenario-1 rows all use M=8 and S=320, so the objective presses the tested upper M and lower S boundaries. W is mixed across the top five, although the leader itself is at W=1. Round 2 should refine W/M/S while extending M upward and S downward once.

## Objective 2 — unrestricted cost-adjusted total return

Leader: E40/BH720/TRW6/K2/W192/M2/S480.

- Gross total return: 27.6884%.
- Cost-adjusted total return: 22.6187%.
- Gross/cost-adjusted average trade: 0.2368% / 0.2012%.
- Gross/cost-adjusted maximum drawdown: 15.0023% / 15.8205%.
- Trades: 114; modeled cost drag: 5.0697 percentage points.
- Gap audit: 45 gap-spanning trades; gross return excluding those trades is -13.9397%. The leader is materially gap-dependent and does not qualify Scenario 1.
- Execution/lifecycle: no waited entries; 10 pending exits; median/P95 holding distance 621.5 / 1,598.35 bars; 52 rebound, 61 speed, and 1 segment-end exit.
- Available-prefix contract: 81 of 114 trades record a final governing W source shorter than requested W=192, so the confirmed available-prefix rule is economically material for this seed.

The second-ranked unrestricted cost row is E40/BH720/TRW6/K2/W48/M8/S480 at 22.5857%, only 0.0330 percentage points behind. The top five all share E40/BH720/TRW6/K2, while W/M vary. Round 2 should separately map the strong entry geometry and refine the W/M/S ridge.

## Gross-versus-cost reversal

The gross leader is E20/BH720/TRW3/K1.25/W1/M0.25/S960:

- 4,481 trades.
- Gross total return: +91.7995%.
- Cost-adjusted total return: -61.0935%.
- Gross/cost-adjusted average trade: +0.01458% / -0.02102%.
- Gross/cost ranks: 1 / 357.
- Cost-adjusted maximum drawdown: 61.0935%.

This is the decisive Round-1 finding: the selected cost model reverses the high-turnover gross frontier. Gross remains a required alternate view, but it cannot seed the cost-adjusted continuation.

## Round 2 — concurrent broad and local design

Planned new-coordinate maximum before the external anti-join: 247. All blocks remain `all_window`, raw `entry_slippage=0`, combined exit, three workers, batch size 12, and the 4,096 MiB free-memory floor.

### R2-BROAD — strong-region entry map (144 coordinates)

- E: 20, 40, 80, 160, 320, 640
- BH: 240, 480, 720
- TRW: 3, 6, 12, 24
- K: 1.0, 2.0
- W: 48
- M: 8.0
- S: 600

Purpose: determine whether the shared E40/BH720/TRW6/K2 unrestricted entry geometry persists away from the anchored Round-1 exit blocks and whether broader Scenario-1-qualified regions appear. S=600 avoids repeating Round-1 coordinates while remaining between tested 480 and 720 scales.

### R2-S1-LOCAL — Scenario-1 W/M/S refinement (47 coordinates)

Seed fixed at E320/BH720/TRW24/K1.25.

- Main block: W={1,2,4}, M={6,8,10,12}, S={240,280,400}: 36.
- S=320, M={6,10,12}, W={1,2,4}: 9.
- S=320, M=8, W={2,4}: 2.

The exact Round-1 seed W1/M8/S320 is omitted. This performs the one permitted outward expansion for upper M and lower S while testing whether the W=1 edge persists.

### R2-UR-LOCAL — unrestricted W/M/S ridge refinement (56 coordinates)

Seed entry geometry fixed at E40/BH720/TRW6/K2.

- S={400,560}, W={24,48,96,192,288}, M={1.5,2,4,8}: 40.
- S=480, W={24,96,288}, M={1.5,2,4,8}: 12.
- S=480, W={48,192}, M={1.5,4}: 4.

The four Round-1 overlaps at S480 with W={48,192} and M={2,8} are omitted. The block tests whether the 0.033-percentage-point W192/M2 versus W48/M8 near-tie is a ridge rather than a point result.

## Round-2 continuation and stop decisions

- Continue both objective branches because each has eligible positive cost-adjusted evidence and neither has received a local refinement round.
- Continue one broad branch because both leaders came from anchored/cross-regime blocks whose entry neighborhoods were not jointly mapped.
- Do not seed any continuation from gross leaders that are cost-negative.
- After Round 2, stop a local branch without strict cost-adjusted primary-metric improvement or if the same interior leader persists in the next refinement.
- If the Scenario-1 leader still presses the newly extended M=12 or S=240 boundary, report unresolved boundary dependence and stop extending that dimension.
- Gap dependence remains a required warning and display-only audit; it cannot rerank, qualify, or stop an objective by itself.
- No parameter is accepted. All evidence remains in-sample and conditional on the selected shadow-cost model.

## Delivery recovery boundary

The original exactly-one delivery invocation completed stage analysis, then rejected cumulative union because the historical temporary validation campaign has a different engine hash. The active multi-round cumulative lineage is explicitly scoped to `results\campaigns\v4_4_cost_adjusted_multiround_20260803`; the old temporary campaign remains preserved evidence outside that union. Cumulative-only recovery is authorized without rerunning stage analysis or deleting the failed partial snapshot.

Recovery completed and atomically published the active-campaign snapshot. The failed status and partial snapshot remain preserved. Stage, snapshot, and stable-route browser QA each exercised 200 states with zero runtime errors, external requests, or layout failures and six screenshots. Cost-adjusted and gross ordering, displayed metrics, and rank headers all changed together as required. Manual review found no overlap, garbled text, broken alignment, or asymmetry.
