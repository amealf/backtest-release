# Round 2 Interpretation and Terminal Round 3 Design

Status: final. Immutable raw closure, fixed-template stage delivery, two-stage cumulative publication, hash QA, browser QA, and manual visual review are complete. This memo authorizes freezing one terminal Round-3 JSON; B retains exclusive ownership of validate-only, exact anti-join, resource gates, and raw stage materialization.

## Evidence identity

- Round-2 plan SHA-256: `f90dbf5563ae9128304d5b48b902db440d9b73a4af3c0a7654079ef73628f7fd`
- Round-2 plan fingerprint: `7ad95dbd7ba9ebc1faffd8cbc1723211273453af0471ab950e5b3d798ee6c4e8`
- Round-2 completion manifest SHA-256: `b97a4811ebf3520fb6086ec26d5dfa149b2154d989ca6d94a1149f8d7a28350c`
- Round-2 stage analysis manifest SHA-256: `be731db4bc274b85271ea5bb26421aba0e8f8b8f538825e3925e5d459aa47164`
- Published cumulative snapshot: `dde99537b4584f0d5d98a70e388cacffd226736a455963a2f54acd47b4bfd847`
- Cumulative current-pointer SHA-256: `2ec7c4b3d820c9f46c35c8a7d82be11433ee1f45af217343ecc18f8b97e5a810`
- Cumulative analysis/completion/trade-review manifest SHA-256: `0d8cb2d45ff7b89515d6ddc10f2bd04b110be3cd1a0b64a65320d59d01980634` / `9fbc16780c2ef9dd953e571c15c9e07ff650badcea4a208387b3d19bf27f6a03` / `1510975149a6617de69b4fc25ae284fb2260b45a27e2634b358e6a4daafe821a`
- Round-2 closed scope: 247 coordinates, 20,629 trades, 21 batches, `all_window`.
- Active cumulative scope: 619 coordinates, 337,027 trades, two compatible stages.
- Round-2 stage audit: 211 waited entries, maximum wait 101 bars; 3,963 rebound exits, 16,538 speed exits, 128 segment-end exits; 8,744 gap-spanning trades; median/P95 holding distance 797 / 2,029 bars.
- Qualification: 51 coordinates qualify Scenario 1, 6 qualify Scenario 2, and zero qualify Scenario 3. Scenario 3 remains diagnostic and is not a campaign target.
- Cost-positive coordinates: 139 of 247 under the selected 3.56-bps per-trade shadow model.
- Delivery QA: 631 hash/size checks with zero mismatches. Browser QA passed 320 stage states, 400 snapshot states, and 400 stable-route states, with zero runtime errors, external requests, or layout failures and six screenshots per surface. Manual review found no display defects.

## Objective 1 — Scenario-1-qualified cost-adjusted total return

Round-2 leader: E320/BH720/TRW24/K1.25/W2/M6/S400.

- Gross total return: 27.9590%.
- Cost-adjusted total return: 25.5775%.
- Strict improvement over the Round-1 Scenario-1 seed: +8.5617 percentage points cost-adjusted.
- Gross/cost-adjusted average trade: 0.4864% / 0.4508%.
- Gross/cost-adjusted maximum drawdown: 7.2411% / 7.5404%.
- Trades: 53; modeled cost drag: 2.3816 percentage points.
- Gap audit: 12 gap-spanning trades; gross return excluding those trades is -0.3462%. Dependence remains slightly negative but is materially lower than the Round-1 seed's -5.0089%; this field remains display-only.
- Execution/lifecycle: 2 waited entries, maximum wait 41 bars; 7 pending exits; median/P95 holding distance 528 / 1,450.8 bars; 23 rebound and 30 speed exits.
- Available-prefix audit: all 53 trades record a final governing source length of W=2; no shorter governing prefix remains at exit for this coordinate.

Interpretation: the Scenario-1 branch earns one terminal refinement because it made a strict primary-metric improvement. W=2 is interior to the tested W={1,2,4} set. The leader presses the lower tested M=6 edge and the upper tested S=400 edge, which are new and specific unresolved boundaries. The Round-2 stop clause for M=12 or S=240 does not fire.

## Objective 2 — unrestricted cost-adjusted total return

Round-2 leader: E40/BH720/TRW6/K2/W48/M4/S400.

- Gross total return: 40.6708%.
- Cost-adjusted total return: 36.0556%.
- Strict improvement over the Round-1 unrestricted seed: +13.4369 percentage points cost-adjusted.
- Gross/cost-adjusted average trade: 0.3954% / 0.3598%.
- Gross/cost-adjusted maximum drawdown: 16.3560% / 16.9845%.
- Trades: 94; modeled cost drag: 4.6152 percentage points.
- Gap audit: 41 gap-spanning trades; gross return excluding those trades is +0.3523%, versus -13.9397% for the Round-1 seed. This substantially reduces, but does not eliminate, concentration in gap-spanning trades.
- Execution/lifecycle: 2 waited entries, maximum wait 81 bars; 15 pending exits; median/P95 holding distance 660 / 1,726.7 bars; 11 rebound, 82 speed, and 1 segment-end exit.
- Available-prefix audit: 2 of 94 trades record a final governing source shorter than requested W=48.

The next four cost rows remain under the same E40/BH720/TRW6/K2 entry geometry. W/M vary across W={24,96,192} and M={2,4}, while S=400 occupies the leading five rows. W48 and M4 are interior to the tested local sets, but S=400 is the lower tested boundary. The branch earns one terminal refinement because it made a strict primary-metric improvement and exposes a specific lower-S boundary.

## Broad-branch stop and gross/cost interpretation

The 144-coordinate Round-2 broad block did not beat either primary-objective leader. Its best cost-adjusted row was E320/BH240/TRW24/K1/W48/M8/S600 at 9.0957%, with 70 trades, 16.1592% cost-adjusted maximum drawdown, and -11.4795% gross return after excluding gap-spanning trades. Only 4 of 144 broad coordinates qualified Scenario 1 and 42 were cost-positive.

Therefore the broad branch stops. Round 3 contains no new broad entry map. The cumulative gross leader remains the Round-1 high-turnover coordinate whose +91.7995% gross return becomes -61.0935% after modeled costs; it remains excluded from cost-adjusted continuation.

## Terminal Round 3 — objective-specific local design

Planned new-coordinate maximum before B's external anti-join: 212, split into 113 Scenario-1 coordinates and 99 unrestricted coordinates. An in-memory tuple audit found 212 unique coordinates, zero internal duplicates, and zero overlap with the frozen Round-1 and Round-2 plan grids. All blocks remain `all_window`, raw `entry_slippage=0`, combined exit, three workers, batch size 12, and the 4,096 MiB free-memory floor.

### R3-S1-LOCAL — 113 coordinates

Entry geometry remains E320/BH720/TRW24/K1.25.

- S={360,440,520,560}, W={1,2,3,4}, M={4,5,6,7,8}: 80.
- S=480, W={1,2,3,4}, M={4,5,6,7}: 16.
- S=480, W={2,3,4}, M=8: 3.
- S=400, W={1,2,4}, M={4,5,7}: 9.
- S=400, W=3, M={4,5,6,7,8}: 5.

The known W1/M8/S480 Round-1 coordinate and all Round-2 coordinates are omitted. This resolves the new lower-M and upper-S boundaries while interpolating W=3 around the interior W=2 leader.

### R3-UR-LOCAL — 99 coordinates

Entry geometry remains E40/BH720/TRW6/K2.

- S={280,320,360,440}, W={32,40,48,56,64}, M={3,4,5,6}: 80.
- S=400, W={32,40,56,64}, M={3,4,5,6}: 16.
- S=400, W=48, M={3,5,6}: 3.

The known W48/M4/S400 Round-2 leader is omitted. This expands below S=400 and interpolates W/M around the interior W48/M4 leader.

## Terminal stop contract

- Round 3 is terminal for this bounded in-sample campaign, even if a new boundary leader appears.
- Interpret only after immutable raw closure, exactly one fixed-template delivery, two-stage-plus-terminal cumulative publication, hash QA, browser QA, and visual QA.
- Rank the Scenario-1 and unrestricted objectives independently in cost-adjusted mode by default; retain gross as the alternate view and never form a combined score.
- Report strict improvement or no improvement against the Round-2 leaders. Do not create Round 4 from this campaign.
- Preserve gap-excluded return as a display-only dependence audit. It cannot rerank, qualify, or authorize continuation.
- No parameter is accepted. All results remain in-sample and conditional on the selected 3.56-bps shadow-cost model.

## Active-campaign delivery boundary

The cumulative lineage remains scoped to `results\campaigns\v4_4_cost_adjusted_multiround_20260803`. The older temporary validation campaign and failed partial snapshot remain preserved outside the active union. Identity disagreement within the active campaign must still fail closed.
