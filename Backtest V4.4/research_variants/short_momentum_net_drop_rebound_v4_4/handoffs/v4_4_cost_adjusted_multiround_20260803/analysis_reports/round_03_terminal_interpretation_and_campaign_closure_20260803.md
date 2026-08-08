# Round 3 Terminal Interpretation and Campaign Closure

Status: final and canonical. Terminal raw closure, fixed-template stage delivery, three-stage cumulative publication, hash QA, browser QA, and manual visual review are complete. The bounded V4.4 in-sample campaign ends here. Round 4 is prohibited and no parameter is accepted.

## Evidence identity

- Round-3 plan SHA-256: `46c95b24feab49b6f260a0e8f1e1125fd74c34c6a0e268b89e0e1fb83a6d9b8c`
- Round-3 plan fingerprint: `6d40e4a35562bbbb16347a63b48442f132025bbc4695f0ac87bc4312eae0955e`
- Round-3 completion manifest SHA-256: `edec2c43ecf4c4035a690f763b6a4d68d8be8f97a9da40ec9b3f3aac20ff25ea`
- Round-3 stage analysis manifest SHA-256: `d163c36e471c732181f93d85d3e9752f8938b3cb14d92c025648e53815044d2f`
- Published final cumulative snapshot: `0fb3e1e5e8ef890f3b225db46288fa4b3957bcb88c7ca2dff72d750679db6922`
- Cumulative current-pointer SHA-256: `f91f5eaee14f4b92d91f2b3b28150a702731c4968c557c7302fa30f22a6478fc`
- Cumulative analysis/completion/trade-review manifest SHA-256: `d8dca67c5631fa3da3cb0dc7630b1f60fd19ec0b1e6368ac84a3242ea5aa13e7` / `32adc2941a99f2c4733dc2ed7da84d44823f903cc3e5958b9ba5443ca3a7f17c` / `06b71481a7348c437807eb3664232948a8c91d81141ccd87cb5fe5606aaf0780`
- Round-3 closed scope: 212 coordinates, 16,847 trades, 18 batches, `all_window`.
- Final active-campaign scope: 831 coordinates, 353,874 trades, three compatible stages.
- Round-3 stage audit: 384 waited entries, maximum wait 81 bars; 5,085 rebound exits, 11,683 speed exits, 79 segment-end exits; 5,593 gap-spanning trades; median/P95 holding distance 535 / 1,404 bars.
- Round-3 qualification: 133 Scenario-1-qualified, zero Scenario-2-qualified, zero Scenario-3-qualified, and 206 cost-positive coordinates.
- Final cumulative qualification: 225 Scenario-1-qualified, 74 Scenario-2-qualified, zero Scenario-3-qualified, and 472 cost-positive coordinates.
- Delivery QA: 843 hash/size checks with zero mismatches. Browser QA passed 360 terminal-stage states, 520 snapshot states, and 520 stable-route states, with zero runtime errors, external requests, or layout failures and six screenshots per surface. Cost/gross sorting, displayed returns, and rank headers were verified. Manual review found no display defect.

## Final Objective 1 — Scenario-1-qualified cost-adjusted total return

Final leader: E320/BH720/TRW24/K1.25/W4/M4/S400, from Round 3.

- Gross total return: 33.3830%.
- Cost-adjusted total return: 30.6696%.
- Strict improvement over the Round-2 leader: +5.0921 percentage points.
- Total improvement over the Round-1 leader: +13.6538 percentage points.
- Gross/cost-adjusted average trade: 0.5159% / 0.4803%.
- Gross/cost-adjusted maximum drawdown: 7.4152% / 7.8125%.
- Trades: 58; modeled cost drag: 2.7134 percentage points.
- Gap audit: 11 gap-spanning trades; gross return excluding those trades is +0.6431%. This is the first Scenario-1 campaign leader with positive gap-excluded gross return, but the field remains display-only.
- Execution/lifecycle: 2 waited entries, maximum wait 41 bars; 6 pending exits; median/P95 holding distance 456.5 / 1,395.3 bars; 32 rebound and 26 speed exits.
- Available-prefix audit: 3 of 58 trades record a final governing source shorter than requested W=4.

Interpretation: the terminal Scenario-1 branch improved materially. W=4 and M=4 sit on the terminal tested boundaries, while S=400 is interior to the terminal speed set. The boundary observation is reported, not extended: Round 3 is terminal and cannot authorize Round 4. This coordinate is an in-sample descriptive leader, not an accepted parameter.

## Final Objective 2 — unrestricted cost-adjusted total return

Final cumulative leader remains the Round-2 coordinate E40/BH720/TRW6/K2/W48/M4/S400.

- Gross total return: 40.6708%.
- Cost-adjusted total return: 36.0556%.
- Gross/cost-adjusted average trade: 0.3954% / 0.3598%.
- Gross/cost-adjusted maximum drawdown: 16.3560% / 16.9845%.
- Trades: 94.
- Gap audit: 41 gap-spanning trades; gross return excluding those trades is +0.3523%.
- Execution/lifecycle: 2 waited entries, maximum wait 81 bars; 15 pending exits; median/P95 holding distance 660 / 1,726.7 bars; 11 rebound, 82 speed, and 1 segment-end exit.
- Available-prefix audit: 2 of 94 trades record a final governing source shorter than requested W=48.

The best new Round-3 unrestricted coordinate is E40/BH720/TRW6/K2/W40/M4/S400 at 32.8804% cost-adjusted, 3.1752 percentage points below the Round-2 leader. It has 96 trades, 17.0761% cost-adjusted maximum drawdown, and -2.4247% gross return after excluding gap-spanning trades. Therefore the terminal unrestricted refinement does not improve its primary metric, and the Round-2 leader remains final descriptive evidence.

## Gross-versus-cost terminal finding

The final cumulative gross leader remains E20/BH720/TRW3/K1.25/W1/M0.25/S960 from Round 1:

- 4,481 trades.
- Gross total return: +91.7995%.
- Cost-adjusted total return: -61.0935%.
- Gross/cost-adjusted average trade: +0.01458% / -0.02102%.
- Gross maximum drawdown: 2.9940%; cost-adjusted maximum drawdown: 61.0935%.
- Final cumulative ranks: gross 1, cost-adjusted 816 of 831.

The selected 3.56-bps per-trade model therefore continues to reverse the high-turnover gross frontier. Gross remains an alternate display and audit mode; it does not determine either cost-adjusted campaign objective.

## Campaign progression

- Round 1 mapped 372 broad coordinates. Its Scenario-1 and unrestricted cost-adjusted leaders were 17.0157% and 22.6187%.
- Round 2 added 247 concurrent broad-plus-local coordinates. Local leaders improved to 25.5775% and 36.0556%; the 144-coordinate broad branch failed to improve either objective and stopped.
- Round 3 added 212 terminal local coordinates. Scenario 1 improved to 30.6696%; unrestricted did not improve and retained its Round-2 leader.
- Final cumulative scope is 831 unique coordinates and 353,874 trades. The active cumulative contains exactly the three compatible multi-round stages. The older temporary validation campaign, prior snapshots, and failed partial snapshot remain preserved outside or behind the final stable routes.

## Final decision and validity boundary

- The bounded V4.4 in-sample multi-round campaign is complete.
- Round 4 is prohibited. Remaining W/M boundary pressure cannot authorize more coordinates in this campaign.
- No parameter is accepted, promoted, or declared production-ready.
- The final leaders are conditional on one instrument, one observed period, the unchanged V4.4 method contract, and the selected 3.56-bps HKD-notional shadow-cost model.
- Gap-excluded return remains a display-only dependence audit and never reranks or qualifies coordinates.
- Any out-of-sample validation, alternative cost model, alternative baseline policy, or production selection requires a new explicit user decision and a separately scoped plan.

## Preserved method and delivery contracts

- W uses available prefixes 1..W; an early prefix maximum may govern a trade. No full-W or minimum-ratio gate applies.
- The exact candidate is `w_open_to_end_low_drop = open[start] - low[end]`, not an internal maximum ordered decline.
- Pending entry is retained-signal `fill_first_real_open` for at most 120 continuous candidate bars, without trigger recross or structural cancellation.
- Cost-adjusted remains the default ranking/display mode; gross remains optional. Switching changes sorting, displayed returns, and the rank header together.
- Raw fills and raw returns remain unchanged; the 3.56-bps model is derived analysis only.
- The final cumulative lineage remains scoped to campaign `v4_4_cost_adjusted_multiround_20260803`; identity disagreement within that campaign must fail closed.
