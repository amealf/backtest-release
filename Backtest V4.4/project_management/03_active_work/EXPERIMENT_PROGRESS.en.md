# Experiment Progress

## 2026-08-07 — Current K200 optimal-parameter initial later-month replay closed

- Design: one schema-4 exact temporal transfer over `2026-07-08 23:52:15` through `2026-08-07 03:21:45`; 100 training-only multi-metric candidates; four workers; no target metric in selection; no intermediate or final HTML.
- Candidate authority: the corrected freeze requires positive cost-adjusted training return. It retains six previously evaluated headline controls and adds 94 exact coordinates not previously run over the later month. The period itself is reused, so the result is parameter-level new evidence and period-level post-hoc evidence.
- Closure: 29/100 later returns are positive, 28/100 are positive with at least ten trades, and 8/100 have positive non-gap return. Median later return is -0.671%; mean is -1.049%; median drawdown is 13.048%; train/later Spearman is -0.346.
- Interpretation: the high-total branch around E320–432/BH200–240/TRW18–23/W6/M4.5 is positive but gap-dependent. The BH612/S308 low-frequency branch contributes the few non-gap-positive results, with only 8–15 trades and high positive-return concentration. Current training leaders do not transfer reliably; `parameter_acceptance=none`.
- Superseded attempt: the earlier root without `_v2_` admitted twelve cost-negative training candidates through an unadjusted non-gap queue. It remains historical, invalid-for-decisions evidence and is replaced by the corrected run.

## 2026-08-07 — K200 train/test/SI triple comparison closed

- Design: retain the completed 250-candidate SI population; select and freeze 100 additional candidates from closed K200 temporal evidence without reading their SI outcomes; evaluate the new freeze on SI; replay all 350 over the K200 test interval. The final page displays K200 training, K200 test, and SI return together.
- Closure: K200 test contains 34,248 trades and 199 positive candidates. SI contains 48,022 trades and 275 positive candidates. Three-column positives total 149; the new 100 contribute 72 K200-test positives, 59 SI positives, and 43 three-column positives.
- Rank evidence: overall Spearman is -0.349 for train/test, +0.319 for train/SI, and +0.003 for test/SI. The new 100 were selected with K200 temporal evidence, while the old 250 had prior SI selection; no candidate has three pristine unseen columns.
- Interpretation: a seven-point neighborhood around E320/BH240/TRW12/K1.25/W6, M4.25–4.75, S340–370 has median returns +50.123%/+8.827%/+14.350%, but median K200-test non-gap return is -12.213%. Preserve the strategy skeleton and test short-window re-estimation on future unseen K200 data; do not accept a static parameter. `parameter_acceptance=none`.

## 2026-08-07 — K200 four-slice temporal migration closed

- Design: training source `2026-05-26 00:00:00` through `2026-07-08 23:52:00`; test begins at the next 15-second bar. R1 uses 400 source-only multi-metric candidates. R2 and R3 freeze repeated, neighboring, and diverse source candidates from closed earlier slices. R4 is frozen before the final unseen week. The full replay evaluates the unchanged R4 freeze and is descriptive.
- Closure: R1/R2/R3/R4 each contain 400 coordinates and 18,886/15,006/13,985/12,505 trades. Their positive-coordinate counts are 296/26/383/25. The full replay contains 400 coordinates and 62,375 trades; 196 are positive over the aggregated interval.
- Stability: 218 candidates have four-slice coverage and two are positive in every slice. Those two have 11 and 13 full-test trades, worst-slice returns of +0.048% and +0.160%, and 100% median Top-2 concentration. The best at-least-20-trade three-of-four candidate is E30/BH456/TRW9/K0.8/W32/M0.5/S144: +10.661% full-test return, -3.095% worst slice, and 7.566% drawdown.
- Transfer diagnosis: training/full-test Spearman is -0.26169. The training leader E480/BH171/TRW12/K1.26/W7/M4.5/S388 records -1.567% full-test return and 14.902% drawdown. Classification: `no_static_general_parameter_found`; retain regime-sensitive short-window fitting as an untested future direction. `parameter_acceptance=none`.

## 2026-08-06 — Six-hour K200 leap/grid cycle closed

- Execution: four workers completed automated Rounds 20–108, comprising 60 leap rounds and 29 adaptive-grid rounds. They added 31,375 anti-joined coordinates and 10,843,398 trades; the three opening rounds added another 360 coordinates and 109,979 trades.
- Final population: 37,058 unique coordinates, 11,749,606 trades, 109 compatible stages, 19,974 cost-positive coordinates, and 4,406 Scenario-1-qualified coordinates. Duplicate coordinates are zero.
- Headline comparison: the unrestricted total-return leader remains E480/BH171/TRW12/K1.26/W7/M4.5/S388 at 82.4664%; the Scenario-1 leader remains E320/BH240/TRW22/K1/W6/M4.5/S330 at 55.8326%; both minimum-10 and minimum-20 average-return views retain E96/BH612/TRW24/K1.6/W16/M2/S308 at 1.23914% average trade.
- Interpretation: broad exploration repeatedly found cost-positive and promising nonadjacent anchors, but it did not improve a headline champion. Stable local evidence supports M around 4.1–4.25 on the E320/BH170/TRW11/K1.4/W6/S96 branch; the high-total ridge around E480/BH171/TRW12/K1.26/W7/M4.5/S388 remains strongly gap-dependent.
- Delivery: cumulative snapshot `eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5` reused 5,320 chunks and generated 31,738. No SI migration followed because K200 global closure was not established. `parameter_acceptance=none`.

## 2026-08-06 — Six-hour K200 leap/grid cycle authorized; planning active

- Mode: `continuation_search` on the current positive-entry K200 lineage, using the frozen K200 data, method semantics, and cost contract.
- Authorization: approximately six hours unattended, four compute workers, memory monitoring, exact completed+active+pending anti-join, no repeated coordinates, and no intermediate cumulative or per-trade HTML.
- Cycle: multi-round leap search across nonadjacent legal regions; select promising mutually nonadjacent anchors from closed multi-metric evidence; run finite one-parameter grids without a fixed point-count cap; return to leap search; expose summaries for user correction between cycles.
- Current status: authority and planning only. No new round is counted as evidence until its raw stage is `IMMUTABLE_CLOSED` and its compact interpretation is written. `parameter_acceptance=none`.

## 2026-08-06 — Stricter-entry K continuation and unified migration publication closed

- Design: hold E320/BH240/TRW20/W6/M4.5/S332 fixed and test K1.5/K1.6/K1.75. Four workers completed three K200 source coordinates in one batch, producing 209 source trades; the frozen coordinates were then evaluated unchanged on SImain.
- Target result: K1.5/K1.6/K1.75 produced 62/57/51 trades, 23.9157%/16.7475%/20.1076% cost-adjusted return, and 11.3028%/14.3036%/14.6663% maximum drawdown. Trade count keeps falling as K rises, but return remains below K1.4's 35.6004%; K1.6 and K1.75 also worsen drawdown.
- Interpretation: `not_improved_target_peak_at_k1p4`. The tested K curve supports K1.4 as a local SImain migration peak under the fixed coordinate. It does not establish a globally accepted parameter and it does not reverse the mixed K200 evidence.
- Delivery: run `k200_20260526_20260708__simain_20260129_20260223__combined_250_stricter_entry_v54_20260806` ranks 250 compatible candidates together and contains 24,194 target trades. Incremental publication reused 247 target trade chunks and generated three. `parameter_acceptance=none`.

## 2026-08-05 — Continuation Round 15 dual-purpose exploration closed

- Plan: 276 exact new K200 coordinates, divided into 192 broad stratified points and 84 one-parameter refinements. Every refinement block changed one of E, BH, TRW, K, W, M, or S at three new values around one of four strong anchors. Plan SHA-256 is `f4bb715f8da1c57b3afb083657e3f423c52448204a1acc09d6c8ec10474e2d55`; plan fingerprint is `9b49f80e28972524a859332a57b35a5912a9302ede9ada7770a045f8e3d40920`.
- Closure: 276 coordinates, 54,314 trades, and 35 batches. Cost-positive rows are 187; Scenario-1 qualification is 29, Scenario-2 is 53, and Scenario-3 is zero. Five new points extend the cumulative return/drawdown frontier.
- Broad branch: 103 of 192 rows are cost-positive, but neither broad block changes a primary leader. Its single new frontier point is dependency-sensitive and is not carried into local refinement. Classification: `mixed`.
- Refinement branch: one block is `improved`, eighteen are `mixed`, and nine are `not_improved`. E96 raises average trade by 0.08661 percentage points and lowers drawdown by 1.21480 points, while total return falls by 3.21962 points and trades fall from 29 to 25. E128 retains 29 trades and improves total return by 0.47847 points, average trade by 0.01207 points, and drawdown by 0.01585 points. Classification: `improved`.
- Delivery: snapshot `5b4e11b4c137028dc0a33d792a47800c8d792f6125e2cc8d2f5796ec6ef4fa94` contains 5,320 coordinates, 797,020 trades, and sixteen stages. A sixty-coordinate handoff preserves broad coverage plus supported average-E, Scenario-BH, and low-drawdown-W single-axis checks. No parameter is accepted and no next compute is authorized.

## 2026-08-05 — Continuation Round 14 large multiblock exploration closed

- Design: 294 exact new K200 coordinates across twelve evidence-led two-parameter surfaces plus sixty remote sparse controls. The reviewed plan SHA-256 is `2674438ab96a2ddee95b709956a32d58b227f748b877d18f30935d39dd5199ef`; four workers completed 37 batches and 28,577 trades.
- Classification: `improved`. Of 294 new rows, 263 are cost-positive; 47 qualify for Scenario 1, 91 for Scenario 2, and none for Scenario 3. Six new points extend the cumulative return/drawdown frontier.
- Unchanged leaders: unrestricted total-return remains E480/BH171/TRW12/K1.26/W7/M4.5/S388 at 82.4664%; Scenario 1 remains E320/BH240/TRW22/K1.0/W6/M4.5/S330 at 55.8326%.
- Improved leader: both minimum-10 and minimum-20 average-return views now select E112/BH612/TRW24/K1.6/W16/M2/S308 with 29 trades, 38.4486% total return, 1.15253% average trade, and 4.23380% drawdown. Its block `average_w_m_surface` is the only block classified `improved`; five blocks are `mixed` and seven are `not_improved`.
- Trade-level boundary: the new leader retains 29 entries, removes one, adds none, and changes ten retained exit times. Median trade, win rate, MFE, MAE, and MFE retention all improve, but sixteen synthetic-signal trades and seven gap trades remain. The evidence supports a W/M exit-behavior interaction, not a parameter acceptance claim.
- Delivery: cumulative snapshot `db85efb36f3de1c1f8255c6108fb365ad9f3d337f77a8d37a0e0ae41982e5699` contains 5,044 coordinates and 742,706 trades. The proposed fourteen-coordinate handoff has zero completed overlap and remains unauthorized for compute.

## 2026-08-05 — Stricter-entry K batch transferred to SImain

- K200 confirmation: E320/BH240/TRW20/W6/M4.5/S332 fixed; K=1.2/1.3/1.4. Trades fall from the K1.1 anchor's 89 to 83/83/77. Cost-adjusted return falls from 55.0133% to 46.9394%/29.3673%/19.7248%. Source classification: `mixed` for fewer trades but lower return.
- SImain refinement: the same frozen points produce 85/73/67 trades and 16.4678%/33.9182%/35.6004% cost-adjusted return. K1.3/K1.4 exceed the K1.1 anchor's 22.0575% while reducing drawdown from 14.6209% to 11.6094%/11.9132%; non-gap return is 23.7683%/26.2978%.
- Classification: `improved_on_target_mixed_across_instruments`. The evidence supports a target-specific stricter-entry direction but does not support reusing that direction as a K200 rule. No parameter is accepted.
- Delivery: 247 combined candidates, 24,024 SImain trades, 244 reused trade chunks, and three newly generated chunks.

## 2026-08-05 — Combined transfer presentation closed; stricter-entry question authorized

- Presentation union: 244 nonoverlapping candidates from the earlier 180-candidate exact transfer and repaired-source 64-candidate exact transfer; 25,129 K200 trades and 23,799 SImain trades.
- Result: 210/244 SImain candidates are cost-positive; 148 are target-positive stable candidates and 57 are isolated positives. Source/target cost-return rank Spearman is 0.060149.
- Evidence boundary: this is a union of completed exact-transfer evidence, not a new target run. No candidate was generated from target results and no parameter was accepted.
- New authorization: investigate whether greater entry difficulty reduces trade count and improves return quality, using small multi-point directional batches and one final HTML publication after the series.

## 2026-08-05 — Repaired-source K200 to SImain transfer closed

- Source freeze: repaired snapshot `0126cd77b436aef1434e7072bac0d6dfa15b3d2ad4dc2cf1b2fafe936ee1e626`, 4,747 coordinates. The top-20% and minimum-10-trade gates, exclusion of 266 previous transfers and three champions, prior W/M/S-family restriction, and within-family return/threshold/trade-count Pareto rule retain 64 candidates from a 225-coordinate eligible pool. Freeze content SHA-256 is `0623937812e4f669799b7eeca30f9f1d7201d05762a6669cbefcd146e2d50d68`.
- Exact transfer: four workers, SIH6 15-second target, 2026-01-29 through 2026-02-23, fixed 3.57-bps cost. The 64 coordinates produce 6,755 trades; 58 are cost-positive, 29 stable-positive, and 26 isolated-positive.
- Comparison: median target return is 4.5883% and median drawdown is 27.1303%. Return-rank Spearman is -0.45728. The target leader is E320/BH240/TRW21/K1.05/W6/M4.5/S340 at 19.6892%; the K200 leader reaches 1.1311% on SImain.
- Decision: positive-transfer frequency is broad, but rank transfer is unsupported. The repaired-source run remains separate from the historical 266-candidate aggregate. No target-local grid or parameter acceptance.

## 2026-08-05 — K200 one-axis series extended through Round 12

- Rounds 8–12 used four workers, three new points per round, and one changed parameter per round. TRW11/10/9, BH145/137/128, M4.75/5.0/5.5, M4.25/4.0/3.75, and S427/466/520 were all exact new coordinates.
- Best results by round were 61.9046%, 69.8267%, 79.6419%, 65.2421%, and 52.6175% cost-adjusted total return, all below the 82.4352% W6 anchor. Every direction is `not_improved`.
- The final one-time cumulative publication created snapshot `0126cd77b436aef1434e7072bac0d6dfa15b3d2ad4dc2cf1b2fafe936ee1e626`: 4,747 coordinates, 713,886 trades, thirteen compatible stages, and zero duplicate coordinates.
- Decision: TRW12, BH171, and M4.5 remain local fixed-anchor peaks; S388 remains stronger than the tested expansion. W7 remains an isolated total-return leader, W6 remains the stable representative anchor, and `parameter_acceptance=none`.

## 2026-08-05 — New-rules K200 five-round one-axis series closed

- Mode and anchor: `continuation_search` on the repaired K200 lineage from E480/BH171/TRW12/K1.26/W6/M4.5/S388. Five rounds used four workers and changed one parameter at a time with three new points.
- Round 3 BH expansion: BH205/257/480 returned 69.6174%/62.2260%/64.9896% cost-adjusted versus the 82.4352% anchor, with higher drawdown. Classification: `not_improved`.
- Round 4 TRW expansion: TRW13/14/15 returned 71.9266%/67.9187%/62.3312% and drawdown worsened monotonically. Classification: `not_improved`.
- Rounds 5–6 K directions: K1.4/1.5/1.6 returned 71.1573%/66.7014%/66.3850%; K1.15/1.05/0.95 returned 62.0151%/56.4756%/45.1925%. K1.26 is a local fixed-anchor peak, not an accepted parameter. Both rounds: `not_improved`.
- Round 7 W expansion: W7 returned 82.4664%, only 0.0312 percentage points above W6. W8/W9 returned 75.2452%/75.4916%; W7's average trade fell from 0.5618% to 0.5571%, drawdown rose from 15.1770% to 15.2433%, and gap-excluded gross return fell from 4.6795% to 3.2002%. Classification: `mixed`; stop the direction.
- Final evidence: fifteen coordinates, 1,707 trades, and one final publication into snapshot `7528265de87be7e855f8e2c80585c52de95c248f0747ca60c1f3a7bcc5ae81b2`, now 4,732 coordinates, 712,108 trades, and eight stages. No combined score, no next plan, and `parameter_acceptance=none`.

## 2026-08-05 — Fixed-parameter E-window sensitivity closed

- Question: does the strong E=320/BH171/TRW11/K1.4/W6/M4.5/S388 result indicate that its entry-search window is too large?
- Design: hold every parameter except E fixed. Add E=304, 256, 192, 160, 136, 112, 96, 80, 64, 48, 32, 24, and 16; compare with completed E=224, 272, 288, and 320 evidence. Four compute workers were used; raw closure added 13 coordinates and 1,388 trades.
- Result: no smaller E exceeded E=320's 78.2952% cost-adjusted total return. The closest smaller point is E=304 at 75.7622%; E=256 records 74.2774%. Below E=224, returns generally step down and reach 7.8332% at E=16.
- Trade-off: smaller windows can reduce exposure and drawdown. E=64 has 108 trades, 62.3971% cost-adjusted return, and 12.4072% drawdown; E=16 has 71 trades, 7.8332% return, and 9.5025% drawdown. E=320 has 114 trades and 15.1499% drawdown.
- Classification: `not_improved` for return; `mixed` for return-versus-drawdown. The oversized-window hypothesis is not supported under fixed remaining parameters. `parameter_acceptance=none`.
- Delivery: snapshot `b077548e654277738e1d953ce7bea01eb184a0e223f064c7728dbc2de4d1a561`, 4,717 coordinates, 710,401 trades, three compatible stages, shared cumulative main and per-trade entries updated.

## 2026-08-05 — Strict-entry K200 to SImain exact-transfer design accepted

- Mode: `transfer_exact`; no SImain local grid or target-driven candidate generation.
- Source universe: all 4,704 completed K200 coordinates. Eligibility uses the top 20% by fixed-3.57-bps cost-adjusted total return plus at least 10 K200 trades.
- Structure: retain only W/M/S families already represented in the original 180-candidate transfer; vary E/BH/TRW/K through already completed K200 coordinates only.
- Strictness: median actual entry threshold is `median(entry_baseline_value × K)`. Within each W/M/S family, retain points nondominated on higher K200 cost-adjusted return, higher threshold median, and lower K200 trade count.
- Exclusions: original 180 transferred coordinates and the current four primary K200 view champions. Candidate cap is 100; the read-only pre-target audit found 300 eligible unseen non-champions and 86 within-family Pareto candidates across 14 W/M/S families.
- Target contract: reuse SIH6, the 2026-01-29 through 2026-02-23 test interval, prior-day warm-up, zero rolls, the established gap/synthetic/low-activity audit treatment, and 3.57 bps. This exact reuse permits one combined SImain rank table with the earlier batch.
- Evaluation: inspect whether higher source thresholds and lower source trade counts reduce SImain trades while improving SImain cost return, median trade, and drawdown. Retain the aggregate five-dimensional source/target Pareto set without a combined score.
- Frozen result: 86/86 selected coordinates match the pre-target audit and have zero overlap with the original 180 or the excluded primary champions. Target evaluation generated 9,171 SImain trades; 81 candidates are positive after 3.57 bps (94.19%).
- Strictness result: source-threshold versus SImain trade-count Spearman is -0.89296. Across threshold quartiles, median SImain trades move 115.5 → 114 → 109 → 87 and median drawdown moves 28.20% → 27.78% → 27.72% → 17.48%. Median total return moves 3.92% → 4.47% → 3.26% → 15.19%; median trade also has a Q3 setback. Trade count and drawdown improve monotonically, while return and median trade do not.
- Best new transfer: E320/BH240/TRW20/K1.1/W6/M4.5/S340, 85 SImain trades, +20.0188% cost-adjusted total return, -0.3684% median trade, and 17.4644% maximum drawdown. The aggregate leader remains the earlier S332 neighbor at 86 trades, +22.0575%, -0.4156% median trade, and 14.6209% drawdown.
- Shared delivery: 266 candidates, 26,215 SImain trades, 233 cost-positive candidates, and 81 five-dimensional cross-instrument Pareto candidates; 39 Pareto candidates come from the strict batch. One shared main entry and one shared per-trade entry compare both transfer batches. Browser and desktop/mobile/manual visual QA passed with zero runtime errors or external requests; 282 declared output/trade artifacts matched hash and size.
- Interpretation: stricter entry can materially reduce target trading and improve drawdown, and the strictest quartile is the most promising. The relationship is threshold-like rather than continuously monotonic. No combined score, SImain local grid, target-driven candidate change, or parameter acceptance was created.

## 2026-08-04 — K200 to SImain frozen-candidate transfer validation

- Presentation refinement: removed the redundant global filter and long candidate-source/identifier display; retained per-column filtering and placed the per-trade action immediately after rank.
- Trade evidence: generated a dedicated historical-template SImain trade review for all 180 frozen candidates from the existing 17,044 target trades. No strategy compute, frozen-candidate record, or migration metric changed.
- Browser evidence: desktop/mobile comparison and trade-review states pass with no runtime error, external request, replacement character, or page-level overflow.
- Source pool: the current 4,451-coordinate compatible K200 cumulative snapshot. Eight source-only selection views and exact one-axis neighbors produced 180 frozen candidates before any SImain result was read.
- Frozen identity: content SHA-256 `14961b09ba28212b77b38b486f65ca20a9d41138d7a071f4a764750f27366ab6`; file SHA-256 `34f355e2dfda1607d85e38e0b1a554e9ffce9ce8be69316a015bc7593e49bb11`.
- Target data: explicit SIH6, America/Chicago, 15-second session-filled OHLC from 2026-01-28 through 2026-02-23. The prior day supplies 960 required warm-up bars; 5,520 bars are available before the test start. Only trades entered from 2026-01-29 through 2026-02-23 are counted.
- Target result: 152/180 candidates have positive 3.57-bps cost-adjusted total return (84.44%). K200/SImain rank Spearman correlation is 0.36477. The adjacency audit identifies 101 candidates in positive local regions and 48 isolated positive candidates.
- Data audit: SIH6 is the main contract throughout the test interval with zero rolls; source data contains 600 synthetic bars, zero synthetic-signal trades, and 1,410 gap-spanning target trades. SImain has no bound instrument-specific low-activity exclusion policy, so zero-trade and synthetic exposure are reported without inventing a filter.
- Delivery: HTML, comparison CSV, frozen JSON/CSV, target/source transaction CSVs, representative transaction CSV, migration report JSON, run config, and post-hoc status are isolated below `results\cross_instrument_comparison`. Every table field supports filtering and sorting. Default order uses target cost total return descending, drawdown ascending, and target cost median trade descending.
- Interpretation boundary: no combined score, no target-driven candidate modification, no parameter acceptance, and no SImain full-grid run.

## 2026-08-04 — Continuation Rounds 12–13 closed; bounded exploration stopped

- Round 12 closed 229 coordinates, 7,845 trades, and 20 batches. It moved the average-return leader from S320 to S310 at E160/BH720/TRW24/K1.6/W10/M2.5: 33 trades, +0.9414% cost-adjusted average return, +35.1509% cost-adjusted total return, and 4.60% maximum drawdown.
- Round 13 closed 172 new coordinates, 5,944 trades, and 15 batches. It resolved S301–S319 at one-bar precision, cross-checked W/M and entry geometry, and retained 20 sparse bounded broad controls.
- The final average-return leader is E160/BH720/TRW24/K1.6/W10/M2.5/S308: 33 trades, +0.9464% cost-adjusted average return, +35.3717% cost-adjusted total return, and 4.54% maximum drawdown. S308–S312 form a continuous high region; the incremental gain over S310 is only 0.0050 percentage points.
- Final shared snapshot `e4a20d1d5bcb8974f4341a3647e2e246c3c1ab855d66ce4b3d7d4998d7fb3d44` compares 4,348 compatible coordinates and 669,694 trades across sixteen stages in the same cumulative main and shared per-trade HTML. Stage, snapshot, and stable browser QA each passed 280 states; 4,352 outputs and 7,778,441,200 bytes passed hash/size reconciliation with zero mismatches.
- The median leader trade remains negative; seven gap-spanning trades compound to +24.85%, while non-gap trades compound to +8.43%. No parameter is accepted. Further one-bar refinement is stopped because improvement has diminished and would increase in-sample overfitting risk.

## 2026-08-04 — Continuation Round 11 delivered; Round 12 designed

- Round 11 closed at 280 coordinates, 13,306 trades, and 24 batches; cumulative snapshot `a82a5a4036d5aaf2e964052d2aa189591d75d3d2c01f25528d4c544ab8e724b3` now contains 3,947 coordinates, 655,905 trades, and fourteen stages.
- The average-return leader improved to E160/BH720/TRW24/K1.6/W10/M2.5/S320: 33 trades, +0.9108% cost-adjusted average return, +33.7945% cost-adjusted total return, and 4.54% maximum drawdown.
- Stage, snapshot, and stable browser QA each passed 280 states. All 3,951 declared delivery outputs and three stable routes passed hash/size reconciliation.
- Round 12 retains 229 new coordinates after exact anti-join: 213 refinement/module rows and 16 bounded broad controls, in 20 expected batches.

## 2026-08-04 — Multi-round exploration reopened; Round 11 designed

- The user explicitly reopened exploration in the existing campaign lineage and requested both reasonable bounded broad spans and local refinement.
- Preserved the prior header-only no-next-round handoff as historical evidence and created a separate reopened Round-11 handoff.
- Round 11 requests 321 rows, contains 291 unique requests, removes 30 internal duplicates and 11 protected overlaps, and retains 280 new coordinates in 24 expected batches.
- The retained set contains 223 local-refinement and 57 broad-jump coordinates. All rows include a hypothesis, expected behavior change, evidence summary, and selection reason.
- No parameter is accepted; the same cumulative main and shared per-trade HTML remain the required delivery.

## 2026-08-04 — Continuation Round 10 closed; current exploration stopped

- Round-10 raw closure passed at 83 coordinates, 2,699 trades, and 7 batches. Completion SHA-256 is `a1456d0ffd2e47d188f1ab1d418be1756e5dacccd689de5f8e655fcab7ee7283`.
- Shared cumulative snapshot `73ea40e633f2ea4d4c70ab906c5b4119fa7fb1ecc41113b8e949a089a8a4fdb3` compares 3,667 compatible coordinates in one ranking and shared per-trade page; total trades are 642,599 across thirteen stages.
- Stage, snapshot, and stable-route QA each passed 280 interaction states. The 3,671-output hash/size audit checked 7,454,481,962 bytes with zero mismatches; mobile visual inspection found no overlap, garbling, clipping, or asymmetry.
- Both average-return views improved to E160/BH720/TRW24/K1.6/W8/M3/S320: 33 trades, +0.8458% average fixed-cost return, +31.0212% fixed-cost total return, and 5.73% maximum drawdown.
- Risk remains concentrated: median trade is negative, top-two contribution is 41.85%, and seven gap-spanning trades compound to +23.45%. No parameter is accepted and no next round is authorized.
- Final anchor diagnostics cover all 236 trades from the unrestricted, Scenario-1, and average-return leaders. The catalog includes fixed-3.57-bps return, MFE, MAE, MFE retention, profit giveback, post-exit continuation, gap/synthetic/low-activity flags, drawdown relief, top/bottom contributors, and a deterministic random control sample. Summary SHA-256 is `e2ce75b4c5729df255c0da8daff755cd13f903c35753ac3b1bcb2d346156a3c2`; catalog SHA-256 is `83fe0134ce124ae0fb2a1280b749fe98de03a5bcaedb001ce8b3c62aad7aa2a3`.
- The required no-next-round handoff is header-only at `.omo\exploration_v13\continuation_round_11_handoff\next_round_parameters.csv`, SHA-256 `435fd47a166ae07d5fa3a1577970ee0e6d1adee3c899c11881aa090a0d756197`; its audit records zero coordinates, zero batches, and the stop reason.

## 2026-08-04 — Continuation Round 9 closed; Round 10 handoff prepared

- Round-9 raw closure passed at 196 coordinates, 15,867 trades, and 17 batches. Completion SHA-256 is `dfcfe2a168b1f475e90b48996af6daf95487c6b7697c958974959374f507b8e6`.
- Shared cumulative snapshot `3f893a17657375bbdf665b238ee737fb2e7709d98d259dfb46ab3655d74c19fa` compares all 3,584 compatible coordinates in one ranking and shared per-trade page; total trades are 639,900 across twelve stages.
- Stage, snapshot, and stable-route QA each passed 280 interaction states. The 3,588-output hash/size audit checked 7,421,931,310 bytes with zero mismatches.
- Unrestricted and Scenario-1 total-return micro surfaces did not beat their Round-8 incumbents and are stopped. Both average-return views improved to E160/BH720/TRW24/K1.6/W8/M4/S320 at +0.7758% average fixed-cost return and 4.91% maximum drawdown.
- Round-10 handoff requested 87 rows, removed 3 internal repeats and 1 completed overlap against 3,585 protected completed coordinates, and retained 83 unique new coordinates in 7 batches. CSV SHA-256 is `81212c034f4f28d00ab049dbb16f768660f094b421784e91370784398224918e`.

## 2026-08-04 — Continuation Round 8 closed; Round 9 handoff prepared

- Round-8 raw closure passed at 637 coordinates, 63,388 trades, and 54 batches. Completion SHA-256 is `99385596e7fbc28a77b29d9299b1e2b1cd2741f90bf87ea46bdd351f20883a64`.
- Shared cumulative snapshot `0569e8b7859ba4f6c896a870dcb20f092e4c55747496dafb64a49552edf56ebe` compares the prior 2,751 and all 637 new coordinates in one 3,388-coordinate ranking and shared per-trade page; total trades are 624,033 across eleven stages.
- Unrestricted BH×S is `mixed`: BH171/S388 improves fixed-3.57-bps return from 78.0292% to 78.4568%, with nearby BH166 and S392 points close, while median trade, concentration, and gap dependence remain weak.
- Scenario-1 TRW×K is `mixed`: TRW22/K1.0 reaches 55.9743%, only 0.0504 percentage points above the incumbent; two compensating neighbors remain close.
- The strict-entry broad block is `improved` for both average-return views: E160/BH720/TRW24/K1.6/W8/M4/S400 has 32 trades, 0.6986% average return, and 6.0223% maximum drawdown, but top-two concentration remains 48.4%.
- Unrestricted TRW×K, both W×M surfaces, Scenario-1 E×BH, and the sub-50-minute broad block are `not_improved` and stop.
- Round-9 handoff requested 253 rows, deduplicated 2 internal repeats, removed 55 completed overlaps against 3,389 protected completed coordinates, and retained 196 unique new coordinates in 17 batches. CSV SHA-256 is `f8e1e97c556320109784c4b432d8407514db4177365c54a2b3059b376bc8f59a`.
- Round-9 reasons: confirm the unrestricted ridge at one-unit resolution, distinguish the tiny Scenario-1 gain from noise, test local stability of the new average-return leader, and retain medium-speed broad controls. No new result branch is created and `parameter_acceptance=none`.

## 2026-08-04 — Continuation Rounds 3–7 reconciled; Round 8 handoff closed

- Round 3 updated combinations because the shared E320/BH240/TRW12/K1.25/W6/M4.5 core improved both objectives while S intervals, fractional M, adjacent W, and entry axes remained unresolved; 381 coordinates tested those gaps.
- Round 4 updated combinations because Round 3 moved Scenario 1 to the TRW18 boundary and unrestricted to interior S390; 317 coordinates refined the two peaks independently.
- Round 5 updated combinations because Round 4 moved Scenario 1 to lower-bound K1.1 and unrestricted to BH180; 165 coordinates tested the adjacent TRW/K surface and BH180/S390 neighborhood.
- Round 6 updated combinations because Round 5 found an interior Scenario-1 S330 and an unrestricted BH180/TRW11/K1.4/S390 improvement; 81 coordinates tested integer speed neighbors and the higher-K boundary.
- Round 7 updated combinations because Scenario 1 stopped improving while unrestricted moved to S388 and K>1.4 failed; 32 coordinates confirmed immediate speed, entry, baseline, and exit neighbors. The current cumulative leader became E320/BH170/TRW11/K1.4/W6/M4.5/S388.
- New user authorization supersedes Round 7's local stop condition while preserving its evidence. Snapshot `8ddc0d2d0a32f3e5a6ec4710a3a8a64774029ce5a6170ccd724bf07282229fb8` contains 2,751 coordinates, 560,645 trades, and ten stages.
- Round 8 requested 693 coordinates in eight evidence-led blocks. Exact anti-join removed 56 completed overlaps, leaving 637 unique new coordinates and 54 expected batches. Protected completed=2,752 including the historical temporary coordinate; active=0 and pending=0. The handoff CSV SHA-256 is `deebea044781f168d4e31f18abe88db0afa42d5930948431ac3a1e9e5e3fdb2c`.
- Round-8 reasons: measure the BH×S unrestricted ridge; test TRW×K entry quality and W×M giveback surfaces for both independent objectives; inspect fine Scenario-1 E×BH stability; retain broad strict-entry coverage; and test sub-50-minute S values that are absent from all completed coordinates. Gap-excluded return remains diagnostic only.

## 2026-08-02 — Temporary single-coordinate comparison

- Scope: E120/BH360/TRW6/K0.75/W1/M0.25/S480, `all_window`, calculated entry.
- Result: 3,882 trades; fixed-template analysis and per-trade page delivered.
- Target evidence: 2026-06-19 11:07 trade exits at real close 1514.850; theoretical line 1514.825; audit basis `same_bar_close_after_strict_new_low_confirmation`.
- Interpretation: implementation validation only. It does not establish parameter quality and does not authorize another experiment.

## 2026-08-03 — Multi-round cost-adjusted exploration authorized

- Question: within the unchanged V4.4 trading rules, which broad regions lead Scenario-1-qualified and unrestricted cost-adjusted total return after 3.56 bps per completed trade?
- Design state: source closure and a non-executable multi-round design precede the first executable plan. Early rounds use multiple broad blocks; later rounds may run new broad coverage and local refinement together only after a closed-round interpretation.
- Execution boundary: `all_window`, raw `entry_slippage=0`, three workers, 12 coordinates per batch, 4,096 MiB minimum free memory. The cost model is derived analysis only.
- Stop boundary: never interpret partial batches; stop a branch on invalid closure, nonfinite primary evidence, fixed-template failure, or lack of eligible improvement. No in-sample leader is an accepted parameter.
- Round-1 plan: frozen at 372 coordinates across four broad blocks; SHA-256 `1424dc17862a2bfe0b8f0439fef061e64efc487c5057b7cff64498ed40a78046`. Validate-only, exact anti-join, the fresh resource gate, and stage materialization followed the frozen-plan handoff.

## 2026-08-03 — Round 1 closed; Round 2 frozen

- Closure: 372 coordinates, 316,398 trades, 31 batches; completion SHA-256 `c9532f77b626f647dcfe7b1fdc09ee76b2b895e851da0b98f6206b26ff1e6539`.
- Cost effect: 127 coordinates remain cost-positive. The gross leader has 4,481 trades and reverses from +91.7995% gross to -61.0935% cost-adjusted.
- Scenario-1 leader: E320/BH720/TRW24/K1.25/W1/M8/S320; 62 trades, +17.0157% cost-adjusted, 10.6737% cost-adjusted maximum drawdown. The top five press M=8 and S=320.
- Unrestricted leader: E40/BH720/TRW6/K2/W192/M2/S480; 114 trades, +22.6187% cost-adjusted, 15.8205% cost-adjusted maximum drawdown. The top five share E40/BH720/TRW6/K2; W192/M2 and W48/M8 are separated by only 0.0330 percentage points.
- Evidence boundary: both leaders are materially gap-dependent, and gap-excluded return remains display-only. No parameter is accepted.
- Delivery: active-campaign snapshot `2020ad7b12d57889f1c1d0cf69f981bcf2b5e3ec5b8a4808c196dbb6cdd51d47`; 384 hash/size checks and browser QA passed. The older temporary campaign remains outside this engine-hash lineage.
- Round 2: 247 new coordinates across one 144-coordinate broad block, 47 Scenario-1 local coordinates, and 56 unrestricted local coordinates; plan SHA-256 `f90dbf5563ae9128304d5b48b902db440d9b73a4af3c0a7654079ef73628f7fd`.

## 2026-08-03 — Round 2 closed; terminal Round 3 frozen

- Closure: 247 coordinates, 20,629 trades, 21 batches; completion SHA-256 `b97a4811ebf3520fb6086ec26d5dfa149b2154d989ca6d94a1149f8d7a28350c`; stage analysis SHA-256 `be731db4bc274b85271ea5bb26421aba0e8f8b8f538825e3925e5d459aa47164`.
- Qualification/cost: 51 Scenario-1-qualified, 6 Scenario-2-qualified, zero Scenario-3-qualified, and 139 cost-positive coordinates.
- Scenario-1 leader: E320/BH720/TRW24/K1.25/W2/M6/S400; 53 trades, +25.5775% cost-adjusted, 7.5404% cost-adjusted maximum drawdown, +8.5617 percentage points over Round 1.
- Unrestricted leader: E40/BH720/TRW6/K2/W48/M4/S400; 94 trades, +36.0556% cost-adjusted, 16.9845% cost-adjusted maximum drawdown, +13.4369 percentage points over Round 1.
- Gap audit: leader gross returns excluding gap-spanning trades are -0.3462% and +0.3523%, respectively, materially less gap-dependent than the Round-1 seeds. The field remains display-only.
- Broad decision: the best Round-2 broad-block row reached only +9.0957% cost-adjusted and did not improve either objective; the broad branch stops.
- Delivery: active-campaign snapshot `dde99537b4584f0d5d98a70e388cacffd226736a455963a2f54acd47b4bfd847`; 619 coordinates, 337,027 trades, 631 artifact checks with zero mismatches, and all stage/snapshot/stable browser and visual QA passed.
- Terminal Round 3: 212 new local coordinates, zero pre-audit overlap with frozen Rounds 1–2, plan SHA-256 `46c95b24feab49b6f260a0e8f1e1125fd74c34c6a0e268b89e0e1fb83a6d9b8c`. Round 4 is prohibited.

## 2026-08-03 — Round 3 terminal closure

- Closure: 212 coordinates, 16,847 trades, 18 batches; completion SHA-256 `edec2c43ecf4c4035a690f763b6a4d68d8be8f97a9da40ec9b3f3aac20ff25ea`; stage analysis SHA-256 `d163c36e471c732181f93d85d3e9752f8938b3cb14d92c025648e53815044d2f`.
- Qualification/cost: 133 Scenario-1-qualified, zero Scenario-2/3-qualified, and 206 cost-positive terminal coordinates. Final cumulative counts are 225 Scenario-1, 74 Scenario-2, zero Scenario-3, and 472 cost-positive of 831.
- Scenario-1 result: E320/BH720/TRW24/K1.25/W4/M4/S400; 58 trades, +30.6696% cost-adjusted, 7.8125% cost-adjusted maximum drawdown, +5.0921 percentage points over Round 2, and +0.6431% gross after excluding gap-spanning trades.
- Unrestricted result: the best new terminal row reached +32.8804%, below the Round-2 leader by 3.1752 percentage points. Final unrestricted leader remains E40/BH720/TRW6/K2/W48/M4/S400 at +36.0556%.
- Gross/cost reversal: the gross leader remains +91.7995% gross and -61.0935% cost-adjusted, ranked 816 of 831 by cost-adjusted return.
- Final delivery: snapshot `0fb3e1e5e8ef890f3b225db46288fa4b3957bcb88c7ca2dff72d750679db6922`; 831 coordinates, 353,874 trades, 843 artifact checks with zero mismatches, and all terminal-stage/snapshot/stable browser and visual QA passed.
- Campaign decision: complete. Round 4 is prohibited; no parameter is accepted. Any new research requires a separate user decision and scope.
- Independent total audit: all 831 coordinates are unique with zero pairwise/cross-round duplicates; all 712 raw artifact records and 843 delivery checks match; every runner/delivery/union lock is released and no matching process remains. Final evidence SHA-256 `58009d4eb357e3022de423d63310c9821fd167e518b43c739dc8efea6a694c0e`.

## 2026-08-03 — Continuation Round 1 frozen

- Authority: a new continuation phase supersedes the prior stop only for separately named `continuation_round_*` stages. The original three-round campaign and its Round-4 prohibition remain closed historical evidence.
- Question: can a materially sized new broad span improve the independent Scenario-1-qualified and unrestricted cost-adjusted total-return leaders under unchanged V4.4 method and 3.56-bps cost contracts?
- Plan: 528 coordinates across a 288-coordinate shared entry map, 180-coordinate Scenario-1 exit span, and 60-coordinate unrestricted exit span; 44 expected batches.
- Anti-join: 528/528 expanded tuples are unique, with zero overlap against the 831 compatible completed coordinates and zero overlap against all 832 protected completed current-V4.4 IDs; active=0 and pending=0 at freeze preparation.
- Identity: plan SHA-256 `481fd28365757f739cb0e260d3cc36a4390db9cde9b1f1ccf3063aefdb8c9bf5`; design memo SHA-256 `94966f8096bb5317e09d2eb178a10cc9b1f31d3871c2bb8a3efa61218f2b9412`.
- Delivery boundary at freeze time: immutable raw closure precedes both analysis and HTML generation. The later pipelined-governance correction allows analysis and the next-plan path to proceed while delivery runs, subject to current source identity, exact anti-join, separate roots/locks, live-process resource gating, and a single cumulative publisher. All four HTML entries and full QA remain mandatory.

## 2026-08-03 — Continuation Round 1 raw/delivery closure; source repair active

- Raw closure: 528 coordinates, 54,842 trades, 44 batches; completion SHA-256 `0990507be75526618663b4e08a3d628fd7af856dd692c9d9c3313de2cd0fdf6d`.
- Old-source delivery: stage analysis SHA-256 `2d87ad79920740980141aaef6f7e5c4b650ebcf99dd5e15c485c1c05986a0f70`; cumulative snapshot `ce1e20f7366135cb92c098dc3db4c3245bdc2374630a89f9adafaa54d715d714` with 1,359 coordinates, 408,716 trades, and four stages. All 1,921 artifact checks and 400/720/720 browser states passed with clean visual QA.
- Evidence boundary update: the user requested presentation corrections after that delivery. Preserve the raw closure and old-source delivery. Shallow V2 was rejected after recursive runtime-input audit; Continuation Round-2 validation and compute remain paused until replacement `SOURCE_FINAL_V3` is independently confirmed. After confirmation, work may pipeline with the exactly-one corrected-source redelivery. The replacement `DELIVERY_FINAL` remains mandatory eventual evidence.
- Execution governance is role-neutral: one executor or multiple executors may perform analysis, compute, and HTML delivery. After immutable raw closure, analysis and delivery may run concurrently. Overlapping next-round compute also requires the current source identity, exact completed+active+pending anti-join, separate process/output/root/lock boundaries, a fresh resource gate that includes the live delivery process, and exactly one cumulative publisher. A result-affecting source inconsistency found during delivery pauses new compute until another source identity closes.
- Current authorization: continue parameter exploration. Complete Continuation Round 2 through immutable raw closure, four-entry delivery and full QA, objective-specific interpretation, bilingual records, and read-only total audit. Its delivered evidence must justify any later round; no later plan or compute is pre-authorized.

## 2026-08-03 — V6-bound Continuation Round 2 frozen

- Source authority: `SOURCE_FINAL_V6`, manifest SHA-256 `0aee46e6edf23eb60e5a2843e4abc5ff33ebfd0fd32e5acd945d66576104b123`; all 47 source/runtime/template bindings matched in the read-only closure check.
- Plan: `v4_4_cost_adjusted_multiround_20260803_continuation_round_02_dual_objective_local_v6_all_window.json`, SHA-256 `d982267710abab0355a37271c18a25df40decc3d9f846f82030a3ecbeab82a07`; 416 unique coordinates and 35 expected batches at the existing 3-worker, batch-12, 4,096-MiB resource contract.
- Anti-join: zero overlap with 1,360 completed coordinates; no active or pending compute. The superseded V4-bound R2 root has exactly four metadata files and no raw progress, batches, trades, completion, or analysis; it remains preserved and excluded only as documented pre-compute evidence.
- State: the new V6 output root is absent. Existing R1 HTML will receive QA-only recovery; raw or delivery redelivery is not part of this plan.

## 2026-08-03 — V6-bound Continuation Round 2 closed and delivered

- Raw authority: 416 unique coordinates, 41,134 trades, and 35 batches. Completion SHA-256 `2c0364fda3fc17cd09419d0a6003e6a3e6d7f1da035b8228c201fd21a6570d6e`; all 350 indexed raw artifact records reconciled with zero errors and the raw lock released.
- Delivery authority: stage analysis SHA-256 `dfc399024c7278a8af90b7dca638e69d3174de1fd2594aaf759f892af9678556`; current snapshot `2da2a0dff4c1890627f78c0556a2d8504ff0f384f77db147da54572367635a52`, 1,775 coordinates, 449,850 trades, and five compatible stages. Stable routes point to that snapshot.
- QA: R1 existing-HTML recovery passed at 400/720/720 states. R2 stage/snapshot/stable QA passed at 320/920/720 states, with zero runtime errors, external requests, or layout failures; manual desktop/mobile inspection found no garbling, clipping, broken alignment, or unintended overflow.
- Scenario-1 cost leader: E320/BH240/TRW12/K1.25/W6/M4.5/S340, 130 trades, gross +49.7181%, cost-adjusted +42.9652%, cost maximum drawdown -18.6941%; +11.2383 pp over the prior cumulative Scenario-1 leader.
- Unrestricted cost leader: E320/BH240/TRW12/K1.25/W6/M4.5/S400, 113 trades, gross +70.8452%, cost-adjusted +64.1373%, cost maximum drawdown -15.7238%; +17.1489 pp over the prior cumulative unrestricted leader. It also leads both average-return views at >=10 and >=20 trades.
- Interpretation: this is an in-sample, gap-sensitive research improvement. Gap-excluded returns remain display-only, no combined score is formed, and parameter acceptance remains `none`. No subsequent round is created in this closure.
- Final read-only audit: 1,775 unique coordinates, 449,850 trades, and 149 batches across five completed compatible stages; 745 raw records, 90 stage-analysis artifacts, and 21 snapshot artifacts all match their hashes and sizes. No relevant compute/delivery process remains.

## 2026-08-03 — Cost-contract transition for continued exploration

- User approved continued exploration while retaining completed raw stages and legacy results.
- Future derived rankings replace fixed HKD 300,000 / 3.56-bps costing with a hash-bound K200M current-notional snapshot: 1,106.70 points × KRW 50,000 = KRW 55,335,000; USD 6 commission at KRW/USD 1,446.7 = 1.568663594470046 bps; total with 2 bps slippage = 3.568663594470046 bps.
- No raw batch is rerun or overwritten. The source closure and cumulative derived analysis must be refreshed under the new reference before the next reviewed parameter plan.

## 2026-08-03 — Delivery evidence boundary update

- Preserve the current five-stage cumulative snapshot and all historical stage pages. No existing result is rebuilt or replaced.
- For a future authorized round, append its completed compatible results to the current cumulative main and single shared per-trade entries. A dedicated per-round per-trade HTML page is no longer required.
- The next backtest waits for an explicit user instruction.

## 2026-08-04 — Continuation Round 14 relative multiscale exploration

- Design: 103 exact new coordinates combining relative E/BH/S timing scales, broad interaction points, module-pair tests, and fine checks used only for stability. Large timing parameters were not densified by arbitrary single-digit steps.
- Closure: 103 coordinates, 8,490 trades, and 9 batches; completion SHA-256 `b1641b13c742454f68e474d1d6acdc325cbccd1e87e70bcf64874229b1ce070d`.
- Result: unrestricted cost-adjusted total return improved to 79.1437% at E320/BH171/TRW12/K1.26/W6/M4.5/S388. Scenario 1 and both average-return views did not improve.
- Delivery: snapshot `f66874f74b868dfc4dd74d30d2b3708161c3eb10607fb04c4cc2cd2075f5b8d9`, 4,451 coordinates, 678,184 trades, and 17 stages. Targeted browser and artifact QA passed.

## 2026-08-04 — Continuation Round 15 ridge interaction exploration

- Design: 48 exact new coordinates after removing 10 completed overlaps: 28 broad E/S interactions, 18 TRW/K ridge pairs, and 2 K stability checks. Compute used two workers after a preserved pre-batch three-worker attempt exceeded the available process-pool memory margin.
- Closure: 48 coordinates, 5,651 trades, and 4 batches; completion SHA-256 `b3d9fece5318d6cea20fdd1cb0c76e4c6fdc033fff3af8902f418f95f0ba87d5`.
- Result: the unrestricted leader moved to E480/BH171/TRW12/K1.26/W6/M4.5/S388: 112 trades, 90.0092% gross, 82.6033% cost-adjusted total return, 0.5627% cost-adjusted average return, and 15.1770% cost-adjusted maximum drawdown. E336/E576/E720 remain strong nearby, supporting a broad plateau.
- Delivery: existing shared entries now point to snapshot `45b9a08396493a53ece45bd62af91070fb6b443539cd2e12ae5aeac5c756faad`, containing 4,499 coordinates, 683,835 trades, and 18 compatible stages. Artifact audit checked 4,523 unique declared files with zero mismatches; targeted browser/desktop/mobile QA passed.
- Decision: the evidence is dependent in-sample and remains gap-sensitive. `parameter_acceptance=none`; stop this exploration session after the published Round 15 result.

## 2026-08-04 — Continuation Round 16 multimetric broad coverage

- Design: 205 exact new coordinates after deduplication and anti-join, spanning unrestricted, Scenario-1, average-return, low-drawdown, and remote-control surfaces. The retained grid contains 170 large-span points and 35 paired-module points; E, BH, and S use relative spacing rather than single-digit refinement.
- Closure: 205 coordinates, 22,635 trades, and 18 batches; completion SHA-256 `0e569a9458af3f2146c6d5874c123e14ad70993d104ca7f4d30dc7ed80946060`.
- Total-return result: the unrestricted incumbent remains E480/BH171/TRW12/K1.26/W6/M4.5/S388 at 82.603275% cost-adjusted return. The Scenario-1 incumbent remains E320/BH240/TRW22/K1/W6/M4.5/S330 at 55.976177%.
- Average-return result: both >=10 and >=20 trade views improve to E112/BH612/TRW24/K1.6/W10/M2.5/S308, 30 trades, 0.999935% cost-adjusted average return, 33.850987% cost-adjusted total return, and 4.823612% drawdown.
- Pareto result: E150/BH504/TRW24/K1.6/W10/M2.5/S310 is a new nondominated moderate-trade point with 30.817684% cost-adjusted return and 3.089401% drawdown.
- Delivery: shared main and per-trade entries point to snapshot `20464535ee48376b73b847ea8454355b2acd58ab4c78c1273f3e97f9e37f76c7`, containing 4,704 coordinates, 706,470 trades, and 19 stages. Artifact and desktop/mobile browser QA passed. `parameter_acceptance=none`.
