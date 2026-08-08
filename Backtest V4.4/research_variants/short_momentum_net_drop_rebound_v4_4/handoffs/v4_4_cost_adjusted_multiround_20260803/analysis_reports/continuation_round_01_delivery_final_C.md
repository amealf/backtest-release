# Continuation Round-1 DELIVERY_FINAL — member C

## Delivery outcome

- Frozen plan: `D:\Code\backtest-release\Backtest V4.4\research_variants\short_momentum_net_drop_rebound_v4_4\plans\v4_4_cost_adjusted_multiround_20260803_continuation_round_01_broad_span_all_window.json`
- Plan SHA-256: `481fd28365757f739cb0e260d3cc36a4390db9cde9b1f1ccf3063aefdb8c9bf5`
- Raw completion SHA-256: `0990507be75526618663b4e08a3d628fd7af856dd692c9d9c3313de2cd0fdf6d`
- Immutable-closure evidence SHA-256: `1a578d1506033dbd1fd576e7031022367bc2d28ffbb497739ef8ab6243b6354d`
- Exactly one scoped `review_workers=4` stage+cumulative delivery job completed: job `608deedbe6df687de6a58e0eab17d9101a01a3e32a3532528ba0456ab3c8daf4`, PID `37348`, reused-existing-job `false`.
- Delivery status is complete; PID exited; matching Python writer count is zero.
- Delivery status/stdout SHA-256: `6e368013461efee0091f6831d68f7bb02c075f56d0c6d08d2d8077f7447ded7c`.
- Stderr is 384 bytes, SHA-256 `a92ac16138738c1ca756ba0294ddd3121edd984230ca61c68f508ba114ce8a8d`, and contains only one non-fatal pandas mixed-type `DtypeWarning`; no traceback or failure.

## Continuation stage closure

- Stage: `D:\Code\backtest-release\Backtest V4.4\results\campaigns\v4_4_cost_adjusted_multiround_20260803\continuation_round_01_broad_span_all_window`
- Plan fingerprint: `5b893814832b88a6a4e8db66ccc204065f5719060cea1f011644ff9dec237f84`
- Stage coordinates/trades: `528` / `54,842`
- Stage analysis manifest SHA-256: `2d87ad79920740980141aaef6f7e5c4b650ebcf99dd5e15c485c1c05986a0f70`
- Stage main HTML SHA-256: `40bf5e4b4f5531ac0bcafcfb0adf9cda8b8f2a22a65d301c079b64697a344ad3`
- Stage trade HTML SHA-256: `125214644c9a0eab53014f86080188ae6f7204f0d318bb535d59d44c10538ed1`
- Stage trade-review manifest SHA-256: `67ce70d61498d728f81fd41de709396ef3b56d8140115415c46c6b8ff19cd011`
- Stage scenario HTML SHA-256: `c7ae6d952567e9f3783feb91e478b09626510121b73797a6c73fb966782cffd1`

## Cumulative publication

- Campaign discovery root: `D:\Code\backtest-release\Backtest V4.4\results\campaigns\v4_4_cost_adjusted_multiround_20260803`
- Union output: `D:\Code\backtest-release\Backtest V4.4\results\all_completed_union_analysis`
- Snapshot id: `ce1e20f7366135cb92c098dc3db4c3245bdc2374630a89f9adafaa54d715d714`
- Snapshot root: `D:\Code\backtest-release\Backtest V4.4\results\all_completed_union_analysis\snapshots\ce1e20f7366135cb92c098dc3db4c3245bdc2374630a89f9adafaa54d715d714`
- Cumulative coordinates/trades/stages: `1,359` / `408,716` / `4`
- Current-pointer SHA-256: `f4afb59508f353e297df9d86ad5444a9515caaa25332ff573b3b363afc6a286f`
- Analysis-manifest SHA-256: `b069f2c0419da44e8d369055a2ead086688b0cfc0128c9796f730391b3a007a6`
- Completion-manifest SHA-256: `611bb61e8e5339d9819338c6503c52976ed38efd2ea934ef1c06e2f8ad137cb3`
- Trade-review-manifest SHA-256: `575cdae3140a15a5e14b14b46297aff56841e1bc13561d1a9f0dba5c515d4330`
- Snapshot main/trade/scenario HTML SHA-256: `40bf5e4b4f5531ac0bcafcfb0adf9cda8b8f2a22a65d301c079b64697a344ad3` / `125214644c9a0eab53014f86080188ae6f7204f0d318bb535d59d44c10538ed1` / `c7ae6d952567e9f3783feb91e478b09626510121b73797a6c73fb966782cffd1`
- Stable main/trade/scenario redirect SHA-256: `5839f30dfb969a086ce60adfab6fca3e9528d414c7186f31dcdc50698b1d113a` / `bf5b96ade9ef478fd1312603d7247bbfd2933378952a4a565839b3b797dfddfd` / `d92057a7e96b0e6c7b26003ebb64a181a5b433d6e21b3d90423a5488fc9ddb17`
- Six stable manifest/data copies are byte-identical to the snapshot copies; all three stable redirects target the current snapshot.
- Included stage keys are the original R1/R2/R3 plus `continuation_round_01_broad_span_all_window`; excluded-stage count is zero because the incompatible temporary campaign remains outside the scoped root.
- Parameter acceptance remains `none`.

## Artifact reconciliation

- Stage analysis-manifest artifact checks: `18`, mismatches: `0`.
- Stage trade-review output checks: `532`, mismatches: `0`.
- Snapshot completion-manifest artifact checks: `8`, mismatches: `0`.
- Snapshot trade-review output checks: `1,363`, mismatches: `0`.
- Total hash/size checks: `1,921`, mismatches: `0`.
- Raw completion, stage manifest, stage summary, batch index, grid, and SOURCE_MANIFEST hashes remain unchanged after delivery.
- Raw runner, stage delivery, and union locks all passed nonblocking acquire/release probes after QA.

## Browser and visual QA

| Surface | QA SHA-256 | States | Runtime errors | External requests | Layout failures | Screenshots |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Continuation stage | `6405e85d2b64ddb02b58c97aced3e5a19ada87dd0a3366606766016f35c7e98d` | 400 | 0 | 0 | 0 | 6 |
| Immutable cumulative snapshot | `cf1ad89f9ee0e3fec5632421bc05b9f95a7ce0d000d19ea17a6d599e9487161e` | 720 | 0 | 0 | 0 | 6 |
| Stable redirects | `5de19e32ea89620b36de698767b74a13a9ccbc9ef9f45f19b75273ea86f9f4ec` | 720 | 0 | 0 | 0 | 6 |

- Cost-adjusted default rank header: `成本后排名 ▲`.
- Gross-mode rank header: `毛收益排名 ▲`.
- Browser automation verified that mode switching changes both ordering and displayed returns, all four primary views work, trade routes preserve query parameters, scenario tabs/chart states reconcile, and trade entry/full-window/theme controls work.
- Snapshot and stable screenshots are byte-identical for all six captures. Stage and cumulative scenario/trade captures are also byte-identical; the stage/cumulative main captures differ only by their expected coordinate populations.
- Manual review inspected eight unique current captures: stage main desktop/mobile, cumulative main desktop/mobile, scenario desktop/mobile, and trade desktop/mobile.
- No blank/loading capture, replacement character, garbled text, unintended overlap, clipping outside the page, broken alignment, or asymmetric layout was found. Long main pages are expected because the complete ranking tables are retained. Mobile panels stack cleanly; wide data remains contained by the template's responsive/scrollable regions.
- Screenshot evidence and automated DOM checks do not by themselves prove full screen-reader semantics, exact contrast ratios, or complete keyboard focus order; no visible accessibility blocker was found in the accepted captures.

## Bounded auxiliary browser limitation

- A supplementary Codex in-app Browser attempt to navigate to the local stable `file://` entry was blocked by the Browser URL policy before navigation.
- This did not affect the three authoritative repository-native Chromium/Playwright suites, which had already captured and passed the stage, immutable snapshot, and stable-route flows.
- The blocked attempt made no project mutation; its blank audit tab was closed. Delivery was not rerun and no gate was weakened.

## Preserved evidence

- SOURCE_MANIFEST remains `6fa3d0c8eb0277066ef5f70fca4a9fbab1d31fbb30e023cd8fd83d233192ae16`.
- Original Round-1, Round-2, and terminal Round-3 snapshots remain present.
- Failed partial snapshot `1ba05465b49a1de45c407bfd9b4456eeb83c92b9dadea8d1c5d060a40ae22d98` remains present.
- Historical temporary implementation-validation campaign remains present and excluded from this cumulative lineage.
- No source, template, raw compute, plan, or project-management file was edited by C.
