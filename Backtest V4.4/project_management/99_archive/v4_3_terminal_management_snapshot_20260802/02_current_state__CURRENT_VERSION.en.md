# Current Version Notes

## Snapshot

| Field | Current value |
| --- | --- |
| Project | Backtest V4.3 Research |
| Context initialized | 2026-08-01 |
| Initialization mode | existing project |
| Confirmed optional modules | data, research |
| Project implementation state | The five-round max-W campaign is terminally closed at 684 cumulative coordinates and 232,225 trades; Round 5 is delivered and interpreted; no Round 6 or parameter acceptance |

## Current behavior

- The entry method remains calculated-threshold with `wait_next_real_trade`.
- A pending entry fills at the first real-trade bar open within the existing wait/continuity boundary. Price retrigger, higher-high cancellation, and structural-reversal cancellation are absent by explicit decision.
- Rebound and speed exits that trigger on a non-real bar freeze the first trigger reason, time, theoretical price, and evidence. The position remains open and fills at the next real-trade bar open; flat reset occurs after the fill.
- The rebound basis is the monotonic maximum of positive finite causal W-window `open[start]-low[end]` candidates from completed in-position bars. Each bar checks against the basis effective through the prior completed bar; actual entry resets the per-trade maximum.
- `rolling_tr_sum` is the sole aggregation method. Baseline sampling is a separate policy axis: `all_window` is the default and treats markers as audit/chart evidence; `exclude_marked` omits `baseline_excluded` atoms and backfills older eligible atoms within the same continuity segment.
- Both policies retain finite synthetic TR15 atoms unless a synthetic atom is also marked and `exclude_marked` is selected. No real-only baseline policy is current.
- Duplicate source timestamps fail before execution. `baseline_pending_atom_count` measures actual `low_activity_state == pending_low_activity_buffer` atoms in the physical baseline span.
- Active runtime inputs and fixed templates resolve from repository-relative, hash-bound files under `runtime_inputs`.
- A user-requested ZIP includes current-version code, project-management files, analysis reports, and 15-second OHLC data; compute-result payloads are excluded.
- Scenario creation and review use the version-neutral shared HTML at `D:\Code\backtest-release\shared_tools\scenario_manager\index.html`. Saving assigns the next monotonic `情景N` name without a name field, counting archived names so deleted numbers are not reused. Active scenarios appear as a two-row button strip above the chart and reload their intervals when clicked; deleted scenarios remain recoverable. The display defaults to 15-second bars and also supports 1, 5, 15, 30, 60, and 120 minutes; CSV/TSV upload is in the lower action row.

## Current form and interfaces

The engine and runner produce immutable raw stage evidence. The analyzer and four-worker review generator create the fixed historical-template trade HTML. The cumulative builder publishes the stable main/trade routes only for completed compatible stages. Stage and cumulative `main_data` expose four independent `highReturnViews`, partitioned by baseline-sampling policy. All five max-W stages and the 684-coordinate/232,225-trade cumulative delivery are closed from immutable raw evidence.

Each raw stage has one baseline-sampling policy. The cumulative delivery may contain both policies, but rankings and identity mappings stay partitioned by policy and method.

## Known gaps

- The terminal leaders are dependent in-sample candidates for external validation/review only. No holdout, new-date, cross-instrument, cost/slippage, or production-acceptance evidence exists.
- Scenario-1 qualification is interaction-specific; the final unrestricted-total leader lies on lower E/W/M boundaries and has 3,743 trades; the average leader retains M10 from Round 3. These are limitations, not gap-audit rejection rules.
- Existing V4.2 results cannot be reclassified as V4.3 results because execution semantics changed.

## Recent changes

| Date | Change | Evidence |
| --- | --- | --- |
| 2026-08-01 | Initialized the neutral project-management structure. | Generated files and initialization report |
| 2026-08-01 | Created V4.3 with pending synthetic-exit execution and repository-relative runtime inputs. | Source manifest, runtime manifest, and regression tests |
| 2026-08-02 | Added and refined the cross-version scenario manager without rewriting the original selector or historical snapshots. It now uses automatic monotonic `情景N` names, a two-row saved-scenario button strip, a lower upload action, and 15-second default display with 30/60/120-minute choices. | Shared HTML, local scenario persistence, and desktop/390px browser QA |
| 2026-08-02 | Added selectable dual baseline sampling, neutral preparation markers, policy-partitioned identities/rankings, truthful pending-atom audit, and duplicate-timestamp rejection. | Current code, regenerated preparation/runtime manifests, and tests |
| 2026-08-02 | Closed the max-completed-W identity and froze a 12-coordinate `all_window` Round-1 plan with four independent high-return views; no coordinate executed. | Source manifest SHA-256 `78fcea5452769fe14f531c4e3036fe7ba3b2513092a0ad1e43a5b7d495c949ec`, plan SHA-256 `36f9f47fd1c49ac1be68abf7a0f146b81d667c115123b385dde31239daccdc80`, source-closure artifact, and full tests |
| 2026-08-02 | Closed Round 1 as a 12-coordinate implementation/smoke probe: 1,196 immutable trades, corrected stage/cumulative delivery, and four full-return views. Gap-excluded return is display-only audit evidence. | Immutable evidence SHA-256 `935fd5e562b54deb8d8334251753a5b04e2ed3b71148d8ad00a9f5137dcdc216`; corrected delivery evidence SHA-256 `d0f88bc6a07c0ec0b5f8a298998aeee900a5a8a47b3dc9c581202bde14703b7c` |
| 2026-08-02 | Froze the deterministic 504-coordinate `all_window` Round-2 Cartesian plan at 42 batches of 12, with S480 and broad E/BH/TRW/K/W/M jumps. | Plan SHA-256 `5735c12e439c8686f1f8e2e5e5d4ffcdb956902aa21825619f5c6d9ab4eebe85`; source manifest SHA-256 `306bf83ca50dab913fc7d8f081ec249dc05e03529ced7ce1c6dc58addab3f946` |
| 2026-08-02 | Closed Round 2 at 504 coordinates and 73,306 trades; independent leaders were 10.8547% Scenario-1 total, 51.3786% unrestricted total, and 0.554842% average trade. | Completion SHA-256 `863eb92646e5ae78d6d2bd72f6c9e5d24acfb311b069e17c83c4fd94894eb2cc`; delivery evidence SHA-256 `af17ff903b66877bcebae41d1bc7e1e6f041847b091945a573f17c8df2059160` |
| 2026-08-02 | Terminally closed the 72-coordinate Round-3 interaction refinement and the 588-coordinate cumulative delivery. Terminal in-sample leaders are 18.2861% Scenario-1 total, 115.8216% unrestricted total, and 0.578419% average trade for both trade-count views. No Round 4 or parameter acceptance. | Completion SHA-256 `d42d806e0e9c4d651b18e03846f300c8f059e70dff987bab9dc544f281e474a7`; terminal delivery evidence SHA-256 `7b8f1450c9b53b84ad5d5d2e3ddd51fa6d84147624b41b4135cbe9d29262bf6f` |
| 2026-08-02 | The user replaced the prior no-Round-4 decision and approved a bounded 72-coordinate boundary continuation; the plan is frozen and compute remains gated behind B's independent anti-join and validate-only. | Plan SHA-256 `f991e8c46b42ccc38677eef3be7f2e67d9e822592bb53d3578a4fbc1cdb77670`; approved pre-run design evidence |
| 2026-08-02 | Closed Round 4 through immutable raw evidence and stable fixed-template delivery at 660 cumulative coordinates/185,493 trades, then approved and froze a terminal 24-coordinate Round-5 plan with two 12-point blocks and no average block. | Round-4 delivery evidence SHA-256 `0d5d3187de8a2a1297bbb35f1de220930d34ed30c117906b936506cd404b7fb8`; Round-5 plan SHA-256 `bb7f323e077ebb919d122764d7fac20b97b0790959d71761b77bcf69e6a23ed8` |
| 2026-08-02 | Terminally closed Round 5 and the five-stage campaign at 684 coordinates/232,225 trades. Final view leaders are Round-4 Scenario-1 total at 37.082743%, Round-5 unrestricted total at 515.916078%, and the Round-3 0.578419% average leader in both trade-count views. | Round-5 immutable evidence SHA-256 `ae53e778944aa3fd4ed41627655fbaeb07aa863632110a0073728bba11b784c3`; delivery evidence SHA-256 `6d3cc38cfe98d16d2b095990f2f6a8c85d69c230b9b873bae61d5c414c6751a0`; snapshot `4bed7f828a7068ee0ee70001a245133c481e785358cfb3f2b043d08713ca7259` |

## Maintenance

Update after a deliverable, current behavior, project structure, interface, dependency, command, or operating-procedure change. Record durable rationale in `04_decisions\DECISIONS.en.md` and current authoritative paths in `SOURCE_OF_TRUTH.en.md`.
