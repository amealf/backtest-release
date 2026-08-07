# Short Momentum Net-Drop Rebound V4.4

V4.4 is the current rolling-TR, calculated-execution-price, combined-exit research identity with a selectable baseline-sampling policy.

## Method contract

- `rolling_tr_sum` is the sole entry-baseline method; `tr_average` is rejected.
- Entry eligibility requires `baseline > 0`, `drop > 0`, and `K × baseline > 0`; positive equality `drop == K × baseline` remains valid.
- Every finite TR15 atom inside the exact continuous BH window ending at strict H participates in the baseline.
- Baseline sampling supports `all_window` (default) and `exclude_marked` (optional). The first treats `baseline_excluded` as audit/chart evidence; the second uses causal `baseline_available_from` timing, including recovery-time backfill. Policy-specific results and rankings never mix.
- Synthetic status remains independent from the marker. Neither supported policy removes every synthetic bar.
- A real completed signal bar fills at `min(signal-bar open, H - K × baseline)`.
- A pending entry is a retained signal under `fill_first_real_open`: it fills at the open of the first continuous real-trade bar within the existing 120-bar boundary. It does not require a trigger recross and is not cancelled by a higher high or structural reversal.
- Rebound is evaluated before the zero-extension speed exit. A prior rebound line fills at open when `open >= trigger`, otherwise at trigger when `high >= trigger`. A strict-new-low real bar uses only earlier completed W information and, when confirmed by its close, fills at that close. A non-real exit remains pending until the next real-trade open inside the sample.
- W source windows begin at `max(H, continuous_start, end-W+1)`. Candidate length is the available prefix from 1 through W; there is no full-W or minimum-window-ratio requirement, and an early prefix maximum can govern the remainder of the position.
- The exact candidate is `w_open_to_end_low_drop = open[start] - low[end]`. It is the window-start-open to window-end-low net drop, not the maximum ordered decline from an internal high to a later low. Legacy raw audit field names remain unchanged for compatibility.
- Scenario qualification is optional and instrument-bound. A current-schema profile without a scenario set has no scenario qualification and cannot inherit a K200 fallback.

## Ranking and cost contract

- Every primary view offers gross and cost-adjusted ranking/display modes; the selected mode changes both ordering and displayed return fields. Cost-adjusted is the default.
- Each campaign loads the cost model bound by its instrument profile. Notional is derived from reference price and point value; commission, slippage, quote currency, and any FX conversion come from that bound model.
- Cost metrics are derived analysis only. Raw fills and raw `return` remain unchanged.

## Current state

K200 remains the primary completed research lineage. Transfer targets and fresh-search instruments use separate profiles, campaign manifests, result roots, and ranking lineages. Completed evidence does not accept a parameter by itself.

## Runtime entry points

- Preparation: `data_preparation\prepare_dataset.py`
- Engine: `code\v4_4_engine.py`
- Resumable runner: `code\run_v4_4_resumable_campaign.py`
- Delivery worker: `code\run_v4_4_delivery_worker.py`
- Stage analyzer: `code\analyze_v4_4_scenario_3_stage.py`
- Trade delivery: `code\build_v4_4_review_delivery.py` through the current analyzer; its compatibility standalone CLI requires explicit stage, validation-stage, and output arguments.
- Cumulative union: `code\build_v4_4_combined_union_analysis.py`
- Browser QA: `code\qa_v4_4_scenario_3_stage.mjs`

Hash-pinned instrument data, preparation artifacts, historical V4 main/trade templates, Plotly asset, and market selector are stored under repository-level `runtime_inputs`. Active code resolves them from the checkout. Old absolute paths are retained only in provenance records.

The user has authorized bounded multi-round exploration after source closure. Every executable round still requires a reviewed, hash-bound plan and immutable closure before interpretation. No parameter is accepted.
