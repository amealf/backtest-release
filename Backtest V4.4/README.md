# Backtest V4.41

This folder is the instrument-neutral V4.41 release workspace for short-momentum
net-drop rebound research. Strategy semantics, instrument properties, and
campaign intent are separate contracts. Every executable instrument profile
binds 15-second market data, cost, gap, low-activity, optional scenario, and
ranking-lineage evidence. V4.2 source and results remain untouched in their
original locations.

## Current identity

- Release version: `V4.41`
- Strategy and ranking major version: `V4.4`
- Compatibility: completed V4.4 and V4.41-minor results remain in the same cumulative ranking when their ranking lineage and result contracts match
- Strategy: policy-specific V4.4 `rolling_tr_sum` identities; `all_window` is the default and `exclude_marked` is optional
- Entry baseline: `rolling_tr_sum` only
- Baseline sampling: `all_window` uses every finite TR15 atom inside one continuity segment; `exclude_marked` omits `baseline_excluded` atoms and backfills older eligible atoms. Rankings and result identities stay separate.
- Entry: `calculated_threshold`
- Pending entry: within the existing 120-bar continuous wait boundary, fill at the open of the first real-trade bar; do not require a price retrigger and do not cancel on a higher high or structural reversal
- Exit: a prior rebound line fills at open when `open >= trigger`, otherwise at trigger when `high >= trigger`; a strict-new-low real bar confirmed by its close fills at that close; a non-real signal remains pending for the next real-trade open
- W candidate: available 1..W prefix with no full-W gate; `w_open_to_end_low_drop = open[start] - low[end]`, not an internal high-to-later-low maximum ordered decline
- Entry qualification: baseline, drop, and `K × baseline` must all be positive; positive equality remains eligible.
- Ranking: gross and cost-adjusted modes change both ordering and displayed returns; cost-adjusted is the default. Each campaign loads its instrument-bound cost model and calculates notional from reference price and point value.
- Scenario: optional and instrument-bound. A schema-v5-or-later profile with no scenario set has no scenario qualification.

`RELEASE.json` declares the presentation release and its major-version compatibility boundary. `research_variants\short_momentum_net_drop_rebound_v4_4\SOURCE_MANIFEST.json` remains the strategy source and identity closure. The current K200 profile and K200 campaign evidence remain under `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles` and `results\campaigns`; other instruments use their own profiles, campaign manifests, and ranking lineages.

## Formal source release

Scheme A publishes the GitHub source, `Backtest_V4.41_source_release_20260809.zip`, its SHA-256 sidecar, its machine audit, and tag `V4.41`. This compact Windows package includes the current source, tests, runtime contracts, project documents, scenario tooling, cumulative browser payload, stable main shell, and one representative per-trade chunk. Multi-gigabyte historical ledgers remain outside the public release assets. The acceptance gate is 112 passed, 2 explicit historical-artifact skips, and 0 failed from an independent extraction.

## Result packages

Completed evaluations are registered under `results\evaluation_packages\<instrument_id>\<start_YYYYMMDDTHHMMSS>__<end_YYYYMMDDTHHMMSS>`. Folder names record only the instrument and exact evaluated interval. Training, test, transfer, holdout, and descriptive roles belong in the package `EXPERIMENT.md` or a comparison plan, so the same package can be reused without renaming.

Every package owns an `evaluation_manifest.json`, experiment note, neutral parameter summary, browser summary, immutable trade-record reference, and per-trade entry. `results\evaluation_packages\catalog.json` is the machine-readable registry. Generic multi-package comparisons live under `results\evaluation_comparison\comparisons` and join package data by exact `combo_id`.

The current retained K200 and SImain evidence can be registered with:

```powershell
.\.venv\Scripts\python.exe tools\build_v4_4_evaluation_framework.py build-current
```

The builder audits exact data parity before changing the stable comparison redirect. It does not rerun the strategy or modify existing main and per-trade artifacts.

For a newly completed instrument and interval, copy `runtime_inputs\templates\EVALUATION_PACKAGE_SPEC.template.json`, bind its source files and column mapping, then register it with:

```powershell
.\.venv\Scripts\python.exe tools\register_v4_4_evaluation_package.py D:\path\to\evaluation_package_spec.json
```

Registration creates one immutable date package and updates the shared package catalog. Experiment roles stay in the spec-generated record and later comparison plans.

## Market scenarios

`runtime_inputs\scenarios\market_catalog.json` registers the fixed market intervals available in the selector. The current catalog contains K200 training, K200 subsequent-test, and the current SI interval. `runtime_inputs\scenarios\scenario_catalog.json` stores named scenarios; each scenario binds one market interval and one or more selected ranges.

Build the selector and apply saved scenarios to existing evaluation packages with:

```powershell
.\.venv\Scripts\python.exe tools\build_v4_41_scenario_manager.py
.\.venv\Scripts\python.exe tools\apply_v4_41_scenario.py --all
.\.venv\Scripts\python.exe tools\apply_v4_41_scenario.py --scenario-id scenario_1
```

The selector is published at `results\market_scenario_manager\index.html`. Scenario rankings are written under `results\scenario_analysis\<scenario_id>` and reuse the current cumulative-main interface and package-owned per-trade routes. Applying a scenario reads completed summaries and immutable trade records; it does not rerun the strategy or change an evaluation package.

## Runtime setup

Use Windows and Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-v4_4.txt
npm ci
npx playwright install chromium
```

See `RUNTIME.md` for repository-relative runtime inputs, hashes, test commands, locks, and the run gate.

## Validation

Run the V4.41 release tests and offline documentation builds:

```powershell
.\.venv\Scripts\python.exe -m pytest research_variants\short_momentum_net_drop_rebound_v4_4 -q
npm run build:dashboard
```

## Current handoff package

The fixed package builder creates the V4.41 source/runtime/management handoff while retaining the V4.4 strategy-major transaction-record identity:

```powershell
powershell -ExecutionPolicy Bypass -File tools\package_v4_4_with_trade_records.ps1
```

The archive includes `RELEASE.json`, the current source and project documents, repository-local runtime inputs, canonical reports, and immutable raw/derived transaction ledgers from completed stages. Full result directories, HTML snapshots, dependencies, caches, `.git`, and `.omo` stay outside the ZIP.

## Run gate

Every campaign declares one mode: `transfer_exact`, `target_local_refinement`,
`continuation_search`, or `fresh_search`. Execution requires a reviewed,
hash-bound plan, immutable raw closure, and fixed-template delivery. No
parameter is accepted by default.
