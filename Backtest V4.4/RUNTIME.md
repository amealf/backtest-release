# V4.4 runtime contract

## Platform and dependencies

The current runner and union writer use `msvcrt`; execution is Windows-only. Use Python 3.12, NumPy 2.3.5, pandas 3.0.1, pytest, Node 20 or newer, and Playwright 1.61.1 with Chromium.

Create local environments inside the checkout. Virtual environments, Node modules, browser profiles, screenshots, logs, locks, and result payloads are ignored and must remain uncommitted.

## Repository-local runtime inputs

All active runtime dependencies resolve from `runtime_inputs` relative to this checkout:

- `runtime_inputs\market_data\k200_clean_15s_session_filled.csv` — clean K200 15-second source.
- `runtime_inputs\market_data\data_preparation_audit.json` — immediate-recovery cleaning audit.
- `runtime_inputs\data_preparation\data_preparation_manifest.json` — V4.4 schema-5 causal-availability manifest; its runtime artifact paths are relative to the manifest. It declares `all_window` as default and causal `exclude_marked` as optional.
- `runtime_inputs\templates\historical_v4_main.html` — required historical main-page template.
- `runtime_inputs\templates\historical_v4_trade.html` — required historical trade-page template.
- `runtime_inputs\templates\plotly.min.js` — required local Plotly asset.
- `runtime_inputs\templates\market-intuition-selector.html` — required market-selector source.

`runtime_inputs\RUNTIME_INPUTS.json` binds the relative paths, sizes, and SHA-256 values. Files under `runtime_inputs\provenance` preserve older source records and may contain historical absolute paths; active Python code never opens those historical locations.

## Optional legacy CLI inputs

The standalone legacy engine CLI may use `runtime_inputs\legacy\canonical_volume_atoms.csv` and `runtime_inputs\legacy\selected_events.csv` when supplied. The compatibility review build requires explicit stage, validation-stage, and output arguments. None of these optional files is required by the current engine, runner, analyzer, or union builder.

## Path and identity rules

Active defaults are derived from `Path(__file__)` and the repository root. Moving the V4.4 folder does not require source edits. Every later plan must use repository-relative or explicitly supplied inputs and must be hash-bound before launch.

`SOURCE_MANIFEST.json` must be rebound after any implementation, scenario-definition, template-policy, or runtime-input change.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest research_variants\short_momentum_net_drop_rebound_v4_4 -q
npm run build:dashboard
```

The repository contains no separate research-hub builder or QA program. The
previous `build:hub`, `test:hub`, and `qa:hub` commands were retired because
their referenced files were absent. The cumulative research and per-trade
entries are produced by the Python stage-analysis and union-delivery tools.

Result-dependent tests may skip while all stage and cumulative output roots are absent. A skip is valid only when it names that no-result dependency.

## Compute and memory gate

The temporary one-coordinate plan is complete. The user has authorized bounded multi-round exploration after the current source identity and each round plan are reviewed and hash-bound.

When authorized, use exactly three workers, 12 coordinates per batch, and a 4,096 MiB minimum-free-memory reserve. Record available memory before launch and in every batch progress/completion handoff. Pause and resume on the same identity if the reserve would be breached; never overcommit memory.

Raw compute owns `.v4_4_runner.lock`, immutable stage delivery owns `.v4_4_delivery.lock`, and cumulative publication owns `.v4_4_union.lock`. One phase has one writer per output.
