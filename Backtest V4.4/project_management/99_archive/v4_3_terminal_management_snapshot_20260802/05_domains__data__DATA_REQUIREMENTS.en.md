# Data Requirements

## Scope

This optional module exists because maintained data sources, schemas, migrations, or pipelines are part of **Backtest V4.3 Research**.

## Source and contract register

| Dataset or contract | Owner/source | Schema or format | Validity range | Status |
| --- | --- | --- | --- | --- |
| K200 15-second bars | `runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | CSV; ordered OHLCV, trade count, synthetic flag | Hash-bound source range | Current |
| Immediate-recovery cleaning audit | `runtime_inputs\market_data\data_preparation_audit.json` | JSON | Must bind the source hash | Current |
| V4.3 preparation contract | `runtime_inputs\data_preparation\data_preparation_manifest.json` | JSON schema v4 with relative artifact paths and supported-policy declaration | Current source hash and preparation code | Current |
| Baseline marker atoms/events | Preparation manifest artifacts | CSV/JSON; policy-neutral `baseline_excluded` plus `eligible_if_excluding_marked` | One-to-one datetime alignment with source | Current |

## Quality and lineage

- The 15-second source must contain datetime, OHLC, volume, trade count, and synthetic-empty-bar status and must match its bound hash.
- Preparation artifacts must align one-to-one by datetime and resolve relative to their manifest.
- Preparation never chooses a baseline-sampling policy. `all_window` ignores markers for eligibility; `exclude_marked` uses the neutral marker. Both policies preserve the same source rows for signals, fills, exits, and charts.
- The source must contain one row per datetime. Duplicate timestamps stop the engine; no keep-first/keep-last merge policy is current.
- Synthetic status is independent of `baseline_excluded`. Neither current policy removes every synthetic bar.
- Historical absolute paths may be retained as provenance but cannot be opened by active runtime code.
- For a user-requested ZIP, include the hash-bound 15-second OHLC source. Exclude campaign roots, raw batches, stage summaries/trades, chunks, snapshots, and other compute-result payloads; include analysis report documents explicitly requested by the packaging contract.

## Validation register

| Check | Scope | Failure action | Evidence |
| --- | --- | --- | --- |
| Source SHA-256 | Full market source | Stop before execution | Runtime/source manifests |
| Preparation identity and artifact SHA-256 | Manifest plus filter files | Stop before execution | Engine and preparation validation |
| Timestamp alignment | Source versus filter atoms | Stop before execution | Engine validation |
| Unique source datetime | Full market source | Stop before execution | Preparation and engine validation |
| Supported/default baseline policies | Preparation contract versus engine/plan | Stop before execution | Runner validation |
| Real-trade execution bar | Entry and signal-driven exit fills | Stop delivery on mismatch | Engine/review regression tests |

## Maintenance

Update when a durable source, schema, contract, quality threshold, lineage rule, access policy, retention period, migration, backup, or recovery procedure changes.
