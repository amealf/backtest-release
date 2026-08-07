# Data Requirements

## Instrument profile binding

- Every new instrument binds its market-data and preparation-manifest path, size, SHA-256, and `bar_seconds` in an instrument profile. Campaign plans consume those bindings; code does not select a product-specific filename.
- The profile records timezone, exchange session, continuous-contract policy, warm-up boundary, gap policy, low-activity policy, and optional scenario set.
- Missing data, gap, or low-activity decisions keep the profile non-executable. K200 remains the only ready profile until another profile is explicitly completed.
- The strategy's authoritative granularity is 15 seconds because the primary K200 research uses 15-second bars. Every transfer target must also use 15-second bars. Profile, preparation manifest, and observed timestamp grid must agree; any mismatch stops before compute and is reported to the user.
- Gap and low-activity configuration/implementation files are hash-bound. Their policy IDs and hashes must match the preparation manifest; `status=ready` alone is insufficient.

## Cross-instrument evaluation data

- SImain comparison data is explicit SIH6 15-second session-filled OHLC in America/Chicago time. The current test interval is 2026-01-29 through 2026-02-23; data begins on 2026-01-28 to supply rolling warm-up.
- Warm-up bars may feed rolling indicators, but the transaction population includes only positions whose entry time is inside the declared test interval.
- Record the exchange session, source timezone, continuous-contract schedule, roll count and adjustment policy, real gaps, synthetic bars, zero-trade exposure, and the presence or absence of an instrument-specific low-activity policy.
- Cross-instrument percentages and bps are comparable. Preserve raw price-point MFE, MAE, and realized point totals for within-instrument inspection.

## Active sources

- K200 session-filled 15-second OHLC under `runtime_inputs/market_data`: 233,368 rows from `2026-05-23T00:00:00+09:00` through `2026-08-07T03:21:45+09:00`, SHA-256 `9760d367a109777c4789ce45d982a6c0708bacddad8f549450ed94f81ad5c405`.
- Prepared baseline atoms, events, and schema-5 manifest under `runtime_inputs/data_preparation`.
- The current ready binding is `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\k200m.json`; SImain and NQ templates are input checklists, not active sources.

## Evaluation-result package contract

- The logical root is `results\evaluation_packages`; physical bytes continue through the existing results junction to F. Package identity is `<instrument_id>\<start_YYYYMMDDTHHMMSS>__<end_YYYYMMDDTHHMMSS>` and uses the evaluated interval rather than warm-up or unused file coverage.
- `evaluation_manifest.json` records the instrument, display name, exact interval, timezone, bar size, status, artifact paths, storage mode, provenance, and experiment-record path. It does not assign train/test/transfer meaning.
- `parameter_summary.csv` contains the package's completed parameter population in a neutral metric namespace. `browser_summaries\<candidate_set_id>.js` is a smaller projection used by a comparison page; it never replaces the authoritative summary or trade records.
- `trade_records\trades.csv` retains immutable transaction evidence. A historical adapter may use a same-volume hard link whose manifest carries the already declared source hash. New native runs append immutable batches and publish one consolidated package summary after closure.
- `trade_review\index.html` is the package-owned per-trade entry. A compatibility package may redirect to a retained historical page while preserving query and hash; a native package owns its process payload, catalog, and deterministic per-coordinate chunks.
- `EXPERIMENT.md`, the comparison plan, and bilingual project records state what the run tested and how a package is used. The date directory remains stable when a later experiment assigns a different role.
- A comparison reaches complete status only when its candidate set has unique exact `combo_id` values and every selected package supplies the declared population. Comparison pages load summaries, not complete trade records.
- New completed evaluations use `runtime_inputs\templates\EVALUATION_PACKAGE_SPEC.template.json` to declare source files, neutral parameter fields, neutral metric mappings, interval identity, and the current experiment description. `tools\register_v4_4_evaluation_package.py` refuses to replace a package whose manifest already exists.

## Per-download README contract

- Every instrument acquisition creates `README.md` inside that exact download directory. Refresh the same file whenever the run is resumed or completed; do not keep the only explanation in project memory or a separate report.
- Record the instrument, upstream source and data type, source/request timezone, half-open requested interval, actual first/last observation when available, creation/resume/completion times, status, and principal raw/derived/audit files.
- State how candidate contracts are compared, which contract is considered main, every effective contract interval with inclusive/exclusive boundaries, and whether adjustment is applied. When a supplement inherits an earlier audited decision instead of recomputing candidates, say so explicitly.
- State the lineage and merge contract: the supplement starts at the parent's exclusive end, raw download directories remain separate, derived rows append in timestamp order with identical schemas, overlaps/duplicates/reversals fail, real gaps remain, and roll price differences remain when the series is unadjusted.
- Keep a chronological update table in the README so later data management can reconstruct the acquisition chain across all retained download directories.

## Baseline availability contract

- Normal atom: available when its bar completes.
- Pending low-volume atom: immediately available and entry-neutral while the run remains unconfirmed.
- Pending interval later recovered: no exclusion is created.
- Confirmed interval: every low-volume atom from the run start carries the confirmation time in `baseline_excluded_from`; it is eligible before that time and unavailable at and after that time.
- `confirmed_low_activity_active` is true from the confirmation atom through the final low-volume atom. The first normal-volume atom is outside the gate.
- `all_window`: every finite TR15 atom inside one continuity segment remains eligible.
- `exclude_marked`: at calculation time t, an atom is eligible only when `baseline_available_from <= t`.
- `confirmed_low_activity_gate`: at calculation time t, an atom is eligible only when `baseline_excluded_from` is empty or greater than t; active gate atoms block new entries and cancel unfilled entry orders.

Duplicate timestamps fail closed. Historical absolute paths are provenance only and are not active runtime dependencies.

## ZIP handoff records

- A user-requested ZIP includes the hash-bound 15-second OHLC input and transaction records for every completed V4.4 stage.
- Package each immutable raw `batches/**/trades.csv` and each derived `analysis/stage_trades.csv` below `trade_records/`, preserving file bytes.
- `trade_records/TRADE_RECORDS_MANIFEST.json` must state the source campaign/stage, record role, relative source path, row count, size, SHA-256, and the source completion/stage-manifest hashes.
- Other result payloads remain outside the archive: campaign files, batch manifests, summaries, grid files, chunks, snapshots, HTML output, logs, locks, and partial or failed stages.
