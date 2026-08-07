# Backtest Management

## Required operation

Before each backtest starts, append one row to this document. The row must state:

- the instrument;
- the exact market-data file used by the run;
- the first timestamp included in the backtest interval;
- the last timestamp included in the backtest interval.

Use the evaluated interval, not the full boundary of a file that also contains warm-up or later unused data. Keep the row when a run stops early so every launched backtest remains visible. HTML generation is unrelated to this record.

After raw compute closes, record the result package in the run's `EXPERIMENT.md` and `evaluation_manifest.json`. The package path is derived from the instrument and exact evaluated interval. Record the candidate-set source, plan, completion status, parameter-summary path, immutable trade-record path, per-trade entry, and every comparison plan that consumes the package. Experiment roles belong in those records and never rename the package directory.

For a newly completed run, fill `runtime_inputs\templates\EVALUATION_PACKAGE_SPEC.template.json` and invoke `tools\register_v4_4_evaluation_package.py`. The declaration maps run-specific summary columns into the shared parameter/metric namespace. An existing package manifest is immutable; another candidate population for the same instrument and interval receives another candidate-set artifact within that package through a reviewed publication change.

This rule applies to every instrument and every evaluation interval. A ready instrument profile, complete data contract, exact start/end, and an authorized campaign plan are the execution boundary; K200-specific naming is not part of the management contract.

## Run log

| Instrument | Market-data file | Backtest start | Backtest end |
| --- | --- | --- | --- |
| K200 temporal migration R1 | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-08 23:52:15` | `2026-07-17 05:59:45` |
| K200 temporal migration R2 | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-20 08:45:00` | `2026-07-25 05:59:45` |
| K200 temporal migration R3 | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-27 08:45:00` | `2026-08-01 05:59:45` |
| K200 temporal migration R4 final holdout | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-08-03 08:45:00` | `2026-08-07 03:21:45` |
| K200 temporal migration full-test descriptive replay | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-08 23:52:15` | `2026-08-07 03:21:45` |
| SI exact transfer of 100 new temporal candidates | `D:\Code\data\ibkr\SImain\SImain_15s_20260128_20260223_session_filled.csv` | `2026-01-29 00:00:00` | `2026-02-23 23:59:45` |
| K200 test replay for the combined 350 candidates | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-08 23:52:15` | `2026-08-07 03:21:45` |
| K200 current-optimal 100-candidate initial forward replay | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-08 23:52:15` | `2026-08-07 03:21:45` |
| K200 current-optimal positive-training 100-candidate corrected replay | `D:\Code\backtest-release\Backtest V4.4\runtime_inputs\market_data\k200_clean_15s_session_filled.csv` | `2026-07-08 23:52:15` | `2026-08-07 03:21:45` |
