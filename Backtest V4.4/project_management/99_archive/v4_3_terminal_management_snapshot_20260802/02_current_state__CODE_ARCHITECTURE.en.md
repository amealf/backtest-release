# Code Architecture

## Scope

This document records the current code structure, execution flow, component responsibilities, and system boundaries for **Backtest V4.3 Research**. Record only architecture supported by inspected code, configuration, generated artifacts, or explicit user confirmation.

## Architecture status

The V4.3 package is an offline Windows/Python backtest pipeline with repository-local runtime inputs and generated HTML review artifacts.

## Current system flow

```text
runtime_inputs + neutral baseline markers + scenario definition + reviewed parameter plan
  -> v4_3_engine.py
  -> run_v4_3_resumable_campaign.py (raw immutable batches/stage)
  -> analyze_v4_3_scenario_3_stage.py
       -> build_v4_3_review_delivery.py (fixed trade template, four workers)
       -> build_v4_3_combined_union_analysis.py (stable cumulative routes)
```

```text
shared scenario workflow:
D:\Code\backtest-release\shared_tools\scenario_manager\index.html
  -> browser-local cross-version scenario library
  -> reviewed version-specific scenario definition / plan input
```

## Component responsibility register

| Path or component | Responsibility | Inputs | Outputs | Status and evidence |
| --- | --- | --- | --- | --- |
| `runtime_inputs\RUNTIME_INPUTS.json` | Bind movable runtime assets. | Repository-relative files | Hash/size closure | Current |
| `data_preparation\prepare_dataset.py` | Produce policy-neutral baseline markers and a relative-path preparation manifest. | K200 15-second bars and cleaning audit | Marker atoms/events/report/manifest | Current |
| `code\v4_3_engine.py` | Select baseline eligibility by coordinate policy, then compute signals, entries, pending exits, fills, and trades. | Prepared bars and one coordinate | Policy-identified audited trade records | Current |
| `code\run_v4_3_resumable_campaign.py` | Validate plan identity and execute resumable batches. | Reviewed plan | Raw stage artifacts and manifests | Current |
| `code\analyze_v4_3_scenario_3_stage.py` | Validate closed stage evidence and compute review tables. | Closed raw stage | Analysis CSV/manifest/HTML | Current |
| `code\build_v4_3_review_delivery.py` | Generate trade review from the required historical template. | Summary, trades, local template assets | Lazy trade chunks and HTML | Current |
| `code\build_v4_3_combined_union_analysis.py` | Anti-join compatible completed stages and publish stable routes. | Completed stage analyses | Cumulative main/trade delivery | Current |
| Cross-version scenario manager | Reuse the named market-selector interface to save multi-interval scenarios under automatic monotonic `情景N` names, select them from a two-row button strip, clear, archive, and restore. | Bundled K200 15-second selector data or a user CSV/TSV | Browser-local shared scenario library | Current shared tool |

## Dependencies and call relationships

- Active runtime dependencies are Python, NumPy, pandas, repository-local data/templates, and Windows `msvcrt` locks.
- Node/Playwright are validation dependencies for dashboards and browser QA.
- Paths inside `runtime_inputs\provenance` do not participate in active calls.

## Interfaces and boundaries

- Raw compute, stage analysis, and cumulative publication have separate writer locks and output ownership.
- Entry and exit audit columns are a durable interface to analysis and trade HTML.
- One raw stage uses one baseline-sampling policy. Fingerprints and stage/batch/completion manifests bind the resolved policy.
- The cumulative builder accepts compatible stages from either supported policy and publishes policy-to-strategy/result identity maps; ranking never crosses policy partitions.
- A completed stage may be delivered while a different stage computes because raw and derived output roots are disjoint.
- No compute begins from a documentation or validation command.
- The shared scenario manager is the human editing and lookup interface. A backtest still requires a reviewed, immutable version-specific scenario definition and identity before compute.

## Architecture change log

| Date | Change | Evidence | Status |
| --- | --- | --- | --- |
| 2026-08-01 | Created the neutral code-architecture record. | Project-management initialization | Unconfirmed until project code is inspected |
| 2026-08-01 | Bound V4.3 runtime inputs to the checkout and added pending synthetic-exit execution. | Current code, manifests, and tests | Current |
| 2026-08-02 | Made preparation markers policy-neutral and propagated the selected baseline policy through compute, identities, manifests, analysis, and fixed-template delivery. | Current code, regenerated manifests, and tests | Current |
| 2026-08-02 | Added one version-neutral scenario-manager entry, then refined it to automatic sequential names, a two-row selector, and 15-second default display while retaining historical result snapshots. | Shared HTML and desktop/mobile browser QA | Current |

## Maintenance

Update this document when a code entry point, component, dependency or call relationship, data flow, interface boundary, generated artifact, or component responsibility is added, renamed, retired, or materially changed. Update `SOURCE_OF_TRUTH.en.md` when an authoritative path changes, `CURRENT_VERSION.en.md` when behavior changes, and `04_decisions\DECISIONS.en.md` when the architecture change represents a durable choice.
