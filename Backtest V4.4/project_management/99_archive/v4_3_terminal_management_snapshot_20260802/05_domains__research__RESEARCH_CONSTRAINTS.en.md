# Research Constraints

## Scope

This optional module exists because research questions, experiments, or result interpretation are maintained within **Backtest V4.3 Research**.

## Research register

| Question or hypothesis | Method | Evidence | Status |
| --- | --- | --- | --- |
| How does the max-completed-W rebound basis affect Scenario-1-qualified total return, unrestricted total return, and unrestricted average return at `>=10` and `>=20` trades? | Five closed rounds: broad coverage, interaction refinement, bounded continuation, and one terminal two-block micro-refinement | 684 unique coordinates, 232,225 trades, five immutable/delivered stages; terminal delivery evidence SHA-256 `6d3cc38cfe98d16d2b095990f2f6a8c85d69c230b9b873bae61d5c414c6751a0` | Terminal in-sample evidence; no Round 6 or acceptance |
| How much do synthetic zero-TR atoms change the baseline? | Parameter-independent diagnostic comparing `all_window` with a real-only 180-atom backfilled reference | Verified on the current training range; not a trading result | Diagnostic complete |

## Method constraints

- Separate hypothesis, method, observed result, interpretation, and limitation.
- Record inputs, preprocessing, sampling, parameter selection, evaluation, and exclusion rules.
- Prevent leakage, future information, or non-independent comparisons when relevant.
- Keep each run traceable to code, configuration, data, environment, and output.
- Treat a selected result as evidence, not automatic proof or production readiness.
- Keep the current scenario definitions unchanged; Scenario-1 filtering does not change trade calculations.
- `rolling_tr_sum` is the sole aggregation method. Rank `all_window` and `exclude_marked` separately and do not create a combined score.
- One stage uses one baseline-sampling policy. The cumulative page may show both only through explicit policy partitions and identity mappings.
- Current high-return analysis has exactly four independent views: Scenario-1-qualified total return without another trade-count filter; unrestricted total return without scenario or trade-count filtering; unrestricted average return at `train_trade_count >= 10`; and unrestricted average return at `train_trade_count >= 20`. Sort each by its primary metric descending and then `combo_id`.
- Keep gap-excluded return as a display-only gap-dependence audit. It cannot filter or rank Scenario-1 qualification, any primary view, continuation, seed selection, or candidate consideration. Keep other diagnostics in tables rather than ranking views or score terms.
- Round 2 used exactly 504 fresh anti-joined coordinates and 42 batches of 12. Round 3 used 72 fresh coordinates in three independent 24-point E×W×K/M blocks. Every round used three workers, a 4,096 MiB free-memory floor, and one why/run/observed/next Markdown.
- Round 4 activated exactly two terminal Round-5 branches: one Scenario-1 block and one unrestricted-total block. Round 5 closed at 24 coordinates with no average block. The five-round campaign is terminal; no Round 6 is authorized.
- Do not infer a real-only or synthetic-excluding policy from the marker comparison. Either choice requires an explicit sampling definition, new result identity, and user approval.
- A terminal view leader may be labelled an in-sample candidate for external validation/review under that view's exact eligibility and primary metric. A non-leader is not selected because its approved primary metric ranks lower or it is ineligible; gap-excluded return cannot accept or reject any row. Never claim acceptance from an in-sample result.
- Every campaign requires a pre-run rationale, bounded axes, anti-join, stop conditions, resumable batches, and post-run main/trade HTML plus analysis.

## Reproducibility

| Requirement | Current source | Status |
| --- | --- | --- |
| Input identity and validity range | `runtime_inputs\RUNTIME_INPUTS.json` and preparation manifest | Current |
| Run configuration and code revision | V4.3 plan, fingerprint, and `SOURCE_MANIFEST.json` | Required before run |
| Evaluation and acceptance rule | Scenario definition, four-view objective contract, and project constraints | Current |
| Output and audit trail | Plan/audit, batch/completion manifests, summary/trades, fixed-template HTML, analysis manifest | Required after run |

## Maintenance

Update when an accepted research question, method, constraint, evidence boundary, reproducibility rule, or interpretation standard changes. Put individual run state in the current task and interpretation-changing events in project history.
