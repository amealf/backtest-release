# Change Reasons

## Purpose

Explain material behavior changes that future work may need to audit. Each entry preserves the reason, prior behavior, updated behavior, and evidence impact. This complements the concise decision register and work-progress log.

## When to add an entry

Add an entry when a repair or other material change alters observable behavior, defaults, validity boundaries, reproducibility, evidence interpretation, or an important user workflow. Use `DECISIONS.en.md` for durable choices and `WORK_PROGRESS.en.md` for completed-work summaries.

Routine status updates and minor presentation edits do not need an entry unless they materially affect review or use.

## 2026-08-02 — Version-neutral scenario library

**Reason:** The scenario viewer had been presented as version-specific, which fragmented scenario lookup and editing.

**Prior behavior:** Version result pages could display scenario requirements, while the named market selector could create intervals but did not persist a reusable scenario or manage saved definitions.

**Updated behavior:** A shared copy of the named selector saves one or more intervals under automatic monotonic `情景N` names, including archived numbers in the sequence so names are not reused. Active scenarios are two rows of buttons above the chart and reload on click. Deleted scenarios remain recoverable. The title/theme header was removed, upload moved to the lower action row, and 15-second display is the default alongside 1/5/15/30/60/120-minute choices.

**Evidence impact:** Future tasks consult one stable HTML and select a saved scenario button before binding a reviewed immutable scenario definition into a version-specific plan. Original selector bytes and historical result snapshots remain unchanged.

**Validity boundary:** Browser-local storage is shared through the stable HTML path on the same browser profile. It does not by itself authorize a backtest or replace plan/manifest identity checks.

## 2026-08-02 — Selectable baseline sampling

**Reason:** V4.3 must retain both the default all-window baseline and the optional marker-excluding baseline. Existing preparation wording incorrectly implied that one choice was universal.

**Prior behavior:** The engine always used every finite TR15 atom; preparation emitted a field named as if marker eligibility were already a backtest decision; downstream identities and rankings had no sampling-policy axis.

**Updated behavior:** Preparation publishes a neutral marker and `eligible_if_excluding_marked`. Plans select one policy per stage. `all_window` remains default; `exclude_marked` backfills older unmarked finite atoms inside the continuity segment. Policy-specific identities and rankings propagate through compute and delivery.

**Evidence impact:** Preparation schema/identity, runtime/source manifests, combo/fingerprint/stage/completion/analysis/catalog identities, and current UI controls changed. Existing V4.3 result count remains zero, so no result migration occurred.

**Validity boundary:** Both policies use `rolling_tr_sum`; neither removes all synthetic bars.

## 2026-08-02 — Audit and input-integrity corrections

**Reason:** `baseline_pending_atom_count` was forced to zero despite real pending low-activity atoms, and engine-level duplicate datetime handling was not fail-closed.

**Prior behavior:** Pending count used an all-false array; duplicate timestamps survived sorting and could distort continuity.

**Updated behavior:** Pending count uses the actual low-activity pending state inside the physical baseline span. Duplicate source timestamps raise before execution. Current UI sources remove `tr_average`, and compatibility ranking includes `>=10`.

**Evidence impact:** Audit fields and input validation change; trading rules are unchanged by the pending-count repair. Duplicate malformed inputs are newly rejected.

**Validity boundary:** No `_rank()` duplicate assignment was found, so ranking code received no speculative cleanup for that report.

## 2026-08-01 — Real-trade exit execution

**Reason:** Rebound and speed conditions could be true on synthetic/no-trade bars, but those bars cannot execute a market fill.

**Prior behavior:** The engine closed immediately using a synthetic bar-derived close or calculated threshold.

**Updated behavior:** The first trigger reason, time, theoretical price, and evidence are frozen as `pending_exit`; the position remains open and fills at the next real-trade bar open, then resets flat.

**Evidence impact:** New pending-exit audit fields, V4.3 identities, engine tests, analyzer checks, and trade-review source validation are required. V4.2 results are not reusable as V4.3 results.

**Validity boundary:** Applies to rebound and speed signal exits. Segment-end closure remains its separate forced-end rule.

## 2026-08-01 — Portable active runtime

**Reason:** Absolute machine paths prevented clean relocation and reproduction.

**Prior behavior:** Active defaults opened data and template assets from specific D:/F: locations.

**Updated behavior:** Active defaults resolve repository-relative files under `runtime_inputs`; historical absolute paths remain provenance records only.

**Evidence impact:** Runtime and source manifests bind local paths and hashes. Moving the folder no longer requires code edits.

**Validity boundary:** Applies to active V4.3 runtime. Provenance JSON may retain historical paths without becoming a dependency.

## Entry format

```markdown
## YYYY-MM-DD — Short change title

**Reason:** Why the change was needed.

**Prior behavior:** What happened before the change.

**Updated behavior:** What happens now.

**Evidence impact:** Which outputs, tests, records, or interpretations are affected.

**Validity boundary:** Where the new explanation applies and what remains unchanged or unconfirmed.
```

## Maintenance

Place new entries above older entries. Link evidence instead of copying large output. Update `CURRENT_VERSION.en.md`, `CODE_ARCHITECTURE.en.md`, or `DECISIONS.en.md` as well when the same change affects current behavior, architecture, or a durable choice.
