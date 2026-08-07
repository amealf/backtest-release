# Project Goal

## Durable objective

Repair execution realism and keep V4.3 reproducible from repository-local runtime inputs.

## Current milestone

Preserve the terminally closed five-round max-W evidence under the frozen four-view contract; create no Round 6 and accept no parameter. Any external validation requires a new user-authorized contract.

## Success criteria

| Criterion | Measure | Status |
| --- | --- | --- |
| Execution realism | Signal-driven exits cannot fill on bars without real trades. | Current |
| Entry continuity | Pending entry retains the confirmed first-real-trade-open behavior. | Current |
| Reproducibility | Active runtime dependencies resolve from repository-relative, hash-bound inputs. | Current |
| Auditability | New V4.3 identities and pending-exit evidence distinguish these results from V4.2. | Current |

## Non-goals

- Do not modify V4.2 source or results.
- Do not run parameter exploration without an explicit user instruction, a frozen V4.3 plan, exact anti-join, and passed validate-only gates.
- Do not claim a parameter has been accepted from implementation tests.

## Maintenance

Update this file only when the user changes the durable objective, priority, or accepted success criteria. Put temporary work in `03_active_work\CURRENT_TASKS.en.md`.
