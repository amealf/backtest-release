# Project Constraints

## Purpose

Record durable constraints that apply across the project. Keep domain-specific rules in an approved optional module rather than assuming that domain exists.

## Confirmed constraints

| Constraint | Rationale | Evidence or owner | Status |
| --- | --- | --- | --- |
| Preserve existing user files during management-folder initialization. | Initialization must remain reviewable and reversible. | Project-management protocol | Current |
| Keep English operational documents and Chinese review mirrors aligned. | Agents need one execution source while reviewers need a readable mirror. | Project-management protocol | Current |
| Preserve V4.2 source and historical results. | V4.3 has a new folder and identity boundary. | User confirmation | Current |
| Use only `rolling_tr_sum`. Support two baseline-sampling policies: `all_window` is the default; `exclude_marked` is optional. Keep their strategy/result/combo identities and rankings separate. | Preserve both confirmed sampling choices without mixing evidence. | User confirmation and engine evidence | Current |
| Pending entry fills at the first real-trade bar open within its established wait and continuity boundary. Do not add price retrigger, higher-high cancellation, or structural-reversal cancellation. | Preserve the confirmed entry behavior. | User confirmation | Current |
| Rebound or speed exits triggered without a real trade become pending and fill at the next real-trade bar open. | Prevent synthetic fills. | User-confirmed P0 repair | Current |
| Active runtime code may use repository-relative inputs only. Historical absolute paths are provenance records, never runtime dependencies. | Keep the V4.3 folder movable and reproducible. | User confirmation | Current |
| Use the established main/trade templates and confirm any requested substitution or scope expansion with the user. | Preserve the accepted review workflow and communication boundary. | User correction | Current |
| When the user requests a ZIP, include current-version code, `project_management`, analysis reports, and the 15-second OHLC dataset; exclude compute-result payloads. | Keep handoff packages reproducible without bundling campaign output. | User instruction | Current |
| When a task needs to use or inspect a scenario, open the shared cross-version scenario-manager HTML; do not infer the definition from a version-specific result page. | Keep one editable scenario library across backtest versions. | User instruction | Current |

## Quality and integrity

- Make current paths, configurations, inputs, outputs, and validity boundaries traceable when they exist.
- Keep every strategy/result/combo/audit/fingerprint identity versioned for V4.3.
- Neither supported sampling policy removes all synthetic bars. A real-only or synthetic-excluding baseline requires a separate approved research contract.
- Do not run a backtest from implementation, documentation, or validation work alone.
- Do not place changing experiment counts or rankings in this durable constraint file.

## Open assumptions

| Assumption | Why it matters | Confirmation needed |
| --- | --- | --- |
| Future parameter grid | Determines the next campaign identity and output root. | Separate user instruction and reviewed plan |

## Maintenance

Update only when the user accepts, changes, or retires a durable constraint. Record the associated rationale in `04_decisions\DECISIONS.en.md` when the change represents a lasting choice.
