# Project Operating Rules

<!-- generated-by: manage-project-context; review before merging into an existing user-owned file -->

This file is the concise operating protocol for agents working on <strong>Backtest V4.41 Research</strong>. V4.41 is a minor release on the V4.4 strategy and ranking lineage. Detailed context and maintenance rules live under `project_management`.

## Required reading

1. Read this file.
2. Read `project_management\00_START_HERE.en.md` for evidence priority and the complete task matrix.
3. Read the English documents selected by that matrix. English is the operational source; `*.zh.md` files are Chinese review mirrors.

## Read by task

- Scope or planning: read `01_goal\PROJECT_GOAL.en.md`, `01_goal\PROJECT_CONSTRAINTS.en.md`, `03_active_work\CURRENT_TASKS.en.md`, and `03_active_work\WORK_PROGRESS.en.md`.
- Parameter exploration design, execution, interpretation, or next-round handoff: also read and follow `05_domains\research\PARAMETER_EXPLORATION_GUIDE.en.md`, `05_domains\research\RESEARCH_CONSTRAINTS.en.md`, and `03_active_work\EXPERIMENT_PROGRESS.en.md`.
- Any backtest execution: also read and update `03_active_work\BACKTEST_MANAGEMENT.en.md`.
- Cross-instrument transfer or a fresh instrument search: also read `00_core\STRATEGY_CONTRACT_V4_4.en.md`, `10_instruments\INSTRUMENT_PROFILE_CONTRACT.en.md`, and `20_campaigns\CAMPAIGN_WORKFLOW.en.md`.
- Deliverable or current-behavior changes: also read `02_current_state\SOURCE_OF_TRUTH.en.md`, `02_current_state\CURRENT_VERSION.en.md`, and relevant `04_decisions\DECISIONS.en.md` entries.
- Behavior-changing repairs or changes whose rationale affects future work: also read `04_decisions\CHANGE_REASONS.en.md`.
- Code entry point, execution flow, component responsibility, dependency, data flow, or interface-boundary changes: also read `02_current_state\CODE_ARCHITECTURE.en.md` and `03_active_work\WORK_PROGRESS.en.md`.
- New direction: read `07_future\NEXT_DIRECTIONS.en.md` plus current goal, constraints, tasks, and work progress.
- Historical or inactive work: read `06_history\PROJECT_HISTORY.en.md` and the relevant `99_archive` note only when the task needs past evidence, reuse, or restoration.
- References: verify provenance in `08_references\README.en.md` before relying on retained source material.

Paths in this section are relative to `project_management`.

## Conversation-driven project memory

- At the start of project-changing work, review the conversation and delegated context available in the current task. Do not claim access to unseen chats.
- Compare the latest confirmed user statements with this file, `00_START_HERE.en.md`, and the task-selected English documents. A newer explicit correction supersedes an older summary.
- When project work establishes an accepted durable change, update the mapped English document and Chinese mirror before finishing the task, even when the user does not separately request documentation maintenance.
- Append completed or materially changed work to `03_active_work\WORK_PROGRESS.en.md` and its Chinese mirror. When a major delivery version or working method changes, archive obsolete detailed progress and leave a concise restoration pointer.
- Do not modify files for explanation-only questions, hypotheticals, or unaccepted ideas.
- Record concise facts, decisions, and useful rationale. Never store raw transcripts, secrets, or private personal data.

## Active domain modules

- Read `project_management\05_domains\data\DATA_REQUIREMENTS.en.md` before changing schemas, migrations, pipelines, lineage, or data contracts.
- Read `project_management\05_domains\research\RESEARCH_CONSTRAINTS.en.md` before changing methods, running experiments, or interpreting results.
- Read and follow `project_management\05_domains\research\PARAMETER_EXPLORATION_GUIDE.en.md` for every parameter-exploration task.

## Execution orchestration

- This is an instrument- and interval-neutral backtest project. Any executable instrument profile may run an independently declared evaluation interval; K200 is one retained evidence lineage, not the project-wide storage identity.
- Publish every completed evaluation under `results\evaluation_packages\<instrument_id>\<start_YYYYMMDDTHHMMSS>__<end_YYYYMMDDTHHMMSS>`. Directory names contain only the instrument and exact evaluated interval. Training, test, transfer, holdout, and descriptive roles belong in `EXPERIMENT.md`, the comparison plan, and bilingual management records.
- Each evaluation package owns `evaluation_manifest.json`, `parameter_summary.csv`, candidate-set browser projections, immutable trade records, an experiment record, and a per-trade entry. Append future compute as immutable batches and generate only missing per-coordinate review chunks.
- `results\evaluation_comparison` is the generic multi-evaluation reader. A comparison plan selects any completed date packages, joins them by exact `combo_id`, and keeps each package's ranking lineage independent. A comparison may describe transfer, temporal validation, or another experiment without changing package names.
- Existing stable entries and completed main/per-trade HTML are compatibility contracts. Build a parallel package-backed page, prove exact row/value parity and browser-visible parity, then change only the stable redirect. Preserve the prior entry and completed pages for immediate rollback.
- Every market-data acquisition directory must contain a `README.md`. Create it with the directory, refresh it on every resume/completion, and record the instrument, source, request/update times, contract split, main-contract selection rule, merge rule, lineage, and principal files.
- Before each backtest starts, append its instrument, exact market-data file, evaluated start time, and evaluated end time to `project_management\03_active_work\BACKTEST_MANAGEMENT.en.md` and its Chinese mirror. Keep the record if the run stops early.
- Backtest compute and evidence analysis are separate from HTML publication. Intermediate exploration rounds close immutable raw evidence and compact summaries without generating cumulative or per-trade HTML.
- Migration HTML uses the fixed cross-instrument builder after the migration series closes. A new combined run declares `incremental_parent_run`; unchanged target per-trade chunks are reused and only new-candidate chunks are generated. Do not regenerate every historical migration chunk for a small append.
- Parameter exploration follows a repeating AI-led cycle: multi-round leap search across nonadjacent legal regions, finite one-parameter grids around promising nonadjacent anchors, then a return to leap search. Individual rounds may belong to one phase; the complete cycle preserves both exploration and refinement evidence.
- A one-parameter grid has no fixed point-count cap. Freeze its finite bounds, step schedule, anchor, and expected coordinate count before compute. Continue, refine, widen, or stop only after the closed grid pattern is interpreted; every later grid must pass the completed+active+pending anti-join.
- The user reviews summaries and may correct objectives, anchors, ranges, or direction between cycles. While the user is unavailable, AI may continue multiple rounds only inside the already authorized time, instrument, method, data, and resource boundary. A method change or authority expansion still requires user approval.
- Publish the shared cumulative main entry `results\all_completed_union_analysis\index.html` and shared per-trade entry `results\all_completed_union_analysis\trade_review\index.html` once, after the exploration series ends or when the user explicitly requests an intermediate publication. The final publication contains all compatible completed rounds.

## Working rules

- Answer questions without changing files unless the user requests a change.
- Before changing any file, behavior, output, workflow, field, or artifact beyond what the user explicitly requested or accepted, stop and obtain user approval. Technical convenience, provenance concerns, adjacent cleanup, and agent interpretation do not expand the authorized scope.
- Prefer the smallest implementation that satisfies the confirmed contract. Add defensive branches only for a concrete, severe, and plausible failure; do not add speculative fallback layers.
- Keep verification proportional and minimal. Do not run broad regression, repeated hash audits, or browser matrices for a small change. Whenever work has a visible result and the environment allows it, capture a representative screenshot for the user and judge basic overlap, garbling, asymmetry, and obvious layout errors. Use Computer Use for local Chrome `file:///` pages when ordinary browser automation cannot open the path.
- When the user names an existing tool, template, or behavior, inspect and use that exact item. Ask for confirmation before substituting a newly authored implementation or expanding the requested scope.
- Treat the project's current real files, behavior, and produced artifacts as execution evidence. Use management documents to explain intent, status, and validity boundaries.
- Keep unknown facts marked as unconfirmed. Do not turn a proposal, future direction, or archived item into current truth.
- Preserve existing project files. Review `*.project-management.proposed*` siblings before any merge.
- Keep English and Chinese management documents aligned when either changes.
- When creating or materially editing human-facing Chinese project documentation or Chinese UI copy, invoke the `qu-ai-wei` skill after facts and technical meaning are stable. Preserve code identifiers, formulas, parameters, numbers, paths, commands, hashes, and English-Chinese semantic alignment exactly.
- Use HTML `<strong>...</strong>` for bold text in managed Markdown. Do not use Markdown `**...**`; inline code, fenced code, and glob paths such as `batches/**/trades.csv` keep their literal asterisks.

## Validation tiers

- Run the complete regression suite only for version upgrades and changes to the backtest engine, entry/exit or fill logic, return/cost calculations, data preparation, schemas, execution contracts, or result semantics.
- Use focused tests or a targeted browser interaction check for frontend behavior such as filters, sorting, selectors, buttons, navigation, or state synchronization. Exercise only the affected interaction and its nearest dependent state unless risk expands the boundary.
- For presentation-only changes such as copy, color, spacing, typography, alignment, responsive layout, or visibility, regenerate the affected HTML and perform one simple functional check. Screenshots are needed only when visual judgment is part of the request or the layout changed materially.
- Escalate to the higher tier whenever a change crosses tiers or its semantic impact is uncertain. Record the chosen tier and evidence in the handoff; do not run a broader suite by habit.

## After changes

- Classify accepted conversation changes with the maintenance-trigger table in `00_START_HERE.en.md`.
- Update the document named by `00_START_HERE.en.md` for the change type.
- Update `WORK_PROGRESS` after completed or materially changed work, and `CHANGE_REASONS` when a behavior change needs durable rationale.
- Run `node "project_management\tools\build_dashboard.mjs"` after changing managed Markdown or the manifest.
- Verify the generated dashboard when a change affects its layout or interaction.
