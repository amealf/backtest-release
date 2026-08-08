# Project Management Entry

## Purpose

This folder is the shared working memory for <strong>Backtest V4.41 Research</strong>. V4.41 is a minor release on the V4.4 strategy and ranking lineage. It records current intent, constraints, valid files, version state, active work, decisions, history, future directions, references, and archive boundaries. It does not duplicate large project files or raw source material.

The folder was initialized for a <strong>existing project</strong> on 2026-08-01.

## Current orientation

- <strong>Durable goal:</strong> Repair execution realism and keep V4.4 reproducible from repository-local runtime inputs.
- <strong>Optional domain modules:</strong> data, research
- <strong>Operational language:</strong> English.
- <strong>Review mirror:</strong> Chinese.
- <strong>Unknown facts:</strong> remain unconfirmed until supported by the user or project evidence.

The management dashboard is an offline document reader. Its use of HTML does not imply that this project contains or needs a frontend.

## Evidence priority

Use the highest applicable source:

1. Current user instruction and accepted decisions.
2. Current real project artifacts and behavior, including configuration, tests, or commands only when they exist.
3. Current English project-management documents.
4. Chinese review mirrors.
5. Historical, proposed, future, and archived material.

When sources disagree, record the discrepancy. Do not silently choose the more convenient source.

## Conversation intake and ongoing maintenance

Before initialization or project-changing work, review the conversation and delegated context available in the current task. Extract confirmed goals, constraints, decisions, tasks, current state, domains, corrections, and unknowns. Do not claim access to conversation that is not available.

For a new project, use those confirmed facts to populate the neutral management scaffold. Build a project implementation scaffold only when the user requests it and the available conversation establishes enough project shape. For an existing project, reconcile the intake with real project evidence and keep every user-owned file protected.

During later work, an accepted durable change in the conversation triggers the matching document update before the task ends. Explanation-only questions, hypotheticals, and unaccepted ideas do not change files. Store conclusions and useful rationale rather than raw transcripts, secrets, or private personal data.

When a request names an existing tool, template, or behavior, that named item defines the implementation boundary. Inspect it before work and obtain user confirmation before substituting a newly authored implementation or adding behavior outside the stated request.

Before changing any file, behavior, output, workflow, field, or artifact beyond the user's explicit request or accepted proposal, stop and obtain approval. Technical convenience, provenance concerns, adjacent cleanup, and agent interpretation do not broaden authorization.

## Role-neutral execution governance

- The project accepts any executable instrument profile and any exact evaluation interval. Completed results use date-based evaluation packages; experiment roles live in plans and records rather than directory names.
- `results\evaluation_comparison` reads package-owned browser summaries and joins exact `combo_id` values. Existing stable main and per-trade pages remain compatibility contracts during framework changes.
- Intermediate exploration rounds close immutable raw evidence and compact summaries only. They do not generate cumulative or per-trade HTML by default.
- Parameter exploration follows a repeating AI-led cycle: multi-round leap search, finite one-parameter grids around promising nonadjacent anchors, then renewed leap search. A round may belong to one phase; reports preserve the phase and keep exploration and refinement evidence separate across the cycle.
- One-axis grids have no fixed point-count cap. Freeze finite bounds, steps, anchor, and expected coordinates before compute; interpret the complete closed pattern before continuing, refining, widening, stopping, or changing direction.
- User observation and correction occur between cycles or at a requested checkpoint. Autonomous rounds while the user is unavailable remain inside the authorized time, instrument, method, data, and resource boundary.
- The shared cumulative main and per-trade entries are published once after the exploration series ends. An intermediate publication requires an explicit user request. Final publication includes every compatible completed round and uses one cumulative writer.

## Validation-tier routing

- <strong>Complete regression:</strong> version upgrades and changes to engine behavior, entry/exit or fills, return/cost calculations, data preparation, schemas, execution contracts, or result semantics.
- <strong>Focused interaction:</strong> frontend filters, sorting, selectors, buttons, navigation, and state synchronization. Check the affected interaction and its closest dependent state.
- <strong>Visual-only:</strong> copy, color, spacing, typography, alignment, responsive layout, and visibility. Regenerate the affected HTML, perform a simple functional check, and retain desktop/mobile screenshots; do not run the complete suite unless behavior or data also changes.
- If scope crosses tiers or semantic impact is uncertain, use the higher tier and record the selected tier in the handoff.

## Default reading order

Read before project changes:

1. `..\AGENTS.md`
2. `00_START_HERE.en.md`
3. `01_goal\PROJECT_GOAL.en.md`
4. `01_goal\PROJECT_CONSTRAINTS.en.md`
5. `02_current_state\SOURCE_OF_TRUTH.en.md`
6. `02_current_state\CURRENT_VERSION.en.md`
7. `03_active_work\CURRENT_TASKS.en.md`
8. `03_active_work\WORK_PROGRESS.en.md`
9. `03_active_work\EXPERIMENT_PROGRESS.en.md` when the task runs, interprets, or closes an experiment.
10. `03_active_work\BACKTEST_MANAGEMENT.en.md` before every backtest run.
11. `05_domains\research\PARAMETER_EXPLORATION_GUIDE.en.md` for every parameter-exploration design, run, interpretation, or next-round CSV handoff.
12. `00_core\STRATEGY_CONTRACT_V4_4.en.md`, `10_instruments\INSTRUMENT_PROFILE_CONTRACT.en.md`, and `20_campaigns\CAMPAIGN_WORKFLOW.en.md` for cross-instrument transfer or a fresh instrument search.

Load other documents only when the task requires them.

## Task selection matrix

| Task | Additional documents |
| --- | --- |
| Explain or answer | Only the files needed for the answer; do not mutate project state unless requested. |
| Change scope or plan | Goal, constraints, current tasks, and work progress. |
| Change a deliverable or current behavior | Goal, constraints, source of truth, current version, current tasks, work progress, and relevant decisions. |
| Repair or materially change behavior | Current version, work progress, change reasons, and relevant decisions. |
| Change code structure, execution flow, components, dependencies, data flow, or interface boundaries | Source of truth, current version, code architecture, current tasks, work progress, and relevant decisions. |
| Change a confirmed domain | The matching active module listed below, plus current state, work progress, and decisions. |
| Design, run, interpret, or hand off parameter exploration | Parameter Exploration Guide, research constraints, experiment progress, current constraints, current tasks, and the current source identity. |
| Run any backtest | Backtest Management, current source identity, instrument profile, campaign workflow, and the documents required by the run's mode. |
| Transfer to another instrument or start a fresh instrument search | Strategy Contract, Instrument Profile Contract, Campaign Workflow, Parameter Exploration Guide, data requirements, and current source identity. |
| Propose a direction | Current goal, constraints, tasks, work progress, decisions, and next directions. |
| Reuse a reference | Reference index and its source file. |
| Inspect, reuse, or restore historical work | Project history, archive guide, and the archived item's note. |

### Active domain reading rules

- Read `project_management\05_domains\data\DATA_REQUIREMENTS.en.md` before changing schemas, migrations, pipelines, lineage, or data contracts.
- Read `project_management\05_domains\research\RESEARCH_CONSTRAINTS.en.md` before changing methods, running experiments, or interpreting results.
- Read and follow `project_management\05_domains\research\PARAMETER_EXPLORATION_GUIDE.en.md` for every parameter-exploration task.

## Document maintenance triggers

| Document | Update trigger |
| --- | --- |
| `PROJECT_GOAL` | The user changes the durable objective or accepted success criteria. |
| `PROJECT_CONSTRAINTS` | A durable cross-project constraint is accepted, changed, or retired. |
| `SOURCE_OF_TRUTH` | An authoritative path, entry point, branch, configuration, or artifact changes. |
| `CURRENT_VERSION` | A deliverable, current behavior, project structure, interface, dependency, or operating procedure changes. |
| `CODE_ARCHITECTURE` | A code entry point, component responsibility, dependency or call relationship, data flow, generated artifact, or interface boundary changes. Mark it not applicable when the project has no code. |
| `CURRENT_TASKS` | A task is declared, completed, blocked, resumed, or changes scope. |
| `WORK_PROGRESS` | Work completes or changes materially enough to affect handoff, review, or the current understanding of the project. |
| `EXPERIMENT_PROGRESS` | An authorized experimental round closes or the experimental evidence boundary changes. |
| `BACKTEST_MANAGEMENT` | Before every backtest starts; keep the row even when the run stops early. |
| `PARAMETER_EXPLORATION_GUIDE` | The accepted parameter-diagnosis, improvement-judgment, experiment-block, or next-round CSV handoff rule changes. |
| `STRATEGY_CONTRACT_V4_4` | Accepted instrument-neutral entry, exit, timing, or state-machine semantics change. |
| `INSTRUMENT_PROFILE_CONTRACT` | Data, cost, gap, low-activity, scenario, or ranking-lineage interfaces change. |
| `CAMPAIGN_WORKFLOW` | Exact transfer, target-local refinement, same-lineage continuation search, fresh search, cross-instrument ranking, or prompt-handoff rules change. |
| `CHANGE_REASONS` | A material behavior change needs a durable reason, prior behavior, updated behavior, evidence impact, and validity boundary. |
| `DECISIONS` | A durable choice is accepted, retired, or reversed. |
| `PROJECT_HISTORY` | An event changes how past or current work should be understood. |
| `NEXT_DIRECTIONS` | A testable research question, strategy hypothesis, or validation path appears, changes, becomes active, or is retired. Execution, delivery, and resource gates belong in constraints, tasks, or operating rules rather than future research directions. |
| `08_references\README` | A retained source or its provenance, relevance, or review status changes. |
| `99_archive\README` | Material becomes inactive, archive policy changes, or an item is restored. |
| Active domain document | A durable rule in that confirmed domain changes. |
| Root agent files and this entry | The reading order or maintenance protocol changes. |

Update the English source and Chinese mirror together. Regenerate `index.html` with:

```powershell
node "project_management\tools\build_dashboard.mjs"
```

## History and archive boundary

The Dashboard separates current guidance and work from project history and inactive material. Do not load history or archive documents during normal work unless the task needs past evidence, reuse, or restoration.

When a major delivery version or working method changes and prior detailed progress would mislead current work, move the superseded body into a dated folder under `99_archive`, keep a concise restoration pointer in `WORK_PROGRESS`, and preserve the archived material.

## Initialization inventory

The initializer detected these pre-existing files, limited to a small orientation sample:

- `.gitignore`
- `.python-version`
- `package-lock.json`
- `package.json`
- `PRODUCT.md`
- `README.md`
- `RELEASE.json`
- `requirements-v4_4.txt`
- `research_variants\short_momentum_net_drop_rebound_v4_4\__init__.py`
- `research_variants\short_momentum_net_drop_rebound_v4_4\README.md`
- `research_variants\short_momentum_net_drop_rebound_v4_4\SOURCE_MANIFEST.json`
- `runtime_inputs\data_preparation\baseline_filter_atoms.csv`
- `runtime_inputs\data_preparation\baseline_filter_events.json`
- `runtime_inputs\data_preparation\data_preparation_manifest.json`
- `runtime_inputs\market_data\data_preparation_audit.json`
- `runtime_inputs\market_data\k200_clean_15s_session_filled.csv`
- `runtime_inputs\provenance\V4_2_SOURCE_MANIFEST.json`
- `runtime_inputs\templates\historical_v4_main.html`
- `runtime_inputs\templates\historical_v4_trade.html`
- `runtime_inputs\templates\market-intuition-selector.html`
- `runtime_inputs\scenarios\market_catalog.json`
- `runtime_inputs\scenarios\scenario_catalog.json`
- `tools\build_v4_41_scenario_manager.py`
- `tools\apply_v4_41_scenario.py`
- `runtime_inputs\templates\plotly.min.js`
- `RUNTIME.md`

Detection is not authority. Confirm current entry points and artifacts in `02_current_state\SOURCE_OF_TRUTH.en.md`.

## Review style

- Keep documents concise and factual.
- Separate current, retired, proposed, and unknown states.
- Link to real paths instead of copying large outputs.
- Record why a fact is believed and when it stops being valid.
- Use HTML `<strong>...</strong>` for bold text. Do not author Markdown `**...**`; this avoids CommonMark delimiter ambiguity around Chinese punctuation. Inline code, fenced code, and glob paths retain literal asterisks.
