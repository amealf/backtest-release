# Experiment Progress

## Purpose

Track the closed max-completed-W campaign by experimental round without duplicating raw results. Each row records why the round ran, what it covered, the observed primary-view outcome, and the next decision.

## Closed campaign

| Round | Why | Run | Observed | Next | Authority |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish implementation/smoke evidence under the max-W identity | 12 coordinates, 1 batch, 1,196 trades | One Scenario-1-qualified row; smoke leaders only | Broad deterministic exploration | Completion `3a3428b5…`; corrected delivery `d0f88bc6…` |
| 2 | Cover broad E/BH/TRW/K/W/M jumps | 504 coordinates, 42 batches, 73,306 trades | Primary leaders improved across all four independent views | Bounded interaction refinement | Completion `863eb926…`; delivery `af17ff90…` |
| 3 | Test interactions around independently recorded Round-2 leaders | 72 coordinates, 6 batches, 37,618 trades | Leaders reached 18.286139% Scenario-1 total, 115.821579% unrestricted total, and 0.578419% average | Initially terminal; later user-authorized Round 4 | Completion `d42d806e…`; delivery `7b8f1450…` |
| 4 | Resolve inspected E/W/K, E/W/M, and M boundaries | 72 coordinates, 6 batches, 73,373 trades | Scenario-1 total reached 37.082743%; unrestricted total reached 340.137206%; both average branches stopped | Exact two-block terminal Round 5 | Completion `24532112…`; delivery `0d5d3187…` |
| 5 | Test only the two Round-4 branches that passed | 24 coordinates, 2 batches, 46,732 trades | Scenario-1 retained Round-4 leader; unrestricted total reached 515.916078%; both average views retained Round-3 leader | Terminal close; no Round 6; no acceptance | Completion `5804d894…`; delivery `6d3cc38c…` |

The cumulative close is five stages, 684 coordinates, and 232,225 trades at snapshot `4bed7f828a7068ee0ee70001a245133c481e785358cfb3f2b043d08713ca7259`.

## Interpretation boundary

- Four independent primary views remain separate; no combined score exists.
- Gap-excluded return is display-only and cannot filter, rank, continue, stop, accept, or reject.
- Final leaders are dependent in-sample candidates only. No holdout, new-date, cross-instrument, fee, nonzero-slippage, or production-acceptance evidence exists.
- Round 5 is terminal. No Round 6 or parameter acceptance is authorized.

## Maintenance

Append only when an authorized experimental round closes or the user changes the experimental evidence boundary. Keep task state in `CURRENT_TASKS.en.md` and durable research rules in `..\05_domains\research\RESEARCH_CONSTRAINTS.en.md`.
