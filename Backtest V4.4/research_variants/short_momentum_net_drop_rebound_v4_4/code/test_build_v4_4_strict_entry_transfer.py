from __future__ import annotations

import pandas as pd

from build_v4_4_strict_entry_transfer import (
    STRICT_BATCH_LABEL,
    _strictness_diagnostics,
    _write_aggregate_candidate_freeze,
    pareto_mask,
)


def test_pareto_mask_keeps_tradeoff_points_without_combined_score() -> None:
    frame = pd.DataFrame(
        [
            {"return": 0.30, "threshold": 9.0, "trades": 60},
            {"return": 0.28, "threshold": 8.0, "trades": 70},
            {"return": 0.25, "threshold": 10.0, "trades": 50},
            {"return": 0.20, "threshold": 7.0, "trades": 80},
        ]
    )
    mask = pareto_mask(
        frame,
        maximize=("return", "threshold"),
        minimize=("trades",),
    )
    assert mask.tolist() == [True, False, True, False]


def test_aggregate_freeze_writes_matching_json_and_csv(tmp_path) -> None:
    candidates = pd.DataFrame(
        [
            {"candidate_order": 1, "combo_id": "combo-a"},
            {"candidate_order": 2, "combo_id": "combo-b"},
        ]
    )
    payload = {
        "schema_version": 1,
        "candidate_count": 2,
        "candidates": candidates.to_dict(orient="records"),
    }

    _write_aggregate_candidate_freeze(tmp_path, candidates, payload)

    written = pd.read_csv(tmp_path / "frozen_candidates.csv")
    assert written.combo_id.tolist() == ["combo-a", "combo-b"]
    assert (tmp_path / "frozen_candidates.json").is_file()


def test_strictness_diagnostics_reports_quartiles_and_monotonic_checks() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "transfer_batch": STRICT_BATCH_LABEL,
                "source_entry_threshold_median": float(index + 1),
                "source_trade_count": 100 - index,
                "target_trade_count": 80 - index,
                "target_cost_total_return": 0.10 + index / 100.0,
                "target_cost_median_trade": 0.001 + index / 10_000.0,
                "target_cost_max_drawdown_abs": 0.20 - index / 100.0,
                "cross_instrument_pareto": index in {6, 7},
            }
        )
    result = _strictness_diagnostics(pd.DataFrame(rows))
    assert result["strict_candidate_count"] == 8
    assert len(result["threshold_quartiles"]) == 4
    assert result["all_four_improve_continuously"] is True
    assert result["cross_instrument_pareto_count_strict_batch"] == 2


def test_strictness_diagnostics_does_not_claim_continuous_improvement() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "transfer_batch": STRICT_BATCH_LABEL,
                "source_entry_threshold_median": float(index + 1),
                "source_trade_count": 100 - index,
                "target_trade_count": 80 - index,
                "target_cost_total_return": 0.20 - index / 100.0,
                "target_cost_median_trade": 0.002 - index / 10_000.0,
                "target_cost_max_drawdown_abs": 0.10 + index / 100.0,
                "cross_instrument_pareto": False,
            }
        )
    result = _strictness_diagnostics(pd.DataFrame(rows))
    assert result["all_four_improve_continuously"] is False
    assert result["continuous_improvement_checks"][
        "target_trade_count_nonincreasing"
    ] is True
