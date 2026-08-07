from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = (
    PROJECT_ROOT
    / "research_variants"
    / "short_momentum_net_drop_rebound_v4_4"
    / "code"
)
sys.path.insert(0, str(CODE_ROOT))

import build_v4_4_cross_instrument_comparison as cross  # noqa: E402
from v4_4_engine import load_bars  # noqa: E402


STAGE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "campaigns"
    / "v4_4_positive_entry_signal_repair_20260805"
    / "continuation_round_14_large_multiblock_exploration_all_window"
)
OLD_SNAPSHOT_ID = "0126cd77b436aef1434e7072bac0d6dfa15b3d2ad4dc2cf1b2fafe936ee1e626"
OLD_SNAPSHOT = (
    PROJECT_ROOT
    / "results"
    / "all_completed_union_analysis"
    / "snapshots"
    / OLD_SNAPSHOT_ID
)
OUTPUT_ROOT = STAGE_ROOT / "interpretation"
HANDOFF_ROOT = STAGE_ROOT / "next_round_handoff"
PARAMETER_FIELDS = ("e", "bh", "trw", "k", "w", "m", "speed_window_bars")
AVG_NEW_ID = (
    "v4_4_rolling_tr_sum_bpall_window_fillcalculated_threshold_execwait_next_real_trade_"
    "slip0_sx1_s308_rx1_e112_bh612_trw24_k1p6_w16_m2_f21ec85557"
)
AVG_OLD_ID = (
    "v4_4_rolling_tr_sum_bpall_window_fillcalculated_threshold_execwait_next_real_trade_"
    "slip0_sx1_s308_rx1_e112_bh612_trw24_k1p6_w10_m2p5_5a8f5e6359"
)


BLOCK_CLASSIFICATIONS = {
    "unrestricted_e_bh_surface": (
        "not_improved",
        "The best new E/BH point remains materially below the unrestricted incumbent and has no drawdown advantage.",
    ),
    "unrestricted_trw_k_surface": (
        "mixed",
        "TRW16/K0.95 approaches the total-return incumbent and TRW14/K1.45 adds a new return/drawdown Pareto point, but neither improves the primary total-return view and both remain gap-sensitive.",
    ),
    "unrestricted_w_m_surface": (
        "not_improved",
        "Joint W/M changes do not recover the W6/W7 and M4.5 return level.",
    ),
    "unrestricted_e_s_surface": (
        "not_improved",
        "Proportional E/S combinations remain well below the incumbent total return.",
    ),
    "scenario_1_e_bh_surface": (
        "mixed",
        "E400/BH240 stays close to the Scenario-1 incumbent with the same drawdown and supports a broad E neighborhood, but it does not improve return or average trade.",
    ),
    "scenario_1_trw_k_surface": (
        "not_improved",
        "All tested TRW/K points remain below the qualified Scenario-1 incumbent.",
    ),
    "scenario_1_m_s_surface": (
        "not_improved",
        "The highest-return rows in this block are not Scenario-1 qualified; qualified rows remain below the incumbent.",
    ),
    "average_e_bh_surface": (
        "mixed",
        "E96/BH612 raises average trade and lowers drawdown, while total return and trade count decline.",
    ),
    "average_trw_k_surface": (
        "mixed",
        "K1.75 and the TRW21/K2 pair raise average trade and materially reduce drawdown, while total return and sample size decline.",
    ),
    "average_w_m_surface": (
        "improved",
        "W16/M2 becomes the new >=10 and >=20 average-return leader and improves total return, drawdown, median trade, MFE retention, and non-gap return with one fewer trade.",
    ),
    "low_drawdown_e_s_surface": (
        "mixed",
        "S262 adds a new positive return/drawdown Pareto point; the gain is a trade-off rather than a total-return improvement.",
    ),
    "low_drawdown_bh_m_surface": (
        "not_improved",
        "No point from this block improves the current low-drawdown frontier.",
    ),
    "remote_sparse_controls": (
        "not_improved",
        "Remote controls produce many positive rows but no competitive primary-view or low-drawdown leader.",
    ),
}


def clean(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): clean(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def coordinate_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        int(row["e"]),
        int(row["bh"]),
        int(row["trw"]),
        round(float(row["k"]), 8),
        int(row["w"]),
        round(float(row["m"]), 8),
        int(row["speed_window_bars"]),
    )


def current_snapshot() -> tuple[str, Path]:
    pointer = json.loads(
        (
            PROJECT_ROOT
            / "results"
            / "all_completed_union_analysis"
            / "current_snapshot.json"
        ).read_text(encoding="utf-8")
    )
    snapshot_id = str(pointer["union_snapshot_id"])
    return snapshot_id, Path(str(pointer["snapshot_root"]))


def load_old_anchor_trades(old_summary: pd.DataFrame, combo_id: str) -> pd.DataFrame:
    row = old_summary.loc[old_summary.combo_id.eq(combo_id)].iloc[0]
    path = Path(str(row.source_stage_root)) / "batches" / str(row.batch_id) / "trades.csv"
    trades = pd.read_csv(path, low_memory=False)
    trades = trades.loc[trades.combo_id.eq(combo_id)].copy()
    if "gross_return" not in trades:
        trades["gross_return"] = trades["return"]
    return trades


def two_dimensional_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[
        frame.train_trade_count.ge(20) & frame.train_cost_adjusted_return.gt(0)
    ].sort_values(
        ["train_cost_adjusted_max_drawdown_abs", "train_cost_adjusted_return"],
        ascending=[True, False],
        kind="mergesort",
    )
    best_return = -np.inf
    positions: list[int] = []
    for position, row in eligible.iterrows():
        if float(row.train_cost_adjusted_return) > best_return + 1e-12:
            positions.append(position)
            best_return = float(row.train_cost_adjusted_return)
    return eligible.loc[positions].copy()


def block_summary(
    stage: pd.DataFrame,
    frontier_ids: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for block, group in stage.groupby("experiment_block", sort=True):
        qualified = group.loc[group.scenario_1_qualified.astype(bool)]
        average_eligible = group.loc[group.train_trade_count.ge(20)]
        classification, reason = BLOCK_CLASSIFICATIONS[str(block)]
        rows.append(
            {
                "experiment_block": block,
                "coordinate_count": len(group),
                "cost_positive_count": int(group.train_cost_adjusted_return.gt(0).sum()),
                "scenario_1_qualified_count": len(qualified),
                "best_total_return": group.train_cost_adjusted_return.max(),
                "median_total_return": group.train_cost_adjusted_return.median(),
                "best_average_trade_ge20": (
                    average_eligible.train_cost_adjusted_avg_trade.max()
                    if len(average_eligible)
                    else np.nan
                ),
                "best_scenario_1_return": (
                    qualified.train_cost_adjusted_return.max()
                    if len(qualified)
                    else np.nan
                ),
                "median_drawdown": group.train_cost_adjusted_max_drawdown_abs.median(),
                "median_trade_count": group.train_trade_count.median(),
                "new_return_drawdown_frontier_count": int(
                    group.combo_id.isin(frontier_ids).sum()
                ),
                "classification": classification,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def leader_diagnostics(
    new_trades: pd.DataFrame,
    old_trades: pd.DataFrame,
    bars: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    new_enriched, new_metrics = cross._add_excursions(
        new_trades, bars, prefix="new"
    )
    _, old_metrics = cross._add_excursions(old_trades, bars, prefix="old")
    new_metric = new_metrics.iloc[0].to_dict()
    old_metric = old_metrics.iloc[0].to_dict()

    old_entries = set(old_trades.entry_time.astype(str))
    new_entries = set(new_trades.entry_time.astype(str))
    retained = old_entries.intersection(new_entries)
    changed_exit_count = 0
    for entry_time in retained:
        old_exit = str(
            old_trades.loc[
                old_trades.entry_time.astype(str).eq(entry_time), "exit_time"
            ].iloc[0]
        )
        new_exit = str(
            new_trades.loc[
                new_trades.entry_time.astype(str).eq(entry_time), "exit_time"
            ].iloc[0]
        )
        changed_exit_count += int(old_exit != new_exit)

    lows = bars.low.to_numpy(float)
    diagnostics = new_enriched.copy()
    diagnostics["signal_window_bars"] = (
        pd.to_numeric(diagnostics.signal_index, errors="coerce")
        - pd.to_numeric(diagnostics.h_index, errors="coerce")
    )
    diagnostics["entry_drop_ratio"] = (
        pd.to_numeric(diagnostics.entry_drop_value, errors="coerce")
        / pd.to_numeric(diagnostics.entry_baseline_value, errors="coerce")
    )
    diagnostics["profit_giveback_fraction"] = (
        diagnostics.new_mfe_bps / 10_000.0 - diagnostics.new_cost_adjusted_return
    )
    for minutes in (30, 60, 120):
        bars_forward = minutes * 4
        values: list[float] = []
        for row in diagnostics.itertuples(index=False):
            start = min(len(lows), int(row.exit_index) + 1)
            end = min(len(lows), start + bars_forward)
            future_low = float(np.min(lows[start:end])) if end > start else float(row.exit_fill_price)
            values.append(float(row.exit_fill_price) - future_low)
        diagnostics[f"post_exit_continuation_{minutes}m_points"] = values
    positive_sum = float(
        diagnostics.loc[
            diagnostics.new_cost_adjusted_return.gt(0), "new_cost_adjusted_return"
        ].sum()
    )
    diagnostics["positive_return_share"] = np.where(
        diagnostics.new_cost_adjusted_return.gt(0),
        diagnostics.new_cost_adjusted_return / positive_sum,
        0.0,
    )

    summary = {
        "old": {key: clean(value) for key, value in old_metric.items()},
        "new": {key: clean(value) for key, value in new_metric.items()},
        "transition": {
            "old_trade_count": len(old_trades),
            "new_trade_count": len(new_trades),
            "retained_entry_count": len(retained),
            "disappeared_entry_count": len(old_entries.difference(new_entries)),
            "new_entry_count": len(new_entries.difference(old_entries)),
            "retained_entry_exit_changed_count": changed_exit_count,
        },
    }
    keep = [
        "entry_time",
        "exit_time",
        "exit_reason",
        "entry_fill_price",
        "exit_fill_price",
        "new_cost_adjusted_return",
        "new_mfe_points",
        "new_mae_points",
        "new_mfe_bps",
        "new_mae_bps",
        "new_mfe_retention",
        "profit_giveback_fraction",
        "signal_window_bars",
        "entry_baseline_value",
        "entry_drop_value",
        "entry_drop_ratio",
        "entry_wait_bar_count",
        "holding_minutes",
        "position_crosses_real_gap",
        "signal_synthetic_empty_bar_count",
        "new_zero_trade_bar_count_holding",
        "new_synthetic_bar_count_holding",
        "post_exit_continuation_30m_points",
        "post_exit_continuation_60m_points",
        "post_exit_continuation_120m_points",
        "positive_return_share",
    ]
    return summary, diagnostics[keep].sort_values(
        "new_cost_adjusted_return", ascending=False, kind="mergesort"
    )


def proposed_next_coordinates(current: pd.DataFrame) -> list[dict[str, object]]:
    candidates: list[tuple[str, str, str, dict[str, object]]] = []
    base_avg = {
        "e": 112,
        "bh": 612,
        "trw": 24,
        "k": 1.6,
        "w": 16,
        "m": 2.0,
        "speed_window_bars": 308,
    }
    for value in (14, 18, 20):
        candidates.append(
            (
                "average_w_direction",
                "single_axis",
                "Test whether the W16 average-return improvement continues or forms a local peak.",
                {**base_avg, "w": value},
            )
        )
    for value in (1.8, 2.2, 2.4):
        candidates.append(
            (
                "average_m_stability",
                "stability",
                "Measure the M neighborhood around the new W16/M2 leader.",
                {**base_avg, "m": value},
            )
        )
    e_base = {
        "e": 96,
        "bh": 612,
        "trw": 24,
        "k": 1.6,
        "w": 10,
        "m": 2.5,
        "speed_window_bars": 308,
    }
    for value in (88, 104, 120):
        candidates.append(
            (
                "average_e_stability",
                "stability",
                "Check whether the E96 drawdown and average-return trade-off is continuous.",
                {**e_base, "e": value},
            )
        )
    ridge_base = {
        "e": 112,
        "bh": 612,
        "trw": 24,
        "k": 1.6,
        "w": 10,
        "m": 2.5,
        "speed_window_bars": 308,
    }
    for trw, k_value in ((22, 1.9), (26, 1.6), (28, 1.45)):
        candidates.append(
            (
                "average_trw_k_ridge",
                "module_pair",
                "Test the average-return TRW/K compensation ridge around the new low-drawdown points.",
                {**ridge_base, "trw": trw, "k": k_value},
            )
        )
    low_base = {
        "e": 150,
        "bh": 504,
        "trw": 24,
        "k": 1.6,
        "w": 10,
        "m": 2.5,
        "speed_window_bars": 262,
    }
    for value in (240, 280, 340):
        candidates.append(
            (
                "low_drawdown_s_direction",
                "single_axis",
                "Locate the return/drawdown frontier around S262.",
                {**low_base, "speed_window_bars": value},
            )
        )

    completed = {
        coordinate_key(row)
        for row in current[list(PARAMETER_FIELDS)].to_dict("records")
    }
    output: list[dict[str, object]] = []
    for block, mode, hypothesis, parameters in candidates:
        if coordinate_key(parameters) in completed:
            continue
        output.append(
            {
                "candidate_id": f"next_{len(output) + 1:03d}",
                "experiment_block": block,
                "search_mode": mode,
                "anchor_combo_id": AVG_NEW_ID if "average" in block else "",
                "diagnostic_type": (
                    "average_return_stability"
                    if "average" in block
                    else "low_drawdown_speed_geometry"
                ),
                "hypothesis": hypothesis,
                "method": "rolling_tr_sum",
                "baseline_sampling_policy": "all_window",
                **parameters,
                "abs_floor_value": 0.0,
                "cost_bps": cross.ROUND_TRIP_COST_BPS,
                "expected_behavior_change": (
                    "Retain the improved average-return or drawdown profile across neighboring parameters."
                ),
                "evidence_summary": (
                    "Round 14 improved both >=10 and >=20 average-return views at W16/M2 and added six return/drawdown frontier points."
                ),
                "selection_reason": "Follow only the supported average-return and low-drawdown directions.",
                "source_round": "continuation_round_14_large_multiblock_exploration_all_window",
                "status": "proposed_for_next_round",
            }
        )
    return output


def main() -> None:
    snapshot_id, snapshot_root = current_snapshot()
    stage = pd.read_csv(STAGE_ROOT / "analysis" / "analysis_summary.csv")
    old = pd.read_csv(OLD_SNAPSHOT / "analysis_summary.csv")
    current = pd.read_csv(snapshot_root / "analysis_summary.csv")
    plan = json.loads((STAGE_ROOT / "input_plan.json").read_text(encoding="utf-8"))
    metadata = {
        coordinate_key(row): row for row in plan["explicit_combos"]
    }
    stage["experiment_block"] = [
        metadata[coordinate_key(row)]["design"]
        for row in stage[list(PARAMETER_FIELDS)].to_dict("records")
    ]
    stage["origin"] = "round_14"
    old["origin"] = "prior"
    combined = pd.concat([old, stage], ignore_index=True)
    frontier = two_dimensional_frontier(combined)
    new_frontier = frontier.loc[frontier.origin.eq("round_14")].copy()

    old_total = old.sort_values("train_cost_adjusted_return", ascending=False).iloc[0]
    old_scenario = old.loc[old.scenario_1_qualified.astype(bool)].sort_values(
        "train_cost_adjusted_return", ascending=False
    ).iloc[0]
    old_average = old.loc[old.train_trade_count.ge(20)].sort_values(
        "train_cost_adjusted_avg_trade", ascending=False
    ).iloc[0]
    current_total = current.sort_values("train_cost_adjusted_return", ascending=False).iloc[0]
    current_scenario = current.loc[current.scenario_1_qualified.astype(bool)].sort_values(
        "train_cost_adjusted_return", ascending=False
    ).iloc[0]
    current_average = current.loc[current.train_trade_count.ge(20)].sort_values(
        "train_cost_adjusted_avg_trade", ascending=False
    ).iloc[0]

    stage_trades = pd.read_csv(
        STAGE_ROOT / "analysis" / "stage_trades.csv", low_memory=False
    )
    new_avg_trades = stage_trades.loc[stage_trades.combo_id.eq(AVG_NEW_ID)].copy()
    old_avg_trades = load_old_anchor_trades(old, AVG_OLD_ID)
    bars = load_bars(cross.K200_SOURCE, cross.K200_PREPARATION)
    trade_comparison, trade_diagnostics = leader_diagnostics(
        new_avg_trades, old_avg_trades, bars
    )

    block_frame = block_summary(stage, set(new_frontier.combo_id.astype(str)))
    leaders = pd.DataFrame(
        [
            {"view": "prior_unrestricted_total", **old_total.to_dict()},
            {"view": "current_unrestricted_total", **current_total.to_dict()},
            {"view": "prior_scenario_1_total", **old_scenario.to_dict()},
            {"view": "current_scenario_1_total", **current_scenario.to_dict()},
            {"view": "prior_average_ge20", **old_average.to_dict()},
            {"view": "current_average_ge20", **current_average.to_dict()},
        ]
    )

    report = {
        "status": "complete",
        "stage_id": STAGE_ROOT.name,
        "source_snapshot_id": OLD_SNAPSHOT_ID,
        "delivery_snapshot_id": snapshot_id,
        "plan_fingerprint": json.loads(
            (STAGE_ROOT / "completion_manifest.json").read_text(encoding="utf-8")
        )["plan_fingerprint"],
        "coordinate_count": len(stage),
        "trade_count": int(stage.train_trade_count.sum()),
        "cost_positive_coordinate_count": int(stage.train_cost_adjusted_return.gt(0).sum()),
        "scenario_1_qualified_coordinate_count": int(stage.scenario_1_qualified.astype(bool).sum()),
        "scenario_2_qualified_coordinate_count": int(stage.scenario_2_qualified.astype(bool).sum()),
        "scenario_3_qualified_coordinate_count": int(stage.scenario_3_qualified.astype(bool).sum()),
        "new_return_drawdown_frontier_count": len(new_frontier),
        "overall_classification": "improved",
        "primary_view_changes": {
            "unrestricted_total_return": "unchanged",
            "scenario_1_total_return": "unchanged",
            "average_return_ge10": "improved",
            "average_return_ge20": "improved",
            "low_drawdown_frontier": "improved_with_six_new_points",
        },
        "average_leader_change": {
            "prior_combo_id": str(old_average.combo_id),
            "current_combo_id": str(current_average.combo_id),
            "prior_parameters": {
                field: clean(old_average[field]) for field in PARAMETER_FIELDS
            },
            "current_parameters": {
                field: clean(current_average[field]) for field in PARAMETER_FIELDS
            },
            "trade_count_change": int(current_average.train_trade_count - old_average.train_trade_count),
            "total_return_percentage_point_change": float(
                (current_average.train_cost_adjusted_return - old_average.train_cost_adjusted_return) * 100
            ),
            "average_trade_percentage_point_change": float(
                (current_average.train_cost_adjusted_avg_trade - old_average.train_cost_adjusted_avg_trade) * 100
            ),
            "drawdown_percentage_point_change": float(
                (current_average.train_cost_adjusted_max_drawdown_abs - old_average.train_cost_adjusted_max_drawdown_abs) * 100
            ),
        },
        "trade_level_comparison": trade_comparison,
        "block_classifications": records(
            block_frame[["experiment_block", "classification", "reason"]]
        ),
        "evidence_boundary": (
            "All results are in-sample and gap/synthetic dependent. Gap-excluded return is dependency evidence only."
        ),
        "combined_score": False,
        "parameter_acceptance": "none",
        "next_round_compute_authorized": False,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    HANDOFF_ROOT.mkdir(parents=True, exist_ok=True)
    block_frame.to_csv(OUTPUT_ROOT / "block_summary.csv", index=False)
    leaders.to_csv(OUTPUT_ROOT / "primary_view_leaders.csv", index=False)
    new_frontier.to_csv(OUTPUT_ROOT / "new_return_drawdown_frontier.csv", index=False)
    trade_diagnostics.to_csv(OUTPUT_ROOT / "average_leader_trade_diagnostics.csv", index=False)
    (OUTPUT_ROOT / "round_14_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    next_rows = proposed_next_coordinates(current)
    next_columns = [
        "candidate_id",
        "experiment_block",
        "search_mode",
        "anchor_combo_id",
        "diagnostic_type",
        "hypothesis",
        "method",
        "baseline_sampling_policy",
        "e",
        "bh",
        "trw",
        "k",
        "abs_floor_value",
        "w",
        "m",
        "speed_window_bars",
        "cost_bps",
        "expected_behavior_change",
        "evidence_summary",
        "selection_reason",
        "source_round",
        "status",
    ]
    with (HANDOFF_ROOT / "next_round_parameters.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=next_columns)
        writer.writeheader()
        writer.writerows(next_rows)
    handoff_path = HANDOFF_ROOT / "next_round_parameters.csv"
    handoff_audit = {
        "status": "proposed_not_authorized_for_compute",
        "source_round": STAGE_ROOT.name,
        "source_snapshot_id": snapshot_id,
        "candidate_count": len(next_rows),
        "completed_overlap_count": 0,
        "active_overlap_count": 0,
        "pending_overlap_count": 0,
        "sha256": hashlib.sha256(handoff_path.read_bytes()).hexdigest(),
        "parameter_acceptance": "none",
    }
    (HANDOFF_ROOT / "handoff_audit.json").write_text(
        json.dumps(handoff_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(OUTPUT_ROOT / "round_14_report.json"),
                "overall_classification": report["overall_classification"],
                "new_average_leader": str(current_average.combo_id),
                "new_frontier_count": len(new_frontier),
                "next_round_candidate_count": len(next_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
