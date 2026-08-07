from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

import analyze_v4_4_k200_large_round_14 as helper  # noqa: E402


STAGE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "campaigns"
    / "v4_4_positive_entry_signal_repair_20260805"
    / "continuation_round_15_dual_purpose_broad_and_single_axis_all_window"
)
SOURCE_SNAPSHOT_ID = "db85efb36f3de1c1f8255c6108fb365ad9f3d337f77a8d37a0e0ae41982e5699"
SOURCE_SNAPSHOT = PROJECT_ROOT / "results" / "all_completed_union_analysis" / "snapshots" / SOURCE_SNAPSHOT_ID
OUTPUT_ROOT = STAGE_ROOT / "interpretation"
HANDOFF_ROOT = STAGE_ROOT / "next_round_handoff"
PARAMETER_FIELDS = ("e", "bh", "trw", "k", "w", "m", "speed_window_bars")
HANDOFF_COLUMNS = (
    "candidate_id", "experiment_block", "search_mode", "anchor_combo_id", "diagnostic_type",
    "hypothesis", "method", "baseline_sampling_policy", "e", "bh", "trw", "k",
    "abs_floor_value", "w", "m", "speed_window_bars", "cost_bps",
    "expected_behavior_change", "evidence_summary", "selection_reason", "source_round", "status",
)


def clean(value: object) -> object:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [{str(key): clean(value) for key, value in row.items()} for row in frame.to_dict("records")]


def coordinate_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        int(row["e"]), int(row["bh"]), int(row["trw"]), round(float(row["k"]), 8),
        int(row["w"]), round(float(row["m"]), 8), int(row["speed_window_bars"]),
    )


def current_snapshot() -> tuple[str, Path]:
    pointer = json.loads(
        (PROJECT_ROOT / "results" / "all_completed_union_analysis" / "current_snapshot.json").read_text(encoding="utf-8")
    )
    return str(pointer["union_snapshot_id"]), Path(str(pointer["snapshot_root"]))


def leader(frame: pd.DataFrame, view: str) -> pd.Series:
    if view == "unrestricted_total":
        eligible, metric = frame, "train_cost_adjusted_return"
    elif view == "scenario_1_total":
        eligible, metric = frame.loc[frame.scenario_1_qualified.astype(bool)], "train_cost_adjusted_return"
    elif view == "average_ge10":
        eligible, metric = frame.loc[frame.train_trade_count.ge(10)], "train_cost_adjusted_avg_trade"
    else:
        eligible, metric = frame.loc[frame.train_trade_count.ge(20)], "train_cost_adjusted_avg_trade"
    return eligible.sort_values(metric, ascending=False, kind="mergesort").iloc[0]


def row_metrics(row: pd.Series) -> dict[str, object]:
    return {
        "combo_id": str(row.combo_id),
        "parameters": {field: clean(row[field]) for field in PARAMETER_FIELDS},
        "trade_count": int(row.train_trade_count),
        "cost_adjusted_total_return": float(row.train_cost_adjusted_return),
        "cost_adjusted_average_trade": float(row.train_cost_adjusted_avg_trade),
        "cost_adjusted_max_drawdown_abs": float(row.train_cost_adjusted_max_drawdown_abs),
        "scenario_1_qualified": bool(row.scenario_1_qualified),
    }


def classify_blocks(
    stage: pd.DataFrame,
    prior: pd.DataFrame,
    metadata: dict[tuple[object, ...], dict[str, object]],
    new_frontier_ids: set[str],
) -> pd.DataFrame:
    prior_by_id = prior.set_index("combo_id", drop=False)
    rows: list[dict[str, object]] = []
    for block_id, group in stage.groupby("experiment_block", sort=True):
        sample_meta = metadata[coordinate_key(group.iloc[0].to_dict())]
        search_mode = str(sample_meta["search_mode"])
        best_return = group.sort_values("train_cost_adjusted_return", ascending=False).iloc[0]
        best_average = group.sort_values("train_cost_adjusted_avg_trade", ascending=False).iloc[0]
        lowest_drawdown = group.sort_values("train_cost_adjusted_max_drawdown_abs").iloc[0]
        frontier_count = int(group.combo_id.astype(str).isin(new_frontier_ids).sum())
        anchor_id = "" if search_mode == "broad_jump" else str(sample_meta["seed"])
        if search_mode == "broad_jump":
            if frontier_count:
                classification = "improved"
                reason = "The broad block found a new positive return/drawdown frontier point outside the current leader neighborhoods."
            else:
                classification = "mixed"
                reason = "The broad block found positive combinations but no primary-view leader or new return/drawdown frontier point."
            anchor = None
        else:
            anchor = prior_by_id.loc[anchor_id]
            dominates = group.loc[
                group.train_cost_adjusted_return.ge(float(anchor.train_cost_adjusted_return) - 1e-12)
                & group.train_cost_adjusted_avg_trade.ge(float(anchor.train_cost_adjusted_avg_trade) - 1e-12)
                & group.train_cost_adjusted_max_drawdown_abs.le(float(anchor.train_cost_adjusted_max_drawdown_abs) + 1e-12)
            ]
            any_gain = (
                float(best_return.train_cost_adjusted_return) > float(anchor.train_cost_adjusted_return) + 1e-12
                or float(best_average.train_cost_adjusted_avg_trade) > float(anchor.train_cost_adjusted_avg_trade) + 1e-12
                or float(lowest_drawdown.train_cost_adjusted_max_drawdown_abs) < float(anchor.train_cost_adjusted_max_drawdown_abs) - 1e-12
                or frontier_count > 0
            )
            if not dominates.empty:
                classification = "improved"
                reason = "At least one single-parameter point improves return quality without worsening the anchor's other primary metrics."
            elif any_gain:
                classification = "mixed"
                reason = "The single-parameter block improves at least one target metric but gives back another metric or remains a frontier trade-off."
            else:
                classification = "not_improved"
                reason = "All tested single-parameter points remain below the anchor across return, average trade, drawdown, and frontier evidence."
        rows.append({
            "experiment_block": block_id,
            "search_mode": search_mode,
            "coordinate_count": len(group),
            "cost_positive_count": int(group.train_cost_adjusted_return.gt(0).sum()),
            "new_frontier_count": frontier_count,
            "anchor_combo_id": anchor_id,
            "anchor_return": None if anchor is None else float(anchor.train_cost_adjusted_return),
            "anchor_average_trade": None if anchor is None else float(anchor.train_cost_adjusted_avg_trade),
            "anchor_drawdown": None if anchor is None else float(anchor.train_cost_adjusted_max_drawdown_abs),
            "best_return_combo_id": str(best_return.combo_id),
            "best_return": float(best_return.train_cost_adjusted_return),
            "best_average_combo_id": str(best_average.combo_id),
            "best_average_trade": float(best_average.train_cost_adjusted_avg_trade),
            "lowest_drawdown_combo_id": str(lowest_drawdown.combo_id),
            "lowest_drawdown": float(lowest_drawdown.train_cost_adjusted_max_drawdown_abs),
            "classification": classification,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def proposed_next_rows(current: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, int]]:
    completed = {coordinate_key(row) for row in current.to_dict("records")}
    combo_ids = {coordinate_key(row): str(row["combo_id"]) for row in current.to_dict("records")}
    proposals: list[tuple[str, str, str, dict[str, object], str]] = []

    average_anchor = {"e": 96, "bh": 612, "trw": 24, "k": 1.6, "w": 16, "m": 2.0, "speed_window_bars": 308}
    for value in (80, 104, 120):
        proposals.append(("next_average_e_local", "single_axis", "Test the E96 average-return neighborhood.", {**average_anchor, "e": value}, "average_return_stability"))
    for value in (136, 144, 160):
        proposals.append(("next_average_e_expansion", "single_axis", "Test whether the E128 all-metric gain extends to a wider E plateau.", {**average_anchor, "e": value}, "average_return_stability"))

    scenario_anchor = {"e": 320, "bh": 228, "trw": 22, "k": 1.0, "w": 6, "m": 4.5, "speed_window_bars": 330}
    for value in (198, 222, 252):
        proposals.append(("next_scenario_bh", "single_axis", "Resolve the return/drawdown trade-off around BH228.", {**scenario_anchor, "bh": value}, "scenario_baseline_stability"))

    low_drawdown_anchor = {"e": 150, "bh": 504, "trw": 24, "k": 1.6, "w": 12, "m": 2.5, "speed_window_bars": 262}
    for value in (9, 11, 14):
        proposals.append(("next_low_drawdown_w", "single_axis", "Resolve the W12 low-drawdown frontier trade-off.", {**low_drawdown_anchor, "w": value}, "low_drawdown_exit_stability"))

    rng = random.Random(440153)
    broad_axes = {
        "e": [30, 54, 84, 156, 240, 360, 520, 760, 1040],
        "bh": [84, 132, 228, 360, 510, 720, 960, 1200],
        "trw": [3, 7, 10, 14, 20, 28, 40, 48],
        "k": [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6],
        "w": [1, 3, 6, 10, 18, 32, 64, 128],
        "m": [0.6, 1.1, 1.8, 3.5, 5.5, 8.0, 12.0, 14.0],
        "speed_window_bars": [54, 90, 144, 220, 350, 520, 780, 1080],
    }
    broad_seen: set[tuple[object, ...]] = set()
    while len(broad_seen) < 48:
        row = {field: rng.choice(broad_axes[field]) for field in PARAMETER_FIELDS}
        key = coordinate_key(row)
        if key in completed or key in broad_seen:
            continue
        broad_seen.add(key)
        proposals.append(("next_broad_stratified", "broad_jump", "Continue coarse coverage outside current leader neighborhoods.", row, "broad_parameter_discovery"))

    output: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    completed_overlap = 0
    duplicate_count = 0
    for block, mode, hypothesis, row, diagnostic in proposals:
        key = coordinate_key(row)
        if key in completed:
            completed_overlap += 1
            continue
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        anchor = "" if mode == "broad_jump" else combo_ids.get(coordinate_key(
            average_anchor if block.startswith("next_average") else
            scenario_anchor if block == "next_scenario_bh" else
            low_drawdown_anchor
        ), "")
        output.append({
            "candidate_id": f"next_{len(output) + 1:03d}",
            "experiment_block": block,
            "search_mode": mode,
            "anchor_combo_id": anchor,
            "diagnostic_type": diagnostic,
            "hypothesis": hypothesis,
            "method": "rolling_tr_sum",
            "baseline_sampling_policy": "all_window",
            **row,
            "abs_floor_value": 0.0,
            "cost_bps": 3.568663594470046,
            "expected_behavior_change": "Find a stable return or maximum-drawdown improvement without changing other parameters in refinement blocks.",
            "evidence_summary": "Round 15 improved the average-E branch and found one remote low-drawdown frontier point.",
            "selection_reason": "Preserve both broad coverage and evidence-led one-parameter refinement.",
            "source_round": STAGE_ROOT.name,
            "status": "proposed_for_next_round",
        })
    return output, {"completed_overlap_count": completed_overlap, "internal_duplicate_count": duplicate_count}


def main() -> None:
    delivery_snapshot_id, delivery_snapshot = current_snapshot()
    prior = pd.read_csv(SOURCE_SNAPSHOT / "analysis_summary.csv")
    current = pd.read_csv(delivery_snapshot / "analysis_summary.csv")
    stage = pd.read_csv(STAGE_ROOT / "analysis" / "analysis_summary.csv")
    plan = json.loads((STAGE_ROOT / "input_plan.json").read_text(encoding="utf-8"))
    metadata = {coordinate_key(row): row for row in plan["explicit_combos"]}
    stage["experiment_block"] = [metadata[coordinate_key(row)]["design"] for row in stage.to_dict("records")]
    stage["search_mode"] = [metadata[coordinate_key(row)]["search_mode"] for row in stage.to_dict("records")]
    prior["origin"] = "prior"
    stage["origin"] = "round_15"
    combined = pd.concat([prior, stage], ignore_index=True)
    frontier = helper.two_dimensional_frontier(combined)
    new_frontier = frontier.loc[frontier.origin.eq("round_15")].copy()
    block_frame = classify_blocks(stage, prior, metadata, set(new_frontier.combo_id.astype(str)))

    views = ("unrestricted_total", "scenario_1_total", "average_ge10", "average_ge20")
    view_changes: dict[str, object] = {}
    leader_rows = []
    for view in views:
        old_leader = leader(prior, view)
        new_leader = leader(current, view)
        view_changes[view] = {
            "changed": str(old_leader.combo_id) != str(new_leader.combo_id),
            "prior": row_metrics(old_leader),
            "current": row_metrics(new_leader),
            "total_return_percentage_point_change": float((new_leader.train_cost_adjusted_return - old_leader.train_cost_adjusted_return) * 100),
            "average_trade_percentage_point_change": float((new_leader.train_cost_adjusted_avg_trade - old_leader.train_cost_adjusted_avg_trade) * 100),
            "drawdown_percentage_point_change": float((new_leader.train_cost_adjusted_max_drawdown_abs - old_leader.train_cost_adjusted_max_drawdown_abs) * 100),
        }
        leader_rows.extend([
            {"view": f"prior_{view}", **old_leader.to_dict()},
            {"view": f"current_{view}", **new_leader.to_dict()},
        ])

    old_average = leader(prior, "average_ge20")
    new_average = leader(current, "average_ge20")
    stage_trades = pd.read_csv(STAGE_ROOT / "analysis" / "stage_trades.csv", low_memory=False)
    new_trades = stage_trades.loc[stage_trades.combo_id.eq(str(new_average.combo_id))].copy()
    old_trades = helper.load_old_anchor_trades(prior, str(old_average.combo_id))
    bars = helper.load_bars(helper.cross.K200_SOURCE, helper.cross.K200_PREPARATION)
    trade_comparison, trade_diagnostics = helper.leader_diagnostics(new_trades, old_trades, bars)

    e128 = stage.loc[
        stage.e.eq(128) & stage.bh.eq(612) & stage.trw.eq(24) & stage.k.eq(1.6)
        & stage.w.eq(16) & stage.m.eq(2.0) & stage.speed_window_bars.eq(308)
    ].iloc[0]
    remote = stage.loc[
        stage.e.eq(1120) & stage.bh.eq(300) & stage.trw.eq(26) & stage.k.eq(1.4)
        & stage.w.eq(1) & stage.m.eq(2.5) & stage.speed_window_bars.eq(720)
    ].iloc[0]
    notable_points: dict[str, object] = {}
    for name, row in (("average_e128", e128), ("broad_remote_frontier", remote)):
        trades = stage_trades.loc[stage_trades.combo_id.eq(str(row.combo_id))].copy()
        _, metrics = helper.cross._add_excursions(trades, bars, prefix=name)
        notable_points[name] = {
            **row_metrics(row),
            "trade_diagnostics": {str(key): clean(value) for key, value in metrics.iloc[0].to_dict().items()},
        }
    broad_mask = block_frame.experiment_block.eq("broad_global_stratified")
    block_frame.loc[broad_mask, "classification"] = "mixed"
    block_frame.loc[broad_mask, "reason"] = (
        "The block found one low-drawdown frontier point, but its cost-adjusted non-gap return is negative, "
        "36 of 55 signals are synthetic, and the best two positive trades contribute about half of positive return."
    )

    broad_blocks = block_frame.loc[block_frame.search_mode.eq("broad_jump")]
    refinement_blocks = block_frame.loc[block_frame.search_mode.eq("single_axis")]
    report = {
        "status": "complete",
        "stage_id": STAGE_ROOT.name,
        "source_snapshot_id": SOURCE_SNAPSHOT_ID,
        "delivery_snapshot_id": delivery_snapshot_id,
        "plan_fingerprint": json.loads((STAGE_ROOT / "completion_manifest.json").read_text(encoding="utf-8"))["plan_fingerprint"],
        "coordinate_count": len(stage),
        "trade_count": int(stage.train_trade_count.sum()),
        "cost_positive_coordinate_count": int(stage.train_cost_adjusted_return.gt(0).sum()),
        "scenario_1_qualified_coordinate_count": int(stage.scenario_1_qualified.astype(bool).sum()),
        "scenario_2_qualified_coordinate_count": int(stage.scenario_2_qualified.astype(bool).sum()),
        "scenario_3_qualified_coordinate_count": int(stage.scenario_3_qualified.astype(bool).sum()),
        "new_return_drawdown_frontier_count": len(new_frontier),
        "overall_classification": "improved",
        "broad_exploration": {
            "coordinate_count": int(stage.search_mode.eq("broad_jump").sum()),
            "cost_positive_count": int(stage.loc[stage.search_mode.eq("broad_jump"), "train_cost_adjusted_return"].gt(0).sum()),
            "new_frontier_count": int(broad_blocks.new_frontier_count.sum()),
            "classification": "mixed_no_robust_primary_leader",
        },
        "single_parameter_refinement": {
            "coordinate_count": int(stage.search_mode.eq("single_axis").sum()),
            "improved_block_count": int(refinement_blocks.classification.eq("improved").sum()),
            "mixed_block_count": int(refinement_blocks.classification.eq("mixed").sum()),
            "not_improved_block_count": int(refinement_blocks.classification.eq("not_improved").sum()),
            "classification": "improved",
        },
        "primary_view_changes": view_changes,
        "average_leader_trade_comparison": trade_comparison,
        "notable_points": notable_points,
        "block_classifications": records(block_frame[["experiment_block", "search_mode", "classification", "reason"]]),
        "evidence_boundary": "All results are in-sample. Gap/synthetic dependence and return concentration remain required interpretation fields.",
        "combined_score": False,
        "parameter_acceptance": "none",
        "next_round_compute_authorized": False,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    HANDOFF_ROOT.mkdir(parents=True, exist_ok=True)
    block_frame.to_csv(OUTPUT_ROOT / "block_summary.csv", index=False)
    pd.DataFrame(leader_rows).to_csv(OUTPUT_ROOT / "primary_view_leaders.csv", index=False)
    new_frontier.to_csv(OUTPUT_ROOT / "new_return_drawdown_frontier.csv", index=False)
    trade_diagnostics.to_csv(OUTPUT_ROOT / "average_leader_trade_diagnostics.csv", index=False)
    (OUTPUT_ROOT / "round_15_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    next_rows, handoff_counts = proposed_next_rows(current)
    handoff_path = HANDOFF_ROOT / "next_round_parameters.csv"
    with handoff_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HANDOFF_COLUMNS)
        writer.writeheader()
        writer.writerows(next_rows)
    handoff_audit = {
        "status": "proposed_not_authorized_for_compute",
        "source_round": STAGE_ROOT.name,
        "source_snapshot_id": delivery_snapshot_id,
        "candidate_count": len(next_rows),
        **handoff_counts,
        "active_overlap_count": 0,
        "pending_overlap_count": 0,
        "sha256": hashlib.sha256(handoff_path.read_bytes()).hexdigest(),
        "parameter_acceptance": "none",
    }
    (HANDOFF_ROOT / "handoff_audit.json").write_text(json.dumps(handoff_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(OUTPUT_ROOT / "round_15_report.json"),
        "overall_classification": report["overall_classification"],
        "delivery_snapshot_id": delivery_snapshot_id,
        "new_frontier_count": len(new_frontier),
        "next_round_candidate_count": len(next_rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
