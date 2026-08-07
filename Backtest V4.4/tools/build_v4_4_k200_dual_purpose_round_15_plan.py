from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_ROOT = PROJECT_ROOT / "research_variants" / "short_momentum_net_drop_rebound_v4_4"
PLAN_ROOT = VARIANT_ROOT / "plans"
CAMPAIGN_ID = "v4_4_positive_entry_signal_repair_20260805"
STAGE_ID = "continuation_round_15_dual_purpose_broad_and_single_axis_all_window"
CAMPAIGN_ROOT = PROJECT_ROOT / "results" / "campaigns" / CAMPAIGN_ID
OUTPUT_ROOT = CAMPAIGN_ROOT / STAGE_ID
PLAN_PATH = PLAN_ROOT / "v4_4_k200_dual_purpose_20260805_round_15.json"
AUDIT_PATH = PLAN_ROOT / "v4_4_k200_dual_purpose_20260805_round_15.audit.json"
PARAMETER_FIELDS = ("e", "bh", "trw", "k", "w", "m", "speed_window_bars")


ANCHORS = {
    "unrestricted": {
        "e": 480, "bh": 171, "trw": 12, "k": 1.26,
        "w": 7, "m": 4.5, "speed_window_bars": 388,
    },
    "scenario_1": {
        "e": 320, "bh": 240, "trw": 22, "k": 1.0,
        "w": 6, "m": 4.5, "speed_window_bars": 330,
    },
    "average_return": {
        "e": 112, "bh": 612, "trw": 24, "k": 1.6,
        "w": 16, "m": 2.0, "speed_window_bars": 308,
    },
    "low_drawdown": {
        "e": 150, "bh": 504, "trw": 24, "k": 1.6,
        "w": 10, "m": 2.5, "speed_window_bars": 262,
    },
}


LOCAL_VALUES = {
    "unrestricted": {
        "e": [384, 432, 576], "bh": [137, 154, 205], "trw": [9, 10, 14],
        "k": [1.05, 1.15, 1.4], "w": [2, 4, 11], "m": [3.6, 4.0, 5.0],
        "speed_window_bars": [310, 350, 466],
    },
    "scenario_1": {
        "e": [232, 304, 448], "bh": [180, 228, 330], "trw": [15, 26, 30],
        "k": [0.85, 1.2, 1.35], "w": [4, 8, 10], "m": [3.6, 4.0, 5.0],
        "speed_window_bars": [264, 300, 420],
    },
    "average_return": {
        "e": [88, 96, 128], "bh": [504, 552, 720], "trw": [20, 22, 28],
        "k": [1.35, 1.45, 1.8], "w": [14, 18, 20], "m": [1.6, 1.8, 2.2],
        "speed_window_bars": [246, 278, 370],
    },
    "low_drawdown": {
        "e": [128, 180, 240], "bh": [400, 456, 600], "trw": [20, 22, 28],
        "k": [1.35, 1.45, 1.8], "w": [7, 8, 12], "m": [2.0, 2.2, 3.0],
        "speed_window_bars": [210, 240, 300],
    },
}


BROAD_BLOCKS = (
    {
        "block_id": "broad_global_stratified",
        "seed": 440151,
        "count": 96,
        "axes": {
            "e": [24, 48, 72, 112, 176, 256, 384, 560, 800, 1120],
            "bh": [72, 120, 192, 300, 456, 660, 900, 1200],
            "trw": [4, 6, 9, 13, 18, 26, 36, 48],
            "k": [0.55, 0.75, 0.95, 1.15, 1.4, 1.7, 2.1, 2.6],
            "w": [1, 2, 4, 7, 12, 20, 40, 80],
            "m": [0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 9.0, 14.0],
            "speed_window_bars": [48, 80, 128, 200, 320, 480, 720, 1080],
        },
        "hypothesis": "A balanced coarse sample may expose a promising region outside every current leader neighborhood.",
    },
    {
        "block_id": "broad_midpoint_stratified",
        "seed": 440152,
        "count": 96,
        "axes": {
            "e": [36, 60, 90, 132, 216, 300, 450, 680, 960],
            "bh": [90, 150, 270, 420, 570, 780, 1080],
            "trw": [5, 8, 11, 16, 22, 30, 42],
            "k": [0.65, 0.85, 1.05, 1.3, 1.55, 1.9, 2.3],
            "w": [2, 3, 5, 9, 16, 28, 56],
            "m": [0.75, 1.25, 1.75, 3.0, 5.0, 7.5, 11.0],
            "speed_window_bars": [60, 100, 160, 260, 380, 560, 840],
        },
        "hypothesis": "Coarse midpoint values omitted by earlier grids may contain a stable return or drawdown region.",
    },
)


OBJECTIVES = {
    "unrestricted": "unrestricted_total_return",
    "scenario_1": "scenario_1_total_return",
    "average_return": "average_return_ge10_ge20",
    "low_drawdown": "low_drawdown_moderate_trade_pareto",
}


def coordinate_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        int(row["e"]), int(row["bh"]), int(row["trw"]), round(float(row["k"]), 8),
        int(row["w"]), round(float(row["m"]), 8), int(row["speed_window_bars"]),
    )


def current_snapshot() -> tuple[str, Path]:
    pointer = json.loads(
        (PROJECT_ROOT / "results" / "all_completed_union_analysis" / "current_snapshot.json").read_text(encoding="utf-8")
    )
    snapshot_id = str(pointer["union_snapshot_id"])
    return snapshot_id, PROJECT_ROOT / "results" / "all_completed_union_analysis" / "snapshots" / snapshot_id


def completed_coordinates(snapshot_root: Path) -> tuple[set[tuple[object, ...]], dict[tuple[object, ...], str]]:
    keys: set[tuple[object, ...]] = set()
    combo_ids: dict[tuple[object, ...], str] = {}
    with (snapshot_root / "analysis_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = coordinate_key(row)
            keys.add(key)
            combo_ids[key] = str(row["combo_id"])
    return keys, combo_ids


def stratified_rows(block: dict[str, object]) -> list[dict[str, object]]:
    rng = random.Random(int(block["seed"]))
    count = int(block["count"])
    columns: dict[str, list[object]] = {}
    for name, values in dict(block["axes"]).items():
        column = [values[index % len(values)] for index in range(count)]
        rng.shuffle(column)
        columns[name] = column
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for index in range(count):
        row = {name: columns[name][index] for name in PARAMETER_FIELDS}
        if coordinate_key(row) not in seen:
            rows.append(row)
            seen.add(coordinate_key(row))
    while len(rows) < count:
        row = {name: rng.choice(dict(block["axes"])[name]) for name in PARAMETER_FIELDS}
        if coordinate_key(row) not in seen:
            rows.append(row)
            seen.add(coordinate_key(row))
    return rows


def local_rows() -> list[tuple[dict[str, object], dict[str, object]]]:
    rows: list[tuple[dict[str, object], dict[str, object]]] = []
    for anchor_name, axes in LOCAL_VALUES.items():
        for parameter, values in axes.items():
            block = {
                "block_id": f"refine_{anchor_name}_{parameter}",
                "anchor": anchor_name,
                "objective": OBJECTIVES[anchor_name],
                "search_mode": "single_axis",
                "diagnostic_type": f"{anchor_name}_{parameter}_sensitivity",
                "hypothesis": f"Changing only {parameter} may improve return or maximum drawdown around the {anchor_name} leader.",
            }
            for value in values:
                row = dict(ANCHORS[anchor_name])
                row[parameter] = value
                rows.append((block, row))
    return rows


def main() -> None:
    snapshot_id, snapshot_root = current_snapshot()
    completed, combo_ids = completed_coordinates(snapshot_root)
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)

    requested = local_rows()
    for broad in BROAD_BLOCKS:
        block = {
            "block_id": broad["block_id"],
            "anchor": "",
            "objective": "broad_multimetric_discovery",
            "search_mode": "broad_jump",
            "diagnostic_type": "remote_parameter_space_control",
            "hypothesis": broad["hypothesis"],
        }
        requested.extend((block, row) for row in stratified_rows(broad))

    retained: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    retained_by_block: dict[str, int] = {}
    internal_duplicate_count = 0
    completed_overlap_count = 0
    requested_by_block: dict[str, int] = {}
    for block, row in requested:
        block_id = str(block["block_id"])
        requested_by_block[block_id] = requested_by_block.get(block_id, 0) + 1
        key = coordinate_key(row)
        if key in seen:
            internal_duplicate_count += 1
            continue
        seen.add(key)
        if key in completed:
            completed_overlap_count += 1
            continue
        retained_by_block[block_id] = retained_by_block.get(block_id, 0) + 1
        anchor_name = str(block.get("anchor", ""))
        seed = combo_ids[coordinate_key(ANCHORS[anchor_name])] if anchor_name else "broad_stratified"
        retained.append({
            "candidate_id": f"r15_{len(retained) + 1:04d}",
            "seed": seed,
            "objective": block["objective"],
            "design": block_id,
            "search_mode": block["search_mode"],
            "method": "rolling_tr_sum",
            "baseline_sampling_policy": "all_window",
            **row,
        })

    block_ids = list(requested_by_block)
    block_audit = [
        {
            "block_id": block_id,
            "requested_count": requested_by_block[block_id],
            "retained_count": retained_by_block.get(block_id, 0),
        }
        for block_id in block_ids
    ]
    exploration_blocks = []
    for block_id in block_ids:
        sample = next(block for block, _ in requested if block["block_id"] == block_id)
        exploration_blocks.append({
            "block_id": block_id,
            "anchor": sample.get("anchor", ""),
            "objective": sample["objective"],
            "search_mode": sample["search_mode"],
            "diagnostic_type": sample["diagnostic_type"],
            "hypothesis": sample["hypothesis"],
            "expected_behavior_change": "Improve cost-adjusted return or maximum drawdown without relying on one isolated trade.",
            "falsifying_outcome": "The block remains below current leaders or any gain is isolated, concentrated, or gap-dependent.",
            "metrics": [
                "cost_adjusted_total_return", "cost_adjusted_average_trade", "cost_adjusted_max_drawdown",
                "trade_count", "neighborhood_stability", "return_concentration", "gap_dependency_display_only",
            ],
            "minimum_trade_count": 10,
            "evidence_boundary": "in_sample_immutable_closed_only",
        })

    plan = {
        "schema_version": 4,
        "status": "approved_for_execution",
        "campaign_id": CAMPAIGN_ID,
        "stage_id": STAGE_ID,
        "stage_kind": "continuation_dual_purpose_exploration",
        "predecessor_stage_ids": ["continuation_round_14_large_multiblock_exploration_all_window"],
        "selection_provenance": (
            "The user defined two continuing purposes for parameter exploration: broad coverage to reduce the chance of "
            "missing promising regions, and one-parameter refinement around strong combinations to seek return or maximum-"
            "drawdown improvement. This plan freezes both purposes before compute and keeps their blocks separate."
        ),
        "source": str(PROJECT_ROOT / "runtime_inputs" / "market_data" / "k200_clean_15s_session_filled.csv"),
        "data_preparation_manifest": str(PROJECT_ROOT / "runtime_inputs" / "data_preparation" / "data_preparation_manifest.json"),
        "scenario_definition": str(PLAN_ROOT / "v4_4_scenario_groups_single_select_combined_exit_20260801.json"),
        "instrument_profile": str(VARIANT_ROOT / "instrument_profiles" / "k200m.future_v2.json"),
        "train_start": "2026-05-26 00:00:00",
        "train_end": "2026-07-08 23:52:00",
        "entry_fill_mode": "calculated_threshold",
        "entry_execution_policy": "wait_next_real_trade",
        "entry_slippage": 0,
        "baseline_sampling_policy": "all_window",
        "exit_mode": "combined",
        "resources": {"workers": 4, "batch_size": 8, "minimum_free_memory_mb": 4096},
        "planned_output_root": str(OUTPUT_ROOT),
        "objective_contract": {
            "objectives": [
                "broad_parameter_space_discovery", "unrestricted_cost_adjusted_total_return",
                "scenario_1_cost_adjusted_total_return", "cost_adjusted_average_return_ge10",
                "cost_adjusted_average_return_ge20", "low_drawdown_moderate_trade_pareto",
            ],
            "combined_score": False,
            "gap_excluded_return_role": "display_only_dependency_audit",
            "parameter_acceptance": "none",
        },
        "exploration_blocks": exploration_blocks,
        "anti_join": {
            "source_snapshot_id": snapshot_id,
            "completed_coordinate_count": len(completed),
            "active_coordinate_count": 0,
            "pending_coordinate_count": 0,
            "requested_coordinate_count": len(requested),
            "unique_requested_coordinate_count": len(seen),
            "internal_duplicate_count": internal_duplicate_count,
            "completed_overlap_count": completed_overlap_count,
            "retained_new_coordinate_count": len(retained),
        },
        "delivery_contract": {
            "intermediate_html": False,
            "final_shared_html_once": True,
            "cumulative_main": str(PROJECT_ROOT / "results" / "all_completed_union_analysis" / "index.html"),
            "shared_trade_review": str(PROJECT_ROOT / "results" / "all_completed_union_analysis" / "trade_review" / "index.html"),
        },
        "grid_blocks": [],
        "explicit_combos": retained,
        "stop_conditions": [
            "source_or_result_semantics_identity_mismatch", "nonfinite_primary_evidence",
            "memory_floor_failure", "partial_batches_are_not_interpreted",
        ],
    }
    PLAN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "status": "frozen_before_compute",
        "plan": str(PLAN_PATH),
        "plan_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "plan_size_bytes": PLAN_PATH.stat().st_size,
        "source_snapshot_id": snapshot_id,
        "output_root": str(OUTPUT_ROOT),
        "output_root_absent": not OUTPUT_ROOT.exists(),
        "completed_coordinate_count": len(completed),
        "requested_coordinate_count": len(requested),
        "unique_requested_coordinate_count": len(seen),
        "internal_duplicate_count": internal_duplicate_count,
        "completed_overlap_count": completed_overlap_count,
        "retained_new_coordinate_count": len(retained),
        "block_counts": block_audit,
        "resources": plan["resources"],
        "parameter_acceptance": "none",
    }
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
