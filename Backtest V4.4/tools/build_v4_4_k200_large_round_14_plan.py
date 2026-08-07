from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_ROOT = (
    PROJECT_ROOT / "research_variants" / "short_momentum_net_drop_rebound_v4_4"
)
PLAN_ROOT = VARIANT_ROOT / "plans"
CAMPAIGN_ROOT = (
    PROJECT_ROOT
    / "results"
    / "campaigns"
    / "v4_4_positive_entry_signal_repair_20260805"
)
STAGE_ID = "continuation_round_14_large_multiblock_exploration_all_window"
PLAN_PATH = PLAN_ROOT / "v4_4_k200_large_multiblock_20260805_round_14.json"
AUDIT_PATH = PLAN_ROOT / "v4_4_k200_large_multiblock_20260805_round_14.audit.json"
OUTPUT_ROOT = CAMPAIGN_ROOT / STAGE_ID
PARAMETER_FIELDS = ("e", "bh", "trw", "k", "w", "m", "speed_window_bars")


ANCHORS = {
    "unrestricted": {
        "e": 480,
        "bh": 171,
        "trw": 12,
        "k": 1.26,
        "w": 6,
        "m": 4.5,
        "speed_window_bars": 388,
    },
    "scenario_1": {
        "e": 320,
        "bh": 240,
        "trw": 22,
        "k": 1.0,
        "w": 6,
        "m": 4.5,
        "speed_window_bars": 330,
    },
    "average_return": {
        "e": 112,
        "bh": 612,
        "trw": 24,
        "k": 1.6,
        "w": 10,
        "m": 2.5,
        "speed_window_bars": 308,
    },
    "low_drawdown": {
        "e": 150,
        "bh": 504,
        "trw": 24,
        "k": 1.6,
        "w": 10,
        "m": 2.5,
        "speed_window_bars": 310,
    },
}


BLOCK_DEFINITIONS = (
    {
        "block_id": "unrestricted_e_bh_surface",
        "anchor": "unrestricted",
        "axes": {"e": [240, 336, 480, 576, 720], "bh": [120, 145, 171, 205, 257]},
        "objective": "unrestricted_total_return",
        "search_mode": "module_pair",
        "diagnostic_type": "entry_baseline_geometry",
        "hypothesis": "The unrestricted E plateau may survive materially different BH horizons.",
        "expected": "A structural ridge should retain strong cost-adjusted return across neighboring proportional E/BH combinations.",
        "falsify": "Most new combinations lose return or create isolated gains with worse drawdown and concentration.",
    },
    {
        "block_id": "unrestricted_trw_k_surface",
        "anchor": "unrestricted",
        "axes": {"trw": [8, 10, 12, 14, 16], "k": [0.95, 1.1, 1.26, 1.45, 1.6]},
        "objective": "unrestricted_total_return",
        "search_mode": "module_pair",
        "diagnostic_type": "entry_threshold_geometry",
        "hypothesis": "TRW and K may compensate for each other outside the previously sampled ridge pairs.",
        "expected": "A coherent diagonal region should preserve return while changing trade count and entry strictness.",
        "falsify": "The K1.26/TRW12 neighborhood remains an isolated fixed-anchor peak.",
    },
    {
        "block_id": "unrestricted_w_m_surface",
        "anchor": "unrestricted",
        "axes": {"w": [3, 5, 6, 8, 10], "m": [3.5, 4.0, 4.5, 5.0, 6.0]},
        "objective": "unrestricted_total_return",
        "search_mode": "module_pair",
        "diagnostic_type": "profit_giveback",
        "hypothesis": "Joint W/M changes may reduce profit giveback without the one-axis M deterioration.",
        "expected": "Neighboring W/M points should improve retention or drawdown without concentrating return in one trade.",
        "falsify": "W6/M4.5 remains dominant and alternatives worsen return quality.",
    },
    {
        "block_id": "unrestricted_e_s_surface",
        "anchor": "unrestricted",
        "axes": {"e": [240, 336, 480, 576, 720], "speed_window_bars": [272, 330, 388, 466, 582]},
        "objective": "unrestricted_total_return",
        "search_mode": "module_pair",
        "diagnostic_type": "entry_speed_interaction",
        "hypothesis": "The broad E plateau may require a proportional speed horizon rather than fixed S388.",
        "expected": "Several proportional E/S pairs should retain strong return with comparable drawdown.",
        "falsify": "Only the existing E/S center remains competitive.",
    },
    {
        "block_id": "scenario_1_e_bh_surface",
        "anchor": "scenario_1",
        "axes": {"e": [208, 256, 320, 400, 512], "bh": [156, 192, 240, 300, 384]},
        "objective": "scenario_1_total_return",
        "search_mode": "broad_jump",
        "diagnostic_type": "scenario_entry_baseline_geometry",
        "hypothesis": "Scenario-1 qualification may persist across a proportional E/BH surface absent from the repaired grid.",
        "expected": "Multiple points should remain qualified and competitive instead of one exact coordinate.",
        "falsify": "Qualification or return collapses outside the incumbent.",
    },
    {
        "block_id": "scenario_1_trw_k_surface",
        "anchor": "scenario_1",
        "axes": {"trw": [16, 19, 22, 25, 28], "k": [0.8, 0.9, 1.0, 1.1, 1.25]},
        "objective": "scenario_1_total_return",
        "search_mode": "module_pair",
        "diagnostic_type": "scenario_entry_threshold_geometry",
        "hypothesis": "Scenario-1 entry quality may follow a TRW/K compensation ridge.",
        "expected": "A neighboring qualified region should balance return, drawdown, and trade count.",
        "falsify": "New points lose qualification or deteriorate across several metrics.",
    },
    {
        "block_id": "scenario_1_m_s_surface",
        "anchor": "scenario_1",
        "axes": {"m": [3.0, 3.75, 4.5, 5.5], "speed_window_bars": [231, 280, 330, 396, 495]},
        "objective": "scenario_1_total_return",
        "search_mode": "module_pair",
        "diagnostic_type": "scenario_exit_timing",
        "hypothesis": "Scenario-1 exits may improve when rebound and speed horizons move together.",
        "expected": "A small stable M/S region should improve retention or drawdown while preserving qualification.",
        "falsify": "The incumbent exit pair remains superior or gains are isolated.",
    },
    {
        "block_id": "average_e_bh_surface",
        "anchor": "average_return",
        "axes": {"e": [80, 96, 112, 136, 160], "bh": [432, 504, 612, 720, 864]},
        "objective": "average_return_ge10_ge20",
        "search_mode": "module_pair",
        "diagnostic_type": "selective_entry_geometry",
        "hypothesis": "The repaired average-return leader may extend across a proportional E/BH neighborhood.",
        "expected": "Several points should retain high average trade and low drawdown with at least 20 trades.",
        "falsify": "The E112/BH612 result is isolated or falls below minimum trade count nearby.",
    },
    {
        "block_id": "average_trw_k_surface",
        "anchor": "average_return",
        "axes": {"trw": [18, 21, 24, 28, 30], "k": [1.3, 1.45, 1.6, 1.75, 2.0]},
        "objective": "average_return_ge10_ge20",
        "search_mode": "module_pair",
        "diagnostic_type": "selective_threshold_geometry",
        "hypothesis": "Average return may improve through a broader TRW/K strictness surface.",
        "expected": "A non-isolated region should improve median trade or drawdown while retaining enough trades.",
        "falsify": "Stricter or looser pairs reduce sample size or overall quality without a stable gain.",
    },
    {
        "block_id": "average_w_m_surface",
        "anchor": "average_return",
        "axes": {"w": [6, 8, 10, 12, 16], "m": [1.5, 2.0, 2.5, 3.0, 3.5]},
        "objective": "average_return_ge10_ge20",
        "search_mode": "module_pair",
        "diagnostic_type": "selective_exit_retention",
        "hypothesis": "Average-return exits may have a W/M region with better profit retention.",
        "expected": "Several W/M neighbors should preserve high average return and low drawdown.",
        "falsify": "Changes merely reduce trades or create one isolated winner.",
    },
    {
        "block_id": "low_drawdown_e_s_surface",
        "anchor": "low_drawdown",
        "axes": {"e": [80, 112, 150, 200, 280], "speed_window_bars": [216, 262, 310, 370, 465]},
        "objective": "low_drawdown_moderate_trade_pareto",
        "search_mode": "broad_jump",
        "diagnostic_type": "drawdown_entry_speed_geometry",
        "hypothesis": "A proportional E/S surface may retain moderate return while lowering drawdown.",
        "expected": "New nondominated points should combine positive return, moderate trades, and lower drawdown.",
        "falsify": "Lower drawdown comes only from collapsing return or trade count.",
    },
    {
        "block_id": "low_drawdown_bh_m_surface",
        "anchor": "low_drawdown",
        "axes": {"bh": [360, 432, 504, 600, 720], "m": [1.5, 2.0, 2.5, 3.0, 3.5]},
        "objective": "low_drawdown_moderate_trade_pareto",
        "search_mode": "module_pair",
        "diagnostic_type": "drawdown_baseline_exit_geometry",
        "hypothesis": "BH and M may jointly control drawdown without sacrificing the moderate-trade profile.",
        "expected": "Several points should improve the return/drawdown frontier with at least 20 trades.",
        "falsify": "Any drawdown reduction is purchased by material return loss or sample collapse.",
    },
)


REMOTE_AXES = {
    "e": [40, 80, 160, 320, 640],
    "bh": [120, 240, 480, 720, 960],
    "trw": [6, 12, 18, 24, 36],
    "k": [0.8, 1.1, 1.4, 1.7, 2.0],
    "w": [1, 4, 10, 32, 128],
    "m": [0.5, 2.0, 4.5, 8.0, 12.0],
    "speed_window_bars": [80, 240, 400, 600, 960],
}


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
        (PROJECT_ROOT / "results" / "all_completed_union_analysis" / "current_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    snapshot_id = str(pointer["union_snapshot_id"])
    return snapshot_id, (
        PROJECT_ROOT
        / "results"
        / "all_completed_union_analysis"
        / "snapshots"
        / snapshot_id
    )


def completed_coordinates(snapshot_root: Path) -> tuple[set[tuple[object, ...]], dict[tuple[object, ...], str]]:
    keys: set[tuple[object, ...]] = set()
    combo_ids: dict[tuple[object, ...], str] = {}
    with (snapshot_root / "analysis_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            current = coordinate_key(row)
            keys.add(current)
            combo_ids[current] = str(row["combo_id"])
    return keys, combo_ids


def pair_rows(block: dict[str, object]) -> list[dict[str, object]]:
    anchor = dict(ANCHORS[str(block["anchor"])])
    axes = dict(block["axes"])
    names = list(axes)
    rows: list[dict[str, object]] = []
    for values in itertools.product(*(axes[name] for name in names)):
        row = dict(anchor)
        row.update(dict(zip(names, values)))
        rows.append(row)
    return rows


def remote_rows(completed: set[tuple[object, ...]]) -> list[dict[str, object]]:
    rng = random.Random(44014)
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    while len(rows) < 60:
        row = {name: rng.choice(values) for name, values in REMOTE_AXES.items()}
        current = coordinate_key(row)
        if current in seen or current in completed:
            continue
        seen.add(current)
        rows.append(row)
    return rows


def main() -> None:
    snapshot_id, snapshot_root = current_snapshot()
    completed, combo_ids = completed_coordinates(snapshot_root)
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)

    requested: list[tuple[dict[str, object], dict[str, object]]] = []
    block_audit: list[dict[str, object]] = []
    for block in BLOCK_DEFINITIONS:
        rows = pair_rows(block)
        requested.extend((block, row) for row in rows)
        block_audit.append({"block_id": block["block_id"], "requested_count": len(rows)})

    remote_block = {
        "block_id": "remote_sparse_controls",
        "anchor": "",
        "objective": "remote_multimetric_control",
        "search_mode": "broad_jump",
        "diagnostic_type": "remote_parameter_space_control",
        "hypothesis": "Sparse remote combinations may expose a missed region unrelated to current leaders.",
        "expected": "Any useful remote result should be supported by nearby economic structure rather than one isolated point.",
        "falsify": "Remote controls remain materially weaker across return, drawdown, and concentration.",
    }
    remote = remote_rows(completed)
    requested.extend((remote_block, row) for row in remote)
    block_audit.append({"block_id": remote_block["block_id"], "requested_count": len(remote)})

    retained: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    internal_duplicate_count = 0
    completed_overlap_count = 0
    retained_by_block: dict[str, int] = {}
    for block, row in requested:
        current = coordinate_key(row)
        if current in seen:
            internal_duplicate_count += 1
            continue
        seen.add(current)
        if current in completed:
            completed_overlap_count += 1
            continue
        block_id = str(block["block_id"])
        retained_by_block[block_id] = retained_by_block.get(block_id, 0) + 1
        anchor_name = str(block.get("anchor", ""))
        anchor_combo_id = (
            combo_ids[coordinate_key(ANCHORS[anchor_name])] if anchor_name else ""
        )
        retained.append(
            {
                "candidate_id": f"r14_{len(retained) + 1:04d}",
                "seed": anchor_combo_id or "remote_sparse_control",
                "objective": str(block["objective"]),
                "design": block_id,
                "search_mode": str(block["search_mode"]),
                "method": "rolling_tr_sum",
                "baseline_sampling_policy": "all_window",
                **row,
            }
        )

    for audit in block_audit:
        audit["retained_count"] = retained_by_block.get(str(audit["block_id"]), 0)

    plan = {
        "schema_version": 4,
        "status": "approved_for_execution",
        "campaign_id": "v4_4_positive_entry_signal_repair_20260805",
        "stage_id": STAGE_ID,
        "stage_kind": "continuation_large_multiblock_exploration",
        "predecessor_stage_ids": ["continuation_round_13_stricter_entry_k_expansion_all_window"],
        "selection_provenance": (
            "The user explicitly authorized another large K200 exploration round. "
            "The current repaired 4,747-coordinate evidence leaves separate unrestricted, "
            "Scenario-1, average-return, and low-drawdown surfaces plus remote parameter-space "
            "controls unresolved. This plan freezes 294 exact new coordinates across twelve "
            "evidence-led pair surfaces and one 60-coordinate remote sparse-control block. "
            "No combined score or parameter acceptance is created."
        ),
        "source": str(PROJECT_ROOT / "runtime_inputs" / "market_data" / "k200_clean_15s_session_filled.csv"),
        "data_preparation_manifest": str(
            PROJECT_ROOT / "runtime_inputs" / "data_preparation" / "data_preparation_manifest.json"
        ),
        "scenario_definition": str(
            PLAN_ROOT / "v4_4_scenario_groups_single_select_combined_exit_20260801.json"
        ),
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
                "unrestricted_cost_adjusted_total_return",
                "scenario_1_cost_adjusted_total_return",
                "cost_adjusted_average_return_ge10",
                "cost_adjusted_average_return_ge20",
                "low_drawdown_moderate_trade_pareto",
            ],
            "combined_score": False,
            "gap_excluded_return_role": "display_only_dependency_audit",
            "parameter_acceptance": "none",
        },
        "exploration_blocks": [
            {
                "block_id": block["block_id"],
                "anchor": block.get("anchor", ""),
                "objective": block["objective"],
                "search_mode": block["search_mode"],
                "diagnostic_type": block["diagnostic_type"],
                "hypothesis": block["hypothesis"],
                "expected_behavior_change": block["expected"],
                "falsifying_outcome": block["falsify"],
                "metrics": [
                    "cost_adjusted_total_return",
                    "cost_adjusted_average_trade",
                    "cost_adjusted_max_drawdown",
                    "trade_count",
                    "scenario_1_qualification",
                    "neighborhood_stability",
                    "return_concentration",
                    "gap_dependency_display_only",
                ],
                "minimum_trade_count": 10,
                "evidence_boundary": "in_sample_immutable_closed_only",
            }
            for block in (*BLOCK_DEFINITIONS, remote_block)
        ],
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
            "cumulative_main": str(
                PROJECT_ROOT / "results" / "all_completed_union_analysis" / "index.html"
            ),
            "shared_trade_review": str(
                PROJECT_ROOT
                / "results"
                / "all_completed_union_analysis"
                / "trade_review"
                / "index.html"
            ),
        },
        "grid_blocks": [],
        "explicit_combos": retained,
        "stop_conditions": [
            "source_or_result_semantics_identity_mismatch",
            "nonfinite_primary_evidence",
            "memory_floor_failure",
            "partial_batches_are_not_interpreted",
        ],
    }
    PLAN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plan_hash = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    audit = {
        "status": "frozen_before_compute",
        "plan": str(PLAN_PATH),
        "plan_sha256": plan_hash,
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
