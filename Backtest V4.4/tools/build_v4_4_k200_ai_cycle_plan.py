from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_ROOT = PROJECT_ROOT / "research_variants" / "short_momentum_net_drop_rebound_v4_4"
PLAN_ROOT = VARIANT_ROOT / "plans" / "generated_ai_cycle_20260806"
CAMPAIGN_ID = "v4_4_positive_entry_signal_repair_20260805"
CAMPAIGN_ROOT = PROJECT_ROOT / "results" / "campaigns" / CAMPAIGN_ID
PARAMETER_FIELDS = ("e", "bh", "trw", "k", "w", "m", "speed_window_bars")

LEAP_AXES = {
    "e": [30, 42, 54, 72, 84, 112, 156, 216, 240, 300, 360, 450, 520, 680, 760, 960, 1040, 1200],
    "bh": [72, 84, 108, 132, 168, 228, 300, 360, 456, 510, 612, 720, 900, 960, 1200],
    "trw": [3, 4, 5, 7, 9, 11, 14, 18, 20, 24, 28, 32, 40, 48],
    "k": [0.55, 0.6, 0.7, 0.8, 0.95, 1.0, 1.2, 1.4, 1.5, 1.8, 2.2, 2.6],
    "w": [1, 2, 3, 4, 6, 8, 10, 14, 18, 24, 32, 48, 64, 96, 128],
    "m": [0.5, 0.6, 0.8, 1.1, 1.5, 1.8, 2.5, 3.5, 4.5, 5.5, 8.0, 10.0, 12.0, 14.0],
    "speed_window_bars": [48, 54, 72, 90, 120, 144, 180, 220, 280, 350, 440, 520, 640, 780, 900, 1080],
}

GRID_SPECS = (
    {
        "block_id": "grid_anchor_a_e",
        "anchor": {"e": 480, "bh": 171, "trw": 12, "k": 1.26, "w": 7, "m": 4.5, "speed_window_bars": 388},
        "parameter": "e",
        "values": [288, 336, 384, 432, 480, 528, 576, 672, 720],
        "objective": "unrestricted_total_return",
    },
    {
        "block_id": "grid_anchor_b_bh",
        "anchor": {"e": 320, "bh": 240, "trw": 22, "k": 1.0, "w": 6, "m": 4.5, "speed_window_bars": 330},
        "parameter": "bh",
        "values": [120, 144, 168, 192, 216, 228, 240, 252, 264, 288, 312, 336, 360, 384],
        "objective": "scenario_1_total_return",
    },
    {
        "block_id": "grid_anchor_c_e",
        "anchor": {"e": 96, "bh": 612, "trw": 24, "k": 1.6, "w": 16, "m": 2.0, "speed_window_bars": 308},
        "parameter": "e",
        "values": [48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224],
        "objective": "average_return_ge10_ge20",
    },
    {
        "block_id": "grid_anchor_d_s",
        "anchor": {"e": 320, "bh": 170, "trw": 11, "k": 1.4, "w": 6, "m": 4.5, "speed_window_bars": 120},
        "parameter": "speed_window_bars",
        "values": [60, 72, 84, 96, 108, 120, 132, 144, 168, 192],
        "objective": "non_gap_return_and_low_concentration",
    },
    {
        "block_id": "grid_anchor_e_s",
        "anchor": {"e": 150, "bh": 504, "trw": 24, "k": 1.6, "w": 10, "m": 2.5, "speed_window_bars": 262},
        "parameter": "speed_window_bars",
        "values": [158, 184, 210, 236, 262, 288, 314, 340, 392],
        "objective": "low_drawdown_frontier",
    },
    {
        "block_id": "grid_anchor_f_s",
        "anchor": {"e": 450, "bh": 228, "trw": 9, "k": 2.6, "w": 128, "m": 14.0, "speed_window_bars": 48},
        "parameter": "speed_window_bars",
        "values": [24, 32, 40, 48, 56, 64, 72, 84, 96],
        "objective": "remote_non_gap_return_and_low_concentration",
    },
    {
        "block_id": "grid_anchor_g_trw",
        "anchor": {"e": 680, "bh": 720, "trw": 48, "k": 1.5, "w": 24, "m": 10.0, "speed_window_bars": 350},
        "parameter": "trw",
        "values": [24, 28, 32, 36, 40, 44, 48, 52, 56, 64],
        "objective": "remote_average_return_ge10",
    },
    {
        "block_id": "grid_anchor_h_k",
        "anchor": {"e": 520, "bh": 510, "trw": 20, "k": 1.5, "w": 128, "m": 12.0, "speed_window_bars": 54},
        "parameter": "k",
        "values": [0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1],
        "objective": "remote_balanced_return_scenario_1",
    },
)


def coordinate_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        int(row["e"]), int(row["bh"]), int(row["trw"]), round(float(row["k"]), 8),
        int(row["w"]), round(float(row["m"]), 8), int(row["speed_window_bars"]),
    )


def read_csv_coordinates(path: Path) -> set[tuple[object, ...]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {coordinate_key(row) for row in csv.DictReader(handle)}


def current_snapshot() -> tuple[str, Path]:
    pointer = json.loads(
        (PROJECT_ROOT / "results" / "all_completed_union_analysis" / "current_snapshot.json").read_text(encoding="utf-8")
    )
    return str(pointer["union_snapshot_id"]), Path(str(pointer["snapshot_root"]))


def coordinate_authority() -> tuple[str, set[tuple[object, ...]], set[tuple[object, ...]], list[str]]:
    snapshot_id, snapshot_root = current_snapshot()
    completed = read_csv_coordinates(snapshot_root / "analysis_summary.csv")
    pending: set[tuple[object, ...]] = set()
    completed_stages: list[str] = []
    if CAMPAIGN_ROOT.exists():
        for stage in sorted(path for path in CAMPAIGN_ROOT.iterdir() if path.is_dir()):
            input_plan = stage / "input_plan.json"
            if not input_plan.exists():
                continue
            payload = json.loads(input_plan.read_text(encoding="utf-8"))
            keys = {coordinate_key(row) for row in payload.get("explicit_combos", [])}
            if (stage / "completion_manifest.json").exists():
                completed.update(read_csv_coordinates(stage / "stage_summary.csv"))
                completed_stages.append(stage.name)
            else:
                pending.update(keys)
    return snapshot_id, completed, pending, completed_stages


def initial_leap_rows() -> list[tuple[dict[str, object], dict[str, object]]]:
    handoff = (
        CAMPAIGN_ROOT
        / "continuation_round_15_dual_purpose_broad_and_single_axis_all_window"
        / "next_round_handoff"
        / "next_round_parameters.csv"
    )
    rows: list[tuple[dict[str, object], dict[str, object]]] = []
    with handoff.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["search_mode"] != "broad_jump":
                continue
            combo = {
                "e": int(row["e"]), "bh": int(row["bh"]), "trw": int(row["trw"]),
                "k": float(row["k"]), "w": int(row["w"]), "m": float(row["m"]),
                "speed_window_bars": int(row["speed_window_bars"]),
            }
            rows.append(({"block_id": "leap_handoff_r15", "objective": "broad_multimetric_discovery"}, combo))
    return rows


def generated_leap_rows(seed: int, count: int) -> list[tuple[dict[str, object], dict[str, object]]]:
    rng = random.Random(seed)
    rows: list[tuple[dict[str, object], dict[str, object]]] = []
    seen: set[tuple[object, ...]] = set()
    while len(rows) < count:
        combo = {field: rng.choice(LEAP_AXES[field]) for field in PARAMETER_FIELDS}
        key = coordinate_key(combo)
        if key in seen:
            continue
        seen.add(key)
        rows.append(({
            "block_id": f"leap_stratified_seed_{seed}",
            "objective": "broad_multimetric_discovery",
        }, combo))
    return rows


def known_grid_rows() -> list[tuple[dict[str, object], dict[str, object]]]:
    rows: list[tuple[dict[str, object], dict[str, object]]] = []
    for spec in GRID_SPECS:
        for value in spec["values"]:
            combo = dict(spec["anchor"])
            combo[str(spec["parameter"])] = value
            rows.append((dict(spec), combo))
    return rows


def value(row: dict[str, str], field: str, default: float = 0.0) -> float:
    raw = row.get(field, "")
    return default if raw in ("", None) else float(raw)


def load_evidence_rows() -> list[dict[str, str]]:
    _, snapshot_root = current_snapshot()
    paths = [snapshot_root / "analysis_summary.csv"]
    paths.extend(sorted(CAMPAIGN_ROOT.glob("continuation_round_*_ai_*_all_window/compact_analysis/analysis_summary.csv")))
    rows: dict[tuple[object, ...], dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows[coordinate_key(row)] = row
    return list(rows.values())


def separated(candidate: dict[str, str], anchors: list[dict[str, str]]) -> bool:
    for anchor in anchors:
        changes = 0
        for field in ("e", "bh", "trw", "w", "speed_window_bars"):
            left, right = max(1.0, value(candidate, field)), max(1.0, value(anchor, field))
            if max(left, right) / min(left, right) >= 1.25:
                changes += 1
        if abs(value(candidate, "k") - value(anchor, "k")) >= 0.2:
            changes += 1
        if abs(value(candidate, "m") - value(anchor, "m")) >= 1.0:
            changes += 1
        if changes < 2:
            return False
    return True


def adaptive_grid_specs(round_number: int) -> tuple[dict[str, object], ...]:
    rows = load_evidence_rows()
    cycle_index = max(0, (round_number - 22) // 3)
    axis_cycles = {
        "total_return": ("e", "k", "bh", "speed_window_bars"),
        "scenario_1_total": ("bh", "trw", "k", "e"),
        "average_ge20": ("trw", "e", "k", "speed_window_bars"),
        "non_gap_return": ("speed_window_bars", "w", "m", "trw"),
        "low_drawdown_positive": ("w", "m", "speed_window_bars", "k"),
    }
    factor_cycles = (
        (0.60, 0.75, 0.85, 1.00, 1.15, 1.30, 1.50),
        (0.50, 0.65, 0.80, 1.00, 1.20, 1.40, 1.70),
        (0.70, 0.82, 0.92, 1.00, 1.08, 1.18, 1.32),
        (0.40, 0.55, 0.70, 1.00, 1.35, 1.65, 2.00),
    )
    views = (
        ("total_return", "train_cost_adjusted_return", False, lambda row: True),
        ("scenario_1_total", "train_cost_adjusted_return", False, lambda row: str(row.get("scenario_1_qualified", "")).lower() == "true"),
        ("average_ge20", "train_cost_adjusted_avg_trade", False, lambda row: value(row, "train_trade_count") >= 20),
        ("non_gap_return", "train_cost_adjusted_return_excluding_gap", False, lambda row: value(row, "train_trade_count") >= 20),
        ("low_drawdown_positive", "train_cost_adjusted_max_drawdown_abs", True, lambda row: value(row, "train_trade_count") >= 20 and value(row, "train_cost_adjusted_return") > 0),
    )
    anchors: list[dict[str, str]] = []
    specs: list[dict[str, object]] = []
    for objective, metric, ascending, predicate in views:
        parameter = axis_cycles[objective][cycle_index % len(axis_cycles[objective])]
        eligible = [row for row in rows if predicate(row) and row.get(metric, "") not in ("", None)]
        eligible.sort(key=lambda row: value(row, metric), reverse=not ascending)
        selected = next((row for row in eligible if separated(row, anchors)), None)
        if selected is None:
            continue
        anchors.append(selected)
        anchor = {
            "e": int(value(selected, "e")), "bh": int(value(selected, "bh")),
            "trw": int(value(selected, "trw")), "k": value(selected, "k"),
            "w": int(value(selected, "w")), "m": value(selected, "m"),
            "speed_window_bars": int(value(selected, "speed_window_bars")),
        }
        center = value(selected, parameter)
        base_factors = factor_cycles[cycle_index % len(factor_cycles)]
        expansion_band = cycle_index // len(factor_cycles)
        factors = tuple(
            round(min(2.50, max(0.25, 1.0 + (factor - 1.0) * (1.0 + 0.15 * expansion_band))), 4)
            for factor in base_factors
        )
        if parameter in ("k", "m"):
            values = sorted({round(max(0.05, center * factor), 3) for factor in factors})
        else:
            values = sorted({max(1, int(round(center * factor))) for factor in factors})
        specs.append({
            "block_id": f"adaptive_cycle_{cycle_index}_{objective}_{parameter}",
            "anchor": anchor,
            "parameter": parameter,
            "values": values,
            "objective": objective,
        })
    if not specs:
        raise RuntimeError("no separated adaptive-grid anchors were found")
    return tuple(specs)


def grid_rows(specs: tuple[dict[str, object], ...]) -> list[tuple[dict[str, object], dict[str, object]]]:
    rows: list[tuple[dict[str, object], dict[str, object]]] = []
    for spec in specs:
        for item in spec["values"]:
            combo = dict(spec["anchor"])
            combo[str(spec["parameter"])] = item
            rows.append((dict(spec), combo))
    return rows


def build_requested(
    args: argparse.Namespace,
    selected_grid_specs: tuple[dict[str, object], ...],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    if args.phase == "initial-leap":
        return initial_leap_rows()
    if args.phase == "generated-leap":
        return generated_leap_rows(args.seed, args.count)
    return grid_rows(selected_grid_specs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--phase", choices=("initial-leap", "generated-leap", "known-grid", "adaptive-grid"), required=True)
    parser.add_argument("--seed", type=int, default=440617)
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()

    stage_suffix = args.phase.replace("-", "_")
    stage_id = f"continuation_round_{args.round:02d}_ai_{stage_suffix}_all_window"
    output_root = CAMPAIGN_ROOT / stage_id
    plan_path = PLAN_ROOT / f"{stage_id}.json"
    audit_path = PLAN_ROOT / f"{stage_id}.audit.json"
    if output_root.exists() or plan_path.exists():
        raise FileExistsError(f"target already exists: {output_root} or {plan_path}")

    snapshot_id, completed, pending, completed_stages = coordinate_authority()
    selected_grid_specs: tuple[dict[str, object], ...] = ()
    if args.phase == "known-grid":
        selected_grid_specs = GRID_SPECS
    elif args.phase == "adaptive-grid":
        selected_grid_specs = adaptive_grid_specs(args.round)
    requested = build_requested(args, selected_grid_specs)
    retained: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    completed_overlap = 0
    pending_overlap = 0
    duplicate_count = 0
    block_counts: dict[str, dict[str, int]] = {}
    for block, combo in requested:
        block_id = str(block["block_id"])
        counts = block_counts.setdefault(block_id, {"requested": 0, "retained": 0})
        counts["requested"] += 1
        key = coordinate_key(combo)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        if key in completed:
            completed_overlap += 1
            continue
        if key in pending:
            pending_overlap += 1
            continue
        counts["retained"] += 1
        retained.append({
            "candidate_id": f"r{args.round}_{len(retained) + 1:04d}",
            "seed": f"ai_cycle_{args.round}",
            "objective": block["objective"],
            "design": block_id,
            "search_mode": "single_axis" if args.phase == "known-grid" else "broad_jump",
            "method": "rolling_tr_sum",
            "baseline_sampling_policy": "all_window",
            **combo,
        })
    if not retained:
        raise RuntimeError("anti-join retained zero new coordinates")

    grid_blocks = []
    if args.phase in ("known-grid", "adaptive-grid"):
        grid_blocks = [
            {
                "block_id": spec["block_id"],
                "phase": "grid_refinement",
                "anchor": spec["anchor"],
                "changed_parameter": spec["parameter"],
                "exact_values": spec["values"],
                "finite_lower_bound": min(spec["values"]),
                "finite_upper_bound": max(spec["values"]),
                "expected_closed_curve_count": len(spec["values"]),
                "all_other_parameters_fixed": True,
                "objective": spec["objective"],
            }
            for spec in selected_grid_specs
        ]

    predecessor = completed_stages[-1:] if completed_stages else []
    plan = {
        "schema_version": 4,
        "status": "approved_for_execution",
        "campaign_id": CAMPAIGN_ID,
        "stage_id": stage_id,
        "stage_kind": "continuation_ai_leap_grid_cycle",
        "predecessor_stage_ids": predecessor,
        "selection_provenance": (
            "User-authorized AI-led cycle: multi-round nonadjacent leap search, finite one-parameter grids around "
            "promising nonadjacent anchors, then renewed leap search. Exact completed and pending coordinates are excluded."
        ),
        "source": str(PROJECT_ROOT / "runtime_inputs" / "market_data" / "k200_clean_15s_session_filled.csv"),
        "data_preparation_manifest": str(PROJECT_ROOT / "runtime_inputs" / "data_preparation" / "data_preparation_manifest.json"),
        "scenario_definition": str(VARIANT_ROOT / "plans" / "v4_4_scenario_groups_single_select_combined_exit_20260801.json"),
        "instrument_profile": str(VARIANT_ROOT / "instrument_profiles" / "k200m.future_v2.json"),
        "train_start": "2026-05-26 00:00:00",
        "train_end": "2026-07-08 23:52:00",
        "entry_fill_mode": "calculated_threshold",
        "entry_execution_policy": "wait_next_real_trade",
        "entry_slippage": 0,
        "baseline_sampling_policy": "all_window",
        "exit_mode": "combined",
        "resources": {"workers": 4, "batch_size": 8, "minimum_free_memory_mb": 4096},
        "planned_output_root": str(output_root),
        "objective_contract": {
            "objectives": ["broad_multimetric_discovery", "single_parameter_refinement"],
            "combined_score": False,
            "gap_excluded_return_role": "display_only_dependency_audit",
            "parameter_acceptance": "none",
        },
        "exploration_cycle": {
            "phase": "grid_refinement" if args.phase in ("known-grid", "adaptive-grid") else "leap_search",
            "round": args.round,
            "next_phase_after_closed_interpretation": "renewed_leap_search" if args.phase in ("known-grid", "adaptive-grid") else "continue_or_grid_from_closed_evidence",
            "user_correction_checkpoint": "between_cycles",
        },
        "anti_join": {
            "source_snapshot_id": snapshot_id,
            "completed_coordinate_count": len(completed),
            "pending_coordinate_count": len(pending),
            "requested_coordinate_count": len(requested),
            "internal_duplicate_count": duplicate_count,
            "completed_overlap_count": completed_overlap,
            "pending_overlap_count": pending_overlap,
            "retained_new_coordinate_count": len(retained),
        },
        "delivery_contract": {
            "intermediate_html": False,
            "final_shared_html_once": True,
            "cumulative_main": str(PROJECT_ROOT / "results" / "all_completed_union_analysis" / "index.html"),
            "shared_trade_review": str(PROJECT_ROOT / "results" / "all_completed_union_analysis" / "trade_review" / "index.html"),
        },
        "grid_blocks": [],
        "declared_grid_blocks": grid_blocks,
        "explicit_combos": retained,
        "stop_conditions": [
            "source_or_result_semantics_identity_mismatch",
            "nonfinite_primary_evidence",
            "memory_floor_failure",
            "partial_batches_are_not_interpreted",
        ],
    }

    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "status": "frozen_before_compute",
        "plan": str(plan_path),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "stage_id": stage_id,
        "phase": args.phase,
        "output_root": str(output_root),
        "completed_coordinate_count": len(completed),
        "pending_coordinate_count": len(pending),
        "requested_coordinate_count": len(requested),
        "internal_duplicate_count": duplicate_count,
        "completed_overlap_count": completed_overlap,
        "pending_overlap_count": pending_overlap,
        "retained_new_coordinate_count": len(retained),
        "block_counts": block_counts,
        "resources": plan["resources"],
        "parameter_acceptance": "none",
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
