"""Materialize the bounded V4.2 high-return Round-1 plan and audit."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from scenario_groups import COMBINED_SCENARIO_SCHEMA_ID, load_scenario_contract
from v4_2_engine import (
    DATA_PREPARATION_MANIFEST_DEFAULT,
    ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    ENTRY_FILL_CALCULATED_THRESHOLD,
    SOURCE_DEFAULT,
    SOURCE_SHA256,
    TRAIN_END,
    TRAIN_START,
    Combo,
)


VARIANT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = VARIANT_ROOT.parents[1]
PLANS_ROOT = VARIANT_ROOT / "plans"
CAMPAIGNS_ROOT = Path(
    r"F:\Backtest test 6.11\K200_short_momentum_dual_exit"
    r"\v4_2_calculated_entry\campaigns"
)
CAMPAIGN_ID = "v4_2_high_return_min10_min20_leap_20260801"
STAGE_ID = "s01_total_average_exit_axis_probe"
PLAN_PATH = PLANS_ROOT / f"{CAMPAIGN_ID}_{STAGE_ID}.json"
AUDIT_PATH = PLAN_PATH.with_suffix(".audit.json")
OUTPUT_ROOT = CAMPAIGNS_ROOT / CAMPAIGN_ID / STAGE_ID
SCENARIO_DEFINITION = (
    PLANS_ROOT / "v4_2_scenario_groups_single_select_combined_exit_20260801.json"
)
SOURCE_MANIFEST = VARIANT_ROOT / "SOURCE_MANIFEST.json"
DESIGN_MEMO = (
    REPOSITORY_ROOT
    / ".omo"
    / "teams"
    / "team-6e3b2f23"
    / "artifacts"
    / "A_v4_2_high_return_min10_min20_round_design_20260801.md"
)
DESIGN_MEMO_SHA256 = "32a4e0e7fd1bc30415a65c7a6e9d7151becda51a75928f7a00efa5815bc2d3f4"
CURRENT_UNION_POINTER = Path(r"F:\V4_2_results\all_completed_union_analysis\current_snapshot.json")
CURRENT_UNION_SNAPSHOT_ID = "35d8e9e2495027d344070f423a2d1fdb2d90c7b195ff5b71262798f5f23b4493"
CURRENT_UNION_ANALYSIS_SHA256 = "d91e90602cdacaab9b6f5cad9756e136c2e288f5d35bdd8ab2a7817b68aee066"
CURRENT_COMPLETED_COUNT = 1470
SPARSE_PLAN_PATH = (
    PLANS_ROOT
    / "v4_2_scenario_3_sparse_leap_20260801_s01_exit_axis_large_step_probe.json"
)
SPARSE_INCOMPLETE_COUNT = 24

ANCHORS = (
    {
        "anchor_id": "total_return_anchor_t",
        "objective": "total_return",
        "combo_id": "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e320_bh180_trw12_k1p16666666667_w6_m4_8d26d6a4ed",
        "e": 320, "bh": 180, "trw": 12, "k": 1.1666666666666667,
        "w": 6, "m": 4.0, "speed_window_bars": 480,
        "anchor_metric": 0.8099857457780448,
    },
    {
        "anchor_id": "average_return_anchor_a",
        "objective": "average_trade_return",
        "combo_id": "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e80_bh360_trw6_k3_w6_m4_17feb8f4ca",
        "e": 80, "bh": 360, "trw": 6, "k": 3.0,
        "w": 6, "m": 4.0, "speed_window_bars": 480,
        "anchor_metric": 0.0037884005655437,
    },
)
M_VALUES = (0.5, 1.5, 3.0, 8.0)
W_VALUES = (2, 3, 12, 24)
S_VALUES = (80, 160, 320, 960)
EXPECTED_COMBO_IDS = (
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e320_bh180_trw12_k1p16666666667_w6_m0p5_586ad922c6",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e320_bh180_trw12_k1p16666666667_w6_m1p5_9f008b908e",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e320_bh180_trw12_k1p16666666667_w6_m3_077a86855d",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e320_bh180_trw12_k1p16666666667_w6_m8_800ba015e6",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e320_bh180_trw12_k1p16666666667_w2_m4_c0a6f7aa96",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e320_bh180_trw12_k1p16666666667_w3_m4_241f6091ef",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e320_bh180_trw12_k1p16666666667_w12_m4_bb6605edb6",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e320_bh180_trw12_k1p16666666667_w24_m4_c3008d70ae",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s80_rx1_e320_bh180_trw12_k1p16666666667_w6_m4_9f4b5d91ab",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s160_rx1_e320_bh180_trw12_k1p16666666667_w6_m4_8ff24461b9",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s320_rx1_e320_bh180_trw12_k1p16666666667_w6_m4_e95ee9b4a9",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s960_rx1_e320_bh180_trw12_k1p16666666667_w6_m4_dda5b40b14",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e80_bh360_trw6_k3_w6_m0p5_000fc8d427",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e80_bh360_trw6_k3_w6_m1p5_03a4e5b651",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e80_bh360_trw6_k3_w6_m3_c422ea1607",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e80_bh360_trw6_k3_w6_m8_61c7d70e4b",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e80_bh360_trw6_k3_w2_m4_687be5edcf",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e80_bh360_trw6_k3_w3_m4_cc18c26260",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e80_bh360_trw6_k3_w12_m4_81a2bc68b7",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s480_rx1_e80_bh360_trw6_k3_w24_m4_9315a88a1e",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s80_rx1_e80_bh360_trw6_k3_w6_m4_2af3011d80",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s160_rx1_e80_bh360_trw6_k3_w6_m4_4d031797a7",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s320_rx1_e80_bh360_trw6_k3_w6_m4_1049476191",
    "v4_2_rolling_tr_sum_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s960_rx1_e80_bh360_trw6_k3_w6_m4_1201f1470e",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved), "size_bytes": resolved.stat().st_size}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def combo_id_from_record(row: dict[str, Any]) -> str:
    return Combo(
        method=str(row["method"]), e=int(row["e"]), bh=int(row["bh"]),
        trw=int(row["trw"]), k=float(row["k"]), w=int(row["w"]), m=float(row["m"]),
        entry_fill_mode=ENTRY_FILL_CALCULATED_THRESHOLD,
        entry_execution_policy=ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        entry_slippage=0.0, speed_window_bars=int(row["speed_window_bars"]),
    ).combo_id


def combo_record(anchor: dict[str, Any], axis: str, value: int | float) -> tuple[str, dict[str, Any]]:
    values = {"w": anchor["w"], "m": anchor["m"], "speed_window_bars": anchor["speed_window_bars"]}
    values[axis] = value
    row = {
        "method": "rolling_tr_sum", "e": anchor["e"], "bh": anchor["bh"],
        "trw": anchor["trw"], "k": anchor["k"], "w": values["w"], "m": values["m"],
        "speed_window_bars": values["speed_window_bars"],
        "entry_fill_mode": ENTRY_FILL_CALCULATED_THRESHOLD,
        "entry_execution_policy": ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        "entry_slippage": 0.0, "seed": anchor["anchor_id"], "objective": anchor["objective"],
        "design": "high_return_round_1_one_axis_exit_probe", "source_combo_id": anchor["combo_id"],
        "leap_round": "round_1", "leap_axis": axis, "leap_value": value,
    }
    return combo_id_from_record(row), row


def stage_grid() -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for anchor in ANCHORS:
        rows.extend(combo_record(anchor, "m", value) for value in M_VALUES)
        rows.extend(combo_record(anchor, "w", value) for value in W_VALUES)
        rows.extend(combo_record(anchor, "speed_window_bars", value) for value in S_VALUES)
    ids = tuple(combo_id for combo_id, _ in rows)
    if ids != EXPECTED_COMBO_IDS:
        raise ValueError("generated coordinate IDs/order differ from the frozen memo")
    return rows


def completed_v4_2_ids() -> tuple[set[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    sources: list[dict[str, Any]] = []
    for completion_path in sorted(CAMPAIGNS_ROOT.rglob("completion_manifest.json")):
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("version_label") != "V4.2":
            continue
        summary_path = completion_path.parent / "stage_summary.csv"
        if not summary_path.is_file():
            raise ValueError(f"compatible completion lacks stage summary: {completion_path}")
        frame = pd.read_csv(summary_path, usecols=["combo_id"])
        stage_ids = set(frame.combo_id.astype(str))
        ids.update(stage_ids)
        sources.append({
            "campaign_id": payload.get("campaign_id"), "stage_id": payload.get("stage_id"),
            "coordinate_count": len(stage_ids), "completion_manifest": artifact(completion_path),
            "stage_summary": artifact(summary_path),
        })
    return ids, sources


def sparse_incomplete_ids() -> tuple[set[str], dict[str, Any]]:
    payload = json.loads(SPARSE_PLAN_PATH.read_text(encoding="utf-8"))
    ids = {combo_id_from_record(row) for row in payload.get("explicit_combos", [])}
    output = Path(str(payload["planned_output_root"]))
    completion = output / "completion_manifest.json"
    if completion.is_file():
        raise ValueError("the separate sparse stage is no longer incomplete")
    return ids, {
        "campaign_id": payload.get("campaign_id"), "stage_id": payload.get("stage_id"),
        "coordinate_count": len(ids), "plan": artifact(SPARSE_PLAN_PATH),
        "completion_manifest_present": False,
    }


def union_evidence() -> tuple[dict[str, Any], Path]:
    pointer = json.loads(CURRENT_UNION_POINTER.read_text(encoding="utf-8"))
    if pointer.get("status") != "complete" or pointer.get("union_snapshot_id") != CURRENT_UNION_SNAPSHOT_ID:
        raise ValueError("stable V4.2 union pointer drifted from the frozen design")
    analysis_path = Path(str(pointer["snapshot_root"])) / "analysis_manifest.json"
    if sha256_file(analysis_path) != CURRENT_UNION_ANALYSIS_SHA256:
        raise ValueError("stable V4.2 union analysis hash drifted from the frozen design")
    manifest = json.loads(analysis_path.read_text(encoding="utf-8"))
    if int(manifest.get("coordinate_count", -1)) != CURRENT_COMPLETED_COUNT:
        raise ValueError("stable V4.2 union coordinate count changed")
    return manifest, analysis_path


def validate_live_boundaries() -> dict[str, Any]:
    if sha256_file(DESIGN_MEMO) != DESIGN_MEMO_SHA256:
        raise ValueError("frozen design memo hash changed")
    if sha256_file(SOURCE_DEFAULT) != SOURCE_SHA256:
        raise ValueError("V4.2 source hash changed")
    preparation = json.loads(DATA_PREPARATION_MANIFEST_DEFAULT.read_text(encoding="utf-8"))
    if preparation.get("status") != "complete" or preparation.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("V4.2 preparation identity changed")
    scenario = load_scenario_contract(SCENARIO_DEFINITION.resolve())
    if scenario["scenario_schema_id"] != COMBINED_SCENARIO_SCHEMA_ID:
        raise ValueError("scenario contract changed")
    union_manifest, analysis_path = union_evidence()
    completed_ids, completed_sources = completed_v4_2_ids()
    sparse_ids, sparse_source = sparse_incomplete_ids()
    proposed_ids = set(EXPECTED_COMBO_IDS)
    if len(completed_ids) != CURRENT_COMPLETED_COUNT:
        raise ValueError("live compatible completion population is not 1,470")
    if len(sparse_ids) != SPARSE_INCOMPLETE_COUNT:
        raise ValueError("incomplete sparse-plan population is not 24")
    completed_overlap = proposed_ids & completed_ids
    sparse_overlap = proposed_ids & sparse_ids
    if completed_overlap or sparse_overlap:
        raise ValueError("proposed coordinates overlap a completed or pending coordinate boundary")
    return {
        "current_union_manifest": union_manifest,
        "current_union_analysis": artifact(analysis_path),
        "completed_ids": completed_ids,
        "completed_sources": completed_sources,
        "sparse_ids": sparse_ids,
        "sparse_source": sparse_source,
        "completed_overlap": sorted(completed_overlap),
        "sparse_overlap": sorted(sparse_overlap),
    }


def plan_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    rows = stage_grid()
    explicit = [row for _, row in rows]
    return {
        "schema_version": 3, "status": "approved_for_execution",
        "campaign_id": CAMPAIGN_ID, "stage_id": STAGE_ID,
        "stage_kind": "high_return_total_average_exit_axis_probe",
        "predecessor_stage_ids": ["s01_short_e_entry_reachability", "s02_long_e_preperiod_lifecycle"],
        "selection_provenance": "Frozen 2026-08-01 V4.2 high-total/high-average design; fresh V4.2 computation only.",
        "source": str(SOURCE_DEFAULT.resolve()),
        "data_preparation_manifest": str(DATA_PREPARATION_MANIFEST_DEFAULT.resolve()),
        "scenario_definition": str(SCENARIO_DEFINITION.resolve()),
        "train_start": TRAIN_START, "train_end": TRAIN_END,
        "entry_fill_mode": ENTRY_FILL_CALCULATED_THRESHOLD,
        "entry_execution_policy": ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        "entry_slippage": 0.0, "exit_mode": "combined",
        "resources": {"workers": 3, "batch_size": 12, "minimum_free_memory_mb": 4096},
        "planned_output_root": str(OUTPUT_ROOT.resolve()),
        "execution_gate": "A new explicit user start instruction is required; plan materialization and validate-only do not authorize compute.",
        "experiment_question": "Can one-axis M/W/S leaps improve compounded total return or average trade return around their separate leaders while preserving the current method and diagnostics?",
        "objective_contract": {
            "views": [
                {"view_id": "min10_total_return", "minimum_trade_count": 10, "metric": "train_return", "direction": "descending"},
                {"view_id": "min10_average_return", "minimum_trade_count": 10, "metric": "train_avg_trade", "direction": "descending"},
                {"view_id": "min20_total_return", "minimum_trade_count": 20, "metric": "train_return", "direction": "descending"},
                {"view_id": "min20_average_return", "minimum_trade_count": 20, "metric": "train_avg_trade", "direction": "descending"},
            ],
            "exact_metric_tie_breaker": "combo_id", "single_composite_score": False,
            "scenario_3_is_separate_promotion_gate": True,
            "parameter_acceptance": "none_from_in_sample_leap_round",
        },
        "dimension_contract": {
            "anchor_combo_ids": [item["combo_id"] for item in ANCHORS],
            "m_values": list(M_VALUES), "w_values": list(W_VALUES), "speed_window_values": list(S_VALUES),
            "one_changed_exit_axis_per_coordinate": True, "entry_dimensions_frozen_per_anchor": True,
            "raw_coordinate_count": 24, "execution_coordinate_count": 24,
            "memo_combo_ids_in_order": list(EXPECTED_COMBO_IDS),
            "batch_partition_in_plan_order": [list(EXPECTED_COMBO_IDS[:12]), list(EXPECTED_COMBO_IDS[12:])],
            "all_other_method_data_scenario_template_and_trading_rules_unchanged": True,
        },
        "round_progression_contract": {
            "coordinate_budget": {"round_1": 24, "round_2_maximum": 12, "round_3_maximum": 12, "entire_direction_maximum": 48},
            "round_2_gate": "At most one same-axis branch per independent view and at most three fresh neighbors per surviving view; deduplicate coincident min10/min20 branches.",
            "round_3_gate": "Only after a fresh Round-2 neighbor reproduces the improvement; at most 12 nearest-neighbor or cross-anchor robustness rows and no new axis.",
        },
        "continuation_gates": {
            "total_return_anchor": ANCHORS[0]["anchor_metric"],
            "average_trade_return_anchor": ANCHORS[1]["anchor_metric"],
            "hard_guards": ["finite metrics", "positive total return", "positive gap-excluded return", "ordered same-axis support across baseline and at least two probes"],
        },
        "stop_conditions": [
            "Stop before execution on snapshot, source, plan, hash, identity, lock, output-isolation, resource, duplicate, or anti-join drift.",
            "Close all 24 coordinates and immutable stage/cumulative delivery before interpretation.",
            "Stop each view when no eligible row beats its exact objective anchor.",
            "Stop on nonfinite evidence, nonpositive gap-excluded return, incomplete closure, lifecycle mismatch, or fixed-template failure.",
            "Stop the direction when all four views stop; Round 2 requires ordered support and Round 3 requires reproduced improvement.",
        ],
        "delivery_contract": {
            "four_independent_analysis_views_required": True,
            "scenario_3_separate": True, "fixed_historical_templates_required": True,
            "stage_and_cumulative_delivery_required_after_completion": True,
            "trade_review_workers": 4,
        },
        "pre_execution_evidence": {
            "design_memo": artifact(DESIGN_MEMO), "source_manifest": artifact(SOURCE_MANIFEST),
            "current_union_pointer": artifact(CURRENT_UNION_POINTER),
            "current_union_analysis": evidence["current_union_analysis"],
        },
        "anti_join": {
            "completed_compatible_coordinate_count": len(evidence["completed_ids"]),
            "incomplete_sparse_plan_coordinate_count": len(evidence["sparse_ids"]),
            "proposed_coordinate_count": len(EXPECTED_COMBO_IDS),
            "proposed_unique_coordinate_count": len(set(EXPECTED_COMBO_IDS)),
            "proposed_completed_overlap_count": len(evidence["completed_overlap"]),
            "proposed_incomplete_sparse_overlap_count": len(evidence["sparse_overlap"]),
            "completed_sources": evidence["completed_sources"],
            "incomplete_sparse_source": evidence["sparse_source"],
        },
        "grid_blocks": [], "explicit_combos": explicit,
    }


def build_plan() -> dict[str, Any]:
    evidence = validate_live_boundaries()
    payload = plan_payload(evidence)
    atomic_json(PLAN_PATH, payload)
    ids = tuple(combo_id_from_record(row) for row in payload["explicit_combos"])
    checks = {
        "frozen_memo_sha256": sha256_file(DESIGN_MEMO) == DESIGN_MEMO_SHA256,
        "exact_24_combo_ids_and_order": ids == EXPECTED_COMBO_IDS,
        "two_12_coordinate_plan_order_batches": len(ids[:12]) == len(ids[12:]) == 12,
        "completed_compatible_count_1470": len(evidence["completed_ids"]) == 1470,
        "incomplete_sparse_count_24": len(evidence["sparse_ids"]) == 24,
        "completed_overlap_zero": not evidence["completed_overlap"],
        "incomplete_sparse_overlap_zero": not evidence["sparse_overlap"],
        "four_independent_analysis_views": len(payload["objective_contract"]["views"]) == 4,
        "resources_3_12_4096": payload["resources"] == {"workers": 3, "batch_size": 12, "minimum_free_memory_mb": 4096},
        "explicit_start_gate_retained": "explicit user start" in payload["execution_gate"],
    }
    if not all(checks.values()):
        raise ValueError(f"high-return plan audit failed: {checks}")
    audit = {
        "schema_version": 1, "status": "passed", "plan_id": f"{CAMPAIGN_ID}_{STAGE_ID}",
        "checks": checks, "plan": artifact(PLAN_PATH), "planner": artifact(Path(__file__)),
        "design_memo": artifact(DESIGN_MEMO), "source_manifest": artifact(SOURCE_MANIFEST),
        "source": artifact(SOURCE_DEFAULT), "data_preparation_manifest": artifact(DATA_PREPARATION_MANIFEST_DEFAULT),
        "scenario_definition": artifact(SCENARIO_DEFINITION), "current_union_analysis": evidence["current_union_analysis"],
        "anti_join": payload["anti_join"], "planned_output_root": str(OUTPUT_ROOT.resolve()),
        "coordinate_count": 24, "batch_count": 2,
    }
    atomic_json(AUDIT_PATH, audit)
    return {
        "status": "passed", "campaign_id": CAMPAIGN_ID, "stage_id": STAGE_ID,
        "coordinate_count": 24, "batch_count": 2, "plan": artifact(PLAN_PATH),
        "audit": artifact(AUDIT_PATH), "output": str(OUTPUT_ROOT.resolve()),
    }


def main() -> None:
    print(json.dumps(build_plan(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
