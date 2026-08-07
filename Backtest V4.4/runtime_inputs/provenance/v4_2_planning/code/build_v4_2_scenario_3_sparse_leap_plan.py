"""Build the audited first sparse exit-axis leap for V4.2 Scenario 3."""
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
PLANS_ROOT = VARIANT_ROOT / "plans"
CAMPAIGNS_ROOT = Path(
    r"F:\Backtest test 6.11\K200_short_momentum_dual_exit"
    r"\v4_2_calculated_entry\campaigns"
)
CAMPAIGN_ID = "v4_2_scenario_3_sparse_leap_20260801"
STAGE_ID = "s01_exit_axis_large_step_probe"
PLAN_PATH = PLANS_ROOT / f"{CAMPAIGN_ID}_{STAGE_ID}.json"
AUDIT_PATH = PLAN_PATH.with_suffix(".audit.json")
OUTPUT_ROOT = CAMPAIGNS_ROOT / CAMPAIGN_ID / STAGE_ID

SCENARIO_DEFINITION = (
    PLANS_ROOT / "v4_2_scenario_groups_single_select_combined_exit_20260801.json"
)
CURRENT_UNION_ANALYSIS = Path(
    r"F:\V4_2_results\all_completed_union_analysis\snapshots"
    r"\d6700c82c5191867f2d9d9b2098f04987c56e575e27c8f7afa92cd50920c8dc7"
    r"\analysis_manifest.json"
)
CURRENT_UNION_ANALYSIS_SHA256 = (
    "4459c0c63cdd31152db87adff9b2a63541dec88af54f44f484485fe9c9d9c2df"
)
CURRENT_UNION_COORDINATES = 1470
CURRENT_UNION_TRADES = 350845

HISTORICAL_V4_1_SPEED_ANALYSIS = Path(
    r"F:\Backtest test 6.11\K200_short_momentum_dual_exit\combined_exit\campaigns"
    r"\v4_1_scenario_3_combined_20260731\s01_speed_window_anchor_screen"
    r"\analysis\analysis_manifest.json"
)
HISTORICAL_V4_1_REBOUND_ANALYSIS = Path(
    r"F:\Backtest test 6.11\K200_short_momentum_dual_exit\combined_exit\campaigns"
    r"\v4_1_scenario_3_rebound_lifecycle_20260731\s01_pre_gap_rebound_exit_screen"
    r"\analysis\analysis_manifest.json"
)
MAIN_DESIGN_SOURCE = Path(
    r"F:\Backtest test 6.11\K200_short_momentum_dual_exit\rebound_only\campaigns"
    r"\rbw13_v4_extreme_activity_filtered_20260729\all_completed_union_analysis"
    r"\analysis_report.html"
)
MAIN_DESIGN_SOURCE_SHA256 = (
    "ef7ea69d9648d6fc84511f9753e7e2c07f36e73272b5e15b7ef606d7db274a72"
)
TRADE_DESIGN_SOURCE = Path(
    r"F:\Backtest test 6.11\K200_short_momentum_dual_exit\rebound_only\campaigns"
    r"\rbw13_v4_extreme_activity_filtered_20260729\all_completed_union_analysis"
    r"\trade_explain\index.html"
)
TRADE_DESIGN_SOURCE_SHA256 = (
    "c5c4964aa8ec14478a4ed60adb8e094d4890cba149006e3c454a55bf5ac7b146"
)

ANCHORS = (
    {
        "anchor_id": "a_strongest_two_of_three",
        "combo_id": (
            "v4_2_rolling_tr_sum_fillcalculated_threshold_"
            "execwait_next_real_trade_slip0_sx1_s480_rx1_"
            "e120_bh360_trw20_k1_w6_m4_431461d58a"
        ),
        "e": 120,
        "bh": 360,
        "trw": 20,
        "k": 1.0,
        "w": 6,
        "m": 4.0,
        "speed_window_bars": 480,
    },
    {
        "anchor_id": "b_distinct_baseline_geometry",
        "combo_id": (
            "v4_2_rolling_tr_sum_fillcalculated_threshold_"
            "execwait_next_real_trade_slip0_sx1_s480_rx1_"
            "e120_bh300_trw8_k2p5_w6_m4_ec8fdb390d"
        ),
        "e": 120,
        "bh": 300,
        "trw": 8,
        "k": 2.5,
        "w": 6,
        "m": 4.0,
        "speed_window_bars": 480,
    },
)
M_LEAPS = (0.5, 1.5, 3.0, 8.0)
W_LEAPS = (2, 3, 12, 24)
S_LEAPS = (80, 160, 320, 960)
EXPECTED_COORDINATE_COUNT = len(ANCHORS) * (
    len(M_LEAPS) + len(W_LEAPS) + len(S_LEAPS)
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validated_artifact(record: dict[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != str(record.get("sha256", "")):
        raise ValueError(f"{label} failed hash validation")
    return path


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token not in {"true", "false"}:
        raise ValueError(f"invalid boolean evidence: {value!r}")
    return token == "true"


def completed_v4_2_ids() -> tuple[set[str], list[dict[str, Any]]]:
    combo_ids: set[str] = set()
    sources: list[dict[str, Any]] = []
    if not CAMPAIGNS_ROOT.is_dir():
        return combo_ids, sources
    for completion_path in sorted(CAMPAIGNS_ROOT.rglob("completion_manifest.json")):
        stage = completion_path.parent
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("version_label") != "V4.2":
            continue
        summary_path = stage / "stage_summary.csv"
        if not summary_path.is_file():
            continue
        stage_frame = pd.read_csv(summary_path, usecols=["combo_id"])
        stage_ids = set(stage_frame.combo_id.astype(str))
        combo_ids.update(stage_ids)
        sources.append(
            {
                "campaign_id": payload.get("campaign_id"),
                "stage_id": payload.get("stage_id"),
                "coordinate_count": len(stage_ids),
                "completion_manifest": artifact(completion_path),
                "stage_summary": artifact(summary_path),
            }
        )
    return combo_ids, sources


def load_union_evidence() -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    if sha256_file(CURRENT_UNION_ANALYSIS) != CURRENT_UNION_ANALYSIS_SHA256:
        raise ValueError("current V4.2 union analysis identity changed")
    manifest = json.loads(CURRENT_UNION_ANALYSIS.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("coordinate_count", -1)) != CURRENT_UNION_COORDINATES
        or int(manifest.get("trade_count", -1)) != CURRENT_UNION_TRADES
        or int(manifest.get("completed_stage_count", -1)) != 2
        or int(manifest.get("duplicate_coordinate_count", -1)) != 0
        or int(manifest.get("excluded_stage_count", -1)) != 0
        or int(manifest.get("scenario_3_qualified_coordinate_count", -1)) != 0
    ):
        raise ValueError("current V4.2 union closure changed")
    summary_path = validated_artifact(
        manifest["artifacts"]["analysis_summary"], "current union summary"
    )
    summary = pd.read_csv(summary_path)
    if len(summary) != CURRENT_UNION_COORDINATES or summary.combo_id.astype(str).duplicated().any():
        raise ValueError("current V4.2 union summary population changed")

    anchor_evidence: dict[str, Any] = {}
    for anchor in ANCHORS:
        rows = summary.loc[summary.combo_id.astype(str).eq(anchor["combo_id"])]
        if len(rows) != 1:
            raise ValueError(f"anchor is not unique: {anchor['anchor_id']}")
        row = rows.iloc[0]
        checks = {
            "method": str(row.method) == "rolling_tr_sum",
            "e": int(row.e) == anchor["e"],
            "bh": int(row.bh) == anchor["bh"],
            "trw": int(row.trw) == anchor["trw"],
            "k": abs(float(row.k) - anchor["k"]) <= 1e-12,
            "w": int(row.w) == anchor["w"],
            "m": abs(float(row.m) - anchor["m"]) <= 1e-12,
            "speed_window_bars": int(row.speed_window_bars)
            == anchor["speed_window_bars"],
            "scenario_1_qualified": truthy(row.scenario_1_qualified),
            "scenario_2_qualified": truthy(row.scenario_2_qualified),
            "scenario_3_not_qualified": not truthy(row.scenario_3_qualified),
            "qualified_segment_count": int(row.scenario_3_qualified_segment_count) == 2,
            "failed_only_market_03": str(row.scenario_3_failed_segment_ids)
            == "market_03",
        }
        if not all(checks.values()):
            raise ValueError(f"anchor evidence changed: {anchor['anchor_id']} {checks}")
        anchor_evidence[anchor["anchor_id"]] = {
            **anchor,
            "train_return": float(row.train_return),
            "train_return_excluding_gap_spanning_trades": float(
                row.train_return_excluding_gap_spanning_trades
            ),
            "train_avg_trade": float(row.train_avg_trade),
            "train_max_drawdown_abs": float(row.train_max_drawdown_abs),
            "scenario_3_failure": "market_03: zero interval entries plus one interval exit",
            "observed_market_03_carry_exit": "2026-07-01 10:02:00 rebound_threshold",
        }
    return manifest, summary, anchor_evidence


def combo_record(
    anchor: dict[str, Any],
    *,
    leap_axis: str,
    leap_value: float | int,
) -> tuple[str, dict[str, Any]]:
    values = {
        "w": int(anchor["w"]),
        "m": float(anchor["m"]),
        "speed_window_bars": int(anchor["speed_window_bars"]),
    }
    values[leap_axis] = leap_value
    combo = Combo(
        method="rolling_tr_sum",
        e=int(anchor["e"]),
        bh=int(anchor["bh"]),
        trw=int(anchor["trw"]),
        k=float(anchor["k"]),
        w=int(values["w"]),
        m=float(values["m"]),
        entry_fill_mode=ENTRY_FILL_CALCULATED_THRESHOLD,
        entry_execution_policy=ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        speed_window_bars=int(values["speed_window_bars"]),
    )
    return combo.combo_id, {
        "method": combo.method,
        "e": combo.e,
        "bh": combo.bh,
        "trw": combo.trw,
        "k": combo.k,
        "w": combo.w,
        "m": combo.m,
        "speed_window_bars": combo.speed_window_bars,
        "entry_fill_mode": ENTRY_FILL_CALCULATED_THRESHOLD,
        "entry_execution_policy": ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        "entry_slippage": 0.0,
        "seed": anchor["anchor_id"],
        "source_combo_id": anchor["combo_id"],
        "objective": "scenario_3_total_return",
        "design": "sparse_exit_axis_large_step_probe",
        "leap_round": "round_1",
        "leap_axis": leap_axis,
        "leap_value": leap_value,
    }


def stage_grid() -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for anchor in ANCHORS:
        for value in M_LEAPS:
            rows.append(combo_record(anchor, leap_axis="m", leap_value=value))
        for value in W_LEAPS:
            rows.append(combo_record(anchor, leap_axis="w", leap_value=value))
        for value in S_LEAPS:
            rows.append(
                combo_record(
                    anchor,
                    leap_axis="speed_window_bars",
                    leap_value=value,
                )
            )
    return rows


def validate_frozen_plan() -> dict[str, Any]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    combo_ids = {
        Combo(
            method=str(row["method"]),
            e=int(row["e"]),
            bh=int(row["bh"]),
            trw=int(row["trw"]),
            k=float(row["k"]),
            w=int(row["w"]),
            m=float(row["m"]),
            entry_fill_mode=ENTRY_FILL_CALCULATED_THRESHOLD,
            entry_execution_policy=ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
            speed_window_bars=int(row["speed_window_bars"]),
        ).combo_id
        for row in plan.get("explicit_combos", [])
    }
    if (
        plan.get("status") != "approved_for_execution"
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("stage_id") != STAGE_ID
        or len(combo_ids) != EXPECTED_COORDINATE_COUNT
        or audit.get("status") != "passed"
        or audit.get("plan", {}).get("sha256") != sha256_file(PLAN_PATH)
    ):
        raise ValueError("frozen sparse-leap plan identity drift")
    return {
        "status": "passed",
        "campaign_id": CAMPAIGN_ID,
        "stage_id": STAGE_ID,
        "coordinate_count": len(combo_ids),
        "plan": artifact(PLAN_PATH),
        "audit": artifact(AUDIT_PATH),
        "output": str(OUTPUT_ROOT.resolve()),
        "reused_frozen_plan": True,
    }


def build_plan() -> dict[str, Any]:
    existing = (PLAN_PATH.is_file(), AUDIT_PATH.is_file())
    if any(existing):
        if not all(existing):
            raise ValueError("sparse-leap plan set is partially materialized")
        return validate_frozen_plan()

    scenario = load_scenario_contract(SCENARIO_DEFINITION.resolve())
    union_manifest, _, anchor_evidence = load_union_evidence()
    completed_ids, completed_sources = completed_v4_2_ids()
    raw = stage_grid()
    raw_ids = [combo_id for combo_id, _ in raw]
    overlap = set(raw_ids).intersection(completed_ids)
    axis_counts = Counter(str(row["leap_axis"]) for _, row in raw)
    anchor_counts = Counter(str(row["seed"]) for _, row in raw)

    if sha256_file(SOURCE_DEFAULT) != SOURCE_SHA256:
        raise ValueError("V4.2 source hash changed")
    preparation = json.loads(
        DATA_PREPARATION_MANIFEST_DEFAULT.read_text(encoding="utf-8")
    )
    if (
        preparation.get("status") != "complete"
        or str(preparation.get("source_sha256", "")) != SOURCE_SHA256
    ):
        raise ValueError("V4.2 preparation identity changed")
    if len(completed_ids) != CURRENT_UNION_COORDINATES:
        raise ValueError("completed V4.2 anti-join population changed")
    if len(raw_ids) != EXPECTED_COORDINATE_COUNT or len(set(raw_ids)) != len(raw_ids):
        raise ValueError("sparse-leap raw coordinate closure failed")
    if overlap:
        raise ValueError(f"sparse-leap design overlaps completed coordinates: {sorted(overlap)}")
    if scenario["scenario_schema_id"] != COMBINED_SCENARIO_SCHEMA_ID:
        raise ValueError("Scenario-3 definition changed")
    if sha256_file(MAIN_DESIGN_SOURCE) != MAIN_DESIGN_SOURCE_SHA256:
        raise ValueError("historical V4 main template changed")
    if sha256_file(TRADE_DESIGN_SOURCE) != TRADE_DESIGN_SOURCE_SHA256:
        raise ValueError("historical V4 trade template changed")

    explicit = [row for _, row in raw]
    plan = {
        "schema_version": 3,
        "status": "approved_for_execution",
        "campaign_id": CAMPAIGN_ID,
        "stage_id": STAGE_ID,
        "stage_kind": "scenario_3_sparse_exit_axis_leap",
        "predecessor_stage_ids": [
            "s01_short_e_entry_reachability",
            "s02_long_e_preperiod_lifecycle",
        ],
        "selection_provenance": (
            "Fresh V4.2 recomputation from two current all-window rolling rows that "
            "qualify markets 1 and 2 but fail market 03 through the shared 10:02 "
            "rebound exit. Historical V4.1 S and W/M screens set large-step bounds "
            "only; no historical metric enters V4.2 ranking."
        ),
        "source": str(SOURCE_DEFAULT.resolve()),
        "data_preparation_manifest": str(DATA_PREPARATION_MANIFEST_DEFAULT.resolve()),
        "scenario_definition": str(SCENARIO_DEFINITION.resolve()),
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "entry_fill_mode": ENTRY_FILL_CALCULATED_THRESHOLD,
        "entry_execution_policy": ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        "entry_slippage": 0.0,
        "exit_mode": "combined",
        "resources": {
            "workers": 3,
            "batch_size": 12,
            "minimum_free_memory_mb": 4096,
        },
        "planned_output_root": str(OUTPUT_ROOT.resolve()),
        "objective_contract": {
            "required_scenario_id": "scenario_3",
            "primary_metric": "train_return",
            "direction": "descending",
            "method_scope": "rolling_tr_sum_only",
            "single_composite_score": False,
            "diagnostic_candidate_gate": (
                "Scenario 3 qualified with positive total and gap-excluded return; "
                "later audited validation remains required."
            ),
            "parameter_acceptance": "none_from_in_sample_leap_round",
        },
        "experiment_question": (
            "Can a sparse large-step change to exactly one exit axis move the shared "
            "market-03 carry exit from 10:02 to at or before 08:58, preserve markets "
            "1 and 2, and permit exactly one market-03 entry with zero interval exits "
            "and a hold past 11:02?"
        ),
        "dimension_contract": {
            "leap_definition": (
                "two fixed 2/3 entry anchors; change exactly one of M, W, or S by a "
                "widely separated value; no dense local neighborhood"
            ),
            "anchor_combo_ids": [anchor["combo_id"] for anchor in ANCHORS],
            "m_values": list(M_LEAPS),
            "w_values": list(W_LEAPS),
            "speed_window_values": list(S_LEAPS),
            "coordinates_per_anchor": EXPECTED_COORDINATE_COUNT // len(ANCHORS),
            "raw_coordinate_count": len(raw),
            "completed_overlap_removed": 0,
            "execution_coordinate_count": len(explicit),
            "fixed_entry_dimensions": ["E", "BH", "TRW", "K"],
            "entry_baseline_uses_all_finite_window_tr15": True,
            "filter_marker_is_audit_only": True,
            "all_other_trading_rules_unchanged": True,
        },
        "round_progression_contract": {
            "coordinate_budget": {
                "round_1": 24,
                "round_2_maximum": 12,
                "round_3_maximum": 12,
                "entire_direction_maximum": 48,
            },
            "round_1": (
                "Run all 24 one-axis leaps and close fixed-template stage plus cumulative "
                "delivery before interpreting a next jump."
            ),
            "round_2_gate": (
                "Create at most 12 fresh result-led coordinates only after a Scenario-3 "
                "hit, or after a row still qualifying markets 1 and 2 moves the carry "
                "exit to/before 08:58, or to/before 09:30 with a clear same-axis ordering "
                "across at least two values and exactly one remaining market-03 failure. "
                "Follow only the demonstrated branch; do not cross all axes."
            ),
            "round_3_gate": (
                "Create at most 12 robustness coordinates only if Round 2 produces at "
                "least one Scenario-3-qualified row. Replicate or test nearest one-axis "
                "neighbors without adding a new axis or wider excursion."
            ),
            "hit_rule": (
                "If Scenario 3 qualifies, stop leap expansion, rank qualified rows by "
                "total return, and retain them as in-sample diagnostics only."
            ),
            "stop_rule": (
                "Stop when a closed round produces no Scenario-3 row and no directional "
                "lifecycle improvement under its predeclared gate. Round 2 stops if it "
                "produces no Scenario-3 row; a second near miss cannot authorize Round 3."
            ),
        },
        "stop_conditions": [
            "Stop before execution on any source, preparation, scenario, current-union, template, identity, plan-hash, duplicate, or anti-join mismatch.",
            "Stop interpretation if any filter marker changes BH/TRW, H, signal, fill, exit, continuity, return, or chart OHLC.",
            "Stop interpretation if calculated entry, wait-next-real-trade, rebound-before-speed, speed fill, or zero-slippage evidence differs from the current V4.2 contract.",
            "Close all 24 coordinates and the fixed-template cumulative delivery before any Round-2 decision.",
            "Do not infer a dense neighborhood or a multi-axis interaction from a one-axis leap.",
            "Reject nonpositive total or gap-excluded return as a candidate; accept no parameter from this in-sample round.",
            "Stop all successor computation if no axis meets the Round-2 progression gate.",
        ],
        "delivery_contract": {
            "runner": "code/run_v4_2_resumable_campaign.py",
            "delivery_worker": "code/run_v4_2_delivery_worker.py",
            "stage_analyzer": "code/analyze_v4_2_scenario_3_stage.py",
            "trade_review_generator": "code/build_v4_2_review_delivery.py",
            "cumulative_union_builder": "code/build_v4_2_combined_union_analysis.py",
            "browser_qa": "code/qa_v4_2_scenario_3_stage.mjs",
            "delivery_mode": "background_after_immutable_completion",
            "trade_review_workers": 4,
            "main_entry_html_required": True,
            "trade_level_html_required": True,
            "cumulative_main_and_trade_refresh_required": True,
            "historical_v4_main_design_source": str(MAIN_DESIGN_SOURCE.resolve()),
            "historical_v4_main_design_sha256": MAIN_DESIGN_SOURCE_SHA256,
            "historical_v4_trade_design_source": str(TRADE_DESIGN_SOURCE.resolve()),
            "historical_v4_trade_design_sha256": TRADE_DESIGN_SOURCE_SHA256,
        },
        "concurrency_contract": {
            "compute_lock": ".v4_2_runner.lock per stage output",
            "delivery_lock": ".v4_2_delivery.lock per completed stage",
            "union_lock": ".v4_2_union.lock per cumulative output",
            "same_output_has_one_writer_per_phase": True,
            "distinct_next_stage_compute_may_overlap_prior_delivery": True,
        },
        "pre_execution_evidence": {
            "current_union_analysis": artifact(CURRENT_UNION_ANALYSIS),
            "current_union_summary": union_manifest["artifacts"]["analysis_summary"],
            "anchors": anchor_evidence,
            "historical_v4_1_selection_provenance_only": {
                "speed_analysis": artifact(HISTORICAL_V4_1_SPEED_ANALYSIS),
                "rebound_analysis": artifact(HISTORICAL_V4_1_REBOUND_ANALYSIS),
                "historical_rows_may_enter_v4_2_ranking": False,
            },
        },
        "anti_join": {
            "completed_v4_2_coordinate_count": len(completed_ids),
            "removed_coordinate_count": len(overlap),
            "removed_combo_ids": sorted(overlap),
            "completed_sources": completed_sources,
        },
        "grid_blocks": [],
        "explicit_combos": explicit,
    }

    checks = {
        "current_union_hash_closed": sha256_file(CURRENT_UNION_ANALYSIS)
        == CURRENT_UNION_ANALYSIS_SHA256,
        "current_union_population_closed": len(completed_ids)
        == CURRENT_UNION_COORDINATES,
        "scenario_schema_current": scenario["scenario_schema_id"]
        == COMBINED_SCENARIO_SCHEMA_ID,
        "source_hash_current": sha256_file(SOURCE_DEFAULT) == SOURCE_SHA256,
        "preparation_closed": preparation.get("status") == "complete",
        "two_current_two_of_three_anchors": len(anchor_evidence) == 2,
        "raw_count_24": len(raw) == EXPECTED_COORDINATE_COUNT == 24,
        "execution_ids_unique": len(set(raw_ids)) == len(raw_ids),
        "execution_ids_anti_joined": not overlap,
        "twelve_coordinates_per_anchor": set(anchor_counts.values()) == {12},
        "eight_coordinates_per_axis": axis_counts
        == Counter({"m": 8, "w": 8, "speed_window_bars": 8}),
        "rolling_only": {row["method"] for row in explicit} == {"rolling_tr_sum"},
        "calculated_entry_only": all(
            row["entry_fill_mode"] == ENTRY_FILL_CALCULATED_THRESHOLD
            for row in explicit
        ),
        "wait_next_real_trade_only": all(
            row["entry_execution_policy"]
            == ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE
            for row in explicit
        ),
        "zero_slippage": all(float(row["entry_slippage"]) == 0.0 for row in explicit),
        "resources_3_12_4096": plan["resources"]
        == {"workers": 3, "batch_size": 12, "minimum_free_memory_mb": 4096},
        "historical_main_template_current": sha256_file(MAIN_DESIGN_SOURCE)
        == MAIN_DESIGN_SOURCE_SHA256,
        "historical_trade_template_current": sha256_file(TRADE_DESIGN_SOURCE)
        == TRADE_DESIGN_SOURCE_SHA256,
        "composite_score_disabled": plan["objective_contract"][
            "single_composite_score"
        ]
        is False,
        "no_parameter_acceptance": plan["objective_contract"]["parameter_acceptance"]
        == "none_from_in_sample_leap_round",
    }
    if not all(checks.values()):
        raise ValueError(f"sparse-leap plan audit failed: {checks}")

    atomic_json(PLAN_PATH, plan)
    audit = {
        "schema_version": 1,
        "status": "passed",
        "plan_id": f"{CAMPAIGN_ID}_{STAGE_ID}",
        "checks": checks,
        "plan": artifact(PLAN_PATH),
        "planner": artifact(Path(__file__)),
        "source": artifact(SOURCE_DEFAULT),
        "data_preparation_manifest": artifact(DATA_PREPARATION_MANIFEST_DEFAULT),
        "scenario_definition": artifact(SCENARIO_DEFINITION),
        "current_union_analysis": artifact(CURRENT_UNION_ANALYSIS),
        "historical_v4_main_template": artifact(MAIN_DESIGN_SOURCE),
        "historical_v4_trade_template": artifact(TRADE_DESIGN_SOURCE),
        "planned_output_root": str(OUTPUT_ROOT.resolve()),
        "coordinate_count": len(explicit),
        "batch_count": 2,
    }
    atomic_json(AUDIT_PATH, audit)
    return {
        "status": "passed",
        "campaign_id": CAMPAIGN_ID,
        "stage_id": STAGE_ID,
        "coordinate_count": len(explicit),
        "batch_count": 2,
        "plan": artifact(PLAN_PATH),
        "audit": artifact(AUDIT_PATH),
        "output": str(OUTPUT_ROOT.resolve()),
        "reused_frozen_plan": False,
    }


def main() -> None:
    print(json.dumps(build_plan(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
