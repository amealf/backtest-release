"""Build the audited two-stage all-window rolling V4.2 Scenario-3 grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
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
CAMPAIGN_ID = "v4_2_scenario_3_all_window_rolling_20260801"
SCENARIO_DEFINITION = (
    PLANS_ROOT / "v4_2_scenario_groups_single_select_combined_exit_20260801.json"
)
PLAN_SPECS = (
    {
        "stage_id": "s01_short_e_entry_reachability",
        "e_values": (80, 120, 160, 200),
        "expected_count": 840,
        "design": "all_window_short_e_entry_reachability",
    },
    {
        "stage_id": "s02_long_e_preperiod_lifecycle",
        "e_values": (240, 280, 320),
        "expected_count": 630,
        "design": "all_window_long_e_preperiod_lifecycle",
    },
)
BH_VALUES = (120, 180, 240, 300, 360)
EFFECTIVE_THRESHOLDS = (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
ROLLING_TRW_VALUES = (6, 8, 10, 12, 16, 20)
SPEED_WINDOW_BARS = 480
W_BARS = 6
M_MULTIPLIER = 4.0
REVIEW_WORKERS = 4

V4_1_REBOUND_ANALYSIS = Path(
    r"F:\Backtest test 6.11\K200_short_momentum_dual_exit\combined_exit\campaigns"
    r"\v4_1_scenario_3_rebound_lifecycle_20260731"
    r"\s01_pre_gap_rebound_exit_screen\analysis\analysis_manifest.json"
)
V4_1_FINAL_ANALYSIS = Path(
    r"F:\Backtest test 6.11\K200_short_momentum_dual_exit\combined_exit\campaigns"
    r"\v4_1_scenario_3_pre_gap_clear_20260731"
    r"\s01_final_pre_gap_position_clearance_screen\analysis\analysis_manifest.json"
)
TRADE_DESIGN_SOURCE = Path(
    r"F:\Backtest test 6.11\K200_short_momentum_dual_exit\rebound_only\campaigns"
    r"\rbw13_v4_extreme_activity_filtered_20260729\all_completed_union_analysis"
    r"\trade_explain\index.html"
)
TRADE_DESIGN_SOURCE_SHA256 = (
    "c5c4964aa8ec14478a4ed60adb8e094d4890cba149006e3c454a55bf5ac7b146"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validated_csv(record: dict[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != str(record.get("sha256", "")):
        raise ValueError(f"{label} failed hash validation")
    return path


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token not in {"true", "false"}:
        raise ValueError(f"invalid boolean evidence: {value!r}")
    return token == "true"


def _load_v4_1_endpoint(
    manifest_path: Path,
    *,
    expected_coordinates: int,
    expected_trades: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("coordinate_count", -1)) != expected_coordinates
        or int(manifest.get("trade_count", -1)) != expected_trades
    ):
        raise ValueError(f"V4.1 endpoint closure mismatch: {manifest_path}")
    summary = pd.read_csv(
        _validated_csv(manifest["source_artifacts"]["stage_summary"], "stage summary")
    )
    segments = pd.read_csv(
        _validated_csv(
            manifest["source_artifacts"]["stage_segment_qualification"],
            "segment qualification",
        )
    )
    return manifest, summary, segments


def _segment_record(segments: pd.DataFrame, combo_id: str) -> dict[str, Any]:
    rows = segments.loc[segments.combo_id.astype(str).eq(combo_id)].copy()
    if len(rows) != 3 or rows.segment_id.astype(str).duplicated().any():
        raise ValueError(f"segment endpoint closure is incomplete for {combo_id}")
    return {
        str(row.segment_id): {
            "qualified": _truthy(row.qualified),
            "entry_count": int(row.entry_count_in_interval),
            "exit_count": int(row.exit_count_in_interval),
            "holds_past_end": _truthy(row.holds_past_segment_end),
        }
        for row in rows.itertuples(index=False)
    }


def _endpoint_evidence() -> dict[str, Any]:
    rebound_manifest, rebound_summary, rebound_segments = _load_v4_1_endpoint(
        V4_1_REBOUND_ANALYSIS, expected_coordinates=29, expected_trades=3603
    )
    final_manifest, final_summary, final_segments = _load_v4_1_endpoint(
        V4_1_FINAL_ANALYSIS, expected_coordinates=36, expected_trades=11624
    )
    lower = rebound_summary.loc[
        pd.to_numeric(rebound_summary.e).eq(120)
        & pd.to_numeric(rebound_summary.bh).eq(300)
        & pd.to_numeric(rebound_summary.w).eq(W_BARS)
        & pd.to_numeric(rebound_summary.m).eq(M_MULTIPLIER)
    ].copy()
    lower = lower.loc[lower.method.astype(str).eq("rolling_tr_sum")].copy()
    if len(lower) != 1:
        raise ValueError("V4.1 rolling lower endpoint is not unique")
    lower_rows = []
    for row in lower.itertuples(index=False):
        segments = _segment_record(rebound_segments, str(row.combo_id))
        if not (
            segments["market_01"]["qualified"]
            and segments["market_02"]["qualified"]
            and not segments["market_03"]["qualified"]
            and segments["market_03"]["entry_count"] == 0
        ):
            raise ValueError("V4.1 lower endpoint lifecycle changed")
        lower_rows.append(
            {
                "combo_id": str(row.combo_id),
                "method": str(row.method),
                "segments": segments,
            }
        )

    upper = final_summary.loc[
        pd.to_numeric(final_summary.e).eq(280)
        & pd.to_numeric(final_summary.w).eq(W_BARS)
        & pd.to_numeric(final_summary.m).eq(M_MULTIPLIER)
    ].copy()
    upper_rows = []
    for row in upper.itertuples(index=False):
        segments = _segment_record(final_segments, str(row.combo_id))
        if segments["market_03"]["qualified"]:
            if not (
                not segments["market_01"]["qualified"]
                and not segments["market_02"]["qualified"]
                and segments["market_01"]["entry_count"] == 3
                and segments["market_02"]["entry_count"] == 2
            ):
                raise ValueError("V4.1 upper endpoint lifecycle changed")
            upper_rows.append(
                {
                    "combo_id": str(row.combo_id),
                    "method": str(row.method),
                    "segments": segments,
                }
            )
    upper_rows = [row for row in upper_rows if row["method"] == "rolling_tr_sum"]
    if len(upper_rows) != 1:
        raise ValueError("V4.1 upper endpoint lacks one rolling market-03 row")
    return {
        "evidence_role": "historical_selection_provenance_only",
        "v4_1_rebound_analysis": artifact(V4_1_REBOUND_ANALYSIS),
        "v4_1_final_analysis": artifact(V4_1_FINAL_ANALYSIS),
        "lower_endpoint": lower_rows,
        "upper_endpoint": upper_rows,
        "v4_1_rows_may_enter_v4_2_rankings": False,
        "v4_2_recomputation_required": True,
    }


def _completed_v4_2_ids() -> tuple[set[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    sources: list[dict[str, Any]] = []
    if not CAMPAIGNS_ROOT.is_dir():
        return ids, sources
    for completion_path in CAMPAIGNS_ROOT.rglob("completion_manifest.json"):
        if "v4_2_all_completed_union_analysis" in completion_path.parts:
            continue
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("version_label") != "V4.2":
            continue
        stage = completion_path.parent
        summary_path = stage / "stage_summary.csv"
        if not summary_path.is_file():
            continue
        frame = pd.read_csv(summary_path, usecols=["combo_id"])
        stage_ids = set(frame.combo_id.astype(str))
        ids.update(stage_ids)
        sources.append(
            {
                "campaign_id": payload.get("campaign_id"),
                "stage_id": payload.get("stage_id"),
                "completion_manifest": artifact(completion_path),
                "coordinate_count": len(stage_ids),
            }
        )
    return ids, sources


def _combo_record(
    method: str,
    e: int,
    bh: int,
    trw: int,
    threshold: float,
    design: str,
) -> tuple[str, dict[str, Any]]:
    if method != "rolling_tr_sum":
        raise ValueError("current V4.2 plan permits rolling_tr_sum only")
    k = float(threshold) / int(trw)
    combo = Combo(
        method=method,
        e=int(e),
        bh=int(bh),
        trw=int(trw),
        k=k,
        w=W_BARS,
        m=M_MULTIPLIER,
        entry_fill_mode=ENTRY_FILL_CALCULATED_THRESHOLD,
        entry_execution_policy=ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        speed_window_bars=SPEED_WINDOW_BARS,
    )
    return combo.combo_id, {
        "method": method,
        "e": int(e),
        "bh": int(bh),
        "trw": int(trw),
        "k": k,
        "w": W_BARS,
        "m": M_MULTIPLIER,
        "speed_window_bars": SPEED_WINDOW_BARS,
        "entry_fill_mode": ENTRY_FILL_CALCULATED_THRESHOLD,
        "entry_execution_policy": ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        "entry_slippage": 0.0,
        "effective_threshold": float(threshold),
        "seed": "v4_1_scenario_3_lifecycle_endpoints",
        "objective": "scenario_3_total_return",
        "design": design,
    }


def _stage_grid(spec: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for e in spec["e_values"]:
        for bh in BH_VALUES:
            for threshold in EFFECTIVE_THRESHOLDS:
                for trw in ROLLING_TRW_VALUES:
                    rows.append(
                        _combo_record(
                            "rolling_tr_sum", e, bh, trw, threshold, spec["design"]
                        )
                    )
    return rows


def _preparation_evidence() -> dict[str, Any]:
    path = DATA_PREPARATION_MANIFEST_DEFAULT.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "complete"
        or not str(payload.get("prepared_identity", "")).startswith(
            "v4_2_static_baseline_"
        )
        or str(payload.get("source_sha256", "")) != SOURCE_SHA256
    ):
        raise ValueError("V4.2 preparation identity is not closed")
    return artifact(path)


def build_plans() -> dict[str, Any]:
    existing_pairs = [
        (
            PLANS_ROOT / f"{CAMPAIGN_ID}_{spec['stage_id']}.json",
            PLANS_ROOT / f"{CAMPAIGN_ID}_{spec['stage_id']}.audit.json",
            spec,
        )
        for spec in PLAN_SPECS
    ]
    existing_flags = [plan.is_file() and audit.is_file() for plan, audit, _ in existing_pairs]
    if any(existing_flags):
        if not all(existing_flags):
            raise ValueError("V4.2 frozen plan set is partially materialized")
        retained: list[dict[str, Any]] = []
        retained_ids: set[str] = set()
        for plan_path, audit_path, spec in existing_pairs:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if (
                plan.get("status") != "approved_for_execution"
                or plan.get("campaign_id") != CAMPAIGN_ID
                or plan.get("stage_id") != spec["stage_id"]
                or plan.get("entry_fill_mode") != ENTRY_FILL_CALCULATED_THRESHOLD
                or audit.get("status") != "passed"
                or audit.get("plan", {}).get("sha256") != sha256_file(plan_path)
            ):
                raise ValueError(f"frozen V4.2 plan identity drift: {plan_path}")
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
                for row in plan["explicit_combos"]
            }
            if (
                len(combo_ids) != int(spec["expected_count"])
                or retained_ids.intersection(combo_ids)
            ):
                raise ValueError(f"frozen V4.2 plan population drift: {plan_path}")
            retained_ids.update(combo_ids)
            retained.append(
                {
                    "stage_id": spec["stage_id"],
                    "plan": artifact(plan_path),
                    "audit": artifact(audit_path),
                    "output": plan["planned_output_root"],
                    "coordinate_count": len(combo_ids),
                }
            )
        if len(retained_ids) != 1470:
            raise ValueError("frozen V4.2 plans no longer close 1470 coordinates")
        return {
            "status": "passed",
            "campaign_id": CAMPAIGN_ID,
            "stage_count": len(retained),
            "coordinate_count": len(retained_ids),
            "stages": retained,
            "reused_frozen_plans": True,
        }

    scenario = load_scenario_contract(SCENARIO_DEFINITION.resolve())
    if scenario["scenario_schema_id"] != COMBINED_SCENARIO_SCHEMA_ID:
        raise ValueError("V4.2 combined Scenario-3 contract mismatch")
    if sha256_file(SOURCE_DEFAULT) != SOURCE_SHA256:
        raise ValueError("V4.2 source hash mismatch")
    if (
        not TRADE_DESIGN_SOURCE.is_file()
        or sha256_file(TRADE_DESIGN_SOURCE) != TRADE_DESIGN_SOURCE_SHA256
    ):
        raise ValueError("historical V4 trade template changed")
    preparation = _preparation_evidence()
    endpoints = _endpoint_evidence()
    completed_ids, completed_sources = _completed_v4_2_ids()
    planned_ids: set[str] = set()
    written: list[dict[str, Any]] = []
    for index, spec in enumerate(PLAN_SPECS, start=1):
        raw = _stage_grid(spec)
        raw_ids = [combo_id for combo_id, _ in raw]
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError(f"{spec['stage_id']} contains duplicate coordinates")
        overlap = set(raw_ids).intersection(completed_ids | planned_ids)
        explicit = [row for combo_id, row in raw if combo_id not in overlap]
        execution_ids = {
            combo_id for combo_id, _ in raw if combo_id not in overlap
        }
        if len(raw) != int(spec["expected_count"]):
            raise ValueError(f"{spec['stage_id']} raw count changed")
        if len(explicit) != len(execution_ids):
            raise ValueError(f"{spec['stage_id']} execution identity closure failed")
        output = CAMPAIGNS_ROOT / CAMPAIGN_ID / str(spec["stage_id"])
        plan_path = PLANS_ROOT / f"{CAMPAIGN_ID}_{spec['stage_id']}.json"
        audit_path = plan_path.with_suffix(".audit.json")
        plan = {
            "schema_version": 3,
            "status": "approved_for_execution",
            "campaign_id": CAMPAIGN_ID,
            "stage_id": spec["stage_id"],
            "stage_kind": "scenario_3_all_window_rolling_entry_bridge",
            "predecessor_stage_ids": (
                [] if index == 1 else [PLAN_SPECS[index - 2]["stage_id"]]
            ),
            "selection_provenance": (
                "V4.1 next-open evidence separates E120 rows retaining markets 1/2 "
                "from E280 rows qualifying market 03. V4.2 recomputes a broad rolling-TR "
                "grid with every finite TR15 atom inside BH/TRW. Filter markers remain "
                "audit evidence only; V4.1 metrics never enter V4.2 ranking."
            ),
            "source": str(SOURCE_DEFAULT.resolve()),
            "data_preparation_manifest": str(
                DATA_PREPARATION_MANIFEST_DEFAULT.resolve()
            ),
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
            "planned_output_root": str(output.resolve()),
            "objective_contract": {
                "required_scenario_id": "scenario_3",
                "primary_metric": "train_return",
                "direction": "descending",
                "method_scope": "rolling_tr_sum_only",
                "methods_ranked_separately": False,
                "single_composite_score": False,
                "diagnostic_candidate_gate": (
                    "Scenario 3 qualified with positive total and gap-excluded return; "
                    "another audited validation is still required."
                ),
                "parameter_acceptance": "none_from_in_sample_stage",
            },
            "experiment_question": (
                "Can an all-window rolling-TR baseline plus signal-bar calculated "
                "execution price at fixed S480/W6/M4 "
                f"and E={list(spec['e_values'])} produce exactly one entry, zero "
                "interval exits, and a hold past the end in all three Scenario-3 markets?"
            ),
            "dimension_contract": {
                "fixed_entry_fill_mode": ENTRY_FILL_CALCULATED_THRESHOLD,
                "fixed_speed_window_bars": SPEED_WINDOW_BARS,
                "fixed_w": W_BARS,
                "fixed_m": M_MULTIPLIER,
                "e_values": list(spec["e_values"]),
                "bh_values": list(BH_VALUES),
                "effective_threshold_values": list(EFFECTIVE_THRESHOLDS),
                "rolling_trw_values": list(ROLLING_TRW_VALUES),
                "raw_coordinate_count": len(raw),
                "completed_or_planned_overlap_removed": len(overlap),
                "execution_coordinate_count": len(explicit),
                "entry_baseline_uses_all_finite_window_tr15": True,
                "filter_marker_is_audit_only": True,
                "all_other_trading_rules_unchanged": True,
            },
            "stop_conditions": [
                "Stop before execution on any source, preparation, scenario, endpoint, template, identity, or plan-hash mismatch.",
                "Stop interpretation if any filter marker changes BH/TRW, H, signal, fill, exit, continuity, or chart OHLC.",
                "Stop interpretation if any real signal-bar calculated fill differs from min(open, H-K*baseline), or any waited fill differs from the next real-trade open within 120 bars.",
                "Stop interpretation if any speed exit differs from the zero-extension current-close contract.",
                "Close after every anti-joined coordinate in this frozen stage completes; never accept a parameter from this in-sample stage.",
                "Reject rows with nonpositive total or gap-excluded return; only rolling_tr_sum is in scope.",
                "If both stages contain zero Scenario-3 rows, stop this entry-baseline slice and diagnose segment/lifecycle failures before any expansion.",
            ],
            "delivery_contract": {
                "runner": "code/run_v4_2_resumable_campaign.py",
                "delivery_worker": "code/run_v4_2_delivery_worker.py",
                "stage_analyzer": "code/analyze_v4_2_scenario_3_stage.py",
                "trade_review_generator": "code/build_v4_2_review_delivery.py",
                "cumulative_union_builder": "code/build_v4_2_combined_union_analysis.py",
                "browser_qa": "code/qa_v4_2_scenario_3_stage.mjs",
                "delivery_mode": "background",
                "trade_review_workers": REVIEW_WORKERS,
                "next_stage_may_compute_during_delivery": True,
                "main_entry_html_required": True,
                "trade_level_html_required": True,
                "cumulative_main_and_trade_refresh_required": True,
                "historical_v4_trade_design_source": str(
                    TRADE_DESIGN_SOURCE.resolve()
                ),
                "historical_v4_trade_design_sha256": TRADE_DESIGN_SOURCE_SHA256,
                "required_artifacts": [
                    "plan",
                    "plan_audit",
                    "grid_manifest",
                    "progress",
                    "batch_manifests",
                    "stage_summary",
                    "stage_trades",
                    "stage_segment_qualification",
                    "stage_scenario_qualification",
                    "completion_manifest",
                    "delivery_job",
                    "delivery_status",
                    "analysis_manifest",
                    "trade_review_manifest",
                    "main_entry_html",
                    "trade_level_html",
                    "cumulative_completion_manifest",
                ],
            },
            "concurrency_contract": {
                "compute_lock": ".v4_2_runner.lock per stage output",
                "delivery_lock": ".v4_2_delivery.lock per completed stage",
                "union_lock": ".v4_2_union.lock per cumulative output",
                "immutable_handoff": "completion_manifest plus hash-bound stage artifacts",
                "distinct_stage_compute_and_prior_delivery_may_overlap": True,
                "same_output_has_one_writer_per_phase": True,
            },
            "pre_execution_evidence": endpoints,
            "anti_join": {
                "completed_v4_2_coordinate_count": len(completed_ids),
                "prior_frozen_stage_coordinate_count": len(planned_ids),
                "removed_coordinate_count": len(overlap),
                "removed_combo_ids": sorted(overlap),
                "completed_sources": completed_sources,
            },
            "grid_blocks": [],
            "explicit_combos": explicit,
        }
        checks = {
            "scenario_schema_current": scenario["scenario_schema_id"]
            == COMBINED_SCENARIO_SCHEMA_ID,
            "source_hash_current": sha256_file(SOURCE_DEFAULT) == SOURCE_SHA256,
            "v4_2_preparation_closed": bool(preparation),
            "calculated_entry_only": plan["entry_fill_mode"]
            == ENTRY_FILL_CALCULATED_THRESHOLD,
            "raw_coordinate_count_expected": len(raw) == spec["expected_count"],
            "execution_ids_unique": len(execution_ids) == len(explicit),
            "execution_ids_anti_joined": execution_ids.isdisjoint(
                completed_ids | planned_ids
            ),
            "rolling_only": {row["method"] for row in explicit}
            == {"rolling_tr_sum"},
            "fixed_s480_w6_m4": all(
                row["speed_window_bars"] == 480
                and row["w"] == 6
                and row["m"] == 4.0
                for row in explicit
            ),
            "historical_template_current": sha256_file(TRADE_DESIGN_SOURCE)
            == TRADE_DESIGN_SOURCE_SHA256,
            "background_delivery_recorded": plan["delivery_contract"][
                "delivery_mode"
            ]
            == "background",
            "next_stage_overlap_allowed": plan["delivery_contract"][
                "next_stage_may_compute_during_delivery"
            ]
            is True,
            "single_method_scope": plan["objective_contract"][
                "methods_ranked_separately"
            ]
            is False,
            "composite_score_disabled": plan["objective_contract"][
                "single_composite_score"
            ]
            is False,
        }
        if not all(checks.values()):
            raise ValueError(f"{spec['stage_id']} plan audit failed: {checks}")
        atomic_json(plan_path, plan)
        audit = {
            "schema_version": 1,
            "status": "passed",
            "plan_id": f"{CAMPAIGN_ID}_{spec['stage_id']}",
            "checks": checks,
            "plan": artifact(plan_path),
            "source": artifact(SOURCE_DEFAULT),
            "data_preparation_manifest": preparation,
            "scenario_definition": artifact(SCENARIO_DEFINITION),
            "historical_trade_template": artifact(TRADE_DESIGN_SOURCE),
            "pre_execution_evidence": endpoints,
            "planned_output_root": str(output.resolve()),
        }
        atomic_json(audit_path, audit)
        written.append(
            {
                "stage_id": spec["stage_id"],
                "plan": artifact(plan_path),
                "audit": artifact(audit_path),
                "output": str(output.resolve()),
                "coordinate_count": len(explicit),
            }
        )
        planned_ids.update(execution_ids)
    if len(planned_ids) != 1470:
        raise ValueError("two-stage V4.2 grid must close exactly 1470 unique coordinates")
    return {
        "status": "passed",
        "campaign_id": CAMPAIGN_ID,
        "stage_count": len(written),
        "coordinate_count": len(planned_ids),
        "stages": written,
    }


def main() -> None:
    argparse.ArgumentParser(
        description="Build the audited all-window rolling V4.2 Scenario-3 plans."
    ).parse_args()
    print(json.dumps(build_plans(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
