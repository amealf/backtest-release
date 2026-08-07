"""Validate and deliver one Scenario-3 V4.4 combined-exit stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from instrument_contracts import load_cost_model, sha256_file
from build_v4_4_review_delivery import (
    MAX_W_TRADE_AUDIT_FIELDS,
    build_stage_trade_review,
)
from run_v4_4_resumable_campaign import (
    FINGERPRINT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    result_semantics_id,
    trade_audit_identity,
)
from v4_4_engine import (
    COMBINED_TRADE_AUDIT_SCHEMA_ID,
    COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
    DOWNSIDE_SPEED_EXIT_REASON,
    ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    ENTRY_FILL_CALCULATED_THRESHOLD,
    EXIT_MODE_COMBINED,
    MAX_REAL_TRADE_WAIT_BARS,
    REBOUND_BASELINE_POLICY_ID,
    baseline_filter_id,
    strategy_id,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_TEMPLATE_ROOT = REPOSITORY_ROOT / "runtime_inputs" / "templates"
COST_REFERENCE_PATH = (
    REPOSITORY_ROOT
    / "runtime_inputs"
    / "cost_models"
    / "k200m_current_notional_cost_reference_20260803.json"
)
LEGACY_V4_MAIN_TEMPLATE_PATH = RUNTIME_TEMPLATE_ROOT / "historical_v4_main.html"
LEGACY_V4_MAIN_TEMPLATE_SHA256 = (
    "ef7ea69d9648d6fc84511f9753e7e2c07f36e73272b5e15b7ef606d7db274a72"
)
LEGACY_V4_MAIN_TEMPLATE_ID = "v4_unified_analysis_v3_style_20260729"
LEGACY_V4_TRADE_DESIGN_PATH = RUNTIME_TEMPLATE_ROOT / "historical_v4_trade.html"
LEGACY_V4_TRADE_DESIGN_SHA256 = (
    "9ffc8fd269173a27eae47f21d993c1f43cc296f0b76b14018ff3fb45a9402b50"
)
MARKET_SELECTOR_SOURCE_PATH = RUNTIME_TEMPLATE_ROOT / "market-intuition-selector.html"
MARKET_SELECTOR_SOURCE_SHA256 = (
    "b14e62ff5b15f20c2d1f4533fee858aa4129c8a626d26ac1e6ae4f3d21e4a214"
)
SCENARIO_REQUIREMENTS_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "review_templates"
    / "scenario_requirements_market_selector.html"
)
ANALYSIS_IDENTITY = (
    "v4_4_max_completed_w_drop_dual_baseline_sampling_rolling_"
    "scenario_3_stage_analysis_v8_derived_k200m_notional_cost_ranking"
)
APPROVED_PLAN_STATUSES = (
    "approved_for_execution",
    "approved_for_execution_after_identity_and_plan_audit",
    "approved_for_exact_result_semantics_repair",
)
def _cost_reference_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_k200m_cost_model() -> dict[str, Any]:
    """Backward-compatible K200 entry point backed by the generic cost loader."""
    return load_cost_model(COST_REFERENCE_PATH)


K200M_COST_MODEL = _load_k200m_cost_model()


def _cost_model_from_stage_manifest(stage_manifest: dict[str, Any]) -> dict[str, Any]:
    stage_instrument = stage_manifest.get("instrument_contract")
    if not isinstance(stage_instrument, dict):
        return K200M_COST_MODEL
    cost_path_value = stage_instrument.get("cost_model_path")
    expected_cost_sha = str(stage_instrument.get("cost_model_sha256", ""))
    if cost_path_value in (None, "") or not expected_cost_sha:
        raise ValueError("closed stage instrument contract lacks cost-model identity")
    cost_path = Path(str(cost_path_value)).resolve()
    if not cost_path.is_file() or sha256_file(cost_path) != expected_cost_sha:
        raise ValueError("closed stage cost-model binding failed validation")
    return load_cost_model(cost_path)
COST_ADJUSTED_RETURN_KEY = "train_cost_adjusted_return"
COST_ADJUSTED_AVG_TRADE_KEY = "train_cost_adjusted_avg_trade"
COST_ADJUSTED_MAX_DRAWDOWN_KEY = "train_cost_adjusted_max_drawdown"
MAIN_SUMMARY_ROW_FIELDS = (
    "method",
    "baseline_sampling_policy",
    "e",
    "bh",
    "trw",
    "k",
    "w",
    "m",
    "speed_window_bars",
    "combo_id",
    "train_trade_count",
    "train_return",
    "train_avg_trade",
    "train_max_drawdown_abs",
    "train_cost_adjusted_return",
    "train_cost_adjusted_avg_trade",
    "train_cost_adjusted_max_drawdown_abs",
    "round_trip_cost_bps",
    "train_return_excluding_gap_spanning_trades",
    "gap_spanning_trade_count",
    "rebound_exit_count",
    "speed_exit_count",
    "segment_end_exit_count",
    "waited_entry_count",
    "maximum_entry_wait_bars",
    "holding_bar_distance_median",
    "holding_bar_distance_p95",
    "scenario_1_qualified",
    "scenario_2_qualified",
    "scenario_3_qualified",
    "scenario_3_qualified_segment_count",
    "scenario_3_failed_segment_ids",
)
MAIN_SUMMARY_TOP_LEVEL_FIELDS = (
    "coordinateCount",
    "tradeCount",
    "scenario3QualifiedCount",
    "highReturnViews",
    "costModel",
    "baselineSamplingPolicies",
    "scopeLabel",
    "strategyId",
    "nativeTradeRoute",
    "scenarioRequirementsRoute",
    "templateProvenance",
    "unionSnapshotId",
)
HIGH_RETURN_VIEWS = (
    {
        "id": "scenario_1_qualified_total_return",
        "label": "情景一 · 总收益",
        "scenario_filter": "scenario_1",
        "minimum_trade_count": 0,
        "metric": "total_return",
        "metric_key": COST_ADJUSTED_RETURN_KEY,
        "gross_metric_key": "train_return",
        "cost_adjusted_metric_key": COST_ADJUSTED_RETURN_KEY,
        "display_metric_key": COST_ADJUSTED_RETURN_KEY,
        "order": "descending",
        "tie_break": "combo_id",
    },
    {
        "id": "unrestricted_total_return",
        "label": "全坐标 · 总收益",
        "scenario_filter": "all",
        "minimum_trade_count": 0,
        "metric": "total_return",
        "metric_key": COST_ADJUSTED_RETURN_KEY,
        "gross_metric_key": "train_return",
        "cost_adjusted_metric_key": COST_ADJUSTED_RETURN_KEY,
        "display_metric_key": COST_ADJUSTED_RETURN_KEY,
        "order": "descending",
        "tie_break": "combo_id",
    },
    {
        "id": "unrestricted_average_return_ge10",
        "label": "全坐标 · 至少 10 笔 · 笔均收益",
        "scenario_filter": "all",
        "minimum_trade_count": 10,
        "metric": "average_trade",
        "metric_key": COST_ADJUSTED_AVG_TRADE_KEY,
        "gross_metric_key": "train_avg_trade",
        "cost_adjusted_metric_key": COST_ADJUSTED_AVG_TRADE_KEY,
        "display_metric_key": COST_ADJUSTED_AVG_TRADE_KEY,
        "order": "descending",
        "tie_break": "combo_id",
    },
    {
        "id": "unrestricted_average_return_ge20",
        "label": "全坐标 · 至少 20 笔 · 笔均收益",
        "scenario_filter": "all",
        "minimum_trade_count": 20,
        "metric": "average_trade",
        "metric_key": COST_ADJUSTED_AVG_TRADE_KEY,
        "gross_metric_key": "train_avg_trade",
        "cost_adjusted_metric_key": COST_ADJUSTED_AVG_TRADE_KEY,
        "display_metric_key": COST_ADJUSTED_AVG_TRADE_KEY,
        "order": "descending",
        "tie_break": "combo_id",
    },
)


TRADE_COLUMNS = [
    "combo_id", "method", "baseline_sampling_policy", "entry_time", "exit_time",
    "exit_reason", "return", "gross_return", "cost_adjusted_return",
    "round_trip_cost_bps", "cost_model_id",
    "entry_price", "exit_price", "signal_time", "h_time", "entry_trigger_price",
    "entry_price_basis", "entry_fill_source", "entry_wait_bar_count",
    "entry_baseline_value", "entry_drop_value", "active_low", "rebound_net_drop",
    "rebound_threshold", "rebound_check_price", "rebound_check_price_basis",
    "rebound_gap_adjusted", "rebound_gap_slippage", "speed_window_bars",
    "speed_reference_time", "speed_reference_low", "speed_current_low",
    "speed_extension", "speed_check_price", "speed_check_price_basis",
    "exit_price_basis", "exit_bar_synthetic", "exit_bar_volume",
    "exit_bar_trade_count", "pending_exit", "pending_exit_trigger_index",
    "pending_exit_trigger_time", "pending_exit_trigger_reason",
    "pending_exit_theoretical_price", "pending_exit_wait_bar_count",
    "pending_exit_fill_policy", "pending_exit_fill_vs_theoretical_delta",
    "strategy_id", "trade_audit_schema_version", "trade_audit_schema_id",
    "rebound_baseline_policy_id", "rebound_max_w_drop",
    "rebound_latest_applied_candidate",
    "rebound_latest_applied_candidate_start_index",
    "rebound_latest_applied_candidate_end_index",
    "rebound_latest_applied_candidate_observed_bar_count",
    "rebound_exit_bar_candidate", "rebound_exit_bar_candidate_start_index",
    "rebound_exit_bar_candidate_end_index",
    "rebound_exit_bar_candidate_observed_bar_count",
    "rebound_candidates_effective_through_index",
    "rebound_window_observed_bar_count", "rebound_baseline_update_rule",
    "position_crosses_real_gap", "entry_bar_synthetic",
    "entry_bar_volume", "entry_bar_trade_count", "e", "bh", "trw", "k", "w", "m",
    "holding_bar_distance", "holding_minutes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def build_scenario_requirements_delivery(
    output_root: Path,
    scenario_definition_path: Path,
    *,
    main_href: str = "../index.html",
) -> dict[str, Any]:
    output_root = output_root.resolve()
    scenario_definition_path = scenario_definition_path.resolve()
    if (
        not MARKET_SELECTOR_SOURCE_PATH.is_file()
        or sha256_file(MARKET_SELECTOR_SOURCE_PATH) != MARKET_SELECTOR_SOURCE_SHA256
    ):
        raise ValueError("market-intuition selector source failed hash validation")
    if not SCENARIO_REQUIREMENTS_TEMPLATE_PATH.is_file():
        raise ValueError("scenario-requirements template is missing")
    if not scenario_definition_path.is_file():
        raise ValueError("scenario definition for requirements page is missing")

    scenario_definition = json.loads(
        scenario_definition_path.read_text(encoding="utf-8")
    )
    if (
        not scenario_definition.get("segments")
        or not scenario_definition.get("scenarios")
        or not scenario_definition.get("qualification_rule")
    ):
        raise ValueError("scenario definition lacks viewer requirements")

    output_root.mkdir(parents=True, exist_ok=True)
    data_path = output_root / "scenario_requirements_data.js"
    data_payload = {
        "schema_version": 1,
        "scenario_schema_id": scenario_definition.get("scenario_schema_id"),
        "selection_mode": scenario_definition.get("selection_mode"),
        "neutral_selection_id": scenario_definition.get("neutral_selection_id"),
        "qualification_rule": scenario_definition["qualification_rule"],
        "segments": scenario_definition["segments"],
        "scenarios": scenario_definition["scenarios"],
        "scenario_definition_sha256": sha256_file(scenario_definition_path),
        "market_selector_source": artifact(MARKET_SELECTOR_SOURCE_PATH),
    }
    _atomic_text(
        data_path,
        "window.V4_4_SCENARIO_REQUIREMENTS="
        + json.dumps(data_payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
    )

    html = SCENARIO_REQUIREMENTS_TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "__BACK_HREF__": main_href,
        "__PLOTLY_HREF__": "../assets/plotly.min.js",
        "__PROCESS_PAYLOAD_HREF__": "../trade_review/process_payload.js",
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    if any(token in html for token in replacements):
        raise ValueError("scenario-requirements template replacement is incomplete")
    index_path = output_root / "index.html"
    _atomic_text(index_path, html)
    return {
        "index": artifact(index_path),
        "data": artifact(data_path),
        "template": artifact(SCENARIO_REQUIREMENTS_TEMPLATE_PATH),
        "selector_source": artifact(MARKET_SELECTOR_SOURCE_PATH),
        "scenario_definition": artifact(scenario_definition_path),
    }


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clean(child) for child in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_clean(record) for record in frame.to_dict("records")]


def main_summary_payload(
    payload: dict[str, Any],
    *,
    native_trade_route: str | None = None,
    scenario_requirements_route: str | None = None,
) -> dict[str, Any]:
    summary = {
        key: _clean(payload[key])
        for key in MAIN_SUMMARY_TOP_LEVEL_FIELDS
        if key in payload
    }
    summary["rows"] = [
        {field: _clean(row.get(field)) for field in MAIN_SUMMARY_ROW_FIELDS}
        for row in payload.get("rows", [])
    ]
    if native_trade_route is not None:
        summary["nativeTradeRoute"] = native_trade_route
    if scenario_requirements_route is not None:
        summary["scenarioRequirementsRoute"] = scenario_requirements_route
    return summary


def _validate_artifact(record: dict[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != str(record.get("sha256", "")):
        raise ValueError(f"artifact failed hash validation: {label}")
    return path


def _load_stage(
    stage_root: Path,
    plan_path: Path,
    *,
    load_trades: bool = True,
) -> dict[str, Any]:
    stage_manifest_path = stage_root / "stage_manifest.json"
    completion_path = stage_root / "completion_manifest.json"
    progress_path = stage_root / "progress.json"
    for path in (stage_manifest_path, completion_path, progress_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    fingerprint = str(stage_manifest.get("plan_fingerprint", ""))
    stage_coordinate_count = int(stage_manifest.get("coordinate_count", -1))
    completion_coordinate_count = int(completion.get("coordinate_count", -1))
    stage_batch_count = int(stage_manifest.get("batch_count", -1))
    completion_batch_count = int(completion.get("batch_count", -1))
    baseline_sampling_policy = str(
        stage_manifest.get("baseline_sampling_policy", "")
    )
    expected_baseline_filter_id = (
        baseline_filter_id(baseline_sampling_policy)
        if baseline_sampling_policy
        else ""
    )
    stage_exit_mode = str(stage_manifest.get("exit_mode", ""))
    expected_strategy_id = (
        strategy_id(
            baseline_sampling_policy,
            combined_exit=stage_exit_mode == EXIT_MODE_COMBINED,
        )
        if baseline_sampling_policy
        else ""
    )
    expected_result_semantics_id = (
        result_semantics_id(
            ENTRY_FILL_CALCULATED_THRESHOLD,
            ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
            0.0,
            stage_exit_mode,
            baseline_sampling_policy,
        )
        if baseline_sampling_policy
        else ""
    )
    checks = {
        "raw_output_schema_consistent": all(
            int(payload.get("schema_version", -1)) == OUTPUT_SCHEMA_VERSION
            for payload in (stage_manifest, completion, progress)
        ),
        "plan_fingerprint_schema_consistent": int(
            stage_manifest.get("plan_fingerprint_schema_version", -1)
        )
        == FINGERPRINT_SCHEMA_VERSION,
        "plan_status_approved": plan.get("status") in APPROVED_PLAN_STATUSES,
        "stage_materialized": stage_manifest.get("status") == "materialized",
        "completion_complete": completion.get("status") == "complete",
        "progress_complete": progress.get("status") == "complete",
        "fingerprint_consistent": fingerprint
        and completion.get("plan_fingerprint") == fingerprint
        and progress.get("plan_fingerprint") == fingerprint,
        "plan_hash_bound": stage_manifest.get("input_plan_sha256") == sha256_file(plan_path),
        "exit_mode_consistent": bool(stage_exit_mode)
        and completion.get("exit_mode") == stage_exit_mode,
        "scenario_contract_available_or_explicitly_none": (
            "scenario_3" in completion.get("scenario_ids", [])
            or stage_manifest.get("scenario_selection_mode") == "none"
        ),
        "baseline_sampling_policy_consistent": bool(baseline_sampling_policy)
        and str(plan.get("baseline_sampling_policy", baseline_sampling_policy))
        == baseline_sampling_policy
        and completion.get("baseline_sampling_policy") == baseline_sampling_policy
        and progress.get("baseline_sampling_policy") == baseline_sampling_policy,
        "baseline_filter_identity_consistent": bool(expected_baseline_filter_id)
        and stage_manifest.get("baseline_filter_id") == expected_baseline_filter_id
        and completion.get("baseline_filter_id") == expected_baseline_filter_id
        and progress.get("baseline_filter_id") == expected_baseline_filter_id,
        "max_w_strategy_identity_consistent": bool(expected_strategy_id)
        and all(
            payload.get("strategy_id") == expected_strategy_id
            for payload in (stage_manifest, completion, progress)
        ),
        "max_w_result_semantics_consistent": bool(expected_result_semantics_id)
        and all(
            payload.get("result_semantics_id") == expected_result_semantics_id
            for payload in (stage_manifest, completion, progress)
        ),
        "max_w_trade_audit_identity_consistent": all(
            int(payload.get("trade_audit_schema_version", -1))
            == trade_audit_identity(stage_exit_mode)[0]
            and payload.get("trade_audit_schema_id")
            == trade_audit_identity(stage_exit_mode)[1]
            for payload in (stage_manifest, completion, progress)
        ),
        "max_w_rebound_policy_consistent": all(
            payload.get("rebound_baseline_policy_id")
            == REBOUND_BASELINE_POLICY_ID
            for payload in (stage_manifest, completion, progress)
        ),
        "coordinate_counts_consistent": (
            stage_coordinate_count
            == completion_coordinate_count
            == int(progress.get("total_coordinate_count", -1))
            == int(progress.get("completed_coordinate_count", -1))
            and int(progress.get("remaining_coordinate_count", -1)) == 0
        ),
        "batch_counts_consistent": (
            stage_batch_count
            == completion_batch_count
            == int(progress.get("total_batches", -1))
            == int(progress.get("completed_batch_count", -1))
            == len(progress.get("completed_batches", []))
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"stage identity audit failed: {checks}")
    summary_path = _validate_artifact(
        completion["artifacts"]["stage_summary"], "stage_summary"
    )
    segment_path = _validate_artifact(
        completion["artifacts"]["stage_segment_qualification"],
        "stage_segment_qualification",
    )
    scenario_path = _validate_artifact(
        completion["artifacts"]["stage_scenario_qualification"],
        "stage_scenario_qualification",
    )
    batch_index_path = _validate_artifact(
        completion["artifacts"]["batch_index"], "batch_index"
    )
    batch_index = json.loads(batch_index_path.read_text(encoding="utf-8"))
    if len(batch_index.get("batches", [])) != completion_batch_count:
        raise ValueError("batch index count differs from the closed stage contract")
    trades: list[pd.DataFrame] = []
    trade_paths: list[Path] = []
    batch_manifests: list[dict[str, Any]] = []
    batch_combo_ids: list[str] = []
    for batch in batch_index.get("batches", []):
        manifest_path = Path(str(batch["manifest"])).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("plan_fingerprint") != fingerprint:
            raise ValueError(f"batch identity mismatch: {manifest_path}")
        if int(manifest.get("schema_version", -1)) != OUTPUT_SCHEMA_VERSION:
            raise ValueError(f"batch output schema mismatch: {manifest_path}")
        if manifest.get("strategy_id") != expected_strategy_id:
            raise ValueError(f"batch max-W strategy identity mismatch: {manifest_path}")
        if (
            int(manifest.get("trade_audit_schema_version", -1))
            != trade_audit_identity(stage_exit_mode)[0]
            or manifest.get("trade_audit_schema_id")
            != trade_audit_identity(stage_exit_mode)[1]
        ):
            raise ValueError(f"batch trade-audit identity mismatch: {manifest_path}")
        if manifest.get("rebound_baseline_policy_id") != REBOUND_BASELINE_POLICY_ID:
            raise ValueError(f"batch rebound-policy identity mismatch: {manifest_path}")
        if manifest.get("baseline_sampling_policy") != baseline_sampling_policy:
            raise ValueError(f"batch baseline policy mismatch: {manifest_path}")
        if manifest.get("baseline_filter_id") != expected_baseline_filter_id:
            raise ValueError(f"batch baseline filter identity mismatch: {manifest_path}")
        for name, record in manifest.get("artifacts", {}).items():
            if not load_trades and name == "trades":
                trade_artifact_path = Path(str(record.get("path", ""))).resolve()
                if not trade_artifact_path.is_file():
                    raise FileNotFoundError(trade_artifact_path)
                continue
            _validate_artifact(record, f"{manifest['batch_id']}:{name}")
        manifest_combo_ids = [str(value) for value in manifest.get("combo_ids", [])]
        if len(manifest_combo_ids) != int(manifest.get("coordinate_count", -1)):
            raise ValueError(f"batch combo count mismatch: {manifest_path}")
        batch_combo_ids.extend(manifest_combo_ids)
        trade_path = Path(str(manifest["artifacts"]["trades"]["path"])).resolve()
        trade_paths.append(trade_path)
        if load_trades:
            trades.append(pd.read_csv(trade_path))
        batch_manifests.append(artifact(manifest_path))
    summary = pd.read_csv(summary_path)
    segment = pd.read_csv(segment_path)
    scenario = pd.read_csv(scenario_path)
    all_trades = pd.concat(trades, ignore_index=True, sort=False) if trades else pd.DataFrame()
    if len(summary) != int(completion["coordinate_count"]):
        raise ValueError("stage summary coordinate count mismatch")
    if load_trades and len(all_trades) != int(completion["trade_count"]):
        raise ValueError("stage trade count mismatch")
    if summary.combo_id.astype(str).duplicated().any():
        raise ValueError("stage summary contains duplicate combo_id")
    grid_path = Path(str(stage_manifest.get("artifacts", {}).get("grid_manifest", ""))).resolve()
    if (
        not grid_path.is_file()
        or sha256_file(grid_path) != str(stage_manifest.get("grid_manifest_sha256", ""))
    ):
        raise ValueError("stage grid manifest failed hash validation")
    grid = pd.read_csv(grid_path)
    if "combo_id" not in grid.columns or grid.combo_id.astype(str).duplicated().any():
        raise ValueError("stage grid manifest lacks unique combo_id rows")
    grid_ids = set(grid.combo_id.astype(str))
    summary_ids = set(summary.combo_id.astype(str))
    if len(grid) != stage_coordinate_count or summary_ids != grid_ids:
        raise ValueError("stage summary does not exactly cover the hash-bound grid")
    if len(batch_combo_ids) != len(set(batch_combo_ids)) or set(batch_combo_ids) != grid_ids:
        raise ValueError("closed batch manifests do not exactly cover the hash-bound grid")
    policy_frames = {
        "grid": grid,
        "summary": summary,
    }
    if stage_manifest.get("scenario_selection_mode") != "none":
        policy_frames.update({"segment": segment, "scenario": scenario})
    if not all_trades.empty:
        policy_frames["trades"] = all_trades
    for label, frame in policy_frames.items():
        if "baseline_sampling_policy" not in frame.columns:
            raise ValueError(f"{label} lacks baseline_sampling_policy")
        values = set(frame["baseline_sampling_policy"].astype(str).unique())
        if values != {baseline_sampling_policy}:
            raise ValueError(f"{label} baseline sampling policy mismatch: {values}")

    scenario_definition_value = stage_manifest.get("scenario_definition")
    scenario_definition_path: Path | None = None
    if scenario_definition_value in (None, ""):
        if str(stage_manifest.get("scenario_selection_mode", "")) != "none":
            raise ValueError("stage without scenario definition must declare scenario mode none")
        segment_ids: set[str] = set()
        scenario_ids: set[str] = set()
    else:
        scenario_definition_path = Path(str(scenario_definition_value)).resolve()
        if (
            not scenario_definition_path.is_file()
            or sha256_file(scenario_definition_path)
            != str(stage_manifest.get("scenario_definition_sha256", ""))
        ):
            raise ValueError("scenario definition failed hash validation")
        scenario_definition = json.loads(
            scenario_definition_path.read_text(encoding="utf-8")
        )
        segment_ids = {
            str(row["segment_id"]) for row in scenario_definition.get("segments", [])
        }
        scenario_ids = {
            str(row["scenario_id"]) for row in scenario_definition.get("scenarios", [])
        }
    expected_segment_keys = {(combo_id, segment_id) for combo_id in grid_ids for segment_id in segment_ids}
    expected_scenario_keys = {(combo_id, scenario_id) for combo_id in grid_ids for scenario_id in scenario_ids}
    actual_segment_keys = (
        set(zip(segment.combo_id.astype(str), segment.segment_id.astype(str)))
        if segment_ids else set()
    )
    actual_scenario_keys = (
        set(zip(scenario.combo_id.astype(str), scenario.scenario_id.astype(str)))
        if scenario_ids else set()
    )
    if len(segment) != len(actual_segment_keys) or actual_segment_keys != expected_segment_keys:
        raise ValueError("segment qualification does not exactly cover every coordinate and segment")
    if len(scenario) != len(actual_scenario_keys) or actual_scenario_keys != expected_scenario_keys:
        raise ValueError("scenario qualification does not exactly cover every coordinate and scenario")
    return {
        "plan": plan,
        "stage_manifest": stage_manifest,
        "completion": completion,
        "summary": summary,
        "segment": segment,
        "scenario": scenario,
        "trades": all_trades,
        "checks": checks,
        "artifacts": {
            "plan": artifact(plan_path),
            "stage_manifest": artifact(stage_manifest_path),
            "completion_manifest": artifact(completion_path),
            "progress": artifact(progress_path),
            "stage_summary": artifact(summary_path),
            "stage_segment_qualification": artifact(segment_path),
            "stage_scenario_qualification": artifact(scenario_path),
            "batch_index": artifact(batch_index_path),
            "batch_manifests": batch_manifests,
            "trade_paths": trade_paths,
            "grid_manifest": artifact(grid_path),
            "scenario_definition": (
                artifact(scenario_definition_path)
                if scenario_definition_path is not None
                else None
            ),
        },
    }


def _validate_trades(
    trades: pd.DataFrame, exit_mode: str = EXIT_MODE_COMBINED
) -> dict[str, Any]:
    if trades.empty:
        return {
            "all_actual_entry_bars_real": True,
            "all_signal_exit_fills_real": True,
            "maximum_wait_within_limit": True,
            "speed_reason_present": False,
            "speed_extension_nonpositive": True,
            "speed_reference_distance_exact": True,
            "speed_fill_policy_valid": True,
            "lifecycle_nonnegative": True,
            "max_w_strategy_identity_valid": True,
            "max_w_trade_audit_identity_valid": True,
            "max_w_update_rule_valid": True,
            "max_w_basis_alias_valid": True,
            "max_w_basis_positive_when_present": True,
            "max_w_source_window_valid": True,
            "max_w_latest_applied_source_valid": True,
            "max_w_exit_candidate_source_valid": True,
            "max_w_closed_bar_timing_valid": True,
            "max_w_threshold_valid": True,
            "trade_count": 0,
            "waited_entry_count": 0,
            "maximum_entry_wait_bars": 0,
            "rebound_exit_count": 0,
            "speed_exit_count": 0,
            "segment_end_exit_count": 0,
            "gap_spanning_trade_count": 0,
            "holding_bar_distance_minimum": None,
            "holding_bar_distance_median": None,
            "holding_bar_distance_p95": None,
            "holding_bar_distance_maximum": None,
            "holding_minutes_minimum": None,
            "holding_minutes_median": None,
            "holding_minutes_p95": None,
            "holding_minutes_maximum": None,
        }
    required = {
        "combo_id", "exit_reason", "entry_wait_bar_count", "entry_bar_synthetic",
        "entry_bar_volume", "entry_bar_trade_count", "exit_price", "speed_window_bars",
        "speed_extension", "exit_price_basis", "speed_reference_index", "entry_index",
        "signal_index", "w",
        "exit_index", "entry_time", "exit_time", "holding_bar_distance", "holding_minutes",
        "exit_bar_synthetic", "exit_bar_volume", "exit_bar_trade_count", "pending_exit",
        "pending_exit_trigger_index", "pending_exit_wait_bar_count", "pending_exit_fill_policy",
    }
    required.update(MAX_W_TRADE_AUDIT_FIELDS)
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"stage trades lack audit fields: {sorted(missing)}")
    real_entry = (
        ~_truthy(trades.entry_bar_synthetic)
        & pd.to_numeric(trades.entry_bar_volume, errors="raise").gt(0)
        & pd.to_numeric(trades.entry_bar_trade_count, errors="raise").gt(0)
    )
    waits = pd.to_numeric(trades.entry_wait_bar_count, errors="raise")
    signal_exit = trades.exit_reason.astype(str).isin(
        ["rebound_threshold", DOWNSIDE_SPEED_EXIT_REASON]
    )
    real_signal_exit = (
        ~_truthy(trades.loc[signal_exit, "exit_bar_synthetic"])
        & pd.to_numeric(trades.loc[signal_exit, "exit_bar_volume"], errors="raise").gt(0)
        & pd.to_numeric(
            trades.loc[signal_exit, "exit_bar_trade_count"], errors="raise"
        ).gt(0)
    )
    pending_signal_exit = _truthy(trades.loc[signal_exit, "pending_exit"])
    pending_fill_policy = trades.loc[signal_exit, "pending_exit_fill_policy"].astype(str)
    pending_wait = pd.to_numeric(
        trades.loc[signal_exit, "pending_exit_wait_bar_count"], errors="raise"
    )
    pending_policy_valid = bool(
        pending_fill_policy.loc[pending_signal_exit].eq("next_real_trade_bar_open").all()
        and pending_wait.loc[pending_signal_exit].gt(0).all()
        and pending_fill_policy.loc[~pending_signal_exit].eq("same_real_trade_bar").all()
        and pending_wait.loc[~pending_signal_exit].eq(0).all()
    )
    audit_version = pd.to_numeric(
        trades["trade_audit_schema_version"], errors="raise"
    )
    max_w_drop = pd.to_numeric(trades["rebound_max_w_drop"], errors="coerce")
    rebound_net_drop = pd.to_numeric(trades["rebound_net_drop"], errors="coerce")
    max_start = pd.to_numeric(trades["rebound_window_start_index"], errors="raise")
    max_end = pd.to_numeric(trades["rebound_window_end_index"], errors="raise")
    max_observed = pd.to_numeric(
        trades["rebound_window_observed_bar_count"], errors="raise"
    )
    latest_candidate = pd.to_numeric(
        trades["rebound_latest_applied_candidate"], errors="coerce"
    )
    latest_start = pd.to_numeric(
        trades["rebound_latest_applied_candidate_start_index"], errors="raise"
    )
    latest_end = pd.to_numeric(
        trades["rebound_latest_applied_candidate_end_index"], errors="raise"
    )
    latest_observed = pd.to_numeric(
        trades["rebound_latest_applied_candidate_observed_bar_count"],
        errors="raise",
    )
    exit_candidate_start = pd.to_numeric(
        trades["rebound_exit_bar_candidate_start_index"], errors="raise"
    )
    exit_candidate_end = pd.to_numeric(
        trades["rebound_exit_bar_candidate_end_index"], errors="raise"
    )
    exit_candidate_observed = pd.to_numeric(
        trades["rebound_exit_bar_candidate_observed_bar_count"], errors="raise"
    )
    effective_through = pd.to_numeric(
        trades["rebound_candidates_effective_through_index"], errors="raise"
    )
    exit_indices = pd.to_numeric(trades["exit_index"], errors="raise")
    trigger_indices = pd.to_numeric(
        trades["pending_exit_trigger_index"], errors="coerce"
    )
    all_pending = _truthy(trades["pending_exit"])
    candidate_end_expected = exit_indices.where(~all_pending, trigger_indices)
    basis_present = max_w_drop.notna()
    effective_present = effective_through.ge(0)
    entry_indices = pd.to_numeric(trades["entry_index"], errors="raise")
    signal_indices = pd.to_numeric(trades["signal_index"], errors="raise")
    h_indices = pd.to_numeric(trades["h_index"], errors="raise")
    rebound_reason = trades.exit_reason.astype(str).eq("rebound_threshold")
    speed_reason = trades.exit_reason.astype(str).eq(DOWNSIDE_SPEED_EXIT_REASON)
    segment_end_reason = trades.exit_reason.astype(str).eq("segment_end")
    same_signal_immediate_rebound = (
        basis_present
        & rebound_reason
        & ~all_pending
        & entry_indices.eq(signal_indices)
        & exit_indices.eq(signal_indices)
        & effective_through.eq(-1)
        & latest_start.eq(-1)
        & latest_end.eq(-1)
        & latest_observed.eq(0)
        & latest_candidate.isna()
        & max_end.eq(signal_indices - 1)
    )
    applied_state_basis = basis_present & ~same_signal_immediate_rebound
    rebound_applied_state_basis = applied_state_basis & rebound_reason
    speed_or_segment_basis = applied_state_basis & (
        speed_reason | segment_end_reason
    )
    w_values = pd.to_numeric(trades["w"], errors="raise")
    max_source_valid = bool(
        max_start.loc[basis_present].ge(0).all()
        and max_start.loc[basis_present].ge(h_indices.loc[basis_present]).all()
        and max_end.loc[basis_present].ge(max_start.loc[basis_present]).all()
        and max_observed.loc[basis_present].gt(0).all()
        and max_observed.loc[basis_present].le(w_values.loc[basis_present]).all()
        and (max_end.loc[basis_present] - max_start.loc[basis_present] + 1)
        .eq(max_observed.loc[basis_present])
        .all()
        and max_end.loc[rebound_applied_state_basis]
        .lt(exit_candidate_end.loc[rebound_applied_state_basis])
        .all()
        and max_end.loc[rebound_applied_state_basis]
        .le(effective_through.loc[rebound_applied_state_basis])
        .all()
        and max_end.loc[speed_or_segment_basis]
        .le(effective_through.loc[speed_or_segment_basis])
        .all()
        and (
            ~basis_present | rebound_reason | speed_reason | segment_end_reason
        ).all()
        and bool(
            (
                same_signal_immediate_rebound
                | ~(
                    basis_present
                    & rebound_reason
                    & effective_through.lt(0)
                )
            ).all()
        )
        and max_start.loc[~basis_present].eq(-1).all()
        and max_end.loc[~basis_present].eq(-1).all()
        and max_observed.loc[~basis_present].eq(0).all()
    )
    latest_source_valid = bool(
        latest_start.loc[effective_present].ge(0).all()
        and latest_start.loc[effective_present]
        .ge(h_indices.loc[effective_present])
        .all()
        and latest_end.loc[effective_present].eq(
            effective_through.loc[effective_present]
        ).all()
        and latest_observed.loc[effective_present].gt(0).all()
        and (latest_end.loc[effective_present] - latest_start.loc[effective_present] + 1)
        .eq(latest_observed.loc[effective_present])
        .all()
        and latest_start.loc[~effective_present].eq(-1).all()
        and latest_end.loc[~effective_present].eq(-1).all()
        and latest_observed.loc[~effective_present].eq(0).all()
        and latest_candidate.loc[~effective_present].isna().all()
    )
    exit_candidate_source_valid = bool(
        exit_candidate_end.eq(candidate_end_expected).all()
        and exit_candidate_start.ge(0).all()
        and exit_candidate_start.ge(h_indices).all()
        and exit_candidate_end.ge(exit_candidate_start).all()
        and exit_candidate_observed.gt(0).all()
        and (exit_candidate_end - exit_candidate_start + 1)
        .eq(exit_candidate_observed)
        .all()
    )
    closed_bar_timing_valid = bool(
        effective_through.loc[rebound_reason]
        .lt(exit_candidate_end.loc[rebound_reason])
        .all()
        and effective_through.loc[speed_reason]
        .eq(exit_candidate_end.loc[speed_reason])
        .all()
        and effective_through.loc[segment_end_reason]
        .eq(exit_candidate_end.loc[segment_end_reason])
        .all()
    )
    threshold = pd.to_numeric(trades["rebound_threshold"], errors="coerce")
    trigger_price = pd.to_numeric(
        trades["rebound_trigger_price"], errors="coerce"
    )
    expected_threshold = (
        pd.to_numeric(trades["active_low"], errors="raise")
        + pd.to_numeric(trades["m"], errors="raise") * max_w_drop
    )
    max_w_checks = {
        "max_w_strategy_identity_valid": bool(
            all(
                trades.loc[
                    trades["baseline_sampling_policy"].astype(str).eq(policy),
                    "strategy_id",
                ]
                .astype(str)
                    .eq(strategy_id(policy, combined_exit=exit_mode == EXIT_MODE_COMBINED))
                .all()
                for policy in trades["baseline_sampling_policy"].astype(str).unique()
            )
        ),
        "max_w_trade_audit_identity_valid": bool(
                audit_version.eq(trade_audit_identity(exit_mode)[0]).all()
                and trades["trade_audit_schema_id"]
                .astype(str)
                .eq(trade_audit_identity(exit_mode)[1])
            .all()
            and trades["rebound_baseline_policy_id"]
            .astype(str)
            .eq(REBOUND_BASELINE_POLICY_ID)
            .all()
        ),
        "max_w_update_rule_valid": bool(
            trades["rebound_baseline_update_rule"]
            .astype(str)
            .eq("maximum_positive_completed_bar_w_candidates_effective_next_bar")
            .all()
        ),
        "max_w_basis_alias_valid": bool(
            np.isclose(
                rebound_net_drop.fillna(0.0),
                max_w_drop.fillna(0.0),
                rtol=0.0,
                atol=1e-12,
            ).all()
            and rebound_net_drop.isna().eq(max_w_drop.isna()).all()
        ),
        "max_w_basis_positive_when_present": bool(
            max_w_drop.loc[basis_present].gt(0).all()
        ),
        "max_w_source_window_valid": max_source_valid,
        "max_w_latest_applied_source_valid": latest_source_valid,
        "max_w_exit_candidate_source_valid": exit_candidate_source_valid,
        "max_w_closed_bar_timing_valid": closed_bar_timing_valid,
        "max_w_threshold_valid": bool(
            np.isclose(
                threshold.loc[basis_present],
                expected_threshold.loc[basis_present],
                rtol=0.0,
                atol=1e-9,
            ).all()
            and threshold.loc[~basis_present].isna().all()
            and np.isclose(
                trigger_price.loc[basis_present],
                threshold.loc[basis_present],
                rtol=0.0,
                atol=1e-9,
            ).all()
            and trigger_price.loc[~basis_present].isna().all()
        ),
    }
    speed = trades.loc[
        trades.exit_reason.astype(str).eq(DOWNSIDE_SPEED_EXIT_REASON),
        [
            "speed_extension",
            "speed_reference_index",
            "exit_index",
            "pending_exit",
            "pending_exit_trigger_index",
            "speed_window_bars",
            "exit_price_basis",
        ],
    ]
    speed_extension = pd.to_numeric(speed.speed_extension, errors="coerce")
    speed_reference = pd.to_numeric(speed.speed_reference_index, errors="coerce")
    speed_exit_index = pd.to_numeric(speed.exit_index, errors="coerce")
    speed_pending = _truthy(speed.pending_exit)
    speed_trigger_index = pd.to_numeric(speed.pending_exit_trigger_index, errors="coerce")
    speed_check_index = speed_exit_index.where(~speed_pending, speed_trigger_index)
    speed_windows = pd.to_numeric(speed.speed_window_bars, errors="coerce")
    holding_bars = pd.to_numeric(trades.holding_bar_distance, errors="raise")
    holding_minutes = pd.to_numeric(trades.holding_minutes, errors="raise")
    speed_checks = {
        "all_actual_entry_bars_real": bool(real_entry.all()),
        "all_signal_exit_fills_real": bool(real_signal_exit.all()),
        "pending_exit_policy_valid": pending_policy_valid,
        "maximum_wait_within_limit": bool(waits.max() <= MAX_REAL_TRADE_WAIT_BARS),
        "speed_reason_present": bool(len(speed)),
        "speed_extension_nonpositive": bool((speed_extension <= 0).all()),
        "speed_reference_distance_exact": bool(
            ((speed_check_index - speed_reference) == speed_windows).all()
        ),
        "speed_fill_policy_valid": bool(
            speed.loc[~speed_pending, "exit_price_basis"].eq("current_bar_close").all()
            and speed.loc[speed_pending, "exit_price_basis"]
            .eq("next_real_trade_bar_open_after_pending_exit")
            .all()
        ),
        "lifecycle_nonnegative": bool(
            holding_bars.ge(0).all() and holding_minutes.ge(0).all()
        ),
        **max_w_checks,
    }
    required_checks = {
        key: value for key, value in speed_checks.items() if key != "speed_reason_present"
    }
    if not all(required_checks.values()):
        raise ValueError(f"trade execution audit failed: {speed_checks}")
    return {
        **speed_checks,
        "trade_count": int(len(trades)),
        "waited_entry_count": int(waits.gt(0).sum()),
        "maximum_entry_wait_bars": int(waits.max()),
        "rebound_exit_count": int(trades.exit_reason.astype(str).eq("rebound_threshold").sum()),
        "speed_exit_count": int(len(speed)),
        "segment_end_exit_count": int(trades.exit_reason.astype(str).eq("segment_end").sum()),
        "gap_spanning_trade_count": int(_truthy(trades.position_crosses_real_gap).sum()),
        "holding_bar_distance_minimum": int(holding_bars.min()),
        "holding_bar_distance_median": float(holding_bars.median()),
        "holding_bar_distance_p95": float(holding_bars.quantile(0.95)),
        "holding_bar_distance_maximum": int(holding_bars.max()),
        "holding_minutes_minimum": float(holding_minutes.min()),
        "holding_minutes_median": float(holding_minutes.median()),
        "holding_minutes_p95": float(holding_minutes.quantile(0.95)),
        "holding_minutes_maximum": float(holding_minutes.max()),
    }


def _augment_trade_lifecycle(
    trades: pd.DataFrame,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Add lifecycle columns; callers may reuse disposable loaded trade frames."""
    result = trades.copy() if copy else trades
    if result.empty:
        result["holding_bar_distance"] = pd.Series(dtype="int64")
        result["holding_minutes"] = pd.Series(dtype="float64")
        return result
    entry_index = pd.to_numeric(result.entry_index, errors="raise")
    exit_index = pd.to_numeric(result.exit_index, errors="raise")
    entry_time = pd.to_datetime(result.entry_time, errors="raise")
    exit_time = pd.to_datetime(result.exit_time, errors="raise")
    result["holding_bar_distance"] = (exit_index - entry_index).astype(int)
    result["holding_minutes"] = (
        (exit_time - entry_time).dt.total_seconds() / 60.0
    )
    return result


def _maximum_drawdown(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="raise").to_numpy(dtype=float)
    if not len(values):
        return 0.0
    if not np.isfinite(values).all() or np.any(values <= -1.0):
        raise ValueError("cost-adjusted returns must be finite and greater than -100%")
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def _apply_cost_adjusted_metrics(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    cost_model: dict[str, Any] | None = None,
    copy: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add derived cost metrics; callers may reuse disposable loaded frames."""
    result_summary = summary.copy() if copy else summary
    result_trades = trades.copy() if copy else trades
    selected_cost_model = cost_model or K200M_COST_MODEL
    cost_bps = float(selected_cost_model["round_trip_total_cost_bps"])
    cost_fraction = cost_bps / 10000.0
    commission_quote = float(selected_cost_model["round_trip_commission_quote"])
    slippage_quote = float(selected_cost_model["round_trip_slippage_quote"])
    total_cost_quote = float(selected_cost_model["round_trip_total_cost_quote"])
    quote_currency = str(selected_cost_model["quote_currency"])

    gross = pd.to_numeric(result_trades["return"], errors="raise")
    result_trades["gross_return"] = gross
    result_trades["cost_adjusted_return"] = gross - cost_fraction
    result_trades["round_trip_cost_bps"] = cost_bps
    result_trades["round_trip_commission_quote"] = commission_quote
    result_trades["round_trip_slippage_quote"] = slippage_quote
    result_trades["round_trip_total_cost_quote"] = total_cost_quote
    result_trades["cost_quote_currency"] = quote_currency
    if quote_currency == "KRW":
        result_trades["round_trip_commission_krw"] = commission_quote
        result_trades["round_trip_slippage_krw"] = slippage_quote
        result_trades["round_trip_total_cost_krw"] = total_cost_quote
    result_trades["cost_model_id"] = str(selected_cost_model["id"])

    metric_frame = result_trades[
        ["combo_id", "entry_index", "exit_index", "gross_return", "cost_adjusted_return"]
    ].copy()
    metric_frame["combo_id"] = metric_frame["combo_id"].astype(str)
    metric_frame = metric_frame.sort_values(
        ["combo_id", "entry_index", "exit_index"], kind="mergesort"
    )
    combo_groups = metric_frame["combo_id"]
    gross_wealth = (1.0 + metric_frame["gross_return"]).groupby(
        combo_groups, sort=False
    ).cumprod()
    adjusted_wealth = (1.0 + metric_frame["cost_adjusted_return"]).groupby(
        combo_groups, sort=False
    ).cumprod()
    adjusted_peak = adjusted_wealth.groupby(combo_groups, sort=False).cummax().clip(
        lower=1.0
    )
    adjusted_drawdown = adjusted_wealth / adjusted_peak - 1.0
    metric_frame["gross_wealth"] = gross_wealth
    metric_frame["adjusted_wealth"] = adjusted_wealth
    metric_frame["adjusted_drawdown"] = adjusted_drawdown
    grouped_metrics = metric_frame.groupby("combo_id", sort=False)
    metrics = grouped_metrics.agg(
        gross_compounded=("gross_wealth", "last"),
        adjusted_wealth=("adjusted_wealth", "last"),
        adjusted_avg=("cost_adjusted_return", "mean"),
        adjusted_drawdown=("adjusted_drawdown", "min"),
        trade_count=("combo_id", "size"),
    ).reset_index()
    metrics["gross_compounded"] -= 1.0
    metrics[COST_ADJUSTED_RETURN_KEY] = metrics.pop("adjusted_wealth") - 1.0
    metrics[COST_ADJUSTED_AVG_TRADE_KEY] = metrics.pop("adjusted_avg")
    metrics[COST_ADJUSTED_MAX_DRAWDOWN_KEY] = metrics.pop("adjusted_drawdown").clip(
        upper=0.0
    )
    metrics["train_cost_adjusted_max_drawdown_abs"] = metrics[
        COST_ADJUSTED_MAX_DRAWDOWN_KEY
    ].abs()
    summary_ids = result_summary["combo_id"].astype(str)
    metrics = metrics.set_index("combo_id").reindex(summary_ids).reset_index()
    empty_mask = metrics["trade_count"].isna()
    metrics.loc[empty_mask, COST_ADJUSTED_RETURN_KEY] = 0.0
    metrics.loc[empty_mask, COST_ADJUSTED_MAX_DRAWDOWN_KEY] = 0.0
    metrics.loc[empty_mask, "train_cost_adjusted_max_drawdown_abs"] = 0.0
    metrics.loc[empty_mask, "trade_count"] = 0
    expected_gross = pd.to_numeric(result_summary["train_return"], errors="raise").to_numpy()
    if not np.allclose(
        metrics["gross_compounded"].fillna(0.0).to_numpy(),
        expected_gross,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("gross trade compounding does not match stage summary")
    metrics.drop(columns="gross_compounded", inplace=True)
    metrics["round_trip_cost_bps"] = cost_bps
    metrics["estimated_total_commission_quote"] = metrics["trade_count"] * commission_quote
    metrics["estimated_total_slippage_quote"] = metrics["trade_count"] * slippage_quote
    metrics["estimated_total_cost_quote"] = metrics["trade_count"] * total_cost_quote
    metrics["cost_quote_currency"] = quote_currency
    metrics["cost_model_id"] = str(selected_cost_model["id"])
    if quote_currency == "KRW":
        metrics["estimated_total_commission_krw"] = metrics[
            "estimated_total_commission_quote"
        ]
        metrics["estimated_total_slippage_krw"] = metrics[
            "estimated_total_slippage_quote"
        ]
        metrics["estimated_total_cost_krw"] = metrics["estimated_total_cost_quote"]
    metrics.drop(columns="trade_count", inplace=True)
    if metrics["combo_id"].astype(str).duplicated().any():
        raise ValueError("cost-adjusted metrics contain duplicate combo IDs")
    result_summary = result_summary.merge(
        metrics, on="combo_id", how="left", validate="one_to_one"
    )
    if result_summary[COST_ADJUSTED_RETURN_KEY].isna().any():
        raise ValueError("cost-adjusted metrics do not cover every coordinate")
    return result_summary, result_trades


def _combo_lifecycle_summary(summary: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            {
                "combo_id": summary.combo_id.astype(str),
                "waited_entry_count": 0,
                "maximum_entry_wait_bars": 0,
                "holding_bar_distance_median": np.nan,
                "holding_bar_distance_p95": np.nan,
            }
        )
    result = (
        trades.assign(
            waited_entry=lambda frame: pd.to_numeric(
                frame.entry_wait_bar_count, errors="raise"
            ).gt(0),
            entry_wait_bar_count_numeric=lambda frame: pd.to_numeric(
                frame.entry_wait_bar_count, errors="raise"
            ),
        )
        .groupby("combo_id", sort=False)
        .agg(
            waited_entry_count=("waited_entry", "sum"),
            maximum_entry_wait_bars=("entry_wait_bar_count_numeric", "max"),
            holding_bar_distance_median=("holding_bar_distance", "median"),
            holding_bar_distance_p95=(
                "holding_bar_distance",
                lambda values: values.quantile(0.95),
            ),
        )
        .reset_index()
    )
    result = summary[["combo_id"]].merge(result, on="combo_id", how="left")
    result[["waited_entry_count", "maximum_entry_wait_bars"]] = result[
        ["waited_entry_count", "maximum_entry_wait_bars"]
    ].fillna(0)
    return result


def _exit_reason_summary(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["method", "speed_window_bars", "exit_reason", "trade_count", "mean_return"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    return (
        trades.groupby(
            ["method", "baseline_sampling_policy", "speed_window_bars", "exit_reason"],
            sort=True,
        )
        .agg(trade_count=("combo_id", "size"), mean_return=("return", "mean"))
        .reset_index()
    )


def _trade_lifecycle_summary(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method", "speed_window_bars", "exit_reason", "trade_count",
        "holding_bar_distance_median", "holding_bar_distance_p95",
        "holding_bar_distance_maximum", "holding_minutes_median",
        "holding_minutes_p95", "holding_minutes_maximum", "waited_entry_count",
        "gap_spanning_trade_count",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    return (
        trades.assign(
            waited_entry=lambda frame: pd.to_numeric(
                frame.entry_wait_bar_count, errors="raise"
            ).gt(0),
            gap_spanning=lambda frame: _truthy(frame.position_crosses_real_gap),
        )
        .groupby(
            ["method", "baseline_sampling_policy", "speed_window_bars", "exit_reason"],
            sort=True,
        )
        .agg(
            trade_count=("combo_id", "size"),
            holding_bar_distance_median=("holding_bar_distance", "median"),
            holding_bar_distance_p95=(
                "holding_bar_distance", lambda values: values.quantile(0.95)
            ),
            holding_bar_distance_maximum=("holding_bar_distance", "max"),
            holding_minutes_median=("holding_minutes", "median"),
            holding_minutes_p95=("holding_minutes", lambda values: values.quantile(0.95)),
            holding_minutes_maximum=("holding_minutes", "max"),
            waited_entry_count=("waited_entry", "sum"),
            gap_spanning_trade_count=("gap_spanning", "sum"),
        )
        .reset_index()
    )


def _rank(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary.copy()
    required_cost_fields = {
        COST_ADJUSTED_RETURN_KEY,
        COST_ADJUSTED_AVG_TRADE_KEY,
        COST_ADJUSTED_MAX_DRAWDOWN_KEY,
    }
    missing_cost_fields = required_cost_fields.difference(rows.columns)
    if missing_cost_fields:
        raise ValueError(
            f"summary lacks cost-adjusted ranking fields: {sorted(missing_cost_fields)}"
        )
    rows["scenario_3_qualified"] = _truthy(rows["scenario_3_qualified"])
    drawdown = (
        rows["train_max_drawdown_abs"]
        if "train_max_drawdown_abs" in rows.columns
        else rows["train_max_drawdown"]
    )
    rows["train_max_drawdown_abs"] = pd.to_numeric(
        drawdown,
        errors="raise",
    ).abs()
    rows["scenario_3_cost_adjusted_return_rank"] = np.nan
    rows["scenario_3_gross_return_rank"] = np.nan
    for _, group in rows.loc[rows.scenario_3_qualified].groupby(
        ["method", "baseline_sampling_policy"], sort=True
    ):
        cost_ordered = group.sort_values(
            [COST_ADJUSTED_RETURN_KEY, "combo_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        rows.loc[
            cost_ordered.index, "scenario_3_cost_adjusted_return_rank"
        ] = np.arange(
            1, len(cost_ordered) + 1
        )
        gross_ordered = group.sort_values(
            ["train_return", "combo_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        rows.loc[gross_ordered.index, "scenario_3_gross_return_rank"] = np.arange(
            1, len(gross_ordered) + 1
        )
    rows["scenario_3_total_return_rank"] = rows[
        "scenario_3_cost_adjusted_return_rank"
    ]
    return rows.sort_values(
        [
            "method",
            "scenario_3_qualified",
            "scenario_3_cost_adjusted_return_rank",
            "combo_id",
        ],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _main_html() -> str:
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V4.41 情景三 combined 回测</title><style>
:root{--bg:#f4f7fb;--p:#fff;--ink:#132238;--muted:#65758b;--line:#d9e2ec;--accent:#1e63c6;--good:#087a4b;--bad:#a34722}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1600px;margin:auto;padding:22px 16px 48px}h1{margin:0}h2{margin:24px 0 9px;font-size:18px}.muted{color:var(--muted)}.controls,.cards{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}.controls label{display:flex;gap:6px;align-items:center}select{padding:8px;border:1px solid var(--line);border-radius:7px;background:var(--p)}.card,.note{background:var(--p);border:1px solid var(--line);border-radius:10px;padding:11px 13px}.card{min-width:150px}.card b{display:block;font-size:22px}.table{overflow:auto;background:var(--p);border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:max-content;min-width:100%}th,td{padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}th:first-child,td:first-child{text-align:left}a{color:var(--accent)}.yes{color:var(--good);font-weight:700}.no{color:var(--bad);font-weight:700}@media(max-width:700px){main{padding:14px 9px}.card{min-width:130px;flex:1}}</style></head><body><main><h1>V4.41 情景三 combined 回测</h1><div class="muted">情景三内部 AND · 总收益目标 · 滚动 TR 总和均值 · baseline 策略分开排名</div><div class="controls"><label>方法<select id="method"><option value="rolling_tr_sum">滚动 TR 总和均值</option></select></label><label>基准采样<select id="baselinePolicy"><option value="all_window">全部</option><option value="exclude_marked">排除标记</option></select></label><label>速度窗口<select id="speed"></select></label><label>结果范围<select id="view"><option value="all">全部坐标</option><option value="qualified">情景三合格</option></select></label></div><div id="cards" class="cards"></div><div id="note" class="note"></div><h2>坐标结果</h2><div id="table" class="table"></div><h2>情景资格</h2><div id="scenarioTable" class="table"></div><h2>分段资格</h2><div id="segmentTable" class="table"></div><h2>退出原因</h2><div id="exitTable" class="table"></div><h2>交易生命周期</h2><div id="lifecycleTable" class="table"></div></main><script src="report_data.js"></script><script>
(()=>{const D=window.V4_4_S3_STAGE,$=x=>document.getElementById(x),pct=v=>v==null?'—':(100*Number(v)).toFixed(3)+'%',num=v=>v==null?'—':Number(v).toFixed(2),flag=v=>v?'<span class="yes">通过</span>':'<span class="no">未通过</span>',m=$('method'),s=$('speed'),v=$('view');s.innerHTML='<option value="all">全部</option>'+D.speedWindows.map(x=>`<option value="${x}">${x} bar / ${x/4} 分钟</option>`).join('');v.value=D.scenario3QualifiedCount?'qualified':'all';function render(){let methodRows=D.rows.filter(r=>r.method===m.value),rows=methodRows;if(s.value!=='all')rows=rows.filter(r=>Number(r.speed_window_bars)===Number(s.value));if(v.value==='qualified')rows=rows.filter(r=>r.scenario_3_qualified);rows.sort((a,b)=>Number(a.scenario_3_diagnostic_rank)-Number(b.scenario_3_diagnostic_rank));const methodQualified=methodRows.filter(r=>r.scenario_3_qualified).length;$('cards').innerHTML=[["当前显示",rows.length],["本方法情景三合格",methodQualified],["阶段总坐标",D.coordinateCount],["阶段总交易",D.tradeCount]].map(x=>`<div class="card"><span class="muted">${x[0]}</span><b>${x[1]}</b></div>`).join('');$('note').textContent=D.note+(D.scenario3QualifiedCount===0?' 本阶段情景三合格数为 0；诊断排名只用于解释失败，不构成候选或参数接受。':'');$('table').innerHTML='<table><thead><tr><th>组合</th><th>诊断序</th><th>S</th><th>情景三分段</th><th>失败行情</th><th>情景一</th><th>情景二</th><th>情景三</th><th>E</th><th>BH</th><th>TRW</th><th>K</th><th>W</th><th>M</th><th>总收益</th><th>缺口剔除收益</th><th>笔均</th><th>最大回撤</th><th>交易数</th><th>回撤退出</th><th>速度退出</th><th>逐笔</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${r.combo_id}</td><td>${r.scenario_3_diagnostic_rank}</td><td>${r.speed_window_bars}</td><td>${r.scenario_3_qualified_segment_count}/3</td><td>${r.scenario_3_failed_segment_ids||'—'}</td><td>${flag(r.scenario_1_qualified)}</td><td>${flag(r.scenario_2_qualified)}</td><td>${flag(r.scenario_3_qualified)}</td><td>${r.e}</td><td>${r.bh}</td><td>${r.trw}</td><td>${r.k}</td><td>${r.w}</td><td>${r.m}</td><td>${pct(r.train_return)}</td><td>${pct(r.train_return_excluding_gap_spanning_trades)}</td><td>${pct(r.train_avg_trade)}</td><td>${pct(r.train_max_drawdown_abs)}</td><td>${r.train_trade_count}</td><td>${r.rebound_exit_count}</td><td>${r.speed_exit_count}</td><td><a target="_blank" href="trade_review/index.html?combo_id=${encodeURIComponent(r.combo_id)}">查看</a></td></tr>`).join('')+'</tbody></table>';const sr=D.scenarioRows.filter(r=>r.method===m.value);$('scenarioTable').innerHTML='<table><thead><tr><th>情景</th><th>合格坐标</th><th>坐标数</th></tr></thead><tbody>'+sr.map(r=>`<tr><td>${r.scenario_id}</td><td>${r.qualified_coordinate_count}</td><td>${r.coordinate_count}</td></tr>`).join('')+'</tbody></table>';const gr=D.segmentRows.filter(r=>r.method===m.value);$('segmentTable').innerHTML='<table><thead><tr><th>行情</th><th>合格坐标</th><th>坐标数</th><th>区间开仓合计</th><th>区间平仓合计</th></tr></thead><tbody>'+gr.map(r=>`<tr><td>${r.segment_id}</td><td>${r.qualified_coordinate_count}</td><td>${r.coordinate_count}</td><td>${r.entry_count_in_interval}</td><td>${r.exit_count_in_interval}</td></tr>`).join('')+'</tbody></table>';let er=D.exitRows.filter(r=>r.method===m.value);if(s.value!=='all')er=er.filter(r=>Number(r.speed_window_bars)===Number(s.value));$('exitTable').innerHTML='<table><thead><tr><th>S</th><th>原因</th><th>交易数</th><th>平均收益</th></tr></thead><tbody>'+er.map(r=>`<tr><td>${r.speed_window_bars}</td><td>${r.exit_reason}</td><td>${r.trade_count}</td><td>${pct(r.mean_return)}</td></tr>`).join('')+'</tbody></table>';let lr=D.lifecycleRows.filter(r=>r.method===m.value);if(s.value!=='all')lr=lr.filter(r=>Number(r.speed_window_bars)===Number(s.value));$('lifecycleTable').innerHTML='<table><thead><tr><th>S</th><th>退出原因</th><th>交易数</th><th>持仓 bar 中位</th><th>持仓 bar P95</th><th>持仓 bar 最大</th><th>持仓分钟中位</th><th>持仓分钟 P95</th><th>持仓分钟最大</th><th>等待成交</th><th>跨缺口</th></tr></thead><tbody>'+lr.map(r=>`<tr><td>${r.speed_window_bars}</td><td>${r.exit_reason}</td><td>${r.trade_count}</td><td>${num(r.holding_bar_distance_median)}</td><td>${num(r.holding_bar_distance_p95)}</td><td>${r.holding_bar_distance_maximum}</td><td>${num(r.holding_minutes_median)}</td><td>${num(r.holding_minutes_p95)}</td><td>${num(r.holding_minutes_maximum)}</td><td>${r.waited_entry_count}</td><td>${r.gap_spanning_trade_count}</td></tr>`).join('')+'</tbody></table>'}m.onchange=render;s.onchange=render;v.onchange=render;render()})();</script></body></html>"""


def _legacy_v4_main_script() -> str:
    return r"""
const DATA=window.V4_ANALYSIS_DATA;
(()=>{
  'use strict';
  if(!DATA){document.body.textContent='分析资料未加载';return;}
  let method='rolling_tr_sum',baselinePolicy=(DATA.baselineSamplingPolicies||[]).includes('all_window')?'all_window':(DATA.baselineSamplingPolicies||['all_window'])[0],scenarioFilter='all',metric='total_return',countFilter='ge10',countGreaterThan='',countLessThan='',returnView='cost_adjusted',sortKey='rank',sortDir=1,page=1;
  const PAGE_SIZE=500;
  const selectedRanges={bh:new Set(),e:new Set(),w:new Set(),speed_window_bars:new Set()};
  const $=id=>document.getElementById(id);
  const esc=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const number=value=>value==null||Number.isNaN(Number(value))?'—':Number(value).toLocaleString(undefined,{maximumFractionDigits:4});
  const percentage=value=>value==null||Number.isNaN(Number(value))?'—':(Number(value)*100).toFixed(3)+'%';
  const text=value=>value===true?'是':value===false?'否':String(value??'—');
  const methods={rolling_tr_sum:'滚动 TR 总和均值'};
  const baselinePolicies=[['all_window','全部'],['exclude_marked','排除标记'],['confirmed_low_activity_gate','确认后过滤']].filter(item=>(DATA.baselineSamplingPolicies||[]).includes(item[0]));
  const scenarios=[['all','全部坐标'],['scenario_1','情景一'],['scenario_2','情景二'],['scenario_3','情景三']];
  const entryBaselineRanges=[
    ['all','全部',null,null],['lt45','＜45 分钟',null,45],['45_60','45–＜60 分钟',45,60],
    ['60_90','60–＜90 分钟',60,90],['90_150','90–＜150 分钟',90,150],['gte150','≥150 分钟',150,null]
  ];
  const entryMarketRanges=[
    ['all','全部',null,null],['lt30','＜30 分钟',null,30],['30_60','30–＜60 分钟',30,60],
    ['60_90','60–＜90 分钟',60,90],['90_120','90–＜120 分钟',90,120],['gte120','≥120 分钟',120,null]
  ];
  const exitBaselineRanges=[
    ['all','全部',null,null],['lt1','＜1 分钟',null,1],['1_2','1–＜2 分钟',1,2],
    ['2_5','2–＜5 分钟',2,5],['5_15','5–＜15 分钟',5,15],['15_30','15–＜30 分钟',15,30],['gte30','≥30 分钟',30,null]
  ];
  const speedRanges=[
    ['all','全部',null,null],
    ['lt5','＜5 分钟',null,5],
    ['5_15','5–＜15 分钟',5,15],
    ['15_30','15–＜30 分钟',15,30],
    ['30_60','30–＜60 分钟',30,60],
    ['60_120','60–＜120 分钟',60,120],
    ['gte120','≥120 分钟',120,null]
  ];
  const metrics=[
    ['total_return','总收益','train_return','train_cost_adjusted_return','desc'],
    ['average_trade','笔均收益','train_avg_trade','train_cost_adjusted_avg_trade','desc']
  ];
  const returnViews=[['cost_adjusted','手续费／滑点后'],['gross','无手续费／滑点']];
  const counts=[['all','不限',0],['ge10','至少 10 笔',10],['ge20','至少 20 笔',20]];
  const highReturnViews=Array.isArray(DATA.highReturnViews)?DATA.highReturnViews:[];
  const cm=DATA.costModel||{};
  const sourceRows=Array.isArray(DATA.rows)?DATA.rows:[];
  const sourceIndexes=sourceRows.map((_,index)=>index);
  const indexesByMethodPolicy=new Map();
  sourceIndexes.forEach(index=>{
    const row=sourceRows[index],key=`${row.method}\u0000${row.baseline_sampling_policy}`;
    if(!indexesByMethodPolicy.has(key))indexesByMethodPolicy.set(key,[]);
    indexesByMethodPolicy.get(key).push(index);
  });
  const currentRanks=new Int32Array(sourceRows.length);
  const percentageKeys=new Set(['train_return','train_avg_trade','train_max_drawdown_abs','train_cost_adjusted_return','train_cost_adjusted_avg_trade','train_cost_adjusted_max_drawdown_abs','train_return_excluding_gap_spanning_trades']);
  const booleanKeys=new Set(['scenario_1_qualified','scenario_2_qualified','scenario_3_qualified']);
  const columns=()=>[
    ['rank','排名'],['speed_window_bars','S'],['e','E'],['bh','BH'],['trw','TRW'],['k','K'],['w','W'],['m','M'],
    ...(returnView==='gross'?[['train_return','总收益'],['train_avg_trade','笔均'],['train_max_drawdown_abs','回撤']]:[['train_cost_adjusted_return','总收益'],['train_cost_adjusted_avg_trade','笔均'],['train_cost_adjusted_max_drawdown_abs','回撤']]),
    ['train_trade_count','笔数'],
    ['rebound_exit_count','回撤退出'],['speed_exit_count','速度退出'],
    ['holding_bar_distance_median','持仓 bar 中位'],['holding_bar_distance_p95','持仓 bar P95'],
    ['scenario_1_qualified','情景一'],['scenario_2_qualified','情景二'],['scenario_3_qualified','情景三'],
    ['scenario_3_qualified_segment_count','情景三分段'],['scenario_3_failed_segment_ids','失败行情'],['combo_id','参数组合'],
    ['round_trip_cost_bps','往返成本 bps'],['gap_spanning_trade_count','跨 gap 笔数']
  ];
  const label=(items,id)=>items.find(item=>item[0]===id)?.[1]||id;
  const metricDefinition=()=>metrics.find(item=>item[0]===metric)||metrics[0];
  const minimumTrades=()=>counts.find(item=>item[0]===countFilter)?.[2]||0;
  const greaterThanTrades=()=>countGreaterThan===''?null:Number(countGreaterThan);
  const lessThanTrades=()=>countLessThan===''?null:Number(countLessThan);
  const tradeCountLabel=()=>[label(counts,countFilter),countGreaterThan===''?'':`大于 ${countGreaterThan} 笔`,countLessThan===''?'':`小于 ${countLessThan} 笔`].filter(Boolean).join(' · ');
  const scenarioQualifies=row=>scenarioFilter==='all'||Boolean(row[scenarioFilter+'_qualified']);
  const rangeQualifies=(row,field,definitions)=>{
    const selected=selectedRanges[field];
    if(!selected||selected.size===0)return true;
    const minutes=Number(row[field])/4;
    if(!Number.isFinite(minutes))return false;
    return [...selected].some(id=>{
      const definition=definitions.find(([rangeId])=>rangeId===id);
      if(!definition)return false;
      const min=definition[2],max=definition[3];
      return (min==null||minutes>=min)&&(max==null||minutes<max);
    });
  };
  const selectedLabel=(field,definitions)=>{
    const selected=selectedRanges[field];
    return !selected||selected.size===0?'全部':[...selected].map(id=>label(definitions,id)).join('、');
  };
  const baselinePolicyQualifies=row=>row.baseline_sampling_policy===baselinePolicy;
  const route=row=>String(DATA.nativeTradeRoute||'trade_review/index.html?combo_id={combo_id}').replace('{combo_id}',encodeURIComponent(row.combo_id));
  const methodPolicyIndexes=()=>indexesByMethodPolicy.get(`${method}\u0000${baselinePolicy}`)||[];
  function rankedIndexes(){
    const definition=metricDefinition(),key=returnView==='gross'?definition[2]:definition[3],direction=definition[4]==='asc'?1:-1;
    const indexes=[];
    for(const index of methodPolicyIndexes()){
      const row=sourceRows[index];
      const tradeCount=Number(row.train_trade_count),greaterThan=greaterThanTrades(),lessThan=lessThanTrades();
      if(baselinePolicyQualifies(row)&&scenarioQualifies(row)&&rangeQualifies(row,'bh',entryBaselineRanges)&&rangeQualifies(row,'e',entryMarketRanges)&&rangeQualifies(row,'w',exitBaselineRanges)&&rangeQualifies(row,'speed_window_bars',speedRanges)&&tradeCount>=minimumTrades()&&(greaterThan==null||tradeCount>greaterThan)&&(lessThan==null||tradeCount<lessThan)&&Number.isFinite(Number(row[key])))indexes.push(index);
    }
    indexes.sort((a,b)=>direction*(Number(sourceRows[a][key])-Number(sourceRows[b][key]))||String(sourceRows[a].combo_id).localeCompare(String(sourceRows[b].combo_id)));
    currentRanks.fill(0);
    indexes.forEach((rowIndex,rankIndex)=>{currentRanks[rowIndex]=rankIndex+1;});
    return indexes;
  }
  function value(row,key,rowIndex){
    if(key==='rank')return `<a class="rank-link" data-combo-id="${esc(row.combo_id)}" target="_blank" rel="noopener" href="${route(row)}">#${currentRanks[rowIndex]}</a>`;
    if(key==='combo_id')return `<a class="combo" target="_blank" rel="noopener" title="${esc(row.combo_id)}" aria-label="${esc(row.combo_id)}" href="${route(row)}">${esc(row.combo_id)}</a>`;
    if(booleanKeys.has(key))return `<span class="${row[key]?'yes':'no'}">${text(row[key])}</span>`;
    if(key==='scenario_3_qualified_segment_count')return `${number(row[key])}/3`;
    if(percentageKeys.has(key))return percentage(row[key]);
    return key==='scenario_3_failed_segment_ids'?esc(row[key]||'—'):number(row[key]);
  }
  function controlButtons(container,items,current,dataName){
    $(container).innerHTML=items.map(([id,currentLabel])=>`<button type="button" class="${current===id?'active':''}" aria-pressed="${current===id}" data-${dataName}="${id}">${esc(currentLabel)}</button>`).join('');
  }
  function rangeButtons(container,items,field,dataName){
    const selected=selectedRanges[field],allActive=selected.size===0;
    $(container).innerHTML=items.map(([id,currentLabel])=>{const active=id==='all'?allActive:selected.has(id);return `<button type="button" class="${active?'active':''}" aria-pressed="${active}" data-${dataName}="${id}">${esc(currentLabel)}</button>`;}).join('');
  }
  function toggleRange(field,id){
    const selected=selectedRanges[field];
    if(id==='all'){selected.clear();}
    else if(selected.has(id)){selected.delete(id);}else{selected.add(id);}
    resetRanking();
  }
  function bindControls(){
    controlButtons('scenario-controls',scenarios,scenarioFilter,'scenario');
    controlButtons('baseline-policy-controls',baselinePolicies,baselinePolicy,'baseline-policy');
    rangeButtons('entry-baseline-controls',entryBaselineRanges,'bh','entry-baseline-range');
    rangeButtons('entry-market-controls',entryMarketRanges,'e','entry-market-range');
    rangeButtons('exit-baseline-controls',exitBaselineRanges,'w','exit-baseline-range');
    rangeButtons('speed-controls',speedRanges,'speed_window_bars','speed-range');
    controlButtons('metric-controls',metrics,metric,'metric');
    controlButtons('count-controls',counts,countFilter,'count');
    const greaterInput=$('count-greater-than'),lessInput=$('count-less-than');
    greaterInput.value=countGreaterThan;
    lessInput.value=countLessThan;
    greaterInput.onchange=()=>{countGreaterThan=greaterInput.value;resetRanking();};
    lessInput.onchange=()=>{countLessThan=lessInput.value;resetRanking();};
    controlButtons('return-view-controls',returnViews,returnView,'return-view');
    const highReturnContainer=$('high-return-view-controls');
    if(highReturnContainer){
      highReturnContainer.innerHTML=highReturnViews.map(view=>`<button type="button" class="${scenarioFilter===view.scenario_filter&&metric===view.metric&&minimumTrades()===Number(view.minimum_trade_count)&&countGreaterThan===''&&countLessThan===''?'active':''}" aria-pressed="${scenarioFilter===view.scenario_filter&&metric===view.metric&&minimumTrades()===Number(view.minimum_trade_count)&&countGreaterThan===''&&countLessThan===''}" data-high-return-view="${esc(view.id)}">${esc(view.label)}</button>`).join('');
      document.querySelectorAll('[data-high-return-view]').forEach(node=>node.onclick=()=>{
        const view=highReturnViews.find(item=>item.id===node.dataset.highReturnView);
        if(!view)return;
        scenarioFilter=view.scenario_filter;
        metric=view.metric;
        countFilter=Number(view.minimum_trade_count)>0?`ge${Number(view.minimum_trade_count)}`:'all';
        countGreaterThan='';
        countLessThan='';
        resetRanking();
      });
    }
    const scenarioLink=$('scenario-requirements-link');
    if(scenarioLink){
      const pattern=String(DATA.scenarioRequirementsRoute||'scenario_requirements/index.html?scenario={scenario_id}');
      scenarioLink.href=pattern.replace('{scenario_id}',encodeURIComponent(scenarioFilter));
      scenarioLink.setAttribute('aria-label',`查看${label(scenarios,scenarioFilter)}要求`);
    }
    document.querySelectorAll('[data-scenario]').forEach(node=>node.onclick=()=>{scenarioFilter=node.dataset.scenario;resetRanking();});
    document.querySelectorAll('[data-baseline-policy]').forEach(node=>node.onclick=()=>{baselinePolicy=node.dataset.baselinePolicy;resetRanking();});
    document.querySelectorAll('[data-entry-baseline-range]').forEach(node=>node.onclick=()=>toggleRange('bh',node.dataset.entryBaselineRange));
    document.querySelectorAll('[data-entry-market-range]').forEach(node=>node.onclick=()=>toggleRange('e',node.dataset.entryMarketRange));
    document.querySelectorAll('[data-exit-baseline-range]').forEach(node=>node.onclick=()=>toggleRange('w',node.dataset.exitBaselineRange));
    document.querySelectorAll('[data-speed-range]').forEach(node=>node.onclick=()=>toggleRange('speed_window_bars',node.dataset.speedRange));
    document.querySelectorAll('[data-metric]').forEach(node=>node.onclick=()=>{metric=node.dataset.metric;resetRanking();});
    document.querySelectorAll('[data-count]').forEach(node=>node.onclick=()=>{countFilter=node.dataset.count;resetRanking();});
    document.querySelectorAll('[data-return-view]').forEach(node=>node.onclick=()=>{returnView=node.dataset.returnView;sortKey='rank';sortDir=1;page=1;render();});
  }
  function resetRanking(){sortKey='rank';sortDir=1;page=1;render();}
  function render(){
    bindControls();
    const allStrategyCount=$('all-strategy-count');
    if(allStrategyCount)allStrategyCount.textContent=`全部策略 ${number(DATA.coordinateCount)}`;
    const methodRows=methodPolicyIndexes(),ranked=rankedIndexes(),scenarioLabel=label(scenarios,scenarioFilter),baselinePolicyLabel=label(baselinePolicies,baselinePolicy),entryBaselineLabel=selectedLabel('bh',entryBaselineRanges),entryMarketLabel=selectedLabel('e',entryMarketRanges),exitBaselineLabel=selectedLabel('w',exitBaselineRanges),speedLabel=selectedLabel('speed_window_bars',speedRanges),definition=metricDefinition(),metricLabel=definition[1],metricKey=returnView==='gross'?definition[2]:definition[3],metricDirection=definition[4]==='asc'?'升序':'降序',countLabel=tradeCountLabel(),returnViewLabel=label(returnViews,returnView);
    $('cards').innerHTML=[
      ['可排参数组合',number(ranked.length),`${esc(baselinePolicyLabel)}策略 ${number(methodRows.length)} · ${esc(DATA.scopeLabel||'当前阶段')}总坐标 ${number(DATA.coordinateCount)}`],
      ['情景过滤',scenarioLabel,scenarioFilter==='all'?'不限制情景资格':'按已保存资格筛选'],
      ['窗口筛选',`BH ${entryBaselineLabel} · E ${entryMarketLabel} · W ${exitBaselineLabel} · S ${speedLabel}`,'组间取交集；组内所选区间取并集'],
      ['当前排序首位',ranked.length?percentage(sourceRows[ranked[0]][metricKey]):'真实空集',`${returnViewLabel} · ${metricLabel} · ${metricDirection} · ${countLabel}`]
    ].map(([cardLabel,current,detail])=>`<div class="metric"><span class="metric-label">${esc(cardLabel)}</span><b>${esc(current)}</b><small>${esc(detail)}</small></div>`).join('');
    const scenarioEvidence=DATA.scenario3QualifiedCount?`情景三合格 ${number(DATA.scenario3QualifiedCount)} 个坐标`:'情景三为真实空集';
    const costText=`${cm.instrument_name||cm.instrument_id} 参考价 ${number(cm.reference_price)} × ${number(cm.point_value)} ${cm.quote_currency}/点＝${number(cm.contract_notional_quote)} ${cm.quote_currency} 名义价值；${number(cm.round_trip_slippage_bps)} bps 往返滑点 + ${number(cm.round_trip_commission)} ${cm.commission_currency} 往返手续费，合计 ${number(cm.round_trip_total_cost_bps)} bps。`;
    $('status').innerHTML=`<strong>${esc(methods[method])} · ${esc(baselinePolicyLabel)} · ${esc(scenarioLabel)} · BH ${esc(entryBaselineLabel)} · E ${esc(entryMarketLabel)} · W ${esc(exitBaselineLabel)} · S ${esc(speedLabel)} · ${esc(returnViewLabel)}${esc(metricLabel)}${esc(metricDirection)} · ${esc(countLabel)}</strong><span class="summary-detail">排序与显示均采用 ${esc(returnViewLabel)}；默认采用手续费／滑点后口径。成本模型：${esc(costText)} 筛选后 ${number(ranked.length)} / ${number(methodRows.length)} 个参数组合；${esc(DATA.scopeLabel||'当前阶段')}共 ${number(DATA.coordinateCount)} 个坐标、${number(DATA.tradeCount)} 笔逐笔交易。${esc(scenarioEvidence)}，不代表参数接受。</span>`;
    let rows=[...ranked];
    rows.sort((a,b)=>{
      const rowA=sourceRows[a],rowB=sourceRows[b],av=sortKey==='rank'?currentRanks[a]:rowA[sortKey],bv=sortKey==='rank'?currentRanks[b]:rowB[sortKey];
      if(av==null&&bv==null)return String(rowA.combo_id).localeCompare(String(rowB.combo_id));
      if(av==null)return 1;if(bv==null)return -1;
      if(typeof av==='string'||typeof bv==='string')return sortDir*String(av).localeCompare(String(bv));
      return sortDir*(Number(av)-Number(bv));
    });
    const pager=$('pager');
    if(!rows.length){pager.hidden=true;$('table').innerHTML='<div class="empty"><h2>当前条件形成真实空集</h2><p>可调整情景、四组窗口区间或交易数；已保存回测结果保持不变。</p></div>';return;}
    const pageCount=Math.ceil(rows.length/PAGE_SIZE);
    page=Math.min(Math.max(1,page),pageCount);
    const pageStart=(page-1)*PAGE_SIZE,pageEnd=Math.min(pageStart+PAGE_SIZE,rows.length),pageRows=rows.slice(pageStart,pageEnd);
    const visibleColumns=columns();
    $('table').innerHTML='<table><thead><tr>'+visibleColumns.map(([columnKey,columnLabel])=>`<th><button type="button" data-sort="${columnKey}">${columnLabel}${sortKey===columnKey?(sortDir===1?' ▲':' ▼'):''}</button></th>`).join('')+'</tr></thead><tbody>'+pageRows.map(rowIndex=>{const row=sourceRows[rowIndex];return '<tr>'+visibleColumns.map(([columnKey])=>`<td>${value(row,columnKey,rowIndex)}</td>`).join('')+'</tr>';}).join('')+'</tbody></table>';
    $('table').scrollTop=0;
    pager.hidden=pageCount<=1;
    $('page-status').textContent=`第 ${number(page)} / ${number(pageCount)} 页 · ${number(pageStart+1)}–${number(pageEnd)} / ${number(rows.length)} · 每页 500 行`;
    $('page-prev').disabled=page===1;
    $('page-next').disabled=page===pageCount;
    $('page-prev').onclick=()=>{if(page>1){page-=1;render();}};
    $('page-next').onclick=()=>{if(page<pageCount){page+=1;render();}};
    document.querySelectorAll('[data-sort]').forEach(node=>node.onclick=()=>{const key=node.dataset.sort;sortDir=sortKey===key?-sortDir:1;sortKey=key;page=1;render();});
  }
  $('method').onchange=()=>{method=$('method').value;resetRanking();};
  render();
  const controlsPanel=$('controls'),controlToggle=$('control-toggle');
  function setControlsCollapsed(collapsed){
    controlsPanel.classList.toggle('is-collapsed',collapsed);
    controlToggle.setAttribute('aria-expanded',String(!collapsed));
    controlToggle.setAttribute('aria-label',collapsed?'Expand filters and sorting':'Collapse filters and sorting');
  }
  setControlsCollapsed(false);
  controlToggle.onclick=()=>setControlsCollapsed(!controlsPanel.classList.contains('is-collapsed'));
  function applyTheme(dark){document.documentElement.dataset.theme=dark?'dark':'';localStorage.setItem('v4-unified-theme',dark?'dark':'light');$('theme').textContent=dark?'浅色 Light':'深色 Dark';}
  applyTheme(localStorage.getItem('v4-unified-theme')==='dark');
  $('theme').onclick=()=>applyTheme(document.documentElement.dataset.theme!=='dark');
})();
"""


def _legacy_v4_main_html(cross_instrument_href: str | None = None) -> str:
    if (
        not LEGACY_V4_MAIN_TEMPLATE_PATH.is_file()
        or sha256_file(LEGACY_V4_MAIN_TEMPLATE_PATH)
        != LEGACY_V4_MAIN_TEMPLATE_SHA256
    ):
        raise ValueError("historical V4 main-page template failed hash validation")
    if (
        not LEGACY_V4_TRADE_DESIGN_PATH.is_file()
        or sha256_file(LEGACY_V4_TRADE_DESIGN_PATH)
        != LEGACY_V4_TRADE_DESIGN_SHA256
    ):
        raise ValueError("historical V4 trade-page design failed hash validation")
    template = LEGACY_V4_MAIN_TEMPLATE_PATH.read_text(encoding="utf-8")
    marker = '<script src="analysis_data.js"></script><script>'
    prefix, separator, _ = template.partition(marker)
    if not separator:
        raise ValueError("historical V4 main-page template lacks its data-script marker")
    replacements = {
        "<title>V4 统一综合分析</title>": "<title>V4.41 K200回测结果排序</title>",
        "<h1>V4 统一综合分析</h1>": (
            '<h1>V4.41 K200回测结果排序</h1>'
            '<div id="all-strategy-count" class="all-strategy-count" '
            'aria-live="polite">全部策略 —</div>'
        ),
        "--accent:#185fb8;": "--accent:#1554a3;",
        "--soft:#eaf3ff;": "--soft:#e2efff;",
        "--accent:#72b7ff;": "--accent:#5ea7ef;",
        "--soft:#172a3e;": "--soft:#14263a;",
        "情景、策略范围、排序指标与交易数门槛彼此独立；每次选择都形成可追溯的训练样本排名。": (
            "情景、四组窗口区间、排序指标与交易数门槛彼此独立；每次选择都形成可追溯的训练样本排名。"
        ),
        "仅筛选事件资格，不改变收益指标。": "仅筛选已保存情景资格，不改变收益指标。",
        '<div id="scenario-controls" class="segmented" role="group" aria-label="情景过滤"></div>': (
            '<div class="scenario-control-row">'
            '<div id="scenario-controls" class="segmented" role="group" aria-label="情景过滤"></div>'
            '<a id="scenario-requirements-link" class="scenario-requirements-link" '
            'target="_blank" rel="noopener">查看</a></div>'
        ),
        '<option value="tr_average">TR 平均值</option>': "",
        '<option value="rolling_tr_sum">滚动 TR 总和</option>': (
            '<option value="rolling_tr_sum">滚动 TR 总和均值</option>'
        ),
        "两种方法维持独立排名。": "仅使用滚动 TR 总和均值。",
        '</fieldset><fieldset class="control-group scenario">': (
            '</fieldset><fieldset class="control-group baseline-policy">'
            '<legend>开仓基准采样</legend><div id="baseline-policy-controls" '
            'class="segmented" role="group" aria-label="开仓基准采样"></div>'
            '<span class="control-note">全部为默认；排除标记单独排名。</span></fieldset>'
            '<fieldset class="control-group scenario">'
        ),
        "策略范围": "速度窗口",
        'id="slice-controls"': 'id="speed-controls"',
        "3–15 分钟短跌要求 E 与 W 同时属于规定集合。": (
        "按 S ÷ 4 所在分钟区间筛选；区间左闭右开，筛选不改变交易结果。"
        ),
        '<fieldset class="control-group count-axis"><legend>最低交易数</legend>'
        '<div id="count-controls" class="segmented" role="group" aria-label="最低交易数"></div>'
        '<span class="control-note">在其他过滤条件之后应用门槛。</span></fieldset>': (
            '<fieldset class="control-group count-axis"><legend>交易数</legend>'
            '<div class="count-filter-row"><div id="count-controls" class="segmented" '
            'role="group" aria-label="交易数"></div><div class="count-range-inputs">'
            '<label>大于 <input id="count-greater-than" type="number" min="0" step="1" '
            'inputmode="numeric" aria-label="交易数大于"> 笔</label>'
            '<label>小于 <input id="count-less-than" type="number" min="0" step="1" '
            'inputmode="numeric" aria-label="交易数小于"> 笔</label></div></div>'
            '<span class="control-note">在其他过滤条件之后应用；两个数字框可同时使用。</span></fieldset>'
            '<fieldset class="control-group return-view-axis"><legend>收益显示</legend>'
            '<div id="return-view-controls" class="segmented" role="group" '
            'aria-label="收益显示"></div><span class="control-note">'
            '排序与显示使用同一口径；默认采用手续费／滑点后。</span></fieldset>'
            '<fieldset class="control-group high-return-axis"><legend>高收益四视图</legend>'
            '<div id="high-return-view-controls" class="segmented" role="group" '
            'aria-label="高收益四视图"></div><span class="control-note">'
            '最少 10／20 笔与总收益／笔均收益形成四个独立排名。</span></fieldset>'
        ),
        '<div id="table" class="table-wrap"></div>': (
            '<div id="table" class="table-wrap"></div>'
            '<nav id="pager" class="pager" aria-label="Table pagination" hidden>'
            '<button id="page-prev" type="button" aria-label="Previous page">‹</button>'
            '<span id="page-status" class="pager-status" aria-live="polite"></span>'
            '<button id="page-next" type="button" aria-label="Next page">›</button>'
            '</nav>'
        ),
        '<section id="contract-panel" class="contract-panel"><div class="contract-head"><h2>研究合同</h2></div><div id="contract-table" class="table-wrap"></div></section>': "",
    }
    for old, new in replacements.items():
        prefix = prefix.replace(old, new)
    speed_fieldset = (
        '<fieldset class="control-group slice"><legend>速度窗口</legend>'
        '<div id="speed-controls" class="segmented" role="group" aria-label="速度窗口"></div>'
        '<span class="control-note">按 S ÷ 4 所在分钟区间筛选；区间左闭右开，筛选不改变交易结果。</span></fieldset>'
    )
    window_fieldsets = (
        '<fieldset class="control-group window-filter entry-baseline"><legend>开仓基准窗口（BH）</legend>'
        '<div id="entry-baseline-controls" class="segmented" role="group" aria-label="开仓基准窗口"></div>'
        '<span class="control-note">按 BH ÷ 4 转为分钟；可同时选择多个区间。</span></fieldset>'
        '<fieldset class="control-group window-filter entry-market"><legend>开仓行情窗口（E）</legend>'
        '<div id="entry-market-controls" class="segmented" role="group" aria-label="开仓行情窗口"></div>'
        '<span class="control-note">按 E ÷ 4 转为分钟；可同时选择多个区间。</span></fieldset>'
        '<fieldset class="control-group window-filter exit-baseline"><legend>平仓基准窗口（W）</legend>'
        '<div id="exit-baseline-controls" class="segmented" role="group" aria-label="平仓基准窗口"></div>'
        '<span class="control-note">按 W ÷ 4 转为分钟；可同时选择多个区间。</span></fieldset>'
        '<fieldset class="control-group window-filter exit-market"><legend>平仓行情窗口（S）</legend>'
        '<div id="speed-controls" class="segmented" role="group" aria-label="平仓行情窗口"></div>'
        '<span class="control-note">按 S ÷ 4 转为分钟；可同时选择多个区间。组内取并集，四组之间取交集。</span></fieldset>'
    )
    if speed_fieldset not in prefix:
        raise ValueError("historical V4 main-page transformed speed fieldset is missing")
    prefix = prefix.replace(speed_fieldset, window_fieldsets, 1)
    control_head = (
        '<div class="control-head"><div><h2 id="control-title">筛选与排序</h2>'
        '<p>过滤条件只改变成员集合；排序指标只决定集合内的名次。</p></div></div>'
    )
    collapsible_control_head = (
        '<div class="control-head"><div><h2 id="control-title">筛选与排序</h2>'
        '<p>过滤条件只改变成员集合；排序指标只决定集合内的名次。</p></div>'
        '<button id="control-toggle" class="control-toggle" type="button" '
        'aria-expanded="true" aria-controls="control-grid status" '
        'aria-label="Collapse filters and sorting">'
        '<svg class="control-toggle-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
        '<path d="m6 9 6 6 6-6"></path></svg></button></div>'
    )
    if control_head not in prefix:
        raise ValueError("historical V4 main-page control heading is missing")
    prefix = prefix.replace(control_head, collapsible_control_head, 1)
    if '<div class="control-grid">' not in prefix:
        raise ValueError("historical V4 main-page control grid is missing")
    prefix = prefix.replace(
        '<div class="control-grid">',
        '<div id="control-grid" class="control-grid">',
        1,
    )
    scenario_style = (
        ".shell{max-width:none;padding-bottom:20px}"
        ".control-group.baseline-policy{grid-column:span 4}"
        ".scenario-control-row{display:flex;align-items:flex-start;gap:8px;flex-wrap:wrap}"
        ".scenario-control-row .segmented{flex:1 1 auto}"
        ".scenario-requirements-link{display:inline-flex;align-items:center;justify-content:center;"
        "min-height:36px;min-width:72px;padding:7px 12px;border:1px solid var(--strong);"
        "border-radius:7px;background:var(--panel);color:var(--accent);font-weight:720;"
        "text-decoration:none}"
        ".scenario-requirements-link:hover{border-color:var(--accent);background:var(--soft)}"
        ".scenario-requirements-link:focus-visible{outline:3px solid var(--focus);outline-offset:2px}"
        ".control-toggle{display:inline-flex;align-items:center;justify-content:center;"
        "width:40px;height:36px;padding:0;border:1px solid var(--strong);border-radius:7px;"
        "background:var(--panel);color:var(--ink);cursor:pointer;flex:0 0 auto}"
        ".control-toggle:hover{border-color:var(--accent)}"
        ".control-toggle:focus-visible{outline:3px solid var(--focus);outline-offset:2px}"
        ".control-toggle-icon{width:20px;height:20px;fill:none;stroke:currentColor;stroke-width:2.25;"
        "stroke-linecap:round;stroke-linejoin:round;transform:none}"
        ".control-toggle[aria-expanded=true] .control-toggle-icon{transform:rotate(180deg)}"
        ".count-filter-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}"
        ".count-range-inputs{display:flex;align-items:center;gap:8px;flex-wrap:wrap}"
        ".count-range-inputs label{display:flex;align-items:center;gap:5px;white-space:nowrap;color:var(--ink)}"
        ".count-range-inputs input{width:82px;min-height:36px;padding:7px 8px;border:1px solid var(--strong);border-radius:7px;background:var(--panel);color:var(--ink);font:inherit}"
        ".count-range-inputs input:hover{border-color:var(--accent)}"
        ".count-range-inputs input:focus-visible{outline:3px solid var(--focus);outline-offset:2px;border-color:var(--accent)}"
        ".rank-link{width:90px}"
        ".pager{display:flex;align-items:center;justify-content:flex-end;gap:10px;padding:10px 2px 0}"
        ".pager[hidden]{display:none}"
        ".pager button{display:inline-flex;align-items:center;justify-content:center;width:36px;height:34px;padding:0;border:1px solid var(--strong);border-radius:7px;background:var(--panel);color:var(--ink);font-size:20px;line-height:1;cursor:pointer}"
        ".pager button:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}"
        ".pager button:focus-visible{outline:3px solid var(--focus);outline-offset:2px}"
        ".pager button:disabled{cursor:default;opacity:.42}"
        ".pager-status{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}"
        ".control-panel.is-collapsed .control-head{align-items:center;border-bottom:0}"
        ".control-panel.is-collapsed .control-head p,.control-panel.is-collapsed .control-grid,.control-panel.is-collapsed .summary{display:none}"
        "#table{height:calc(100vh - 58px);max-height:none;overflow:auto;contain:layout paint}"
        "#table th{top:0;z-index:3;box-shadow:0 1px 0 var(--line)}"
        "#table th,#table th:first-child,#table th:last-child{text-align:center}"
        ".control-group.return-view-axis{grid-column:span 12}"
        ".control-group.high-return-axis{grid-column:span 12}"
        ".control-group.window-filter{grid-column:1/-1}"
        "#entry-baseline-controls,#entry-market-controls,#exit-baseline-controls,#speed-controls{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:6px}"
        "#entry-baseline-controls button,#entry-market-controls button,#exit-baseline-controls button,#speed-controls button{width:100%;min-width:0;white-space:nowrap}"
        ".cards{display:none}"
        ".all-strategy-count{margin-top:6px;color:var(--ink);font-size:14px;font-weight:760}"
        "@media(max-width:1050px){.control-group.window-filter{grid-column:1/-1}}"
        "@media(max-width:720px){#entry-baseline-controls,#entry-market-controls,#exit-baseline-controls,#speed-controls{grid-template-columns:repeat(2,minmax(0,1fr))}"
        ".control-panel.is-collapsed .control-head{align-items:center;flex-direction:row;gap:10px}"
        ".control-panel.is-collapsed .control-head h2{white-space:nowrap}}"
        "@media(max-width:620px){.scenario-requirements-link{flex:1 1 100%}}"
    )
    prefix = prefix.replace("</style>", scenario_style + "</style>", 1)
    if cross_instrument_href is not None:
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", cross_instrument_href):
            raise ValueError("cross-instrument navigation href is unsafe")
        navigation_style = (
            ".page-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}"
            ".cross-instrument-link{display:inline-flex;align-items:center;justify-content:center;"
            "min-height:36px;padding:7px 11px;border:1px solid var(--strong);border-radius:7px;"
            "background:var(--panel);color:var(--accent);font-weight:720;text-decoration:none}"
            ".cross-instrument-link:hover{border-color:var(--accent);background:var(--soft)}"
            ".cross-instrument-link:focus-visible{outline:3px solid var(--focus);outline-offset:2px}"
            "@media(max-width:720px){.page-actions{align-items:stretch}.cross-instrument-link,.page-actions .theme{flex:1 1 auto}}"
        )
        prefix = prefix.replace("</style>", navigation_style + "</style>", 1)
        theme_button = '<button class="theme" id="theme" type="button">深色 Dark</button>'
        navigation = (
            '<div class="page-actions">'
            f'<a class="cross-instrument-link" href="{cross_instrument_href}" '
            'target="_blank" rel="noopener">跨品种对比</a>'
            + theme_button
            + "</div>"
        )
        if theme_button not in prefix:
            raise ValueError("transformed V4 main page lacks the theme button")
        prefix = prefix.replace(theme_button, navigation, 1)
    return prefix + marker + _legacy_v4_main_script() + "</script></body></html>"


def _trade_html() -> str:
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V4.4 逐笔分析</title><style>
:root{--bg:#f4f7fb;--p:#fff;--ink:#132238;--muted:#65758b;--line:#d9e2ec;--good:#087a4b;--bad:#a34722}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1600px;margin:auto;padding:20px 14px}h1{margin:0}.toolbar{margin:15px 0}select{max-width:100%;padding:8px;border:1px solid var(--line);border-radius:7px}.cards{display:flex;gap:9px;flex-wrap:wrap;margin:12px 0}.card,.state{background:var(--p);border:1px solid var(--line);border-radius:9px;padding:10px 12px}.card b{display:block;font-size:18px}.muted{color:var(--muted)}.table{overflow:auto;background:var(--p);border:1px solid var(--line);border-radius:9px}table{border-collapse:collapse;width:max-content;min-width:100%}th,td{padding:7px 9px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}th:first-child,td:first-child{text-align:left}.good{color:var(--good)}.bad{color:var(--bad)}@media(max-width:700px){main{padding:12px 8px}}</style></head><body><main><h1>V4.4 逐笔分析</h1><div class="muted">combined exit · 回撤优先检查 · 速度零延伸按当前 close 成交</div><div class="toolbar"><select id="combo"></select></div><div id="cards" class="cards"></div><div id="state" class="state">加载中</div><div id="table" class="table" style="margin-top:12px"></div></main><script src="trade_catalog.js"></script><script>
(()=>{const D=window.V4_4_TRADE_CATALOG,$=x=>document.getElementById(x),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),pct=v=>v==null?'—':(100*Number(v)).toFixed(3)+'%',num=v=>v==null?'—':Number(v).toFixed(5),sel=$('combo'),by=new Map(D.rows.map(r=>[r.combo_id,r]));sel.innerHTML=D.rows.map(r=>`<option value="${esc(r.combo_id)}">${esc(r.method)} · S${r.speed_window_bars} · E${r.e}/BH${r.bh}/TRW${r.trw}/K${r.k}/W${r.w}/M${r.m}</option>`).join('');const wanted=new URLSearchParams(location.search).get('combo_id');if(wanted&&by.has(wanted))sel.value=wanted;function render(p){const r=by.get(sel.value);$('cards').innerHTML=[["方法",r.method],["S",r.speed_window_bars],["总收益",pct(r.train_return)],["情景三",r.scenario_3_qualified?'通过':'未通过'],["交易数",r.train_trade_count]].map(x=>`<div class="card"><span class="muted">${x[0]}</span><b>${x[1]}</b></div>`).join('');$('table').innerHTML='<table><thead><tr><th>#</th><th>开仓</th><th>平仓</th><th>持仓 bar</th><th>持仓分钟</th><th>原因</th><th>收益</th><th>开仓价</th><th>平仓价</th><th>等待</th><th>触发价</th><th>基准</th><th>跌幅</th><th>活动低点</th><th>回撤阈值</th><th>回撤检查</th><th>S</th><th>速度参考时刻</th><th>参考低点</th><th>当前低点</th><th>延伸</th><th>跨缺口</th></tr></thead><tbody>'+p.trades.map((t,i)=>`<tr><td>${i+1}</td><td>${esc(t.entry_time)}</td><td>${esc(t.exit_time)}</td><td>${t.holding_bar_distance}</td><td>${num(t.holding_minutes)}</td><td>${esc(t.exit_reason)}</td><td class="${Number(t.return)>=0?'good':'bad'}">${pct(t.return)}</td><td>${num(t.entry_price)}</td><td>${num(t.exit_price)}</td><td>${t.entry_wait_bar_count}</td><td>${num(t.entry_trigger_price)}</td><td>${num(t.entry_baseline_value)}</td><td>${num(t.entry_drop_value)}</td><td>${num(t.active_low)}</td><td>${num(t.rebound_threshold)}</td><td>${num(t.rebound_check_price)} / ${esc(t.rebound_check_price_basis)}</td><td>${t.speed_window_bars}</td><td>${esc(t.speed_reference_time)}</td><td>${num(t.speed_reference_low)}</td><td>${num(t.speed_current_low)}</td><td>${num(t.speed_extension)}</td><td>${esc(t.position_crosses_real_gap)}</td></tr>`).join('')+'</tbody></table>';$('state').textContent=`已加载 ${p.trades.length} 笔交易`}function load(){window.V4_4_TRADE_CHUNK=null;const old=document.getElementById('chunk');if(old)old.remove();const r=by.get(sel.value),s=document.createElement('script');s.id='chunk';s.src='trade_chunks/'+r.chunk;s.onload=()=>{if(!window.V4_4_TRADE_CHUNK||window.V4_4_TRADE_CHUNK.comboId!==r.combo_id){$('state').textContent='逐笔资料身份不匹配';return}render(window.V4_4_TRADE_CHUNK)};s.onerror=()=>{$('state').textContent='逐笔资料加载失败'};document.body.appendChild(s);history.replaceState(null,'','?combo_id='+encodeURIComponent(r.combo_id))}sel.onchange=load;load()})();</script></body></html>"""


def analyze(
    plan_path: Path,
    stage_root: Path,
    analysis_root: Path,
    *,
    review_workers: int = 4,
) -> dict[str, Any]:
    plan_path, stage_root, analysis_root = (
        plan_path.resolve(), stage_root.resolve(), analysis_root.resolve()
    )
    loaded = _load_stage(stage_root, plan_path)
    stage_instrument = loaded["stage_manifest"].get("instrument_contract")
    instrument_profile = stage_instrument if isinstance(stage_instrument, dict) else None
    selected_cost_model = _cost_model_from_stage_manifest(loaded["stage_manifest"])
    summary = loaded["summary"].copy()
    scenario_enabled = loaded["artifacts"]["scenario_definition"] is not None
    if scenario_enabled:
        scenario_3_detail = loaded["scenario"].loc[
            loaded["scenario"].scenario_id.astype(str).eq("scenario_3"),
            ["combo_id", "qualified_segment_count", "failed_segment_ids"],
        ].copy()
        if (
            len(scenario_3_detail) != len(summary)
            or scenario_3_detail.combo_id.astype(str).duplicated().any()
        ):
            raise ValueError(
                "Scenario 3 detail does not cover every coordinate exactly once"
            )
        scenario_3_detail = scenario_3_detail.rename(
            columns={
                "qualified_segment_count": "scenario_3_qualified_segment_count",
                "failed_segment_ids": "scenario_3_failed_segment_ids",
            }
        )
        summary = summary.merge(
            scenario_3_detail,
            on="combo_id",
            how="left",
            validate="one_to_one",
        )
    else:
        for scenario_id in ("scenario_1", "scenario_2", "scenario_3"):
            summary[f"{scenario_id}_qualified"] = False
        summary["scenario_3_qualified_segment_count"] = 0
        summary["scenario_3_failed_segment_ids"] = ""
    trades = _augment_trade_lifecycle(loaded["trades"], copy=False)
    summary, trades = _apply_cost_adjusted_metrics(
        summary, trades, cost_model=selected_cost_model, copy=False
    )
    summary = _rank(summary)
    trade_audit = _validate_trades(
        trades, str(loaded["stage_manifest"].get("exit_mode", EXIT_MODE_COMBINED))
    )
    known = set(summary.combo_id.astype(str))
    unknown_trade_ids = set(trades.combo_id.astype(str)).difference(known)
    if unknown_trade_ids:
        raise ValueError(f"trades contain unknown coordinates: {sorted(unknown_trade_ids)[:3]}")
    combo_lifecycle = _combo_lifecycle_summary(summary, trades)
    summary = summary.merge(
        combo_lifecycle, on="combo_id", how="left", validate="one_to_one"
    )
    eligible = summary.loc[summary.scenario_3_qualified].copy()
    rankings = eligible.sort_values(
        ["method", "baseline_sampling_policy", "scenario_3_total_return_rank"],
        kind="mergesort",
    )
    diagnostics = summary.copy()
    diagnostics["scenario_3_diagnostic_rank"] = np.nan
    for _, group in diagnostics.groupby(
        ["method", "baseline_sampling_policy"], sort=True
    ):
        ordered = group.sort_values(
            [
                "scenario_3_qualified_segment_count",
                COST_ADJUSTED_RETURN_KEY,
                "train_cost_adjusted_max_drawdown_abs",
                "train_trade_count",
                "combo_id",
            ],
            ascending=[False, False, True, False, True],
            kind="mergesort",
        )
        diagnostics.loc[ordered.index, "scenario_3_diagnostic_rank"] = np.arange(
            1, len(ordered) + 1
        )
    diagnostics = diagnostics.sort_values(
        ["method", "baseline_sampling_policy", "scenario_3_diagnostic_rank"],
        kind="mergesort",
    ).reset_index(drop=True)
    speed_summary = (
        summary.groupby(
            ["method", "baseline_sampling_policy", "speed_window_bars"], sort=True
        )
        .agg(
            coordinate_count=("combo_id", "size"),
            scenario_3_qualified_count=("scenario_3_qualified", "sum"),
            best_train_return=("train_return", "max"),
            median_train_return=("train_return", "median"),
            best_train_cost_adjusted_return=(
                COST_ADJUSTED_RETURN_KEY,
                "max",
            ),
            median_train_cost_adjusted_return=(
                COST_ADJUSTED_RETURN_KEY,
                "median",
            ),
            trade_count=("train_trade_count", "sum"),
            speed_exit_count=("speed_exit_count", "sum"),
        )
        .reset_index()
    )
    exit_summary = _exit_reason_summary(trades)
    lifecycle_summary = _trade_lifecycle_summary(trades)
    if scenario_enabled:
        segment_summary = (
            loaded["segment"]
            .assign(qualified=lambda frame: _truthy(frame.qualified))
            .groupby(["method", "baseline_sampling_policy", "segment_id"], sort=True)
            .agg(
                coordinate_count=("combo_id", "size"),
                qualified_coordinate_count=("qualified", "sum"),
                entry_count_in_interval=("entry_count_in_interval", "sum"),
                exit_count_in_interval=("exit_count_in_interval", "sum"),
            )
            .reset_index()
        )
        scenario_summary = (
            loaded["scenario"]
            .assign(qualified=lambda frame: _truthy(frame.qualified))
            .groupby(["method", "baseline_sampling_policy", "scenario_id"], sort=True)
            .agg(
                coordinate_count=("combo_id", "size"),
                qualified_coordinate_count=("qualified", "sum"),
            )
            .reset_index()
        )
    else:
        segment_summary = pd.DataFrame(
            columns=[
                "method",
                "baseline_sampling_policy",
                "segment_id",
                "coordinate_count",
                "qualified_coordinate_count",
                "entry_count_in_interval",
                "exit_count_in_interval",
            ]
        )
        scenario_summary = pd.DataFrame(
            columns=[
                "method",
                "baseline_sampling_policy",
                "scenario_id",
                "coordinate_count",
                "qualified_coordinate_count",
            ]
        )
    leaders = diagnostics.groupby(
        ["method", "baseline_sampling_policy"], sort=True
    ).head(1).copy()
    leaders["evidence_role"] = (
        "scenario_3_segment_coverage_then_cost_adjusted_return_diagnostic"
    )
    leaders["parameter_accepted"] = False
    active_high_return_views = (
        HIGH_RETURN_VIEWS
        if scenario_enabled
        else tuple(
            view for view in HIGH_RETURN_VIEWS if view["scenario_filter"] == "all"
        )
    )

    analysis_root.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_root / "analysis_summary.csv"
    ranking_path = analysis_root / "scenario_3_total_return_rankings.csv"
    speed_summary_path = analysis_root / "speed_window_summary.csv"
    exit_summary_path = analysis_root / "exit_reason_summary.csv"
    lifecycle_summary_path = analysis_root / "trade_lifecycle_summary.csv"
    segment_summary_path = analysis_root / "segment_qualification_summary.csv"
    scenario_summary_path = analysis_root / "scenario_qualification_summary.csv"
    leader_path = analysis_root / "diagnostic_leaders.csv"
    trades_path = analysis_root / "stage_trades.csv"
    _atomic_csv(summary_path, summary)
    _atomic_csv(ranking_path, rankings)
    _atomic_csv(speed_summary_path, speed_summary)
    _atomic_csv(exit_summary_path, exit_summary)
    _atomic_csv(lifecycle_summary_path, lifecycle_summary)
    _atomic_csv(segment_summary_path, segment_summary)
    _atomic_csv(scenario_summary_path, scenario_summary)
    _atomic_csv(leader_path, leaders)
    _atomic_csv(trades_path, trades)

    main_data = {
        "coordinateCount": int(len(summary)),
        "tradeCount": int(len(trades)),
        "scopeLabel": "当前阶段",
        "scenario3QualifiedCount": int(len(eligible)),
        "highReturnViews": [dict(view) for view in active_high_return_views],
        "costModel": dict(selected_cost_model),
        "instrumentProfile": instrument_profile,
        "speedWindows": sorted(int(value) for value in summary.speed_window_bars.unique()),
        "note": (
            "情景三要求三段行情各自只有一次区间内开仓、区间内零平仓；区间后允许回撤或速度退出。"
            "当前仅使用滚动 TR 总和均值；全部窗口与排除标记策略分别排名。"
            if scenario_enabled
            else "当前品种没有绑定人工情景集；主排名只使用全坐标视图。"
        ),
        "rows": _records(diagnostics),
        "segmentRows": _records(segment_summary),
        "scenarioRows": _records(scenario_summary),
        "exitRows": _records(exit_summary),
        "lifecycleRows": _records(lifecycle_summary),
        "strategyId": loaded["completion"]["strategy_id"],
        "resultSemanticsId": loaded["completion"]["result_semantics_id"],
        "rawOutputSchemaVersion": OUTPUT_SCHEMA_VERSION,
        "planFingerprintSchemaVersion": FINGERPRINT_SCHEMA_VERSION,
        "tradeAuditSchemaVersion": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "tradeAuditSchemaId": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "reboundBaselinePolicyId": REBOUND_BASELINE_POLICY_ID,
        "baselineSamplingPolicies": sorted(
            str(value) for value in summary.baseline_sampling_policy.unique()
        ),
        "nativeTradeRoute": "trade_review/index.html?combo_id={combo_id}",
        "scenarioRequirementsRoute": (
            "scenario_requirements/index.html?scenario={scenario_id}"
            if scenario_enabled
            else None
        ),
        "templateProvenance": {
            "id": LEGACY_V4_MAIN_TEMPLATE_ID,
            "path": str(LEGACY_V4_MAIN_TEMPLATE_PATH.resolve()),
            "sha256": LEGACY_V4_MAIN_TEMPLATE_SHA256,
        },
    }
    _atomic_text(
        analysis_root / "report_data.js",
        "window.V4_4_S3_STAGE="
        + json.dumps(main_data, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
    )
    _atomic_text(
        analysis_root / "analysis_data.js",
        "window.V4_ANALYSIS_DATA="
        + json.dumps(
            main_summary_payload(main_data),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + ";\n",
    )
    main_html = _legacy_v4_main_html()
    _atomic_text(analysis_root / "index.html", main_html)
    _atomic_text(analysis_root / "analysis_report.html", main_html)

    review_root = analysis_root / "trade_review"
    review_delivery = build_stage_trade_review(
        review_root,
        diagnostics,
        trades,
        loaded["stage_manifest"],
        loaded["completion"],
        analysis_identity=ANALYSIS_IDENTITY,
        workers=review_workers,
    )
    catalog_path = Path(review_delivery["catalog"])
    trade_html_path = Path(review_delivery["index"])
    chunk_manifest_path = Path(review_delivery["manifest"])
    scenario_requirements_delivery = None
    if scenario_enabled:
        scenario_definition_path = Path(
            str(loaded["artifacts"]["scenario_definition"]["path"])
        ).resolve()
        scenario_requirements_delivery = build_scenario_requirements_delivery(
            analysis_root / "scenario_requirements",
            scenario_definition_path,
        )

    output_artifacts = {
        "analysis_summary": artifact(summary_path),
        "scenario_3_rankings": artifact(ranking_path),
        "speed_window_summary": artifact(speed_summary_path),
        "exit_reason_summary": artifact(exit_summary_path),
        "trade_lifecycle_summary": artifact(lifecycle_summary_path),
        "segment_qualification_summary": artifact(segment_summary_path),
        "scenario_qualification_summary": artifact(scenario_summary_path),
        "diagnostic_leaders": artifact(leader_path),
        "stage_trades": artifact(trades_path),
        "main_report_data": artifact(analysis_root / "analysis_data.js"),
        "simple_report_data": artifact(analysis_root / "report_data.js"),
        "main_entry_html": artifact(analysis_root / "index.html"),
        "analysis_report_html": artifact(analysis_root / "analysis_report.html"),
        "trade_catalog": artifact(catalog_path),
        "trade_level_html": artifact(trade_html_path),
        "trade_review_manifest": artifact(chunk_manifest_path),
    }
    if scenario_requirements_delivery is not None:
        output_artifacts["scenario_requirements_html"] = scenario_requirements_delivery[
            "index"
        ]
        output_artifacts["scenario_requirements_data"] = scenario_requirements_delivery[
            "data"
        ]
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "analysis_identity": ANALYSIS_IDENTITY,
        "campaign_id": loaded["completion"]["campaign_id"],
        "stage_id": loaded["completion"]["stage_id"],
        "plan_fingerprint": loaded["completion"]["plan_fingerprint"],
        "strategy_id": loaded["completion"]["strategy_id"],
        "result_semantics_id": loaded["completion"]["result_semantics_id"],
        "raw_output_schema_version": OUTPUT_SCHEMA_VERSION,
        "plan_fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "baseline_sampling_policy": loaded["completion"][
            "baseline_sampling_policy"
        ],
        "exit_mode": loaded["completion"]["exit_mode"],
        "scenario_schema_id": loaded["completion"]["scenario_schema_id"],
        "coordinate_count": int(len(summary)),
        "trade_count": int(len(trades)),
        "scenario_3_qualified_coordinate_count": int(len(eligible)),
        "method_policy_qualified_counts": {
            f"{method}|{policy}": int(len(group))
            for (method, policy), group in eligible.groupby(
                ["method", "baseline_sampling_policy"], sort=True
            )
        },
        "speed_window_bars": sorted(
            int(value) for value in summary.speed_window_bars.unique()
        ),
        "ranking_policy": (
            "baseline sampling policies are ranked separately; four independent "
            "views cover Scenario-1 total return, unrestricted total return, and "
            "unrestricted average trade return at min10/min20; each view offers gross "
            "and cost-adjusted ordering/display modes with combo_id as the exact-tie "
            "breaker; the 2 bps slippage plus USD 6 round-trip commission cost-adjusted "
            "mode is the default; Scenario 3 remains a separate diagnostic filter"
        ),
        "cost_model": dict(selected_cost_model),
        "instrument_profile": instrument_profile,
        "high_return_views": [dict(view) for view in active_high_return_views],
        "single_composite_score": False,
        "parameter_acceptance": "none",
        "stage_checks": loaded["checks"],
        "trade_audit": trade_audit,
        "presentation_contract": {
            "main_template_id": LEGACY_V4_MAIN_TEMPLATE_ID,
            "main_template": artifact(LEGACY_V4_MAIN_TEMPLATE_PATH),
            "trade_template": artifact(
                Path(__file__).resolve().parents[1]
                / "review_templates"
                / "trade_v4_explain_reuse.html"
            ),
            "trade_design_source": artifact(LEGACY_V4_TRADE_DESIGN_PATH),
            "trade_generator": artifact(
                Path(__file__).resolve().parent / "build_v4_4_review_delivery.py"
            ),
            "historical_v4_main_template_reused": True,
            "historical_v4_trade_template_reused": True,
            "main_entry": "index.html",
            "main_alias": "analysis_report.html",
            "trade_entry": "trade_review/index.html",
            "scenario_requirements_entry": (
                "scenario_requirements/index.html" if scenario_enabled else None
            ),
            "scenario_requirements_template": (
                scenario_requirements_delivery["template"]
                if scenario_requirements_delivery is not None
                else None
            ),
            "market_selector_source": (
                scenario_requirements_delivery["selector_source"]
                if scenario_requirements_delivery is not None
                else None
            ),
            "scenario_definition": (
                scenario_requirements_delivery["scenario_definition"]
                if scenario_requirements_delivery is not None
                else None
            ),
            "market_selector_interaction_reused": scenario_enabled,
        },
        "source_artifacts": loaded["artifacts"],
        "artifacts": output_artifacts,
    }
    manifest_path = analysis_root / "analysis_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {**manifest, "analysis_manifest": artifact(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--union-campaigns-root", type=Path)
    parser.add_argument("--union-output", type=Path)
    parser.add_argument("--review-workers", type=int, default=4)
    parser.add_argument(
        "--skip-union-refresh",
        action="store_true",
        help="Retain only the stage delivery; reserved for controlled repair work.",
    )
    args = parser.parse_args()
    plan = args.plan.resolve()
    stage = args.stage.resolve()
    analysis = (args.analysis or (stage / "analysis")).resolve()
    from run_v4_4_resumable_campaign import _exclusive_stage_writer

    with _exclusive_stage_writer(stage):
        result = analyze(
            plan, stage, analysis, review_workers=args.review_workers
        )
        union_result = None
        if not args.skip_union_refresh:
            from build_v4_4_combined_union_analysis import build_union

            campaigns_root = (
                args.union_campaigns_root.resolve()
                if args.union_campaigns_root
                else next(
                    (parent for parent in (stage, *stage.parents) if parent.name == "campaigns"),
                    stage.parent,
                )
            )
            union_result = build_union(
                campaigns_root=campaigns_root,
                output_root=(
                    args.union_output.resolve()
                    if args.union_output
                    else REPOSITORY_ROOT
                    / "results"
                    / "all_completed_union_analysis"
                ),
                review_workers=args.review_workers,
            )
    print(
        json.dumps(
            {
                "analysis": str(analysis.resolve()),
                "coordinate_count": result["coordinate_count"],
                "trade_count": result["trade_count"],
                "scenario_3_qualified_coordinate_count": result[
                    "scenario_3_qualified_coordinate_count"
                ],
                "status": result["status"],
                "union_delivery": (
                    {
                        "output": union_result["output"],
                        "coordinate_count": union_result["coordinate_count"],
                        "trade_count": union_result["trade_count"],
                        "completed_stage_count": union_result["completed_stage_count"],
                    }
                    if union_result is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
