"""Resumable, bounded-memory parameter execution for the isolated V4.4 engine.

The runner deliberately separates coordinate execution from candidate selection.
An input plan can describe a coarse Cartesian grid or an explicit refinement set,
but the runner never chooses leaders or imposes a business ranking.  Every batch
keeps the raw V4 summary metrics, event qualification rows, and immutable trades.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import json
import math
import msvcrt
import os
import re
import time
import traceback
import uuid
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from instrument_contracts import (
    EXPERIMENT_MODES,
    load_campaign_manifest,
    load_instrument_profile,
    sha256_file as instrument_sha256_file,
)
from v4_4_engine import (
    BASELINE_SAMPLING_POLICIES,
    COMBINED_TRADE_AUDIT_SCHEMA_ID,
    COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
    COMBINED_STRATEGY_ID,
    DATA_PREPARATION_PIPELINE_VERSION,
    DATA_PREPARATION_MANIFEST_DEFAULT,
    DEFAULT_BASELINE_SAMPLING_POLICY,
    ENTRY_FILL_CALCULATED_THRESHOLD,
    ENTRY_FILL_MODES,
    ENTRY_FILL_NEXT_BAR_OPEN,
    ENTRY_EXECUTION_POLICIES,
    ENTRY_EXECUTION_REJECT_SYNTHETIC_FILL,
    ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    ENTRY_METHOD_ROLLING,
    ENTRY_SIGNAL_POLICY_ID,
    EXIT_MODE_COMBINED,
    EXIT_MODE_REBOUND_ONLY,
    EVENTS_DEFAULT,
    METHODS,
    REBOUND_BASELINE_POLICY_ID,
    SOURCE_DEFAULT,
    SOURCE_SHA256,
    STRATEGY_ID,
    TRADE_AUDIT_SCHEMA_ID,
    TRADE_AUDIT_SCHEMA_VERSION,
    TRAIN_END,
    TRAIN_START,
    VERSION_LABEL,
    Combo,
    _event_metrics,
    _jsonable,
    _sha256,
    _summary,
    load_bars,
    simulate_combo,
    baseline_filter_id,
    strategy_id,
)
from scenario_groups import (
    COMBINED_SCENARIO_SCHEMA_ID,
    SCENARIO_SCHEMA_ID,
    SELECTION_MODE as SCENARIO_SELECTION_MODE,
    attach_scenario_groups,
    evaluate_segment_qualification,
    load_scenario_contract,
    segments_frame,
)


PLAN_SCHEMA_VERSION = 6
SUPPORTED_PLAN_SCHEMA_VERSIONS = (2, 3, 4, 5, PLAN_SCHEMA_VERSION)
OUTPUT_SCHEMA_VERSION = 7
FINGERPRINT_SCHEMA_VERSION = 8
DEFAULT_WORKERS = 2
DEFAULT_BATCH_SIZE = 16
DEFAULT_MINIMUM_FREE_MEMORY_MB = 4096
LEGACY_MAX_WORKERS = 2
LEGACY_MINIMUM_FREE_MEMORY_MB = 4096
def result_semantics_id(
    entry_fill_mode: str,
    entry_execution_policy: str,
    entry_slippage: float,
    exit_mode: str = EXIT_MODE_REBOUND_ONLY,
    baseline_sampling_policy: str = DEFAULT_BASELINE_SAMPLING_POLICY,
) -> str:
    if baseline_sampling_policy not in BASELINE_SAMPLING_POLICIES:
        raise ValueError(
            "unsupported V4.4 baseline sampling policy: "
            f"{baseline_sampling_policy}"
        )
    slip = format(float(entry_slippage), ".12g").replace(".", "p")
    base = (
        f"v4_4_{baseline_sampling_policy}_rolling_tr_sum_"
        f"{entry_fill_mode}_{entry_execution_policy}_short_adverse_slippage_{slip}_"
        "h_bounded_max_completed_w_drop_rebound_strict_low_close_fill_"
        "sample_end_close_pending_exit_next_real_trade_open_"
        "positive_entry_signal_results_v4"
    )
    if exit_mode == EXIT_MODE_REBOUND_ONLY:
        return base
    if exit_mode == EXIT_MODE_COMBINED:
        return base.replace("_results_v4", "_combined_zero_extension_exit_results_v4")
    raise ValueError(f"unsupported V4.4 exit_mode: {exit_mode}")


def trade_audit_identity(exit_mode: str) -> tuple[int, str]:
    if exit_mode == EXIT_MODE_REBOUND_ONLY:
        return TRADE_AUDIT_SCHEMA_VERSION, TRADE_AUDIT_SCHEMA_ID
    if exit_mode == EXIT_MODE_COMBINED:
        return COMBINED_TRADE_AUDIT_SCHEMA_VERSION, COMBINED_TRADE_AUDIT_SCHEMA_ID
    raise ValueError(f"unsupported V4.4 exit_mode: {exit_mode}")


RESULT_SEMANTICS_ID = result_semantics_id(
    ENTRY_FILL_CALCULATED_THRESHOLD,
    ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    0.0,
)
RESULT_SEMANTICS_BY_ENTRY_FILL = {
    mode: result_semantics_id(mode, ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE, 0.0)
    for mode in ENTRY_FILL_MODES
}
SCENARIO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
COORDINATE_LABEL_FIELDS = ("seed", "objective", "design", "search_mode")


@dataclass(frozen=True)
class ResourceSettings:
    workers: int = DEFAULT_WORKERS
    batch_size: int = DEFAULT_BATCH_SIZE
    minimum_free_memory_mb: int = DEFAULT_MINIMUM_FREE_MEMORY_MB


@dataclass(frozen=True)
class RuntimeSpec:
    source: Path
    data_preparation_manifest: Path
    train_start: pd.Timestamp
    train_end: pd.Timestamp


@dataclass(frozen=True)
class EffectivePlan:
    campaign_id: str
    stage_id: str
    stage_kind: str
    predecessor_stage_ids: tuple[str, ...]
    selection_provenance: str
    source: Path
    data_preparation_manifest: Path
    events: Path
    scenario_definition: Path | None
    scenario_contract: dict[str, Any] | None
    scenario_selection_mode: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    scenario_ids: tuple[str, ...]
    entry_fill_mode: str
    entry_execution_policy: str
    entry_slippage: float
    baseline_sampling_policy: str
    exit_mode: str
    strategy_id: str
    resources: ResourceSettings
    combos: tuple[Combo, ...]
    coordinate_labels: dict[str, dict[str, str]]
    duplicate_coordinate_count: int
    input_plan_sha256: str
    experiment_mode: str | None
    campaign_manifest: Path | None
    campaign_manifest_contract: dict[str, Any] | None
    instrument_profile: Path | None
    instrument_profile_contract: dict[str, Any] | None
    ranking_lineage_id: str | None
    scenario_policy: str
    mode_validation_contract: dict[str, Any] | None


_WORKER_FRAME: pd.DataFrame | None = None
_WORKER_START: pd.Timestamp | None = None
_WORKER_END: pd.Timestamp | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_identifier(value: Any, field: str) -> str:
    result = str(value).strip()
    if not result or not SCENARIO_ID_PATTERN.fullmatch(result):
        raise ValueError(f"{field} must contain only letters, numbers, underscore, or hyphen")
    return result


def _label_tokens(value: Any, field: str) -> set[str]:
    if value in (None, ""):
        return set()
    tokens = {item.strip() for item in str(value).split("|") if item.strip()}
    for token in tokens:
        _safe_identifier(token, field)
    return tokens


def _resolve_plan_path(value: Any, plan_dir: Path, default: Path) -> Path:
    raw = Path(str(value)) if value not in (None, "") else default
    return raw.resolve() if raw.is_absolute() else (plan_dir / raw).resolve()


def _number_list(block: dict[str, Any], name: str) -> list[Any]:
    values = block.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"grid block field {name!r} must be a non-empty list")
    return values


def _combo_from_record(
    record: dict[str, Any],
    default_entry_fill_mode: str = ENTRY_FILL_CALCULATED_THRESHOLD,
    default_entry_execution_policy: str = ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    default_entry_slippage: float = 0.0,
    default_baseline_sampling_policy: str = DEFAULT_BASELINE_SAMPLING_POLICY,
) -> Combo:
    required = {"method", "e", "bh", "trw", "k", "w", "m"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"combo record lacks fields: {sorted(missing)}")
    combo = Combo(
        method=str(record["method"]),
        e=int(record["e"]),
        bh=int(record["bh"]),
        trw=int(record["trw"]),
        k=float(record["k"]),
        w=int(record["w"]),
        m=float(record["m"]),
        entry_fill_mode=str(record.get("entry_fill_mode", default_entry_fill_mode)),
        entry_execution_policy=str(
            record.get("entry_execution_policy", default_entry_execution_policy)
        ),
        entry_slippage=float(record.get("entry_slippage", default_entry_slippage)),
        speed_window_bars=int(record.get("speed_window_bars", 0)),
        baseline_sampling_policy=str(
            record.get("baseline_sampling_policy", default_baseline_sampling_policy)
        ),
    )
    _validate_combo(combo)
    return combo


def _validate_combo(combo: Combo) -> None:
    if combo.method not in METHODS:
        raise ValueError(f"unsupported V4 method: {combo.method}")
    for name in ("e", "bh", "trw", "w"):
        if int(getattr(combo, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("k", "m"):
        value = float(getattr(combo, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if combo.method == ENTRY_METHOD_ROLLING and combo.trw > combo.bh:
        raise ValueError("rolling_tr_sum requires TRW <= BH")


def _expand_grid_block(
    block: dict[str, Any],
    default_entry_fill_mode: str = ENTRY_FILL_CALCULATED_THRESHOLD,
    default_entry_execution_policy: str = ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    default_entry_slippage: float = 0.0,
    default_baseline_sampling_policy: str = DEFAULT_BASELINE_SAMPLING_POLICY,
) -> Iterable[Combo]:
    method = str(block.get("method", ""))
    if method not in METHODS:
        raise ValueError(f"grid block has unsupported method: {method}")
    axes = {name: _number_list(block, name) for name in ("e", "bh", "trw", "k", "w", "m")}
    axes["speed_window_bars"] = block.get("speed_window_bars", [0])
    if not isinstance(axes["speed_window_bars"], list) or not axes["speed_window_bars"]:
        raise ValueError("grid block field 'speed_window_bars' must be a non-empty list")
    axis_names = ("e", "bh", "trw", "k", "w", "m", "speed_window_bars")
    for values in itertools.product(*(axes[name] for name in axis_names)):
        yield _combo_from_record(
            dict(zip(axis_names, values), method=method),
            str(block.get("entry_fill_mode", default_entry_fill_mode)),
            str(block.get("entry_execution_policy", default_entry_execution_policy)),
            float(block.get("entry_slippage", default_entry_slippage)),
            str(
                block.get(
                    "baseline_sampling_policy",
                    default_baseline_sampling_policy,
                )
            ),
        )


def _normalise_resources(
    payload: dict[str, Any],
    workers_override: int | None,
    batch_size_override: int | None,
    minimum_free_memory_mb_override: int | None,
) -> ResourceSettings:
    resources = payload.get("resources", {})
    if resources is None:
        resources = {}
    if not isinstance(resources, dict):
        raise ValueError("resources must be an object")
    workers = int(workers_override if workers_override is not None else resources.get("workers", DEFAULT_WORKERS))
    batch_size = int(batch_size_override if batch_size_override is not None else resources.get("batch_size", DEFAULT_BATCH_SIZE))
    minimum = int(
        minimum_free_memory_mb_override
        if minimum_free_memory_mb_override is not None
        else resources.get("minimum_free_memory_mb", DEFAULT_MINIMUM_FREE_MEMORY_MB)
    )
    if workers < 1:
        raise ValueError("workers must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if minimum < 0:
        raise ValueError("minimum_free_memory_mb cannot be negative")
    return ResourceSettings(workers=workers, batch_size=batch_size, minimum_free_memory_mb=minimum)


def _candidate_records(payload: dict[str, Any], field: str) -> list[dict[str, Any]]:
    records = payload.get("candidates")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{field} must contain a non-empty candidates list")
    if payload.get("candidate_count") not in (None, len(records)):
        raise ValueError(f"{field} candidate_count differs from candidates")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"{field}.candidates[{index}] must be an object")
        parameters = item.get("parameters", item)
        if not isinstance(parameters, dict):
            raise ValueError(f"{field}.candidates[{index}].parameters must be an object")
        result.append({**parameters, "combo_id": item.get("combo_id", parameters.get("combo_id"))})
    return result


def _validated_candidate_combos(
    payload: dict[str, Any], field: str, defaults: tuple[str, str, float, str]
) -> dict[str, Combo]:
    result: dict[str, Combo] = {}
    for record in _candidate_records(payload, field):
        combo = _combo_from_record(record, *defaults)
        declared = str(record.get("combo_id", ""))
        if not declared or declared != combo.combo_id:
            raise ValueError(f"{field} contains an absent or incorrect combo_id")
        if declared in result:
            raise ValueError(f"{field} contains duplicate combo_id: {declared}")
        result[declared] = combo
    return result


def _completed_parent(
    parent: dict[str, Any], instrument_id: str, field: str
) -> None:
    if str(parent.get("status", "")) not in {"complete", "completed", "immutable_closed"}:
        raise ValueError(f"{field} is not complete")
    target = str(
        parent.get("target_instrument_id", parent.get("instrument_id", ""))
    ).strip()
    if target != instrument_id:
        raise ValueError(f"{field} target instrument differs from current profile")


def _rule_accepts(value: float, anchor: float, rule: dict[str, Any]) -> bool:
    mode = str(rule.get("mode", ""))
    if mode == "fixed":
        return value == anchor
    if mode == "relative":
        return abs(value - anchor) <= abs(anchor) * float(rule.get("max_fraction", -1)) + 1e-12
    if mode == "absolute":
        return abs(value - anchor) <= float(rule.get("max_delta", -1)) + 1e-12
    if mode == "values":
        values = rule.get("values")
        return isinstance(values, list) and value in [float(item) for item in values]
    raise ValueError(f"unsupported neighborhood rule mode: {mode}")


def _parameter_values(combo: Combo) -> dict[str, float]:
    return {
        "e": float(combo.e), "bh": float(combo.bh), "trw": float(combo.trw),
        "k": float(combo.k), "w": float(combo.w), "m": float(combo.m),
        "speed_window_bars": float(combo.speed_window_bars),
    }


def _validate_parameter_space(combo: Combo, rules: dict[str, Any], field: str) -> None:
    values = _parameter_values(combo)
    for name, value in values.items():
        rule = rules.get(name)
        if not isinstance(rule, dict):
            raise ValueError(f"{field} lacks rule for {name}")
        if "values" in rule:
            allowed = [float(item) for item in rule["values"]]
            if value not in allowed:
                raise ValueError(f"coordinate {combo.combo_id} is outside {field}.{name}")
        else:
            low, high = float(rule.get("min", math.inf)), float(rule.get("max", -math.inf))
            if not low <= value <= high:
                raise ValueError(f"coordinate {combo.combo_id} is outside {field}.{name}")


def _validate_mode_coordinates(
    mode: str | None,
    manifest: dict[str, Any] | None,
    combos: tuple[Combo, ...],
    requested_count: int,
    labels: dict[str, dict[str, str]],
    defaults: tuple[str, str, float, str],
) -> dict[str, Any] | None:
    if mode is None or manifest is None or int(manifest.get("manifest_schema_version", 1)) < 2:
        return None
    resolved = manifest["resolved_mode_contract"]
    profile = manifest["instrument_profile_contract"]
    instrument_id = str(profile["instrument_id"])
    combo_ids = {combo.combo_id for combo in combos}
    search = manifest.get("search") or {}
    budget = int(search.get("maximum_coordinate_count", 0))
    if budget <= 0 or len(combos) > budget:
        raise ValueError("campaign coordinate count exceeds or lacks maximum_coordinate_count")
    if requested_count != len(combos):
        raise ValueError("new mode contracts reject duplicate plan coordinates")
    if mode == "transfer_exact":
        source_bar_seconds = int(resolved["candidate_freeze_payload"].get("bar_seconds", 0))
        target_bar_seconds = int(profile.get("bar_seconds", 0))
        if source_bar_seconds != target_bar_seconds or target_bar_seconds != 15:
            raise ValueError(
                f"transfer bar_seconds differs: expected=15, source={source_bar_seconds}, target={target_bar_seconds}"
            )
        if str(resolved["candidate_freeze_payload"].get("status", "")) != "frozen_before_target_evaluation":
            raise ValueError("candidate freeze was not frozen before target evaluation")
        if not str(resolved["candidate_freeze_payload"].get("frozen_at", "")).strip():
            raise ValueError("candidate freeze lacks frozen_at")
        if resolved["candidate_freeze_payload"].get("target_evaluation_started_at") not in (None, ""):
            raise ValueError("candidate freeze was created after target evaluation began")
        if resolved["candidate_freeze_payload"].get("target_results_present") is not False:
            raise ValueError("candidate freeze lacks target-result absence proof")
        frozen = _validated_candidate_combos(
            resolved["candidate_freeze_payload"], "candidate freeze", defaults
        )
        frozen_ids = set(frozen)
        if combo_ids != frozen_ids:
            raise ValueError(
                f"transfer_exact plan differs from candidate freeze: missing={len(frozen_ids-combo_ids)}, extra={len(combo_ids-frozen_ids)}"
            )
        return {"mode": mode, "candidate_count": len(frozen_ids), "coordinate_set_exact": True}
    if mode == "target_local_refinement":
        parent = resolved["parent_payload"]
        _completed_parent(parent, instrument_id, "parent transfer")
        frozen_payload = resolved.get("candidate_freeze_payload")
        if int(frozen_payload.get("bar_seconds", 0)) != int(profile.get("bar_seconds", 0)):
            raise ValueError("parent candidate freeze bar_seconds differs from target")
        frozen = _validated_candidate_combos(frozen_payload, "parent candidate freeze", defaults)
        parent_freeze_sha = str(parent.get("candidate_freeze_sha256", ""))
        if not parent_freeze_sha:
            raise ValueError("parent transfer lacks candidate_freeze_sha256")
        if parent_freeze_sha != instrument_sha256_file(Path(resolved["candidate_freeze_path"])):
            raise ValueError("parent transfer binds a different candidate freeze")
        neighborhood = search["neighborhood"]
        anchor_ids = neighborhood.get("anchor_combo_ids")
        rules = neighborhood.get("parameter_rules")
        if not isinstance(anchor_ids, list) or not anchor_ids or not isinstance(rules, dict):
            raise ValueError("bounded neighborhood requires anchors and parameter_rules")
        if any(str(item) not in frozen for item in anchor_ids):
            raise ValueError("neighborhood anchor is absent from parent candidate freeze")
        anchors = [frozen[str(item)] for item in anchor_ids]
        for combo in combos:
            values = _parameter_values(combo)
            if not any(
                combo.method == anchor.method
                and combo.entry_fill_mode == anchor.entry_fill_mode
                and combo.entry_execution_policy == anchor.entry_execution_policy
                and combo.entry_slippage == anchor.entry_slippage
                and combo.baseline_sampling_policy == anchor.baseline_sampling_policy
                and all(
                    isinstance(rules.get(name), dict)
                    and _rule_accepts(value, _parameter_values(anchor)[name], rules[name])
                    for name, value in values.items()
                )
                for anchor in anchors
            ):
                raise ValueError(f"coordinate {combo.combo_id} is outside every declared neighborhood")
        return {"mode": mode, "anchor_count": len(anchors), "all_coordinates_local": True}
    if mode == "continuation_search":
        _completed_parent(resolved["parent_payload"], instrument_id, "parent stage")
        parent_lineage = str(resolved["parent_payload"].get("ranking_lineage_id", ""))
        if parent_lineage != str(profile["ranking_lineage_id"]):
            raise ValueError("continuation parent ranking lineage differs")
    if mode in {"continuation_search", "fresh_search"}:
        rules = search.get("parameter_space")
        if not isinstance(rules, dict):
            raise ValueError(f"{mode} requires parameter_space")
        for combo in combos:
            _validate_parameter_space(combo, rules, "parameter_space")
            if not labels[combo.combo_id].get("search_mode"):
                raise ValueError(f"{mode} requires search_mode on every coordinate")
        return {"mode": mode, "coordinate_count": len(combos), "parameter_space_valid": True}
    raise ValueError(f"unsupported experiment mode: {mode}")


def load_plan(
    plan_path: Path,
    *,
    workers: int | None = None,
    batch_size: int | None = None,
    minimum_free_memory_mb: int | None = None,
) -> EffectivePlan:
    plan_path = plan_path.resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_schema_version = int(payload.get("schema_version", 0))
    if plan_schema_version not in SUPPORTED_PLAN_SCHEMA_VERSIONS:
        raise ValueError(
            f"plan schema_version must be one of {SUPPORTED_PLAN_SCHEMA_VERSIONS}"
        )
    campaign_id = _safe_identifier(payload.get("campaign_id"), "campaign_id")
    stage_id = _safe_identifier(payload.get("stage_id"), "stage_id")
    stage_kind = _safe_identifier(payload.get("stage_kind", "explicit"), "stage_kind")
    predecessor_ids = tuple(
        _safe_identifier(item, "predecessor_stage_ids item")
        for item in payload.get("predecessor_stage_ids", [])
    )
    selection_provenance = str(payload.get("selection_provenance", "")).strip()
    plan_dir = plan_path.parent
    campaign_manifest: Path | None = None
    campaign_manifest_contract: dict[str, Any] | None = None
    campaign_manifest_value = payload.get("campaign_manifest")
    if campaign_manifest_value not in (None, ""):
        campaign_manifest = _resolve_plan_path(
            campaign_manifest_value,
            plan_dir,
            Path(str(campaign_manifest_value)),
        )
        campaign_manifest_contract = load_campaign_manifest(campaign_manifest)
        if campaign_manifest_contract["campaign_id"] != campaign_id:
            raise ValueError("plan campaign_id differs from campaign manifest")

    experiment_mode_value = payload.get("experiment_mode")
    experiment_mode = (
        str(experiment_mode_value).strip()
        if experiment_mode_value not in (None, "")
        else None
    )
    if campaign_manifest_contract is not None:
        manifest_mode = str(campaign_manifest_contract["mode"])
        if experiment_mode is None:
            experiment_mode = manifest_mode
        elif experiment_mode != manifest_mode:
            raise ValueError("plan experiment_mode differs from campaign manifest")
    if experiment_mode is not None and experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(f"unsupported experiment_mode: {experiment_mode}")
    if plan_schema_version >= 5 and campaign_manifest is None:
        raise ValueError("schema-v5+ plans require campaign_manifest")
    if plan_schema_version >= 6 and int(campaign_manifest_contract.get("manifest_schema_version", 0)) < 2:
        raise ValueError("schema-v6 plans require a schema-v2 campaign manifest")

    instrument_profile: Path | None = None
    instrument_profile_contract: dict[str, Any] | None = None
    instrument_profile_value = payload.get("instrument_profile")
    if instrument_profile_value in (None, "") and campaign_manifest_contract is not None:
        instrument_profile_value = campaign_manifest_contract[
            "resolved_instrument_profile_path"
        ]
    if instrument_profile_value not in (None, ""):
        instrument_profile = _resolve_plan_path(
            instrument_profile_value,
            plan_dir,
            Path(str(instrument_profile_value)),
        )
        instrument_profile_contract = load_instrument_profile(instrument_profile)
        if campaign_manifest_contract is not None and instrument_profile != Path(
            campaign_manifest_contract["resolved_instrument_profile_path"]
        ):
            raise ValueError("plan instrument_profile differs from campaign manifest")

    ranking_lineage_value = payload.get("ranking_lineage_id")
    ranking_lineage_id = (
        _safe_identifier(ranking_lineage_value, "ranking_lineage_id")
        if ranking_lineage_value not in (None, "")
        else None
    )
    if instrument_profile_contract is not None:
        profile_lineage = str(instrument_profile_contract["ranking_lineage_id"])
        if ranking_lineage_id is None:
            ranking_lineage_id = profile_lineage
        elif ranking_lineage_id != profile_lineage:
            raise ValueError("plan ranking lineage differs from instrument profile")

    profile_source = (
        Path(str(instrument_profile_contract["resolved_market_data_path"]))
        if instrument_profile_contract is not None
        else SOURCE_DEFAULT
    )
    profile_preparation = (
        Path(str(instrument_profile_contract["resolved_preparation_manifest_path"]))
        if instrument_profile_contract is not None
        else DATA_PREPARATION_MANIFEST_DEFAULT
    )
    source = _resolve_plan_path(payload.get("source"), plan_dir, profile_source)
    preparation = _resolve_plan_path(
        payload.get("data_preparation_manifest"), plan_dir, profile_preparation
    )
    if instrument_profile_contract is not None:
        if source != profile_source:
            raise ValueError("plan source differs from instrument profile")
        if preparation != profile_preparation:
            raise ValueError("plan preparation manifest differs from instrument profile")
    events = _resolve_plan_path(payload.get("events"), plan_dir, EVENTS_DEFAULT)
    train_start = pd.Timestamp(payload.get("train_start", TRAIN_START))
    train_end = pd.Timestamp(payload.get("train_end", TRAIN_END))
    if train_start > train_end:
        raise ValueError("train_start must be earlier than or equal to train_end")
    scenario_definition: Path | None = None
    scenario_contract: dict[str, Any] | None = None
    scenario_policy = str(payload.get("scenario_policy", "legacy_or_explicit")).strip()
    if scenario_policy not in {"legacy_or_explicit", "profile_optional", "none"}:
        raise ValueError(f"unsupported scenario_policy: {scenario_policy}")
    scenario_definition_value = payload.get("scenario_definition")
    if (
        scenario_definition_value in (None, "")
        and scenario_policy == "profile_optional"
        and instrument_profile_contract is not None
    ):
        scenario_definition_value = instrument_profile_contract.get(
            "resolved_scenario_set_path"
        )
    if scenario_policy == "none":
        if scenario_definition_value not in (None, "") or "scenario_ids" in payload:
            raise ValueError("scenario_policy none cannot bind scenario definitions or ids")
        scenario_ids = ()
        scenario_selection_mode = "none"
    elif scenario_definition_value not in (None, ""):
        if "scenario_ids" in payload:
            raise ValueError(
                "scenario-group plans derive all available scenarios from scenario_definition; "
                "scenario_ids multi-selection is not allowed"
            )
        scenario_definition = _resolve_plan_path(
            scenario_definition_value,
            plan_dir,
            Path(str(scenario_definition_value)),
        )
        scenario_contract = load_scenario_contract(scenario_definition)
        scenario_ids = tuple(
            _safe_identifier(item["scenario_id"], "scenario_id")
            for item in scenario_contract["scenarios"]
        )
        scenario_selection_mode = str(scenario_contract["selection_mode"])
    elif scenario_policy == "profile_optional" and plan_schema_version >= 5:
        scenario_ids = ()
        scenario_selection_mode = "none"
    else:
        raw_scenarios = payload.get("scenario_ids", ["event_01", "event_02"])
        if not isinstance(raw_scenarios, list) or not raw_scenarios:
            raise ValueError("scenario_ids must be a non-empty list")
        scenario_ids = tuple(_safe_identifier(item, "scenario_ids item") for item in raw_scenarios)
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario_ids contains duplicates")
        scenario_selection_mode = "legacy_multiple"
    entry_fill_mode = str(
        payload.get("entry_fill_mode", ENTRY_FILL_CALCULATED_THRESHOLD)
    )
    if entry_fill_mode != ENTRY_FILL_CALCULATED_THRESHOLD:
        raise ValueError(
            "V4.4 requires calculated_threshold entry; next_bar_open plans belong to V4.1"
        )
    entry_execution_policy = str(
        payload.get("entry_execution_policy", ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE)
    )
    if entry_execution_policy not in ENTRY_EXECUTION_POLICIES:
        raise ValueError(
            f"unsupported V4.4 entry_execution_policy: {entry_execution_policy}"
        )
    entry_slippage = float(payload.get("entry_slippage", 0.0))
    if not math.isfinite(entry_slippage) or entry_slippage < 0:
        raise ValueError("entry_slippage must be finite and nonnegative")
    baseline_sampling_policy = str(
        payload.get(
            "baseline_sampling_policy",
            DEFAULT_BASELINE_SAMPLING_POLICY,
        )
    )
    if baseline_sampling_policy not in BASELINE_SAMPLING_POLICIES:
        raise ValueError(
            "unsupported V4.4 baseline_sampling_policy: "
            f"{baseline_sampling_policy}"
        )
    exit_mode = str(payload.get("exit_mode", EXIT_MODE_REBOUND_ONLY))
    if exit_mode not in {EXIT_MODE_REBOUND_ONLY, EXIT_MODE_COMBINED}:
        raise ValueError(f"unsupported V4.4 exit_mode: {exit_mode}")
    selected_strategy_id = strategy_id(
        baseline_sampling_policy,
        combined_exit=exit_mode == EXIT_MODE_COMBINED,
    )

    requested: list[Combo] = []
    label_sets: dict[str, dict[str, set[str]]] = {}

    def add(combo: Combo, metadata: dict[str, Any]) -> None:
        requested.append(combo)
        target = label_sets.setdefault(
            combo.combo_id, {field: set() for field in COORDINATE_LABEL_FIELDS}
        )
        for field in COORDINATE_LABEL_FIELDS:
            target[field].update(_label_tokens(metadata.get(field), field))

    grid_blocks = payload.get("grid_blocks", [])
    explicit_combos = payload.get("explicit_combos", [])
    if not isinstance(grid_blocks, list) or not isinstance(explicit_combos, list):
        raise ValueError("grid_blocks and explicit_combos must be lists")
    for block in grid_blocks:
        if not isinstance(block, dict):
            raise ValueError("each grid_blocks item must be an object")
        for combo in _expand_grid_block(
            block,
            entry_fill_mode,
            entry_execution_policy,
            entry_slippage,
            baseline_sampling_policy,
        ):
            if (
                combo.entry_fill_mode != entry_fill_mode
                or combo.entry_execution_policy != entry_execution_policy
                or combo.entry_slippage != entry_slippage
                or combo.baseline_sampling_policy != baseline_sampling_policy
                or combo.exit_mode != exit_mode
            ):
                raise ValueError("all coordinates in one V4.4 stage must share its execution and exit contract")
            add(combo, block)
    for record in explicit_combos:
        if not isinstance(record, dict):
            raise ValueError("each explicit_combos item must be an object")
        combo = _combo_from_record(
            record,
            entry_fill_mode,
            entry_execution_policy,
            entry_slippage,
            baseline_sampling_policy,
        )
        if (
            combo.entry_fill_mode != entry_fill_mode
            or combo.entry_execution_policy != entry_execution_policy
            or combo.entry_slippage != entry_slippage
            or combo.baseline_sampling_policy != baseline_sampling_policy
            or combo.exit_mode != exit_mode
        ):
            raise ValueError("all coordinates in one V4.4 stage must share its execution and exit contract")
        add(combo, record)
    if not requested:
        raise ValueError("the stage plan contains no parameter coordinates")

    unique = {combo.combo_id: combo for combo in requested}
    combos = tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.entry_fill_mode, item.entry_execution_policy, item.entry_slippage,
                item.baseline_sampling_policy,
                item.method, item.e, item.bh, item.trw, item.k, item.w, item.m,
                item.speed_window_bars,
                item.combo_id,
            ),
        )
    )
    settings = _normalise_resources(payload, workers, batch_size, minimum_free_memory_mb)
    coordinate_labels = {
        combo.combo_id: {
            field: "|".join(sorted(label_sets[combo.combo_id][field]))
            for field in COORDINATE_LABEL_FIELDS
        }
        for combo in combos
    }
    defaults = (
        entry_fill_mode,
        entry_execution_policy,
        entry_slippage,
        baseline_sampling_policy,
    )
    mode_validation_contract = _validate_mode_coordinates(
        experiment_mode,
        campaign_manifest_contract,
        combos,
        len(requested),
        coordinate_labels,
        defaults,
    )
    return EffectivePlan(
        campaign_id=campaign_id,
        stage_id=stage_id,
        stage_kind=stage_kind,
        predecessor_stage_ids=predecessor_ids,
        selection_provenance=selection_provenance,
        source=source,
        data_preparation_manifest=preparation,
        events=events,
        scenario_definition=scenario_definition,
        scenario_contract=scenario_contract,
        scenario_selection_mode=scenario_selection_mode,
        train_start=train_start,
        train_end=train_end,
        scenario_ids=scenario_ids,
        entry_fill_mode=entry_fill_mode,
        entry_execution_policy=entry_execution_policy,
        entry_slippage=entry_slippage,
        baseline_sampling_policy=baseline_sampling_policy,
        exit_mode=exit_mode,
        strategy_id=selected_strategy_id,
        resources=settings,
        combos=combos,
        coordinate_labels=coordinate_labels,
        duplicate_coordinate_count=len(requested) - len(combos),
        input_plan_sha256=_sha256(plan_path),
        experiment_mode=experiment_mode,
        campaign_manifest=campaign_manifest,
        campaign_manifest_contract=campaign_manifest_contract,
        instrument_profile=instrument_profile,
        instrument_profile_contract=instrument_profile_contract,
        ranking_lineage_id=ranking_lineage_id,
        scenario_policy=scenario_policy,
        mode_validation_contract=mode_validation_contract,
    )


def _available_memory_mb() -> int | None:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys // (1024 * 1024))
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return pages * page_size // (1024 * 1024)
        except (OSError, TypeError, ValueError):
            return None
    return None


def _enforce_memory_floor(minimum_free_memory_mb: int) -> int | None:
    available = _available_memory_mb()
    if minimum_free_memory_mb and available is None:
        raise RuntimeError("available physical memory could not be measured")
    if available is not None and available < minimum_free_memory_mb:
        raise MemoryError(
            f"available physical memory {available} MiB is below the required floor "
            f"{minimum_free_memory_mb} MiB"
        )
    return available


def _worker_init(source: str, preparation_manifest: str, train_start: str, train_end: str) -> None:
    global _WORKER_FRAME, _WORKER_START, _WORKER_END
    _WORKER_FRAME = load_bars(Path(source), Path(preparation_manifest))
    _WORKER_START = pd.Timestamp(train_start)
    _WORKER_END = pd.Timestamp(train_end)


def _worker_run(combo: Combo) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if _WORKER_FRAME is None or _WORKER_START is None or _WORKER_END is None:
        raise RuntimeError("V4 campaign worker is not initialized")
    trades = simulate_combo(_WORKER_FRAME, combo, _WORKER_START, _WORKER_END)
    return _summary(combo, trades), trades


def _atomic_target(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")


def _atomic_write_text(path: Path, value: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_target(path)
    temporary.write_text(value, encoding=encoding)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_target(path)
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": int(path.stat().st_size)}


def _resolve_manifest_path(manifest_path: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _extreme_output_matches_runtime_source(
    preparation_manifest: Path,
    extreme_contract: dict[str, Any],
    cleaned_source: dict[str, Any],
    runtime_source: Path,
    runtime_source_sha256: str,
) -> bool:
    audited_path = Path(str(cleaned_source.get("path", "")))
    audited_path_matches = bool(
        str(audited_path) and audited_path.resolve() == runtime_source.resolve()
    )
    relocated_path = _resolve_manifest_path(
        preparation_manifest,
        extreme_contract.get("runtime_source_path", ""),
    )
    relocated_path_matches = bool(
        extreme_contract.get("source_path_relocated") is True
        and str(extreme_contract.get("runtime_source_path", ""))
        and relocated_path == runtime_source.resolve()
    )
    return bool(
        (audited_path_matches or relocated_path_matches)
        and str(cleaned_source.get("sha256", "")).lower()
        == runtime_source_sha256.lower()
    )


def _effective_preparation_bar_seconds(
    preparation: dict[str, Any],
    instrument_profile_contract: dict[str, Any],
) -> int:
    """Resolve legacy K200 granularity through its hash-verified attestation."""
    declared = int(preparation.get("bar_seconds", 0))
    if declared:
        return declared
    attestation_path = instrument_profile_contract.get(
        "resolved_policy_attestation_path"
    )
    if not attestation_path:
        return 0
    attestation = json.loads(Path(str(attestation_path)).read_text(encoding="utf-8"))
    return int(attestation.get("bar_seconds", 0))


def _validate_contract(plan: EffectivePlan) -> tuple[pd.DataFrame, dict[str, str]]:
    required_paths = [
        (plan.source, "source"),
        (plan.data_preparation_manifest, "data_preparation_manifest"),
    ]
    if plan.scenario_definition is not None:
        required_paths.append((plan.scenario_definition, "scenario_definition"))
    elif plan.scenario_selection_mode != "none":
        required_paths.append((plan.events, "events"))
    for path, label in required_paths:
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    source_hash = _sha256(plan.source)
    if plan.instrument_profile_contract is None:
        if plan.source.resolve() != SOURCE_DEFAULT.resolve():
            raise ValueError(
                "legacy V4.4 plans permit only the audited current K200 source"
            )
        expected_source_hash = SOURCE_SHA256
    else:
        expected_source_hash = str(plan.instrument_profile_contract["data"]["sha256"])
    if source_hash.lower() != expected_source_hash.lower():
        raise ValueError("V4.4 source hash does not match its instrument contract")
    preparation = json.loads(plan.data_preparation_manifest.read_text(encoding="utf-8"))
    if preparation.get("status") != "complete":
        raise ValueError("data-preparation manifest is not complete")
    profile_schema_version = int(
        (plan.instrument_profile_contract or {}).get("schema_version", 1)
    )
    if profile_schema_version >= 2:
        if str(preparation.get("source_sha256", "")).lower() != source_hash.lower():
            raise ValueError("data-preparation manifest source hash does not match the source")
        prepared_bar_seconds = _effective_preparation_bar_seconds(
            preparation, plan.instrument_profile_contract
        )
        if prepared_bar_seconds != int(plan.instrument_profile_contract["bar_seconds"]):
            raise ValueError("data-preparation bar_seconds differs from instrument profile")
        if plan.scenario_contract is not None:
            selected = segments_frame(plan.scenario_contract)
        elif plan.scenario_selection_mode == "none":
            selected = pd.DataFrame(columns=["event_id", "start_time", "end_time"])
        else:
            events = pd.read_csv(plan.events)
            selected = events.loc[
                events.event_id.astype(str).isin(plan.scenario_ids)
            ].copy()
        hashes = {
            "source_sha256": source_hash,
            "data_preparation_manifest_sha256": _sha256(plan.data_preparation_manifest),
            "events_sha256": (
                _sha256(plan.events) if plan.scenario_selection_mode != "none" and plan.scenario_contract is None else ""
            ),
            "scenario_definition_sha256": (
                _sha256(plan.scenario_definition) if plan.scenario_definition is not None else ""
            ),
            "engine_sha256": _sha256(Path(__file__).with_name("v4_4_engine.py")),
            "runner_sha256": _sha256(Path(__file__)),
            "instrument_profile_sha256": instrument_sha256_file(plan.instrument_profile),
            "cost_model_sha256": instrument_sha256_file(
                Path(plan.instrument_profile_contract["resolved_cost_model_path"])
            ),
        }
        if plan.campaign_manifest is not None:
            hashes["campaign_manifest_sha256"] = instrument_sha256_file(plan.campaign_manifest)
        return selected.reset_index(drop=True), hashes
    if int(preparation.get("schema_version", 0)) < 5:
        raise ValueError("V4.4 causal baseline policies require a schema-v5 preparation manifest")
    if preparation.get("pipeline_version") != DATA_PREPARATION_PIPELINE_VERSION:
        raise ValueError("data-preparation pipeline identity does not match V4.4")
    if not str(preparation.get("prepared_identity", "")).startswith(
        "v4_4_policy_neutral_baseline_marker_"
    ):
        raise ValueError("prepared data identity does not belong to V4.4")
    if str(preparation.get("source_sha256", "")).lower() != source_hash.lower():
        raise ValueError("data-preparation manifest source hash does not match the V4.4 source")
    sampling_contract = preparation.get("rule_contract", {}).get(
        "baseline_sampling_contract", {}
    )
    if sampling_contract.get("marker_generation_is_policy_neutral") is not True:
        raise ValueError("data preparation does not declare a policy-neutral marker")
    if sampling_contract.get("default_baseline_sampling_policy") != DEFAULT_BASELINE_SAMPLING_POLICY:
        raise ValueError("data preparation default baseline policy differs from V4.4")
    if set(sampling_contract.get("supported_baseline_sampling_policies", {})) != set(
        BASELINE_SAMPLING_POLICIES
    ):
        raise ValueError("data preparation baseline policy set differs from V4.4")
    extreme = preparation.get("extreme_cleaning", {})
    if extreme.get("status") != "passed" or extreme.get("passed") is not True:
        raise ValueError("V4.4 campaign requires a passed extreme-cleaning audit")
    if extreme.get("method") != "same_bar_immediate_tick_recovery":
        raise ValueError("V4.4 campaign extreme-cleaning method does not match the current contract")
    extreme_audit_path = Path(str(extreme.get("audit_path", "")))
    if not extreme_audit_path.is_absolute():
        extreme_audit_path = (
            plan.data_preparation_manifest.parent / extreme_audit_path
        ).resolve()
    extreme_audit_sha256 = str(extreme.get("audit_sha256", "")).lower()
    if (
        not extreme_audit_path.is_file()
        or not extreme_audit_sha256
        or _sha256(extreme_audit_path).lower() != extreme_audit_sha256
    ):
        raise ValueError("extreme-cleaning audit failed hash validation")
    extreme_audit = json.loads(extreme_audit_path.read_text(encoding="utf-8"))
    cleaned_source = extreme_audit.get("outputs", {}).get("session_filled_15s", {})
    transient_filter = extreme_audit.get("transient_tail_filter", {})
    if (
        extreme_audit.get("passed") is not True
        or transient_filter.get("enabled") is not True
        or transient_filter.get("method") != "same_bar_immediate_tick_recovery"
    ):
        raise ValueError("extreme-cleaning audit does not prove the required immediate-recovery rule")
    if not _extreme_output_matches_runtime_source(
        plan.data_preparation_manifest,
        extreme,
        cleaned_source,
        plan.source,
        source_hash,
    ):
        raise ValueError("extreme-cleaning audit output does not match the V4.4 source")
    preparation_artifacts = preparation.get("artifacts", {})
    audited_preparation_artifacts: dict[str, str] = {}
    for artifact_name in ("filter_atoms", "filter_events"):
        entry = preparation_artifacts.get(artifact_name, {})
        artifact_path = _resolve_manifest_path(
            plan.data_preparation_manifest,
            entry.get("path", ""),
        )
        expected_hash = str(entry.get("sha256", ""))
        if not artifact_path.is_file() or not expected_hash or _sha256(artifact_path).lower() != expected_hash.lower():
            raise ValueError(f"data-preparation artifact failed hash validation: {artifact_name}")
        audited_preparation_artifacts[f"{artifact_name}_sha256"] = expected_hash.lower()
    if plan.train_start != pd.Timestamp(TRAIN_START) or plan.train_end != pd.Timestamp(TRAIN_END):
        raise ValueError("the current V4.4 campaign runner requires the fixed all-data training interval")
    if plan.scenario_contract is not None:
        selected = segments_frame(plan.scenario_contract)
        if plan.scenario_selection_mode != SCENARIO_SELECTION_MODE:
            raise ValueError("current V4.4 scenario-group plans must be single-select")
        scenario_schema_id = str(plan.scenario_contract["scenario_schema_id"])
        expected_scenario_schema_id = (
            COMBINED_SCENARIO_SCHEMA_ID
            if plan.exit_mode == EXIT_MODE_COMBINED
            else SCENARIO_SCHEMA_ID
        )
        if scenario_schema_id != expected_scenario_schema_id:
            raise ValueError(
                "scenario schema identity does not match the stage exit contract"
            )
        selected["start_time"] = pd.to_datetime(selected["start_time"], errors="raise")
        selected["end_time"] = pd.to_datetime(selected["end_time"], errors="raise")
    elif plan.scenario_selection_mode != "none":
        events = pd.read_csv(plan.events)
        required = {"event_id", "start_time", "end_time"}
        missing = required.difference(events.columns)
        if missing:
            raise ValueError(f"events file lacks fields: {sorted(missing)}")
        event_ids = set(events["event_id"].astype(str))
        absent = set(plan.scenario_ids).difference(event_ids)
        if absent:
            raise ValueError(f"events file lacks requested scenario ids: {sorted(absent)}")
        selected = events.loc[events.event_id.astype(str).isin(plan.scenario_ids)].copy()
        selected["start_time"] = pd.to_datetime(selected["start_time"], errors="raise")
        selected["end_time"] = pd.to_datetime(selected["end_time"], errors="raise")
        if selected.event_id.astype(str).duplicated().any():
            raise ValueError("events file contains duplicate requested scenario ids")
    else:
        selected = pd.DataFrame(columns=["event_id", "start_time", "end_time"])
    hashes = {
        "source_sha256": source_hash,
        "data_preparation_manifest_sha256": _sha256(plan.data_preparation_manifest),
        "extreme_cleaning_audit_sha256": extreme_audit_sha256,
        "events_sha256": (
            _sha256(plan.events)
            if plan.scenario_definition is None and plan.scenario_selection_mode != "none"
            else ""
        ),
        "scenario_definition_sha256": (
            _sha256(plan.scenario_definition)
            if plan.scenario_definition is not None
            else ""
        ),
        "engine_sha256": _sha256(Path(__file__).with_name("v4_4_engine.py")),
        "runner_sha256": _sha256(Path(__file__)),
        **audited_preparation_artifacts,
    }
    if plan.instrument_profile is not None:
        hashes["instrument_profile_sha256"] = instrument_sha256_file(
            plan.instrument_profile
        )
        cost_path = Path(
            str(plan.instrument_profile_contract["resolved_cost_model_path"])
        )
        hashes["cost_model_sha256"] = instrument_sha256_file(cost_path)
    if plan.campaign_manifest is not None:
        hashes["campaign_manifest_sha256"] = instrument_sha256_file(
            plan.campaign_manifest
        )
    return selected.sort_values("event_id", kind="mergesort").reset_index(drop=True), hashes


def _grid_rows(plan: EffectivePlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch_size = plan.resources.batch_size
    for ordinal, combo in enumerate(plan.combos, start=1):
        rows.append(
            {
                "grid_ordinal": ordinal,
                "batch_id": f"batch_{((ordinal - 1) // batch_size) + 1:05d}",
                **plan.coordinate_labels[combo.combo_id],
                **asdict(combo),
                "combo_id": combo.combo_id,
            }
        )
    return rows


def _fingerprint_payload(plan: EffectivePlan, hashes: dict[str, str]) -> dict[str, Any]:
    audit_schema_version, audit_schema_id = trade_audit_identity(plan.exit_mode)
    payload = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "campaign_id": plan.campaign_id,
        "stage_id": plan.stage_id,
        "stage_kind": plan.stage_kind,
        "predecessor_stage_ids": list(plan.predecessor_stage_ids),
        "selection_provenance": plan.selection_provenance,
        "source": str(plan.source),
        "data_preparation_manifest": str(plan.data_preparation_manifest),
        "events": str(plan.events) if plan.scenario_definition is None else None,
        "scenario_definition": (
            str(plan.scenario_definition) if plan.scenario_definition is not None else None
        ),
        "scenario_schema_id": (
            plan.scenario_contract["scenario_schema_id"]
            if plan.scenario_contract is not None
            else None
        ),
        "scenario_selection_mode": plan.scenario_selection_mode,
        "train_start": str(plan.train_start),
        "train_end": str(plan.train_end),
        "scenario_ids": list(plan.scenario_ids),
        "entry_fill_mode": plan.entry_fill_mode,
        "entry_execution_policy": plan.entry_execution_policy,
        "entry_slippage": plan.entry_slippage,
        "baseline_sampling_policy": plan.baseline_sampling_policy,
        "baseline_filter_id": baseline_filter_id(plan.baseline_sampling_policy),
        "resources": {
            "workers": plan.resources.workers,
            "batch_size": plan.resources.batch_size,
            "minimum_free_memory_mb": plan.resources.minimum_free_memory_mb,
        },
        "combo_ids": [combo.combo_id for combo in plan.combos],
        "coordinate_labels": [
            {"combo_id": combo.combo_id, **plan.coordinate_labels[combo.combo_id]}
            for combo in plan.combos
        ],
        "hashes": hashes,
        "version_label": VERSION_LABEL,
        "strategy_id": plan.strategy_id,
        "trade_audit_schema_version": audit_schema_version,
        "trade_audit_schema_id": audit_schema_id,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "entry_baseline_scope": (
            "all_finite_tr15_inside_one_continuous_segment"
            if plan.baseline_sampling_policy == DEFAULT_BASELINE_SAMPLING_POLICY
            else "finite_tr15_excluding_baseline_excluded_inside_one_continuous_segment"
        ),
        "filter_marker_role": (
            "audit_and_trade_chart_only"
            if plan.baseline_sampling_policy == DEFAULT_BASELINE_SAMPLING_POLICY
            else "baseline_eligibility_and_trade_chart_audit"
        ),
        "exit_mode": plan.exit_mode,
        "result_semantics_id": result_semantics_id(
            plan.entry_fill_mode,
            plan.entry_execution_policy,
            plan.entry_slippage,
            plan.exit_mode,
            plan.baseline_sampling_policy,
        ),
        "entry_signal_policy_id": ENTRY_SIGNAL_POLICY_ID,
    }
    if plan.instrument_profile is not None:
        payload["instrument_contract"] = {
            "experiment_mode": plan.experiment_mode,
            "campaign_manifest": str(plan.campaign_manifest),
            "instrument_profile": str(plan.instrument_profile),
            "instrument_id": plan.instrument_profile_contract["instrument_id"],
            "display_name": plan.instrument_profile_contract.get("display_name", plan.instrument_profile_contract["instrument_id"]),
            "strategy_contract_id": plan.instrument_profile_contract[
                "strategy_contract_id"
            ],
            "ranking_lineage_id": plan.ranking_lineage_id,
            "scenario_policy": plan.scenario_policy,
            "campaign_manifest_sha256": hashes.get("campaign_manifest_sha256"),
            "instrument_profile_sha256": hashes["instrument_profile_sha256"],
            "cost_model_sha256": hashes["cost_model_sha256"],
            "gap_policy": plan.instrument_profile_contract["gap_policy_contract"],
            "low_activity_policy": plan.instrument_profile_contract[
                "low_activity_policy_contract"
            ],
            "mode_validation": plan.mode_validation_contract,
        }
    return payload


def _fingerprint_payload_v1(plan: EffectivePlan, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": plan.campaign_id,
        "stage_id": plan.stage_id,
        "stage_kind": plan.stage_kind,
        "predecessor_stage_ids": list(plan.predecessor_stage_ids),
        "selection_provenance": plan.selection_provenance,
        "source": str(plan.source),
        "data_preparation_manifest": str(plan.data_preparation_manifest),
        "events": str(plan.events),
        "train_start": str(plan.train_start),
        "train_end": str(plan.train_end),
        "scenario_ids": list(plan.scenario_ids),
        "batch_size": plan.resources.batch_size,
        "combo_ids": [combo.combo_id for combo in plan.combos],
        "coordinate_labels": [
            {"combo_id": combo.combo_id, **plan.coordinate_labels[combo.combo_id]}
            for combo in plan.combos
        ],
        "hashes": hashes,
        "version_label": VERSION_LABEL,
        "strategy_id": plan.strategy_id,
        "exit_mode": plan.exit_mode,
    }


def _materialize_stage_contract(
    plan_path: Path,
    output: Path,
    plan: EffectivePlan,
    hashes: dict[str, str],
) -> tuple[str, pd.DataFrame]:
    grid = pd.DataFrame(_grid_rows(plan))
    fingerprint = _sha256_text(_canonical_json(_fingerprint_payload(plan, hashes)))
    audit_schema_version, audit_schema_id = trade_audit_identity(plan.exit_mode)
    stage_manifest_path = output / "stage_manifest.json"
    grid_path = output / "grid_manifest.csv"
    input_definition_path = output / "input_plan.json"
    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "materialized",
        "created_at": _utc_now(),
        "campaign_id": plan.campaign_id,
        "stage_id": plan.stage_id,
        "stage_kind": plan.stage_kind,
        "predecessor_stage_ids": list(plan.predecessor_stage_ids),
        "selection_provenance": plan.selection_provenance,
        "plan_fingerprint": fingerprint,
        "plan_fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "input_plan": str(plan_path.resolve()),
        "input_plan_sha256": plan.input_plan_sha256,
        "version_label": VERSION_LABEL,
        "strategy_id": plan.strategy_id,
        "trade_audit_schema_version": audit_schema_version,
        "trade_audit_schema_id": audit_schema_id,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "exit_mode": plan.exit_mode,
        "source": str(plan.source),
        "source_sha256": hashes["source_sha256"],
        "data_preparation_manifest": str(plan.data_preparation_manifest),
        "data_preparation_manifest_sha256": hashes["data_preparation_manifest_sha256"],
        "baseline_filter_atoms_sha256": hashes.get("filter_atoms_sha256"),
        "baseline_filter_events_sha256": hashes.get("filter_events_sha256"),
        "events": str(plan.events) if plan.scenario_definition is None else None,
        "events_sha256": hashes.get("events_sha256") or None,
        "scenario_definition": (
            str(plan.scenario_definition) if plan.scenario_definition is not None else None
        ),
        "scenario_definition_sha256": hashes.get("scenario_definition_sha256") or None,
        "scenario_schema_id": (
            plan.scenario_contract["scenario_schema_id"]
            if plan.scenario_contract is not None
            else None
        ),
        "scenario_selection_mode": plan.scenario_selection_mode,
        "engine_sha256": hashes["engine_sha256"],
        "runner_sha256": hashes["runner_sha256"],
        "entry_fill_mode": plan.entry_fill_mode,
        "entry_execution_policy": plan.entry_execution_policy,
        "entry_slippage": plan.entry_slippage,
        "baseline_sampling_policy": plan.baseline_sampling_policy,
        "baseline_filter_id": baseline_filter_id(plan.baseline_sampling_policy),
        "result_semantics_id": result_semantics_id(
            plan.entry_fill_mode,
            plan.entry_execution_policy,
            plan.entry_slippage,
            plan.exit_mode,
            plan.baseline_sampling_policy,
        ),
        "entry_signal_policy_id": ENTRY_SIGNAL_POLICY_ID,
        "train_start": str(plan.train_start),
        "train_end": str(plan.train_end),
        "scenario_ids": list(plan.scenario_ids),
        "coordinate_count": len(plan.combos),
        "duplicate_coordinate_count_removed": plan.duplicate_coordinate_count,
        "batch_count": int(math.ceil(len(plan.combos) / plan.resources.batch_size)),
        "batch_size": plan.resources.batch_size,
        "worker_cap": None,
        "initial_workers": plan.resources.workers,
        "minimum_free_memory_mb": plan.resources.minimum_free_memory_mb,
        "concurrency_contract": {
            "writer_scope": "one_process_per_stage_output_directory",
            "lock_file": str((output / ".v4_4_runner.lock").resolve()),
            "distinct_stage_outputs_may_run_concurrently": True,
            "shared_source_and_preparation_are_read_only": True,
        },
        "ranking_policy": "none; batch and stage summaries preserve raw metrics and scenario flags",
        "trade_storage": "immutable per-batch trades.csv; no monolithic stage trade copy",
        "artifacts": {
            "grid_manifest": str(grid_path.resolve()),
            "input_plan_copy": str(input_definition_path.resolve()),
        },
    }
    if plan.instrument_profile is not None:
        manifest["instrument_contract"] = {
            "experiment_mode": plan.experiment_mode,
            "campaign_manifest": str(plan.campaign_manifest),
            "campaign_manifest_sha256": hashes.get("campaign_manifest_sha256"),
            "instrument_profile": str(plan.instrument_profile),
            "instrument_profile_sha256": hashes["instrument_profile_sha256"],
            "instrument_id": plan.instrument_profile_contract["instrument_id"],
            "display_name": plan.instrument_profile_contract.get("display_name", plan.instrument_profile_contract["instrument_id"]),
            "strategy_contract_id": plan.instrument_profile_contract[
                "strategy_contract_id"
            ],
            "ranking_lineage_id": plan.ranking_lineage_id,
            "cost_model_path": plan.instrument_profile_contract[
                "resolved_cost_model_path"
            ],
            "cost_model_sha256": hashes["cost_model_sha256"],
            "scenario_policy": plan.scenario_policy,
            "gap_policy": plan.instrument_profile_contract["gap_policy_contract"],
            "low_activity_policy": plan.instrument_profile_contract[
                "low_activity_policy_contract"
            ],
            "mode_validation": plan.mode_validation_contract,
        }
    if stage_manifest_path.is_file():
        existing = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
        fingerprint_schema = int(existing.get("plan_fingerprint_schema_version", 1))
        if fingerprint_schema != FINGERPRINT_SCHEMA_VERSION:
            raise ValueError("V4.4 refuses to resume a V4 or unknown stage identity")
        if existing.get("plan_fingerprint") != fingerprint:
            raise ValueError("output directory contains a different stage contract")
        if not grid_path.is_file() or _sha256(grid_path) != existing.get("grid_manifest_sha256"):
            raise ValueError("existing grid manifest failed integrity validation")
        if not input_definition_path.is_file() or _sha256(input_definition_path) != existing.get("input_plan_copy_sha256"):
            raise ValueError("existing input-plan copy failed integrity validation")
        existing_grid = pd.read_csv(grid_path)
        if existing_grid.combo_id.astype(str).tolist() != grid.combo_id.astype(str).tolist():
            raise ValueError("existing grid manifest coordinates differ from the effective plan")
        return fingerprint, existing_grid

    output.mkdir(parents=True, exist_ok=True)
    input_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    _atomic_write_json(input_definition_path, input_payload)
    _atomic_write_csv(grid_path, grid)
    manifest["grid_manifest_sha256"] = _sha256(grid_path)
    manifest["input_plan_copy_sha256"] = _sha256(input_definition_path)
    _atomic_write_json(stage_manifest_path, manifest)
    return fingerprint, grid


def _batch_specs(plan: EffectivePlan, grid: pd.DataFrame) -> list[tuple[str, list[Combo], pd.DataFrame]]:
    output: list[tuple[str, list[Combo], pd.DataFrame]] = []
    by_id = {combo.combo_id: combo for combo in plan.combos}
    for batch_id, rows in grid.groupby("batch_id", sort=False):
        batch_rows = rows.sort_values("grid_ordinal", kind="mergesort").reset_index(drop=True)
        output.append((str(batch_id), [by_id[str(combo_id)] for combo_id in batch_rows.combo_id], batch_rows))
    return output


def _batch_manifest_is_complete(
    batch_dir: Path,
    batch_id: str,
    expected_combo_ids: Sequence[str],
    plan_fingerprint: str,
    baseline_sampling_policy: str,
) -> tuple[bool, dict[str, Any] | None]:
    manifest_path = batch_dir / "batch_manifest.json"
    if not manifest_path.is_file():
        return False, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return False, None
    if manifest.get("plan_fingerprint") != plan_fingerprint or manifest.get("batch_id") != batch_id:
        raise ValueError(f"completed batch contract mismatch: {batch_id}")
    if list(manifest.get("combo_ids", [])) != list(expected_combo_ids):
        raise ValueError(f"completed batch coordinate mismatch: {batch_id}")
    if manifest.get("baseline_sampling_policy") != baseline_sampling_policy:
        raise ValueError(f"completed batch baseline policy mismatch: {batch_id}")
    if manifest.get("baseline_filter_id") != baseline_filter_id(baseline_sampling_policy):
        raise ValueError(f"completed batch baseline filter identity mismatch: {batch_id}")
    if int(manifest.get("coordinate_count", -1)) != len(expected_combo_ids):
        raise ValueError(f"completed batch coordinate count mismatch: {batch_id}")
    artifacts = manifest.get("artifacts", {})
    required_artifacts = {
        "grid", "summary", "trades", "segment_qualification", "scenario_qualification"
    }
    if not isinstance(artifacts, dict) or not required_artifacts.issubset(artifacts):
        raise ValueError(f"completed batch artifact manifest is incomplete: {batch_id}")
    for artifact in artifacts.values():
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or _sha256(path) != artifact.get("sha256"):
            raise ValueError(f"completed batch artifact failed integrity validation: {batch_id} / {path.name}")
    return True, manifest


def _empty_trade_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["combo_id", "entry_time", "exit_time", "exit_reason"])


def _attach_scenario_flags(
    summaries: pd.DataFrame,
    details: pd.DataFrame,
    scenario_ids: Sequence[str],
    scenario_contract: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scenario_contract is not None:
        return attach_scenario_groups(summaries, details, scenario_contract)
    result = summaries.copy()
    if details.empty:
        for event_id in scenario_ids:
            result[f"{event_id}_qualified"] = False
    else:
        flags = details.pivot(index="combo_id", columns="event_id", values="qualified")
        for event_id in scenario_ids:
            values = flags[event_id] if event_id in flags else pd.Series(dtype=bool)
            result[f"{event_id}_qualified"] = result.combo_id.map(values).fillna(False).astype(bool)
    columns = [f"{event_id}_qualified" for event_id in scenario_ids]
    result["scenario_qualified_count"] = result[columns].sum(axis=1).astype(int)
    result["all_requested_scenarios_qualified"] = result[columns].all(axis=1).astype(bool)
    result["train_max_drawdown_abs"] = result["train_max_drawdown"].abs()
    scenario_rows: list[dict[str, Any]] = []
    for row in result[["combo_id", "method"]].itertuples(index=False):
        for scenario_id in scenario_ids:
            flag = bool(
                result.loc[
                    result.combo_id.astype(str).eq(str(row.combo_id)),
                    f"{scenario_id}_qualified",
                ].iloc[0]
            )
            scenario_rows.append(
                {
                    "combo_id": str(row.combo_id),
                    "method": str(row.method),
                    "scenario_id": scenario_id,
                    "scenario_label_zh": scenario_id,
                    "scenario_label_en": scenario_id,
                    "aggregation": "legacy_single_event",
                    "required_segment_ids": scenario_id,
                    "required_segment_count": 1,
                    "qualified_segment_count": int(flag),
                    "qualified": flag,
                    "failed_segment_ids": "" if flag else scenario_id,
                }
            )
    return result, pd.DataFrame(scenario_rows)


def _execute_batch_results(
    combos: list[Combo],
    workers: int,
    executor: ProcessPoolExecutor | None,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    if workers == 1:
        return [_worker_run(combo) for combo in combos]
    if executor is None:
        raise RuntimeError("process executor is unavailable")
    return list(executor.map(_worker_run, combos, chunksize=1))


BatchExecutor = Callable[[list[Combo]], list[tuple[dict[str, Any], list[dict[str, Any]]]]]


def _write_batch(
    batch_dir: Path,
    batch_id: str,
    grid_rows: pd.DataFrame,
    results: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    events: pd.DataFrame,
    scenario_ids: Sequence[str],
    scenario_contract: dict[str, Any] | None,
    plan_fingerprint: str,
    workers: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if len(results) != len(grid_rows):
        raise AssertionError(f"{batch_id} returned an unexpected result count")
    summaries = pd.DataFrame([summary for summary, _ in results])
    trades = pd.DataFrame([trade for _, combo_trades in results for trade in combo_trades])
    if trades.empty:
        trades = _empty_trade_frame()
    if scenario_contract is not None:
        details = evaluate_segment_qualification(summaries, trades, scenario_contract)
        qualified, scenario_details = _attach_scenario_flags(
            summaries,
            details,
            scenario_ids,
            scenario_contract,
        )
        qualified["train_max_drawdown_abs"] = qualified["train_max_drawdown"].abs()
    else:
        details, raw_qualified = _event_metrics(summaries, trades, events, ())
        qualified, scenario_details = _attach_scenario_flags(
            raw_qualified,
            details,
            scenario_ids,
        )
    ordinal_map = dict(zip(grid_rows.combo_id.astype(str), grid_rows.grid_ordinal.astype(int), strict=True))
    qualified.insert(0, "grid_ordinal", qualified.combo_id.astype(str).map(ordinal_map).astype(int))
    qualified.insert(1, "batch_id", batch_id)
    qualified = qualified.sort_values("grid_ordinal", kind="mergesort").reset_index(drop=True)
    details.insert(0, "batch_id", batch_id)
    scenario_details.insert(0, "batch_id", batch_id)
    trades.insert(0, "batch_id", batch_id)
    strategy_ids = summaries["strategy_id"].astype(str).unique().tolist()
    exit_modes = summaries["exit_mode"].astype(str).unique().tolist()
    if len(strategy_ids) != 1 or len(exit_modes) != 1:
        raise AssertionError("one batch must contain one strategy and exit contract")
    audit_schema_version, audit_schema_id = trade_audit_identity(exit_modes[0])
    if not trades.empty:
        trade_audit_ids = trades["trade_audit_schema_id"].astype(str).unique().tolist()
        trade_audit_versions = trades["trade_audit_schema_version"].astype(int).unique().tolist()
        rebound_policy_ids = trades["rebound_baseline_policy_id"].astype(str).unique().tolist()
        if (
            trade_audit_ids != [audit_schema_id]
            or trade_audit_versions != [audit_schema_version]
            or rebound_policy_ids != [REBOUND_BASELINE_POLICY_ID]
        ):
            raise AssertionError("batch trades do not match the max-W audit identity")

    summary_path = batch_dir / "summary.csv"
    trades_path = batch_dir / "trades.csv"
    details_path = batch_dir / "segment_qualification.csv"
    scenario_details_path = batch_dir / "scenario_qualification.csv"
    grid_path = batch_dir / "grid.csv"
    _atomic_write_csv(grid_path, grid_rows)
    _atomic_write_csv(summary_path, qualified)
    _atomic_write_csv(trades_path, trades)
    _atomic_write_csv(details_path, details)
    _atomic_write_csv(scenario_details_path, scenario_details)
    artifacts = {
        "grid": _artifact_record(grid_path),
        "summary": _artifact_record(summary_path),
        "trades": _artifact_record(trades_path),
        "segment_qualification": _artifact_record(details_path),
        "scenario_qualification": _artifact_record(scenario_details_path),
    }
    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "complete",
        "completed_at": _utc_now(),
        "plan_fingerprint": plan_fingerprint,
        "batch_id": batch_id,
        "combo_ids": grid_rows.combo_id.astype(str).tolist(),
        "coordinate_count": int(len(qualified)),
        "trade_count": int(len(trades)),
        "segment_qualification_count": int(len(details)),
        "scenario_qualification_count": int(len(scenario_details)),
        "scenario_ids": list(scenario_ids),
        "scenario_selection_mode": (
            scenario_contract["selection_mode"]
            if scenario_contract is not None
            else "legacy_multiple"
        ),
        "scenario_schema_id": (
            scenario_contract["scenario_schema_id"]
            if scenario_contract is not None
            else None
        ),
        "strategy_id": strategy_ids[0],
        "trade_audit_schema_version": audit_schema_version,
        "trade_audit_schema_id": audit_schema_id,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "baseline_sampling_policy": str(
            summaries["baseline_sampling_policy"].iloc[0]
        ),
        "baseline_filter_id": baseline_filter_id(
            str(summaries["baseline_sampling_policy"].iloc[0])
        ),
        "exit_mode": exit_modes[0],
        "speed_window_bars": sorted(
            int(value) for value in summaries["speed_window_bars"].unique()
        ),
        "workers": workers,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "raw_metric_columns": [
            "train_return",
            "train_avg_trade",
            "train_max_drawdown",
            "train_max_drawdown_abs",
            "train_trade_count",
            "train_return_excluding_gap_spanning_trades",
        ],
        "ranking_policy": "none",
        "artifacts": artifacts,
    }
    _atomic_write_json(batch_dir / "batch_manifest.json", manifest)
    return manifest


def _write_progress(
    path: Path,
    *,
    plan: EffectivePlan,
    plan_fingerprint: str,
    status: str,
    completed_manifests: Sequence[dict[str, Any]],
    current_batch: str | None,
    started_at: str,
    available_memory_mb: int | None,
    last_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed_ids = [str(item["batch_id"]) for item in completed_manifests]
    completed_coordinates = int(sum(int(item["coordinate_count"]) for item in completed_manifests))
    audit_schema_version, audit_schema_id = trade_audit_identity(plan.exit_mode)
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "campaign_id": plan.campaign_id,
        "stage_id": plan.stage_id,
        "version_label": VERSION_LABEL,
        "strategy_id": plan.strategy_id,
        "trade_audit_schema_version": audit_schema_version,
        "trade_audit_schema_id": audit_schema_id,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "exit_mode": plan.exit_mode,
        "entry_fill_mode": plan.entry_fill_mode,
        "entry_execution_policy": plan.entry_execution_policy,
        "entry_slippage": plan.entry_slippage,
        "baseline_sampling_policy": plan.baseline_sampling_policy,
        "baseline_filter_id": baseline_filter_id(plan.baseline_sampling_policy),
        "result_semantics_id": result_semantics_id(
            plan.entry_fill_mode,
            plan.entry_execution_policy,
            plan.entry_slippage,
            plan.exit_mode,
            plan.baseline_sampling_policy,
        ),
        "plan_fingerprint": plan_fingerprint,
        "scenario_ids": list(plan.scenario_ids),
        "scenario_selection_mode": plan.scenario_selection_mode,
        "scenario_schema_id": (
            plan.scenario_contract["scenario_schema_id"]
            if plan.scenario_contract is not None
            else None
        ),
        "status": status,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "current_batch": current_batch,
        "total_batches": int(math.ceil(len(plan.combos) / plan.resources.batch_size)),
        "completed_batches": completed_ids,
        "completed_batch_count": len(completed_ids),
        "total_coordinate_count": len(plan.combos),
        "completed_coordinate_count": completed_coordinates,
        "remaining_coordinate_count": len(plan.combos) - completed_coordinates,
        "workers": plan.resources.workers,
        "worker_cap": None,
        "batch_size": plan.resources.batch_size,
        "minimum_free_memory_mb": plan.resources.minimum_free_memory_mb,
        "available_memory_mb": available_memory_mb,
        "last_error": last_error,
    }
    _atomic_write_json(path, payload)
    return payload


def _finalize_stage(
    output: Path,
    plan: EffectivePlan,
    plan_fingerprint: str,
    completed_manifests: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    audit_schema_version, audit_schema_id = trade_audit_identity(plan.exit_mode)
    summaries: list[pd.DataFrame] = []
    details: list[pd.DataFrame] = []
    scenario_details: list[pd.DataFrame] = []
    batch_index: list[dict[str, Any]] = []
    for manifest in completed_manifests:
        summary_path = Path(manifest["artifacts"]["summary"]["path"])
        details_path = Path(manifest["artifacts"]["segment_qualification"]["path"])
        scenario_details_path = Path(manifest["artifacts"]["scenario_qualification"]["path"])
        summaries.append(pd.read_csv(summary_path))
        details.append(pd.read_csv(details_path))
        scenario_details.append(pd.read_csv(scenario_details_path))
        batch_index.append(
            {
                "batch_id": manifest["batch_id"],
                "coordinate_count": manifest["coordinate_count"],
                "trade_count": manifest["trade_count"],
                "manifest": str((output / "batches" / manifest["batch_id"] / "batch_manifest.json").resolve()),
                "artifacts": manifest["artifacts"],
            }
        )
    summary = pd.concat(summaries, ignore_index=True, sort=False).sort_values("grid_ordinal", kind="mergesort")
    qualification = pd.concat(details, ignore_index=True, sort=False)
    scenario_qualification = pd.concat(scenario_details, ignore_index=True, sort=False)
    if summary.combo_id.astype(str).duplicated().any() or len(summary) != len(plan.combos):
        raise AssertionError("completed stage summary does not contain one row per planned coordinate")
    summary_path = output / "stage_summary.csv"
    details_path = output / "stage_segment_qualification.csv"
    scenario_details_path = output / "stage_scenario_qualification.csv"
    index_path = output / "batch_index.json"
    _atomic_write_csv(summary_path, summary)
    _atomic_write_csv(details_path, qualification)
    _atomic_write_csv(scenario_details_path, scenario_qualification)
    _atomic_write_json(index_path, {"schema_version": OUTPUT_SCHEMA_VERSION, "batches": batch_index})
    completion = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "complete",
        "completed_at": _utc_now(),
        "campaign_id": plan.campaign_id,
        "stage_id": plan.stage_id,
        "stage_kind": plan.stage_kind,
        "version_label": VERSION_LABEL,
        "strategy_id": plan.strategy_id,
        "trade_audit_schema_version": audit_schema_version,
        "trade_audit_schema_id": audit_schema_id,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "exit_mode": plan.exit_mode,
        "entry_fill_mode": plan.entry_fill_mode,
        "entry_execution_policy": plan.entry_execution_policy,
        "entry_slippage": plan.entry_slippage,
        "baseline_sampling_policy": plan.baseline_sampling_policy,
        "baseline_filter_id": baseline_filter_id(plan.baseline_sampling_policy),
        "result_semantics_id": result_semantics_id(
            plan.entry_fill_mode,
            plan.entry_execution_policy,
            plan.entry_slippage,
            plan.exit_mode,
            plan.baseline_sampling_policy,
        ),
        "plan_fingerprint": plan_fingerprint,
        "coordinate_count": int(len(summary)),
        "trade_count": int(sum(int(item["trade_count"]) for item in completed_manifests)),
        "batch_count": len(completed_manifests),
        "scenario_ids": list(plan.scenario_ids),
        "scenario_selection_mode": plan.scenario_selection_mode,
        "scenario_schema_id": (
            plan.scenario_contract["scenario_schema_id"]
            if plan.scenario_contract is not None
            else None
        ),
        "scenario_definition": (
            str(plan.scenario_definition) if plan.scenario_definition is not None else None
        ),
        "ranking_policy": "none; use stage_summary.csv to apply an explicitly approved research view",
        "artifacts": {
            "stage_summary": _artifact_record(summary_path),
            "stage_segment_qualification": _artifact_record(details_path),
            "stage_scenario_qualification": _artifact_record(scenario_details_path),
            "batch_index": _artifact_record(index_path),
        },
    }
    _atomic_write_json(output / "completion_manifest.json", completion)
    return completion


def _load_valid_completion(output: Path, plan: EffectivePlan, plan_fingerprint: str) -> dict[str, Any] | None:
    path = output / "completion_manifest.json"
    if not path.is_file():
        return None
    completion = json.loads(path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("plan_fingerprint") != plan_fingerprint:
        raise ValueError("completion manifest does not match the current stage contract")
    if int(completion.get("coordinate_count", -1)) != len(plan.combos):
        raise ValueError("completion manifest coordinate count does not match the stage grid")
    if completion.get("baseline_sampling_policy") != plan.baseline_sampling_policy:
        raise ValueError("completion manifest baseline policy does not match the stage plan")
    if completion.get("baseline_filter_id") != baseline_filter_id(plan.baseline_sampling_policy):
        raise ValueError("completion manifest baseline filter identity does not match the stage plan")
    artifacts = completion.get("artifacts", {})
    required = {
        "stage_summary", "stage_segment_qualification", "stage_scenario_qualification", "batch_index"
    }
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise ValueError("completion artifact manifest is incomplete")
    for artifact in artifacts.values():
        artifact_path = Path(str(artifact.get("path", "")))
        if not artifact_path.is_file() or _sha256(artifact_path) != artifact.get("sha256"):
            raise ValueError(f"completion artifact failed integrity validation: {artifact_path.name}")
    return completion


@contextmanager
def _exclusive_stage_writer(output: Path) -> Iterable[Path]:
    """Allow one writer per stage output while preserving cross-stage concurrency."""
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".v4_4_runner.lock"
    handle = lock_path.open("a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            locked = True
        except OSError as error:
            raise RuntimeError(
                "another V4.4 runner is already writing this stage output; "
                "use a distinct campaign/stage/output identity for concurrent work"
            ) from error
        yield lock_path
    finally:
        if locked:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def _run_stage_locked(
    plan_path: Path,
    output: Path,
    *,
    workers: int | None = None,
    batch_size: int | None = None,
    minimum_free_memory_mb: int | None = None,
    validate_only: bool = False,
    stop_after_new_batches: int | None = None,
    _batch_executor: BatchExecutor | None = None,
    _contract_validator: Callable[[EffectivePlan], tuple[pd.DataFrame, dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Execute or resume one explicit stage plan.

    ``_batch_executor`` and ``_contract_validator`` are narrow test seams.  CLI
    runs always use V4 simulation and the audited source/preparation contract.
    """
    if stop_after_new_batches is not None and stop_after_new_batches < 1:
        raise ValueError("stop_after_new_batches must be positive")
    plan_path = plan_path.resolve()
    output = output.resolve()
    plan = load_plan(
        plan_path,
        workers=workers,
        batch_size=batch_size,
        minimum_free_memory_mb=minimum_free_memory_mb,
    )
    validator = _contract_validator or _validate_contract
    events, hashes = validator(plan)
    plan_fingerprint, grid = _materialize_stage_contract(plan_path, output, plan, hashes)
    batch_specs = _batch_specs(plan, grid)
    completed_manifests: list[dict[str, Any]] = []
    pending: list[tuple[str, list[Combo], pd.DataFrame]] = []
    for batch_id, combos, grid_rows in batch_specs:
        batch_dir = output / "batches" / batch_id
        complete, manifest = _batch_manifest_is_complete(
            batch_dir,
            batch_id,
            [combo.combo_id for combo in combos],
            plan_fingerprint,
            plan.baseline_sampling_policy,
        )
        if complete:
            assert manifest is not None
            completed_manifests.append(manifest)
        else:
            pending.append((batch_id, combos, grid_rows))
    started_at = _utc_now()
    available = _enforce_memory_floor(plan.resources.minimum_free_memory_mb)
    if validate_only:
        progress = _write_progress(
            output / "progress.json",
            plan=plan,
            plan_fingerprint=plan_fingerprint,
            status="ready" if pending else "complete",
            completed_manifests=completed_manifests,
            current_batch=None,
            started_at=started_at,
            available_memory_mb=available,
        )
        return {
            "status": progress["status"],
            "output": str(output),
            "plan_fingerprint": plan_fingerprint,
            "coordinate_count": len(plan.combos),
            "completed_coordinate_count": progress["completed_coordinate_count"],
            "pending_batch_count": len(pending),
        }
    if not pending:
        completion = _load_valid_completion(output, plan, plan_fingerprint)
        if completion is None:
            completion = _finalize_stage(output, plan, plan_fingerprint, completed_manifests)
        _write_progress(
            output / "progress.json",
            plan=plan,
            plan_fingerprint=plan_fingerprint,
            status="complete",
            completed_manifests=completed_manifests,
            current_batch=None,
            started_at=started_at,
            available_memory_mb=available,
        )
        return {**completion, "output": str(output), "resumed_without_execution": True}

    runtime = RuntimeSpec(plan.source, plan.data_preparation_manifest, plan.train_start, plan.train_end)
    executor: ProcessPoolExecutor | None = None
    new_batch_count = 0
    try:
        if _batch_executor is None:
            initargs = (
                str(runtime.source),
                str(runtime.data_preparation_manifest),
                str(runtime.train_start),
                str(runtime.train_end),
            )
            if plan.resources.workers == 1:
                _worker_init(*initargs)
            else:
                executor = ProcessPoolExecutor(
                    max_workers=plan.resources.workers,
                    initializer=_worker_init,
                    initargs=initargs,
                )
        for batch_id, combos, grid_rows in pending:
            available = _enforce_memory_floor(plan.resources.minimum_free_memory_mb)
            _write_progress(
                output / "progress.json",
                plan=plan,
                plan_fingerprint=plan_fingerprint,
                status="running",
                completed_manifests=completed_manifests,
                current_batch=batch_id,
                started_at=started_at,
                available_memory_mb=available,
            )
            batch_started = time.perf_counter()
            results = (
                _batch_executor(combos)
                if _batch_executor is not None
                else _execute_batch_results(combos, plan.resources.workers, executor)
            )
            manifest = _write_batch(
                output / "batches" / batch_id,
                batch_id,
                grid_rows,
                results,
                events,
                plan.scenario_ids,
                plan.scenario_contract,
                plan_fingerprint,
                plan.resources.workers,
                time.perf_counter() - batch_started,
            )
            completed_manifests.append(manifest)
            new_batch_count += 1
            print(
                json.dumps(
                    {
                        "stage_id": plan.stage_id,
                        "batch_id": batch_id,
                        "completed_batches": len(completed_manifests),
                        "total_batches": len(batch_specs),
                        "completed_coordinates": sum(int(item["coordinate_count"]) for item in completed_manifests),
                        "total_coordinates": len(plan.combos),
                        "available_memory_mb": available,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if stop_after_new_batches is not None and new_batch_count >= stop_after_new_batches:
                progress = _write_progress(
                    output / "progress.json",
                    plan=plan,
                    plan_fingerprint=plan_fingerprint,
                    status="paused",
                    completed_manifests=completed_manifests,
                    current_batch=None,
                    started_at=started_at,
                    available_memory_mb=available,
                )
                return {
                    "status": "paused",
                    "output": str(output),
                    "plan_fingerprint": plan_fingerprint,
                    "completed_coordinate_count": progress["completed_coordinate_count"],
                    "remaining_coordinate_count": progress["remaining_coordinate_count"],
                    "new_batch_count": new_batch_count,
                }
        completion = _finalize_stage(output, plan, plan_fingerprint, completed_manifests)
        _write_progress(
            output / "progress.json",
            plan=plan,
            plan_fingerprint=plan_fingerprint,
            status="complete",
            completed_manifests=completed_manifests,
            current_batch=None,
            started_at=started_at,
            available_memory_mb=available,
        )
        return {**completion, "output": str(output), "new_batch_count": new_batch_count}
    except KeyboardInterrupt:
        _write_progress(
            output / "progress.json",
            plan=plan,
            plan_fingerprint=plan_fingerprint,
            status="interrupted",
            completed_manifests=completed_manifests,
            current_batch=None,
            started_at=started_at,
            available_memory_mb=_available_memory_mb(),
        )
        raise
    except BaseException as error:
        _write_progress(
            output / "progress.json",
            plan=plan,
            plan_fingerprint=plan_fingerprint,
            status="failed",
            completed_manifests=completed_manifests,
            current_batch=None,
            started_at=started_at,
            available_memory_mb=_available_memory_mb(),
            last_error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(limit=12),
            },
        )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)


def run_stage(
    plan_path: Path,
    output: Path,
    *,
    workers: int | None = None,
    batch_size: int | None = None,
    minimum_free_memory_mb: int | None = None,
    validate_only: bool = False,
    stop_after_new_batches: int | None = None,
    deliver_html: bool = False,
    delivery_mode: str = "background",
    review_workers: int = 4,
    union_campaigns_root: Path | None = None,
    union_output_root: Path | None = None,
    _batch_executor: BatchExecutor | None = None,
    _contract_validator: Callable[
        [EffectivePlan], tuple[pd.DataFrame, dict[str, str]]
    ]
    | None = None,
) -> dict[str, Any]:
    """Execute one stage and publish HTML only when explicitly requested.

    Intermediate exploration rounds close immutable raw evidence without
    rebuilding the cumulative pages.  The final round may opt into one HTML
    publication after the exploration series is complete.
    """
    resolved_plan = Path(plan_path).resolve()
    resolved_output = output.resolve()
    if delivery_mode not in {"background", "synchronous"}:
        raise ValueError("delivery_mode must be background or synchronous")
    if int(review_workers) < 1 or int(review_workers) > 32:
        raise ValueError("review_workers must be between 1 and 32")
    with _exclusive_stage_writer(resolved_output):
        result = _run_stage_locked(
            resolved_plan,
            resolved_output,
            workers=workers,
            batch_size=batch_size,
            minimum_free_memory_mb=minimum_free_memory_mb,
            validate_only=validate_only,
            stop_after_new_batches=stop_after_new_batches,
            _batch_executor=_batch_executor,
            _contract_validator=_contract_validator,
        )
    if result.get("status") == "complete" and not validate_only and deliver_html:
        from run_v4_4_delivery_worker import deliver, launch_delivery

        campaigns_root = union_campaigns_root.resolve() if union_campaigns_root else None
        if campaigns_root is None:
            campaigns_root = next(
                (
                    parent
                    for parent in (resolved_output, *resolved_output.parents)
                    if parent.name == "campaigns"
                ),
                resolved_output.parent,
            )
        cumulative_output = (
            union_output_root.resolve()
            if union_output_root
            else Path(__file__).resolve().parents[3]
            / "results"
            / "all_completed_union_analysis"
        )
        if delivery_mode == "synchronous":
            delivery_result = deliver(
                resolved_plan,
                resolved_output,
                campaigns_root,
                cumulative_output,
                review_workers=int(review_workers),
            )
        else:
            delivery_result = launch_delivery(
                resolved_plan,
                resolved_output,
                campaigns_root,
                cumulative_output,
                review_workers=int(review_workers),
            )
        result["html_delivery"] = delivery_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or resume a bounded-memory V4 parameter stage.")
    parser.add_argument("--plan", required=True, help="Path to the JSON stage/grid plan")
    parser.add_argument("--output", required=True, help="Output directory for this stage")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--minimum-free-memory-mb", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--stop-after-new-batches", type=int)
    parser.add_argument(
        "--publish-html",
        action="store_true",
        help="Publish the cumulative main and shared per-trade HTML after this final round",
    )
    parser.add_argument(
        "--delivery-mode",
        choices=("background", "synchronous"),
        default="background",
    )
    parser.add_argument("--review-workers", type=int, default=4)
    args = parser.parse_args()
    result = run_stage(
        Path(args.plan),
        Path(args.output),
        workers=args.workers,
        batch_size=args.batch_size,
        minimum_free_memory_mb=args.minimum_free_memory_mb,
        validate_only=args.validate_only,
        stop_after_new_batches=args.stop_after_new_batches,
        deliver_html=args.publish_html,
        delivery_mode=args.delivery_mode,
        review_workers=args.review_workers,
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
