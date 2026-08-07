"""Versioned single-select scenario groups for V4.4 qualification."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SCENARIO_SCHEMA_VERSION = 2
SCENARIO_SCHEMA_ID = "v4_4_scenario_groups_single_select_v2_20260801"
COMBINED_SCENARIO_SCHEMA_VERSION = 3
COMBINED_SCENARIO_SCHEMA_ID = (
    "v4_4_scenario_groups_single_select_combined_exit_v3_20260801"
)
SELECTION_MODE = "single"
NEUTRAL_SELECTION_ID = "all"
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_SCENARIO_DEFINITION = (
    Path(__file__).resolve().parent.parent
    / "plans"
    / "v4_4_scenario_groups_single_select_combined_exit_20260801.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_id(value: Any, field: str) -> str:
    result = str(value).strip()
    if not result or not ID_PATTERN.fullmatch(result):
        raise ValueError(f"{field} must contain only letters, numbers, underscore, or hyphen")
    return result


def load_scenario_contract(path: Path = DEFAULT_SCENARIO_DEFINITION) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", 0))
    schema_id = str(payload.get("scenario_schema_id", ""))
    supported_schemas = {
        SCENARIO_SCHEMA_ID: SCENARIO_SCHEMA_VERSION,
        COMBINED_SCENARIO_SCHEMA_ID: COMBINED_SCENARIO_SCHEMA_VERSION,
    }
    if schema_id not in supported_schemas:
        raise ValueError("scenario schema identity is not supported by V4.4")
    if schema_version != supported_schemas[schema_id]:
        raise ValueError("scenario schema_version does not match its schema identity")
    if payload.get("selection_mode") != SELECTION_MODE:
        raise ValueError("V4.4 scenario selection must be single-select")
    if payload.get("neutral_selection_id") != NEUTRAL_SELECTION_ID:
        raise ValueError("scenario neutral selection identity does not match the active contract")

    rule = payload.get("qualification_rule")
    required_common_rule = {
        "entry_interval": "start_exclusive_end_inclusive",
        "exit_interval": "start_exclusive_end_inclusive",
        "required_entry_count": 1,
        "required_exit_count": 0,
        "must_hold_past_segment_end": True,
    }
    if not isinstance(rule, dict) or any(
        rule.get(name) != value for name, value in required_common_rule.items()
    ):
        raise ValueError("scenario qualification rule differs from the approved V4.4 contract")
    if schema_id == SCENARIO_SCHEMA_ID:
        if set(rule) != {*required_common_rule, "required_eventual_exit_reason"}:
            raise ValueError("rebound-only scenario qualification fields differ from schema v2")
        exit_reasons = [str(rule.get("required_eventual_exit_reason", ""))]
        if exit_reasons != ["rebound_threshold"]:
            raise ValueError("schema-v2 scenarios require eventual rebound_threshold exit")
    else:
        if set(rule) != {*required_common_rule, "required_eventual_exit_reasons"}:
            raise ValueError("combined-exit scenario qualification fields differ from schema v3")
        raw_exit_reasons = rule.get("required_eventual_exit_reasons")
        if not isinstance(raw_exit_reasons, list):
            raise ValueError("combined-exit scenarios require an eventual-exit-reason list")
        exit_reasons = [str(reason) for reason in raw_exit_reasons]
        if exit_reasons != ["rebound_threshold", "downside_speed_below_threshold"]:
            raise ValueError("schema-v3 scenarios require rebound and speed eventual exits")

    segments = payload.get("segments")
    scenarios = payload.get("scenarios")
    if not isinstance(segments, list) or not segments:
        raise ValueError("scenario definition requires at least one market segment")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenario definition requires at least one scenario")

    segment_ids: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("each market segment must be an object")
        segment_id = _safe_id(segment.get("segment_id"), "segment_id")
        begin = pd.Timestamp(segment.get("start_time"))
        finish = pd.Timestamp(segment.get("end_time"))
        if begin >= finish:
            raise ValueError(f"market segment must have start < end: {segment_id}")
        segment_ids.append(segment_id)
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("market segment identifiers must be unique")

    known_segments = set(segment_ids)
    scenario_ids: list[str] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("each scenario must be an object")
        scenario_id = _safe_id(scenario.get("scenario_id"), "scenario_id")
        if scenario_id == NEUTRAL_SELECTION_ID:
            raise ValueError("a scenario cannot reuse the neutral selection identity")
        if scenario.get("aggregation") != "all":
            raise ValueError("every V4.4 scenario must use internal AND aggregation")
        members = scenario.get("segment_ids")
        if not isinstance(members, list) or not members:
            raise ValueError(f"scenario must contain one or more market segments: {scenario_id}")
        normalised = [_safe_id(item, "scenario segment_id") for item in members]
        if len(set(normalised)) != len(normalised):
            raise ValueError(f"scenario contains duplicate market segments: {scenario_id}")
        unknown = set(normalised).difference(known_segments)
        if unknown:
            raise ValueError(f"scenario contains unknown market segments: {sorted(unknown)}")
        scenario_ids.append(scenario_id)
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario identifiers must be unique")

    payload["definition_path"] = str(path)
    payload["definition_sha256"] = sha256_file(path)
    payload["qualification_exit_reasons"] = exit_reasons
    return payload


def segments_frame(contract: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": segment["segment_id"],
                "segment_id": segment["segment_id"],
                "label_zh": segment["label_zh"],
                "label_en": segment["label_en"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
            }
            for segment in contract["segments"]
        ]
    )


def evaluate_segment_qualification(
    coordinates: pd.DataFrame,
    trades: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    required_coordinates = {"combo_id", "method", "baseline_sampling_policy"}
    required_trades = {"combo_id", "entry_time", "exit_time", "exit_reason"}
    missing_coordinates = required_coordinates.difference(coordinates.columns)
    missing_trades = required_trades.difference(trades.columns)
    if missing_coordinates:
        raise ValueError(f"coordinate rows lack fields: {sorted(missing_coordinates)}")
    if missing_trades:
        raise ValueError(f"trade rows lack fields: {sorted(missing_trades)}")
    if coordinates.combo_id.astype(str).duplicated().any():
        raise ValueError("coordinate rows must contain one row per combo_id")

    working = trades.copy()
    working["entry_time"] = pd.to_datetime(working["entry_time"], errors="raise")
    working["exit_time"] = pd.to_datetime(working["exit_time"], errors="raise")
    trade_groups = {
        str(combo_id): group.copy()
        for combo_id, group in working.groupby("combo_id", sort=False)
    }
    empty = working.iloc[0:0].copy()
    allowed_exit_reasons = set(contract["qualification_exit_reasons"])
    rows: list[dict[str, Any]] = []
    for coordinate in coordinates[
        ["combo_id", "method", "baseline_sampling_policy"]
    ].itertuples(index=False):
        combo_id = str(coordinate.combo_id)
        combo_trades = trade_groups.get(combo_id, empty)
        for segment in contract["segments"]:
            begin = pd.Timestamp(segment["start_time"])
            finish = pd.Timestamp(segment["end_time"])
            entered = combo_trades.loc[
                combo_trades.entry_time.gt(begin) & combo_trades.entry_time.le(finish)
            ]
            exited = combo_trades.loc[
                combo_trades.exit_time.gt(begin) & combo_trades.exit_time.le(finish)
            ]
            selected = entered.iloc[0] if len(entered) == 1 else None
            selected_exit_reason = "" if selected is None else str(selected.exit_reason)
            holds_past_end = bool(
                selected is not None and pd.Timestamp(selected.exit_time) > finish
            )
            qualified = bool(
                len(entered) == 1
                and len(exited) == 0
                and holds_past_end
                and selected_exit_reason in allowed_exit_reasons
            )
            failures: list[str] = []
            if len(entered) != 1:
                failures.append(f"entry_count_{len(entered)}")
            if len(exited) != 0:
                failures.append(f"exit_count_inside_{len(exited)}")
            if selected is not None and not holds_past_end:
                failures.append("selected_trade_does_not_hold_past_end")
            if selected is not None and selected_exit_reason not in allowed_exit_reasons:
                failures.append(f"selected_exit_reason_{selected_exit_reason}")
            rows.append(
                {
                    "combo_id": combo_id,
                    "method": str(coordinate.method),
                    "baseline_sampling_policy": str(
                        coordinate.baseline_sampling_policy
                    ),
                    "segment_id": segment["segment_id"],
                    "segment_label_zh": segment["label_zh"],
                    "segment_label_en": segment["label_en"],
                    "segment_start_time": segment["start_time"],
                    "segment_end_time": segment["end_time"],
                    "qualified": qualified,
                    "entry_count_in_interval": int(len(entered)),
                    "exit_count_in_interval": int(len(exited)),
                    "holds_past_segment_end": holds_past_end,
                    "selected_entry_time": "" if selected is None else str(selected.entry_time),
                    "selected_exit_time": "" if selected is None else str(selected.exit_time),
                    "selected_exit_reason": selected_exit_reason,
                    "failure_reasons": "|".join(failures),
                }
            )
    return pd.DataFrame(rows)


def attach_scenario_groups(
    coordinates: pd.DataFrame,
    segment_details: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = coordinates.copy()
    expected_detail_rows = len(result) * len(contract["segments"])
    if len(segment_details) != expected_detail_rows:
        raise ValueError("segment qualification does not cover every coordinate/segment pair")
    if segment_details.duplicated(["combo_id", "segment_id"]).any():
        raise ValueError("segment qualification contains duplicate coordinate/segment rows")

    flag_lookup = {
        (str(row.combo_id), str(row.segment_id)): bool(row.qualified)
        for row in segment_details.itertuples(index=False)
    }
    scenario_rows: list[dict[str, Any]] = []
    for coordinate in result[
        ["combo_id", "method", "baseline_sampling_policy"]
    ].itertuples(index=False):
        combo_id = str(coordinate.combo_id)
        for scenario in contract["scenarios"]:
            members = [str(item) for item in scenario["segment_ids"]]
            member_flags = [flag_lookup[(combo_id, segment_id)] for segment_id in members]
            failed = [
                segment_id
                for segment_id, qualified in zip(members, member_flags, strict=True)
                if not qualified
            ]
            scenario_rows.append(
                {
                    "combo_id": combo_id,
                    "method": str(coordinate.method),
                    "baseline_sampling_policy": str(
                        coordinate.baseline_sampling_policy
                    ),
                    "scenario_id": scenario["scenario_id"],
                    "scenario_label_zh": scenario["label_zh"],
                    "scenario_label_en": scenario["label_en"],
                    "aggregation": "all",
                    "required_segment_ids": "|".join(members),
                    "required_segment_count": int(len(members)),
                    "qualified_segment_count": int(sum(member_flags)),
                    "qualified": bool(all(member_flags)),
                    "failed_segment_ids": "|".join(failed),
                }
            )
    scenario_details = pd.DataFrame(scenario_rows)
    for scenario in contract["scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        flags = scenario_details.loc[
            scenario_details.scenario_id.eq(scenario_id), ["combo_id", "qualified"]
        ].set_index("combo_id")["qualified"]
        result[f"{scenario_id}_qualified"] = (
            result.combo_id.astype(str).map(flags).fillna(False).astype(bool)
        )
    scenario_columns = [
        f"{scenario['scenario_id']}_qualified" for scenario in contract["scenarios"]
    ]
    result["qualified_scenario_count"] = result[scenario_columns].sum(axis=1).astype(int)
    result["scenario_schema_id"] = contract["scenario_schema_id"]
    result["scenario_selection_mode"] = contract["selection_mode"]
    return result, scenario_details


def filter_single_scenario(
    coordinates: pd.DataFrame,
    scenario_id: str | None,
    contract: dict[str, Any],
) -> pd.DataFrame:
    if scenario_id in (None, "", contract["neutral_selection_id"]):
        return coordinates.copy()
    if not isinstance(scenario_id, str):
        raise TypeError("scenario selection accepts exactly one scenario identifier")
    known = {str(item["scenario_id"]) for item in contract["scenarios"]}
    if scenario_id not in known:
        raise ValueError(f"unknown scenario selection: {scenario_id}")
    column = f"{scenario_id}_qualified"
    if column not in coordinates.columns:
        raise ValueError(f"coordinate rows lack scenario qualification column: {column}")
    return coordinates.loc[coordinates[column].fillna(False).astype(bool)].copy()
