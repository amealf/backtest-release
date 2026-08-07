"""Build the closed V4.4 user home and lazy trade-analysis delivery."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from run_v4_4_resumable_campaign import (
    FINGERPRINT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
)
from v4_4_engine import (
    COMBINED_TRADE_AUDIT_SCHEMA_ID,
    COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
    REBOUND_BASELINE_POLICY_ID,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
RUNTIME_TEMPLATE_ROOT = PROJECT_ROOT / "runtime_inputs" / "templates"
TEMPLATE_ROOT = PACKAGE_ROOT / "review_templates"
DEFAULT_SOURCE_MANIFEST = PACKAGE_ROOT / "SOURCE_MANIFEST.json"
LEGACY_STANDALONE_ROOT = PROJECT_ROOT / "external_inputs" / "legacy_review_delivery"
LEGACY_STANDALONE_STAGE = LEGACY_STANDALONE_ROOT / "stage"
LEGACY_STANDALONE_VALIDATION_STAGE = LEGACY_STANDALONE_ROOT / "validation_stage"
OWNER_MARKER = "generated-by: v4.4-review-delivery"
RELEASE_VERSION_LABEL = "V4.41"

HOME_TEMPLATE = TEMPLATE_ROOT / "home_v4_hub_reuse.html"
TRADE_TEMPLATE = TEMPLATE_ROOT / "trade_v4_explain_reuse.html"
STYLE_TEMPLATE = TEMPLATE_ROOT / "v4_template_adapter.css"
HOME_SCRIPT_TEMPLATE = TEMPLATE_ROOT / "home.js"
TRADE_SCRIPT_TEMPLATE = TEMPLATE_ROOT / "trade_analysis.js"
HUB_DESIGN_SOURCE = PROJECT_ROOT / "project_management" / "research_hub.html"
TRADE_DESIGN_SOURCE = RUNTIME_TEMPLATE_ROOT / "historical_v4_trade.html"
TRADE_DESIGN_SOURCE_SHA256 = (
    "9ffc8fd269173a27eae47f21d993c1f43cc296f0b76b14018ff3fb45a9402b50"
)
TRADE_PLOTLY_SOURCE = RUNTIME_TEMPLATE_ROOT / "plotly.min.js"
TRADE_PLOTLY_SOURCE_SHA256 = (
    "91c4c2879c1cee9f5cab0693f4d7c27773cbe7e22ab6252e5eef0ccfc7f2a7ef"
)
FILTER_OVERLAY_ID = "v4_4_filter_audit_palette_causal_baseline_v2"
BASELINE_SAMPLING_POLICY_CONTRACTS = {
    "all_window": (
        "every finite TR15 atom inside one continuous segment is eligible; "
        "baseline_excluded and filter intervals are audit/chart-coloring only"
    ),
    "exclude_marked": (
        "finite TR15 atoms inside one continuous segment are eligible only after "
        "baseline_available_from; recovered pending atoms become available at the "
        "recovery confirmation time and confirmed exclusions remain unavailable"
    ),
    "confirmed_low_activity_gate": (
        "pending low activity has no strategy effect; confirmation removes the run "
        "from its first atom in all later baseline calculations and blocks new entries "
        "until the first normal-volume atom"
    ),
}
FILTER_OVERLAY_SCOPE = (
    "baseline_excluded and filter intervals always remain visible audit/chart evidence; "
    "all_window ignores them, exclude_marked applies the legacy availability lifecycle, "
    "and confirmed_low_activity_gate applies confirmation-time exclusion and entry gating"
)

NATIVE_METHOD_CONTRACTS = {
    "rolling_tr_sum": {
        "slug": "mean_rolling_tr_sum_w10",
        "label": "BH 内滚动 TR 总和均值（15秒 TR 原子）",
        "family": "true_range_sum",
        "formula": (
            "mean of every overlapping TRW sum inside all finite 15-second "
            "TR atoms selected under the policy at the current calculation time, "
            "with the source side ending at strict H"
        ),
    },
}

PROMISING_FILES = {
    "stage_summary": "stage_summary",
    "comparison": "comparison_summary",
    "rankings": "rankings",
    "shortlist": "shortlist",
    "objective_summary": "objective_summary",
    "trades": "trades",
}

SUMMARY_FIELDS = (
    "combo_id",
    "method",
    "baseline_sampling_policy",
    "e",
    "bh",
    "trw",
    "k",
    "w",
    "m",
    "speed_window_bars",
    "speed_exit_enabled",
    "rebound_exit_enabled",
    "entry_fill_mode",
    "entry_execution_policy",
    "entry_slippage",
    "train_trade_count",
    "train_return",
    "train_cost_adjusted_return",
    "train_return_excluding_gap_spanning_trades",
    "train_avg_trade",
    "train_cost_adjusted_avg_trade",
    "train_max_drawdown_abs",
    "train_cost_adjusted_max_drawdown",
    "train_cost_adjusted_max_drawdown_abs",
    "round_trip_cost_bps",
    "estimated_total_commission_krw",
    "estimated_total_slippage_krw",
    "estimated_total_cost_krw",
    "cost_model_id",
    "gap_spanning_trade_count",
    "synthetic_signal_trade_count",
    "segment_end_exit_count",
    "rebound_exit_count",
    "speed_exit_count",
    "scenario_1_qualified",
    "scenario_2_qualified",
    "scenario_3_qualified",
    "event_01_qualified",
    "event_02_qualified",
    "short_drop_3_15m_member",
)

TRADE_DETAIL_FIELDS = (
    "batch_id",
    "combo_id",
    "method",
    "baseline_sampling_policy",
    "entry_index",
    "entry_time",
    "entry_price",
    "entry_price_before_slippage",
    "entry_bar_synthetic",
    "entry_bar_volume",
    "entry_bar_trade_count",
    "entry_fill_mode",
    "entry_execution_policy",
    "entry_slippage",
    "entry_fill_source",
    "initial_entry_index",
    "initial_entry_time",
    "entry_wait_bar_count",
    "initial_entry_bar_synthetic",
    "initial_entry_bar_volume",
    "initial_entry_bar_trade_count",
    "signal_index",
    "signal_time",
    "h_index",
    "h_time",
    "exit_index",
    "exit_time",
    "exit_price",
    "exit_reason",
    "exit_bar_synthetic",
    "exit_bar_volume",
    "exit_bar_trade_count",
    "pending_exit",
    "pending_exit_trigger_index",
    "pending_exit_trigger_time",
    "pending_exit_trigger_reason",
    "pending_exit_theoretical_price",
    "pending_exit_wait_bar_count",
    "pending_exit_fill_policy",
    "pending_exit_fill_vs_theoretical_delta",
    "strategy_id",
    "trade_audit_schema_version",
    "trade_audit_schema_id",
    "return",
    "gross_return",
    "cost_adjusted_return",
    "round_trip_cost_bps",
    "round_trip_commission_krw",
    "round_trip_slippage_krw",
    "round_trip_total_cost_krw",
    "cost_model_id",
    "active_low_index",
    "active_low",
    "rebound_net_drop",
    "e",
    "bh",
    "trw",
    "k",
    "w",
    "m",
    "rebound_window_start_index",
    "rebound_window_end_index",
    "rebound_window_observed_bar_count",
    "rebound_max_w_drop",
    "rebound_latest_applied_candidate",
    "rebound_latest_applied_candidate_start_index",
    "rebound_latest_applied_candidate_end_index",
    "rebound_latest_applied_candidate_observed_bar_count",
    "rebound_exit_bar_candidate",
    "rebound_exit_bar_candidate_start_index",
    "rebound_exit_bar_candidate_end_index",
    "rebound_exit_bar_candidate_observed_bar_count",
    "rebound_candidates_effective_through_index",
    "rebound_baseline_policy_id",
    "rebound_threshold",
    "rebound_trigger_price",
    "rebound_check_price",
    "rebound_check_price_basis",
    "rebound_gap_adjusted",
    "rebound_gap_slippage",
    "rebound_baseline_update_rule",
    "speed_exit_enabled",
    "speed_window_bars",
    "speed_reference_index",
    "speed_reference_time",
    "speed_reference_low",
    "speed_current_low",
    "speed_extension",
    "speed_check_price",
    "speed_check_price_basis",
    "downside_speed_exit_fill_policy",
    "exit_price_basis",
    "entry_continuous_segment_id",
    "exit_continuous_segment_id",
    "position_crosses_real_gap",
    "signal_single_bar_drop_share",
    "signal_synthetic_empty_bar_count",
    "entry_baseline_value",
    "entry_drop_value",
    "baseline_history_start_index",
    "baseline_history_end_index",
    "baseline_eligible_atom_count",
    "baseline_physical_span_bars",
    "baseline_excluded_atom_count",
    "baseline_pending_atom_count",
    "baseline_confirmed_excluded_atom_count",
    "baseline_filter_id",
    "baseline_filter_stage",
    "holding_bar_distance",
    "holding_minutes",
)

MAX_W_TRADE_AUDIT_FIELDS = (
    "strategy_id",
    "trade_audit_schema_version",
    "trade_audit_schema_id",
    "rebound_baseline_policy_id",
    "rebound_net_drop",
    "rebound_max_w_drop",
    "rebound_window_start_index",
    "rebound_window_end_index",
    "rebound_window_observed_bar_count",
    "rebound_latest_applied_candidate",
    "rebound_latest_applied_candidate_start_index",
    "rebound_latest_applied_candidate_end_index",
    "rebound_latest_applied_candidate_observed_bar_count",
    "rebound_exit_bar_candidate",
    "rebound_exit_bar_candidate_start_index",
    "rebound_exit_bar_candidate_end_index",
    "rebound_exit_bar_candidate_observed_bar_count",
    "rebound_candidates_effective_through_index",
    "rebound_threshold",
    "rebound_trigger_price",
    "rebound_check_price",
    "rebound_check_price_basis",
    "rebound_gap_adjusted",
    "rebound_gap_slippage",
    "rebound_baseline_update_rule",
)

SOURCE_COLUMNS = (
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "is_synthetic_empty_bar",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }
    if root is not None:
        result["relative_path"] = resolved.relative_to(root.resolve()).as_posix()
    return result


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(current) for key, current in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(current) for current in value]
    if pd.isna(value):
        return None
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".v43-{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".v43-{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _json_script(variable: str, payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    encoded = encoded.replace("</script", "<\\/script")
    return f"window.{variable}={encoded};\n"


def _compact_json(payload: Any) -> str:
    return json.dumps(
        _jsonable(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("</script", "<\\/script")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"unrecognized boolean value: {value!r}")


def _number(value: Any) -> int | float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def _record(row: pd.Series, fields: Iterable[str]) -> dict[str, Any]:
    return {field: _jsonable(row[field]) for field in fields if field in row.index}


def _bound_artifact(manifest: dict[str, Any], key: str) -> Path:
    entry = manifest.get("artifacts", {}).get(key)
    if not isinstance(entry, dict):
        raise ValueError(f"manifest lacks artifact {key}")
    path = Path(str(entry.get("path", "")))
    if not path.is_file():
        raise ValueError(f"bound artifact is missing: {path}")
    if _sha256(path) != str(entry.get("sha256", "")):
        raise ValueError(f"bound artifact hash mismatch: {path}")
    if int(path.stat().st_size) != int(entry.get("size_bytes", -1)):
        raise ValueError(f"bound artifact size mismatch: {path}")
    return path


def _combo_key(combo_id: str) -> str:
    return hashlib.sha256(combo_id.encode("utf-8")).hexdigest()[:14]


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def _validate_output_owner(output: Path) -> None:
    index = output / "index.html"
    if index.exists() and OWNER_MARKER not in index.read_text(encoding="utf-8"):
        raise ValueError(f"refusing to replace an unrelated page: {index}")


def _summary_record(row: pd.Series) -> dict[str, Any]:
    result = _record(row, SUMMARY_FIELDS)
    result["key"] = _combo_key(str(row["combo_id"]))
    for field in (
        "event_01_qualified",
        "event_02_qualified",
        "short_drop_3_15m_member",
        "scenario_1_qualified",
        "scenario_2_qualified",
        "scenario_3_qualified",
        "speed_exit_enabled",
        "rebound_exit_enabled",
    ):
        result[field] = _bool(row.get(field, False))
    return result


def _source_bar(source: pd.DataFrame, index: int) -> dict[str, Any]:
    row = source.iloc[index]
    return {
        "i": int(index),
        "t": str(row["datetime"]),
        "o": _number(row["open"]),
        "h": _number(row["high"]),
        "l": _number(row["low"]),
        "c": _number(row["close"]),
        "v": _number(row["volume"]),
        "n": _number(row["trade_count"]),
        "s": _bool(row["is_synthetic_empty_bar"]),
    }


def _assert_close(left: Any, right: Any, label: str, *, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} mismatch: {left!r} != {right!r}")


def _validate_trade_entry(row: pd.Series, source: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    entry_index = int(row["entry_index"])
    initial_index = int(row["initial_entry_index"])
    if entry_index < 0 or entry_index >= len(source) or initial_index < 0 or initial_index >= len(source):
        raise ValueError("trade entry index lies outside the source")
    actual = _source_bar(source, entry_index)
    initial = _source_bar(source, initial_index)
    if actual["t"] != str(row["entry_time"]):
        raise ValueError("actual entry timestamp does not match source evidence")
    if initial["t"] != str(row["initial_entry_time"]):
        raise ValueError("initial entry timestamp does not match source evidence")
    if actual["s"] or float(actual["v"] or 0) <= 0 or float(actual["n"] or 0) <= 0:
        raise ValueError("actual entry source bar is not a real-trade bar")
    if _bool(row["entry_bar_synthetic"]) != actual["s"]:
        raise ValueError("actual entry synthetic flag does not match source")
    _assert_close(row["entry_bar_volume"], actual["v"], "actual entry volume")
    _assert_close(row["entry_bar_trade_count"], actual["n"], "actual entry trade count")
    fill_source = str(row["entry_fill_source"])
    if fill_source == "calculated_threshold":
        if entry_index != initial_index or int(row["entry_wait_bar_count"]) != 0:
            raise ValueError("calculated-threshold entry cannot contain a wait")
        if entry_index != int(row["signal_index"]):
            raise ValueError("calculated-threshold entry must fill on the signal bar")
        expected_price = min(float(actual["o"]), float(row["entry_trigger_price"]))
        _assert_close(
            row["entry_price_before_slippage"],
            expected_price,
            "calculated entry price",
        )
    else:
        _assert_close(row["entry_price_before_slippage"], actual["o"], "actual entry open")
    if _bool(row["initial_entry_bar_synthetic"]) != initial["s"]:
        raise ValueError("initial entry synthetic flag does not match source")
    _assert_close(row["initial_entry_bar_volume"], initial["v"], "initial entry volume")
    _assert_close(
        row["initial_entry_bar_trade_count"], initial["n"], "initial entry trade count"
    )
    waited = int(row["entry_wait_bar_count"]) > 0
    expected_source = (
        "waited_real_trade_open"
        if waited
        else "calculated_threshold"
        if str(row["entry_fill_mode"]) == "calculated_threshold"
        else "initial_real_trade_open"
    )
    if fill_source != expected_source:
        raise ValueError("entry fill source does not match the wait state")
    return initial, actual


def _validate_trade_exit(row: pd.Series, source: pd.DataFrame) -> dict[str, Any]:
    exit_index = int(row["exit_index"])
    if exit_index < 0 or exit_index >= len(source):
        raise ValueError("trade exit index lies outside the source")
    actual = _source_bar(source, exit_index)
    if actual["t"] != str(row["exit_time"]):
        raise ValueError("actual exit timestamp does not match source evidence")
    if _bool(row["exit_bar_synthetic"]) != actual["s"]:
        raise ValueError("actual exit synthetic flag does not match source")
    _assert_close(row["exit_bar_volume"], actual["v"], "actual exit volume")
    _assert_close(row["exit_bar_trade_count"], actual["n"], "actual exit trade count")
    reason = str(row["exit_reason"])
    if reason in {"rebound_threshold", "downside_speed_below_threshold"}:
        if actual["s"] or float(actual["v"] or 0) <= 0 or float(actual["n"] or 0) <= 0:
            raise ValueError("signal-driven exit source bar is not a real-trade bar")
        pending = _bool(row["pending_exit"])
        if pending:
            if str(row["pending_exit_fill_policy"]) != "next_real_trade_bar_open":
                raise ValueError("pending exit fill policy is not next-real-trade open")
            if int(row["pending_exit_wait_bar_count"]) <= 0:
                raise ValueError("pending exit did not wait for a later real-trade bar")
            if int(row["pending_exit_trigger_index"]) >= exit_index:
                raise ValueError("pending exit trigger does not precede its real fill")
            _assert_close(row["exit_price"], actual["o"], "pending exit open")
        elif str(row["pending_exit_fill_policy"]) != "same_real_trade_bar":
            raise ValueError("same-bar exit has inconsistent pending-exit audit")
    return actual


def _trade_catalog_record(row: pd.Series, sequence: int) -> dict[str, Any]:
    combo_id = str(row["combo_id"])
    wait_bars = int(row["entry_wait_bar_count"])
    signal_synthetic_count = int(row["signal_synthetic_empty_bar_count"])
    return {
        "id": f"{_combo_key(combo_id)}-{int(row['entry_index'])}",
        "sequence": int(sequence),
        "combo_key": _combo_key(combo_id),
        "combo_id": combo_id,
        "method": str(row["method"]),
        "baseline_sampling_policy": str(row["baseline_sampling_policy"]),
        "e": _number(row["e"]),
        "bh": _number(row["bh"]),
        "trw": _number(row["trw"]),
        "k": _number(row["k"]),
        "w": _number(row["w"]),
        "m": _number(row["m"]),
        "speed_window_bars": _number(row.get("speed_window_bars")),
        "signal_time": str(row["signal_time"]),
        "initial_entry_time": str(row["initial_entry_time"]),
        "entry_time": str(row["entry_time"]),
        "exit_time": str(row["exit_time"]),
        "entry_price": _number(row["entry_price"]),
        "exit_price": _number(row["exit_price"]),
        "return": _number(row["return"]),
        "exit_reason": str(row["exit_reason"]),
        "wait_bars": wait_bars,
        "waited": wait_bars > 0,
        "crosses_gap": _bool(row["position_crosses_real_gap"]),
        "synthetic_signal": signal_synthetic_count > 0,
        "synthetic_signal_bar_count": signal_synthetic_count,
        "actual_entry_real": (
            not _bool(row["entry_bar_synthetic"])
            and float(row["entry_bar_volume"]) > 0
            and float(row["entry_bar_trade_count"]) > 0
        ),
        "entry_fill_source": str(row["entry_fill_source"]),
        "pending_exit": _bool(row.get("pending_exit", False)),
        "pending_exit_wait_bars": int(row.get("pending_exit_wait_bar_count", 0)),
    }


def _trade_detail(
    row: pd.Series,
    sequence: int,
    source: pd.DataFrame,
) -> tuple[dict[str, Any], tuple[int, int], tuple[int, int]]:
    initial, actual = _validate_trade_entry(row, source)
    actual_exit = _validate_trade_exit(row, source)
    detail = _record(row, TRADE_DETAIL_FIELDS)
    catalog = _trade_catalog_record(row, sequence)
    detail.update(
        {
            "id": catalog["id"],
            "sequence": int(sequence),
            "combo_key": catalog["combo_key"],
            "waited": catalog["waited"],
            "synthetic_signal": catalog["synthetic_signal"],
            "actual_entry_real": catalog["actual_entry_real"],
            "initial_entry_evidence": initial,
            "actual_entry_evidence": actual,
            "actual_exit_evidence": actual_exit,
        }
    )
    signal_index = int(row["signal_index"])
    entry_index = int(row["entry_index"])
    exit_index = int(row["exit_index"])
    entry_range = (max(0, signal_index - 6), min(len(source) - 1, entry_index + 6))
    exit_range = (max(0, exit_index - 8), min(len(source) - 1, exit_index + 6))
    detail["entry_bar_range"] = list(entry_range)
    detail["exit_bar_range"] = list(exit_range)
    return detail, entry_range, exit_range


def _bar_indices(ranges: Iterable[tuple[int, int]]) -> list[int]:
    values: set[int] = set()
    for start, end in ranges:
        values.update(range(start, end + 1))
    return sorted(values)


def _extract_style(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"design template is missing: {path}")
    source = path.read_text(encoding="utf-8")
    match = re.search(r"<style>(.*?)</style>", source, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        raise ValueError(f"design template has no inline style block: {path}")
    return match.group(1).strip()


def _render_template(
    path: Path,
    values: dict[str, str],
    *,
    raw_values: dict[str, str] | None = None,
) -> str:
    text = path.read_text(encoding="utf-8")
    for key, value in (raw_values or {}).items():
        text = text.replace(f"{{{{{{{key}}}}}}}", value)
    for key, value in values.items():
        text = text.replace(f"{{{{{key}}}}}", html.escape(value, quote=True))
    unresolved = [segment.split("}}", 1)[0] for segment in text.split("{{")[1:] if "}}" in segment]
    if unresolved:
        raise ValueError(f"unresolved template values in {path}: {unresolved}")
    return text


def _load_inputs(
    source_manifest_path: Path,
    stage: Path,
    validation_stage: Path,
) -> dict[str, Any]:
    source_manifest = _read_json(source_manifest_path)
    promising_manifest_path = stage / "v4_4_promising_manifest.json"
    validation_manifest_path = validation_stage / "v4_4_validation_manifest.json"
    stage_manifest_path = stage / "stage_manifest.json"
    completion_path = stage / "completion_manifest.json"
    for path in (
        promising_manifest_path,
        validation_manifest_path,
        stage_manifest_path,
        completion_path,
    ):
        if not path.is_file():
            raise ValueError(f"required closed V4.4 artifact is missing: {path}")
    promising_manifest = _read_json(promising_manifest_path)
    validation_manifest = _read_json(validation_manifest_path)
    stage_manifest = _read_json(stage_manifest_path)
    completion = _read_json(completion_path)
    expected = source_manifest["promising_exploration"]
    if _sha256(promising_manifest_path) != expected["analysis_manifest_sha256"]:
        raise ValueError("source manifest does not bind the current promising manifest")
    if _sha256(validation_manifest_path) != source_manifest["validation"]["validation_manifest_sha256"]:
        raise ValueError("source manifest does not bind the current validation manifest")
    source_identity = {
        "version_label": source_manifest.get("version_label"),
        "strategy_id": source_manifest.get(
            "rebound_only_strategy_id", source_manifest.get("strategy_id")
        ),
        "result_semantics_id": source_manifest.get(
            "rebound_only_result_semantics_id",
            source_manifest.get("result_semantics_id"),
        ),
    }
    for field in ("version_label", "strategy_id", "result_semantics_id"):
        values = {
            source_identity[field],
            promising_manifest.get(field),
            validation_manifest.get(field),
            stage_manifest.get(field),
            completion.get(field),
        }
        if len(values) != 1:
            raise ValueError(f"V4.4 identity mismatch for {field}: {values}")
    if promising_manifest.get("status") != "complete" or completion.get("status") != "complete":
        raise ValueError("promising campaign is not closed")
    if int(promising_manifest.get("coordinate_count", -1)) != 24:
        raise ValueError("promising manifest coordinate count is not 24")
    if int(promising_manifest.get("trade_count", -1)) != 2385:
        raise ValueError("promising manifest trade count is not 2,385")
    artifacts = {
        name: _bound_artifact(promising_manifest, manifest_key)
        for name, manifest_key in PROMISING_FILES.items()
    }
    validation_summary_path = _bound_artifact(validation_manifest, "validation_summary")
    preparation_manifest_path = Path(str(stage_manifest["data_preparation_manifest"]))
    source_path = Path(str(stage_manifest["source"]))
    if not preparation_manifest_path.is_file() or _sha256(preparation_manifest_path) != str(
        stage_manifest["data_preparation_manifest_sha256"]
    ):
        raise ValueError("V4.4 preparation manifest failed stage binding")
    if not source_path.is_file() or _sha256(source_path) != str(stage_manifest["source_sha256"]):
        raise ValueError("V4.4 market source failed stage binding")
    preparation_manifest = _read_json(preparation_manifest_path)
    if preparation_manifest.get("prepared_identity") != promising_manifest.get("prepared_identity"):
        raise ValueError("prepared identity differs from the promising manifest")
    return {
        "source_manifest": source_manifest,
        "source_manifest_path": source_manifest_path,
        "promising_manifest": promising_manifest,
        "promising_manifest_path": promising_manifest_path,
        "validation_manifest": validation_manifest,
        "validation_manifest_path": validation_manifest_path,
        "stage_manifest": stage_manifest,
        "stage_manifest_path": stage_manifest_path,
        "completion": completion,
        "completion_path": completion_path,
        "artifacts": artifacts,
        "validation_summary_path": validation_summary_path,
        "preparation_manifest": preparation_manifest,
        "preparation_manifest_path": preparation_manifest_path,
        "source_path": source_path,
    }


def build(
    source_manifest_path: Path,
    stage: Path,
    validation_stage: Path,
    output: Path,
) -> dict[str, Any]:
    source_manifest_path = source_manifest_path.resolve()
    stage = stage.resolve()
    validation_stage = validation_stage.resolve()
    output = output.resolve()
    _validate_output_owner(output)
    inputs = _load_inputs(source_manifest_path, stage, validation_stage)
    artifacts = inputs["artifacts"]

    summary = pd.read_csv(artifacts["stage_summary"])
    comparison = pd.read_csv(artifacts["comparison"])
    rankings = pd.read_csv(artifacts["rankings"])
    shortlist = pd.read_csv(artifacts["shortlist"])
    objective_summary = pd.read_csv(artifacts["objective_summary"])
    trades = pd.read_csv(artifacts["trades"])
    validation_summary = pd.read_csv(inputs["validation_summary_path"])
    if len(summary) != 24 or summary["combo_id"].astype(str).duplicated().any():
        raise ValueError("V4.4 review requires 24 unique summary rows")
    if len(trades) != 2385:
        raise ValueError("V4.4 review requires all 2,385 closed trades")
    if set(trades["combo_id"].astype(str)) != set(summary["combo_id"].astype(str)):
        raise ValueError("trade and summary combo populations differ")

    source = pd.read_csv(inputs["source_path"], usecols=list(SOURCE_COLUMNS))
    combo_rows = [_summary_record(row) for _, row in summary.iterrows()]
    combo_by_id = {str(row["combo_id"]): row for row in combo_rows}
    chunk_directory = output / "assets" / "trade_chunks"
    catalog_rows: list[dict[str, Any]] = []
    chunk_files: list[Path] = []
    verified_entry_count = 0
    waited_entry_count = 0
    maximum_wait = 0

    for combo_id, combo_trades in trades.groupby("combo_id", sort=False):
        combo_id = str(combo_id)
        combo_trades = combo_trades.sort_values(["entry_index", "exit_index"], kind="mergesort")
        details: list[dict[str, Any]] = []
        ranges: list[tuple[int, int]] = []
        for sequence, (_, trade) in enumerate(combo_trades.iterrows(), start=1):
            detail, entry_range, exit_range = _trade_detail(trade, sequence, source)
            details.append(detail)
            ranges.extend((entry_range, exit_range))
            catalog_rows.append(_trade_catalog_record(trade, sequence))
            verified_entry_count += int(detail["actual_entry_real"])
            waited_entry_count += int(detail["waited"])
            maximum_wait = max(maximum_wait, int(trade["entry_wait_bar_count"]))
        key = _combo_key(combo_id)
        bars = [_source_bar(source, index) for index in _bar_indices(ranges)]
        payload = {
            "combo": combo_by_id[combo_id],
            "trades": details,
            "bars": bars,
            "source_sha256": inputs["stage_manifest"]["source_sha256"],
        }
        chunk_path = chunk_directory / f"{key}.js"
        _atomic_text(
            chunk_path,
            "window.V41_TRADE_CHUNKS=window.V41_TRADE_CHUNKS||{};"
            f"window.V41_TRADE_CHUNKS[{json.dumps(key)}]="
            + json.dumps(
                _jsonable(payload),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).replace("</script", "<\\/script")
            + ";\n",
        )
        chunk_files.append(chunk_path)

    if verified_entry_count != 2385:
        raise ValueError("not every actual entry bar passed source evidence validation")
    if waited_entry_count != int(inputs["promising_manifest"]["waited_entry_trade_count"]):
        raise ValueError("waited-entry count differs from the closed promising manifest")
    if maximum_wait != int(inputs["promising_manifest"]["maximum_observed_entry_wait_bars"]):
        raise ValueError("maximum entry wait differs from the closed promising manifest")

    ranking_rows = []
    for _, row in rankings.iterrows():
        ranking_rows.append(
            {
                "objective": str(row["ranking_objective"]),
                "minimum_trade_count": int(row["minimum_trade_count"]),
                "rank": int(row["rank"]),
                "combo_id": str(row["combo_id"]),
                "objective_metric": str(row["objective_metric"]),
            }
        )
    shortlist_rows = []
    for _, row in shortlist.iterrows():
        record = _summary_record(row)
        record["shortlist_reasons"] = str(row["shortlist_reasons"])
        record["shortlist_reason_count"] = int(row["shortlist_reason_count"])
        shortlist_rows.append(record)
    validation_rows = []
    validation_fields = (
        "source_v4_combo_id",
        "v4_4_combo_id",
        "method_v4_4",
        "e_v4_4",
        "bh_v4_4",
        "trw_v4_4",
        "k_v4_4",
        "w_v4_4",
        "m_v4_4",
        "v4_train_trade_count",
        "train_trade_count",
        "v4_train_return",
        "train_return",
        "v4_train_max_drawdown_abs",
        "train_max_drawdown_abs",
        "trade_count_delta_v4_4_minus_v4",
        "return_delta_v4_4_minus_v4",
        "max_drawdown_abs_delta_v4_4_minus_v4",
        "v4_4_waited_entry_trade_count",
        "v4_4_max_entry_wait_bar_count",
    )
    for _, row in validation_summary.iterrows():
        validation_rows.append(_record(row, validation_fields))

    preparation = inputs["preparation_manifest"]
    promising = inputs["promising_manifest"]
    validation = inputs["validation_manifest"]
    identity = {
        "version_label": promising["version_label"],
        "strategy_id": promising["strategy_id"],
        "baseline_filter_id": promising["baseline_filter_id"],
        "result_semantics_id": promising["result_semantics_id"],
        "campaign_id": promising["campaign_id"],
        "stage_id": promising["stage_id"],
        "prepared_identity": promising["prepared_identity"],
        "source_sha256": inputs["stage_manifest"]["source_sha256"],
        "promising_manifest_sha256": _sha256(inputs["promising_manifest_path"]),
        "validation_manifest_sha256": _sha256(inputs["validation_manifest_path"]),
        "preparation_manifest_sha256": _sha256(inputs["preparation_manifest_path"]),
        "parameter_acceptance": "none",
    }
    home_data = {
        "identity": identity,
        "counts": {
            "validation_coordinates": int(validation["selected_coordinate_count"]),
            "validation_trades": int(validation["trade_count"]),
            "validation_waited_entries": int(validation["waited_entry_trade_count"]),
            "validation_maximum_wait": int(validation["maximum_observed_entry_wait_bars"]),
            "promising_coordinates": int(promising["coordinate_count"]),
            "promising_trades": int(promising["trade_count"]),
            "promising_waited_entries": int(promising["waited_entry_trade_count"]),
            "promising_maximum_wait": int(promising["maximum_observed_entry_wait_bars"]),
            "shortlist_coordinates": int(promising["shortlist_coordinate_count"]),
        },
        "preparation": {
            "atom_count": int(preparation["low_activity_summary"]["atom_count"]),
            "baseline_excluded_atom_count": int(
                preparation["low_activity_summary"]["baseline_excluded_atom_count"]
            ),
            "baseline_excluded_minutes": float(
                preparation["low_activity_summary"]["baseline_excluded_minutes"]
            ),
            "buffer_reinserted_atom_count": int(
                preparation["low_activity_summary"]["buffer_reinserted_atom_count"]
            ),
            "recovery_confirmation_count": int(
                preparation["low_activity_summary"]["recovery_confirmation_count"]
            ),
            "audit_url": _file_url(Path(preparation["report_audit"]["index"])),
        },
        "combos": combo_rows,
        "rankings": ranking_rows,
        "shortlist": shortlist_rows,
        "validation": validation_rows,
        "objective_summary": [_jsonable(row) for row in objective_summary.to_dict("records")],
        "comparison": [
            {
                "combo_id": str(row["combo_id"]),
                "selection_objective": str(row["selection_objective"]),
                "selection_reason": str(row["selection_reason"]),
                "trade_count_delta": _number(row["trade_count_delta_v4_4_minus_v4"]),
                "return_delta": _number(row["return_delta_v4_4_minus_v4"]),
                "gap_return_delta": _number(
                    row["gap_excluded_return_delta_v4_4_minus_v4"]
                ),
                "drawdown_delta": _number(
                    row["max_drawdown_abs_delta_v4_4_minus_v4"]
                ),
            }
            for _, row in comparison.iterrows()
        ],
    }
    catalog_data = {
        "identity": identity,
        "trade_count": len(catalog_rows),
        "combo_count": len(combo_rows),
        "combos": [
            {
                **row,
                "chunk": f"../assets/trade_chunks/{row['key']}.js",
            }
            for row in combo_rows
        ],
        "trades": catalog_rows,
    }

    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    template_values = {
        "VERSION_LABEL": str(identity["version_label"]),
        "STRATEGY_ID": str(identity["strategy_id"]),
        "RESULT_SEMANTICS_ID": str(identity["result_semantics_id"]),
        "SOURCE_SHA_SHORT": str(identity["source_sha256"])[:12],
        "PREPARED_IDENTITY": str(identity["prepared_identity"]),
        "PROMISING_MANIFEST_SHA_SHORT": str(identity["promising_manifest_sha256"])[:12],
        "COORDINATE_COUNT": str(len(combo_rows)),
        "TRADE_COUNT": str(len(catalog_rows)),
        "CAMPAIGN_ID": str(identity["campaign_id"]),
        "STAGE_ID": str(identity["stage_id"]),
        "MANIFEST_HREF": "../v4_4_review_manifest.json",
        "MAIN_HREF": "../index.html",
        "ASSET_PREFIX": "../assets",
        "REVIEW_TITLE": f"{RELEASE_VERSION_LABEL} 净跌幅回撤逐笔查看",
        "VALIDATION_MANIFEST_SHA_SHORT": str(identity["validation_manifest_sha256"])[:12],
        "LOW_ACTIVITY_AUDIT_URL": str(home_data["preparation"]["audit_url"]),
    }
    design_styles = {
        "HUB_REFERENCE_STYLE": _extract_style(HUB_DESIGN_SOURCE),
        "TRADE_REFERENCE_STYLE": _extract_style(TRADE_DESIGN_SOURCE),
    }
    index_path = output / "index.html"
    trade_page_path = output / "trade_analysis" / "index.html"
    style_path = assets / "v4_4_review.css"
    home_script_path = assets / "home.js"
    trade_script_path = assets / "trade_analysis.js"
    home_data_path = assets / "home_data.js"
    catalog_path = assets / "trade_catalog.js"
    _atomic_text(
        index_path,
        _render_template(
            HOME_TEMPLATE,
            template_values,
            raw_values={"HUB_REFERENCE_STYLE": design_styles["HUB_REFERENCE_STYLE"]},
        ),
    )
    _atomic_text(
        trade_page_path,
        _render_template(
            TRADE_TEMPLATE,
            template_values,
            raw_values={"TRADE_REFERENCE_STYLE": design_styles["TRADE_REFERENCE_STYLE"]},
        ),
    )
    _atomic_text(style_path, STYLE_TEMPLATE.read_text(encoding="utf-8"))
    _atomic_text(home_script_path, HOME_SCRIPT_TEMPLATE.read_text(encoding="utf-8"))
    _atomic_text(trade_script_path, TRADE_SCRIPT_TEMPLATE.read_text(encoding="utf-8"))
    _atomic_text(home_data_path, _json_script("V41_HOME_DATA", home_data))
    _atomic_text(catalog_path, _json_script("V41_TRADE_CATALOG", catalog_data))
    native_review = build_stage_trade_review(
        trade_page_path.parent,
        summary,
        trades,
        inputs["stage_manifest"],
        inputs["completion"],
        analysis_identity=(
            "v4_4_promising_review_exact_historical_v4_trade_implementation_v1"
        ),
        manifest_href="../v4_4_review_manifest.json",
        main_href="../index.html",
    )

    output_files = [
        index_path,
        trade_page_path,
        style_path,
        home_script_path,
        trade_script_path,
        home_data_path,
        catalog_path,
        *chunk_files,
        native_review["process_payload"],
        native_review["catalog"],
        native_review["manifest"],
        native_review["plotly"],
        *native_review["chunks"],
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "evidence_role": "v4_4_closed_diagnostic_review_delivery",
        "parameter_acceptance": "none",
        **identity,
        "generator": _artifact(Path(__file__)),
        "template_sources": {
            path.name: _artifact(path)
            for path in (
                HOME_TEMPLATE,
                TRADE_TEMPLATE,
                STYLE_TEMPLATE,
                HOME_SCRIPT_TEMPLATE,
                TRADE_SCRIPT_TEMPLATE,
            )
        },
        "design_template_sources": {
            "v4_research_hub": _artifact(HUB_DESIGN_SOURCE),
            "v4_trade_explain": _artifact(TRADE_DESIGN_SOURCE),
        },
        "authoritative_inputs": {
            "source_manifest": _artifact(inputs["source_manifest_path"]),
            "promising_manifest": _artifact(inputs["promising_manifest_path"]),
            "validation_manifest": _artifact(inputs["validation_manifest_path"]),
            "stage_manifest": _artifact(inputs["stage_manifest_path"]),
            "completion_manifest": _artifact(inputs["completion_path"]),
            "stage_summary": _artifact(artifacts["stage_summary"]),
            "promising_trades": _artifact(artifacts["trades"]),
            "rankings": _artifact(artifacts["rankings"]),
            "shortlist": _artifact(artifacts["shortlist"]),
            "preparation_manifest": _artifact(inputs["preparation_manifest_path"]),
            "market_source": _artifact(inputs["source_path"]),
        },
        "routes": {
            "home": str(index_path),
            "trade_analysis": str(trade_page_path),
            "trade_query_contract": "trade_analysis/index.html?combo_id=<V4.4 combo_id>&trade=<entry_index>",
            "return_navigation": "trade_analysis/index.html -> ../index.html",
        },
        "closure": {
            "coordinate_count": len(combo_rows),
            "trade_count": len(catalog_rows),
            "trade_chunk_count": len(chunk_files),
            "verified_real_entry_bar_count": verified_entry_count,
            "waited_entry_trade_count": waited_entry_count,
            "maximum_observed_entry_wait_bars": maximum_wait,
            "lazy_loading": {
                "startup": "HTML shell and compact catalog only",
                "trade_details": "one combo chunk loads after a trade selection",
                "file_protocol_safe": True,
            },
            "identity_checks": {
                "all_combo_ids_are_v4_4": all(
                    row["combo_id"].startswith("v4_4_") for row in catalog_rows
                ),
                "all_actual_entry_bars_match_source": verified_entry_count == len(catalog_rows),
                "no_v4_result_row_in_delivery": all(
                    row["combo_id"].startswith("v4_4_") for row in combo_rows
                ),
                "no_parameter_accepted": True,
                "v4_hub_design_template_reused": bool(design_styles["HUB_REFERENCE_STYLE"]),
                "v4_trade_design_template_reused": bool(
                    design_styles["TRADE_REFERENCE_STYLE"]
                ),
                "historical_trade_html_css_javascript_reused": True,
                "historical_plotly_candlestick_reused": True,
                "adapter_shell_removed_from_active_trade_route": True,
            },
        },
        "outputs": [_artifact(path, root=output) for path in output_files],
        "resource_count": len(output_files),
        "resource_bytes": sum(path.stat().st_size for path in output_files),
    }
    if not all(manifest["closure"]["identity_checks"].values()):
        raise AssertionError("V4.4 review delivery identity closure failed")
    manifest_path = output / "v4_4_review_manifest.json"
    _atomic_text(
        manifest_path,
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return {
        "home": str(index_path),
        "trade_analysis": str(trade_page_path),
        "trade_catalog": str(native_review["catalog"]),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "generator_sha256": _sha256(Path(__file__)),
        "coordinate_count": len(combo_rows),
        "trade_count": len(catalog_rows),
        "trade_chunk_count": len(chunk_files),
        "verified_real_entry_bar_count": verified_entry_count,
        "resource_count": len(output_files),
        "resource_bytes": sum(path.stat().st_size for path in output_files),
    }


def _native_method_contract(method: str) -> dict[str, str]:
    if method not in NATIVE_METHOD_CONTRACTS:
        raise ValueError(f"unsupported V4.4 entry method: {method}")
    return NATIVE_METHOD_CONTRACTS[method]


def _baseline_sampling_contract(policy: str) -> str:
    if policy not in BASELINE_SAMPLING_POLICY_CONTRACTS:
        raise ValueError(f"unsupported V4.4 baseline sampling policy: {policy}")
    return BASELINE_SAMPLING_POLICY_CONTRACTS[policy]


def _policy_identity(
    manifest: dict[str, Any],
    policy: str,
    field: str,
) -> str:
    mapping = manifest.get(f"{field}s_by_baseline_sampling_policy")
    if isinstance(mapping, dict) and policy in mapping:
        return str(mapping[policy])
    return str(manifest.get(field, ""))


def _native_combo_record(
    row: pd.Series,
    stage_manifest: dict[str, Any],
    completion_manifest: dict[str, Any],
) -> dict[str, Any]:
    raw = _summary_record(row)
    method = str(row["method"])
    contract = _native_method_contract(method)
    baseline_sampling_policy = str(row["baseline_sampling_policy"])
    baseline_sampling_contract = _baseline_sampling_contract(
        baseline_sampling_policy
    )
    bh = int(row["bh"])
    trw = int(row["trw"])
    e = int(row["e"])
    speed_window = int(_number(row.get("speed_window_bars")) or 0)
    trade_count = int(row["train_trade_count"])
    rolling_count = bh - trw + 1 if method == "rolling_tr_sum" else 0
    return {
        **raw,
        "period": 1,
        "entry_window_multiplier": e,
        "entry_window_bars": e,
        "baseline_history_multiplier": bh,
        "baseline_history_bars": bh,
        "baseline_window_multiplier": trw if method == "rolling_tr_sum" else 1,
        "baseline_window_bars": trw if method == "rolling_tr_sum" else 1,
        "baseline_sample_count_target": None,
        "baseline_sample_count": bh,
        "baseline_required_sample_count": bh,
        "k_drop": float(row["k"]),
        "k_context": float(row["k"]),
        "abs_floor": 0.0,
        "daily_floor_mult": 0.0,
        "exit_wait_multiplier": float(speed_window),
        "exit_wait_bars": speed_window,
        "reverse_up_window_multiplier": 0.0,
        "reverse_up_window_bars": 0,
        "reverse_up_limit_multiplier": 0.0,
        "rebound_tr_window_bars": int(row["w"]),
        "rebound_multiplier": float(row["m"]),
        "speed_exit_enabled": _bool(row.get("speed_exit_enabled", False)),
        "rebound_exit_enabled": _bool(row.get("rebound_exit_enabled", True)),
        "rebound_baseline_mode": "fixed_window_open_to_active_low_net_drop",
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "rebound_baseline_update_rule": (
            "maximum_positive_completed_bar_w_candidates_effective_next_bar"
        ),
        "rebound_duration_insufficient_history_policy": "unavailable",
        "execution_bar_seconds": 15,
        "volatility_tr_bar_seconds": 15,
        "strategy_major_version": "V4.4",
        "baseline_sampling_policy": baseline_sampling_policy,
        "baseline_sampling_policy_label": {
            "all_window": "全部",
            "exclude_marked": "排除标记",
            "confirmed_low_activity_gate": "确认后过滤",
        }[baseline_sampling_policy],
        "strategy_id": _policy_identity(
            completion_manifest, baseline_sampling_policy, "strategy_id"
        ),
        "result_semantics_id": _policy_identity(
            stage_manifest, baseline_sampling_policy, "result_semantics_id"
        ),
        "entry_signal_state_version": "v4_4_strict_h_wait_next_real_trade",
        "entry_signal_high_tie_policy": "strict_greater_equal_high_retains_earlier_H",
        "entry_baseline_anchor_policy": "signal_high_inclusive",
        "baseline_tr_atom_policy": (
            "BH ends at and includes the complete 15-second TR atom containing strict H"
        ),
        "baseline_history_collection_policy": baseline_sampling_contract,
        "baseline_window_continuity_policy": "real gaps reset every continuous window",
        "entry_confirmation_policy": (
            "completed 15-second signal; wait at most 120 bars for the next real-trade open"
        ),
        "entry_fill_policy": "wait_next_real_trade; fill actual real-trade bar open",
        "entry_signal_state_policy": "strict highest H inside E; equal highs retain earlier H",
        "entry_signal_reset_policy": "position close resets the flat entry lifecycle",
        "entry_signal_freshness_policy": "completed 15-second signal confirmation only",
        "entry_signal_interval_start_policy": "one fixed full-training sample",
        "holding_pre_exit_bars_excluded_from_next_entry": True,
        "baseline_reset_on_exit": False,
        "exit_mode": str(completion_manifest.get("exit_mode", "combined")),
        "rebound_baseline_method": "h_bounded_max_completed_w_open_to_low_net_drop",
        "rebound_baseline_history_end_policy": (
            "monotonic maximum of positive completed-bar W candidates; current "
            "bar becomes effective on the next bar"
        ),
        "rebound_duration_unit": "15_second_execution_bar",
        "rebound_current_bar_excluded": True,
        "rebound_active_low_bar_included": True,
        "rebound_window_requires_exact_execution_interval": True,
        "rebound_equal_low_resets": False,
        "baseline_history_span_bars": bh,
        "baseline_history_span_formula": "exact BH eligible TR15 atoms",
        "baseline_history_span_policy": "physical gaps are forbidden",
        "execution_mode": "research",
        "execution_mode_label": "研究模式",
        "fill_mode": "wait_next_real_trade_open_combined_exit",
        "downside_speed_exit_fill_policy": "current_bar_close",
        "rebound_exit_fill_policy": (
            "open>=prior trigger fills open; otherwise high>=prior trigger fills trigger; "
            "strict-new-low close confirmation fills that real bar close; equality exits"
        ),
        "segment_end_exit_fill_policy": "sample end bar close without later-price lookup",
        "fee_bps_per_side": 0.0,
        "slippage_bps_per_side": 0.0,
        "ranking_cost_model_id": row.get("cost_model_id"),
        "ranking_round_trip_cost_bps": _number(row.get("round_trip_cost_bps")),
        "ranking_return_basis": "switchable_cost_adjusted_default_and_gross",
        "default_display_return_basis": "cost_adjusted",
        "entry_baseline_family": contract["family"],
        "entry_baseline_method": (
            f"{contract['slug']}__{baseline_sampling_policy}"
        ),
        "entry_baseline_method_label": (
            f"{contract['label']} · "
            f"{ {'all_window': '全部', 'exclude_marked': '排除标记', 'confirmed_low_activity_gate': '确认后过滤'}[baseline_sampling_policy] }"
        ),
        "entry_baseline_formula": contract["formula"],
        "baseline_value_method": (
            f"{contract['slug']}__{baseline_sampling_policy}"
        ),
        "tr_component_formula": (
            "max(high15, previous_15s_close)-min(low15, previous_15s_close)"
        ),
        "tr_component_unit": "15-second price-point TR atom",
        "tr_normalization": "none",
        "tr_square_root": False,
        "tr_anchor_price_conversion": False,
        "tr_history_bars": bh,
        "tr_raw_component_count": bh,
        "tr_component_count": bh,
        "tr_sum_window_bars": trw if method == "rolling_tr_sum" else 0,
        "tr_sum_window_count": rolling_count,
        "tr_sum_window_step_bars": 1,
        "tr_sum_windows_overlap": method == "rolling_tr_sum",
        "tr_rolling_window_bars": trw if method == "rolling_tr_sum" else 0,
        "tr_rolling_window_count": rolling_count,
        "entry_rule_unchanged_except_baseline": (
            "strict H, completed 15-second signal, wait_next_real_trade open fill"
        ),
        "exit_rule": (
            "combined downside-speed close and strict-new-low frozen W net-drop rebound"
        ),
        "rebound_evidence_status": "active_v4_4_frozen_net_drop",
        "parent_engine_sha256": stage_manifest.get("engine_sha256"),
        "raw_output_schema_version": int(stage_manifest["schema_version"]),
        "plan_fingerprint_schema_version": int(
            stage_manifest["plan_fingerprint_schema_version"]
        ),
        "trade_audit_schema_version": int(
            stage_manifest["trade_audit_schema_version"]
        ),
        "trade_audit_schema_id": str(stage_manifest["trade_audit_schema_id"]),
        "train_return": _number(row.get("train_return")),
        "train_cost_adjusted_return": _number(
            row.get("train_cost_adjusted_return")
        ),
        "test_return": None,
        "all_return": _number(row.get("train_return")),
        "train_avg_trade": _number(row.get("train_avg_trade")),
        "train_cost_adjusted_avg_trade": _number(
            row.get("train_cost_adjusted_avg_trade")
        ),
        "test_avg_trade": None,
        "all_avg_trade": _number(row.get("train_avg_trade")),
        "train_max_drawdown": _number(row.get("train_max_drawdown")),
        "train_cost_adjusted_max_drawdown": _number(
            row.get("train_cost_adjusted_max_drawdown")
        ),
        "test_max_drawdown": None,
        "all_max_drawdown": _number(row.get("train_max_drawdown")),
        "train_trade_count": trade_count,
        "test_trade_count": 0,
        "all_trade_count": trade_count,
        "original_html_base": "__hidden__",
    }


def _native_source_value(source: pd.DataFrame, index: int, field: str) -> Any:
    if index < 0 or index >= len(source):
        return None
    return _jsonable(source.at[index, field])


def _native_source_time(source: pd.DataFrame, index: int) -> str | None:
    value = _native_source_value(source, index, "datetime")
    return str(value) if value is not None else None


def _native_strict_low_reset_count(source: pd.DataFrame, entry: int, low: int) -> int:
    if entry < 0 or low < entry or low >= len(source):
        return 0
    cache = source.attrs.setdefault("strict_low_reset_count_cache", {})
    cache_key = (entry, low)
    cached = cache.get(cache_key)
    if cached is not None:
        return int(cached)
    low_values = source.attrs.get("native_low_values")
    if low_values is None:
        low_values = source["low"].to_numpy(dtype=float)
        source.attrs["native_low_values"] = low_values
    values = low_values[entry : low + 1]
    if len(values) < 2:
        return 0
    current = values[0]
    count = 0
    for value in values[1:]:
        if value < current:
            current = value
            count += 1
    if len(cache) < 500_000:
        cache[cache_key] = count
    return count


def _native_trade_record(
    row: pd.Series,
    combo: dict[str, Any],
    source: pd.DataFrame,
    *,
    research_start_index: int,
    previous_exit_index: int | None,
    validate_source: bool = True,
) -> dict[str, Any]:
    if validate_source:
        _validate_trade_entry(row, source)
        _validate_trade_exit(row, source)
    detail = _record(row, TRADE_DETAIL_FIELDS)
    entry = int(row["entry_index"])
    signal = int(row["signal_index"])
    high_index = int(row["h_index"])
    exit_index = int(row["exit_index"])
    active_low_index = int(row["active_low_index"])
    rebound_start = int(row["rebound_window_start_index"])
    rebound_end = int(row["rebound_window_end_index"])
    e = int(row["e"])
    bh = int(row["bh"])
    trw = int(row["trw"])
    reset_index = previous_exit_index if previous_exit_index is not None else research_start_index
    signal_start = max(0, signal - e + 1, int(reset_index))
    baseline = float(row["entry_baseline_value"])
    drop = float(row["entry_drop_value"])
    entry_threshold = float(row["k"]) * baseline
    active_low = float(row["active_low"])
    rebound_net_drop = float(row["rebound_net_drop"])
    rebound_threshold_value = rebound_net_drop * float(row["m"])
    speed_window = int(_number(row.get("speed_window_bars")) or 0)
    speed_reference_low = _number(row.get("speed_reference_low"))
    speed_current_low = _number(row.get("speed_current_low"))
    speed_delta = (
        float(speed_reference_low) - float(speed_current_low)
        if speed_reference_low is not None and speed_current_low is not None
        else _number(row.get("speed_extension"))
    )
    exit_check = _number(row.get("rebound_check_price"))
    rebound_value = (
        float(exit_check) - active_low if exit_check is not None else None
    )
    baseline_start = int(row["baseline_history_start_index"])
    baseline_end = int(row["baseline_history_end_index"])
    latest_candidate_start = int(row["rebound_latest_applied_candidate_start_index"])
    latest_candidate_end = int(row["rebound_latest_applied_candidate_end_index"])
    exit_candidate_start = int(row["rebound_exit_bar_candidate_start_index"])
    exit_candidate_end = int(row["rebound_exit_bar_candidate_end_index"])
    candidates_effective_through = int(row["rebound_candidates_effective_through_index"])
    exit_open = _native_source_value(source, exit_index, "open")
    exit_high = _native_source_value(source, exit_index, "high")
    exit_low = _native_source_value(source, exit_index, "low")
    exit_close = _native_source_value(source, exit_index, "close")
    entry_open = _native_source_value(source, entry, "open")
    entry_high = _native_source_value(source, entry, "high")
    entry_low = _native_source_value(source, entry, "low")
    entry_close = _native_source_value(source, entry, "close")
    contract = _native_method_contract(str(row["method"]))
    rolling_count = bh - trw + 1 if str(row["method"]) == "rolling_tr_sum" else 0
    return {
        **detail,
        "segment": "train",
        "period": 1,
        "baseline_history_multiplier": bh,
        "baseline_window_multiplier": trw if str(row["method"]) == "rolling_tr_sum" else 1,
        "baseline_window_bars": trw if str(row["method"]) == "rolling_tr_sum" else 1,
        "baseline_sample_count_target": None,
        "baseline_required_sample_count": bh,
        "entry_baseline_anchor_policy": "signal_high_inclusive",
        "baseline_tr_atom_policy": combo["baseline_tr_atom_policy"],
        "baseline_anchor_index": high_index,
        "baseline_anchor_time": _native_source_time(source, high_index),
        "baseline_price_anchor_index": high_index,
        "baseline_price_anchor_time": _native_source_time(source, high_index),
        "baseline_tr_atom_start_index": baseline_start,
        "baseline_tr_atom_start_time": _native_source_time(source, baseline_start),
        "baseline_tr_atom_end_index": baseline_end,
        "baseline_tr_atom_end_time": _native_source_time(source, baseline_end),
        "baseline_value_lookup_index": high_index,
        "baseline_history_start_time": _native_source_time(source, baseline_start),
        "baseline_history_end_time": _native_source_time(source, baseline_end),
        "baseline_history_row_span": int(row["baseline_physical_span_bars"]),
        "baseline_history_elapsed_minutes": (baseline_end - baseline_start) * 0.25,
        "baseline_history_collection_policy": combo["baseline_history_collection_policy"],
        "k_drop": float(row["k"]),
        "k_context": float(row["k"]),
        "abs_floor_value": 0.0,
        "daily_floor_mult": 0.0,
        "exit_wait_multiplier": float(speed_window),
        "exit_wait_bars": speed_window,
        "exit_mode": str(row.get("exit_mode", combo["exit_mode"])),
        "speed_exit_enabled": _bool(row.get("speed_exit_enabled", False)),
        "rebound_exit_enabled": _bool(row.get("rebound_exit_enabled", True)),
        "rebound_tr_window_bars": int(row["w"]),
        "rebound_multiplier": float(row["m"]),
        "rebound_baseline_mode": "fixed_window_open_to_active_low_net_drop",
        "rebound_duration_insufficient_history_policy": "unavailable",
        "reverse_up_window_multiplier": 0.0,
        "reverse_up_window_bars": 0,
        "reverse_up_limit_multiplier": 0.0,
        "execution_mode": "research",
        "execution_mode_label": "研究模式",
        "fill_mode": "wait_next_real_trade_open_combined_exit",
        "entry_fill_policy": combo["entry_fill_policy"],
        "segment_end_exit_fill_policy": "segment end bar close",
        "fee_bps_per_side": 0.0,
        "slippage_bps_per_side": 0.0,
        "entry_signal_window_start_index": signal_start,
        "entry_window_start_time": _native_source_time(source, signal_start),
        "entry_window_end_time": _native_source_time(source, signal),
        "entry_effective_window_bars": signal - signal_start + 1,
        "entry_signal_confirmation_index": signal,
        "entry_signal_confirmation_time": _native_source_time(source, signal),
        "entry_signal_confirmation_reason": "completed_15s_signal_bar",
        "entry_fill_delay_bars": entry - signal,
        "entry_fill_timing_policy": "wait_next_real_trade_open_maximum_120_bars",
        "entry_signal_state_policy": combo["entry_signal_state_policy"],
        "entry_signal_reset_policy": combo["entry_signal_reset_policy"],
        "entry_signal_freshness_policy": combo["entry_signal_freshness_policy"],
        "entry_signal_source": "rolling_e_window",
        "entry_fresh_drop_event_type": "completed_15s_net_drop",
        "entry_signal_reset_applied": True,
        "entry_signal_reset_index": int(reset_index),
        "entry_signal_reset_time": _native_source_time(source, int(reset_index)),
        "entry_signal_reset_reason": (
            "previous_exit" if previous_exit_index is not None else "research_start"
        ),
        "previous_exit_index": previous_exit_index,
        "previous_exit_time": (
            _native_source_time(source, previous_exit_index)
            if previous_exit_index is not None
            else None
        ),
        "entry_window_crosses_prior_holding": False,
        "entry_window_includes_reset_boundary_bar": signal_start == int(reset_index),
        "entry_window_metadata_version": "v4_4_wait_next_real_trade_v1",
        "entry_window_continuous": True,
        "entry_after_time_break": False,
        "reverse_up_passed": True,
        "entry_signal_high_index": high_index,
        "entry_signal_high_time": _native_source_time(source, high_index),
        "entry_signal_high_price": _native_source_value(source, high_index, "high"),
        "entry_signal_low_index": signal,
        "entry_signal_low_time": _native_source_time(source, signal),
        "entry_signal_low_price": _native_source_value(source, signal, "low"),
        "signal_high_index": high_index,
        "signal_high_time": _native_source_time(source, high_index),
        "signal_high_price": _native_source_value(source, high_index, "high"),
        "signal_low_index": signal,
        "signal_low_time": _native_source_time(source, signal),
        "signal_low_price": _native_source_value(source, signal, "low"),
        "signal_drop_bars": signal - high_index + 1,
        "signal_low_lag_bars": 0,
        "entry_trigger_price": _number(row.get("entry_trigger_price")),
        "entry_gap_adjusted": _bool(row.get("entry_gap_adjusted")),
        "entry_bar_open": entry_open,
        "entry_bar_high": entry_high,
        "entry_bar_low": entry_low,
        "entry_close": entry_close,
        "drop_value": drop,
        "drop_effective_window_bars": signal - signal_start + 1,
        "entry_current_bar_fresh_ordered_drop": True,
        "drop_high_price": _native_source_value(source, high_index, "high"),
        "drop_low_price": _native_source_value(source, signal, "low"),
        "local_baseline_value": baseline,
        "entry_local_threshold_value": entry_threshold,
        "entry_abs_floor_value": 0.0,
        "entry_daily_floor_value": 0.0,
        "entry_threshold_value": entry_threshold,
        "drop_to_baseline_ratio": drop / baseline if baseline > 0 else None,
        "baseline_sample_count": int(row["baseline_eligible_atom_count"]),
        "exit_bar_open": exit_open,
        "exit_bar_high": exit_high,
        "exit_bar_low": exit_low,
        "exit_bar_close": exit_close,
        "return_rate": float(row["return"]),
        "hold_minutes": float(
            _number(row.get("holding_minutes"))
            or _number(row.get("hold_minutes"))
            or ((exit_index - entry) * 0.25)
        ),
        "hold_time_break_count": int(_bool(row["position_crosses_real_gap"])),
        "hold_cross_time_break": _bool(row["position_crosses_real_gap"]),
        "exit_window_continuous": not _bool(row["position_crosses_real_gap"]),
        "post_entry_low_index": active_low_index,
        "post_entry_low_time": _native_source_time(source, active_low_index),
        "post_entry_low_price": active_low,
        "trade_low_time": _native_source_time(source, active_low_index),
        "trade_low_price": active_low,
        "trade_low_basis": "strict_active_low",
        "rebound_anchor_high_index": high_index,
        "rebound_anchor_high_time": _native_source_time(source, high_index),
        "rebound_anchor_high_price": _native_source_value(source, high_index, "high"),
        "rebound_active_low_index": active_low_index,
        "rebound_active_low_time": _native_source_time(source, active_low_index),
        "rebound_active_low_price": active_low,
        "rebound_active_low_reset_count": _native_strict_low_reset_count(
            source, entry, active_low_index
        ),
        "rebound_active_low_reset_on_exit_bar": active_low_index == exit_index,
        "rebound_baseline_start_index": rebound_start,
        "rebound_baseline_start_time": _native_source_time(source, rebound_start),
        "rebound_baseline_end_index": rebound_end,
        "rebound_baseline_end_time": _native_source_time(source, rebound_end),
        "rebound_baseline_tr_count": int(row["w"]),
        "rebound_baseline_tr_sum": rebound_net_drop,
        "rebound_baseline_tr_mean": rebound_net_drop,
        "rebound_baseline_value": rebound_net_drop,
        "rebound_baseline_window_bars_requested": int(row["w"]),
        "rebound_baseline_observed_bar_count": int(
            row["rebound_window_observed_bar_count"]
        ),
        "rebound_baseline_window_first_open": _native_source_value(
            source, rebound_start, "open"
        ),
        "rebound_baseline_active_low_index": rebound_end,
        "rebound_baseline_active_low_time": _native_source_time(source, rebound_end),
        "rebound_baseline_active_low_price": _native_source_value(
            source, rebound_end, "low"
        ),
        "rebound_baseline_net_drop_value": rebound_net_drop,
        "rebound_baseline_exact_contiguous": (
            int(row["rebound_window_observed_bar_count"])
            == rebound_end - rebound_start + 1
        ),
        "rebound_baseline_gap_count": 0,
        "rebound_baseline_unavailable_reason": None,
        "rebound_baseline_reset_index": rebound_end,
        "rebound_baseline_reset_time": _native_source_time(source, rebound_end),
        "rebound_baseline_history_tr_count": 0,
        "rebound_duration_bars": exit_index - active_low_index,
        "rebound_baseline_rolling_window_count": 0,
        "rebound_baseline_extrapolated": False,
        "rebound_baseline_available": True,
        "rebound_threshold_value": rebound_threshold_value,
        "rebound_max_w_drop": _number(row.get("rebound_max_w_drop")),
        "rebound_latest_applied_candidate": _number(
            row.get("rebound_latest_applied_candidate")
        ),
        "rebound_latest_applied_candidate_start_time": _native_source_time(
            source, latest_candidate_start
        ),
        "rebound_latest_applied_candidate_end_time": _native_source_time(
            source, latest_candidate_end
        ),
        "rebound_exit_bar_candidate": _number(
            row.get("rebound_exit_bar_candidate")
        ),
        "rebound_exit_bar_candidate_start_time": _native_source_time(
            source, exit_candidate_start
        ),
        "rebound_exit_bar_candidate_end_time": _native_source_time(
            source, exit_candidate_end
        ),
        "rebound_candidates_effective_through_time": _native_source_time(
            source, candidates_effective_through
        ),
        "rebound_trigger_price": _number(row.get("rebound_trigger_price")),
        "rebound_check_price_basis": _jsonable(row.get("rebound_check_price_basis")),
        "rebound_check_price": exit_check,
        "rebound_value": rebound_value,
        "rebound_gap_adjusted": _bool(row.get("rebound_gap_adjusted")),
        "rebound_fill_policy": combo["rebound_exit_fill_policy"],
        "exit_speed_reference_end_time": _jsonable(row.get("speed_reference_time")),
        "exit_speed_previous_low_price": speed_reference_low,
        "exit_speed_current_low_price": speed_current_low,
        "exit_speed_delta": speed_delta,
        "entry_baseline_family": contract["family"],
        "entry_baseline_method": contract["slug"],
        "entry_baseline_method_label": contract["label"],
        "entry_baseline_formula": contract["formula"],
        "tr_history_bars": bh,
        "tr_component_count": bh,
        "tr_rolling_window_bars": trw if str(row["method"]) == "rolling_tr_sum" else 0,
        "tr_rolling_window_count": rolling_count,
        "entry_rule_unchanged_except_baseline": combo["entry_rule_unchanged_except_baseline"],
        "parent_engine_sha256": combo["parent_engine_sha256"],
        "entry_tr_sum_value": baseline,
    }


_PROCESS_CHUNK_SOURCE: pd.DataFrame | None = None
_PROCESS_CHUNK_RESEARCH_START_INDEX = 0


def _source_take(source: pd.DataFrame, indices: pd.Series, field: str) -> pd.Series:
    numeric = pd.to_numeric(indices, errors="coerce").fillna(-1).astype(int).to_numpy()
    values = source[field].to_numpy()
    valid = (numeric >= 0) & (numeric < len(values))
    result = np.empty(len(numeric), dtype=object)
    result[:] = None
    result[valid] = values[numeric[valid]]
    return pd.Series(result, index=indices.index)


def _native_trade_frame_fast(
    trades: pd.DataFrame,
    combo: dict[str, Any],
    source: pd.DataFrame,
    research_start_index: int,
) -> pd.DataFrame:
    ordered = trades.sort_values(["entry_index", "exit_index"], kind="mergesort").copy()
    out = ordered.copy()
    entry = pd.to_numeric(ordered["entry_index"], errors="raise").astype(int)
    signal = pd.to_numeric(ordered["signal_index"], errors="raise").astype(int)
    high = pd.to_numeric(ordered["h_index"], errors="raise").astype(int)
    exit_index = pd.to_numeric(ordered["exit_index"], errors="raise").astype(int)
    active_low_index = pd.to_numeric(ordered["active_low_index"], errors="raise").astype(int)
    rebound_start = pd.to_numeric(
        ordered["rebound_window_start_index"], errors="raise"
    ).astype(int)
    rebound_end = pd.to_numeric(
        ordered["rebound_window_end_index"], errors="raise"
    ).astype(int)
    previous_exit = exit_index.shift(1)
    reset = previous_exit.fillna(research_start_index).astype(int)
    e = pd.to_numeric(ordered["e"], errors="raise").astype(int)
    bh = pd.to_numeric(ordered["bh"], errors="raise").astype(int)
    trw = pd.to_numeric(ordered["trw"], errors="raise").astype(int)
    signal_start = pd.Series(
        np.maximum.reduce(
            [
                np.zeros(len(ordered), dtype=int),
                signal.to_numpy() - e.to_numpy() + 1,
                reset.to_numpy(),
            ]
        ),
        index=ordered.index,
    )
    baseline = pd.to_numeric(ordered["entry_baseline_value"], errors="raise")
    drop = pd.to_numeric(ordered["entry_drop_value"], errors="raise")
    active_low = pd.to_numeric(ordered["active_low"], errors="raise")
    rebound_drop = pd.to_numeric(ordered["rebound_net_drop"], errors="raise")
    multiplier = pd.to_numeric(ordered["m"], errors="raise")
    method = ordered["method"].astype(str)
    method_contract = _native_method_contract(str(method.iloc[0]))

    out["segment"] = "train"
    out["execution_mode"] = "research"
    out["slippage_bps_per_side"] = 0.0
    out["entry_baseline_anchor_policy"] = "signal_high_inclusive"
    out["baseline_anchor_index"] = high
    out["baseline_anchor_time"] = _source_take(source, high, "datetime")
    out["baseline_history_start_time"] = _source_take(
        source, ordered["baseline_history_start_index"], "datetime"
    )
    out["baseline_history_end_time"] = _source_take(
        source, ordered["baseline_history_end_index"], "datetime"
    )
    out["baseline_sample_count"] = ordered["baseline_eligible_atom_count"]
    out["entry_signal_window_start_index"] = signal_start
    out["entry_window_end_time"] = _source_take(source, signal, "datetime")
    out["entry_effective_window_bars"] = signal - signal_start + 1
    out["drop_effective_window_bars"] = signal - signal_start + 1
    out["entry_signal_reset_applied"] = True
    out["entry_signal_reset_time"] = _source_take(source, reset, "datetime")
    out["entry_signal_reset_reason"] = np.where(
        previous_exit.notna(), "previous_exit", "research_start"
    )
    out["previous_exit_time"] = _source_take(source, previous_exit, "datetime")
    out["entry_window_crosses_prior_holding"] = False
    out["entry_fresh_drop_event_type"] = "completed_15s_net_drop"
    out["entry_signal_high_index"] = high
    out["entry_signal_high_time"] = _source_take(source, high, "datetime")
    out["entry_signal_high_price"] = _source_take(source, high, "high")
    out["entry_signal_low_index"] = signal
    out["entry_signal_low_time"] = _source_take(source, signal, "datetime")
    out["entry_signal_low_price"] = _source_take(source, signal, "low")
    out["signal_high_index"] = high
    out["signal_high_time"] = out["entry_signal_high_time"]
    out["signal_high_price"] = out["entry_signal_high_price"]
    out["signal_low_index"] = signal
    out["signal_low_time"] = out["entry_signal_low_time"]
    out["signal_low_price"] = out["entry_signal_low_price"]
    out["entry_bar_open"] = _source_take(source, entry, "open")
    out["entry_close"] = _source_take(source, entry, "close")
    out["entry_gap_adjusted"] = ordered["entry_gap_adjusted"].map(_bool)
    out["drop_value"] = drop
    out["local_baseline_value"] = baseline
    out["entry_local_threshold_value"] = baseline * pd.to_numeric(
        ordered["k"], errors="raise"
    )
    out["entry_abs_floor_value"] = 0.0
    out["entry_daily_floor_value"] = 0.0
    out["entry_threshold_value"] = out["entry_local_threshold_value"]
    out["drop_to_baseline_ratio"] = drop / baseline.where(baseline.gt(0))
    out["exit_bar_open"] = _source_take(source, exit_index, "open")
    out["exit_bar_high"] = _source_take(source, exit_index, "high")
    out["exit_bar_low"] = _source_take(source, exit_index, "low")
    out["exit_bar_close"] = _source_take(source, exit_index, "close")
    out["return_rate"] = ordered["return"]
    out["exit_window_continuous"] = ~ordered["position_crosses_real_gap"].map(_bool)
    out["post_entry_low_index"] = active_low_index
    out["post_entry_low_time"] = _source_take(source, active_low_index, "datetime")
    out["post_entry_low_price"] = active_low
    out["trade_low_time"] = out["post_entry_low_time"]
    out["trade_low_price"] = active_low
    out["rebound_anchor_high_index"] = high
    out["rebound_anchor_high_time"] = out["entry_signal_high_time"]
    out["rebound_anchor_high_price"] = out["entry_signal_high_price"]
    out["rebound_active_low_index"] = active_low_index
    out["rebound_active_low_time"] = out["post_entry_low_time"]
    out["rebound_active_low_price"] = active_low
    out["rebound_active_low_reset_count"] = [
        _native_strict_low_reset_count(source, int(start), int(end))
        for start, end in zip(entry, active_low_index)
    ]
    out["rebound_active_low_reset_on_exit_bar"] = active_low_index.eq(exit_index)
    out["rebound_baseline_mode"] = "fixed_window_open_to_active_low_net_drop"
    out["rebound_duration_insufficient_history_policy"] = "unavailable"
    out["rebound_baseline_start_index"] = rebound_start
    out["rebound_baseline_start_time"] = _source_take(source, rebound_start, "datetime")
    out["rebound_baseline_end_index"] = rebound_end
    out["rebound_baseline_end_time"] = _source_take(source, rebound_end, "datetime")
    out["rebound_baseline_tr_count"] = ordered["w"]
    out["rebound_baseline_tr_sum"] = rebound_drop
    out["rebound_baseline_tr_mean"] = rebound_drop
    out["rebound_baseline_value"] = rebound_drop
    out["rebound_baseline_window_bars_requested"] = ordered["w"]
    out["rebound_baseline_observed_bar_count"] = ordered[
        "rebound_window_observed_bar_count"
    ]
    out["rebound_baseline_window_first_open"] = _source_take(source, rebound_start, "open")
    out["rebound_baseline_active_low_price"] = _source_take(source, rebound_end, "low")
    out["rebound_baseline_active_low_index"] = rebound_end
    out["rebound_baseline_active_low_time"] = _source_take(source, rebound_end, "datetime")
    out["rebound_baseline_net_drop_value"] = rebound_drop
    out["rebound_baseline_exact_contiguous"] = pd.to_numeric(
        ordered["rebound_window_observed_bar_count"], errors="raise"
    ).eq(rebound_end - rebound_start + 1)
    out["rebound_baseline_unavailable_reason"] = None
    out["rebound_baseline_history_tr_count"] = 0
    out["rebound_duration_bars"] = exit_index - active_low_index
    out["rebound_baseline_rolling_window_count"] = 0
    out["rebound_baseline_extrapolated"] = False
    out["rebound_baseline_available"] = True
    out["rebound_threshold_value"] = rebound_drop * multiplier
    out["rebound_latest_applied_candidate_start_time"] = _source_take(
        source, ordered["rebound_latest_applied_candidate_start_index"], "datetime"
    )
    out["rebound_latest_applied_candidate_end_time"] = _source_take(
        source, ordered["rebound_latest_applied_candidate_end_index"], "datetime"
    )
    out["rebound_exit_bar_candidate_start_time"] = _source_take(
        source, ordered["rebound_exit_bar_candidate_start_index"], "datetime"
    )
    out["rebound_exit_bar_candidate_end_time"] = _source_take(
        source, ordered["rebound_exit_bar_candidate_end_index"], "datetime"
    )
    out["rebound_candidates_effective_through_time"] = _source_take(
        source, ordered["rebound_candidates_effective_through_index"], "datetime"
    )
    out["rebound_value"] = pd.to_numeric(
        ordered["rebound_check_price"], errors="coerce"
    ) - active_low
    out["rebound_gap_adjusted"] = ordered["rebound_gap_adjusted"].map(_bool)
    out["exit_speed_reference_end_time"] = ordered["speed_reference_time"]
    out["exit_speed_previous_low_price"] = ordered["speed_reference_low"]
    out["exit_speed_current_low_price"] = ordered["speed_current_low"]
    out["exit_speed_delta"] = pd.to_numeric(
        ordered["speed_reference_low"], errors="coerce"
    ) - pd.to_numeric(ordered["speed_current_low"], errors="coerce")
    out["entry_baseline_method"] = method_contract["slug"]
    out["entry_baseline_method_label"] = method_contract["label"]
    out["rebound_tr_window_bars"] = ordered["w"]
    out["rebound_multiplier"] = ordered["m"]
    out["exit_wait_bars"] = ordered["speed_window_bars"]
    out["tr_history_bars"] = bh
    out["tr_component_count"] = bh
    out["tr_rolling_window_bars"] = np.where(method.eq("rolling_tr_sum"), trw, 0)
    out["tr_rolling_window_count"] = np.where(
        method.eq("rolling_tr_sum"), bh - trw + 1, 0
    )
    out["entry_tr_sum_value"] = baseline
    out["reverse_up_window_multiplier"] = 0.0
    out["reverse_up_window_bars"] = 0
    out["reverse_up_limit_multiplier"] = 0.0
    return out


def _initialize_chunk_process(
    source_path: str,
    train_start: str,
    train_end: str,
) -> None:
    global _PROCESS_CHUNK_SOURCE, _PROCESS_CHUNK_RESEARCH_START_INDEX
    source = pd.read_csv(Path(source_path), usecols=list(SOURCE_COLUMNS))
    source = source.loc[source["datetime"].astype(str).le(train_end)].reset_index(drop=True)
    research_positions = np.flatnonzero(
        source["datetime"].astype(str).ge(train_start).to_numpy()
    )
    if not len(research_positions):
        raise ValueError("V4.4 training start is outside the review source")
    _PROCESS_CHUNK_SOURCE = source
    _PROCESS_CHUNK_RESEARCH_START_INDEX = int(research_positions[0])


def _build_combo_chunk_process(
    task: tuple[str, dict[str, Any], pd.DataFrame, str],
) -> tuple[str, str, Path, int, int, int]:
    combo_id, combo, combo_trades, chunk_path_value = task
    source = _PROCESS_CHUNK_SOURCE
    if source is None:
        raise RuntimeError("native trade chunk process has no source")
    native_frame = _native_trade_frame_fast(
        combo_trades,
        combo,
        source,
        _PROCESS_CHUNK_RESEARCH_START_INDEX,
    )
    waits = pd.to_numeric(combo_trades["entry_wait_bar_count"], errors="raise")
    waited_count = int(waits.gt(0).sum())
    maximum_wait = int(waits.max()) if len(waits) else 0
    native_json = native_frame.to_json(
        orient="records",
        force_ascii=False,
        double_precision=15,
    )
    chunk_path = Path(chunk_path_value)
    filename = chunk_path.name
    _atomic_text(
        chunk_path,
        "window.NATIVE_COMBO="
        + _compact_json(combo)
        + ";window.NATIVE_TRADES="
        + native_json
        + ";\n",
    )
    return (
        combo_id,
        filename,
        chunk_path,
        int(len(native_frame)),
        waited_count,
        maximum_wait,
    )


def _native_process_features(
    source: pd.DataFrame,
    combos: list[dict[str, Any]],
    stage_manifest: dict[str, Any],
    filter_events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "datetime": source["datetime"].astype(str).tolist(),
        "open": source["open"].astype(float).tolist(),
        "high": source["high"].astype(float).tolist(),
        "low": source["low"].astype(float).tolist(),
        "close": source["close"].astype(float).tolist(),
        "daily_floor_basis_value": [],
        "research_intervals": [{
            "segment": "train",
            "start": stage_manifest.get("train_start"),
            "end": stage_manifest.get("train_end"),
        }],
        "entry_signal_state_version": "v4_4_strict_h_wait_next_real_trade",
        "entry_signal_state_policy": "strict H in E with post-exit lifecycle reset",
        "entry_signal_reset_policy": "reset at every position close",
        "entry_signal_freshness_policy": "completed 15-second signal confirmation",
        "entry_signal_interval_start_policy": "one fixed full-training sample",
        "process_drop_series_scope": "trade-audit records; chart path uses source OHLC",
        "baseline_filter_overlay_id": FILTER_OVERLAY_ID,
        "baseline_filter_events": filter_events,
        "unified_n1": True,
        "entry_baseline_anchor_policy": "signal_high_inclusive",
        "baseline_history_collection_policy": {
            policy: _baseline_sampling_contract(policy)
            for policy in sorted(
                {str(item["baseline_sampling_policy"]) for item in combos}
            )
        },
        "baseline_sampling_policies": sorted(
            {str(item["baseline_sampling_policy"]) for item in combos}
        ),
        "periods": [1],
        "entry_window_multipliers": sorted({item["entry_window_multiplier"] for item in combos}),
        "baseline_history_multipliers": sorted({item["baseline_history_multiplier"] for item in combos}),
        "baseline_window_multipliers": sorted({item["baseline_window_multiplier"] for item in combos}),
        "baseline_sample_counts": [],
        "k_drop": sorted({item["k_drop"] for item in combos}),
        "k_context": sorted({item["k_context"] for item in combos}),
        "abs_floors": [0.0],
        "daily_floors": [0.0],
        "exit_wait_multipliers": sorted({item["exit_wait_bars"] for item in combos}),
        "reverse_up_window_multipliers": [0.0],
        "reverse_up_limit_multipliers": [0.0],
        "combos": combos,
        "default_combo_id": combos[0]["combo_id"],
        "entry_baseline_family": combos[0]["entry_baseline_family"],
        "entry_baseline_method": combos[0]["entry_baseline_method"],
    }


def _historical_trade_html(
    main_href: str,
    *,
    default_start_date: str = "2026-05-26",
    instrument_label: str = "K200",
    peer_review_href: str | None = None,
    peer_review_label: str | None = None,
    peer_research_contract_id: str | None = None,
) -> str:
    if not TRADE_DESIGN_SOURCE.is_file():
        raise FileNotFoundError(f"historical V4 trade template is missing: {TRADE_DESIGN_SOURCE}")
    if _sha256(TRADE_DESIGN_SOURCE) != TRADE_DESIGN_SOURCE_SHA256:
        raise ValueError("historical V4 trade template hash changed")
    source = TRADE_DESIGN_SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        "<!doctype html>",
        f"<!doctype html>\n<!-- {OWNER_MARKER}; exact-template-source-sha256: {TRADE_DESIGN_SOURCE_SHA256} -->",
        1,
    )
    source = source.replace(
        "V4 净跌幅回撤逐笔查看",
        f"{RELEASE_VERSION_LABEL} 组合平仓逐笔查看",
    )
    source = source.replace('../outcome_native/index.html', main_href)
    peer_review = (
        {
            "href": peer_review_href,
            "label": peer_review_label,
            "researchContractId": peer_research_contract_id,
        }
        if peer_review_href and peer_review_label
        else None
    )
    source = source.replace(
        "const peerReview = null;",
        "const peerReview = "
        + json.dumps(peer_review, ensure_ascii=False, separators=(",", ":"))
        + ";",
        1,
    )
    source = source.replace(
        'const DEFAULT_START_DATE = "2026-05-26";',
        f"const DEFAULT_START_DATE = {json.dumps(default_start_date, ensure_ascii=False)};",
        1,
    )
    source = source.replace(
        'name:"K200",increasing:',
        f"name:{json.dumps(instrument_label, ensure_ascii=False)},increasing:",
        1,
    )
    max_w_audit_function = r'''
function v43MaxWAuditText(trade) {
  const reason = canonicalExitReason(trade?.exit_reason);
  const policy = trade?.rebound_baseline_policy_id || "未记录";
  const effective = trade?.rebound_max_w_drop ?? trade?.rebound_net_drop;
  const maxSource = `${trade?.rebound_baseline_start_time ?? "—"} 至 ${trade?.rebound_baseline_end_time ?? "—"}（index ${trade?.rebound_window_start_index ?? "—"} 至 ${trade?.rebound_window_end_index ?? "—"}，observed=${trade?.rebound_window_observed_bar_count ?? "—"}）`;
  const latestSource = `${trade?.rebound_latest_applied_candidate_start_time ?? "—"} 至 ${trade?.rebound_latest_applied_candidate_end_time ?? "—"}（index ${trade?.rebound_latest_applied_candidate_start_index ?? "—"} 至 ${trade?.rebound_latest_applied_candidate_end_index ?? "—"}，observed=${trade?.rebound_latest_applied_candidate_observed_bar_count ?? "—"}）`;
  const exitCandidateSource = `${trade?.rebound_exit_bar_candidate_start_time ?? "—"} 至 ${trade?.rebound_exit_bar_candidate_end_time ?? "—"}（index ${trade?.rebound_exit_bar_candidate_start_index ?? "—"} 至 ${trade?.rebound_exit_bar_candidate_end_index ?? "—"}，observed=${trade?.rebound_exit_bar_candidate_observed_bar_count ?? "—"}）`;
  const timing = reason === "rebound_threshold"
    ? "当前退出／触发 bar 的 W candidate 仅作审计，不进入同 bar 的回撤阈值；它从下一根 bar 才可能生效。"
    : reason === "downside_speed_below_threshold"
      ? "当前退出／触发 bar 未触发回撤后，其 W candidate 已应用，再执行同 bar 的速度检查。"
      : "区间末 bar 未触发回撤后，其 W candidate 已应用，再执行 segment_end 平仓。";
  return ` 闭合 bar max-W 审计：policy=${esc(policy)}；本次退出采用 max-W drop=${fmtRecorded(effective,4)}，max 来源=${esc(maxSource)}；最新已应用 candidate=${fmtRecorded(trade?.rebound_latest_applied_candidate,4)}，来源=${esc(latestSource)}；当前退出／触发 bar candidate=${fmtRecorded(trade?.rebound_exit_bar_candidate,4)}，来源=${esc(exitCandidateSource)}；候选已应用至 ${esc(trade?.rebound_candidates_effective_through_time ?? "—")}（index ${esc(trade?.rebound_candidates_effective_through_index ?? "—")}）。${timing}`;
}
'''
    source = source.replace(
        "function renderExitReason(trade, combo) {",
        max_w_audit_function + "\nfunction renderExitReason(trade, combo) {",
        1,
    )
    max_w_text_replacements = {
        "精确连续窗口结束于 active low，首根 open=${fmtRecorded(trade.rebound_baseline_window_first_open,4)}，active low=${fmtRecorded(trade.rebound_baseline_active_low_price ?? reboundLowPrice,4)}，净下跌基准=${fmtRecorded(reboundBaseline,4)}": (
            "max-W 来源窗口首根 open=${fmtRecorded(trade.rebound_baseline_window_first_open,4)}，窗口末端 low=${fmtRecorded(trade.rebound_baseline_active_low_price,4)}，有效 max-W drop=${fmtRecorded(reboundBaseline,4)}"
        ),
        "V3 退出 W 与开仓 H 无关；W 以严格 active low 为右端。最后 active low：": (
        "V4.4 max-W 来源窗口不得早于本次交易 H；仅使用上一已完成 bar 及更早候选。当前 active low："
        ),
        "窗口含 active low bar，并要求每相邻 bar 恰好相差 ${esc(executionBarSeconds)} 秒。窗口首根 open=${fmtRecorded(trade.rebound_baseline_window_first_open,4)}，active low=${fmtRecorded(trade.rebound_baseline_active_low_price ?? activeLowPrice,4)}，净下跌基准=${fmtRecorded(trade.rebound_baseline_window_first_open,4)} − ${fmtRecorded(trade.rebound_baseline_active_low_price ?? activeLowPrice,4)} = ${fmtRecorded(baselineMean,4)}。实际 bar 数=${esc(trade.rebound_baseline_observed_bar_count)}／${esc(trade.rebound_baseline_window_bars_requested)}，精确连续=${esc(trade.rebound_baseline_exact_contiguous)}。严格新低重算，相同低点不重算，反弹期间冻结。": (
            "来源为 H 及以后正的已完成 bar W candidate 单调最大值，并要求每相邻 bar 恰好相差 ${esc(executionBarSeconds)} 秒。窗口首根 open=${fmtRecorded(trade.rebound_baseline_window_first_open,4)}，窗口末端 low=${fmtRecorded(trade.rebound_baseline_active_low_price,4)}，max-W drop=${fmtRecorded(trade.rebound_baseline_window_first_open,4)} − ${fmtRecorded(trade.rebound_baseline_active_low_price,4)} = ${fmtRecorded(baselineMean,4)}。实际 bar 数=${esc(trade.rebound_baseline_observed_bar_count)}／${esc(trade.rebound_baseline_window_bars_requested)}，精确连续=${esc(trade.rebound_baseline_exact_contiguous)}。当前完成 bar 的候选从下一根 bar 才生效。"
        ),
        "V3 回撤基准绑定当前严格 active low，并向前取固定 W 根执行 bar；开仓信号高点 H 只属于开仓逻辑，不进入本次回撤基准公式。": (
            "V4.4 回撤基准取 H 及以后正的已完成 bar W candidate 单调最大值；当前检查 bar 的候选不进入同 bar 回撤阈值。"
        ),
        " 这里是最终冻结点 L；从 L 到平仓经过 ${esc(frozenInterval)}，期间没有严格新低，W 窗口、净下跌基准和回撤阈值保持不变。": (
            " active low 仍是回撤量的价格锚点；max-W 基准由独立来源窗口给出，并只吸收上一已完成 bar 及更早候选。"
        ),
        "V3 退出 W：以严格 active low bar 为右端，向前取精确连续 W 根 ${timeframeUnit(executionBarSeconds)} 执行 bar；基准=窗口首根 open−active low，M 乘以该净下跌基准。严格新低重算，相同低点不重算，反弹期间冻结。": (
            "V4.4 max-W：每根已完成 bar 形成一个起点不早于 H、最多 W 根的连续候选，candidate=窗口首根 open−窗口末端 low；仅保留正有限值，下一根 bar 生效，交易内取单调最大值。"
        ),
        "当前为 ${reboundWindowLabel(combo.rebound_tr_window_bars)}。窗口结束于并包含严格 active low bar；要求每相邻 bar 恰好相差 ${esc(executionBarSeconds)} 秒。基准=窗口首根 open−active low，少于 W 根、时间缺口、非有限价格或非正净下跌都会使该次基准不可用。": (
            "当前为 ${reboundWindowLabel(combo.rebound_tr_window_bars)}。每根已完成 bar 形成起点不早于 H、最多 W 根的连续窗口；candidate=窗口首根 open−窗口末端 low。仅正有限候选参与单调最大值，当前 bar 候选从下一根 bar 才生效。"
        ),
        "持仓期间逐 bar 检查。严格新低会重置 active low，并在该 bar 收盘时重算 W 窗口净下跌；相同低点不会重置。新低 bar 用 close − low，普通 bar 用 high − active low；反弹期间基准冻结。": (
            "持仓期间逐 bar 检查。旧阈值满足 open≥trigger 时按 open 平仓，否则 high≥trigger 时按 trigger 平仓；相等也触发。若该 bar 严格创新低且前两项未触发，则用 close−active low 判断，满足时按该真实 bar close 平仓。检查结束且未触发回撤后，当前 bar candidate 才加入后续状态。"
        ),
        "W=精确连续 15 秒执行 bar，结束于 active low<br>净下跌基准=窗口首根 open−active low=": (
            "W=起点不早于 H 的已完成 15 秒连续候选，当前 bar 下一根生效<br>max-W drop=来源窗口首根 open−末端 low="
        ),
        "if (basis === \"segment_end_bar_close\") return \"研究区间末 bar close\";": (
            "if (basis === \"segment_end_bar_close\" || basis === \"sample_end_bar_close\") return \"研究区间末 bar close\";\n"
            "  if (basis === \"exit_bar_open_at_or_above_rebound_trigger\") return \"open 大于或等于理论回撤线，按 open\";\n"
            "  if (basis === \"calculated_rebound_threshold\") return \"high 大于或等于理论回撤线，按理论线\";\n"
            "  if (basis === \"same_bar_close_after_strict_new_low_confirmation\") return \"严格新低 bar 由 close 确认，按 close\";"
        ),
        "[\"回撤成交\", `理论 cover=active low + threshold。普通 bar 的 open 已高于理论 cover 时按 open；其它达到阈值的情形按理论 cover。当前 policy=${esc(combo.rebound_fill_policy || \"未记录\")}。`]": (
            "[\"回撤成交\", `旧理论线满足 open≥trigger 时按 open；否则 high≥trigger 时按 trigger，相等也退出。严格新低 bar 在 close≥新理论线时按该真实 bar close；非真实 bar 进入 pending_exit。当前 policy=${esc(combo.rebound_fill_policy || \"未记录\")}。`]"
        ),
    }
    for old, new in max_w_text_replacements.items():
        if old not in source:
            raise ValueError(f"historical V4 trade template lacks max-W adapter marker: {old}")
        source = source.replace(old, new)
    overlay_functions = r'''
function v43FilterColors(eventType) {
  const dark = currentTheme === "dark";
  if (eventType === "universal_low_volume") return dark
    ? {line:"#4de1bd",fill:"rgba(77,225,189,.12)",label:"通用低成交量"}
    : {line:"#0f9f83",fill:"rgba(15,159,131,.13)",label:"通用低成交量"};
  if (String(eventType).includes("lock")) return dark
    ? {line:"#d887ee",fill:"rgba(216,135,238,.13)",label:"涨跌停锁价候选"}
    : {line:"#a23bb9",fill:"rgba(162,59,185,.14)",label:"涨跌停锁价候选"};
  if (eventType === "circuit_breaker_candidate") return dark
    ? {line:"#ff9c3d",fill:"rgba(255,156,61,.14)",label:"熔断阶段"}
    : {line:"#e37a12",fill:"rgba(227,122,18,.15)",label:"熔断阶段"};
  return dark
    ? {line:"#8995a5",fill:"rgba(137,149,165,.12)",label:"过滤区间"}
    : {line:"#7b8798",fill:"rgba(123,135,152,.13)",label:"过滤区间"};
}
function v43FilterRangeShape(range, startTime, endTime, colors) {
  const datetimes = features?.datetime || [];
  const start = datetimeLowerBound(datetimes, String(startTime || ""));
  const end = datetimeUpperBound(datetimes, String(endTime || "")) - 1;
  if (end < range.start || start > range.end || start > end) return null;
  const startX = indexToVisibleX(Math.max(range.start, start), range);
  const endX = indexToVisibleX(Math.min(range.end, end), range);
  return xRangeBand(Number(startX)-.5,Number(endX)+.5,colors.fill);
}
function v43FilterEventShapes(range) {
  const events = (features?.baseline_filter_events || []).filter(item => item.apply_to_baseline === true);
  return events.flatMap(event => {
    const colors = v43FilterColors(event.event_type);
    if (event.event_type === "circuit_breaker_candidate" && event.halt_end && event.call_auction_start) {
      const dark = currentTheme === "dark";
      const auction = dark
        ? {line:"#ff6f61",fill:"rgba(255,111,97,.12)"}
        : {line:"#c44725",fill:"rgba(196,71,37,.13)"};
      return cleanShapes([
        v43FilterRangeShape(range,event.start,event.halt_end,colors),
        v43FilterRangeShape(range,event.call_auction_start,event.end,auction)
      ]);
    }
    return cleanShapes([v43FilterRangeShape(range,event.start,event.end,colors)]);
  });
}
function v43FilterLegendTraces() {
  const seen = new Set();
  return (features?.baseline_filter_events || [])
    .filter(item => item.apply_to_baseline === true)
    .flatMap(event => {
      const colors = v43FilterColors(event.event_type);
      if (seen.has(colors.label)) return [];
      seen.add(colors.label);
      return [{type:"scatter",mode:"markers",x:[null],y:[null],name:colors.label,
        marker:{symbol:"square",size:9,color:colors.line},hoverinfo:"skip",yaxis:"y",showlegend:true}];
    });
}
'''
    source = source.replace(
        "function quietActivityShapes(range) {",
        overlay_functions + "\nfunction quietActivityShapes(range) {",
        1,
    )
    source = source.replace(
        "  const quietShapes = quietActivityShapes(range);",
        "  const filterShapes = v43FilterEventShapes(range);\n"
        "  data.push(...v43FilterLegendTraces());\n"
        "  const quietShapes = quietActivityShapes(range);",
        1,
    )
    source = source.replace(
        "  const shapes = quietShapes.slice();",
        "  const shapes = [...filterShapes, ...quietShapes];",
        1,
    )
    source = source.replace(
        "。${reboundRangeText}`",
        "。${reboundRangeText}${v43MaxWAuditText(trade)}`",
        1,
    )
    source = source.replace(
        "）。${baselineAvailabilityText}`",
        "）。${baselineAvailabilityText}${v43MaxWAuditText(trade)}`",
    )
    if (
        FILTER_OVERLAY_ID not in source
        or "v43FilterEventShapes" not in source
        or "v43MaxWAuditText" not in source
    ):
        source = source.replace(
            OWNER_MARKER,
            f"{OWNER_MARKER}; {FILTER_OVERLAY_ID}",
            1,
        )
    return source


def refresh_trade_review_shell(
    output: Path,
    *,
    main_href: str = "../index.html",
    default_start_date: str = "2026-05-26",
    instrument_label: str = "K200",
    peer_review_href: str | None = None,
    peer_review_label: str | None = None,
    peer_research_contract_id: str | None = None,
) -> dict[str, Any]:
    """Refresh presentation-only trade-review files without rebuilding data payloads."""
    output = output.resolve()
    manifest_path = output / "trade_review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    peer_review = (
        {
            "href": peer_review_href,
            "label": peer_review_label,
            "research_contract_id": peer_research_contract_id,
        }
        if peer_review_href and peer_review_label
        else None
    )
    if (
        str(manifest.get("template_source", {}).get("sha256", ""))
        == TRADE_DESIGN_SOURCE_SHA256
        and manifest.get("peer_review") == peer_review
        and manifest.get("release_version_label") == RELEASE_VERSION_LABEL
    ):
        return {"refreshed": False, "index": output / "index.html", "manifest": manifest_path}

    index_path = output / "index.html"
    resource_audit_path = output / "resource_audit.json"
    _atomic_text(
        index_path,
        _historical_trade_html(
            main_href,
            default_start_date=default_start_date,
            instrument_label=instrument_label,
            peer_review_href=peer_review_href,
            peer_review_label=peer_review_label,
            peer_research_contract_id=peer_research_contract_id,
        ),
    )
    resource_audit = json.loads(resource_audit_path.read_text(encoding="utf-8"))
    resource_audit["template_source"] = str(TRADE_DESIGN_SOURCE)
    resource_audit["template_source_sha256"] = TRADE_DESIGN_SOURCE_SHA256
    resource_audit["peer_review"] = peer_review
    resource_audit["release_version_label"] = RELEASE_VERSION_LABEL
    resource_audit["range_statistics_policy"] = (
        "compute once from already-loaded visible OHLC after horizontal selection; "
        "compact flat panel in the right detail column; clear native selection state, preserve "
        "normal chart opacity, and redraw the pale borderless interval as an ordinary Plotly shape; "
        "close clears selection and restores zoom; "
        "no startup precomputation, cache, or additional request"
    )
    resource_audit["holding_check_policy"] = (
        "after explicit activation, keep Plotly zoom drag and pair chart pointer-down with "
        "document-level pointer-up; treat movement beyond five pixels as a drag and map a short "
        "left press to the nearest visible candle; mark the selected close with a blue Plotly marker "
        "that matches the entry marker's pixel size and zoom behavior, then derive "
        "the recorded trade's "
        "holding state from loaded OHLC and fixed combo parameters; mirror max completed-W "
        "candidate next-bar effectiveness and speed-window continuity; for an aggregated candle "
        "use its final source bar; shade the baseline start through the selected-time active low "
        "with a darker borderless blue fill; draw a pale-red dense dashed horizontal guide at "
        "active-low plus rebound threshold and a labeled vertical guide at active-low-index plus S, "
        "the no-new-low theoretical speed-exit position; place the blue selection marker at the "
        "displayed candle's aggregate high and center; render the panel in the right detail column; no startup computation "
        "or additional request"
    )
    _atomic_text(
        resource_audit_path,
        json.dumps(resource_audit, ensure_ascii=False, indent=2) + "\n",
    )

    manifest["generator"] = _artifact(Path(__file__))
    manifest["template_source"] = _artifact(TRADE_DESIGN_SOURCE)
    manifest["peer_review"] = peer_review
    manifest["release_version_label"] = RELEASE_VERSION_LABEL
    refreshed_outputs = {
        index_path.resolve(): _artifact(index_path, root=output),
        resource_audit_path.resolve(): _artifact(resource_audit_path, root=output),
    }
    manifest["outputs"] = [
        refreshed_outputs.get(
            (output / Path(str(record["path"]))).resolve(), record
        )
        for record in manifest.get("outputs", [])
    ]
    _atomic_text(
        manifest_path,
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )
    return {
        "refreshed": True,
        "index": index_path,
        "manifest": manifest_path,
        "resource_audit": resource_audit_path,
    }


def _stage_filter_bundle(
    stage_manifest: dict[str, Any],
    source_sha256: str,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    manifest_path = Path(str(stage_manifest["data_preparation_manifest"])).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("V4.4 preparation manifest is not complete for trade review")
    prepared_identity = str(payload.get("prepared_identity", ""))
    if not prepared_identity.startswith(
        ("v4_4_policy_neutral_baseline_marker_", "v4_4_confirmed_low_activity_gate_")
    ):
        raise ValueError("V4.4 trade review requires an approved baseline preparation")
    if str(payload.get("source_sha256", "")).lower() != source_sha256.lower():
        raise ValueError("V4.4 trade-review preparation source hash mismatch")
    marker_entry = payload.get("artifacts", {}).get("filter_atoms", {})
    events_entry = payload.get("artifacts", {}).get("filter_events", {})
    def resolve_preparation_artifact(value: Any) -> Path:
        path = Path(str(value))
        return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()

    marker_path = resolve_preparation_artifact(marker_entry.get("path", ""))
    events_path = resolve_preparation_artifact(events_entry.get("path", ""))
    for path, expected in (
        (marker_path, marker_entry.get("sha256")),
        (events_path, events_entry.get("sha256")),
    ):
        if not path.is_file() or not expected or _sha256(path) != str(expected):
            raise ValueError(f"V4.4 trade-review filter artifact failed hash validation: {path}")
    events_payload = json.loads(events_path.read_text(encoding="utf-8"))
    events = [
        dict(item) for item in events_payload.get("events", [])
        if bool(item.get("apply_to_baseline", False))
    ]
    return marker_path, events_path, events


def build_stage_trade_review(
    output: Path,
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    stage_manifest: dict[str, Any],
    completion_manifest: dict[str, Any],
    *,
    analysis_identity: str,
    manifest_href: str = "../analysis_manifest.json",
    main_href: str = "../index.html",
    research_contract_label: str = "V4.4 情景三全样本训练",
    default_start_date: str = "2026-05-26",
    instrument_label: str = "K200",
    peer_review_href: str | None = None,
    peer_review_label: str | None = None,
    peer_research_contract_id: str | None = None,
    workers: int = 4,
    reuse_trade_review: Path | None = None,
    reuse_chunk_directory: Path | None = None,
    expected_trade_count: int | None = None,
    reused_trade_stats: dict[str, int] | None = None,
    publication_chunk_counts: dict[str, int] | None = None,
    source_frame: pd.DataFrame | None = None,
    chunks_only: bool = False,
) -> dict[str, Any]:
    """Generate the reused V4 trade-review shell for one closed V4.4 stage."""
    output = output.resolve()
    source_path = Path(str(stage_manifest["source"])).resolve()
    source_sha256 = str(stage_manifest["source_sha256"])
    if source_frame is None and (
        not source_path.is_file() or _sha256(source_path) != source_sha256
    ):
        raise ValueError("V4.4 stage market source failed hash validation")
    filter_marker_path, filter_events_path, filter_events = _stage_filter_bundle(
        stage_manifest, source_sha256
    )
    if summary.empty or summary["combo_id"].astype(str).duplicated().any():
        raise ValueError("V4.4 trade review requires unique non-empty summary rows")
    if "baseline_sampling_policy" not in summary:
        raise ValueError("V4.4 summary lacks baseline_sampling_policy")
    policies = sorted(summary["baseline_sampling_policy"].astype(str).unique())
    unknown_policies = set(policies).difference(BASELINE_SAMPLING_POLICY_CONTRACTS)
    if unknown_policies:
        raise ValueError(f"unsupported V4.4 baseline sampling policies: {unknown_policies}")
    declared_policies = stage_manifest.get("baseline_sampling_policies")
    if declared_policies is None:
        declared_policies = [stage_manifest.get("baseline_sampling_policy")]
    if sorted(str(value) for value in declared_policies) != policies:
        raise ValueError("V4.4 summary and stage baseline policies disagree")
    if not trades.empty:
        if "baseline_sampling_policy" not in trades:
            raise ValueError("V4.4 trades lack baseline_sampling_policy")
        trade_policy_by_combo = trades.groupby("combo_id")[
            "baseline_sampling_policy"
        ].nunique()
        if trade_policy_by_combo.gt(1).any():
            raise ValueError("one combo_id contains multiple baseline sampling policies")
    summary_combo_ids = set(summary["combo_id"].astype(str))
    trade_combo_ids = set(trades["combo_id"].astype(str)) if "combo_id" in trades else set()
    if trade_combo_ids.difference(summary_combo_ids):
        raise ValueError("V4.4 trade review contains trades for an unknown summary combo")
    if int(completion_manifest["coordinate_count"]) != len(summary):
        raise ValueError("V4.4 trade review coordinate count differs from completion")
    resolved_trade_count = (
        len(trades) if expected_trade_count is None else int(expected_trade_count)
    )
    if int(completion_manifest["trade_count"]) != resolved_trade_count:
        raise ValueError("V4.4 trade review trade count differs from completion")
    identity_checks = {
        "raw_output_schema_version": int(stage_manifest.get("schema_version", -1))
        == OUTPUT_SCHEMA_VERSION
        and int(completion_manifest.get("schema_version", -1))
        == OUTPUT_SCHEMA_VERSION,
        "plan_fingerprint_schema_version": int(
            stage_manifest.get("plan_fingerprint_schema_version", -1)
        )
        == FINGERPRINT_SCHEMA_VERSION,
        "trade_audit_schema": int(
            stage_manifest.get("trade_audit_schema_version", -1)
        )
        == COMBINED_TRADE_AUDIT_SCHEMA_VERSION
        and str(stage_manifest.get("trade_audit_schema_id", ""))
        == COMBINED_TRADE_AUDIT_SCHEMA_ID
        and int(completion_manifest.get("trade_audit_schema_version", -1))
        == COMBINED_TRADE_AUDIT_SCHEMA_VERSION
        and str(completion_manifest.get("trade_audit_schema_id", ""))
        == COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "rebound_baseline_policy": str(
            stage_manifest.get("rebound_baseline_policy_id", "")
        )
        == REBOUND_BASELINE_POLICY_ID
        and str(completion_manifest.get("rebound_baseline_policy_id", ""))
        == REBOUND_BASELINE_POLICY_ID,
    }
    if not all(identity_checks.values()):
        raise ValueError(f"V4.4 max-W review identity mismatch: {identity_checks}")
    if not trades.empty:
        missing_audit_fields = set(MAX_W_TRADE_AUDIT_FIELDS).difference(trades.columns)
        if missing_audit_fields:
            raise ValueError(
                "V4.4 trades lack max-W audit fields: "
                f"{sorted(missing_audit_fields)}"
            )
        trade_identity_checks = {
            "trade_audit_schema_version": pd.to_numeric(
                trades["trade_audit_schema_version"], errors="raise"
            ).eq(COMBINED_TRADE_AUDIT_SCHEMA_VERSION).all(),
            "trade_audit_schema_id": trades["trade_audit_schema_id"]
            .astype(str)
            .eq(COMBINED_TRADE_AUDIT_SCHEMA_ID)
            .all(),
            "rebound_baseline_policy_id": trades["rebound_baseline_policy_id"]
            .astype(str)
            .eq(REBOUND_BASELINE_POLICY_ID)
            .all(),
        }
        if not all(bool(value) for value in trade_identity_checks.values()):
            raise ValueError(
                f"V4.4 trade max-W identity mismatch: {trade_identity_checks}"
            )

    if source_frame is None:
        source = pd.read_csv(source_path, usecols=list(SOURCE_COLUMNS))
        source = source.loc[
            source["datetime"].astype(str).le(str(stage_manifest["train_end"]))
        ].reset_index(drop=True)
    else:
        source = source_frame
    if source.empty:
        raise ValueError("V4.4 trade review source is empty through train_end")
    if not trades.empty:
        maximum_trade_index = max(
            int(trades["entry_index"].max()), int(trades["exit_index"].max())
        )
        if maximum_trade_index >= len(source):
            raise ValueError("V4.4 trade review source does not cover every saved trade")
    research_positions = np.flatnonzero(
        source["datetime"].astype(str).ge(str(stage_manifest["train_start"])).to_numpy()
    )
    if not len(research_positions):
        raise ValueError("V4.4 training start is outside the review source")
    research_start_index = int(research_positions[0])

    combo_rows = [
        _native_combo_record(row, stage_manifest, completion_manifest)
        for _, row in summary.iterrows()
    ]
    combo_by_id = {str(row["combo_id"]): row for row in combo_rows}
    worker_count = int(workers)
    if worker_count < 1 or worker_count > 32:
        raise ValueError("trade-review workers must be between 1 and 32")
    chunk_directory = output / "v3_native_trades_js"
    chunk_files: list[Path] = []
    filename_by_combo: dict[str, str] = {}
    verified_entry_count = 0
    waited_entry_count = 0
    maximum_wait = 0
    empty_positions = np.asarray([], dtype=np.intp)
    trade_positions_by_combo = (
        {
            str(combo_id): positions
            for combo_id, positions in trades.groupby(
                trades["combo_id"].astype(str), sort=False
            ).indices.items()
        }
        if not trades.empty
        else {}
    )

    reusable_chunk_directory: Path | None = None
    if reuse_chunk_directory is not None:
        reusable_chunk_directory = reuse_chunk_directory.resolve()
    if reuse_trade_review is not None:
        if reusable_chunk_directory is not None:
            raise ValueError("choose one incremental chunk source")
        reuse_root = reuse_trade_review.resolve()
        reuse_manifest_path = reuse_root / "trade_review_manifest.json"
        reuse_manifest = json.loads(reuse_manifest_path.read_text(encoding="utf-8"))
        if (
            reuse_manifest.get("status") != "complete"
            or str(reuse_manifest.get("source_sha256")) != source_sha256
            or str(reuse_manifest.get("strategy_id"))
            != str(completion_manifest["strategy_id"])
        ):
            raise ValueError("incremental trade-review parent has incompatible source or strategy identity")
        reusable_chunk_directory = reuse_root / "v3_native_trades_js"

    def chunk_filename(combo_id: str) -> str:
        return f"c_{hashlib.sha256(combo_id.encode('utf-8')).hexdigest()[:16]}.js"

    def build_combo_chunk(
        combo_id: str,
    ) -> tuple[str, str, Path, int, int, int]:
        combo_trades = trades.iloc[
            trade_positions_by_combo.get(combo_id, empty_positions)
        ]
        combo_trades = combo_trades.sort_values(
            ["entry_index", "exit_index"], kind="mergesort"
        ) if not combo_trades.empty else combo_trades
        native_trades: list[dict[str, Any]] = []
        previous_exit_index: int | None = None
        combo_verified_count = 0
        combo_waited_count = 0
        combo_maximum_wait = 0
        for _, trade in combo_trades.iterrows():
            native_trades.append(
                _native_trade_record(
                    trade,
                    combo_by_id[combo_id],
                    source,
                    research_start_index=research_start_index,
                    previous_exit_index=previous_exit_index,
                )
            )
            previous_exit_index = int(trade["exit_index"])
            combo_verified_count += 1
            combo_waited_count += int(int(trade["entry_wait_bar_count"]) > 0)
            combo_maximum_wait = max(
                combo_maximum_wait, int(trade["entry_wait_bar_count"])
            )
        filename = chunk_filename(combo_id)
        chunk_path = chunk_directory / filename
        _atomic_text(
            chunk_path,
            "window.NATIVE_COMBO="
            + _compact_json(combo_by_id[combo_id])
            + ";window.NATIVE_TRADES="
            + _compact_json(native_trades)
            + ";\n",
        )
        return (
            combo_id,
            filename,
            chunk_path,
            combo_verified_count,
            combo_waited_count,
            combo_maximum_wait,
        )

    combo_ids = list(combo_by_id)
    reused_chunk_count = 0
    generated_combo_ids: list[str] = []
    catalog_only_reuse = resolved_trade_count != len(trades)
    for combo_id in combo_ids:
        filename = chunk_filename(combo_id)
        source_chunk = (
            reusable_chunk_directory / filename
            if reusable_chunk_directory is not None
            else None
        )
        if source_chunk is None or not source_chunk.is_file():
            generated_combo_ids.append(combo_id)
            continue
        chunk_path = chunk_directory / filename
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        if source_chunk.resolve() == chunk_path.resolve():
            pass
        elif chunk_path.exists():
            if _sha256(chunk_path) != _sha256(source_chunk):
                raise ValueError(f"existing incremental chunk differs from parent: {chunk_path}")
        else:
            os.link(source_chunk, chunk_path)
        positions = trade_positions_by_combo.get(combo_id, empty_positions)
        combo_trades = trades.iloc[positions]
        filename_by_combo[combo_id] = filename
        chunk_files.append(chunk_path)
        if not catalog_only_reuse:
            verified_entry_count += int(len(combo_trades))
        waited = pd.to_numeric(
            combo_trades.get("entry_wait_bar_count", pd.Series(dtype=int)),
            errors="coerce",
        ).fillna(0).astype(int)
        if not catalog_only_reuse:
            waited_entry_count += int(waited.gt(0).sum())
            maximum_wait = max(maximum_wait, int(waited.max()) if len(waited) else 0)
        reused_chunk_count += 1
    generation_workers = min(worker_count, max(1, len(generated_combo_ids)))
    if generation_workers > 1 and generated_combo_ids:
        process_tasks = (
            (
                combo_id,
                combo_by_id[combo_id],
                trades.iloc[
                    trade_positions_by_combo.get(combo_id, empty_positions)
                ].copy(),
                str(chunk_directory / chunk_filename(combo_id)),
            )
            for combo_id in generated_combo_ids
        )
        with ProcessPoolExecutor(
            max_workers=generation_workers,
            initializer=_initialize_chunk_process,
            initargs=(
                str(source_path),
                str(stage_manifest["train_start"]),
                str(stage_manifest["train_end"]),
            ),
        ) as executor:
            chunk_results = list(executor.map(_build_combo_chunk_process, process_tasks))
    else:
        with ThreadPoolExecutor(
            max_workers=generation_workers,
            thread_name_prefix="v4_4_trade_html",
        ) as executor:
            chunk_results = list(executor.map(build_combo_chunk, generated_combo_ids))
    for (
        combo_id,
        filename,
        chunk_path,
        combo_verified_count,
        combo_waited_count,
        combo_maximum_wait,
    ) in chunk_results:
        filename_by_combo[combo_id] = filename
        chunk_files.append(chunk_path)
        verified_entry_count += combo_verified_count
        waited_entry_count += combo_waited_count
        maximum_wait = max(maximum_wait, combo_maximum_wait)

    if catalog_only_reuse:
        if generated_combo_ids:
            raise ValueError("catalog-only incremental review is missing reusable chunks")
        stats = reused_trade_stats or {}
        verified_entry_count = resolved_trade_count
        waited_entry_count = int(stats.get("waited_entry_count", 0))
        maximum_wait = int(stats.get("maximum_entry_wait_bars", 0))
    if verified_entry_count != resolved_trade_count:
        raise ValueError("not every V4.4 actual entry bar passed source validation")
    if chunks_only:
        return {
            "chunks": chunk_files,
            "verified_real_entry_bar_count": verified_entry_count,
            "waited_entry_trade_count": waited_entry_count,
            "maximum_observed_entry_wait_bars": maximum_wait,
            "reused_chunk_count": reused_chunk_count,
            "generated_chunk_count": len(generated_combo_ids),
        }
    published_reused_chunk_count = (
        int(publication_chunk_counts["reused_chunk_count"])
        if publication_chunk_counts is not None
        else reused_chunk_count
    )
    published_generated_chunk_count = (
        int(publication_chunk_counts["generated_chunk_count"])
        if publication_chunk_counts is not None
        else len(generated_combo_ids)
    )
    baseline_filter_id = str(
        trades.iloc[0].get("baseline_filter_id", "unconfirmed")
        if not trades.empty
        else stage_manifest.get("baseline_filter_id", "unconfirmed")
    )
    prepared_identity = str(
        stage_manifest.get("prepared_identity")
        or stage_manifest.get("data_preparation_manifest_sha256")
        or "unconfirmed"
    )
    identity = {
        "version_label": str(stage_manifest.get("version_label", "V4.4")),
        "strategy_id": str(completion_manifest["strategy_id"]),
        "result_semantics_id": str(stage_manifest.get("result_semantics_id", "")),
        "campaign_id": str(completion_manifest["campaign_id"]),
        "stage_id": str(completion_manifest["stage_id"]),
        "exit_mode": str(completion_manifest.get("exit_mode", "")),
        "baseline_sampling_policy": (
            policies[0] if len(policies) == 1 else "multiple"
        ),
        "baseline_sampling_policies": policies,
        "strategy_ids_by_baseline_sampling_policy": {
            policy: _policy_identity(
                completion_manifest, policy, "strategy_id"
            )
            for policy in policies
        },
        "result_semantics_ids_by_baseline_sampling_policy": {
            policy: _policy_identity(
                stage_manifest, policy, "result_semantics_id"
            )
            for policy in policies
        },
        "baseline_filter_id": baseline_filter_id,
        "prepared_identity": prepared_identity,
        "source_sha256": source_sha256,
        "analysis_identity": analysis_identity,
        "raw_output_schema_version": OUTPUT_SCHEMA_VERSION,
        "plan_fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "parameter_acceptance": "none",
    }
    catalog_rows = []
    for combo in combo_rows:
        combo_id = str(combo["combo_id"])
        catalog_rows.append({
            "catalog_key": f"{identity['campaign_id']}::{combo_id}",
            "research_contract_id": identity["campaign_id"],
            "research_contract_label": research_contract_label,
            "strategy_contract_id": combo["strategy_id"],
            "result_semantics_id": combo["result_semantics_id"],
            "evaluation_mode": "full_training",
            "train_start": stage_manifest.get("train_start"),
            "train_end": stage_manifest.get("train_end"),
            "test_start": stage_manifest.get("train_end"),
            "test_end": stage_manifest.get("train_end"),
            "process_payload_href": "process_payload.js",
            "trade_js_base": "v3_native_trades_js",
            "trade_js_file": filename_by_combo[combo_id],
            **{
                key: combo.get(key)
                for key in (
                    "combo_id", "baseline_sampling_policy",
                    "baseline_sampling_policy_label", "period", "entry_window_multiplier",
                    "entry_window_bars", "baseline_history_multiplier",
                    "baseline_history_bars", "baseline_window_multiplier",
                    "baseline_sample_count_target", "k_drop", "abs_floor",
                    "daily_floor_mult", "exit_wait_bars",
                    "rebound_tr_window_bars", "rebound_multiplier",
                    "reverse_up_window_multiplier", "reverse_up_limit_multiplier",
                    "entry_baseline_method", "entry_baseline_method_label",
                    "entry_baseline_anchor_policy", "baseline_tr_atom_policy",
                    "baseline_history_collection_policy", "entry_confirmation_policy",
                    "entry_fill_policy", "entry_signal_freshness_policy",
                    "entry_signal_high_tie_policy", "entry_signal_interval_start_policy",
                    "entry_signal_reset_policy", "entry_signal_state_version",
                    "execution_bar_seconds", "volatility_tr_bar_seconds", "exit_mode",
                    "speed_exit_enabled", "rebound_exit_enabled", "rebound_baseline_mode",
                    "strategy_major_version", "parent_engine_sha256", "train_avg_trade",
                    "train_cost_adjusted_return", "train_cost_adjusted_avg_trade",
                    "ranking_cost_model_id", "ranking_round_trip_cost_bps",
                    "ranking_return_basis", "default_display_return_basis",
                    "raw_output_schema_version", "plan_fingerprint_schema_version",
                    "trade_audit_schema_version", "trade_audit_schema_id",
                    "rebound_baseline_policy_id", "rebound_baseline_update_rule",
                    "train_max_drawdown", "train_trade_count", "all_trade_count",
                    "original_html_base",
                )
            },
        })
    catalog_data = {
        "schema_version": "v4_4_exact_historical_v4_trade_template_catalog_v2",
        "strategy_contract": identity["strategy_id"],
        "strategy_contract_id": identity["strategy_id"],
        "baseline_sampling_policies": policies,
        "strategy_contract_ids_by_baseline_sampling_policy": identity[
            "strategy_ids_by_baseline_sampling_policy"
        ],
        "research_samples": [{
            "research_contract_id": identity["campaign_id"],
            "label": research_contract_label,
            "train_start": stage_manifest.get("train_start"),
            "train_end": stage_manifest.get("train_end"),
        }],
        "source_result_count": 1,
        "combo_count": len(catalog_rows),
        "rows": catalog_rows,
    }
    index_path = output / "index.html"
    process_path = output / "process_payload.js"
    catalog_path = output / "all_results_catalog.js"
    resource_audit_path = output / "resource_audit.json"
    plotly_path = output.parent / "assets" / "plotly.min.js"
    if not TRADE_PLOTLY_SOURCE.is_file() or _sha256(TRADE_PLOTLY_SOURCE) != TRADE_PLOTLY_SOURCE_SHA256:
        raise ValueError("historical V4 Plotly dependency is missing or changed")
    _atomic_text(
        index_path,
        _historical_trade_html(
            main_href,
            default_start_date=default_start_date,
            instrument_label=instrument_label,
            peer_review_href=peer_review_href,
            peer_review_label=peer_review_label,
            peer_research_contract_id=peer_research_contract_id,
        ),
    )
    _atomic_text(
        process_path,
        _json_script(
            "PROCESS_PAYLOAD",
            {
                "features": _native_process_features(
                    source, combo_rows, stage_manifest, filter_events
                )
            },
        ),
    )
    _atomic_text(
        catalog_path,
        _json_script("ALL_RESULTS_TRADE_EXPLAIN_CATALOG", catalog_data),
    )
    _atomic_bytes(plotly_path, TRADE_PLOTLY_SOURCE.read_bytes())
    resource_audit = {
        "schema_version": 1,
        "row_count": len(catalog_rows),
        "trade_count": resolved_trade_count,
        "chunk_count": len(chunk_files),
        "process_payload_count": 1,
        "chunk_trade_count": resolved_trade_count,
        "loading_contract": (
            "exact supplied historical V4 UI; source OHLC at startup; one "
            "NATIVE_TRADES chunk per selected V4.4 combo"
        ),
        "template_source": str(TRADE_DESIGN_SOURCE),
        "template_source_sha256": TRADE_DESIGN_SOURCE_SHA256,
        "peer_review": (
            {
                "href": peer_review_href,
                "label": peer_review_label,
                "research_contract_id": peer_research_contract_id,
            }
            if peer_review_href and peer_review_label
            else None
        ),
        "plotly_source_sha256": TRADE_PLOTLY_SOURCE_SHA256,
        "adapter_contains_html": False,
        "feature_series_policy": (
            "full source OHLC plus exact saved per-trade audit fields; aggregate "
            "drop/baseline arrays are omitted because they are not stored stage evidence"
        ),
        "filter_overlay": {
            "id": FILTER_OVERLAY_ID,
            "event_count": len(filter_events),
            "scope": FILTER_OVERLAY_SCOPE,
            "palette_source": "V4.4 low-activity audit HTML",
        },
        "baseline_sampling_policies": policies,
        "baseline_sampling_policy_contracts": {
            policy: _baseline_sampling_contract(policy) for policy in policies
        },
        "max_w_audit_contract": {
            "raw_output_schema_version": OUTPUT_SCHEMA_VERSION,
            "plan_fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
            "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
            "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
            "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
            "retained_trade_fields": list(MAX_W_TRADE_AUDIT_FIELDS),
            "historical_template_alias": (
                "rebound_baseline_active_low_* identifies the max-source window "
                "end in the generated historical-template payload; "
                "rebound_active_low_* remains the threshold anchor"
            ),
        },
    }
    _atomic_text(
        resource_audit_path,
        json.dumps(resource_audit, ensure_ascii=False, indent=2) + "\n",
    )

    output_files = [index_path, process_path, catalog_path, resource_audit_path, *chunk_files]
    manifest = {
        "schema_version": 5,
        "status": "complete",
        "evidence_role": "v4_4_closed_stage_native_trade_review",
        **identity,
        "generator": _artifact(Path(__file__)),
        "template_source": _artifact(TRADE_DESIGN_SOURCE),
        "plotly_source": _artifact(TRADE_PLOTLY_SOURCE),
        "plotly_output": _artifact(plotly_path),
        "source_market_data": _artifact(source_path),
        "baseline_filter_marker": _artifact(filter_marker_path),
        "baseline_filter_events": _artifact(filter_events_path),
        "routes": {
            "main": main_href,
            "manifest": manifest_href,
            "peer_review": (
                {
                    "href": peer_review_href,
                    "label": peer_review_label,
                    "research_contract_id": peer_research_contract_id,
                }
                if peer_review_href and peer_review_label
                else None
            ),
            "query_contract": (
                "index.html?combo_id=<V4.4 combo id>&research_contract_id="
                f"{identity['campaign_id']}&reason=<entry|exit>"
            ),
        },
        "closure": {
            "coordinate_count": len(combo_rows),
            "trade_count": resolved_trade_count,
            "trade_chunk_count": len(chunk_files),
            "generation_worker_count": min(worker_count, max(1, len(combo_ids))),
            "verified_real_entry_bar_count": verified_entry_count,
            "waited_entry_trade_count": waited_entry_count,
            "maximum_observed_entry_wait_bars": maximum_wait,
            "historical_v4_trade_template_reused": True,
            "historical_v4_html_css_javascript_reused": True,
            "historical_v4_plotly_candlestick_reused": True,
            "adapter_shell_removed": True,
            "source_ohlc_bar_count": len(source),
            "filter_overlay_event_count": len(filter_events),
            "filter_overlay_matches_audit_palette": True,
            "filter_overlay_preserves_full_ohlc": True,
            "one_combo_chunk_loaded_on_selection": True,
            "max_w_audit_fields_retained": True,
            "closed_bar_policy_id": REBOUND_BASELINE_POLICY_ID,
            "incremental_publication": {
                "parent_trade_review": (
                    str(reuse_trade_review.resolve())
                    if reuse_trade_review is not None
                    else None
                ),
                "reused_chunk_count": published_reused_chunk_count,
                "generated_chunk_count": published_generated_chunk_count,
                "reuse_mechanism": "same-volume hard link",
            },
        },
        "outputs": [_artifact(path, root=output) for path in output_files],
    }
    manifest_path = output / "trade_review_manifest.json"
    _atomic_text(
        manifest_path,
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )
    return {
        "index": index_path,
        "catalog": catalog_path,
        "process_payload": process_path,
        "plotly": plotly_path,
        "manifest": manifest_path,
        "chunks": chunk_files,
        "verified_real_entry_bar_count": verified_entry_count,
        "waited_entry_trade_count": waited_entry_count,
        "maximum_observed_entry_wait_bars": maximum_wait,
        "reused_chunk_count": published_reused_chunk_count,
        "generated_chunk_count": published_generated_chunk_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the closed V4.4 home and lazy trade-analysis delivery."
    )
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--stage", required=True)
    parser.add_argument("--validation-stage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(
        Path(args.source_manifest),
        Path(args.stage),
        Path(args.validation_stage),
        Path(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
