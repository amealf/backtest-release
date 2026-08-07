"""Isolated V4.4 rolling-TR calculated-price research engine.

The preparation layer attaches causal low-activity lifecycle fields. Each
coordinate selects its baseline sampling and entry-gating policy while
retaining the rolling-TR-sum formula.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research_variants.short_momentum_net_drop_rebound_v4_4.data_preparation.prepare_dataset import (  # noqa: E402
    PIPELINE_VERSION as DATA_PREPARATION_PIPELINE_VERSION,
    prepare_dataset,
    sha256_file as preparation_sha256,
)
from research_variants.short_momentum_net_drop_rebound_v4_4.data_preparation.low_activity import (  # noqa: E402
    LOW_ACTIVITY_STATE_CONFIRMED,
    LOW_ACTIVITY_STATE_NORMAL,
    LOW_ACTIVITY_STATE_PENDING,
)


VERSION_LABEL = "V4.4"
BASELINE_SAMPLING_ALL_WINDOW = "all_window"
BASELINE_SAMPLING_EXCLUDE_MARKED = "exclude_marked"
BASELINE_SAMPLING_CONFIRMED_LOW_ACTIVITY_GATE = "confirmed_low_activity_gate"
BASELINE_SAMPLING_POLICIES = (
    BASELINE_SAMPLING_ALL_WINDOW,
    BASELINE_SAMPLING_EXCLUDE_MARKED,
    BASELINE_SAMPLING_CONFIRMED_LOW_ACTIVITY_GATE,
)
DEFAULT_BASELINE_SAMPLING_POLICY = BASELINE_SAMPLING_CONFIRMED_LOW_ACTIVITY_GATE
REBOUND_BASELINE_POLICY_ID = "max_completed_w_h_bounded_open_to_low_v2"
BASELINE_FILTER_IDS = {
    BASELINE_SAMPLING_ALL_WINDOW: "all_window_market_no_baseline_exclusion_v4_4",
    BASELINE_SAMPLING_EXCLUDE_MARKED: "causal_low_activity_lifecycle_exclusion_v4_4",
    BASELINE_SAMPLING_CONFIRMED_LOW_ACTIVITY_GATE: "confirmed_low_activity_retroactive_baseline_exclusion_and_entry_gate_v4_4",
}


def baseline_filter_id(policy: str) -> str:
    if policy not in BASELINE_FILTER_IDS:
        raise ValueError(f"unsupported V4.4 baseline sampling policy: {policy}")
    return BASELINE_FILTER_IDS[policy]


def strategy_id(policy: str, *, combined_exit: bool) -> str:
    baseline_filter_id(policy)
    suffix = "_combined_exit" if combined_exit else ""
    return (
        f"net_drop_rebound_entry_peak_anchor_v4_4_{policy}_rolling_tr_sum_"
        "calculated_execution_price_h_bounded_max_completed_w_drop_rebound_"
        "strict_low_close_fill_sample_end_close_"
        "pending_exit_next_real_trade_open"
        f"{suffix}"
    )


BASELINE_FILTER_ID = baseline_filter_id(DEFAULT_BASELINE_SAMPLING_POLICY)
STRATEGY_ID = strategy_id(DEFAULT_BASELINE_SAMPLING_POLICY, combined_exit=False)
COMBINED_STRATEGY_ID = strategy_id(DEFAULT_BASELINE_SAMPLING_POLICY, combined_exit=True)
EXIT_MODE_REBOUND_ONLY = "rebound_only"
EXIT_MODE_COMBINED = "combined"
DOWNSIDE_SPEED_EXIT_REASON = "downside_speed_below_threshold"
ENTRY_METHOD_ROLLING = "rolling_tr_sum"
METHODS = (ENTRY_METHOD_ROLLING,)
ENTRY_SIGNAL_POLICY_ID = "positive_baseline_drop_threshold_with_inclusive_boundary_v1"
ENTRY_FILL_NEXT_BAR_OPEN = "next_bar_open"
ENTRY_FILL_CALCULATED_THRESHOLD = "calculated_threshold"
ENTRY_FILL_MODES = (ENTRY_FILL_CALCULATED_THRESHOLD,)
ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE = "wait_next_real_trade"
ENTRY_EXECUTION_REJECT_SYNTHETIC_FILL = "reject_synthetic_fill"
ENTRY_EXECUTION_POLICIES = (
    ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    ENTRY_EXECUTION_REJECT_SYNTHETIC_FILL,
)
MAX_REAL_TRADE_WAIT_BARS = 120
TRADE_AUDIT_SCHEMA_VERSION = 3
TRADE_AUDIT_SCHEMA_ID = "v4_4_h_bounded_w_causal_baseline_execution_audit_v3"
COMBINED_TRADE_AUDIT_SCHEMA_VERSION = 3
COMBINED_TRADE_AUDIT_SCHEMA_ID = "v4_4_h_bounded_w_causal_baseline_combined_exit_audit_v3"
RUNTIME_INPUTS_ROOT = REPOSITORY_ROOT / "runtime_inputs"
SOURCE_DEFAULT = RUNTIME_INPUTS_ROOT / "market_data" / "k200_clean_15s_session_filled.csv"
EXTERNAL_INPUTS_ROOT = RUNTIME_INPUTS_ROOT / "legacy"
QUIET_ACTIVITY_ATOMS_DEFAULT = EXTERNAL_INPUTS_ROOT / "canonical_volume_atoms.csv"
QUIET_ACTIVITY_ATOMS_SHA256 = "1d30c0f6019506acfa5bf6a6127c8f8acb46a25c4028990b0a5d230ed0befbf8"
SOURCE_SHA256 = "ec04981dcf7fa74a7c8266f7d678913ba6ad5dc8c9463194882ecdbd0d4121a4"
DATA_PREPARATION_MANIFEST_DEFAULT = (
    RUNTIME_INPUTS_ROOT / "data_preparation" / "data_preparation_manifest.json"
)
RESULT_DEFAULT = REPOSITORY_ROOT / "results" / "validation" / "initial_validation"
EVENTS_DEFAULT = EXTERNAL_INPUTS_ROOT / "selected_events.csv"
TRAIN_START = "2026-05-26 00:00:00"
TRAIN_END = "2026-07-08 23:52:00"
FUTURE_TEST_CANDIDATE_START = "2026-07-23 02:44:00"


@dataclass(frozen=True)
class Combo:
    method: str
    e: int
    bh: int
    trw: int
    k: float
    w: int
    m: float
    entry_fill_mode: str = ENTRY_FILL_CALCULATED_THRESHOLD
    entry_execution_policy: str = ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE
    entry_slippage: float = 0.0
    speed_window_bars: int = 0
    baseline_sampling_policy: str = DEFAULT_BASELINE_SAMPLING_POLICY

    def __post_init__(self) -> None:
        if self.method != ENTRY_METHOD_ROLLING:
            raise ValueError("V4.4 current method requires rolling_tr_sum")
        if self.baseline_sampling_policy not in BASELINE_SAMPLING_POLICIES:
            raise ValueError(
                "unsupported V4.4 baseline sampling policy: "
                f"{self.baseline_sampling_policy}"
            )
        if self.entry_fill_mode != ENTRY_FILL_CALCULATED_THRESHOLD:
            raise ValueError(
                "V4.4 requires calculated_threshold entry; next_bar_open belongs to V4.1"
            )
        if self.entry_execution_policy not in ENTRY_EXECUTION_POLICIES:
            raise ValueError(
                f"unsupported V4.4 entry execution policy: {self.entry_execution_policy}"
            )
        if not math.isfinite(float(self.entry_slippage)) or float(self.entry_slippage) < 0:
            raise ValueError("entry_slippage must be finite and nonnegative")
        if int(self.speed_window_bars) != self.speed_window_bars or int(self.speed_window_bars) < 0:
            raise ValueError("speed_window_bars must be a nonnegative integer")

    @property
    def speed_exit_enabled(self) -> bool:
        return int(self.speed_window_bars) > 0

    @property
    def rebound_exit_enabled(self) -> bool:
        return True

    @property
    def exit_mode(self) -> str:
        return EXIT_MODE_COMBINED if self.speed_exit_enabled else EXIT_MODE_REBOUND_ONLY

    @property
    def strategy_id(self) -> str:
        return strategy_id(
            self.baseline_sampling_policy,
            combined_exit=self.speed_exit_enabled,
        )

    @property
    def baseline_filter_id(self) -> str:
        return baseline_filter_id(self.baseline_sampling_policy)

    @property
    def combo_id(self) -> str:
        token = lambda value: format(float(value), ".12g").replace(".", "p")
        coordinate = (
            self.method, self.e, self.bh, self.trw, self.k, self.w, self.m,
            self.entry_fill_mode, self.entry_execution_policy, self.entry_slippage,
            int(self.speed_window_bars), self.baseline_sampling_policy,
        )
        identity = [
            VERSION_LABEL,
            self.strategy_id,
            self.baseline_filter_id,
            *map(str, coordinate),
        ]
        digest = hashlib.sha256("|".join(identity).encode()).hexdigest()[:10]
        speed_token = (
            f"_sx1_s{int(self.speed_window_bars)}_rx1"
            if self.speed_exit_enabled
            else ""
        )
        return (
            f"v4_4_{self.method}_bp{self.baseline_sampling_policy}_fill{self.entry_fill_mode}"
            f"_exec{self.entry_execution_policy}_slip{token(self.entry_slippage)}"
            f"{speed_token}"
            f"_e{self.e}_bh{self.bh}_trw{self.trw}_k{token(self.k)}"
            f"_w{self.w}_m{token(self.m)}_{digest}"
        )


def _finite(value: float) -> bool:
    return bool(math.isfinite(float(value)))


def entry_signal_qualifies(baseline: float, drop: float, k: float) -> bool:
    """Require positive signal economics while retaining inclusive equality."""
    threshold = float(k) * float(baseline)
    return bool(
        _finite(baseline)
        and _finite(drop)
        and _finite(threshold)
        and baseline > 0.0
        and drop > 0.0
        and threshold > 0.0
        and drop >= threshold
    )


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna(False).map(
        lambda value: value is True or str(value).strip().lower() in {"1", "true", "yes"}
    ).astype(bool)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_filter_bundle(path: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    mask = pd.read_csv(
        path,
        usecols=["datetime", "effective_filter_mask", "causal_active"],
    )
    mask["datetime"] = pd.to_datetime(mask["datetime"], errors="raise")
    if len(mask) != len(frame) or not mask["datetime"].equals(frame["datetime"]):
        raise ValueError("legacy quiet-activity atoms do not align one-to-one with the 15-second source")
    mask = mask.rename(columns={"effective_filter_mask": "baseline_excluded"})
    mask["filter_reason_codes"] = np.where(mask["baseline_excluded"], "universal_low_volume", "")
    mask["filter_event_ids"] = ""
    events: list[dict[str, Any]] = []
    excluded = mask["baseline_excluded"].astype(bool)
    run_id = excluded.ne(excluded.shift(fill_value=False)).cumsum()
    for number, positions in enumerate(mask.loc[excluded].groupby(run_id[excluded], sort=True).groups.values(), start=1):
        indexes = np.asarray(list(positions), dtype=int)
        active = indexes[mask.loc[indexes, "causal_active"].to_numpy(bool)]
        if not len(active):
            raise ValueError("legacy effective interval lacks a causal confirmation atom")
        event_id = f"legacy_universal_low_volume_{number:02d}"
        mask.loc[indexes, "filter_event_ids"] = event_id
        events.append(
            {
                "event_id": event_id,
                "event_type": "universal_low_volume",
                "start": str(mask.loc[indexes[0], "datetime"]),
                "confirmation_time": str(mask.loc[active[0], "datetime"]),
                "end": str(mask.loc[indexes[-1], "datetime"]),
                "apply_to_baseline": True,
            }
        )
    return mask, events


def _prepared_filter_bundle(
    manifest_path: Path,
    source: Path,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("data-preparation manifest is not complete")
    if int(manifest.get("schema_version", 0)) < 5:
        raise ValueError("V4.4 causal baseline policies require a schema-v5 preparation manifest")
    if manifest.get("pipeline_version") != DATA_PREPARATION_PIPELINE_VERSION:
        raise ValueError("data-preparation pipeline identity does not match V4.4")
    if not str(manifest.get("prepared_identity", "")).startswith(
        "v4_4_confirmed_low_activity_gate_"
    ):
        raise ValueError("prepared data identity does not belong to V4.4")
    sampling_contract = manifest.get("rule_contract", {}).get(
        "baseline_sampling_contract", {}
    )
    if sampling_contract.get("marker_generation_is_policy_neutral") is not True:
        raise ValueError("prepared data marker is not policy neutral")
    if sampling_contract.get("default_baseline_sampling_policy") != DEFAULT_BASELINE_SAMPLING_POLICY:
        raise ValueError("prepared data default baseline policy differs from V4.4")
    if set(sampling_contract.get("supported_baseline_sampling_policies", {})) != set(
        BASELINE_SAMPLING_POLICIES
    ):
        raise ValueError("prepared data baseline policy set differs from V4.4")
    extreme = manifest.get("extreme_cleaning", {})
    legacy_current_source = (
        extreme.get("status") == "legacy_preexisting_source"
        and extreme.get("passed") is False
        and _sha256(source).lower() == SOURCE_SHA256
    )
    passed_current_method = (
        extreme.get("status") == "passed"
        and extreme.get("passed") is True
        and extreme.get("method") == "same_bar_immediate_tick_recovery"
    )
    if not (legacy_current_source or passed_current_method):
        raise ValueError("V4 requires the current-source legacy record or a passed extreme-cleaning audit")
    if str(manifest.get("source_sha256", "")).lower() != _sha256(source).lower():
        raise ValueError("data-preparation source hash does not match the V4 source")
    atoms_entry = manifest.get("artifacts", {}).get("filter_atoms", {})
    events_entry = manifest.get("artifacts", {}).get("filter_events", {})

    def resolve_artifact(value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else (manifest_path.parent / path).resolve()

    atoms_path = resolve_artifact(atoms_entry.get("path", ""))
    events_path = resolve_artifact(events_entry.get("path", ""))
    for path, expected in (
        (atoms_path, atoms_entry.get("sha256")),
        (events_path, events_entry.get("sha256")),
    ):
        if not path.is_file() or not expected or _sha256(path).lower() != str(expected).lower():
            raise ValueError(f"data-preparation artifact failed hash validation: {path}")
    mask = pd.read_csv(atoms_path, low_memory=False)
    mask["datetime"] = pd.to_datetime(mask["datetime"], errors="raise")
    if len(mask) != len(frame) or not mask["datetime"].equals(frame["datetime"]):
        raise ValueError("prepared filter atoms do not align one-to-one with the 15-second source")
    required_audit = {
        "low_activity_state", "pending_buffer_start", "pending_buffer_count",
        "buffer_reinserted", "buffer_confirmed_excluded",
        "recovery_confirmation_time", "baseline_available_from",
        "low_activity_confirmation_time", "baseline_excluded_from",
        "confirmed_low_activity_active",
        "eligible_if_excluding_marked",
    }
    missing_audit = required_audit.difference(mask.columns)
    if missing_audit:
        raise ValueError(f"V4.4 filter atoms lack lifecycle audit fields: {sorted(missing_audit)}")
    events_payload = json.loads(events_path.read_text(encoding="utf-8"))
    events = [event for event in events_payload.get("events", []) if bool(event.get("apply_to_baseline", False))]
    return mask, events, manifest


def load_bars(
    source: Path,
    filter_artifact: Path | None = DATA_PREPARATION_MANIFEST_DEFAULT,
) -> pd.DataFrame:
    """Read 15-second OHLC and attach immutable audit/chart markers.

    Every row remains a physical observation for signals, fills, exits, time,
    and charts. The coordinate policy decides whether the neutral marker
    participates in BH/TRW sample eligibility.
    """
    frame = pd.read_csv(source)
    required = {
        "datetime", "open", "high", "low", "close", "volume", "trade_count",
        "is_synthetic_empty_bar",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"source lacks required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    duplicate_times = frame.loc[
        frame["datetime"].duplicated(keep=False), "datetime"
    ]
    if not duplicate_times.empty:
        examples = duplicate_times.astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            "source contains duplicate datetime rows; V4.4 requires one row per 15-second "
            f"timestamp; examples={examples}"
        )
    frame = frame.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    for name in ("open", "high", "low", "close", "volume", "trade_count"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    frame["is_synthetic_empty_bar"] = _coerce_bool(frame["is_synthetic_empty_bar"])
    delta = frame["datetime"].diff().dt.total_seconds()
    frame["continuous"] = delta.eq(15.0)
    frame.loc[0, "continuous"] = False
    frame["continuous_segment_id"] = (~frame["continuous"]).cumsum().astype(int)
    frame["continuous_run"] = frame["continuous"].astype(int).groupby((~frame["continuous"]).cumsum()).cumsum() + 1
    previous_close = frame["close"].shift(1).to_numpy(dtype=float)
    tr = np.maximum(frame["high"].to_numpy(float), previous_close) - np.minimum(frame["low"].to_numpy(float), previous_close)
    tr[~frame["continuous"].to_numpy(bool)] = np.nan
    frame["tr15"] = tr
    preparation_manifest: dict[str, Any] | None = None
    if filter_artifact is None:
        mask = pd.DataFrame({
            "datetime": frame["datetime"],
            "baseline_excluded": False,
            "filter_reason_codes": "",
            "filter_event_ids": "",
            "causal_active": False,
            "low_activity_state": LOW_ACTIVITY_STATE_NORMAL,
            "pending_buffer_start": pd.NaT,
            "pending_buffer_count": 0,
            "buffer_reinserted": False,
            "buffer_confirmed_excluded": False,
            "recovery_confirmation_time": pd.NaT,
            "baseline_available_from": frame["datetime"],
            "low_activity_confirmation_time": pd.NaT,
            "baseline_excluded_from": pd.NaT,
            "confirmed_low_activity_active": False,
            "eligible_if_excluding_marked": True,
        })
        events: list[dict[str, Any]] = []
    elif filter_artifact.suffix.lower() == ".json":
        candidate_manifest = json.loads(filter_artifact.read_text(encoding="utf-8"))
        if int(candidate_manifest.get("schema_version", 0)) >= 6:
            preparation_manifest = candidate_manifest
            artifacts = candidate_manifest.get("artifacts") or {}
            if artifacts:
                def generic_artifact(name: str) -> Path:
                    entry = artifacts.get(name, {})
                    artifact_path = Path(str(entry.get("path", "")))
                    artifact_path = artifact_path if artifact_path.is_absolute() else (filter_artifact.parent / artifact_path).resolve()
                    if not artifact_path.is_file() or _sha256(artifact_path) != str(entry.get("sha256", "")):
                        raise ValueError(f"generic preparation artifact failed hash validation: {name}")
                    return artifact_path
                mask = pd.read_csv(generic_artifact("filter_atoms"), low_memory=False)
                mask["datetime"] = pd.to_datetime(mask["datetime"], errors="raise")
                if len(mask) != len(frame) or not mask["datetime"].equals(frame["datetime"]):
                    raise ValueError("generic filter atoms do not align with market data")
                events_payload = json.loads(generic_artifact("filter_events").read_text(encoding="utf-8"))
                events = [dict(item) for item in events_payload.get("events", []) if bool(item.get("apply_to_baseline", False))]
            else:
                mask = pd.DataFrame({
                    "datetime": frame["datetime"], "baseline_excluded": False,
                    "filter_reason_codes": "", "filter_event_ids": "", "causal_active": False,
                    "low_activity_state": LOW_ACTIVITY_STATE_NORMAL, "pending_buffer_start": pd.NaT,
                    "pending_buffer_count": 0, "buffer_reinserted": False,
                    "buffer_confirmed_excluded": False, "recovery_confirmation_time": pd.NaT,
                    "baseline_available_from": frame["datetime"],
                    "low_activity_confirmation_time": pd.NaT,
                    "baseline_excluded_from": pd.NaT,
                    "confirmed_low_activity_active": False,
                    "eligible_if_excluding_marked": True,
                })
                events = []
        else:
            mask, events, preparation_manifest = _prepared_filter_bundle(filter_artifact, source, frame)
    else:
        raise ValueError("V4.4 accepts only its schema-v2 data-preparation manifest")

    excluded = _coerce_bool(mask["baseline_excluded"]).reset_index(drop=True)
    marker_eligible = _coerce_bool(mask["eligible_if_excluding_marked"]).reset_index(drop=True)
    if not marker_eligible.equals(~excluded):
        raise ValueError("prepared baseline eligibility is not the inverse of baseline_excluded")
    causal_active = _coerce_bool(
        mask.get("causal_active", pd.Series(False, index=mask.index))
    ).reset_index(drop=True)
    event_stage = np.zeros(len(frame), dtype=int)
    ordered_events = sorted(events, key=lambda event: (pd.Timestamp(event["confirmation_time"]), str(event["event_id"])))
    mechanism_events = [
        event for event in ordered_events
        if str(event.get("event_type")) != "universal_low_volume"
    ]
    confirmation_indices: list[int] = []
    for stage, event in enumerate(mechanism_events, start=1):
        confirmation_index = int(frame["datetime"].searchsorted(pd.Timestamp(event["confirmation_time"]), side="left"))
        if confirmation_index >= len(frame):
            continue
        confirmation_indices.append(confirmation_index)
        positions = frame.index[frame["datetime"].between(pd.Timestamp(event["start"]), pd.Timestamp(event["end"]))]
        event_stage[np.asarray(list(positions), dtype=int)] = stage
    confirmation_indices_array = np.asarray(confirmation_indices, dtype=int)
    row_indices = np.arange(len(frame), dtype=int)
    filter_stage = np.searchsorted(confirmation_indices_array, row_indices, side="right")
    frame["baseline_excluded"] = excluded
    frame["baseline_filter_reason_codes"] = mask.get("filter_reason_codes", "").fillna("").astype(str).reset_index(drop=True)
    frame["baseline_filter_event_ids"] = mask.get("filter_event_ids", "").fillna("").astype(str).reset_index(drop=True)
    for column in (
        "universal_low_volume_excluded",
        "k200_price_lock_excluded",
        "k200_circuit_breaker_excluded",
    ):
        frame[column] = _coerce_bool(
            mask.get(column, pd.Series(False, index=mask.index))
        ).reset_index(drop=True)
    frame["low_activity_state"] = (
        mask.get("low_activity_state", pd.Series(LOW_ACTIVITY_STATE_NORMAL, index=mask.index))
        .fillna(LOW_ACTIVITY_STATE_NORMAL)
        .astype(str)
        .reset_index(drop=True)
    )
    frame["low_volume_atom"] = _coerce_bool(
        mask.get("low_volume_atom", pd.Series(False, index=mask.index))
    ).reset_index(drop=True)
    frame["volume_threshold"] = pd.to_numeric(
        mask.get("volume_threshold", pd.Series(np.nan, index=mask.index)),
        errors="coerce",
    ).reset_index(drop=True)
    frame["pending_buffer_start"] = pd.to_datetime(
        mask.get("pending_buffer_start", pd.Series(pd.NaT, index=mask.index)),
        errors="coerce",
    ).reset_index(drop=True)
    frame["pending_buffer_count"] = pd.to_numeric(
        mask.get("pending_buffer_count", pd.Series(0, index=mask.index)),
        errors="raise",
    ).fillna(0).astype(int).reset_index(drop=True)
    frame["buffer_reinserted"] = _coerce_bool(
        mask.get("buffer_reinserted", pd.Series(False, index=mask.index))
    ).reset_index(drop=True)
    frame["buffer_confirmed_excluded"] = _coerce_bool(
        mask.get("buffer_confirmed_excluded", pd.Series(False, index=mask.index))
    ).reset_index(drop=True)
    frame["recovery_confirmation_time"] = pd.to_datetime(
        mask.get("recovery_confirmation_time", pd.Series(pd.NaT, index=mask.index)),
        errors="coerce",
    ).reset_index(drop=True)
    frame["baseline_available_from"] = pd.to_datetime(
        mask.get("baseline_available_from", frame["datetime"]),
        errors="coerce",
    ).reset_index(drop=True)
    frame["low_activity_confirmation_time"] = pd.to_datetime(
        mask.get("low_activity_confirmation_time", pd.Series(pd.NaT, index=mask.index)),
        errors="coerce",
    ).reset_index(drop=True)
    frame["baseline_excluded_from"] = pd.to_datetime(
        mask.get("baseline_excluded_from", pd.Series(pd.NaT, index=mask.index)),
        errors="coerce",
    ).reset_index(drop=True)
    frame["confirmed_low_activity_active"] = _coerce_bool(
        mask.get("confirmed_low_activity_active", pd.Series(False, index=mask.index))
    ).reset_index(drop=True)
    if frame.loc[~marker_eligible, "baseline_available_from"].notna().any():
        raise ValueError("confirmed baseline exclusions must never become available")
    reinserted = frame["buffer_reinserted"] & marker_eligible
    if reinserted.any() and not frame.loc[
        reinserted, "baseline_available_from"
    ].astype("datetime64[ns]").equals(
        frame.loc[reinserted, "recovery_confirmation_time"].astype(
            "datetime64[ns]"
        )
    ):
        raise ValueError("reinserted baseline atoms must become available at recovery")

    frame["quiet_activity_excluded"] = excluded
    frame["quiet_activity_causal_active"] = causal_active
    frame["quiet_filter_event_stage"] = event_stage
    frame["quiet_filter_stage"] = filter_stage
    frame["eligible_if_excluding_marked"] = marker_eligible
    frame["baseline_eligible_all_window"] = frame["continuous"] & frame["tr15"].notna()
    frame["baseline_eligible_exclude_marked"] = (
        frame["baseline_eligible_all_window"] & marker_eligible
    )
    frame["source_index"] = np.arange(len(frame), dtype=int)
    frame.attrs["baseline_filter_events"] = ordered_events
    frame.attrs["mechanism_filter_events"] = mechanism_events
    frame.attrs["data_preparation_manifest"] = preparation_manifest
    frame.attrs["filter_artifact"] = str(filter_artifact) if filter_artifact else None
    return frame


def current_sample_scope_guard(
    frame: pd.DataFrame,
    source: Path = SOURCE_DEFAULT,
) -> dict[str, Any]:
    if source.resolve() != SOURCE_DEFAULT.resolve():
        raise ValueError(
            "current V4 contract only permits the audited current source; "
            "an extended source requires a new test research contract"
        )
    actual_hash = _sha256(source).lower()
    if actual_hash != SOURCE_SHA256:
        raise ValueError("current V4 source hash changed")
    start = pd.Timestamp(TRAIN_START)
    end = pd.Timestamp(TRAIN_END)
    selected = frame.loc[frame.datetime.between(start, end, inclusive="both")]
    if selected.empty:
        raise ValueError("current V4 sample is absent from the source")
    future_start = pd.Timestamp(FUTURE_TEST_CANDIDATE_START)
    future_rows_used = int(
        selected.datetime.ge(future_start).sum()
    )
    if future_rows_used:
        raise ValueError("future-test candidate dates entered the current sample")
    return {
        "policy": "fixed current sample; extended 15-second dates require a separate future test research contract",
        "source_path": str(source),
        "source_sha256": actual_hash,
        "source_first_datetime": str(frame.datetime.iloc[0]),
        "source_last_datetime": str(frame.datetime.iloc[-1]),
        "sample_start": TRAIN_START,
        "sample_end": TRAIN_END,
        "sample_bar_count": int(len(selected)),
        "rows_before_sample_excluded": int(frame.datetime.lt(start).sum()),
        "rows_after_sample_excluded": int(frame.datetime.gt(end).sum()),
        "future_test_candidate_start": FUTURE_TEST_CANDIDATE_START,
        "future_test_candidate_rows_used": future_rows_used,
        "passed": True,
    }


def baseline_atom_indices(
    h_index: int,
    bh: int,
    eligible_indices: np.ndarray,
    segment_ids: np.ndarray,
    available_from_indices: np.ndarray | None = None,
    as_of_index: int | None = None,
    excluded_from_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return the last BH atoms at H that were known by the calculation time."""
    if as_of_index is None and (
        available_from_indices is not None or excluded_from_indices is not None
    ):
        raise ValueError(
            "as_of_index is required for causal availability or exclusion"
        )
    if as_of_index is not None and (
        available_from_indices is None and excluded_from_indices is None
    ):
        raise ValueError("as_of_index requires a causal availability or exclusion array")
    endpoint = int(np.searchsorted(eligible_indices, h_index, side="right"))
    if endpoint < 1:
        return np.empty(0, dtype=int)
    target_segment = int(segment_ids[h_index])
    segment_start = int(np.searchsorted(segment_ids, target_segment, side="left"))
    eligible_start = int(np.searchsorted(eligible_indices, segment_start, side="left"))
    candidates = eligible_indices[eligible_start:endpoint]
    if available_from_indices is not None:
        candidates = candidates[
            available_from_indices[candidates] <= int(as_of_index)
        ]
    if excluded_from_indices is not None:
        candidates = candidates[
            excluded_from_indices[candidates] > int(as_of_index)
        ]
    if len(candidates) < bh:
        return np.empty(0, dtype=int)
    return candidates[-bh:]


def entry_baseline(
    tr15: np.ndarray,
    h_index: int,
    combo: Combo,
    eligible_indices: np.ndarray | None = None,
    segment_ids: np.ndarray | None = None,
    available_from_indices: np.ndarray | None = None,
    as_of_index: int | None = None,
    excluded_from_indices: np.ndarray | None = None,
) -> float:
    """Compute the scale from exactly BH policy-eligible TR atoms.

    With no eligibility arrays this helper retains its compact contiguous form
    for formula fixtures.
    """
    if eligible_indices is None and segment_ids is None:
        start = h_index - combo.bh + 1
        if start < 0:
            return math.nan
        values = tr15[start : h_index + 1]
    elif eligible_indices is not None and segment_ids is not None:
        indices = baseline_atom_indices(
            h_index,
            combo.bh,
            eligible_indices,
            segment_ids,
            available_from_indices,
            as_of_index,
            excluded_from_indices,
        )
        if len(indices) != combo.bh:
            return math.nan
        values = tr15[indices]
    else:
        raise ValueError("eligible_indices and segment_ids must be supplied together")
    return entry_baseline_from_values(values, combo)


def entry_baseline_from_values(values: np.ndarray, combo: Combo) -> float:
    """Evaluate one baseline from an already selected immutable BH sample."""
    if len(values) != combo.bh or not np.isfinite(values).all():
        return math.nan
    if combo.method != ENTRY_METHOD_ROLLING:
        raise ValueError(f"unsupported entry method: {combo.method}")
    if not 1 <= combo.trw <= combo.bh:
        raise ValueError("TRW must be between 1 and BH")
    sums = np.convolve(values, np.ones(combo.trw, dtype=float), mode="valid")
    return float(sums.mean())


def _window_net_drop(
    frame: pd.DataFrame,
    low_index: int,
    w: int,
    h_index: int = 0,
) -> float:
    continuous_start = low_index - int(frame.iloc[low_index]["continuous_run"]) + 1
    start = max(int(h_index), continuous_start, low_index - int(w) + 1)
    if start > low_index:
        return math.nan
    value = float(frame.iloc[start]["open"]) - float(frame.iloc[low_index]["low"])
    return value if _finite(value) and value > 0 else math.nan


def _window_net_drop_arrays(opens: np.ndarray, lows: np.ndarray, continuous_runs: np.ndarray, low_index: int, w: int, h_index: int = 0) -> float:
    """Fast equivalent of the public DataFrame helper used by the full grid."""
    return _window_net_drop_sample_arrays(
        opens, lows, continuous_runs, low_index, w, h_index
    )[0]


def _window_net_drop_sample_arrays(
    opens: np.ndarray,
    lows: np.ndarray,
    continuous_runs: np.ndarray,
    end_index: int,
    w: int,
    h_index: int,
) -> tuple[float, int, int, int]:
    """Return ``w_open_to_end_low_drop`` for the available 1..W prefix.

    This is ``open[start] - low[end]``. It is not the maximum ordered decline
    from any internal high to a later low. ``start`` is bounded by H, the
    continuity segment, and W; no full-W or minimum-window-ratio gate applies.
    """
    if end_index < 0 or end_index < h_index:
        return math.nan, -1, -1, 0
    continuous_start = end_index - int(continuous_runs[end_index]) + 1
    start = max(int(h_index), continuous_start, end_index - int(w) + 1)
    if start > end_index:
        return math.nan, -1, -1, 0
    observed = end_index - start + 1
    value = float(opens[start] - lows[end_index])
    candidate = value if _finite(value) and value > 0 else math.nan
    return candidate, start, end_index, observed


def _max_drawdown(returns: Iterable[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def simulate_combo(frame: pd.DataFrame, combo: Combo, train_start: pd.Timestamp, train_end: pd.Timestamp) -> list[dict[str, Any]]:
    """Execute one all-window rolling-TR V4.4 combo under declared policies."""
    times = frame["datetime"].to_numpy()
    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    closes = frame["close"].to_numpy(float)
    tr15 = frame["tr15"].to_numpy(float)
    continuous = frame["continuous"].to_numpy(bool)
    continuous_runs = frame["continuous_run"].to_numpy(int)
    segment_ids = frame["continuous_segment_id"].to_numpy(int)
    baseline_available_times = pd.to_datetime(
        frame.get("baseline_available_from", frame["datetime"]),
        errors="coerce",
    ).to_numpy(dtype="datetime64[ns]")
    baseline_available_from_indices = np.full(len(frame), len(frame) + 1, dtype=int)
    known_availability = ~np.isnat(baseline_available_times)
    baseline_available_from_indices[known_availability] = np.searchsorted(
        times,
        baseline_available_times[known_availability],
        side="left",
    )
    baseline_excluded_times = pd.to_datetime(
        frame.get("baseline_excluded_from", pd.Series(pd.NaT, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype="datetime64[ns]")
    baseline_excluded_from_indices = np.full(len(frame), len(frame) + 1, dtype=int)
    known_exclusions = ~np.isnat(baseline_excluded_times)
    baseline_excluded_from_indices[known_exclusions] = np.searchsorted(
        times,
        baseline_excluded_times[known_exclusions],
        side="left",
    )
    synthetic_empty = frame["is_synthetic_empty_bar"].to_numpy(bool)
    volumes = frame["volume"].to_numpy(float)
    trade_counts = frame["trade_count"].to_numpy(float)
    real_trade_bar = ~synthetic_empty & (volumes > 0) & (trade_counts > 0)
    confirmed_universal = (
        frame.get("buffer_confirmed_excluded", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
        .to_numpy()
        & frame.get("low_volume_atom", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    final_baseline_excluded = frame.get(
        "baseline_excluded", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool).to_numpy()
    marker_eligible = frame.get(
        "eligible_if_excluding_marked", pd.Series(True, index=frame.index)
    ).fillna(False).astype(bool).to_numpy()
    base_eligible = continuous & np.isfinite(tr15)
    if combo.baseline_sampling_policy == BASELINE_SAMPLING_EXCLUDE_MARKED:
        base_eligible &= marker_eligible
    eligible_indices = np.flatnonzero(base_eligible)
    confirmed_low_activity_active = frame.get(
        "confirmed_low_activity_active", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool).to_numpy()
    pending_low_activity = frame.get(
        "low_activity_state",
        pd.Series(LOW_ACTIVITY_STATE_NORMAL, index=frame.index),
    ).astype(str).eq(LOW_ACTIVITY_STATE_PENDING).to_numpy()
    start_positions = np.flatnonzero(frame["datetime"].ge(train_start).to_numpy())
    end_positions = np.flatnonzero(frame["datetime"].le(train_end).to_numpy())
    if not len(start_positions) or not len(end_positions):
        raise ValueError("training interval is outside source data")
    start, end = int(start_positions[0]), int(end_positions[-1])
    trades: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    pending_exit: dict[str, Any] | None = None
    in_position = False
    flat_reset = start
    entry_index = -1
    entry_price = math.nan
    entry_h = -1
    active_low = math.inf
    active_low_index = -1
    rebound_basis = math.nan
    rebound_max_window_start_index = -1
    rebound_max_window_end_index = -1
    rebound_max_window_observed_bar_count = 0
    rebound_latest_candidate = math.nan
    rebound_latest_candidate_start_index = -1
    rebound_latest_candidate_end_index = -1
    rebound_latest_candidate_observed_bar_count = 0
    rebound_candidates_effective_through_index = -1
    entry_bar_check_basis = math.nan
    entry_bar_check_window_start_index = -1
    entry_bar_check_window_end_index = -1
    entry_bar_check_window_observed_bar_count = 0
    entry_baseline_value = math.nan
    entry_drop_value = math.nan
    baseline_start_index = -1
    baseline_end_index = -1
    baseline_physical_span_bars = 0
    baseline_excluded_atom_count = 0
    baseline_pending_atom_count = 0
    baseline_confirmed_excluded_atom_count = 0
    entry_filter_stage = 0
    initial_entry_index = -1
    entry_price_before_slippage = math.nan
    entry_trigger_price = math.nan
    entry_price_basis = ""
    entry_gap_adjusted: bool | None = None
    entry_gap_slippage = math.nan
    entry_wait_bar_count = 0
    entry_fill_source = ""
    entry_initial_bar_synthetic = False
    entry_initial_bar_volume = math.nan
    entry_initial_bar_trade_count = math.nan
    position_running_lows: dict[int, float] = {}

    def apply_completed_rebound_candidate(index: int) -> None:
        """Make one completed-bar candidate effective for later bars only."""
        nonlocal rebound_basis
        nonlocal rebound_max_window_start_index, rebound_max_window_end_index
        nonlocal rebound_max_window_observed_bar_count
        nonlocal rebound_latest_candidate, rebound_latest_candidate_start_index
        nonlocal rebound_latest_candidate_end_index
        nonlocal rebound_latest_candidate_observed_bar_count
        nonlocal rebound_candidates_effective_through_index
        candidate, window_start, window_end, observed = _window_net_drop_sample_arrays(
            opens, lows, continuous_runs, index, combo.w, entry_h
        )
        rebound_latest_candidate = candidate
        rebound_latest_candidate_start_index = window_start
        rebound_latest_candidate_end_index = window_end
        rebound_latest_candidate_observed_bar_count = observed
        rebound_candidates_effective_through_index = index
        if _finite(candidate) and (
            not _finite(rebound_basis) or candidate > rebound_basis
        ):
            rebound_basis = candidate
            rebound_max_window_start_index = window_start
            rebound_max_window_end_index = window_end
            rebound_max_window_observed_bar_count = observed

    def close_trade(
        index: int,
        price: float,
        reason: str,
        *,
        rebound_check_price: float | None = None,
        rebound_check_price_basis: str | None = None,
        exit_price_basis: str,
        rebound_gap_adjusted: bool | None = None,
        rebound_gap_slippage: float | None = None,
        speed_reference_index: int | None = None,
        speed_reference_low: float | None = None,
        speed_current_low: float | None = None,
        speed_extension: float | None = None,
        speed_check_price: float | None = None,
        speed_check_price_basis: str | None = None,
        pending_exit_trigger_index: int | None = None,
        pending_exit_theoretical_price: float | None = None,
        pending_exit_wait_bar_count: int = 0,
        rebound_basis_for_exit: float | None = None,
        rebound_basis_window_start_index: int | None = None,
        rebound_basis_window_end_index: int | None = None,
        rebound_basis_window_observed_bar_count: int | None = None,
    ) -> None:
        nonlocal in_position, flat_reset
        assert in_position
        effective_rebound_basis = (
            float(rebound_basis_for_exit)
            if rebound_basis_for_exit is not None
            else rebound_basis
        )
        rebound_window_start_index = (
            int(rebound_basis_window_start_index)
            if rebound_basis_window_start_index is not None
            else rebound_max_window_start_index
        )
        rebound_window_end_index = (
            int(rebound_basis_window_end_index)
            if rebound_basis_window_end_index is not None
            else rebound_max_window_end_index
        )
        rebound_window_observed_bar_count = (
            int(rebound_basis_window_observed_bar_count)
            if rebound_basis_window_observed_bar_count is not None
            else rebound_max_window_observed_bar_count
        )
        candidate_index = (
            int(pending_exit_trigger_index)
            if pending_exit_trigger_index is not None
            else index
        )
        (
            exit_bar_candidate,
            exit_bar_candidate_start,
            exit_bar_candidate_end,
            exit_bar_candidate_observed,
        ) = _window_net_drop_sample_arrays(
            opens, lows, continuous_runs, candidate_index, combo.w, entry_h
        )
        rebound_threshold = (
            active_low + combo.m * effective_rebound_basis
            if _finite(effective_rebound_basis)
            else math.nan
        )
        one_bar_drops = np.maximum(
            closes[entry_h:signal_index] - lows[entry_h + 1 : signal_index + 1],
            0.0,
        )
        signal_single_bar_drop_share = (
            float(one_bar_drops.max() / entry_drop_value)
            if len(one_bar_drops) and entry_drop_value > 0
            else math.nan
        )
        trades.append({
            "combo_id": combo.combo_id, "method": combo.method,
            "baseline_sampling_policy": combo.baseline_sampling_policy,
            "entry_index": entry_index,
            "entry_time": str(pd.Timestamp(times[entry_index])), "entry_price": entry_price,
            "entry_fill_price": entry_price,
            "entry_price_before_slippage": entry_price_before_slippage,
            "entry_trigger_price": entry_trigger_price,
            "entry_price_basis": entry_price_basis,
            "entry_gap_adjusted": entry_gap_adjusted,
            "entry_gap_slippage": entry_gap_slippage,
            "entry_fill_vs_trigger_delta": entry_price_before_slippage - entry_trigger_price,
            "entry_bar_synthetic": bool(synthetic_empty[entry_index]),
            "entry_bar_volume": float(volumes[entry_index]),
            "entry_bar_trade_count": float(trade_counts[entry_index]),
            "entry_fill_mode": combo.entry_fill_mode,
            "entry_execution_policy": combo.entry_execution_policy,
            "entry_slippage": combo.entry_slippage,
            "strategy_id": combo.strategy_id,
            "exit_mode": combo.exit_mode,
            "speed_exit_enabled": combo.speed_exit_enabled,
            "speed_window_bars": int(combo.speed_window_bars),
            "rebound_exit_enabled": combo.rebound_exit_enabled,
            "entry_fill_source": entry_fill_source,
            "initial_entry_index": initial_entry_index,
            "initial_entry_time": str(pd.Timestamp(times[initial_entry_index])),
            "entry_wait_bar_count": entry_wait_bar_count,
            "initial_entry_bar_synthetic": entry_initial_bar_synthetic,
            "initial_entry_bar_volume": entry_initial_bar_volume,
            "initial_entry_bar_trade_count": entry_initial_bar_trade_count,
            "signal_index": signal_index, "signal_time": str(pd.Timestamp(times[signal_index])),
            "h_index": entry_h, "h_time": str(pd.Timestamp(times[entry_h])),
            "exit_index": index, "exit_time": str(pd.Timestamp(times[index])), "exit_price": price,
            "exit_fill_price": price,
            "exit_price_basis": exit_price_basis,
            "exit_reason": reason, "return": (entry_price - price) / entry_price,
            "exit_bar_synthetic": bool(synthetic_empty[index]),
            "exit_bar_volume": float(volumes[index]),
            "exit_bar_trade_count": float(trade_counts[index]),
            "pending_exit": pending_exit_trigger_index is not None,
            "pending_exit_trigger_index": pending_exit_trigger_index,
            "pending_exit_trigger_time": (
                ""
                if pending_exit_trigger_index is None
                else str(pd.Timestamp(times[pending_exit_trigger_index]))
            ),
            "pending_exit_trigger_reason": (
                "" if pending_exit_trigger_index is None else reason
            ),
            "pending_exit_theoretical_price": pending_exit_theoretical_price,
            "pending_exit_wait_bar_count": int(pending_exit_wait_bar_count),
            "pending_exit_fill_policy": (
                "next_real_trade_bar_open"
                if pending_exit_trigger_index is not None
                else "same_real_trade_bar"
            ),
            "pending_exit_fill_vs_theoretical_delta": (
                None
                if pending_exit_theoretical_price is None
                else float(price - pending_exit_theoretical_price)
            ),
            "active_low_index": active_low_index, "active_low": active_low,
            "rebound_net_drop": effective_rebound_basis,
            "rebound_max_w_drop": effective_rebound_basis,
            "rebound_latest_applied_candidate": rebound_latest_candidate,
            "rebound_latest_applied_candidate_start_index": rebound_latest_candidate_start_index,
            "rebound_latest_applied_candidate_end_index": rebound_latest_candidate_end_index,
            "rebound_latest_applied_candidate_observed_bar_count": rebound_latest_candidate_observed_bar_count,
            "rebound_exit_bar_candidate": exit_bar_candidate,
            "rebound_exit_bar_candidate_start_index": exit_bar_candidate_start,
            "rebound_exit_bar_candidate_end_index": exit_bar_candidate_end,
            "rebound_exit_bar_candidate_observed_bar_count": exit_bar_candidate_observed,
            "rebound_candidates_effective_through_index": rebound_candidates_effective_through_index,
            "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
            "e": combo.e, "bh": combo.bh, "trw": combo.trw,
            "k": combo.k, "w": combo.w, "m": combo.m,
            "rebound_window_start_index": rebound_window_start_index,
            "rebound_window_end_index": rebound_window_end_index,
            "rebound_window_observed_bar_count": rebound_window_observed_bar_count,
            "rebound_threshold": rebound_threshold,
            "rebound_trigger_price": rebound_threshold,
            "rebound_check_price": rebound_check_price,
            "rebound_check_price_basis": rebound_check_price_basis,
            "rebound_gap_adjusted": rebound_gap_adjusted,
            "rebound_gap_slippage": rebound_gap_slippage,
            "rebound_baseline_update_rule": (
                "maximum_positive_completed_bar_w_candidates_effective_next_bar"
            ),
            "speed_reference_index": speed_reference_index,
            "speed_reference_time": (
                ""
                if speed_reference_index is None
                else str(pd.Timestamp(times[speed_reference_index]))
            ),
            "speed_reference_low": speed_reference_low,
            "speed_current_low": speed_current_low,
            "speed_extension": speed_extension,
            "speed_check_price": (
                speed_check_price
                if reason == DOWNSIDE_SPEED_EXIT_REASON
                else None
            ),
            "speed_check_price_basis": (
                speed_check_price_basis
                if reason == DOWNSIDE_SPEED_EXIT_REASON
                else None
            ),
            "downside_speed_exit_fill_policy": (
                (
                    "next_real_trade_bar_open_after_pending_signal"
                    if pending_exit_trigger_index is not None
                    else "current_real_trade_bar_close"
                )
                if combo.speed_exit_enabled
                else "disabled"
            ),
            "entry_continuous_segment_id": int(segment_ids[entry_index]),
            "exit_continuous_segment_id": int(segment_ids[index]),
            "position_crosses_real_gap": bool(segment_ids[entry_index] != segment_ids[index]),
            "signal_single_bar_drop_share": signal_single_bar_drop_share,
            "signal_synthetic_empty_bar_count": int(synthetic_empty[entry_h : signal_index + 1].sum()),
            "entry_baseline_value": entry_baseline_value, "entry_drop_value": entry_drop_value,
            "baseline_history_start_index": baseline_start_index,
            "baseline_history_end_index": baseline_end_index,
            "baseline_eligible_atom_count": combo.bh,
            "baseline_physical_span_bars": baseline_physical_span_bars,
            "baseline_excluded_atom_count": baseline_excluded_atom_count,
            "baseline_pending_atom_count": baseline_pending_atom_count,
            "baseline_confirmed_excluded_atom_count": baseline_confirmed_excluded_atom_count,
            "baseline_filter_id": combo.baseline_filter_id,
            "baseline_filter_stage": entry_filter_stage,
            "trade_audit_schema_version": (
                COMBINED_TRADE_AUDIT_SCHEMA_VERSION
                if combo.speed_exit_enabled
                else TRADE_AUDIT_SCHEMA_VERSION
            ),
            "trade_audit_schema_id": (
                COMBINED_TRADE_AUDIT_SCHEMA_ID
                if combo.speed_exit_enabled
                else TRADE_AUDIT_SCHEMA_ID
            ),
        })
        in_position = False
        flat_reset = index
        position_running_lows.clear()

    def evaluate_rebound_exit(index: int) -> bool:
        nonlocal active_low, active_low_index, pending_exit
        same_signal_entry_bar = index == entry_index == signal_index
        basis_for_check = (
            entry_bar_check_basis if same_signal_entry_bar else rebound_basis
        )
        basis_window_start = (
            entry_bar_check_window_start_index
            if same_signal_entry_bar
            else rebound_max_window_start_index
        )
        basis_window_end = (
            entry_bar_check_window_end_index
            if same_signal_entry_bar
            else rebound_max_window_end_index
        )
        basis_window_observed = (
            entry_bar_check_window_observed_bar_count
            if same_signal_entry_bar
            else rebound_max_window_observed_bar_count
        )
        def finish_rebound(
            trigger: float,
            check_price: float,
            check_price_basis: str,
            fill: float,
            exit_price_basis: str,
            gap_adjusted: bool,
            gap_slippage: float,
        ) -> bool:
            nonlocal pending_exit
            if not bool(real_trade_bar[index]):
                pending_exit = {
                    "trigger_index": index,
                    "reason": "rebound_threshold",
                    "theoretical_price": float(trigger),
                    "rebound_check_price": check_price,
                    "rebound_check_price_basis": check_price_basis,
                    "rebound_gap_adjusted": None,
                    "rebound_gap_slippage": None,
                    "speed_reference_index": None,
                    "speed_reference_low": None,
                    "speed_current_low": None,
                    "speed_extension": None,
                    "speed_check_price": None,
                    "speed_check_price_basis": None,
                    "rebound_basis_for_exit": basis_for_check,
                    "rebound_basis_window_start_index": basis_window_start,
                    "rebound_basis_window_end_index": basis_window_end,
                    "rebound_basis_window_observed_bar_count": basis_window_observed,
                }
                return True
            close_trade(
                index,
                float(fill),
                "rebound_threshold",
                rebound_check_price=check_price,
                rebound_check_price_basis=check_price_basis,
                exit_price_basis=exit_price_basis,
                rebound_gap_adjusted=gap_adjusted,
                rebound_gap_slippage=gap_slippage,
                rebound_basis_for_exit=basis_for_check,
                rebound_basis_window_start_index=basis_window_start,
                rebound_basis_window_end_index=basis_window_end,
                rebound_basis_window_observed_bar_count=basis_window_observed,
            )
            return True

        prior_trigger = (
            active_low + combo.m * basis_for_check
            if _finite(active_low) and _finite(basis_for_check)
            else math.nan
        )
        if _finite(prior_trigger) and float(opens[index]) >= prior_trigger:
            return finish_rebound(
                prior_trigger,
                float(opens[index]),
                "bar_open_at_or_above_prior_trigger",
                float(opens[index]),
                "exit_bar_open_at_or_above_rebound_trigger",
                True,
                float(opens[index] - prior_trigger),
            )
        if _finite(prior_trigger) and float(highs[index]) >= prior_trigger:
            return finish_rebound(
                prior_trigger,
                float(highs[index]),
                "bar_high_reaches_prior_trigger",
                prior_trigger,
                "calculated_rebound_threshold",
                False,
                0.0,
            )

        strict_low = float(lows[index]) < active_low
        if strict_low:
            active_low, active_low_index = float(lows[index]), index
            trigger = (
                active_low + combo.m * basis_for_check
                if _finite(basis_for_check)
                else math.nan
            )
            if _finite(trigger) and float(closes[index]) >= trigger:
                return finish_rebound(
                    trigger,
                    float(closes[index]),
                    "bar_close_after_strict_new_low",
                    float(closes[index]),
                    "same_bar_close_after_strict_new_low_confirmation",
                    False,
                    0.0,
                )
        apply_completed_rebound_candidate(index)
        return False

    def evaluate_speed_exit(index: int) -> bool:
        nonlocal pending_exit
        if not combo.speed_exit_enabled:
            return False
        position_running_lows[index] = float(active_low)
        speed_bars = int(combo.speed_window_bars)
        if index - entry_index < speed_bars:
            return False
        reference_index = index - speed_bars
        if reference_index not in position_running_lows:
            return False
        if int(segment_ids[reference_index]) != int(segment_ids[index]):
            return False
        reference_low = float(position_running_lows[reference_index])
        current_low = float(active_low)
        extension = reference_low - current_low
        if _finite(extension) and extension <= 0:
            if not bool(real_trade_bar[index]):
                pending_exit = {
                    "trigger_index": index,
                    "reason": DOWNSIDE_SPEED_EXIT_REASON,
                    "theoretical_price": float(closes[index]),
                    "rebound_check_price": None,
                    "rebound_check_price_basis": None,
                    "rebound_gap_adjusted": None,
                    "rebound_gap_slippage": None,
                    "speed_reference_index": reference_index,
                    "speed_reference_low": reference_low,
                    "speed_current_low": current_low,
                    "speed_extension": extension,
                    "speed_check_price": float(closes[index]),
                    "speed_check_price_basis": "bar_close",
                }
                return True
            close_trade(
                index,
                float(closes[index]),
                DOWNSIDE_SPEED_EXIT_REASON,
                exit_price_basis="current_bar_close",
                speed_reference_index=reference_index,
                speed_reference_low=reference_low,
                speed_current_low=current_low,
                speed_extension=extension,
                speed_check_price=float(closes[index]),
                speed_check_price_basis="bar_close",
            )
            return True
        return False

    def activate_entry(
        index: int,
        raw_price: float,
        fill_source: str,
        signal_state: dict[str, Any],
        initial_index: int,
    ) -> None:
        nonlocal entry_h, signal_index, entry_baseline_value, entry_drop_value
        nonlocal baseline_start_index, baseline_end_index, baseline_physical_span_bars
        nonlocal baseline_excluded_atom_count, baseline_pending_atom_count
        nonlocal baseline_confirmed_excluded_atom_count, entry_filter_stage
        nonlocal entry_index, entry_price, entry_price_before_slippage, in_position
        nonlocal entry_trigger_price, entry_price_basis, entry_gap_adjusted
        nonlocal entry_gap_slippage
        nonlocal active_low, active_low_index, rebound_basis, initial_entry_index
        nonlocal rebound_max_window_start_index, rebound_max_window_end_index
        nonlocal rebound_max_window_observed_bar_count
        nonlocal rebound_latest_candidate, rebound_latest_candidate_start_index
        nonlocal rebound_latest_candidate_end_index
        nonlocal rebound_latest_candidate_observed_bar_count
        nonlocal rebound_candidates_effective_through_index
        nonlocal entry_bar_check_basis, entry_bar_check_window_start_index
        nonlocal entry_bar_check_window_end_index
        nonlocal entry_bar_check_window_observed_bar_count
        nonlocal entry_wait_bar_count, entry_fill_source, entry_initial_bar_synthetic
        nonlocal entry_initial_bar_volume, entry_initial_bar_trade_count

        slipped_price = float(raw_price) - float(combo.entry_slippage)
        if not _finite(slipped_price) or slipped_price <= 0:
            raise ValueError("adverse short-entry slippage produced a non-positive fill")
        entry_h = int(signal_state["h_index"])
        signal_index = int(signal_state["signal_index"])
        entry_baseline_value = float(signal_state["baseline"])
        entry_drop_value = float(signal_state["drop"])
        baseline_start_index = int(signal_state["baseline_start_index"])
        baseline_end_index = int(signal_state["baseline_end_index"])
        baseline_physical_span_bars = int(signal_state["baseline_physical_span_bars"])
        baseline_excluded_atom_count = int(signal_state["baseline_excluded_atom_count"])
        baseline_pending_atom_count = int(signal_state["baseline_pending_atom_count"])
        baseline_confirmed_excluded_atom_count = int(
            signal_state["baseline_confirmed_excluded_atom_count"]
        )
        entry_filter_stage = int(signal_state["baseline_filter_stage"])
        initial_entry_index = int(initial_index)
        entry_index = int(index)
        entry_price_before_slippage = float(raw_price)
        entry_price = slipped_price
        entry_trigger_price = float(signal_state["entry_trigger_price"])
        if combo.entry_fill_mode == ENTRY_FILL_CALCULATED_THRESHOLD and fill_source == "calculated_threshold":
            entry_gap_adjusted = bool(float(raw_price) < entry_trigger_price)
            entry_price_basis = (
                "signal_bar_open_after_down_gap"
                if entry_gap_adjusted
                else "calculated_entry_threshold"
            )
            entry_gap_slippage = (
                float(entry_trigger_price - raw_price) if entry_gap_adjusted else 0.0
            )
        else:
            entry_gap_adjusted = None
            entry_price_basis = (
                "waited_real_trade_bar_open"
                if fill_source == "waited_real_trade_open"
                else "initial_real_trade_bar_open"
            )
            entry_gap_slippage = math.nan
        entry_wait_bar_count = int(index - initial_index)
        entry_fill_source = fill_source
        entry_initial_bar_synthetic = bool(synthetic_empty[initial_index])
        entry_initial_bar_volume = float(volumes[initial_index])
        entry_initial_bar_trade_count = float(trade_counts[initial_index])
        in_position = True
        active_low, active_low_index, rebound_basis = math.inf, -1, math.nan
        rebound_max_window_start_index = -1
        rebound_max_window_end_index = -1
        rebound_max_window_observed_bar_count = 0
        rebound_latest_candidate = math.nan
        rebound_latest_candidate_start_index = -1
        rebound_latest_candidate_end_index = -1
        rebound_latest_candidate_observed_bar_count = 0
        rebound_candidates_effective_through_index = -1
        entry_bar_check_basis = math.nan
        entry_bar_check_window_start_index = -1
        entry_bar_check_window_end_index = -1
        entry_bar_check_window_observed_bar_count = 0
        if entry_index == signal_index:
            (
                entry_bar_check_basis,
                entry_bar_check_window_start_index,
                entry_bar_check_window_end_index,
                entry_bar_check_window_observed_bar_count,
            ) = _window_net_drop_sample_arrays(
                opens, lows, continuous_runs, signal_index - 1, combo.w, entry_h
            )
        else:
            apply_completed_rebound_candidate(signal_index)
        position_running_lows.clear()

    signal_index = -1
    for i in range(start, end + 1):
        if pending_exit is not None:
            if bool(real_trade_bar[i]):
                state = pending_exit
                pending_exit = None
                trigger_index = int(state["trigger_index"])
                theoretical_price = float(state["theoretical_price"])
                actual_price = float(opens[i])
                close_trade(
                    i,
                    actual_price,
                    str(state["reason"]),
                    rebound_check_price=state["rebound_check_price"],
                    rebound_check_price_basis=state["rebound_check_price_basis"],
                    exit_price_basis="next_real_trade_bar_open_after_pending_exit",
                    rebound_gap_adjusted=(
                        bool(actual_price > theoretical_price)
                        if state["reason"] == "rebound_threshold"
                        else None
                    ),
                    rebound_gap_slippage=(
                        max(0.0, actual_price - theoretical_price)
                        if state["reason"] == "rebound_threshold"
                        else None
                    ),
                    speed_reference_index=state["speed_reference_index"],
                    speed_reference_low=state["speed_reference_low"],
                    speed_current_low=state["speed_current_low"],
                    speed_extension=state["speed_extension"],
                    speed_check_price=state["speed_check_price"],
                    speed_check_price_basis=state["speed_check_price_basis"],
                    pending_exit_trigger_index=trigger_index,
                    pending_exit_theoretical_price=theoretical_price,
                    pending_exit_wait_bar_count=i - trigger_index,
                    rebound_basis_for_exit=state.get("rebound_basis_for_exit"),
                    rebound_basis_window_start_index=state.get(
                        "rebound_basis_window_start_index"
                    ),
                    rebound_basis_window_end_index=state.get(
                        "rebound_basis_window_end_index"
                    ),
                    rebound_basis_window_observed_bar_count=state.get(
                        "rebound_basis_window_observed_bar_count"
                    ),
                )
            continue
        cancelled_entry_order = False
        entry_gate_active = (
            combo.baseline_sampling_policy
            == BASELINE_SAMPLING_CONFIRMED_LOW_ACTIVITY_GATE
            and bool(confirmed_low_activity_active[i])
        )
        if entry_gate_active and pending is not None:
            pending = None
            cancelled_entry_order = True
        if pending is not None and int(pending["next_check_index"]) == i:
            initial_index = int(pending["initial_fill_index"])
            candidate_count = i - initial_index + 1
            if not bool(continuous[i]):
                pending = None
                cancelled_entry_order = True
            elif bool(real_trade_bar[i]):
                activate_entry(
                    i,
                    float(opens[i]),
                    "initial_real_trade_open" if i == initial_index else "waited_real_trade_open",
                    pending,
                    initial_index,
                )
                pending = None
            elif (
                combo.entry_execution_policy == ENTRY_EXECUTION_REJECT_SYNTHETIC_FILL
                or candidate_count >= MAX_REAL_TRADE_WAIT_BARS
            ):
                pending = None
                cancelled_entry_order = True
            elif i < end:
                pending["next_check_index"] = i + 1
        if in_position:
            if evaluate_rebound_exit(i):
                continue
            if evaluate_speed_exit(i):
                continue
        if (
            not in_position
            and pending is None
            and not cancelled_entry_order
            and not entry_gate_active
        ):
            # Signal state is both reset-aware and limited to a contiguous E-bar physical window.
            window_start = max(flat_reset, i - combo.e + 1)
            if window_start >= i or int(continuous_runs[i]) < min(combo.e, i - flat_reset + 1):
                continue
            local_highs = highs[window_start : i + 1]
            h_rel = int(np.argmax(local_highs))  # first maximum retains the earlier equal H.
            h_index = window_start + h_rel
            if h_index >= i:
                continue  # avoid unknown intrabar high-to-low ordering.
            filter_stage = (
                0
                if combo.baseline_sampling_policy == BASELINE_SAMPLING_ALL_WINDOW
                else 1
            )
            baseline_indices = baseline_atom_indices(
                h_index,
                combo.bh,
                eligible_indices,
                segment_ids,
                (
                    baseline_available_from_indices
                    if combo.baseline_sampling_policy
                    == BASELINE_SAMPLING_EXCLUDE_MARKED
                    else None
                ),
                (
                    i
                    if combo.baseline_sampling_policy
                    in (
                        BASELINE_SAMPLING_EXCLUDE_MARKED,
                        BASELINE_SAMPLING_CONFIRMED_LOW_ACTIVITY_GATE,
                    )
                    else None
                ),
                (
                    baseline_excluded_from_indices
                    if combo.baseline_sampling_policy
                    == BASELINE_SAMPLING_CONFIRMED_LOW_ACTIVITY_GATE
                    else None
                ),
            )
            baseline = entry_baseline_from_values(tr15[baseline_indices], combo)
            drop = float(highs[h_index] - lows[i])
            if entry_signal_qualifies(baseline, drop, combo.k):
                history_start = int(baseline_indices[0])
                # H remains the physical right edge of the complete BH window.
                history_end = h_index
                permanently_excluded = final_baseline_excluded
                signal_state = {
                    "h_index": h_index,
                    "baseline": baseline,
                    "drop": drop,
                    "entry_trigger_price": float(highs[h_index] - combo.k * baseline),
                    "signal_index": i,
                    "baseline_start_index": history_start,
                    "baseline_end_index": history_end,
                    "baseline_physical_span_bars": history_end - history_start + 1,
                    "baseline_excluded_atom_count": int(
                        permanently_excluded[history_start : history_end + 1].sum()
                    ),
                    "baseline_pending_atom_count": int(
                        pending_low_activity[history_start : history_end + 1].sum()
                    ),
                    "baseline_confirmed_excluded_atom_count": int(
                        confirmed_universal[history_start : history_end + 1].sum()
                    ),
                    "baseline_filter_stage": filter_stage,
                }
                if combo.entry_fill_mode == ENTRY_FILL_NEXT_BAR_OPEN:
                    # Both H and the current bar are observed only at close. The
                    # next continuous bar is the initial execution candidate.
                    if i + 1 <= end and bool(continuous[i + 1]):
                        pending = {
                            "initial_fill_index": i + 1,
                            "next_check_index": i + 1,
                            **signal_state,
                        }
                else:
                    calculated_price = float(signal_state["entry_trigger_price"])
                    # H predates the signal bar. A downward gap therefore
                    # fills at the lower signal-bar open; otherwise the resting
                    # threshold order fills at its calculated price.
                    if bool(real_trade_bar[i]):
                        activate_entry(
                            i,
                            min(float(opens[i]), calculated_price),
                            "calculated_threshold",
                            signal_state,
                            i,
                        )
                        if evaluate_rebound_exit(i):
                            continue
                        if evaluate_speed_exit(i):
                            continue
                    elif (
                        combo.entry_execution_policy
                        == ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE
                        and i < end
                    ):
                        pending = {
                            "initial_fill_index": i,
                            "next_check_index": i + 1,
                            **signal_state,
                        }
    if in_position:
        pending_exit = None
        close_trade(
            end,
            float(closes[end]),
            "segment_end",
            exit_price_basis="sample_end_bar_close",
        )
    return trades


def _event_metrics(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    events: pd.DataFrame,
    hard_event_ids: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score all events, optionally using a requested subset as the hard gate.

    An empty ``hard_event_ids`` tuple makes every coordinate ranking-eligible
    while retaining each event as a diagnostic column.  ``None`` preserves the
    historical behavior in which every catalog event is a hard gate.
    """
    event_ids = tuple(events["event_id"].astype(str))
    hard_ids = event_ids if hard_event_ids is None else tuple(hard_event_ids)
    unknown = set(hard_ids).difference(event_ids)
    if unknown:
        raise ValueError(f"hard event ids must be a subset of the event catalog: {sorted(unknown)}")
    details: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []
    empty_trades = trades.iloc[0:0].copy()
    trade_groups = {
        str(combo_id): group.copy()
        for combo_id, group in trades.groupby("combo_id", sort=False)
    }
    for row in summary.to_dict("records"):
        combo_trades = trade_groups.get(str(row["combo_id"]), empty_trades)
        flags: list[bool] = []
        for event in events.itertuples(index=False):
            begin, finish = pd.Timestamp(event.start_time), pd.Timestamp(event.end_time)
            entered = combo_trades.loc[(pd.to_datetime(combo_trades["entry_time"]) > begin) & (pd.to_datetime(combo_trades["entry_time"]) <= finish)]
            exited = combo_trades.loc[(pd.to_datetime(combo_trades["exit_time"]) > begin) & (pd.to_datetime(combo_trades["exit_time"]) <= finish)]
            selected = entered.iloc[0] if len(entered) == 1 else None
            reason = "" if selected is None else str(selected.exit_reason)
            holds = bool(selected is not None and pd.Timestamp(selected.exit_time) > finish)
            qualified = bool(len(entered) == 1 and len(exited) == 0 and holds and reason == "rebound_threshold")
            failures: list[str] = []
            if len(entered) != 1: failures.append(f"entry_count_{len(entered)}")
            if len(exited): failures.append(f"exit_count_inside_{len(exited)}")
            if selected is not None and not holds: failures.append("selected_trade_does_not_hold_past_end")
            if selected is not None and reason != "rebound_threshold": failures.append(f"selected_exit_reason_{reason}")
            details.append({"combo_id": row["combo_id"], "method": row["method"], "event_id": event.event_id,
                            "qualified": qualified, "entry_count_in_interval": len(entered), "exit_count_in_interval": len(exited),
                            "selected_exit_reason": reason, "failure_reasons": "|".join(failures)})
            flags.append(qualified)
        by_event = dict(zip(event_ids, flags, strict=True))
        combo_rows.append({
            **row,
            "required_event_count": len(hard_ids),
            "qualified_event_count": int(sum(by_event[event_id] for event_id in hard_ids)),
            "diagnostic_qualified_event_count": int(sum(flags)),
            "hard_qualified": bool(all(by_event[event_id] for event_id in hard_ids)),
        })
    return pd.DataFrame(details), pd.DataFrame(combo_rows)


def _rank(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["train_max_drawdown_abs"] = rows["train_max_drawdown"].abs()
    rows["ranking_eligible_ge10"] = rows.hard_qualified & rows.train_trade_count.ge(10)
    rows["ranking_eligible_ge20"] = rows.hard_qualified & rows.train_trade_count.ge(20)
    rows["total_return_rank"] = np.nan
    rows["total_return_rank_ge10"] = np.nan
    rows["total_return_rank_ge20"] = np.nan
    rows["avg_trade_rank"] = np.nan
    rows["max_drawdown_rank"] = np.nan
    rows["balanced_rank_ge10"] = np.nan
    rows["balanced_rank_ge20"] = np.nan
    rows["pareto_frontier"] = False
    rows["return_pareto_frontier"] = False
    for _, group in rows.groupby(
        ["method", "baseline_sampling_policy"], sort=True
    ):
        eligible = group.loc[group.hard_qualified].copy()
        if eligible.empty: continue
        for minimum, column in (
            (0, "total_return_rank"),
            (10, "total_return_rank_ge10"),
            (20, "total_return_rank_ge20"),
        ):
            view = eligible.loc[eligible.train_trade_count.ge(minimum)].copy()
            if view.empty: continue
            order = view.sort_values(
                ["train_return", "train_max_drawdown_abs", "train_trade_count", "combo_id"],
                ascending=[False, True, False, True],
                kind="mergesort",
            )
            rows.loc[order.index, column] = np.arange(1, len(order) + 1)
        rows.loc[eligible.index, "avg_trade_rank"] = eligible.train_avg_trade.rank(ascending=False, method="min")
        rows.loc[eligible.index, "max_drawdown_rank"] = eligible.train_max_drawdown_abs.rank(ascending=True, method="min")
        for minimum in (10, 20):
            view = eligible.loc[eligible.train_trade_count.ge(minimum)].copy()
            if view.empty: continue
            score = 0.5 * view.train_avg_trade.rank(pct=True) + 0.5 * view.train_max_drawdown_abs.rank(pct=True, ascending=False)
            order = view.assign(_score=score).sort_values(["_score", "train_trade_count", "combo_id"], ascending=[False, False, True], kind="mergesort")
            rows.loc[order.index, f"balanced_rank_ge{minimum}"] = np.arange(1, len(order) + 1)
        values = eligible[["train_avg_trade", "train_max_drawdown_abs"]].to_numpy(float)
        frontier = []
        for index, value in zip(eligible.index, values):
            no_worse = (values[:, 0] >= value[0]) & (values[:, 1] <= value[1])
            better = (values[:, 0] > value[0]) | (values[:, 1] < value[1])
            frontier.append(not bool(np.any(no_worse & better)))
        rows.loc[eligible.index, "pareto_frontier"] = frontier
        return_values = eligible[["train_return", "train_max_drawdown_abs"]].to_numpy(float)
        return_frontier = []
        for value in return_values:
            no_worse = (return_values[:, 0] >= value[0]) & (return_values[:, 1] <= value[1])
            better = (return_values[:, 0] > value[0]) | (return_values[:, 1] < value[1])
            return_frontier.append(not bool(np.any(no_worse & better)))
        rows.loc[eligible.index, "return_pareto_frontier"] = return_frontier
    return rows


def _summary(combo: Combo, trades: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([trade["return"] for trade in trades], dtype=float)
    non_gap_returns = np.asarray(
        [trade["return"] for trade in trades if not trade.get("position_crosses_real_gap", False)],
        dtype=float,
    )
    single_bar_shares = np.asarray(
        [trade.get("signal_single_bar_drop_share", math.nan) for trade in trades],
        dtype=float,
    )
    finite_single_bar_shares = single_bar_shares[np.isfinite(single_bar_shares)]
    return {**asdict(combo), "combo_id": combo.combo_id,
            "strategy_id": combo.strategy_id, "exit_mode": combo.exit_mode,
            "speed_exit_enabled": combo.speed_exit_enabled,
            "rebound_exit_enabled": combo.rebound_exit_enabled,
            "train_trade_count": int(len(returns)),
            "train_return": float(np.prod(1 + returns) - 1) if len(returns) else 0.0,
            "train_return_excluding_gap_spanning_trades": float(np.prod(1 + non_gap_returns) - 1) if len(non_gap_returns) else 0.0,
            "train_avg_trade": float(returns.mean()) if len(returns) else math.nan,
            "train_max_drawdown": _max_drawdown(returns),
            "gap_spanning_trade_count": int(sum(bool(t.get("position_crosses_real_gap", False)) for t in trades)),
            "high_single_bar_share_ge_0p8_count": int(np.sum(finite_single_bar_shares >= 0.8)),
            "signal_single_bar_share_median": float(np.median(finite_single_bar_shares)) if len(finite_single_bar_shares) else math.nan,
            "signal_single_bar_share_p95": float(np.quantile(finite_single_bar_shares, 0.95)) if len(finite_single_bar_shares) else math.nan,
            "synthetic_signal_trade_count": int(sum(int(t.get("signal_synthetic_empty_bar_count", 0)) > 0 for t in trades)),
            "segment_end_exit_count": int(sum(t["exit_reason"] == "segment_end" for t in trades)),
            "rebound_exit_count": int(sum(t["exit_reason"] == "rebound_threshold" for t in trades)),
            "speed_exit_count": int(sum(t["exit_reason"] == DOWNSIDE_SPEED_EXIT_REASON for t in trades))}


def default_coarse_grid(
    entry_fill_mode: str = ENTRY_FILL_CALCULATED_THRESHOLD,
    entry_execution_policy: str = ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    entry_slippage: float = 0.0,
    baseline_sampling_policy: str = DEFAULT_BASELINE_SAMPLING_POLICY,
) -> list[Combo]:
    """Retained rolling-TR smoke grid for the current V4.4 contract."""
    output: list[Combo] = []
    for method, ks in ((ENTRY_METHOD_ROLLING, (1.25, 2.25)),):
        for e in (56, 80):
            for bh in (360, 480):
                for trw in (40,):
                    for k in ks:
                        for w in (32, 128):
                            for m in (4.0, 10.0):
                                output.append(Combo(
                                    method, e, bh, trw, k, w, m, entry_fill_mode,
                                    entry_execution_policy, entry_slippage,
                                    0, baseline_sampling_policy,
                                ))
    return output


def local_grid(seed_rows: pd.DataFrame, known: set[str]) -> list[Combo]:
    """One structured, anti-joined local expansion around best hard-qualified seeds."""
    output: list[Combo] = []
    for method in METHODS:
        candidates = seed_rows.loc[(seed_rows.method == method) & seed_rows.hard_qualified].sort_values("balanced_rank_ge10")
        if candidates.empty: continue
        row = candidates.iloc[0]
        entry_fill_mode = str(
            row.get("entry_fill_mode", ENTRY_FILL_CALCULATED_THRESHOLD)
        )
        entry_execution_policy = str(
            row.get("entry_execution_policy", ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE)
        )
        entry_slippage = float(row.get("entry_slippage", 0.0))
        baseline_sampling_policy = str(
            row.get(
                "baseline_sampling_policy",
                DEFAULT_BASELINE_SAMPLING_POLICY,
            )
        )
        k_step = 0.25
        for k in (max(k_step, float(row.k) - k_step), float(row.k) + k_step):
            for w in (max(3, int(row.w) - 16), int(row.w) + 16):
                combo = Combo(
                    method, int(row.e), int(row.bh), int(row.trw), k, w,
                    float(row.m), entry_fill_mode, entry_execution_policy,
                    entry_slippage,
                    0,
                    baseline_sampling_policy,
                )
                if combo.combo_id not in known: output.append(combo)
    return output


def execute(
    frame: pd.DataFrame,
    combos: list[Combo],
    events: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    hard_event_ids: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []; trades: list[dict[str, Any]] = []
    for combo in combos:
        result = simulate_combo(frame, combo, train_start, train_end)
        summaries.append(_summary(combo, result)); trades.extend(result)
    summary = pd.DataFrame(summaries)
    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        trade_frame = pd.DataFrame(columns=["combo_id", "entry_time", "exit_time", "exit_reason"])
    details, qualified = _event_metrics(summary, trade_frame, events, hard_event_ids)
    return _rank(qualified), trade_frame, details


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list): return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp,)): return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def write_reports(
    output: Path,
    rows: pd.DataFrame,
    trades: pd.DataFrame,
    details: pd.DataFrame,
    events: pd.DataFrame,
    continuity: dict[str, Any],
    manifest: dict[str, Any],
    frame: pd.DataFrame | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "ranking.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output / "trades.csv", index=False, encoding="utf-8-sig")
    details.to_csv(output / "event_qualification.csv", index=False, encoding="utf-8-sig")
    (output / "run_manifest.json").write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "version": VERSION_LABEL,
        "rows": _jsonable(rows.to_dict("records")),
        "trades": _jsonable(trades.to_dict("records")),
        "events": _jsonable(events.to_dict("records")),
        "continuity": continuity,
        "hardEvents": manifest.get("hard_event_ids", ["event_01", "event_02"]),
        "primaryObjective": manifest.get("primary_objective", "balanced"),
    }
    payload_text = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    (output / "report_data.json").write_text(payload_text, encoding="utf-8")
    # The report is opened from file:/// in normal research review.  A JSON
    # fetch is blocked by Chromium's local-origin policy, so expose the same
    # immutable payload through a same-directory script as well.
    (output / "report_data.js").write_text(f"window.V4_4_REPORT_DATA={payload_text};\n", encoding="utf-8")
    html = """<!doctype html><meta charset=utf-8><title>V4.4 综合分析</title><style>
:root{--bg:#f4f7fb;--panel:#fff;--ink:#132238;--muted:#566b85;--line:#d9e2ee;--accent:#165fc6;--good:#087744;--bad:#ad5410}[data-theme=dark]{--bg:#11161d;--panel:#000;--ink:#e5f1ff;--muted:#9aafc5;--line:#283340;--accent:#70b7ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1550px;margin:auto;padding:24px 20px 48px}.top,.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.top{justify-content:space-between}.cards{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:10px;margin:14px 0}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px}.card b{display:block;font-size:21px;margin-top:3px}.muted{color:var(--muted)}button,select{color:var(--ink);background:var(--panel);border:1px solid var(--line);padding:7px 10px;border-radius:7px;font:inherit;cursor:pointer}button.active{background:var(--accent);color:white}.table{overflow:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel);margin-top:12px}table{border-collapse:collapse;width:max-content;min-width:100%}th,td{padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}th:first-child,td:first-child{text-align:left}.yes{color:var(--good);font-weight:700}.no{color:var(--bad);font-weight:700}@media(max-width:800px){main{padding:16px 12px}.cards{grid-template-columns:repeat(2,minmax(130px,1fr))}}</style><main><div class=top><div><h1>V4</h1><div class=muted>全样本训练 · 15秒 TR 原子 · 硬资格与诊断场景在页面卡片中明确显示</div></div><button id=theme>深色 Dark</button></div><div id=cards class=cards></div><div class=toolbar><select id=method><option value=rolling_tr_sum>滚动TR总和</option></select><button class=active data-view=ge10>≥10 笔平衡</button><button data-view=ge20>≥20 笔平衡</button><button data-view=avg>笔均收益</button><button data-view=dd>最大回撤</button><button data-view=pareto>Pareto</button><button data-view=all>全部</button></div><div class=panel id=info style="margin-top:12px"></div><div id=table class=table></div><div class=panel style="margin-top:16px"><h2>逐笔记录</h2><select id=tradeCombo></select><div id=trades class=table></div></div></main><script>
fetch('report_data.json').then(r=>r.json()).then(D=>{let method='rolling_tr_sum',view='ge10';const $=x=>document.getElementById(x),pct=x=>x==null?'—':(100*Number(x)).toFixed(3)+'%',flag=x=>x?'<span class=yes>通过</span>':'<span class=no>未通过</span>',gate=D.hardEvents.join(' + '),event2Label=D.hardEvents.includes('event_02')?'event_02':'event_02诊断';const active=()=>D.rows.filter(r=>r.method===method);function rank(r){return view==='ge10'?r.balanced_rank_ge10:view==='ge20'?r.balanced_rank_ge20:view==='avg'?r.avg_trade_rank:view==='dd'?r.max_drawdown_rank:view==='pareto'?(r.pareto_frontier?1:null):1}function render(){const rows=active().filter(r=>rank(r)!=null).sort((a,b)=>Number(rank(a))-Number(rank(b))||a.combo_id.localeCompare(b.combo_id));const qual=active().filter(r=>r.hard_qualified).length;$('cards').innerHTML=[['方法',method==='rolling_tr_sum'?'滚动TR总和（默认）':'TR平均值（独立K网格）'],['组合',active().length],['硬资格 '+gate,qual],['≥10',active().filter(r=>r.balanced_rank_ge10!=null).length],['session-filled零TR',pct(D.continuity.session_filled_zero_tr_share)]].map(x=>`<div class=card><span class=muted>${x[0]}</span><b>${x[1]}</b></div>`).join('');$('info').innerHTML=`<b>契约：</b>BH/TRW 都以 15 秒 TR 原子计数；W 是连续 15 秒净下跌窗口；速度退出关闭。真实时间缺口数 ${D.continuity.real_gap_count}，会使连续窗口重置。${D.hardEvents.includes('event_02')?'event_02 参与硬资格。':'event_02 仅作为诊断。'}${qual?'':'<span class=no> 当前方法没有 '+gate+' 合格组合；表格保留失败边界。</span>'}`;$('table').innerHTML='<table><thead><tr><th>排名</th><th>组合</th><th>E</th><th>BH(15s)</th><th>TRW(15s)</th><th>K</th><th>W(15s)</th><th>M</th><th>训练笔均</th><th>训练回撤</th><th>笔数</th><th>event_01</th><th>'+event2Label+'</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${rank(r)}</td><td>${r.combo_id}</td><td>${r.e}</td><td>${r.bh}</td><td>${r.trw}</td><td>${r.k}</td><td>${r.w}</td><td>${r.m}</td><td>${pct(r.train_avg_trade)}</td><td>${pct(r.train_max_drawdown_abs)}</td><td>${r.train_trade_count}</td><td>${flag(r.event_01_qualified)}</td><td>${flag(r.event_02_qualified)}</td></tr>`).join('')+'</tbody></table>';const combos=active().map(r=>r.combo_id);$('tradeCombo').innerHTML=combos.map(id=>`<option>${id}</option>`).join('');renderTrades();}function renderTrades(){const id=$('tradeCombo').value;$('trades').innerHTML='<table><thead><tr><th>入场</th><th>出场</th><th>原因</th><th>收益</th><th>H</th><th>活动低点</th></tr></thead><tbody>'+D.trades.filter(t=>t.combo_id===id).map(t=>`<tr><td>${t.entry_time}</td><td>${t.exit_time}</td><td>${t.exit_reason}</td><td>${pct(t.return)}</td><td>${t.h_time}</td><td>${t.active_low}</td></tr>`).join('')+'</tbody></table>'}$('method').onchange=e=>{method=e.target.value;render()};$('tradeCombo').onchange=renderTrades;document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{view=b.dataset.view;document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x===b));render()});let dark=false;$('theme').onclick=()=>{dark=!dark;document.documentElement.dataset.theme=dark?'dark':'';$('theme').textContent=dark?'浅色 Light':'深色 Dark'};render();});</script>"""
    html = html.replace("<h1>V4</h1>", "<h1>V4.4</h1>").replace(
        "['session-filled零TR',pct(D.continuity.session_filled_zero_tr_share)]",
        "['低活跃／机制原子排除',pct(D.continuity.baseline_excluded_bar_share)]",
    ).replace(
        "BH/TRW 都以 15 秒 TR 原子计数；W 是连续 15 秒净下跌窗口；速度退出关闭。",
        "BH/TRW 都以合格 15 秒 TR 原子计数；已确认的通用低成交量、K200 锁价与熔断原子跳过，并在同一连续段内向更早历史补足；每类排除原因保持独立；W 下跌基准窗口以持仓后的严格活动低点 L 为终点，仅在创新低时更新，其余执行柱保持冻结；速度退出关闭。",
    ).replace(
        "<th>W(15s)</th>",
        "<th>W下跌基准(15s)</th>",
    ).replace(
        "<td>${r.combo_id}</td>",
        "<td><a href=trade_explain/index.html?combo_id=${encodeURIComponent(r.combo_id)}>${r.combo_id}</a></td>",
    ).replace(
        "<script>\nfetch('report_data.json').then(r=>r.json()).then(D=>{",
        "<script src=report_data.js></script><script>\n(()=>{const D=window.V4_4_REPORT_DATA;if(!D){document.getElementById('info').textContent='报告数据未加载：请确认 report_data.js 与本页位于同一目录。';return;}",
    ).replace("});</script>", "})();</script>")
    (output / "analysis_report.html").write_text(html, encoding="utf-8")
    # V4.4 validation keeps immutable CSV/JSON evidence in its isolated result root.


def run(
    output: Path,
    source: Path = SOURCE_DEFAULT,
    events_path: Path = EVENTS_DEFAULT,
    hard_event_ids: tuple[str, ...] = ("event_01",),
    data_preparation_manifest: Path | None = None,
    extreme_cleaning_audit: Path | None = None,
    entry_fill_mode: str = ENTRY_FILL_CALCULATED_THRESHOLD,
    entry_execution_policy: str = ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    entry_slippage: float = 0.0,
    baseline_sampling_policy: str = DEFAULT_BASELINE_SAMPLING_POLICY,
) -> dict[str, Any]:
    if baseline_sampling_policy not in BASELINE_SAMPLING_POLICIES:
        raise ValueError(
            f"unsupported V4.4 baseline sampling policy: {baseline_sampling_policy}"
        )
    preparation_result: dict[str, Any] | None = None
    if data_preparation_manifest is None:
        source_hash = _sha256(source)
        if (
            source.resolve() == SOURCE_DEFAULT.resolve()
            and source_hash.lower() == SOURCE_SHA256.lower()
            and DATA_PREPARATION_MANIFEST_DEFAULT.is_file()
        ):
            data_preparation_manifest = DATA_PREPARATION_MANIFEST_DEFAULT
        else:
            preparation_result = prepare_dataset(
                source,
                "K200",
                extreme_audit=extreme_cleaning_audit,
                allow_legacy_preexisting_source=False,
            )
            data_preparation_manifest = Path(preparation_result["manifest_path"])
    frame = load_bars(source, data_preparation_manifest)
    events = pd.read_csv(events_path)
    events = events.loc[events.event_id.isin(["event_01", "event_02"])].copy()
    if events.event_id.nunique() != 2: raise ValueError("event_01 and event_02 are both required")
    start, end = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    sample_scope = current_sample_scope_guard(frame, source)
    coarse = default_coarse_grid(
        entry_fill_mode,
        entry_execution_policy,
        entry_slippage,
        baseline_sampling_policy,
    )
    coarse_rows, coarse_trades, coarse_details = execute(frame, coarse, events, start, end, hard_event_ids)
    extras = local_grid(coarse_rows, {combo.combo_id for combo in coarse})
    if extras:
        extra_rows, extra_trades, extra_details = execute(frame, extras, events, start, end, hard_event_ids)
        rows = pd.concat([coarse_rows, extra_rows], ignore_index=True); trades = pd.concat([coarse_trades, extra_trades], ignore_index=True); details = pd.concat([coarse_details, extra_details], ignore_index=True)
        rows = _rank(rows)
    else: rows, trades, details = coarse_rows, coarse_trades, coarse_details
    event_flags = details.pivot(index="combo_id", columns="event_id", values="qualified")
    for event_id in ("event_01", "event_02"):
        rows[f"{event_id}_qualified"] = rows.combo_id.map(event_flags.get(event_id, pd.Series(dtype=bool))).fillna(False).astype(bool)
    synthetic = frame.is_synthetic_empty_bar.astype(bool)
    excluded = frame.baseline_excluded.astype(bool)
    continuity = {
        "session_filled_bar_count": int(synthetic.sum()),
        "session_filled_zero_tr_count": int((synthetic & frame.tr15.eq(0)).sum()),
        "session_filled_zero_tr_share": float((synthetic & frame.tr15.eq(0)).sum()/synthetic.sum()) if synthetic.any() else 0.0,
        "baseline_excluded_bar_count": int(excluded.sum()),
        "baseline_excluded_bar_share": float(excluded.mean()),
        "quiet_activity_excluded_bar_count": int(excluded.sum()),
        "quiet_activity_excluded_bar_share": float(excluded.mean()),
        "universal_low_volume_excluded_bar_count": int(frame.universal_low_volume_excluded.sum()),
        "k200_price_lock_excluded_bar_count": int(frame.k200_price_lock_excluded.sum()),
        "k200_circuit_breaker_excluded_bar_count": int(frame.k200_circuit_breaker_excluded.sum()),
        "baseline_eligible_all_window_bar_count": int(
            frame.baseline_eligible_all_window.sum()
        ),
        "baseline_eligible_exclude_marked_bar_count": int(
            frame.baseline_eligible_exclude_marked.sum()
        ),
        "real_gap_count": int((~frame.continuous & frame.index.to_series().ne(0)).sum()),
    }
    engine_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "version_label": VERSION_LABEL,
        "strategy_major_version": VERSION_LABEL,
        "baseline_filter_id": baseline_filter_id(baseline_sampling_policy),
        "baseline_sampling_policy": baseline_sampling_policy,
        "entry_fill_mode": entry_fill_mode,
        "entry_execution_policy": entry_execution_policy,
        "entry_slippage": entry_slippage,
        "maximum_real_trade_wait_bars": MAX_REAL_TRADE_WAIT_BARS,
        "engine_sha256": engine_sha,
        "source": str(source),
        "source_sha256": SOURCE_SHA256,
        "data_preparation_manifest": str(data_preparation_manifest),
        "data_preparation_manifest_sha256": _sha256(data_preparation_manifest),
        "data_preparation_status": preparation_result["status"] if preparation_result else "provided_manifest",
        "baseline_filter_contract": {
            "universal_method_id": "strict_median20_30m",
            "k200_mechanism_methods": ["price_lock_candidate", "circuit_breaker_candidate"],
            "reference_elapsed_hours": 84.0,
            "volume_median_ratio": 0.2,
            "duration_minutes": 30,
            "marker_policy": "pending low activity has no strategy effect; confirmation publishes a retroactive baseline_excluded_from timestamp and activates the entry gate until the first normal-volume atom",
            "baseline_policy": "at each calculation time collect the latest BH finite atoms inside one continuity segment that have not reached their causal exclusion time",
        },
        "evaluation": "full_training_all_user_dates",
        "train_start": str(start),
        "train_end": str(end),
        "sample_scope_guard": sample_scope,
        "entry_methods": {
            ENTRY_METHOD_ROLLING: "mean of all overlapping TRW sums in exact BH eligible 15s TR atoms",
        },
        "baseline_sampling_policies": {
            "default": DEFAULT_BASELINE_SAMPLING_POLICY,
            "supported": list(BASELINE_SAMPLING_POLICIES),
            "selected": baseline_sampling_policy,
        },
        "entry_execution": {
            "default_policy": ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
            "reject_mode": ENTRY_EXECUTION_REJECT_SYNTHETIC_FILL,
            "real_trade_bar": "not synthetic and volume > 0 and trade_count > 0",
            "wait_limit": f"{MAX_REAL_TRADE_WAIT_BARS} continuous 15-second candidate bars",
            "slippage": "absolute price units subtracted from a short entry fill; current validation value is 0",
        },
        "exit": "V4 15s net-drop rebound only; speed disabled",
        "event_gate": " AND ".join(hard_event_ids),
        "hard_event_ids": list(hard_event_ids),
        "diagnostic_event_ids": [event_id for event_id in events.event_id.astype(str) if event_id not in hard_event_ids],
        "coarse_combo_count": len(coarse),
        "local_combo_count": len(extras),
        "anti_joined": True,
        "continuity": continuity,
    }
    write_reports(output, rows, trades, details, events, continuity, manifest, frame)
    return {"output": str(output), "rows": int(len(rows)), "hard_qualified": int(rows.hard_qualified.sum()), "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated V4 research.")
    parser.add_argument("--output", default=str(RESULT_DEFAULT)); parser.add_argument("--source", default=str(SOURCE_DEFAULT)); parser.add_argument("--events", default=str(EVENTS_DEFAULT)); parser.add_argument("--hard-events", default="event_01"); parser.add_argument("--data-preparation-manifest"); parser.add_argument("--extreme-cleaning-audit")
    parser.add_argument(
        "--entry-fill-mode",
        choices=ENTRY_FILL_MODES,
        default=ENTRY_FILL_CALCULATED_THRESHOLD,
    )
    parser.add_argument(
        "--entry-execution-policy",
        choices=ENTRY_EXECUTION_POLICIES,
        default=ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    )
    parser.add_argument("--entry-slippage", type=float, default=0.0)
    parser.add_argument(
        "--baseline-sampling-policy",
        choices=BASELINE_SAMPLING_POLICIES,
        default=DEFAULT_BASELINE_SAMPLING_POLICY,
    )
    args = parser.parse_args(); hard_events = tuple(item.strip() for item in args.hard_events.split(",") if item.strip()); print(json.dumps(run(Path(args.output), Path(args.source), Path(args.events), hard_events, Path(args.data_preparation_manifest) if args.data_preparation_manifest else None, Path(args.extreme_cleaning_audit) if args.extreme_cleaning_audit else None, args.entry_fill_mode, args.entry_execution_policy, args.entry_slippage, args.baseline_sampling_policy), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
