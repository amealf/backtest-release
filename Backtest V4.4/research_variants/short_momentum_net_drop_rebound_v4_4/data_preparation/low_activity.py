"""V4.4 one-time low-activity and K200 market-mechanism inference.

The universal volume rule is instrument-neutral.  K200 price-lock and circuit
events are intentionally separate, data-inferred mechanism candidates.  The
strategy adapter may union events that have ``apply_to_baseline=True`` while
retaining their independent reason fields.  Universal low-volume atoms use a
pending buffer inside preparation: transient runs are retained in the final
static marker after the first normal 15-second recovery bar completes, while
runs that reach the existing 30-minute threshold are marked excluded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FILTER_RULE_VERSION = "low_activity_v4_4_confirmed_gate_20260806"
ATOM_SECONDS = 15
REFERENCE_HOURS = 84
MINIMUM_REFERENCE_POSITIVE_ATOMS = 500
VOLUME_MEDIAN_RATIO = 0.20
LOW_VOLUME_DURATION_MINUTES = 30
PRICE_LOCK_WINDOW_MINUTES = 30
PRICE_LOCK_MINIMUM_TRADED_MINUTES = 10
PRICE_LOCK_MAX_RANGE_TICKS = 4
PRICE_LOCK_MAX_TR_RATIO = 0.05
PRICE_LOCK_MAX_RANGE_RATIO = 0.05
PRICE_LOCK_REFERENCE_MINIMUM_WINDOWS = 500
CIRCUIT_BREAKER_GAP_SECONDS = 30 * 60
CIRCUIT_HALT_MINUTES = 20
EXTENDED_INTERNAL_PAUSE_MINUTES = 60
LOW_ACTIVITY_STATE_NORMAL = "normal"
LOW_ACTIVITY_STATE_PENDING = "pending_low_activity_buffer"
LOW_ACTIVITY_STATE_CONFIRMED = "confirmed_excluded"


@dataclass(frozen=True)
class LowActivityResult:
    atoms: pd.DataFrame
    events: list[dict[str, Any]]
    tick_size: float | None
    summary: dict[str, Any]


def load_15s_bars(path: Path) -> pd.DataFrame:
    required = {
        "datetime", "open", "high", "low", "close", "volume",
        "trade_count", "source", "is_synthetic_empty_bar", "bar_seconds",
    }
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"15-second source lacks required fields: {sorted(missing)}")
    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame = frame.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    if frame["datetime"].duplicated().any():
        raise ValueError("15-second source contains duplicate datetimes")
    for column in ("open", "high", "low", "close", "volume", "trade_count"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["is_synthetic_empty_bar"] = (
        pd.to_numeric(frame["is_synthetic_empty_bar"], errors="raise").astype(int)
    )
    exact_slot = (
        pd.to_numeric(frame["bar_seconds"], errors="raise").eq(ATOM_SECONDS)
        & frame["datetime"].dt.second.isin((0, 15, 30, 45))
        & frame["datetime"].dt.microsecond.eq(0)
    )
    if not bool(exact_slot.all()):
        raise ValueError("source contains rows outside the exact 15-second grid")
    if not np.isfinite(frame[["open", "high", "low", "close"]].to_numpy(float)).all():
        raise ValueError("source contains non-finite OHLC")
    if bool((frame[["open", "high", "low", "close"]] <= 0).any().any()):
        raise ValueError("source contains non-positive OHLC")
    source_semantic = frame["source"].fillna("missing").astype(str)
    canonical_ibkr = source_semantic.str.startswith("ibkr_historical_ticks_trades_")
    frame["volume_semantic"] = source_semantic.where(
        ~canonical_ibkr,
        "ibkr_historical_trades_volume_with_marked_synthetic_empties",
    )
    frame["atom_segment"] = _segments(frame, ATOM_SECONDS)
    return frame


def _segments(frame: pd.DataFrame, seconds: int) -> pd.Series:
    time_break = frame["datetime"].diff().dt.total_seconds().ne(seconds)
    semantic_break = frame["volume_semantic"].ne(frame["volume_semantic"].shift())
    breaks = time_break | semantic_break
    if len(breaks):
        breaks.iloc[0] = True
    return breaks.cumsum().astype(int)


def infer_tick_size(atoms: pd.DataFrame) -> float:
    prices = np.unique(
        np.concatenate(
            [
                pd.to_numeric(atoms[column], errors="coerce")
                .dropna()
                .round(8)
                .to_numpy()
                for column in ("open", "high", "low", "close")
            ]
        )
    )
    prices.sort()
    differences = np.diff(prices)
    differences = differences[differences > 1e-8]
    if not len(differences):
        raise ValueError("cannot infer a positive price tick")
    return float(np.round(np.quantile(differences, 0.01), 8))


def _minute_atoms(atoms: pd.DataFrame) -> pd.DataFrame:
    work = atoms.copy()
    work["minute"] = work["datetime"].dt.floor("min")
    minute = (
        work.groupby(["atom_segment", "minute"], sort=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trade_count=("trade_count", "sum"),
            traded=("volume", lambda values: int(pd.to_numeric(values).gt(0).any())),
            atom_count=("datetime", "size"),
        )
        .reset_index()
        .rename(columns={"minute": "datetime"})
    )
    minute = minute.loc[minute["atom_count"].eq(4)].reset_index(drop=True)
    minute["minute_segment"] = minute["datetime"].diff().dt.total_seconds().ne(60).cumsum().astype(int)
    previous_close = minute.groupby("minute_segment", sort=False)["close"].shift(1)
    minute["tr"] = np.maximum(minute["high"], previous_close) - np.minimum(minute["low"], previous_close)
    minute.loc[previous_close.isna(), "tr"] = (
        minute.loc[previous_close.isna(), "high"] - minute.loc[previous_close.isna(), "low"]
    )
    return minute


def _causal_reference_median(metric: pd.Series, datetimes: pd.Series) -> pd.Series:
    indexed = pd.Series(
        pd.to_numeric(metric, errors="coerce").to_numpy(float),
        index=pd.DatetimeIndex(datetimes),
    )
    return (
        indexed.rolling(
            pd.Timedelta(hours=REFERENCE_HOURS),
            closed="left",
            min_periods=PRICE_LOCK_REFERENCE_MINIMUM_WINDOWS,
        )
        .median()
        .reset_index(drop=True)
    )


def _price_lock_events(atoms: pd.DataFrame, tick_size: float) -> list[dict[str, Any]]:
    minute = _minute_atoms(atoms)
    grouped = minute.groupby("minute_segment", sort=False)
    window = PRICE_LOCK_WINDOW_MINUTES
    minute["traded_minutes"] = grouped["traded"].rolling(window, min_periods=window).sum().reset_index(level=0, drop=True)
    minute["tr_sum"] = grouped["tr"].rolling(window, min_periods=window).sum().reset_index(level=0, drop=True)
    minute["window_high"] = grouped["high"].rolling(window, min_periods=window).max().reset_index(level=0, drop=True)
    minute["window_low"] = grouped["low"].rolling(window, min_periods=window).min().reset_index(level=0, drop=True)
    minute["price_range"] = minute["window_high"] - minute["window_low"]
    minute["tr_reference"] = _causal_reference_median(minute["tr_sum"], minute["datetime"])
    minute["range_reference"] = _causal_reference_median(minute["price_range"], minute["datetime"])
    minute["tr_ratio"] = minute["tr_sum"] / minute["tr_reference"]
    minute["range_ratio"] = minute["price_range"] / minute["range_reference"]
    minute["confirmed"] = (
        minute["traded_minutes"].ge(PRICE_LOCK_MINIMUM_TRADED_MINUTES)
        & minute["price_range"].le(tick_size * PRICE_LOCK_MAX_RANGE_TICKS + 1e-9)
        & minute["tr_ratio"].le(PRICE_LOCK_MAX_TR_RATIO)
        & minute["range_ratio"].le(PRICE_LOCK_MAX_RANGE_RATIO)
    )
    provisional: list[dict[str, int]] = []
    for index in minute.index[minute["confirmed"]]:
        start_index = int(index) - window + 1
        if start_index < 0 or int(minute.loc[start_index, "minute_segment"]) != int(minute.loc[index, "minute_segment"]):
            continue
        if provisional and start_index <= provisional[-1]["end_index"] + 1 and int(minute.loc[index, "minute_segment"]) == provisional[-1]["segment"]:
            provisional[-1]["end_index"] = int(index)
        else:
            provisional.append(
                {
                    "segment": int(minute.loc[index, "minute_segment"]),
                    "start_index": start_index,
                    "confirmation_index": int(index),
                    "end_index": int(index),
                }
            )
    events: list[dict[str, Any]] = []
    for number, item in enumerate(provisional, start=1):
        start = pd.Timestamp(minute.loc[item["start_index"], "datetime"])
        confirmation = pd.Timestamp(minute.loc[item["confirmation_index"], "datetime"]) + pd.Timedelta(seconds=45)
        seed = minute.loc[item["start_index"] : item["end_index"]]
        lock_price = float(seed["close"].median())
        tolerance = tick_size * PRICE_LOCK_MAX_RANGE_TICKS + 1e-9
        atom_segment = int(atoms.loc[atoms["datetime"].eq(start), "atom_segment"].iloc[0])
        candidates = atoms.loc[atoms["atom_segment"].eq(atom_segment) & atoms["datetime"].ge(start)]
        within_lock = candidates["high"].le(lock_price + tolerance) & candidates["low"].ge(lock_price - tolerance)
        outside = candidates.index[~within_lock]
        end_index = int(outside[0] - 1) if len(outside) else int(candidates.index[-1])
        event_atoms = atoms.loc[int(candidates.index[0]) : end_index]
        earlier = atoms.loc[
            atoms["datetime"].lt(start)
            & atoms["datetime"].ge(start - pd.Timedelta(hours=1))
            & atoms["volume"].gt(0)
        ]
        prior_price = None if earlier.empty else float(earlier["close"].median())
        event_type = "unclassified_price_lock"
        if prior_price is not None and lock_price < prior_price * 0.997:
            event_type = "lower_limit_lock_candidate"
        elif prior_price is not None and lock_price > prior_price * 1.003:
            event_type = "upper_limit_lock_candidate"
        if event_type == "unclassified_price_lock" and events:
            previous = events[-1]
            gap_hours = (start - pd.Timestamp(previous["end"])).total_seconds() / 3600
            if gap_hours <= 72 and abs(lock_price - float(previous["lock_price"])) <= tick_size * 2:
                event_type = str(previous["event_type"])
        row = minute.loc[item["confirmation_index"]]
        events.append(
            {
                "event_id": f"k200_price_lock_{number:02d}",
                "family": "k200_market_mechanism",
                "event_type": event_type,
                "label": "跌停锁价候选" if event_type == "lower_limit_lock_candidate" else "涨停锁价候选" if event_type == "upper_limit_lock_candidate" else "价格锁定候选",
                "start": start,
                "confirmation_time": confirmation,
                "end": pd.Timestamp(event_atoms["datetime"].iloc[-1]),
                "duration_minutes": float(len(event_atoms) * ATOM_SECONDS / 60),
                "apply_to_baseline": True,
                "reason_code": "k200_price_lock",
                "reason": "30分钟窗口内至少10个真实成交分钟，TR与价格跨度均不高于更早84小时正常中位数的5%，且价格跨度不超过4个tick。",
                "confidence": "high_shape_confidence",
                "lock_price": lock_price,
                "prior_price": prior_price,
                "tr_ratio_at_confirmation": float(row["tr_ratio"]),
                "range_ratio_at_confirmation": float(row["range_ratio"]),
            }
        )
    return events


def _pause_events(atoms: pd.DataFrame) -> list[dict[str, Any]]:
    observed = atoms["volume"].gt(0) & atoms["is_synthetic_empty_bar"].ne(1)
    zero = ~observed
    run = (atoms["atom_segment"].ne(atoms["atom_segment"].shift()) | zero.ne(zero.shift())).cumsum()
    events: list[dict[str, Any]] = []
    circuit_count = 0
    extended_count = 0
    for _, positions in atoms.loc[zero].groupby(run[zero], sort=False).groups.items():
        indexes = list(positions)
        start_index, end_index = int(indexes[0]), int(indexes[-1])
        if start_index == 0 or end_index + 1 >= len(atoms):
            continue
        if int(atoms.loc[start_index - 1, "atom_segment"]) != int(atoms.loc[start_index, "atom_segment"]):
            continue
        if int(atoms.loc[end_index + 1, "atom_segment"]) != int(atoms.loc[end_index, "atom_segment"]):
            continue
        before = atoms.loc[start_index - 1]
        after = atoms.loc[end_index + 1]
        gap_seconds = int((pd.Timestamp(after["datetime"]) - pd.Timestamp(before["datetime"])).total_seconds())
        zero_minutes = float(len(indexes) * ATOM_SECONDS / 60)
        if gap_seconds == CIRCUIT_BREAKER_GAP_SECONDS:
            circuit_count += 1
            start = pd.Timestamp(atoms.loc[start_index, "datetime"])
            events.append(
                {
                    "event_id": f"k200_circuit_{circuit_count:02d}",
                    "family": "k200_market_mechanism",
                    "event_type": "circuit_breaker_candidate",
                    "label": "熔断／集合竞价候选",
                    "start": start,
                    "confirmation_time": pd.Timestamp(after["datetime"]),
                    "halt_end": start + pd.Timedelta(minutes=CIRCUIT_HALT_MINUTES) - pd.Timedelta(seconds=15),
                    "call_auction_start": start + pd.Timedelta(minutes=CIRCUIT_HALT_MINUTES),
                    "end": pd.Timestamp(atoms.loc[end_index, "datetime"]),
                    "duration_minutes": float(gap_seconds / 60),
                    "apply_to_baseline": True,
                    "reason_code": "k200_circuit_breaker",
                    "reason": "连续15秒数据内部的前后两根真实成交柱相隔恰好30分钟；数据形状按20分钟暂停与10分钟集合竞价解释。",
                    "confidence": "high_timing_confidence",
                    "before_price": float(before["close"]),
                    "after_price": float(after["open"]),
                }
            )
        elif zero_minutes >= EXTENDED_INTERNAL_PAUSE_MINUTES:
            extended_count += 1
            start = pd.Timestamp(atoms.loc[start_index, "datetime"])
            events.append(
                {
                    "event_id": f"k200_extended_pause_{extended_count:02d}",
                    "family": "k200_market_mechanism",
                    "event_type": "extended_internal_pause_candidate",
                    "label": "长时内部停牌候选",
                    "start": start,
                    "confirmation_time": start + pd.Timedelta(minutes=EXTENDED_INTERNAL_PAUSE_MINUTES),
                    "end": pd.Timestamp(atoms.loc[end_index, "datetime"]),
                    "duration_minutes": zero_minutes,
                    "apply_to_baseline": False,
                    "reason_code": "uncertain_extended_pause",
                    "reason": "连续数据内部存在超过60分钟的完整零成交；仅凭OHLCV无法区分停牌、锁价无成交与数据来源缺失，因此只展示而不自动排除。",
                    "confidence": "medium_data_only_confidence",
                }
            )
    return events


def _low_activity_lifecycle(
    frame: pd.DataFrame,
    duration_atoms: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply the V4.4 pending/reinsert/confirm lifecycle one completed bar at a time."""
    output = frame.copy()
    row_count = len(output)
    states = np.full(row_count, LOW_ACTIVITY_STATE_NORMAL, dtype=object)
    pending_starts = np.full(row_count, np.datetime64("NaT"), dtype="datetime64[ns]")
    pending_counts = np.zeros(row_count, dtype=int)
    buffer_reinserted = np.zeros(row_count, dtype=bool)
    buffer_confirmed = np.zeros(row_count, dtype=bool)
    recovery_times = np.full(row_count, np.datetime64("NaT"), dtype="datetime64[ns]")
    confirmation_times = np.full(row_count, np.datetime64("NaT"), dtype="datetime64[ns]")
    baseline_excluded_from = np.full(row_count, np.datetime64("NaT"), dtype="datetime64[ns]")
    confirmed_active = np.zeros(row_count, dtype=bool)

    events: list[dict[str, Any]] = []
    pending_indices: list[int] = []
    pending_start: pd.Timestamp | None = None
    current_segment: int | None = None
    confirmed = False
    current_confirmation: pd.Timestamp | None = None
    current_event: dict[str, Any] | None = None

    def finish_confirmed_event(end_index: int, end_reason: str) -> None:
        nonlocal current_event
        if current_event is None:
            return
        current_event["end"] = pd.Timestamp(output.loc[end_index, "datetime"])
        current_event["duration_minutes"] = float(
            len(pending_indices) * ATOM_SECONDS / 60
        )
        current_event["end_reason"] = end_reason
        current_event = None

    def reset_run() -> None:
        nonlocal pending_indices, pending_start, confirmed, current_confirmation, current_event
        pending_indices = []
        pending_start = None
        confirmed = False
        current_confirmation = None
        current_event = None

    for index, row in output.iterrows():
        position = int(index)
        segment = int(row["atom_segment"])
        timestamp = pd.Timestamp(row["datetime"])
        low_volume = bool(row["low_volume_atom"])

        if current_segment is None or segment != current_segment:
            if pending_indices and confirmed:
                finish_confirmed_event(pending_indices[-1], "segment_break")
            reset_run()
            current_segment = segment

        if low_volume:
            if not pending_indices:
                pending_start = timestamp
            pending_indices.append(position)
            assert pending_start is not None
            pending_starts[position] = pending_start.to_datetime64()
            pending_counts[position] = len(pending_indices)
            if not confirmed and len(pending_indices) >= duration_atoms:
                confirmed = True
                buffer_confirmed[np.asarray(pending_indices, dtype=int)] = True
                confirmation = timestamp
                current_confirmation = confirmation
                resolved = np.asarray(pending_indices, dtype=int)
                confirmation_times[resolved] = confirmation.to_datetime64()
                baseline_excluded_from[resolved] = confirmation.to_datetime64()
                current_event = {
                    "event_id": f"universal_low_volume_{len(events) + 1:02d}",
                    "family": "universal_low_volume",
                    "event_type": "universal_low_volume",
                    "label": "通用低成交量",
                    "start": pending_start,
                    "confirmation_time": confirmation,
                    "end": timestamp,
                    "duration_minutes": float(len(pending_indices) * ATOM_SECONDS / 60),
                    "apply_to_baseline": True,
                    "reason_code": "universal_low_volume",
                    "reason": (
                        "每根15秒成交量均不高于更早84小时可信正成交量中位数的20%；"
                        "异常起点进入临时缓冲，连续30分钟后确认永久排除，确认前恢复则在"
                        "首根正常15秒bar完成后按时间顺序回填。"
                    ),
                    "confidence": "rule_confirmed",
                    "end_reason": "source_end",
                    "baseline_lifecycle": "pending_then_confirmed_excluded",
                }
                events.append(current_event)
            if confirmed:
                states[position] = LOW_ACTIVITY_STATE_CONFIRMED
                buffer_confirmed[position] = True
                confirmed_active[position] = True
                assert current_confirmation is not None
                confirmation_times[position] = current_confirmation.to_datetime64()
                baseline_excluded_from[position] = current_confirmation.to_datetime64()
                assert current_event is not None
                current_event["end"] = timestamp
                current_event["duration_minutes"] = float(
                    len(pending_indices) * ATOM_SECONDS / 60
                )
            else:
                states[position] = LOW_ACTIVITY_STATE_PENDING
            continue

        if pending_indices:
            resolved = np.asarray(pending_indices, dtype=int)
            recovery_times[resolved] = timestamp.to_datetime64()
            assert pending_start is not None
            pending_starts[position] = pending_start.to_datetime64()
            pending_counts[position] = len(pending_indices)
            recovery_times[position] = timestamp.to_datetime64()
            if confirmed:
                buffer_confirmed[resolved] = True
                buffer_confirmed[position] = True
                finish_confirmed_event(pending_indices[-1], "high_volume_atom")
            else:
                buffer_reinserted[resolved] = True
                buffer_reinserted[position] = True
            reset_run()

    if pending_indices and confirmed:
        finish_confirmed_event(pending_indices[-1], "source_end")

    output["low_activity_state"] = states
    output["pending_buffer_start"] = pd.to_datetime(pending_starts)
    output["pending_buffer_count"] = pending_counts
    output["buffer_reinserted"] = buffer_reinserted
    output["buffer_confirmed_excluded"] = buffer_confirmed
    output["recovery_confirmation_time"] = pd.to_datetime(recovery_times)
    output["low_activity_confirmation_time"] = pd.to_datetime(confirmation_times)
    output["baseline_excluded_from"] = pd.to_datetime(baseline_excluded_from)
    output["confirmed_low_activity_active"] = confirmed_active
    return output, events


def _volume_events(atoms: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame = atoms.sort_values("datetime").reset_index(drop=True).copy()
    frame["atom_segment"] = _segments(frame, ATOM_SECONDS)
    synthetic = frame["is_synthetic_empty_bar"].eq(1)
    frame["trustworthy_positive"] = (
        ~synthetic & frame["volume"].gt(0) & frame["trade_count"].gt(0)
    )
    threshold = pd.Series(np.nan, index=frame.index, dtype=float)
    reference_count = pd.Series(0, index=frame.index, dtype=int)
    elapsed = pd.Timedelta(hours=REFERENCE_HOURS)
    for _, group in frame.groupby("volume_semantic", sort=False):
        values = group["volume"].where(group["trustworthy_positive"])
        indexed = pd.Series(values.to_numpy(float), index=pd.DatetimeIndex(group["datetime"]))
        rolling = indexed.rolling(elapsed, closed="left", min_periods=1)
        counts = rolling.count()
        medians = rolling.median() * VOLUME_MEDIAN_RATIO
        medians = medians.where(counts.ge(MINIMUM_REFERENCE_POSITIVE_ATOMS))
        threshold.loc[group.index] = medians.to_numpy(float)
        reference_count.loc[group.index] = counts.fillna(0).to_numpy(int)
    frame["volume_threshold"] = threshold
    frame["reference_positive_count"] = reference_count
    frame["low_volume_atom"] = threshold.notna() & (
        (frame["trustworthy_positive"] & frame["volume"].le(threshold))
        | (synthetic & frame["volume"].eq(0))
    )
    duration_atoms = LOW_VOLUME_DURATION_MINUTES * 60 // ATOM_SECONDS
    return _low_activity_lifecycle(frame, duration_atoms)


def _event_mask(frame: pd.DataFrame, event: dict[str, Any]) -> pd.Series:
    return frame["datetime"].between(pd.Timestamp(event["start"]), pd.Timestamp(event["end"]))


def detect_low_activity(atoms: pd.DataFrame, instrument: str) -> LowActivityResult:
    frame = atoms.sort_values("datetime").reset_index(drop=True).copy()
    instrument_key = instrument.strip().upper()
    tick_size: float | None = None
    mechanism_events: list[dict[str, Any]] = []
    mechanism_reference_mask = pd.Series(False, index=frame.index, dtype=bool)
    if instrument_key == "K200":
        tick_size = infer_tick_size(frame)
        mechanism_events = sorted(
            _price_lock_events(frame, tick_size) + _pause_events(frame),
            key=lambda event: (pd.Timestamp(event["start"]), str(event["event_type"])),
        )
        for event in mechanism_events:
            if bool(event.get("apply_to_baseline", False)):
                mechanism_reference_mask |= _event_mask(frame, event)

    universal_input = frame.loc[~mechanism_reference_mask].copy().reset_index(drop=True)
    universal_frame, universal_events = _volume_events(universal_input)
    events = sorted(
        universal_events + mechanism_events,
        key=lambda event: (pd.Timestamp(event["start"]), str(event["event_type"])),
    )

    output = frame.copy()
    output["universal_low_volume_excluded"] = False
    output["k200_price_lock_excluded"] = False
    output["k200_circuit_breaker_excluded"] = False
    output["baseline_excluded"] = False
    event_ids: list[list[str]] = [[] for _ in range(len(output))]
    reasons: list[list[str]] = [[] for _ in range(len(output))]
    for event in events:
        mask = _event_mask(output, event)
        event_type = str(event["event_type"])
        if event_type == "universal_low_volume":
            output.loc[mask, "universal_low_volume_excluded"] = True
        elif "price_lock" in event_type or "limit_lock" in event_type:
            output.loc[mask, "k200_price_lock_excluded"] = True
        elif event_type == "circuit_breaker_candidate":
            output.loc[mask, "k200_circuit_breaker_excluded"] = True
        if bool(event.get("apply_to_baseline", False)):
            output.loc[mask, "baseline_excluded"] = True
        for index in output.index[mask]:
            event_ids[int(index)].append(str(event["event_id"]))
            reasons[int(index)].append(str(event["reason_code"]))
    output["filter_event_ids"] = ["|".join(values) for values in event_ids]
    output["filter_reason_codes"] = ["|".join(values) for values in reasons]
    # Publish a policy-neutral marker. The backtest selects whether the marker
    # participates in baseline eligibility.
    output["eligible_if_excluding_marked"] = ~output["baseline_excluded"].astype(bool)

    universal_diag = universal_frame.set_index("datetime")
    output["volume_threshold"] = output["datetime"].map(universal_diag["volume_threshold"])
    output["low_volume_atom"] = output["datetime"].map(universal_diag["low_volume_atom"]).fillna(False).astype(bool)
    output["low_activity_state"] = (
        output["datetime"].map(universal_diag["low_activity_state"])
        .fillna(LOW_ACTIVITY_STATE_NORMAL)
        .astype(str)
    )
    output["pending_buffer_start"] = pd.to_datetime(
        output["datetime"].map(universal_diag["pending_buffer_start"])
    )
    output["pending_buffer_count"] = (
        output["datetime"].map(universal_diag["pending_buffer_count"])
        .fillna(0)
        .astype(int)
    )
    for column in ("buffer_reinserted", "buffer_confirmed_excluded"):
        output[column] = (
            output["datetime"].map(universal_diag[column]).fillna(False).astype(bool)
        )
    output["recovery_confirmation_time"] = pd.to_datetime(
        output["datetime"].map(universal_diag["recovery_confirmation_time"])
    )
    output["low_activity_confirmation_time"] = pd.to_datetime(
        output["datetime"].map(universal_diag["low_activity_confirmation_time"])
    )
    output["baseline_excluded_from"] = pd.to_datetime(
        output["datetime"].map(universal_diag["baseline_excluded_from"])
    )
    output["confirmed_low_activity_active"] = (
        output["datetime"]
        .map(universal_diag["confirmed_low_activity_active"])
        .fillna(False)
        .astype(bool)
    )
    baseline_available_from = pd.to_datetime(output["datetime"]).copy()
    reinserted = output["buffer_reinserted"].astype(bool)
    baseline_available_from.loc[reinserted] = output.loc[
        reinserted, "recovery_confirmation_time"
    ]
    unresolved_pending = (
        output["low_activity_state"].eq(LOW_ACTIVITY_STATE_PENDING)
        & ~reinserted
    )
    baseline_available_from.loc[unresolved_pending] = pd.NaT
    baseline_available_from.loc[
        ~output["eligible_if_excluding_marked"].astype(bool)
    ] = pd.NaT
    output["baseline_available_from"] = baseline_available_from
    applied = [event for event in events if bool(event.get("apply_to_baseline", False))]
    summary = {
        "rule_version": FILTER_RULE_VERSION,
        "instrument": instrument_key,
        "atom_count": int(len(output)),
        "baseline_excluded_atom_count": int(output["baseline_excluded"].sum()),
        "baseline_excluded_minutes": float(output["baseline_excluded"].sum() * ATOM_SECONDS / 60),
        "eligible_if_excluding_marked_atom_count": int(
            output["eligible_if_excluding_marked"].sum()
        ),
        "universal_event_count": int(sum(event["event_type"] == "universal_low_volume" for event in events)),
        "price_lock_event_count": int(sum("price_lock" in str(event["event_id"]) for event in events)),
        "circuit_event_count": int(sum(event["event_type"] == "circuit_breaker_candidate" for event in events)),
        "extended_pause_display_only_count": int(sum(event["event_type"] == "extended_internal_pause_candidate" for event in events)),
        "applied_event_count": int(len(applied)),
        "buffer_reinserted_atom_count": int(output["buffer_reinserted"].sum()),
        "buffer_confirmed_excluded_atom_count": int(output["buffer_confirmed_excluded"].sum()),
        "confirmed_low_activity_active_atom_count": int(
            output["confirmed_low_activity_active"].sum()
        ),
        "recovery_confirmation_count": int((
            output["recovery_confirmation_time"].notna()
            & output["low_activity_state"].eq(LOW_ACTIVITY_STATE_NORMAL)
        ).sum()),
        "baseline_available_immediately_atom_count": int(
            output["baseline_available_from"].eq(output["datetime"]).sum()
        ),
        "baseline_available_after_recovery_atom_count": int((
            output["baseline_available_from"].notna()
            & output["baseline_available_from"].gt(output["datetime"])
        ).sum()),
        "baseline_never_available_atom_count": int(
            output["baseline_available_from"].isna().sum()
        ),
        "tick_size": tick_size,
    }
    return LowActivityResult(output, events, tick_size, summary)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or (not isinstance(value, (str, bool)) and pd.isna(value)):
        return None
    return value
