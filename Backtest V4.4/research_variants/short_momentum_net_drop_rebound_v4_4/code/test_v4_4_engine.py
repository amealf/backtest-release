from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from v4_4_engine import (  # noqa: E402
    BASELINE_FILTER_ID,
    BASELINE_SAMPLING_CONFIRMED_LOW_ACTIVITY_GATE,
    BASELINE_SAMPLING_EXCLUDE_MARKED,
    COMBINED_STRATEGY_ID,
    COMBINED_TRADE_AUDIT_SCHEMA_ID,
    COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
    DOWNSIDE_SPEED_EXIT_REASON,
    ENTRY_EXECUTION_REJECT_SYNTHETIC_FILL,
    ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    ENTRY_FILL_CALCULATED_THRESHOLD,
    ENTRY_METHOD_ROLLING,
    EVENTS_DEFAULT,
    MAX_REAL_TRADE_WAIT_BARS,
    QUIET_ACTIVITY_ATOMS_DEFAULT,
    REPOSITORY_ROOT,
    REBOUND_BASELINE_POLICY_ID,
    STRATEGY_ID,
    TRADE_AUDIT_SCHEMA_ID,
    TRADE_AUDIT_SCHEMA_VERSION,
    VERSION_LABEL,
    Combo,
    baseline_atom_indices,
    entry_baseline,
    entry_signal_qualifies,
    load_bars,
    _window_net_drop,
    simulate_combo,
)


def test_optional_legacy_cli_inputs_are_repository_local_placeholders() -> None:
    external_inputs = REPOSITORY_ROOT / "runtime_inputs" / "legacy"
    assert QUIET_ACTIVITY_ATOMS_DEFAULT == external_inputs / "canonical_volume_atoms.csv"
    assert EVENTS_DEFAULT == external_inputs / "selected_events.csv"


def _frame(periods: int = 9) -> pd.DataFrame:
    opens = [10.0, 10.0, 11.0, 11.0, 9.0, 8.0, 8.0, 8.0, 8.0] + [8.0] * max(0, periods - 9)
    highs = [10.0, 10.0, 11.0, 11.0, 9.0, 8.0, 8.0, 8.0, 8.0] + [8.0] * max(0, periods - 9)
    lows = [9.0, 9.0, 10.0, 10.0, 8.0, 7.0, 7.0, 8.0, 8.0] + [8.0] * max(0, periods - 9)
    times = pd.date_range("2026-01-01 09:00:00", periods=periods, freq="15s")
    frame = pd.DataFrame(
        {
            "datetime": times,
            "open": opens[:periods],
            "high": highs[:periods],
            "low": lows[:periods],
            "close": lows[:periods],
            "volume": 10.0,
            "trade_count": 1.0,
            "is_synthetic_empty_bar": False,
        }
    )
    frame["continuous"] = [False] + [True] * (periods - 1)
    frame["continuous_segment_id"] = 1
    frame["continuous_run"] = np.arange(1, periods + 1)
    previous = np.r_[np.nan, frame.close.to_numpy()[:-1]]
    frame["tr15"] = np.maximum(frame.high, previous) - np.minimum(frame.low, previous)
    frame.loc[0, "tr15"] = np.nan
    frame["quiet_filter_event_stage"] = 0
    frame["quiet_filter_stage"] = 0
    frame["buffer_confirmed_excluded"] = False
    frame["low_volume_atom"] = False
    frame["baseline_excluded"] = False
    frame["eligible_if_excluding_marked"] = True
    frame["baseline_available_from"] = frame["datetime"]
    frame["baseline_excluded_from"] = pd.NaT
    frame["confirmed_low_activity_active"] = False
    frame["low_activity_state"] = "normal"
    frame.attrs["mechanism_filter_events"] = []
    return frame


def _combo(**overrides: object) -> Combo:
    values: dict[str, object] = {
        "method": ENTRY_METHOD_ROLLING,
        "e": 4,
        "bh": 2,
        "trw": 1,
        "k": 0.1,
        "w": 3,
        "m": 0.1,
    }
    values.update(overrides)
    return Combo(**values)  # type: ignore[arg-type]


def test_long_synthetic_flat_run_cannot_enter_on_zero_threshold() -> None:
    periods = 40
    frame = _frame(periods)
    for field in ("open", "high", "low", "close"):
        frame[field] = 10.0
    frame["tr15"] = 0.0
    frame["is_synthetic_empty_bar"] = True
    trades = simulate_combo(
        frame, _combo(e=4, bh=2, trw=1, k=1.0),
        frame.datetime.iloc[0], frame.datetime.iloc[-1],
    )
    assert trades == []
    assert entry_signal_qualifies(2.0, 2.0, 1.0) is True


def test_load_bars_without_filter_accepts_empty_reinserted_set(tmp_path: Path) -> None:
    source = tmp_path / "instrument_15s.csv"
    pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01 09:00:00", periods=3, freq="15s"),
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [1.0, 1.0, 1.0],
            "trade_count": [1, 1, 1],
            "is_synthetic_empty_bar": [False, False, False],
        }
    ).to_csv(source, index=False)

    loaded = load_bars(source, None)

    assert not loaded["buffer_reinserted"].any()
    pd.testing.assert_series_equal(
        loaded["baseline_available_from"],
        loaded["datetime"],
        check_names=False,
        check_dtype=False,
    )


def test_v4_4_identity_is_separate_and_execution_parameters_enter_combo_id() -> None:
    default = _combo()
    rejected = _combo(entry_execution_policy=ENTRY_EXECUTION_REJECT_SYNTHETIC_FILL)
    slipped = _combo(entry_slippage=0.25)
    combined = _combo(speed_window_bars=320)
    exclude_marked = _combo(
        baseline_sampling_policy=BASELINE_SAMPLING_EXCLUDE_MARKED
    )
    assert VERSION_LABEL == "V4.4"
    assert "v4_4" in STRATEGY_ID and BASELINE_FILTER_ID.endswith("v4_4")
    assert "max_completed_w_drop_rebound" in STRATEGY_ID
    assert REBOUND_BASELINE_POLICY_ID == "max_completed_w_h_bounded_open_to_low_v2"
    assert default.combo_id.startswith("v4_4_")
    assert len({
        default.combo_id,
        rejected.combo_id,
        slipped.combo_id,
        combined.combo_id,
        exclude_marked.combo_id,
    }) == 5
    assert exclude_marked.strategy_id != default.strategy_id
    assert "_sx1_s320_rx1_" in combined.combo_id
    assert combined.strategy_id == COMBINED_STRATEGY_ID
    assert default.speed_window_bars == 0 and not default.speed_exit_enabled


def test_combined_exit_uses_zero_extension_after_rebound_check() -> None:
    frame = _frame()
    trade = simulate_combo(
        frame,
        _combo(m=100.0, speed_window_bars=2),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )[0]
    assert trade["exit_reason"] == DOWNSIDE_SPEED_EXIT_REASON
    assert trade["exit_index"] == 7
    assert trade["exit_price"] == frame.close.iloc[7]
    assert trade["exit_price_basis"] == "current_bar_close"
    assert trade["speed_reference_index"] == 5
    assert trade["speed_reference_low"] == 7.0
    assert trade["speed_current_low"] == 7.0
    assert trade["speed_extension"] == 0.0
    assert trade["speed_check_price_basis"] == "bar_close"
    assert trade["trade_audit_schema_version"] == COMBINED_TRADE_AUDIT_SCHEMA_VERSION
    assert trade["trade_audit_schema_id"] == COMBINED_TRADE_AUDIT_SCHEMA_ID


def test_all_window_baseline_uses_every_finite_atom() -> None:
    tr = np.array([np.nan, 1.0, 2.0, 100.0, 200.0, 3.0])
    eligible = np.array([1, 2, 3, 4, 5])
    segments = np.ones(len(tr), dtype=int)
    combo = _combo(bh=2)
    assert baseline_atom_indices(4, 2, eligible, segments).tolist() == [3, 4]
    assert entry_baseline(tr, 4, combo, eligible, segments) == 150.0
    assert baseline_atom_indices(5, 2, eligible, segments).tolist() == [4, 5]
    assert entry_baseline(tr, 5, combo, eligible, segments) == 101.5


def test_exclude_marked_uses_only_atoms_available_by_calculation_time() -> None:
    eligible = np.array([1, 2, 3, 4], dtype=int)
    segments = np.ones(5, dtype=int)
    available_from = np.array([0, 1, 4, 3, 4], dtype=int)
    assert baseline_atom_indices(
        3, 2, eligible, segments, available_from, 3
    ).tolist() == [1, 3]
    assert baseline_atom_indices(
        3, 2, eligible, segments, available_from, 4
    ).tolist() == [2, 3]


def test_confirmed_gate_excludes_pending_run_only_from_confirmation_time() -> None:
    eligible = np.array([1, 2, 3, 4], dtype=int)
    segments = np.ones(5, dtype=int)
    excluded_from = np.array([6, 4, 4, 6, 6], dtype=int)
    assert baseline_atom_indices(
        3, 3, eligible, segments, None, 3, excluded_from
    ).tolist() == [1, 2, 3]
    assert baseline_atom_indices(
        3, 1, eligible, segments, None, 4, excluded_from
    ).tolist() == [3]


def test_confirmed_gate_blocks_entries_but_pending_phase_has_no_effect() -> None:
    pending_frame = _frame()
    pending_frame.loc[1:3, "low_activity_state"] = "pending_low_activity_buffer"
    pending_trades = simulate_combo(
        pending_frame,
        _combo(baseline_sampling_policy=BASELINE_SAMPLING_CONFIRMED_LOW_ACTIVITY_GATE),
        pending_frame.datetime.iloc[0],
        pending_frame.datetime.iloc[-1],
    )
    assert pending_trades

    confirmed_frame = pending_frame.copy()
    confirmed_frame.loc[3:, "confirmed_low_activity_active"] = True
    confirmed_trades = simulate_combo(
        confirmed_frame,
        _combo(baseline_sampling_policy=BASELINE_SAMPLING_CONFIRMED_LOW_ACTIVITY_GATE),
        confirmed_frame.datetime.iloc[0],
        confirmed_frame.datetime.iloc[-1],
    )
    assert confirmed_trades == []


def test_confirmation_cancels_an_unfilled_entry_order() -> None:
    frame = _frame()
    frame.loc[3, ["volume", "trade_count"]] = 0
    frame.loc[3, "is_synthetic_empty_bar"] = True
    frame.loc[4:, "confirmed_low_activity_active"] = True
    trades = simulate_combo(
        frame,
        _combo(baseline_sampling_policy=BASELINE_SAMPLING_CONFIRMED_LOW_ACTIVITY_GATE),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )
    assert trades == []


def test_filtered_bar_still_participates_in_h_and_signal_detection() -> None:
    frame = _frame()
    control = simulate_combo(
        frame, _combo(bh=2), frame.datetime.iloc[0], frame.datetime.iloc[-1]
    )[0]
    frame.loc[2, "baseline_excluded"] = True
    frame.loc[2, "eligible_if_excluding_marked"] = False
    trade = simulate_combo(
        frame, _combo(bh=2), frame.datetime.iloc[0], frame.datetime.iloc[-1]
    )[0]
    assert trade["h_index"] == 2
    assert trade["signal_index"] == 3
    assert trade["entry_baseline_value"] == control["entry_baseline_value"]
    assert trade["baseline_excluded_atom_count"] == 1


def test_exclude_marked_changes_only_baseline_sample_eligibility() -> None:
    frame = _frame()
    frame.loc[2, "baseline_excluded"] = True
    frame.loc[2, "eligible_if_excluding_marked"] = False
    all_window = simulate_combo(
        frame,
        _combo(bh=1, k=0.75),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )[0]
    exclude_marked = simulate_combo(
        frame,
        _combo(
            bh=1,
            k=0.75,
            baseline_sampling_policy=BASELINE_SAMPLING_EXCLUDE_MARKED,
        ),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )[0]
    assert all_window["h_index"] == exclude_marked["h_index"] == 2
    assert exclude_marked["signal_index"] < all_window["signal_index"]
    assert exclude_marked["baseline_sampling_policy"] == "exclude_marked"


def test_pending_low_activity_count_is_a_real_audit_measure() -> None:
    frame = _frame()
    frame.loc[1, "low_activity_state"] = "pending_low_activity_buffer"
    trade = simulate_combo(
        frame,
        _combo(bh=2),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )[0]
    assert trade["baseline_pending_atom_count"] == 1


def test_wait_policy_uses_the_next_real_trade_open() -> None:
    frame = _frame()
    frame.loc[3:4, ["volume", "trade_count"]] = 0
    frame.loc[3:4, "is_synthetic_empty_bar"] = True
    trade = simulate_combo(frame, _combo(m=100.0), frame.datetime.iloc[0], frame.datetime.iloc[-1])[0]
    assert trade["initial_entry_index"] == trade["signal_index"] == 3
    assert trade["entry_index"] == 5
    assert trade["entry_price"] == frame.open.iloc[5]
    assert trade["entry_wait_bar_count"] == 2
    assert trade["entry_fill_source"] == "waited_real_trade_open"
    assert trade["entry_trigger_price"] == 11.0 - 0.1 * trade["entry_baseline_value"]
    assert trade["entry_fill_price"] == trade["entry_price"]
    assert trade["entry_price_basis"] == "waited_real_trade_bar_open"
    assert trade["entry_gap_adjusted"] is None
    assert np.isnan(trade["entry_gap_slippage"])


def test_pending_entry_uses_h_bounded_completed_signal_candidate() -> None:
    frame = _frame()
    frame.loc[3:4, ["volume", "trade_count"]] = 0.0
    frame.loc[3:4, "is_synthetic_empty_bar"] = True
    frame.loc[3, ["low", "close"]] = [9.0, 9.0]
    frame.loc[5, ["open", "high", "low", "close"]] = [8.0, 9.0, 7.0, 8.5]
    trade = simulate_combo(
        frame,
        _combo(m=1.0),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )[0]
    assert trade["signal_index"] == 3
    assert trade["entry_index"] == 5
    assert trade["h_index"] == 2
    assert trade["rebound_max_w_drop"] == 4.0
    assert trade["rebound_window_start_index"] == 3
    assert trade["rebound_window_end_index"] == 5
    assert trade["rebound_window_start_index"] >= trade["h_index"]
    assert trade["trade_audit_schema_version"] == TRADE_AUDIT_SCHEMA_VERSION
    assert trade["trade_audit_schema_id"] == TRADE_AUDIT_SCHEMA_ID


def test_pending_entry_uses_first_real_open_without_retrigger_or_structure_cancel() -> None:
    frame = _frame()
    frame.loc[3, ["volume", "trade_count"]] = 0
    frame.loc[3, "is_synthetic_empty_bar"] = True
    frame.loc[4, ["open", "high", "low", "close"]] = [12, 13, 12, 12]
    trade = simulate_combo(
        frame, _combo(m=100.0), frame.datetime.iloc[0], frame.datetime.iloc[-1]
    )[0]
    assert frame.low.iloc[4] > trade["entry_trigger_price"]
    assert frame.high.iloc[4] > frame.high.iloc[int(trade["h_index"])]
    assert trade["entry_index"] == 4
    assert trade["entry_price"] == frame.open.iloc[4]
    assert trade["entry_fill_source"] == "waited_real_trade_open"


def test_synthetic_rebound_exit_waits_for_next_real_trade_open() -> None:
    frame = _frame()
    frame.loc[6, ["volume", "trade_count"]] = 0
    frame.loc[6, "is_synthetic_empty_bar"] = True
    frame.loc[7, ["open", "high", "low", "close"]] = [10, 10, 10, 10]
    trade = simulate_combo(frame, _combo(), frame.datetime.iloc[0], frame.datetime.iloc[-1])[0]
    assert trade["exit_reason"] == "rebound_threshold"
    assert trade["pending_exit"] is True
    assert trade["pending_exit_trigger_index"] == 6
    assert trade["exit_index"] == 7
    assert trade["exit_price"] == frame.open.iloc[7]
    assert trade["pending_exit_wait_bar_count"] == 1
    assert trade["pending_exit_fill_policy"] == "next_real_trade_bar_open"
    assert trade["exit_bar_synthetic"] is False


def test_synthetic_speed_exit_waits_for_next_real_trade_open() -> None:
    frame = _frame()
    frame.loc[7, ["volume", "trade_count"]] = 0
    frame.loc[7, "is_synthetic_empty_bar"] = True
    frame.loc[8, ["open", "high", "low", "close"]] = [9.0, 9.0, 9.0, 9.0]
    trade = simulate_combo(
        frame,
        _combo(m=100.0, speed_window_bars=2),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )[0]
    assert trade["exit_reason"] == DOWNSIDE_SPEED_EXIT_REASON
    assert trade["pending_exit"] is True
    assert trade["pending_exit_trigger_index"] == 7
    assert trade["exit_index"] == 8
    assert trade["exit_price"] == frame.open.iloc[8]
    assert trade["pending_exit_wait_bar_count"] == 1
    assert trade["pending_exit_fill_policy"] == "next_real_trade_bar_open"


def test_zero_volume_non_synthetic_fill_also_waits() -> None:
    frame = _frame()
    frame.loc[3, ["volume", "trade_count"]] = 0
    trade = simulate_combo(frame, _combo(m=100.0), frame.datetime.iloc[0], frame.datetime.iloc[-1])[0]
    assert trade["initial_entry_bar_synthetic"] is False
    assert trade["initial_entry_index"] == trade["signal_index"] == 3
    assert trade["entry_index"] == 4
    assert trade["entry_price"] == frame.open.iloc[4]


def test_reject_policy_cancels_an_invalid_initial_fill() -> None:
    frame = _frame(6)
    frame.loc[3, ["volume", "trade_count"]] = 0
    frame.loc[3, "is_synthetic_empty_bar"] = True
    trades = simulate_combo(
        frame,
        _combo(entry_execution_policy=ENTRY_EXECUTION_REJECT_SYNTHETIC_FILL),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )
    assert all(trade["signal_index"] != 3 for trade in trades)


def test_real_time_gap_cancels_a_waiting_signal() -> None:
    frame = _frame(6)
    frame.loc[3, ["volume", "trade_count"]] = 0
    frame.loc[3, "is_synthetic_empty_bar"] = True
    frame.loc[4, "datetime"] = frame.datetime.iloc[4] + pd.Timedelta(seconds=15)
    frame.loc[4, "continuous"] = False
    frame.loc[4, "continuous_segment_id"] = 2
    frame.loc[4, "continuous_run"] = 1
    trades = simulate_combo(
        frame, _combo(), frame.datetime.iloc[0], frame.datetime.iloc[-1]
    )
    assert all(trade["signal_index"] != 3 for trade in trades)


def test_wait_limit_cancels_before_the_121st_candidate_bar() -> None:
    periods = 130
    frame = _frame(periods)
    initial = 3
    last_allowed = initial + MAX_REAL_TRADE_WAIT_BARS - 1
    frame.loc[initial:last_allowed, ["volume", "trade_count"]] = 0
    frame.loc[initial:last_allowed, "is_synthetic_empty_bar"] = True
    frame.loc[last_allowed + 1 :, ["open", "high", "low", "close"]] = 8
    trades = simulate_combo(frame, _combo(m=100.0), frame.datetime.iloc[0], frame.datetime.iloc[-1])
    assert all(trade["signal_index"] != 3 for trade in trades)


def test_short_entry_slippage_is_adverse_and_persisted() -> None:
    frame = _frame()
    baseline_trade = simulate_combo(frame, _combo(m=100.0), frame.datetime.iloc[0], frame.datetime.iloc[-1])[0]
    slipped_trade = simulate_combo(
        frame,
        _combo(m=100.0, entry_slippage=0.25),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )[0]
    assert slipped_trade["entry_price_before_slippage"] == baseline_trade["entry_price"]
    assert slipped_trade["entry_price"] == baseline_trade["entry_price"] - 0.25
    assert slipped_trade["entry_slippage"] == 0.25


def test_calculated_fill_waits_from_an_invalid_signal_bar() -> None:
    frame = _frame()
    frame.loc[3, ["volume", "trade_count"]] = 0
    frame.loc[3, "is_synthetic_empty_bar"] = True
    trade = simulate_combo(
        frame,
        _combo(entry_fill_mode=ENTRY_FILL_CALCULATED_THRESHOLD, m=100.0),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )[0]
    assert trade["initial_entry_index"] == trade["signal_index"] == 3
    assert trade["entry_index"] == 4
    assert trade["entry_price"] == frame.open.iloc[4]
    assert trade["entry_fill_source"] == "waited_real_trade_open"
    assert trade["entry_gap_adjusted"] is None
    assert trade["entry_price_basis"] == "waited_real_trade_bar_open"


def test_calculated_threshold_entry_audit_distinguishes_threshold_and_down_gap() -> None:
    regular = _frame()
    regular_trade = simulate_combo(
        regular,
        _combo(entry_fill_mode=ENTRY_FILL_CALCULATED_THRESHOLD, m=100.0),
        regular.datetime.iloc[0],
        regular.datetime.iloc[-1],
    )[0]
    assert regular_trade["entry_price_basis"] == "calculated_entry_threshold"
    assert regular_trade["entry_gap_adjusted"] is False
    assert regular_trade["entry_gap_slippage"] == 0.0
    assert regular_trade["entry_fill_price"] == regular_trade["entry_trigger_price"]

    down_gap = _frame()
    down_gap.loc[3, ["open", "high", "low", "close"]] = [9, 10, 8, 9]
    gap_trade = simulate_combo(
        down_gap,
        _combo(entry_fill_mode=ENTRY_FILL_CALCULATED_THRESHOLD, m=100.0),
        down_gap.datetime.iloc[0],
        down_gap.datetime.iloc[-1],
    )[0]
    assert gap_trade["entry_price_basis"] == "signal_bar_open_after_down_gap"
    assert gap_trade["entry_gap_adjusted"] is True
    assert gap_trade["entry_fill_price"] == 9.0
    assert gap_trade["entry_gap_slippage"] == gap_trade["entry_trigger_price"] - 9.0


def test_rebound_audit_uses_close_on_strict_new_low() -> None:
    frame = _frame()
    frame.loc[5, "close"] = 8.0
    trade = simulate_combo(frame, _combo(), frame.datetime.iloc[0], frame.datetime.iloc[-1])[0]
    assert trade["exit_index"] == 5
    assert trade["rebound_check_price"] == frame.close.iloc[5]
    assert trade["rebound_check_price_basis"] == "bar_close_after_strict_new_low"
    assert trade["exit_price_basis"] == "same_bar_close_after_strict_new_low_confirmation"
    assert trade["exit_price"] == frame.close.iloc[5]
    assert trade["rebound_gap_adjusted"] is False
    assert trade["rebound_gap_slippage"] == 0.0


def test_w_candidate_uses_available_continuous_prefix_up_to_w() -> None:
    frame = _frame()
    assert _window_net_drop(frame, 1, 100) == 1.0
    assert _window_net_drop(frame, 5, 3) == 4.0
    assert _window_net_drop(frame, 5, 100, h_index=4) == 2.0


def test_w_candidate_is_start_open_to_end_low_not_internal_ordered_drop() -> None:
    frame = _frame(4)
    frame.loc[:, "open"] = [100.0, 110.0, 109.0, 108.0]
    frame.loc[:, "high"] = [100.0, 110.0, 109.0, 108.0]
    frame.loc[:, "low"] = [100.0, 109.0, 108.0, 95.0]
    frame.loc[:, "continuous_run"] = [1, 2, 3, 4]

    assert _window_net_drop(frame, 3, 4) == 5.0


def test_available_prefix_maximum_can_remain_effective_below_requested_w() -> None:
    frame = _frame()
    trade = simulate_combo(
        frame,
        _combo(w=100, m=100.0),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )[0]
    applied = [
        _window_net_drop(frame, index, 100, h_index=int(trade["h_index"]))
        for index in range(int(trade["signal_index"]), int(trade["exit_index"]) + 1)
    ]

    assert trade["rebound_max_w_drop"] == max(value for value in applied if np.isfinite(value))
    assert trade["rebound_window_observed_bar_count"] < 100


def test_long_lower_wick_uses_prior_completed_max_for_same_bar_exit() -> None:
    frame = _frame()
    frame.loc[5, ["high", "close"]] = [10.5, 10.5]
    trade = simulate_combo(
        frame,
        _combo(m=1.0),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )[0]
    assert trade["exit_index"] == 5
    assert trade["active_low"] == 7.0
    assert trade["rebound_net_drop"] == 3.0
    assert trade["rebound_exit_bar_candidate"] == 4.0
    assert trade["rebound_latest_applied_candidate"] == 3.0
    assert trade["rebound_candidates_effective_through_index"] == 4
    assert trade["rebound_baseline_policy_id"] == REBOUND_BASELINE_POLICY_ID
    assert trade["rebound_threshold"] == 10.0
    assert trade["exit_price"] == frame.close.iloc[5] == 10.5
    assert trade["rebound_window_start_index"] == 2
    assert trade["rebound_window_end_index"] == 4


def test_completed_non_new_low_bar_can_raise_max_for_later_bars() -> None:
    frame = _frame()
    frame.loc[4, ["open", "high", "low", "close"]] = [20.0, 20.0, 8.0, 8.0]
    trade = simulate_combo(
        frame,
        _combo(m=100.0),
        frame.datetime.iloc[0],
        frame.datetime.iloc[-1],
    )[0]
    assert trade["exit_reason"] == "segment_end"
    assert trade["rebound_max_w_drop"] == 13.0
    assert trade["rebound_window_start_index"] == 4
    assert trade["rebound_window_end_index"] == 6
    assert trade["rebound_window_observed_bar_count"] == 3
    assert trade["rebound_candidates_effective_through_index"] == trade["exit_index"]


def test_rebound_audit_records_up_gap_exit_fill() -> None:
    frame = _frame()
    frame.loc[6, ["open", "high", "low", "close"]] = [9, 10, 8, 9]
    trade = simulate_combo(frame, _combo(), frame.datetime.iloc[0], frame.datetime.iloc[-1])[0]
    assert trade["exit_index"] == 6
    assert trade["rebound_check_price"] == frame.open.iloc[6]
    assert trade["rebound_check_price_basis"] == "bar_open_at_or_above_prior_trigger"
    assert trade["exit_fill_price"] == frame.open.iloc[6]
    assert trade["exit_price_basis"] == "exit_bar_open_at_or_above_rebound_trigger"
    assert trade["rebound_gap_adjusted"] is True
    assert trade["rebound_gap_slippage"] == trade["exit_fill_price"] - trade["rebound_trigger_price"]


def test_rebound_exits_when_open_or_high_equals_theoretical_trigger() -> None:
    open_touch = _frame()
    open_touch.loc[6, ["open", "high", "low", "close"]] = [7.4, 7.4, 7.2, 7.2]
    open_trade = simulate_combo(
        open_touch, _combo(), open_touch.datetime.iloc[0], open_touch.datetime.iloc[-1]
    )[0]
    assert open_trade["exit_index"] == 6
    assert open_trade["exit_price"] == open_trade["rebound_trigger_price"] == 7.4
    assert open_trade["rebound_gap_adjusted"] is True

    high_touch = _frame()
    high_touch.loc[6, ["open", "high", "low", "close"]] = [7.2, 7.4, 7.2, 7.2]
    high_trade = simulate_combo(
        high_touch, _combo(), high_touch.datetime.iloc[0], high_touch.datetime.iloc[-1]
    )[0]
    assert high_trade["exit_index"] == 6
    assert high_trade["exit_price"] == high_trade["rebound_trigger_price"] == 7.4
    assert high_trade["rebound_gap_adjusted"] is False


def test_sample_end_forces_close_without_reading_later_prices() -> None:
    frame = _frame(12)
    trade = simulate_combo(
        frame,
        _combo(m=100.0),
        frame.datetime.iloc[0],
        frame.datetime.iloc[6],
    )[0]
    assert trade["exit_index"] == 6
    assert trade["exit_reason"] == "segment_end"
    assert trade["exit_price_basis"] == "sample_end_bar_close"


def test_all_window_baseline_matches_contiguous_finite_history() -> None:
    eligible = np.arange(1, 11, dtype=int)
    segments = np.ones(11, dtype=int)
    for index in range(2, 11):
        expected_pool = eligible[eligible <= index]
        size = min(3, len(expected_pool))
        expected = expected_pool[-size:].tolist()
        actual = baseline_atom_indices(index, size, eligible, segments).tolist()
        assert actual == expected


def test_batch_backtest_matches_stepwise_prefix_delivery() -> None:
    frame = _frame(12)
    frame.loc[6, "high"] = 9
    batch = simulate_combo(frame, _combo(), frame.datetime.iloc[0], frame.datetime.iloc[-1])
    delivered: dict[tuple[int, int], dict[str, object]] = {}
    for end_index in range(1, len(frame)):
        prefix = frame.iloc[: end_index + 1].copy()
        trades = simulate_combo(
            prefix,
            _combo(),
            prefix.datetime.iloc[0],
            prefix.datetime.iloc[-1],
        )
        for trade in trades:
            forced_open_position = (
                trade["exit_reason"] == "segment_end" and end_index < len(frame) - 1
            )
            if not forced_open_position:
                delivered[(int(trade["signal_index"]), int(trade["entry_index"]))] = trade
    stepwise = [delivered[key] for key in sorted(delivered)]
    assert stepwise == batch
