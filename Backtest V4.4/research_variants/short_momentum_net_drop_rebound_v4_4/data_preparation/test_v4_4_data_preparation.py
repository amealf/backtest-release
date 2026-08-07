from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research_variants.short_momentum_net_drop_rebound_v4_4.data_preparation import low_activity

from research_variants.short_momentum_net_drop_rebound_v4_4.data_preparation.low_activity import (
    LOW_ACTIVITY_STATE_CONFIRMED,
    LOW_ACTIVITY_STATE_NORMAL,
    LOW_ACTIVITY_STATE_PENDING,
    _low_activity_lifecycle,
    detect_low_activity,
    load_15s_bars,
)
from research_variants.short_momentum_net_drop_rebound_v4_4.data_preparation.prepare_dataset import (
    PIPELINE_VERSION,
    prepare_dataset,
)


def _lifecycle_frame(low_flags: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01 09:00:00", periods=len(low_flags), freq="15s"),
            "atom_segment": 1,
            "low_volume_atom": low_flags,
        }
    )


def _bars(periods: int) -> pd.DataFrame:
    times = pd.date_range("2026-01-01 09:00:00", periods=periods, freq="15s")
    prices = [100 + (index % 7) * 0.05 for index in range(periods)]
    return pd.DataFrame(
        {
            "datetime": times,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 10,
            "trade_count": 1,
            "source": "test_ticks",
            "is_synthetic_empty_bar": 0,
            "bar_seconds": 15,
        }
    )


def test_transient_low_activity_buffer_is_fully_reinserted_on_recovery() -> None:
    frame, events = _low_activity_lifecycle(
        _lifecycle_frame([False, True, True, True, False, False]),
        duration_atoms=4,
    )
    assert events == []
    assert frame.low_activity_state.tolist() == [
        LOW_ACTIVITY_STATE_NORMAL,
        LOW_ACTIVITY_STATE_PENDING,
        LOW_ACTIVITY_STATE_PENDING,
        LOW_ACTIVITY_STATE_PENDING,
        LOW_ACTIVITY_STATE_NORMAL,
        LOW_ACTIVITY_STATE_NORMAL,
    ]
    assert frame.buffer_reinserted.tolist() == [False, True, True, True, True, False]
    assert not bool(frame.buffer_confirmed_excluded.any())
    assert frame.baseline_excluded_from.isna().all()
    assert not bool(frame.confirmed_low_activity_active.any())
    recovery = frame.datetime.iloc[4]
    assert frame.loc[1:4, "recovery_confirmation_time"].eq(recovery).all()
    assert frame.loc[1:4, "pending_buffer_start"].eq(frame.datetime.iloc[1]).all()
    assert frame.pending_buffer_count.tolist() == [0, 1, 2, 3, 3, 0]


def test_sustained_low_activity_buffer_becomes_permanently_excluded() -> None:
    frame, events = _low_activity_lifecycle(
        _lifecycle_frame([False, True, True, True, True, True, False]),
        duration_atoms=4,
    )
    assert len(events) == 1
    event = events[0]
    assert event["start"] == frame.datetime.iloc[1]
    assert event["confirmation_time"] == frame.datetime.iloc[4]
    assert event["end"] == frame.datetime.iloc[5]
    assert event["end_reason"] == "high_volume_atom"
    assert frame.low_activity_state.tolist()[1:6] == [
        LOW_ACTIVITY_STATE_PENDING,
        LOW_ACTIVITY_STATE_PENDING,
        LOW_ACTIVITY_STATE_PENDING,
        LOW_ACTIVITY_STATE_CONFIRMED,
        LOW_ACTIVITY_STATE_CONFIRMED,
    ]
    assert frame.loc[1:6, "buffer_confirmed_excluded"].all()
    assert not bool(frame.buffer_reinserted.any())
    assert frame.loc[1:6, "recovery_confirmation_time"].eq(frame.datetime.iloc[6]).all()
    confirmation = frame.datetime.iloc[4]
    assert frame.loc[1:5, "baseline_excluded_from"].eq(confirmation).all()
    assert frame.loc[1:5, "low_activity_confirmation_time"].eq(confirmation).all()
    assert frame.confirmed_low_activity_active.tolist() == [
        False, False, False, False, True, True, False
    ]


def test_preparation_manifest_and_atoms_use_isolated_v4_4_identity(tmp_path: Path) -> None:
    frame = _bars(625)
    frame.loc[501:620, "volume"] = 1
    source = tmp_path / "bars.csv"
    frame.to_csv(source, index=False)
    plotly = tmp_path / "plotly.min.js"
    plotly.write_text("window.Plotly={};", encoding="utf-8")
    output = tmp_path / "prepared_v4_4"
    registry = tmp_path / "PROCESSED_DATASETS.json"
    result = prepare_dataset(
        source,
        "GENERIC",
        output_dir=output,
        plotly_source=plotly,
        registry=registry,
        allow_legacy_preexisting_source=True,
    )
    assert result["status"] == "generated"
    manifest = json.loads((output / "data_preparation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 5
    assert manifest["pipeline_version"] == PIPELINE_VERSION
    assert manifest["dataset_id"].startswith("GENERIC_v4_4_")
    assert manifest["prepared_identity"].startswith("v4_4_confirmed_low_activity_gate_")
    contract = manifest["rule_contract"]["baseline_sampling_contract"]
    assert contract["default_baseline_sampling_policy"] == "confirmed_low_activity_gate"
    assert set(contract["supported_baseline_sampling_policies"]) == {
        "all_window",
        "exclude_marked",
        "confirmed_low_activity_gate",
    }
    atoms = pd.read_csv(output / "baseline_filter_atoms.csv")
    for field in (
        "low_activity_state",
        "pending_buffer_start",
        "pending_buffer_count",
        "buffer_reinserted",
        "buffer_confirmed_excluded",
        "recovery_confirmation_time",
        "baseline_available_from",
        "low_activity_confirmation_time",
        "baseline_excluded_from",
        "confirmed_low_activity_active",
        "eligible_if_excluding_marked",
    ):
        assert field in atoms.columns
    assert atoms.loc[501:620, "buffer_confirmed_excluded"].all()
    assert atoms.loc[501:620, "baseline_available_from"].isna().all()
    confirmation = pd.Timestamp(atoms.loc[620, "datetime"])
    assert pd.to_datetime(atoms.loc[501:620, "baseline_excluded_from"]).eq(confirmation).all()
    assert atoms.loc[501:619, "confirmed_low_activity_active"].eq(False).all()
    assert bool(atoms.loc[620, "confirmed_low_activity_active"])
    assert manifest["low_activity_summary"]["universal_event_count"] == 1


def test_full_detector_reinserts_a_short_real_threshold_run() -> None:
    frame = _bars(510)
    frame.loc[501:503, "volume"] = 1
    result = detect_low_activity(frame.assign(volume_semantic="test_ticks", atom_segment=1), "GENERIC")
    assert result.events == []
    assert result.atoms.loc[501:503, "buffer_reinserted"].all()
    assert bool(result.atoms.loc[504, "buffer_reinserted"])
    assert result.atoms.loc[501:504, "recovery_confirmation_time"].notna().all()
    recovery = result.atoms.loc[504, "datetime"]
    assert pd.to_datetime(
        result.atoms.loc[501:504, "baseline_available_from"]
    ).eq(recovery).all()


def test_display_only_pause_remains_in_universal_low_volume_input(monkeypatch) -> None:
    frame = _bars(745)
    frame.loc[501:620, ["volume", "trade_count"]] = 0
    frame.loc[501:620, "is_synthetic_empty_bar"] = 1
    pause = {
        "event_id": "k200_extended_pause_01",
        "family": "k200_market_mechanism",
        "event_type": "extended_internal_pause_candidate",
        "label": "长时内部停牌候选",
        "start": frame.datetime.iloc[501],
        "confirmation_time": frame.datetime.iloc[501] + pd.Timedelta(minutes=60),
        "end": frame.datetime.iloc[620],
        "duration_minutes": 30.0,
        "apply_to_baseline": False,
        "reason_code": "uncertain_extended_pause",
        "reason": "display only",
        "confidence": "test",
    }
    monkeypatch.setattr(low_activity, "infer_tick_size", lambda _: 0.05)
    monkeypatch.setattr(low_activity, "_price_lock_events", lambda *_: [])
    monkeypatch.setattr(low_activity, "_pause_events", lambda *_: [pause])
    result = detect_low_activity(
        frame.assign(volume_semantic="test_ticks", atom_segment=1), "K200"
    )
    universal = [
        event for event in result.events
        if event["event_type"] == "universal_low_volume"
    ]
    assert len(universal) == 1
    assert universal[0]["start"] == frame.datetime.iloc[501]
    assert result.atoms.loc[501:620, "baseline_excluded"].all()
    assert not result.atoms.loc[501:620, "eligible_if_excluding_marked"].any()
