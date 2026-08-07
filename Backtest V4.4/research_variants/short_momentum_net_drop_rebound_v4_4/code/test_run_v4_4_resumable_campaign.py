from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import run_v4_4_resumable_campaign as campaign  # noqa: E402
from scenario_groups import (  # noqa: E402
    COMBINED_SCENARIO_SCHEMA_ID,
    DEFAULT_SCENARIO_DEFINITION,
    segments_frame,
)
from v4_4_engine import (  # noqa: E402
    COMBINED_STRATEGY_ID,
    ENTRY_EXECUTION_REJECT_SYNTHETIC_FILL,
    ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    ENTRY_FILL_CALCULATED_THRESHOLD,
    REBOUND_BASELINE_POLICY_ID,
    STRATEGY_ID,
    TRADE_AUDIT_SCHEMA_ID,
    TRADE_AUDIT_SCHEMA_VERSION,
    VERSION_LABEL,
)


def test_legacy_preparation_uses_verified_policy_attestation_bar_seconds(
    tmp_path: Path,
) -> None:
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps({"status": "complete", "bar_seconds": 15}), encoding="utf-8"
    )
    assert campaign._effective_preparation_bar_seconds(
        {"schema_version": 5},
        {"resolved_policy_attestation_path": str(attestation)},
    ) == 15
    assert campaign._effective_preparation_bar_seconds(
        {"schema_version": 5, "bar_seconds": 30},
        {"resolved_policy_attestation_path": str(attestation)},
    ) == 30


def _plan_payload(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "campaign_id": "v4_4_validation_test",
        "stage_id": "selected_v4_coordinates",
        "stage_kind": "v4_4_validation",
        "selection_provenance": "test fixture",
        "source": str(tmp_path / "source.csv"),
        "data_preparation_manifest": str(tmp_path / "preparation.json"),
        "scenario_definition": str(DEFAULT_SCENARIO_DEFINITION),
        "train_start": "2026-05-26 00:00:00",
        "train_end": "2026-07-08 23:52:00",
        "entry_fill_mode": ENTRY_FILL_CALCULATED_THRESHOLD,
        "entry_execution_policy": ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        "entry_slippage": 0.0,
        "resources": {"workers": 1, "batch_size": 2, "minimum_free_memory_mb": 0},
        "grid_blocks": [],
        "explicit_combos": [
            {
                "method": "rolling_tr_sum",
                "e": 58,
                "bh": 480,
                "trw": 16,
                "k": 0.65,
                "w": 4,
                "m": 0.625,
                "seed": "v4_trade_count_high",
                "objective": "validation",
                "design": "v4_4_repair_check",
            },
            {
                "method": "rolling_tr_sum",
                "e": 96,
                "bh": 420,
                "trw": 20,
                "k": 0.8,
                "w": 4,
                "m": 0.625,
                "seed": "v4_drawdown_low",
                "objective": "validation",
                "design": "v4_4_repair_check",
            },
        ],
    }


def _write_plan(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fake_validator(_: campaign.EffectivePlan) -> tuple[pd.DataFrame, dict[str, str]]:
    events = pd.DataFrame(
        [
            {"event_id": "market_01", "start_time": "2026-06-01", "end_time": "2026-06-02"},
            {"event_id": "market_02", "start_time": "2026-06-03", "end_time": "2026-06-04"},
            {"event_id": "market_03", "start_time": "2026-06-05", "end_time": "2026-06-06"},
        ]
    )
    return events, {
        "source_sha256": "source-v4-3",
        "data_preparation_manifest_sha256": "prepared-v4-3",
        "extreme_cleaning_audit_sha256": "extreme",
        "events_sha256": "",
        "scenario_definition_sha256": "scenarios-v2",
        "engine_sha256": "engine-v4-3",
        "runner_sha256": "runner-v4-3",
        "filter_atoms_sha256": "atoms-v4-3",
        "filter_events_sha256": "filter-events-v4-3",
    }


def test_plan_propagates_execution_policy_slippage_and_v4_4_combo_identity(tmp_path: Path) -> None:
    plan = campaign.load_plan(_write_plan(tmp_path, _plan_payload(tmp_path)))
    assert plan.entry_execution_policy == ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE
    assert plan.entry_slippage == 0.0
    assert plan.scenario_selection_mode == "single"
    assert plan.scenario_ids == ("scenario_1", "scenario_2", "scenario_3")
    assert all(combo.combo_id.startswith("v4_4_") for combo in plan.combos)
    assert all(combo.entry_execution_policy == plan.entry_execution_policy for combo in plan.combos)
    assert all(combo.entry_slippage == plan.entry_slippage for combo in plan.combos)


def test_result_semantics_identity_changes_with_policy_and_slippage() -> None:
    default = campaign.result_semantics_id(
        ENTRY_FILL_CALCULATED_THRESHOLD,
        ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        0.0,
    )
    rejected = campaign.result_semantics_id(
        ENTRY_FILL_CALCULATED_THRESHOLD,
        ENTRY_EXECUTION_REJECT_SYNTHETIC_FILL,
        0.0,
    )
    slipped = campaign.result_semantics_id(
        ENTRY_FILL_CALCULATED_THRESHOLD,
        ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        0.25,
    )
    assert default == campaign.RESULT_SEMANTICS_ID
    assert len({default, rejected, slipped}) == 3
    assert default.startswith("v4_4_confirmed_low_activity_gate_rolling_tr_sum_")
    assert "max_completed_w_drop_rebound" in default
    assert default.endswith("_positive_entry_signal_results_v4")


def test_relocated_extreme_audit_source_requires_declared_matching_runtime_path(
    tmp_path: Path,
) -> None:
    preparation_manifest = tmp_path / "runtime_inputs" / "data_preparation" / "manifest.json"
    runtime_source = tmp_path / "runtime_inputs" / "market_data" / "source.csv"
    runtime_source.parent.mkdir(parents=True)
    runtime_source.write_text("audited source", encoding="utf-8")
    source_hash = campaign._sha256(runtime_source)
    cleaned_source = {
        "path": r"F:\historical_provenance\source.csv",
        "sha256": source_hash.upper(),
    }
    relocated = {
        "source_path_relocated": True,
        "runtime_source_path": "../market_data/source.csv",
    }
    assert campaign._extreme_output_matches_runtime_source(
        preparation_manifest,
        relocated,
        cleaned_source,
        runtime_source,
        source_hash,
    )
    assert not campaign._extreme_output_matches_runtime_source(
        preparation_manifest,
        {**relocated, "source_path_relocated": False},
        cleaned_source,
        runtime_source,
        source_hash,
    )
    assert campaign._resolve_manifest_path(
        preparation_manifest,
        "baseline_filter_atoms.csv",
    ) == preparation_manifest.parent / "baseline_filter_atoms.csv"


def test_validate_only_materializes_an_isolated_v4_4_stage_manifest(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path, _plan_payload(tmp_path))
    output = tmp_path / "v4_4_output"
    result = campaign.run_stage(
        plan_path,
        output,
        validate_only=True,
        _contract_validator=_fake_validator,
    )
    assert result["status"] == "ready"
    manifest = json.loads((output / "stage_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 7
    assert manifest["plan_fingerprint_schema_version"] == 8
    assert manifest["version_label"] == VERSION_LABEL == "V4.4"
    assert manifest["strategy_id"] == STRATEGY_ID
    assert manifest["trade_audit_schema_version"] == TRADE_AUDIT_SCHEMA_VERSION
    assert manifest["trade_audit_schema_id"] == TRADE_AUDIT_SCHEMA_ID
    assert manifest["rebound_baseline_policy_id"] == REBOUND_BASELINE_POLICY_ID
    assert manifest["entry_execution_policy"] == ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE
    assert manifest["entry_fill_mode"] == ENTRY_FILL_CALCULATED_THRESHOLD
    assert manifest["baseline_sampling_policy"] == "confirmed_low_activity_gate"
    assert manifest["baseline_filter_id"] == (
        "confirmed_low_activity_retroactive_baseline_exclusion_and_entry_gate_v4_4"
    )
    assert manifest["entry_slippage"] == 0.0
    assert manifest["result_semantics_id"] == campaign.RESULT_SEMANTICS_ID
    assert manifest["scenario_selection_mode"] == "single"
    assert manifest["scenario_schema_id"] == COMBINED_SCENARIO_SCHEMA_ID
    assert manifest["scenario_ids"] == ["scenario_1", "scenario_2", "scenario_3"]
    assert all(row.startswith("v4_4_") for row in pd.read_csv(output / "grid_manifest.csv").combo_id)


def test_intermediate_round_defers_html_by_default() -> None:
    signature = inspect.signature(campaign.run_stage)
    assert signature.parameters["deliver_html"].default is False
    source = Path(campaign.__file__).read_text(encoding="utf-8")
    assert '"--publish-html"' in source
    assert "deliver_html=args.publish_html" in source
    assert 'result["mandatory_html_delivery"]' not in source


def test_v4_4_refuses_to_resume_a_v4_fingerprint_schema(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path, _plan_payload(tmp_path))
    plan = campaign.load_plan(plan_path)
    output = tmp_path / "foreign_v4_output"
    output.mkdir()
    (output / "stage_manifest.json").write_text(
        json.dumps({"plan_fingerprint_schema_version": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refuses to resume a V4"):
        campaign._materialize_stage_contract(
            plan_path,
            output,
            plan,
            _fake_validator(plan)[1],
        )


def test_plan_rejects_coordinate_execution_identity_override(tmp_path: Path) -> None:
    payload = _plan_payload(tmp_path)
    payload["explicit_combos"][0]["entry_execution_policy"] = (
        ENTRY_EXECUTION_REJECT_SYNTHETIC_FILL
    )
    with pytest.raises(ValueError, match="share its execution and exit contract"):
        campaign.load_plan(_write_plan(tmp_path, payload))


def test_scenario_group_plan_rejects_legacy_multi_scenario_request(tmp_path: Path) -> None:
    payload = _plan_payload(tmp_path)
    payload["scenario_ids"] = ["scenario_1", "scenario_2"]
    with pytest.raises(ValueError, match="multi-selection is not allowed"):
        campaign.load_plan(_write_plan(tmp_path, payload))


def test_batch_outputs_internal_and_scenario_qualification(tmp_path: Path) -> None:
    plan = campaign.load_plan(_write_plan(tmp_path, _plan_payload(tmp_path)))
    assert plan.scenario_contract is not None
    grid = pd.DataFrame(campaign._grid_rows(plan))
    combo_ids = grid.combo_id.astype(str).tolist()
    summaries = [
        {
                "combo_id": combo_id,
                "method": method,
                "baseline_sampling_policy": "all_window",
                "train_return": 0.1,
            "train_avg_trade": 0.01,
            "train_max_drawdown": -0.02,
            "train_trade_count": 3,
            "train_return_excluding_gap_spanning_trades": 0.08,
            "strategy_id": STRATEGY_ID,
            "exit_mode": "rebound_only",
            "speed_window_bars": 0,
        }
        for combo_id, method in zip(combo_ids, grid.method.astype(str), strict=True)
    ]
    segment_times = (
        ("2026-06-23 10:00:00", "2026-06-23 15:30:00"),
        ("2026-06-26 10:00:00", "2026-06-26 13:00:00"),
        ("2026-07-01 10:00:00", "2026-07-01 11:30:00"),
    )
    all_three = [
        {
            "combo_id": combo_ids[0],
            "entry_time": entry,
            "exit_time": exit_time,
            "exit_reason": "rebound_threshold",
        }
        for entry, exit_time in segment_times
    ]
    first_only = [
        {
            "combo_id": combo_ids[1],
            "entry_time": segment_times[0][0],
            "exit_time": segment_times[0][1],
            "exit_reason": "rebound_threshold",
        }
    ]
    for trade in [*all_three, *first_only]:
        trade.update(
            {
                "trade_audit_schema_version": TRADE_AUDIT_SCHEMA_VERSION,
                "trade_audit_schema_id": TRADE_AUDIT_SCHEMA_ID,
                "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
            }
        )
    manifest = campaign._write_batch(
        tmp_path / "batch",
        "batch_00001",
        grid,
        [(summaries[0], all_three), (summaries[1], first_only)],
        segments_frame(plan.scenario_contract),
        plan.scenario_ids,
        plan.scenario_contract,
        "fingerprint",
        1,
        0.1,
    )
    assert set(manifest["artifacts"]) == {
        "grid",
        "summary",
        "trades",
        "segment_qualification",
        "scenario_qualification",
    }
    assert manifest["trade_audit_schema_version"] == TRADE_AUDIT_SCHEMA_VERSION
    assert manifest["trade_audit_schema_id"] == TRADE_AUDIT_SCHEMA_ID
    assert manifest["rebound_baseline_policy_id"] == REBOUND_BASELINE_POLICY_ID
    summary = pd.read_csv(manifest["artifacts"]["summary"]["path"])
    assert "all_requested_scenarios_qualified" not in summary.columns
    indexed = summary.set_index("combo_id")
    assert bool(indexed.loc[combo_ids[0], "scenario_3_qualified"])
    assert not bool(indexed.loc[combo_ids[1], "scenario_3_qualified"])


def test_combined_plan_binds_speed_identity_and_schema(tmp_path: Path) -> None:
    payload = _plan_payload(tmp_path)
    payload["schema_version"] = 3
    payload["exit_mode"] = "combined"
    payload["scenario_definition"] = str(
        DEFAULT_SCENARIO_DEFINITION.parent
        / "v4_4_scenario_groups_single_select_combined_exit_20260801.json"
    )
    for combo in payload["explicit_combos"]:
        combo["speed_window_bars"] = 320
    plan = campaign.load_plan(_write_plan(tmp_path, payload))
    assert plan.exit_mode == "combined"
    assert plan.strategy_id == COMBINED_STRATEGY_ID
    assert all(combo.speed_window_bars == 320 for combo in plan.combos)
    assert all("_sx1_s320_rx1_" in combo.combo_id for combo in plan.combos)
    semantics = campaign.result_semantics_id(
        plan.entry_fill_mode,
        plan.entry_execution_policy,
        plan.entry_slippage,
        plan.exit_mode,
    )
    assert "combined_zero_extension_exit" in semantics
