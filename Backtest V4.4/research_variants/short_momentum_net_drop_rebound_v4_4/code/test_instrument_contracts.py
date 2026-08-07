from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import pandas as pd
import run_v4_4_resumable_campaign as campaign_runner

from build_v4_4_combined_union_analysis import _apply_stage_bound_costs
from analyze_v4_4_scenario_3_stage import _cost_model_from_stage_manifest, analyze
from instrument_contracts import (
    load_campaign_manifest,
    load_cost_model,
    load_instrument_profile,
    sha256_file,
)
from run_v4_4_resumable_campaign import load_plan
from v4_4_engine import Combo, entry_signal_qualifies


VARIANT_ROOT = Path(__file__).resolve().parent.parent
K200_PROFILE = VARIANT_ROOT / "instrument_profiles" / "k200m.json"
SIMAIN_TEMPLATE = VARIANT_ROOT / "instrument_profiles" / "simain.template.json"


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def test_legacy_k200_cost_is_normalized_without_numeric_change() -> None:
    profile = load_instrument_profile(K200_PROFILE)
    model = profile["normalized_cost_model"]
    assert profile["instrument_id"] == "k200m"
    assert profile["ranking_lineage_id"] == "k200m_v4_4_positive_entry_future_lineage"
    assert model["quote_currency"] == "KRW"
    assert model["point_value"] == model["contract_multiplier_krw_per_point"] == 50000.0
    assert model["contract_notional_quote"] == model["contract_notional_krw"] == 55335000.0
    assert math.isclose(
        model["round_trip_total_cost_bps"],
        3.568663594470046,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_unresolved_instrument_template_cannot_execute() -> None:
    template = load_instrument_profile(SIMAIN_TEMPLATE, require_ready=False)
    assert template["status"] == "requires_user_input"
    with pytest.raises(ValueError, match="not ready"):
        load_instrument_profile(SIMAIN_TEMPLATE)


def test_generic_cost_uses_price_times_point_value_for_notional(tmp_path: Path) -> None:
    path = tmp_path / "cost.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cost_model_id": "test_cost",
                "instrument_id": "test_future",
                "contract": {
                    "instrument": "Test Future",
                    "point_value": 20,
                    "quote_currency": "USD",
                },
                "market_price": {"price": 18000, "currency": "points"},
                "cost_inputs": {
                    "round_trip_slippage_bps": 2,
                    "round_trip_commission": 5,
                    "commission_currency": "USD",
                },
            }
        ),
        encoding="utf-8",
    )
    model = load_cost_model(path)
    assert model["contract_notional_quote"] == 360000
    assert math.isclose(
        model["round_trip_total_cost_bps"],
        2 + 10000 * 5 / 360000,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_campaign_modes_enforce_transfer_and_lineage_contracts(tmp_path: Path) -> None:
    candidate_freeze = tmp_path / "candidates.json"
    candidate_freeze.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ready",
                "campaign_id": "simain_transfer_check",
                "mode": "transfer_exact",
                "instrument_profile": _binding(K200_PROFILE),
                "ranking": {
                    "lineage_id": "k200m_v4_4_positive_entry_future_lineage",
                    "merge_policy": "same_instrument_compatible_lineage_only",
                },
                "source": {"candidate_freeze": _binding(candidate_freeze)},
                "target_tuning_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_campaign_manifest(manifest)
    assert loaded["mode"] == "transfer_exact"
    changed = json.loads(manifest.read_text(encoding="utf-8"))
    changed["target_tuning_allowed"] = True
    manifest.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="forbid target tuning"):
        load_campaign_manifest(manifest)


def test_schema_v5_plan_binds_profile_mode_and_ranking_lineage(tmp_path: Path) -> None:
    candidate_freeze = tmp_path / "candidates.json"
    candidate_freeze.write_text("{}\n", encoding="utf-8")
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ready",
                "campaign_id": "k200_contract_test",
                "mode": "transfer_exact",
                "instrument_profile": _binding(K200_PROFILE),
                "ranking": {
                    "lineage_id": "k200m_v4_4_positive_entry_future_lineage",
                    "merge_policy": "same_instrument_compatible_lineage_only",
                },
                "source": {"candidate_freeze": _binding(candidate_freeze)},
                "target_tuning_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "campaign_id": "k200_contract_test",
                "stage_id": "transfer_stage",
                "campaign_manifest": str(campaign),
                "experiment_mode": "transfer_exact",
                "scenario_policy": "profile_optional",
                "entry_fill_mode": "calculated_threshold",
                "entry_execution_policy": "wait_next_real_trade",
                "baseline_sampling_policy": "all_window",
                "exit_mode": "combined",
                "explicit_combos": [
                    {
                        "method": "rolling_tr_sum",
                        "e": 320,
                        "bh": 240,
                        "trw": 12,
                        "k": 1.25,
                        "w": 6,
                        "m": 4.5,
                        "speed_window_bars": 340,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = load_plan(plan_path)
    assert plan.experiment_mode == "transfer_exact"
    assert plan.instrument_profile == K200_PROFILE.resolve()
    assert plan.ranking_lineage_id == "k200m_v4_4_positive_entry_future_lineage"
    assert plan.scenario_selection_mode == "single"


def test_union_retains_each_source_stage_cost_model() -> None:
    summary = pd.DataFrame(
        [
            {"combo_id": "a", "train_return": 0.01, "source_stage_key": "c::s1"},
            {"combo_id": "b", "train_return": 0.01, "source_stage_key": "c::s2"},
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "combo_id": "a",
                "return": 0.01,
                "entry_index": 1,
                "exit_index": 2,
                "source_stage_key": "c::s1",
            },
            {
                "combo_id": "b",
                "return": 0.01,
                "entry_index": 1,
                "exit_index": 2,
                "source_stage_key": "c::s2",
            },
        ]
    )

    def model(identity: str, cost_bps: float) -> dict[str, object]:
        return {
            "id": identity,
            "reference_sha256": identity * 8,
            "round_trip_total_cost_bps": cost_bps,
            "round_trip_commission_quote": 1.0,
            "round_trip_slippage_quote": 1.0,
            "round_trip_total_cost_quote": 2.0,
            "quote_currency": "USD",
        }

    included = [
        {
            "campaign_id": "c",
            "stage_id": "s1",
            "ranking_lineage_id": "lineage",
            "cost_model": model("model_one", 2.0),
        },
        {
            "campaign_id": "c",
            "stage_id": "s2",
            "ranking_lineage_id": "lineage",
            "cost_model": model("model_two", 5.0),
        },
    ]
    costed_summary, costed_trades, by_stage, union_model = _apply_stage_bound_costs(
        summary, trades, included
    )
    costs = costed_summary.set_index("combo_id")["round_trip_cost_bps"].to_dict()
    returns = costed_trades.set_index("combo_id")["cost_adjusted_return"].to_dict()
    assert costs == {"a": 2.0, "b": 5.0}
    assert returns["a"] == pytest.approx(0.01 - 0.0002)
    assert returns["b"] == pytest.approx(0.01 - 0.0005)
    assert len(by_stage) == 2
    assert union_model["id"] == "multiple_stage_bound_cost_models"


def _record(combo: Combo) -> dict[str, object]:
    return {
        "combo_id": combo.combo_id,
        "method": combo.method,
        "e": combo.e,
        "bh": combo.bh,
        "trw": combo.trw,
        "k": combo.k,
        "w": combo.w,
        "m": combo.m,
        "speed_window_bars": combo.speed_window_bars,
    }


def _mode_manifest(mode: str, profile: dict[str, object], resolved: dict[str, object], search: dict[str, object]) -> dict[str, object]:
    return {
        "manifest_schema_version": 2,
        "mode": mode,
        "instrument_profile_contract": profile,
        "resolved_mode_contract": resolved,
        "search": search,
    }


def test_transfer_exact_rejects_empty_freeze_and_plan_difference() -> None:
    combo = Combo("rolling_tr_sum", 320, 240, 12, 1.25, 6, 4.5, baseline_sampling_policy="all_window")
    profile = {"instrument_id": "simain", "bar_seconds": 15}
    empty = _mode_manifest(
        "transfer_exact", profile,
        {"candidate_freeze_payload": {"status": "frozen_before_target_evaluation", "frozen_at": "2026-08-05T00:00:00Z", "target_evaluation_started_at": None, "target_results_present": False, "bar_seconds": 15, "candidates": []}},
        {"maximum_coordinate_count": 10},
    )
    with pytest.raises(ValueError, match="non-empty candidates"):
        campaign_runner._validate_mode_coordinates(
            "transfer_exact", empty, (combo,), 1, {combo.combo_id: {"search_mode": ""}},
            ("calculated_threshold", "wait_next_real_trade", 0.0, "all_window"),
        )
    other = Combo("rolling_tr_sum", 200, 240, 12, 1.25, 6, 4.5, baseline_sampling_policy="all_window")
    mismatch = _mode_manifest(
        "transfer_exact", profile,
        {"candidate_freeze_payload": {"status": "frozen_before_target_evaluation", "frozen_at": "2026-08-05T00:00:00Z", "target_evaluation_started_at": None, "target_results_present": False, "bar_seconds": 15, "candidate_count": 1, "candidates": [_record(other)]}},
        {"maximum_coordinate_count": 10},
    )
    with pytest.raises(ValueError, match="differs from candidate freeze"):
        campaign_runner._validate_mode_coordinates(
            "transfer_exact", mismatch, (combo,), 1, {combo.combo_id: {"search_mode": ""}},
            ("calculated_threshold", "wait_next_real_trade", 0.0, "all_window"),
        )


def test_target_local_refinement_rejects_far_coordinate(tmp_path: Path) -> None:
    anchor = Combo("rolling_tr_sum", 320, 240, 12, 1.25, 6, 4.5, speed_window_bars=340, baseline_sampling_policy="all_window")
    far = Combo("rolling_tr_sum", 20, 720, 3, 0.5, 192, 0.25, speed_window_bars=960, baseline_sampling_policy="all_window")
    freeze_path = tmp_path / "freeze.json"
    freeze_payload = {"bar_seconds": 15, "candidate_count": 1, "candidates": [_record(anchor)]}
    freeze_path.write_text(json.dumps(freeze_payload), encoding="utf-8")
    rules = {name: {"mode": "relative", "max_fraction": 0.1} for name in ("e", "bh", "trw", "k")}
    rules.update({name: {"mode": "fixed"} for name in ("w", "m", "speed_window_bars")})
    manifest = _mode_manifest(
        "target_local_refinement",
        {"instrument_id": "simain", "ranking_lineage_id": "si_lineage", "bar_seconds": 15},
        {
            "parent_payload": {"status": "complete", "target_instrument_id": "simain", "candidate_freeze_sha256": sha256_file(freeze_path)},
            "candidate_freeze_payload": freeze_payload,
            "candidate_freeze_path": str(freeze_path),
        },
        {"maximum_coordinate_count": 10, "neighborhood": {"anchor_combo_ids": [anchor.combo_id], "parameter_rules": rules}},
    )
    with pytest.raises(ValueError, match="outside every declared neighborhood"):
        campaign_runner._validate_mode_coordinates(
            "target_local_refinement", manifest, (far,), 1, {far.combo_id: {"search_mode": "local"}},
            ("calculated_threshold", "wait_next_real_trade", 0.0, "all_window"),
        )


def test_fresh_search_requires_space_budget_and_search_mode() -> None:
    combo = Combo("rolling_tr_sum", 320, 240, 12, 1.25, 6, 4.5)
    profile = {"instrument_id": "nq", "ranking_lineage_id": "nq_lineage", "bar_seconds": 15}
    space = {name: {"values": [value]} for name, value in campaign_runner._parameter_values(combo).items()}
    manifest = _mode_manifest(
        "fresh_search", profile, {},
        {"maximum_coordinate_count": 1, "parameter_space": space},
    )
    with pytest.raises(ValueError, match="requires search_mode"):
        campaign_runner._validate_mode_coordinates(
            "fresh_search", manifest, (combo,), 1, {combo.combo_id: {"search_mode": ""}},
            ("calculated_threshold", "wait_next_real_trade", 0.0, "all_window"),
        )
    space["e"] = {"values": [20]}
    with pytest.raises(ValueError, match="outside parameter_space.e"):
        campaign_runner._validate_mode_coordinates(
            "fresh_search", manifest, (combo,), 1, {combo.combo_id: {"search_mode": "broad_jump"}},
            ("calculated_threshold", "wait_next_real_trade", 0.0, "all_window"),
        )


def test_positive_entry_signal_gate_rejects_zero_and_keeps_equality() -> None:
    assert entry_signal_qualifies(0.0, 0.0, 1.0) is False
    assert entry_signal_qualifies(2.0, 2.0, 1.0) is True


def test_schema_v5_profile_optional_without_scenario_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text("{}", encoding="utf-8")
    profile = load_instrument_profile(K200_PROFILE)
    profile = {**profile, "resolved_scenario_set_path": None}
    manifest = {
        "campaign_id": "scenario_free",
        "mode": "transfer_exact",
        "manifest_schema_version": 1,
        "resolved_instrument_profile_path": str(K200_PROFILE),
        "instrument_profile_contract": profile,
        "resolved_mode_contract": {},
    }
    monkeypatch.setattr(campaign_runner, "load_campaign_manifest", lambda _: manifest)
    monkeypatch.setattr(campaign_runner, "load_instrument_profile", lambda _: profile)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "campaign_id": "scenario_free",
                "stage_id": "stage",
                "campaign_manifest": str(campaign_path),
                "scenario_policy": "profile_optional",
                "explicit_combos": [_record(Combo("rolling_tr_sum", 320, 240, 12, 1.25, 6, 4.5))],
            }
        ),
        encoding="utf-8",
    )
    plan = load_plan(plan_path)
    assert plan.scenario_selection_mode == "none"
    assert plan.scenario_ids == ()


def test_analyzer_uses_closed_stage_target_cost(tmp_path: Path) -> None:
    cost_path = tmp_path / "target_cost.json"
    cost_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cost_model_id": "simain_cost",
                "instrument_id": "simain",
                "contract": {"instrument": "SI", "point_value": 5000, "quote_currency": "USD"},
                "market_price": {"price": 30, "currency": "USD"},
                "cost_inputs": {"round_trip_slippage_bps": 2, "round_trip_commission": 6, "commission_currency": "USD"},
            }
        ),
        encoding="utf-8",
    )
    model = _cost_model_from_stage_manifest(
        {"instrument_contract": {"cost_model_path": str(cost_path), "cost_model_sha256": sha256_file(cost_path)}}
    )
    assert model["instrument_id"] == "simain"
    assert model["id"] == "simain_cost"


def test_schema_v2_profile_rejects_policy_and_bar_interval_mismatch(tmp_path: Path) -> None:
    data_path = tmp_path / "bars.csv"
    pd.DataFrame({
        "datetime": pd.date_range("2026-01-01 00:00:00", periods=9, freq="15s"),
        "open": [10, 10, 11, 11, 9, 8, 8, 8, 8],
        "high": [10, 10, 11, 11, 9, 8, 8, 8, 8],
        "low": [9, 9, 10, 10, 8, 7, 7, 8, 8],
        "close": [9, 9, 10, 10, 8, 7, 7, 8, 8],
        "volume": 1, "trade_count": 1, "is_synthetic_empty_bar": False,
    }).to_csv(data_path, index=False)
    config = tmp_path / "policy.json"
    implementation = tmp_path / "policy.py"
    config.write_text("{}", encoding="utf-8")
    implementation.write_text("# policy\n", encoding="utf-8")
    cost = tmp_path / "cost.json"
    cost.write_text(
        json.dumps({
            "cost_model_id": "si_cost", "instrument_id": "simain",
            "contract": {"instrument": "SI", "point_value": 5000, "quote_currency": "USD"},
            "market_price": {"price": 30},
            "cost_inputs": {"round_trip_slippage_bps": 2, "round_trip_commission": 6, "commission_currency": "USD"},
        }), encoding="utf-8"
    )
    gap_contract = {"policy_id": "gap_v1", "mode": "none", "config_sha256": sha256_file(config), "implementation_sha256": sha256_file(implementation)}
    low_contract = {"policy_id": "low_v1", "mode": "none", "config_sha256": sha256_file(config), "implementation_sha256": sha256_file(implementation)}
    marker = tmp_path / "filter_atoms.csv"
    pd.DataFrame({
        "datetime": pd.date_range("2026-01-01 00:00:00", periods=9, freq="15s"),
        "baseline_excluded": False, "filter_reason_codes": "", "filter_event_ids": "",
        "causal_active": False, "low_activity_state": "normal", "pending_buffer_start": "",
        "pending_buffer_count": 0, "buffer_reinserted": False,
        "buffer_confirmed_excluded": False, "recovery_confirmation_time": "",
        "baseline_available_from": pd.date_range("2026-01-01 00:00:00", periods=9, freq="15s"),
        "eligible_if_excluding_marked": True,
    }).to_csv(marker, index=False)
    events = tmp_path / "filter_events.json"
    events.write_text(json.dumps({"events": []}), encoding="utf-8")
    preparation = tmp_path / "prep.json"
    preparation.write_text(json.dumps({
        "schema_version": 6, "status": "complete", "bar_seconds": 15,
        "source_sha256": sha256_file(data_path),
        "prepared_identity": "v4_4_policy_neutral_baseline_marker_smoke",
        "policy_contracts": {"gap_policy": gap_contract, "low_activity_policy": low_contract},
        "artifacts": {
            "filter_atoms": {"path": str(marker), "sha256": sha256_file(marker)},
            "filter_events": {"path": str(events), "sha256": sha256_file(events)},
        },
    }), encoding="utf-8")
    binding = lambda target: _binding(target)
    policy = lambda identity: {"status": "ready", "policy_id": identity, "mode": "none", "config": binding(config), "implementation": binding(implementation)}
    profile_payload = {
        "schema_version": 2, "status": "ready", "instrument_id": "simain", "display_name": "SI",
        "strategy_contract_id": "core_v2", "ranking_lineage_id": "si_lineage",
        "bar_seconds": 15,
        "data": {**binding(data_path), "preparation_manifest": binding(preparation)},
        "cost_model": binding(cost), "gap_policy": policy("gap_v1"),
        "low_activity_policy": policy("low_v1"), "scenario_set": None,
    }
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(profile_payload), encoding="utf-8")
    assert load_instrument_profile(profile)["bar_seconds"] == 15
    combo = Combo("rolling_tr_sum", 4, 2, 1, 0.1, 3, 0.1, speed_window_bars=4)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "status": "frozen_before_target_evaluation", "frozen_at": "2026-01-01T00:00:00Z",
        "target_evaluation_started_at": None, "target_results_present": False,
        "bar_seconds": 15, "candidate_count": 1, "candidates": [_record(combo)],
    }), encoding="utf-8")
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({
        "schema_version": 2, "status": "ready", "campaign_id": "si_smoke",
        "mode": "transfer_exact", "instrument_profile": binding(profile),
        "ranking": {"lineage_id": "si_lineage", "merge_policy": "same_instrument_compatible_lineage_only"},
        "source": {"candidate_freeze": binding(freeze)}, "target_tuning_allowed": False,
        "search": {"maximum_coordinate_count": 1},
    }), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema_version": 6, "status": "approved_for_execution", "campaign_id": "si_smoke", "stage_id": "transfer",
        "campaign_manifest": str(campaign), "scenario_policy": "profile_optional",
        "exit_mode": "combined",
        "train_start": "2026-01-01 00:00:00", "train_end": "2026-01-01 00:02:00",
        "explicit_combos": [_record(combo)],
    }), encoding="utf-8")
    plan = load_plan(plan_path)
    selected, hashes = campaign_runner._validate_contract(plan)
    assert selected.empty
    assert plan.instrument_profile_contract["display_name"] == "SI"
    assert hashes["cost_model_sha256"] == sha256_file(cost)
    stage_root = tmp_path / "stage"
    result = campaign_runner.run_stage(
        plan_path, stage_root, workers=1, batch_size=1,
        minimum_free_memory_mb=0, deliver_html=False,
    )
    assert result["status"] == "complete"
    stage_manifest = json.loads((stage_root / "stage_manifest.json").read_text(encoding="utf-8"))
    assert stage_manifest["instrument_contract"]["display_name"] == "SI"
    assert stage_manifest["instrument_contract"]["ranking_lineage_id"] == "si_lineage"
    assert stage_manifest["instrument_contract"]["ranking_lineage_id"] != "k200m_v4_4_training_20260526_20260708"
    analysis_root = tmp_path / "analysis"
    analyzed = analyze(plan_path, stage_root, analysis_root, review_workers=1)
    assert analyzed["status"] == "complete"
    analysis_data = (analysis_root / "analysis_data.js").read_text(encoding="utf-8")
    assert '"display_name":"SI"' in analysis_data
    assert '"ranking_lineage_id":"si_lineage"' in analysis_data
    original_freeze = freeze.read_bytes()
    freeze.write_text(json.dumps({"candidates": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_freeze SHA-256"):
        load_plan(plan_path)
    freeze.write_bytes(original_freeze)

    invalid_mode_profile = json.loads(profile.read_text(encoding="utf-8"))
    invalid_mode_profile["gap_policy"]["mode"] = "anything"
    profile.write_text(json.dumps(invalid_mode_profile), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported gap_policy.mode"):
        load_instrument_profile(profile)
    invalid_hash_profile = profile_payload.copy()
    invalid_hash_profile["gap_policy"] = dict(profile_payload["gap_policy"])
    invalid_hash_profile["gap_policy"]["config"] = dict(profile_payload["gap_policy"]["config"])
    invalid_hash_profile["gap_policy"]["config"]["sha256"] = "0" * 64
    profile.write_text(json.dumps(invalid_hash_profile), encoding="utf-8")
    with pytest.raises(ValueError, match="gap_policy.config SHA-256"):
        load_instrument_profile(profile)

    preparation_without_contracts = json.loads(preparation.read_text(encoding="utf-8"))
    preparation_without_contracts.pop("policy_contracts")
    preparation.write_text(json.dumps(preparation_without_contracts), encoding="utf-8")
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps({
        "status": "complete", "bar_seconds": 15,
        "preparation_manifest_sha256": "0" * 64,
        "policy_contracts": {"gap_policy": gap_contract, "low_activity_policy": low_contract},
    }), encoding="utf-8")
    attestation_profile = json.loads(json.dumps(profile_payload))
    attestation_profile["data"]["preparation_manifest"] = binding(preparation)
    attestation_profile["data"]["policy_attestation"] = binding(attestation)
    profile.write_text(json.dumps(attestation_profile), encoding="utf-8")
    with pytest.raises(ValueError, match="attestation binds a different preparation"):
        load_instrument_profile(profile)
    changed = json.loads(preparation.read_text(encoding="utf-8"))
    changed["policy_contracts"] = {"gap_policy": {**gap_contract, "policy_id": "wrong"}, "low_activity_policy": low_contract}
    preparation.write_text(json.dumps(changed), encoding="utf-8")
    profile_payload["data"]["preparation_manifest"] = binding(preparation)
    profile.write_text(json.dumps(profile_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from preparation manifest"):
        load_instrument_profile(profile)
