"""Focused contract tests for the V4.2 high-return Round-1 planner."""
from __future__ import annotations

import json

import build_v4_2_high_return_round_1_plan as planner


def test_frozen_coordinate_ids_order_and_batches() -> None:
    rows = planner.stage_grid()
    ids = tuple(combo_id for combo_id, _ in rows)
    assert ids == planner.EXPECTED_COMBO_IDS
    assert len(ids) == len(set(ids)) == 24
    assert {row["seed"] for _, row in rows[:12]} == {"total_return_anchor_t"}
    assert {row["seed"] for _, row in rows[12:]} == {"average_return_anchor_a"}


def test_materialized_plan_and_audit_match_live_contract() -> None:
    result = planner.build_plan()
    plan = json.loads(planner.PLAN_PATH.read_text(encoding="utf-8"))
    audit = json.loads(planner.AUDIT_PATH.read_text(encoding="utf-8"))
    ids = tuple(planner.combo_id_from_record(row) for row in plan["explicit_combos"])
    assert result["status"] == audit["status"] == "passed"
    assert ids == planner.EXPECTED_COMBO_IDS
    assert plan["resources"] == {"workers": 3, "batch_size": 12, "minimum_free_memory_mb": 4096}
    assert len(plan["objective_contract"]["views"]) == 4
    assert plan["anti_join"]["completed_compatible_coordinate_count"] == 1470
    assert plan["anti_join"]["incomplete_sparse_plan_coordinate_count"] == 24
    assert plan["anti_join"]["proposed_completed_overlap_count"] == 0
    assert plan["anti_join"]["proposed_incomplete_sparse_overlap_count"] == 0
    assert audit["plan"]["sha256"] == planner.sha256_file(planner.PLAN_PATH)
