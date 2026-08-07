from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_v4_4_combined_union_analysis import (
    RANKING_LINEAGE_ID,
    _shared_stage_manifest,
)


def _stage_manifest(policy: str, suffix: str) -> dict[str, object]:
    return {
        "source": "k200.csv",
        "source_sha256": "source",
        "data_preparation_manifest": f"prepared_{suffix}.json",
        "data_preparation_manifest_sha256": f"prepared_{suffix}",
        "prepared_identity": f"prepared_{suffix}",
        "engine_sha256": f"engine_{suffix}",
        "version_label": "V4.4",
        "scenario_schema_id": "scenario",
        "scenario_definition_sha256": "scenario_hash",
        "scenario_selection_mode": "single",
        "baseline_filter_atoms_sha256": f"atoms_{suffix}",
        "baseline_filter_events_sha256": f"events_{suffix}",
        "events_sha256": "events",
        "train_start": "2026-05-26 00:00:00",
        "train_end": "2026-07-08 23:52:00",
        "entry_fill_mode": "calculated_threshold",
        "entry_execution_policy": "wait_next_real_trade",
        "entry_slippage": 0.0,
        "exit_mode": "combined",
        "schema_version": 7,
        "plan_fingerprint_schema_version": 8,
        "trade_audit_schema_version": 3,
        "trade_audit_schema_id": "trade_audit",
        "rebound_baseline_policy_id": "rebound",
        "baseline_sampling_policy": policy,
        "strategy_id": f"strategy_{suffix}",
        "baseline_filter_id": f"filter_{suffix}",
        "result_semantics_id": f"semantics_{suffix}",
    }


def test_minor_implementation_hashes_do_not_split_v4_4_union() -> None:
    first = _stage_manifest("all_window", "old")
    second = deepcopy(first)
    second.update(
        {
            "data_preparation_manifest": "prepared_new.json",
            "data_preparation_manifest_sha256": "prepared_new",
            "prepared_identity": "prepared_new",
            "engine_sha256": "engine_new",
            "baseline_filter_atoms_sha256": "atoms_new",
            "baseline_filter_events_sha256": "filter_events_new",
            "strategy_id": "strategy_new",
            "baseline_filter_id": "filter_new",
            "result_semantics_id": "semantics_new",
        }
    )
    included = [
        {"ranking_lineage_id": RANKING_LINEAGE_ID, "loaded": {"stage_manifest": first}},
        {"ranking_lineage_id": RANKING_LINEAGE_ID, "loaded": {"stage_manifest": second}},
    ]

    shared = _shared_stage_manifest(included)

    assert shared["ranking_major_version"] == "V4.4"
    assert shared["ranking_lineage_id"] == RANKING_LINEAGE_ID
    assert shared["strategy_id"] == "multiple_within_v4_4_major_ranking_lineage"
    assert shared["strategy_id_variants_by_baseline_sampling_policy"]["all_window"] == [
        "strategy_new",
        "strategy_old",
    ]
