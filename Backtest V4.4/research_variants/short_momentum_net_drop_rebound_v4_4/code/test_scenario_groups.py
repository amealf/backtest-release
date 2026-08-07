from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from scenario_groups import (  # noqa: E402
    COMBINED_SCENARIO_SCHEMA_ID,
    SCENARIO_SCHEMA_ID,
    attach_scenario_groups,
    evaluate_segment_qualification,
    filter_single_scenario,
    load_scenario_contract,
)


def _contract(tmp_path: Path) -> dict[str, object]:
    payload = {
        "schema_version": 2,
        "status": "current",
        "scenario_schema_id": SCENARIO_SCHEMA_ID,
        "selection_mode": "single",
        "neutral_selection_id": "all",
        "qualification_rule": {
            "entry_interval": "start_exclusive_end_inclusive",
            "exit_interval": "start_exclusive_end_inclusive",
            "required_entry_count": 1,
            "required_exit_count": 0,
            "must_hold_past_segment_end": True,
            "required_eventual_exit_reason": "rebound_threshold",
        },
        "segments": [
            {
                "segment_id": f"market_0{number}",
                "label_zh": f"行情 {number}",
                "label_en": f"Market {number}",
                "start_time": f"2026-06-0{number} 09:00:00",
                "end_time": f"2026-06-0{number} 10:00:00",
            }
            for number in (1, 2, 3)
        ],
        "scenarios": [
            {
                "scenario_id": "scenario_1",
                "label_zh": "情景一",
                "label_en": "Scenario 1",
                "aggregation": "all",
                "segment_ids": ["market_01"],
            },
            {
                "scenario_id": "scenario_2",
                "label_zh": "情景二",
                "label_en": "Scenario 2",
                "aggregation": "all",
                "segment_ids": ["market_02"],
            },
            {
                "scenario_id": "scenario_3",
                "label_zh": "情景三",
                "label_en": "Scenario 3",
                "aggregation": "all",
                "segment_ids": ["market_01", "market_02", "market_03"],
            },
        ],
    }
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return load_scenario_contract(path)


def test_scenario_three_requires_all_three_market_segments(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    coordinates = pd.DataFrame(
        [
            {"combo_id": "all_three", "method": "rolling_tr_sum", "baseline_sampling_policy": "all_window"},
            {"combo_id": "first_two", "method": "rolling_tr_sum", "baseline_sampling_policy": "exclude_marked"},
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "combo_id": combo_id,
                "entry_time": f"2026-06-0{number} 09:30:00",
                "exit_time": f"2026-06-0{number} 10:30:00",
                "exit_reason": "rebound_threshold",
            }
            for combo_id, numbers in (("all_three", (1, 2, 3)), ("first_two", (1, 2)))
            for number in numbers
        ]
    )
    details = evaluate_segment_qualification(coordinates, trades, contract)
    classified, scenario_details = attach_scenario_groups(coordinates, details, contract)
    indexed = classified.set_index("combo_id")
    assert bool(indexed.loc["all_three", "scenario_3_qualified"])
    assert not bool(indexed.loc["first_two", "scenario_3_qualified"])
    assert bool(indexed.loc["first_two", "scenario_1_qualified"])
    assert bool(indexed.loc["first_two", "scenario_2_qualified"])
    rejected = scenario_details.loc[
        scenario_details.combo_id.eq("first_two")
        & scenario_details.scenario_id.eq("scenario_3")
    ].iloc[0]
    assert rejected.failed_segment_ids == "market_03"


def test_single_scenario_filter_rejects_multi_selection(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    rows = pd.DataFrame(
        [
            {
                "combo_id": "a",
                "scenario_1_qualified": True,
                "scenario_2_qualified": False,
                "scenario_3_qualified": False,
            },
            {
                "combo_id": "b",
                "scenario_1_qualified": False,
                "scenario_2_qualified": True,
                "scenario_3_qualified": False,
            },
        ]
    )
    assert filter_single_scenario(rows, "scenario_1", contract).combo_id.tolist() == ["a"]
    assert len(filter_single_scenario(rows, "all", contract)) == 2
    with pytest.raises(TypeError, match="exactly one"):
        filter_single_scenario(rows, ["scenario_1", "scenario_2"], contract)  # type: ignore[arg-type]


def test_combined_scenario_accepts_speed_exit_after_segment_end(tmp_path: Path) -> None:
    payload = {
        "schema_version": 3,
        "status": "current",
        "scenario_schema_id": COMBINED_SCENARIO_SCHEMA_ID,
        "selection_mode": "single",
        "neutral_selection_id": "all",
        "qualification_rule": {
            "entry_interval": "start_exclusive_end_inclusive",
            "exit_interval": "start_exclusive_end_inclusive",
            "required_entry_count": 1,
            "required_exit_count": 0,
            "must_hold_past_segment_end": True,
            "required_eventual_exit_reasons": [
                "rebound_threshold",
                "downside_speed_below_threshold",
            ],
        },
        "segments": [
            {
                "segment_id": "market_01",
                "label_zh": "行情一",
                "label_en": "Market 1",
                "start_time": "2026-06-01 09:00:00",
                "end_time": "2026-06-01 10:00:00",
            }
        ],
        "scenarios": [
            {
                "scenario_id": "scenario_1",
                "label_zh": "情景一",
                "label_en": "Scenario 1",
                "aggregation": "all",
                "segment_ids": ["market_01"],
            }
        ],
    }
    path = tmp_path / "combined_scenarios.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    contract = load_scenario_contract(path)
    coordinates = pd.DataFrame([{
        "combo_id": "speed",
        "method": "rolling_tr_sum",
        "baseline_sampling_policy": "all_window",
    }])
    trades = pd.DataFrame(
        [
            {
                "combo_id": "speed",
                "entry_time": "2026-06-01 09:30:00",
                "exit_time": "2026-06-01 10:30:00",
                "exit_reason": "downside_speed_below_threshold",
            }
        ]
    )
    details = evaluate_segment_qualification(coordinates, trades, contract)
    assert bool(details.iloc[0].qualified)
