from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = Path(__file__).with_name("apply_v4_41_scenario.py")
SPEC = importlib.util.spec_from_file_location("apply_scenario", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
apply_scenario = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(apply_scenario)


def test_catalog_registers_three_markets_and_current_scenarios() -> None:
    market_catalog = json.loads(
        (ROOT / "runtime_inputs" / "scenarios" / "market_catalog.json").read_text(
            encoding="utf-8"
        )
    )
    scenario_catalog = json.loads(
        (ROOT / "runtime_inputs" / "scenarios" / "scenario_catalog.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["label_zh"] for row in market_catalog["markets"]] == [
        "K200 训",
        "K200 测",
        "SI 当前阶段",
    ]
    assert [row["scenario_id"] for row in scenario_catalog["scenarios"]] == [
        "scenario_1",
        "scenario_2",
        "scenario_3",
    ]
    assert len(scenario_catalog["scenarios"][2]["segments"]) == 3


def test_trade_application_requires_every_selected_segment(tmp_path: Path) -> None:
    trades = pd.DataFrame(
        [
            {
                "combo_id": "pass",
                "entry_time": "2026-01-01 10:05:00",
                "exit_time": "2026-01-01 11:00:00",
                "exit_reason": "rebound_threshold",
            },
            {
                "combo_id": "pass",
                "entry_time": "2026-01-02 10:05:00",
                "exit_time": "2026-01-02 11:00:00",
                "exit_reason": "downside_speed_below_threshold",
            },
            {
                "combo_id": "fail",
                "entry_time": "2026-01-01 10:05:00",
                "exit_time": "2026-01-01 10:08:00",
                "exit_reason": "rebound_threshold",
            },
        ]
    )
    trade_path = tmp_path / "trades.csv"
    trades.to_csv(trade_path, index=False)
    scenario = {
        "segments": [
            {
                "segment_id": "a",
                "start_time": "2026-01-01 10:00:00",
                "end_time": "2026-01-01 10:10:00",
            },
            {
                "segment_id": "b",
                "start_time": "2026-01-02 10:00:00",
                "end_time": "2026-01-02 10:10:00",
            },
        ]
    }
    rule = {
        "required_entry_count": 1,
        "required_exit_count": 0,
        "required_eventual_exit_reasons": [
            "rebound_threshold",
            "downside_speed_below_threshold",
        ],
    }
    qualified, details = apply_scenario.evaluate_from_trades(
        scenario, rule, trade_path, {"pass", "fail"}
    )
    assert qualified == {"pass"}
    assert len(details) == 4


def test_generated_page_keeps_current_main_controls(tmp_path: Path) -> None:
    market = {"label_zh": "K200 测"}
    scenario = {"label_zh": "测试场景"}
    html = apply_scenario.scenario_html(ROOT / "results" / "_scenario_test", market, scenario)
    assert "交易数" in html
    assert "大于" in html and "小于" in html
    assert "const scenarios=Array.isArray(DATA.scenarios)" in html
    assert "V4.41 K200 测 · 测试场景结果排序" in html
