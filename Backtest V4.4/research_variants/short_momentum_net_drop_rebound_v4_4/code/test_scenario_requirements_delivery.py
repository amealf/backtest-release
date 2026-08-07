from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analyze_v4_4_scenario_3_stage import (  # noqa: E402
    COST_ADJUSTED_AVG_TRADE_KEY,
    COST_ADJUSTED_MAX_DRAWDOWN_KEY,
    COST_ADJUSTED_RETURN_KEY,
    K200M_COST_MODEL,
    MARKET_SELECTOR_SOURCE_SHA256,
    SCENARIO_REQUIREMENTS_TEMPLATE_PATH,
    HIGH_RETURN_VIEWS,
    _legacy_v4_main_html,
    _apply_cost_adjusted_metrics,
    _rank,
    build_scenario_requirements_delivery,
    sha256_file,
)


SCENARIO_DEFINITION = (
    CODE_DIR.parent
    / "plans"
    / "v4_4_scenario_groups_single_select_combined_exit_20260801.json"
)


def test_main_entry_exposes_current_scenario_requirements_in_a_new_page() -> None:
    html = _legacy_v4_main_html()
    assert 'id="scenario-requirements-link"' in html
    assert 'target="_blank"' in html
    assert "scenarioRequirementsRoute" in html
    assert "scenario_requirements/index.html?scenario={scenario_id}" in html
    assert "scenario-control-row" in html


def test_main_entry_exposes_four_separately_labeled_high_return_views() -> None:
    assert [view["id"] for view in HIGH_RETURN_VIEWS] == [
        "scenario_1_qualified_total_return",
        "unrestricted_total_return",
        "unrestricted_average_return_ge10",
        "unrestricted_average_return_ge20",
    ]
    assert [
        (view["scenario_filter"], view["minimum_trade_count"], view["metric"])
        for view in HIGH_RETURN_VIEWS
    ] == [
        ("scenario_1", 0, "total_return"),
        ("all", 0, "total_return"),
        ("all", 10, "average_trade"),
        ("all", 20, "average_trade"),
    ]
    assert all(view["tie_break"] == "combo_id" for view in HIGH_RETURN_VIEWS)
    assert [view["metric_key"] for view in HIGH_RETURN_VIEWS] == [
        COST_ADJUSTED_RETURN_KEY,
        COST_ADJUSTED_RETURN_KEY,
        COST_ADJUSTED_AVG_TRADE_KEY,
        COST_ADJUSTED_AVG_TRADE_KEY,
    ]
    assert [view["gross_metric_key"] for view in HIGH_RETURN_VIEWS] == [
        "train_return",
        "train_return",
        "train_avg_trade",
        "train_avg_trade",
    ]

    html = _legacy_v4_main_html()
    assert 'id="high-return-view-controls"' in html
    assert 'data-high-return-view="${esc(view.id)}"' in html
    assert "['ge10','至少 10 笔',10]" in html
    assert "['gap_excluded_return','排除 gap 收益'" not in html
    assert "['maximum_drawdown','最大回撤'" not in html
    assert "gap 依赖审计" not in html
    assert "V4.41 K200回测结果排序" in html
    assert "#table{height:calc(100vh - 58px);max-height:none;overflow:auto" in html
    assert "#table th{top:0;z-index:3" in html
    assert '>查看 #${currentRanks[rowIndex]}</a>' not in html
    assert '>#${currentRanks[rowIndex]}</a>' in html
    assert ".rank-link{width:90px}" in html
    assert "const PAGE_SIZE=500;" in html
    assert "rows.slice(pageStart,pageEnd)" in html
    assert 'id="pager"' in html
    assert "['rank','排名']" in html
    assert "['segment_end_exit_count','segment_end']" not in html
    assert "['waited_entry_count','等待成交']" not in html
    assert "['maximum_entry_wait_bars','最长等待']" not in html
    assert 'id="contract-panel"' not in html
    assert "function renderContract()" not in html
    assert ".shell{max-width:none" in html
    assert 'class="cross-instrument-link"' not in html
    assert "['train_cost_adjusted_return','总收益']" in html
    assert "['train_cost_adjusted_avg_trade','笔均']" in html
    assert "['train_cost_adjusted_max_drawdown_abs','回撤']" in html
    assert html.index("['combo_id','参数组合']") < html.index("['round_trip_cost_bps','往返成本 bps']")
    assert html.index("['round_trip_cost_bps','往返成本 bps']") < html.index("['gap_spanning_trade_count','跨 gap 笔数']")
    assert 'id="all-strategy-count"' in html
    assert "全部策略 ${number(DATA.coordinateCount)}" in html
    assert ".cards{display:none}" in html
    assert ".control-group.window-filter{grid-column:1/-1}" in html
    assert html.count("const cm=DATA.costModel||{};") == 1
    assert html.index("const cm=DATA.costModel||{};") < html.index("function render(){")
    assert 'id="return-view-controls"' in html
    assert "排序与显示使用同一口径；默认采用手续费／滑点后" in html
    assert "cm.round_trip_slippage_bps" in html
    assert "cm.round_trip_commission" in html
    assert "cm.contract_notional_quote" in html
    assert "cm.quote_currency" in html
    assert "成本后排名" not in html
    for view in HIGH_RETURN_VIEWS:
        assert view["label"] in html or "highReturnViews" in html


def test_scenario_3_cost_adjusted_return_exact_ties_use_combo_id_only() -> None:
    rows = pd.DataFrame(
        [
            {
                "combo_id": "combo_b",
                "method": "rolling_tr_sum",
                "baseline_sampling_policy": "all_window",
                "scenario_3_qualified": True,
                "train_return": 0.2,
                COST_ADJUSTED_RETURN_KEY: 0.1,
                COST_ADJUSTED_AVG_TRADE_KEY: 0.01,
                COST_ADJUSTED_MAX_DRAWDOWN_KEY: -0.01,
                "train_max_drawdown_abs": 0.01,
                "train_trade_count": 30,
            },
            {
                "combo_id": "combo_a",
                "method": "rolling_tr_sum",
                "baseline_sampling_policy": "all_window",
                "scenario_3_qualified": True,
                "train_return": 0.2,
                COST_ADJUSTED_RETURN_KEY: 0.1,
                COST_ADJUSTED_AVG_TRADE_KEY: 0.01,
                COST_ADJUSTED_MAX_DRAWDOWN_KEY: -0.9,
                "train_max_drawdown_abs": 0.9,
                "train_trade_count": 10,
            },
            {
                "combo_id": "combo_c",
                "method": "rolling_tr_sum",
                "baseline_sampling_policy": "all_window",
                "scenario_3_qualified": True,
                "train_return": 0.1,
                COST_ADJUSTED_RETURN_KEY: 0.05,
                COST_ADJUSTED_AVG_TRADE_KEY: 0.005,
                COST_ADJUSTED_MAX_DRAWDOWN_KEY: -0.001,
                "train_max_drawdown_abs": 0.001,
                "train_trade_count": 100,
            },
        ]
    )
    ranked = _rank(rows).sort_values("scenario_3_total_return_rank")
    assert ranked.combo_id.tolist() == ["combo_a", "combo_b", "combo_c"]


def test_shadow_cost_model_penalizes_turnover_and_can_reverse_gross_order() -> None:
    many_returns = [0.001] * 100
    few_returns = [0.08]
    summary = pd.DataFrame(
        [
            {
                "combo_id": "many",
                "method": "rolling_tr_sum",
                "baseline_sampling_policy": "all_window",
                "scenario_3_qualified": True,
                "train_return": (1.001**100) - 1.0,
                "train_avg_trade": 0.001,
                "train_max_drawdown": 0.0,
                "train_trade_count": 100,
            },
            {
                "combo_id": "few",
                "method": "rolling_tr_sum",
                "baseline_sampling_policy": "all_window",
                "scenario_3_qualified": True,
                "train_return": 0.08,
                "train_avg_trade": 0.08,
                "train_max_drawdown": 0.0,
                "train_trade_count": 1,
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "combo_id": combo_id,
                "entry_index": index * 2,
                "exit_index": index * 2 + 1,
                "return": value,
            }
            for combo_id, values in (("many", many_returns), ("few", few_returns))
            for index, value in enumerate(values)
        ]
    )
    adjusted, adjusted_trades = _apply_cost_adjusted_metrics(summary, trades)
    ranked = _rank(adjusted).sort_values("scenario_3_cost_adjusted_return_rank")

    expected_commission_bps = (
        10000.0
        * K200M_COST_MODEL["round_trip_commission_usd"]
        * K200M_COST_MODEL["usdkrw"]
        / K200M_COST_MODEL["contract_notional_krw"]
    )
    assert math.isclose(
        K200M_COST_MODEL["round_trip_commission_bps"],
        expected_commission_bps,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        K200M_COST_MODEL["round_trip_total_cost_bps"],
        K200M_COST_MODEL["round_trip_slippage_bps"] + expected_commission_bps,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert adjusted.loc[
        adjusted.combo_id.eq("many"), "train_return"
    ].iloc[0] > adjusted.loc[adjusted.combo_id.eq("few"), "train_return"].iloc[0]
    assert ranked.combo_id.tolist() == ["few", "many"]
    assert adjusted_trades.round_trip_cost_bps.eq(
        K200M_COST_MODEL["round_trip_total_cost_bps"]
    ).all()
    assert adjusted_trades.round_trip_commission_krw.eq(
        K200M_COST_MODEL["round_trip_commission_krw"]
    ).all()


def test_browser_qa_reconciles_scenario_3_and_exercises_all_four_views() -> None:
    qa_source = (CODE_DIR / "qa_v4_4_scenario_3_stage.mjs").read_text(
        encoding="utf-8"
    )
    assert "Scenario 3 is not a rendered empty set" not in qa_source
    assert "Scenario 3 manifest/data reconciliation failed" in qa_source
    assert "expectedHighReturnViews" in qa_source
    assert "high_return_view_states" in qa_source


def test_scenario_requirements_delivery_reuses_selector_contract(
    tmp_path: Path,
) -> None:
    result = build_scenario_requirements_delivery(
        tmp_path / "scenario_requirements",
        SCENARIO_DEFINITION,
    )
    html = Path(result["index"]["path"]).read_text(encoding="utf-8")
    data_text = Path(result["data"]["path"]).read_text(encoding="utf-8")
    payload = json.loads(data_text.removeprefix(
        "window.V4_4_SCENARIO_REQUIREMENTS="
    ).removesuffix(";\n"))

    assert SCENARIO_REQUIREMENTS_TEMPLATE_PATH.is_file()
    assert "__BACK_HREF__" not in html
    assert "__PLOTLY_HREF__" not in html
    assert "__PROCESS_PAYLOAD_HREF__" not in html
    assert '../assets/plotly.min.js' in html
    assert '../trade_review/process_payload.js' in html
    assert 'id="scenario-tabs"' in html
    assert 'id="chart"' in html
    assert "Plotly.react" in html
    assert payload["qualification_rule"]["required_entry_count"] == 1
    assert payload["qualification_rule"]["required_exit_count"] == 0
    assert payload["qualification_rule"]["must_hold_past_segment_end"] is True
    assert [row["scenario_id"] for row in payload["scenarios"]] == [
        "scenario_1",
        "scenario_2",
        "scenario_3",
    ]
    assert len(payload["segments"]) == 3
    assert payload["market_selector_source"]["sha256"] == (
        MARKET_SELECTOR_SOURCE_SHA256
    )
    assert sha256_file(Path(result["selector_source"]["path"])) == (
        MARKET_SELECTOR_SOURCE_SHA256
    )
