from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

import build_v4_4_cross_instrument_comparison as cross
from analyze_v4_4_scenario_3_stage import _legacy_v4_main_html


def test_cross_page_contract_contains_required_controls_and_no_score() -> None:
    html = cross._comparison_html(base_css=":root{--bg:#fff}")
    for token in (
        'id="scope-strip"',
        'id="return-view-controls"',
        'id="source-instrument"',
        'id="source-interval"',
        'id="target-instrument"',
        'id="target-interval"',
        'id="global-filter"',
        'id="filter-field"',
        'id="filter-operator"',
        'id="filter-value"',
        'id="active-filters"',
        'id="comparison-table"',
        "returnView==='gross'",
        "_total_return",
        "_median_trade",
        "_max_drawdown_abs",
        "_gross_win_rate",
        "正收益一致性",
        "排名百分位变化",
        "稳定区域",
        "孤立正收益点",
        "事后全网格诊断",
        "综合 score",
        "手续费／滑点后",
        "无手续费／滑点",
        "${sourceName} 总收益",
        "${targetName} 总收益",
        "${sourceName} 回测区间为 ${sourceRange}",
        "`#${num(value)}`",
        'class="rank-link"',
        'target="_blank"',
        'rel="noopener"',
        "matches.find(item=>item.run_id===DATA.runId)||matches[0]",
    ):
        assert token in html
    for token in (
        "候选来源",
        "查看逐笔",
        "['trade_review','逐笔'",
        "['combo_id','combo_id'",
        "SImain 成本后总收益",
        "K200 成本后总收益",
    ):
        assert token not in html
    source_total = html.index("[source.total,`${sourceName} 总收益`")
    target_total = html.index("[target.total,`${targetName} 总收益`")
    target_median = html.index("[target.median,`${targetName} 中位单笔`")
    assert source_total < target_total < target_median
    assert "target_mfe_points_median" in html
    assert "target_mae_points_median" in html
    assert "source_mfe_points_median" in html
    assert "source_mae_points_median" in html


def test_cross_page_navigation_preserves_the_historical_main_template() -> None:
    historical = _legacy_v4_main_html()
    assert "跨品种对比" not in historical
    source = inspect.getsource(cross.publish_current_main_standalone_view)
    assert "entry-nav" not in source
    assert "location.replace" in source
    assert "redirect_to_stable_main" in source
    assert 'relative_main = "main/index.html"' in source
    assert "publish_stable_main_assets" in source
    assert "historical snapshot index changed" in source


def test_run_catalog_includes_current_and_excludes_partial_prior_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    current = runs / "current"
    complete = runs / "complete"
    partial = runs / "partial"
    for root, run_id in ((current, "current"), (complete, "complete"), (partial, "partial")):
        root.mkdir(parents=True)
        (root / "run_config.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "source": {
                        "instrument": "K200",
                        "sample_start": "2026-05-26",
                        "sample_end": "2026-07-08",
                    },
                    "target": {
                        "instrument": "SImain",
                        "sample_start": "2026-01-29",
                        "sample_end": "2026-02-23",
                    },
                }
            ),
            encoding="utf-8",
        )
    (complete / "index.html").write_text("complete", encoding="utf-8")
    (complete / "cross_instrument_manifest.json").write_text("{}", encoding="utf-8")
    (partial / "index.html").write_text("partial", encoding="utf-8")

    catalog = cross._run_catalog(tmp_path, current)

    assert [item["run_id"] for item in catalog] == ["complete", "current"]


def test_cross_qa_declares_the_candidate_count_argument() -> None:
    qa_source = (Path(cross.__file__).with_name("qa_v4_4_cross_instrument_comparison.mjs")).read_text(
        encoding="utf-8"
    )
    assert "await trade.waitForFunction((count) => (" in qa_source
    assert "), expectedCandidateCount, { timeout: 180000 });" in qa_source


def test_rank_percentile_uses_one_hundred_as_best() -> None:
    values = pd.Series([0.1, 0.3, 0.2])
    percentiles = cross._rank_percentile(values, ascending=False)
    assert percentiles.tolist() == [0.0, 100.0, 50.0]


def test_adjacency_requires_one_axis_change() -> None:
    common = {
        "method": "rolling_tr_sum",
        "baseline_sampling_policy": "all_window",
        "entry_fill_mode": "calculated_threshold",
        "entry_execution_policy": "wait_next_real_trade",
        "entry_slippage": 0.0,
        "e": 320,
        "bh": 240,
        "trw": 12,
        "k": 1.25,
        "w": 6,
        "m": 4.5,
        "speed_window_bars": 400,
    }
    frame = pd.DataFrame(
        [
            {**common, "combo_id": "a"},
            {**common, "combo_id": "b", "speed_window_bars": 420},
            {**common, "combo_id": "c", "speed_window_bars": 440},
            {**common, "combo_id": "d", "e": 300, "speed_window_bars": 420},
        ]
    )
    adjacency = cross._adjacency(frame)
    assert "b" in adjacency["a"]
    assert "c" not in adjacency["a"]
    assert "d" not in adjacency["a"]


def test_target_evaluation_reads_frozen_candidates_and_never_calls_freeze() -> None:
    source = inspect.getsource(cross.evaluate_target)
    assert "load_frozen_candidates()" in source
    assert "freeze_candidates(" not in source
    assert "SIMAIN_SOURCE" in source


def test_cost_contract_comes_from_price_derived_instrument_profile() -> None:
    model = cross.K200_PROFILE["normalized_cost_model"]
    assert cross.ROUND_TRIP_COST_BPS == pytest.approx(
        model["round_trip_total_cost_bps"]
    )
    assert model["contract_notional_quote"] == pytest.approx(
        model["reference_price"] * model["point_value"]
    )


def test_gross_view_metrics_are_derived_from_trade_records(tmp_path: Path) -> None:
    path = tmp_path / "trades.csv"
    pd.DataFrame(
        [
            {"combo_id": "a", "gross_return": 0.10},
            {"combo_id": "a", "gross_return": -0.05},
            {"combo_id": "b", "gross_return": 0.02},
        ]
    ).to_csv(path, index=False)
    metrics = cross._gross_view_metrics_from_trade_records(path, prefix="target").set_index(
        "combo_id"
    )
    assert metrics.loc["a", "target_gross_total_return"] == pytest.approx(0.045)
    assert metrics.loc["a", "target_gross_median_trade"] == pytest.approx(0.025)
    assert metrics.loc["a", "target_gross_win_rate"] == pytest.approx(0.5)
