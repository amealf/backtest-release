from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_v4_4_review_delivery import (  # noqa: E402
    BASELINE_SAMPLING_POLICY_CONTRACTS,
    DEFAULT_SOURCE_MANIFEST,
    FILTER_OVERLAY_ID,
    FILTER_OVERLAY_SCOPE,
    FINGERPRINT_SCHEMA_VERSION,
    HOME_SCRIPT_TEMPLATE,
    HOME_TEMPLATE,
    HUB_DESIGN_SOURCE,
    LEGACY_STANDALONE_ROOT,
    LEGACY_STANDALONE_STAGE,
    LEGACY_STANDALONE_VALIDATION_STAGE,
    OUTPUT_SCHEMA_VERSION,
    STYLE_TEMPLATE,
    TRADE_DESIGN_SOURCE,
    TRADE_DESIGN_SOURCE_SHA256,
    TRADE_SCRIPT_TEMPLATE,
    TRADE_TEMPLATE,
    TRADE_DETAIL_FIELDS,
    TRADE_PLOTLY_SOURCE_SHA256,
    _combo_key,
    _historical_trade_html,
    _sha256,
    _trade_catalog_record,
    _validate_trade_entry,
    _validate_trade_exit,
    build,
    build_stage_trade_review,
)
from analyze_v4_4_scenario_3_stage import (  # noqa: E402
    APPROVED_PLAN_STATUSES,
    _legacy_v4_main_html,
    _validate_trades,
)
from v4_4_engine import (  # noqa: E402
    COMBINED_TRADE_AUDIT_SCHEMA_ID,
    COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
    REBOUND_BASELINE_POLICY_ID,
    strategy_id,
)


def _source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": ["2026-06-01 10:00:00", "2026-06-01 10:00:15"],
            "open": [100.0, 98.0],
            "high": [101.0, 99.0],
            "low": [97.0, 96.0],
            "close": [98.5, 97.0],
            "volume": [10.0, 12.0],
            "trade_count": [5.0, 6.0],
            "is_synthetic_empty_bar": [False, False],
        }
    )


def test_calculated_entry_validation_accepts_threshold_fill_below_signal_open() -> None:
    row = pd.Series(
        {
            "entry_index": 0,
            "initial_entry_index": 0,
            "signal_index": 0,
            "entry_time": "2026-06-01 10:00:00",
            "initial_entry_time": "2026-06-01 10:00:00",
            "entry_bar_synthetic": False,
            "entry_bar_volume": 10.0,
            "entry_bar_trade_count": 5.0,
            "initial_entry_bar_synthetic": False,
            "initial_entry_bar_volume": 10.0,
            "initial_entry_bar_trade_count": 5.0,
            "entry_price_before_slippage": 99.0,
            "entry_trigger_price": 99.0,
            "entry_wait_bar_count": 0,
            "entry_fill_mode": "calculated_threshold",
            "entry_fill_source": "calculated_threshold",
        }
    )
    _validate_trade_entry(row, _source_rows())


def test_calculated_entry_validation_accepts_lower_signal_open_after_gap() -> None:
    row = pd.Series(
        {
            "entry_index": 1,
            "initial_entry_index": 1,
            "signal_index": 1,
            "entry_time": "2026-06-01 10:00:15",
            "initial_entry_time": "2026-06-01 10:00:15",
            "entry_bar_synthetic": False,
            "entry_bar_volume": 12.0,
            "entry_bar_trade_count": 6.0,
            "initial_entry_bar_synthetic": False,
            "initial_entry_bar_volume": 12.0,
            "initial_entry_bar_trade_count": 6.0,
            "entry_price_before_slippage": 98.0,
            "entry_trigger_price": 99.0,
            "entry_wait_bar_count": 0,
            "entry_fill_mode": "calculated_threshold",
            "entry_fill_source": "calculated_threshold",
        }
    )
    _validate_trade_entry(row, _source_rows())


def test_pending_exit_validation_requires_next_real_trade_open() -> None:
    row = pd.Series(
        {
            "exit_index": 1,
            "exit_time": "2026-06-01 10:00:15",
            "exit_price": 98.0,
            "exit_reason": "rebound_threshold",
            "exit_bar_synthetic": False,
            "exit_bar_volume": 12.0,
            "exit_bar_trade_count": 6.0,
            "pending_exit": True,
            "pending_exit_trigger_index": 0,
            "pending_exit_wait_bar_count": 1,
            "pending_exit_fill_policy": "next_real_trade_bar_open",
        }
    )
    evidence = _validate_trade_exit(row, _source_rows())
    assert evidence["o"] == 98.0


def test_signal_exit_validation_rejects_a_synthetic_fill_bar() -> None:
    source = _source_rows()
    source.loc[1, ["volume", "trade_count"]] = 0
    source.loc[1, "is_synthetic_empty_bar"] = True
    row = pd.Series(
        {
            "exit_index": 1,
            "exit_time": "2026-06-01 10:00:15",
            "exit_price": 98.0,
            "exit_reason": "downside_speed_below_threshold",
            "exit_bar_synthetic": True,
            "exit_bar_volume": 0.0,
            "exit_bar_trade_count": 0.0,
            "pending_exit": False,
            "pending_exit_trigger_index": None,
            "pending_exit_wait_bar_count": 0,
            "pending_exit_fill_policy": "same_real_trade_bar",
        }
    )
    with pytest.raises(ValueError, match="not a real-trade bar"):
        _validate_trade_exit(row, source)

CURRENT_COMBINED_STAGE = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "fixtures"
    / "combined_stage"
)


def _trade() -> pd.Series:
    return pd.Series(
        {
            "combo_id": "v4_4_rolling_tr_sum_demo",
            "method": "rolling_tr_sum",
            "baseline_sampling_policy": "all_window",
            "entry_index": 120,
            "signal_time": "2026-06-01 10:00:00",
            "initial_entry_time": "2026-06-01 10:00:15",
            "entry_time": "2026-06-01 10:01:00",
            "exit_time": "2026-06-01 10:20:00",
            "entry_price": 1234.5,
            "exit_price": 1220.0,
            "return": 0.0117,
            "exit_reason": "rebound_threshold",
            "entry_wait_bar_count": 3,
            "position_crosses_real_gap": False,
            "signal_synthetic_empty_bar_count": 2,
            "entry_bar_synthetic": False,
            "entry_bar_volume": 10,
            "entry_bar_trade_count": 7,
            "entry_fill_source": "waited_real_trade_open",
            "e": 60,
            "bh": 480,
            "trw": 17,
            "k": 0.9,
            "w": 24,
            "m": 3.5,
        }
    )


def test_trade_catalog_record_exposes_every_requested_filter_and_real_fill_state() -> None:
    row = _trade_catalog_record(_trade(), 4)
    assert row["id"] == f"{_combo_key(row['combo_id'])}-120"
    assert row["sequence"] == 4
    assert row["waited"] is True
    assert row["wait_bars"] == 3
    assert row["crosses_gap"] is False
    assert row["synthetic_signal"] is True
    assert row["synthetic_signal_bar_count"] == 2
    assert row["actual_entry_real"] is True
    assert {"e", "bh", "trw", "k", "w", "m", "method", "exit_reason"}.issubset(row)


def test_review_templates_are_offline_and_keep_v4_4_navigation_identity() -> None:
    texts = {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            HOME_TEMPLATE,
            TRADE_TEMPLATE,
            STYLE_TEMPLATE,
            HOME_SCRIPT_TEMPLATE,
            TRADE_SCRIPT_TEMPLATE,
        )
    }
    combined = "\n".join(texts.values())
    assert "https://" not in combined
    assert "http://" not in combined
    assert "trade_analysis/index.html" in texts[HOME_TEMPLATE.name]
    assert 'href="{{MAIN_HREF}}"' in texts[TRADE_TEMPLATE.name]
    assert 'id="filter-speed"' in texts[TRADE_TEMPLATE.name]
    assert "design-reuse:" in texts[HOME_TEMPLATE.name]
    assert "report-grid" in texts[HOME_TEMPLATE.name]
    assert "design-reuse:" in texts[TRADE_TEMPLATE.name]
    assert "controls-drawer" in texts[TRADE_TEMPLATE.name]
    assert "detail-layout" in texts[TRADE_TEMPLATE.name]
    assert "真实成交柱 · 通过" in texts[TRADE_SCRIPT_TEMPLATE.name]
    assert "v4_4_" in combined
    assert "rbw13" not in texts[HOME_TEMPLATE.name]
    assert "rbw14" not in texts[TRADE_TEMPLATE.name]


def test_historical_trade_template_adds_static_filter_overlay_without_replacing_layout() -> None:
    html = _historical_trade_html(
        "../index.html",
        peer_review_href="../peer/index.html",
        peer_review_label="显示测试集",
        peer_research_contract_id="peer_contract",
    )
    assert FILTER_OVERLAY_ID in html
    assert "v43MaxWAuditText" in html
    assert "rebound_exit_bar_candidate" in html
    assert "rebound_latest_applied_candidate" in html
    assert "闭合 bar max-W 审计" in html
    assert "当前 bar 候选从下一根 bar 才生效" in html
    assert "V3 回撤基准绑定当前严格 active low" not in html
    assert "反弹期间基准冻结" not in html
    assert "v43FilterEventShapes" in html
    assert "baseline_filter_events" in html
    assert 'id="chart"' in html
    assert "Plotly.react" in html
    assert 'id="peerReviewLink" class="view-btn peer-review-toggle"' in html
    assert 'id="peerReviewOverlay" class="peer-review-overlay"' in html
    assert 'id="peerReviewFrame" class="peer-review-frame"' in html
    assert 'target.searchParams.set("embedded", "1")' in html
    assert 'els.peerReviewFrame.removeAttribute("src")' in html
    assert '"label":"显示测试集"' in html
    assert html.index("initPeerReview();") < html.index("if (allResultsCatalog)")
    assert 'html[data-embedded-review="true"] .selection-card{display:none!important}' in html
    assert 'html[data-peer-review-open="true"] .wrap > :not(.peer-review-overlay){visibility:hidden}' in html


def _valid_max_w_rebound_trade() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "combo_id": "max_w_delivery_contract",
            "method": "rolling_tr_sum",
            "baseline_sampling_policy": "all_window",
            "strategy_id": strategy_id("all_window", combined_exit=True),
            "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
            "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
            "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
            "exit_reason": "rebound_threshold",
            "entry_index": 1,
            "h_index": 1,
            "signal_index": 1,
            "exit_index": 4,
            "entry_time": "2026-06-01 10:00:15",
            "exit_time": "2026-06-01 10:01:00",
            "entry_wait_bar_count": 0,
            "entry_bar_synthetic": False,
            "entry_bar_volume": 10.0,
            "entry_bar_trade_count": 5.0,
            "exit_bar_synthetic": False,
            "exit_bar_volume": 12.0,
            "exit_bar_trade_count": 6.0,
            "exit_price": 91.0,
            "exit_price_basis": "calculated_rebound_threshold",
            "pending_exit": False,
            "pending_exit_trigger_index": None,
            "pending_exit_wait_bar_count": 0,
            "pending_exit_fill_policy": "same_real_trade_bar",
            "speed_window_bars": 2,
            "speed_extension": None,
            "speed_reference_index": None,
            "position_crosses_real_gap": False,
            "holding_bar_distance": 3,
            "holding_minutes": 0.75,
            "active_low": 90.0,
            "w": 2,
            "m": 0.2,
            "rebound_net_drop": 5.0,
            "rebound_max_w_drop": 5.0,
            "rebound_window_start_index": 2,
            "rebound_window_end_index": 3,
            "rebound_window_observed_bar_count": 2,
            "rebound_latest_applied_candidate": 5.0,
            "rebound_latest_applied_candidate_start_index": 2,
            "rebound_latest_applied_candidate_end_index": 3,
            "rebound_latest_applied_candidate_observed_bar_count": 2,
            "rebound_exit_bar_candidate": 4.0,
            "rebound_exit_bar_candidate_start_index": 3,
            "rebound_exit_bar_candidate_end_index": 4,
            "rebound_exit_bar_candidate_observed_bar_count": 2,
            "rebound_candidates_effective_through_index": 3,
            "rebound_threshold": 91.0,
            "rebound_trigger_price": 91.0,
            "rebound_check_price": 91.0,
            "rebound_check_price_basis": "bar_high",
            "rebound_gap_adjusted": False,
            "rebound_gap_slippage": 0.0,
            "rebound_baseline_update_rule": (
                "maximum_positive_completed_bar_w_candidates_effective_next_bar"
            ),
        }]
    )


def test_max_w_delivery_audit_accepts_prior_closed_bar_basis() -> None:
    trade = _valid_max_w_rebound_trade()
    audit = _validate_trades(trade)
    assert audit["max_w_closed_bar_timing_valid"] is True
    assert audit["max_w_exit_candidate_source_valid"] is True
    assert set((
        "rebound_max_w_drop",
        "rebound_latest_applied_candidate",
        "rebound_exit_bar_candidate",
        "rebound_baseline_policy_id",
    )).issubset(TRADE_DETAIL_FIELDS)


def test_max_w_delivery_audit_rejects_same_bar_rebound_candidate_as_effective() -> None:
    trade = _valid_max_w_rebound_trade()
    trade.loc[0, "rebound_candidates_effective_through_index"] = 4
    with pytest.raises(ValueError, match="max_w_closed_bar_timing_valid"):
        _validate_trades(trade)


def test_post_audit_plan_status_is_an_exact_approved_state() -> None:
    assert APPROVED_PLAN_STATUSES == (
        "approved_for_execution",
        "approved_for_execution_after_identity_and_plan_audit",
        "approved_for_exact_result_semantics_repair",
    )


def test_same_signal_immediate_rebound_uses_pre_signal_source_without_applied_state() -> None:
    trade = _valid_max_w_rebound_trade()
    trade.loc[0, ["entry_index", "signal_index", "exit_index"]] = 4
    trade.loc[0, "entry_time"] = trade.loc[0, "exit_time"]
    trade.loc[0, ["holding_bar_distance", "holding_minutes"]] = 0
    trade.loc[0, [
        "rebound_window_start_index",
        "rebound_window_end_index",
        "rebound_window_observed_bar_count",
    ]] = [3, 3, 1]
    trade.loc[0, "rebound_latest_applied_candidate"] = None
    trade.loc[0, [
        "rebound_latest_applied_candidate_start_index",
        "rebound_latest_applied_candidate_end_index",
        "rebound_latest_applied_candidate_observed_bar_count",
        "rebound_candidates_effective_through_index",
    ]] = [-1, -1, 0, -1]
    audit = _validate_trades(trade)
    assert audit["max_w_source_window_valid"] is True
    assert audit["max_w_latest_applied_source_valid"] is True


def test_same_signal_immediate_rebound_rejects_nonadjacent_source_end() -> None:
    trade = _valid_max_w_rebound_trade()
    trade.loc[0, ["entry_index", "signal_index", "exit_index"]] = 4
    trade.loc[0, "entry_time"] = trade.loc[0, "exit_time"]
    trade.loc[0, ["holding_bar_distance", "holding_minutes"]] = 0
    trade.loc[0, [
        "rebound_window_start_index",
        "rebound_window_end_index",
        "rebound_window_observed_bar_count",
    ]] = [2, 2, 1]
    trade.loc[0, "rebound_latest_applied_candidate"] = None
    trade.loc[0, [
        "rebound_latest_applied_candidate_start_index",
        "rebound_latest_applied_candidate_end_index",
        "rebound_latest_applied_candidate_observed_bar_count",
        "rebound_candidates_effective_through_index",
    ]] = [-1, -1, 0, -1]
    with pytest.raises(ValueError, match="max_w_source_window_valid"):
        _validate_trades(trade)


def test_closed_max_w_review_uses_four_workers_and_hash_gated_historical_assets(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.csv"
    _source_rows().to_csv(source_path, index=False)
    marker_path = tmp_path / "filter_atoms.csv"
    marker_path.write_text("datetime,baseline_excluded\n", encoding="utf-8")
    events_path = tmp_path / "filter_events.json"
    events_path.write_text('{"events":[]}\n', encoding="utf-8")
    preparation_path = tmp_path / "data_preparation_manifest.json"
    preparation_path.write_text(
        json.dumps({
            "status": "complete",
            "prepared_identity": "v4_4_policy_neutral_baseline_marker_test",
            "source_sha256": _sha256(source_path),
            "artifacts": {
                "filter_atoms": {
                    "path": str(marker_path),
                    "sha256": _sha256(marker_path),
                },
                "filter_events": {
                    "path": str(events_path),
                    "sha256": _sha256(events_path),
                },
            },
        }),
        encoding="utf-8",
    )
    summary = pd.DataFrame([
        {
            "combo_id": f"max_w_combo_{index}",
            "method": "rolling_tr_sum",
            "baseline_sampling_policy": "all_window",
            "e": 4,
            "bh": 8,
            "trw": 2,
            "k": 0.9,
            "w": 2,
            "m": 0.2,
            "speed_window_bars": 2,
            "speed_exit_enabled": True,
            "rebound_exit_enabled": True,
            "entry_fill_mode": "calculated_threshold",
            "entry_execution_policy": "wait_next_real_trade",
            "entry_slippage": 0.0,
            "train_trade_count": 0,
            "train_return": 0.0,
            "train_return_excluding_gap_spanning_trades": 0.0,
            "train_avg_trade": None,
            "train_max_drawdown": 0.0,
            "train_max_drawdown_abs": 0.0,
            "gap_spanning_trade_count": 0,
            "synthetic_signal_trade_count": 0,
            "segment_end_exit_count": 0,
            "rebound_exit_count": 0,
            "speed_exit_count": 0,
            "scenario_1_qualified": False,
            "scenario_2_qualified": False,
            "scenario_3_qualified": False,
            "event_01_qualified": False,
            "event_02_qualified": False,
            "short_drop_3_15m_member": False,
        }
        for index in range(4)
    ])
    stage_manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "plan_fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "version_label": "V4.4",
        "strategy_id": strategy_id("all_window", combined_exit=True),
        "result_semantics_id": "max_completed_w_drop_rebound_results_v2_test",
        "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "campaign_id": "max_w_review_test",
        "stage_id": "round_01",
        "exit_mode": "combined",
        "baseline_sampling_policy": "all_window",
        "baseline_sampling_policies": ["all_window"],
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "data_preparation_manifest": str(preparation_path),
        "engine_sha256": "engine-test-hash",
        "train_start": "2026-06-01 10:00:00",
        "train_end": "2026-06-01 10:00:15",
    }
    completion = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "strategy_id": stage_manifest["strategy_id"],
        "result_semantics_id": stage_manifest["result_semantics_id"],
        "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "campaign_id": stage_manifest["campaign_id"],
        "stage_id": stage_manifest["stage_id"],
        "exit_mode": "combined",
        "coordinate_count": 4,
        "trade_count": 0,
    }
    result = build_stage_trade_review(
        tmp_path / "trade_review",
        summary,
        pd.DataFrame(),
        stage_manifest,
        completion,
        analysis_identity="max_w_review_test",
        workers=4,
    )
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    resource_audit = json.loads(
        (Path(result["index"]).parent / "resource_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["closure"]["generation_worker_count"] == 4
    assert manifest["closure"]["max_w_audit_fields_retained"] is True
    assert manifest["template_source"]["sha256"] == TRADE_DESIGN_SOURCE_SHA256
    assert manifest["plotly_output"]["sha256"] == TRADE_PLOTLY_SOURCE_SHA256
    assert manifest["raw_output_schema_version"] == OUTPUT_SCHEMA_VERSION
    assert manifest["plan_fingerprint_schema_version"] == FINGERPRINT_SCHEMA_VERSION
    assert resource_audit["max_w_audit_contract"]["rebound_baseline_policy_id"] == (
        REBOUND_BASELINE_POLICY_ID
    )
    assert len(result["chunks"]) == 4
    html = Path(result["index"]).read_text(encoding="utf-8")
    assert "v43MaxWAuditText" in html
    assert "top:calc(84px + 8px)" in html
    assert "top:calc(84px + 14px)" not in html
    assert (
        'function pointShape(cx, cy, range, color)' in html
        and 'fillcolor:"rgba(0,0,0,0)",line:{color,width:1.2}' in html
    )
    assert 'function vline(cx, color)' in html
    assert 'function hlineY3(value, color)' in html
    assert 'dash:"dash"' not in html
    assert 'dash:"dot"' not in html
    assert "虚线提示" not in html
    assert "· 紫色" not in html
    assert "· 橙色" not in html
    assert "pushAuditAnnotation(`cover=" not in html
    assert "pushAuditAnnotation(`理论线=" not in html
    assert "· 实际成交=" not in html
    assert "pushAuditAnnotation(`L=${fmtRecorded(reboundLowPrice,3)}`" not in html
    assert "pushAuditAnnotation(`L=${fmtRecorded(reboundLowPrice,3)} ·" not in html
    assert "function xRangeBand(x0, x1, color)" in html
    assert 'line:{color:"rgba(0,0,0,0)",width:0}' in html
    assert "function xRangeHighlight(x0, x1, color)" in html
    assert "xRangeOutline" not in html
    assert "colors.fill,colors.line,1" not in html


def test_delivery_metadata_declares_supported_baseline_sampling_policies() -> None:
    assert set(BASELINE_SAMPLING_POLICY_CONTRACTS) == {
        "all_window",
        "exclude_marked",
        "confirmed_low_activity_gate",
    }


def test_stage_browser_qa_normalizes_transparent_rgba_whitespace() -> None:
    qa_source = (
        Path(__file__).with_name("qa_v4_4_scenario_3_stage.mjs").read_text(
            encoding="utf-8"
        )
    )
    assert (
        'String(shape.fillcolor || "").replace(/\\s+/g, "").toLowerCase() '
        '!== "rgba(0,0,0,0)"'
    ) in qa_source
    assert 'shape.fillcolor !== "rgba(0,0,0,0)"' not in qa_source
    assert "audit/chart-coloring only" in BASELINE_SAMPLING_POLICY_CONTRACTS[
        "all_window"
    ]
    assert "baseline_available_from" in BASELINE_SAMPLING_POLICY_CONTRACTS[
        "exclude_marked"
    ]
    assert "blocks new entries" in BASELINE_SAMPLING_POLICY_CONTRACTS[
        "confirmed_low_activity_gate"
    ]
    assert "all_window" in FILTER_OVERLAY_SCOPE
    assert "exclude_marked" in FILTER_OVERLAY_SCOPE


def test_cumulative_main_groups_speed_windows_into_seven_responsive_ranges() -> None:
    html = _legacy_v4_main_html()
    for control_id, label in (
        ("all", "全部"),
        ("lt5", "＜5 分钟"),
        ("5_15", "5–＜15 分钟"),
        ("15_30", "15–＜30 分钟"),
        ("30_60", "30–＜60 分钟"),
        ("60_120", "60–＜120 分钟"),
        ("gte120", "≥120 分钟"),
    ):
        assert f"['{control_id}','{label}'" in html
    assert "const speedRanges=[" in html
    assert "const minutes=Number(row[field])/4" in html
    assert "按 S ÷ 4 转为分钟；可同时选择多个区间。组内取并集，四组之间取交集。" in html
    assert "按精确 S 筛选" not in html
    assert (
        "#speed-controls{display:grid;grid-template-columns:"
        "repeat(7,minmax(0,1fr));gap:6px}"
    ) in html
    assert "@media(max-width:720px){" in html
    assert (
        "#entry-baseline-controls,#entry-market-controls,"
        "#exit-baseline-controls,#speed-controls{grid-template-columns:"
        "repeat(2,minmax(0,1fr))}"
    ) in html


def test_cumulative_main_filter_panel_can_collapse_to_one_row_and_expand() -> None:
    html = _legacy_v4_main_html()
    assert 'id="control-toggle"' in html
    assert 'aria-expanded="true"' in html
    assert 'aria-controls="control-grid status"' in html
    assert 'id="control-grid" class="control-grid"' in html
    assert ".control-panel.is-collapsed .control-grid" in html
    assert '<svg class="control-toggle-icon" viewBox="0 0 24 24"' in html
    assert '<path d="m6 9 6 6 6-6"></path>' in html
    assert 'id="control-toggle-label"' not in html
    assert "width:40px;height:36px" in html
    assert "contain:layout paint" in html
    assert "setControlsCollapsed(false)" in html
    assert "controlToggle.onclick=()=>setControlsCollapsed" in html
    assert "collapsed?'Expand filters and sorting':'Collapse filters and sorting'" in html


def test_cumulative_main_uses_precise_rolling_tr_sum_mean_label() -> None:
    html = _legacy_v4_main_html()
    assert '<option value="rolling_tr_sum">滚动 TR 总和均值</option>' in html
    assert "const methods={rolling_tr_sum:'滚动 TR 总和均值'}" in html
    assert "仅使用滚动 TR 总和均值。" in html
    assert '<option value="rolling_tr_sum">滚动 TR 总和</option>' not in html


def test_stage_browser_qa_uses_manifest_dynamic_cost_contract() -> None:
    qa_source = Path(__file__).with_name("qa_v4_4_scenario_3_stage.mjs").read_text(
        encoding="utf-8"
    )
    assert (
        "Number(mainData.costModel?.round_trip_total_cost_bps) !== "
        "Number(manifest.cost_model?.round_trip_total_cost_bps)"
    ) in qa_source
    assert "round_trip_total_cost_bps) !== 3.56" not in qa_source


def test_hub_design_source_is_owned_by_the_current_checkout() -> None:
    project_root = Path(__file__).resolve().parents[3]
    assert HUB_DESIGN_SOURCE == project_root / "project_management" / "research_hub.html"


def test_legacy_standalone_delivery_is_quarantined_under_external_inputs() -> None:
    project_root = Path(__file__).resolve().parents[3]
    assert LEGACY_STANDALONE_ROOT == project_root / "external_inputs" / "legacy_review_delivery"
    assert LEGACY_STANDALONE_STAGE == LEGACY_STANDALONE_ROOT / "stage"
    assert LEGACY_STANDALONE_VALIDATION_STAGE == LEGACY_STANDALONE_ROOT / "validation_stage"


@pytest.mark.skipif(
    not (
        DEFAULT_SOURCE_MANIFEST.is_file()
        and LEGACY_STANDALONE_STAGE.is_dir()
        and LEGACY_STANDALONE_VALIDATION_STAGE.is_dir()
    ),
    reason="closed local V4.4 artifacts are unavailable",
)
def test_closed_delivery_builds_24_lazy_combo_chunks_and_2385_source_verified_trades(
    tmp_path: Path,
) -> None:
    output = tmp_path / "v4_4_review"
    result = build(
        DEFAULT_SOURCE_MANIFEST,
        LEGACY_STANDALONE_STAGE,
        LEGACY_STANDALONE_VALIDATION_STAGE,
        output,
    )
    assert result["coordinate_count"] == 24
    assert result["trade_count"] == 2385
    assert result["trade_chunk_count"] == 24
    assert result["verified_real_entry_bar_count"] == 2385
    manifest = json.loads((output / "v4_4_review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["version_label"] == "V4.4"
    assert manifest["parameter_acceptance"] == "none"
    assert manifest["design_template_sources"]["v4_research_hub"]["path"] == str(
        HUB_DESIGN_SOURCE.resolve()
    )
    assert manifest["design_template_sources"]["v4_trade_explain"]["path"] == str(
        TRADE_DESIGN_SOURCE.resolve()
    )
    assert manifest["closure"]["identity_checks"]["v4_hub_design_template_reused"] is True
    assert manifest["closure"]["identity_checks"]["v4_trade_design_template_reused"] is True
    assert manifest["closure"]["lazy_loading"]["file_protocol_safe"] is True
    assert all(manifest["closure"]["identity_checks"].values())
    assert len(list((output / "assets" / "trade_chunks").glob("*.js"))) == 24
    assert "V4.41 研究审阅工作台" in (output / "index.html").read_text(encoding="utf-8")
    trade_html = (output / "trade_analysis" / "index.html").read_text(encoding="utf-8")
    assert "V4.41 组合平仓逐笔查看" in trade_html
    assert 'type:"candlestick"' in trade_html
    assert manifest["closure"]["identity_checks"][
        "historical_trade_html_css_javascript_reused"
    ] is True


@pytest.mark.skipif(
    not (
        (CURRENT_COMBINED_STAGE / "stage_manifest.json").is_file()
        and (CURRENT_COMBINED_STAGE / "completion_manifest.json").is_file()
        and (CURRENT_COMBINED_STAGE / "analysis" / "analysis_summary.csv").is_file()
        and (CURRENT_COMBINED_STAGE / "analysis" / "stage_trades.csv").is_file()
    ),
    reason="closed combined V4.4 stage is unavailable",
)
def test_combined_stage_delivery_reuses_exact_historical_template_and_native_resources(
    tmp_path: Path,
) -> None:
    stage_manifest = json.loads(
        (CURRENT_COMBINED_STAGE / "stage_manifest.json").read_text(encoding="utf-8")
    )
    completion = json.loads(
        (CURRENT_COMBINED_STAGE / "completion_manifest.json").read_text(encoding="utf-8")
    )
    summary = pd.read_csv(
        CURRENT_COMBINED_STAGE / "analysis" / "analysis_summary.csv"
    )
    trades = pd.read_csv(CURRENT_COMBINED_STAGE / "analysis" / "stage_trades.csv")
    result = build_stage_trade_review(
        tmp_path / "trade_review",
        summary,
        trades,
        stage_manifest,
        completion,
        analysis_identity="test_combined_stage_review",
    )
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["closure"]["coordinate_count"] == 144
    assert manifest["closure"]["trade_count"] == 27662
    assert manifest["closure"]["trade_chunk_count"] == 144
    assert manifest["closure"]["verified_real_entry_bar_count"] == 27662
    assert manifest["closure"]["historical_v4_html_css_javascript_reused"] is True
    assert manifest["closure"]["historical_v4_plotly_candlestick_reused"] is True
    assert manifest["closure"]["adapter_shell_removed"] is True
    assert manifest["template_source"]["sha256"] == TRADE_DESIGN_SOURCE_SHA256
    assert manifest["plotly_output"]["sha256"] == TRADE_PLOTLY_SOURCE_SHA256
    html = Path(result["index"]).read_text(encoding="utf-8")
    assert "V4.41 组合平仓逐笔查看" in html
    assert 'type:"candlestick"' in html
    assert 'id="controlsDrawer"' in html
    assert 'id="chart"' in html
    assert "top:calc(84px + 8px)" in html
    assert "top:calc(84px + 14px)" not in html
    assert 'fillcolor:"rgba(0,0,0,0)",line:{color,width:1.2}' in html
    assert 'dash:"dash"' not in html
    assert 'dash:"dot"' not in html
    assert "虚线提示" not in html
    assert "function xRangeBand(x0, x1, color)" in html
    assert 'line:{color:"rgba(0,0,0,0)",width:0}' in html
    assert "function xRangeHighlight(x0, x1, color)" in html
    assert "xRangeOutline" not in html
    assert "colors.fill,colors.line,1" not in html
    assert "pushAuditAnnotation(`理论线=" not in html
    assert "pushAuditAnnotation(`L=${fmtRecorded(reboundLowPrice,3)} ·" not in html
    assert "assets/v4_4_review.css" not in html
    assert "assets/trade_analysis.js" not in html
    assert (Path(result["index"]).parent / "process_payload.js").is_file()
    assert (Path(result["index"]).parent / "all_results_catalog.js").is_file()
    assert len(list((Path(result["index"]).parent / "v3_native_trades_js").glob("*.js"))) == 144
