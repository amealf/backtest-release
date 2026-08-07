from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "research_variants" / "short_momentum_net_drop_rebound_v4_4"
CODE = VARIANT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "tools"))

import run_v4_4_k200_temporal_migration as temporal  # noqa: E402


CAMPAIGN_ID = "v4_4_k200_current_optimal_forward_initial_v2_20260807"
PLAN_PATH = (
    VARIANT
    / "plans"
    / "k200_current_optimal_forward_initial_v2_20260807.json"
)
FREEZE_PATH = PLAN_PATH.with_name(
    "k200_current_optimal_forward_initial_v2_20260807_candidate_freeze.csv"
)
RESULT_ROOT = ROOT / "results" / "temporal_migration" / CAMPAIGN_ID
STAGE_ROOT = RESULT_ROOT / "initial_month"
TEST_START = "2026-07-08 23:52:15"
TEST_END = "2026-08-07 03:21:45"
CANDIDATE_COUNT = 100
PARAMETERS = ("e", "bh", "trw", "k", "w", "m", "speed_window_bars")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prior_test_ids() -> set[str]:
    ids: set[str] = set()
    temporal_freezes = (
        VARIANT / "plans" / "k200_temporal_migration_20260807"
    ).glob("*_candidate_freeze.csv")
    for path in temporal_freezes:
        ids.update(pd.read_csv(path, usecols=["combo_id"]).combo_id.astype(str))
    comparison_paths = [
        ROOT
        / "results"
        / "cross_instrument_comparison"
        / "runs"
        / "k200_train_test_si__combined_350_v56_20260807"
        / "migration_comparison.csv",
    ]
    for path in comparison_paths:
        if path.is_file():
            ids.update(pd.read_csv(path, usecols=["combo_id"]).combo_id.astype(str))
    return ids


def selection_queues(source: pd.DataFrame) -> list[pd.DataFrame]:
    positive = source.loc[source.train_cost_adjusted_return.gt(0)].copy()
    source = positive
    scenario_1 = source.loc[source.scenario_1_qualified.astype(bool)]
    scenario_2 = source.loc[source.scenario_2_qualified.astype(bool)]
    moderate = source.loc[source.train_trade_count.between(20, 300)]
    low_drawdown = positive.loc[positive.train_trade_count.ge(20)]
    family = source.copy()
    family["e_family"] = pd.qcut(
        family.e.rank(method="first"), 8, labels=False
    )
    family["bh_family"] = pd.qcut(
        family.bh.rank(method="first"), 8, labels=False
    )
    family = (
        family.sort_values("train_cost_adjusted_return", ascending=False)
        .groupby(["e_family", "bh_family"], as_index=False, sort=False)
        .head(8)
    )
    return [
        source.sort_values("train_cost_adjusted_return", ascending=False),
        source.sort_values("train_cost_adjusted_avg_trade", ascending=False),
        scenario_1.sort_values("train_cost_adjusted_return", ascending=False),
        scenario_2.sort_values("train_cost_adjusted_return", ascending=False),
        low_drawdown.sort_values(
            ["train_cost_adjusted_max_drawdown_abs", "train_cost_adjusted_return"],
            ascending=[True, False],
        ),
        source.sort_values("train_return_excluding_gap_spanning_trades", ascending=False),
        moderate.sort_values("train_cost_adjusted_return", ascending=False),
        family.sort_values("train_cost_adjusted_return", ascending=False),
    ]


def select_candidates() -> pd.DataFrame:
    source = temporal.load_source()
    queues = selection_queues(source)
    exposed = prior_test_ids()
    controls = temporal.ordered_unique(queues, 8)
    controls["selection_role"] = "training_headline_control"
    blocked = exposed | set(controls.combo_id.astype(str))
    fresh = temporal.ordered_unique(
        queues,
        CANDIDATE_COUNT - len(controls),
        blocked=blocked,
    )
    fresh["selection_role"] = "training_multimetric_new_exact"
    selected = pd.concat([controls, fresh], ignore_index=True)
    selected["previous_test_evaluation"] = selected.combo_id.astype(str).isin(exposed)
    selected.insert(0, "candidate_order", np.arange(1, len(selected) + 1))
    if len(selected) != CANDIDATE_COUNT or selected.combo_id.duplicated().any():
        raise ValueError("candidate freeze must contain 100 unique coordinates")
    return selected


def explicit_combos(candidates: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for _, row in candidates.iterrows():
        rows.append(
            {
                "candidate_id": f"initial_{int(row.candidate_order):04d}",
                "seed": str(row.selection_role),
                "objective": "current_k200_optimal_forward_behavior",
                "design": "training_only_multimetric_round_robin",
                "search_mode": "transfer_exact",
                "method": "rolling_tr_sum",
                "baseline_sampling_policy": "all_window",
                **{
                    name: (
                        float(row[name])
                        if name in {"k", "m"}
                        else int(row[name])
                    )
                    for name in PARAMETERS
                },
            }
        )
    return rows


def freeze() -> pd.DataFrame:
    if PLAN_PATH.is_file() and FREEZE_PATH.is_file():
        return pd.read_csv(FREEZE_PATH, low_memory=False)
    candidates = select_candidates()
    plan = {
        "schema_version": 4,
        "status": "approved_for_execution",
        "campaign_id": CAMPAIGN_ID,
        "stage_id": "initial_month",
        "stage_kind": "k200_current_optimal_exact_temporal_transfer",
        "predecessor_stage_ids": [],
        "selection_provenance": (
            "Frozen from cost-positive rows in the current K200 training summary without reading target metrics. "
            "Eight training-view controls are retained; all other exact coordinates are "
            "anti-joined against earlier temporal evaluations."
        ),
        "source": str(temporal.MARKET_DATA),
        "data_preparation_manifest": str(temporal.PREPARATION),
        "scenario_definition": str(temporal.SCENARIOS),
        "instrument_profile": str(temporal.PROFILE),
        "train_start": TEST_START,
        "train_end": TEST_END,
        "entry_fill_mode": "calculated_threshold",
        "entry_execution_policy": "wait_next_real_trade",
        "entry_slippage": 0,
        "baseline_sampling_policy": "all_window",
        "exit_mode": "combined",
        "resources": {
            "workers": 4,
            "batch_size": 8,
            "minimum_free_memory_mb": 4096,
        },
        "migration_contract": {
            "source_instrument": "K200 training period",
            "target_instrument": "K200 subsequent month",
            "source_interval": ["2026-05-26 00:00:00", "2026-07-08 23:52:00"],
            "target_interval": [TEST_START, TEST_END],
            "target_role": "period_reused_parameter_level_initial_replay",
            "candidate_count": int(len(candidates)),
            "previously_evaluated_control_count": int(
                candidates.previous_test_evaluation.sum()
            ),
            "new_exact_coordinate_count": int(
                (~candidates.previous_test_evaluation).sum()
            ),
            "target_metrics_used_for_selection": False,
            "combined_score": False,
            "parameter_acceptance": "none",
        },
        "delivery_contract": {
            "intermediate_html": False,
            "initial_markdown_report": True,
        },
        "grid_blocks": [],
        "explicit_combos": explicit_combos(candidates),
        "stop_conditions": [
            "source_or_result_semantics_identity_mismatch",
            "memory_floor_failure",
            "partial_batches_are_not_interpreted",
        ],
        "frozen_at_utc": utc_now(),
    }
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporal.json_write(PLAN_PATH, plan)
    candidates.to_csv(FREEZE_PATH, index=False)
    return candidates


def run() -> None:
    freeze()
    if not (STAGE_ROOT / "completion_manifest.json").is_file():
        temporal.run_stage(PLAN_PATH, STAGE_ROOT)


def pct(value: object) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value) * 100:.3f}%"


def analyze() -> pd.DataFrame:
    if not (STAGE_ROOT / "completion_manifest.json").is_file():
        raise FileNotFoundError("completed initial-month stage is required")
    candidates = freeze()
    target = temporal.analyze_stage(STAGE_ROOT, "initial_month")
    target = target.rename(
        columns={
            "train_trade_count": "test_trade_count",
            "train_cost_adjusted_return": "test_cost_adjusted_return",
            "train_cost_adjusted_avg_trade": "test_cost_adjusted_avg_trade",
            "train_cost_adjusted_max_drawdown_abs": "test_cost_max_drawdown_abs",
            "median_cost_adjusted_trade": "test_median_cost_adjusted_trade",
            "win_rate": "test_win_rate",
            "positive_return_top2_share": "test_positive_return_top2_share",
            "cost_adjusted_return_excluding_gap": "test_non_gap_return",
            "gap_spanning_trade_count": "test_gap_trade_count",
        }
    )
    target_columns = [
        "combo_id",
        "test_trade_count",
        "test_cost_adjusted_return",
        "test_cost_adjusted_avg_trade",
        "test_cost_max_drawdown_abs",
        "test_median_cost_adjusted_trade",
        "test_win_rate",
        "test_positive_return_top2_share",
        "test_non_gap_return",
        "test_gap_trade_count",
    ]
    comparison = candidates.merge(
        target[target_columns], on="combo_id", how="left", validate="one_to_one"
    )
    comparison["minimum_train_test_return"] = comparison[
        ["train_cost_adjusted_return", "test_cost_adjusted_return"]
    ].min(axis=1)
    comparison["test_positive"] = comparison.test_cost_adjusted_return.gt(0)
    comparison["test_non_gap_positive"] = comparison.test_non_gap_return.gt(0)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(RESULT_ROOT / "comparison.csv", index=False)

    fresh = comparison.loc[~comparison.previous_test_evaluation.astype(bool)]
    controls = comparison.loc[comparison.previous_test_evaluation.astype(bool)]
    corr = comparison.train_cost_adjusted_return.rank().corr(
        comparison.test_cost_adjusted_return.rank(), method="pearson"
    )
    robust = comparison.loc[
        comparison.test_trade_count.ge(10)
        & comparison.test_cost_adjusted_return.gt(0)
    ].sort_values(
        ["minimum_train_test_return", "test_cost_max_drawdown_abs"],
        ascending=[False, True],
    )
    top_lines = []
    for _, row in robust.head(10).iterrows():
        top_lines.append(
            f"- E={int(row.e)}，BH={int(row.bh)}，TRW={int(row.trw)}，"
            f"K={row.k:g}，W={int(row.w)}，M={row.m:g}，"
            f"S={int(row.speed_window_bars)}：训练 {pct(row.train_cost_adjusted_return)}，"
            f"后续 {pct(row.test_cost_adjusted_return)}，非 gap {pct(row.test_non_gap_return)}，"
            f"{int(row.test_trade_count)} 笔，回撤 {pct(row.test_cost_max_drawdown_abs)}。"
        )
    top_total = comparison.sort_values(
        "test_cost_adjusted_return", ascending=False
    ).iloc[0]
    training_leader = comparison.sort_values(
        "train_cost_adjusted_return", ascending=False
    ).iloc[0]
    non_gap_candidates = comparison.loc[
        comparison.test_cost_adjusted_return.gt(0)
        & comparison.test_non_gap_return.gt(0)
        & comparison.test_trade_count.ge(10)
    ].sort_values("test_non_gap_return", ascending=False)
    non_gap_line = (
        f"E={int(non_gap_candidates.iloc[0].e)}、BH={int(non_gap_candidates.iloc[0].bh)}、"
        f"TRW={int(non_gap_candidates.iloc[0].trw)}、K={non_gap_candidates.iloc[0].k:g}、"
        f"W={int(non_gap_candidates.iloc[0].w)}、M={non_gap_candidates.iloc[0].m:g}、"
        f"S={int(non_gap_candidates.iloc[0].speed_window_bars)}，后续 "
        f"{pct(non_gap_candidates.iloc[0].test_cost_adjusted_return)}，非 gap "
        f"{pct(non_gap_candidates.iloc[0].test_non_gap_return)}，"
        f"{int(non_gap_candidates.iloc[0].test_trade_count)} 笔，Top2 正收益占比 "
        f"{pct(non_gap_candidates.iloc[0].test_positive_return_top2_share)}"
        if len(non_gap_candidates)
        else "当前没有总收益、非 gap 收益同时为正且至少 10 笔的坐标"
    )
    report = f"""# V4.4 当前 K200 优参数后续一个月初步迁移

## 范围与证据边界

- 训练期：`2026-05-26 00:00:00` 至 `2026-07-08 23:52:00`。
- 后续区间：`{TEST_START}` 至 `{TEST_END}`，约一个月。
- 候选：100 组，全部按训练期多指标冻结；其中 {len(controls)} 组是已评价过的训练视图对照，{len(fresh)} 组是此前没有在后续区间运行过的精确坐标。
- 这个时间区间已经参与早期时间迁移研究，所以本轮属于参数层面的新增初步重放，不能重新称为完全未见测试。

## 初步结果

- 成本后正收益：{int(comparison.test_positive.sum())}/100；新增精确坐标为 {int(fresh.test_positive.sum())}/{len(fresh)}。
- 至少 10 笔且成本后为正：{int((comparison.test_positive & comparison.test_trade_count.ge(10)).sum())}/100。
- 非 gap 收益为正：{int(comparison.test_non_gap_positive.sum())}/100。
- 后续收益中位数：{pct(comparison.test_cost_adjusted_return.median())}；交易数中位数：{comparison.test_trade_count.median():.1f}。
- 训练／后续收益排名 Spearman：{corr:.3f}。

## 训练与后续同时较强的代表

{chr(10).join(top_lines) if top_lines else '- 当前没有满足条件的代表。'}

## 初步判断

整体迁移表现偏弱。100 组训练期正收益参数只有 {int(comparison.test_positive.sum())} 组在后续仍为正，收益中位数为 {pct(comparison.test_cost_adjusted_return.median())}，训练／后续排名相关性为 {corr:.3f}。训练期名次没有稳定延续到后续一个月。

后续总收益最高的是 E={int(top_total.e)}／BH={int(top_total.bh)}／TRW={int(top_total.trw)}／K={top_total.k:g}／W={int(top_total.w)}／M={top_total.m:g}／S={int(top_total.speed_window_bars)}，收益 {pct(top_total.test_cost_adjusted_return)}，但非 gap 收益为 {pct(top_total.test_non_gap_return)}。训练期总收益冠军在后续得到 {pct(training_leader.test_cost_adjusted_return)}，非 gap 收益为 {pct(training_leader.test_non_gap_return)}。高总收益分支继续明显依赖跨时段 gap。

较少数不依赖 gap 的代表为：{non_gap_line}。这个方向交易较少且收益集中，需要更多未见数据，当前只保留为观察对象。

初步结论：当前 K200 训练期优参数整体不能稳定迁移到随后一个月。E320–432／BH200–240／TRW18–23／W6／M4.5 一带保留总收益，但 gap 依赖明显；BH612／S308 的低频分支有少量非 gap 正收益，证据仍稀疏。下一次真正新增 K200 数据更适合冻结这两个分支作对照，不应继续用本月结果调整本月参数。`parameter_acceptance=none`。
"""
    (RESULT_ROOT / "INITIAL_REPORT.md").write_text(report, encoding="utf-8")
    temporal.json_write(
        RESULT_ROOT / "summary.json",
        {
            "status": "complete",
            "candidate_count": int(len(comparison)),
            "previously_evaluated_control_count": int(len(controls)),
            "new_exact_coordinate_count": int(len(fresh)),
            "positive_count": int(comparison.test_positive.sum()),
            "new_exact_positive_count": int(fresh.test_positive.sum()),
            "non_gap_positive_count": int(comparison.test_non_gap_positive.sum()),
            "median_test_return": float(comparison.test_cost_adjusted_return.median()),
            "median_test_trade_count": float(comparison.test_trade_count.median()),
            "train_test_spearman": float(corr),
            "parameter_acceptance": "none",
        },
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("freeze", "run", "analyze", "all"), nargs="?", default="all"
    )
    args = parser.parse_args()
    freeze()
    if args.action in {"run", "all"}:
        run()
    if args.action in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
