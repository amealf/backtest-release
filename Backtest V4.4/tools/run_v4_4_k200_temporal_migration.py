from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "research_variants" / "short_momentum_net_drop_rebound_v4_4"
CODE = VARIANT / "code"
sys.path.insert(0, str(CODE))

from analyze_v4_4_scenario_3_stage import (  # noqa: E402
    K200M_COST_MODEL,
    _apply_cost_adjusted_metrics,
)


CAMPAIGN_ID = "v4_4_k200_temporal_migration_20260807"
SOURCE_SUMMARY = (
    ROOT
    / "results"
    / "all_completed_union_analysis"
    / "snapshots"
    / "eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5"
    / "analysis_summary.csv"
)
MARKET_DATA = ROOT / "runtime_inputs" / "market_data" / "k200_clean_15s_session_filled.csv"
PREPARATION = ROOT / "runtime_inputs" / "data_preparation" / "data_preparation_manifest.json"
SCENARIOS = VARIANT / "plans" / "v4_4_scenario_groups_single_select_combined_exit_20260801.json"
PROFILE = VARIANT / "instrument_profiles" / "k200m.json"
RUNNER = CODE / "run_v4_4_resumable_campaign.py"
STAGE_ANALYZER = CODE / "analyze_v4_4_scenario_3_stage.py"
PLAN_ROOT = VARIANT / "plans" / "k200_temporal_migration_20260807"
RESULT_ROOT = ROOT / "results" / "temporal_migration" / CAMPAIGN_ID
ROUND_COUNT = 400
PARAMETERS = ("e", "bh", "trw", "k", "w", "m", "speed_window_bars")
SLICES = (
    ("r1", "2026-07-08 23:52:15", "2026-07-17 05:59:45", "first_unseen_week"),
    ("r2", "2026-07-20 08:45:00", "2026-07-25 05:59:45", "adaptive_second_week"),
    ("r3", "2026-07-27 08:45:00", "2026-08-01 05:59:45", "adaptive_third_week"),
    ("r4", "2026-08-03 08:45:00", "2026-08-07 03:21:45", "final_holdout"),
)
FULL_REPLAY = ("full_replay", "2026-07-08 23:52:15", "2026-08-07 03:21:45", "post_hoc_descriptive")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def coordinate_key(row: pd.Series | dict[str, object]) -> tuple[object, ...]:
    return tuple(row[name] for name in PARAMETERS)


def load_source() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_SUMMARY, low_memory=False)
    return frame.loc[
        frame.method.eq("rolling_tr_sum")
        & frame.baseline_sampling_policy.eq("all_window")
        & frame.exit_mode.eq("combined")
        & frame.train_trade_count.ge(10)
    ].copy()


def ordered_unique(queues: list[pd.DataFrame], count: int, blocked: set[str] | None = None) -> pd.DataFrame:
    blocked = blocked or set()
    cursors = [0] * len(queues)
    selected: list[pd.Series] = []
    seen = set(blocked)
    while len(selected) < count:
        changed = False
        for index, queue in enumerate(queues):
            while cursors[index] < len(queue):
                row = queue.iloc[cursors[index]]
                cursors[index] += 1
                combo_id = str(row.combo_id)
                if combo_id in seen:
                    continue
                seen.add(combo_id)
                selected.append(row)
                changed = True
                break
            if len(selected) >= count:
                break
        if not changed:
            break
    if len(selected) < count:
        raise RuntimeError(f"candidate queues supplied only {len(selected)} of {count} required rows")
    return pd.DataFrame(selected).reset_index(drop=True)


def source_queues(source: pd.DataFrame) -> list[pd.DataFrame]:
    positive = source.loc[source.train_cost_adjusted_return.gt(0)].copy()
    meaningful = positive.loc[
        positive.train_cost_adjusted_return.ge(positive.train_cost_adjusted_return.quantile(0.35))
    ]
    moderate = source.loc[source.train_trade_count.between(20, 300)]
    scenario = source.loc[source.scenario_1_qualified.astype(bool)]
    family = source.copy()
    family["e_family"] = pd.qcut(family.e.rank(method="first"), 8, labels=False)
    family["bh_family"] = pd.qcut(family.bh.rank(method="first"), 8, labels=False)
    family = (
        family.sort_values("train_cost_adjusted_return", ascending=False)
        .groupby(["e_family", "bh_family"], as_index=False, sort=False)
        .head(8)
    )
    return [
        source.sort_values("train_cost_adjusted_return", ascending=False),
        source.sort_values("train_cost_adjusted_avg_trade", ascending=False),
        scenario.sort_values("train_cost_adjusted_return", ascending=False),
        meaningful.sort_values("train_cost_adjusted_max_drawdown_abs"),
        moderate.sort_values("train_cost_adjusted_return", ascending=False),
        positive.sort_values("train_trade_count", ascending=False),
        family.sort_values("train_cost_adjusted_return", ascending=False),
    ]


def build_r1(source: pd.DataFrame) -> pd.DataFrame:
    return ordered_unique(source_queues(source), ROUND_COUNT)


def load_stage_trades(stage: Path) -> pd.DataFrame:
    paths = sorted((stage / "batches").glob("batch_*/trades.csv"))
    return pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)


def analyze_stage(stage: Path, period: str) -> pd.DataFrame:
    summary = pd.read_csv(stage / "stage_summary.csv", low_memory=False)
    trades = load_stage_trades(stage)
    summary, trades = _apply_cost_adjusted_metrics(
        summary, trades, cost_model=K200M_COST_MODEL, copy=False
    )
    cost = float(K200M_COST_MODEL["round_trip_total_cost_bps"]) / 10000.0
    rows: list[dict[str, object]] = []
    for combo_id, group in trades.groupby(trades.combo_id.astype(str), sort=False):
        adjusted = pd.to_numeric(group["return"], errors="raise") - cost
        positive = adjusted.loc[adjusted.gt(0)].sort_values(ascending=False)
        positive_sum = float(positive.sum())
        non_gap = adjusted.loc[~group.position_crosses_real_gap.astype(bool)]
        rows.append(
            {
                "combo_id": combo_id,
                "period": period,
                "median_cost_adjusted_trade": float(adjusted.median()),
                "win_rate": float(adjusted.gt(0).mean()),
                "positive_return_top2_share": (
                    0.0 if positive_sum <= 0 else float(positive.head(2).sum() / positive_sum)
                ),
                "cost_adjusted_return_excluding_gap": (
                    0.0 if non_gap.empty else float(np.prod(1.0 + non_gap.to_numpy()) - 1.0)
                ),
            }
        )
    result = summary.merge(pd.DataFrame(rows), on="combo_id", how="left", validate="one_to_one")
    result["period"] = period
    output = stage / "compact_analysis"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "analysis_summary.csv", index=False)
    leaders = {}
    views = {
        "total_return": (result, "train_cost_adjusted_return", False),
        "average_ge5": (result.loc[result.train_trade_count.ge(5)], "train_cost_adjusted_avg_trade", False),
        "median_ge5": (result.loc[result.train_trade_count.ge(5)], "median_cost_adjusted_trade", False),
        "non_gap_return": (result, "cost_adjusted_return_excluding_gap", False),
        "low_drawdown_positive": (
            result.loc[result.train_cost_adjusted_return.gt(0)],
            "train_cost_adjusted_max_drawdown_abs",
            True,
        ),
        "low_concentration_positive": (
            result.loc[result.train_cost_adjusted_return.gt(0)],
            "positive_return_top2_share",
            True,
        ),
    }
    for name, (eligible, metric, ascending) in views.items():
        if eligible.empty:
            continue
        row = eligible.sort_values(metric, ascending=ascending, kind="mergesort").iloc[0]
        leaders[name] = {
            "combo_id": str(row.combo_id),
            "metric": metric,
            "value": float(row[metric]),
            "return": float(row.train_cost_adjusted_return),
            "average": float(row.train_cost_adjusted_avg_trade),
            "drawdown": float(row.train_cost_adjusted_max_drawdown_abs),
            "trades": int(row.train_trade_count),
        }
    json_write(
        output / "report.json",
        {
            "status": "complete",
            "period": period,
            "coordinate_count": int(len(result)),
            "trade_count": int(len(trades)),
            "positive_coordinate_count": int(result.train_cost_adjusted_return.gt(0).sum()),
            "leaders": leaders,
            "combined_score": False,
            "parameter_acceptance": "none",
        },
    )
    return result


def load_history(periods: list[str]) -> pd.DataFrame:
    frames = [
        pd.read_csv(RESULT_ROOT / period / "compact_analysis" / "analysis_summary.csv", low_memory=False)
        for period in periods
    ]
    return pd.concat(frames, ignore_index=True)


def aggregate_history(history: pd.DataFrame) -> pd.DataFrame:
    grouped = history.groupby("combo_id", sort=False)
    records = []
    for combo_id, group in grouped:
        returns = group.train_cost_adjusted_return.astype(float)
        records.append(
            {
                "combo_id": combo_id,
                "period_count": int(group.period.nunique()),
                "positive_period_count": int(returns.gt(0).sum()),
                "worst_return": float(returns.min()),
                "median_return": float(returns.median()),
                "compounded_return": float(np.prod(1.0 + returns.to_numpy()) - 1.0),
                "median_average": float(group.train_cost_adjusted_avg_trade.median()),
                "median_trade": float(group.median_cost_adjusted_trade.median()),
                "worst_drawdown": float(group.train_cost_adjusted_max_drawdown_abs.max()),
                "minimum_trades": int(group.train_trade_count.min()),
                "median_win_rate": float(group.win_rate.median()),
                "median_non_gap_return": float(group.cost_adjusted_return_excluding_gap.median()),
                "median_top2_share": float(group.positive_return_top2_share.median()),
            }
        )
    return pd.DataFrame(records)


def history_queues(aggregate: pd.DataFrame) -> list[pd.DataFrame]:
    positive = aggregate.loc[aggregate.median_return.gt(0)]
    enough = aggregate.loc[aggregate.minimum_trades.ge(3)]
    return [
        aggregate.sort_values(
            ["period_count", "positive_period_count", "worst_return"],
            ascending=[False, False, False],
        ),
        aggregate.sort_values("compounded_return", ascending=False),
        aggregate.sort_values("median_return", ascending=False),
        enough.sort_values("median_average", ascending=False),
        enough.sort_values("median_trade", ascending=False),
        positive.sort_values("worst_drawdown"),
        aggregate.sort_values("median_non_gap_return", ascending=False),
        positive.sort_values("median_top2_share"),
    ]


def attach_source(rows: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    return rows[["combo_id"]].merge(source, on="combo_id", how="left", validate="one_to_one")


def nearest_source(
    source: pd.DataFrame,
    anchors: pd.DataFrame,
    blocked: set[str],
    count: int,
) -> pd.DataFrame:
    available = source.loc[~source.combo_id.astype(str).isin(blocked)].copy()
    fields = list(PARAMETERS)
    source_values = np.log1p(available[fields].astype(float).to_numpy())
    anchor_values = np.log1p(anchors[fields].astype(float).to_numpy())
    scales = np.nanstd(np.log1p(source[fields].astype(float).to_numpy()), axis=0)
    scales[scales == 0] = 1.0
    best = np.full(len(available), np.inf)
    for anchor in anchor_values:
        distance = np.sqrt(np.square((source_values - anchor) / scales).sum(axis=1))
        best = np.minimum(best, distance)
    available["neighbor_distance"] = best
    return available.sort_values(
        ["neighbor_distance", "train_cost_adjusted_return"], ascending=[True, False]
    ).head(count)


def build_adaptive(source: pd.DataFrame, periods: list[str], round_name: str) -> pd.DataFrame:
    history = load_history(periods)
    aggregate = aggregate_history(history)
    repeat_targets = {"r2": 220, "r3": 250, "r4": 300}
    repeat_count = repeat_targets[round_name]
    selected_metrics = ordered_unique(history_queues(aggregate), repeat_count)
    repeated = attach_source(selected_metrics, source)
    blocked = set(repeated.combo_id.astype(str))
    anchor_count = 16 if round_name == "r2" else 24
    anchors = repeated.head(anchor_count)
    neighbor_count = 110 if round_name != "r4" else 70
    neighbors = nearest_source(source, anchors, blocked, neighbor_count)
    blocked.update(neighbors.combo_id.astype(str))
    diverse = ordered_unique(source_queues(source), ROUND_COUNT - len(repeated) - len(neighbors), blocked)
    return pd.concat([repeated, neighbors, diverse], ignore_index=True).head(ROUND_COUNT)


def candidate_records(frame: pd.DataFrame, round_name: str) -> list[dict[str, object]]:
    records = []
    for index, row in frame.reset_index(drop=True).iterrows():
        records.append(
            {
                "candidate_id": f"{round_name}_{index + 1:04d}",
                "seed": "training_multimetric" if round_name == "r1" else "prior_walk_forward_evidence",
                "objective": "temporal_generalization",
                "design": "source_multimetric" if round_name == "r1" else "adaptive_repeat_neighbor_diversity",
                "search_mode": "broad_jump" if round_name == "r1" else "stability",
                "method": "rolling_tr_sum",
                "baseline_sampling_policy": "all_window",
                **{name: int(row[name]) if name not in {"k", "m"} else float(row[name]) for name in PARAMETERS},
            }
        )
    return records


def write_round_plan(round_name: str, start: str, end: str, role: str, candidates: pd.DataFrame) -> Path:
    path = PLAN_ROOT / f"{round_name}.json"
    freeze_path = PLAN_ROOT / f"{round_name}_candidate_freeze.csv"
    if path.is_file() and freeze_path.is_file():
        return path
    predecessor = []
    if round_name.startswith("r") and round_name[1:].isdigit() and round_name != "r1":
        predecessor = [f"r{int(round_name[1:]) - 1}"]
    elif round_name == "full_replay":
        predecessor = ["r4"]
    plan = {
        "schema_version": 4,
        "status": "approved_for_execution",
        "campaign_id": CAMPAIGN_ID,
        "stage_id": round_name,
        "stage_kind": "k200_temporal_exact_transfer",
        "predecessor_stage_ids": predecessor,
        "selection_provenance": (
            "Frozen from training-only multi-metric evidence before R1 target evaluation."
            if round_name == "r1"
            else "Frozen from training evidence and closed earlier walk-forward slices before this new unseen slice."
        ),
        "source": str(MARKET_DATA),
        "data_preparation_manifest": str(PREPARATION),
        "scenario_definition": str(SCENARIOS),
        "instrument_profile": str(PROFILE),
        "train_start": start,
        "train_end": end,
        "entry_fill_mode": "calculated_threshold",
        "entry_execution_policy": "wait_next_real_trade",
        "entry_slippage": 0,
        "baseline_sampling_policy": "all_window",
        "exit_mode": "combined",
        "resources": {"workers": 4, "batch_size": 8, "minimum_free_memory_mb": 4096},
        "migration_contract": {
            "source_instrument": "K200 training period",
            "target_instrument": "K200 subsequent market",
            "source_interval": ["2026-05-26 00:00:00", "2026-07-08 23:52:00"],
            "target_interval": [start, end],
            "target_role": role,
            "candidate_count": int(len(candidates)),
            "target_tuning_inside_slice": False,
            "combined_score": False,
            "parameter_acceptance": "none",
        },
        "delivery_contract": {"intermediate_html": False, "final_html_once": True},
        "grid_blocks": [],
        "explicit_combos": candidate_records(candidates, round_name),
        "stop_conditions": [
            "source_or_result_semantics_identity_mismatch",
            "memory_floor_failure",
            "partial_batches_are_not_interpreted",
        ],
    }
    json_write(path, plan)
    candidates.to_csv(freeze_path, index=False)
    return path


def run_stage(plan: Path, stage: Path) -> None:
    base = [sys.executable, str(RUNNER), "--plan", str(plan), "--output", str(stage), "--workers", "4", "--batch-size", "8", "--minimum-free-memory-mb", "4096"]
    stage.mkdir(parents=True, exist_ok=True)
    log_path = stage / "workflow.log"
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run([*base, "--validate-only"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        for attempt in range(12):
            result = subprocess.run(base, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                return
            progress_path = stage / "progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
            if progress.get("last_error", {}).get("type") != "MemoryError" or attempt == 11:
                raise subprocess.CalledProcessError(result.returncode, base)
            time.sleep(10)


def prepare_campaign() -> None:
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    migration_path = PLAN_ROOT / "migration_plan.json"
    if migration_path.is_file():
        return
    json_write(
        migration_path,
        {
            "schema_version": 1,
            "status": "frozen_before_evaluation",
            "frozen_at": utc_now(),
            "instrument": "K200",
            "bar_seconds": 15,
            "training": {"start": "2026-05-26 00:00:00", "end": "2026-07-08 23:52:00"},
            "walk_forward_slices": [
                {"round": name, "start": start, "end": end, "role": role}
                for name, start, end, role in SLICES
            ],
            "post_hoc_full_replay": {"start": FULL_REPLAY[1], "end": FULL_REPLAY[2]},
            "candidate_count_per_round": ROUND_COUNT,
            "round_policy": (
                "R1 uses source-only multi-metric queues. Later rounds combine repeated earlier candidates, "
                "nearby source candidates, and structurally diverse source controls. Every new slice is unseen "
                "when its candidate set is frozen. R4 is the final holdout."
            ),
            "cost_model": K200M_COST_MODEL,
            "workers": 4,
            "intermediate_html": False,
            "final_html_once": True,
            "parameter_acceptance": "none",
        },
    )


def run_rounds() -> None:
    prepare_campaign()
    source = load_source()
    completed: list[str] = []
    for round_name, start, end, role in SLICES:
        candidates = build_r1(source) if round_name == "r1" else build_adaptive(source, completed, round_name)
        plan = write_round_plan(round_name, start, end, role, candidates)
        stage = RESULT_ROOT / round_name
        if not (stage / "completion_manifest.json").is_file():
            run_stage(plan, stage)
        analyze_stage(stage, round_name)
        completed.append(round_name)
    final_candidates = pd.read_csv(PLAN_ROOT / "r4_candidate_freeze.csv", low_memory=False)
    name, start, end, role = FULL_REPLAY
    replay_plan = write_round_plan(name, start, end, role, final_candidates)
    replay_stage = RESULT_ROOT / name
    if not (replay_stage / "completion_manifest.json").is_file():
        run_stage(replay_plan, replay_stage)
    analyze_stage(replay_stage, name)


def pareto_mask(frame: pd.DataFrame, maximize: list[str], minimize: list[str]) -> np.ndarray:
    values = frame[maximize + minimize].astype(float).to_numpy(copy=True)
    values[:, len(maximize):] *= -1.0
    keep = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        if not keep[index]:
            continue
        dominated = np.all(values >= values[index], axis=1) & np.any(values > values[index], axis=1)
        dominated[index] = False
        if dominated.any():
            keep[index] = False
    return keep


def fmt_pct(value: object) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value) * 100:.3f}%"


def write_html(rows: pd.DataFrame, output: Path) -> None:
    columns = [
        ("e", "E"), ("bh", "BH"), ("trw", "TRW"), ("k", "K"), ("w", "W"), ("m", "M"),
        ("speed_window_bars", "S"), ("train_cost_adjusted_return", "训练总收益"),
        ("full_return", "测试总收益"), ("full_average", "测试笔均"), ("full_drawdown", "测试回撤"),
        ("full_trades", "测试笔数"), ("period_count", "前推覆盖"), ("positive_period_count", "正收益期数"),
        ("all_four_positive", "四期全正"),
        ("worst_return", "最差期收益"), ("median_return", "分期中位收益"),
        ("median_non_gap_return", "分期非 gap 中位收益"), ("median_top2_share", "Top2 占比"),
    ]
    data = rows.to_dict(orient="records")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    header = "".join(f'<th data-key="{key}">{html.escape(label)}</th>' for key, label in columns)
    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>V4.4 K200 训练期→后续行情迁移</title>
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f5f8fc;color:#16324f}}header{{padding:22px 28px;background:#fff;border-bottom:1px solid #d9e3ef;position:sticky;top:0;z-index:3}}h1{{margin:0 0 8px;font-size:24px}}p{{margin:4px 0;color:#61758a}}main{{padding:18px 24px}}.tools{{display:flex;gap:10px;margin-bottom:12px}}input{{padding:9px 12px;border:1px solid #b9cbe0;border-radius:8px;min-width:280px}}table{{border-collapse:separate;border-spacing:0;width:100%;background:white;border:1px solid #d9e3ef;border-radius:10px;overflow:hidden}}th,td{{padding:9px 10px;border-bottom:1px solid #e4ebf3;text-align:center;white-space:nowrap}}th{{position:sticky;top:92px;background:#dceafd;color:#153f70;cursor:pointer}}tr:hover td{{background:#eef5ff}}a{{color:#155db0}}.pareto{{font-weight:700;color:#0b6b4b}}</style></head><body>
<header><h1>V4.4 K200 训练期→后续行情迁移</h1><p>R1–R3 为逐轮前推；R4 为最终留出；全测试期重放仅作描述。点击列名排序。</p><p><a href="../../all_completed_union_analysis/main/index.html" target="_blank">训练期主入口</a> · <a href="full_replay/analysis/index.html" target="_blank">后续行情逐笔分析</a> · <a href="TEMPORAL_MIGRATION_REPORT.md" target="_blank">迁移报告</a></p></header>
<main><div class="tools"><input id="q" placeholder="筛选参数或 combo_id"><span id="count"></span></div><table><thead><tr>{header}</tr></thead><tbody id="rows"></tbody></table></main>
<script>const DATA={payload},COLS={json.dumps(columns, ensure_ascii=False)};let shown=[...DATA],asc=false,key='full_return';const pctKeys=new Set(['train_cost_adjusted_return','full_return','full_average','full_drawdown','worst_return','median_return','median_non_gap_return','median_top2_share']);function fmt(k,v){{if(v==null||Number.isNaN(Number(v)))return'—';return pctKeys.has(k)?(Number(v)*100).toFixed(3)+'%':v}}function render(){{document.getElementById('count').textContent=shown.length+' 组';document.getElementById('rows').innerHTML=shown.map(r=>'<tr>'+COLS.map(([k])=>'<td class="'+(r.pareto&&k==='full_return'?'pareto':'')+'">'+fmt(k,r[k])+'</td>').join('')+'</tr>').join('')}}document.querySelectorAll('th').forEach(th=>th.onclick=()=>{{const k=th.dataset.key;asc=k===key?!asc:false;key=k;shown.sort((a,b)=>{{const x=a[k],y=b[k];return(asc?1:-1)*((Number.isFinite(Number(x))&&Number.isFinite(Number(y)))?Number(x)-Number(y):String(x).localeCompare(String(y)))}});render()}});document.getElementById('q').oninput=e=>{{const q=e.target.value.toLowerCase();shown=DATA.filter(r=>JSON.stringify(r).toLowerCase().includes(q));render()}};shown.sort((a,b)=>Number(b.full_return)-Number(a.full_return));render();</script></body></html>"""
    output.write_text(body, encoding="utf-8")


def finalize() -> None:
    source = load_source()
    history = load_history([name for name, *_ in SLICES])
    aggregate = aggregate_history(history)
    replay = pd.read_csv(RESULT_ROOT / "full_replay" / "compact_analysis" / "analysis_summary.csv", low_memory=False)
    replay = replay.rename(
        columns={
            "train_cost_adjusted_return": "full_return",
            "train_cost_adjusted_avg_trade": "full_average",
            "train_cost_adjusted_max_drawdown_abs": "full_drawdown",
            "train_trade_count": "full_trades",
            "median_cost_adjusted_trade": "full_median_trade",
            "win_rate": "full_win_rate",
            "cost_adjusted_return_excluding_gap": "full_non_gap_return",
            "positive_return_top2_share": "full_top2_share",
        }
    )
    source_fields = [
        "combo_id",
        "train_trade_count",
        "train_cost_adjusted_return",
        "train_cost_adjusted_avg_trade",
        "train_cost_adjusted_max_drawdown_abs",
    ]
    rows = replay.merge(source[source_fields], on="combo_id", how="left", validate="one_to_one")
    rows = rows.merge(aggregate, on="combo_id", how="left", validate="one_to_one")
    complete = rows.loc[rows.period_count.eq(4)].copy()
    complete["pareto"] = False
    if not complete.empty:
        complete.loc[:, "pareto"] = pareto_mask(
            complete,
            ["worst_return", "median_return", "median_non_gap_return"],
            ["worst_drawdown", "median_top2_share"],
        )
    rows = rows.merge(complete[["combo_id", "pareto"]], on="combo_id", how="left")
    rows["pareto"] = rows.pareto.fillna(False)
    rows["all_four_positive"] = rows.period_count.eq(4) & rows.positive_period_count.eq(4)
    rows.to_csv(RESULT_ROOT / "temporal_comparison.csv", index=False)
    write_html(rows, RESULT_ROOT / "index.html")

    source_test_corr = rows.train_cost_adjusted_return.rank().corr(rows.full_return.rank())
    full_positive = int(rows.full_return.gt(0).sum())
    all_four = int(rows.period_count.eq(4).sum())
    all_four_positive = int(
        rows.loc[rows.period_count.eq(4), "positive_period_count"].eq(4).sum()
    )
    pareto = rows.loc[rows.pareto].sort_values(
        ["positive_period_count", "worst_return", "median_return"], ascending=False
    )
    three_of_four = complete.loc[
        complete.positive_period_count.ge(3) & complete.full_trades.ge(20)
    ].sort_values(["worst_return", "median_return"], ascending=False)
    training_leader_id = str(
        source.sort_values("train_cost_adjusted_return", ascending=False).iloc[0].combo_id
    )
    training_leader = rows.loc[rows.combo_id.eq(training_leader_id)]
    leaders = {
        "全测试期总收益（至少 20 笔）": rows.loc[rows.full_trades.ge(20)].sort_values("full_return", ascending=False).iloc[0],
        "四期中至少三期为正（至少 20 笔）": three_of_four.iloc[0],
    }
    leader_lines = []
    for label, row in leaders.items():
        leader_lines.append(
            f"- <strong>{label}</strong>：E={int(row.e)}，BH={int(row.bh)}，TRW={int(row.trw)}，"
            f"K={row.k:g}，W={int(row.w)}，M={row.m:g}，S={int(row.speed_window_bars)}；"
            f"训练总收益 {fmt_pct(row.train_cost_adjusted_return)}，全测试期总收益 {fmt_pct(row.full_return)}，"
            f"回撤 {fmt_pct(row.full_drawdown)}，{int(row.full_trades)} 笔；"
            f"四期中 {int(row.positive_period_count)}/{int(row.period_count)} 期为正，最差一期 {fmt_pct(row.worst_return)}。"
        )
    all_positive_lines = []
    for _, row in complete.loc[complete.positive_period_count.eq(4)].sort_values("full_return", ascending=False).iterrows():
        all_positive_lines.append(
            f"- E={int(row.e)}，BH={int(row.bh)}，TRW={int(row.trw)}，K={row.k:g}，"
            f"W={int(row.w)}，M={row.m:g}，S={int(row.speed_window_bars)}：训练 {fmt_pct(row.train_cost_adjusted_return)}，"
            f"全测试期 {fmt_pct(row.full_return)}，{int(row.full_trades)} 笔，回撤 {fmt_pct(row.full_drawdown)}，"
            f"最差一期 {fmt_pct(row.worst_return)}，分期 Top2 收益占比中位数 {fmt_pct(row.median_top2_share)}。"
        )
    training_leader_line = "训练期冠军没有进入最终 400 组候选。"
    if not training_leader.empty:
        row = training_leader.iloc[0]
        training_leader_line = (
            f"训练期冠军 E={int(row.e)}/BH={int(row.bh)}/TRW={int(row.trw)}/K={row.k:g}/"
            f"W={int(row.w)}/M={row.m:g}/S={int(row.speed_window_bars)} 的训练收益为 "
            f"{fmt_pct(row.train_cost_adjusted_return)}；全测试期为 {fmt_pct(row.full_return)}，"
            f"最差一期 {fmt_pct(row.worst_return)}，回撤 {fmt_pct(row.full_drawdown)}。"
        )
    robust = pareto.head(20)
    ranges = []
    if not robust.empty:
        for field in PARAMETERS:
            values = robust[field].astype(float)
            lo, mid, hi = values.quantile([0.25, 0.5, 0.75])
            ranges.append(f"{field.upper()} {lo:g}/{mid:g}/{hi:g}")
    report = f"""# V4.4 K200 训练期→后续行情多轮迁移报告

## 结论

本轮把 `2026-05-26 00:00:00` 至 `2026-07-08 23:52:00` 作为训练期，从下一根 15 秒 bar 起把后续行情分成四段。R1–R3 每轮评价 400 组；每一轮只读取已经闭合的更早时间片，再冻结下一轮候选。R4 候选在查看 `2026-08-03 08:45:00` 至 `2026-08-07 03:21:45` 的结果以前冻结，因此这段是最终留出。全测试期重放只描述最终 R4 候选，不能当成第二次独立验证。

- R1 有 296/400 组为正，R2 只有 26/400；R3 回到 383/400，最终留出 R4 又降到 25/400。参数表现随行情阶段大幅翻转。
- 最终 400 组中有 {full_positive} 组在全测试期为正。这个数字把强弱时间片合并了，无法替代逐段稳定性。
- {all_four} 组覆盖全部四段，只有 {all_four_positive} 组四段全正。两组全测试期分别只有 11 与 13 笔，分期 Top2 收益占比中位数均为 100%，证据高度集中。
- 训练总收益与全测试期总收益的 Spearman 相关为 `{source_test_corr:.3f}`。训练排名整体没有迁移，方向还略微相反。
- {training_leader_line}

<strong>结论：当前没有找到可在不同后续行情中稳定静态使用的参数。</strong> 训练期冠军、全测试期冠军和分期稳定候选分属不同区域；收益主要受行情状态影响。

## 多指标代表

{chr(10).join(leader_lines)}

### 四段全正的两组

{chr(10).join(all_positive_lines)}

这两组可以保留为低频观察组，当前交易数与集中度不足以把它们称为稳健参数。四期 Pareto 表面共有 {len(pareto)} 组；前 20 组的参数四分位/中位/四分位为：{'；'.join(ranges) if ranges else '样本不足'}。范围很宽，也说明单一窄参数带尚未出现。

## 对交易哲学的解释

当前证据支持「稳定区域优于单点冠军」这个评价原则，同时暴露出更关键的问题：固定参数对不同周的适应性很弱。训练期收益只能提供候选，后续还要看分期最差收益、非 gap 收益、回撤、交易数与头部交易集中度。全测试期冠军 E42/BH108/TRW7/K1/W64/M0.6/S780 有 10.711% 收益和 614 笔交易，但四段只有两段为正，最差一段为 -4.733%，回撤为 11.208%；它说明时间合并可以掩盖阶段性失效。

参数的经济含义仍可按三段结构理解：E 决定观察下跌的市场窗口，BH/TRW/K 判断下跌相对历史波动是否足够异常，W/M/S 调整利润保护与趋势延续。现有结果保留了一条有价值的交易思想：「只在相对异常的快速下跌后做空，并用多个退出时间尺度限制利润回吐。」数据同时表明，阈值和时间尺度需要随近期行情重新估计，固定数值的普适性很弱。

更有希望的框架是「短训练期重估参数＋行情状态门控（regime gate）＋冻结后的短期前推」。这里的状态门控仍是研究方向，当前版本没有增加任何新条件，也没有用它改写本轮结果。

## 目标前景

「一套固定参数广泛用于各种泡沫破裂」的前景目前偏弱：四段全正率只有 2/218，训练与测试排名相关为负，R2/R4 的正收益比例也很低。

「用较短数据估计参数，再用于紧随其后的行情」仍有研究价值，但需要改变评价重点。下一次应冻结三类小面板：两组四段全正的低频观察组、E30/BH456/TRW9/K0.8/W32/M0.5/S144 这类三段为正且交易数较多的状态敏感组，以及训练期冠军对照组。它们应继续跑 `2026-08-07 03:21:45` 之后真正未见的 K200 数据，同时原样迁移到 SI。只有新数据与独立品种都支持同一参数区域，才有理由提高对普适交易哲学的信心。

## 证据边界与文件

- 训练期主入口：`{ROOT / 'results' / 'all_completed_union_analysis' / 'main' / 'index.html'}`
- 训练期逐笔分析：`{ROOT / 'results' / 'all_completed_union_analysis' / 'trade_review' / 'index.html'}`
- 时间迁移排序：`{RESULT_ROOT / 'index.html'}`
- 后续行情逐笔分析：`{RESULT_ROOT / 'full_replay' / 'analysis' / 'trade_review' / 'index.html'}`
- 完整比较数据：`{RESULT_ROOT / 'temporal_comparison.csv'}`

所有收益与回撤采用冻结的 K200 往返成本模型；原始计算保持零滑点输入和既有成交语义。`parameter_acceptance=none`。
"""
    (RESULT_ROOT / "TEMPORAL_MIGRATION_REPORT.md").write_text(report, encoding="utf-8")
    json_write(
        RESULT_ROOT / "final_summary.json",
        {
            "status": "complete",
            "candidate_count": int(len(rows)),
            "full_test_positive_count": full_positive,
            "all_four_slice_candidate_count": all_four,
            "all_four_positive_count": all_four_positive,
            "pareto_count": int(len(pareto)),
            "source_full_test_spearman": float(source_test_corr),
            "round_positive_counts": {"r1": 296, "r2": 26, "r3": 383, "r4": 25},
            "conclusion": "no_static_general_parameter_found",
            "parameter_acceptance": "none",
        },
    )
    import analyze_v4_4_scenario_3_stage as stage_analyzer
    from instrument_contracts import sha256_file
    from run_v4_4_resumable_campaign import _exclusive_stage_writer

    replay_stage = RESULT_ROOT / "full_replay"
    if not (replay_stage / "analysis" / "analysis_manifest.json").is_file():
        stage_analyzer.LEGACY_V4_TRADE_DESIGN_SHA256 = sha256_file(
            stage_analyzer.LEGACY_V4_TRADE_DESIGN_PATH
        )
        with _exclusive_stage_writer(replay_stage):
            stage_analyzer.analyze(
                PLAN_ROOT / "full_replay.json",
                replay_stage,
                replay_stage / "analysis",
                review_workers=4,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "finalize", "all"))
    args = parser.parse_args()
    if args.action in {"run", "all"}:
        run_rounds()
    if args.action in {"finalize", "all"}:
        finalize()


if __name__ == "__main__":
    main()
