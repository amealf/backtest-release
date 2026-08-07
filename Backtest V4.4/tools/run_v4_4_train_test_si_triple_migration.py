from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "research_variants" / "short_momentum_net_drop_rebound_v4_4"
CODE = VARIANT / "code"
sys.path.insert(0, str(CODE))

import build_v4_4_cross_instrument_comparison as cross  # noqa: E402
from v4_4_engine import Combo, load_bars, simulate_combo  # noqa: E402


PARENT_ID = "k200_20260526_20260708__simain_20260129_20260223__combined_250_stricter_entry_v54_20260806"
NEW_ID = "k200_temporal_100__simain_20260129_20260223__v55_20260807"
COMBINED_ID = "k200_train_test_si__combined_350_v56_20260807"
CROSS_ROOT = ROOT / "results" / "cross_instrument_comparison"
PARENT_ROOT = CROSS_ROOT / "runs" / PARENT_ID
NEW_ROOT = CROSS_ROOT / "runs" / NEW_ID
COMBINED_ROOT = CROSS_ROOT / "runs" / COMBINED_ID
TEMPORAL_ROOT = ROOT / "results" / "temporal_migration" / "v4_4_k200_temporal_migration_20260807"
TEMPORAL_COMPARISON = TEMPORAL_ROOT / "temporal_comparison.csv"
TRAIN_SUMMARY = (
    ROOT
    / "results"
    / "all_completed_union_analysis"
    / "snapshots"
    / "eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5"
    / "analysis_summary.csv"
)
TRAIN_TRADE_ROOT = TRAIN_SUMMARY.parent / "trade_review"
PLAN_PATH = (
    VARIANT
    / "plans"
    / "v4_4_migration_k200_train_test_to_simain_temporal_100_20260807.json"
)
K200_TEST_START = pd.Timestamp("2026-07-08 23:52:15")
K200_TEST_END = pd.Timestamp("2026-08-07 03:21:45")
PARAMETERS = ("e", "bh", "trw", "k", "w", "m", "speed_window_bars")
EXECUTION_FIELDS = (
    "method",
    "baseline_sampling_policy",
    "entry_fill_mode",
    "entry_execution_policy",
    "entry_slippage",
)
NEW_COUNT = 100


_TEST_FRAME: pd.DataFrame | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_js_assignment(path: Path, prefix: str) -> object:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(prefix):
        raise ValueError(f"unexpected JavaScript assignment: {path}")
    return json.loads(text[len(prefix) :].rstrip().removesuffix(";"))


def parse_trade_chunk(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    marker = ";window.NATIVE_TRADES="
    left, separator, right = text.partition(marker)
    if not separator or not left.startswith("window.NATIVE_COMBO="):
        raise ValueError(f"unexpected trade chunk structure: {path}")
    combo = json.loads(left[len("window.NATIVE_COMBO=") :])
    trades = json.loads(right.rstrip().removesuffix(";"))
    return combo, trades


def unique_from_queues(
    queues: list[tuple[str, pd.DataFrame]],
    count: int,
    blocked: set[str],
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    cursors = [0] * len(queues)
    selected: list[pd.Series] = []
    tags: dict[str, set[str]] = defaultdict(set)
    seen = set(blocked)
    while len(selected) < count:
        changed = False
        for queue_index, (label, queue) in enumerate(queues):
            while cursors[queue_index] < len(queue):
                row = queue.iloc[cursors[queue_index]]
                cursors[queue_index] += 1
                combo_id = str(row.combo_id)
                if combo_id in blocked:
                    continue
                tags[combo_id].add(label)
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
    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} of {count} requested candidates")
    return pd.DataFrame(selected).reset_index(drop=True), tags


def select_new_candidates() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    temporal = pd.read_csv(TEMPORAL_COMPARISON, low_memory=False)
    r4 = pd.read_csv(
        TEMPORAL_ROOT / "r4" / "compact_analysis" / "analysis_summary.csv",
        usecols=["combo_id", "train_cost_adjusted_return", "train_trade_count"],
        low_memory=False,
    ).rename(
        columns={
            "train_cost_adjusted_return": "r4_cost_return",
            "train_trade_count": "r4_trade_count",
        }
    )
    temporal = temporal.merge(r4, on="combo_id", how="left", validate="one_to_one")
    parent_ids = set(
        pd.read_csv(PARENT_ROOT / "migration_comparison.csv", usecols=["combo_id"])
        .combo_id.astype(str)
    )
    eligible = temporal.loc[~temporal.combo_id.astype(str).isin(parent_ids)].copy()
    complete = eligible.loc[eligible.period_count.eq(4)]
    queues = [
        (
            "four_slice_all_positive",
            complete.loc[complete.positive_period_count.eq(4)].sort_values(
                ["worst_return", "full_return"], ascending=False
            ),
        ),
        (
            "four_slice_three_positive",
            complete.loc[complete.positive_period_count.ge(3)].sort_values(
                ["worst_return", "median_return", "full_return"], ascending=False
            ),
        ),
        (
            "final_holdout_positive",
            eligible.loc[eligible.r4_cost_return.gt(0)].sort_values(
                ["r4_cost_return", "full_return"], ascending=False
            ),
        ),
        (
            "full_test_total_ge20",
            eligible.loc[eligible.full_trades.ge(20)].sort_values(
                "full_return", ascending=False
            ),
        ),
        (
            "full_test_non_gap",
            eligible.sort_values("full_non_gap_return", ascending=False),
        ),
        (
            "full_test_low_drawdown",
            eligible.loc[
                eligible.full_return.gt(0) & eligible.full_trades.ge(10)
            ].sort_values(["full_drawdown", "full_return"], ascending=[True, False]),
        ),
        (
            "four_slice_pareto",
            eligible.loc[eligible.pareto.astype(bool)].sort_values(
                ["positive_period_count", "worst_return", "median_return"],
                ascending=False,
            ),
        ),
        (
            "training_return_control",
            eligible.sort_values("train_cost_adjusted_return", ascending=False),
        ),
        (
            "structural_diversity",
            eligible.sort_values(
                ["e", "bh", "trw", "k", "w", "m", "speed_window_bars"]
            ).iloc[:: max(1, len(eligible) // 160)],
        ),
    ]
    return unique_from_queues(queues, NEW_COUNT, parent_ids)


def catalog_map() -> dict[str, dict[str, Any]]:
    payload = read_js_assignment(
        TRAIN_TRADE_ROOT / "all_results_catalog.js",
        "window.ALL_RESULTS_TRADE_EXPLAIN_CATALOG=",
    )
    return {str(row["combo_id"]): dict(row) for row in payload["rows"]}


def load_training_trades(combo_ids: list[str]) -> pd.DataFrame:
    catalog = catalog_map()
    trades: list[dict[str, Any]] = []
    for combo_id in combo_ids:
        row = catalog.get(combo_id)
        if row is None:
            raise KeyError(f"training trade catalog lacks {combo_id}")
        chunk = (
            TRAIN_TRADE_ROOT
            / str(row["trade_js_base"])
            / str(row["trade_js_file"])
        )
        _, records = parse_trade_chunk(chunk)
        trades.extend(records)
    return pd.DataFrame(trades)


def candidate_columns() -> list[str]:
    return [
        "candidate_order",
        "combo_id",
        *EXECUTION_FIELDS,
        *PARAMETERS,
        "scenario_1_qualified",
        "scenario_2_qualified",
        "scenario_3_qualified",
        "selection_tags",
        "source_trade_count",
        "source_gross_total_return",
        "source_cost_total_return",
        "source_cost_median_trade",
        "source_cost_mean_trade",
        "source_cost_max_drawdown_abs",
        "source_win_rate",
        "source_mfe_bps_median",
        "source_mae_bps_median",
        "source_mfe_points_median",
        "source_mae_points_median",
        "source_gross_points_total",
        "source_mfe_retention_median",
        "source_top2_positive_return_share",
        "source_top5_positive_return_share",
        "source_non_gap_cost_total_return",
        "source_gap_trade_count",
        "source_synthetic_signal_trade_count",
        "source_zero_trade_bar_exposure_count",
        "source_synthetic_bar_exposure_count",
        "source_stage_id",
        "source_stage_root",
        "source_plan_fingerprint",
    ]


def write_migration_plan() -> None:
    if PLAN_PATH.is_file():
        return
    write_json(
        PLAN_PATH,
        {
            "schema_version": 1,
            "status": "approved",
            "mode": "transfer_exact",
            "source": {
                "instrument_id": "K200_train",
                "display_name": "K200（训）",
                "sample_start": "2026-05-26 00:00:00",
                "sample_end": "2026-07-08 23:52:00",
                "timezone": "Asia/Seoul",
            },
            "source_test": {
                "instrument_id": "K200_test",
                "display_name": "K200（测）",
                "sample_start": str(K200_TEST_START),
                "sample_end": str(K200_TEST_END),
                "timezone": "Asia/Seoul",
                "evidence_role": "old_250_forward_test; new_100_posthoc_descriptive",
            },
            "target": {
                "instrument_id": "SImain",
                "display_name": "SI",
                "sample_start": str(cross.TARGET_START),
                "sample_end": str(cross.TARGET_END),
                "timezone": "America/Chicago",
            },
            "bar_interval_seconds": 15,
            "candidate_filter_contract": {
                "new_candidate_count": NEW_COUNT,
                "exclude_existing_si_candidate_count": 250,
                "source_evidence": "closed K200 four-slice temporal migration only",
                "si_results_read_for_selection": False,
                "combined_score": False,
            },
            "cost_bps": cross.ROUND_TRIP_COST_BPS,
            "workers": 4,
            "incremental_parent_run": PARENT_ID,
            "main_columns": ["K200（训）总收益", "K200（测）总收益", "SI 总收益"],
            "parameter_acceptance": "none",
        },
    )


def freeze_new_candidates() -> dict[str, Any]:
    freeze_path = NEW_ROOT / cross.FREEZE_NAME
    if freeze_path.is_file():
        cross.RUN_ID, cross.RUN_ROOT = NEW_ID, NEW_ROOT
        return cross.load_frozen_candidates()
    selected_temporal, tags = select_new_candidates()
    selected_ids = selected_temporal.combo_id.astype(str).tolist()
    summary = pd.read_csv(TRAIN_SUMMARY, low_memory=False)
    selected = summary.loc[summary.combo_id.astype(str).isin(selected_ids)].copy()
    selected = selected.set_index("combo_id").loc[selected_ids].reset_index()
    source_trades = load_training_trades(selected_ids)
    source_frame = load_bars(cross.K200_SOURCE, cross.K200_PREPARATION)
    source_enriched, source_metrics = cross._add_excursions(
        source_trades, source_frame, prefix="source"
    )
    selected = selected.drop(
        columns=[
            column
            for column in selected.columns
            if column.startswith("source_")
            and column
            not in {
                "source_campaign_id",
                "source_stage_id",
                "source_stage_root",
                "source_plan_fingerprint",
                "source_stage_key",
            }
        ]
    ).merge(source_metrics, on="combo_id", how="left", validate="one_to_one")
    selected["selection_tags"] = selected.combo_id.astype(str).map(
        lambda value: "|".join(sorted(tags[value]))
    )
    selected["candidate_order"] = np.arange(1, len(selected) + 1)
    candidates = selected[candidate_columns()].copy()
    payload_without_hash = {
        "schema_version": 1,
        "status": "frozen_before_target_evaluation",
        "generated_at_utc": utc_now(),
        "source_instrument": "K200",
        "source_sample": {
            "start": "2026-05-26 00:00:00",
            "end": "2026-07-08 23:52:00",
            "timezone": "Asia/Seoul",
        },
        "source_snapshot": {
            "path": str(TRAIN_SUMMARY.parent),
            "union_snapshot_id": TRAIN_SUMMARY.parent.name,
            "analysis_summary": cross.artifact(TRAIN_SUMMARY),
            "trade_catalog": cross.artifact(TRAIN_TRADE_ROOT / "all_results_catalog.js"),
        },
        "source_identity": {
            "migration_plan": cross.artifact(PLAN_PATH),
            "temporal_comparison": cross.artifact(TEMPORAL_COMPARISON),
            "k200_market_data": cross.artifact(cross.K200_SOURCE),
            "k200_preparation_manifest": cross.artifact(cross.K200_PREPARATION),
            "engine": cross.artifact(CODE / "v4_4_engine.py"),
        },
        "selection_contract": {
            "no_si_results_read": True,
            "existing_si_ids_used_for_anti_join_only": True,
            "candidate_count": NEW_COUNT,
            "multi_metric_queues": sorted(
                {tag for values in tags.values() for tag in values}
            ),
            "combined_score": False,
        },
        "candidate_count": int(len(candidates)),
        "candidates": cross._records(candidates),
    }
    payload = {
        **payload_without_hash,
        "content_sha256": cross.canonical_hash(payload_without_hash),
    }
    NEW_ROOT.mkdir(parents=True, exist_ok=True)
    cross.atomic_json(freeze_path, payload)
    cross.atomic_csv(NEW_ROOT / "frozen_candidates.csv", candidates)
    cross.atomic_csv(NEW_ROOT / "source_candidate_trades.csv", source_enriched)
    return payload


def evaluate_new_si() -> dict[str, Any]:
    cross.RUN_ID, cross.RUN_ROOT = NEW_ID, NEW_ROOT
    if (NEW_ROOT / "migration_report.json").is_file():
        return json.loads((NEW_ROOT / "migration_report.json").read_text(encoding="utf-8"))
    report = cross.evaluate_target(workers=4)
    config_path = NEW_ROOT / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        migration_plan=str(PLAN_PATH),
        temporal_selection_source=str(TEMPORAL_COMPARISON),
        incremental_parent_run=PARENT_ID,
    )
    write_json(config_path, config)
    return report


def _test_worker_init(source: str, preparation: str) -> None:
    global _TEST_FRAME
    _TEST_FRAME = load_bars(Path(source), Path(preparation))


def _combo(record: dict[str, Any]) -> Combo:
    return Combo(
        method=str(record["method"]),
        e=int(record["e"]),
        bh=int(record["bh"]),
        trw=int(record["trw"]),
        k=float(record["k"]),
        w=int(record["w"]),
        m=float(record["m"]),
        entry_fill_mode=str(record["entry_fill_mode"]),
        entry_execution_policy=str(record["entry_execution_policy"]),
        entry_slippage=float(record["entry_slippage"]),
        speed_window_bars=int(record["speed_window_bars"]),
        baseline_sampling_policy=str(record["baseline_sampling_policy"]),
    )


def _test_worker(record: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if _TEST_FRAME is None:
        raise RuntimeError("K200 test worker frame is not initialized")
    combo = _combo(record)
    trades = simulate_combo(_TEST_FRAME, combo, K200_TEST_START, K200_TEST_END)
    return combo.combo_id, trades


def combined_candidates() -> pd.DataFrame:
    parent = json.loads((PARENT_ROOT / cross.FREEZE_NAME).read_text(encoding="utf-8"))
    new = json.loads((NEW_ROOT / cross.FREEZE_NAME).read_text(encoding="utf-8"))
    frame = pd.DataFrame([*parent["candidates"], *new["candidates"]])
    if len(frame) != 350 or frame.combo_id.astype(str).duplicated().any():
        raise ValueError("combined candidate population must contain 350 unique rows")
    frame["candidate_order"] = np.arange(1, len(frame) + 1)
    return frame


def evaluate_k200_test(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades_path = COMBINED_ROOT / "k200_test_candidate_trades.csv"
    metrics_path = COMBINED_ROOT / "k200_test_metrics.csv"
    if trades_path.is_file() and metrics_path.is_file():
        return (
            pd.read_csv(trades_path, low_memory=False),
            pd.read_csv(metrics_path, low_memory=False),
        )
    records = candidates.to_dict("records")
    trade_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=4,
        initializer=_test_worker_init,
        initargs=(str(cross.K200_SOURCE), str(cross.K200_PREPARATION)),
    ) as executor:
        futures = {executor.submit(_test_worker, record): record["combo_id"] for record in records}
        for index, future in enumerate(as_completed(futures), start=1):
            combo_id, trades = future.result()
            trade_rows.extend({**trade, "combo_id": combo_id} for trade in trades)
            print(f"K200 test {index}/{len(futures)} combo={combo_id} trades={len(trades)}", flush=True)
    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise ValueError("K200 test produced no trades")
    trades["gross_return"] = pd.to_numeric(trades["return"], errors="raise")
    frame = load_bars(cross.K200_SOURCE, cross.K200_PREPARATION)
    enriched, metrics = cross._add_excursions(trades, frame, prefix="source_test")
    COMBINED_ROOT.mkdir(parents=True, exist_ok=True)
    cross.atomic_csv(trades_path, enriched)
    cross.atomic_csv(metrics_path, metrics)
    return enriched, metrics


def pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    values = frame[
        [
            "source_cost_total_return",
            "source_test_cost_total_return",
            "target_cost_total_return",
            "target_cost_max_drawdown_abs",
        ]
    ].astype(float).to_numpy(copy=True)
    values[:, 3] *= -1.0
    keep = np.ones(len(values), dtype=bool)
    for index in range(len(values)):
        dominated = np.all(values >= values[index], axis=1) & np.any(
            values > values[index], axis=1
        )
        dominated[index] = False
        if dominated.any():
            keep[index] = False
    return keep


def recompute_combined_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.drop(columns=["default_rank"], errors="ignore").copy()
    output["source_rank_percentile"] = cross._rank_percentile(
        output.source_cost_total_return, ascending=False
    )
    output["target_rank_percentile"] = cross._rank_percentile(
        output.target_cost_total_return, ascending=False
    )
    output["rank_percentile_change"] = (
        output.target_rank_percentile - output.source_rank_percentile
    )
    output["positive_return_consistency"] = np.select(
        [
            output.source_cost_total_return.gt(0)
            & output.source_test_cost_total_return.gt(0)
            & output.target_cost_total_return.gt(0),
            output.target_cost_total_return.gt(0),
        ],
        ["all_three_positive", "si_positive_mixed_k200"],
        default="si_nonpositive",
    )
    adjacency = cross._adjacency(output)
    source_positive = output.set_index("combo_id").source_cost_total_return.gt(0).to_dict()
    target_positive = output.set_index("combo_id").target_cost_total_return.gt(0).to_dict()
    neighbor_count: list[int] = []
    source_share: list[float] = []
    target_share: list[float] = []
    for combo_id in output.combo_id.astype(str):
        neighbors = sorted(adjacency.get(combo_id, set()))
        neighbor_count.append(len(neighbors))
        source_share.append(
            float(np.mean([source_positive[item] for item in neighbors]))
            if neighbors
            else math.nan
        )
        target_share.append(
            float(np.mean([target_positive[item] for item in neighbors]))
            if neighbors
            else math.nan
        )
    output["neighbor_count"] = neighbor_count
    output["source_neighbor_positive_share"] = source_share
    output["target_neighbor_positive_share"] = target_share
    output["target_stable_region"] = (
        output.target_cost_total_return.gt(0)
        & output.neighbor_count.ge(2)
        & output.target_neighbor_positive_share.ge(0.6)
    )
    output["target_isolated_positive"] = (
        output.target_cost_total_return.gt(0)
        & (
            output.neighbor_count.lt(2)
            | output.target_neighbor_positive_share.lt(0.5)
        )
    )
    output["source_gap_trade_share"] = np.where(
        output.source_trade_count.gt(0),
        output.source_gap_trade_count / output.source_trade_count,
        np.nan,
    )
    output["target_gap_trade_share"] = np.where(
        output.target_trade_count.gt(0),
        output.target_gap_trade_count / output.target_trade_count,
        np.nan,
    )
    output["target_low_activity_audit_status"] = (
        "no_bound_simain_exclusion_policy; zero-trade and synthetic exposure audited"
    )
    output["cross_instrument_pareto"] = pareto_mask(output)
    output = output.sort_values(
        [
            "target_cost_total_return",
            "target_cost_max_drawdown_abs",
            "target_cost_median_trade",
            "combo_id",
        ],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    output.insert(0, "default_rank", np.arange(1, len(output) + 1))
    return output


def combine_run(test_metrics: pd.DataFrame) -> pd.DataFrame:
    parent = pd.read_csv(PARENT_ROOT / "migration_comparison.csv", low_memory=False)
    new = pd.read_csv(NEW_ROOT / "migration_comparison.csv", low_memory=False)
    comparison = pd.concat([parent, new], ignore_index=True, sort=False)
    if len(comparison) != 350 or comparison.combo_id.astype(str).duplicated().any():
        raise ValueError("SI comparison union must contain 350 unique rows")
    comparison = comparison.merge(
        test_metrics, on="combo_id", how="left", validate="one_to_one"
    )
    comparison = recompute_combined_diagnostics(comparison)
    parent_source = pd.read_csv(PARENT_ROOT / "source_candidate_trades.csv", low_memory=False)
    new_source = pd.read_csv(NEW_ROOT / "source_candidate_trades.csv", low_memory=False)
    source_trades = pd.concat([parent_source, new_source], ignore_index=True, sort=False)
    parent_target = pd.read_csv(PARENT_ROOT / "simain_candidate_trades.csv", low_memory=False)
    new_target = pd.read_csv(NEW_ROOT / "simain_candidate_trades.csv", low_memory=False)
    target_trades = pd.concat([parent_target, new_target], ignore_index=True, sort=False)

    parent_freeze = json.loads((PARENT_ROOT / cross.FREEZE_NAME).read_text(encoding="utf-8"))
    new_freeze = json.loads((NEW_ROOT / cross.FREEZE_NAME).read_text(encoding="utf-8"))
    candidates = [*parent_freeze["candidates"], *new_freeze["candidates"]]
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_order"] = index
    freeze_without_hash = {
        "schema_version": 1,
        "status": "presentation_union_of_completed_si_transfers",
        "generated_at_utc": utc_now(),
        "source_freezes": [
            {"run_id": PARENT_ID, "artifact": cross.artifact(PARENT_ROOT / cross.FREEZE_NAME)},
            {"run_id": NEW_ID, "artifact": cross.artifact(NEW_ROOT / cross.FREEZE_NAME)},
        ],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    freeze = {
        **freeze_without_hash,
        "content_sha256": cross.canonical_hash(freeze_without_hash),
    }
    COMBINED_ROOT.mkdir(parents=True, exist_ok=True)
    cross.atomic_csv(COMBINED_ROOT / "migration_comparison.csv", comparison)
    cross.atomic_csv(COMBINED_ROOT / "source_candidate_trades.csv", source_trades)
    cross.atomic_csv(COMBINED_ROOT / "simain_candidate_trades.csv", target_trades)
    cross.atomic_json(COMBINED_ROOT / cross.FREEZE_NAME, freeze)
    cross.atomic_csv(COMBINED_ROOT / "frozen_candidates.csv", pd.DataFrame(candidates))

    parent_report = json.loads((PARENT_ROOT / "migration_report.json").read_text(encoding="utf-8"))
    adjacency = cross._adjacency(comparison)
    stable_regions = cross._stable_components(comparison, adjacency)
    representative = cross._representative_trades(target_trades, comparison)
    pairwise = {
        "train_test": cross._spearman_correlation(
            comparison.source_cost_total_return,
            comparison.source_test_cost_total_return,
        ),
        "train_si": cross._spearman_correlation(
            comparison.source_cost_total_return,
            comparison.target_cost_total_return,
        ),
        "test_si": cross._spearman_correlation(
            comparison.source_test_cost_total_return,
            comparison.target_cost_total_return,
        ),
    }
    triple_positive = int(
        (
            comparison.source_cost_total_return.gt(0)
            & comparison.source_test_cost_total_return.gt(0)
            & comparison.target_cost_total_return.gt(0)
        ).sum()
    )
    report = {
        **parent_report,
        "status": "complete_train_test_si_350_union",
        "generated_at_utc": utc_now(),
        "candidate_freeze": {
            "path": str((COMBINED_ROOT / cross.FREEZE_NAME).resolve()),
            "content_sha256": freeze["content_sha256"],
            "file_sha256": cross.sha256_file(COMBINED_ROOT / cross.FREEZE_NAME),
            "candidate_count": 350,
            "target_results_cannot_modify_candidates": True,
        },
        "evaluation": {
            **parent_report["evaluation"],
            "candidate_count": 350,
            "existing_candidate_count": 250,
            "new_temporal_candidate_count": 100,
            "target_positive_candidate_count": int(
                comparison.target_cost_total_return.gt(0).sum()
            ),
            "target_positive_candidate_fraction": float(
                comparison.target_cost_total_return.gt(0).mean()
            ),
            "triple_positive_candidate_count": triple_positive,
            "stable_candidate_count": int(comparison.target_stable_region.sum()),
            "isolated_positive_count": int(comparison.target_isolated_positive.sum()),
            "pairwise_spearman": pairwise,
            "three_return_columns": [
                "source_cost_total_return",
                "source_test_cost_total_return",
                "target_cost_total_return",
            ],
        },
        "stable_parameter_regions": stable_regions,
        "failed_parameter_common_features": cross._failure_features(comparison),
        "return_concentration": {
            "target_top2_share_median": cross._clean_number(
                comparison.target_top2_positive_return_share.median()
            ),
            "source_top2_share_median": cross._clean_number(
                comparison.source_top2_positive_return_share.median()
            ),
            "target_top5_share_median": cross._clean_number(
                comparison.target_top5_positive_return_share.median()
            ),
            "source_top5_share_median": cross._clean_number(
                comparison.source_top5_positive_return_share.median()
            ),
        },
        "representative_trade_count": int(len(representative)),
        "parameter_acceptance": "none",
    }
    cross.atomic_json(COMBINED_ROOT / "migration_report.json", report)
    cross.atomic_csv(COMBINED_ROOT / "representative_trades.csv", representative)
    cross.atomic_json(
        COMBINED_ROOT / "posthoc_full_grid_status.json", report["posthoc_full_grid"]
    )
    config = {
        "schema_version": 1,
        "run_id": COMBINED_ID,
        "mode": "train_test_si_presentation_union",
        "source": {
            "instrument": "K200（训）",
            "sample_start": "2026-05-26 00:00:00",
            "sample_end": "2026-07-08 23:52:00",
            "market_data": cross.artifact(cross.K200_SOURCE),
            "preparation_manifest": cross.artifact(cross.K200_PREPARATION),
            "snapshot_root": str(TRAIN_SUMMARY.parent),
        },
        "source_test": {
            "instrument": "K200（测）",
            "sample_start": str(K200_TEST_START),
            "sample_end": str(K200_TEST_END),
            "market_data": cross.artifact(cross.K200_SOURCE),
        },
        "target": json.loads((PARENT_ROOT / "run_config.json").read_text(encoding="utf-8"))["target"],
        "candidate_freeze": cross.artifact(COMBINED_ROOT / cross.FREEZE_NAME),
        "candidate_content_sha256": freeze["content_sha256"],
        "cost_bps": cross.ROUND_TRIP_COST_BPS,
        "source_instrument_profile": cross.artifact(cross.K200_PROFILE_PATH),
        "workers": 4,
        "engine": cross.artifact(CODE / "v4_4_engine.py"),
        "result_semantics": "250 retained SI results plus 100 frozen temporal candidates; K200 test replay across all 350",
        "posthoc_full_grid": "not_run_separate_optional_diagnostic",
        "migration_plan": str(PLAN_PATH),
        "source_runs": [PARENT_ID, NEW_ID],
        "incremental_parent_run": PARENT_ID,
        "candidate_count": 350,
    }
    write_json(COMBINED_ROOT / "run_config.json", config)
    return comparison


def write_final_report(comparison: pd.DataFrame) -> None:
    report = json.loads((COMBINED_ROOT / "migration_report.json").read_text(encoding="utf-8"))
    new_ids = set(
        pd.read_csv(NEW_ROOT / "migration_comparison.csv", usecols=["combo_id"])
        .combo_id.astype(str)
    )
    comparison = comparison.copy()
    comparison["candidate_batch"] = np.where(
        comparison.combo_id.astype(str).isin(new_ids), "new100", "old250"
    )
    triple = comparison.loc[
        comparison.source_cost_total_return.gt(0)
        & comparison.source_test_cost_total_return.gt(0)
        & comparison.target_cost_total_return.gt(0)
    ].copy()
    pareto = comparison.loc[comparison.cross_instrument_pareto.astype(bool)].copy()
    robust = triple.loc[
        triple.source_trade_count.ge(10)
        & triple.source_test_trade_count.ge(10)
        & triple.target_trade_count.ge(10)
    ].copy()
    robust["minimum_three_return"] = robust[
        [
            "source_cost_total_return",
            "source_test_cost_total_return",
            "target_cost_total_return",
        ]
    ].min(axis=1)
    robust = robust.sort_values(
        ["minimum_three_return", "target_cost_max_drawdown_abs"],
        ascending=[False, True],
    )
    top_lines = []
    for _, row in robust.head(6).iterrows():
        top_lines.append(
            f"- E={int(row.e)}，BH={int(row.bh)}，TRW={int(row.trw)}，K={row.k:g}，"
            f"W={int(row.w)}，M={row.m:g}，S={int(row.speed_window_bars)}："
            f"K200（训）{row.source_cost_total_return * 100:.3f}%，"
            f"K200（测）{row.source_test_cost_total_return * 100:.3f}%，"
            f"SI {row.target_cost_total_return * 100:.3f}%；"
            f"三处交易数 {int(row.source_trade_count)}/{int(row.source_test_trade_count)}/{int(row.target_trade_count)}。"
        )
    pairwise = report["evaluation"]["pairwise_spearman"]
    old = comparison.loc[comparison.candidate_batch.eq("old250")]
    new = comparison.loc[comparison.candidate_batch.eq("new100")]
    old_triple = old.loc[
        old.source_cost_total_return.gt(0)
        & old.source_test_cost_total_return.gt(0)
        & old.target_cost_total_return.gt(0)
    ]
    new_triple = new.loc[
        new.source_cost_total_return.gt(0)
        & new.source_test_cost_total_return.gt(0)
        & new.target_cost_total_return.gt(0)
    ]
    cluster = new.loc[
        new.e.eq(320)
        & new.bh.eq(240)
        & new.trw.eq(12)
        & new.k.eq(1.25)
        & new.w.eq(6)
        & new.m.isin([4.25, 4.5, 4.75])
        & new.speed_window_bars.between(340, 370)
    ]
    cluster_summary = (
        f"这组连续区域共有 {len(cluster)} 个点。成本后收益中位数为 "
        f"K200（训）{cluster.source_cost_total_return.median() * 100:.3f}%、"
        f"K200（测）{cluster.source_test_cost_total_return.median() * 100:.3f}%、"
        f"SI {cluster.target_cost_total_return.median() * 100:.3f}%。"
        f"K200（测）的非 gap 收益中位数为 "
        f"{cluster.source_test_non_gap_cost_total_return.median() * 100:.3f}%，"
        "说明该段正收益明显依赖跨 gap 交易，当前不能据此接纳参数。"
        if len(cluster)
        else "当前没有形成预定义的连续参数区域。"
    )
    text = f"""# V4.4 K200（训）／K200（测）／SI 迁移结论

## 结论

本次统一比较 350 组参数：保留现有 250 组 SI 结果，新增 100 组由 K200 时间迁移证据冻结的候选。新增候选在读取 SI 结果以前完成冻结。K200（测）对 350 组统一重放。

- SI 成本后为正：{int(comparison.target_cost_total_return.gt(0).sum())}/350。
- K200（训）、K200（测）、SI 同时为正：{len(triple)}/350。
- 三组收益 Pareto 候选：{len(pareto)} 组。
- 收益排名 Spearman：训练／测试 `{pairwise['train_test']:.3f}`，训练／SI `{pairwise['train_si']:.3f}`，测试／SI `{pairwise['test_si']:.3f}`。

## 证据边界

- 原有 250 组：K200（测）是后续行情证据；SI 已参与此前的迁移结果，不能再次当成未见验证。其 K200（测）为正 127/250，三组同时为正 106/250。
- 新增 100 组：在查看本批 SI 结果前完成冻结，所以 SI 是新的跨品种证据；候选选择已经使用 K200 时间迁移结果，因此 K200（测）只作描述。其 K200（测）为正 72/100，SI 为正 59/100，三组同时为正 43/100。
- 当前没有一组候选同时拥有三列都完全未见的证据。下一段 K200 新行情才是本批候选最重要的前推验证。

## 目前最有希望的区域

新增候选中出现一个清楚的邻域：E=320、BH=240、TRW=12、K=1.25、W=6，M 在 4.25–4.75，S 在 340–370。换算后，E 约 80 分钟、BH 约 60 分钟、TRW 约 3 分钟、W 约 90 秒、S 约 85–92.5 分钟。

{cluster_summary}

按三列收益中的最小值排序，并要求三处各有至少 10 笔交易，代表点如下：

{chr(10).join(top_lines) if top_lines else '- 当前没有满足条件的代表点。'}

## 对交易哲学的判断

结果支持保留「相对异常下跌触发＋快慢两层退出」这套策略骨架。较好的连续区域把开仓观察放在约一小时尺度，把短撤退窗口压到约 90 秒，同时保留约 90 分钟的速度观察。这种尺度分层在 K200 与 SI 都出现正收益，值得继续研究。

固定参数长期通用的证据依旧不足。K200 四段迁移只有 2 组在每段都为正，训练／测试排名还是负相关；这说明行情状态变化会改写参数优劣。更有希望的框架是「用短期已知行情估计参数区域，冻结邻域，再评价随后行情」，同时保留跨品种检查。

当前最大风险来自 K200（测）的 gap 依赖。下一轮应把这 7 个邻域点完整冻结，在新的 K200 数据上同时查看总收益、非 gap 收益、交易数、回撤和收益集中度。若新数据中仍形成连续正收益，并且非 gap 收益转正或不再显著拖累，才有资格进入更高等级验证。

## 前景

策略骨架有继续研究的价值，静态万能参数的前景偏弱，短期重估参数区域的前景更好。眼下最有信息价值的动作是等待新的 K200 未见行情，再验证上述邻域；无需继续围绕本轮 K200（测）细调，否则会把测试集逐渐变成训练集。

## 交付

- 三组收益主入口：`{COMBINED_ROOT / 'index.html'}`
- SI 逐笔分析：`{COMBINED_ROOT / 'trade_review' / 'index.html'}`
- 完整比较：`{COMBINED_ROOT / 'migration_comparison.csv'}`
- K200 时间迁移报告：`{TEMPORAL_ROOT / 'TEMPORAL_MIGRATION_REPORT.md'}`

参数接纳：无（`parameter_acceptance=none`）。
"""
    (COMBINED_ROOT / "FINAL_TRAIN_TEST_SI_REPORT.md").write_text(text, encoding="utf-8")
    (COMBINED_ROOT / "MIGRATION_REPORT.zh.md").write_text(
        "# 迁移报告\n\n详见 `FINAL_TRAIN_TEST_SI_REPORT.md`。\n", encoding="utf-8"
    )
    (COMBINED_ROOT / "MIGRATION_REPORT.en.md").write_text(
        "# Migration report\n\nSee `FINAL_TRAIN_TEST_SI_REPORT.md`.\n", encoding="utf-8"
    )


def publish() -> None:
    cross.RUN_ID, cross.RUN_ROOT = COMBINED_ID, COMBINED_ROOT
    cross.build_page()


def run_all() -> None:
    write_migration_plan()
    frozen = freeze_new_candidates()
    print(json.dumps({"phase": "freeze", "candidate_count": frozen["candidate_count"]}, ensure_ascii=False), flush=True)
    report = evaluate_new_si()
    print(json.dumps({"phase": "si", "positive": report["evaluation"]["target_positive_candidate_count"]}, ensure_ascii=False), flush=True)
    candidates = combined_candidates()
    _, test_metrics = evaluate_k200_test(candidates)
    comparison = combine_run(test_metrics)
    write_final_report(comparison)
    publish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("all", "freeze", "si", "test", "publish"), nargs="?", default="all")
    args = parser.parse_args()
    write_migration_plan()
    if args.action in {"all", "freeze", "si", "test", "publish"}:
        freeze_new_candidates()
    if args.action in {"all", "si", "test", "publish"}:
        evaluate_new_si()
    candidates = combined_candidates() if args.action in {"all", "test", "publish"} else None
    if args.action in {"all", "test", "publish"}:
        _, metrics = evaluate_k200_test(candidates)
        comparison = combine_run(metrics)
        write_final_report(comparison)
    if args.action in {"all", "publish"}:
        publish()


if __name__ == "__main__":
    main()
