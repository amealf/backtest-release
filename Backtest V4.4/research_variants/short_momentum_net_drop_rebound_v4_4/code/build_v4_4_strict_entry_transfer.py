from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import build_v4_4_cross_instrument_comparison as base


OLD_RUN_ID = "k200_20260526_20260708__simain_20260129_20260223"
STRICT_RUN_ID = (
    "k200_20260526_20260708__simain_20260129_20260223"
    "__strict_entry_transfer_v40_20260805"
)
AGGREGATE_RUN_ID = (
    "k200_20260526_20260708__simain_20260129_20260223"
    "__all_exact_transfers_v46_20260805"
)
MAX_STRICT_CANDIDATES = 100
FIXED_RESEARCH_COST_BPS = 3.57
STRICT_BATCH_LABEL = "严格开仓迁移"
ORIGINAL_BATCH_LABEL = "原始迁移"


def pareto_mask(
    frame: pd.DataFrame,
    *,
    maximize: Iterable[str],
    minimize: Iterable[str],
) -> pd.Series:
    maximize = tuple(maximize)
    minimize = tuple(minimize)
    values = frame.loc[:, [*maximize, *minimize]].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(float)
    split = len(maximize)
    output = np.ones(len(frame), dtype=bool)
    for index, row in enumerate(values):
        at_least_as_good = np.ones(len(frame), dtype=bool)
        strictly_better = np.zeros(len(frame), dtype=bool)
        if split:
            at_least_as_good &= np.all(values[:, :split] >= row[:split], axis=1)
            strictly_better |= np.any(values[:, :split] > row[:split], axis=1)
        if split < values.shape[1]:
            at_least_as_good &= np.all(values[:, split:] <= row[split:], axis=1)
            strictly_better |= np.any(values[:, split:] < row[split:], axis=1)
        output[index] = not bool(np.any(at_least_as_good & strictly_better))
    return pd.Series(output, index=frame.index)


def _write_aggregate_candidate_freeze(
    aggregate_root: Path,
    candidate_union: pd.DataFrame,
    aggregate_freeze: dict[str, Any],
) -> None:
    base.atomic_csv(aggregate_root / "frozen_candidates.csv", candidate_union)
    base.atomic_json(aggregate_root / base.FREEZE_NAME, aggregate_freeze)


def _set_run(run_id: str) -> Path:
    run_root = base.CROSS_ROOT / "runs" / run_id
    base.RUN_ID = run_id
    base.RUN_ROOT = run_root
    base.ROUND_TRIP_COST_BPS = FIXED_RESEARCH_COST_BPS
    return run_root


def _threshold_metrics(snapshot: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        snapshot / "union_trades.csv",
        usecols=["combo_id", "entry_baseline_value", "k"],
        chunksize=100_000,
    ):
        chunk["source_entry_threshold"] = (
            pd.to_numeric(chunk["entry_baseline_value"], errors="raise")
            * pd.to_numeric(chunk["k"], errors="raise")
        )
        rows.append(chunk[["combo_id", "source_entry_threshold"]])
    trades = pd.concat(rows, ignore_index=True)
    return (
        trades.groupby("combo_id", sort=False)["source_entry_threshold"]
        .median()
        .rename("source_entry_threshold_median")
        .reset_index()
    )


def _champion_ids(population: pd.DataFrame) -> set[str]:
    champions: set[str] = set()
    total = population.sort_values(
        ["source_cost_total_return", "combo_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    if not total.empty:
        champions.add(str(total.iloc[0].combo_id))
    scenario = population.loc[base._truthy(population["scenario_1_qualified"])]
    scenario = scenario.sort_values(
        ["source_cost_total_return", "combo_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    if not scenario.empty:
        champions.add(str(scenario.iloc[0].combo_id))
    for minimum in (10, 20):
        average = population.loc[population.source_trade_count.ge(minimum)].sort_values(
            ["source_cost_mean_trade", "source_cost_total_return", "combo_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        if not average.empty:
            champions.add(str(average.iloc[0].combo_id))
    return champions


def select_strict_candidates(
    population: pd.DataFrame,
    previous_candidates: pd.DataFrame,
    *,
    maximum: int = MAX_STRICT_CANDIDATES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ranked = population.sort_values(
        ["source_cost_total_return", "combo_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["source_cost_rank"] = np.arange(1, len(ranked) + 1)
    ranked["source_cost_rank_percentile"] = (
        ranked.source_cost_rank / float(len(ranked))
    )
    previous_ids = set(previous_candidates.combo_id.astype(str))
    permitted_wms = set(
        zip(
            pd.to_numeric(previous_candidates.w, errors="raise").astype(int),
            pd.to_numeric(previous_candidates.m, errors="raise").astype(float),
            pd.to_numeric(
                previous_candidates.speed_window_bars, errors="raise"
            ).astype(int),
        )
    )
    champions = _champion_ids(ranked)
    pool = ranked.loc[
        ranked.source_cost_rank_percentile.le(0.20)
        & ranked.source_trade_count.ge(10)
        & ~ranked.combo_id.astype(str).isin(previous_ids | champions)
    ].copy()
    pool = pool.loc[
        pool.apply(
            lambda row: (
                int(row.w),
                float(row.m),
                int(row.speed_window_bars),
            )
            in permitted_wms,
            axis=1,
        )
    ].copy()

    frontier_parts: list[pd.DataFrame] = []
    for _, family in pool.groupby(
        ["w", "m", "speed_window_bars"], sort=True, dropna=False
    ):
        frontier_parts.append(
            family.loc[
                pareto_mask(
                    family,
                    maximize=(
                        "source_cost_total_return",
                        "source_entry_threshold_median",
                    ),
                    minimize=("source_trade_count",),
                )
            ]
        )
    frontier = (
        pd.concat(frontier_parts, ignore_index=True)
        if frontier_parts
        else pool.iloc[0:0].copy()
    )
    frontier = frontier.sort_values(
        [
            "w",
            "m",
            "speed_window_bars",
            "source_entry_threshold_median",
            "source_trade_count",
            "source_cost_total_return",
            "combo_id",
        ],
        ascending=[True, True, True, False, True, False, True],
        kind="mergesort",
    )

    if len(frontier) > maximum:
        grouped = [
            group.reset_index(drop=True)
            for _, group in frontier.groupby(
                ["w", "m", "speed_window_bars"], sort=True, dropna=False
            )
        ]
        selected_rows: list[pd.Series] = []
        offset = 0
        while len(selected_rows) < maximum:
            added = False
            for group in grouped:
                if offset < len(group):
                    selected_rows.append(group.iloc[offset])
                    added = True
                    if len(selected_rows) == maximum:
                        break
            if not added:
                break
            offset += 1
        frontier = pd.DataFrame(selected_rows)

    frontier = frontier.reset_index(drop=True)
    frontier["candidate_order"] = np.arange(1, len(frontier) + 1)
    frontier["selection_tags"] = (
        "k200_top20_ge10|unseen_in_previous_transfers|"
        "non_champion|within_wms_entry_strictness_pareto"
    )
    frontier["transfer_batch"] = STRICT_BATCH_LABEL
    audit = {
        "population_count": int(len(ranked)),
        "top_20_percent_rank_max": int(math.floor(len(ranked) * 0.20)),
        "minimum_source_trade_count": 10,
        "previous_candidate_count": int(len(previous_ids)),
        "permitted_wms_family_count": int(len(permitted_wms)),
        "excluded_champion_count": int(len(champions)),
        "eligible_pool_count": int(len(pool)),
        "within_wms_pareto_count_before_cap": int(
            sum(len(part) for part in frontier_parts)
        ),
        "maximum_candidate_count": int(maximum),
        "selected_candidate_count": int(len(frontier)),
        "selected_wms_family_count": int(
            frontier.groupby(["w", "m", "speed_window_bars"]).ngroups
        ),
        "internal_duplicate_combo_count": int(
            frontier.combo_id.astype(str).duplicated().sum()
        ),
        "overlap_with_previous_count": int(
            frontier.combo_id.astype(str).isin(previous_ids).sum()
        ),
        "champion_overlap_count": int(
            frontier.combo_id.astype(str).isin(champions).sum()
        ),
    }
    return frontier, audit


def freeze_strict_candidates() -> dict[str, Any]:
    run_root = _set_run(STRICT_RUN_ID)
    if run_root.exists():
        raise FileExistsError(f"strict transfer output already exists: {run_root}")
    snapshot = base.current_snapshot_root()
    summary_path = snapshot / "analysis_summary.csv"
    summary = pd.read_csv(summary_path)
    fixed_metrics, source_trades = base._source_trade_metrics(snapshot)
    thresholds = _threshold_metrics(snapshot)
    population = (
        summary.merge(fixed_metrics, on="combo_id", how="inner", validate="one_to_one")
        .merge(thresholds, on="combo_id", how="inner", validate="one_to_one")
    )
    if len(population) != len(summary):
        raise ValueError("source summary, trade metrics, and threshold metrics differ")
    old_root = base.CROSS_ROOT / "runs" / OLD_RUN_ID
    old_freeze_path = old_root / base.FREEZE_NAME
    old_freeze = json.loads(old_freeze_path.read_text(encoding="utf-8"))
    old_candidates = pd.DataFrame(old_freeze["candidates"])
    selected, selection_audit = select_strict_candidates(
        population,
        old_candidates,
        maximum=MAX_STRICT_CANDIDATES,
    )
    if selected.empty:
        raise ValueError("strict-entry source selection produced no candidates")

    selected_ids = set(selected.combo_id.astype(str))
    selected_source_trades = source_trades.loc[
        source_trades.combo_id.astype(str).isin(selected_ids)
    ].copy()
    source_frame = base.load_bars(base.K200_SOURCE, base.K200_PREPARATION)
    source_enriched, source_excursions = base._add_excursions(
        selected_source_trades,
        source_frame,
        prefix="source",
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
                "source_entry_threshold_median",
                "source_cost_rank",
                "source_cost_rank_percentile",
            }
        ]
    ).merge(source_excursions, on="combo_id", how="left", validate="one_to_one")

    candidate_columns = [
        "candidate_order",
        "combo_id",
        *base.EXECUTION_FIELDS,
        *base.PARAMETER_FIELDS,
        "transfer_batch",
        "scenario_1_qualified",
        "scenario_2_qualified",
        "scenario_3_qualified",
        "selection_tags",
        "source_entry_threshold_median",
        "source_cost_rank",
        "source_cost_rank_percentile",
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
    candidates = selected[candidate_columns].copy()
    payload_without_hash = {
        "schema_version": 2,
        "status": "frozen_before_target_evaluation",
        "generated_at_utc": base.utc_now(),
        "mode": "transfer_exact",
        "source_instrument": "K200",
        "target_instrument": "SImain",
        "source_sample": {
            "start": str(base.SOURCE_START),
            "end": str(base.SOURCE_END),
            "timezone": "Asia/Seoul",
        },
        "source_snapshot": {
            "path": str(snapshot),
            "union_snapshot_id": snapshot.name,
            "analysis_summary": base.artifact(summary_path),
            "union_trades": base.artifact(snapshot / "union_trades.csv"),
        },
        "source_identity": {
            "source_manifest": base.artifact(base.SOURCE_MANIFEST),
            "engine": base.artifact(base.CODE_DIR / "v4_4_engine.py"),
            "strict_transfer_builder": base.artifact(Path(__file__).resolve()),
            "k200_market_data": base.artifact(base.K200_SOURCE),
            "k200_preparation_manifest": base.artifact(base.K200_PREPARATION),
        },
        "selection_contract": {
            "no_target_results_read": True,
            "source_cost_universe": (
                f"all {len(population)} completed K200 coordinates in the frozen source snapshot"
            ),
            "source_cost_top_fraction": 0.20,
            "minimum_source_trade_count": 10,
            "w_m_s_policy": "only W/M/S families present in the previous frozen transfers",
            "varying_fields": ["E", "BH", "TRW", "K"],
            "strictness_fields": [
                "higher K200 median actual entry threshold",
                "lower K200 trade count",
            ],
            "actual_entry_threshold": "entry_baseline_value * K",
            "pareto_fields": {
                "maximize": [
                    "source_cost_total_return",
                    "source_entry_threshold_median",
                ],
                "minimize": ["source_trade_count"],
            },
            "previous_candidates_excluded": base.artifact(old_freeze_path),
            "primary_K200_champions_excluded": True,
            "maximum_candidates": MAX_STRICT_CANDIDATES,
            "combined_score": False,
            "fixed_research_cost_bps": FIXED_RESEARCH_COST_BPS,
        },
        "selection_audit": selection_audit,
        "candidate_count": int(len(candidates)),
        "candidates": base._records(candidates),
    }
    content_hash = base.canonical_hash(payload_without_hash)
    payload = {**payload_without_hash, "content_sha256": content_hash}
    run_root.mkdir(parents=True, exist_ok=False)
    base.atomic_json(run_root / base.FREEZE_NAME, payload)
    base.atomic_csv(run_root / "frozen_candidates.csv", candidates)
    base.atomic_csv(run_root / "source_candidate_trades.csv", source_enriched)
    base.atomic_csv(
        run_root / "source_strictness_population.csv",
        population[
            [
                "combo_id",
                "w",
                "m",
                "speed_window_bars",
                "source_trade_count",
                "source_cost_total_return",
                "source_cost_median_trade",
                "source_entry_threshold_median",
            ]
        ],
    )
    base.atomic_json(run_root / "selection_audit.json", selection_audit)
    return payload


def evaluate_strict_target(*, workers: int) -> dict[str, Any]:
    _set_run(STRICT_RUN_ID)
    report = base.evaluate_target(workers=workers)
    comparison_path = base.RUN_ROOT / "migration_comparison.csv"
    comparison = pd.read_csv(comparison_path)
    comparison["transfer_batch"] = STRICT_BATCH_LABEL
    base.atomic_csv(comparison_path, comparison)
    return report


def _recompute_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for field in (
        "default_rank",
        "source_rank_percentile",
        "target_rank_percentile",
        "rank_percentile_change",
        "positive_return_consistency",
        "neighbor_count",
        "source_neighbor_positive_share",
        "target_neighbor_positive_share",
        "target_stable_region",
        "target_isolated_positive",
        "source_gap_trade_share",
        "target_gap_trade_share",
        "cross_instrument_pareto",
    ):
        if field in frame.columns:
            frame = frame.drop(columns=[field])
    frame["source_rank_percentile"] = base._rank_percentile(
        frame.source_cost_total_return, ascending=False
    )
    frame["target_rank_percentile"] = base._rank_percentile(
        frame.target_cost_total_return, ascending=False
    )
    frame["rank_percentile_change"] = (
        frame.target_rank_percentile - frame.source_rank_percentile
    )
    frame["positive_return_consistency"] = np.select(
        [
            frame.source_cost_total_return.gt(0)
            & frame.target_cost_total_return.gt(0),
            frame.source_cost_total_return.gt(0)
            & frame.target_cost_total_return.le(0),
            frame.source_cost_total_return.le(0)
            & frame.target_cost_total_return.gt(0),
        ],
        [
            "both_positive",
            "source_positive_target_nonpositive",
            "source_nonpositive_target_positive",
        ],
        default="both_nonpositive",
    )
    adjacency = base._adjacency(frame)
    source_positive = (
        frame.set_index("combo_id").source_cost_total_return.gt(0).to_dict()
    )
    target_positive = (
        frame.set_index("combo_id").target_cost_total_return.gt(0).to_dict()
    )
    neighbor_count: list[int] = []
    source_share: list[float] = []
    target_share: list[float] = []
    for combo_id in frame.combo_id.astype(str):
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
    frame["neighbor_count"] = neighbor_count
    frame["source_neighbor_positive_share"] = source_share
    frame["target_neighbor_positive_share"] = target_share
    frame["target_stable_region"] = (
        frame.target_cost_total_return.gt(0)
        & frame.neighbor_count.ge(2)
        & frame.target_neighbor_positive_share.ge(0.6)
    )
    frame["target_isolated_positive"] = (
        frame.target_cost_total_return.gt(0)
        & (
            frame.neighbor_count.lt(2)
            | frame.target_neighbor_positive_share.lt(0.5)
        )
    )
    frame["source_gap_trade_share"] = np.where(
        frame.source_trade_count.gt(0),
        frame.source_gap_trade_count / frame.source_trade_count,
        np.nan,
    )
    frame["target_gap_trade_share"] = np.where(
        frame.target_trade_count.gt(0),
        frame.target_gap_trade_count / frame.target_trade_count,
        np.nan,
    )
    frame["cross_instrument_pareto"] = pareto_mask(
        frame,
        maximize=(
            "source_cost_total_return",
            "target_cost_total_return",
            "target_cost_median_trade",
        ),
        minimize=("target_trade_count", "target_cost_max_drawdown_abs"),
    )
    frame = frame.sort_values(
        [
            "target_cost_total_return",
            "target_cost_max_drawdown_abs",
            "target_cost_median_trade",
            "combo_id",
        ],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "default_rank", np.arange(1, len(frame) + 1))
    return frame


def _strictness_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    strict = frame.loc[frame.transfer_batch.eq(STRICT_BATCH_LABEL)].copy()
    strict["threshold_quartile"] = pd.qcut(
        strict.source_entry_threshold_median.rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    quartiles: list[dict[str, Any]] = []
    for label, group in strict.groupby("threshold_quartile", observed=True):
        quartiles.append(
            {
                "threshold_quartile": str(label),
                "candidate_count": int(len(group)),
                "source_entry_threshold_median": float(
                    group.source_entry_threshold_median.median()
                ),
                "target_trade_count_median": float(group.target_trade_count.median()),
                "target_cost_total_return_median": float(
                    group.target_cost_total_return.median()
                ),
                "target_cost_median_trade_median": float(
                    group.target_cost_median_trade.median()
                ),
                "target_cost_max_drawdown_abs_median": float(
                    group.target_cost_max_drawdown_abs.median()
                ),
            }
        )
    trades = [item["target_trade_count_median"] for item in quartiles]
    returns = [item["target_cost_total_return_median"] for item in quartiles]
    medians = [item["target_cost_median_trade_median"] for item in quartiles]
    drawdowns = [
        item["target_cost_max_drawdown_abs_median"] for item in quartiles
    ]
    monotonic = {
        "target_trade_count_nonincreasing": all(
            left >= right for left, right in zip(trades, trades[1:])
        ),
        "target_cost_total_return_nondecreasing": all(
            left <= right for left, right in zip(returns, returns[1:])
        ),
        "target_cost_median_trade_nondecreasing": all(
            left <= right for left, right in zip(medians, medians[1:])
        ),
        "target_drawdown_nonincreasing": all(
            left >= right for left, right in zip(drawdowns, drawdowns[1:])
        ),
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "strict_candidate_count": int(len(strict)),
        "source_threshold_vs_target_trade_count_spearman": base._spearman_correlation(
            strict.source_entry_threshold_median,
            strict.target_trade_count,
        ),
        "source_trade_count_vs_target_trade_count_spearman": base._spearman_correlation(
            strict.source_trade_count,
            strict.target_trade_count,
        ),
        "threshold_quartiles": quartiles,
        "continuous_improvement_checks": monotonic,
        "all_four_improve_continuously": bool(all(monotonic.values())),
        "cross_instrument_pareto_count_all_transfers": int(
            frame.cross_instrument_pareto.sum()
        ),
        "cross_instrument_pareto_count_strict_batch": int(
            strict.cross_instrument_pareto.sum()
        ),
        "combined_score": False,
    }


def build_aggregate_page() -> dict[str, Any]:
    old_root = base.CROSS_ROOT / "runs" / OLD_RUN_ID
    strict_root = base.CROSS_ROOT / "runs" / STRICT_RUN_ID
    aggregate_root = base.CROSS_ROOT / "runs" / AGGREGATE_RUN_ID
    if aggregate_root.exists():
        raise FileExistsError(f"aggregate transfer output already exists: {aggregate_root}")
    aggregate_root.mkdir(parents=True, exist_ok=False)

    old_comparison = pd.read_csv(old_root / "migration_comparison.csv")
    strict_comparison = pd.read_csv(strict_root / "migration_comparison.csv")
    old_comparison["transfer_batch"] = ORIGINAL_BATCH_LABEL
    strict_comparison["transfer_batch"] = STRICT_BATCH_LABEL
    threshold_population = pd.read_csv(
        strict_root / "source_strictness_population.csv"
    )[["combo_id", "source_entry_threshold_median"]]
    if "source_entry_threshold_median" not in old_comparison.columns:
        old_comparison = old_comparison.merge(
            threshold_population, on="combo_id", how="left", validate="one_to_one"
        )
    combined = pd.concat([old_comparison, strict_comparison], ignore_index=True)
    if combined.combo_id.astype(str).duplicated().any():
        raise ValueError("aggregate transfer contains duplicate combo_id values")
    combined = _recompute_diagnostics(combined)

    old_source = pd.read_csv(old_root / "source_candidate_trades.csv", low_memory=False)
    strict_source = pd.read_csv(
        strict_root / "source_candidate_trades.csv", low_memory=False
    )
    old_target = pd.read_csv(old_root / "simain_candidate_trades.csv", low_memory=False)
    strict_target = pd.read_csv(
        strict_root / "simain_candidate_trades.csv", low_memory=False
    )
    source_trades = pd.concat([old_source, strict_source], ignore_index=True)
    target_trades = pd.concat([old_target, strict_target], ignore_index=True)
    expected_ids = set(combined.combo_id.astype(str))
    if set(source_trades.combo_id.astype(str)) != expected_ids:
        raise ValueError("aggregate source trades do not cover every candidate")
    if set(target_trades.combo_id.astype(str)) != expected_ids:
        raise ValueError("aggregate target trades do not cover every candidate")

    old_freeze = json.loads((old_root / base.FREEZE_NAME).read_text(encoding="utf-8"))
    strict_freeze = json.loads(
        (strict_root / base.FREEZE_NAME).read_text(encoding="utf-8")
    )
    candidate_union = pd.concat(
        [pd.DataFrame(old_freeze["candidates"]), pd.DataFrame(strict_freeze["candidates"])],
        ignore_index=True,
    )
    candidate_union["candidate_order"] = np.arange(1, len(candidate_union) + 1)
    aggregate_freeze = {
        "schema_version": 1,
        "status": "presentation_union_of_completed_frozen_transfers",
        "generated_at_utc": base.utc_now(),
        "source_freezes": [
            {
                "run_id": OLD_RUN_ID,
                "artifact": base.artifact(old_root / base.FREEZE_NAME),
                "content_sha256": old_freeze["content_sha256"],
            },
            {
                "run_id": STRICT_RUN_ID,
                "artifact": base.artifact(strict_root / base.FREEZE_NAME),
                "content_sha256": strict_freeze["content_sha256"],
            },
        ],
        "candidate_count": int(len(candidate_union)),
        "candidates": base._records(candidate_union),
        "target_results_modified_source_freezes": False,
    }
    aggregate_freeze["content_sha256"] = base.canonical_hash(aggregate_freeze)

    strict_report = json.loads(
        (strict_root / "migration_report.json").read_text(encoding="utf-8")
    )
    strictness = _strictness_diagnostics(combined)
    adjacency = base._adjacency(combined)
    report = {
        **strict_report,
        "status": "complete_union_of_exact_transfer_batches",
        "generated_at_utc": base.utc_now(),
        "candidate_freeze": {
            "path": str((aggregate_root / base.FREEZE_NAME).resolve()),
            "content_sha256": aggregate_freeze["content_sha256"],
            "candidate_count": int(len(combined)),
            "source_freeze_count": 2,
            "target_results_cannot_modify_candidates": True,
        },
        "evaluation": {
            **strict_report["evaluation"],
            "candidate_count": int(len(combined)),
            "original_candidate_count": int(len(old_comparison)),
            "strict_entry_candidate_count": int(len(strict_comparison)),
            "target_positive_candidate_count": int(
                combined.target_cost_total_return.gt(0).sum()
            ),
            "target_positive_candidate_fraction": float(
                combined.target_cost_total_return.gt(0).mean()
            ),
            "rank_spearman_correlation": base._spearman_correlation(
                combined.source_cost_total_return,
                combined.target_cost_total_return,
            ),
            "stable_candidate_count": int(combined.target_stable_region.sum()),
            "isolated_positive_count": int(
                combined.target_isolated_positive.sum()
            ),
        },
        "stable_parameter_regions": base._stable_components(combined, adjacency),
        "isolated_parameter_points": combined.loc[
            combined.target_isolated_positive, "combo_id"
        ].astype(str).head(30).tolist(),
        "failed_parameter_common_features": base._failure_features(combined),
        "return_concentration": {
            "target_top2_share_median": base._clean_number(
                combined.target_top2_positive_return_share.median()
            ),
            "source_top2_share_median": base._clean_number(
                combined.source_top2_positive_return_share.median()
            ),
            "target_top5_share_median": base._clean_number(
                combined.target_top5_positive_return_share.median()
            ),
            "source_top5_share_median": base._clean_number(
                combined.source_top5_positive_return_share.median()
            ),
        },
        "strictness_diagnostics": strictness,
        "cross_instrument_pareto": {
            "candidate_count": int(combined.cross_instrument_pareto.sum()),
            "maximize": [
                "K200 cost-adjusted total return",
                "SImain cost-adjusted total return",
                "SImain cost-adjusted median trade",
            ],
            "minimize": ["SImain trade count", "SImain maximum drawdown"],
            "combined_score": False,
        },
        "parameter_acceptance": "none",
    }
    report["audits"]["gap"]["target_gap_trade_count"] = int(
        pd.to_numeric(target_trades.position_crosses_real_gap, errors="coerce")
        .fillna(0)
        .astype(bool)
        .sum()
    )

    representative = pd.concat(
        [
            pd.read_csv(old_root / "representative_trades.csv", low_memory=False),
            pd.read_csv(strict_root / "representative_trades.csv", low_memory=False),
        ],
        ignore_index=True,
    )
    old_config = json.loads((old_root / "run_config.json").read_text(encoding="utf-8"))
    run_config = {
        **old_config,
        "run_id": AGGREGATE_RUN_ID,
        "mode": "transfer_exact_presentation_union",
        "source_runs": [OLD_RUN_ID, STRICT_RUN_ID],
        "candidate_count": int(len(combined)),
        "cost_bps": FIXED_RESEARCH_COST_BPS,
        "result_semantics": (
            "presentation union of two independently frozen exact-transfer batches; "
            "no target-driven candidate generation"
        ),
    }

    base.atomic_csv(aggregate_root / "migration_comparison.csv", combined)
    base.atomic_csv(aggregate_root / "source_candidate_trades.csv", source_trades)
    base.atomic_csv(aggregate_root / "simain_candidate_trades.csv", target_trades)
    base.atomic_csv(aggregate_root / "representative_trades.csv", representative)
    base.atomic_csv(
        aggregate_root / "cross_instrument_pareto.csv",
        combined.loc[combined.cross_instrument_pareto].copy(),
    )
    _write_aggregate_candidate_freeze(
        aggregate_root,
        candidate_union,
        aggregate_freeze,
    )
    base.atomic_json(aggregate_root / "migration_report.json", report)
    base.atomic_json(
        aggregate_root / "strictness_diagnostics.json", strictness
    )
    base.atomic_json(
        aggregate_root / "posthoc_full_grid_status.json",
        report["posthoc_full_grid"],
    )
    base.atomic_json(aggregate_root / "run_config.json", run_config)

    _set_run(AGGREGATE_RUN_ID)
    manifest = base.build_page()
    manifest["aggregate_transfer_runs"] = [OLD_RUN_ID, STRICT_RUN_ID]
    manifest["strictness_diagnostics"] = base.artifact(
        aggregate_root / "strictness_diagnostics.json"
    )
    manifest["cross_instrument_pareto"] = base.artifact(
        aggregate_root / "cross_instrument_pareto.csv"
    )
    base.atomic_json(aggregate_root / "cross_instrument_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze strict-entry K200 candidates, run exact SImain transfer, "
            "and publish one aggregate comparison/trade-review entry."
        )
    )
    parser.add_argument(
        "command", choices=("freeze", "evaluate", "build", "all")
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command in {"freeze", "all"}:
        payload = freeze_strict_candidates()
        print(
            json.dumps(
                {
                    "phase": "freeze",
                    "run_id": STRICT_RUN_ID,
                    "candidate_count": payload["candidate_count"],
                    "content_sha256": payload["content_sha256"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.command in {"evaluate", "all"}:
        report = evaluate_strict_target(workers=args.workers)
        print(
            json.dumps(
                {"phase": "evaluate", **report["evaluation"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.command in {"build", "all"}:
        manifest = build_aggregate_page()
        print(
            json.dumps(
                {
                    "phase": "build",
                    "run_id": AGGREGATE_RUN_ID,
                    "output": str(base.CROSS_ROOT / "runs" / AGGREGATE_RUN_ID),
                    "manifest": manifest,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
