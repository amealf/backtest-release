from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
VARIANT_ROOT = CODE_DIR.parent
PROJECT_ROOT = VARIANT_ROOT.parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
RUNTIME_ROOT = PROJECT_ROOT / "runtime_inputs"
SOURCE_MANIFEST = VARIANT_ROOT / "SOURCE_MANIFEST.json"
K200_PROFILE_PATH = VARIANT_ROOT / "instrument_profiles" / "k200m.json"
UNION_ROOT = RESULTS_ROOT / "all_completed_union_analysis"
CROSS_ROOT = RESULTS_ROOT / "cross_instrument_comparison"
RUN_ID = "k200_20260526_20260708__simain_20260129_20260223"
RUN_ROOT = CROSS_ROOT / "runs" / RUN_ID
SIMAIN_SOURCE = Path(
    r"D:\Code\data\ibkr\SImain\SImain_15s_20260128_20260223_session_filled.csv"
)
SIMAIN_SOURCE_MANIFEST = Path(
    r"D:\Code\data\ibkr\SImain\SImain_15s_20260128_20260223_manifest.json"
)
SIMAIN_MAIN_SCHEDULE = Path(
    r"D:\Code\LPPL\Data\SImain\SImain_1_min_main_contract_schedule.csv"
)
SOURCE_START = pd.Timestamp("2026-05-26 00:00:00")
SOURCE_END = pd.Timestamp("2026-07-08 23:52:00")
TARGET_START = pd.Timestamp("2026-01-29 00:00:00")
TARGET_END = pd.Timestamp("2026-02-23 23:59:45")
MAX_FROZEN_CANDIDATES = 180
TOP_PER_VIEW = 12
PARAMETER_FIELDS = ("e", "bh", "trw", "k", "w", "m", "speed_window_bars")
EXECUTION_FIELDS = (
    "method",
    "baseline_sampling_policy",
    "entry_fill_mode",
    "entry_execution_policy",
    "entry_slippage",
)
FREEZE_NAME = "frozen_candidates.json"


if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from instrument_contracts import load_instrument_profile  # noqa: E402
from build_v4_4_combined_union_analysis import (  # noqa: E402
    publish_stable_main_assets,
)
from build_v4_4_review_delivery import build_stage_trade_review  # noqa: E402
from run_v4_4_resumable_campaign import (  # noqa: E402
    FINGERPRINT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    result_semantics_id,
)
from v4_4_engine import (  # noqa: E402
    COMBINED_TRADE_AUDIT_SCHEMA_ID,
    COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
    ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    ENTRY_FILL_CALCULATED_THRESHOLD,
    EXIT_MODE_COMBINED,
    REBOUND_BASELINE_POLICY_ID,
    Combo,
    _max_drawdown,
    load_bars,
    simulate_combo,
)


K200_PROFILE = load_instrument_profile(K200_PROFILE_PATH)
K200_SOURCE = Path(str(K200_PROFILE["resolved_market_data_path"]))
K200_PREPARATION = Path(
    str(K200_PROFILE["resolved_preparation_manifest_path"])
)
ROUND_TRIP_COST_BPS = float(
    K200_PROFILE["normalized_cost_model"]["round_trip_total_cost_bps"]
)


_WORKER_FRAME: pd.DataFrame | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def current_snapshot_root() -> Path:
    pointer_path = UNION_ROOT / "current_snapshot.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("status") != "complete":
        raise ValueError("current cumulative pointer is not complete")
    snapshot = Path(str(pointer.get("snapshot_root", "")))
    if not snapshot.is_dir():
        snapshot_id = str(pointer["union_snapshot_id"])
        snapshot = UNION_ROOT / "snapshots" / snapshot_id
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    return snapshot.resolve()


def _clean_number(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _clean_number(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna(False).map(
        lambda value: value is True or str(value).strip().lower() in {"1", "true", "yes"}
    )


def _source_trade_metrics(snapshot: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_path = snapshot / "union_trades.csv"
    columns = [
        "combo_id",
        "entry_index",
        "exit_index",
        "entry_time",
        "exit_time",
        "entry_fill_price",
        "exit_fill_price",
        "gross_return",
        "position_crosses_real_gap",
        "signal_synthetic_empty_bar_count",
        "source_stage_id",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(trade_path, usecols=columns, chunksize=100_000):
        chunk["cost_adjusted_return_fixed_3p57"] = (
            pd.to_numeric(chunk["gross_return"], errors="raise")
            - ROUND_TRIP_COST_BPS / 10_000.0
        )
        chunk["position_crosses_real_gap"] = _truthy(
            chunk["position_crosses_real_gap"]
        )
        chunks.append(chunk)
    trades = pd.concat(chunks, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for combo_id, group in trades.groupby("combo_id", sort=False):
        returns = group["cost_adjusted_return_fixed_3p57"].to_numpy(float)
        gross = group["gross_return"].to_numpy(float)
        non_gap = group.loc[
            ~group["position_crosses_real_gap"],
            "cost_adjusted_return_fixed_3p57",
        ].to_numpy(float)
        positives = np.sort(returns[returns > 0])[::-1]
        positive_sum = float(positives.sum())
        rows.append(
            {
                "combo_id": str(combo_id),
                "source_trade_count": int(len(returns)),
                "source_gross_total_return": float(np.prod(1.0 + gross) - 1.0),
                "source_cost_total_return": float(np.prod(1.0 + returns) - 1.0),
                "source_cost_median_trade": float(np.median(returns)) if len(returns) else math.nan,
                "source_cost_mean_trade": float(np.mean(returns)) if len(returns) else math.nan,
                "source_cost_max_drawdown_abs": abs(float(_max_drawdown(returns))),
                "source_win_rate": float(np.mean(returns > 0)) if len(returns) else math.nan,
                "source_non_gap_cost_total_return": (
                    float(np.prod(1.0 + non_gap) - 1.0) if len(non_gap) else 0.0
                ),
                "source_gap_trade_count": int(group["position_crosses_real_gap"].sum()),
                "source_synthetic_signal_trade_count": int(
                    pd.to_numeric(
                        group["signal_synthetic_empty_bar_count"], errors="coerce"
                    ).fillna(0).gt(0).sum()
                ),
                "source_top2_positive_return_share": (
                    float(positives[:2].sum() / positive_sum)
                    if positive_sum > 0
                    else math.nan
                ),
                "source_top5_positive_return_share": (
                    float(positives[:5].sum() / positive_sum)
                    if positive_sum > 0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows), trades


def _selection_views(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    at_least_10 = frame.loc[frame.source_trade_count.ge(10)]
    at_least_20 = frame.loc[frame.source_trade_count.ge(20)]
    positive_20 = at_least_20.loc[at_least_20.source_cost_total_return.gt(0)]
    scenario_1 = frame.loc[_truthy(frame["scenario_1_qualified"])]
    return [
        (
            "unrestricted_cost_total_ge10",
            at_least_10.sort_values(
                ["source_cost_total_return", "source_cost_max_drawdown_abs", "source_cost_median_trade", "combo_id"],
                ascending=[False, True, False, True],
                kind="mergesort",
            ),
        ),
        (
            "scenario_1_cost_total",
            scenario_1.sort_values(
                ["source_cost_total_return", "source_cost_max_drawdown_abs", "source_cost_median_trade", "combo_id"],
                ascending=[False, True, False, True],
                kind="mergesort",
            ),
        ),
        (
            "cost_mean_trade_ge10",
            at_least_10.sort_values(
                ["source_cost_mean_trade", "source_cost_total_return", "combo_id"],
                ascending=[False, False, True],
                kind="mergesort",
            ),
        ),
        (
            "cost_median_trade_ge10",
            at_least_10.sort_values(
                ["source_cost_median_trade", "source_cost_total_return", "combo_id"],
                ascending=[False, False, True],
                kind="mergesort",
            ),
        ),
        (
            "win_rate_ge20",
            at_least_20.sort_values(
                ["source_win_rate", "source_cost_total_return", "combo_id"],
                ascending=[False, False, True],
                kind="mergesort",
            ),
        ),
        (
            "low_drawdown_cost_positive_ge20",
            positive_20.sort_values(
                ["source_cost_max_drawdown_abs", "source_cost_total_return", "combo_id"],
                ascending=[True, False, True],
                kind="mergesort",
            ),
        ),
        (
            "low_concentration_cost_positive_ge20",
            positive_20.sort_values(
                ["source_top2_positive_return_share", "source_cost_total_return", "combo_id"],
                ascending=[True, False, True],
                kind="mergesort",
            ),
        ),
        (
            "non_gap_cost_total_ge10",
            at_least_10.sort_values(
                ["source_non_gap_cost_total_return", "source_cost_total_return", "combo_id"],
                ascending=[False, False, True],
                kind="mergesort",
            ),
        ),
    ]


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bool(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12))
    return left == right


def _neighbor_ids(seed: pd.Series, population: pd.DataFrame) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for axis in PARAMETER_FIELDS:
        others = [field for field in PARAMETER_FIELDS if field != axis]
        mask = pd.Series(True, index=population.index)
        for field in (*EXECUTION_FIELDS, *others):
            mask &= population[field].map(lambda value, expected=seed[field]: _same_value(value, expected))
        candidates = population.loc[mask].copy()
        if candidates.empty:
            continue
        candidates["_distance"] = (
            pd.to_numeric(candidates[axis], errors="raise") - float(seed[axis])
        ).abs()
        candidates = candidates.loc[candidates["_distance"].gt(0)].sort_values(
            ["_distance", axis, "combo_id"], kind="mergesort"
        )
        if candidates.empty:
            continue
        below = candidates.loc[pd.to_numeric(candidates[axis]) < float(seed[axis])].head(1)
        above = candidates.loc[pd.to_numeric(candidates[axis]) > float(seed[axis])].head(1)
        for row in pd.concat([below, above]).itertuples(index=False):
            output.append((str(row.combo_id), axis))
    return output


def _add_excursions(
    trades: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = trades.copy()
    lows = frame["low"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    trade_counts = frame["trade_count"].to_numpy(float)
    synthetic = frame["is_synthetic_empty_bar"].to_numpy(bool)
    mfe_points: list[float] = []
    mae_points: list[float] = []
    zero_trade_counts: list[int] = []
    synthetic_counts: list[int] = []
    for row in enriched.itertuples(index=False):
        start = max(0, int(row.entry_index))
        end = min(len(frame) - 1, int(row.exit_index))
        if end < start:
            raise ValueError(f"trade index order is invalid for {row.combo_id}")
        entry = float(row.entry_fill_price)
        mfe_points.append(max(0.0, entry - float(np.min(lows[start : end + 1]))))
        mae_points.append(max(0.0, float(np.max(highs[start : end + 1])) - entry))
        zero_trade_counts.append(int(np.sum(trade_counts[start : end + 1] <= 0)))
        synthetic_counts.append(int(np.sum(synthetic[start : end + 1])))
    enriched[f"{prefix}_mfe_points"] = mfe_points
    enriched[f"{prefix}_mae_points"] = mae_points
    entry_prices = pd.to_numeric(enriched["entry_fill_price"], errors="raise")
    enriched[f"{prefix}_mfe_bps"] = enriched[f"{prefix}_mfe_points"] / entry_prices * 10_000.0
    enriched[f"{prefix}_mae_bps"] = enriched[f"{prefix}_mae_points"] / entry_prices * 10_000.0
    enriched[f"{prefix}_gross_points"] = entry_prices - pd.to_numeric(
        enriched["exit_fill_price"], errors="raise"
    )
    enriched[f"{prefix}_cost_adjusted_return"] = (
        pd.to_numeric(enriched["gross_return"], errors="raise")
        - ROUND_TRIP_COST_BPS / 10_000.0
    )
    mfe_fraction = enriched[f"{prefix}_mfe_bps"] / 10_000.0
    enriched[f"{prefix}_mfe_retention"] = np.where(
        mfe_fraction.gt(0),
        enriched[f"{prefix}_cost_adjusted_return"] / mfe_fraction,
        np.nan,
    )
    enriched[f"{prefix}_zero_trade_bar_count_holding"] = zero_trade_counts
    enriched[f"{prefix}_synthetic_bar_count_holding"] = synthetic_counts

    metrics: list[dict[str, Any]] = []
    for combo_id, group in enriched.groupby("combo_id", sort=False):
        returns = group[f"{prefix}_cost_adjusted_return"].to_numpy(float)
        gross = pd.to_numeric(group["gross_return"], errors="raise").to_numpy(float)
        positives = np.sort(returns[returns > 0])[::-1]
        positive_sum = float(positives.sum())
        non_gap = group.loc[
            ~_truthy(group["position_crosses_real_gap"]),
            f"{prefix}_cost_adjusted_return",
        ].to_numpy(float)
        metrics.append(
            {
                "combo_id": str(combo_id),
                f"{prefix}_trade_count": int(len(group)),
                f"{prefix}_gross_total_return": float(np.prod(1.0 + gross) - 1.0),
                f"{prefix}_cost_total_return": float(np.prod(1.0 + returns) - 1.0),
                f"{prefix}_cost_median_trade": float(np.median(returns)) if len(returns) else math.nan,
                f"{prefix}_cost_mean_trade": float(np.mean(returns)) if len(returns) else math.nan,
                f"{prefix}_cost_max_drawdown_abs": abs(float(_max_drawdown(returns))),
                f"{prefix}_win_rate": float(np.mean(returns > 0)) if len(returns) else math.nan,
                f"{prefix}_mfe_bps_median": float(group[f"{prefix}_mfe_bps"].median()),
                f"{prefix}_mae_bps_median": float(group[f"{prefix}_mae_bps"].median()),
                f"{prefix}_mfe_points_median": float(group[f"{prefix}_mfe_points"].median()),
                f"{prefix}_mae_points_median": float(group[f"{prefix}_mae_points"].median()),
                f"{prefix}_gross_points_total": float(group[f"{prefix}_gross_points"].sum()),
                f"{prefix}_mfe_retention_median": float(group[f"{prefix}_mfe_retention"].median()),
                f"{prefix}_top2_positive_return_share": (
                    float(positives[:2].sum() / positive_sum) if positive_sum > 0 else math.nan
                ),
                f"{prefix}_top5_positive_return_share": (
                    float(positives[:5].sum() / positive_sum) if positive_sum > 0 else math.nan
                ),
                f"{prefix}_non_gap_cost_total_return": (
                    float(np.prod(1.0 + non_gap) - 1.0) if len(non_gap) else 0.0
                ),
                f"{prefix}_gap_trade_count": int(_truthy(group["position_crosses_real_gap"]).sum()),
                f"{prefix}_synthetic_signal_trade_count": int(
                    pd.to_numeric(group["signal_synthetic_empty_bar_count"], errors="coerce").fillna(0).gt(0).sum()
                ),
                f"{prefix}_zero_trade_bar_exposure_count": int(
                    group[f"{prefix}_zero_trade_bar_count_holding"].gt(0).sum()
                ),
                f"{prefix}_synthetic_bar_exposure_count": int(
                    group[f"{prefix}_synthetic_bar_count_holding"].gt(0).sum()
                ),
            }
        )
    return enriched, pd.DataFrame(metrics)


def freeze_candidates() -> dict[str, Any]:
    snapshot = current_snapshot_root()
    summary_path = snapshot / "analysis_summary.csv"
    summary = pd.read_csv(summary_path)
    fixed_metrics, source_trades = _source_trade_metrics(snapshot)
    population = summary.merge(fixed_metrics, on="combo_id", how="inner", validate="one_to_one")
    if len(population) != len(summary):
        raise ValueError("source summary and trade metrics do not reconcile one-to-one")

    selection_tags: dict[str, set[str]] = defaultdict(set)
    ordered_seed_ids: list[str] = []
    for view_id, view in _selection_views(population):
        for combo_id in view.head(TOP_PER_VIEW)["combo_id"].astype(str):
            selection_tags[combo_id].add(view_id)
            if combo_id not in ordered_seed_ids:
                ordered_seed_ids.append(combo_id)

    selected_ids = list(ordered_seed_ids)
    for combo_id in ordered_seed_ids:
        if len(selected_ids) >= MAX_FROZEN_CANDIDATES:
            break
        seed = population.loc[population.combo_id.astype(str).eq(combo_id)].iloc[0]
        for neighbor_id, axis in _neighbor_ids(seed, population):
            selection_tags[neighbor_id].add(f"one_axis_neighbor:{combo_id}:{axis}")
            if neighbor_id not in selected_ids:
                selected_ids.append(neighbor_id)
            if len(selected_ids) >= MAX_FROZEN_CANDIDATES:
                break

    selected = population.set_index("combo_id").loc[selected_ids].reset_index()
    selected_source_trades = source_trades.loc[
        source_trades.combo_id.astype(str).isin(selected_ids)
    ].copy()
    source_frame = load_bars(K200_SOURCE, K200_PREPARATION)
    source_enriched, source_excursions = _add_excursions(
        selected_source_trades,
        source_frame,
        prefix="source",
    )
    selected = selected.drop(
        columns=[
            column
            for column in selected.columns
            if column.startswith("source_")
            and column not in {"source_campaign_id", "source_stage_id", "source_stage_root", "source_plan_fingerprint", "source_stage_key"}
        ]
    ).merge(source_excursions, on="combo_id", how="left", validate="one_to_one")
    selected["selection_tags"] = selected.combo_id.map(
        lambda value: "|".join(sorted(selection_tags[str(value)]))
    )
    selected["candidate_order"] = np.arange(1, len(selected) + 1)

    source_manifest_sha = sha256_file(SOURCE_MANIFEST)
    selection_contract = {
        "no_target_results_read": True,
        "top_per_view": TOP_PER_VIEW,
        "maximum_frozen_candidates": MAX_FROZEN_CANDIDATES,
        "views": [view_id for view_id, _ in _selection_views(population)],
        "neighbor_rule": "nearest lower and upper completed value on one axis while every other parameter and execution field is equal",
        "cost_bps": ROUND_TRIP_COST_BPS,
        "minimum_trade_counts": {"general": 10, "win_drawdown_concentration": 20},
        "combined_score": False,
    }
    candidate_columns = [
        "candidate_order",
        "combo_id",
        *EXECUTION_FIELDS,
        *PARAMETER_FIELDS,
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
    candidates = selected[candidate_columns].sort_values("candidate_order")
    payload_without_hash = {
        "schema_version": 1,
        "status": "frozen_before_target_evaluation",
        "generated_at_utc": utc_now(),
        "source_instrument": "K200",
        "source_sample": {
            "start": str(SOURCE_START),
            "end": str(SOURCE_END),
            "timezone": "Asia/Seoul",
        },
        "source_snapshot": {
            "path": str(snapshot),
            "union_snapshot_id": snapshot.name,
            "analysis_summary": artifact(summary_path),
            "union_trades": artifact(snapshot / "union_trades.csv"),
        },
        "source_identity": {
            "source_manifest_path": str(SOURCE_MANIFEST),
            "source_manifest_sha256": source_manifest_sha,
            "implementation_artifacts": {
                "engine": artifact(CODE_DIR / "v4_4_engine.py"),
                "stage_analyzer": artifact(CODE_DIR / "analyze_v4_4_scenario_3_stage.py"),
                "cumulative_builder": artifact(CODE_DIR / "build_v4_4_combined_union_analysis.py"),
                "cross_instrument_builder": artifact(Path(__file__).resolve()),
            },
            "k200_market_data": artifact(K200_SOURCE),
            "k200_preparation_manifest": artifact(K200_PREPARATION),
        },
        "selection_contract": selection_contract,
        "candidate_count": int(len(candidates)),
        "candidates": _records(candidates),
    }
    content_hash = canonical_hash(payload_without_hash)
    payload = {**payload_without_hash, "content_sha256": content_hash}
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(RUN_ROOT / FREEZE_NAME, payload)
    atomic_csv(RUN_ROOT / "frozen_candidates.csv", candidates)
    atomic_csv(RUN_ROOT / "source_candidate_trades.csv", source_enriched)
    return payload


def load_frozen_candidates() -> dict[str, Any]:
    path = RUN_ROOT / FREEZE_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = str(payload.pop("content_sha256"))
    actual = canonical_hash(payload)
    payload["content_sha256"] = expected
    if actual != expected:
        raise ValueError(f"frozen candidate content hash mismatch: {actual} != {expected}")
    if payload.get("status") != "frozen_before_target_evaluation":
        raise ValueError("candidate file is not frozen for target evaluation")
    return payload


def _combo_from_candidate(record: dict[str, Any]) -> Combo:
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


def _worker_initialize(source: str) -> None:
    global _WORKER_FRAME
    _WORKER_FRAME = load_bars(Path(source), None)


def _worker_simulate(record: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if _WORKER_FRAME is None:
        raise RuntimeError("target worker frame is not initialized")
    combo = _combo_from_candidate(record)
    trades = simulate_combo(_WORKER_FRAME, combo, TARGET_START, TARGET_END)
    return combo.combo_id, trades


def _rank_percentile(series: pd.Series, *, ascending: bool = False) -> pd.Series:
    if len(series) <= 1:
        return pd.Series(100.0, index=series.index)
    rank = series.rank(method="min", ascending=ascending)
    return (1.0 - (rank - 1.0) / (len(series) - 1.0)) * 100.0


def _spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    pairs = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(pairs) < 2:
        return math.nan
    left_rank = pairs["left"].rank(method="average")
    right_rank = pairs["right"].rank(method="average")
    return float(left_rank.corr(right_rank, method="pearson"))


def _adjacency(frame: pd.DataFrame) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {str(value): set() for value in frame.combo_id}
    for _, seed in frame.iterrows():
        seed_id = str(seed.combo_id)
        for neighbor_id, _ in _neighbor_ids(seed, frame):
            adjacency[seed_id].add(neighbor_id)
            adjacency.setdefault(neighbor_id, set()).add(seed_id)
    return adjacency


def _stable_components(frame: pd.DataFrame, adjacency: dict[str, set[str]]) -> list[dict[str, Any]]:
    by_id = frame.set_index("combo_id")
    stable_ids = set(frame.loc[frame.target_stable_region, "combo_id"].astype(str))
    seen: set[str] = set()
    components: list[dict[str, Any]] = []
    for start in sorted(stable_ids):
        if start in seen:
            continue
        queue: deque[str] = deque([start])
        members: list[str] = []
        seen.add(start)
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor in stable_ids and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        member_rows = by_id.loc[members]
        components.append(
            {
                "member_count": len(members),
                "combo_ids": sorted(members),
                "best_target_cost_total_return": float(member_rows.target_cost_total_return.max()),
                "median_target_cost_total_return": float(member_rows.target_cost_total_return.median()),
                "parameter_ranges": {
                    field: [
                        _clean_number(member_rows[field].min()),
                        _clean_number(member_rows[field].max()),
                    ]
                    for field in PARAMETER_FIELDS
                },
            }
        )
    return sorted(
        components,
        key=lambda item: (-item["member_count"], -item["best_target_cost_total_return"]),
    )


def _failure_features(frame: pd.DataFrame) -> list[dict[str, Any]]:
    failed = frame.loc[
        frame.source_cost_total_return.gt(0) & frame.target_cost_total_return.le(0)
    ]
    retained = frame.loc[
        frame.source_cost_total_return.gt(0) & frame.target_cost_total_return.gt(0)
    ]
    output: list[dict[str, Any]] = []
    for field in PARAMETER_FIELDS:
        counts = failed[field].value_counts(dropna=False).head(3)
        output.append(
            {
                "field": field,
                "failed_top_values": [
                    {"value": _clean_number(value), "count": int(count)}
                    for value, count in counts.items()
                ],
                "failed_median": _clean_number(pd.to_numeric(failed[field], errors="coerce").median()),
                "retained_median": _clean_number(pd.to_numeric(retained[field], errors="coerce").median()),
            }
        )
    return output


def _representative_trades(
    target_trades: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    leaders = comparison.sort_values(
        ["target_cost_total_return", "target_cost_max_drawdown_abs", "target_cost_median_trade"],
        ascending=[False, True, False],
        kind="mergesort",
    ).head(5)
    rows: list[pd.Series] = []
    for combo_id in leaders.combo_id.astype(str):
        group = target_trades.loc[target_trades.combo_id.astype(str).eq(combo_id)].copy()
        if group.empty:
            continue
        group["target_profit_giveback_bps"] = (
            group["target_mfe_bps"]
            - group["target_cost_adjusted_return"] * 10_000.0
        )
        selections = {
            "best_cost_trade": group["target_cost_adjusted_return"].idxmax(),
            "worst_cost_trade": group["target_cost_adjusted_return"].idxmin(),
            "largest_profit_giveback": group["target_profit_giveback_bps"].idxmax(),
            "largest_mae": group["target_mae_bps"].idxmax(),
        }
        for role, index in selections.items():
            row = group.loc[index].copy()
            row["representative_role"] = role
            rows.append(row)
    if not rows:
        return target_trades.iloc[0:0].copy()
    return pd.DataFrame(rows).drop_duplicates(
        subset=["combo_id", "entry_time", "exit_time", "representative_role"]
    )


def evaluate_target(*, workers: int = 3) -> dict[str, Any]:
    frozen = load_frozen_candidates()
    if not SIMAIN_SOURCE.is_file() or not SIMAIN_SOURCE_MANIFEST.is_file():
        raise FileNotFoundError("SImain warm-up source or its manifest is missing")
    candidates = pd.DataFrame(frozen["candidates"])
    required_warmup_bars = int(
        max(
            candidates.e.max(),
            (candidates.bh + candidates.trw - 1).max(),
            candidates.w.max(),
            candidates.speed_window_bars.max(),
            120,
        )
    )
    target_frame = load_bars(SIMAIN_SOURCE, None)
    prior_rows = int((target_frame.datetime < TARGET_START).sum())
    if prior_rows < required_warmup_bars:
        raise ValueError(
            f"target warm-up has {prior_rows} bars, requires {required_warmup_bars}"
        )
    if target_frame.datetime.min() >= TARGET_START or target_frame.datetime.max() < TARGET_END:
        raise ValueError("target source does not cover warm-up plus the full evaluation range")

    records = candidates.to_dict("records")
    trade_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_initialize,
        initargs=(str(SIMAIN_SOURCE),),
    ) as executor:
        futures = {executor.submit(_worker_simulate, record): record["combo_id"] for record in records}
        for index, future in enumerate(as_completed(futures), start=1):
            combo_id, trades = future.result()
            for trade in trades:
                trade_rows.append({**trade, "combo_id": combo_id})
            print(f"target evaluation {index}/{len(futures)} combo={combo_id} trades={len(trades)}", flush=True)
    target_trades = pd.DataFrame(trade_rows)
    if target_trades.empty:
        raise ValueError("frozen candidates produced no target trades")
    target_trades["gross_return"] = pd.to_numeric(
        target_trades["return"], errors="raise"
    )
    if not pd.to_datetime(target_trades.entry_time).between(TARGET_START, TARGET_END).all():
        raise ValueError("target statistics contain a trade entered outside the evaluation interval")
    target_trades["position_crosses_real_gap"] = _truthy(
        target_trades["position_crosses_real_gap"]
    )
    target_enriched, target_metrics = _add_excursions(
        target_trades,
        target_frame,
        prefix="target",
    )
    comparison = candidates.merge(target_metrics, on="combo_id", how="left", validate="one_to_one")

    comparison["source_rank_percentile"] = _rank_percentile(
        comparison.source_cost_total_return,
        ascending=False,
    )
    comparison["target_rank_percentile"] = _rank_percentile(
        comparison.target_cost_total_return,
        ascending=False,
    )
    comparison["rank_percentile_change"] = (
        comparison.target_rank_percentile - comparison.source_rank_percentile
    )
    comparison["positive_return_consistency"] = np.select(
        [
            comparison.source_cost_total_return.gt(0) & comparison.target_cost_total_return.gt(0),
            comparison.source_cost_total_return.gt(0) & comparison.target_cost_total_return.le(0),
            comparison.source_cost_total_return.le(0) & comparison.target_cost_total_return.gt(0),
        ],
        ["both_positive", "source_positive_target_nonpositive", "source_nonpositive_target_positive"],
        default="both_nonpositive",
    )

    adjacency = _adjacency(comparison)
    source_positive = comparison.set_index("combo_id").source_cost_total_return.gt(0).to_dict()
    target_positive = comparison.set_index("combo_id").target_cost_total_return.gt(0).to_dict()
    neighbor_count: list[int] = []
    source_neighbor_positive_share: list[float] = []
    target_neighbor_positive_share: list[float] = []
    for combo_id in comparison.combo_id.astype(str):
        neighbors = sorted(adjacency.get(combo_id, set()))
        neighbor_count.append(len(neighbors))
        source_neighbor_positive_share.append(
            float(np.mean([source_positive[item] for item in neighbors])) if neighbors else math.nan
        )
        target_neighbor_positive_share.append(
            float(np.mean([target_positive[item] for item in neighbors])) if neighbors else math.nan
        )
    comparison["neighbor_count"] = neighbor_count
    comparison["source_neighbor_positive_share"] = source_neighbor_positive_share
    comparison["target_neighbor_positive_share"] = target_neighbor_positive_share
    comparison["target_stable_region"] = (
        comparison.target_cost_total_return.gt(0)
        & comparison.neighbor_count.ge(2)
        & comparison.target_neighbor_positive_share.ge(0.6)
    )
    comparison["target_isolated_positive"] = (
        comparison.target_cost_total_return.gt(0)
        & (
            comparison.neighbor_count.lt(2)
            | comparison.target_neighbor_positive_share.lt(0.5)
        )
    )
    comparison["source_gap_trade_share"] = np.where(
        comparison.source_trade_count.gt(0),
        comparison.source_gap_trade_count / comparison.source_trade_count,
        np.nan,
    )
    comparison["target_gap_trade_share"] = np.where(
        comparison.target_trade_count.gt(0),
        comparison.target_gap_trade_count / comparison.target_trade_count,
        np.nan,
    )
    comparison["target_low_activity_audit_status"] = (
        "no_bound_simain_exclusion_policy; zero-trade and synthetic exposure audited"
    )

    comparison = comparison.sort_values(
        ["target_cost_total_return", "target_cost_max_drawdown_abs", "target_cost_median_trade", "combo_id"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    comparison.insert(0, "default_rank", np.arange(1, len(comparison) + 1))
    stable_regions = _stable_components(comparison, adjacency)
    isolated = comparison.loc[comparison.target_isolated_positive].sort_values(
        "target_cost_total_return", ascending=False
    )
    rank_correlation = _spearman_correlation(
        comparison.source_cost_total_return,
        comparison.target_cost_total_return,
    )
    positive_fraction = float(comparison.target_cost_total_return.gt(0).mean())
    representative = _representative_trades(target_enriched, comparison)

    schedule = pd.read_csv(SIMAIN_MAIN_SCHEDULE)
    schedule = schedule.loc[
        schedule.trade_date.astype(str).between("2026-01-29", "2026-02-23")
    ]
    rollover_symbols = sorted(schedule.main_local_symbol.dropna().astype(str).unique())
    if rollover_symbols != ["SIH6"]:
        raise ValueError(f"unexpected SImain rollover schedule: {rollover_symbols}")

    report = {
        "schema_version": 1,
        "status": "complete_frozen_candidate_validation",
        "generated_at_utc": utc_now(),
        "candidate_freeze": {
            "path": str((RUN_ROOT / FREEZE_NAME).resolve()),
            "content_sha256": frozen["content_sha256"],
            "file_sha256": sha256_file(RUN_ROOT / FREEZE_NAME),
            "candidate_count": int(len(comparison)),
            "target_results_cannot_modify_candidates": True,
        },
        "evaluation": {
            "source_instrument": "K200",
            "source_interval": [str(SOURCE_START), str(SOURCE_END)],
            "target_instrument": "SImain",
            "target_interval": [str(TARGET_START), str(TARGET_END)],
            "cost_bps": ROUND_TRIP_COST_BPS,
            "target_positive_candidate_fraction": positive_fraction,
            "target_positive_candidate_count": int(comparison.target_cost_total_return.gt(0).sum()),
            "rank_spearman_correlation": rank_correlation,
            "stable_candidate_count": int(comparison.target_stable_region.sum()),
            "isolated_positive_count": int(comparison.target_isolated_positive.sum()),
            "combined_score": False,
            "default_sort": [
                {"field": "target_cost_total_return", "direction": "desc"},
                {"field": "target_cost_max_drawdown_abs", "direction": "asc"},
                {"field": "target_cost_median_trade", "direction": "desc"},
            ],
        },
        "stable_parameter_regions": stable_regions,
        "isolated_parameter_points": isolated.combo_id.astype(str).head(30).tolist(),
        "failed_parameter_common_features": _failure_features(comparison),
        "return_concentration": {
            "target_top2_share_median": _clean_number(comparison.target_top2_positive_return_share.median()),
            "source_top2_share_median": _clean_number(comparison.source_top2_positive_return_share.median()),
            "target_top5_share_median": _clean_number(comparison.target_top5_positive_return_share.median()),
            "source_top5_share_median": _clean_number(comparison.source_top5_positive_return_share.median()),
        },
        "audits": {
            "timezone": {
                "source": "Asia/Seoul",
                "target": "America/Chicago",
                "comparison_basis": "local wall-clock sample boundaries; returns and durations normalized to percentages, bps, and minutes",
            },
            "trading_session": {
                "target": "COMEX 17:00 previous day through 16:00 session day, with 16:00-17:00 maintenance break",
                "target_session_count": int(target_frame.loc[target_frame.datetime.between(TARGET_START, TARGET_END), "session_date"].nunique()),
            },
            "continuous_contract_roll": {
                "schedule": artifact(SIMAIN_MAIN_SCHEDULE),
                "symbols": rollover_symbols,
                "roll_count": 0,
                "price_adjustment": "none; explicit main contract SIH6",
            },
            "gap": {
                "trade_field": "position_crosses_real_gap",
                "target_gap_trade_count": int(target_enriched.position_crosses_real_gap.sum()),
            },
            "synthetic": {
                "source_file_synthetic_rows": int(target_frame.is_synthetic_empty_bar.sum()),
                "target_signal_trade_count": int(pd.to_numeric(target_enriched.signal_synthetic_empty_bar_count, errors="coerce").fillna(0).gt(0).sum()),
            },
            "low_activity": {
                "status": "audit_only_no_bound_simain_low_activity_exclusion_policy",
                "zero_trade_rows": int(target_frame.loc[target_frame.datetime.between(TARGET_START, TARGET_END), "trade_count"].le(0).sum()),
                "candidate_trade_exposure_count": int(target_enriched.target_zero_trade_bar_count_holding.gt(0).sum()),
            },
            "warmup": {
                "required_bars": required_warmup_bars,
                "available_bars": prior_rows,
                "available_start": str(target_frame.datetime.min()),
                "statistics_start": str(TARGET_START),
                "statistics_end": str(TARGET_END),
                "only_test_interval_entries_counted": True,
            },
        },
        "representative_trade_count": int(len(representative)),
        "posthoc_full_grid": {
            "status": "not_run",
            "separate_from_frozen_validation": True,
            "may_not_modify_frozen_candidates": True,
        },
        "parameter_acceptance": "none",
    }

    atomic_csv(RUN_ROOT / "migration_comparison.csv", comparison)
    atomic_csv(RUN_ROOT / "simain_candidate_trades.csv", target_enriched)
    atomic_csv(RUN_ROOT / "representative_trades.csv", representative)
    atomic_json(RUN_ROOT / "migration_report.json", report)
    atomic_json(RUN_ROOT / "posthoc_full_grid_status.json", report["posthoc_full_grid"])
    run_config = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "mode": "transfer_exact",
        "source": {
            "instrument": "K200",
            "sample_start": str(SOURCE_START),
            "sample_end": str(SOURCE_END),
            "market_data": artifact(K200_SOURCE),
            "preparation_manifest": artifact(K200_PREPARATION),
            "snapshot_root": str(current_snapshot_root()),
        },
        "target": {
            "instrument": "SImain",
            "sample_start": str(TARGET_START),
            "sample_end": str(TARGET_END),
            "warmup_source": artifact(SIMAIN_SOURCE),
            "source_manifest": artifact(SIMAIN_SOURCE_MANIFEST),
            "main_contract_schedule": artifact(SIMAIN_MAIN_SCHEDULE),
        },
        "candidate_freeze": artifact(RUN_ROOT / FREEZE_NAME),
        "candidate_content_sha256": frozen["content_sha256"],
        "cost_bps": ROUND_TRIP_COST_BPS,
        "source_instrument_profile": artifact(K200_PROFILE_PATH),
        "workers": workers,
        "engine": artifact(CODE_DIR / "v4_4_engine.py"),
        "result_semantics": "target evaluation only; no candidate generation from target results",
        "posthoc_full_grid": "not_run_separate_optional_diagnostic",
    }
    atomic_json(RUN_ROOT / "run_config.json", run_config)
    return report


def _css_from_current_main() -> str:
    html = (current_snapshot_root() / "index.html").read_text(encoding="utf-8")
    match = re.search(r"<style>(.*?)</style>", html, flags=re.DOTALL)
    if match is None:
        raise ValueError("current cumulative main lacks reusable CSS")
    return match.group(1)


def build_cross_trade_review(comparison: pd.DataFrame) -> dict[str, Any]:
    trades_path = RUN_ROOT / "simain_candidate_trades.csv"
    trades = pd.read_csv(trades_path, low_memory=False)
    if trades.empty or trades["combo_id"].astype(str).nunique() != len(comparison):
        raise ValueError("SImain trade review requires trades for every frozen candidate")
    trades["batch_id"] = "simain_frozen_candidate_validation"
    trades["cost_adjusted_return"] = pd.to_numeric(
        trades["target_cost_adjusted_return"], errors="raise"
    )
    trades["round_trip_cost_bps"] = ROUND_TRIP_COST_BPS
    trades["cost_model_id"] = "v4_4_fixed_3p57_bps_cross_instrument_transfer"
    trades["holding_minutes"] = (
        pd.to_numeric(trades["exit_index"], errors="raise")
        - pd.to_numeric(trades["entry_index"], errors="raise")
    ) * 0.25

    grouped = trades.groupby("combo_id", sort=False)
    trade_means = grouped["return"].mean()
    exit_counts = (
        trades.assign(value=1)
        .pivot_table(
            index="combo_id",
            columns="exit_reason",
            values="value",
            aggfunc="sum",
            fill_value=0,
        )
    )
    summary = comparison.copy()
    summary["speed_exit_enabled"] = True
    summary["rebound_exit_enabled"] = True
    summary["train_trade_count"] = summary["target_trade_count"].astype(int)
    summary["train_return"] = summary["target_gross_total_return"]
    summary["train_cost_adjusted_return"] = summary["target_cost_total_return"]
    summary["train_return_excluding_gap_spanning_trades"] = summary[
        "target_non_gap_cost_total_return"
    ]
    summary["train_avg_trade"] = summary["combo_id"].map(trade_means)
    summary["train_cost_adjusted_avg_trade"] = summary["target_cost_mean_trade"]
    summary["train_max_drawdown_abs"] = summary["target_cost_max_drawdown_abs"]
    summary["train_cost_adjusted_max_drawdown"] = -summary[
        "target_cost_max_drawdown_abs"
    ]
    summary["train_cost_adjusted_max_drawdown_abs"] = summary[
        "target_cost_max_drawdown_abs"
    ]
    summary["round_trip_cost_bps"] = ROUND_TRIP_COST_BPS
    summary["cost_model_id"] = "v4_4_fixed_3p57_bps_cross_instrument_transfer"
    summary["estimated_total_commission_krw"] = None
    summary["estimated_total_slippage_krw"] = None
    summary["estimated_total_cost_krw"] = None
    summary["gap_spanning_trade_count"] = summary["target_gap_trade_count"].astype(int)
    summary["synthetic_signal_trade_count"] = summary[
        "target_synthetic_signal_trade_count"
    ].astype(int)
    summary["segment_end_exit_count"] = summary["combo_id"].map(
        exit_counts.get("segment_end", pd.Series(dtype=int))
    ).fillna(0).astype(int)
    summary["rebound_exit_count"] = summary["combo_id"].map(
        exit_counts.get("rebound_threshold", pd.Series(dtype=int))
    ).fillna(0).astype(int)
    summary["speed_exit_count"] = summary["combo_id"].map(
        exit_counts.get("downside_speed_below_threshold", pd.Series(dtype=int))
    ).fillna(0).astype(int)
    for field in ("event_01_qualified", "event_02_qualified", "short_drop_3_15m_member"):
        summary[field] = False

    preparation_root = RUN_ROOT / "simain_trade_review_source_contract"
    filter_atoms_path = preparation_root / "filter_atoms.csv"
    filter_events_path = preparation_root / "filter_events.json"
    preparation_manifest_path = preparation_root / "data_preparation_manifest.json"
    atomic_csv(
        filter_atoms_path,
        pd.DataFrame(columns=["datetime", "baseline_excluded"]),
    )
    atomic_json(
        filter_events_path,
        {
            "schema_version": 1,
            "status": "complete",
            "events": [],
            "policy": "audit_only_no_bound_simain_low_activity_exclusion_policy",
        },
    )
    source_sha = sha256_file(SIMAIN_SOURCE)
    prepared_identity = (
        "v4_4_policy_neutral_baseline_marker_simain_audit_only_"
        + source_sha[:16]
    )
    atomic_json(
        preparation_manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "prepared_identity": prepared_identity,
            "source": str(SIMAIN_SOURCE),
            "source_sha256": source_sha,
            "low_activity_policy": (
                "audit_only_no_bound_simain_low_activity_exclusion_policy"
            ),
            "artifacts": {
                "filter_atoms": artifact(filter_atoms_path),
                "filter_events": artifact(filter_events_path),
            },
        },
    )

    strategy = str(trades["strategy_id"].iloc[0])
    semantics = result_semantics_id(
        ENTRY_FILL_CALCULATED_THRESHOLD,
        ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        0.0,
        EXIT_MODE_COMBINED,
        "all_window",
    )
    stage_manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "version_label": "V4.4",
        "campaign_id": "v4_4_cross_instrument_comparison",
        "stage_id": RUN_ID,
        "source": str(SIMAIN_SOURCE),
        "source_sha256": source_sha,
        "data_preparation_manifest": str(preparation_manifest_path),
        "prepared_identity": prepared_identity,
        "data_preparation_manifest_sha256": sha256_file(preparation_manifest_path),
        "train_start": "2026-01-28 00:00:00",
        "train_end": str(TARGET_END),
        "engine_sha256": sha256_file(CODE_DIR / "v4_4_engine.py"),
        "plan_fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "baseline_sampling_policy": "all_window",
        "baseline_sampling_policies": ["all_window"],
        "baseline_filter_id": "simain_audit_only_no_exclusion",
        "result_semantics_id": semantics,
        "result_semantics_ids_by_baseline_sampling_policy": {
            "all_window": semantics
        },
    }
    completion_manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "complete",
        "campaign_id": "v4_4_cross_instrument_comparison",
        "stage_id": RUN_ID,
        "coordinate_count": int(len(summary)),
        "trade_count": int(len(trades)),
        "strategy_id": strategy,
        "strategy_ids_by_baseline_sampling_policy": {"all_window": strategy},
        "result_semantics_id": semantics,
        "exit_mode": EXIT_MODE_COMBINED,
        "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
    }
    run_config = json.loads((RUN_ROOT / "run_config.json").read_text(encoding="utf-8"))
    incremental_parent_run = run_config.get("incremental_parent_run")
    reuse_trade_review = (
        CROSS_ROOT / "runs" / str(incremental_parent_run) / "trade_review"
        if incremental_parent_run
        else None
    )
    review = build_stage_trade_review(
        RUN_ROOT / "trade_review",
        summary,
        trades,
        stage_manifest,
        completion_manifest,
        analysis_identity=sha256_file(RUN_ROOT / "migration_report.json"),
        manifest_href="../cross_instrument_manifest.json",
        main_href="../index.html",
        research_contract_label="SImain 冻结候选迁移验证",
        default_start_date="2026-01-29",
        instrument_label="SImain",
        workers=4,
        reuse_trade_review=reuse_trade_review,
    )
    return {
        "index": artifact(review["index"]),
        "manifest": artifact(review["manifest"]),
        "catalog": artifact(review["catalog"]),
        "process_payload": artifact(review["process_payload"]),
        "chunk_count": len(review["chunks"]),
        "reused_chunk_count": int(review.get("reused_chunk_count", 0)),
        "generated_chunk_count": int(review.get("generated_chunk_count", len(review["chunks"]))),
        "trade_count": int(len(trades)),
        "source_contract": artifact(preparation_manifest_path),
    }


def build_k200_trade_review(
    comparison: pd.DataFrame,
    *,
    role: str,
) -> dict[str, Any]:
    settings = {
        "source": {
            "trades": "source_candidate_trades.csv",
            "output": "trade_review_k200_train",
            "prefix": "source",
            "cost_return": "source_cost_adjusted_return",
            "label": "K200（训）冻结候选训练区间",
            "instrument": "K200（训）",
            "start": "2026-05-26 00:00:00",
            "end": "2026-07-08 23:52:00",
            "peer_href": "../trade_review_k200_test/index.html",
            "peer_label": "显示测试集",
            "peer_research_contract_id": "v4_4_cross_instrument_comparison",
        },
        "source_test": {
            "trades": "k200_test_candidate_trades.csv",
            "output": "trade_review_k200_test",
            "prefix": "source_test",
            "cost_return": "source_test_cost_adjusted_return",
            "label": "K200（测）后续行情重放",
            "instrument": "K200（测）",
            "start": "2026-07-08 23:52:15",
            "end": "2026-08-07 03:21:45",
            "peer_href": "../../../../all_completed_union_analysis/trade_review/index.html",
            "peer_label": "显示训练集",
            "peer_research_contract_id": "v4_4_all_completed_combined_union",
        },
    }
    config = settings[role]
    trades = pd.read_csv(RUN_ROOT / config["trades"], low_memory=False)
    if trades.empty or trades["combo_id"].astype(str).nunique() != len(comparison):
        raise ValueError(f"{config['instrument']} trade review requires every candidate")
    trades["batch_id"] = f"{role}_frozen_candidate_review"
    trades["return"] = pd.to_numeric(trades["gross_return"], errors="raise")
    trades["cost_adjusted_return"] = pd.to_numeric(
        trades[config["cost_return"]], errors="raise"
    )
    trades["round_trip_cost_bps"] = ROUND_TRIP_COST_BPS
    trades["cost_model_id"] = "v4_4_fixed_3p57_bps_k200_train_test"
    trades["trade_audit_schema_version"] = COMBINED_TRADE_AUDIT_SCHEMA_VERSION
    trades["trade_audit_schema_id"] = COMBINED_TRADE_AUDIT_SCHEMA_ID
    trades["rebound_baseline_policy_id"] = REBOUND_BASELINE_POLICY_ID
    trades["holding_minutes"] = (
        pd.to_numeric(trades["exit_index"], errors="raise")
        - pd.to_numeric(trades["entry_index"], errors="raise")
    ) * 0.25

    prefix = str(config["prefix"])
    trade_means = trades.groupby("combo_id", sort=False)["return"].mean()
    exit_counts = (
        trades.assign(value=1)
        .pivot_table(
            index="combo_id",
            columns="exit_reason",
            values="value",
            aggfunc="sum",
            fill_value=0,
        )
    )
    summary = comparison.copy()
    summary["speed_exit_enabled"] = True
    summary["rebound_exit_enabled"] = True
    summary["train_trade_count"] = summary[f"{prefix}_trade_count"].astype(int)
    summary["train_return"] = summary[f"{prefix}_gross_total_return"]
    summary["train_cost_adjusted_return"] = summary[f"{prefix}_cost_total_return"]
    summary["train_return_excluding_gap_spanning_trades"] = summary[
        f"{prefix}_non_gap_cost_total_return"
    ]
    summary["train_avg_trade"] = summary["combo_id"].map(trade_means)
    summary["train_cost_adjusted_avg_trade"] = summary[f"{prefix}_cost_mean_trade"]
    summary["train_max_drawdown_abs"] = summary[f"{prefix}_cost_max_drawdown_abs"]
    summary["train_cost_adjusted_max_drawdown"] = -summary[
        f"{prefix}_cost_max_drawdown_abs"
    ]
    summary["train_cost_adjusted_max_drawdown_abs"] = summary[
        f"{prefix}_cost_max_drawdown_abs"
    ]
    summary["round_trip_cost_bps"] = ROUND_TRIP_COST_BPS
    summary["cost_model_id"] = "v4_4_fixed_3p57_bps_k200_train_test"
    summary["estimated_total_commission_krw"] = None
    summary["estimated_total_slippage_krw"] = None
    summary["estimated_total_cost_krw"] = None
    summary["gap_spanning_trade_count"] = summary[f"{prefix}_gap_trade_count"].astype(int)
    summary["synthetic_signal_trade_count"] = summary[
        f"{prefix}_synthetic_signal_trade_count"
    ].astype(int)
    summary["segment_end_exit_count"] = summary["combo_id"].map(
        exit_counts.get("segment_end", pd.Series(dtype=int))
    ).fillna(0).astype(int)
    summary["rebound_exit_count"] = summary["combo_id"].map(
        exit_counts.get("rebound_threshold", pd.Series(dtype=int))
    ).fillna(0).astype(int)
    summary["speed_exit_count"] = summary["combo_id"].map(
        exit_counts.get("downside_speed_below_threshold", pd.Series(dtype=int))
    ).fillna(0).astype(int)
    for field in ("event_01_qualified", "event_02_qualified", "short_drop_3_15m_member"):
        summary[field] = False

    preparation = json.loads(K200_PREPARATION.read_text(encoding="utf-8"))
    source_sha = sha256_file(K200_SOURCE)
    strategy = str(trades["strategy_id"].iloc[0])
    semantics = result_semantics_id(
        ENTRY_FILL_CALCULATED_THRESHOLD,
        ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        0.0,
        EXIT_MODE_COMBINED,
        "all_window",
    )
    stage_manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "version_label": "V4.4",
        "campaign_id": "v4_4_cross_instrument_comparison",
        "stage_id": f"{RUN_ID}_{role}",
        "source": str(K200_SOURCE),
        "source_sha256": source_sha,
        "data_preparation_manifest": str(K200_PREPARATION),
        "prepared_identity": str(preparation["prepared_identity"]),
        "data_preparation_manifest_sha256": sha256_file(K200_PREPARATION),
        "train_start": config["start"],
        "train_end": config["end"],
        "engine_sha256": sha256_file(CODE_DIR / "v4_4_engine.py"),
        "plan_fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "baseline_sampling_policy": "all_window",
        "baseline_sampling_policies": ["all_window"],
        "baseline_filter_id": "all_window_market_no_baseline_exclusion_v4_4",
        "result_semantics_id": semantics,
        "result_semantics_ids_by_baseline_sampling_policy": {"all_window": semantics},
    }
    completion_manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "complete",
        "campaign_id": "v4_4_cross_instrument_comparison",
        "stage_id": f"{RUN_ID}_{role}",
        "coordinate_count": int(len(summary)),
        "trade_count": int(len(trades)),
        "strategy_id": strategy,
        "strategy_ids_by_baseline_sampling_policy": {"all_window": strategy},
        "result_semantics_id": semantics,
        "exit_mode": EXIT_MODE_COMBINED,
        "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
    }
    output = RUN_ROOT / str(config["output"])
    reuse_trade_review = output if (output / "trade_review_manifest.json").is_file() else None
    review = build_stage_trade_review(
        output,
        summary,
        trades,
        stage_manifest,
        completion_manifest,
        analysis_identity=sha256_file(RUN_ROOT / "migration_report.json"),
        manifest_href="../cross_instrument_manifest.json",
        main_href="../index.html",
        research_contract_label=str(config["label"]),
        default_start_date=str(config["start"])[:10],
        instrument_label=str(config["instrument"]),
        peer_review_href=str(config["peer_href"]),
        peer_review_label=str(config["peer_label"]),
        peer_research_contract_id=str(config["peer_research_contract_id"]),
        workers=4,
        reuse_trade_review=reuse_trade_review,
    )
    return {
        "index": artifact(review["index"]),
        "manifest": artifact(review["manifest"]),
        "catalog": artifact(review["catalog"]),
        "process_payload": artifact(review["process_payload"]),
        "chunk_count": len(review["chunks"]),
        "reused_chunk_count": int(review.get("reused_chunk_count", 0)),
        "generated_chunk_count": int(review.get("generated_chunk_count", len(review["chunks"]))),
        "trade_count": int(len(trades)),
        "source_contract": artifact(K200_PREPARATION),
    }


def _gross_view_metrics_from_trade_records(path: Path, *, prefix: str) -> pd.DataFrame:
    trades = pd.read_csv(path, usecols=["combo_id", "gross_return"])
    trades["gross_return"] = pd.to_numeric(trades["gross_return"], errors="raise")
    rows: list[dict[str, Any]] = []
    for combo_id, group in trades.groupby("combo_id", sort=False):
        gross = group["gross_return"].to_numpy(float)
        rows.append(
            {
                "combo_id": str(combo_id),
                f"{prefix}_gross_total_return": float(np.prod(1.0 + gross) - 1.0),
                f"{prefix}_gross_median_trade": float(np.median(gross)),
                f"{prefix}_gross_max_drawdown_abs": abs(float(_max_drawdown(gross))),
                f"{prefix}_gross_win_rate": float(np.mean(gross > 0)),
            }
        )
    return pd.DataFrame(rows)


def _add_gross_view_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    adjacency = _adjacency(output)
    source_positive = output.set_index("combo_id").source_gross_total_return.gt(0).to_dict()
    target_positive = output.set_index("combo_id").target_gross_total_return.gt(0).to_dict()
    source_shares: list[float] = []
    target_shares: list[float] = []
    for combo_id in output.combo_id.astype(str):
        neighbors = sorted(adjacency.get(combo_id, set()))
        source_shares.append(
            float(np.mean([source_positive[item] for item in neighbors])) if neighbors else math.nan
        )
        target_shares.append(
            float(np.mean([target_positive[item] for item in neighbors])) if neighbors else math.nan
        )
    output["source_gross_rank_percentile"] = _rank_percentile(
        output.source_gross_total_return, ascending=False
    )
    output["target_gross_rank_percentile"] = _rank_percentile(
        output.target_gross_total_return, ascending=False
    )
    output["gross_rank_percentile_change"] = (
        output.target_gross_rank_percentile - output.source_gross_rank_percentile
    )
    output["gross_positive_return_consistency"] = np.select(
        [
            output.source_gross_total_return.gt(0) & output.target_gross_total_return.gt(0),
            output.source_gross_total_return.gt(0) & output.target_gross_total_return.le(0),
            output.source_gross_total_return.le(0) & output.target_gross_total_return.gt(0),
        ],
        ["both_positive", "source_positive_target_nonpositive", "source_nonpositive_target_positive"],
        default="both_nonpositive",
    )
    output["source_gross_neighbor_positive_share"] = source_shares
    output["target_gross_neighbor_positive_share"] = target_shares
    output["target_gross_stable_region"] = (
        output.target_gross_total_return.gt(0)
        & output.neighbor_count.ge(2)
        & output.target_gross_neighbor_positive_share.ge(0.6)
    )
    output["target_gross_isolated_positive"] = (
        output.target_gross_total_return.gt(0)
        & (
            output.neighbor_count.lt(2)
            | output.target_gross_neighbor_positive_share.lt(0.5)
        )
    )
    return output


def _comparison_html() -> str:
    base_css = _css_from_current_main()
    extra_css = r"""
.shell{max-width:1580px;padding:22px 20px 48px}.page-head{align-items:flex-start;margin-bottom:14px}.page-title{max-width:74ch}.page-title h1{font-size:23px;letter-spacing:-.02em;text-wrap:balance}.page-title p{max-width:72ch;line-height:1.55;text-wrap:pretty}.scope-strip{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 14px;padding:11px 14px;border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--muted)}.scope-strip strong{color:var(--ink);font-weight:740}.scope-strip span+span::before{content:"·";margin-right:10px;color:var(--strong)}.comparison-selector{display:grid;grid-template-columns:repeat(4,minmax(170px,1fr)) auto;gap:10px;align-items:end;margin:0 0 14px;padding:14px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}.comparison-selector label,.filter-builder label{display:flex;flex-direction:column;gap:6px;color:var(--muted);font-size:12px;font-weight:700}.comparison-selector select,.filter-builder select,.filter-builder input{min-height:39px;padding:8px 10px;border:1px solid var(--strong);border-radius:7px;background:var(--panel);color:var(--ink);font:inherit}.comparison-selector button,.filter-builder button,.filter-chip button{min-height:39px;padding:8px 13px;border:1px solid var(--strong);border-radius:7px;background:var(--panel);color:var(--accent);font-weight:740}.comparison-selector button:hover,.filter-builder button:hover,.filter-chip button:hover{border-color:var(--accent);background:var(--soft)}.comparison-selector :focus-visible,.filter-builder :focus-visible,.filter-chip :focus-visible{outline:3px solid var(--focus);outline-offset:2px}.selection-status{grid-column:1/-1;margin:0;color:var(--muted);font-size:12px}.summary-band{display:flex;gap:0;align-items:stretch;flex-wrap:wrap;margin:0 0 18px;border-block:1px solid var(--line);background:transparent}.summary-item{min-width:190px;padding:13px 22px 13px 0;margin-right:22px}.summary-item span{display:block;color:var(--muted);font-size:12px}.summary-item strong{display:block;margin:2px 0;font-size:18px;letter-spacing:-.01em}.table-toolbar{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin:22px 0 8px}.table-toolbar h2,.section-head h2{margin:0;font-size:18px;letter-spacing:-.01em}.table-toolbar p,.section-head p{margin:4px 0 0;color:var(--muted);font-size:12px;line-height:1.55}.table-note{max-width:58ch;text-align:right;color:var(--muted);font-size:12px}.filter-panel{margin:0 0 10px;padding:12px 14px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}.filter-builder{display:grid;grid-template-columns:minmax(190px,1.3fr) minmax(130px,.7fr) minmax(180px,1fr) auto auto;gap:8px;align-items:end}.global-filter{grid-column:1/-1}.active-filters{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}.filter-chip{display:inline-flex;align-items:center;gap:6px;padding:5px 7px 5px 10px;border:1px solid var(--line);border-radius:999px;background:var(--subtle);color:var(--ink);font-size:12px}.filter-chip button{min-height:26px;padding:2px 7px;border-radius:999px}.result-count{margin:8px 0 0;color:var(--muted);font-size:12px}.table-wrap.comparison{max-height:70vh;border-radius:10px}.table-wrap.comparison table{font-variant-numeric:tabular-nums}.table-wrap.comparison th button{font-weight:760}.table-wrap.comparison th:first-child,.table-wrap.comparison td:first-child{position:sticky;left:0;z-index:2;background:var(--panel);text-align:left}.table-wrap.comparison th:first-child{z-index:3}.rank-link{min-width:92px;padding:7px 12px;font-size:13px}.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:24px 0 8px}.report-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(260px,.85fr);gap:10px}.report-panel{padding:14px 16px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}.report-panel:first-child{grid-row:span 2}.report-panel h3{margin:0 0 8px;font-size:15px}.report-panel p,.report-panel li{color:var(--muted);line-height:1.55}.report-panel ul{margin:0;padding-left:18px}.notice{padding:12px 14px;border:1px solid var(--strong);border-radius:9px;background:var(--subtle);line-height:1.55}.contract-panel{margin-top:22px}@media(max-width:1100px){.comparison-selector{grid-template-columns:repeat(2,minmax(170px,1fr))}.comparison-selector>button{justify-self:start}.filter-builder{grid-template-columns:repeat(2,minmax(0,1fr))}.filter-builder>button{justify-self:start}}@media(max-width:900px){.report-grid{grid-template-columns:1fr}.report-panel:first-child{grid-row:auto}.table-note{text-align:left}}@media(max-width:620px){.shell{padding:12px 8px 30px}.page-head{align-items:stretch}.theme{align-self:flex-start}.table-toolbar,.section-head{align-items:flex-start;flex-direction:column;gap:4px}.summary-item{min-width:50%;padding:10px 12px 10px 0;margin:0}.scope-strip{align-items:flex-start;flex-direction:column;gap:3px}.scope-strip span+span::before{content:"";margin:0}.comparison-selector,.filter-builder{grid-template-columns:1fr}.page-title h1{font-size:21px}}
    """
    extra_css += r"""
.return-view-bar{display:flex;align-items:center;justify-content:space-between;gap:18px;margin:0 0 14px;padding:10px 2px;border-block:1px solid var(--line)}.return-view-copy{display:flex;flex-direction:column;gap:2px}.return-view-copy strong{font-size:14px}.return-view-copy span{color:var(--muted);font-size:12px;line-height:1.45}.return-view-bar .segmented{flex-wrap:nowrap}.return-view-bar .segmented button{min-width:132px}.header-info{margin-left:4px;color:var(--accent);font-size:12px}@media(max-width:620px){.return-view-bar{align-items:flex-start;flex-direction:column;gap:6px}.return-view-bar .segmented{width:100%}.return-view-bar .segmented button{min-width:0;flex:1 1 0}}
.page-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.main-entry-link{display:inline-flex;align-items:center;justify-content:center;min-height:36px;padding:7px 11px;border:1px solid var(--strong);border-radius:7px;background:var(--panel);color:var(--accent);font-weight:720;text-decoration:none}.main-entry-link:hover{border-color:var(--accent);background:var(--soft)}.main-entry-link:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
"""
    extra_css += r"""
.table-wrap.comparison th,.table-wrap.comparison td,.table-wrap.comparison th:first-child,.table-wrap.comparison td:first-child{text-align:center}
"""
    extra_css += r"""
.comparison-selector{grid-template-columns:repeat(4,minmax(0,1fr)) auto;align-items:end;gap:12px;padding:18px 20px 16px}.comparison-selector .comparison-heading{grid-column:1/-1;margin:0 0 2px;color:var(--ink);font-size:18px;line-height:1.25;font-weight:760;letter-spacing:-.01em}.comparison-selector label{min-width:0}.comparison-selector select{width:100%;min-height:42px}.comparison-selector>#open-run{min-height:42px;white-space:nowrap}.comparison-selector .selection-status{padding-top:1px}
@media(max-width:1100px){.comparison-selector{grid-template-columns:repeat(2,minmax(0,1fr))}.comparison-selector>#open-run{justify-self:start}}
@media(max-width:620px){.comparison-selector{grid-template-columns:1fr;padding:14px}.comparison-selector .comparison-heading{font-size:17px}.comparison-selector>#open-run{width:100%;justify-self:stretch}}
"""
    return fr"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V4.41 跨品种对比</title>
<style>{base_css}{extra_css}</style></head><body><main class="shell">
<header class="page-head"><div class="page-title"><h1>V4.41 跨品种迁移验证</h1><p>用原品种证据冻结候选，再以被比较品种作独立评价。目标结果不能生成或修改候选；点击排名可打开该参数的逐笔证据。</p></div><div class="page-actions"><a class="main-entry-link" href="../../../all_completed_union_analysis/index.html" target="_blank" rel="noopener">主品种入口</a><button class="theme" id="theme" type="button">深色 Dark</button></div></header>
<section class="comparison-selector" aria-labelledby="comparison-selector-title"><h2 id="comparison-selector-title" class="comparison-heading">比较范围</h2><label>原品种<select id="source-instrument"></select></label><label>原样本区间<select id="source-interval"></select></label><label>被比较品种<select id="target-instrument"></select></label><label>被比较区间<select id="target-interval"></select></label><button id="open-run" type="button">加载比较</button><p id="selection-status" class="selection-status" aria-live="polite"></p></section>
<section id="scope-strip" class="scope-strip" aria-label="固定比较范围"></section>
<section class="return-view-bar" aria-labelledby="return-view-title"><div class="return-view-copy"><strong id="return-view-title">排名与收益口径</strong><span id="return-view-description"></span></div><div id="return-view-controls" class="segmented" role="group" aria-label="排名与收益口径"></div></section>
<section id="summary-band" class="summary-band" aria-label="迁移摘要"></section>
<section class="table-toolbar"><div><h2>冻结候选排名</h2><p id="ranking-help"></p></div><div id="table-note" class="table-note"></div></section>
<section class="filter-panel" aria-label="全字段过滤"><div class="filter-builder"><label class="global-filter">全文过滤<input id="global-filter" type="search" placeholder="搜索任意参数、指标或迁移诊断" autocomplete="off"></label><label>字段<select id="filter-field"></select></label><label>条件<select id="filter-operator"></select></label><label>值<input id="filter-value" type="text" inputmode="decimal" placeholder="输入过滤值"></label><button id="add-filter" type="button">添加条件</button><button id="clear-filters" type="button">清除全部</button></div><div id="active-filters" class="active-filters" aria-live="polite"></div><p id="result-count" class="result-count"></p></section>
<div id="comparison-table" class="table-wrap comparison"></div>
<section class="section-head"><div><h2>迁移分析（固定成本后审计）</h2><p>稳定区域、孤立点、失效特征和代表性交易均来自冻结候选验证。</p></div></section><div id="report-grid" class="report-grid"></div>
<section class="section-head"><div><h2>事后全网格诊断</h2><p>与冻结候选验证分离，不能改写候选。</p></div></section><div id="posthoc" class="notice"></div>
<section class="contract-panel"><div class="contract-head"><h2>证据与审计合同</h2></div><div id="contract-table" class="table-wrap"></div></section>
</main><script src="comparison_data.js"></script><script>
const DATA=window.V4_4_CROSS_INSTRUMENT_DATA;
(()=>{{'use strict';if(!DATA){{document.body.textContent='跨品种资料未加载';return;}}
const $=id=>document.getElementById(id);const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const pct=value=>value==null||!Number.isFinite(Number(value))?'—':(Number(value)*100).toFixed(3)+'%';const bps=value=>value==null||!Number.isFinite(Number(value))?'—':Number(value).toFixed(2);const num=value=>value==null||!Number.isFinite(Number(value))?'—':Number(value).toLocaleString(undefined,{{maximumFractionDigits:4}});const yes=value=>value===true?'是':value===false?'否':String(value??'—');
const returnViews=[['cost_adjusted','手续费／滑点后'],['gross','无手续费／滑点']];const runCatalog=DATA.runCatalog||[];
const migration=DATA.migrationPlan||{{}},sourcePlan=migration.source||{{}},sourceTestPlan=migration.source_test||{{}},targetPlan=migration.target||{{}};
const sourceName=sourcePlan.display_name||sourcePlan.instrument_id||'迁移前品种',sourceTestName=sourceTestPlan.display_name||sourceTestPlan.instrument_id||'K200（测）',targetName=targetPlan.display_name||targetPlan.instrument_id||'迁移后品种';
const dateLabel=value=>String(value||'').slice(0,10);const sourceRange=`${{dateLabel(sourcePlan.sample_start)}}—${{dateLabel(sourcePlan.sample_end)}}`,sourceTestRange=`${{dateLabel(sourceTestPlan.sample_start)}}—${{dateLabel(sourceTestPlan.sample_end)}}`,targetRange=`${{dateLabel(targetPlan.sample_start)}}—${{dateLabel(targetPlan.sample_end)}}`;
const sourceReturnHelp=`${{sourceName}} 回测区间为 ${{sourceRange}}；请结合区间长度理解收益差异。`,sourceTestReturnHelp=`${{sourceTestName}} 区间为 ${{sourceTestRange}}；其中从该区间结果选出的新增候选属于事后描述。`;
let rows=[...DATA.rows],returnView='cost_adjusted',sortKey=null,sortDir=0,globalQuery='',filterRules=[];
function modeFields(prefix){{const gross=returnView==='gross';return{{total:`${{prefix}}_${{gross?'gross':'cost'}}_total_return`,median:`${{prefix}}_${{gross?'gross':'cost'}}_median_trade`,drawdown:`${{prefix}}_${{gross?'gross':'cost'}}_max_drawdown_abs`,win:gross?`${{prefix}}_gross_win_rate`:`${{prefix}}_win_rate`}};}}
function diagnosticFields(){{const gross=returnView==='gross';return{{consistency:gross?'gross_positive_return_consistency':'positive_return_consistency',rankChange:gross?'gross_rank_percentile_change':'rank_percentile_change',targetNeighbor:gross?'target_gross_neighbor_positive_share':'target_neighbor_positive_share',sourceNeighbor:gross?'source_gross_neighbor_positive_share':'source_neighbor_positive_share',stable:gross?'target_gross_stable_region':'target_stable_region',isolated:gross?'target_gross_isolated_positive':'target_isolated_positive',sourceRank:gross?'source_gross_rank_percentile':'source_rank_percentile',targetRank:gross?'target_gross_rank_percentile':'target_rank_percentile'}};}}
function columns(){{const source=modeFields('source'),sourceTest=modeFields('source_test'),target=modeFields('target'),diagnostic=diagnosticFields(),hasSourceTest=rows.some(row=>row.source_test_cost_total_return!=null||row.source_test_gross_total_return!=null);return[
['view_rank','排名','parameter','rank',null,'target'],['e','E','parameter','number'],['bh','BH','parameter','number'],['trw','TRW','parameter','number'],['k','K','parameter','number'],['w','W','parameter','number'],['m','M','parameter','number'],['speed_window_bars','S','parameter','number'],
[source.total,`${{sourceName}} 总收益`,'source','percent',sourceReturnHelp,'source'],...(hasSourceTest?[[sourceTest.total,`${{sourceTestName}} 总收益`,'source_test','percent',sourceTestReturnHelp,'source_test']]:[]),[target.total,`${{targetName}} 总收益`,'target','percent',null,'target'],
[target.median,`${{targetName}} 中位单笔`,'target','percent'],[target.drawdown,`${{targetName}} 最大回撤`,'target','percent'],['target_trade_count',`${{targetName}} 交易数`,'target','number'],[target.win,`${{targetName}} 胜率`,'target','percent'],['target_mfe_bps_median',`${{targetName}} MFE 中位 bps`,'target','bps'],['target_mae_bps_median',`${{targetName}} MAE 中位 bps`,'target','bps'],['target_mfe_points_median',`${{targetName}} MFE 中位点数`,'target','number'],['target_mae_points_median',`${{targetName}} MAE 中位点数`,'target','number'],['target_mfe_retention_median',`${{targetName}} MFE 保留率`,'target','percent'],['target_top2_positive_return_share',`${{targetName}} 收益集中度 Top2`,'target','percent'],['target_gross_points_total',`${{targetName}} 原始点数合计`,'target','number'],
[source.median,`${{sourceName}} 中位单笔`,'source','percent'],[source.drawdown,`${{sourceName}} 最大回撤`,'source','percent'],['source_trade_count',`${{sourceName}} 交易数`,'source','number'],['source_entry_threshold_median',`${{sourceName}} 实际开仓阈值中位数`,'source','number'],[source.win,`${{sourceName}} 胜率`,'source','percent'],['source_mfe_bps_median',`${{sourceName}} MFE 中位 bps`,'source','bps'],['source_mae_bps_median',`${{sourceName}} MAE 中位 bps`,'source','bps'],['source_mfe_points_median',`${{sourceName}} MFE 中位点数`,'source','number'],['source_mae_points_median',`${{sourceName}} MAE 中位点数`,'source','number'],['source_mfe_retention_median',`${{sourceName}} MFE 保留率`,'source','percent'],['source_top2_positive_return_share',`${{sourceName}} 收益集中度 Top2`,'source','percent'],['source_gross_points_total',`${{sourceName}} 原始点数合计`,'source','number'],
[diagnostic.consistency,'正收益一致性','diagnostic','text'],[diagnostic.rankChange,'排名百分位变化','diagnostic','number'],['cross_instrument_pareto',hasSourceTest?'三组收益 Pareto':'跨品种 Pareto','diagnostic','boolean'],['neighbor_count','相邻候选数','diagnostic','number'],[diagnostic.targetNeighbor,`${{targetName}} 邻居正收益率`,'diagnostic','percent'],[diagnostic.sourceNeighbor,`${{sourceName}} 邻居正收益率`,'diagnostic','percent'],[diagnostic.stable,'稳定区域','diagnostic','boolean'],[diagnostic.isolated,'孤立正收益点','diagnostic','boolean'],['target_gap_trade_share',`${{targetName}} 跨 gap 交易占比`,'diagnostic','percent'],['source_gap_trade_share',`${{sourceName}} 跨 gap 交易占比`,'diagnostic','percent']];}}
function defaultOrderedRows(){{const target=modeFields('target');return [...rows].sort((a,b)=>Number(b[target.total])-Number(a[target.total])||Number(a[target.drawdown])-Number(b[target.drawdown])||Number(b[target.median])-Number(a[target.median])||String(a.combo_id).localeCompare(String(b.combo_id)));}}
function viewRankMap(){{return new Map(defaultOrderedRows().map((row,index)=>[String(row.combo_id),index+1]));}}
function tradeRoute(row,role){{const roots={{source:DATA.artifacts.sourceTradeReview,source_test:DATA.artifacts.sourceTestTradeReview,target:DATA.artifacts.targetTradeReview||DATA.artifacts.tradeReview}},contracts={{source:'v4_4_all_completed_combined_union',source_test:'v4_4_cross_instrument_comparison',target:'v4_4_cross_instrument_comparison'}},root=roots[role]||roots.target,contract=contracts[role]||contracts.target;return `${{root}}?combo_id=${{encodeURIComponent(row.combo_id)}}&research_contract_id=${{contract}}`;}}
function metricLink(row,role,label){{return `<a class="rank-link" target="_blank" rel="noopener" href="${{tradeRoute(row,role)}}">${{label}}</a>`;}}
function format(row,key,type,ranks,routeRole){{const value=key==='view_rank'?ranks.get(String(row.combo_id)):row[key];if(type==='rank')return metricLink(row,routeRole,`#${{num(value)}}`);if(type==='percent')return routeRole?metricLink(row,routeRole,pct(value)):pct(value);if(type==='bps')return bps(value);if(type==='number')return num(value);if(type==='boolean')return `<span class="${{value?'yes':'no'}}">${{yes(value)}}</span>`;return esc(value);}}
function matchesRule(row,rule){{const type=columns().find(item=>item[0]===rule.key)?.[3]||'text',value=rule.key==='view_rank'?viewRankMap().get(String(row.combo_id)):row[rule.key];if(['number','percent','bps','rank'].includes(type)){{const actual=Number(value),expected=Number(rule.value);if(!Number.isFinite(actual)||!Number.isFinite(expected))return false;return rule.op==='gt'?actual>expected:rule.op==='gte'?actual>=expected:rule.op==='lt'?actual<expected:rule.op==='lte'?actual<=expected:actual===expected;}}const actual=String(value??'').toLocaleLowerCase(),expected=String(rule.value??'').toLocaleLowerCase();return rule.op==='eq'?actual===expected:actual.includes(expected);}}
function filteredRows(){{const cols=columns(),ranks=viewRankMap();return rows.filter(row=>{{if(globalQuery){{const haystack=cols.map(([key])=>String(key==='view_rank'?ranks.get(String(row.combo_id)):row[key]??'')).join(' ').toLocaleLowerCase();if(!haystack.includes(globalQuery))return false;}}return filterRules.every(rule=>matchesRule(row,rule));}});}}
function sortedRows(){{if(sortKey==null)return filteredRows().sort((a,b)=>viewRankMap().get(String(a.combo_id))-viewRankMap().get(String(b.combo_id)));const ranks=viewRankMap();return filteredRows().sort((a,b)=>{{const av=sortKey==='view_rank'?ranks.get(String(a.combo_id)):a[sortKey],bv=sortKey==='view_rank'?ranks.get(String(b.combo_id)):b[sortKey];if(av==null&&bv==null)return String(a.combo_id).localeCompare(String(b.combo_id));if(av==null)return 1;if(bv==null)return -1;const delta=(typeof av==='number'&&typeof bv==='number')?av-bv:String(av).localeCompare(String(bv));return delta?sortDir*delta:String(a.combo_id).localeCompare(String(b.combo_id));}});}}
function sortIndicator(key){{if(sortKey==null)return key==='view_rank'?' ▲':'';return sortKey===key?(sortDir===1?' ▲':' ▼'):'';}}
function changeSort(key){{if(sortKey!==key){{sortKey=key;sortDir=1;}}else if(sortDir===1){{sortDir=-1;}}else{{sortKey=null;sortDir=0;}}renderTable();}}
function renderTable(){{const visible=sortedRows(),cols=columns(),ranks=viewRankMap(),viewLabel=returnViews.find(item=>item[0]===returnView)?.[1]||returnView;$('table-note').textContent=`当前口径：${{viewLabel}}；默认按 ${{targetName}} 总收益降序 → 最大回撤升序 → 中位单笔降序`;$('result-count').textContent=`显示 ${{num(visible.length)}} / ${{num(rows.length)}} 个冻结候选`;$('comparison-table').innerHTML='<table><thead><tr>'+cols.map(([key,label,, ,tooltip])=>{{const attrs=tooltip?` title="${{esc(tooltip)}}" aria-label="${{esc(label+'：'+tooltip)}}"`:'';const info=tooltip?'<span class="header-info" aria-hidden="true">ⓘ</span>':'';return `<th><button type="button" data-sort="${{key}}"${{attrs}}>${{esc(label)}}${{info}}${{sortIndicator(key)}}</button></th>`;}}).join('')+'</tr></thead><tbody>'+visible.map(row=>'<tr>'+cols.map(([key,, ,type,,routeRole])=>`<td>${{format(row,key,type,ranks,routeRole)}}</td>`).join('')+'</tr>').join('')+'</tbody></table>';document.querySelectorAll('[data-sort]').forEach(node=>node.onclick=()=>changeSort(node.dataset.sort));}}
function unique(values){{return [...new Set(values.filter(Boolean))];}}function optionHtml(values,current){{return values.map(value=>`<option value="${{esc(value)}}"${{value===current?' selected':''}}>${{esc(value)}}</option>`).join('');}}
function renderSelectors(){{const current=runCatalog.find(item=>item.run_id===DATA.runId)||runCatalog[0];if(!current)return;$('source-instrument').innerHTML=optionHtml(unique(runCatalog.map(item=>item.source_instrument)),current.source_instrument);$('source-interval').innerHTML=optionHtml(unique(runCatalog.map(item=>item.source_interval_label)),current.source_interval_label);$('target-instrument').innerHTML=optionHtml(unique(runCatalog.map(item=>item.target_instrument)),current.target_instrument);$('target-interval').innerHTML=optionHtml(unique(runCatalog.map(item=>item.target_interval_label)),current.target_interval_label);$('selection-status').textContent=`当前冻结验证：${{current.source_instrument}} ${{current.source_interval_label}} → ${{current.target_instrument}} ${{current.target_interval_label}}`;$('open-run').onclick=()=>{{const matches=runCatalog.filter(item=>item.source_instrument===$('source-instrument').value&&item.source_interval_label===$('source-interval').value&&item.target_instrument===$('target-instrument').value&&item.target_interval_label===$('target-interval').value),match=matches.find(item=>item.run_id===DATA.runId)||matches[0];if(!match){{$('selection-status').textContent='当前目录没有这一组合的冻结候选验证结果。';return;}}if(match.run_id===DATA.runId){{$('selection-status').textContent='当前页面已经是所选比较范围。';return;}}location.href=match.href;}};}}
function operatorOptions(type){{return ['number','percent','bps','rank'].includes(type)?[['eq','等于'],['gt','大于'],['gte','大于等于'],['lt','小于'],['lte','小于等于']]:[['contains','包含'],['eq','等于']];}}function renderOperatorOptions(){{const type=columns().find(item=>item[0]===$('filter-field').value)?.[3]||'text';$('filter-operator').innerHTML=operatorOptions(type).map(([value,label])=>`<option value="${{value}}">${{label}}</option>`).join('');}}
function refreshFilterFields(){{$('filter-field').innerHTML=columns().map(([key,label])=>`<option value="${{key}}">${{esc(label)}}</option>`).join('');renderOperatorOptions();}}
function renderFilterChips(){{$('active-filters').innerHTML=filterRules.map((rule,index)=>{{const item=columns().find(column=>column[0]===rule.key),label=item?.[1]||rule.key,opLabel=operatorOptions(item?.[3]||'text').find(option=>option[0]===rule.op)?.[1]||rule.op;return `<span class="filter-chip">${{esc(label)}} ${{esc(opLabel)}} ${{esc(rule.value)}}<button type="button" data-remove-filter="${{index}}" aria-label="移除 ${{esc(label)}} 过滤">×</button></span>`;}}).join('');document.querySelectorAll('[data-remove-filter]').forEach(node=>node.onclick=()=>{{filterRules.splice(Number(node.dataset.removeFilter),1);renderFilterChips();renderTable();}});}}
function bindFilters(){{refreshFilterFields();$('filter-field').onchange=renderOperatorOptions;$('global-filter').oninput=event=>{{globalQuery=event.target.value.trim().toLocaleLowerCase();renderTable();}};$('add-filter').onclick=()=>{{const value=$('filter-value').value.trim();if(!value)return;filterRules.push({{key:$('filter-field').value,op:$('filter-operator').value,value}});$('filter-value').value='';renderFilterChips();renderTable();}};$('filter-value').onkeydown=event=>{{if(event.key==='Enter')$('add-filter').click();}};$('clear-filters').onclick=()=>{{globalQuery='';filterRules=[];$('global-filter').value='';$('filter-value').value='';renderFilterChips();renderTable();}};}}
function renderReturnViews(){{const triple=rows.some(row=>row.source_test_cost_total_return!=null||row.source_test_gross_total_return!=null);$('return-view-description').textContent=triple?`同步切换 ${{sourceName}}／${{sourceTestName}}／${{targetName}} 的总收益；其余审计字段继续使用对应口径。`:`同步切换 ${{sourceName}}／${{targetName}} 的总收益、中位单笔、最大回撤和胜率；默认采用手续费／滑点后。`;$('ranking-help').textContent=`点击同一列第三次恢复默认排序；蓝色排名打开 ${{targetName}} 逐笔分析，三列蓝色总收益分别打开对应品种与区间的逐笔分析。`;$('return-view-controls').innerHTML=returnViews.map(([value,label])=>`<button type="button" data-return-view="${{value}}" class="${{returnView===value?'active':''}}" aria-pressed="${{returnView===value}}">${{esc(label)}}</button>`).join('');document.querySelectorAll('[data-return-view]').forEach(node=>node.onclick=()=>{{returnView=node.dataset.returnView;sortKey=null;sortDir=0;globalQuery='';filterRules=[];$('global-filter').value='';refreshFilterFields();renderFilterChips();renderView();}});}}
function renderScope(){{const testPart=rows.some(row=>row.source_test_cost_total_return!=null)?`<span>${{esc(sourceTestName)}}：${{esc(sourceTestRange)}}</span>`:'';$('scope-strip').innerHTML=`<strong>${{esc(sourceName)}} → ${{esc(targetName)}} 冻结候选验证</strong><span>${{esc(sourceName)}}：${{esc(sourceRange)}}</span>${{testPart}}<span>${{esc(targetName)}}：${{esc(targetRange)}}</span><span>候选 ${{num(DATA.rows.length)}} 个</span><span>研究成本 ${{num(DATA.report.evaluation.cost_bps)}} bps</span>`;}}
function correlation(left,right){{const pairs=left.map((value,index)=>[Number(value),Number(right[index])]).filter(([a,b])=>Number.isFinite(a)&&Number.isFinite(b));if(pairs.length<2)return NaN;const meanA=pairs.reduce((sum,[a])=>sum+a,0)/pairs.length,meanB=pairs.reduce((sum,[,b])=>sum+b,0)/pairs.length;let numerator=0,sumA=0,sumB=0;for(const [a,b] of pairs){{const da=a-meanA,db=b-meanB;numerator+=da*db;sumA+=da*da;sumB+=db*db;}}return sumA&&sumB?numerator/Math.sqrt(sumA*sumB):NaN;}}
function renderSummary(){{const target=modeFields('target'),diagnostic=diagnosticFields(),positive=rows.filter(row=>Number(row[target.total])>0).length,rankCorrelation=correlation(rows.map(row=>row[diagnostic.sourceRank]),rows.map(row=>row[diagnostic.targetRank])),stable=rows.filter(row=>row[diagnostic.stable]).length,isolated=rows.filter(row=>row[diagnostic.isolated]).length,viewLabel=returnViews.find(item=>item[0]===returnView)?.[1]||returnView;$('summary-band').innerHTML=[[`${{targetName}} 正收益候选`,`${{num(positive)}} / ${{num(rows.length)}}`,`${{viewLabel}} · ${{pct(positive/rows.length)}}`],[`${{sourceName}}／${{targetName}} 排名相关性`,num(rankCorrelation),`${{viewLabel}} · Spearman`],['稳定候选',num(stable),`${{viewLabel}} · 相邻参数共同为正`],['孤立正收益点',num(isolated),`${{viewLabel}} · 需谨慎解释`]].map(([label,value,detail])=>`<div class="summary-item"><span>${{esc(label)}}</span><strong>${{esc(value)}}</strong><span>${{esc(detail)}}</span></div>`).join('');}}
function renderReport(){{const report=DATA.report,regions=report.stable_parameter_regions||[],failures=report.failed_parameter_common_features||[];$('report-grid').innerHTML=`<article class="report-panel"><h3>稳定参数区域</h3>${{regions.length?`<ul>${{regions.slice(0,6).map(item=>`<li>${{item.member_count}} 个候选；最佳 ${{pct(item.best_target_cost_total_return)}}；中位 ${{pct(item.median_target_cost_total_return)}}</li>`).join('')}}</ul>`:'<p>没有满足当前邻域定义的稳定区域。</p>'}}</article><article class="report-panel"><h3>失效参数共同特征</h3><ul>${{failures.slice(0,7).map(item=>`<li>${{esc(item.field)}}：失效中位 ${{num(item.failed_median)}}；保留中位 ${{num(item.retained_median)}}</li>`).join('')}}</ul></article><article class="report-panel"><h3>收益集中度</h3><p>${{esc(targetName)}} Top2 正收益贡献中位：${{pct(report.return_concentration.target_top2_share_median)}}；${{esc(sourceName)}}：${{pct(report.return_concentration.source_top2_share_median)}}。</p><p><a class="nav-link" href="representative_trades.csv">代表性交易 CSV</a></p></article>`;$('posthoc').textContent=report.posthoc_full_grid.status==='not_run'?`${{targetName}} 全参数网格尚未运行。若以后运行，它只作为事后诊断，输出与冻结候选验证分离，也不能修改 frozen_candidates.json。`:report.posthoc_full_grid.status;}}
function renderContract(){{const a=DATA.report.audits,viewLabel=returnViews.find(item=>item[0]===returnView)?.[1]||returnView;$('contract-table').innerHTML='<table><tbody>'+[['候选冻结',`内容 SHA ${{DATA.report.candidate_freeze.content_sha256}}；${{targetName}} 结果不能修改候选。`],['预热',`需要 ${{num(a.warmup.required_bars)}} 根，实际 ${{num(a.warmup.available_bars)}} 根；仅统计测试区间开仓交易。`],['时区',`${{sourceName}} ${{sourcePlan.timezone||'—'}}；${{targetName}} ${{targetPlan.timezone||'—'}}；比较使用百分比、bps 与分钟。`],['交易时段',a.trading_session.target],['连续合约',`SIH6；区间换月 0 次；不调价。`],['gap',`按 position_crosses_real_gap 审计；${{targetName}} 跨 gap 交易 ${{num(a.gap.target_gap_trade_count)}} 笔。`],['synthetic',`源文件 synthetic ${{num(a.synthetic.source_file_synthetic_rows)}} 根；信号相关交易 ${{num(a.synthetic.target_signal_trade_count)}} 笔。`],['低活动',`${{targetName}} 尚无绑定的专用品种排除政策；只审计零成交和 synthetic 暴露。`],['排序',`当前表格采用${{viewLabel}}，默认按 ${{targetName}} 总收益降序、最大回撤升序、中位单笔降序；同一列第三次点击恢复默认排序。`],['综合 score','未创建。'],['参数接受','none']].map(([label,value])=>`<tr><th>${{esc(label)}}</th><td>${{esc(value)}}</td></tr>`).join('')+'</tbody></table>';}}
function renderView(){{renderReturnViews();renderSummary();renderContract();renderTable();}}
renderSelectors();bindFilters();renderFilterChips();renderScope();renderReport();renderView();function applyTheme(dark){{document.documentElement.dataset.theme=dark?'dark':'';localStorage.setItem('v4-unified-theme',dark?'dark':'light');$('theme').textContent=dark?'浅色 Light':'深色 Dark';}}applyTheme(localStorage.getItem('v4-unified-theme')==='dark');$('theme').onclick=()=>applyTheme(document.documentElement.dataset.theme!=='dark');
}})();</script></body></html>"""


def _run_catalog(
    cross_root: Path,
    current_run_root: Path,
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    current_resolved = current_run_root.resolve()
    for run_dir in sorted((cross_root / "runs").glob("*")):
        config_path = run_dir / "run_config.json"
        page_path = run_dir / "index.html"
        manifest_path = run_dir / "cross_instrument_manifest.json"
        if not run_dir.is_dir() or not config_path.exists():
            continue
        is_current = run_dir.resolve() == current_resolved
        if not is_current and (not page_path.exists() or not manifest_path.exists()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source = dict(config.get("source") or {})
        target = dict(config.get("target") or {})
        source_start = str(source.get("sample_start") or "")[:10]
        source_end = str(source.get("sample_end") or "")[:10]
        target_start = str(target.get("sample_start") or "")[:10]
        target_end = str(target.get("sample_end") or "")[:10]
        catalog.append(
            {
                "run_id": str(config.get("run_id") or run_dir.name),
                "source_instrument": str(source.get("instrument") or ""),
                "source_interval_label": f"{source_start}—{source_end}",
                "target_instrument": str(target.get("instrument") or ""),
                "target_interval_label": f"{target_start}—{target_end}",
                "href": f"../{run_dir.name}/index.html",
            }
        )
    return catalog


def _migration_presentation_plan(run_root: Path) -> tuple[dict[str, Any], Path | None]:
    config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    plan_value = config.get("migration_plan")
    if plan_value:
        plan_path = Path(str(plan_value)).resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        return plan, plan_path
    source = dict(config.get("source") or {})
    target = dict(config.get("target") or {})
    return {
        "schema_version": 0,
        "status": "historical_run_without_migration_plan",
        "source": {
            "instrument_id": str(source.get("instrument") or "source"),
            "display_name": str(source.get("instrument") or "迁移前品种"),
            "sample_start": str(source.get("sample_start") or ""),
            "sample_end": str(source.get("sample_end") or ""),
        },
        "target": {
            "instrument_id": str(target.get("instrument") or "target"),
            "display_name": str(target.get("instrument") or "迁移后品种"),
            "sample_start": str(target.get("sample_start") or ""),
            "sample_end": str(target.get("sample_end") or ""),
        },
    }, None


def build_page() -> dict[str, Any]:
    report = json.loads((RUN_ROOT / "migration_report.json").read_text(encoding="utf-8"))
    migration_plan, migration_plan_path = _migration_presentation_plan(RUN_ROOT)
    comparison = pd.read_csv(RUN_ROOT / "migration_comparison.csv")
    source_gross = _gross_view_metrics_from_trade_records(
        RUN_ROOT / "source_candidate_trades.csv", prefix="source"
    )
    target_gross = _gross_view_metrics_from_trade_records(
        RUN_ROOT / "simain_candidate_trades.csv", prefix="target"
    )
    for prefix, metrics in (("source", source_gross), ("target", target_gross)):
        total_key = f"{prefix}_gross_total_return"
        expected = comparison.set_index("combo_id")[total_key].sort_index()
        actual = metrics.set_index("combo_id")[total_key].sort_index()
        if not expected.index.equals(actual.index) or not np.allclose(
            expected.to_numpy(float), actual.to_numpy(float), rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{prefix} gross-return records do not reconcile with migration CSV")
        comparison = comparison.drop(columns=[total_key]).merge(
            metrics, on="combo_id", how="left", validate="one_to_one"
        )
    comparison = _add_gross_view_diagnostics(comparison)
    source_test_trade_review = build_k200_trade_review(comparison, role="source_test")
    trade_review = build_cross_trade_review(comparison)
    run_catalog = _run_catalog(CROSS_ROOT, RUN_ROOT)
    data = {
        "schemaVersion": 1,
        "runId": RUN_ID,
        "runCatalog": run_catalog,
        "migrationPlan": migration_plan,
        "rows": _records(comparison),
        "report": report,
        "artifacts": {
            "frozenCandidates": "frozen_candidates.json",
            "comparisonCsv": "migration_comparison.csv",
            "targetTradesCsv": "simain_candidate_trades.csv",
            "representativeTradesCsv": "representative_trades.csv",
            "runConfig": "run_config.json",
            "tradeReview": "trade_review/index.html",
            "sourceTradeReview": "../../../all_completed_union_analysis/trade_review/index.html",
            "sourceTestTradeReview": "trade_review_k200_test/index.html",
            "targetTradeReview": "trade_review/index.html",
        },
    }
    if (RUN_ROOT / "k200_test_candidate_trades.csv").is_file():
        data["artifacts"]["sourceTestTradesCsv"] = "k200_test_candidate_trades.csv"
    if (RUN_ROOT / "k200_test_metrics.csv").is_file():
        data["artifacts"]["sourceTestMetricsCsv"] = "k200_test_metrics.csv"
    if (RUN_ROOT / "FINAL_TRAIN_TEST_SI_REPORT.md").is_file():
        data["artifacts"]["finalTrainTestSiReport"] = "FINAL_TRAIN_TEST_SI_REPORT.md"
    atomic_text(
        RUN_ROOT / "comparison_data.js",
        "window.V4_4_CROSS_INSTRUMENT_DATA="
        + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
    )
    atomic_text(RUN_ROOT / "index.html", _comparison_html())
    atomic_text(
        CROSS_ROOT / "index.html",
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<meta http-equiv=\"refresh\" content=\"0; url=runs/{RUN_ID}/index.html\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>V4.41 跨品种对比</title></head><body>"
        f"<p><a href=\"runs/{RUN_ID}/index.html\">打开跨品种对比</a></p></body></html>",
    )
    navigation_audit = publish_current_main_standalone_view()
    outputs = {
        name: artifact(RUN_ROOT / name)
        for name in (
            "index.html",
            "comparison_data.js",
            "frozen_candidates.json",
            "frozen_candidates.csv",
            "migration_comparison.csv",
            "migration_report.json",
            "simain_candidate_trades.csv",
            "source_candidate_trades.csv",
            "representative_trades.csv",
            "run_config.json",
            "posthoc_full_grid_status.json",
        )
    }
    for report_name in ("MIGRATION_REPORT.en.md", "MIGRATION_REPORT.zh.md"):
        report_path = RUN_ROOT / report_name
        if report_path.is_file():
            outputs[report_name] = artifact(report_path)
    for extra_name in (
        "k200_test_candidate_trades.csv",
        "k200_test_metrics.csv",
        "FINAL_TRAIN_TEST_SI_REPORT.md",
    ):
        extra_path = RUN_ROOT / extra_name
        if extra_path.is_file():
            outputs[extra_name] = artifact(extra_path)
    run_config = json.loads((RUN_ROOT / "run_config.json").read_text(encoding="utf-8"))
    build_manifest = {
        "schema_version": 1,
        "status": "complete",
        "generated_at_utc": utc_now(),
        "run_id": RUN_ID,
        "presentation_source_identity": artifact(SOURCE_MANIFEST),
        "migration_plan": (
            artifact(migration_plan_path) if migration_plan_path is not None else None
        ),
        "outputs": outputs,
        "stable_entry": artifact(CROSS_ROOT / "index.html"),
        "existing_result_data_modified": False,
        "existing_main_navigation_added": True,
        "existing_main_cross_instrument_switch_removed": False,
        "navigation_audit": navigation_audit,
        "source_test_trade_review": source_test_trade_review,
        "trade_review": trade_review,
        "incremental_parent_run": run_config.get("incremental_parent_run"),
        "parameter_acceptance": "none",
    }
    atomic_json(RUN_ROOT / "cross_instrument_manifest.json", build_manifest)
    return build_manifest


def publish_current_main_standalone_view() -> dict[str, Any]:
    snapshot = current_snapshot_root()
    stable_index = UNION_ROOT / "index.html"
    snapshot_index = snapshot / "index.html"
    snapshot_analysis_data = snapshot / "analysis_data.js"
    snapshot_union_trades = snapshot / "union_trades.csv"
    snapshot_manifest = json.loads(
        (snapshot / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    declared_union_trades = dict(snapshot_manifest["artifacts"]["union_trades"])
    if Path(declared_union_trades["path"]).resolve() != snapshot_union_trades.resolve():
        raise RuntimeError("snapshot manifest union-trades path disagrees with current snapshot")
    if snapshot_union_trades.stat().st_size != int(declared_union_trades["size_bytes"]):
        raise RuntimeError("snapshot union-trades size disagrees with immutable manifest")
    before = {
        "stable_index": artifact(stable_index),
        "snapshot_index": artifact(snapshot_index),
        "snapshot_analysis_data": artifact(snapshot_analysis_data),
        "snapshot_union_trades": declared_union_trades,
    }
    stable_main = publish_stable_main_assets(UNION_ROOT, snapshot, snapshot.name)
    relative_main = "main/index.html"
    encoded_target = json.dumps(relative_main, ensure_ascii=False)
    standalone_html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V4.41 K200回测结果排序</title></head><body><script>const target={encoded_target};location.replace(target+location.search+location.hash);</script></body></html>"""
    atomic_text(stable_index, standalone_html)
    after = {
        "stable_index": artifact(stable_index),
        "snapshot_index": artifact(snapshot_index),
        "snapshot_analysis_data": artifact(snapshot_analysis_data),
        "snapshot_union_trades": declared_union_trades,
    }
    if before["snapshot_index"] != after["snapshot_index"]:
        raise RuntimeError("historical snapshot index changed while publishing navigation")
    if before["snapshot_analysis_data"] != after["snapshot_analysis_data"]:
        raise RuntimeError("historical snapshot analysis data changed while publishing navigation")
    if before["snapshot_union_trades"] != after["snapshot_union_trades"]:
        raise RuntimeError("historical snapshot trades changed while publishing navigation")
    return {
        "stable_entry": str(stable_index),
        "snapshot": str(snapshot),
        "before": before,
        "after": after,
        "historical_snapshot_unchanged": True,
        "analysis_data_unchanged": True,
        "union_trades_unchanged": True,
        "stable_main_mode": "redirect_to_stable_main",
        "stable_main": stable_main,
        "cross_instrument_switch_present": True,
    }


def main() -> int:
    global RUN_ID, RUN_ROOT
    parser = argparse.ArgumentParser(
        description="Freeze K200 candidates, evaluate them on SImain, and build the V4.4 cross-instrument page."
    )
    parser.add_argument(
        "command",
        choices=("freeze", "evaluate", "build", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--run-id",
        help=(
            "Build an existing migration run. The run_config incremental_parent_run "
            "field enables chunk reuse from the prior delivered run."
        ),
    )
    args = parser.parse_args()
    if args.run_id:
        if args.command != "build":
            raise ValueError("--run-id is available only with the build command")
        RUN_ID = str(args.run_id)
        RUN_ROOT = CROSS_ROOT / "runs" / RUN_ID
    if args.command in {"freeze", "all"}:
        frozen = freeze_candidates()
        print(
            json.dumps(
                {
                    "phase": "freeze",
                    "candidate_count": frozen["candidate_count"],
                    "content_sha256": frozen["content_sha256"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.command in {"evaluate", "all"}:
        report = evaluate_target(workers=args.workers)
        print(json.dumps({"phase": "evaluate", **report["evaluation"]}, ensure_ascii=False), flush=True)
    if args.command in {"build", "all"}:
        manifest = build_page()
        print(json.dumps({"phase": "build", "output": str(RUN_ROOT), "manifest": manifest}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
