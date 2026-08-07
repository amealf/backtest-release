from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_CODE = PROJECT_ROOT / "research_variants" / "short_momentum_net_drop_rebound_v4_4" / "code"
sys.path.insert(0, str(VARIANT_CODE))

from analyze_v4_4_scenario_3_stage import K200M_COST_MODEL, _apply_cost_adjusted_metrics  # noqa: E402


PARAMETER_FIELDS = ("e", "bh", "trw", "k", "w", "m", "speed_window_bars")


def clean(value: object) -> object:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def row_record(row: pd.Series) -> dict[str, object]:
    return {str(key): clean(value) for key, value in row.items()}


def load_trades(stage_root: Path) -> pd.DataFrame:
    paths = sorted((stage_root / "batches").glob("batch_*/trades.csv"))
    return pd.concat(
        [
            pd.read_csv(
                path,
                usecols=["combo_id", "return", "entry_index", "exit_index", "position_crosses_real_gap"],
                low_memory=False,
            )
            for path in paths
        ],
        ignore_index=True,
    )


def add_dependency_metrics(summary: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    cost_fraction = float(K200M_COST_MODEL["round_trip_total_cost_bps"]) / 10000.0
    records: list[dict[str, object]] = []
    for combo_id, group in trades.groupby(trades.combo_id.astype(str), sort=False):
        adjusted = pd.to_numeric(group["return"], errors="raise") - cost_fraction
        positives = adjusted.loc[adjusted.gt(0)].sort_values(ascending=False)
        positive_sum = float(positives.sum())
        top2 = 0.0 if positive_sum <= 0 else float(positives.head(2).sum() / positive_sum)
        non_gap = adjusted.loc[~group.position_crosses_real_gap.astype(bool)]
        non_gap_total = 0.0 if non_gap.empty else float(np.prod(1.0 + non_gap.to_numpy()) - 1.0)
        records.append({
            "combo_id": str(combo_id),
            "train_cost_adjusted_return_excluding_gap": non_gap_total,
            "positive_return_top2_share": top2,
        })
    return result.merge(pd.DataFrame(records), on="combo_id", how="left", validate="one_to_one")


def select_views(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    positive = frame.loc[frame.train_cost_adjusted_return.gt(0)].copy()
    views: dict[str, tuple[pd.DataFrame, str, bool]] = {
        "total_return": (frame, "train_cost_adjusted_return", False),
        "scenario_1_total": (frame.loc[frame.scenario_1_qualified.astype(bool)], "train_cost_adjusted_return", False),
        "average_ge10": (frame.loc[frame.train_trade_count.ge(10)], "train_cost_adjusted_avg_trade", False),
        "average_ge20": (frame.loc[frame.train_trade_count.ge(20)], "train_cost_adjusted_avg_trade", False),
        "non_gap_return": (frame, "train_cost_adjusted_return_excluding_gap", False),
        "low_drawdown_positive": (positive, "train_cost_adjusted_max_drawdown_abs", True),
    }
    output: dict[str, dict[str, object]] = {}
    for name, (eligible, metric, ascending) in views.items():
        if eligible.empty:
            continue
        row = eligible.sort_values(metric, ascending=ascending, kind="mergesort").iloc[0]
        output[name] = row_record(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    stage_root = Path(args.stage).resolve()

    completion = json.loads((stage_root / "completion_manifest.json").read_text(encoding="utf-8"))
    if completion["status"] != "complete":
        raise RuntimeError("stage is not complete")
    summary = pd.read_csv(stage_root / "stage_summary.csv", low_memory=False)
    trades = load_trades(stage_root)
    summary, trades = _apply_cost_adjusted_metrics(summary, trades, cost_model=K200M_COST_MODEL, copy=False)
    summary = add_dependency_metrics(summary, trades)

    plan = json.loads((stage_root / "input_plan.json").read_text(encoding="utf-8"))
    labels = pd.DataFrame(
        [
            {
                **{field: row[field] for field in PARAMETER_FIELDS},
                "experiment_block": row.get("design", ""),
                "search_mode": row.get("search_mode", ""),
                "objective": row.get("objective", ""),
            }
            for row in plan["explicit_combos"]
        ]
    )
    summary = summary.merge(labels, on=list(PARAMETER_FIELDS), how="left", validate="one_to_one")

    output_root = stage_root / "compact_analysis"
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "analysis_summary.csv", index=False)
    views = select_views(summary)
    pd.DataFrame(
        [{"view": name, **record} for name, record in views.items()]
    ).to_csv(output_root / "view_leaders.csv", index=False)
    report = {
        "status": "complete",
        "stage_id": stage_root.name,
        "phase": plan.get("exploration_cycle", {}).get("phase", "unclassified"),
        "coordinate_count": int(len(summary)),
        "trade_count": int(len(trades)),
        "cost_positive_coordinate_count": int(summary.train_cost_adjusted_return.gt(0).sum()),
        "scenario_1_qualified_coordinate_count": int(summary.scenario_1_qualified.astype(bool).sum()),
        "view_leaders": views,
        "cost_model_id": K200M_COST_MODEL["id"],
        "combined_score": False,
        "parameter_acceptance": "none",
        "evidence_boundary": "in_sample_immutable_closed_only",
    }
    (output_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    concise = {
        "stage_id": stage_root.name,
        "coordinate_count": report["coordinate_count"],
        "trade_count": report["trade_count"],
        "cost_positive_coordinate_count": report["cost_positive_coordinate_count"],
        "scenario_1_qualified_coordinate_count": report["scenario_1_qualified_coordinate_count"],
        "leaders": {
            name: {
                "combo_id": record["combo_id"],
                "parameters": {field: record[field] for field in PARAMETER_FIELDS},
                "return": record["train_cost_adjusted_return"],
                "average": record["train_cost_adjusted_avg_trade"],
                "drawdown": record["train_cost_adjusted_max_drawdown_abs"],
                "trades": record["train_trade_count"],
                "non_gap_return": record["train_cost_adjusted_return_excluding_gap"],
                "top2_share": record["positive_return_top2_share"],
            }
            for name, record in views.items()
        },
    }
    print(json.dumps(concise, ensure_ascii=False))


if __name__ == "__main__":
    main()
