from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_v4_4_cross_instrument_comparison as base
import build_v4_4_strict_entry_transfer as strict


RAW_RUN_ID = (
    "k200_20260526_20260708__simain_20260129_20260223"
    "__stricter_entry_k_refinement_v51_20260805"
)
PARENT_RUN_ID = (
    "k200_20260526_20260708__simain_20260129_20260223"
    "__original_180_plus_repaired_64_v50_20260805"
)
FINAL_RUN_ID = (
    "k200_20260526_20260708__simain_20260129_20260223"
    "__combined_247_stricter_entry_v52_20260805"
)
SOURCE_STAGE = (
    base.RESULTS_ROOT
    / "campaigns"
    / "v4_4_positive_entry_signal_repair_20260805"
    / "continuation_round_13_stricter_entry_k_expansion_all_window"
)
MIGRATION_PLAN = (
    base.VARIANT_ROOT
    / "plans"
    / "v4_4_migration_k200_to_simain_stricter_entry_20260805.json"
)
CLASSIFICATION = "improved_on_target_mixed_across_instruments"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _planned_k_values() -> list[float]:
    return [
        float(value)
        for value in _load_json(MIGRATION_PLAN)["bounded_neighborhood"]["values"]
    ]


def _configure(args: argparse.Namespace) -> None:
    global RAW_RUN_ID, PARENT_RUN_ID, FINAL_RUN_ID, SOURCE_STAGE, MIGRATION_PLAN
    global CLASSIFICATION
    if args.raw_run_id:
        RAW_RUN_ID = args.raw_run_id
    if args.parent_run_id:
        PARENT_RUN_ID = args.parent_run_id
    if args.final_run_id:
        FINAL_RUN_ID = args.final_run_id
    if args.source_stage:
        SOURCE_STAGE = args.source_stage.resolve()
    if args.migration_plan:
        MIGRATION_PLAN = args.migration_plan.resolve()
    if args.classification:
        CLASSIFICATION = args.classification


def _set_run(run_id: str) -> Path:
    base.RUN_ID = run_id
    base.RUN_ROOT = base.CROSS_ROOT / "runs" / run_id
    base.ROUND_TRIP_COST_BPS = 3.57
    return base.RUN_ROOT


def freeze() -> dict:
    run_root = _set_run(RAW_RUN_ID)
    run_root.mkdir(parents=True, exist_ok=False)
    summary = pd.read_csv(SOURCE_STAGE / "stage_summary.csv")
    trades = pd.concat(
        [
            pd.read_csv(path, low_memory=False)
            for path in sorted((SOURCE_STAGE / "batches").glob("batch_*/trades.csv"))
        ],
        ignore_index=True,
    )
    trades["gross_return"] = pd.to_numeric(trades["return"], errors="raise")
    trades["source_stage_id"] = SOURCE_STAGE.name
    trades["source_stage_root"] = str(SOURCE_STAGE.resolve())
    completion = _load_json(SOURCE_STAGE / "completion_manifest.json")
    trades["source_plan_fingerprint"] = completion["plan_fingerprint"]
    source_frame = base.load_bars(base.K200_SOURCE, base.K200_PREPARATION)
    source_enriched, source_metrics = base._add_excursions(
        trades, source_frame, prefix="source"
    )
    threshold = (
        trades.assign(
            source_entry_threshold=pd.to_numeric(
                trades["entry_baseline_value"], errors="raise"
            )
            * pd.to_numeric(trades["k"], errors="raise")
        )
        .groupby("combo_id", sort=False)["source_entry_threshold"]
        .median()
        .rename("source_entry_threshold_median")
        .reset_index()
    )
    candidates = (
        summary.merge(source_metrics, on="combo_id", how="left", validate="one_to_one")
        .merge(threshold, on="combo_id", how="left", validate="one_to_one")
        .sort_values("k")
        .reset_index(drop=True)
    )
    candidates["candidate_order"] = np.arange(1, len(candidates) + 1)
    candidates["selection_tags"] = "approved_stricter_entry_k_confirmation"
    candidates["source_stage_id"] = SOURCE_STAGE.name
    candidates["source_stage_root"] = str(SOURCE_STAGE.resolve())
    candidates["source_plan_fingerprint"] = completion["plan_fingerprint"]
    columns = [
        "candidate_order",
        "combo_id",
        *base.EXECUTION_FIELDS,
        *base.PARAMETER_FIELDS,
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
        "source_entry_threshold_median",
        "source_stage_id",
        "source_stage_root",
        "source_plan_fingerprint",
    ]
    candidates = candidates[columns]
    payload = {
        "schema_version": 1,
        "status": "frozen_before_target_evaluation",
        "generated_at_utc": base.utc_now(),
        "source_instrument": "K200",
        "source_sample": {
            "start": str(base.SOURCE_START),
            "end": str(base.SOURCE_END),
            "timezone": "Asia/Seoul",
        },
        "source_stage": base.artifact(SOURCE_STAGE / "completion_manifest.json"),
        "migration_plan": base.artifact(MIGRATION_PLAN),
        "selection_contract": {
            "mode": "target_local_refinement",
            "changed_field": "k",
            "values": _planned_k_values(),
            "target_results_cannot_modify_candidates": True,
            "combined_score": False,
        },
        "candidate_count": int(len(candidates)),
        "candidates": base._records(candidates),
    }
    payload["content_sha256"] = base.canonical_hash(payload)
    base.atomic_json(run_root / base.FREEZE_NAME, payload)
    base.atomic_csv(run_root / "frozen_candidates.csv", candidates)
    base.atomic_csv(run_root / "source_candidate_trades.csv", source_enriched)
    return payload


def evaluate(*, workers: int) -> dict:
    _set_run(RAW_RUN_ID)
    report = base.evaluate_target(workers=workers)
    config_path = base.RUN_ROOT / "run_config.json"
    config = _load_json(config_path)
    config.update(
        {
            "mode": "target_local_refinement",
            "migration_plan": str(MIGRATION_PLAN.resolve()),
            "parent_transfer_run": PARENT_RUN_ID,
            "source_stage": str(SOURCE_STAGE.resolve()),
            "result_semantics": (
                f"{len(_planned_k_values())} frozen K200 stricter-entry coordinates evaluated on SImain; "
                "target results cannot modify candidates"
            ),
        }
    )
    config["source"]["snapshot_root"] = str(SOURCE_STAGE.resolve())
    base.atomic_json(config_path, config)
    return report


def _report_markdown(report: dict, *, language: str) -> str:
    e = report["evaluation"]
    new_rows = pd.DataFrame(report["stricter_entry_results"])
    lines = []
    for row in new_rows.sort_values("k").to_dict("records"):
        lines.append(
            f"K={row['k']}: K200 {row['source_trade_count']} trades / "
            f"{row['source_cost_total_return']:.2%}; SImain {row['target_trade_count']} trades / "
            f"{row['target_cost_total_return']:.2%}."
        )
    if language == "en":
        return "\n".join(
            [
                "# K200 to SImain Stricter-Entry Migration Report",
                "",
                f"Combined candidates: {e['candidate_count']}.",
                f"SImain cost-positive candidates: {e['target_positive_candidate_count']} ({e['target_positive_candidate_fraction']:.2%}).",
                "",
                *[f"- {line}" for line in lines],
                "",
                f"Classification: {report['stricter_entry_classification']}. No parameter is accepted.",
                "",
            ]
        )
    zh_lines = [
        line.replace("trades", "笔").replace("K200", "K200").replace("SImain", "SImain")
        for line in lines
    ]
    return "\n".join(
        [
            "# K200 → SImain 更严格开仓迁移报告",
            "",
            f"合并候选：{e['candidate_count']} 个。",
            f"SImain 成本后正收益候选：{e['target_positive_candidate_count']} 个（{e['target_positive_candidate_fraction']:.2%}）。",
            "",
            *[f"- {line}" for line in zh_lines],
            "",
            f"结论分类：{report['stricter_entry_classification']}。没有接受参数。",
            "",
        ]
    )


def publish() -> dict:
    parent_root = base.CROSS_ROOT / "runs" / PARENT_RUN_ID
    new_root = base.CROSS_ROOT / "runs" / RAW_RUN_ID
    final_root = base.CROSS_ROOT / "runs" / FINAL_RUN_ID
    final_root.mkdir(parents=True, exist_ok=False)

    parent_comparison = pd.read_csv(parent_root / "migration_comparison.csv")
    new_comparison = pd.read_csv(new_root / "migration_comparison.csv")
    combined = strict._recompute_diagnostics(
        pd.concat([parent_comparison, new_comparison], ignore_index=True, sort=False)
        .drop(columns=["transfer_batch"], errors="ignore")
    )
    source_trades = pd.concat(
        [
            pd.read_csv(parent_root / "source_candidate_trades.csv", low_memory=False),
            pd.read_csv(new_root / "source_candidate_trades.csv", low_memory=False),
        ],
        ignore_index=True,
        sort=False,
    )
    target_trades = pd.concat(
        [
            pd.read_csv(parent_root / "simain_candidate_trades.csv", low_memory=False),
            pd.read_csv(new_root / "simain_candidate_trades.csv", low_memory=False),
        ],
        ignore_index=True,
        sort=False,
    )
    representative = pd.concat(
        [
            pd.read_csv(parent_root / "representative_trades.csv", low_memory=False),
            pd.read_csv(new_root / "representative_trades.csv", low_memory=False),
        ],
        ignore_index=True,
        sort=False,
    )

    parent_freeze = _load_json(parent_root / base.FREEZE_NAME)
    new_freeze = _load_json(new_root / base.FREEZE_NAME)
    candidates = pd.concat(
        [
            pd.DataFrame(parent_freeze["candidates"]),
            pd.DataFrame(new_freeze["candidates"]),
        ],
        ignore_index=True,
        sort=False,
    ).drop(columns=["transfer_batch"], errors="ignore")
    candidates["candidate_order"] = np.arange(1, len(candidates) + 1)
    freeze = {
        "schema_version": 1,
        "status": "presentation_union_of_completed_transfer_and_local_refinement",
        "generated_at_utc": base.utc_now(),
        "source_freezes": [
            {
                "run_id": PARENT_RUN_ID,
                "artifact": base.artifact(parent_root / base.FREEZE_NAME),
                "content_sha256": parent_freeze["content_sha256"],
            },
            {
                "run_id": RAW_RUN_ID,
                "artifact": base.artifact(new_root / base.FREEZE_NAME),
                "content_sha256": new_freeze["content_sha256"],
            },
        ],
        "candidate_count": int(len(candidates)),
        "candidates": base._records(candidates),
        "target_results_modified_source_freezes": False,
    }
    freeze["content_sha256"] = base.canonical_hash(freeze)

    new_report = _load_json(new_root / "migration_report.json")
    adjacency = base._adjacency(combined)
    new_result_rows = combined.loc[
        combined.combo_id.astype(str).isin(new_comparison.combo_id.astype(str))
    ]
    report = {
        **new_report,
        "status": "complete_combined_transfer_with_stricter_entry_refinement",
        "generated_at_utc": base.utc_now(),
        "candidate_freeze": {
            "path": str((final_root / base.FREEZE_NAME).resolve()),
            "content_sha256": freeze["content_sha256"],
            "candidate_count": int(len(combined)),
            "source_freeze_count": 2,
            "target_results_cannot_modify_candidates": True,
        },
        "evaluation": {
            **new_report["evaluation"],
            "candidate_count": int(len(combined)),
            "parent_candidate_count": int(len(parent_comparison)),
            "stricter_entry_candidate_count": int(len(new_comparison)),
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
        "stricter_entry_results": base._records(
            new_result_rows[
                [
                    "combo_id",
                    "k",
                    "source_trade_count",
                    "source_cost_total_return",
                    "source_cost_max_drawdown_abs",
                    "target_trade_count",
                    "target_cost_total_return",
                    "target_cost_median_trade",
                    "target_cost_max_drawdown_abs",
                ]
            ]
        ),
        "stricter_entry_classification": CLASSIFICATION,
        "parameter_acceptance": "none",
    }
    report["audits"]["gap"]["target_gap_trade_count"] = int(
        base._truthy(target_trades["position_crosses_real_gap"]).sum()
    )

    config = _load_json(new_root / "run_config.json")
    config.update(
        {
            "run_id": FINAL_RUN_ID,
            "mode": "target_local_refinement_presentation_union",
            "migration_plan": str(MIGRATION_PLAN.resolve()),
            "source_runs": [PARENT_RUN_ID, RAW_RUN_ID],
            "incremental_parent_run": PARENT_RUN_ID,
            "candidate_count": int(len(combined)),
            "result_semantics": (
                f"combined presentation of {len(parent_comparison)} compatible migration candidates and "
                f"{len(new_comparison)} frozen stricter-entry target-local refinement candidates"
            ),
        }
    )

    base.atomic_csv(final_root / "migration_comparison.csv", combined)
    base.atomic_csv(final_root / "source_candidate_trades.csv", source_trades)
    base.atomic_csv(final_root / "simain_candidate_trades.csv", target_trades)
    base.atomic_csv(final_root / "representative_trades.csv", representative)
    base.atomic_csv(final_root / "frozen_candidates.csv", candidates)
    base.atomic_json(final_root / base.FREEZE_NAME, freeze)
    base.atomic_json(final_root / "migration_report.json", report)
    base.atomic_json(
        final_root / "posthoc_full_grid_status.json", report["posthoc_full_grid"]
    )
    base.atomic_json(final_root / "run_config.json", config)
    base.atomic_text(
        final_root / "MIGRATION_REPORT.en.md",
        _report_markdown(report, language="en"),
    )
    base.atomic_text(
        final_root / "MIGRATION_REPORT.zh.md",
        _report_markdown(report, language="zh"),
    )

    _set_run(FINAL_RUN_ID)
    manifest = base.build_page()
    manifest["source_runs"] = [PARENT_RUN_ID, RAW_RUN_ID]
    manifest["migration_reports"] = {
        "en": base.artifact(final_root / "MIGRATION_REPORT.en.md"),
        "zh": base.artifact(final_root / "MIGRATION_REPORT.zh.md"),
    }
    base.atomic_json(final_root / "cross_instrument_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "evaluate", "publish", "all"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--raw-run-id")
    parser.add_argument("--parent-run-id")
    parser.add_argument("--final-run-id")
    parser.add_argument("--source-stage", type=Path)
    parser.add_argument("--migration-plan", type=Path)
    parser.add_argument("--classification")
    args = parser.parse_args()
    _configure(args)
    if args.command in {"freeze", "all"}:
        print(json.dumps({"phase": "freeze", **freeze()}, ensure_ascii=False))
    if args.command in {"evaluate", "all"}:
        report = evaluate(workers=args.workers)
        print(json.dumps({"phase": "evaluate", **report["evaluation"]}, ensure_ascii=False))
    if args.command in {"publish", "all"}:
        manifest = publish()
        print(json.dumps({"phase": "publish", "run_id": FINAL_RUN_ID, "manifest": manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
