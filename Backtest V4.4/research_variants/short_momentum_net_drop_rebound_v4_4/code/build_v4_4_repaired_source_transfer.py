from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_v4_4_cross_instrument_comparison as base
import build_v4_4_strict_entry_transfer as strict


RUN_ID = (
    "k200_repaired_v48_20260526_20260708"
    "__simain_20260129_20260223"
    "__promising_exact_transfer_v49_20260805"
)
PREVIOUS_TRANSFER_RUN_ID = (
    "k200_20260526_20260708__simain_20260129_20260223"
    "__all_exact_transfers_v46_20260805"
)
TRANSFER_BATCH_LABEL = "修复后 K200 精确迁移"
ORIGINAL_RUN_ID = "k200_20260526_20260708__simain_20260129_20260223"
COMBINED_RUN_ID = (
    "k200_20260526_20260708__simain_20260129_20260223"
    "__original_180_plus_repaired_64_v50_20260805"
)
MIGRATION_PLAN = (
    base.VARIANT_ROOT
    / "plans"
    / "v4_4_migration_k200_to_simain_20260805.json"
)


def configure() -> None:
    strict.STRICT_RUN_ID = RUN_ID
    strict.OLD_RUN_ID = PREVIOUS_TRANSFER_RUN_ID
    strict.STRICT_BATCH_LABEL = TRANSFER_BATCH_LABEL


def freeze() -> dict:
    configure()
    return strict.freeze_strict_candidates()


def evaluate(*, workers: int) -> dict:
    configure()
    return strict.evaluate_strict_target(workers=workers)


def build() -> dict:
    configure()
    strict._set_run(RUN_ID)
    return base.build_page()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _combined_report_markdown(report: dict, *, language: str) -> str:
    evaluation = report["evaluation"]
    if language == "en":
        return f"""# K200 to SImain Migration Report

- Migration plan: `v4_4_migration_k200_to_simain_20260805.json`
- Source instrument: K200
- Target instrument: SImain
- Combined candidate count: {evaluation['candidate_count']} ({evaluation['original_candidate_count']} earlier + {evaluation['repaired_candidate_count']} repaired)
- Target-positive candidates: {evaluation['target_positive_candidate_count']} ({evaluation['target_positive_candidate_fraction']:.2%})
- Source/target Spearman rank correlation: {evaluation['rank_spearman_correlation']:.6f}
- Stable candidates: {evaluation['stable_candidate_count']}
- Isolated positive candidates: {evaluation['isolated_positive_count']}

The combined page is a presentation union of two completed exact-transfer result sets. Their candidate coordinates do not overlap. Target results did not generate or modify the source candidate sets.
"""
    return f"""# K200 → SImain 迁移报告

- 迁移方案：`v4_4_migration_k200_to_simain_20260805.json`
- 迁移前品种：K200
- 迁移后品种：SImain
- 合并候选数：{evaluation['candidate_count']}（原结果 {evaluation['original_candidate_count']} + 修复后结果 {evaluation['repaired_candidate_count']}）
- SImain 正收益候选：{evaluation['target_positive_candidate_count']}（{evaluation['target_positive_candidate_fraction']:.2%}）
- K200／SImain Spearman 排名相关性：{evaluation['rank_spearman_correlation']:.6f}
- 稳定候选：{evaluation['stable_candidate_count']}
- 孤立正收益候选：{evaluation['isolated_positive_count']}

当前页面合并展示两组已经完成的精确迁移结果；两组候选坐标没有重合。SImain 结果没有生成或修改 K200 候选集合。
"""


def combine() -> dict:
    original_root = base.CROSS_ROOT / "runs" / ORIGINAL_RUN_ID
    repaired_root = base.CROSS_ROOT / "runs" / RUN_ID
    combined_root = base.CROSS_ROOT / "runs" / COMBINED_RUN_ID
    combined_root.mkdir(parents=True, exist_ok=False)

    original_comparison = pd.read_csv(original_root / "migration_comparison.csv")
    repaired_comparison = pd.read_csv(repaired_root / "migration_comparison.csv")
    combined = pd.concat(
        [original_comparison, repaired_comparison], ignore_index=True, sort=False
    ).drop(columns=["transfer_batch"], errors="ignore")
    if combined.combo_id.astype(str).duplicated().any():
        raise ValueError("the requested result sets contain duplicate combo_id values")
    combined = strict._recompute_diagnostics(combined)

    source_trades = pd.concat(
        [
            pd.read_csv(original_root / "source_candidate_trades.csv", low_memory=False),
            pd.read_csv(repaired_root / "source_candidate_trades.csv", low_memory=False),
        ],
        ignore_index=True,
        sort=False,
    )
    target_trades = pd.concat(
        [
            pd.read_csv(original_root / "simain_candidate_trades.csv", low_memory=False),
            pd.read_csv(repaired_root / "simain_candidate_trades.csv", low_memory=False),
        ],
        ignore_index=True,
        sort=False,
    )
    representative = pd.concat(
        [
            pd.read_csv(original_root / "representative_trades.csv", low_memory=False),
            pd.read_csv(repaired_root / "representative_trades.csv", low_memory=False),
        ],
        ignore_index=True,
        sort=False,
    )

    original_freeze = _load_json(original_root / base.FREEZE_NAME)
    repaired_freeze = _load_json(repaired_root / base.FREEZE_NAME)
    candidates = pd.concat(
        [
            pd.DataFrame(original_freeze["candidates"]),
            pd.DataFrame(repaired_freeze["candidates"]),
        ],
        ignore_index=True,
        sort=False,
    ).drop(columns=["transfer_batch"], errors="ignore")
    candidates["candidate_order"] = np.arange(1, len(candidates) + 1)
    freeze = {
        "schema_version": 1,
        "status": "presentation_union_of_completed_exact_transfers",
        "generated_at_utc": base.utc_now(),
        "source_freezes": [
            {
                "run_id": ORIGINAL_RUN_ID,
                "artifact": base.artifact(original_root / base.FREEZE_NAME),
                "content_sha256": original_freeze["content_sha256"],
            },
            {
                "run_id": RUN_ID,
                "artifact": base.artifact(repaired_root / base.FREEZE_NAME),
                "content_sha256": repaired_freeze["content_sha256"],
            },
        ],
        "candidate_count": int(len(candidates)),
        "candidates": base._records(candidates),
        "target_results_modified_source_freezes": False,
    }
    freeze["content_sha256"] = base.canonical_hash(freeze)

    repaired_report = _load_json(repaired_root / "migration_report.json")
    adjacency = base._adjacency(combined)
    report = {
        **repaired_report,
        "status": "complete_presentation_union_of_exact_transfers",
        "generated_at_utc": base.utc_now(),
        "candidate_freeze": {
            "path": str((combined_root / base.FREEZE_NAME).resolve()),
            "content_sha256": freeze["content_sha256"],
            "candidate_count": int(len(combined)),
            "source_freeze_count": 2,
            "target_results_cannot_modify_candidates": True,
        },
        "evaluation": {
            **repaired_report["evaluation"],
            "candidate_count": int(len(combined)),
            "original_candidate_count": int(len(original_comparison)),
            "repaired_candidate_count": int(len(repaired_comparison)),
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
        "parameter_acceptance": "none",
    }
    report["audits"]["gap"]["target_gap_trade_count"] = int(
        pd.to_numeric(target_trades.position_crosses_real_gap, errors="coerce")
        .fillna(0)
        .astype(bool)
        .sum()
    )

    config = _load_json(repaired_root / "run_config.json")
    config.update(
        {
            "run_id": COMBINED_RUN_ID,
            "mode": "transfer_exact_presentation_union",
            "migration_plan": str(MIGRATION_PLAN.resolve()),
            "source_runs": [ORIGINAL_RUN_ID, RUN_ID],
            "candidate_count": int(len(combined)),
            "result_semantics": (
                "presentation union of two completed exact-transfer result sets; "
                "no target-driven candidate generation"
            ),
        }
    )

    base.atomic_csv(combined_root / "migration_comparison.csv", combined)
    base.atomic_csv(combined_root / "source_candidate_trades.csv", source_trades)
    base.atomic_csv(combined_root / "simain_candidate_trades.csv", target_trades)
    base.atomic_csv(combined_root / "representative_trades.csv", representative)
    base.atomic_csv(combined_root / "frozen_candidates.csv", candidates)
    base.atomic_json(combined_root / base.FREEZE_NAME, freeze)
    base.atomic_json(combined_root / "migration_report.json", report)
    base.atomic_json(
        combined_root / "posthoc_full_grid_status.json", report["posthoc_full_grid"]
    )
    base.atomic_json(combined_root / "run_config.json", config)
    base.atomic_text(
        combined_root / "MIGRATION_REPORT.en.md",
        _combined_report_markdown(report, language="en"),
    )
    base.atomic_text(
        combined_root / "MIGRATION_REPORT.zh.md",
        _combined_report_markdown(report, language="zh"),
    )

    strict._set_run(COMBINED_RUN_ID)
    manifest = base.build_page()
    manifest["source_runs"] = [ORIGINAL_RUN_ID, RUN_ID]
    manifest["migration_reports"] = {
        "en": base.artifact(combined_root / "MIGRATION_REPORT.en.md"),
        "zh": base.artifact(combined_root / "MIGRATION_REPORT.zh.md"),
    }
    base.atomic_json(combined_root / "cross_instrument_manifest.json", manifest)

    redirect = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url=../{COMBINED_RUN_ID}/index.html">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>V4.41 跨品种迁移验证</title></head><body>'
        f'<p><a href="../{COMBINED_RUN_ID}/index.html">打开合并后的跨品种排序</a></p>'
        '</body></html>'
    )
    base.atomic_text(repaired_root / "index.html", redirect)
    repaired_manifest = _load_json(repaired_root / "cross_instrument_manifest.json")
    repaired_manifest["presentation_redirect"] = f"../{COMBINED_RUN_ID}/index.html"
    repaired_manifest["outputs"]["index.html"] = base.artifact(
        repaired_root / "index.html"
    )
    base.atomic_json(repaired_root / "cross_instrument_manifest.json", repaired_manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze promising candidates from the repaired K200 source snapshot, "
            "evaluate the exact coordinates on SImain, and publish an independent comparison."
        )
    )
    parser.add_argument(
        "command", choices=("freeze", "evaluate", "build", "combine", "all")
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.command in {"freeze", "all"}:
        payload = freeze()
        print(
            json.dumps(
                {
                    "phase": "freeze",
                    "run_id": RUN_ID,
                    "source_snapshot": payload["source_snapshot"]["union_snapshot_id"],
                    "candidate_count": payload["candidate_count"],
                    "content_sha256": payload["content_sha256"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.command in {"evaluate", "all"}:
        report = evaluate(workers=args.workers)
        print(
            json.dumps(
                {"phase": "evaluate", **report["evaluation"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.command in {"build", "all"}:
        manifest = build()
        print(
            json.dumps(
                {
                    "phase": "build",
                    "run_id": RUN_ID,
                    "output": str(base.CROSS_ROOT / "runs" / RUN_ID),
                    "manifest": manifest,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.command in {"combine", "all"}:
        manifest = combine()
        print(
            json.dumps(
                {
                    "phase": "combine",
                    "run_id": COMBINED_RUN_ID,
                    "output": str(base.CROSS_ROOT / "runs" / COMBINED_RUN_ID),
                    "manifest": manifest,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
