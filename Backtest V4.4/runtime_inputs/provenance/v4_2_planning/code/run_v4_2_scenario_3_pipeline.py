"""Run two V4.2 Scenario-3 stages while prior delivery continues in parallel."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_v4_2_scenario_3_calculated_entry_plans import (
    CAMPAIGN_ID,
    CAMPAIGNS_ROOT,
    PLANS_ROOT,
    PLAN_SPECS,
    build_plans,
    sha256_file,
)
from run_v4_2_delivery_worker import wait_for_delivery
from run_v4_2_resumable_campaign import run_stage


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _plan_path(stage_id: str) -> Path:
    return PLANS_ROOT / f"{CAMPAIGN_ID}_{stage_id}.json"


def _stage_root(stage_id: str) -> Path:
    return CAMPAIGNS_ROOT / CAMPAIGN_ID / stage_id


def run_pipeline(
    *,
    workers: int = 3,
    batch_size: int = 12,
    minimum_free_memory_mb: int = 4096,
    review_workers: int = 4,
    validate_only: bool = False,
) -> dict[str, Any]:
    plans = build_plans()
    union_output = Path(r"F:\V4_2_results\all_completed_union_analysis")
    stage_results: list[dict[str, Any]] = []
    delivery_jobs: list[dict[str, Any]] = []
    for spec in PLAN_SPECS:
        stage_id = str(spec["stage_id"])
        plan = _plan_path(stage_id)
        stage = _stage_root(stage_id)
        result = run_stage(
            plan,
            stage,
            workers=int(workers),
            batch_size=int(batch_size),
            minimum_free_memory_mb=int(minimum_free_memory_mb),
            validate_only=validate_only,
            delivery_mode="background",
            review_workers=int(review_workers),
            union_campaigns_root=CAMPAIGNS_ROOT,
            union_output_root=union_output,
        )
        stage_results.append(result)
        if validate_only:
            continue
        delivery = result.get("mandatory_html_delivery")
        if not isinstance(delivery, dict) or not delivery.get("job_id"):
            raise RuntimeError(f"{stage_id} did not launch mandatory delivery")
        delivery_jobs.append(
            {
                "stage_id": stage_id,
                "stage": str(stage.resolve()),
                "job_id": str(delivery["job_id"]),
                "launched_at": delivery.get("launched_at"),
                "pid": delivery.get("pid"),
            }
        )

    if validate_only:
        return {
            "status": "ready",
            "campaign_id": CAMPAIGN_ID,
            "coordinate_count": plans["coordinate_count"],
            "stage_results": stage_results,
        }

    delivery_results = [
        wait_for_delivery(
            Path(job["stage"]),
            job["job_id"],
            timeout_seconds=7200,
            poll_seconds=1,
        )
        for job in delivery_jobs
    ]
    stage_two_progress = json.loads(
        (_stage_root(str(PLAN_SPECS[1]["stage_id"])) / "progress.json").read_text(
            encoding="utf-8"
        )
    )
    stage_one_delivery = delivery_results[0]
    overlap_observed = str(stage_two_progress["started_at"]) < str(
        stage_one_delivery["completed_at"]
    )
    if not overlap_observed:
        raise RuntimeError(
            "pipeline did not observe stage-2 compute overlapping stage-1 delivery"
        )

    campaign_root = CAMPAIGNS_ROOT / CAMPAIGN_ID
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": _utc_now(),
        "campaign_id": CAMPAIGN_ID,
        "version_label": "V4.2",
        "entry_fill_mode": "calculated_threshold",
        "pipeline_contract": {
            "stage_compute_lock": ".v4_2_runner.lock",
            "stage_delivery_lock": ".v4_2_delivery.lock",
            "cumulative_union_lock": ".v4_2_union.lock",
            "review_worker_count": int(review_workers),
            "stage_2_started_before_stage_1_delivery_completed": overlap_observed,
            "immutable_handoff": "completion_manifest and hash-bound stage artifacts",
        },
        "stages": [],
        "cumulative_output": str(union_output.resolve()),
        "parameter_acceptance": "none",
    }
    for spec, delivery in zip(PLAN_SPECS, delivery_results):
        stage_id = str(spec["stage_id"])
        stage = _stage_root(stage_id)
        manifest["stages"].append(
            {
                "stage_id": stage_id,
                "plan": {
                    "path": str(_plan_path(stage_id).resolve()),
                    "sha256": sha256_file(_plan_path(stage_id)),
                },
                "completion_manifest": {
                    "path": str((stage / "completion_manifest.json").resolve()),
                    "sha256": sha256_file(stage / "completion_manifest.json"),
                },
                "delivery_status": {
                    "path": str((stage / "delivery_status.json").resolve()),
                    "sha256": sha256_file(stage / "delivery_status.json"),
                },
                "coordinate_count": int(delivery["stage_coordinate_count"]),
                "trade_count": int(delivery["stage_trade_count"]),
                "scenario_3_qualified_coordinate_count": int(
                    delivery["stage_scenario_3_qualified_coordinate_count"]
                ),
            }
        )
    manifest["coordinate_count"] = sum(
        row["coordinate_count"] for row in manifest["stages"]
    )
    manifest["trade_count"] = sum(row["trade_count"] for row in manifest["stages"])
    manifest["cumulative_coordinate_count"] = int(
        delivery_results[-1]["union_coordinate_count"]
    )
    manifest["cumulative_trade_count"] = int(
        delivery_results[-1]["union_trade_count"]
    )
    manifest_path = campaign_root / "pipeline_completion_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the audited two-stage V4.2 Scenario-3 pipeline."
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--minimum-free-memory-mb", type=int, default=4096)
    parser.add_argument("--review-workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(
        workers=args.workers,
        batch_size=args.batch_size,
        minimum_free_memory_mb=args.minimum_free_memory_mb,
        review_workers=args.review_workers,
        validate_only=args.validate_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
