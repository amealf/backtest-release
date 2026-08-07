"""Run V4.4 stage and cumulative HTML delivery outside the compute lock."""
from __future__ import annotations

import argparse
import hashlib
import json
import msvcrt
import os
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from analyze_v4_4_scenario_3_stage import analyze, sha256_file
from build_v4_4_combined_union_analysis import build_union


DELIVERY_STATUS_SCHEMA_VERSION = 1
DELIVERY_LOCK_NAME = ".v4_4_delivery.lock"
DELIVERY_STATUS_NAME = "delivery_status.json"
DELIVERY_JOB_NAME = "delivery_job.json"


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


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _job_id(plan: Path, stage: Path, completion: Path, review_workers: int) -> str:
    digest = hashlib.sha256()
    for value in (
        str(plan.resolve()),
        sha256_file(plan),
        str(stage.resolve()),
        sha256_file(completion),
        str(int(review_workers)),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@contextmanager
def _exclusive_delivery_writer(stage: Path) -> Iterator[Path]:
    lock_path = stage / DELIVERY_LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError(
                f"another V4.4 delivery writer owns this stage: {lock_path}"
            ) from error
        yield lock_path
    finally:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()


def deliver(
    plan: Path,
    stage: Path,
    campaigns_root: Path,
    union_output: Path,
    *,
    review_workers: int = 4,
) -> dict[str, Any]:
    plan = plan.resolve()
    stage = stage.resolve()
    campaigns_root = campaigns_root.resolve()
    union_output = union_output.resolve()
    completion = stage / "completion_manifest.json"
    if not plan.is_file() or not completion.is_file():
        raise FileNotFoundError("V4.4 delivery requires a plan and completion manifest")
    workers = int(review_workers)
    if workers < 1 or workers > 32:
        raise ValueError("review_workers must be between 1 and 32")
    job_id = _job_id(plan, stage, completion, workers)
    status_path = stage / DELIVERY_STATUS_NAME
    with _exclusive_delivery_writer(stage) as lock_path:
        running = {
            "schema_version": DELIVERY_STATUS_SCHEMA_VERSION,
            "job_id": job_id,
            "status": "running",
            "started_at": _utc_now(),
            "pid": os.getpid(),
            "plan": str(plan),
            "plan_sha256": sha256_file(plan),
            "stage": str(stage),
            "completion_manifest": str(completion),
            "completion_manifest_sha256": sha256_file(completion),
            "campaigns_root": str(campaigns_root),
            "union_output": str(union_output),
            "review_workers": workers,
            "delivery_lock": str(lock_path),
        }
        _atomic_json(status_path, running)
        try:
            stage_result = analyze(
                plan,
                stage,
                stage / "analysis",
                review_workers=workers,
            )
            union_result = build_union(
                campaigns_root=campaigns_root,
                output_root=union_output,
                review_workers=workers,
            )
            complete = {
                **running,
                "status": "complete",
                "completed_at": _utc_now(),
                "stage_analysis": str((stage / "analysis").resolve()),
                "stage_coordinate_count": int(stage_result["coordinate_count"]),
                "stage_trade_count": int(stage_result["trade_count"]),
                "stage_scenario_3_qualified_coordinate_count": int(
                    stage_result["scenario_3_qualified_coordinate_count"]
                ),
                "union_snapshot_output": union_result["snapshot_output"],
                "union_coordinate_count": int(union_result["coordinate_count"]),
                "union_trade_count": int(union_result["trade_count"]),
                "union_completed_stage_count": int(
                    union_result["completed_stage_count"]
                ),
            }
            _atomic_json(status_path, complete)
            return complete
        except BaseException as error:
            failed = {
                **running,
                "status": "failed",
                "failed_at": _utc_now(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(limit=20),
                },
            }
            _atomic_json(status_path, failed)
            raise


def launch_delivery(
    plan: Path,
    stage: Path,
    campaigns_root: Path,
    union_output: Path,
    *,
    review_workers: int = 4,
) -> dict[str, Any]:
    plan = plan.resolve()
    stage = stage.resolve()
    campaigns_root = campaigns_root.resolve()
    union_output = union_output.resolve()
    completion = stage / "completion_manifest.json"
    if not completion.is_file():
        raise FileNotFoundError(f"V4.4 completion manifest is missing: {completion}")
    workers = int(review_workers)
    job_id = _job_id(plan, stage, completion, workers)
    status_path = stage / DELIVERY_STATUS_NAME
    existing = _load_json(status_path)
    if existing and existing.get("job_id") == job_id and existing.get("status") in {
        "queued",
        "running",
        "complete",
    }:
        return {**existing, "reused_existing_job": True}

    job_path = stage / DELIVERY_JOB_NAME
    stdout_path = stage / "delivery_worker.stdout.log"
    stderr_path = stage / "delivery_worker.stderr.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--plan",
        str(plan),
        "--stage",
        str(stage),
        "--campaigns-root",
        str(campaigns_root),
        "--union-output",
        str(union_output),
        "--review-workers",
        str(workers),
    ]
    queued = {
        "schema_version": DELIVERY_STATUS_SCHEMA_VERSION,
        "job_id": job_id,
        "status": "queued",
        "queued_at": _utc_now(),
        "plan": str(plan),
        "stage": str(stage),
        "campaigns_root": str(campaigns_root),
        "union_output": str(union_output),
        "review_workers": workers,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": command,
    }
    _atomic_json(status_path, queued)
    with stdout_path.open("ab") as stdout_handle, stderr_path.open("ab") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            ),
            close_fds=True,
        )
    job = {**queued, "pid": process.pid, "launched_at": _utc_now()}
    _atomic_json(job_path, job)
    return {**job, "job_manifest": str(job_path), "reused_existing_job": False}


def wait_for_delivery(
    stage: Path,
    job_id: str,
    *,
    timeout_seconds: float = 3600.0,
    poll_seconds: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_seconds)
    status_path = stage.resolve() / DELIVERY_STATUS_NAME
    while time.monotonic() < deadline:
        status = _load_json(status_path)
        if status and status.get("job_id") == job_id:
            if status.get("status") == "complete":
                return status
            if status.get("status") == "failed":
                raise RuntimeError(
                    f"V4.4 delivery failed: {status.get('error', {}).get('message')}"
                )
        time.sleep(max(0.1, float(poll_seconds)))
    raise TimeoutError(f"timed out waiting for V4.4 delivery: {status_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one V4.4 stage delivery and refresh its cumulative union."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--campaigns-root", type=Path, required=True)
    parser.add_argument("--union-output", type=Path, required=True)
    parser.add_argument("--review-workers", type=int, default=4)
    args = parser.parse_args()
    result = deliver(
        args.plan,
        args.stage,
        args.campaigns_root,
        args.union_output,
        review_workers=args.review_workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
