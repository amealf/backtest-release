from __future__ import annotations

import argparse
import csv
import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
BUILDER = PROJECT_ROOT / "tools" / "build_v4_4_k200_ai_cycle_plan.py"
ANALYZER = PROJECT_ROOT / "tools" / "analyze_v4_4_k200_ai_cycle_stage.py"
RUNNER = (
    PROJECT_ROOT
    / "research_variants"
    / "short_momentum_net_drop_rebound_v4_4"
    / "code"
    / "run_v4_4_resumable_campaign.py"
)
PLAN_ROOT = (
    PROJECT_ROOT
    / "research_variants"
    / "short_momentum_net_drop_rebound_v4_4"
    / "plans"
    / "generated_ai_cycle_20260806"
)
CAMPAIGN_ROOT = (
    PROJECT_ROOT
    / "results"
    / "campaigns"
    / "v4_4_positive_entry_signal_repair_20260805"
)
SESSION_ROOT = PROJECT_ROOT / "results" / "ai_exploration" / "k200_leap_grid_cycle_20260806"


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def available_memory_mb() -> int:
    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.available_physical // (1024 * 1024))


def run(command: list[str], *, retry: bool = False) -> None:
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode == 0:
        return
    if retry:
        print(json.dumps({"event": "retry_resumable_command", "returncode": result.returncode, "at": utc_now()}), flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def next_round_number() -> int:
    numbers = []
    for path in PLAN_ROOT.glob("continuation_round_*_ai_*_all_window.json"):
        try:
            numbers.append(int(path.name.split("_")[2]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=16) + 1


def stage_paths(round_number: int, phase: str) -> tuple[Path, Path]:
    suffix = phase.replace("-", "_")
    stage_id = f"continuation_round_{round_number:02d}_ai_{suffix}_all_window"
    return PLAN_ROOT / f"{stage_id}.json", CAMPAIGN_ROOT / stage_id


def write_state(state: dict[str, object]) -> None:
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    (SESSION_ROOT / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_round(record: dict[str, object]) -> None:
    path = SESSION_ROOT / "rounds.csv"
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=5.5)
    parser.add_argument("--leap-count", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--minimum-free-memory-mb", type=int, default=4096)
    parser.add_argument("--publish-reserve-minutes", type=int, default=35)
    args = parser.parse_args()

    started = time.time()
    finish_by = started + args.hours * 3600
    compute_until = finish_by - args.publish_reserve_minutes * 60
    sequence = ("generated-leap", "generated-leap", "adaptive-grid")
    sequence_index = 0
    completed: list[dict[str, object]] = []
    last_plan: Path | None = None
    last_stage: Path | None = None
    state: dict[str, object] = {
        "status": "running",
        "started_at": utc_now(),
        "requested_hours": args.hours,
        "compute_until_epoch": compute_until,
        "finish_by_epoch": finish_by,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "minimum_free_memory_mb": args.minimum_free_memory_mb,
        "leap_count": args.leap_count,
        "completed_rounds": completed,
        "parameter_acceptance": "none",
    }
    write_state(state)

    while time.time() < compute_until:
        while available_memory_mb() < args.minimum_free_memory_mb:
            state["status"] = "waiting_for_memory"
            state["available_memory_mb"] = available_memory_mb()
            state["updated_at"] = utc_now()
            write_state(state)
            time.sleep(60)
        state["status"] = "running"

        round_number = next_round_number()
        phase = sequence[sequence_index % len(sequence)]
        sequence_index += 1
        count = args.leap_count if phase == "generated-leap" else 0
        build_command = [
            str(PYTHON), str(BUILDER), "--round", str(round_number), "--phase", phase,
            "--seed", str(440600 + round_number),
        ]
        if count:
            build_command.extend(["--count", str(count)])
        run(build_command)
        plan_path, stage_root = stage_paths(round_number, phase)

        runner_base = [
            str(PYTHON), str(RUNNER), "--plan", str(plan_path), "--output", str(stage_root),
            "--workers", str(args.workers), "--batch-size", str(args.batch_size),
            "--minimum-free-memory-mb", str(args.minimum_free_memory_mb),
        ]
        run(runner_base + ["--validate-only"])
        if available_memory_mb() < args.minimum_free_memory_mb:
            raise RuntimeError("memory fell below the frozen floor after validation")
        round_started = time.time()
        run(runner_base, retry=True)
        run([str(PYTHON), str(ANALYZER), "--stage", str(stage_root)])
        elapsed = time.time() - round_started
        summary = json.loads((stage_root / "compact_analysis" / "report.json").read_text(encoding="utf-8"))
        record = {
            "round": round_number,
            "phase": phase,
            "stage_id": stage_root.name,
            "coordinate_count": int(summary["coordinate_count"]),
            "trade_count": int(summary["trade_count"]),
            "cost_positive_coordinate_count": int(summary["cost_positive_coordinate_count"]),
            "scenario_1_qualified_coordinate_count": int(summary["scenario_1_qualified_coordinate_count"]),
            "elapsed_seconds": round(elapsed, 3),
            "available_memory_mb": available_memory_mb(),
            "completed_at": utc_now(),
        }
        completed.append(record)
        append_round(record)
        state["completed_rounds"] = completed
        state["last_completed_stage"] = str(stage_root)
        state["available_memory_mb"] = record["available_memory_mb"]
        state["updated_at"] = utc_now()
        write_state(state)
        last_plan, last_stage = plan_path, stage_root

    if last_plan is not None and last_stage is not None:
        state["status"] = "publishing_final_html"
        state["updated_at"] = utc_now()
        write_state(state)
        publish_command = [
            str(PYTHON), str(RUNNER), "--plan", str(last_plan), "--output", str(last_stage),
            "--workers", str(args.workers), "--batch-size", str(args.batch_size),
            "--minimum-free-memory-mb", str(args.minimum_free_memory_mb),
            "--publish-html", "--delivery-mode", "synchronous", "--review-workers", str(args.workers),
        ]
        run(publish_command, retry=True)

    state["status"] = "complete"
    state["completed_at"] = utc_now()
    state["elapsed_seconds"] = round(time.time() - started, 3)
    state["final_published_stage"] = "" if last_stage is None else str(last_stage)
    state["si_migration_started"] = False
    state["si_migration_reason"] = "K200 global convergence was not assumed from finite cyclic exploration."
    write_state(state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
