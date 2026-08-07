from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

from write_k200_download_readme import write_readme


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_RUN = Path(
    r"F:\Backtest test 6.11\02_DATA_AND_AUDITS\market_data"
    r"\k200_historical_ticks_supplements"
    r"\k200_postroll_supplement_20260723T024400_to_20260728T161430_20260728_151540"
)
DEFAULT_BASE_15S = (
    DEFAULT_PARENT_RUN
    / "data_extended"
    / "k200_clean_15s_session_filled_through_20260728.csv"
)
DEFAULT_OUTPUT_BASE = Path(
    r"F:\Backtest test 6.11\02_DATA_AND_AUDITS\market_data"
    r"\k200_historical_ticks_supplements"
)
SUPPLEMENT_RUNNER = Path(
    r"D:\CodexHome\worktrees\v3-15s-branch\research_variants"
    r"\short_momentum_net_drop_rebound_v3_15s_branch\code"
    r"\download_k200_postroll_supplement.py"
)
PIPELINE_SOURCE = Path(
    r"D:\Code\backtest-release\Backtest v2 ratio"
    r"\download_k200_ibkr_historical_ticks.py"
)
CLEAN_SOURCE = Path(
    r"D:\Code\backtest-release\Backtest v2 ratio\research_variants"
    r"\short_momentum_15s_equivalence\code\build_clean_k200_subminute_pair.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("k200_incremental_runner", SUPPLEMENT_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load supplement runner: {SUPPLEMENT_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def merge_extended_15s(runner, output: Path, base_source: Path, supplement_source: Path):
    extended_dir = output / "data_extended"
    extended_dir.mkdir(parents=True, exist_ok=True)
    destination = extended_dir / "k200_clean_15s_session_filled_latest.csv"
    temporary = destination.with_name(f".{destination.name}.tmp")
    counts = {"base": 0, "supplement": 0}
    first_epochs: dict[str, int | None] = {"base": None, "supplement": None}
    last_epochs: dict[str, int | None] = {"base": None, "supplement": None}
    duplicate_epoch_count = 0
    nonmonotonic_epoch_count = 0
    previous_epoch: int | None = None
    fieldnames: list[str] | None = None

    with temporary.open("w", newline="", encoding="utf-8-sig") as target:
        writer: csv.DictWriter | None = None
        for label, source in (("base", base_source), ("supplement", supplement_source)):
            with source.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                current_fields = list(reader.fieldnames or [])
                if fieldnames is None:
                    fieldnames = current_fields
                    writer = csv.DictWriter(target, fieldnames=fieldnames)
                    writer.writeheader()
                elif current_fields != fieldnames:
                    raise ValueError("base and supplement 15-second schemas differ")
                assert writer is not None
                for row in reader:
                    epoch = int(row["epoch_seconds"])
                    if first_epochs[label] is None:
                        first_epochs[label] = epoch
                    last_epochs[label] = epoch
                    counts[label] += 1
                    if previous_epoch is not None:
                        duplicate_epoch_count += int(epoch == previous_epoch)
                        nonmonotonic_epoch_count += int(epoch <= previous_epoch)
                    previous_epoch = epoch
                    writer.writerow(row)
    temporary.replace(destination)

    boundary_gap_seconds = (
        int(first_epochs["supplement"]) - int(last_epochs["base"])
        if first_epochs["supplement"] is not None and last_epochs["base"] is not None
        else None
    )
    audit = {
        "schema_version": 1,
        "status": "active_v4_4_extension_candidate",
        "base_source": {
            "path": str(base_source),
            "sha256": runner.sha256_file(base_source),
            "bar_count": counts["base"],
            "first_epoch": first_epochs["base"],
            "last_epoch": last_epochs["base"],
        },
        "supplement_source": {
            "path": str(supplement_source),
            "sha256": runner.sha256_file(supplement_source),
            "bar_count": counts["supplement"],
            "first_epoch": first_epochs["supplement"],
            "last_epoch": last_epochs["supplement"],
        },
        "extended_output": {
            "path": str(destination),
            "sha256": runner.sha256_file(destination),
            "bar_count": counts["base"] + counts["supplement"],
        },
        "boundary_gap_seconds": boundary_gap_seconds,
        "duplicate_epoch_count": duplicate_epoch_count,
        "nonmonotonic_epoch_count": nonmonotonic_epoch_count,
    }
    audit["passed"] = (
        counts["base"] > 0
        and counts["supplement"] > 0
        and boundary_gap_seconds is not None
        and boundary_gap_seconds > 0
        and boundary_gap_seconds % 15 == 0
        and duplicate_epoch_count == 0
        and nonmonotonic_epoch_count == 0
    )
    runner.write_json(extended_dir / "extended_15s_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("extended 15-second series audit failed")
    return audit


def build_clean_derivative(
    runner,
    output: Path,
    pipeline,
    cleaner,
    start_epoch: int,
    end_epoch: int,
    raw_tick_count: int,
):
    clean_dir = output / "data_clean"
    clean_dir.mkdir(parents=True, exist_ok=False)
    observed = clean_dir / "k200_clean_15s_trade_only.csv"
    filled = clean_dir / "k200_clean_15s_session_filled.csv"
    minute = clean_dir / "k200_clean_1min_from_15s_session_filled.csv"
    tail_filter = cleaner.ImmediateTickRecoveryFilter(
        Decimal("0.03"), Decimal("0.001"), 1
    )
    (
        source_positive_ticks,
        used_ticks,
        observed_count,
        invalid_rows,
        filtered_records,
    ) = cleaner.build_clean_observed_15s(
        output / "k200_main_historical_ticks.csv.gz",
        observed,
        list(pipeline.BAR_COLUMNS),
        tail_filter,
    )
    fill_stats = pipeline.export_session_filled_bars(
        observed, filled, 15, start_epoch, end_epoch
    )
    minute_stats = cleaner.build_parent_60s(filled, minute)
    observed_audit = runner.csv_bar_audit(observed)
    filled_audit = runner.csv_bar_audit(filled)
    minute_audit = runner.csv_bar_audit(minute)
    pair_audit = runner.validate_clean_15_to_60(filled, minute)
    filtered_tick_count = sum(
        int(item["removed_tick_count"]) for item in filtered_records
    )
    audit = {
        "schema_version": 1,
        "source_tick_file": str(output / "k200_main_historical_ticks.csv.gz"),
        "source_tick_sha256": runner.sha256_file(
            output / "k200_main_historical_ticks.csv.gz"
        ),
        "source_tick_count": raw_tick_count,
        "source_positive_tick_count": source_positive_ticks,
        "valid_tick_count_used": used_ticks,
        "invalid_tick_count_excluded": len(invalid_rows),
        "invalid_ticks": invalid_rows,
        "transient_tail_filter": {
            "enabled": True,
            "method": "same_bar_immediate_tick_recovery",
            "tick_deviation": "0.03",
            "recovery_tolerance": "0.001",
            "max_recovery_seconds": 1,
            "filtered_tick_count": filtered_tick_count,
            "records": filtered_records,
        },
        "outputs": {
            "observed_15s": {**observed_audit, "builder_count": observed_count},
            "session_filled_15s": {**filled_audit, "builder_stats": fill_stats},
            "paired_1min": {**minute_audit, "builder_stats": minute_stats},
        },
        "pair_15s_to_1min": pair_audit,
    }
    audit["passed"] = (
        source_positive_ticks + len(invalid_rows) == raw_tick_count
        and used_ticks + filtered_tick_count == source_positive_ticks
        and observed_audit["passed"]
        and filled_audit["passed"]
        and minute_audit["passed"]
        and pair_audit["passed"]
    )
    runner.write_json(clean_dir / "data_preparation_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("clean supplement audit failed")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the next K200 tick supplement and extend the clean 15-second candidate."
    )
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT_RUN)
    parser.add_argument("--base-15s", type=Path, default=DEFAULT_BASE_15S)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--end")
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int)
    parser.add_argument("--max-ticks", type=int, default=1000)
    parser.add_argument("--pace-seconds", type=float, default=10.25)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--timeout-retries", type=int, default=32)
    parser.add_argument("--max-pages", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = load_runner()
    base_source = args.base_15s.resolve()
    if not base_source.is_file():
        raise FileNotFoundError(base_source)

    runner.CURRENT_V3_15S_SOURCE = base_source
    runner.DEFAULT_PROTECTED_PATHS = (
        base_source,
        PROJECT_ROOT / "runtime_inputs" / "market_data" / "k200_clean_15s_session_filled.csv",
    )
    original_merge = runner.build_extended_15s_series
    original_clean = runner.build_clean_derivative
    original_write_json = runner.write_json
    runner.build_extended_15s_series = (
        lambda output, current_source, supplement_source: merge_extended_15s(
            runner, output, base_source, supplement_source
        )
    )
    runner.build_clean_derivative = (
        lambda output, pipeline, cleaner, start_epoch, end_epoch, raw_tick_count:
        build_clean_derivative(
            runner,
            output,
            pipeline,
            cleaner,
            start_epoch,
            end_epoch,
            raw_tick_count,
        )
    )
    runner.write_json = lambda path, payload: (
        original_write_json(path, payload),
        write_readme(Path(path).parent)
        if Path(path).name == "run_manifest.json"
        else None,
    )[0]
    try:
        if args.resume_dir:
            manifest_path = args.resume_dir.resolve() / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            request = manifest["request"]
            history = request.setdefault(
                "max_ticks_per_request_history",
                [int(request["max_ticks_per_request"])],
            )
            if args.max_ticks not in history:
                history.append(args.max_ticks)
            timeout_history = request.setdefault(
                "request_timeout_seconds_history",
                [],
            )
            if args.request_timeout not in timeout_history:
                timeout_history.append(args.request_timeout)
            original_write_json(manifest_path, manifest)
        run_args = argparse.Namespace(
            parent_run=args.parent_run,
            output_base=args.output_base,
            pipeline_source=PIPELINE_SOURCE,
            clean_source=CLEAN_SOURCE,
            start=None,
            end=args.end,
            resume_dir=args.resume_dir,
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            max_ticks=args.max_ticks,
            pace_seconds=args.pace_seconds,
            request_timeout=args.request_timeout,
            empty_advance_seconds=21600,
            progress_every=5,
            max_pages=args.max_pages,
            finalize_only=False,
        )
        timeout_failures = 0
        while True:
            try:
                output = runner.run(run_args)
                break
            except TimeoutError:
                timeout_failures += 1
                if args.resume_dir is None or timeout_failures > args.timeout_retries:
                    raise
                fallback_args = argparse.Namespace(**vars(run_args))
                fallback_args.resume_dir = args.resume_dir
                fallback_args.max_ticks = min(500, args.max_ticks)
                fallback_args.max_pages = 1
                fallback_output = runner.run(fallback_args)
                fallback_manifest = json.loads(
                    (fallback_output / "run_manifest.json").read_text(encoding="utf-8")
                )
                if fallback_manifest.get("status") == "complete":
                    output = fallback_output
                    break
                run_args.resume_dir = args.resume_dir
    finally:
        runner.build_extended_15s_series = original_merge
        runner.build_clean_derivative = original_clean
        runner.write_json = original_write_json
    print(json.dumps({"output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
