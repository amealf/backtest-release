from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_variants.short_momentum_net_drop_rebound_v4_4.data_preparation.prepare_dataset import (
    prepare_dataset,
)


ACTIVE_DATA = ROOT / "runtime_inputs" / "market_data" / "k200_clean_15s_session_filled.csv"
ACTIVE_AUDIT = ROOT / "runtime_inputs" / "market_data" / "data_preparation_audit.json"
PREPARATION_DIR = ROOT / "runtime_inputs" / "data_preparation"
RUNTIME_MANIFEST = ROOT / "runtime_inputs" / "RUNTIME_INPUTS.json"
PROFILE = (
    ROOT
    / "research_variants"
    / "short_momentum_net_drop_rebound_v4_4"
    / "instrument_profiles"
    / "k200m.json"
)
ATTESTATION = PROFILE.parent / "policies" / "k200m_preparation_policy_attestation_v2.json"
SOURCE_MANIFEST = PROFILE.parents[1] / "SOURCE_MANIFEST.json"
ORIGINAL_AUDIT = Path(
    r"F:\Backtest test 6.11\02_DATA_AND_AUDITS\market_data\k200_historical_ticks"
    r"\k200_ticks_20260523_to_20260723_20260723_014534"
    r"\data_clean_immediate_tick_recovery_filter_v2_20260729\data_preparation_audit.json"
)
PREVIOUS_SUPPLEMENT = Path(
    r"F:\Backtest test 6.11\02_DATA_AND_AUDITS\market_data"
    r"\k200_historical_ticks_supplements"
    r"\k200_postroll_supplement_20260723T024400_to_20260728T161430_20260728_151540"
)
PREVIOUS_CLEAN = (
    PREVIOUS_SUPPLEMENT
    / "data_clean_immediate_tick_recovery_20260807"
    / "k200_clean_15s_session_filled.csv"
)
PREVIOUS_AUDIT = PREVIOUS_CLEAN.parent / "data_preparation_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def csv_boundary(path: Path) -> tuple[list[str], int, int, int]:
    count = 0
    first_epoch: int | None = None
    last_epoch: int | None = None
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            epoch = int(row["epoch_seconds"])
            if first_epoch is None:
                first_epoch = epoch
            last_epoch = epoch
            count += 1
    if first_epoch is None or last_epoch is None:
        raise ValueError(f"empty 15-second source: {path}")
    return fields, count, first_epoch, last_epoch


def append_csv_bytes(target, source: Path, expected_header: bytes, *, include_header: bool) -> None:
    with source.open("rb") as handle:
        header = handle.readline()
        if header.lstrip(b"\xef\xbb\xbf").rstrip(b"\r\n") != expected_header:
            raise ValueError(f"15-second schema differs: {source}")
        if include_header:
            target.write(header)
        shutil.copyfileobj(handle, target, length=1024 * 1024)


def activate(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    run_manifest = read_json(run_dir / "run_manifest.json")
    if run_manifest.get("status") != "complete":
        raise ValueError("incremental tick run is not complete")
    if not read_json(run_dir / "data_quality_report.json").get("passed"):
        raise ValueError("incremental tick quality report did not pass")

    new_clean = run_dir / "data_clean" / "k200_clean_15s_session_filled.csv"
    new_audit = run_dir / "data_clean" / "data_preparation_audit.json"
    sources = (ACTIVE_DATA, PREVIOUS_CLEAN, new_clean)
    audits = (ORIGINAL_AUDIT, PREVIOUS_AUDIT, new_audit)
    boundaries = [csv_boundary(path) for path in sources]
    component_identities = [
        {
            "data_path": str(data_path),
            "data_sha256": sha256_file(data_path),
            "data_size_bytes": data_path.stat().st_size,
            "bar_count": boundary[1],
            "first_epoch": boundary[2],
            "last_epoch": boundary[3],
            "audit_path": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
        }
        for data_path, audit_path, boundary in zip(sources, audits, boundaries)
    ]
    if not all(boundary[0] == boundaries[0][0] for boundary in boundaries[1:]):
        raise ValueError("15-second source schemas differ")
    for left, right in zip(boundaries, boundaries[1:]):
        gap = right[2] - left[3]
        if gap <= 0 or gap % 15:
            raise ValueError("15-second append boundary is invalid")

    with ACTIVE_DATA.open("rb") as handle:
        expected_header = handle.readline().lstrip(b"\xef\xbb\xbf").rstrip(b"\r\n")
    temporary_data = ACTIVE_DATA.with_name(ACTIVE_DATA.name + ".tmp")
    with temporary_data.open("wb") as target:
        for index, source in enumerate(sources):
            append_csv_bytes(target, source, expected_header, include_header=index == 0)
    os.replace(temporary_data, ACTIVE_DATA)

    component_audits = [read_json(path) for path in audits]
    filtered_count = sum(
        int(payload.get("transient_tail_filter", {}).get("filtered_tick_count", 0))
        for payload in component_audits
    )
    output_identity = artifact(ACTIVE_DATA)
    composite_audit = {
        "schema_version": 2,
        "status": "complete",
        "method": "incremental_concat_of_independently_cleaned_tick_ranges",
        "components": component_identities,
        "transient_tail_filter": {
            "enabled": True,
            "method": "same_bar_immediate_tick_recovery",
            "tick_deviation": "0.03",
            "recovery_tolerance": "0.001",
            "max_recovery_seconds": 1,
            "filtered_tick_count": filtered_count,
        },
        "outputs": {
            "session_filled_15s": {
                "path": str(ACTIVE_DATA),
                "sha256": output_identity["sha256"],
                "size_bytes": output_identity["size_bytes"],
                "bar_count": sum(item[1] for item in boundaries),
                "first_epoch": boundaries[0][2],
                "last_epoch": boundaries[-1][3],
            }
        },
        "passed": True,
    }
    write_json(ACTIVE_AUDIT, composite_audit)

    prepared = prepare_dataset(
        ACTIVE_DATA,
        "K200",
        output_dir=PREPARATION_DIR,
        extreme_audit=ACTIVE_AUDIT,
        force=True,
    )["manifest"]
    preparation_manifest = PREPARATION_DIR / "data_preparation_manifest.json"

    attestation = read_json(ATTESTATION)
    attestation["preparation_manifest_sha256"] = sha256_file(preparation_manifest)
    write_json(ATTESTATION, attestation)

    profile = read_json(PROFILE)
    profile["data"].update(artifact(ACTIVE_DATA))
    profile["data"]["preparation_manifest"].update(artifact(preparation_manifest))
    profile["data"]["policy_attestation"].update(artifact(ATTESTATION))
    write_json(PROFILE, profile)

    runtime = read_json(RUNTIME_MANIFEST)
    runtime["runtime_identity"] = "backtest_v4_4_extended_k200_data_20260807"
    runtime_paths = {
        "market_data_15s": ACTIVE_DATA,
        "market_data_cleaning_audit": ACTIVE_AUDIT,
        "data_preparation_manifest": preparation_manifest,
        "baseline_filter_atoms": PREPARATION_DIR / "baseline_filter_atoms.csv",
        "baseline_filter_events": PREPARATION_DIR / "baseline_filter_events.json",
    }
    for key, path in runtime_paths.items():
        runtime["artifacts"][key].update(artifact(path))
    write_json(RUNTIME_MANIFEST, runtime)

    source_manifest = read_json(SOURCE_MANIFEST)
    source_manifest["status"] = "v4_4_confirmed_low_activity_gate_extended_k200_data_20260807"
    source_manifest["runtime_inputs"].update(
        {
            "manifest_sha256": sha256_file(RUNTIME_MANIFEST),
            "manifest_size_bytes": RUNTIME_MANIFEST.stat().st_size,
        }
    )
    source_manifest["data_preparation"].update(
        {
            "manifest_sha256": sha256_file(preparation_manifest),
            "prepared_identity": prepared["prepared_identity"],
            "pipeline_version": prepared["pipeline_version"],
            "manifest_size_bytes": preparation_manifest.stat().st_size,
        }
    )
    profile_artifact = artifact(PROFILE)
    attestation_artifact = artifact(ATTESTATION)
    for key in ("ready_k200_profile", "future_k200_positive_entry_profile"):
        source_manifest["instrument_campaign_contract"][key].update(profile_artifact)
    source_manifest["instrument_campaign_contract"]["policy_contracts"][
        "k200_preparation_attestation"
    ].update(attestation_artifact)
    source_manifest["implementation"]["incremental_tick_update"] = {
        "path": "tools/download_k200_incremental_ticks.py",
        **artifact(ROOT / "tools" / "download_k200_incremental_ticks.py"),
    }
    source_manifest["implementation"]["incremental_15s_activation"] = {
        "path": "tools/activate_k200_incremental_15s.py",
        **artifact(ROOT / "tools" / "activate_k200_incremental_15s.py"),
    }
    source_manifest["implementation"]["market_data_download_readme"] = {
        "path": "tools/write_k200_download_readme.py",
        **artifact(ROOT / "tools" / "write_k200_download_readme.py"),
    }
    write_json(SOURCE_MANIFEST, source_manifest)

    return {
        "active_data": str(ACTIVE_DATA),
        "bar_count": composite_audit["outputs"]["session_filled_15s"]["bar_count"],
        "first_epoch": boundaries[0][2],
        "last_epoch": boundaries[-1][3],
        "data_sha256": sha256_file(ACTIVE_DATA),
        "preparation_manifest": str(preparation_manifest),
        "prepared_identity": prepared["prepared_identity"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activate one completed K200 tick supplement in the V4.4 15-second source."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(activate(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
