"""Versioned, idempotent first-use research-data preparation for V4.4."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .audit_report import build_audit_report
from .low_activity import FILTER_RULE_VERSION, detect_low_activity, json_ready, load_15s_bars


PIPELINE_VERSION = "research_data_preparation_v4_4_confirmed_low_activity_gate_20260806"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = ROOT / "project_management" / "05_domains" / "data" / "PROCESSED_DATASETS.json"
DEFAULT_PLOTLY = ROOT / "runtime_inputs" / "templates" / "plotly.min.js"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_hash(value: Any) -> str:
    text = json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rule_contract(instrument: str) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "low_activity_rule_version": FILTER_RULE_VERSION,
        "instrument": instrument.upper(),
        "extreme_tick_rule": {
            "method": "same_bar_immediate_tick_recovery",
            "preceding_trade_deviation": 0.03,
            "anchor_return_tolerance": 0.001,
            "maximum_recovery_seconds": 1,
        },
        "universal_low_volume": {
            "reference_hours": 84,
            "positive_volume_median_ratio": 0.20,
            "continuous_minutes": 30,
            "minimum_low_share": 1.0,
            "end_policy": "first_high_volume_atom",
            "baseline_lifecycle": {
                "suspected": "pending low-volume atoms remain baseline-eligible and do not block entries",
                "transient_recovery": "the first normal 15-second bar cancels the pending run with no strategy effect",
                "confirmed": "at 30 continuous minutes, exclude the run from its first atom for all later baseline calculations and block new entries until the first normal bar",
            },
        },
        "k200_mechanism": {
            "price_lock": "30 complete minutes; >=10 traded minutes; <=4 ticks; TR and range <=5% of earlier 84h median",
            "circuit": "internal real-trade separation exactly 30 minutes; 20-minute halt plus 10-minute call auction",
            "extended_pause": "display-only mechanism candidate because OHLCV cannot disambiguate the cause; remains eligible for universal low-volume detection",
            "universal_input_policy": "remove only mechanism intervals with apply_to_baseline=true; retain display-only candidates in universal input",
        },
        "baseline_sampling_contract": {
            "marker_generation_is_policy_neutral": True,
            "default_baseline_sampling_policy": "confirmed_low_activity_gate",
            "supported_baseline_sampling_policies": {
                "all_window": (
                    "use every finite TR15 atom inside one continuous segment; "
                    "baseline_excluded remains audit and chart-coloring evidence"
                ),
                "exclude_marked": (
                    "use finite TR15 atoms inside one continuous segment only when "
                    "baseline_available_from is known and no later than the current "
                    "calculation time; recovered pending atoms become available at the "
                    "recovery confirmation time; confirmed exclusions never become available"
                ),
                "confirmed_low_activity_gate": (
                    "pending atoms have no strategy effect; when the run confirms, use "
                    "baseline_excluded_from to remove it from its first atom in all later "
                    "baseline calculations and block new entries until the first normal atom"
                ),
            },
            "synthetic_bar_policy": (
                "synthetic status is independent from baseline_excluded; finite synthetic TR15 "
                "atoms remain eligible unless a future explicitly approved policy says otherwise"
            ),
        },
    }


def implementation_sha256() -> str:
    files = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("low_activity.py"),
        Path(__file__).resolve().with_name("audit_report.py"),
    )
    return _json_hash({path.name: sha256_file(path) for path in files})


def default_output_dir(source: Path, source_sha256: str, implementation_hash: str) -> Path:
    return (
        source.parent
        / "research_preparation"
        / f"{source.stem}_{source_sha256[:12]}_{FILTER_RULE_VERSION}_{implementation_hash[:8]}"
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _extreme_cleaning_contract(
    source: Path,
    source_sha256: str,
    extreme_audit: Path | None,
    *,
    allow_legacy_preexisting_source: bool,
) -> dict[str, Any]:
    if extreme_audit is None:
        if not allow_legacy_preexisting_source:
            raise ValueError(
                "a new dataset requires an immediate-recovery data_preparation_audit.json before low-activity preparation"
            )
        return {
            "status": "legacy_preexisting_source",
            "passed": False,
            "limitation": "This source predates the required immediate-recovery rule. New datasets may not use this exception.",
        }
    payload = json.loads(extreme_audit.read_text(encoding="utf-8"))
    transient = payload.get("transient_tail_filter", {})
    output = payload.get("outputs", {}).get("session_filled_15s", {})
    if not payload.get("passed"):
        raise ValueError("extreme-move preparation audit did not pass")
    if not transient.get("enabled") or transient.get("method") != "same_bar_immediate_tick_recovery":
        raise ValueError("extreme-move audit does not prove the required immediate-recovery rule")
    if str(output.get("sha256", "")).lower() != source_sha256.lower():
        raise ValueError("extreme-move audit output hash does not match the low-activity source")
    audited_source_path = Path(output.get("path", ""))
    source_path_relocated = audited_source_path.resolve() != source.resolve()
    return {
        "status": "passed",
        "passed": True,
        "audit_path": str(extreme_audit.resolve()),
        "audit_sha256": sha256_file(extreme_audit),
        "audited_source_path": str(audited_source_path),
        "runtime_source_path": str(source),
        "source_path_relocated": source_path_relocated,
        "filtered_tick_count": int(transient.get("filtered_tick_count", 0)),
        "method": transient.get("method"),
        "parameters": {
            "tick_deviation": transient.get("tick_deviation"),
            "recovery_tolerance": transient.get("recovery_tolerance"),
            "max_recovery_seconds": transient.get("max_recovery_seconds"),
        },
    }


def _artifact_paths(
    manifest: dict[str, Any], manifest_path: Path
) -> list[tuple[Path, str]]:
    output: list[tuple[Path, str]] = []
    for item in manifest.get("artifacts", {}).values():
        if isinstance(item, dict) and item.get("path") and item.get("sha256"):
            path = Path(item["path"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            output.append((path, str(item["sha256"])))
    return output


def validate_manifest(
    manifest_path: Path,
    *,
    source_sha256: str,
    contract_hash: str,
    implementation_hash: str,
) -> tuple[bool, dict[str, Any] | None, str]:
    if not manifest_path.is_file():
        return False, None, "manifest_missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None, "manifest_unreadable"
    if manifest.get("status") != "complete":
        return False, manifest, "status_not_complete"
    if str(manifest.get("source_sha256", "")).lower() != source_sha256.lower():
        return False, manifest, "source_hash_changed"
    if manifest.get("rule_contract_sha256") != contract_hash:
        return False, manifest, "rule_contract_changed"
    if manifest.get("implementation_sha256") != implementation_hash:
        return False, manifest, "implementation_changed"
    for path, expected in _artifact_paths(manifest, manifest_path):
        if not path.is_file():
            return False, manifest, f"artifact_missing:{path.name}"
        if sha256_file(path).lower() != expected.lower():
            return False, manifest, f"artifact_hash_changed:{path.name}"
    return True, manifest, "valid"


def _registry_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "datasets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _update_registry(registry: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    payload = _registry_payload(registry)
    key = str(manifest["dataset_id"])
    manifest_root = manifest_path.resolve().parent

    def repository_relative(value: str | Path, *, base: Path = manifest_root) -> str:
        path = Path(value)
        resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
        try:
            return resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return Path(os.path.relpath(resolved, registry.resolve().parent)).as_posix()

    payload.setdefault("datasets", {})[key] = {
        "instrument": manifest["instrument"],
        "source_path": repository_relative(manifest["source_path"]),
        "source_sha256": manifest["source_sha256"],
        "pipeline_version": manifest["pipeline_version"],
        "prepared_identity": manifest["prepared_identity"],
        "rule_contract_sha256": manifest["rule_contract_sha256"],
        "implementation_sha256": manifest["implementation_sha256"],
        "status": manifest["status"],
        "manifest_path": repository_relative(manifest_path),
        "report_path": repository_relative(
            manifest["artifacts"]["report_index"]["path"]
        ),
        "processed_at": manifest["processed_at"],
        "extreme_cleaning_status": manifest["extreme_cleaning"]["status"],
    }
    payload["updated_at"] = _utc_now()
    _atomic_json(registry, payload)


def prepare_dataset(
    source: Path,
    instrument: str,
    *,
    output_dir: Path | None = None,
    extreme_audit: Path | None = None,
    plotly_source: Path = DEFAULT_PLOTLY,
    registry: Path = DEFAULT_REGISTRY,
    allow_legacy_preexisting_source: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha256 = sha256_file(source)
    contract = rule_contract(instrument)
    contract_hash = _json_hash(contract)
    implementation_hash = implementation_sha256()
    output = (output_dir or default_output_dir(source, source_sha256, implementation_hash)).resolve()
    manifest_path = output / "data_preparation_manifest.json"
    if not force:
        valid, manifest, reason = validate_manifest(
            manifest_path,
            source_sha256=source_sha256,
            contract_hash=contract_hash,
            implementation_hash=implementation_hash,
        )
        if valid and manifest is not None:
            _update_registry(registry, manifest_path, manifest)
            return {
                "status": "reused",
                "reason": reason,
                "manifest_path": str(manifest_path),
                "manifest": manifest,
            }

    extreme = _extreme_cleaning_contract(
        source,
        source_sha256,
        extreme_audit,
        allow_legacy_preexisting_source=allow_legacy_preexisting_source,
    )
    if extreme_audit is not None:
        extreme["audit_path"] = Path(
            os.path.relpath(extreme_audit.resolve(), output)
        ).as_posix()
    extreme["runtime_source_path"] = Path(
        os.path.relpath(source, output)
    ).as_posix()
    bars = load_15s_bars(source)
    result = detect_low_activity(bars, instrument)
    output.mkdir(parents=True, exist_ok=True)
    atoms_path = output / "baseline_filter_atoms.csv"
    atoms_temp = atoms_path.with_name(atoms_path.name + ".tmp")
    result.atoms[[
        "datetime", "universal_low_volume_excluded", "k200_price_lock_excluded",
        "k200_circuit_breaker_excluded", "baseline_excluded", "filter_event_ids",
        "filter_reason_codes", "volume_threshold", "low_volume_atom",
        "low_activity_state", "pending_buffer_start", "pending_buffer_count",
        "buffer_reinserted", "buffer_confirmed_excluded",
        "recovery_confirmation_time", "baseline_available_from",
        "low_activity_confirmation_time", "baseline_excluded_from",
        "confirmed_low_activity_active",
        "eligible_if_excluding_marked",
    ]].to_csv(atoms_temp, index=False, encoding="utf-8")
    os.replace(atoms_temp, atoms_path)
    events_path = output / "baseline_filter_events.json"
    _atomic_json(events_path, {"schema_version": 2, "events": result.events})
    report_output = output / "low_activity_report"
    report_audit = build_audit_report(result, report_output, plotly_source)
    report_audit["index"] = Path(report_audit["index"]).relative_to(output).as_posix()
    def portable_artifact(path: Path, **extra: Any) -> dict[str, Any]:
        return {
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path),
            **extra,
        }

    artifacts = {
        "filter_atoms": portable_artifact(atoms_path, rows=int(len(result.atoms))),
        "filter_events": portable_artifact(events_path, events=int(len(result.events))),
        "report_index": portable_artifact(report_output / "index.html"),
        "report_universal": portable_artifact(report_output / "sections" / "universal.js"),
        "report_mechanism": portable_artifact(report_output / "sections" / "mechanism.js"),
        "report_plotly": portable_artifact(report_output / "assets" / "plotly.min.js"),
    }
    manifest = {
        "schema_version": 5,
        "status": "complete",
        "dataset_id": f"{instrument.upper()}_v4_4_{source_sha256[:16]}",
        "prepared_identity": f"v4_4_confirmed_low_activity_gate_{source_sha256[:16]}_{contract_hash[:12]}_{implementation_hash[:12]}",
        "instrument": instrument.upper(),
        "source_path": Path(os.path.relpath(source, output)).as_posix(),
        "source_sha256": source_sha256,
        "source_bytes": source.stat().st_size,
        "pipeline_version": PIPELINE_VERSION,
        "rule_contract": contract,
        "rule_contract_sha256": contract_hash,
        "implementation_sha256": implementation_hash,
        "processed_at": _utc_now(),
        "extreme_cleaning": extreme,
        "low_activity_summary": result.summary,
        "report_loading_contract": {
            "startup": "HTML shell and summary only",
            "section_payloads": "loaded after the matching title is expanded",
            "plotly": "loaded after the first title is expanded",
        },
        "report_audit": report_audit,
        "artifacts": artifacts,
    }
    _atomic_json(manifest_path, manifest)
    _update_registry(registry, manifest_path, manifest)
    return {
        "status": "generated",
        "reason": "first_use_or_contract_change",
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one dataset before research use.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--output")
    parser.add_argument("--extreme-audit")
    parser.add_argument("--plotly", default=str(DEFAULT_PLOTLY))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--allow-legacy-preexisting-source", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_dataset(
        Path(args.source),
        args.instrument,
        output_dir=Path(args.output) if args.output else None,
        extreme_audit=Path(args.extreme_audit) if args.extreme_audit else None,
        plotly_source=Path(args.plotly),
        registry=Path(args.registry),
        allow_legacy_preexisting_source=args.allow_legacy_preexisting_source,
        force=args.force,
    )
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
