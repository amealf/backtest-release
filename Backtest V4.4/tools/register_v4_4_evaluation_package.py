from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from build_v4_4_evaluation_framework import (
    PACKAGE_ROOT,
    artifact,
    atomic_csv,
    atomic_json,
    atomic_text,
    browser_summary_script,
    compact_json,
    ensure_hard_link,
    experiment_markdown,
    package_path,
    redirect_html,
    relative_href,
    sha256,
)


def source_path(spec_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else spec_path.parent / path


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("evaluation package spec schema_version must be 1")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(spec["instrument_id"])):
        raise ValueError("instrument_id contains unsupported characters")
    return spec


def projection_from_summary(
    summary: pd.DataFrame,
    *,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parameter_columns = list(spec["columns"]["parameters"])
    metric_mapping = dict(spec["columns"]["metrics"])
    required = set(parameter_columns) | set(metric_mapping.values())
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"parameter summary is missing declared columns: {missing}")
    if "combo_id" not in parameter_columns:
        raise ValueError("columns.parameters must include combo_id")
    if summary["combo_id"].astype(str).duplicated().any():
        raise ValueError("parameter summary contains duplicate combo_id values")

    neutral = summary[parameter_columns].copy()
    for metric_name, source_name in metric_mapping.items():
        neutral[metric_name] = summary[source_name]
    records = json.loads(neutral.to_json(orient="records", date_format="iso"))
    evaluation_id = (
        f"{spec['instrument_id']}_"
        f"{package_path(str(spec['instrument_id']), str(spec['sample_start']), str(spec['sample_end'])).name}"
    )
    rows = [
        {
            "combo_id": str(record["combo_id"]),
            "parameters": {name: record[name] for name in parameter_columns},
            "metrics": {name: record[name] for name in metric_mapping},
        }
        for record in records
    ]
    projection = {
        "schemaVersion": 1,
        "evaluationId": evaluation_id,
        "instrumentId": str(spec["instrument_id"]),
        "displayName": str(spec["display_name"]),
        "sampleStart": str(spec["sample_start"]),
        "sampleEnd": str(spec["sample_end"]),
        "candidateSetId": str(spec["candidate_set_id"]),
        "rowCount": len(rows),
        "rows": rows,
    }
    return neutral, projection


def update_catalog(entry: dict[str, Any]) -> None:
    catalog_path = PACKAGE_ROOT / "catalog.json"
    catalog = (
        json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog_path.exists()
        else {"schema_version": 1, "status": "complete", "evaluations": []}
    )
    evaluations = [
        item
        for item in catalog["evaluations"]
        if item["evaluation_id"] != entry["evaluation_id"]
    ]
    evaluations.append(entry)
    evaluations.sort(key=lambda item: (item["instrument_id"], item["sample_start"], item["sample_end"]))
    catalog["evaluations"] = evaluations
    atomic_json(catalog_path, catalog)
    atomic_text(
        PACKAGE_ROOT / "catalog.js",
        "window.V4_4_EVALUATION_CATALOG=" + compact_json(evaluations) + ";\n",
    )


def register(spec_path: Path) -> Path:
    spec = load_spec(spec_path)
    sources = {
        name: source_path(spec_path, value)
        for name, value in spec["source"].items()
    }
    root = package_path(
        str(spec["instrument_id"]),
        str(spec["sample_start"]),
        str(spec["sample_end"]),
    )
    manifest_path = root / "evaluation_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"evaluation package already exists: {manifest_path}")

    summary = pd.read_csv(sources["parameter_summary"], low_memory=False)
    neutral, projection = projection_from_summary(summary, spec=spec)
    parameter_summary = root / "parameter_summary.csv"
    browser_summary = root / "browser_summaries" / f"{spec['candidate_set_id']}.js"
    trade_records = root / "trade_records" / "trades.csv"
    trade_review = root / "trade_review" / "index.html"
    experiment_record = root / "EXPERIMENT.md"

    atomic_csv(parameter_summary, neutral)
    atomic_text(browser_summary, browser_summary_script(projection))
    ensure_hard_link(sources["trade_records"], trade_records)
    atomic_text(
        trade_review,
        redirect_html(
            relative_href(trade_review.parent, sources["trade_review"]),
            f"V4.4 {spec['display_name']}逐笔分析",
        ),
    )
    atomic_text(
        experiment_record,
        experiment_markdown(
            display_name=str(spec["display_name"]),
            start=str(spec["sample_start"]),
            end=str(spec["sample_end"]),
            current_role=str(spec["experiment"]["role"]),
            result_description=str(spec["experiment"]["description"]),
        ),
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "evaluation_id": projection["evaluationId"],
        "identity": {
            "instrument_id": str(spec["instrument_id"]),
            "display_name": str(spec["display_name"]),
            "sample_start": str(spec["sample_start"]),
            "sample_end": str(spec["sample_end"]),
            "timezone": str(spec["timezone"]),
            "bar_seconds": int(spec["bar_seconds"]),
            "directory_naming_rule": "instrument_id/evaluation_start__evaluation_end",
        },
        "candidate_set_id": str(spec["candidate_set_id"]),
        "role_policy": "experiment roles are declared in experiment records and comparison plans",
        "artifacts": {
            "parameter_summary": artifact(parameter_summary),
            "browser_summary": artifact(browser_summary),
            "trade_records": {
                **artifact(trade_records),
                "storage": "same-volume hard link to immutable source",
            },
            "trade_review": artifact(trade_review),
            "experiment_record": artifact(experiment_record),
        },
        "provenance": {
            "registration_spec": str(spec_path.resolve()),
            "source_parameter_summary": {
                "path": str(sources["parameter_summary"].resolve()),
                "sha256": sha256(sources["parameter_summary"]),
            },
            "source_trade_records": {
                "path": str(sources["trade_records"].resolve()),
                "sha256": sha256(sources["trade_records"]),
            },
            "source_trade_review": str(sources["trade_review"].resolve()),
        },
    }
    atomic_json(manifest_path, manifest)
    update_catalog(
        {
            "evaluation_id": projection["evaluationId"],
            "instrument_id": str(spec["instrument_id"]),
            "display_name": str(spec["display_name"]),
            "sample_start": str(spec["sample_start"]),
            "sample_end": str(spec["sample_end"]),
            "manifest": relative_href(PACKAGE_ROOT, manifest_path),
            "browser_summary": relative_href(PACKAGE_ROOT, browser_summary),
            "trade_review": relative_href(PACKAGE_ROOT, trade_review),
        }
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register one completed instrument/time evaluation as a V4.4 result package."
    )
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    print(register(args.spec).resolve())


if __name__ == "__main__":
    main()
