"""Build the cumulative V4.4 combined-exit main and trade-review delivery.

Completed stages in the accepted V4.4 major ranking lineage are included even
when minor implementations have different hashes. Hashes still close each
stage's own evidence. A single union-writer lock serializes refreshes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import msvcrt
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from analyze_v4_4_scenario_3_stage import (
    ANALYSIS_IDENTITY,
    LEGACY_V4_MAIN_TEMPLATE_ID,
    LEGACY_V4_MAIN_TEMPLATE_PATH,
    LEGACY_V4_TRADE_DESIGN_PATH,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _apply_cost_adjusted_metrics,
    _augment_trade_lifecycle,
    _combo_lifecycle_summary,
    _cost_model_from_stage_manifest,
    _exit_reason_summary,
    _legacy_v4_main_html,
    _load_stage,
    _rank,
    _records,
    _truthy,
    _trade_lifecycle_summary,
    _validate_trades,
    COST_ADJUSTED_RETURN_KEY,
    K200M_COST_MODEL,
    HIGH_RETURN_VIEWS,
    artifact,
    build_scenario_requirements_delivery,
    main_summary_payload,
    sha256_file,
)
from build_v4_4_review_delivery import (
    SOURCE_COLUMNS,
    build_stage_trade_review,
    refresh_trade_review_shell,
)
from run_v4_4_resumable_campaign import (
    FINGERPRINT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    result_semantics_id,
)
from v4_4_engine import (
    BASELINE_SAMPLING_POLICIES,
    COMBINED_TRADE_AUDIT_SCHEMA_ID,
    COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
    ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
    ENTRY_FILL_CALCULATED_THRESHOLD,
    EXIT_MODE_COMBINED,
    REBOUND_BASELINE_POLICY_ID,
    baseline_filter_id,
    strategy_id,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMPAIGNS_ROOT = REPOSITORY_ROOT / "results" / "campaigns"
DEFAULT_PLANS_ROOT = Path(__file__).resolve().parents[1] / "plans"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "all_completed_union_analysis"

REQUIRED_STRATEGY_IDS = {
    policy: strategy_id(policy, combined_exit=True)
    for policy in BASELINE_SAMPLING_POLICIES
}
REQUIRED_BASELINE_FILTER_IDS = {
    policy: baseline_filter_id(policy) for policy in BASELINE_SAMPLING_POLICIES
}
REQUIRED_RESULT_SEMANTICS_IDS = {
    policy: result_semantics_id(
        ENTRY_FILL_CALCULATED_THRESHOLD,
        ENTRY_EXECUTION_WAIT_NEXT_REAL_TRADE,
        0.0,
        EXIT_MODE_COMBINED,
        policy,
    )
    for policy in BASELINE_SAMPLING_POLICIES
}
REQUIRED_SCENARIO_SCHEMA_ID = (
    "v4_4_scenario_groups_single_select_combined_exit_v3_20260801"
)
REQUIRED_SCENARIO_DEFINITION_SHA256 = (
    "020bb4df6c535ea0e0fd5fff412fc05c39e24d40a9c35a2ada1ccdd7d7ad7af2"
)
UNION_ANALYSIS_IDENTITY = (
    "v4_4_max_completed_w_drop_dual_baseline_sampling_rolling_"
    "all_completed_union_v5_derived_k200m_notional_cost_ranking"
)
UNION_CAMPAIGN_ID = "v4_4_all_completed_combined_union"
UNION_STAGE_ID = "all_completed_union_analysis"
RANKING_MAJOR_VERSION = "V4.4"
RANKING_LINEAGE_ID = "k200m_v4_4_positive_entry_future_lineage"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clean(child) for child in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return value


def _snapshot_id(stage_records: list[dict[str, Any]]) -> str:
    identity = {
        "stages": [
            {
                "campaign_id": row["campaign_id"],
                "stage_id": row["stage_id"],
                "plan_fingerprint": row["plan_fingerprint"],
                "completion_manifest_sha256": row["completion_manifest_sha256"],
            }
            for row in stage_records
        ],
        "generators": {
            "union": sha256_file(Path(__file__)),
            "stage_analysis": sha256_file(
                Path(__file__).resolve().parent / "analyze_v4_4_scenario_3_stage.py"
            ),
            "trade_review": sha256_file(
                Path(__file__).resolve().parent / "build_v4_4_review_delivery.py"
            ),
        },
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _redirect_html(relative_target: str, title: str) -> str:
    encoded_target = json.dumps(relative_target, ensure_ascii=False)
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{title}</title></head><body><p>正在打开当前完整快照…</p><script>"
        f"const target={encoded_target};location.replace(target+location.search+location.hash);"
        "</script></body></html>"
    )


def publish_stable_main_assets(
    output_root: Path,
    delivery_root: Path,
    snapshot_id: str,
) -> dict[str, Any]:
    source = (delivery_root / "analysis_data.js").read_text(encoding="utf-8")
    prefix = "window.V4_ANALYSIS_DATA="
    if not source.startswith(prefix):
        raise ValueError("snapshot main data lacks the expected variable prefix")
    payload = json.loads(source[len(prefix):].rstrip().removesuffix(";"))
    summary = main_summary_payload(
        payload,
        native_trade_route=(
            f"../snapshots/{snapshot_id}/trade_review/index.html?combo_id={{combo_id}}"
        ),
        scenario_requirements_route=(
            f"../snapshots/{snapshot_id}/scenario_requirements/"
            "index.html?scenario={scenario_id}"
        ),
    )
    main_root = output_root / "main"
    main_root.mkdir(parents=True, exist_ok=True)
    data_path = main_root / "analysis_data.js"
    _atomic_text(
        data_path,
        "window.V4_ANALYSIS_DATA="
        + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
    )
    main_html = _legacy_v4_main_html("../../cross_instrument_comparison/index.html")
    index_path = main_root / "index.html"
    alias_path = main_root / "analysis_report.html"
    _atomic_text(index_path, main_html)
    _atomic_text(alias_path, main_html)
    return {
        "index": artifact(index_path),
        "alias": artifact(alias_path),
        "data": artifact(data_path),
        "row_count": len(summary["rows"]),
        "row_field_count": len(summary["rows"][0]) if summary["rows"] else 0,
        "source_data_size_bytes": (delivery_root / "analysis_data.js").stat().st_size,
    }


def _current_trade_review_for_reuse(
    output_root: Path,
    delivery_root: Path,
) -> Path | None:
    pointer_path = output_root / "current_snapshot.json"
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    previous_root = Path(str(pointer.get("snapshot_root", ""))).resolve()
    previous_review = previous_root / "trade_review"
    if previous_root == delivery_root.resolve() or not (
        previous_review / "trade_review_manifest.json"
    ).is_file():
        return None
    return previous_review


def _publish_stable_routes(
    output_root: Path,
    delivery_root: Path,
    snapshot_id: str,
    excluded: list[dict[str, Any]],
) -> dict[str, Any]:
    relative_snapshot = f"snapshots/{snapshot_id}"
    stable_main = publish_stable_main_assets(output_root, delivery_root, snapshot_id)
    stable_index = output_root / "index.html"
    if not stable_index.is_file() or "location.replace" not in stable_index.read_text(
        encoding="utf-8"
    ):
        _atomic_text(
            stable_index,
            _redirect_html("main/index.html", "V4.41 累计总入口"),
        )
    _atomic_text(
        output_root / "analysis_report.html",
        _redirect_html("main/analysis_report.html", "V4.41 累计总入口"),
    )
    _atomic_text(
        output_root / "trade_review" / "index.html",
        _redirect_html(
            f"../{relative_snapshot}/trade_review/index.html", "V4.41 累计逐笔分析"
        ),
    )
    _atomic_text(
        output_root / "scenario_requirements" / "index.html",
        _redirect_html(
            f"../{relative_snapshot}/scenario_requirements/index.html",
            "V4.41 场景要求查看器",
        ),
    )
    for relative in (
        "analysis_manifest.json",
        "completion_manifest.json",
        "source_stages.csv",
        "excluded_stages.csv",
        "duplicate_coordinate_audit.json",
    ):
        source = delivery_root / relative
        target = output_root / relative
        _atomic_text(target, source.read_text(encoding="utf-8"))
    _atomic_text(
        output_root / "trade_review" / "trade_review_manifest.json",
        (delivery_root / "trade_review" / "trade_review_manifest.json").read_text(
            encoding="utf-8"
        ),
    )
    current = {
        "schema_version": 1,
        "status": "complete",
        "union_snapshot_id": snapshot_id,
        "snapshot_root": str(delivery_root.resolve()),
        "stable_main_entry": str((output_root / "index.html").resolve()),
        "stable_main_data": stable_main["data"],
        "stable_trade_entry": str((output_root / "trade_review" / "index.html").resolve()),
        "stable_scenario_requirements_entry": str(
            (output_root / "scenario_requirements" / "index.html").resolve()
        ),
        "excluded_stage_count": len(excluded),
        "published_at": _utc_now(),
    }
    _atomic_json(output_root / "current_snapshot.json", current)
    return current


def _load_complete_snapshot(delivery_root: Path, snapshot_id: str) -> dict[str, Any] | None:
    completion_path = delivery_root / "completion_manifest.json"
    if not completion_path.is_file():
        return None
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "complete"
        or completion.get("union_snapshot_id") != snapshot_id
    ):
        return None
    for record in completion.get("artifacts", {}).values():
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != str(record.get("sha256", "")):
            return None
    review_manifest_path = Path(
        str(completion["artifacts"]["trade_review_manifest"]["path"])
    ).resolve()
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    review_root = review_manifest_path.parent
    for record in review_manifest.get("outputs", []):
        path = Path(str(record.get("path", "")))
        path = path.resolve() if path.is_absolute() else (review_root / path).resolve()
        if not path.is_file() or sha256_file(path) != str(record.get("sha256", "")):
            return None
    return completion


def _refresh_completed_snapshot_trade_shell(
    delivery_root: Path,
    snapshot_id: str,
    peer_review_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    completion_path = delivery_root / "completion_manifest.json"
    if not completion_path.is_file():
        return None
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("union_snapshot_id") != snapshot_id:
        return None
    refreshed = refresh_trade_review_shell(
        delivery_root / "trade_review", **(peer_review_kwargs or {})
    )
    if not refreshed["refreshed"]:
        return refreshed

    analysis_path = delivery_root / "analysis_manifest.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["artifacts"]["trade_level_html"] = artifact(refreshed["index"])
    analysis["artifacts"]["trade_review_manifest"] = artifact(refreshed["manifest"])
    _atomic_json(analysis_path, analysis)

    completion["artifacts"]["analysis_manifest"] = artifact(analysis_path)
    completion["artifacts"]["trade_level_html"] = artifact(refreshed["index"])
    completion["artifacts"]["trade_review_manifest"] = artifact(refreshed["manifest"])
    _atomic_json(completion_path, completion)
    return refreshed


def _trade_review_peer_kwargs(output_root: Path) -> dict[str, Any]:
    path = output_root / "trade_review_peer.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "peer_review_href": str(payload["href"]),
        "peer_review_label": str(payload["label"]),
        "peer_research_contract_id": str(payload["research_contract_id"]),
    }


@contextmanager
def _exclusive_union_writer(
    output_root: Path, timeout_seconds: float | None = None
) -> Iterable[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".v4_4_union.lock"
    handle = lock_path.open("a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
                break
            except OSError as error:
                if deadline is not None and time.monotonic() >= deadline:
                    raise RuntimeError(
                        "another V4.4 union refresh is still writing the cumulative delivery"
                    ) from error
                time.sleep(0.25)
        yield lock_path
    finally:
        if locked:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def _plan_index(plans_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in sorted(plans_root.rglob("*.json")):
        if path.name.endswith(".audit.json"):
            continue
        result.setdefault(sha256_file(path), []).append(path.resolve())
    return result


def _discovery_record(stage_manifest_path: Path, stage_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage_root": str(stage_manifest_path.parent.resolve()),
        "campaign_id": str(stage_manifest.get("campaign_id", "")),
        "stage_id": str(stage_manifest.get("stage_id", "")),
        "stage_status": str(stage_manifest.get("status", "")),
        "version_label": str(stage_manifest.get("version_label", "")),
        "plan_fingerprint": str(stage_manifest.get("plan_fingerprint", "")),
        "input_plan_sha256": str(stage_manifest.get("input_plan_sha256", "")),
        "coordinate_count": int(stage_manifest.get("coordinate_count", 0) or 0),
        "strategy_id": str(stage_manifest.get("strategy_id", "")),
        "result_semantics_id": str(stage_manifest.get("result_semantics_id", "")),
        "raw_output_schema_version": int(stage_manifest.get("schema_version", -1)),
        "plan_fingerprint_schema_version": int(
            stage_manifest.get("plan_fingerprint_schema_version", -1)
        ),
        "trade_audit_schema_version": int(
            stage_manifest.get("trade_audit_schema_version", -1)
        ),
        "trade_audit_schema_id": str(
            stage_manifest.get("trade_audit_schema_id", "")
        ),
        "rebound_baseline_policy_id": str(
            stage_manifest.get("rebound_baseline_policy_id", "")
        ),
        "baseline_sampling_policy": str(
            stage_manifest.get("baseline_sampling_policy", "")
        ),
        "baseline_filter_id": str(stage_manifest.get("baseline_filter_id", "")),
        "scenario_schema_id": str(stage_manifest.get("scenario_schema_id", "")),
        "scenario_definition_sha256": str(
            stage_manifest.get("scenario_definition_sha256", "")
        ),
        "scenario_selection_mode": str(stage_manifest.get("scenario_selection_mode", "")),
        "exit_mode": str(stage_manifest.get("exit_mode", "")),
    }


def _discover_closed_stages(
    campaigns_root: Path, plans_root: Path, output_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not campaigns_root.is_dir():
        raise FileNotFoundError(campaigns_root)
    plan_index = _plan_index(plans_root)
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for stage_manifest_path in sorted(campaigns_root.rglob("stage_manifest.json")):
        if output_root in stage_manifest_path.parents:
            continue
        stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
        record = _discovery_record(stage_manifest_path, stage_manifest)
        completion_path = stage_manifest_path.parent / "completion_manifest.json"
        progress_path = stage_manifest_path.parent / "progress.json"
        policy = record["baseline_sampling_policy"]
        identity_matches = (
            record["version_label"] == RANKING_MAJOR_VERSION
            and policy in BASELINE_SAMPLING_POLICIES
            and record["raw_output_schema_version"] == OUTPUT_SCHEMA_VERSION
            and record["plan_fingerprint_schema_version"]
            == FINGERPRINT_SCHEMA_VERSION
            and record["trade_audit_schema_version"]
            == COMBINED_TRADE_AUDIT_SCHEMA_VERSION
            and record["trade_audit_schema_id"]
            == COMBINED_TRADE_AUDIT_SCHEMA_ID
            and record["rebound_baseline_policy_id"]
            == REBOUND_BASELINE_POLICY_ID
            and record["scenario_schema_id"] == REQUIRED_SCENARIO_SCHEMA_ID
            and record["scenario_definition_sha256"]
            == REQUIRED_SCENARIO_DEFINITION_SHA256
            and record["scenario_selection_mode"] == "single"
            and record["exit_mode"] == EXIT_MODE_COMBINED
        )
        if not identity_matches:
            excluded.append({**record, "reason": "incompatible_result_identity"})
            continue
        if not completion_path.is_file() or not progress_path.is_file():
            excluded.append({**record, "reason": "stage_not_complete"})
            continue
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if completion.get("status") != "complete" or progress.get("status") != "complete":
            excluded.append(
                {
                    **record,
                    "reason": "stage_not_complete",
                    "progress_status": str(progress.get("status", "")),
                    "completion_status": str(completion.get("status", "")),
                }
            )
            continue
        candidates = list(plan_index.get(record["input_plan_sha256"], []))
        retained_plan_copy = stage_manifest_path.parent / "input_plan.json"
        if (
            retained_plan_copy.is_file()
            and sha256_file(retained_plan_copy) == record["input_plan_sha256"]
        ):
            candidates = [retained_plan_copy.resolve()]
        if len(candidates) != 1:
            raise ValueError(
                f"completed stage must resolve one exact input plan: {stage_manifest_path.parent}"
            )
        plan_payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        ranking_lineage_id = (
            str(plan_payload.get("ranking_lineage_id", ""))
            or str(
                (stage_manifest.get("instrument_contract") or {}).get(
                    "ranking_lineage_id", ""
                )
            )
        )
        if ranking_lineage_id != RANKING_LINEAGE_ID:
            excluded.append(
                {
                    **record,
                    "ranking_lineage_id": ranking_lineage_id,
                    "reason": "outside_v4_4_major_ranking_lineage",
                }
            )
            continue
        loaded = _load_stage(
            stage_manifest_path.parent,
            candidates[0],
            load_trades=False,
        )
        stage_cost_model = _cost_model_from_stage_manifest(stage_manifest)
        included.append(
            {
                **record,
                "plan_path": str(candidates[0]),
                "completion_manifest": str(completion_path.resolve()),
                "completion_manifest_sha256": sha256_file(completion_path),
                "completion_coordinate_count": int(completion["coordinate_count"]),
                "completion_trade_count": int(completion["trade_count"]),
                "completed_at": str(completion.get("completed_at", "")),
                "ranking_lineage_id": ranking_lineage_id,
                "cost_model_id": str(stage_cost_model["id"]),
                "cost_model_reference_sha256": str(
                    stage_cost_model["reference_sha256"]
                ),
                "cost_model": stage_cost_model,
                "loaded": loaded,
            }
        )
    if not included:
        raise ValueError("no completed compatible V4.4 combined-exit stages were discovered")
    included.sort(key=lambda row: (row["campaign_id"], row["stage_id"], row["stage_root"]))
    excluded.sort(key=lambda row: (row["campaign_id"], row["stage_id"], row["stage_root"]))
    return included, excluded


def _with_provenance(frame: pd.DataFrame, record: dict[str, Any]) -> pd.DataFrame:
    # Stage frames are freshly loaded for this union build and are not reused.
    # Enriching them in place avoids retaining a second full trade population
    # before the cumulative concat.  This is material at 600k+ saved trades.
    result = frame
    result["source_campaign_id"] = record["campaign_id"]
    result["source_stage_id"] = record["stage_id"]
    result["source_stage_root"] = record["stage_root"]
    result["source_plan_fingerprint"] = record["plan_fingerprint"]
    result["source_stage_key"] = f"{record['campaign_id']}::{record['stage_id']}"
    return result


def _normalized_combo_rows(frame: pd.DataFrame, combo_id: str, sort_columns: list[str]) -> dict[str, pd.DataFrame]:
    ignored = {
        "batch_id",
        "grid_ordinal",
        "source_campaign_id",
        "source_stage_id",
        "source_stage_root",
        "source_plan_fingerprint",
        "source_stage_key",
    }
    result: dict[str, pd.DataFrame] = {}
    for source_key, group in frame.loc[frame.combo_id.astype(str).eq(combo_id)].groupby(
        "source_stage_key", sort=True
    ):
        columns = [column for column in group.columns if column not in ignored]
        normalized = group.loc[:, columns].copy()
        available_sort = [column for column in sort_columns if column in normalized.columns]
        if available_sort:
            normalized = normalized.sort_values(available_sort, kind="mergesort")
        normalized = normalized.reindex(sorted(normalized.columns), axis=1).reset_index(drop=True)
        result[str(source_key)] = normalized
    return result


def _deduplicate_exact_coordinates(
    summary: pd.DataFrame,
    segment: pd.DataFrame,
    scenario: pd.DataFrame,
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    duplicate_ids = sorted(
        summary.loc[summary.combo_id.astype(str).duplicated(keep=False), "combo_id"]
        .astype(str)
        .unique()
    )
    duplicate_audit: list[dict[str, Any]] = []
    for combo_id in duplicate_ids:
        owners = sorted(
            summary.loc[summary.combo_id.astype(str).eq(combo_id), "source_stage_key"]
            .astype(str)
            .unique()
        )
        retained = owners[0]
        comparisons = (
            (summary, ["combo_id"]),
            (segment, ["segment_id"]),
            (scenario, ["scenario_id"]),
        )
        for frame, sort_columns in comparisons:
            normalized = _normalized_combo_rows(frame, combo_id, sort_columns)
            if not normalized:
                continue
            if set(normalized) != set(owners):
                raise ValueError(
                    f"duplicate combo_id has a missing evidence population: {combo_id}"
                )
            baseline = normalized[retained]
            for owner in owners[1:]:
                try:
                    assert_frame_equal(
                        baseline,
                        normalized[owner],
                        check_dtype=False,
                        check_exact=False,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                except AssertionError as error:
                    raise ValueError(
                        f"duplicate combo_id has conflicting saved evidence: {combo_id}"
                    ) from error
        duplicate_audit.append(
            {
                "combo_id": combo_id,
                "retained_source_stage_key": retained,
                "equivalent_duplicate_stage_keys": owners[1:],
            }
        )
        for frame in (summary, segment, scenario):
            drop_mask = frame.combo_id.astype(str).eq(combo_id) & ~frame.source_stage_key.astype(
                str
            ).eq(retained)
            frame.drop(index=frame.index[drop_mask], inplace=True)
    return (
        summary.reset_index(drop=True),
        segment.reset_index(drop=True),
        scenario.reset_index(drop=True),
        trades.reset_index(drop=True),
        duplicate_audit,
    )


def _shared_stage_manifest(included: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "source",
        "source_sha256",
        "version_label",
        "scenario_schema_id",
        "scenario_definition_sha256",
        "scenario_selection_mode",
        "events_sha256",
        "train_start",
        "train_end",
        "entry_fill_mode",
        "entry_execution_policy",
        "entry_slippage",
        "exit_mode",
        "schema_version",
        "plan_fingerprint_schema_version",
        "trade_audit_schema_version",
        "trade_audit_schema_id",
        "rebound_baseline_policy_id",
    )
    base = included[0]["loaded"]["stage_manifest"].copy()
    for field in fields:
        values = {str(row["loaded"]["stage_manifest"].get(field)) for row in included}
        if len(values) != 1:
            raise ValueError(f"completed stages disagree on union identity field: {field}")
    policies = sorted(
        {
            str(row["loaded"]["stage_manifest"]["baseline_sampling_policy"])
            for row in included
        }
    )
    if base.get("version_label") != RANKING_MAJOR_VERSION:
        raise ValueError("union stage is outside the V4.4 major version")
    lineages = {str(row["ranking_lineage_id"]) for row in included}
    if lineages != {RANKING_LINEAGE_ID}:
        raise ValueError("completed stages disagree on the V4.4 major ranking lineage")

    def variants(field: str, policy: str) -> list[str]:
        return sorted(
            {
                str(row["loaded"]["stage_manifest"].get(field, ""))
                for row in included
                if str(
                    row["loaded"]["stage_manifest"].get(
                        "baseline_sampling_policy", ""
                    )
                )
                == policy
            }
        )

    strategy_variants = {policy: variants("strategy_id", policy) for policy in policies}
    filter_variants = {policy: variants("baseline_filter_id", policy) for policy in policies}
    semantics_variants = {
        policy: variants("result_semantics_id", policy) for policy in policies
    }
    base["baseline_sampling_policies"] = policies
    base["strategy_ids_by_baseline_sampling_policy"] = {
        policy: values[0] if len(values) == 1 else "multiple_within_v4_4"
        for policy, values in strategy_variants.items()
    }
    base["baseline_filter_ids_by_baseline_sampling_policy"] = {
        policy: values[0] if len(values) == 1 else "multiple_within_v4_4"
        for policy, values in filter_variants.items()
    }
    base["result_semantics_ids_by_baseline_sampling_policy"] = {
        policy: values[0] if len(values) == 1 else "multiple_within_v4_4"
        for policy, values in semantics_variants.items()
    }
    base["strategy_id_variants_by_baseline_sampling_policy"] = strategy_variants
    base["baseline_filter_id_variants_by_baseline_sampling_policy"] = filter_variants
    base["result_semantics_id_variants_by_baseline_sampling_policy"] = semantics_variants
    base["baseline_sampling_policy"] = policies[0] if len(policies) == 1 else "multiple"
    base["strategy_id"] = (
        next(iter(strategy_variants.values()))[0]
        if len(policies) == 1 and len(next(iter(strategy_variants.values()))) == 1
        else "multiple_within_v4_4_major_ranking_lineage"
    )
    base["result_semantics_id"] = (
        next(iter(semantics_variants.values()))[0]
        if len(policies) == 1 and len(next(iter(semantics_variants.values()))) == 1
        else "multiple_within_v4_4_major_ranking_lineage"
    )
    base["ranking_major_version"] = RANKING_MAJOR_VERSION
    base["ranking_lineage_id"] = RANKING_LINEAGE_ID
    base["minor_identity_policy"] = (
        "engine, data-preparation, strategy, result-semantics, and artifact hashes "
        "remain stage provenance and do not split the V4.4 cumulative ranking"
    )
    base["campaign_id"] = UNION_CAMPAIGN_ID
    base["stage_id"] = UNION_STAGE_ID
    return base


def _apply_stage_bound_costs(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    included: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Apply each retained coordinate owner's frozen source-stage cost model."""
    costed_summaries: list[pd.DataFrame] = []
    costed_trades: list[pd.DataFrame] = []
    cost_models_by_stage: list[dict[str, Any]] = []
    for row in included:
        stage_key = f"{row['campaign_id']}::{row['stage_id']}"
        stage_summary = summary.loc[
            summary.source_stage_key.astype(str).eq(stage_key)
        ].copy()
        stage_trades = trades.loc[
            trades.source_stage_key.astype(str).eq(stage_key)
        ].copy()
        if stage_summary.empty:
            continue
        stage_summary, stage_trades = _apply_cost_adjusted_metrics(
            stage_summary,
            stage_trades,
            cost_model=row["cost_model"],
            copy=False,
        )
        costed_summaries.append(stage_summary)
        costed_trades.append(stage_trades)
        cost_models_by_stage.append(
            {
                "source_stage_key": stage_key,
                "ranking_lineage_id": row["ranking_lineage_id"],
                "cost_model": dict(row["cost_model"]),
            }
        )
    result_summary = pd.concat(costed_summaries, ignore_index=True, sort=False)
    result_trades = pd.concat(costed_trades, ignore_index=True, sort=False)
    unique_cost_models = {
        (record["cost_model"]["id"], record["cost_model"]["reference_sha256"]): record[
            "cost_model"
        ]
        for record in cost_models_by_stage
    }
    if len(unique_cost_models) == 1:
        union_cost_model = dict(next(iter(unique_cost_models.values())))
    else:
        union_cost_model = {
            "id": "multiple_stage_bound_cost_models",
            "role": "each_coordinate_uses_its_source_stage_cost_model",
            "instrument_name": "按阶段绑定的成本模型",
            "ranking_display_default": "cost_adjusted",
            "ranking_basis": "cost_adjusted",
            "available_ranking_display_modes": ["cost_adjusted", "gross"],
            "model_count": len(unique_cost_models),
        }
    return result_summary, result_trades, cost_models_by_stage, union_cost_model


def _trade_chunk_filename(combo_id: str) -> str:
    digest = hashlib.sha256(str(combo_id).encode("utf-8")).hexdigest()[:16]
    return f"c_{digest}.js"


def _combined_trade_audit(
    audits: list[dict[str, Any]],
    holding_bars: list[np.ndarray],
    holding_minutes: list[np.ndarray],
) -> dict[str, Any]:
    boolean_keys = {
        key
        for audit in audits
        for key, value in audit.items()
        if isinstance(value, (bool, np.bool_))
    }
    count_keys = {
        "trade_count",
        "waited_entry_count",
        "rebound_exit_count",
        "speed_exit_count",
        "segment_end_exit_count",
        "gap_spanning_trade_count",
    }
    bars = np.concatenate(holding_bars) if holding_bars else np.asarray([], dtype=float)
    minutes = (
        np.concatenate(holding_minutes)
        if holding_minutes
        else np.asarray([], dtype=float)
    )
    result: dict[str, Any] = {
        key: all(bool(audit.get(key, True)) for audit in audits)
        for key in sorted(boolean_keys)
    }
    result.update(
        {
            key: int(sum(int(audit.get(key, 0)) for audit in audits))
            for key in sorted(count_keys)
        }
    )
    result["maximum_entry_wait_bars"] = max(
        (int(audit.get("maximum_entry_wait_bars", 0)) for audit in audits),
        default=0,
    )
    for values, prefix in ((bars, "holding_bar_distance"), (minutes, "holding_minutes")):
        result[f"{prefix}_minimum"] = float(values.min()) if len(values) else None
        result[f"{prefix}_median"] = float(np.median(values)) if len(values) else None
        result[f"{prefix}_p95"] = float(np.quantile(values, 0.95)) if len(values) else None
        result[f"{prefix}_maximum"] = float(values.max()) if len(values) else None
    return result


def _closed_stage_trade_statistics(trades: pd.DataFrame) -> dict[str, Any]:
    waits = pd.to_numeric(trades["entry_wait_bar_count"], errors="raise")
    reasons = trades["exit_reason"].astype(str)
    return {
        "closed_stage_trade_audit_reused": True,
        "trade_count": int(len(trades)),
        "waited_entry_count": int(waits.gt(0).sum()),
        "maximum_entry_wait_bars": int(waits.max()) if len(waits) else 0,
        "rebound_exit_count": int(reasons.eq("rebound_threshold").sum()),
        "speed_exit_count": int(reasons.eq("downside_speed").sum()),
        "segment_end_exit_count": int(reasons.eq("segment_end").sum()),
        "gap_spanning_trade_count": int(_truthy(trades["position_crosses_real_gap"]).sum()),
    }


def _combined_exit_summary(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True, sort=False)
    frame["return_sum"] = frame["mean_return"] * frame["trade_count"]
    keys = ["method", "baseline_sampling_policy", "speed_window_bars", "exit_reason"]
    result = frame.groupby(keys, sort=True).agg(
        trade_count=("trade_count", "sum"),
        return_sum=("return_sum", "sum"),
    ).reset_index()
    result["mean_return"] = result["return_sum"] / result["trade_count"]
    return result.drop(columns="return_sum")


def _combined_lifecycle_summary(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True, sort=False)
    keys = ["method", "baseline_sampling_policy", "speed_window_bars", "exit_reason"]
    weighted = (
        "holding_bar_distance_median",
        "holding_bar_distance_p95",
        "holding_minutes_median",
        "holding_minutes_p95",
    )
    for column in weighted:
        frame[f"{column}_weighted"] = frame[column] * frame["trade_count"]
    aggregations: dict[str, tuple[str, str]] = {
        "trade_count": ("trade_count", "sum"),
        "holding_bar_distance_maximum": ("holding_bar_distance_maximum", "max"),
        "holding_minutes_maximum": ("holding_minutes_maximum", "max"),
        "waited_entry_count": ("waited_entry_count", "sum"),
        "gap_spanning_trade_count": ("gap_spanning_trade_count", "sum"),
    }
    aggregations.update(
        {f"{column}_weighted": (f"{column}_weighted", "sum") for column in weighted}
    )
    result = frame.groupby(keys, sort=True).agg(**aggregations).reset_index()
    for column in weighted:
        result[column] = result[f"{column}_weighted"] / result["trade_count"]
        result.drop(columns=f"{column}_weighted", inplace=True)
    return result


def _build_union_locked(
    campaigns_root: Path,
    plans_root: Path,
    output_root: Path,
    review_workers: int,
) -> dict[str, Any]:
    included, excluded = _discover_closed_stages(campaigns_root, plans_root, output_root)
    stage_records = [
        {
            key: value
            for key, value in row.items()
            if key not in {"loaded", "cost_model"}
        }
        for row in included
    ]
    snapshot_id = _snapshot_id(stage_records)
    delivery_root = output_root / "snapshots" / snapshot_id
    peer_review_kwargs = _trade_review_peer_kwargs(output_root)
    presentation_refresh = _refresh_completed_snapshot_trade_shell(
        delivery_root, snapshot_id, peer_review_kwargs
    )
    existing_snapshot = _load_complete_snapshot(delivery_root, snapshot_id)
    if existing_snapshot is not None:
        stable_snapshot = _publish_stable_routes(
            output_root, delivery_root, snapshot_id, excluded
        )
        manifest_path = delivery_root / "analysis_manifest.json"
        completion_path = delivery_root / "completion_manifest.json"
        return {
            **existing_snapshot,
            "output": str(output_root.resolve()),
            "snapshot_output": str(delivery_root.resolve()),
            "stable_snapshot": stable_snapshot,
            "analysis_manifest": artifact(manifest_path),
            "completion_manifest": artifact(completion_path),
            "excluded_stages": excluded,
            "reused_complete_snapshot": True,
            "presentation_refresh": presentation_refresh,
        }

    summaries = [_with_provenance(row["loaded"]["summary"], row) for row in included]
    segments = [_with_provenance(row["loaded"]["segment"], row) for row in included]
    scenarios = [_with_provenance(row["loaded"]["scenario"], row) for row in included]
    shared_stage_manifest = _shared_stage_manifest(included)
    summary = pd.concat(summaries, ignore_index=True, sort=False)
    segment = pd.concat(segments, ignore_index=True, sort=False)
    scenario = pd.concat(scenarios, ignore_index=True, sort=False)
    trades = pd.DataFrame()
    del summaries, segments, scenarios
    raw_coordinate_count = int(len(summary))
    raw_trade_count = int(sum(row["completion_trade_count"] for row in included))
    summary, segment, scenario, trades, duplicate_audit = _deduplicate_exact_coordinates(
        summary, segment, scenario, trades
    )

    scenario_3_detail = scenario.loc[
        scenario.scenario_id.astype(str).eq("scenario_3"),
        ["combo_id", "qualified_segment_count", "failed_segment_ids"],
    ].copy()
    if len(scenario_3_detail) != len(summary) or scenario_3_detail.combo_id.astype(
        str
    ).duplicated().any():
        raise ValueError("union Scenario 3 detail does not cover every unique coordinate")
    scenario_3_detail = scenario_3_detail.rename(
        columns={
            "qualified_segment_count": "scenario_3_qualified_segment_count",
            "failed_segment_ids": "scenario_3_failed_segment_ids",
        }
    )
    summary = summary.merge(
        scenario_3_detail, on="combo_id", how="left", validate="one_to_one"
    )
    delivery_root.mkdir(parents=True, exist_ok=True)
    trade_review_root = delivery_root / "trade_review"
    chunk_directory = trade_review_root / "v3_native_trades_js"
    chunk_directory.mkdir(parents=True, exist_ok=True)
    previous_review = _current_trade_review_for_reuse(output_root, delivery_root)
    reused_chunk_count = 0
    if previous_review is not None:
        for source_chunk in sorted((previous_review / "v3_native_trades_js").glob("c_*.js")):
            target_chunk = chunk_directory / source_chunk.name
            if not target_chunk.exists():
                os.link(source_chunk, target_chunk)
            reused_chunk_count += 1

    source_path = Path(str(shared_stage_manifest["source"])).resolve()
    source = pd.read_csv(source_path, usecols=list(SOURCE_COLUMNS))
    source = source.loc[
        source["datetime"].astype(str).le(str(shared_stage_manifest["train_end"]))
    ].reset_index(drop=True)
    union_trades_path = delivery_root / "union_trades.csv"
    union_trades_temp = union_trades_path.with_suffix(".csv.incremental.tmp")
    costed_summaries: list[pd.DataFrame] = []
    combo_lifecycle_parts: list[pd.DataFrame] = []
    exit_parts: list[pd.DataFrame] = []
    lifecycle_parts: list[pd.DataFrame] = []
    stage_trade_audits: list[dict[str, Any]] = []
    holding_bar_arrays: list[np.ndarray] = []
    holding_minute_arrays: list[np.ndarray] = []
    cost_models_by_stage: list[dict[str, Any]] = []
    generated_chunk_count = 0
    retained_trade_count = 0
    wrote_trade_header = False
    for row in included:
        stage_key = f"{row['campaign_id']}::{row['stage_id']}"
        stage_summary = summary.loc[
            summary["source_stage_key"].astype(str).eq(stage_key)
        ].copy()
        if stage_summary.empty:
            continue
        stage_frames = [pd.read_csv(path) for path in row["loaded"]["artifacts"]["trade_paths"]]
        stage_trades = (
            pd.concat(stage_frames, ignore_index=True, sort=False)
            if stage_frames
            else pd.DataFrame()
        )
        del stage_frames
        _with_provenance(stage_trades, row)
        retained_ids = set(stage_summary["combo_id"].astype(str))
        stage_trades = stage_trades.loc[
            stage_trades["combo_id"].astype(str).isin(retained_ids)
        ].reset_index(drop=True)
        stage_trades = _augment_trade_lifecycle(stage_trades, copy=False)
        stage_summary, stage_trades = _apply_cost_adjusted_metrics(
            stage_summary,
            stage_trades,
            cost_model=row["cost_model"],
            copy=False,
        )
        combo_lifecycle_parts.append(_combo_lifecycle_summary(stage_summary, stage_trades))
        costed_summaries.append(stage_summary)
        exit_parts.append(_exit_reason_summary(stage_trades))
        lifecycle_parts.append(_trade_lifecycle_summary(stage_trades))
        audit = _closed_stage_trade_statistics(stage_trades)
        stage_trade_audits.append(audit)
        holding_bar_arrays.append(
            pd.to_numeric(stage_trades["holding_bar_distance"], errors="raise").to_numpy()
        )
        holding_minute_arrays.append(
            pd.to_numeric(stage_trades["holding_minutes"], errors="raise").to_numpy()
        )
        retained_trade_count += int(len(stage_trades))
        stage_trades.to_csv(
            union_trades_temp,
            mode="w" if not wrote_trade_header else "a",
            header=not wrote_trade_header,
            index=False,
            encoding="utf-8",
        )
        wrote_trade_header = True
        missing_ids = {
            combo_id
            for combo_id in retained_ids
            if not (chunk_directory / _trade_chunk_filename(combo_id)).is_file()
        }
        if missing_ids:
            missing_summary = stage_summary.loc[
                stage_summary["combo_id"].astype(str).isin(missing_ids)
            ].copy()
            missing_trades = stage_trades.loc[
                stage_trades["combo_id"].astype(str).isin(missing_ids)
            ].copy()
            chunk_completion = {
                "status": "complete",
                "campaign_id": UNION_CAMPAIGN_ID,
                "stage_id": UNION_STAGE_ID,
                "strategy_id": row["loaded"]["stage_manifest"]["strategy_id"],
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "plan_fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
                "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
                "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
                "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
                "exit_mode": "combined",
                "coordinate_count": int(len(missing_summary)),
                "trade_count": int(len(missing_trades)),
            }
            chunk_delivery = build_stage_trade_review(
                trade_review_root,
                missing_summary,
                missing_trades,
                row["loaded"]["stage_manifest"],
                chunk_completion,
                analysis_identity=UNION_ANALYSIS_IDENTITY,
                workers=review_workers,
                source_frame=source,
                chunks_only=True,
            )
            generated_chunk_count += int(chunk_delivery["generated_chunk_count"])
        cost_models_by_stage.append(
            {
                "source_stage_key": stage_key,
                "ranking_lineage_id": row["ranking_lineage_id"],
                "cost_model": dict(row["cost_model"]),
            }
        )
        del stage_trades
        gc.collect()
    os.replace(union_trades_temp, union_trades_path)
    summary = pd.concat(costed_summaries, ignore_index=True, sort=False)
    combo_lifecycle = pd.concat(combo_lifecycle_parts, ignore_index=True, sort=False)
    summary = summary.merge(combo_lifecycle, on="combo_id", how="left", validate="one_to_one")
    trade_audit = _combined_trade_audit(
        stage_trade_audits,
        holding_bar_arrays,
        holding_minute_arrays,
    )
    unique_cost_models = {
        (record["cost_model"]["id"], record["cost_model"]["reference_sha256"]): record["cost_model"]
        for record in cost_models_by_stage
    }
    union_cost_model = (
        dict(next(iter(unique_cost_models.values())))
        if len(unique_cost_models) == 1
        else {
            "id": "multiple_stage_bound_cost_models",
            "role": "each_coordinate_uses_its_source_stage_cost_model",
            "instrument_name": "按阶段绑定的成本模型",
            "ranking_display_default": "cost_adjusted",
            "ranking_basis": "cost_adjusted",
            "available_ranking_display_modes": ["cost_adjusted", "gross"],
            "model_count": len(unique_cost_models),
        }
    )
    summary = _rank(summary)
    eligible = summary.loc[_truthy(summary.scenario_3_qualified)].copy()
    rankings = eligible.sort_values(
        ["method", "baseline_sampling_policy", "scenario_3_total_return_rank"],
        kind="mergesort",
    )
    diagnostics = summary.copy()
    diagnostics["scenario_3_diagnostic_rank"] = np.nan
    for _, group in diagnostics.groupby(
        ["method", "baseline_sampling_policy"], sort=True
    ):
        ordered = group.sort_values(
            [
                "scenario_3_qualified_segment_count",
                COST_ADJUSTED_RETURN_KEY,
                "train_cost_adjusted_max_drawdown_abs",
                "train_trade_count",
                "combo_id",
            ],
            ascending=[False, False, True, False, True],
            kind="mergesort",
        )
        diagnostics.loc[ordered.index, "scenario_3_diagnostic_rank"] = np.arange(
            1, len(ordered) + 1
        )
    diagnostics = diagnostics.sort_values(
        ["method", "baseline_sampling_policy", "scenario_3_diagnostic_rank"],
        kind="mergesort",
    ).reset_index(drop=True)

    speed_summary = (
        summary.groupby(
            ["method", "baseline_sampling_policy", "speed_window_bars"], sort=True
        )
        .agg(
            coordinate_count=("combo_id", "size"),
            scenario_3_qualified_count=("scenario_3_qualified", "sum"),
            best_train_return=("train_return", "max"),
            median_train_return=("train_return", "median"),
            best_train_cost_adjusted_return=(
                COST_ADJUSTED_RETURN_KEY,
                "max",
            ),
            median_train_cost_adjusted_return=(
                COST_ADJUSTED_RETURN_KEY,
                "median",
            ),
            trade_count=("train_trade_count", "sum"),
            speed_exit_count=("speed_exit_count", "sum"),
        )
        .reset_index()
    )
    exit_summary = _combined_exit_summary(exit_parts)
    lifecycle_summary = _combined_lifecycle_summary(lifecycle_parts)
    segment_summary = (
        segment.assign(qualified=lambda frame: _truthy(frame.qualified))
        .groupby(["method", "baseline_sampling_policy", "segment_id"], sort=True)
        .agg(
            coordinate_count=("combo_id", "size"),
            qualified_coordinate_count=("qualified", "sum"),
            entry_count_in_interval=("entry_count_in_interval", "sum"),
            exit_count_in_interval=("exit_count_in_interval", "sum"),
        )
        .reset_index()
    )
    scenario_summary = (
        scenario.assign(qualified=lambda frame: _truthy(frame.qualified))
        .groupby(["method", "baseline_sampling_policy", "scenario_id"], sort=True)
        .agg(
            coordinate_count=("combo_id", "size"),
            qualified_coordinate_count=("qualified", "sum"),
        )
        .reset_index()
    )
    leaders = diagnostics.groupby(
        ["method", "baseline_sampling_policy"], sort=True
    ).head(1).copy()
    leaders["evidence_role"] = (
        "scenario_3_segment_coverage_then_cost_adjusted_return_diagnostic"
    )
    leaders["parameter_accepted"] = False

    delivery_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "analysis_summary": delivery_root / "analysis_summary.csv",
        "scenario_3_rankings": delivery_root / "scenario_3_total_return_rankings.csv",
        "speed_window_summary": delivery_root / "speed_window_summary.csv",
        "exit_reason_summary": delivery_root / "exit_reason_summary.csv",
        "trade_lifecycle_summary": delivery_root / "trade_lifecycle_summary.csv",
        "segment_qualification_summary": delivery_root / "segment_qualification_summary.csv",
        "scenario_qualification_summary": delivery_root / "scenario_qualification_summary.csv",
        "diagnostic_leaders": delivery_root / "diagnostic_leaders.csv",
        "union_trades": delivery_root / "union_trades.csv",
        "source_stages": delivery_root / "source_stages.csv",
        "excluded_stages": delivery_root / "excluded_stages.csv",
        "duplicate_coordinate_audit": delivery_root / "duplicate_coordinate_audit.json",
    }
    csv_payloads = {
        "analysis_summary": diagnostics,
        "scenario_3_rankings": rankings,
        "speed_window_summary": speed_summary,
        "exit_reason_summary": exit_summary,
        "trade_lifecycle_summary": lifecycle_summary,
        "segment_qualification_summary": segment_summary,
        "scenario_qualification_summary": scenario_summary,
        "diagnostic_leaders": leaders,
        "source_stages": pd.DataFrame(stage_records),
        "excluded_stages": pd.DataFrame(excluded),
    }
    for name, frame in csv_payloads.items():
        _atomic_csv(paths[name], frame)
    _atomic_json(paths["duplicate_coordinate_audit"], duplicate_audit)

    union_completion = {
        "status": "complete",
        "campaign_id": UNION_CAMPAIGN_ID,
        "stage_id": UNION_STAGE_ID,
        "ranking_major_version": RANKING_MAJOR_VERSION,
        "ranking_lineage_id": RANKING_LINEAGE_ID,
        "strategy_id": shared_stage_manifest["strategy_id"],
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "plan_fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "baseline_sampling_policy": shared_stage_manifest[
            "baseline_sampling_policy"
        ],
        "strategy_ids_by_baseline_sampling_policy": shared_stage_manifest[
            "strategy_ids_by_baseline_sampling_policy"
        ],
        "baseline_filter_ids_by_baseline_sampling_policy": shared_stage_manifest[
            "baseline_filter_ids_by_baseline_sampling_policy"
        ],
        "result_semantics_ids_by_baseline_sampling_policy": shared_stage_manifest[
            "result_semantics_ids_by_baseline_sampling_policy"
        ],
        "exit_mode": "combined",
        "scenario_schema_id": REQUIRED_SCENARIO_SCHEMA_ID,
        "coordinate_count": int(len(diagnostics)),
        "trade_count": retained_trade_count,
    }
    review_delivery = build_stage_trade_review(
        delivery_root / "trade_review",
        diagnostics,
        pd.DataFrame(),
        shared_stage_manifest,
        union_completion,
        analysis_identity=UNION_ANALYSIS_IDENTITY,
        manifest_href="../analysis_manifest.json",
        main_href="../index.html",
        **peer_review_kwargs,
        workers=review_workers,
        reuse_chunk_directory=chunk_directory,
        expected_trade_count=retained_trade_count,
        reused_trade_stats={
            "waited_entry_count": int(trade_audit["waited_entry_count"]),
            "maximum_entry_wait_bars": int(trade_audit["maximum_entry_wait_bars"]),
        },
        publication_chunk_counts={
            "reused_chunk_count": reused_chunk_count,
            "generated_chunk_count": generated_chunk_count,
        },
        source_frame=source,
    )
    scenario_definition_path = Path(
        str(shared_stage_manifest.get("scenario_definition", ""))
    ).resolve()
    scenario_requirements_delivery = build_scenario_requirements_delivery(
        delivery_root / "scenario_requirements",
        scenario_definition_path,
    )

    speed_windows = sorted(int(value) for value in diagnostics.speed_window_bars.unique())
    main_data = {
        "coordinateCount": int(len(diagnostics)),
        "rawCoordinateCount": raw_coordinate_count,
        "tradeCount": retained_trade_count,
        "rawTradeCount": raw_trade_count,
        "completedStageCount": len(included),
        "excludedIncompleteStageCount": sum(
            row.get("reason") == "stage_not_complete" for row in excluded
        ),
        "duplicateCoordinateCount": len(duplicate_audit),
        "scenario3QualifiedCount": int(len(eligible)),
        "highReturnViews": [dict(view) for view in HIGH_RETURN_VIEWS],
        "costModel": union_cost_model,
        "costModelsByStage": cost_models_by_stage,
        "speedWindows": speed_windows,
        "baselineSamplingPolicies": sorted(
            str(value) for value in diagnostics.baseline_sampling_policy.unique()
        ),
        "scopeLabel": "累计完成阶段并集",
        "note": (
            f"累计入口包含 {len(included)} 个身份一致且清单闭合的阶段；"
            "仍在运行或未完成的阶段不会进入当前快照。情景三要求三段行情各自只有一次区间内开仓、"
            "区间内零平仓；区间后允许回撤或速度退出。两种方法保持独立排名。"
        ),
        "rows": _records(diagnostics),
        "segmentRows": _records(segment_summary),
        "scenarioRows": _records(scenario_summary),
        "exitRows": _records(exit_summary),
        "lifecycleRows": _records(lifecycle_summary),
        "strategyId": shared_stage_manifest["strategy_id"],
        "resultSemanticsId": shared_stage_manifest["result_semantics_id"],
        "rawOutputSchemaVersion": OUTPUT_SCHEMA_VERSION,
        "planFingerprintSchemaVersion": FINGERPRINT_SCHEMA_VERSION,
        "tradeAuditSchemaVersion": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "tradeAuditSchemaId": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "reboundBaselinePolicyId": REBOUND_BASELINE_POLICY_ID,
        "nativeTradeRoute": "trade_review/index.html?combo_id={combo_id}",
        "scenarioRequirementsRoute": (
            "scenario_requirements/index.html?scenario={scenario_id}"
        ),
        "templateProvenance": {
            "id": LEGACY_V4_MAIN_TEMPLATE_ID,
            "path": str(LEGACY_V4_MAIN_TEMPLATE_PATH.resolve()),
            "sha256": sha256_file(LEGACY_V4_MAIN_TEMPLATE_PATH),
        },
        "unionSnapshotId": snapshot_id,
    }
    _atomic_text(
        delivery_root / "report_data.js",
        "window.V4_4_S3_STAGE="
        + json.dumps(main_data, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
    )
    _atomic_text(
        delivery_root / "analysis_data.js",
        "window.V4_ANALYSIS_DATA="
        + json.dumps(
            main_summary_payload(main_data),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + ";\n",
    )
    main_html = _legacy_v4_main_html(
        "../../../cross_instrument_comparison/index.html"
    )
    _atomic_text(delivery_root / "index.html", main_html)
    _atomic_text(delivery_root / "analysis_report.html", main_html)

    output_artifacts = {name: artifact(path) for name, path in paths.items()}
    output_artifacts.update(
        {
            "main_report_data": artifact(delivery_root / "analysis_data.js"),
            "simple_report_data": artifact(delivery_root / "report_data.js"),
            "main_entry_html": artifact(delivery_root / "index.html"),
            "analysis_report_html": artifact(delivery_root / "analysis_report.html"),
            "trade_catalog": artifact(Path(review_delivery["catalog"])),
            "trade_level_html": artifact(Path(review_delivery["index"])),
            "trade_review_manifest": artifact(Path(review_delivery["manifest"])),
            "scenario_requirements_html": scenario_requirements_delivery["index"],
            "scenario_requirements_data": scenario_requirements_delivery["data"],
        }
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": _utc_now(),
        "analysis_identity": UNION_ANALYSIS_IDENTITY,
        "union_snapshot_id": snapshot_id,
        "campaign_id": UNION_CAMPAIGN_ID,
        "stage_id": UNION_STAGE_ID,
        "ranking_major_version": RANKING_MAJOR_VERSION,
        "ranking_lineage_id": RANKING_LINEAGE_ID,
        "strategy_id": shared_stage_manifest["strategy_id"],
        "result_semantics_id": shared_stage_manifest["result_semantics_id"],
        "raw_output_schema_version": OUTPUT_SCHEMA_VERSION,
        "plan_fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "trade_audit_schema_version": COMBINED_TRADE_AUDIT_SCHEMA_VERSION,
        "trade_audit_schema_id": COMBINED_TRADE_AUDIT_SCHEMA_ID,
        "rebound_baseline_policy_id": REBOUND_BASELINE_POLICY_ID,
        "baseline_sampling_policy": shared_stage_manifest[
            "baseline_sampling_policy"
        ],
        "strategy_ids_by_baseline_sampling_policy": shared_stage_manifest[
            "strategy_ids_by_baseline_sampling_policy"
        ],
        "baseline_filter_ids_by_baseline_sampling_policy": shared_stage_manifest[
            "baseline_filter_ids_by_baseline_sampling_policy"
        ],
        "result_semantics_ids_by_baseline_sampling_policy": shared_stage_manifest[
            "result_semantics_ids_by_baseline_sampling_policy"
        ],
        "exit_mode": "combined",
        "scenario_schema_id": REQUIRED_SCENARIO_SCHEMA_ID,
        "completed_stage_count": len(included),
        "excluded_stage_count": len(excluded),
        "excluded_incomplete_stage_count": sum(
            row.get("reason") == "stage_not_complete" for row in excluded
        ),
        "raw_coordinate_count": raw_coordinate_count,
        "coordinate_count": int(len(diagnostics)),
        "duplicate_coordinate_count": len(duplicate_audit),
        "raw_trade_count": raw_trade_count,
        "trade_count": retained_trade_count,
        "scenario_3_qualified_coordinate_count": int(len(eligible)),
        "method_policy_coordinate_counts": {
            f"{method}|{policy}": int(len(group))
            for (method, policy), group in diagnostics.groupby(
                ["method", "baseline_sampling_policy"], sort=True
            )
        },
        "method_policy_qualified_counts": {
            f"{method}|{policy}": int(len(group))
            for (method, policy), group in eligible.groupby(
                ["method", "baseline_sampling_policy"], sort=True
            )
        },
        "speed_window_bars": speed_windows,
        "ranking_policy": (
            "separate baseline-sampling-policy and method rankings; four views each "
            "offer gross and cost-adjusted ordering/display modes; the 2 bps slippage "
            "plus USD 6 round-trip commission mode is the default; no composite score"
        ),
        "cost_model": union_cost_model,
        "cost_models_by_stage": cost_models_by_stage,
        "single_composite_score": False,
        "parameter_acceptance": "none",
        "stage_inclusion_policy": (
            "discover compatible stage manifests; include only complete progress and completion "
            "manifests after exact plan and artifact hash validation"
        ),
        "concurrency_contract": {
            "union_lock": str((output_root / ".v4_4_union.lock").resolve()),
            "one_union_writer": True,
            "distinct_stage_outputs_may_run_concurrently": True,
            "incomplete_stages_excluded": True,
        },
        "publication_contract": {
            "immutable_snapshot_root": str(delivery_root.resolve()),
            "stable_main_entry": str((output_root / "index.html").resolve()),
            "stable_trade_entry": str(
                (output_root / "trade_review" / "index.html").resolve()
            ),
            "stable_routes_redirect_to_one_complete_snapshot": True,
            "existing_complete_snapshot_is_reused": True,
            "incremental_trade_review": {
                "reused_chunk_count": review_delivery["reused_chunk_count"],
                "generated_chunk_count": review_delivery["generated_chunk_count"],
                "reuse_mechanism": (
                    "hard_link_same_volume"
                    if review_delivery["reused_chunk_count"]
                    else "none"
                ),
            },
        },
        "trade_audit": trade_audit,
        "presentation_contract": {
            "main_template": artifact(LEGACY_V4_MAIN_TEMPLATE_PATH),
            "trade_design_source": artifact(LEGACY_V4_TRADE_DESIGN_PATH),
            "historical_v4_main_template_reused": True,
            "historical_v4_trade_template_reused": True,
            "main_entry": "index.html",
            "main_alias": "analysis_report.html",
            "trade_entry": "trade_review/index.html",
            "scenario_requirements_entry": "scenario_requirements/index.html",
            "scenario_requirements_template": scenario_requirements_delivery[
                "template"
            ],
            "market_selector_source": scenario_requirements_delivery[
                "selector_source"
            ],
            "scenario_definition": scenario_requirements_delivery[
                "scenario_definition"
            ],
            "market_selector_interaction_reused": True,
        },
        "source_stages": stage_records,
        "excluded_stages": excluded,
        "artifacts": output_artifacts,
    }
    manifest_path = delivery_root / "analysis_manifest.json"
    _atomic_json(manifest_path, _clean(manifest))
    completion_manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": manifest["completed_at"],
        "union_snapshot_id": snapshot_id,
        "coordinate_count": int(len(diagnostics)),
        "trade_count": retained_trade_count,
        "completed_stage_count": len(included),
        "parameter_acceptance": "none",
        "artifacts": {
            "analysis_manifest": artifact(manifest_path),
            "main_entry_html": artifact(delivery_root / "index.html"),
            "trade_level_html": artifact(Path(review_delivery["index"])),
            "trade_review_manifest": artifact(Path(review_delivery["manifest"])),
            "scenario_requirements_html": scenario_requirements_delivery["index"],
            "scenario_requirements_data": scenario_requirements_delivery["data"],
            "source_stages": artifact(paths["source_stages"]),
            "excluded_stages": artifact(paths["excluded_stages"]),
        },
    }
    completion_path = delivery_root / "completion_manifest.json"
    _atomic_json(completion_path, completion_manifest)
    stable_snapshot = _publish_stable_routes(
        output_root, delivery_root, snapshot_id, excluded
    )
    return {
        **completion_manifest,
        "output": str(output_root.resolve()),
        "snapshot_output": str(delivery_root.resolve()),
        "stable_snapshot": stable_snapshot,
        "analysis_manifest": artifact(manifest_path),
        "completion_manifest": artifact(completion_path),
        "excluded_stages": excluded,
    }


def build_union(
    campaigns_root: Path = DEFAULT_CAMPAIGNS_ROOT,
    plans_root: Path = DEFAULT_PLANS_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    review_workers: int = 4,
) -> dict[str, Any]:
    campaigns_root = campaigns_root.resolve()
    plans_root = plans_root.resolve()
    output_root = output_root.resolve()
    with _exclusive_union_writer(output_root):
        return _build_union_locked(
            campaigns_root, plans_root, output_root, int(review_workers)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the cumulative completed-stage V4.4 combined-exit delivery."
    )
    parser.add_argument("--campaigns-root", type=Path, default=DEFAULT_CAMPAIGNS_ROOT)
    parser.add_argument("--plans-root", type=Path, default=DEFAULT_PLANS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--review-workers", type=int, default=4)
    args = parser.parse_args()
    result = build_union(
        args.campaigns_root,
        args.plans_root,
        args.output,
        review_workers=args.review_workers,
    )
    print(json.dumps(_clean(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
