from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
PACKAGE_ROOT = RESULTS_ROOT / "evaluation_packages"
COMPARISON_ROOT = RESULTS_ROOT / "evaluation_comparison"
CROSS_ROOT = RESULTS_ROOT / "cross_instrument_comparison"
UNION_ROOT = RESULTS_ROOT / "all_completed_union_analysis"

CURRENT_RUN_ID = "k200_train_test_si__combined_350_v56_20260807"
CURRENT_RUN_ROOT = CROSS_ROOT / "runs" / CURRENT_RUN_ID
CURRENT_SNAPSHOT_ID = "eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5"
CURRENT_SNAPSHOT_ROOT = UNION_ROOT / "snapshots" / CURRENT_SNAPSHOT_ID
COMPARISON_ID = (
    "K200_20260526T000000__20260708T235200__"
    "K200_20260708T235215__20260807T032145__"
    "SImain_20260129T000000__20260223T235945"
)
CANDIDATE_SET_ID = "k200_train_test_si_350_v56"

PARAMETER_FIELDS = (
    "combo_id",
    "method",
    "baseline_sampling_policy",
    "entry_fill_mode",
    "entry_execution_policy",
    "entry_slippage",
    "e",
    "bh",
    "trw",
    "k",
    "w",
    "m",
    "speed_window_bars",
)

ROLE_METRIC_SUFFIXES = (
    "trade_count",
    "gross_total_return",
    "cost_total_return",
    "cost_median_trade",
    "cost_mean_trade",
    "cost_max_drawdown_abs",
    "win_rate",
    "mfe_bps_median",
    "mae_bps_median",
    "mfe_points_median",
    "mae_points_median",
    "gross_points_total",
    "mfe_retention_median",
    "top2_positive_return_share",
    "top5_positive_return_share",
    "non_gap_cost_total_return",
    "gap_trade_count",
    "synthetic_signal_trade_count",
    "zero_trade_bar_exposure_count",
    "synthetic_bar_exposure_count",
    "gross_median_trade",
    "gross_max_drawdown_abs",
    "gross_win_rate",
    "low_activity_audit_status",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def read_js_assignment(path: Path, prefix: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(prefix):
        raise ValueError(f"unexpected JavaScript assignment: {path}")
    return json.loads(text[len(prefix) :].rstrip().removesuffix(";"))


def interval_slug(start: str, end: str) -> str:
    def clean(value: str) -> str:
        digits = re.sub(r"[^0-9]", "", value)
        if len(digits) < 14:
            raise ValueError(f"date-time needs seconds: {value}")
        return f"{digits[:8]}T{digits[8:14]}"

    return f"{clean(start)}__{clean(end)}"


def relative_href(origin: Path, target: Path) -> str:
    return os.path.relpath(target, origin).replace("\\", "/")


def redirect_html(target_href: str, title: str) -> str:
    encoded = json.dumps(target_href, ensure_ascii=False)
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title></head><body><script>"
        f"const target={encoded};location.replace(target+location.search+location.hash);"
        "</script></body></html>"
    )


def ensure_hard_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise ValueError(f"existing trade record is not the declared immutable source: {destination}")
        return
    os.link(source, destination)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def package_path(instrument_id: str, start: str, end: str) -> Path:
    return PACKAGE_ROOT / instrument_id / interval_slug(start, end)


def role_projection(
    rows: list[dict[str, Any]],
    *,
    evaluation_id: str,
    prefix: str,
    instrument_id: str,
    display_name: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    projected: list[dict[str, Any]] = []
    for row in rows:
        metrics = {
            suffix: row[f"{prefix}_{suffix}"]
            for suffix in ROLE_METRIC_SUFFIXES
            if f"{prefix}_{suffix}" in row
        }
        parameters = {field: row.get(field) for field in PARAMETER_FIELDS if field in row}
        projected.append(
            {
                "combo_id": str(row["combo_id"]),
                "parameters": parameters,
                "metrics": metrics,
            }
        )
    return {
        "schemaVersion": 1,
        "evaluationId": evaluation_id,
        "instrumentId": instrument_id,
        "displayName": display_name,
        "sampleStart": start,
        "sampleEnd": end,
        "candidateSetId": CANDIDATE_SET_ID,
        "rowCount": len(projected),
        "rows": projected,
    }


def neutral_projection_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in payload["rows"]:
        rows.append({**item["parameters"], **item["metrics"]})
    return pd.DataFrame(rows)


def full_k200_training_summary() -> pd.DataFrame:
    source = pd.read_csv(CURRENT_SNAPSHOT_ROOT / "analysis_summary.csv", low_memory=False)
    output = source[
        [field for field in PARAMETER_FIELDS if field in source.columns]
    ].copy()
    mapping = {
        "train_trade_count": "trade_count",
        "train_return": "gross_total_return",
        "train_avg_trade": "gross_mean_trade",
        "train_max_drawdown_abs": "gross_max_drawdown_abs",
        "train_cost_adjusted_return": "cost_total_return",
        "train_cost_adjusted_avg_trade": "cost_mean_trade",
        "train_cost_adjusted_max_drawdown_abs": "cost_max_drawdown_abs",
        "gap_spanning_trade_count": "gap_trade_count",
        "synthetic_signal_trade_count": "synthetic_signal_trade_count",
        "round_trip_cost_bps": "round_trip_cost_bps",
        "source_campaign_id": "source_campaign_id",
        "source_stage_id": "source_stage_id",
        "source_stage_root": "source_stage_root",
        "source_plan_fingerprint": "source_plan_fingerprint",
        "source_stage_key": "source_stage_key",
    }
    for source_field, target_field in mapping.items():
        if source_field in source.columns:
            output[target_field] = source[source_field]
    return output


def browser_summary_script(payload: dict[str, Any]) -> str:
    key = json.dumps(str(payload["evaluationId"]), ensure_ascii=False)
    return (
        "window.V4_4_EVALUATION_PACKAGES=window.V4_4_EVALUATION_PACKAGES||{};"
        f"window.V4_4_EVALUATION_PACKAGES[{key}]={compact_json(payload)};\n"
    )


def experiment_markdown(
    *,
    display_name: str,
    start: str,
    end: str,
    current_role: str,
    result_description: str,
) -> str:
    return f"""# 回测结果记录

## 数据范围

- 品种：{display_name}
- 评价开始：{start}
- 评价结束：{end}
- K线周期：15秒

## 当前实验中的用途

{current_role}

目录身份只由品种和实际评价时间决定。训练、测试、跨品种验证等用途写在实验方案中，同一个日期结果包可以被不同实验重复引用。

## 已完成内容

{result_description}

## 保存规则

参数汇总、浏览器精简数据、逐笔交易记录和逐笔入口都归入这个日期结果包。当前历史逐笔页面通过兼容入口保持原有显示；以后生成的结果在自己的日期目录中增量追加批次和逐参数交易分块。
"""


def package_manifest(
    *,
    evaluation_id: str,
    instrument_id: str,
    display_name: str,
    start: str,
    end: str,
    timezone_name: str,
    package_root: Path,
    source_summary: Path,
    source_trades: Path,
    source_trade_sha256: str,
    legacy_trade_review: Path,
    parameter_summary: Path,
    browser_summary: Path,
    experiment_record: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete",
        "evaluation_id": evaluation_id,
        "identity": {
            "instrument_id": instrument_id,
            "display_name": display_name,
            "sample_start": start,
            "sample_end": end,
            "timezone": timezone_name,
            "bar_seconds": 15,
            "directory_naming_rule": "instrument_id/evaluation_start__evaluation_end",
        },
        "role_policy": "experiment roles are declared in comparison plans and experiment records, never in the directory name",
        "storage_mode": "date_based_compatibility_package",
        "artifacts": {
            "parameter_summary": artifact(parameter_summary),
            "browser_summary": artifact(browser_summary),
            "trade_records": {
                "path": str((package_root / "trade_records" / "trades.csv").resolve()),
                "size_bytes": source_trades.stat().st_size,
                "sha256": source_trade_sha256,
                "storage": "same_volume_hard_link_to_immutable_source",
            },
            "trade_review": artifact(package_root / "trade_review" / "index.html"),
            "experiment_record": artifact(experiment_record),
        },
        "provenance": {
            "source_summary": str(source_summary.resolve()),
            "source_trade_records": str(source_trades.resolve()),
            "legacy_trade_review": str(legacy_trade_review.resolve()),
            "comparison_run": str(CURRENT_RUN_ROOT.resolve()),
        },
    }


def strip_role_metrics(row: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, Any]:
    removed = {
        f"{prefix}_{suffix}"
        for prefix in prefixes
        for suffix in ROLE_METRIC_SUFFIXES
    }
    return {key: value for key, value in row.items() if key not in removed}


def reconstruct_rows(
    base_rows: list[dict[str, Any]],
    roles: list[dict[str, str]],
    packages: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    row_maps = {
        role["roleKey"]: {
            str(item["combo_id"]): item for item in packages[role["evaluationId"]]["rows"]
        }
        for role in roles
    }
    output: list[dict[str, Any]] = []
    for base in base_rows:
        combo_id = str(base["combo_id"])
        row = dict(base)
        for role in roles:
            item = row_maps[role["roleKey"]].get(combo_id)
            if item is None:
                raise ValueError(
                    f"evaluation {role['evaluationId']} is missing comparison coordinate {combo_id}"
                )
            prefix = role["outputPrefix"]
            for suffix, value in item["metrics"].items():
                row[f"{prefix}_{suffix}"] = value
        output.append(row)
    return output


def composer_script() -> str:
    return """(()=>{const config=window.V4_4_EVALUATION_COMPARISON_CONFIG,packages=window.V4_4_EVALUATION_PACKAGES||{},maps={};for(const role of config.roles){const payload=packages[role.evaluationId];if(!payload)throw new Error(`缺少回测结果包：${role.evaluationId}`);maps[role.roleKey]=new Map(payload.rows.map(row=>[String(row.combo_id),row]));}const rows=config.baseRows.map(base=>{const values={...base},comboId=String(base.combo_id);for(const role of config.roles){const item=maps[role.roleKey].get(comboId);if(!item)throw new Error(`回测结果包 ${role.evaluationId} 缺少参数 ${comboId}`);for(const [suffix,value] of Object.entries(item.metrics))values[`${role.outputPrefix}_${suffix}`]=value;}const row={};for(const field of config.rowFieldOrder)row[field]=values[field];return row;});window.V4_4_CROSS_INSTRUMENT_DATA={...config.baseData,rows};})();\n"""


def build_current_framework() -> dict[str, Any]:
    original_data = read_js_assignment(
        CURRENT_RUN_ROOT / "comparison_data.js",
        "window.V4_4_CROSS_INSTRUMENT_DATA=",
    )
    original_rows = list(original_data["rows"])
    run_config = json.loads((CURRENT_RUN_ROOT / "run_config.json").read_text(encoding="utf-8"))
    cross_manifest = json.loads(
        (CURRENT_RUN_ROOT / "cross_instrument_manifest.json").read_text(encoding="utf-8")
    )
    snapshot_manifest = json.loads(
        (CURRENT_SNAPSHOT_ROOT / "analysis_manifest.json").read_text(encoding="utf-8")
    )

    specs = [
        {
            "role_key": "source",
            "prefix": "source",
            "instrument_id": "K200",
            "display_name": "K200",
            "start": str(run_config["source"]["sample_start"]),
            "end": str(run_config["source"]["sample_end"]),
            "timezone": "Asia/Seoul",
            "source_summary": CURRENT_SNAPSHOT_ROOT / "analysis_summary.csv",
            "source_trades": CURRENT_SNAPSHOT_ROOT / "union_trades.csv",
            "source_trade_sha256": str(
                snapshot_manifest["artifacts"]["union_trades"]["sha256"]
            ),
            "legacy_review": UNION_ROOT / "trade_review" / "index.html",
            "role_text": "在当前K200／K200／SI比较中，这段数据作为K200训练区间。",
            "description": "保存当前累计训练结果，共37,058组参数；当前比较从中读取350组参数。",
        },
        {
            "role_key": "source_test",
            "prefix": "source_test",
            "instrument_id": "K200",
            "display_name": "K200",
            "start": str(run_config["source_test"]["sample_start"]),
            "end": str(run_config["source_test"]["sample_end"]),
            "timezone": "Asia/Seoul",
            "source_summary": CURRENT_RUN_ROOT / "k200_test_metrics.csv",
            "source_trades": CURRENT_RUN_ROOT / "k200_test_candidate_trades.csv",
            "source_trade_sha256": str(
                cross_manifest["outputs"]["k200_test_candidate_trades.csv"]["sha256"]
            ),
            "legacy_review": CURRENT_RUN_ROOT / "trade_review_k200_test" / "index.html",
            "role_text": "在当前K200／K200／SI比较中，这段数据作为K200后续行情验证区间。",
            "description": "保存当前350组参数在该日期区间的回测结果。",
        },
        {
            "role_key": "target",
            "prefix": "target",
            "instrument_id": "SImain",
            "display_name": "SI",
            "start": str(run_config["target"]["sample_start"]),
            "end": str(run_config["target"]["sample_end"]),
            "timezone": "America/Chicago",
            "source_summary": CURRENT_RUN_ROOT / "migration_comparison.csv",
            "source_trades": CURRENT_RUN_ROOT / "simain_candidate_trades.csv",
            "source_trade_sha256": str(
                cross_manifest["outputs"]["simain_candidate_trades.csv"]["sha256"]
            ),
            "legacy_review": CURRENT_RUN_ROOT / "trade_review" / "index.html",
            "role_text": "在当前K200／K200／SI比较中，这段数据作为SI跨品种验证区间。",
            "description": "保存当前350组参数在SI日期区间的回测结果。",
        },
    ]

    packages: dict[str, dict[str, Any]] = {}
    package_catalog: list[dict[str, Any]] = []
    for spec in specs:
        start = spec["start"]
        end = spec["end"]
        evaluation_id = f"{spec['instrument_id']}_{interval_slug(start, end)}"
        root = package_path(spec["instrument_id"], start, end)
        projection = role_projection(
            original_rows,
            evaluation_id=evaluation_id,
            prefix=spec["prefix"],
            instrument_id=spec["instrument_id"],
            display_name=spec["display_name"],
            start=start,
            end=end,
        )
        browser_summary = root / "browser_summaries" / f"{CANDIDATE_SET_ID}.js"
        atomic_text(browser_summary, browser_summary_script(projection))
        parameter_summary = root / "parameter_summary.csv"
        if spec["role_key"] == "source":
            atomic_csv(parameter_summary, full_k200_training_summary())
        else:
            atomic_csv(parameter_summary, neutral_projection_frame(projection))
        ensure_hard_link(spec["source_trades"], root / "trade_records" / "trades.csv")
        trade_review = root / "trade_review" / "index.html"
        atomic_text(
            trade_review,
            redirect_html(
                relative_href(trade_review.parent, spec["legacy_review"]),
                f"V4.4 {spec['display_name']}逐笔分析",
            ),
        )
        experiment_record = root / "EXPERIMENT.md"
        atomic_text(
            experiment_record,
            experiment_markdown(
                display_name=spec["display_name"],
                start=start,
                end=end,
                current_role=spec["role_text"],
                result_description=spec["description"],
            ),
        )
        manifest = package_manifest(
            evaluation_id=evaluation_id,
            instrument_id=spec["instrument_id"],
            display_name=spec["display_name"],
            start=start,
            end=end,
            timezone_name=spec["timezone"],
            package_root=root,
            source_summary=spec["source_summary"],
            source_trades=spec["source_trades"],
            source_trade_sha256=spec["source_trade_sha256"],
            legacy_trade_review=spec["legacy_review"],
            parameter_summary=parameter_summary,
            browser_summary=browser_summary,
            experiment_record=experiment_record,
        )
        atomic_json(root / "evaluation_manifest.json", manifest)
        packages[evaluation_id] = projection
        package_catalog.append(
            {
                "evaluation_id": evaluation_id,
                "instrument_id": spec["instrument_id"],
                "display_name": spec["display_name"],
                "sample_start": start,
                "sample_end": end,
                "manifest": relative_href(PACKAGE_ROOT, root / "evaluation_manifest.json"),
                "browser_summary": relative_href(PACKAGE_ROOT, browser_summary),
                "trade_review": relative_href(PACKAGE_ROOT, trade_review),
            }
        )
        spec["evaluation_id"] = evaluation_id
        spec["package_root"] = root
        spec["browser_summary"] = browser_summary
        spec["trade_review"] = trade_review

    atomic_json(
        PACKAGE_ROOT / "catalog.json",
        {"schema_version": 1, "status": "complete", "evaluations": package_catalog},
    )
    atomic_text(
        PACKAGE_ROOT / "catalog.js",
        "window.V4_4_EVALUATION_CATALOG=" + compact_json(package_catalog) + ";\n",
    )

    comparison_dir = COMPARISON_ROOT / "comparisons" / COMPARISON_ID
    roles = [
        {
            "roleKey": spec["role_key"],
            "outputPrefix": spec["prefix"],
            "evaluationId": spec["evaluation_id"],
            "displayLabel": spec["role_text"],
        }
        for spec in specs
    ]
    base_rows = strip_role_metrics(original_rows[0], tuple(spec["prefix"] for spec in specs))
    base_keys = set(base_rows)
    base_rows = [
        {key: value for key, value in row.items() if key in base_keys}
        for row in original_rows
    ]
    base_data = {key: value for key, value in original_data.items() if key != "rows"}
    run_root_href = relative_href(comparison_dir, CURRENT_RUN_ROOT)
    base_data["artifacts"] = {
        "frozenCandidates": f"{run_root_href}/frozen_candidates.json",
        "comparisonCsv": f"{run_root_href}/migration_comparison.csv",
        "targetTradesCsv": f"{run_root_href}/simain_candidate_trades.csv",
        "representativeTradesCsv": f"{run_root_href}/representative_trades.csv",
        "runConfig": f"{run_root_href}/run_config.json",
        "tradeReview": relative_href(comparison_dir, specs[2]["trade_review"]),
        "sourceTradeReview": relative_href(comparison_dir, specs[0]["trade_review"]),
        "sourceTestTradeReview": relative_href(comparison_dir, specs[1]["trade_review"]),
        "targetTradeReview": relative_href(comparison_dir, specs[2]["trade_review"]),
        "sourceTestTradesCsv": f"{run_root_href}/k200_test_candidate_trades.csv",
        "sourceTestMetricsCsv": f"{run_root_href}/k200_test_metrics.csv",
        "finalTrainTestSiReport": f"{run_root_href}/FINAL_TRAIN_TEST_SI_REPORT.md",
    }
    for item in base_data.get("runCatalog", []):
        run_id = str(item["run_id"])
        item["href"] = relative_href(
            comparison_dir, CROSS_ROOT / "runs" / run_id / "index.html"
        )
    config = {
        "schemaVersion": 1,
        "comparisonId": COMPARISON_ID,
        "candidateSetId": CANDIDATE_SET_ID,
        "roles": roles,
        "rowFieldOrder": list(original_rows[0]),
        "baseData": base_data,
        "baseRows": base_rows,
    }
    atomic_text(
        comparison_dir / "comparison_config.js",
        "window.V4_4_EVALUATION_COMPARISON_CONFIG=" + compact_json(config) + ";\n",
    )
    atomic_text(comparison_dir / "comparison_data.js", composer_script())

    current_html = (CURRENT_RUN_ROOT / "index.html").read_text(encoding="utf-8")
    old_script = '<script src="comparison_data.js"></script>'
    script_tags = "".join(
        f'<script src="{relative_href(comparison_dir, spec["browser_summary"])}"></script>'
        for spec in specs
    )
    script_tags += '<script src="comparison_config.js"></script><script src="comparison_data.js"></script>'
    if current_html.count(old_script) != 1:
        raise ValueError("current comparison page no longer has the expected data-script boundary")
    framework_html = current_html.replace(old_script, script_tags)
    atomic_text(comparison_dir / "index.html", framework_html)

    comparison_plan = {
        "schema_version": 1,
        "status": "complete",
        "comparison_id": COMPARISON_ID,
        "candidate_set_id": CANDIDATE_SET_ID,
        "directory_identity_rule": "instrument and exact evaluation interval",
        "role_identity_rule": "roles belong to this experiment plan and do not change package directory names",
        "evaluations": [
            {
                "role": spec["role_key"],
                "evaluation_id": spec["evaluation_id"],
                "manifest": str((spec["package_root"] / "evaluation_manifest.json").resolve()),
                "experiment_role": spec["role_text"],
            }
            for spec in specs
        ],
        "source_comparison_run": str(CURRENT_RUN_ROOT.resolve()),
        "visible_behavior_contract": "same rows, values, controls, sorting, and per-trade destinations as the current 350-coordinate page",
    }
    atomic_json(comparison_dir / "comparison_plan.json", comparison_plan)
    atomic_text(
        comparison_dir / "EXPERIMENT_REPORT.md",
        "# 实验比较记录\n\n"
        "本次比较读取三个按品种和日期保存的回测结果包。目录名称不承担训练、测试或迁移角色；"
        "这些用途由 `comparison_plan.json` 声明。页面保持当前350组K200训练、K200后续行情和SI结果的显示与逐笔链接。\n",
    )

    reconstructed = reconstruct_rows(base_rows, roles, packages)
    if reconstructed != original_rows:
        raise ValueError("date-based package composition differs from the current 350-row comparison")
    normalized_framework_html = framework_html.replace(script_tags, old_script)
    if normalized_framework_html != current_html:
        raise ValueError("comparison page changed outside the data-source script boundary")

    protected_paths = [
        UNION_ROOT / "index.html",
        UNION_ROOT / "trade_review" / "index.html",
        CURRENT_RUN_ROOT / "index.html",
        CURRENT_RUN_ROOT / "trade_review" / "index.html",
        CURRENT_RUN_ROOT / "trade_review_k200_test" / "index.html",
    ]
    protected_hashes = {str(path.resolve()): sha256(path) for path in protected_paths}
    old_cross_entry = (CROSS_ROOT / "index.html").read_text(encoding="utf-8")
    compatibility_dir = COMPARISON_ROOT / "compatibility"
    atomic_text(compatibility_dir / "cross_instrument_entry_before_switch.html", old_cross_entry)
    atomic_text(
        COMPARISON_ROOT / "index.html",
        redirect_html(
            relative_href(COMPARISON_ROOT, comparison_dir / "index.html"),
            "V4.41 回测结果比较",
        ),
    )
    atomic_text(
        COMPARISON_ROOT / "comparison_catalog.js",
        "window.V4_4_COMPARISON_CATALOG="
        + compact_json(
            [
                {
                    "comparison_id": COMPARISON_ID,
                    "candidate_set_id": CANDIDATE_SET_ID,
                    "href": relative_href(COMPARISON_ROOT, comparison_dir / "index.html"),
                    "evaluation_ids": [spec["evaluation_id"] for spec in specs],
                }
            ]
        )
        + ";\n",
    )
    atomic_text(
        CROSS_ROOT / "index.html",
        redirect_html(
            relative_href(CROSS_ROOT, COMPARISON_ROOT / "index.html"),
            "V4.41 跨品种对比",
        ),
    )
    protected_after = {str(path.resolve()): sha256(path) for path in protected_paths}
    if protected_hashes != protected_after:
        raise ValueError("an existing main or per-trade page changed during framework publication")

    audit = {
        "schema_version": 1,
        "status": "passed",
        "comparison_id": COMPARISON_ID,
        "candidate_count": len(original_rows),
        "row_data_exact_match": True,
        "comparison_html_unchanged_outside_script_sources": True,
        "existing_main_and_trade_html_unchanged": True,
        "protected_hashes": protected_after,
        "stable_cross_entry": artifact(CROSS_ROOT / "index.html"),
        "framework_entry": artifact(COMPARISON_ROOT / "index.html"),
        "comparison_entry": artifact(comparison_dir / "index.html"),
    }
    atomic_json(COMPARISON_ROOT / "compatibility_audit.json", audit)
    atomic_json(
        COMPARISON_ROOT / "framework_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "comparison": comparison_plan,
            "evaluation_catalog": str((PACKAGE_ROOT / "catalog.json").resolve()),
            "compatibility_audit": str((COMPARISON_ROOT / "compatibility_audit.json").resolve()),
            "stable_entry": str((CROSS_ROOT / "index.html").resolve()),
            "legacy_run_retained": str(CURRENT_RUN_ROOT.resolve()),
        },
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build date-based V4.4 evaluation packages and a compatibility comparison entry."
    )
    parser.add_argument("command", choices=("build-current",), nargs="?", default="build-current")
    args = parser.parse_args()
    if args.command == "build-current":
        print(json.dumps(build_current_framework(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
