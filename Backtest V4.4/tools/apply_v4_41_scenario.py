from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = ROOT / "runtime_inputs" / "scenarios"
DEFAULT_MARKET_CATALOG = SCENARIO_ROOT / "market_catalog.json"
DEFAULT_SCENARIO_CATALOG = SCENARIO_ROOT / "scenario_catalog.json"
EVALUATION_ROOT = ROOT / "results" / "evaluation_packages"
CURRENT_MAIN_ROOT = ROOT / "results" / "all_completed_union_analysis" / "main"
CURRENT_MAIN_HTML = CURRENT_MAIN_ROOT / "index.html"
CURRENT_MAIN_DATA = CURRENT_MAIN_ROOT / "analysis_data.js"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "scenario_analysis"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def read_assignment(path: Path, marker: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if marker not in text:
        raise ValueError(f"JavaScript payload marker is missing: {path}")
    payload = text.split(marker, 1)[1].strip()
    if not payload.endswith(";"):
        raise ValueError(f"JavaScript payload is incomplete: {path}")
    return json.loads(payload[:-1])


def evaluation_directory(evaluation_id: str) -> Path:
    catalog = json.loads((EVALUATION_ROOT / "catalog.json").read_text(encoding="utf-8"))
    item = next(
        (row for row in catalog["evaluations"] if row["evaluation_id"] == evaluation_id),
        None,
    )
    if item is None:
        raise ValueError(f"evaluation package is not registered: {evaluation_id}")
    return (EVALUATION_ROOT / Path(item["manifest"]).parent).resolve()


def browser_projection(package_root: Path, evaluation_id: str) -> dict[str, Any]:
    candidates = sorted((package_root / "browser_summaries").glob("*.js"))
    if not candidates:
        raise ValueError(f"evaluation package has no browser summary: {evaluation_id}")
    marker = f'window.V4_4_EVALUATION_PACKAGES["{evaluation_id}"]='
    return read_assignment(candidates[0], marker)


def current_main_payload() -> dict[str, Any]:
    return read_assignment(CURRENT_MAIN_DATA, "window.V4_ANALYSIS_DATA=")


def map_projection_rows(
    projection: dict[str, Any], market: dict[str, Any]
) -> list[dict[str, Any]]:
    cost_bps = float(market["round_trip_cost_bps"])
    cost_fraction = cost_bps / 10000.0
    rows: list[dict[str, Any]] = []
    for item in projection["rows"]:
        parameters = dict(item["parameters"])
        metrics = item["metrics"]
        cost_mean = metrics.get("cost_mean_trade")
        gross_mean = metrics.get("gross_mean_trade")
        if gross_mean is None and cost_mean is not None:
            gross_mean = float(cost_mean) + cost_fraction
        row = {
            **parameters,
            "combo_id": item["combo_id"],
            "train_trade_count": metrics.get("trade_count", 0),
            "train_return": metrics.get("gross_total_return", 0.0),
            "train_avg_trade": gross_mean,
            "train_max_drawdown_abs": metrics.get(
                "gross_max_drawdown_abs", metrics.get("cost_max_drawdown_abs", 0.0)
            ),
            "train_cost_adjusted_return": metrics.get("cost_total_return", 0.0),
            "train_cost_adjusted_avg_trade": cost_mean,
            "train_cost_adjusted_max_drawdown_abs": metrics.get(
                "cost_max_drawdown_abs", 0.0
            ),
            "round_trip_cost_bps": cost_bps,
            "train_return_excluding_gap_spanning_trades": metrics.get(
                "non_gap_cost_total_return"
            ),
            "gap_spanning_trade_count": metrics.get("gap_trade_count", 0),
            "rebound_exit_count": metrics.get("rebound_exit_count"),
            "speed_exit_count": metrics.get("speed_exit_count"),
            "holding_bar_distance_median": metrics.get("holding_bar_distance_median"),
            "holding_bar_distance_p95": metrics.get("holding_bar_distance_p95"),
        }
        rows.append(row)
    return rows


def base_rows(market: dict[str, Any], package_root: Path) -> tuple[list[dict], dict]:
    if market["evaluation_id"] == "K200_20260526T000000__20260708T235200":
        payload = current_main_payload()
        return list(payload["rows"]), payload
    projection = browser_projection(package_root, market["evaluation_id"])
    rows = map_projection_rows(projection, market)
    return rows, {
        "baselineSamplingPolicies": sorted(
            {str(row["baseline_sampling_policy"]) for row in rows}
        ),
        "costModel": {
            "instrument_name": market["label_zh"],
            "round_trip_total_cost_bps": market["round_trip_cost_bps"],
            "description": (
                f"{market['label_zh']}：采用评估包保存的 "
                f"{market['round_trip_cost_bps']:.4f} bps 往返成本。"
            ),
        },
    }


def evaluate_from_trades(
    scenario: dict[str, Any],
    rule: dict[str, Any],
    trade_path: Path,
    combo_ids: set[str],
) -> tuple[set[str], pd.DataFrame]:
    required_columns = {"combo_id", "entry_time", "exit_time", "exit_reason"}
    columns = set(pd.read_csv(trade_path, nrows=0).columns)
    missing = required_columns.difference(columns)
    if missing:
        raise ValueError(f"trade records lack scenario fields: {sorted(missing)}")

    state: dict[tuple[str, str], dict[str, Any]] = {}
    segments = []
    for segment in scenario["segments"]:
        begin = pd.Timestamp(segment["start_time"])
        finish = pd.Timestamp(segment["end_time"])
        if begin >= finish:
            raise ValueError(f"scenario segment start must precede end: {segment['segment_id']}")
        segments.append((segment, begin, finish))

    for chunk in pd.read_csv(
        trade_path,
        usecols=sorted(required_columns),
        chunksize=250_000,
        low_memory=False,
    ):
        chunk["combo_id"] = chunk["combo_id"].astype(str)
        chunk = chunk.loc[chunk["combo_id"].isin(combo_ids)]
        if chunk.empty:
            continue
        chunk["entry_time"] = pd.to_datetime(chunk["entry_time"], errors="raise")
        chunk["exit_time"] = pd.to_datetime(chunk["exit_time"], errors="raise")
        for segment, begin, finish in segments:
            segment_id = str(segment["segment_id"])
            entered = chunk.loc[
                chunk["entry_time"].gt(begin) & chunk["entry_time"].le(finish)
            ]
            exited = chunk.loc[
                chunk["exit_time"].gt(begin) & chunk["exit_time"].le(finish)
            ]
            if not entered.empty:
                grouped = entered.groupby("combo_id", sort=False)
                for combo_id, group in grouped:
                    key = (str(combo_id), segment_id)
                    item = state.setdefault(
                        key,
                        {"entry_count": 0, "exit_count": 0, "selected": None},
                    )
                    if item["entry_count"] == 0:
                        first = group.iloc[0]
                        item["selected"] = {
                            "entry_time": str(first["entry_time"]),
                            "exit_time": str(first["exit_time"]),
                            "exit_reason": str(first["exit_reason"]),
                        }
                    item["entry_count"] += int(len(group))
            if not exited.empty:
                for combo_id, count in exited.groupby("combo_id", sort=False).size().items():
                    key = (str(combo_id), segment_id)
                    item = state.setdefault(
                        key,
                        {"entry_count": 0, "exit_count": 0, "selected": None},
                    )
                    item["exit_count"] += int(count)

    allowed_reasons = set(rule["required_eventual_exit_reasons"])
    details: list[dict[str, Any]] = []
    qualified: set[str] = set()
    for combo_id in sorted(combo_ids):
        segment_flags = []
        for segment, _begin, finish in segments:
            segment_id = str(segment["segment_id"])
            item = state.get(
                (combo_id, segment_id),
                {"entry_count": 0, "exit_count": 0, "selected": None},
            )
            selected = item["selected"] if item["entry_count"] == 1 else None
            holds_past_end = bool(
                selected is not None and pd.Timestamp(selected["exit_time"]) > finish
            )
            flag = bool(
                item["entry_count"] == int(rule["required_entry_count"])
                and item["exit_count"] == int(rule["required_exit_count"])
                and holds_past_end
                and selected["exit_reason"] in allowed_reasons
            )
            segment_flags.append(flag)
            details.append(
                {
                    "combo_id": combo_id,
                    "segment_id": segment_id,
                    "entry_count": item["entry_count"],
                    "exit_count": item["exit_count"],
                    "holds_past_end": holds_past_end,
                    "eventual_exit_reason": "" if selected is None else selected["exit_reason"],
                    "qualified": flag,
                }
            )
        if all(segment_flags):
            qualified.add(combo_id)
    return qualified, pd.DataFrame(details)


def qualified_ids(
    scenario: dict[str, Any],
    rule: dict[str, Any],
    package_root: Path,
    rows: list[dict[str, Any]],
) -> tuple[set[str], pd.DataFrame, str]:
    combo_ids = {str(row["combo_id"]) for row in rows}
    precomputed = scenario.get("precomputed_qualification_field")
    if precomputed:
        qualified = {
            str(row["combo_id"]) for row in rows if bool(row.get(str(precomputed)))
        }
        details = pd.DataFrame(
            {
                "combo_id": sorted(combo_ids),
                "qualified": [combo_id in qualified for combo_id in sorted(combo_ids)],
                "qualification_source": str(precomputed),
            }
        )
        return qualified, details, "precomputed_compatible_v4_4_field"
    qualified, details = evaluate_from_trades(
        scenario,
        rule,
        package_root / "trade_records" / "trades.csv",
        combo_ids,
    )
    return qualified, details, "streamed_immutable_trade_records"


def relative_href(source_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), source_dir.resolve())).as_posix()


def scenario_html(
    output_dir: Path,
    market: dict[str, Any],
    scenario: dict[str, Any],
) -> str:
    html = CURRENT_MAIN_HTML.read_text(encoding="utf-8")
    title = f"V4.41 {market['label_zh']} · {scenario['label_zh']}结果排序"
    html = html.replace("V4.41 K200回测结果排序", title)
    html = html.replace(
        "const scenarios=[['all','全部坐标'],['scenario_1','情景一'],['scenario_2','情景二'],['scenario_3','情景三']];",
        "const scenarios=Array.isArray(DATA.scenarios)?DATA.scenarios:[['all','全部坐标']];",
    )
    html = html.replace(
        "const booleanKeys=new Set(['scenario_1_qualified','scenario_2_qualified','scenario_3_qualified']);",
        "const booleanKeys=new Set(scenarios.filter(item=>item[0]!=='all').map(item=>item[0]+'_qualified'));",
    )
    fixed_columns = (
        "['scenario_1_qualified','情景一'],['scenario_2_qualified','情景二'],['scenario_3_qualified','情景三'],\n"
        "    ['scenario_3_qualified_segment_count','情景三分段'],['scenario_3_failed_segment_ids','失败行情'],['combo_id','参数组合'],"
    )
    dynamic_columns = (
        "...scenarios.filter(item=>item[0]!=='all').map(item=>[item[0]+'_qualified',item[1]]),\n"
        "    ['combo_id','参数组合'],"
    )
    html = html.replace(fixed_columns, dynamic_columns)
    html = html.replace(
        "const scenarioEvidence=DATA.scenario3QualifiedCount?`情景三合格 ${number(DATA.scenario3QualifiedCount)} 个坐标`:'情景三为真实空集';",
        "const scenarioEvidence=DATA.sceneSummary||'当前页面仅包含符合所选场景的参数';",
    )
    html = html.replace(
        "const costText=`${cm.instrument_name||cm.instrument_id} 参考价 ${number(cm.reference_price)} × ${number(cm.point_value)} ${cm.quote_currency}/点＝${number(cm.contract_notional_quote)} ${cm.quote_currency} 名义价值；${number(cm.round_trip_slippage_bps)} bps 往返滑点 + ${number(cm.round_trip_commission)} ${cm.commission_currency} 往返手续费，合计 ${number(cm.round_trip_total_cost_bps)} bps。`;",
        "const costText=cm.description||`${cm.instrument_name||cm.instrument_id} 参考价 ${number(cm.reference_price)} × ${number(cm.point_value)} ${cm.quote_currency}/点＝${number(cm.contract_notional_quote)} ${cm.quote_currency} 名义价值；${number(cm.round_trip_slippage_bps)} bps 往返滑点 + ${number(cm.round_trip_commission)} ${cm.commission_currency} 往返手续费，合计 ${number(cm.round_trip_total_cost_bps)} bps。`;",
    )
    cross_target = (
        ROOT / "results" / "evaluation_comparison" / "index.html"
    ).resolve()
    html = re.sub(
        r'href="[^\"]*cross_instrument_comparison/index\.html"',
        f'href="{relative_href(output_dir, cross_target)}"',
        html,
        count=1,
    )
    return html


def high_return_views(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "scene_total_return",
            "label": f"{scenario['label_zh']} · 总收益",
            "scenario_filter": "all",
            "minimum_trade_count": 0,
            "metric": "total_return",
        },
        {
            "id": "scene_average_return_ge10",
            "label": f"{scenario['label_zh']} · 至少 10 笔 · 笔均收益",
            "scenario_filter": "all",
            "minimum_trade_count": 10,
            "metric": "average_trade",
        },
        {
            "id": "scene_average_return_ge20",
            "label": f"{scenario['label_zh']} · 至少 20 笔 · 笔均收益",
            "scenario_filter": "all",
            "minimum_trade_count": 20,
            "metric": "average_trade",
        },
    ]


def apply_one(
    scenario: dict[str, Any],
    rule: dict[str, Any],
    markets: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    scenario_id = str(scenario["scenario_id"])
    if not SAFE_ID.fullmatch(scenario_id):
        raise ValueError(f"unsafe scenario_id: {scenario_id}")
    market = markets.get(str(scenario["market_id"]))
    if market is None:
        raise ValueError(f"scenario references unknown market: {scenario['market_id']}")
    if scenario.get("evaluation_id") != market["evaluation_id"]:
        raise ValueError("scenario evaluation_id differs from its market catalog entry")
    if scenario.get("aggregation") != "all" or not scenario.get("segments"):
        raise ValueError("scenario must contain one or more AND-aggregated segments")

    package_root = evaluation_directory(market["evaluation_id"])
    rows, source_payload = base_rows(market, package_root)
    qualified, details, qualification_source = qualified_ids(
        scenario, rule, package_root, rows
    )
    scene_field = f"{scenario_id}_qualified"
    selected_rows = []
    for row in rows:
        combo_id = str(row["combo_id"])
        if combo_id not in qualified:
            continue
        item = dict(row)
        item[scene_field] = True
        selected_rows.append(item)

    output_dir = (output_root / scenario_id).resolve()
    trade_review = package_root / "trade_review" / "index.html"
    payload = {
        "coordinateCount": len(selected_rows),
        "tradeCount": int(sum(int(row.get("train_trade_count") or 0) for row in selected_rows)),
        "scenario3QualifiedCount": len(selected_rows),
        "highReturnViews": high_return_views(scenario),
        "baselineSamplingPolicies": source_payload.get("baselineSamplingPolicies", []),
        "costModel": source_payload.get("costModel", {}),
        "scopeLabel": f"{market['label_zh']} · {scenario['label_zh']}",
        "nativeTradeRoute": relative_href(output_dir, trade_review)
        + "?combo_id={combo_id}",
        "scenarioRequirementsRoute": "scenario.json?scenario={scenario_id}",
        "scenarios": [["all", "全部符合参数"], [scenario_id, scenario["label_zh"]]],
        "sceneSummary": (
            f"{scenario['label_zh']}包含 {len(scenario['segments'])} 段行情；"
            f"符合 {len(selected_rows)} / {len(rows)} 个参数。"
        ),
        "rows": selected_rows,
    }
    atomic_text(
        output_dir / "analysis_data.js",
        "window.V4_ANALYSIS_DATA="
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
    )
    atomic_text(output_dir / "index.html", scenario_html(output_dir, market, scenario))
    atomic_text(
        output_dir / "scenario.json",
        json.dumps(scenario, ensure_ascii=False, indent=2) + "\n",
    )
    details.to_csv(output_dir / "qualification_details.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "scenario_id": scenario_id,
        "market_id": market["market_id"],
        "evaluation_id": market["evaluation_id"],
        "qualification_source": qualification_source,
        "source_coordinate_count": len(rows),
        "qualified_coordinate_count": len(selected_rows),
        "entry": str((output_dir / "index.html").resolve()),
        "source_evaluation_package": str(package_root),
    }
    atomic_text(
        output_dir / "scenario_application_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply saved market scenarios to existing evaluation packages."
    )
    parser.add_argument("--scenario-id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--scenario-catalog", type=Path, default=DEFAULT_SCENARIO_CATALOG)
    parser.add_argument("--market-catalog", type=Path, default=DEFAULT_MARKET_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.all == bool(args.scenario_id):
        parser.error("choose exactly one of --scenario-id or --all")

    scenario_catalog = json.loads(args.scenario_catalog.read_text(encoding="utf-8"))
    market_catalog = json.loads(args.market_catalog.read_text(encoding="utf-8"))
    scenarios = list(scenario_catalog["scenarios"])
    if args.scenario_id:
        scenarios = [
            row for row in scenarios if row["scenario_id"] == args.scenario_id
        ]
        if not scenarios:
            raise ValueError(f"scenario is not registered: {args.scenario_id}")
    markets = {row["market_id"]: row for row in market_catalog["markets"]}
    results = [
        apply_one(
            scenario,
            scenario_catalog["qualification_rule"],
            markets,
            args.output_root,
        )
        for scenario in scenarios
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
