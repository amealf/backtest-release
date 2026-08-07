from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SEOUL_ROLL = "2026-06-11T18:00:00+09:00"
ORIGINAL_RUN = Path(
    r"F:\Backtest test 6.11\02_DATA_AND_AUDITS\market_data"
    r"\k200_historical_ticks\k200_ticks_20260523_to_20260723_20260723_014534"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_retained_run(path: Path) -> Path:
    if path.is_dir():
        return path
    old_root = r"F:\Backtest test 6.11\data"
    retained_root = r"F:\Backtest test 6.11\02_DATA_AND_AUDITS\market_data"
    relocated = Path(str(path).replace(old_root, retained_root, 1))
    return relocated if relocated.is_dir() else path


def lineage_manifests(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    chain: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    current = run_dir.resolve()
    while current.is_dir() and str(current).casefold() not in seen:
        seen.add(str(current).casefold())
        manifest_path = current / "run_manifest.json"
        if not manifest_path.is_file():
            break
        manifest = read_json(manifest_path)
        chain.append((current, manifest))
        parent_text = manifest.get("lineage", {}).get("parent_run")
        if not parent_text:
            break
        current = resolve_retained_run(Path(parent_text))
    if not any(path == ORIGINAL_RUN for path, _ in chain) and ORIGINAL_RUN.is_dir():
        chain.append((ORIGINAL_RUN, read_json(ORIGINAL_RUN / "run_manifest.json")))
    chain.reverse()
    return chain


def contract_label(manifest: dict[str, Any]) -> str:
    policy = manifest.get("main_contract_policy")
    if policy:
        return f"{policy['before_roll']} → {policy['from_roll']}"
    contract = manifest.get("contract", {})
    return str(contract.get("key", "未记录"))


def update_time(manifest: dict[str, Any]) -> str:
    for key in ("completed_at_utc", "last_started_at_utc", "created_at_utc"):
        if manifest.get(key):
            return str(manifest[key])
    return "未记录"


def seoul_time(epoch: Any, fallback: str) -> str:
    if epoch is None:
        return fallback
    return datetime.fromtimestamp(int(epoch), ZoneInfo("Asia/Seoul")).isoformat()


def write_readme(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    manifest = read_json(run_dir / "run_manifest.json")
    request = manifest.get("request", {})
    contract = manifest.get("contract", {})
    outputs = manifest.get("outputs") or {}
    state = manifest.get("download_state") or {}
    chain = lineage_manifests(run_dir)

    history_rows = []
    for path, item in chain:
        item_request = item.get("request", {})
        history_rows.append(
            "| {time} | `{start}` | `{end}` | `{contract}` | `{status}` | `{path}` |".format(
                time=update_time(item),
                start=item_request.get("start_seoul", "未记录"),
                end=item_request.get("end_seoul_exclusive", "未记录"),
                contract=contract_label(item),
                status=item.get("status", "未记录"),
                path=path,
            )
        )

    tick_count = outputs.get("tick_count", state.get("tick_count", "下载中"))
    actual_first = seoul_time(
        outputs.get("first_tick_epoch"), "下载完成后记录"
    )
    actual_last = seoul_time(
        outputs.get("last_tick_epoch"), "下载完成后记录"
    )
    clean_path = run_dir / "data_clean" / "k200_clean_15s_session_filled.csv"
    active_candidate = run_dir / "data_extended" / "k200_clean_15s_session_filled_latest.csv"

    lines = [
        "# K200 数据下载说明",
        "",
        "## 本次下载",
        "",
        f"- 状态：`{manifest.get('status', '未记录')}`",
        "- 数据源：IBKR Historical `TRADES` Tick",
        "- 时区：`Asia/Seoul`",
        f"- 请求区间：`{request.get('start_seoul', '未记录')}` 至 `{request.get('end_seoul_exclusive', '未记录')}`（右端不包含）",
        f"- 实际首笔：`{actual_first}`",
        f"- 实际末笔：`{actual_last}`",
        f"- Tick 数：`{tick_count}`",
        f"- 本次合约：`{contract.get('key', contract_label(manifest))}`",
        f"- 是否后复权：`{contract.get('back_adjustment', False)}`",
        f"- 创建时间：`{manifest.get('created_at_utc', '未记录')}`",
        f"- 最近启动时间：`{manifest.get('last_started_at_utc', '未记录')}`",
        f"- 完成时间：`{manifest.get('completed_at_utc', '尚未完成')}`",
        "",
        "## 主力合约规则",
        "",
        "K200 的一个交易日按 `18:00` 至次日 `17:59`（Asia/Seoul）计算。候选合约分别汇总该交易日已下载的一分钟成交量；总成交量最大的合约为主力合约，成交量相同则选择到期日更早的合约。",
        "",
        f"当前连续序列复用已审计的切割点 `{SEOUL_ROLL}`：该时刻以前使用 `016M_20260611`，从该时刻起使用 `016U_20260910`。当前补充区间沿用 `016U_20260910`，没有重新执行候选合约成交量比较。发生下一次换月以前，必须重新下载候选合约并生成独立的成交量审计，确认新切割点。",
        "",
        "## 合约切割",
        "",
        "| 区间 | 使用合约 | 边界规则 |",
        "|---|---|---|",
        f"| 数据起点至 `{SEOUL_ROLL}` | `016M_20260611` | 右端不包含 |",
        f"| `{SEOUL_ROLL}` 起 | `016U_20260910` | 左端包含 |",
        "",
        "连续序列不做后复权，换月处的真实价差保留。",
        "",
        "## 合并规则",
        "",
        "1. 每次增量下载从上一批请求区间的右端不包含时间开始，两个请求区间不能重叠。",
        "2. 各批原始 Tick、SQLite 检查点和审计文件保留在各自目录；原始批次不互相覆盖。",
        "3. Tick 生成 15 秒 OHLCV；只在已有成交的 KRX 交易时段片段内补空柱，补空柱沿用前收盘价，成交量与成交笔数为零。",
        "4. 清理流程排除非正价格/成交量记录，并应用同一 15 秒柱内的一秒即时恢复异常 Tick 规则。",
        "5. 活动 15 秒序列按 `epoch_seconds` 递增追加；字段结构必须一致，重复或倒序时间戳禁止合并，真实时段缺口保留。",
        "",
        "## 更新时间记录",
        "",
        "| 更新时间（UTC） | 请求开始（Asia/Seoul） | 请求结束，不包含（Asia/Seoul） | 合约 | 状态 | 下载目录 |",
        "|---|---|---|---|---|---|",
        *history_rows,
        "",
        "## 主要文件",
        "",
        f"- 运行清单：`{run_dir / 'run_manifest.json'}`",
        f"- 下载进度：`{run_dir / 'download_progress.json'}`",
        f"- 原始 Tick：`{run_dir / 'k200_main_historical_ticks.csv.gz'}`",
        f"- 清理后 15 秒数据：`{clean_path}`",
        f"- 合并候选：`{active_candidate}`",
        "",
        "README 在下载创建、恢复和完成时刷新；运行清单与审计 JSON 保存机器可读的精确字段和哈希。",
        "",
    ]
    destination = run_dir / "README.md"
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a K200 data-download README.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(write_readme(args.run_dir))


if __name__ == "__main__":
    main()
