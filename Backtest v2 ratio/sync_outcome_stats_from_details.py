from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_DIR = Path(
    r"D:\Code\backtest-release\Backtest v2 ratio\result\xagusd_30s_all long_momentum outcome"
)
RESULT_DIR_FALLBACKS = [
    RESULT_DIR,
    Path(
        r"D:\Code\backtest-release\Backtest v2 ratio\result\xagusd_30s_all long outcome"
    ),
]
RUN_NAME_FILTER = "period_5min 20250601-20250615"
DRY_RUN = False
PROGRESS_EVERY = 20


SUMMARY_COLUMNS = [
    "open_bar",
    "close_bar",
    "open_threshold",
    "open_continous_threshold",
    "open_withdrawal_threshold",
    "close_threshold",
    "close_withdrawal_threshold",
    "withdrawal_close_count",
    "speed_close_count",
    "final_capital",
    "total_return_pct",
    "outcome_high",
    "biggest_wd_abs",
    "biggest_wd_pct",
    "trade_num",
    "win_rate_pct",
    "avg_trade_return_pct",
    "median_trade_return_pct",
    "payoff_ratio",
    "profit_factor",
    "sharpe_ratio",
    "capital",
    "biggest_wd",
]

PARAM_PREFIX_TO_COLUMN = {
    "om": "open_bar",
    "cm": "close_bar",
    "o": "open_threshold",
    "opm": "open_threshold",
    "oc": "open_continous_threshold",
    "ocpm": "open_continous_threshold",
    "ow": "open_withdrawal_threshold",
    "owm": "open_withdrawal_threshold",
    "c": "close_threshold",
    "cpm": "close_threshold",
    "cw": "close_withdrawal_threshold",
    "cwm": "close_withdrawal_threshold",
}

SUMMARY_PROGRAM_TAGS = (
    "long_momentum_ratio",
    "long_momentum_ATR",
    "long_momentum_ARCH",
    "long_momentum",
)


@dataclass(frozen=True)
class DetailFile:
    path: Path
    run_name: str
    param_tag: str
    count_tag: str
    final_capital_from_name: float
    mtime_ns: int


def resolve_result_dir(result_dir_value) -> Path:
    if result_dir_value not in (None, ""):
        candidate = Path(result_dir_value).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"result dir does not exist: {candidate}")

    for candidate in RESULT_DIR_FALLBACKS:
        resolved = Path(candidate).expanduser().resolve()
        if resolved.exists():
            return resolved

    checked = "\n".join(str(Path(candidate).expanduser().resolve()) for candidate in RESULT_DIR_FALLBACKS)
    raise FileNotFoundError(
        "result dir does not exist. Checked:\n" + checked
    )


def detect_summary_program_tag(result_dir: Path) -> str:
    lowered = result_dir.name.lower()
    for tag in SUMMARY_PROGRAM_TAGS:
        if tag.lower() in lowered:
            return tag
    return "long_momentum"


def resolve_summary_path(
    outcome_stats_dir: Path,
    run_name: str,
    result_dir: Path,
) -> Path:
    expected_name = (
        detect_summary_program_tag(result_dir)
        + " "
        + run_name
        + " outcome_stats.xlsx"
    )
    preferred = outcome_stats_dir / expected_name
    if preferred.exists():
        return preferred

    exact_legacy = outcome_stats_dir / f"{run_name} outcome_stats.xlsx"
    if exact_legacy.exists():
        return exact_legacy

    matches = sorted(
        outcome_stats_dir.glob(f"* {run_name} outcome_stats.xlsx"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if matches:
        return matches[0]
    return preferred


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(col).strip().lower(): str(col) for col in df.columns}
    for name in candidates:
        hit = lower_map.get(name.lower())
        if hit:
            return hit
    return None


def parse_number(text: str) -> float:
    return float(text)


def parse_detail_filename(path: Path, kind: str) -> DetailFile | None:
    suffix = f" {kind}.xlsx"
    name = path.name
    if not name.lower().endswith(suffix):
        return None

    stem = name[: -len(suffix)]
    left, sep, right = stem.rpartition(" Long ")
    if not sep:
        return None

    right_parts = right.rsplit(" ", 1)
    if len(right_parts) != 2:
        return None
    run_name, final_capital_text = right_parts

    prefix_parts = left.split()
    count_tag = ""
    if prefix_parts and re.fullmatch(r"\d+\+\d+", prefix_parts[-1]):
        count_tag = prefix_parts[-1]
        param_tag = " ".join(prefix_parts[:-1]).strip()
    else:
        param_tag = left.strip()

    if not param_tag or not run_name:
        return None

    try:
        final_capital_from_name = parse_number(final_capital_text)
    except ValueError:
        final_capital_from_name = math.nan

    return DetailFile(
        path=path,
        run_name=run_name.strip(),
        param_tag=param_tag,
        count_tag=count_tag,
        final_capital_from_name=final_capital_from_name,
        mtime_ns=path.stat().st_mtime_ns,
    )


def parse_param_tag(param_tag: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for token in str(param_tag).split():
        match = re.fullmatch(r"([A-Za-z_]+)([-+]?\d+(?:\.\d+)?)", token)
        if not match:
            continue
        prefix, raw_value = match.groups()
        column = PARAM_PREFIX_TO_COLUMN.get(prefix.lower())
        if not column:
            continue
        values[column] = float(raw_value)

    if "close_bar" not in values and "open_bar" in values:
        values["close_bar"] = values["open_bar"]
    if "open_withdrawal_threshold" not in values and "open_threshold" in values:
        values["open_withdrawal_threshold"] = values["open_threshold"]
    if "close_threshold" not in values and "open_threshold" in values:
        values["close_threshold"] = values["open_threshold"]
    if "close_withdrawal_threshold" not in values:
        if "open_withdrawal_threshold" in values:
            values["close_withdrawal_threshold"] = values["open_withdrawal_threshold"]
        elif "close_threshold" in values:
            values["close_withdrawal_threshold"] = values["close_threshold"]
    return values


def get_outcome_withdrawal(series: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if clean.empty:
        return math.nan, math.nan
    running_high = clean.cummax()
    biggest_wd_abs = float((running_high - clean).max())
    outcome_high = float(clean.max())
    return outcome_high, biggest_wd_abs


def read_excel_subset(path: Path, allowed_columns: set[str]) -> pd.DataFrame:
    return pd.read_excel(
        path,
        usecols=lambda column: str(column).strip().lower() in allowed_columns,
    )


def extract_capital_curve(perf_df: pd.DataFrame, trans_df: pd.DataFrame) -> pd.Series:
    capital_col = pick_col(perf_df, ["Capital", "capital"])
    if capital_col:
        capital_curve = pd.to_numeric(perf_df[capital_col], errors="coerce").dropna().reset_index(drop=True)
        if not capital_curve.empty:
            return capital_curve

    capital_col = pick_col(trans_df, ["Capital", "capital"])
    if capital_col:
        sell_curve = pd.to_numeric(trans_df[capital_col], errors="coerce").dropna().reset_index(drop=True)
        if not sell_curve.empty:
            return pd.concat([pd.Series([100.0]), sell_curve], ignore_index=True)

    return pd.Series(dtype=float)


def build_summary_row(perf_path: Path, trans_path: Path, param_tag: str) -> dict[str, float]:
    perf_df = read_excel_subset(perf_path, {"capital"})
    trans_df = read_excel_subset(trans_path, {"type", "close_type", "capital"})

    capital_curve = extract_capital_curve(perf_df, trans_df)
    initial_capital = float(capital_curve.iloc[0]) if len(capital_curve) else 100.0
    final_capital = float(capital_curve.iloc[-1]) if len(capital_curve) else float("nan")
    total_return_pct = (
        (final_capital / initial_capital - 1.0) * 100.0
        if initial_capital and not pd.isna(final_capital)
        else math.nan
    )

    outcome_high, biggest_wd_abs = get_outcome_withdrawal(capital_curve)
    biggest_wd_pct = math.nan
    if not pd.isna(outcome_high) and outcome_high != 0 and not pd.isna(biggest_wd_abs):
        biggest_wd_pct = (biggest_wd_abs / outcome_high) * 100.0

    type_col = pick_col(trans_df, ["Type", "type"])
    close_type_col = pick_col(trans_df, ["Close_type", "close_type"])
    if not type_col:
        raise ValueError(f"trans file is missing Type column: {trans_path}")

    trans_df = trans_df.copy()
    trans_df["_type"] = trans_df[type_col].astype(str).str.lower()
    closed_trades = trans_df[trans_df["_type"] != "long"].copy()
    trade_num = int(len(closed_trades))

    withdrawal_close_count = 0
    speed_close_count = 0
    if close_type_col:
        close_types = pd.to_numeric(closed_trades[close_type_col], errors="coerce")
        withdrawal_close_count = int((close_types == 1).sum())
        speed_close_count = int((close_types == 2).sum())

    row = {column: math.nan for column in SUMMARY_COLUMNS}
    row.update(parse_param_tag(param_tag))
    row.update(
        {
            "withdrawal_close_count": withdrawal_close_count,
            "speed_close_count": speed_close_count,
            "final_capital": final_capital,
            "total_return_pct": total_return_pct,
            "outcome_high": float(outcome_high) if not pd.isna(outcome_high) else math.nan,
            "biggest_wd_abs": float(biggest_wd_abs) if not pd.isna(biggest_wd_abs) else math.nan,
            "biggest_wd_pct": biggest_wd_pct,
            "trade_num": trade_num,
            "capital": final_capital,
            "biggest_wd": float(biggest_wd_abs) if not pd.isna(biggest_wd_abs) else math.nan,
        }
    )
    return row


def choose_latest(store: dict[tuple[str, str], DetailFile], detail: DetailFile) -> None:
    key = (detail.run_name, detail.param_tag)
    current = store.get(key)
    if current is None or detail.mtime_ns > current.mtime_ns:
        store[key] = detail


def load_summary(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_excel(path, index_col=0)
    else:
        df = pd.DataFrame(columns=SUMMARY_COLUMNS)
    df.index = df.index.astype(str)
    df = df[~df.index.duplicated(keep="last")]
    for column in SUMMARY_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    ordered_columns = SUMMARY_COLUMNS + [column for column in df.columns if column not in SUMMARY_COLUMNS]
    df = df[ordered_columns]
    df.index.name = "param_tag"
    return df


def sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    sortable = df.copy()
    sortable["__sort_param_tag"] = sortable.index.astype(str)
    sort_columns = [
        "open_bar",
        "open_threshold",
        "open_continous_threshold",
        "open_withdrawal_threshold",
        "close_bar",
        "close_threshold",
        "close_withdrawal_threshold",
        "__sort_param_tag",
    ]
    sortable = sortable.sort_values(
        by=sort_columns,
        ascending=True,
        na_position="last",
        kind="mergesort",
    )
    sortable = sortable.drop(columns=["__sort_param_tag"])
    sortable.index.name = "param_tag"
    return sortable


def sync_result_dir(
    result_dir: Path,
    dry_run: bool = False,
    run_name_filter: str | None = None,
) -> int:
    perf_dir = result_dir / "perf"
    trans_dir = result_dir / "trans"
    outcome_stats_dir = result_dir / "outcome stats"

    if not perf_dir.exists() or not trans_dir.exists():
        raise FileNotFoundError(f"perf/trans directory is missing under: {result_dir}")

    perf_files: dict[tuple[str, str], DetailFile] = {}
    trans_files: dict[tuple[str, str], DetailFile] = {}

    for path in perf_dir.glob("*.xlsx"):
        detail = parse_detail_filename(path, "perf")
        if detail:
            choose_latest(perf_files, detail)
    for path in trans_dir.glob("*.xlsx"):
        detail = parse_detail_filename(path, "trans")
        if detail:
            choose_latest(trans_files, detail)

    common_keys = sorted(set(perf_files) & set(trans_files))
    missing_perf = sorted(set(trans_files) - set(perf_files))
    missing_trans = sorted(set(perf_files) - set(trans_files))

    if missing_perf:
        print(f"[warn] missing perf pair count: {len(missing_perf)}")
    if missing_trans:
        print(f"[warn] missing trans pair count: {len(missing_trans)}")

    outcome_stats_dir.mkdir(parents=True, exist_ok=True)

    by_run: dict[str, list[tuple[str, DetailFile, DetailFile]]] = {}
    for run_name, param_tag in common_keys:
        if run_name_filter and run_name != run_name_filter:
            continue
        by_run.setdefault(run_name, []).append((param_tag, perf_files[(run_name, param_tag)], trans_files[(run_name, param_tag)]))

    total_added = 0
    for run_name, items in sorted(by_run.items()):
        summary_path = resolve_summary_path(outcome_stats_dir, run_name, result_dir)
        summary_df = load_summary(summary_path)
        existing_tags = set(summary_df.index.astype(str))
        new_rows: list[tuple[str, dict[str, float]]] = []
        pending_items = [
            (param_tag, perf_detail, trans_detail)
            for param_tag, perf_detail, trans_detail in items
            if param_tag not in existing_tags
        ]

        if pending_items:
            print(f"[run] {run_name}: pending {len(pending_items)} row(s)", flush=True)
        else:
            print(f"[skip] {run_name}: no missing rows", flush=True)

        started_at = time.perf_counter()
        pending_count = len(pending_items)
        for index, (param_tag, perf_detail, trans_detail) in enumerate(pending_items, start=1):
            row = build_summary_row(perf_detail.path, trans_detail.path, param_tag)
            new_rows.append((param_tag, row))
            if (
                index == 1
                or index == pending_count
                or index % PROGRESS_EVERY == 0
            ):
                elapsed = time.perf_counter() - started_at
                print(
                    f"[progress] {run_name}: {index}/{pending_count} "
                    f"({index / pending_count:.1%}) elapsed {elapsed:.1f}s "
                    f"current={param_tag}",
                    flush=True,
                )

        if new_rows:
            new_df = pd.DataFrame(
                [row for _, row in new_rows],
                index=[param_tag for param_tag, _ in new_rows],
            )
            new_df.index.name = "param_tag"
            if summary_df.empty:
                merged_df = new_df.copy()
            else:
                merged_df = pd.concat([summary_df, new_df], axis=0)
            merged_df = merged_df[~merged_df.index.duplicated(keep="last")]

            print(f"[update] {run_name}: add {len(new_rows)} row(s)", flush=True)
            total_added += len(new_rows)
        else:
            merged_df = summary_df.copy()

        sorted_df = sort_summary(merged_df)
        sort_changed = list(sorted_df.index) != list(summary_df.index)
        if not dry_run and (new_rows or sort_changed):
            sorted_df.to_excel(summary_path)
            if sort_changed:
                print(f"[sort] {run_name}: reordered by open_bar/open_threshold/open_continous_threshold")

    if dry_run:
        print(f"[done] dry run, missing rows found: {total_added}")
    else:
        print(f"[done] rows written: {total_added}")
    return total_added


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill missing outcome_stats rows from existing perf/trans files."
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Path to one strategy outcome directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan only, do not write outcome_stats.xlsx.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Only sync one batch, for example: period_5min 20250601-20250615",
    )
    args, _ = parser.parse_known_args()

    result_dir = resolve_result_dir(args.result_dir)
    print(f"[path] using result dir: {result_dir}")

    sync_result_dir(
        result_dir=result_dir,
        dry_run=bool(args.dry_run or DRY_RUN),
        run_name_filter=args.run_name or RUN_NAME_FILTER,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
