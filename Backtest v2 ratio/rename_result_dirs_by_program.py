from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


RESULT_ROOT = Path(r"D:\Code\backtest-release\Backtest v2 ratio\result")
DRY_RUN = False

OLD_SUFFIX = " long outcome"
PROGRAM_FOLDER_NAMES = {
    "long_momentum": "long_momentum outcome",
    "long_momentum_ATR": "long_momentum_ATR outcome",
    "long_momentum_ratio": "long_momentum_ratio outcome",
}
RATIO_PATTERN = re.compile(r"(^|\s)(opm|ocpm|cpm|cwm|adt|bs|bw)\S*", re.IGNORECASE)
ATR_PATTERN = re.compile(r"(^|\s)(oa|oca|owa|ca)\S*", re.IGNORECASE)


def pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower_map = {str(col).strip().lower(): str(col) for col in df.columns}
    for name in names:
        hit = lower_map.get(name.lower())
        if hit:
            return hit
    return None


def collect_summary_texts(result_dir: Path) -> list[str]:
    texts: list[str] = []
    outcome_stats_dir = result_dir / "outcome stats"
    if not outcome_stats_dir.exists():
        return texts
    for path in sorted(outcome_stats_dir.glob("*outcome_stats.xlsx")):
        texts.append(path.name)
        try:
            df = pd.read_excel(path, nrows=8)
        except Exception as exc:
            print(f"[warn] cannot read summary: {path.name} ({exc})")
            continue
        param_col = pick_col(df, ["param_tag", "Unnamed: 0"]) or (str(df.columns[0]) if len(df.columns) else None)
        if not param_col:
            continue
        for value in df[param_col].tolist():
            text = str(value or "").strip()
            if text:
                texts.append(text)
    return texts


def collect_detail_name_texts(result_dir: Path) -> list[str]:
    texts: list[str] = []
    for subdir in ("perf", "trans", "html"):
        folder = result_dir / subdir
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file():
                texts.append(path.stem)
            if len(texts) >= 24:
                return texts
    return texts


def detect_program_id(result_dir: Path) -> str | None:
    candidate_texts = collect_summary_texts(result_dir)
    if not candidate_texts:
        candidate_texts = collect_detail_name_texts(result_dir)
    if not candidate_texts:
        return None
    merged_text = "\n".join(candidate_texts)
    if RATIO_PATTERN.search(merged_text):
        return "long_momentum_ratio"
    if ATR_PATTERN.search(merged_text):
        return "long_momentum_ATR"
    return "long_momentum"


def target_dir_for(result_dir: Path, program_id: str) -> Path:
    prefix = result_dir.name[: -len(OLD_SUFFIX)]
    return result_dir.with_name(f"{prefix} {PROGRAM_FOLDER_NAMES[program_id]}")


def is_strategy_result_dir(path: Path) -> bool:
    if not path.is_dir() or not path.name.endswith(OLD_SUFFIX):
        return False
    expected_subdirs = ("perf", "trans", "html", "outcome stats", "stats excel")
    return any((path / name).exists() for name in expected_subdirs)


def rename_result_dirs(result_root: Path, dry_run: bool = False) -> None:
    old_dirs = sorted(
        path for path in result_root.iterdir()
        if is_strategy_result_dir(path)
    )
    if not old_dirs:
        print(f"[done] no old-format result dirs under: {result_root}")
        return

    unresolved: list[Path] = []
    renamed_count = 0
    skipped_count = 0

    for result_dir in old_dirs:
        program_id = detect_program_id(result_dir)
        if not program_id:
            unresolved.append(result_dir)
            print(f"[skip] unresolved: {result_dir.name}")
            continue

        target_dir = target_dir_for(result_dir, program_id)
        if target_dir == result_dir:
            skipped_count += 1
            print(f"[skip] already named: {result_dir.name}")
            continue
        if target_dir.exists():
            skipped_count += 1
            print(f"[skip] target exists: {target_dir.name}")
            continue

        print(f"[rename] {result_dir.name} -> {target_dir.name}")
        if not dry_run:
            result_dir.rename(target_dir)
        renamed_count += 1

    print(f"[done] renamed={renamed_count} skipped={skipped_count} unresolved={len(unresolved)}")
    if unresolved:
        print("[unresolved] please inspect these directories manually:")
        for path in unresolved:
            print(f"  - {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", help="Path to the ./result root directory.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only.")
    args = parser.parse_args()

    result_root = Path(args.result_root or RESULT_ROOT).expanduser().resolve()
    if not result_root.exists():
        raise FileNotFoundError(f"result root does not exist: {result_root}")

    rename_result_dirs(result_root=result_root, dry_run=bool(args.dry_run or DRY_RUN))


if __name__ == "__main__":
    main()
