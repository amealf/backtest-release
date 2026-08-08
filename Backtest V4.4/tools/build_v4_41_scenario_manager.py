from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = ROOT / "runtime_inputs" / "scenarios"
MARKET_CATALOG_PATH = SCENARIO_ROOT / "market_catalog.json"
SCENARIO_CATALOG_PATH = SCENARIO_ROOT / "scenario_catalog.json"
TEMPLATE_PATH = ROOT / "runtime_inputs" / "templates" / "market-intuition-selector.html"
PLOTLY_PATH = ROOT / "runtime_inputs" / "templates" / "plotly.min.js"
OUTPUT_ROOT = ROOT / "results" / "market_scenario_manager"
TIME_ALIASES = ("datetime", "timestamp", "time", "date", "日期", "时间")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def resolve_data_file(item: dict) -> Path:
    declared = item["data_file"]
    path = Path(declared["path"])
    if declared["path_base"] == "repository_root":
        path = ROOT / path
    return path.resolve()


def parse_local_time(value: str, timezone_name: str) -> datetime:
    text = value.strip().replace(" ", "T")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(ZoneInfo(timezone_name)).replace(tzinfo=None)
    return parsed


def select_market_csv(item: dict) -> bytes:
    source = resolve_data_file(item)
    declared = item["data_file"]
    if source.stat().st_size != int(declared["size_bytes"]):
        raise ValueError(f"market-data size differs from catalog: {item['market_id']}")
    if sha256_file(source) != declared["sha256"]:
        raise ValueError(f"market-data hash differs from catalog: {item['market_id']}")

    start = datetime.fromisoformat(item["start_time"].replace(" ", "T"))
    end = datetime.fromisoformat(item["end_time"].replace(" ", "T"))
    output = io.StringIO(newline="")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader)
        normalized = [field.strip().lower() for field in header]
        time_index = next(
            (index for index, field in enumerate(normalized) if field in TIME_ALIASES),
            None,
        )
        if time_index is None:
            raise ValueError(f"market data has no recognized time field: {source}")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(header)
        selected = 0
        for row in reader:
            if len(row) <= time_index:
                continue
            timestamp = parse_local_time(row[time_index], item["timezone"])
            if timestamp < start:
                continue
            if timestamp > end:
                break
            writer.writerow(row)
            selected += 1
    if selected == 0:
        raise ValueError(f"market interval selected no rows: {item['market_id']}")
    return output.getvalue().encode("utf-8")


def browser_asset(item: dict, raw: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(raw).hexdigest()
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    payload = {
        "encoding": "gzip+base64",
        "sha256": digest,
        "row_interval": [item["start_time"], item["end_time"]],
        "gzip_base64": base64.b64encode(compressed).decode("ascii"),
    }
    script = (
        "window.V4_41_MARKET_DATA=window.V4_41_MARKET_DATA||{};"
        f"window.V4_41_MARKET_DATA[{json.dumps(item['market_id'])}]="
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
    )
    return digest, script


def build() -> dict:
    market_catalog = json.loads(MARKET_CATALOG_PATH.read_text(encoding="utf-8"))
    scenario_catalog = json.loads(SCENARIO_CATALOG_PATH.read_text(encoding="utf-8"))
    derived_catalog = deepcopy(market_catalog)
    artifacts = []
    for market in derived_catalog["markets"]:
        raw = select_market_csv(market)
        digest, script = browser_asset(market, raw)
        asset_name = f"{market['market_id']}.js"
        asset_path = OUTPUT_ROOT / "assets" / asset_name
        atomic_text(asset_path, script)
        market["data_asset"] = f"assets/{asset_name}"
        market["browser_data_sha256"] = digest
        market["browser_row_count"] = raw.count(b"\n") - 1
        artifacts.append(
            {
                "market_id": market["market_id"],
                "asset": str(asset_path.resolve()),
                "row_count": market["browser_row_count"],
                "size_bytes": asset_path.stat().st_size,
            }
        )

    atomic_text(
        OUTPUT_ROOT / "market_catalog.js",
        "window.V4_41_MARKET_CATALOG="
        + json.dumps(derived_catalog, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
    )
    atomic_text(
        OUTPUT_ROOT / "scenario_catalog.js",
        "window.V4_41_SCENARIO_CATALOG="
        + json.dumps(scenario_catalog, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
    )
    atomic_text(OUTPUT_ROOT / "index.html", TEMPLATE_PATH.read_text(encoding="utf-8"))
    atomic_bytes(OUTPUT_ROOT / "assets" / "plotly.min.js", PLOTLY_PATH.read_bytes())
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "market_catalog": str(MARKET_CATALOG_PATH.resolve()),
        "scenario_catalog": str(SCENARIO_CATALOG_PATH.resolve()),
        "entry": str((OUTPUT_ROOT / "index.html").resolve()),
        "artifacts": artifacts,
    }
    atomic_text(
        OUTPUT_ROOT / "manager_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
