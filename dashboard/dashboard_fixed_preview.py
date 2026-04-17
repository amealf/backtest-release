#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import mimetypes
import webbrowser
import warnings
from email.parser import BytesParser
from email.policy import default
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import cgi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8765
AUTO_OPEN_BROWSER = True
# 鎵嬪姩鎸囧畾缁撴灉澶ф枃浠跺す銆?# 渚? PRESET_RESULT_DIR = r"D:\Code\backtest-release\Backtest v2 ratio\result\xagusd_30s_all long_momentum outcome"
# 鐣欑┖鏃讹紝椤甸潰缁х画浣跨敤鈥滈€夋嫨缁撴灉鏂囦欢澶光€濇寜閽€?PRESET_RESULT_DIR = r"D:\Code\backtest-release\Backtest v2 ratio\result\xagusd_30s_all long shock multi outcome"

PROGRAM_ID_CLASSIC = "classic"
PROGRAM_ID_CLASSIC_ATR = "classic_atr"
PROGRAM_ID_RATIO = "ratio"
PROGRAM_ID_GARCH = "garch"
PROGRAM_TAG_CLASSIC = "long_momentum"
PROGRAM_TAG_GARCH = "long_momentum_GARCH"
PROGRAM_TAG_ARCH_SHOCK_MULTI = "long_momentum_ARCH_shock_multi"
PROGRAM_TAG_ARCH_SHOCK = "long_momentum_ARCH_shock"
PROGRAM_TAG_ARCH = "long_momentum_ARCH"
PROGRAM_TAG_CLASSIC_ATR = "long_momentum_ATR"
PROGRAM_TAG_RATIO = "long_momentum_ratio"
SUMMARY_PROGRAM_TAGS = (
    PROGRAM_TAG_GARCH,
    PROGRAM_TAG_ARCH_SHOCK_MULTI,
    PROGRAM_TAG_ARCH_SHOCK,
    PROGRAM_TAG_RATIO,
    PROGRAM_TAG_CLASSIC_ATR,
    PROGRAM_TAG_ARCH,
    PROGRAM_TAG_CLASSIC,
)
RESULT_DIR_SUFFIXES = (
    " long shock multi outcome",
    " long outcome",
    " long shock outcome",
    " long_momentum_GARCH outcome",
    " long_momentum outcome",
    " long_momentum_ARCH outcome",
    " long_momentum_ATR outcome",
    " long_momentum_ratio outcome",
)


def _num(value):
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if abs(num - round(num)) < 1e-10:
        return int(round(num))
    return num


def _num_key(value) -> str:
    num = _num(value)
    if num is None:
        return ""
    if isinstance(num, int):
        return str(num)
    return f"{num:.10f}".rstrip("0").rstrip(".")


def _pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for name in names:
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def _strip_summary_program_tag(text: str) -> str:
    stripped = str(text or "").strip()
    lowered = stripped.lower()
    for tag in SUMMARY_PROGRAM_TAGS:
        prefix = tag.lower() + " "
        if lowered.startswith(prefix):
            return stripped[len(tag):].strip()
    return stripped


def _batch_label(file_name: str) -> str:
    stem = _strip_summary_program_tag(Path(file_name).stem.strip())
    lower = stem.lower()
    for suffix in (" outcome_stats",):
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)].strip()
            break
    parts = stem.split()
    if parts and parts[-1].isdigit():
        stem = " ".join(parts[:-1]).strip()
    return stem


def _detect_program_id(text: str | None) -> str | None:
    lowered = str(text or "").strip().lower()
    if PROGRAM_TAG_GARCH.lower() in lowered:
        return PROGRAM_ID_GARCH
    if "long_momentum_garch outcome" in lowered:
        return PROGRAM_ID_GARCH
    if PROGRAM_TAG_ARCH_SHOCK_MULTI.lower() in lowered:
        return PROGRAM_ID_CLASSIC
    if "long_momentum_arch shock multi" in lowered:
        return PROGRAM_ID_CLASSIC
    if PROGRAM_TAG_ARCH_SHOCK.lower() in lowered:
        return PROGRAM_ID_CLASSIC
    if "long_momentum_arch shock" in lowered:
        return PROGRAM_ID_CLASSIC
    if PROGRAM_TAG_RATIO.lower() in lowered:
        return PROGRAM_ID_RATIO
    if PROGRAM_TAG_CLASSIC_ATR.lower() in lowered:
        return PROGRAM_ID_CLASSIC_ATR
    if PROGRAM_TAG_ARCH.lower() in lowered:
        return PROGRAM_ID_CLASSIC
    if PROGRAM_TAG_CLASSIC.lower() in lowered:
        return PROGRAM_ID_CLASSIC
    if "long_momentum_ratio outcome" in lowered:
        return PROGRAM_ID_RATIO
    if "long_momentum_atr outcome" in lowered:
        return PROGRAM_ID_CLASSIC_ATR
    if "long_momentum outcome" in lowered:
        return PROGRAM_ID_CLASSIC
    return None


def _effective_program_id(program_id: str | None) -> str:
    if program_id == PROGRAM_ID_GARCH:
        return PROGRAM_ID_GARCH
    if program_id == PROGRAM_ID_RATIO:
        return PROGRAM_ID_RATIO
    if program_id == PROGRAM_ID_CLASSIC_ATR:
        return PROGRAM_ID_CLASSIC_ATR
    return PROGRAM_ID_CLASSIC


def _summary_fourth_field(program_id: str) -> str:
    if _effective_program_id(program_id) == PROGRAM_ID_RATIO:
        return "close_withdrawal_threshold"
    return "open_withdrawal_threshold"


def _step(values: list[float | int | None]):
    nums = []
    seen = set()
    for value in sorted(v for v in values if v is not None):
        key = _num_key(value)
        if key in seen:
            continue
        seen.add(key)
        nums.append(value)
    if len(nums) < 2:
        return nums, None
    diffs = []
    for left, right in zip(nums, nums[1:]):
        diff = float(right) - float(left)
        if diff > 0:
            diffs.append(diff)
    if not diffs:
        return nums, None
    best = min(diffs)
    if abs(best - round(best)) < 1e-10:
        return nums, int(round(best))
    return nums, float(f"{best:.10f}".rstrip("0").rstrip("."))


def _parse_param_tag(param_tag: str) -> dict:
    result = {
        "open_bar": None,
        "open_threshold": None,
        "open_continous_threshold": None,
        "open_withdrawal_threshold": None,
        "close_withdrawal_threshold": None,
    }
    specs = [
        ("open_bar", ("cb", "om")),
        ("open_continous_threshold", ("csm", "ocpm", "oc")),
        ("open_threshold", ("sha", "opm", "o")),
        ("open_withdrawal_threshold", ("owm", "ow")),
        ("close_withdrawal_threshold", ("cwd", "cwm", "cw")),
    ]
    for token in str(param_tag).strip().split():
        lowered = token.lower()
        for key, prefixes in specs:
            matched = False
            for prefix in prefixes:
                if lowered.startswith(prefix):
                    value = _num(token[len(prefix):])
                    if value is not None:
                        result[key] = value
                    matched = True
                    break
            if matched:
                break
    return result


def _parse_summary(content: bytes, file_name: str, program_id: str | None = None) -> dict:
    resolved_program_id = _effective_program_id(program_id or _detect_program_id(file_name))
    fourth_field = _summary_fourth_field(resolved_program_id)
    df = pd.read_excel(BytesIO(content))
    if df.empty:
        return {
            "batch_label": _batch_label(file_name),
            "records": [],
            "controls": {},
            "default_key": "",
            "program_id": resolved_program_id,
            "fourth_field": fourth_field,
        }

    param_col = _pick_col(df, ["param_tag", "Unnamed: 0"]) or df.columns[0]
    capital_col = _pick_col(df, ["capital", "final_capital"])
    trade_col = _pick_col(df, ["trade_num"])
    wd_col = _pick_col(df, ["biggest_wd", "biggest_wd_abs"])
    high_col = _pick_col(df, ["outcome_high"])
    ob_col = _pick_col(df, ["open_bar"])
    ot_col = _pick_col(df, ["open_threshold"])
    oc_col = _pick_col(df, ["open_continous_threshold"])
    ow_col = _pick_col(df, ["open_withdrawal_threshold"])
    cw_col = _pick_col(df, ["close_withdrawal_threshold"])

    records = []
    for idx, (_, row) in enumerate(df.iterrows()):
        param_tag = str(row[param_col] or "").strip()
        if not param_tag:
            continue
        parsed = _parse_param_tag(param_tag)
        open_bar = _num(row[ob_col]) if ob_col else parsed["open_bar"]
        open_threshold = _num(row[ot_col]) if ot_col else parsed["open_threshold"]
        open_cont = _num(row[oc_col]) if oc_col else parsed["open_continous_threshold"]
        open_withdrawal = _num(row[ow_col]) if ow_col else parsed["open_withdrawal_threshold"]
        close_withdrawal = _num(row[cw_col]) if cw_col else parsed["close_withdrawal_threshold"]
        withdrawal_limit = close_withdrawal if fourth_field == "close_withdrawal_threshold" else open_withdrawal
        records.append({
            "order_index": idx,
            "param_tag": param_tag,
            "selection_key": "|".join([
                _num_key(open_bar),
                _num_key(open_threshold),
                _num_key(open_cont),
                _num_key(withdrawal_limit),
            ]),
            "open_bar": open_bar,
            "open_threshold": open_threshold,
            "open_continous_threshold": open_cont,
            "open_withdrawal_threshold": open_withdrawal,
            "close_withdrawal_threshold": close_withdrawal,
            "withdrawal_limit": withdrawal_limit,
            "capital": float(row[capital_col]) if capital_col and not pd.isna(row[capital_col]) else None,
            "trade_num": int(row[trade_col]) if trade_col and not pd.isna(row[trade_col]) else None,
            "biggest_wd": float(row[wd_col]) if wd_col and not pd.isna(row[wd_col]) else None,
            "outcome_high": float(row[high_col]) if high_col and not pd.isna(row[high_col]) else None,
        })

    default_key = ""
    if records:
        default_key = max(
            records,
            key=lambda item: item["capital"] if item["capital"] is not None else float("-inf"),
        )["selection_key"]

    ob_values, ob_step = _step([row["open_bar"] for row in records])
    ot_values, ot_step = _step([row["open_threshold"] for row in records])
    oc_values, oc_step = _step([row["open_continous_threshold"] for row in records])
    ow_values, ow_step = _step([row["open_withdrawal_threshold"] for row in records])
    cw_values, cw_step = _step([row["close_withdrawal_threshold"] for row in records])
    wd_values, wd_step = _step([row["withdrawal_limit"] for row in records])
    return {
        "batch_label": _batch_label(file_name),
        "records": records,
        "controls": {
            "open_bar": {"values": ob_values, "step": ob_step},
            "open_threshold": {"values": ot_values, "step": ot_step},
            "open_continous_threshold": {"values": oc_values, "step": oc_step},
            "open_withdrawal_threshold": {"values": ow_values, "step": ow_step},
            "close_withdrawal_threshold": {"values": cw_values, "step": cw_step},
            "withdrawal_limit": {"values": wd_values, "step": wd_step},
        },
        "default_key": default_key,
        "program_id": resolved_program_id,
        "fourth_field": fourth_field,
    }


def _parse_price(content: bytes) -> dict:
    df = pd.read_excel(BytesIO(content))
    date_col = _pick_col(df, ["Date", "date"])
    open_col = _pick_col(df, ["open"])
    high_col = _pick_col(df, ["high"])
    low_col = _pick_col(df, ["low"])
    close_col = _pick_col(df, ["close"])
    capital_col = _pick_col(df, ["Capital", "capital"])
    if not all([date_col, open_col, high_col, low_col, close_col]):
        raise ValueError("perf.xlsx 缂哄皯 OHLC 鍒椼€?)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    valid = dates.notna()
    clean = df.loc[valid].copy()
    clean_dates = dates.loc[valid]
    open_series = pd.to_numeric(clean[open_col], errors="coerce")
    high_series = pd.to_numeric(clean[high_col], errors="coerce")
    low_series = pd.to_numeric(clean[low_col], errors="coerce")
    close_series = pd.to_numeric(clean[close_col], errors="coerce")
    base_price = None
    if open_series.notna().any():
        base_price = float(open_series.loc[open_series.notna()].iloc[0])
    if base_price not in (None, 0):
        scale = 100.0 / float(base_price)
        open_series = open_series * scale
        high_series = high_series * scale
        low_series = low_series * scale
        close_series = close_series * scale
    capital_x = []
    capital_y = []
    if capital_col:
        capital_series = pd.to_numeric(clean[capital_col], errors="coerce")
        capital_valid = capital_series.notna()
        capital_x = clean_dates.loc[capital_valid].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        capital_y = capital_series.loc[capital_valid].astype(float).tolist()
    return {
        "x": clean_dates.dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "open": open_series.tolist(),
        "high": high_series.tolist(),
        "low": low_series.tolist(),
        "close": close_series.tolist(),
        "price_base": base_price,
        "capital_x": capital_x,
        "capital_y": capital_y,
    }


def _parse_trans(content: bytes) -> dict:
    df = pd.read_excel(BytesIO(content))
    date_col = _pick_col(df, ["Date", "date"])
    type_col = _pick_col(df, ["Type", "type"])
    price_col = _pick_col(df, ["Price", "price"])
    close_type_col = _pick_col(df, ["Close_type", "close_type"])
    capital_col = _pick_col(df, ["Capital", "capital"])
    if not all([date_col, type_col, price_col]):
        raise ValueError("trans.xlsx 缂哄皯浜ゆ槗鍒椼€?)

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).reset_index(drop=True)
    df["_type"] = df[type_col].astype(str).str.lower()
    df["_price"] = pd.to_numeric(df[price_col], errors="coerce")
    df["_close_type"] = pd.to_numeric(df[close_type_col], errors="coerce") if close_type_col else pd.NA
    df["_capital"] = pd.to_numeric(df[capital_col], errors="coerce") if capital_col else pd.NA

    long_df = df[df["_type"] == "long"]
    sell_df = df[df["_type"] == "sell"]
    wd_df = sell_df[sell_df["_close_type"] == 1]
    speed_df = sell_df[sell_df["_close_type"] == 2]

    capital_x = []
    capital_y = []
    if len(df):
        capital_x.append(df.iloc[0][date_col].strftime("%Y-%m-%d %H:%M:%S"))
        capital_y.append(100.0)
    for _, row in sell_df.dropna(subset=["_capital"]).iterrows():
        capital_x.append(row[date_col].strftime("%Y-%m-%d %H:%M:%S"))
        capital_y.append(float(row["_capital"]))

    link_x = []
    link_y = []
    entry = None
    for _, row in df.iterrows():
        if row["_type"] == "long":
            entry = row
        elif row["_type"] == "sell" and entry is not None:
            link_x.extend([
                entry[date_col].strftime("%Y-%m-%d %H:%M:%S"),
                row[date_col].strftime("%Y-%m-%d %H:%M:%S"),
                None,
            ])
            link_y.extend([float(entry["_price"]), float(row["_price"]), None])
            entry = None

    def pack(source_df: pd.DataFrame) -> dict:
        return {
            "x": source_df[date_col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "y": pd.to_numeric(source_df["_price"], errors="coerce").tolist(),
        }

    return {
        "buy_points": pack(long_df),
        "sell_wd_points": pack(wd_df),
        "sell_speed_points": pack(speed_df),
        "capital_x": capital_x,
        "capital_y": capital_y,
        "trade_link_x": link_x,
        "trade_link_y": link_y,
    }


def _preset_root() -> Path | None:
    if PRESET_RESULT_DIR in (None, ""):
        return None
    root = Path(PRESET_RESULT_DIR).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    root = root.resolve()
    if root.exists() and root.is_dir():
        return root
    for suffix in RESULT_DIR_SUFFIXES:
        if not root.name.endswith(suffix):
            continue
        prefix = root.name[: -len(suffix)]
        for alt_suffix in RESULT_DIR_SUFFIXES:
            candidate = root.with_name(prefix + alt_suffix)
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()
        break
    return root


def _is_ignored_result_file_name(name: str) -> bool:
    lowered = str(name or "").strip().lower()
    return bool(lowered) and lowered.startswith("~")


def _file_entry(path: Path, root: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "relative_path": path.relative_to(root).as_posix(),
        "last_modified": int(stat.st_mtime * 1000),
    }


def _preset_files() -> dict:
    root = _preset_root()
    if root is None:
        return {"enabled": False}
    if not root.exists() or not root.is_dir():
        return {
            "enabled": False,
            "error": f"preset folder not found: {root}",
        }

    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_ignored_result_file_name(path.name):
            continue
        files.append(_file_entry(path, root))
    files.sort(key=lambda item: item["last_modified"], reverse=True)
    return {
        "enabled": True,
        "folder_label": root.name,
        "folder_path": str(root),
        "program_id": _detect_program_id(root.name),
        "files": files,
    }


def _resolve_preset_file(relative_path: str) -> Path:
    root = _preset_root()
    if root is None:
        raise ValueError("preset folder is not configured")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("preset file is outside the configured folder") from exc
    if not candidate.exists() or not candidate.is_file():
        raise ValueError(f"preset file not found: {relative_path}")
    return candidate


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _file_cgi_legacy(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "file" not in form:
            raise ValueError("娌℃湁鏀跺埌鏂囦欢銆?)
        item = form["file"]
        if not getattr(item, "file", None):
            raise ValueError("涓婁紶鏂囦欢涓虹┖銆?)
        content = item.file.read()
        if not content:
            raise ValueError("涓婁紶鏂囦欢涓虹┖銆?)
        return item.filename or "upload.bin", content

    def _file(self):
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            raise ValueError("涓婁紶鏂囦欢涓虹┖銆?)

        body = self.rfile.read(content_length)
        if not body:
            raise ValueError("涓婁紶鏂囦欢涓虹┖銆?)

        message = BytesParser(policy=default).parsebytes(
            (
                f"Content-Type: {content_type}\r\n"
                "MIME-Version: 1.0\r\n\r\n"
            ).encode("utf-8")
            + body
        )
        if not message.is_multipart():
            raise ValueError("请求内容不是 multipart/form-data銆?)

        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            if part.get_param("name", header="content-disposition") != "file":
                continue
            content = part.get_payload(decode=True) or b""
            if not content:
                raise ValueError("涓婁紶鏂囦欢涓虹┖銆?)
            return part.get_filename() or "upload.bin", content

        raise ValueError("娌℃湁鏀跺埌鏂囦欢銆?)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/preset-index":
            self._send_json(200, _preset_files())
            return
        if parsed.path == "/api/preset-summary":
            relative_path = parse_qs(parsed.query).get("path", [""])[0]
            program_id = parse_qs(parsed.query).get("program_id", [""])[0] or _detect_program_id(_preset_root().name if _preset_root() else "")
            path = _resolve_preset_file(relative_path)
            self._send_json(200, _parse_summary(path.read_bytes(), path.name, program_id))
            return
        if parsed.path == "/api/preset-price":
            relative_path = parse_qs(parsed.query).get("path", [""])[0]
            path = _resolve_preset_file(relative_path)
            self._send_json(200, _parse_price(path.read_bytes()))
            return
        if parsed.path == "/api/preset-trans":
            relative_path = parse_qs(parsed.query).get("path", [""])[0]
            path = _resolve_preset_file(relative_path)
            self._send_json(200, _parse_trans(path.read_bytes()))
            return
        if parsed.path == "/api/preset-html":
            relative_path = parse_qs(parsed.query).get("path", [""])[0]
            path = _resolve_preset_file(relative_path)
            content_type = mimetypes.guess_type(path.name)[0] or "text/html"
            self._send(200, path.read_bytes(), f"{content_type}; charset=utf-8")
            return
        super().do_GET()

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/summary":
                program_id = parse_qs(parsed.query).get("program_id", [""])[0] or None
                file_name, content = self._file()
                self._send_json(200, _parse_summary(content, file_name, program_id))
                return
            if parsed.path == "/api/price":
                _, content = self._file()
                self._send_json(200, _parse_price(content))
                return
            if parsed.path == "/api/trans":
                _, content = self._file()
                self._send_json(200, _parse_trans(content))
                return
            self._send_json(404, {"error": "鏈煡鎺ュ彛銆?})
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})


PAGE_LEGACY = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest Dashboard</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{margin:0;padding:16px;background:#f5f7fb;font-family:"Segoe UI","Microsoft YaHei UI",sans-serif;color:#152033}
.wrap{max-width:1480px;margin:0 auto;display:grid;gap:14px}.panel{background:#fff;border:1px solid #d9e1ee;border-radius:16px;padding:16px;box-shadow:0 16px 40px rgba(28,49,88,.08)}
.top{display:grid;grid-template-columns:auto 1fr 320px auto;gap:10px;align-items:center}.title{font-size:20px;font-weight:700}.sub{font-size:12px;color:#66758c;margin-top:4px}
.box,.btn,.sel,.num{height:40px;border:1px solid #d9e1ee;border-radius:12px;padding:0 12px;font-size:13px}.box{display:flex;align-items:center;color:#66758c;background:#fbfcff;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.btn{background:#fff;cursor:pointer;font-weight:600}.btn.primary{background:#2f6bff;color:#fff;border-color:#2f6bff}.btn:disabled{cursor:not-allowed;color:#99a7bf;background:#f5f7fb}
.tabs{display:flex;gap:8px}.tab{height:36px;padding:0 18px;border-radius:999px;border:1px solid #d9e1ee;background:#fff;color:#66758c;font-weight:600;cursor:pointer}.tab.active{background:rgba(47,107,255,.1);color:#2f6bff;border-color:rgba(47,107,255,.25)}
.page{display:none;gap:14px}.page.active{display:grid}.chartbox{padding:14px;border:1px solid #d9e1ee;border-radius:14px;background:#fbfcff}.charttitle{font-size:14px;font-weight:700;margin-bottom:10px}.chart{min-height:340px}.empty{min-height:120px;border:1px dashed #c7d2e5;border-radius:14px;display:grid;place-items:center;text-align:center;color:#66758c;padding:16px;line-height:1.7;background:rgba(255,255,255,.7)}
.detail{display:grid;grid-template-columns:1.1fr .9fr;gap:14px}.left,.right{display:grid;gap:14px;min-width:0}.controls{display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:10px;align-items:end}.label{font-size:12px;color:#66758c;font-weight:600;margin-bottom:6px}.meta{display:grid;gap:8px;font-size:13px}.row{display:grid;grid-template-columns:120px 1fr;gap:10px}.name{color:#66758c;font-weight:600}
.iframe{display:none;border:1px solid #d9e1ee;border-radius:14px;overflow:hidden;min-height:520px;background:#fff}.iframe.active{display:block}iframe{width:100%;min-height:520px;border:0;display:block}
@media(max-width:1200px){.top,.detail,.controls{grid-template-columns:1fr}}
</style></head><body><div class="wrap"><div class="panel"><div class="top"><div><div class="title">Backtest Dashboard</div><div class="sub">总览读取 outcome_stats锛岃鎯呭垏鎹㈣鍙?trans.xlsx锛屽師 HTML 浠嶅彲闅忔椂鎵撳紑銆?/div></div><div id="folderName" class="box">尚未选择结果大文件夹</div><select id="batchSelect" class="sel" disabled><option value="">璇烽€夋嫨鍥炴祴鎵规</option></select><button id="pickBtn" class="btn primary">閫夋嫨缁撴灉鏂囦欢澶?/button></div><div style="margin-top:14px" class="tabs"><button id="tabOverview" class="tab active">涓婚〉闈?/button><button id="tabDetail" class="tab">璇︽儏椤?/button></div>
<div id="pageOverview" class="page active" style="margin-top:14px"><div class="chartbox"><div class="charttitle">收益总览</div><div id="overviewEmpty" class="empty">璇烽€夋嫨涓€涓甫鏈?outcome_stats.xlsx 鐨勭粨鏋滃ぇ鏂囦欢澶广€?/div><div id="overviewChart" class="chart" style="display:none"></div></div></div>
<div id="pageDetail" class="page" style="margin-top:14px"><div class="detail"><div class="left"><div class="chartbox"><div class="controls"><div><div class="label">窗口大小</div><input id="inputBar" class="num" type="number" disabled></div><div><div class="label">閫熷害骞呭害闄愬埗</div><input id="inputThreshold" class="num" type="number" disabled></div><div><div class="label">持续门槛</div><input id="inputCont" class="num" type="number" disabled></div><button id="openHtml" class="btn" disabled>鎵撳紑鍘?HTML</button></div></div><div class="chartbox"><div class="charttitle">鍏叡琛屾儏鍥句笌涔板崠鐐?/div><div id="priceEmpty" class="empty">璇烽€夋嫨涓€涓壒娆★紝鍐嶄粠涓婚〉闈㈢偣閫変竴涓弬鏁扮偣銆?/div><div id="priceChart" class="chart" style="display:none"></div></div><div class="chartbox"><div class="charttitle">交易资金曲线</div><div id="capitalEmpty" class="empty">当前还没有可用的 trans.xlsx銆?/div><div id="capitalChart" class="chart" style="display:none;min-height:260px"></div></div><div id="htmlWrap" class="iframe"><iframe id="htmlFrame"></iframe></div></div>
<div class="right"><div class="chartbox"><div class="charttitle">当前参数</div><div class="meta"><div class="row"><div class="name">缁撴灉鏂囦欢澶?/div><div id="metaFolder">-</div></div><div class="row"><div class="name">批次</div><div id="metaBatch">-</div></div><div class="row"><div class="name">参数标签</div><div id="metaTag">-</div></div><div class="row"><div class="name">鏈€缁堣祫鏈?/div><div id="metaCapital">-</div></div><div class="row"><div class="name">交易次数</div><div id="metaTrade">-</div></div><div class="row"><div class="name">鏈€澶у洖鎾?/div><div id="metaWd">-</div></div><div class="row"><div class="name">宄板€艰祫鏈?/div><div id="metaHigh">-</div></div></div></div><div class="chartbox"><div class="charttitle">鐘舵€?/div><div id="status" class="empty" style="min-height:100px">等待目录</div></div></div></div></div></div></div><input id="folderInput" type="file" webkitdirectory directory multiple style="display:none">
<script>
const state={files:[],summaryFiles:[],transFiles:[],perfFiles:[],htmlFiles:[],summary:null,activeRecord:null,folderLabel:"",priceCache:new Map(),transCache:new Map(),htmlUrlCache:new Map()};
const $=id=>document.getElementById(id);
function setStatus(text){$("status").innerHTML=text}
function setTab(name){$("tabOverview").classList.toggle("active",name==="overview");$("tabDetail").classList.toggle("active",name==="detail");$("pageOverview").classList.toggle("active",name==="overview");$("pageDetail").classList.toggle("active",name==="detail")}
function pathOf(file){return String(file.relative_path||file.webkitRelativePath||"")}
function rel(file){return String(pathOf(file)||file.name).replaceAll("\\\\","/")}
function inDir(file,dir){const path=rel(file).toLowerCase();const prefix=dir.toLowerCase()+"/";return path.startsWith(prefix)||path.includes("/"+prefix)}
function esc(text){return String(text).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")}
function keyOf(v){if(v===null||v===undefined||v==="")return"";const n=Number(v);if(Number.isNaN(n))return"";return String(Number(n.toFixed(10))).replace(/\\.0+$/,"")}
function selKey(){return [keyOf($("inputBar").value),keyOf($("inputThreshold").value),keyOf($("inputCont").value)].join("|")}
function isSummary(file){return inDir(file,"outcome stats")&&file.name.toLowerCase().endsWith("outcome_stats.xlsx")}
function isTrans(file){return inDir(file,"trans")&&file.name.toLowerCase().endsWith("trans.xlsx")}
function isPerf(file){return inDir(file,"perf")&&file.name.toLowerCase().endsWith("perf.xlsx")}
function isHtml(file){return inDir(file,"html")&&file.name.toLowerCase().endsWith(".html")}
function fileMtime(file){return Number(file.last_modified??file.lastModified??0)}
async function upload(url,file){const fd=new FormData();fd.append("file",file,file.name);const res=await fetch(url,{method:"POST",body:fd});const data=await res.json();if(!res.ok)throw new Error(data.error||"请求失败");return data}
async function fetchJson(url){const res=await fetch(url);const data=await res.json();if(!res.ok)throw new Error(data.error||"request failed");return data}
function resetUi(){for(const url of state.htmlUrlCache.values()){URL.revokeObjectURL(url)}state.summary=null;state.activeRecord=null;state.folderLabel="";state.priceCache.clear();state.transCache.clear();state.htmlUrlCache.clear();$("batchSelect").innerHTML='<option value="">璇烽€夋嫨鍥炴祴鎵规</option>';$("batchSelect").disabled=true;$("overviewChart").style.display="none";$("priceChart").style.display="none";$("capitalChart").style.display="none";$("overviewEmpty").style.display="grid";$("priceEmpty").style.display="grid";$("capitalEmpty").style.display="grid";$("openHtml").disabled=true;$("inputBar").disabled=true;$("inputThreshold").disabled=true;$("inputCont").disabled=true;$("htmlWrap").classList.remove("active");$("metaFolder").textContent="-";$("metaBatch").textContent="-";$("metaTag").textContent="-";$("metaCapital").textContent="-";$("metaTrade").textContent="-";$("metaWd").textContent="-";$("metaHigh").textContent="-"}
function configInput(el,meta){const has=meta&&meta.values&&meta.values.length;el.disabled=!has;if(!has){el.value="";el.removeAttribute("min");el.removeAttribute("max");el.removeAttribute("step");return}el.min=meta.values[0];el.max=meta.values[meta.values.length-1];el.step=meta.step??"any"}
function batchFile(files,label,suffix){const arr=files.filter(file=>file.name.toLowerCase().endsWith(suffix)&&file.name.includes(label)).sort((a,b)=>fileMtime(b)-fileMtime(a));return arr[0]||null}
function detailFile(files,label,paramTag,suffix){const arr=files.filter(file=>file.name.toLowerCase().endsWith(suffix)&&file.name.includes(label)&&file.name.includes(paramTag)).sort((a,b)=>fileMtime(b)-fileMtime(a));return arr[0]||null}
function recordByKey(key){return state.summary?.records.find(item=>item.selection_key===key)||null}
function meta(record){$("metaFolder").textContent=state.folderLabel||"-";if(!record){$("metaBatch").textContent="-";$("metaTag").textContent="-";$("metaCapital").textContent="-";$("metaTrade").textContent="-";$("metaWd").textContent="-";$("metaHigh").textContent="-";return}$("metaBatch").textContent=state.summary.batch_label;$("metaTag").textContent=record.param_tag;$("metaCapital").textContent=record.capital??"-";$("metaTrade").textContent=record.trade_num??"-";$("metaWd").textContent=record.biggest_wd??"-";$("metaHigh").textContent=record.outcome_high??"-"}
function inputsFrom(record){$("inputBar").value=record.open_bar??"";$("inputThreshold").value=record.open_threshold??"";$("inputCont").value=record.open_continous_threshold??""}
function overview(){if(!state.summary||!state.summary.records.length){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";return}const rows=state.summary.records;$("overviewEmpty").style.display="none";$("overviewChart").style.display="block";Plotly.newPlot($("overviewChart"),[{x:rows.map((_,i)=>i+1),y:rows.map(r=>r.capital),mode:"lines+markers",line:{color:"#2f6bff",width:2},marker:{size:6,color:"#2f6bff"},customdata:rows.map(r=>r.selection_key),text:rows.map(r=>["param_tag: "+esc(r.param_tag),"open_bar: "+(r.open_bar??"-"),"open_threshold: "+(r.open_threshold??"-"),"open_cont: "+(r.open_continous_threshold??"-"),"capital: "+(r.capital??"-"),"trade_num: "+(r.trade_num??"-")].join("<br>")),hovertemplate:"%{text}<extra></extra>",name:"capital"}],{margin:{l:44,r:18,t:16,b:40},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:{title:"参数组合顺序",showgrid:false,zeroline:false},yaxis:{title:"capital",gridcolor:"#e8eef8",zeroline:false},showlegend:false},{responsive:true,displayModeBar:false});if($("overviewChart").removeAllListeners){$("overviewChart").removeAllListeners("plotly_click")}$("overviewChart").on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record)setRecord(record,true)})}
async function ensurePrice(){const label=state.summary?.batch_label;if(!label)return null;if(state.priceCache.has(label))return state.priceCache.get(label);const file=batchFile(state.perfFiles,label,"perf.xlsx");if(!file){state.priceCache.set(label,null);return null}setStatus("正在读取公共行情数据");const data=file.relative_path?await fetchJson("/api/preset-price?path="+encodeURIComponent(file.relative_path)):await upload("/api/price",file);state.priceCache.set(label,data);return data}
async function ensureTrans(record){if(!record||!state.summary)return null;if(state.transCache.has(record.selection_key))return state.transCache.get(record.selection_key);const file=detailFile(state.transFiles,state.summary.batch_label,record.param_tag,"trans.xlsx");if(!file){state.transCache.set(record.selection_key,null);return null}setStatus("正在读取 trans.xlsx");const data=file.relative_path?await fetchJson("/api/preset-trans?path="+encodeURIComponent(file.relative_path)):await upload("/api/trans",file);state.transCache.set(record.selection_key,data);return data}
function priceChart(price,trans){if(!price){$("priceChart").style.display="none";$("priceEmpty").style.display="grid";$("priceEmpty").innerHTML="褰撳墠鎵规缂哄皯鍙敤鐨?perf.xlsx銆?;return}$("priceEmpty").style.display="none";$("priceChart").style.display="block";const traces=[{type:"candlestick",x:price.x,open:price.open,high:price.high,low:price.low,close:price.close,name:"price",increasing:{line:{color:"#a7a7a7",width:1},fillcolor:"rgba(245,245,245,.95)"},decreasing:{line:{color:"#666",width:1},fillcolor:"rgba(125,125,125,.95)"}}];if(trans){if(trans.trade_link_x.length)traces.push({type:"scatter",mode:"lines",x:trans.trade_link_x,y:trans.trade_link_y,line:{color:"#2f6bff",width:2},hoverinfo:"skip",name:"trade_link"});if(trans.buy_points.x.length)traces.push({type:"scatter",mode:"markers",x:trans.buy_points.x,y:trans.buy_points.y,marker:{color:"#d14f5c",size:7},name:"buy",hovertemplate:"buy<br>%{x}<br>price=%{y}<extra></extra>"});if(trans.sell_wd_points.x.length)traces.push({type:"scatter",mode:"markers",x:trans.sell_wd_points.x,y:trans.sell_wd_points.y,marker:{color:"#1d8b5c",size:7},name:"sell_wd",hovertemplate:"sell_wd<br>%{x}<br>price=%{y}<extra></extra>"});if(trans.sell_speed_points.x.length)traces.push({type:"scatter",mode:"markers",x:trans.sell_speed_points.x,y:trans.sell_speed_points.y,marker:{color:"#c98212",size:7},name:"sell_speed",hovertemplate:"sell_speed<br>%{x}<br>price=%{y}<extra></extra>"})}Plotly.newPlot($("priceChart"),traces,{margin:{l:44,r:18,t:16,b:30},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:{showgrid:false,rangeslider:{visible:false}},yaxis:{title:"price",gridcolor:"#e8eef8",zeroline:false},legend:{orientation:"h",yanchor:"bottom",y:1.02,xanchor:"left",x:0}},{responsive:true,displayModeBar:false})}
function capitalChart(trans){if(!trans||!trans.capital_x.length){$("capitalChart").style.display="none";$("capitalEmpty").style.display="grid";$("capitalEmpty").innerHTML="当前参数还没有可用的 trans.xlsx銆?;return}$("capitalEmpty").style.display="none";$("capitalChart").style.display="block";Plotly.newPlot($("capitalChart"),[{x:trans.capital_x,y:trans.capital_y,type:"scatter",mode:"lines+markers",line:{color:"#2f6bff",width:2},marker:{size:6,color:"#2f6bff"},hovertemplate:"%{x}<br>capital=%{y}<extra></extra>"}],{margin:{l:44,r:18,t:16,b:34},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:{showgrid:false},yaxis:{title:"capital",gridcolor:"#e8eef8",zeroline:false},showlegend:false},{responsive:true,displayModeBar:false})}
function htmlEnabled(record){if(!record||!state.summary){$("openHtml").disabled=true;return}$("openHtml").disabled=!detailFile(state.htmlFiles,state.summary.batch_label,record.param_tag,".html")}
async function setRecord(record,switchTab){state.activeRecord=record;inputsFrom(record);meta(record);htmlEnabled(record);$("htmlWrap").classList.remove("active");if(switchTab)setTab("detail");try{const [price,trans]=await Promise.all([ensurePrice(),ensureTrans(record)]);priceChart(price,trans);capitalChart(trans);setStatus("璇︽儏宸插悓姝?)}catch(error){$("priceChart").style.display="none";$("capitalChart").style.display="none";$("priceEmpty").style.display="grid";$("capitalEmpty").style.display="grid";$("priceEmpty").innerHTML="璇︽儏杞藉叆澶辫触锛?br>"+esc(error.message);$("capitalEmpty").innerHTML="璇︽儏杞藉叆澶辫触锛?br>"+esc(error.message);setStatus("详情载入失败")}}
async function loadSummary(file){setStatus("正在读取 outcome_stats");state.summary=file.relative_path?await fetchJson("/api/preset-summary?path="+encodeURIComponent(file.relative_path)):await upload("/api/summary",file);configInput($("inputBar"),state.summary.controls.open_bar);configInput($("inputThreshold"),state.summary.controls.open_threshold);configInput($("inputCont"),state.summary.controls.open_continous_threshold);overview();const record=recordByKey(state.summary.default_key);if(record)await setRecord(record,false);setStatus("鎵规宸茶浇鍏?)}
function folderChanged(files,folderLabel){resetUi();state.files=[...files];state.summaryFiles=state.files.filter(isSummary);state.transFiles=state.files.filter(isTrans);state.perfFiles=state.files.filter(isPerf);state.htmlFiles=state.files.filter(isHtml);if(!state.files.length){$("folderName").textContent="尚未选择结果大文件夹";setStatus("等待目录");return}state.folderLabel=folderLabel||(rel(state.files[0]).split("/")[0]||"宸查€夋嫨鐩綍");$("folderName").textContent=state.folderLabel;$("metaFolder").textContent=state.folderLabel;const select=$("batchSelect");select.innerHTML='<option value="">璇烽€夋嫨鍥炴祴鎵规</option>';for(const file of [...state.summaryFiles].sort((a,b)=>fileMtime(b)-fileMtime(a))){const opt=document.createElement("option");opt.value=pathOf(file)||file.name;opt.textContent=file.name.replace(/\\.xlsx$/i,"");select.appendChild(opt)}select.disabled=!state.summaryFiles.length;if(!state.summaryFiles.length){setStatus("缂哄皯 outcome_stats");$("overviewEmpty").innerHTML="褰撳墠鐩綍閲屾病鏈?outcome_stats.xlsx銆?br>璇蜂娇鐢ㄦ洿鏂板悗鐨勫洖娴嬬▼搴忛噸鏂拌緭鍑?outcome stats 姹囨€绘枃浠躲€?;return}setStatus("鐩綍宸茶浇鍏?);$("overviewEmpty").innerHTML="璇烽€夋嫨涓€涓洖娴嬫壒娆°€?}
$("pickBtn").addEventListener("click",()=>{$("folderInput").click()});$("folderInput").addEventListener("change",evt=>folderChanged(evt.target.files));$("batchSelect").addEventListener("change",async()=>{const file=state.summaryFiles.find(item=>(pathOf(item)||item.name)===$("batchSelect").value);if(!file)return;try{await loadSummary(file)}catch(error){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";$("overviewEmpty").innerHTML="姹囨€绘枃浠惰鍙栧け璐ワ細<br>"+esc(error.message);setStatus("批次读取失败")}});$("tabOverview").addEventListener("click",()=>setTab("overview"));$("tabDetail").addEventListener("click",()=>setTab("detail"));$("inputBar").addEventListener("change",async()=>{const record=recordByKey(selKey());if(record)await setRecord(record,false);else{state.activeRecord=null;meta(null);htmlEnabled(null);$("priceChart").style.display="none";$("capitalChart").style.display="none";$("priceEmpty").style.display="grid";$("capitalEmpty").style.display="grid";$("priceEmpty").innerHTML="褰撳墠涓夊厓鍙傛暟缁勫悎娌℃湁瀵瑰簲缁撴灉銆?;$("capitalEmpty").innerHTML="当前三元参数组合没有对应 trans.xlsx銆?;setStatus("璇ュ弬鏁扮粍鍚堟病鏈夌粨鏋?)}});$("inputThreshold").addEventListener("change",async()=>{$("inputBar").dispatchEvent(new Event("change"))});$("inputCont").addEventListener("change",async()=>{$("inputBar").dispatchEvent(new Event("change"))});$("openHtml").addEventListener("click",()=>{if(!state.activeRecord||!state.summary)return;const file=detailFile(state.htmlFiles,state.summary.batch_label,state.activeRecord.param_tag,".html");if(!file)return;let url=file.relative_path?"/api/preset-html?path="+encodeURIComponent(file.relative_path):state.htmlUrlCache.get(state.activeRecord.selection_key);if(!file.relative_path&&!url){url=URL.createObjectURL(file);state.htmlUrlCache.set(state.activeRecord.selection_key,url)}$("htmlFrame").src=url;$("htmlWrap").classList.add("active")});async function bootstrapPreset(){try{const preset=await fetchJson("/api/preset-index");if(preset.error){$("folderName").textContent="预设目录无效";setStatus("棰勮鐩綍璇诲彇澶辫触锛?br>"+esc(preset.error));return}if(!preset.enabled){return}folderChanged(preset.files||[],preset.folder_label||preset.folder_path||"预设目录");$("pickBtn").textContent="鏀归€夌粨鏋滄枃浠跺す";if(state.summaryFiles.length===1){const first=state.summaryFiles[0];$("batchSelect").value=pathOf(first)||first.name;$("batchSelect").dispatchEvent(new Event("change"))}}catch(error){setStatus("棰勮鐩綍璇诲彇澶辫触锛?br>"+esc(error.message))}}bootstrapPreset();
</script></body></html>"""


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{margin:0;padding:16px;background:#f5f7fb;font-family:"Segoe UI","Microsoft YaHei UI",sans-serif;color:#152033}
.wrap{max-width:1860px;margin:0 auto}
.app-shell{display:grid;grid-template-columns:minmax(0,1fr) 430px;gap:16px;align-items:start}
.panel{min-width:0;background:#fff;border:1px solid #d9e1ee;border-radius:18px;padding:16px;box-shadow:0 16px 40px rgba(28,49,88,.08)}
.title{font-size:20px;font-weight:700}
.sub{font-size:12px;color:#66758c;margin-top:4px}
.box,.btn,.sel,.num{height:42px;border:1px solid #d9e1ee;border-radius:12px;padding:0 12px;font-size:13px}
.box{display:flex;align-items:center;color:#66758c;background:#fbfcff;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.btn{background:#fff;cursor:pointer;font-weight:600}
.btn.primary{background:#2f6bff;color:#fff;border-color:#2f6bff}
.btn:disabled{cursor:not-allowed;color:#99a7bf;background:#f5f7fb}
.tabs{display:flex;gap:8px;flex-wrap:wrap}
.tab{height:36px;padding:0 18px;border-radius:999px;border:1px solid #d9e1ee;background:#fff;color:#66758c;font-weight:600;cursor:pointer}
.tab.active{background:rgba(47,107,255,.1);color:#2f6bff;border-color:rgba(47,107,255,.25)}
.page{display:none}
.page.active{display:block}
.chartbox{padding:14px;border:1px solid #d9e1ee;border-radius:14px;background:#fbfcff}
.charttitle{font-size:14px;font-weight:700;margin-bottom:10px}
.chartsub{font-size:12px;color:#66758c;margin-top:-4px;margin-bottom:10px}
.chart{min-height:340px}
.empty{min-height:120px;border:1px dashed #c7d2e5;border-radius:14px;display:grid;place-items:center;text-align:center;color:#66758c;padding:16px;line-height:1.7;background:rgba(255,255,255,.7)}
.detail-shell{display:grid;gap:16px}
.price-box .chart{min-height:656px}
.controls-card{display:grid;gap:12px}
.controls-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;align-items:end}
.label{font-size:12px;color:#66758c;font-weight:600;margin-bottom:6px}
.meta{display:grid;gap:8px;font-size:13px}
.row{display:grid;grid-template-columns:110px 1fr;gap:10px}
.name{color:#66758c;font-weight:600}
.rail-stack{display:grid;gap:12px}
.rail-stack>.box,.rail-stack>.sel,.rail-stack>.btn{width:100%;box-sizing:border-box}
.drawer{position:sticky;top:16px;border:1px solid #d9e1ee;border-radius:16px;background:#fbfcff;padding:14px;transition:width .18s ease,padding .18s ease;min-width:0}
.app-drawer{width:430px;box-sizing:border-box}
.drawer.collapsed{width:64px;padding:14px 8px}
.drawer-head{display:flex;align-items:center;justify-content:space-between;gap:10px}
.drawer-body{display:grid;gap:14px;margin-top:12px}
.drawer.collapsed .drawer-body,.drawer.collapsed .drawer-title-wrap{display:none}
.drawer-toggle{min-width:70px}
.drawer.collapsed .drawer-toggle{min-width:0;width:100%;padding:0}
.iframe{display:none;border:1px solid #d9e1ee;border-radius:14px;overflow:hidden;min-height:420px;background:#fff}
.iframe.active{display:block}
iframe{width:100%;min-height:420px;border:0;display:block;background:#fff}
@media(max-width:1320px){
  .app-shell{grid-template-columns:1fr}
  .drawer{position:static}
  .app-drawer{width:auto}
}
@media(max-width:900px){
  .controls-grid{grid-template-columns:1fr}
  .price-box .chart{min-height:520px}
}
</style>
</head>
<body>
<div class="wrap">
  <div class="panel">
    <div class="top">
      <div>
        <div class="title">Backtest Dashboard</div>
        <div class="sub">总览读取 outcome_stats锛岃鎯呭垏鎹㈣鍙?trans.xlsx锛屽師 HTML 浠嶅彲闅忔椂鎵撳紑銆?/div>
      </div>
      <div id="folderName" class="box">尚未选择结果大文件夹</div>
      <select id="batchSelect" class="sel" disabled>
        <option value="">璇烽€夋嫨鍥炴祴鎵规</option>
      </select>
      <button id="pickBtn" class="btn primary">閫夋嫨缁撴灉鏂囦欢澶?/button>
    </div>

    <div class="tabs">
      <button id="tabOverview" class="tab active">涓婚〉闈?/button>
      <button id="tabDetail" class="tab">璇︽儏椤?/button>
    </div>

    <div id="pageOverview" class="page active">
      <div class="chartbox">
        <div class="charttitle">收益总览</div>
        <div id="overviewEmpty" class="empty">璇烽€夋嫨涓€涓甫鏈?outcome_stats.xlsx 鐨勭粨鏋滃ぇ鏂囦欢澶广€?/div>
        <div id="overviewChart" class="chart" style="display:none"></div>
      </div>
    </div>

    <div id="pageDetail" class="page">
      <div class="detail-shell">
        <div class="chartbox price-box">
            <div class="charttitle">鍏叡琛屾儏鍥句笌涔板崠鐐?/div>
            <div class="chartsub">K 绾块噰鐢ㄥ師 HTML 鐨勯粦鐧介鏍硷紝涓嶆樉绀烘病鏈夋暟鎹殑鏃ユ湡闂撮殭銆?/div>
            <div id="priceEmpty" class="empty">璇烽€夋嫨涓€涓壒娆★紝鍐嶄粠涓婚〉闈㈢偣閫変竴涓弬鏁扮偣銆?/div>
            <div id="priceChart" class="chart" style="display:none"></div>
          </div>

        <div class="detail-lower">
          <div class="info-stack">
            <div class="chartbox controls-card">
            <div class="charttitle">参数切换</div>
            <div class="controls-grid">
              <div>
                <div class="label">鏃堕棿绐楀彛</div>
                <input id="inputBar" class="num" type="number" disabled>
              </div>
              <div>
                <div class="label">閫熷害闄愬埗</div>
                <input id="inputThreshold" class="num" type="number" disabled>
              </div>
              <div>
                <div class="label">寮€浠撻棬妲?/div>
                <input id="inputCont" class="num" type="number" disabled>
              </div>
              <div>
                <div class="label">鍘?HTML</div>
                <button id="openHtml" class="btn" disabled>鎵撳紑鍘?HTML</button>
              </div>
            </div>
          </div>

            <div class="chartbox">
            <div class="charttitle">交易资金曲线</div>
            <div id="capitalEmpty" class="empty">当前还没有可用的 trans.xlsx銆?/div>
            <div id="capitalChart" class="chart" style="display:none;min-height:280px"></div>
            </div>
          </div>

          <aside id="settingsDrawer" class="drawer">
          <div class="drawer-head">
            <div class="drawer-title-wrap">
              <div class="charttitle" style="margin-bottom:4px">璁剧疆椤?/div>
              <div class="sub">褰撳墠鍙傛暟銆佺姸鎬佷笌鍘?HTML 棰勮銆?/div>
            </div>
            <button id="settingsToggle" class="btn drawer-toggle">收起</button>
          </div>

          <div class="drawer-body">
            <div class="chartbox">
              <div class="charttitle">当前参数</div>
              <div class="meta">
                <div class="row"><div class="name">缁撴灉鏂囦欢澶?/div><div id="metaFolder">-</div></div>
                <div class="row"><div class="name">批次</div><div id="metaBatch">-</div></div>
                <div class="row"><div class="name">参数标签</div><div id="metaTag">-</div></div>
                <div class="row"><div class="name">鏈€缁堣祫鏈?/div><div id="metaCapital">-</div></div>
                <div class="row"><div class="name">交易次数</div><div id="metaTrade">-</div></div>
                <div class="row"><div class="name">鏈€澶у洖鎾?/div><div id="metaWd">-</div></div>
                <div class="row"><div class="name">宄板€艰祫鏈?/div><div id="metaHigh">-</div></div>
              </div>
            </div>

            <div class="chartbox">
              <div class="charttitle">鐘舵€?/div>
              <div id="status" class="empty" style="min-height:90px">等待目录</div>
            </div>

            <div class="chartbox">
              <div class="charttitle">鍘?HTML 预览</div>
              <div id="htmlEmpty" class="empty">閫夋嫨涓€涓弬鏁板悗锛屽彲鍦ㄨ繖閲岄瑙堝搴旂殑鍘?HTML銆?/div>
              <div id="htmlWrap" class="iframe"><iframe id="htmlFrame"></iframe></div>
            </div>
          </div>
          </aside>
        </div>
      </div>
    </div>
  </div>
</div>
<input id="folderInput" type="file" webkitdirectory directory multiple style="display:none">
<script>
const state={files:[],summaryFiles:[],transFiles:[],perfFiles:[],htmlFiles:[],summary:null,activeRecord:null,folderLabel:"",priceCache:new Map(),transCache:new Map(),htmlUrlCache:new Map(),settingsCollapsed:false};
const $=id=>document.getElementById(id);
const CANDLE_UP_EDGE="rgba(185, 185, 185, 0.9)";
const CANDLE_DOWN_EDGE="rgba(85, 85, 85, 0.9)";
const CANDLE_UP_FILL="rgba(245, 245, 245, 0.9)";
const CANDLE_DOWN_FILL="rgba(120, 120, 120, 0.9)";
const ACCENT_BLUE="#1F77B4";
const SELL_WD_COLOR="green";
const SELL_SPEED_COLOR="black";

function setStatus(text){$("status").innerHTML=text}
function setupLayout(){const wrap=document.querySelector(".wrap");const panel=document.querySelector(".panel");const drawer=$("settingsDrawer");if(!wrap||!panel||!drawer)return;let shell=wrap.querySelector(".app-shell");if(!shell){shell=document.createElement("div");shell.className="app-shell";wrap.insertBefore(shell,panel)}if(panel.parentElement!==shell){shell.appendChild(panel)}drawer.classList.add("app-drawer");if(drawer.parentElement!==shell){shell.appendChild(drawer)}const titleNode=drawer.querySelector(".drawer-title-wrap .charttitle");if(titleNode){titleNode.className="title";titleNode.textContent="Backtest Dashboard"}const subNode=drawer.querySelector(".drawer-title-wrap .sub");if(subNode){subNode.textContent="总览读取 outcome_stats锛岃鎯呭垏鎹㈣鍙?trans.xlsx锛屽師 HTML 浠嶅彲闅忔椂鎵撳紑銆?}let workspaceCard=document.getElementById("workspaceCard");if(!workspaceCard){workspaceCard=document.createElement("div");workspaceCard.id="workspaceCard";workspaceCard.className="chartbox";workspaceCard.innerHTML='<div class="charttitle">宸ヤ綔鍖?/div><div id="workspaceControls" class="rail-stack"></div>';const drawerBody=drawer.querySelector(".drawer-body");if(drawerBody){drawerBody.insertBefore(workspaceCard,drawerBody.firstChild)}}const workspaceControls=document.getElementById("workspaceControls");if(workspaceControls){for(const id of ["folderName","batchSelect","pickBtn"]){const node=$(id);if(node&&node.parentElement!==workspaceControls){workspaceControls.appendChild(node)}}const tabs=document.querySelector(".tabs");if(tabs&&tabs.parentElement!==workspaceControls){workspaceControls.appendChild(tabs)}}const top=document.querySelector(".top");if(top){top.remove()}const detailShell=document.querySelector("#pageDetail .detail-shell");const priceBox=detailShell?.querySelector(".price-box");const controlsCard=document.querySelector("#pageDetail .controls-card");if(detailShell&&priceBox&&controlsCard&&detailShell.firstElementChild!==controlsCard){detailShell.insertBefore(controlsCard,priceBox)}}
function setTab(name){$("tabOverview").classList.toggle("active",name==="overview");$("tabDetail").classList.toggle("active",name==="detail");$("pageOverview").classList.toggle("active",name==="overview");$("pageDetail").classList.toggle("active",name==="detail")}
function setSettingsCollapsed(collapsed){state.settingsCollapsed=collapsed;$("settingsDrawer").classList.toggle("collapsed",collapsed);$("settingsToggle").textContent=collapsed?"灞曞紑":"收起"}
function pathOf(file){return String(file.relative_path||file.webkitRelativePath||"")}
function rel(file){return String(pathOf(file)||file.name).replaceAll("\\\\","/")}
function inDir(file,dir){const path=rel(file).toLowerCase();const prefix=dir.toLowerCase()+"/";return path.startsWith(prefix)||path.includes("/"+prefix)}
function esc(text){return String(text).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")}
function keyOf(v){if(v===null||v===undefined||v==="")return"";const n=Number(v);if(Number.isNaN(n))return"";return String(Number(n.toFixed(10))).replace(/\\.0+$/,"")}
function selKey(){return [keyOf($("inputBar").value),keyOf($("inputThreshold").value),keyOf($("inputCont").value)].join("|")}
function isSummary(file){return inDir(file,"outcome stats")&&file.name.toLowerCase().endsWith("outcome_stats.xlsx")}
function isTrans(file){return inDir(file,"trans")&&file.name.toLowerCase().endsWith("trans.xlsx")}
function isPerf(file){return inDir(file,"perf")&&file.name.toLowerCase().endsWith("perf.xlsx")}
function isHtml(file){return inDir(file,"html")&&file.name.toLowerCase().endsWith(".html")}
function fileMtime(file){return Number(file.last_modified??file.lastModified??0)}
function shortTick(value){const text=String(value||"");return text.length>16?text.slice(5,16):text}
function axisLabel(value){return String(value??"")}
function sampleIndexTicks(values,count=8){if(!values||!values.length)return{vals:[],labels:[]};if(values.length<=count){return{vals:values.map((_,index)=>index),labels:values.map(shortTick)}}const step=Math.max(1,Math.ceil(values.length/count));const vals=[];const labels=[];for(let index=0;index<values.length;index+=step){vals.push(index);labels.push(shortTick(values[index]))}const lastIndex=values.length-1;if(vals[vals.length-1]!==lastIndex){vals.push(lastIndex);labels.push(shortTick(values[lastIndex]))}return{vals,labels}}
function numericAxis(values){const ticks=sampleIndexTicks(values);return{type:"linear",tickmode:"array",tickvals:ticks.vals,ticktext:ticks.labels,showgrid:false,zeroline:false,rangeslider:{visible:false}}}
function buildIndexMap(values){const map=new Map();for(let index=0;index<(values||[]).length;index+=1){const key=axisLabel(values[index]);if(key&&!map.has(key)){map.set(key,index)}}return map}
function axisPoints(xs,ys,indexMap,keepBreaks=false){const outX=[];const outY=[];const labels=[];for(let index=0;index<(xs||[]).length;index+=1){const label=axisLabel(xs[index]);if(!label){if(keepBreaks){outX.push(null);outY.push(null);labels.push("")}continue}const axisX=indexMap.get(label);if(axisX===undefined){if(keepBreaks){outX.push(null);outY.push(null);labels.push("")}continue}outX.push(axisX);outY.push(ys[index]);labels.push(label)}return{x:outX,y:outY,text:labels}}
function gapShapes(values){const shapes=[];const dayMs=24*60*60*1000;for(let index=1;index<(values||[]).length;index+=1){const leftMs=Date.parse(String(values[index-1]||"").replace(" ","T"));const rightMs=Date.parse(String(values[index]||"").replace(" ","T"));if(!Number.isFinite(leftMs)||!Number.isFinite(rightMs))continue;if(rightMs-leftMs<=dayMs)continue;shapes.push({type:"line",xref:"x",yref:"paper",x0:index-0.5,x1:index-0.5,y0:0,y1:1,line:{color:"rgba(120,120,120,0.3)",width:1,dash:"dash"}})}return shapes}
async function upload(url,file){const fd=new FormData();fd.append("file",file,file.name);const res=await fetch(url,{method:"POST",body:fd});const data=await res.json();if(!res.ok)throw new Error(data.error||"请求失败");return data}
async function fetchJson(url){const res=await fetch(url);const data=await res.json();if(!res.ok)throw new Error(data.error||"request failed");return data}
function clearHtmlPreview(message){$("htmlFrame").removeAttribute("src");$("htmlWrap").classList.remove("active");$("htmlEmpty").style.display="grid";$("htmlEmpty").innerHTML=message}
function showHtmlPreview(url){$("htmlEmpty").style.display="none";$("htmlWrap").classList.add("active");$("htmlFrame").src=url}
function resetUi(){for(const url of state.htmlUrlCache.values()){URL.revokeObjectURL(url)}state.summary=null;state.activeRecord=null;state.folderLabel="";state.priceCache.clear();state.transCache.clear();state.htmlUrlCache.clear();$("batchSelect").innerHTML='<option value="">璇烽€夋嫨鍥炴祴鎵规</option>';$("batchSelect").disabled=true;$("overviewChart").style.display="none";$("priceChart").style.display="none";$("capitalChart").style.display="none";$("overviewEmpty").style.display="grid";$("priceEmpty").style.display="grid";$("capitalEmpty").style.display="grid";$("openHtml").disabled=true;$("inputBar").disabled=true;$("inputThreshold").disabled=true;$("inputCont").disabled=true;clearHtmlPreview("閫夋嫨涓€涓弬鏁板悗锛屽彲鍦ㄨ繖閲岄瑙堝搴旂殑鍘?HTML銆?);$("metaFolder").textContent="-";$("metaBatch").textContent="-";$("metaTag").textContent="-";$("metaCapital").textContent="-";$("metaTrade").textContent="-";$("metaWd").textContent="-";$("metaHigh").textContent="-"}
function configInput(el,meta){const has=meta&&meta.values&&meta.values.length;el.disabled=!has;if(!has){el.value="";el.removeAttribute("min");el.removeAttribute("max");el.removeAttribute("step");return}el.min=meta.values[0];el.max=meta.values[meta.values.length-1];el.step=meta.step??"any"}
function batchFile(files,label,suffix){const arr=files.filter(file=>file.name.toLowerCase().endsWith(suffix)&&file.name.includes(label)).sort((a,b)=>fileMtime(b)-fileMtime(a));return arr[0]||null}
function detailFile(files,label,paramTag,suffix){const arr=files.filter(file=>file.name.toLowerCase().endsWith(suffix)&&file.name.includes(label)&&file.name.includes(paramTag)).sort((a,b)=>fileMtime(b)-fileMtime(a));return arr[0]||null}
function recordByKey(key){return state.summary?.records.find(item=>item.selection_key===key)||null}
function meta(record){$("metaFolder").textContent=state.folderLabel||"-";if(!record){$("metaBatch").textContent="-";$("metaTag").textContent="-";$("metaCapital").textContent="-";$("metaTrade").textContent="-";$("metaWd").textContent="-";$("metaHigh").textContent="-";return}$("metaBatch").textContent=state.summary.batch_label;$("metaTag").textContent=record.param_tag;$("metaCapital").textContent=record.capital??"-";$("metaTrade").textContent=record.trade_num??"-";$("metaWd").textContent=record.biggest_wd??"-";$("metaHigh").textContent=record.outcome_high??"-"}
function inputsFrom(record){$("inputBar").value=record.open_bar??"";$("inputThreshold").value=record.open_threshold??"";$("inputCont").value=record.open_continous_threshold??""}
function overview(){if(!state.summary||!state.summary.records.length){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";return}const rows=state.summary.records;$("overviewEmpty").style.display="none";$("overviewChart").style.display="block";Plotly.newPlot($("overviewChart"),[{x:rows.map((_,i)=>i+1),y:rows.map(r=>r.capital),mode:"lines+markers",line:{color:"#2f6bff",width:2},marker:{size:6,color:"#2f6bff"},customdata:rows.map(r=>r.selection_key),text:rows.map(r=>["param_tag: "+esc(r.param_tag),"open_bar: "+(r.open_bar??"-"),"open_threshold: "+(r.open_threshold??"-"),"open_cont: "+(r.open_continous_threshold??"-"),"capital: "+(r.capital??"-"),"trade_num: "+(r.trade_num??"-")].join("<br>")),hovertemplate:"%{text}<extra></extra>",name:"capital"}],{margin:{l:44,r:18,t:16,b:40},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:{title:"参数组合顺序",showgrid:false,zeroline:false},yaxis:{title:"capital",gridcolor:"#e8eef8",zeroline:false},showlegend:false},{responsive:true,displayModeBar:false});if($("overviewChart").removeAllListeners){$("overviewChart").removeAllListeners("plotly_click")}$("overviewChart").on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record)setRecord(record,true)})}
async function ensurePrice(){const label=state.summary?.batch_label;if(!label)return null;if(state.priceCache.has(label))return state.priceCache.get(label);const file=batchFile(state.perfFiles,label,"perf.xlsx");if(!file){state.priceCache.set(label,null);return null}setStatus("正在读取公共行情数据");const data=file.relative_path?await fetchJson("/api/preset-price?path="+encodeURIComponent(file.relative_path)):await upload("/api/price",file);state.priceCache.set(label,data);return data}
async function ensureTrans(record){if(!record||!state.summary)return null;if(state.transCache.has(record.selection_key))return state.transCache.get(record.selection_key);const file=detailFile(state.transFiles,state.summary.batch_label,record.param_tag,"trans.xlsx");if(!file){state.transCache.set(record.selection_key,null);return null}setStatus("正在读取 trans.xlsx");const data=file.relative_path?await fetchJson("/api/preset-trans?path="+encodeURIComponent(file.relative_path)):await upload("/api/trans",file);state.transCache.set(record.selection_key,data);return data}
function priceChart(price,trans){if(!price||!price.x.length){$("priceChart").style.display="none";$("priceEmpty").style.display="grid";$("priceEmpty").innerHTML="褰撳墠鎵规缂哄皯鍙敤鐨?perf.xlsx銆?;return}$("priceEmpty").style.display="none";$("priceChart").style.display="block";const candleX=price.x.map((_,index)=>index);const priceIndexMap=buildIndexMap(price.x);const traces=[{type:"candlestick",x:candleX,open:price.open,high:price.high,low:price.low,close:price.close,text:price.x,name:"price",increasing:{line:{color:CANDLE_UP_EDGE,width:0.8},fillcolor:CANDLE_UP_FILL},decreasing:{line:{color:CANDLE_DOWN_EDGE,width:0.8},fillcolor:CANDLE_DOWN_FILL},hovertemplate:"%{text}<br>open=%{open}<br>high=%{high}<br>low=%{low}<br>close=%{close}<extra></extra>"}];if(trans){const capitalPoints=axisPoints(trans.capital_x,trans.capital_y,priceIndexMap);const tradeLink=axisPoints(trans.trade_link_x,trans.trade_link_y,priceIndexMap,true);const buyPoints=axisPoints(trans.buy_points.x,trans.buy_points.y,priceIndexMap);const sellWdPoints=axisPoints(trans.sell_wd_points.x,trans.sell_wd_points.y,priceIndexMap);const sellSpeedPoints=axisPoints(trans.sell_speed_points.x,trans.sell_speed_points.y,priceIndexMap);if(capitalPoints.x.length)traces.unshift({type:"scatter",mode:"lines",x:capitalPoints.x,y:capitalPoints.y,text:capitalPoints.text,line:{color:ACCENT_BLUE,width:1.2,shape:"hv"},name:"capital",yaxis:"y2",hovertemplate:"%{text}<br>capital=%{y}<extra></extra>"});if(tradeLink.x.length)traces.push({type:"scatter",mode:"lines",x:tradeLink.x,y:tradeLink.y,line:{color:ACCENT_BLUE,width:2},hoverinfo:"skip",name:"trade_link"});if(buyPoints.x.length)traces.push({type:"scatter",mode:"markers",x:buyPoints.x,y:buyPoints.y,text:buyPoints.text,marker:{color:"red",size:4},name:"buy",hovertemplate:"buy<br>%{text}<br>price=%{y}<extra></extra>"});if(sellWdPoints.x.length)traces.push({type:"scatter",mode:"markers",x:sellWdPoints.x,y:sellWdPoints.y,text:sellWdPoints.text,marker:{color:SELL_WD_COLOR,size:4},name:"sell_wd",hovertemplate:"sell_wd<br>%{text}<br>price=%{y}<extra></extra>"});if(sellSpeedPoints.x.length)traces.push({type:"scatter",mode:"markers",x:sellSpeedPoints.x,y:sellSpeedPoints.y,text:sellSpeedPoints.text,marker:{color:SELL_SPEED_COLOR,size:4},name:"sell_speed",hovertemplate:"sell_speed<br>%{text}<br>price=%{y}<extra></extra>"})}Plotly.newPlot($("priceChart"),traces,{margin:{l:48,r:52,t:12,b:36},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:numericAxis(price.x),yaxis:{title:"price",gridcolor:"#e8eef8",zeroline:false},yaxis2:{title:"capital",overlaying:"y",side:"right",showgrid:false,zeroline:false},legend:{orientation:"h",yanchor:"bottom",y:1.02,xanchor:"left",x:0}},{responsive:true,displayModeBar:false})}
function capitalChart(trans){if(!trans||!trans.capital_x.length){$("capitalChart").style.display="none";$("capitalEmpty").style.display="grid";$("capitalEmpty").innerHTML="当前参数还没有可用的 trans.xlsx銆?;return}$("capitalEmpty").style.display="none";$("capitalChart").style.display="block";const axisX=trans.capital_x.map((_,index)=>index);Plotly.newPlot($("capitalChart"),[{x:axisX,y:trans.capital_y,type:"scatter",mode:"lines",text:trans.capital_x,line:{color:ACCENT_BLUE,width:1.2,shape:"hv"},hovertemplate:"%{text}<br>capital=%{y}<extra></extra>"}],{margin:{l:48,r:18,t:12,b:36},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:numericAxis(trans.capital_x),yaxis:{title:"capital",gridcolor:"#e8eef8",zeroline:false},showlegend:false},{responsive:true,displayModeBar:false})}
function htmlFileFor(record){if(!record||!state.summary)return null;return detailFile(state.htmlFiles,state.summary.batch_label,record.param_tag,".html")}
function htmlUrlFor(record){const file=htmlFileFor(record);if(!file)return"";if(file.relative_path){return"/api/preset-html?path="+encodeURIComponent(file.relative_path)}let url=state.htmlUrlCache.get(record.selection_key);if(!url){url=URL.createObjectURL(file);state.htmlUrlCache.set(record.selection_key,url)}return url}
function htmlEnabled(record){$("openHtml").disabled=!htmlFileFor(record)}
async function setRecord(record,switchTab){state.activeRecord=record;inputsFrom(record);meta(record);htmlEnabled(record);clearHtmlPreview("閫夋嫨涓€涓弬鏁板悗锛屽彲鍦ㄨ繖閲岄瑙堝搴旂殑鍘?HTML銆?);if(switchTab)setTab("detail");try{const [price,trans]=await Promise.all([ensurePrice(),ensureTrans(record)]);priceChart(price,trans);capitalChart(trans);setStatus("璇︽儏宸插悓姝?)}catch(error){$("priceChart").style.display="none";$("capitalChart").style.display="none";$("priceEmpty").style.display="grid";$("capitalEmpty").style.display="grid";$("priceEmpty").innerHTML="璇︽儏杞藉叆澶辫触锛?br>"+esc(error.message);$("capitalEmpty").innerHTML="璇︽儏杞藉叆澶辫触锛?br>"+esc(error.message);setStatus("详情载入失败")}}
async function loadSummary(file){setStatus("正在读取 outcome_stats");state.summary=file.relative_path?await fetchJson("/api/preset-summary?path="+encodeURIComponent(file.relative_path)):await upload("/api/summary",file);configInput($("inputBar"),state.summary.controls.open_bar);configInput($("inputThreshold"),state.summary.controls.open_threshold);configInput($("inputCont"),state.summary.controls.open_continous_threshold);overview();const record=recordByKey(state.summary.default_key);if(record)await setRecord(record,false);setStatus("鎵规宸茶浇鍏?)}
function folderChanged(files,folderLabel){resetUi();state.files=[...files];state.summaryFiles=state.files.filter(isSummary);state.transFiles=state.files.filter(isTrans);state.perfFiles=state.files.filter(isPerf);state.htmlFiles=state.files.filter(isHtml);if(!state.files.length){$("folderName").textContent="尚未选择结果大文件夹";setStatus("等待目录");return}state.folderLabel=folderLabel||(rel(state.files[0]).split("/")[0]||"宸查€夋嫨鐩綍");$("folderName").textContent=state.folderLabel;$("metaFolder").textContent=state.folderLabel;const select=$("batchSelect");select.innerHTML='<option value="">璇烽€夋嫨鍥炴祴鎵规</option>';for(const file of [...state.summaryFiles].sort((a,b)=>fileMtime(b)-fileMtime(a))){const opt=document.createElement("option");opt.value=pathOf(file)||file.name;opt.textContent=file.name.replace(/\\.xlsx$/i,"");select.appendChild(opt)}select.disabled=!state.summaryFiles.length;if(!state.summaryFiles.length){setStatus("缂哄皯 outcome_stats");$("overviewEmpty").innerHTML="褰撳墠鐩綍閲屾病鏈?outcome_stats.xlsx銆?br>璇蜂娇鐢ㄦ洿鏂板悗鐨勫洖娴嬬▼搴忛噸鏂拌緭鍑?outcome stats 姹囨€绘枃浠躲€?;return}setStatus("鐩綍宸茶浇鍏?);$("overviewEmpty").innerHTML="璇烽€夋嫨涓€涓洖娴嬫壒娆°€?}

$("pickBtn").addEventListener("click",()=>{$("folderInput").click()});
$("folderInput").addEventListener("change",evt=>folderChanged(evt.target.files));
$("batchSelect").addEventListener("change",async()=>{const file=state.summaryFiles.find(item=>(pathOf(item)||item.name)===$("batchSelect").value);if(!file)return;try{await loadSummary(file)}catch(error){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";$("overviewEmpty").innerHTML="姹囨€绘枃浠惰鍙栧け璐ワ細<br>"+esc(error.message);setStatus("批次读取失败")}});
$("tabOverview").addEventListener("click",()=>setTab("overview"));
$("tabDetail").addEventListener("click",()=>setTab("detail"));
$("settingsToggle").addEventListener("click",()=>setSettingsCollapsed(!state.settingsCollapsed));
$("inputBar").addEventListener("change",async()=>{const record=recordByKey(selKey());if(record)await setRecord(record,false);else{state.activeRecord=null;meta(null);htmlEnabled(null);$("priceChart").style.display="none";$("capitalChart").style.display="none";$("priceEmpty").style.display="grid";$("capitalEmpty").style.display="grid";$("priceEmpty").innerHTML="褰撳墠涓夊厓鍙傛暟缁勫悎娌℃湁瀵瑰簲缁撴灉銆?;$("capitalEmpty").innerHTML="当前三元参数组合没有对应 trans.xlsx銆?;clearHtmlPreview("当前三元参数组合没有对应的原 HTML銆?);setStatus("璇ュ弬鏁扮粍鍚堟病鏈夌粨鏋?)}});
$("inputThreshold").addEventListener("change",async()=>{$("inputBar").dispatchEvent(new Event("change"))});
$("inputCont").addEventListener("change",async()=>{$("inputBar").dispatchEvent(new Event("change"))});
$("openHtml").addEventListener("click",()=>{if(!state.activeRecord||!state.summary)return;const url=htmlUrlFor(state.activeRecord);if(!url){setStatus("褰撳墠鍙傛暟缂哄皯鍘?HTML");return}showHtmlPreview(url);setSettingsCollapsed(false);setStatus("鍘?HTML 宸叉墦寮€");try{const popup=window.open(url,"_blank","noopener");if(!popup){window.location.href=url}}catch(error){window.location.href=url}});

async function bootstrapPreset(){try{const preset=await fetchJson("/api/preset-index");if(preset.error){$("folderName").textContent="预设目录无效";setStatus("棰勮鐩綍璇诲彇澶辫触锛?br>"+esc(preset.error));return}if(!preset.enabled){return}folderChanged(preset.files||[],preset.folder_label||preset.folder_path||"预设目录");$("pickBtn").textContent="鏀归€夌粨鏋滄枃浠跺す";if(state.summaryFiles.length){const first=state.summaryFiles[0];$("batchSelect").value=pathOf(first)||first.name;$("batchSelect").dispatchEvent(new Event("change"))}}catch(error){setStatus("棰勮鐩綍璇诲彇澶辫触锛?br>"+esc(error.message))}}

setupLayout();
setSettingsCollapsed(false);
bootstrapPreset();
</script>
</body>
</html>"""

PAGE = PAGE.replace(
    'async function upload(url,file){',
    'function gapShapes(values){const shapes=[];const dayMs=24*60*60*1000;for(let index=1;index<(values||[]).length;index+=1){const leftMs=Date.parse(String(values[index-1]||"").replace(" ","T"));const rightMs=Date.parse(String(values[index]||"").replace(" ","T"));if(!Number.isFinite(leftMs)||!Number.isFinite(rightMs))continue;if(rightMs-leftMs<=dayMs)continue;shapes.push({type:"line",xref:"x",yref:"paper",x0:index-0.5,x1:index-0.5,y0:0,y1:1,line:{color:"rgba(120,120,120,0.3)",width:1,dash:"dash"}})}return shapes}\nasync function upload(url,file){'
)
PAGE = PAGE.replace(
    'const priceIndexMap=buildIndexMap(price.x);const traces=',
    'const priceIndexMap=buildIndexMap(price.x);const shapes=gapShapes(price.x);const traces='
)
PAGE = PAGE.replace(
    'legend:{orientation:"h",yanchor:"bottom",y:1.02,xanchor:"left",x:0}},{responsive:true,displayModeBar:false})}',
    'legend:{orientation:"h",yanchor:"bottom",y:1.02,xanchor:"left",x:0},shapes:shapes},{responsive:true,displayModeBar:false})}'
)
PAGE = PAGE.replace(
    "</style>",
    """
.app-shell{display:block}
.controls-grid{grid-template-columns:repeat(4,minmax(0,220px));align-items:start}
.param-block{display:grid;gap:6px}
.param-host{display:grid;grid-template-columns:minmax(0,1fr) 36px;gap:8px;align-items:stretch}
.param-select{width:100%;height:42px;border:1px solid #d9e1ee;border-radius:12px;padding:0 12px;background:#fff;font-size:13px;color:#152033}
.param-select:disabled{background:#f5f7fb;color:#99a7bf;cursor:not-allowed}
.param-stepper{display:grid;grid-template-rows:1fr 1fr;gap:6px}
.step-btn{padding:0;border:1px solid #d9e1ee;border-radius:10px;background:#fff;color:#3a4a62;font-size:11px;font-weight:700;cursor:pointer}
.step-btn:disabled{cursor:not-allowed;color:#99a7bf;background:#f5f7fb}
.num-hidden{display:none!important}
.drawer{transition:transform .18s ease,box-shadow .18s ease}
.app-drawer{position:fixed;left:16px;top:16px;z-index:40;width:360px;max-height:calc(100vh - 32px);overflow:auto;box-shadow:0 24px 60px rgba(28,49,88,.18)}
.drawer-body{transition:opacity .18s ease}
.drawer.collapsed{transform:translateX(calc(-100% + 58px));padding:14px}
.drawer.collapsed .drawer-head{justify-content:flex-end}
.drawer.collapsed .drawer-toggle{width:44px;min-width:44px}
@media(max-width:1320px){
  .app-drawer{position:static;left:auto;top:auto;width:auto;max-height:none;margin-bottom:16px}
  .drawer.collapsed{transform:none}
}
@media(max-width:900px){
  .controls-grid{grid-template-columns:1fr}
  .param-host{grid-template-columns:minmax(0,1fr) 40px}
}
</style>"""
)
PAGE = PAGE.replace(
    'setupLayout();\nsetSettingsCollapsed(false);\nbootstrapPreset();',
    '''const legacyResetUi=resetUi;
function ensureParamUi(el){if(!el)return null;if(el._paramUi)return el._paramUi;const host=el.parentElement;if(!host)return null;host.classList.add("param-block");const shell=document.createElement("div");shell.className="param-host";const select=document.createElement("select");select.className="param-select";select.disabled=true;const stepper=document.createElement("div");stepper.className="param-stepper";const up=document.createElement("button");up.type="button";up.className="step-btn";up.textContent="鈻?;up.title="澧炲ぇ";const down=document.createElement("button");down.type="button";down.className="step-btn";down.textContent="鈻?;down.title="鍑忓皬";stepper.appendChild(up);stepper.appendChild(down);shell.appendChild(select);shell.appendChild(stepper);host.insertBefore(shell,el);el.type="hidden";el.classList.add("num-hidden");el.setAttribute("aria-hidden","true");const ui={select,up,down};select.addEventListener("change",()=>{if(select.disabled)return;applyParamValue(el,select.value,true)});up.addEventListener("click",()=>stepParam(el,1));down.addEventListener("click",()=>stepParam(el,-1));el._paramUi=ui;return ui}
function paramValues(el){try{return JSON.parse(el?.dataset?.values||"[]")}catch(error){return[]}}
function syncParamUi(el){const ui=ensureParamUi(el);if(!ui)return;const values=paramValues(el);const current=keyOf(el.value);if(current&&ui.select.value!==current){ui.select.value=current}const index=values.indexOf(current);const disabled=!!el.disabled||!values.length;ui.select.disabled=disabled;ui.up.disabled=disabled||index<0||index>=values.length-1;ui.down.disabled=disabled||index<=0}
function applyParamValue(el,value,emitChange){const next=keyOf(value);el.value=next;const ui=ensureParamUi(el);if(ui&&ui.select.value!==next){ui.select.value=next}syncParamUi(el);if(emitChange){el.dispatchEvent(new Event("change",{bubbles:true}))}}
function stepParam(el,direction){const values=paramValues(el);if(!values.length||el.disabled)return;const current=keyOf(el.value);let index=values.indexOf(current);if(index<0){index=0}const nextIndex=Math.max(0,Math.min(values.length-1,index+direction));if(nextIndex===index)return;applyParamValue(el,values[nextIndex],true)}
setupLayout=function(){const wrap=document.querySelector(".wrap");const panel=document.querySelector(".panel");const drawer=$("settingsDrawer");if(!wrap||!panel||!drawer)return;let shell=wrap.querySelector(".app-shell");if(!shell){shell=document.createElement("div");shell.className="app-shell";wrap.insertBefore(shell,panel)}if(panel.parentElement!==shell){shell.appendChild(panel)}drawer.classList.add("app-drawer");if(drawer.parentElement!==wrap){wrap.appendChild(drawer)}const titleNode=drawer.querySelector(".drawer-title-wrap .charttitle");if(titleNode){titleNode.className="title";titleNode.textContent="Backtest Dashboard"}const subNode=drawer.querySelector(".drawer-title-wrap .sub");if(subNode){subNode.textContent="总览读取 outcome_stats锛岃鎯呭垏鎹㈣鍙?trans.xlsx锛屽師 HTML 浠嶅彲闅忔椂鎵撳紑銆?}let workspaceCard=document.getElementById("workspaceCard");if(!workspaceCard){workspaceCard=document.createElement("div");workspaceCard.id="workspaceCard";workspaceCard.className="chartbox";workspaceCard.innerHTML='<div class="charttitle">宸ヤ綔鍖?/div><div id="workspaceControls" class="rail-stack"></div>';const drawerBody=drawer.querySelector(".drawer-body");if(drawerBody){drawerBody.insertBefore(workspaceCard,drawerBody.firstChild)}}const workspaceControls=document.getElementById("workspaceControls");if(workspaceControls){for(const id of ["folderName","batchSelect","pickBtn"]){const node=$(id);if(node&&node.parentElement!==workspaceControls){workspaceControls.appendChild(node)}}const tabs=document.querySelector(".tabs");if(tabs&&tabs.parentElement!==workspaceControls){workspaceControls.appendChild(tabs)}}const top=document.querySelector(".top");if(top){top.remove()}const detailShell=document.querySelector("#pageDetail .detail-shell");const priceBox=detailShell?.querySelector(".price-box");const controlsCard=document.querySelector("#pageDetail .controls-card");if(detailShell&&priceBox&&controlsCard&&priceBox.previousElementSibling!==controlsCard){detailShell.insertBefore(controlsCard,priceBox)}for(const id of ["inputBar","inputThreshold","inputCont"]){ensureParamUi($(id))}}
setSettingsCollapsed=function(collapsed){state.settingsCollapsed=collapsed;const drawer=$("settingsDrawer");const toggle=$("settingsToggle");if(!drawer||!toggle)return;drawer.classList.toggle("collapsed",collapsed);toggle.textContent=collapsed?"灞曞紑":"收起";toggle.title=collapsed?"灞曞紑杈规爮":"收起边栏"}
configInput=function(el,meta){const ui=ensureParamUi(el);const values=[...(meta?.values||[])].map(keyOf).filter(Boolean).filter((value,index,array)=>array.indexOf(value)===index);el.dataset.values=JSON.stringify(values);const has=values.length>0;el.disabled=!has;if(!has){el.value="";el.removeAttribute("min");el.removeAttribute("max");el.removeAttribute("step");if(ui){ui.select.innerHTML="";const option=document.createElement("option");option.value="";option.textContent="暂无参数";ui.select.appendChild(option)}syncParamUi(el);return}el.min=values[0];el.max=values[values.length-1];el.step=meta?.step??"any";if(ui){ui.select.innerHTML="";for(const value of values){const option=document.createElement("option");option.value=value;option.textContent=value;ui.select.appendChild(option)}}const current=values.includes(keyOf(el.value))?keyOf(el.value):values[0];applyParamValue(el,current,false)}
inputsFrom=function(record){applyParamValue($("inputBar"),record?.open_bar??"",false);applyParamValue($("inputThreshold"),record?.open_threshold??"",false);applyParamValue($("inputCont"),record?.open_continous_threshold??"",false)}
resetUi=function(){legacyResetUi();configInput($("inputBar"),null);configInput($("inputThreshold"),null);configInput($("inputCont"),null)}
setupLayout();
setSettingsCollapsed(false);
bootstrapPreset();'''
)
PAGE = PAGE.replace(
    "</style>",
    """
#pageOverview .charttitle{text-align:center}
#overviewChart.chart{min-height:510px}
.drawer-toggle{display:grid;place-items:center;width:38px;min-width:38px;height:38px;padding:0;border-radius:999px;background:rgba(255,255,255,.34);border-color:rgba(217,225,238,.48);color:transparent;box-shadow:none}
.drawer-toggle::before{content:"鈼€";font-size:14px;line-height:1;color:rgba(21,32,51,.72)}
.drawer.collapsed .drawer-toggle::before{content:"鈻?}
.drawer.collapsed{transform:translateX(calc(-100% + 28px));padding:0;background:transparent;border-color:transparent;box-shadow:none;overflow:visible}
.drawer.collapsed .drawer-head{padding:0;background:transparent}
.drawer.collapsed .drawer-toggle{width:28px;min-width:28px;height:88px;border-radius:0 16px 16px 0;background:rgba(255,255,255,.18);border-color:rgba(217,225,238,.24)}
.drawer-toggle:hover{background:rgba(255,255,255,.46)}
</style>"""
)
PAGE = PAGE.replace(
    'showHtmlPreview(url);setSettingsCollapsed(false);setStatus("鍘?HTML 宸叉墦寮€");try{const popup=window.open(url,"_blank","noopener");if(!popup){window.location.href=url}}catch(error){window.location.href=url}',
    'const popup=window.open(url,"_blank","noopener,noreferrer");if(popup&&popup.focus){popup.focus()}setStatus(popup?"鍘?HTML 宸插湪鏂扮獥鍙ｆ墦寮€":"娴忚鍣ㄦ嫤鎴簡鏂扮獥鍙?)'
)
PAGE = PAGE.replace(
    '$("pickBtn").textContent="鏀归€夌粨鏋滄枃浠跺す";',
    '$("pickBtn").textContent="淇敼鐩爣鏂囦欢澶?;'
)
PAGE = PAGE.replace(
    'function keyOf(v){if(v===null||v===undefined||v==="")return"";const n=Number(v);if(Number.isNaN(n))return"";return String(Number(n.toFixed(10))).replace(/\\.0+$/,"")}',
    'function keyOf(v){if(v===null||v===undefined||v==="")return"";const n=Number(v);if(Number.isNaN(n))return"";return String(Number(n.toFixed(10))).replace(/\\.0+$/,"")}function formatParamDisplay(value){if(value===null||value===undefined||value==="")return"";const n=Number(value);if(Number.isNaN(n))return String(value);if(Number.isInteger(n))return String(n);return n.toFixed(3)}function formatParamTag(tag){return String(tag??"").replace(/-?\\d+\\.\\d+/g,match=>formatParamDisplay(match))}'
)
PAGE = PAGE.replace(
    '$("metaBatch").textContent=state.summary.batch_label;$("metaTag").textContent=record.param_tag;$("metaCapital").textContent=record.capital??"-";',
    '$("metaBatch").textContent=state.summary.batch_label;$("metaTag").textContent=formatParamTag(record.param_tag);$("metaCapital").textContent=record.capital??"-";'
)
PAGE = PAGE.replace(
    'option.textContent=value;ui.select.appendChild(option)',
    'option.textContent=formatParamDisplay(value);ui.select.appendChild(option)'
)
PAGE = PAGE.replace(
    'function setStatus(text){$("status").innerHTML=text}',
    'function setStatus(text){const node=$("status");if(node){node.innerHTML=text}}'
)
PAGE = PAGE.replace(
    'function clearHtmlPreview(message){$("htmlFrame").removeAttribute("src");$("htmlWrap").classList.remove("active");$("htmlEmpty").style.display="grid";$("htmlEmpty").innerHTML=message}',
    'function clearHtmlPreview(message){const frame=$("htmlFrame");const wrap=$("htmlWrap");const empty=$("htmlEmpty");if(frame){frame.removeAttribute("src")}if(wrap){wrap.classList.remove("active")}if(empty){empty.style.display="grid";empty.innerHTML=message}}'
)
PAGE = PAGE.replace(
    'function showHtmlPreview(url){$("htmlEmpty").style.display="none";$("htmlWrap").classList.add("active");$("htmlFrame").src=url}',
    'function showHtmlPreview(url){const empty=$("htmlEmpty");const wrap=$("htmlWrap");const frame=$("htmlFrame");if(empty){empty.style.display="none"}if(wrap){wrap.classList.add("active")}if(frame){frame.src=url}}'
)
PAGE = PAGE.replace(
    'for(const id of ["inputBar","inputThreshold","inputCont"]){ensureParamUi($(id))}}',
    'for(const id of ["inputBar","inputThreshold","inputCont"]){ensureParamUi($(id))}const statusCard=$("status")?.closest(".chartbox");if(statusCard){statusCard.remove()}const htmlCard=$("htmlEmpty")?.closest(".chartbox")||$("htmlWrap")?.closest(".chartbox");if(htmlCard){htmlCard.remove()}}'
)
PAGE = PAGE.replace(
    'const state={files:[],summaryFiles:[],transFiles:[],perfFiles:[],htmlFiles:[],summary:null,activeRecord:null,folderLabel:"",priceCache:new Map(),transCache:new Map(),htmlUrlCache:new Map(),settingsCollapsed:false};',
    'const state={files:[],summaryFiles:[],transFiles:[],perfFiles:[],htmlFiles:[],summary:null,activeRecord:null,folderLabel:"",priceCache:new Map(),transCache:new Map(),htmlUrlCache:new Map(),settingsCollapsed:false,showCapitalOverlay:true};try{const savedCapitalOverlay=localStorage.getItem("dashboard_show_capital_overlay");if(savedCapitalOverlay!==null){state.showCapitalOverlay=savedCapitalOverlay==="1"}}catch(error){}'
)
PAGE = PAGE.replace(
    'function setStatus(text){const node=$("status");if(node){node.innerHTML=text}}',
    'function setStatus(text){const node=$("status");if(node){node.innerHTML=text}}function syncCapitalOverlayToggle(){const node=$("toggleCapitalOverlay");if(node){node.checked=!!state.showCapitalOverlay}}function saveCapitalOverlay(value){state.showCapitalOverlay=!!value;syncCapitalOverlayToggle();try{localStorage.setItem("dashboard_show_capital_overlay",state.showCapitalOverlay?"1":"0")}catch(error){}}'
)
PAGE = PAGE.replace(
    '#overviewChart.chart{min-height:510px}',
    '#overviewChart.chart{min-height:510px}.controls-grid{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}.capital-toggle{display:flex;align-items:center;gap:8px;height:42px;padding:0 12px;border:1px solid #d9e1ee;border-radius:12px;background:#fff;color:#152033;font-size:13px;box-sizing:border-box}.capital-toggle input{margin:0}'
)
PAGE = PAGE.replace(
    'if(capitalPoints.x.length)traces.unshift({type:"scatter",mode:"lines",x:capitalPoints.x,y:capitalPoints.y,text:capitalPoints.text,line:{color:ACCENT_BLUE,width:1.2,shape:"hv"},name:"capital",yaxis:"y2",hovertemplate:"%{text}<br>capital=%{y}<extra></extra>"});',
    'if(state.showCapitalOverlay&&capitalPoints.x.length)traces.unshift({type:"scatter",mode:"lines",x:capitalPoints.x,y:capitalPoints.y,text:capitalPoints.text,line:{color:ACCENT_BLUE,width:1.2,shape:"hv"},name:"capital",yaxis:"y2",hovertemplate:"%{text}<br>capital=%{y}<extra></extra>"});'
)
PAGE = PAGE.replace(
    'yaxis2:{title:"capital",overlaying:"y",side:"right",showgrid:false,zeroline:false}',
    'yaxis2:{title:"capital",overlaying:"y",side:"right",showgrid:false,zeroline:false,visible:!!state.showCapitalOverlay}'
)
PAGE = PAGE.replace(
    'function capitalChart(trans){if(!trans||!trans.capital_x.length){$("capitalChart").style.display="none";$("capitalEmpty").style.display="grid";$("capitalEmpty").innerHTML="当前参数还没有可用的 trans.xlsx銆?;return}$("capitalEmpty").style.display="none";$("capitalChart").style.display="block";const axisX=trans.capital_x.map((_,index)=>index);Plotly.newPlot($("capitalChart"),[{x:axisX,y:trans.capital_y,type:"scatter",mode:"lines",text:trans.capital_x,line:{color:ACCENT_BLUE,width:1.2,shape:"hv"},hovertemplate:"%{text}<br>capital=%{y}<extra></extra>"}],{margin:{l:48,r:18,t:12,b:36},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:numericAxis(trans.capital_x),yaxis:{title:"capital",gridcolor:"#e8eef8",zeroline:false},showlegend:false},{responsive:true,displayModeBar:false})}',
    'function capitalChart(trans){const card=$("capitalChart")?.closest(".chartbox")||$("capitalEmpty")?.closest(".chartbox");if(card){card.style.display="none"}}'
)
PAGE = PAGE.replace(
    'resetUi=function(){legacyResetUi();configInput($("inputBar"),null);configInput($("inputThreshold"),null);configInput($("inputCont"),null)}\nsetupLayout();\nsetSettingsCollapsed(false);\nbootstrapPreset();',
    'resetUi=function(){legacyResetUi();configInput($("inputBar"),null);configInput($("inputThreshold"),null);configInput($("inputCont"),null);syncCapitalOverlayToggle()}\nconst baseSetupLayout=setupLayout;setupLayout=function(){baseSetupLayout();const controlsGrid=document.querySelector("#pageDetail .controls-grid");const htmlCell=$("openHtml")?.parentElement;if(controlsGrid&&!$("toggleCapitalOverlay")){const block=document.createElement("div");block.id="capitalOverlayBlock";block.innerHTML=\'<div class="label">资金曲线</div><label class="capital-toggle"><input id="toggleCapitalOverlay" type="checkbox"><span>主图显示</span></label>\';if(htmlCell){controlsGrid.insertBefore(block,htmlCell)}else{controlsGrid.appendChild(block)}}const toggle=$("toggleCapitalOverlay");if(toggle&&!toggle.dataset.bound){toggle.dataset.bound="1";toggle.addEventListener("change",()=>{saveCapitalOverlay(toggle.checked);if(state.activeRecord){setRecord(state.activeRecord,false)}})}syncCapitalOverlayToggle();const capitalCard=$("capitalChart")?.closest(".chartbox")||$("capitalEmpty")?.closest(".chartbox");if(capitalCard){capitalCard.style.display="none"}}\nsetupLayout();\nsetSettingsCollapsed(false);\nbootstrapPreset();'
)


PAGE = PAGE.replace(
    "</style>",
    """
.drawer-toggle{position:relative;font-size:0;color:transparent}
.drawer-toggle::before{content:"\\25C0";position:absolute;inset:0;display:grid;place-items:center;color:rgba(21,32,51,.7);font-size:14px;line-height:1}
.drawer.collapsed .drawer-toggle::before{content:"\\25B6"}
.drawer.collapsed .drawer-toggle{display:grid;place-items:center}
.controls-card{width:min(1280px,100%);max-width:1280px}
.price-actions{display:flex;align-items:center;justify-content:flex-end;gap:12px;flex-wrap:wrap;margin-top:12px}
.price-actions #openHtml{min-width:160px}
.price-actions #capitalOverlayBlock{min-width:220px}
</style>"""
)
PAGE = PAGE.replace(
    'setupLayout();\nsetSettingsCollapsed(false);\nbootstrapPreset();',
    '''function formatMetric3(value){if(value===null||value===undefined||value==="")return"-";const n=Number(value);if(Number.isNaN(n))return String(value);return n.toFixed(3)}
function currentPriceTitle(){const product=String(state.folderLabel||"").replace(/\\s+long_momentum(?:_atr|_ratio)? outcome$/i,"").trim();const match=String(state.summary?.batch_label||"").match(/\\d{8}-\\d{8}/);const dateText=match?match[0]:"";return[product,dateText].filter(Boolean).join(" ")||"品种 鏃ユ湡"}
function updatePriceHeader(){const title=document.querySelector("#pageDetail .price-box .charttitle");const sub=document.querySelector("#pageDetail .price-box .chartsub");if(sub){sub.remove()}if(title){title.textContent=currentPriceTitle()}}
const wrappedMeta=meta;meta=function(record){wrappedMeta(record);if(!record)return;$("metaCapital").textContent=formatMetric3(record.capital);$("metaWd").textContent=formatMetric3(record.biggest_wd);$("metaHigh").textContent=formatMetric3(record.outcome_high)}
const wrappedFolderChanged=folderChanged;folderChanged=function(files,folderLabel){const result=wrappedFolderChanged(files,folderLabel);updatePriceHeader();return result}
const wrappedLoadSummary=loadSummary;loadSummary=async function(file){const result=await wrappedLoadSummary(file);updatePriceHeader();return result}
const wrappedResetUi=resetUi;resetUi=function(){wrappedResetUi();updatePriceHeader()}
const wrappedSetupLayout=setupLayout;setupLayout=function(){wrappedSetupLayout();updatePriceHeader();const priceBox=document.querySelector("#pageDetail .price-box");const chartNode=$("priceChart");if(priceBox&&chartNode){let actions=$("priceActions");if(!actions){actions=document.createElement("div");actions.id="priceActions";actions.className="price-actions";chartNode.insertAdjacentElement("afterend",actions)}const toggleBlock=$("capitalOverlayBlock");if(toggleBlock&&toggleBlock.parentElement!==actions){actions.appendChild(toggleBlock)}const openBtn=$("openHtml");const oldHtmlCell=openBtn?.parentElement;if(openBtn&&openBtn.parentElement!==actions){actions.appendChild(openBtn)}if(oldHtmlCell&&oldHtmlCell!==actions&&oldHtmlCell.childElementCount===0){oldHtmlCell.remove()}const capitalCard=$("capitalChart")?.closest(".chartbox")||$("capitalEmpty")?.closest(".chartbox");if(capitalCard){capitalCard.style.display="none"}}}
setupLayout();
setSettingsCollapsed(false);
bootstrapPreset();'''
)
PAGE = PAGE.replace(
    'if(oldHtmlCell&&oldHtmlCell!==actions&&oldHtmlCell.childElementCount===0){oldHtmlCell.remove()}',
    'if(oldHtmlCell&&oldHtmlCell!==actions&&oldHtmlCell.parentElement?.classList.contains("controls-grid")){oldHtmlCell.remove()}'
)
PAGE = PAGE.replace(
    'const candleX=price.x.map((_,index)=>index);const priceIndexMap=buildIndexMap(price.x);const shapes=gapShapes(price.x);const traces=[{type:"candlestick"',
    'const candleX=price.x.map((_,index)=>index);const priceIndexMap=buildIndexMap(price.x);const shapes=gapShapes(price.x);const perfCapitalPoints=axisPoints(price.capital_x||[],price.capital_y||[],priceIndexMap);const traces=[{type:"candlestick"'
)
PAGE = PAGE.replace(
    'if(trans){const capitalPoints=axisPoints(trans.capital_x,trans.capital_y,priceIndexMap);const tradeLink=',
    'const capitalPoints=perfCapitalPoints.x.length?perfCapitalPoints:(trans?axisPoints(trans.capital_x,trans.capital_y,priceIndexMap):{x:[],y:[],text:[]});if(state.showCapitalOverlay&&capitalPoints.x.length)traces.unshift({type:"scatter",mode:"lines",x:capitalPoints.x,y:capitalPoints.y,text:capitalPoints.text,line:{color:ACCENT_BLUE,width:1.2},name:"capital",yaxis:"y2",hovertemplate:"%{text}<br>capital=%{y}<extra></extra>"});if(trans){const tradeLink='
)
PAGE = PAGE.replace(
    'if(state.showCapitalOverlay&&capitalPoints.x.length)traces.unshift({type:"scatter",mode:"lines",x:capitalPoints.x,y:capitalPoints.y,text:capitalPoints.text,line:{color:ACCENT_BLUE,width:1.2,shape:"hv"},name:"capital",yaxis:"y2",hovertemplate:"%{text}<br>capital=%{y}<extra></extra>"});',
    ''
)

PAGE = PAGE.replace(
    '<div id="overviewChart" class="chart" style="display:none"></div>\n      </div>',
    '<div id="overviewChart" class="chart" style="display:none"></div>\n      </div>\n      <div class="chartbox" style="margin-top:16px">\n        <div class="charttitle">鍙傛暟涓夌淮鍥?/div>\n        <div id="overview3dEmpty" class="empty">璇烽€夋嫨涓€涓洖娴嬫壒娆°€?/div>\n        <div id="overview3dChart" class="chart" style="display:none"></div>\n      </div>'
)
PAGE = PAGE.replace(
    "</style>",
    """
#overview3dChart.chart{min-height:620px}
</style>"""
)
PAGE = PAGE.replace(
    'setupLayout();\nsetSettingsCollapsed(false);\nbootstrapPreset();',
    '''function setOverview3dEmpty(message){const empty=$("overview3dEmpty");const chart=$("overview3dChart");if(chart){chart.style.display="none"}if(empty){empty.style.display="grid";empty.innerHTML=message}}
function renderOverview3d(){const chart=$("overview3dChart");const empty=$("overview3dEmpty");if(!chart||!empty){return}if(!state.summary||!state.summary.records.length){setOverview3dEmpty("璇烽€夋嫨涓€涓洖娴嬫壒娆°€?);return}const rows=state.summary.records.filter(row=>row.capital!==null&&row.capital!==undefined&&row.open_bar!==null&&row.open_bar!==undefined);if(!rows.length){setOverview3dEmpty("褰撳墠鎵规缂哄皯鍙敤浜庝笁缁村浘鐨勫弬鏁版暟鎹€?);return}const order=rows.map((_,index)=>index+1);empty.style.display="none";chart.style.display="block";const text=rows.map((row,index)=>["椤哄簭: "+(index+1),"param_tag: "+esc(row.param_tag),"鏃堕棿绐楀彛: "+formatParamDisplay(row.open_bar),"閫熷害闄愬埗: "+formatParamDisplay(row.open_threshold),"寮€浠撻棬妲? "+formatParamDisplay(row.open_continous_threshold),"capital: "+formatMetric3(row.capital),"trade_num: "+(row.trade_num??"-")].join("<br>"));Plotly.newPlot(chart,[{type:"scatter3d",mode:"lines+markers",x:order,y:rows.map(row=>row.capital),z:rows.map(row=>row.open_bar),customdata:rows.map(row=>row.selection_key),text:text,hovertemplate:"%{text}<extra></extra>",line:{color:"#2f6bff",width:3},marker:{size:5,color:rows.map(row=>row.capital),colorscale:"Viridis",opacity:0.95,colorbar:{title:"capital"}}}],{margin:{l:0,r:0,t:10,b:0},paper_bgcolor:"rgba(0,0,0,0)",scene:{xaxis:{title:"参数组合顺序",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},yaxis:{title:"capital",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},zaxis:{title:"鏃堕棿绐楀彛",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"}},showlegend:false},{responsive:true,displayModeBar:false});if(chart.removeAllListeners){chart.removeAllListeners("plotly_click")}chart.on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record){setRecord(record,true)}})}
const overviewBase3d=overview;overview=function(){overviewBase3d();renderOverview3d()}
const resetUiBase3d=resetUi;resetUi=function(){resetUiBase3d();setOverview3dEmpty("璇烽€夋嫨涓€涓甫鏈?outcome_stats.xlsx 鐨勭粨鏋滃ぇ鏂囦欢澶广€?)}
const folderChangedBase3d=folderChanged;folderChanged=function(files,folderLabel){const result=folderChangedBase3d(files,folderLabel);if(state.summaryFiles?.length){setOverview3dEmpty("璇烽€夋嫨涓€涓洖娴嬫壒娆°€?)}else{setOverview3dEmpty("褰撳墠鐩綍閲屾病鏈?outcome_stats.xlsx銆?br>璇蜂娇鐢ㄦ洿鏂板悗鐨勫洖娴嬬▼搴忛噸鏂拌緭鍑?outcome stats 姹囨€绘枃浠躲€?)}return result}
setupLayout();
setSettingsCollapsed(false);
bootstrapPreset();'''
)
PAGE = PAGE.replace(
    'async function ensurePrice(){const label=state.summary?.batch_label;if(!label)return null;if(state.priceCache.has(label))return state.priceCache.get(label);const file=batchFile(state.perfFiles,label,"perf.xlsx");if(!file){state.priceCache.set(label,null);return null}setStatus("正在读取公共行情数据");const data=file.relative_path?await fetchJson("/api/preset-price?path="+encodeURIComponent(file.relative_path)):await upload("/api/price",file);state.priceCache.set(label,data);return data}',
    'async function ensurePrice(record){if(!record||!state.summary)return null;const cacheKey=record.selection_key||state.summary.batch_label;if(state.priceCache.has(cacheKey))return state.priceCache.get(cacheKey);const file=detailFile(state.perfFiles,state.summary.batch_label,record.param_tag,"perf.xlsx")||batchFile(state.perfFiles,state.summary.batch_label,"perf.xlsx");if(!file){state.priceCache.set(cacheKey,null);return null}setStatus("正在读取 perf.xlsx");const data=file.relative_path?await fetchJson("/api/preset-price?path="+encodeURIComponent(file.relative_path)):await upload("/api/price",file);state.priceCache.set(cacheKey,data);return data}'
)
PAGE = PAGE.replace(
    'const [price,trans]=await Promise.all([ensurePrice(),ensureTrans(record)]);',
    'const [price,trans]=await Promise.all([ensurePrice(record),ensureTrans(record)]);'
)
PAGE = PAGE.replace(
    'function overview(){if(!state.summary||!state.summary.records.length){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";return}const rows=state.summary.records;$("overviewEmpty").style.display="none";$("overviewChart").style.display="block";Plotly.newPlot($("overviewChart"),[{x:rows.map((_,i)=>i+1),y:rows.map(r=>r.capital),mode:"lines+markers",line:{color:"#2f6bff",width:2},marker:{size:6,color:"#2f6bff"},customdata:rows.map(r=>r.selection_key),text:rows.map(r=>["param_tag: "+esc(r.param_tag),"open_bar: "+(r.open_bar??"-"),"open_threshold: "+(r.open_threshold??"-"),"open_cont: "+(r.open_continous_threshold??"-"),"capital: "+(r.capital??"-"),"trade_num: "+(r.trade_num??"-")].join("<br>")),hovertemplate:"%{text}<extra></extra>",name:"capital"}],{margin:{l:44,r:18,t:16,b:40},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:{title:"参数组合顺序",showgrid:false,zeroline:false},yaxis:{title:"capital",gridcolor:"#e8eef8",zeroline:false},showlegend:false},{responsive:true,displayModeBar:false});if($("overviewChart").removeAllListeners){$("overviewChart").removeAllListeners("plotly_click")}$("overviewChart").on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record)setRecord(record,true)})}',
    'function overview(){if(!state.summary||!state.summary.records.length){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";return}const numOrInf=value=>{const n=Number(value);return Number.isFinite(n)?n:Infinity};const rows=[...state.summary.records].sort((a,b)=>numOrInf(a.open_bar)-numOrInf(b.open_bar)||numOrInf(a.open_threshold)-numOrInf(b.open_threshold)||numOrInf(a.open_continous_threshold)-numOrInf(b.open_continous_threshold)||String(a.param_tag||"").localeCompare(String(b.param_tag||"")));const groupTicks=[];const separators=[];let groupStart=0;for(let i=1;i<=rows.length;i+=1){const sameGroup=i<rows.length&&keyOf(rows[i]?.open_bar)===keyOf(rows[groupStart]?.open_bar);if(sameGroup)continue;const left=groupStart+1;const right=i;groupTicks.push({value:(left+right)/2,label:formatParamDisplay(rows[groupStart]?.open_bar)});if(i<rows.length){separators.push({type:"line",xref:"x",yref:"paper",x0:i+0.5,x1:i+0.5,y0:0,y1:1,line:{color:"rgba(120,120,120,0.2)",width:1,dash:"dash"}})}groupStart=i}$("overviewEmpty").style.display="none";$("overviewChart").style.display="block";Plotly.newPlot($("overviewChart"),[{x:rows.map((_,i)=>i+1),y:rows.map(r=>r.capital),mode:"lines+markers",line:{color:"#2f6bff",width:2},marker:{size:6,color:"#2f6bff"},customdata:rows.map(r=>r.selection_key),text:rows.map(r=>["param_tag: "+esc(r.param_tag),"open_bar: "+formatParamDisplay(r.open_bar),"open_threshold: "+formatParamDisplay(r.open_threshold),"open_cont: "+formatParamDisplay(r.open_continous_threshold),"capital: "+(r.capital??"-"),"trade_num: "+(r.trade_num??"-")].join("<br>")),hovertemplate:"%{text}<extra></extra>",name:"capital"}],{margin:{l:44,r:18,t:16,b:40},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:{title:"om",showgrid:false,zeroline:false,tickmode:"array",tickvals:groupTicks.map(item=>item.value),ticktext:groupTicks.map(item=>item.label)},yaxis:{title:"capital",gridcolor:"#e8eef8",zeroline:false},shapes:separators,showlegend:false},{responsive:true,displayModeBar:false});if($("overviewChart").removeAllListeners){$("overviewChart").removeAllListeners("plotly_click")}$("overviewChart").on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record)setRecord(record,true)})}'
)
PAGE = PAGE.replace(
    "</style>",
    """
.price-actions{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-top:12px}
.price-actions-left,.price-actions-right{display:flex;align-items:flex-end;gap:12px;flex-wrap:wrap}
.price-actions-right{margin-left:auto;justify-content:flex-end}
.price-actions #capitalOverlayBlock{display:flex;flex-direction:column;justify-content:flex-end;min-width:220px}
.price-actions #capitalOverlayBlock .label{margin-bottom:8px}
.price-actions #openHtml,.price-actions #backOverviewBtn{min-width:160px}
</style>"""
)
PAGE = PAGE.replace(
    'mode:"lines+markers",line:{color:"#2f6bff",width:2},marker:{size:6,color:"#2f6bff"}',
    'mode:"markers",marker:{size:6,color:"#2f6bff"}'
)
PAGE = PAGE.replace(
    'line:{color:"rgba(120,120,120,0.2)",width:1,dash:"dash"}',
    'line:{color:"rgba(120,120,120,0.1)",width:1}'
)
PAGE = PAGE.replace(
    'yaxis:{title:"capital",gridcolor:"#e8eef8",zeroline:false}',
    'yaxis:{title:"capital",showgrid:false,zeroline:false}'
)
PAGE = PAGE.replace(
    'type:"scatter3d",mode:"lines+markers",x:order,y:rows.map(row=>row.capital),z:rows.map(row=>row.open_bar),customdata:rows.map(row=>row.selection_key),text:text,hovertemplate:"%{text}<extra></extra>",line:{color:"#2f6bff",width:3},marker:{size:5,color:rows.map(row=>row.capital),colorscale:"Viridis",opacity:0.95,colorbar:{title:"capital"}}',
    'type:"scatter3d",mode:"markers",x:order,y:rows.map(row=>row.capital),z:rows.map(row=>row.open_bar),customdata:rows.map(row=>row.selection_key),text:text,hovertemplate:"%{text}<extra></extra>",marker:{size:5,color:rows.map(row=>row.capital),colorscale:"Viridis",opacity:0.95,colorbar:{title:"capital"}}'
)
PAGE = PAGE.replace(
    'setupLayout();\nsetSettingsCollapsed(false);\nbootstrapPreset();',
    '''const detailActionsSetup=setupLayout;setupLayout=function(){detailActionsSetup();const actions=$("priceActions");if(!actions){return}let left=$("priceActionsLeft");if(!left){left=document.createElement("div");left.id="priceActionsLeft";left.className="price-actions-left";actions.prepend(left)}let right=$("priceActionsRight");if(!right){right=document.createElement("div");right.id="priceActionsRight";right.className="price-actions-right";actions.appendChild(right)}let backBtn=$("backOverviewBtn");if(!backBtn){backBtn=document.createElement("button");backBtn.id="backOverviewBtn";backBtn.className="btn";backBtn.textContent="Back"}if(!backBtn.dataset.bound){backBtn.dataset.bound="1";backBtn.addEventListener("click",()=>{setTab("overview");window.scrollTo({top:0,behavior:"smooth"})})}if(backBtn.parentElement!==left){left.appendChild(backBtn)}const toggleBlock=$("capitalOverlayBlock");if(toggleBlock&&toggleBlock.parentElement!==right){right.appendChild(toggleBlock)}const openBtn=$("openHtml");if(openBtn&&openBtn.parentElement!==right){right.appendChild(openBtn)}}\nsetupLayout();\nsetSettingsCollapsed(false);\nbootstrapPreset();'''
)
PAGE = PAGE.replace(
    'function renderOverview3d(){const chart=$("overview3dChart");const empty=$("overview3dEmpty");if(!chart||!empty){return}if(!state.summary||!state.summary.records.length){setOverview3dEmpty("璇烽€夋嫨涓€涓洖娴嬫壒娆°€?);return}const rows=state.summary.records.filter(row=>row.capital!==null&&row.capital!==undefined&&row.open_bar!==null&&row.open_bar!==undefined);if(!rows.length){setOverview3dEmpty("褰撳墠鎵规缂哄皯鍙敤浜庝笁缁村浘鐨勫弬鏁版暟鎹€?);return}const order=rows.map((_,index)=>index+1);empty.style.display="none";chart.style.display="block";const text=rows.map((row,index)=>["椤哄簭: "+(index+1),"param_tag: "+esc(row.param_tag),"鏃堕棿绐楀彛: "+formatParamDisplay(row.open_bar),"閫熷害闄愬埗: "+formatParamDisplay(row.open_threshold),"寮€浠撻棬妲? "+formatParamDisplay(row.open_continous_threshold),"capital: "+formatMetric3(row.capital),"trade_num: "+(row.trade_num??"-")].join("<br>"));Plotly.newPlot(chart,[{type:"scatter3d",mode:"markers",x:order,y:rows.map(row=>row.capital),z:rows.map(row=>row.open_bar),customdata:rows.map(row=>row.selection_key),text:text,hovertemplate:"%{text}<extra></extra>",marker:{size:5,color:rows.map(row=>row.capital),colorscale:"Viridis",opacity:0.95,colorbar:{title:"capital"}}}],{margin:{l:0,r:0,t:10,b:0},paper_bgcolor:"rgba(0,0,0,0)",scene:{xaxis:{title:"参数组合顺序",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},yaxis:{title:"capital",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},zaxis:{title:"鏃堕棿绐楀彛",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"}},showlegend:false},{responsive:true,displayModeBar:false});if(chart.removeAllListeners){chart.removeAllListeners("plotly_click")}chart.on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record){setRecord(record,true)}})}',
    'function renderOverview3d(){const chart=$("overview3dChart");const empty=$("overview3dEmpty");if(!chart||!empty){return}if(!state.summary||!state.summary.records.length){setOverview3dEmpty("璇烽€夋嫨涓€涓洖娴嬫壒娆°€?);return}const rows=state.summary.records.filter(row=>{const capital=Number(row?.capital);return row.capital!==null&&row.capital!==undefined&&row.open_bar!==null&&row.open_bar!==undefined&&row.open_threshold!==null&&row.open_threshold!==undefined&&(!Number.isFinite(capital)||Math.abs(capital-100)>1e-9)});if(!rows.length){setOverview3dEmpty("褰撳墠鎵规缂哄皯鍙敤浜庝笁缁村浘鐨勫弬鏁版暟鎹€?);return}empty.style.display="none";chart.style.display="block";const text=rows.map(row=>["param_tag: "+esc(row.param_tag),"鏃堕棿绐楀彛: "+formatParamDisplay(row.open_bar),"閫熷害闄愬埗: "+formatParamDisplay(row.open_threshold),"寮€浠撻棬妲? "+formatParamDisplay(row.open_continous_threshold),"capital: "+formatMetric3(row.capital),"trade_num: "+(row.trade_num??"-")].join("<br>"));Plotly.newPlot(chart,[{type:"scatter3d",mode:"markers",x:rows.map(row=>row.open_threshold),y:rows.map(row=>row.capital),z:rows.map(row=>row.open_bar),customdata:rows.map(row=>row.selection_key),text:text,hovertemplate:"%{text}<extra></extra>",marker:{size:5,color:rows.map(row=>row.capital),colorscale:"Viridis",opacity:0.95,colorbar:{title:"capital"}}}],{margin:{l:0,r:0,t:10,b:0},paper_bgcolor:"rgba(0,0,0,0)",scene:{xaxis:{title:"閫熷害闄愬埗",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},yaxis:{title:"capital",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},zaxis:{title:"鏃堕棿绐楀彛",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"}},showlegend:false},{responsive:true,displayModeBar:false});if(chart.removeAllListeners){chart.removeAllListeners("plotly_click")}chart.on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record){setRecord(record,true)}})}'
)


PAGE = PAGE.replace(
    'const text=rows.map(row=>["param_tag: "+esc(row.param_tag),"鏃堕棿绐楀彛: "+formatParamDisplay(row.open_bar),"閫熷害闄愬埗: "+formatParamDisplay(row.open_threshold),"寮€浠撻棬妲? "+formatParamDisplay(row.open_continous_threshold),"capital: "+formatMetric3(row.capital),"trade_num: "+(row.trade_num??"-")].join("<br>"));Plotly.newPlot(chart,[{type:"scatter3d",mode:"markers",x:rows.map(row=>row.open_threshold),y:rows.map(row=>row.capital),z:rows.map(row=>row.open_bar),customdata:rows.map(row=>row.selection_key),text:text,hovertemplate:"%{text}<extra></extra>",marker:{size:5,color:rows.map(row=>row.capital),colorscale:"Viridis",opacity:0.95,colorbar:{title:"capital"}}}],{margin:{l:0,r:0,t:10,b:0},paper_bgcolor:"rgba(0,0,0,0)",scene:{xaxis:{title:"閫熷害闄愬埗",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},yaxis:{title:"capital",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},zaxis:{title:"鏃堕棿绐楀彛",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"}},showlegend:false},{responsive:true,displayModeBar:false});',
    'const xValues=rows.map(row=>Number(row.open_threshold)).filter(Number.isFinite);const zValues=rows.map(row=>Number(row.capital)).filter(Number.isFinite);const timeWindows=[...new Set(rows.map(row=>Number(row.open_bar)).filter(Number.isFinite))].sort((a,b)=>a-b);const separatorYs=[];for(let index=1;index<timeWindows.length;index+=1){separatorYs.push((timeWindows[index-1]+timeWindows[index])/2)}const xMin=xValues.length?Math.min(...xValues):0;const xMax=xValues.length?Math.max(...xValues):1;const zMin=zValues.length?Math.min(...zValues):0;const zMax=zValues.length?Math.max(...zValues):1;const text=rows.map(row=>["param_tag: "+esc(row.param_tag),"鏃堕棿绐楀彛: "+formatParamDisplay(row.open_bar),"閫熷害闄愬埗: "+formatParamDisplay(row.open_threshold),"寮€浠撻棬妲? "+formatParamDisplay(row.open_continous_threshold),"capital: "+formatMetric3(row.capital),"trade_num: "+(row.trade_num??"-")].join("<br>"));const planeTraces=separatorYs.map(value=>({type:"surface",x:[[xMin,xMax],[xMin,xMax]],y:[[value,value],[value,value]],z:[[zMin,zMin],[zMax,zMax]],surfacecolor:[[0,0],[0,0]],colorscale:[[0,"rgba(120,120,120,0.1)"],[1,"rgba(120,120,120,0.1)"]],showscale:false,opacity:0.1,hoverinfo:"skip"}));Plotly.newPlot(chart,[...planeTraces,{type:"scatter3d",mode:"markers",x:rows.map(row=>row.open_threshold),y:rows.map(row=>row.open_bar),z:rows.map(row=>row.capital),customdata:rows.map(row=>row.selection_key),text:text,hovertemplate:"%{text}<extra></extra>",marker:{size:5,color:rows.map(row=>row.capital),colorscale:"Viridis",opacity:0.95,colorbar:{title:"capital"}}}],{margin:{l:0,r:0,t:10,b:0},paper_bgcolor:"rgba(0,0,0,0)",scene:{xaxis:{title:"閫熷害闄愬埗",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},yaxis:{title:"鏃堕棿绐楀彛",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},zaxis:{title:"capital",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"}},showlegend:false},{responsive:true,displayModeBar:false});'
)

PAGE = PAGE.replace(
    'colorscale:[[0,"rgba(120,120,120,0.1)"],[1,"rgba(120,120,120,0.1)"]],showscale:false,opacity:0.1,hoverinfo:"skip"',
    'colorscale:[[0,"rgb(120,120,120)"],[1,"rgb(120,120,120)"]],showscale:false,opacity:0.18,hoverinfo:"skip"'
)
PAGE = PAGE.replace(
    'const xValues=rows.map(row=>Number(row.open_threshold)).filter(Number.isFinite);const zValues=rows.map(row=>Number(row.capital)).filter(Number.isFinite);const timeWindows=[...new Set(rows.map(row=>Number(row.open_bar)).filter(Number.isFinite))].sort((a,b)=>a-b);const separatorYs=[];for(let index=1;index<timeWindows.length;index+=1){separatorYs.push((timeWindows[index-1]+timeWindows[index])/2)}const xMin=xValues.length?Math.min(...xValues):0;const xMax=xValues.length?Math.max(...xValues):1;const zMin=zValues.length?Math.min(...zValues):0;const zMax=zValues.length?Math.max(...zValues):1;',
    'const xValues=[...new Set(rows.map(row=>Number(row.open_bar)).filter(Number.isFinite))].sort((a,b)=>a-b);const yValues=rows.map(row=>Number(row.open_threshold)).filter(Number.isFinite);const zValues=rows.map(row=>Number(row.capital)).filter(Number.isFinite);const separatorXs=[];for(let index=1;index<xValues.length;index+=1){separatorXs.push((xValues[index-1]+xValues[index])/2)}const yMin=yValues.length?Math.min(...yValues):0;const yMax=yValues.length?Math.max(...yValues):1;const zMin=zValues.length?Math.min(...zValues):0;const zMax=zValues.length?Math.max(...zValues):1;'
)
PAGE = PAGE.replace(
    'const planeTraces=separatorYs.map(value=>({type:"surface",x:[[xMin,xMax],[xMin,xMax]],y:[[value,value],[value,value]],z:[[zMin,zMin],[zMax,zMax]],surfacecolor:[[0,0],[0,0]],colorscale:[[0,"rgb(120,120,120)"],[1,"rgb(120,120,120)"]],showscale:false,opacity:0.18,hoverinfo:"skip"}));',
    'const planeTraces=separatorXs.map(value=>({type:"surface",x:[[value,value],[value,value]],y:[[yMin,yMax],[yMin,yMax]],z:[[zMin,zMin],[zMax,zMax]],surfacecolor:[[0,0],[0,0]],colorscale:[[0,"rgb(120,120,120)"],[1,"rgb(120,120,120)"]],showscale:false,opacity:0.18,hoverinfo:"skip"}));'
)
PAGE = PAGE.replace(
    'x:rows.map(row=>row.open_threshold),y:rows.map(row=>row.open_bar),z:rows.map(row=>row.capital)',
    'x:rows.map(row=>row.open_bar),y:rows.map(row=>row.open_threshold),z:rows.map(row=>row.capital)'
)
PAGE = PAGE.replace(
    'scene:{xaxis:{title:"閫熷害闄愬埗",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},yaxis:{title:"鏃堕棿绐楀彛",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},zaxis:{title:"capital",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"}}',
    'scene:{xaxis:{title:"鏃堕棿绐楀彛",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},yaxis:{title:"閫熷害闄愬埗",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},zaxis:{title:"capital",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"}}'
)

PAGE = PAGE.replace(
    '#overviewChart.chart{min-height:510px}.controls-grid{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}.capital-toggle{display:flex;align-items:center;gap:8px;height:42px;padding:0 12px;border:1px solid #d9e1ee;border-radius:12px;background:#fff;color:#152033;font-size:13px;box-sizing:border-box}.capital-toggle input{margin:0}',
    '#overviewChart.chart{min-height:510px}.controls-card{width:min(920px,100%);max-width:920px}.controls-grid{grid-template-columns:repeat(auto-fit,minmax(0,250px));justify-content:start}.capital-toggle{display:flex;align-items:center;gap:8px;height:42px;padding:0 12px;border:1px solid #d9e1ee;border-radius:12px;background:#fff;color:#152033;font-size:13px;box-sizing:border-box}.capital-toggle input{margin:0}'
)
PAGE = PAGE.replace(
    '.price-actions-left,.price-actions-right{display:flex;align-items:flex-end;gap:12px;flex-wrap:wrap}',
    '.price-actions-left,.price-actions-right{display:flex;gap:12px;flex-wrap:wrap}.price-actions-left{align-items:center;align-self:center}.price-actions-right{align-items:flex-end}'
)
PAGE = PAGE.replace(
    '.price-actions #openHtml,.price-actions #backOverviewBtn{min-width:160px}',
    '.price-actions #openHtml{min-width:160px}.price-actions #backOverviewBtn{min-width:112px;display:inline-flex;align-items:center;justify-content:center;padding:0 16px}'
)
PAGE = PAGE.replace(
    '<div class="charttitle">收益总览</div>\n        <div id="overviewEmpty" class="empty">璇烽€夋嫨涓€涓甫鏈?outcome_stats.xlsx 鐨勭粨鏋滃ぇ鏂囦欢澶广€?/div>',
    '<div class="overview-head"><div class="charttitle">收益总览</div><div class="overview-sort-wrap"><span class="overview-sort-label">鎺掑簭</span><select id="overviewSortMode" class="sel overview-sort-select"><option value="current">当前序号</option><option value="capital">收益排序</option></select></div></div>\n        <div id="overviewEmpty" class="empty">璇烽€夋嫨涓€涓甫鏈?outcome_stats.xlsx 鐨勭粨鏋滃ぇ鏂囦欢澶广€?/div>'
)
PAGE = PAGE.replace(
    '.price-actions #openHtml{min-width:160px}.price-actions #backOverviewBtn{min-width:112px;display:inline-flex;align-items:center;justify-content:center;padding:0 16px}',
    '.price-actions #openHtml{min-width:160px}.price-actions #backOverviewBtn{min-width:112px;display:inline-flex;align-items:center;justify-content:center;padding:0 16px}.overview-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:10px}.overview-head .charttitle{margin-bottom:0;text-align:left}.overview-sort-wrap{display:flex;align-items:center;gap:8px}.overview-sort-label{font-size:12px;color:#66758c;font-weight:600}.overview-sort-select{min-width:148px}'
)
PAGE = PAGE.replace(
    'function overview(){if(!state.summary||!state.summary.records.length){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";return}const numOrInf=value=>{const n=Number(value);return Number.isFinite(n)?n:Infinity};const rows=[...state.summary.records].sort((a,b)=>numOrInf(a.open_bar)-numOrInf(b.open_bar)||numOrInf(a.open_threshold)-numOrInf(b.open_threshold)||numOrInf(a.open_continous_threshold)-numOrInf(b.open_continous_threshold)||String(a.param_tag||"").localeCompare(String(b.param_tag||"")));const groupTicks=[];const separators=[];let groupStart=0;for(let i=1;i<=rows.length;i+=1){const sameGroup=i<rows.length&&keyOf(rows[i]?.open_bar)===keyOf(rows[groupStart]?.open_bar);if(sameGroup)continue;const left=groupStart+1;const right=i;groupTicks.push({value:(left+right)/2,label:formatParamDisplay(rows[groupStart]?.open_bar)});if(i<rows.length){separators.push({type:"line",xref:"x",yref:"paper",x0:i+0.5,x1:i+0.5,y0:0,y1:1,line:{color:"rgba(120,120,120,0.1)",width:1}})}groupStart=i}$("overviewEmpty").style.display="none";$("overviewChart").style.display="block";Plotly.newPlot($("overviewChart"),[{x:rows.map((_,i)=>i+1),y:rows.map(r=>r.capital),mode:"markers",marker:{size:6,color:"#2f6bff"},customdata:rows.map(r=>r.selection_key),text:rows.map(r=>["param_tag: "+esc(r.param_tag),"open_bar: "+formatParamDisplay(r.open_bar),"open_threshold: "+formatParamDisplay(r.open_threshold),"open_cont: "+formatParamDisplay(r.open_continous_threshold),"capital: "+(r.capital??"-"),"trade_num: "+(r.trade_num??"-")].join("<br>")),hovertemplate:"%{text}<extra></extra>",name:"capital"}],{margin:{l:44,r:18,t:16,b:40},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:{title:"om",showgrid:false,zeroline:false,tickmode:"array",tickvals:groupTicks.map(item=>item.value),ticktext:groupTicks.map(item=>item.label)},yaxis:{title:"capital",showgrid:false,zeroline:false},shapes:separators,showlegend:false},{responsive:true,displayModeBar:false});if($("overviewChart").removeAllListeners){$("overviewChart").removeAllListeners("plotly_click")}$("overviewChart").on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record)setRecord(record,true)})}',
    'function overview(){if(!state.summary||!state.summary.records.length){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";return}const sortMode=$("overviewSortMode")?.value||"current";const numOrInf=value=>{const n=Number(value);return Number.isFinite(n)?n:Infinity};const numOrNegInf=value=>{const n=Number(value);return Number.isFinite(n)?n:-Infinity};const visibleRows=state.summary.records.filter(row=>{const capital=Number(row?.capital);return !Number.isFinite(capital)||Math.abs(capital-100)>1e-9});if(!visibleRows.length){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";$("overviewEmpty").innerHTML="当前批次过滤掉 capital=100 之后没有可显示的数据。";return}const baseRows=[...visibleRows].sort((a,b)=>numOrInf(a.open_bar)-numOrInf(b.open_bar)||numOrInf(a.open_threshold)-numOrInf(b.open_threshold)||numOrInf(a.open_continous_threshold)-numOrInf(b.open_continous_threshold)||String(a.param_tag||"").localeCompare(String(b.param_tag||""))).map((row,index)=>({...row,__currentOrder:index+1}));const rows=sortMode==="capital"?[...baseRows].sort((a,b)=>numOrNegInf(b.capital)-numOrNegInf(a.capital)||a.__currentOrder-b.__currentOrder):baseRows;const groupTicks=[];const separators=[];if(sortMode==="current"){let groupStart=0;for(let i=1;i<=rows.length;i+=1){const sameGroup=i<rows.length&&keyOf(rows[i]?.open_bar)===keyOf(rows[groupStart]?.open_bar);if(sameGroup)continue;const left=groupStart+1;const right=i;groupTicks.push({value:(left+right)/2,label:formatParamDisplay(rows[groupStart]?.open_bar)});if(i<rows.length){separators.push({type:"line",xref:"x",yref:"paper",x0:i+0.5,x1:i+0.5,y0:0,y1:1,line:{color:"rgba(120,120,120,0.1)",width:1}})}groupStart=i}}$("overviewEmpty").style.display="none";$("overviewChart").style.display="block";Plotly.newPlot($("overviewChart"),[{x:rows.map((_,i)=>i+1),y:rows.map(r=>r.capital),mode:"markers",marker:{size:6,color:"#2f6bff"},customdata:rows.map(r=>r.selection_key),text:rows.map((r,index)=>["param_tag: "+esc(r.param_tag),"当前序号: "+(r.__currentOrder??"-"),"收益排序: "+(sortMode==="capital"?index+1:"-"),"open_bar: "+formatParamDisplay(r.open_bar),"open_threshold: "+formatParamDisplay(r.open_threshold),"open_cont: "+formatParamDisplay(r.open_continous_threshold),"capital: "+(r.capital??"-"),"trade_num: "+(r.trade_num??"-")].join("<br>")),hovertemplate:"%{text}<extra></extra>",name:"capital"}],{margin:{l:44,r:18,t:16,b:40},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:sortMode==="capital"?{title:"收益排名",showgrid:false,zeroline:false}:{title:"om",showgrid:false,zeroline:false,tickmode:"array",tickvals:groupTicks.map(item=>item.value),ticktext:groupTicks.map(item=>item.label)},yaxis:{title:"capital",showgrid:false,zeroline:false},shapes:separators,showlegend:false},{responsive:true,displayModeBar:false});if($("overviewChart").removeAllListeners){$("overviewChart").removeAllListeners("plotly_click")}$("overviewChart").on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record)setRecord(record,true)})}'
)
PAGE = PAGE.replace(
    'setupLayout();\nsetSettingsCollapsed(false);\nbootstrapPreset();',
    'setupLayout();\nconst overviewSortModeNode=$("overviewSortMode");if(overviewSortModeNode&&!overviewSortModeNode.dataset.bound){overviewSortModeNode.dataset.bound="1";overviewSortModeNode.addEventListener("change",()=>overview())}\nsetSettingsCollapsed(false);\nbootstrapPreset();'
)

PAGE = PAGE.replace(
    '<option value="current">当前序号</option><option value="capital">收益排序</option></select>',
    '<option value="current">当前序号</option><option value="capital">收益排序</option><option value="drawdown">鏈€灏忓洖鎾ゆ帓搴?/option></select>'
)
PAGE = PAGE.replace(
    'const rows=sortMode==="capital"?[...baseRows].sort((a,b)=>numOrNegInf(b.capital)-numOrNegInf(a.capital)||a.__currentOrder-b.__currentOrder):baseRows;',
    'const rows=sortMode==="capital"?[...baseRows].sort((a,b)=>numOrNegInf(b.capital)-numOrNegInf(a.capital)||a.__currentOrder-b.__currentOrder):sortMode==="drawdown"?[...baseRows].sort((a,b)=>numOrInf(a.biggest_wd)-numOrInf(b.biggest_wd)||a.__currentOrder-b.__currentOrder):baseRows;'
)
PAGE = PAGE.replace(
    'text:rows.map((r,index)=>["param_tag: "+esc(r.param_tag),"当前序号: "+(r.__currentOrder??"-"),"收益排序: "+(sortMode==="capital"?index+1:"-"),"open_bar: "+formatParamDisplay(r.open_bar),"open_threshold: "+formatParamDisplay(r.open_threshold),"open_cont: "+formatParamDisplay(r.open_continous_threshold),"capital: "+(r.capital??"-"),"trade_num: "+(r.trade_num??"-")].join("<br>"))',
    'text:rows.map((r,index)=>["param_tag: "+esc(r.param_tag),"当前序号: "+(r.__currentOrder??"-"),"收益排序: "+(sortMode==="capital"?index+1:"-"),"回撤排序: "+(sortMode==="drawdown"?index+1:"-"),"open_bar: "+formatParamDisplay(r.open_bar),"open_threshold: "+formatParamDisplay(r.open_threshold),"open_cont: "+formatParamDisplay(r.open_continous_threshold),"capital: "+(r.capital??"-"),"biggest_wd: "+formatMetric3(r.biggest_wd),"trade_num: "+(r.trade_num??"-")].join("<br>"))'
)
PAGE = PAGE.replace(
    'xaxis:sortMode==="capital"?{title:"收益排名",showgrid:false,zeroline:false}:{title:"om",showgrid:false,zeroline:false,tickmode:"array",tickvals:groupTicks.map(item=>item.value),ticktext:groupTicks.map(item=>item.label)}',
    'xaxis:sortMode==="capital"?{title:"收益排名",showgrid:false,zeroline:false}:sortMode==="drawdown"?{title:"鏈€灏忓洖鎾ゆ帓鍚?,showgrid:false,zeroline:false}:{title:"om",showgrid:false,zeroline:false,tickmode:"array",tickvals:groupTicks.map(item=>item.value),ticktext:groupTicks.map(item=>item.label)}'
)

PAGE = PAGE.replace(
    'setupLayout();\nconst overviewSortModeNode=$("overviewSortMode");if(overviewSortModeNode&&!overviewSortModeNode.dataset.bound){overviewSortModeNode.dataset.bound="1";overviewSortModeNode.addEventListener("change",()=>overview())}\nsetSettingsCollapsed(false);\nbootstrapPreset();',
    '''function detailParamDefs(){return[{id:"inputBar",field:"open_bar"},{id:"inputThreshold",field:"open_threshold"},{id:"inputCont",field:"open_continous_threshold"}]}
function sortParamKeys(values){return[...new Set((values||[]).map(keyOf).filter(Boolean))].sort((a,b)=>{const an=Number(a);const bn=Number(b);if(Number.isFinite(an)&&Number.isFinite(bn))return an-bn;return String(a).localeCompare(String(b))})}
function readParamSelections(){const out={};for(const def of detailParamDefs()){out[def.field]=keyOf($(def.id)?.value)}return out}
function filterParamRows(selections,ignoreField){const rows=state.summary?.records||[];return rows.filter(row=>detailParamDefs().every(def=>{if(def.field===ignoreField)return true;const selected=selections[def.field];return !selected||keyOf(row[def.field])===selected}))}
function renderParamOptions(el,values){const ui=ensureParamUi(el);const allowed=sortParamKeys(values);el.dataset.values=JSON.stringify(allowed);const has=allowed.length>0;el.disabled=!has;if(!has){el.value="";el.removeAttribute("min");el.removeAttribute("max");if(ui){ui.select.innerHTML="";const option=document.createElement("option");option.value="";option.textContent="暂无参数";ui.select.appendChild(option)}syncParamUi(el);return allowed}el.min=allowed[0];el.max=allowed[allowed.length-1];if(ui){ui.select.innerHTML="";for(const value of allowed){const option=document.createElement("option");option.value=value;option.textContent=formatParamDisplay(value);ui.select.appendChild(option)}}return allowed}
function syncAvailableParams(){if(!state.summary?.records?.length)return null;const defs=detailParamDefs();const selections=readParamSelections();for(let round=0;round<6;round+=1){let changed=false;for(const def of defs){const allowed=sortParamKeys(filterParamRows(selections,def.field).map(row=>row[def.field]));const current=selections[def.field];if(allowed.length&&!allowed.includes(current)){selections[def.field]=allowed[0];changed=true;continue}if(!allowed.length&&current){selections[def.field]="";changed=true}}if(!changed)break}for(const def of defs){const el=$(def.id);if(!el)continue;const allowed=renderParamOptions(el,filterParamRows(selections,def.field).map(row=>row[def.field]));const next=allowed.includes(selections[def.field])?selections[def.field]:(allowed[0]||"");selections[def.field]=next;if(keyOf(el.value)!==next){applyParamValue(el,next,false)}else{syncParamUi(el)}}const exactKey=defs.map(def=>selections[def.field]).join("|");return recordByKey(exactKey)||state.summary.records.find(row=>defs.every(def=>{const selected=selections[def.field];return !selected||keyOf(row[def.field])===selected}))||null}
selKey=function(){const record=syncAvailableParams();return record?.selection_key||[keyOf($("inputBar").value),keyOf($("inputThreshold").value),keyOf($("inputCont").value)].join("|")}
inputsFrom=function(record){applyParamValue($("inputBar"),record?.open_bar??"",false);applyParamValue($("inputThreshold"),record?.open_threshold??"",false);applyParamValue($("inputCont"),record?.open_continous_threshold??"",false);syncAvailableParams()}
setupLayout();
const overviewSortModeNode=$("overviewSortMode");if(overviewSortModeNode&&!overviewSortModeNode.dataset.bound){overviewSortModeNode.dataset.bound="1";overviewSortModeNode.addEventListener("change",()=>overview())}
setSettingsCollapsed(false);
bootstrapPreset();'''
)
PAGE = PAGE.replace(
    '.overview-sort-select{min-width:148px}',
    '.overview-sort-select{min-width:148px}.price-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:10px}.price-head .charttitle{margin-bottom:0}.price-head-stats{display:flex;align-items:center;justify-content:flex-end;gap:14px;flex-wrap:wrap;margin-left:auto}.price-head-stat{display:flex;align-items:center;gap:6px;font-size:12px;color:#66758c;white-space:nowrap}.price-head-stat strong{font-size:13px;color:#152033;font-weight:700}'
)
PAGE = PAGE.replace(
    'bootstrapPreset();\n</script>',
    '''function syncPriceHeaderStats(record){const tradeNode=$("priceHeadTrade");const wdNode=$("priceHeadWd");if(tradeNode){tradeNode.textContent=record?.trade_num??"-"}if(wdNode){wdNode.textContent=formatMetric3(record?.biggest_wd)}}
const baseUpdatePriceHeaderForStats=updatePriceHeader;
updatePriceHeader=function(){baseUpdatePriceHeaderForStats();const priceBox=document.querySelector("#pageDetail .price-box");if(!priceBox)return;let head=$("priceHead");if(!head){head=document.createElement("div");head.id="priceHead";head.className="price-head";const title=document.querySelector("#pageDetail .price-box .charttitle");const stats=document.createElement("div");stats.className="price-head-stats";stats.innerHTML='<div class="price-head-stat"><span>交易次数</span><strong id="priceHeadTrade">-</strong></div><div class="price-head-stat"><span>鏈€澶у洖鎾?/span><strong id="priceHeadWd">-</strong></div>';if(title){head.appendChild(title)}head.appendChild(stats);priceBox.insertBefore(head,priceBox.firstChild)}const titleNode=head.querySelector(".charttitle");if(titleNode){titleNode.textContent=currentPriceTitle()}syncPriceHeaderStats(state.activeRecord)}
const baseMetaForPriceStats=meta;
meta=function(record){baseMetaForPriceStats(record);syncPriceHeaderStats(record)}
updatePriceHeader();
syncPriceHeaderStats(state.activeRecord);
bootstrapPreset();
</script>'''
)

PAGE = PAGE.replace(
    'syncPriceHeaderStats(state.activeRecord);\nbootstrapPreset();\n</script>',
    '''syncPriceHeaderStats(state.activeRecord);
function ensureWithdrawalControl(){const grid=document.querySelector("#pageDetail .controls-grid");if(!grid)return;let input=$("inputWd");if(!input){const host=document.createElement("div");host.id="inputWdHost";host.innerHTML='<div class="label">回撤限制</div><input id="inputWd" class="num" type="number" disabled>';const contHost=$("inputCont")?.parentElement;if(contHost&&contHost.nextSibling){grid.insertBefore(host,contHost.nextSibling)}else{grid.appendChild(host)}input=$("inputWd")}if(input){ensureParamUi(input);if(!input.dataset.bound){input.dataset.bound="1";input.addEventListener("change",async()=>{$("inputBar").dispatchEvent(new Event("change"))})}}}
const setupLayoutWithWithdrawalControl=setupLayout;
setupLayout=function(){setupLayoutWithWithdrawalControl();ensureWithdrawalControl()}
detailParamDefs=function(){return[{id:"inputBar",field:"open_bar"},{id:"inputThreshold",field:"open_threshold"},{id:"inputCont",field:"open_continous_threshold"},{id:"inputWd",field:"open_withdrawal_threshold"}]}
selKey=function(){const record=syncAvailableParams();return record?.selection_key||detailParamDefs().map(def=>keyOf($(def.id)?.value)).join("|")}
inputsFrom=function(record){applyParamValue($("inputBar"),record?.open_bar??"",false);applyParamValue($("inputThreshold"),record?.open_threshold??"",false);applyParamValue($("inputCont"),record?.open_continous_threshold??"",false);applyParamValue($("inputWd"),record?.open_withdrawal_threshold??"",false);syncAvailableParams()}
const loadSummaryWithWithdrawalControl=loadSummary;
loadSummary=async function(file){const result=await loadSummaryWithWithdrawalControl(file);const wdInput=$("inputWd");if(wdInput){configInput(wdInput,state.summary?.controls?.open_withdrawal_threshold)}if(state.activeRecord){inputsFrom(state.activeRecord)}else{syncAvailableParams()}return result}
const resetUiWithWithdrawalControl=resetUi;
resetUi=function(){resetUiWithWithdrawalControl();const wdInput=$("inputWd");if(wdInput){configInput(wdInput,null)}}
setupLayout();
updatePriceHeader();
syncPriceHeaderStats(state.activeRecord);
bootstrapPreset();
</script>'''
)

PAGE = PAGE.replace(
    'bootstrapPreset();\n</script>',
    '''state.programId=state.programId||"";
function detectProgramId(name){const lowered=String(name||"").toLowerCase();if(lowered.includes("long_momentum_ratio outcome"))return "ratio";if(lowered.includes("long_momentum_atr outcome"))return "classic_atr";if(lowered.includes("long_momentum outcome"))return "classic";return ""}
function unsupportedFolderMessage(){return "褰撳墠缁撴灉鐩綍杩樻病璇嗗埆鍑虹瓥鐣ョ▼搴忋€?br>璇锋鏌?outcome_stats.xlsx 鐨勬枃浠跺悕銆?}
const resetUiBaseProgram=resetUi;
resetUi=function(){resetUiBaseProgram();state.programId=""}
const folderChangedBaseProgram=folderChanged;
folderChanged=function(files,folderLabel){const nextLabel=folderLabel||((files&&files.length)?(rel(files[0]).split("/")[0]||"宸查€夋嫨鐩綍"):"");const programId=detectProgramId(nextLabel)||state.programId||(typeof detectProgramIdByFiles==="function"?detectProgramIdByFiles(files,nextLabel):"");if(files&&files.length&&!programId){resetUiBaseProgram();state.files=[...files];state.folderLabel=nextLabel;$("folderName").textContent=nextLabel||"鏈瘑鍒洰褰?;$("metaFolder").textContent=nextLabel||"-";$("batchSelect").disabled=true;$("overviewEmpty").style.display="grid";$("overviewEmpty").innerHTML=unsupportedFolderMessage();if(typeof setOverview3dEmpty==="function"){setOverview3dEmpty(unsupportedFolderMessage())}setStatus("还没认出当前策略程序");return}state.programId=programId;const result=folderChangedBaseProgram(files,nextLabel);state.programId=programId;return result}
detailParamDefs=function(){return[{id:"inputBar",field:"open_bar"},{id:"inputThreshold",field:"open_threshold"},{id:"inputCont",field:"open_continous_threshold"},{id:"inputWd",field:"withdrawal_limit"}]}
selKey=function(){const record=syncAvailableParams();return record?.selection_key||detailParamDefs().map(def=>keyOf($(def.id)?.value)).join("|")}
inputsFrom=function(record){applyParamValue($("inputBar"),record?.open_bar??"",false);applyParamValue($("inputThreshold"),record?.open_threshold??"",false);applyParamValue($("inputCont"),record?.open_continous_threshold??"",false);applyParamValue($("inputWd"),record?.withdrawal_limit??"",false);syncAvailableParams()}
loadSummary=async function(file){setStatus("正在读取 outcome_stats");const programId=state.programId||detectProgramId(state.folderLabel)||"classic";const queryProgram="program_id="+encodeURIComponent(programId);state.summary=file.relative_path?await fetchJson("/api/preset-summary?path="+encodeURIComponent(pathOf(file)||file.name)+"&"+queryProgram):await upload("/api/summary?"+queryProgram,file);state.programId=state.summary?.program_id||programId;configInput($("inputBar"),state.summary.controls.open_bar);configInput($("inputThreshold"),state.summary.controls.open_threshold);configInput($("inputCont"),state.summary.controls.open_continous_threshold);const wdInput=$("inputWd");if(wdInput){configInput(wdInput,state.summary.controls.withdrawal_limit)}overview();const record=recordByKey(state.summary.default_key);if(record){await setRecord(record,false)}else{syncAvailableParams()}setStatus("鎵规宸茶浇鍏?);return state.summary}
bootstrapPreset();
</script>'''
)

PAGE = PAGE.replace(
    'bootstrapPreset();\n</script>',
    '''function detectProgramIdFromText(text){const lowered=String(text||"").toLowerCase();if(lowered.includes("long_momentum_ratio outcome"))return "ratio";if(lowered.includes("long_momentum_atr outcome"))return "classic_atr";if(lowered.includes("long_momentum outcome"))return "classic";if(/(^|\\s)(opm|ocpm|cpm|cwm|adt|bs|bw)\\S*/i.test(lowered))return "ratio";if(/(^|\\s)(oa|oca|owa|ca)\\S*/i.test(lowered))return "classic_atr";return ""}
function detectProgramIdByFiles(files,folderLabel){const labelHit=detectProgramIdFromText(folderLabel);if(labelHit)return labelHit;const joined=(files||[]).slice(0,120).map(file=>String(file.name||"")+" "+String(pathOf(file)||"")).join("\\n");const fileHit=detectProgramIdFromText(joined);if(fileHit)return fileHit;return files&&files.length?"classic":""}
folderChanged=function(files,folderLabel){const nextLabel=folderLabel||((files&&files.length)?(rel(files[0]).split("/")[0]||"宸查€夋嫨鐩綍"):"");const programId=detectProgramIdByFiles(files,nextLabel)||"classic";state.programId=programId;const result=folderChangedBaseProgram(files,nextLabel);state.programId=programId;return result}
loadSummary=async function(file){setStatus("正在读取 outcome_stats");const programId=state.programId||detectProgramIdByFiles(state.files,state.folderLabel)||"classic";const queryProgram="program_id="+encodeURIComponent(programId);state.summary=file.relative_path?await fetchJson("/api/preset-summary?path="+encodeURIComponent(pathOf(file)||file.name)+"&"+queryProgram):await upload("/api/summary?"+queryProgram,file);state.programId=state.summary?.program_id||programId;configInput($("inputBar"),state.summary.controls.open_bar);configInput($("inputThreshold"),state.summary.controls.open_threshold);configInput($("inputCont"),state.summary.controls.open_continous_threshold);const wdInput=$("inputWd");if(wdInput){configInput(wdInput,state.summary.controls.withdrawal_limit)}overview();const record=recordByKey(state.summary.default_key);if(record){await setRecord(record,false)}else{syncAvailableParams()}setStatus("鎵规宸茶浇鍏?);return state.summary}
bootstrapPreset();
</script>'''
)

PAGE = PAGE.replace(
    'function priceChart(price,trans){if(!price||!price.x.length){$("priceChart").style.display="none";$("priceEmpty").style.display="grid";$("priceEmpty").innerHTML="褰撳墠鎵规缂哄皯鍙敤鐨?perf.xlsx銆?;return}$("priceEmpty").style.display="none";$("priceChart").style.display="block";const candleX=price.x.map((_,index)=>index);const priceIndexMap=buildIndexMap(price.x);const shapes=gapShapes(price.x);const perfCapitalPoints=axisPoints(price.capital_x||[],price.capital_y||[],priceIndexMap);const traces=[{type:"candlestick",x:candleX,open:price.open,high:price.high,low:price.low,close:price.close,text:price.x,name:"price",increasing:{line:{color:CANDLE_UP_EDGE,width:0.8},fillcolor:CANDLE_UP_FILL},decreasing:{line:{color:CANDLE_DOWN_EDGE,width:0.8},fillcolor:CANDLE_DOWN_FILL},hovertemplate:"%{text}<br>open=%{open}<br>high=%{high}<br>low=%{low}<br>close=%{close}<extra></extra>"}];const capitalPoints=perfCapitalPoints.x.length?perfCapitalPoints:(trans?axisPoints(trans.capital_x,trans.capital_y,priceIndexMap):{x:[],y:[],text:[]});if(state.showCapitalOverlay&&capitalPoints.x.length)traces.unshift({type:"scatter",mode:"lines",x:capitalPoints.x,y:capitalPoints.y,text:capitalPoints.text,line:{color:ACCENT_BLUE,width:1.2},name:"capital",yaxis:"y2",hovertemplate:"%{text}<br>capital=%{y}<extra></extra>"});if(trans){const tradeLink=axisPoints(trans.trade_link_x,trans.trade_link_y,priceIndexMap,true);const buyPoints=axisPoints(trans.buy_points.x,trans.buy_points.y,priceIndexMap);const sellWdPoints=axisPoints(trans.sell_wd_points.x,trans.sell_wd_points.y,priceIndexMap);const sellSpeedPoints=axisPoints(trans.sell_speed_points.x,trans.sell_speed_points.y,priceIndexMap);if(tradeLink.x.length)traces.push({type:"scatter",mode:"lines",x:tradeLink.x,y:tradeLink.y,line:{color:ACCENT_BLUE,width:2},hoverinfo:"skip",name:"trade_link"});if(buyPoints.x.length)traces.push({type:"scatter",mode:"markers",x:buyPoints.x,y:buyPoints.y,text:buyPoints.text,marker:{color:"red",size:4},name:"buy",hovertemplate:"buy<br>%{text}<br>price=%{y}<extra></extra>"});if(sellWdPoints.x.length)traces.push({type:"scatter",mode:"markers",x:sellWdPoints.x,y:sellWdPoints.y,text:sellWdPoints.text,marker:{color:SELL_WD_COLOR,size:4},name:"sell_wd",hovertemplate:"sell_wd<br>%{text}<br>price=%{y}<extra></extra>"});if(sellSpeedPoints.x.length)traces.push({type:"scatter",mode:"markers",x:sellSpeedPoints.x,y:sellSpeedPoints.y,text:sellSpeedPoints.text,marker:{color:SELL_SPEED_COLOR,size:4},name:"sell_speed",hovertemplate:"sell_speed<br>%{text}<br>price=%{y}<extra></extra>"})}Plotly.newPlot($("priceChart"),traces,{margin:{l:48,r:52,t:12,b:36},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:numericAxis(price.x),yaxis:{title:"price",gridcolor:"#e8eef8",zeroline:false},yaxis2:{title:"capital",overlaying:"y",side:"right",showgrid:false,zeroline:false,visible:!!state.showCapitalOverlay},legend:{orientation:"h",yanchor:"bottom",y:1.02,xanchor:"left",x:0},shapes:shapes},{responsive:true,displayModeBar:false})}',
    'function priceChart(price,trans){if(!price||!price.x.length){$("priceChart").style.display="none";$("priceEmpty").style.display="grid";$("priceEmpty").innerHTML="褰撳墠鎵规缂哄皯鍙敤鐨?perf.xlsx銆?;return}$("priceEmpty").style.display="none";$("priceChart").style.display="block";const candleX=price.x.map((_,index)=>index);const priceIndexMap=buildIndexMap(price.x);const shapes=gapShapes(price.x);const perfCapitalPoints=axisPoints(price.capital_x||[],price.capital_y||[],priceIndexMap);const basePrice=Number(price.price_base);const scaleValue=value=>{if(value===null||value===undefined||value===\"\")return value;const num=Number(value);if(!Number.isFinite(num)||!Number.isFinite(basePrice)||basePrice===0)return num;return num/basePrice*100};const scaleAxisPoints=points=>({x:[...(points?.x||[])],y:(points?.y||[]).map(scaleValue),text:[...(points?.text||[])]});const traces=[{type:\"candlestick\",x:candleX,open:price.open,high:price.high,low:price.low,close:price.close,text:price.x,name:\"price\",increasing:{line:{color:CANDLE_UP_EDGE,width:0.8},fillcolor:CANDLE_UP_FILL},decreasing:{line:{color:CANDLE_DOWN_EDGE,width:0.8},fillcolor:CANDLE_DOWN_FILL},hovertemplate:\"%{text}<br>open=%{open:.4f}<br>high=%{high:.4f}<br>low=%{low:.4f}<br>close=%{close:.4f}<extra></extra>\"}];const capitalPoints=perfCapitalPoints.x.length?perfCapitalPoints:(trans?axisPoints(trans.capital_x,trans.capital_y,priceIndexMap):{x:[],y:[],text:[]});if(state.showCapitalOverlay&&capitalPoints.x.length)traces.unshift({type:\"scatter\",mode:\"lines\",x:capitalPoints.x,y:capitalPoints.y,text:capitalPoints.text,line:{color:ACCENT_BLUE,width:1.2},name:\"capital\",hovertemplate:\"%{text}<br>capital=%{y:.4f}<extra></extra>\"});if(trans){const tradeLink=scaleAxisPoints(axisPoints(trans.trade_link_x,trans.trade_link_y,priceIndexMap,true));const buyPoints=scaleAxisPoints(axisPoints(trans.buy_points.x,trans.buy_points.y,priceIndexMap));const sellWdPoints=scaleAxisPoints(axisPoints(trans.sell_wd_points.x,trans.sell_wd_points.y,priceIndexMap));const sellSpeedPoints=scaleAxisPoints(axisPoints(trans.sell_speed_points.x,trans.sell_speed_points.y,priceIndexMap));if(tradeLink.x.length)traces.push({type:\"scatter\",mode:\"lines\",x:tradeLink.x,y:tradeLink.y,line:{color:ACCENT_BLUE,width:2},hoverinfo:\"skip\",name:\"trade_link\"});if(buyPoints.x.length)traces.push({type:\"scatter\",mode:\"markers\",x:buyPoints.x,y:buyPoints.y,text:buyPoints.text,marker:{color:\"red\",size:4},name:\"buy\",hovertemplate:\"buy<br>%{text}<br>base100=%{y:.4f}<extra></extra>\"});if(sellWdPoints.x.length)traces.push({type:\"scatter\",mode:\"markers\",x:sellWdPoints.x,y:sellWdPoints.y,text:sellWdPoints.text,marker:{color:SELL_WD_COLOR,size:4},name:\"sell_wd\",hovertemplate:\"sell_wd<br>%{text}<br>base100=%{y:.4f}<extra></extra>\"});if(sellSpeedPoints.x.length)traces.push({type:\"scatter\",mode:\"markers\",x:sellSpeedPoints.x,y:sellSpeedPoints.y,text:sellSpeedPoints.text,marker:{color:SELL_SPEED_COLOR,size:4},name:\"sell_speed\",hovertemplate:\"sell_speed<br>%{text}<br>base100=%{y:.4f}<extra></extra>\"})}Plotly.newPlot($(\"priceChart\"),traces,{margin:{l:48,r:18,t:12,b:36},paper_bgcolor:\"rgba(0,0,0,0)\",plot_bgcolor:\"#fff\",xaxis:numericAxis(price.x),yaxis:{title:\"price (base=100)\",gridcolor:\"#e8eef8\",zeroline:false},legend:{orientation:\"h\",yanchor:\"bottom\",y:1.02,xanchor:\"left\",x:0},shapes:shapes},{responsive:true,displayModeBar:false})}'
)

PAGE = PAGE.replace(
    'folderChanged=function(files,folderLabel){const nextLabel=folderLabel||((files&&files.length)?(rel(files[0]).split("/")[0]||"瀹告煡鈧瀚ㄩ惄顔肩秿"):"");const programId=detectProgramIdByFiles(files,nextLabel)||"classic";state.programId=programId;const result=folderChangedBaseProgram(files,nextLabel);state.programId=programId;return result}',
    'function isIgnoredResultFile(file){const name=String(file?.name||"").trim().toLowerCase();return !!name&&name.startsWith("~")}\nfolderChanged=function(files,folderLabel){const cleanFiles=[...(files||[])].filter(file=>!isIgnoredResultFile(file));const nextLabel=folderLabel||((cleanFiles&&cleanFiles.length)?(rel(cleanFiles[0]).split("/")[0]||"瀹告煡鈧瀚ㄩ惄顔肩秿"):"");const programId=detectProgramIdByFiles(cleanFiles,nextLabel)||"classic";state.programId=programId;const result=folderChangedBaseProgram(cleanFiles,nextLabel);state.programId=programId;return result}'
)


PAGE = PAGE.replace(
    'folderChanged=function(files,folderLabel){const nextLabel=folderLabel||((files&&files.length)?(rel(files[0]).split("/")[0]||"宸查€夋嫨鐩綍"):"");const programId=detectProgramIdByFiles(files,nextLabel)||"classic";state.programId=programId;const result=folderChangedBaseProgram(files,nextLabel);state.programId=programId;return result}',
    'function isIgnoredResultFile(file){const name=String(file?.name||"").trim().toLowerCase();return !!name&&name.startsWith("~")}\nfolderChanged=function(files,folderLabel){const cleanFiles=[...(files||[])].filter(file=>!isIgnoredResultFile(file));const nextLabel=folderLabel||((cleanFiles&&cleanFiles.length)?(rel(cleanFiles[0]).split("/")[0]||"宸查€夋嫨鐩綍"):"");const programId=detectProgramIdByFiles(cleanFiles,nextLabel)||"classic";state.programId=programId;const result=folderChangedBaseProgram(cleanFiles,nextLabel);state.programId=programId;return result}'
)

PAGE = PAGE.replace(
    '.rail-stack>.box,.rail-stack>.sel,.rail-stack>.btn{width:100%;box-sizing:border-box}',
    '.rail-stack>.box,.rail-stack>.sel,.rail-stack>.btn{width:100%;box-sizing:border-box}.workspace-filter{display:grid;gap:6px}.workspace-filter .label{margin-bottom:0}.workspace-filter .sel{width:100%;box-sizing:border-box}'
)

PAGE = PAGE.replace(
    'bootstrapPreset();\n</script>',
    '''const WORKSPACE_PROGRAMS=[{tag:"long_momentum_GARCH",id:"garch",label:"long_momentum_GARCH.py"},{tag:"long_momentum",id:"classic",label:"long_momentum.py"},{tag:"long_momentum_ARCH_shock_multi",id:"classic",label:"long_momentum_ARCH_shock_multi.py"},{tag:"long_momentum_ARCH_shock",id:"classic",label:"long_momentum_ARCH_shock.py"},{tag:"long_momentum_ARCH",id:"classic",label:"long_momentum_ARCH.py"},{tag:"long_momentum_ATR",id:"classic_atr",label:"long_momentum_ATR.py"},{tag:"long_momentum_ratio",id:"ratio",label:"long_momentum_ratio.py"}];
function workspaceProgramLabel(tag){const hit=WORKSPACE_PROGRAMS.find(item=>item.tag===tag);return hit?hit.label:tag||"-"}
function workspaceProgramId(tag){const hit=WORKSPACE_PROGRAMS.find(item=>item.tag===tag);return hit?hit.id:"classic"}
function detectSummaryProgramTag(text){const lowered=String(text||"").trim().toLowerCase();if(lowered.includes("long_momentum_garch"))return "long_momentum_GARCH";if(lowered.includes("long_momentum_arch_shock_multi")||lowered.includes("long_momentum_arch shock multi"))return "long_momentum_ARCH_shock_multi";if(lowered.includes("long_momentum_arch_shock")||lowered.includes("long_momentum_arch shock"))return "long_momentum_ARCH_shock";if(lowered.includes("long_momentum_ratio"))return "long_momentum_ratio";if(lowered.includes("long_momentum_atr"))return "long_momentum_ATR";if(lowered.includes("long_momentum_arch"))return "long_momentum_ARCH";if(lowered.includes("long_momentum"))return "long_momentum";return ""}
function summaryStem(file){return String(file?.name||"").replace(/\\.xlsx$/i,"").trim()}
function stripOutcomeStatsSuffix(text){return String(text||"").replace(/\\s+outcome_stats$/i,"").trim()}
function stripProgramPrefix(text,programTag){const value=String(text||"").trim();if(!programTag)return value;const candidates=[programTag,String(programTag||"").replace(/_/g," "),String(programTag||"").replace(/ /g,"_")].filter(Boolean);for(const candidate of candidates){const prefix=candidate.toLowerCase()+" ";if(value.toLowerCase().startsWith(prefix)){return value.slice(candidate.length).trim()}}return value}
function extractPeriodKey(text){const match=String(text||"").match(/\\bperiod_[^ ]+/i);return match?match[0]:""}
function summaryFileMeta(file){const stem=stripOutcomeStatsSuffix(summaryStem(file));const programTag=detectSummaryProgramTag(file?.name)||detectSummaryProgramTag(pathOf(file))||detectSummaryProgramTag(state.folderLabel)||WORKSPACE_PROGRAMS[0].tag;const withoutProgram=stripProgramPrefix(stem,programTag);const periodKey=extractPeriodKey(withoutProgram);let fileLabel=withoutProgram;if(periodKey&&fileLabel.toLowerCase().startsWith(periodKey.toLowerCase()+" ")){fileLabel=fileLabel.slice(periodKey.length).trim()}if(!fileLabel){fileLabel=withoutProgram||stem}return{file:file,programTag:programTag,periodKey:periodKey||"鏈爣璁板懆鏈?,fileLabel:fileLabel,sortName:withoutProgram||stem}}
function uniqueKeepOrder(items){const out=[];const seen=new Set();for(const item of items){const key=String(item||"");if(seen.has(key))continue;seen.add(key);out.push(item)}return out}
function setSimpleSelectOptions(node,options,placeholder){if(!node)return;node.innerHTML="";const first=document.createElement("option");first.value="";first.textContent=placeholder;node.appendChild(first);for(const item of options){const opt=document.createElement("option");opt.value=item.value;opt.textContent=item.label;node.appendChild(opt)}node.disabled=!options.length}
function ensureWorkspaceFieldHost(hostId,labelText){let host=$(hostId);if(!host){host=document.createElement("div");host.id=hostId;host.className="workspace-filter";host.innerHTML='<div class="label">'+labelText+"</div>"}return host}
function ensureWorkspaceSelectors(){const workspaceControls=$("workspaceControls");if(!workspaceControls)return;let batchSelect=$("batchSelect");if(!batchSelect){batchSelect=document.createElement("select");batchSelect.id="batchSelect";batchSelect.className="sel";batchSelect.disabled=true;batchSelect.innerHTML='<option value="">璇烽€夋嫨缁熻鏂囦欢</option>'}const programHost=ensureWorkspaceFieldHost("programSelectHost","回撤程序");let programSelect=$("programSelect");if(!programSelect){programSelect=document.createElement("select");programSelect.id="programSelect";programSelect.className="sel";programHost.appendChild(programSelect)}const periodHost=ensureWorkspaceFieldHost("periodSelectHost","数据周期");let periodSelect=$("periodSelect");if(!periodSelect){periodSelect=document.createElement("select");periodSelect.id="periodSelect";periodSelect.className="sel";periodHost.appendChild(periodSelect)}const batchHost=ensureWorkspaceFieldHost("batchSelectHost","统计文件");if(batchSelect.parentElement!==batchHost){batchHost.appendChild(batchSelect)}const pickBtn=$("pickBtn");if(batchHost.parentElement!==workspaceControls){if(pickBtn&&pickBtn.parentElement===workspaceControls){workspaceControls.insertBefore(batchHost,pickBtn)}else{workspaceControls.appendChild(batchHost)}}if(programHost.parentElement!==workspaceControls){workspaceControls.insertBefore(programHost,batchHost)}if(periodHost.parentElement!==workspaceControls){workspaceControls.insertBefore(periodHost,batchHost)}if(!programSelect.dataset.bound){programSelect.dataset.bound="1";programSelect.addEventListener("change",()=>refreshWorkspaceSelectors(true))}if(!periodSelect.dataset.bound){periodSelect.dataset.bound="1";periodSelect.addEventListener("change",()=>refreshWorkspaceSelectors(true))}if(!batchSelect.dataset.bound){batchSelect.dataset.bound="1";batchSelect.addEventListener("change",async()=>{const file=state.summaryFiles.find(item=>(pathOf(item)||item.name)===batchSelect.value);if(!file)return;try{await loadSummary(file)}catch(error){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";$("overviewEmpty").innerHTML="姹囨€绘枃浠惰鍙栧け璐ワ細<br>"+esc(error.message);setStatus("批次读取失败")}})}}
function refreshWorkspaceSelectors(autoLoad){ensureWorkspaceSelectors();const programSelect=$("programSelect");const periodSelect=$("periodSelect");const batchSelect=$("batchSelect");if(!programSelect||!periodSelect||!batchSelect)return;const metas=[...(state.summaryFiles||[])].map(summaryFileMeta).sort((left,right)=>fileMtime(right.file)-fileMtime(left.file));const previousProgram=programSelect.value;const previousPeriod=periodSelect.value;const previousBatch=batchSelect.value;const programTags=uniqueKeepOrder(metas.map(meta=>meta.programTag));const programOptions=programTags.map(tag=>({value:tag,label:workspaceProgramLabel(tag)}));setSimpleSelectOptions(programSelect,programOptions,"璇烽€夋嫨鍥炴挙绋嬪簭");const selectedProgram=programTags.includes(previousProgram)?previousProgram:(programTags[0]||"");programSelect.value=selectedProgram;const programMetas=selectedProgram?metas.filter(meta=>meta.programTag===selectedProgram):metas;const periodKeys=uniqueKeepOrder(programMetas.map(meta=>meta.periodKey));const periodOptions=periodKeys.map(periodKey=>({value:periodKey,label:periodKey}));setSimpleSelectOptions(periodSelect,periodOptions,"璇烽€夋嫨鏁版嵁鍛ㄦ湡");const selectedPeriod=periodKeys.includes(previousPeriod)?previousPeriod:(periodKeys[0]||"");periodSelect.value=selectedPeriod;const batchMetas=programMetas.filter(meta=>!selectedPeriod||meta.periodKey===selectedPeriod);setSimpleSelectOptions(batchSelect,batchMetas.map(meta=>({value:pathOf(meta.file)||meta.file.name,label:meta.fileLabel})), "璇烽€夋嫨缁熻鏂囦欢");const batchValues=batchMetas.map(meta=>pathOf(meta.file)||meta.file.name);const nextBatch=batchValues.includes(previousBatch)?previousBatch:(batchValues[0]||"");batchSelect.value=nextBatch;if(nextBatch){state.programId=workspaceProgramId(selectedProgram);const shouldAutoLoad=!!nextBatch&&(autoLoad||!state.summary||previousBatch!==nextBatch||previousProgram!==selectedProgram||previousPeriod!==selectedPeriod);if(shouldAutoLoad){batchSelect.dispatchEvent(new Event("change"))}}}
const setupLayoutForWorkspaceSelectors=setupLayout;setupLayout=function(){setupLayoutForWorkspaceSelectors();ensureWorkspaceSelectors();refreshWorkspaceSelectors(false)}
const resetUiForWorkspaceSelectors=resetUi;resetUi=function(){resetUiForWorkspaceSelectors();ensureWorkspaceSelectors();setSimpleSelectOptions($("programSelect"),[],"璇烽€夋嫨鍥炴挙绋嬪簭");setSimpleSelectOptions($("periodSelect"),[],"璇烽€夋嫨鏁版嵁鍛ㄦ湡");setSimpleSelectOptions($("batchSelect"),[],"璇烽€夋嫨缁熻鏂囦欢")}
const folderChangedForWorkspaceSelectors=folderChanged;folderChanged=function(files,folderLabel){const result=folderChangedForWorkspaceSelectors(files,folderLabel);refreshWorkspaceSelectors(false);return result}
const loadSummaryForWorkspaceSelectors=loadSummary;loadSummary=async function(file){const meta=summaryFileMeta(file);state.programId=workspaceProgramId(meta.programTag);return await loadSummaryForWorkspaceSelectors(file)}
setupLayout();
refreshWorkspaceSelectors(false);
bootstrapPreset();
</script>'''
)

PAGE = PAGE.replace(
    'bootstrapPreset();\n</script>',
    '''function withdrawalLinkedToThreshold(){if(!state.summary?.records?.length){return false}const programId=state.summary?.program_id||state.programId||"";if(programId==="ratio"){return false}let seen=false;for(const row of state.summary.records){const threshold=keyOf(row?.open_threshold);const withdrawal=keyOf(row?.withdrawal_limit);if(!threshold||!withdrawal){return false}if(threshold!==withdrawal){return false}seen=true}return seen}
function syncWithdrawalUiState(){const wdInput=$("inputWd");if(!wdInput){return}const label=wdInput.parentElement?.querySelector(".label");const linked=withdrawalLinkedToThreshold();if(label){label.textContent=linked?"鍥炴挙闄愬埗锛堥殢閫熷害锛?:"回撤限制"}const ui=ensureParamUi(wdInput);if(linked){wdInput.disabled=true;if(ui){ui.select.disabled=true;ui.up.disabled=true;ui.down.disabled=true}return}const values=paramValues(wdInput);wdInput.disabled=!values.length;syncParamUi(wdInput)}
const readParamSelectionsLinkedWithdrawal=readParamSelections;readParamSelections=function(){const out=readParamSelectionsLinkedWithdrawal();if(withdrawalLinkedToThreshold()){out.withdrawal_limit=out.open_threshold||out.withdrawal_limit||""}return out}
const filterParamRowsLinkedWithdrawal=filterParamRows;filterParamRows=function(selections,ignoreField){if(!withdrawalLinkedToThreshold()){return filterParamRowsLinkedWithdrawal(selections,ignoreField)}const rows=state.summary?.records||[];const ignorePair=ignoreField==="open_threshold"||ignoreField==="withdrawal_limit";return rows.filter(row=>detailParamDefs().every(def=>{if(ignorePair&&(def.field==="open_threshold"||def.field==="withdrawal_limit")){return true}if(def.field===ignoreField){return true}const selected=def.field==="withdrawal_limit"?(selections.open_threshold||selections.withdrawal_limit):selections[def.field];return !selected||keyOf(row[def.field])===selected}))}
const syncAvailableParamsLinkedWithdrawal=syncAvailableParams;syncAvailableParams=function(){const record=syncAvailableParamsLinkedWithdrawal();if(withdrawalLinkedToThreshold()){const threshold=keyOf($("inputThreshold")?.value);const wdInput=$("inputWd");if(wdInput&&threshold&&keyOf(wdInput.value)!==threshold){applyParamValue(wdInput,threshold,false)}}syncWithdrawalUiState();return record}
const inputsFromLinkedWithdrawal=inputsFrom;inputsFrom=function(record){inputsFromLinkedWithdrawal(record);if(withdrawalLinkedToThreshold()){$("inputWd")&&applyParamValue($("inputWd"),record?.open_threshold??record?.withdrawal_limit??"",false)}syncWithdrawalUiState()}
const loadSummaryLinkedWithdrawal=loadSummary;loadSummary=async function(file){const result=await loadSummaryLinkedWithdrawal(file);syncWithdrawalUiState();return result}
const resetUiLinkedWithdrawal=resetUi;resetUi=function(){resetUiLinkedWithdrawal();const wdInput=$("inputWd");if(!wdInput){return}wdInput.disabled=false;const label=wdInput.parentElement?.querySelector(".label");if(label){label.textContent="回撤限制"}}
const setupLayoutLinkedWithdrawal=setupLayout;setupLayout=function(){setupLayoutLinkedWithdrawal();syncWithdrawalUiState()}
setupLayout();
syncWithdrawalUiState();
bootstrapPreset();
</script>'''
)

PAGE = PAGE.replace(
    "</style>",
    """
.param-select option.param-option-unavailable{color:#d99;background:#fff4f4}
</style>"""
)

PAGE = PAGE.replace(
    'bootstrapPreset();\n</script>',
    '''function paramFieldMeta(el){const id=el?.id;return detailParamDefs().find(def=>def.id===id)||null}
function paramUniverseValues(el){const meta=paramFieldMeta(el);if(!meta){return paramValues(el)}const controlValues=state.summary?.controls?.[meta.field]?.values;if(Array.isArray(controlValues)&&controlValues.length){return sortParamKeys(controlValues)}try{const allValues=JSON.parse(el?.dataset?.allValues||"[]");return sortParamKeys(allValues)}catch(error){return paramValues(el)}}
paramValues=function(el){try{const allowed=JSON.parse(el?.dataset?.allowedValues||"[]");if(Array.isArray(allowed)&&allowed.length){return allowed}}catch(error){}try{return JSON.parse(el?.dataset?.values||"[]")}catch(error){return[]}}
syncParamUi=function(el){const ui=ensureParamUi(el);if(!ui)return;const allowedValues=paramValues(el);const current=keyOf(el.value);if(current&&ui.select.value!==current){ui.select.value=current}const index=allowedValues.indexOf(current);const disabled=!!el.disabled||!allowedValues.length;ui.select.disabled=!!el.disabled;ui.up.disabled=disabled||index<0||index>=allowedValues.length-1;ui.down.disabled=disabled||index<=0}
renderParamOptions=function(el,values){const ui=ensureParamUi(el);const allowed=sortParamKeys(values);const universe=paramUniverseValues(el);el.dataset.values=JSON.stringify(allowed);el.dataset.allowedValues=JSON.stringify(allowed);el.dataset.allValues=JSON.stringify(universe);const hasUniverse=universe.length>0;el.disabled=!hasUniverse;if(!hasUniverse){el.value="";el.removeAttribute("min");el.removeAttribute("max");if(ui){ui.select.innerHTML="";const option=document.createElement("option");option.value="";option.textContent="暂无参数";ui.select.appendChild(option)}syncParamUi(el);return allowed}el.min=universe[0];el.max=universe[universe.length-1];if(ui){ui.select.innerHTML="";for(const value of universe){const option=document.createElement("option");const enabled=allowed.includes(value);option.value=value;option.textContent=formatParamDisplay(value);option.className=enabled?"param-option-available":"param-option-unavailable";if(!enabled){option.disabled=true;option.style.color="#d99";option.style.backgroundColor="#fff4f4"}ui.select.appendChild(option)}}return allowed}
const syncAvailableParamsShowUnavailable=syncAvailableParams;syncAvailableParams=function(){const record=syncAvailableParamsShowUnavailable();for(const def of detailParamDefs()){const el=$(def.id);if(el){syncParamUi(el)}}return record}
bootstrapPreset();
</script>'''
)

PAGE = PAGE.replace(
    "</style>",
    """
.param-select option[data-unavailable="1"]{color:#e5a2a2;background:#fff3f3}
</style>"""
)

PAGE = PAGE.replace(
    'bootstrapPreset();\n</script>',
    '''function paramFieldMetaFinal(el){const id=el?.id;return detailParamDefs().find(def=>def.id===id)||null}
function paramUniverseValuesFinal(el){const meta=paramFieldMetaFinal(el);if(!meta){return[]}const controlValues=state.summary?.controls?.[meta.field]?.values;if(Array.isArray(controlValues)&&controlValues.length){return sortParamKeys(controlValues)}try{return sortParamKeys(JSON.parse(el?.dataset?.allValues||"[]"))}catch(error){return[]}}
function paramAllowedValuesFinal(el,selections){const meta=paramFieldMetaFinal(el);if(!meta){return[]}return sortParamKeys(filterParamRows(selections||readParamSelections(),meta.field).map(row=>row?.[meta.field]))}
function rebuildParamSelectFinal(el,allowed){const ui=ensureParamUi(el);if(!ui){return}const universe=paramUniverseValuesFinal(el);el.dataset.allowedValues=JSON.stringify(allowed);el.dataset.values=JSON.stringify(allowed);el.dataset.allValues=JSON.stringify(universe);el.disabled=!universe.length;if(!universe.length){ui.select.innerHTML="";const option=document.createElement("option");option.value="";option.textContent="暂无参数";ui.select.appendChild(option);syncParamUi(el);return}ui.select.innerHTML="";const allowedSet=new Set(allowed);for(const value of universe){const option=document.createElement("option");const enabled=allowedSet.has(value);option.value=value;option.textContent=formatParamDisplay(value);option.className=enabled?"param-option-available":"param-option-unavailable";option.dataset.unavailable=enabled?"0":"1";if(!enabled){option.disabled=true;option.style.color="#e5a2a2";option.style.backgroundColor="#fff3f3"}ui.select.appendChild(option)}const current=keyOf(el.value);if(current&&ui.select.value!==current){ui.select.value=current}syncParamUi(el)}
function refreshUnavailableParamOptionsFinal(){if(!state.summary?.records?.length){return null}const selections=readParamSelections();for(const def of detailParamDefs()){const el=$(def.id);if(!el){continue}const allowed=paramAllowedValuesFinal(el,selections);rebuildParamSelectFinal(el,allowed)}return true}
paramFieldMeta=paramFieldMetaFinal;
paramUniverseValues=paramUniverseValuesFinal;
renderParamOptions=function(el,values){const allowed=sortParamKeys(values);rebuildParamSelectFinal(el,allowed);return allowed}
const syncAvailableParamsUnavailableFinal=syncAvailableParams;syncAvailableParams=function(){const record=syncAvailableParamsUnavailableFinal();refreshUnavailableParamOptionsFinal();return record}
const loadSummaryUnavailableFinal=loadSummary;loadSummary=async function(file){const result=await loadSummaryUnavailableFinal(file);refreshUnavailableParamOptionsFinal();return result}
const resetUiUnavailableFinal=resetUi;resetUi=function(){resetUiUnavailableFinal();for(const def of detailParamDefs()){const el=$(def.id);if(!el){continue}delete el.dataset.allowedValues;delete el.dataset.allValues}}
window.paramFieldMeta=paramFieldMetaFinal;
window.paramUniverseValues=paramUniverseValuesFinal;
bootstrapPreset();
</script>'''
)

PAGE = PAGE.replace(
    '褰撳墠涓夊厓鍙傛暟缁勫悎娌℃湁瀵瑰簲缁撴灉銆?,
    '鏃犳暟鎹?
)
PAGE = PAGE.replace(
    '当前三元参数组合没有对应 trans.xlsx銆?,
    '鏃犳暟鎹?
)
PAGE = PAGE.replace(
    '璇ュ弬鏁扮粍鍚堟病鏈夌粨鏋?,
    '鏃犳暟鎹?
)

PAGE = PAGE.replace(
    "</style>",
    """
.param-select option.param-option-unavailable,.param-select option[data-unavailable="1"]{color:#152033 !important;background:#fff3f3 !important}
</style>"""
)

PAGE = PAGE.replace(
    'bootstrapPreset();\n</script>',
    '''function paramUniverseSelectableFinal(el){if(typeof paramUniverseValuesFinal==="function"){return paramUniverseValuesFinal(el)}if(typeof paramUniverseValues==="function"){return paramUniverseValues(el)}return[]}
function rebuildParamSelectSelectableFinal(el,allowed){const ui=ensureParamUi(el);if(!ui){return}const universe=paramUniverseSelectableFinal(el);el.dataset.allowedValues=JSON.stringify(allowed);el.dataset.values=JSON.stringify(allowed);el.dataset.allValues=JSON.stringify(universe);el.disabled=!universe.length;if(!universe.length){ui.select.innerHTML="";const option=document.createElement("option");option.value="";option.textContent="暂无参数";ui.select.appendChild(option);syncParamUi(el);return}ui.select.innerHTML="";const allowedSet=new Set(allowed);for(const value of universe){const option=document.createElement("option");const enabled=allowedSet.has(value);option.value=value;option.textContent=formatParamDisplay(value);option.className=enabled?"param-option-available":"param-option-unavailable";option.dataset.unavailable=enabled?"0":"1";if(!enabled){option.style.backgroundColor="#fff3f3"}ui.select.appendChild(option)}const current=keyOf(el.value);if(current&&ui.select.value!==current){ui.select.value=current}syncParamUi(el)}
function syncAvailableParamsSelectableFinal(){if(!state.summary?.records?.length){return null}const defs=detailParamDefs();const selections=readParamSelections();for(const def of defs){const el=$(def.id);if(!el){continue}const universe=paramUniverseSelectableFinal(el);const current=selections[def.field];const next=universe.includes(current)?current:(universe[0]||"");selections[def.field]=next;if(keyOf(el.value)!==next){applyParamValue(el,next,false)}}if(typeof withdrawalLinkedToThreshold==="function"&&withdrawalLinkedToThreshold()){const threshold=keyOf($("inputThreshold")?.value);const wdInput=$("inputWd");if(wdInput&&threshold&&keyOf(wdInput.value)!==threshold){applyParamValue(wdInput,threshold,false)}selections.withdrawal_limit=threshold||selections.withdrawal_limit||""}for(const def of defs){const el=$(def.id);if(!el){continue}const allowed=sortParamKeys(filterParamRows(selections,def.field).map(row=>row?.[def.field]));rebuildParamSelectSelectableFinal(el,allowed)}if(typeof syncWithdrawalUiState==="function"){syncWithdrawalUiState()}const exactKey=defs.map(def=>keyOf($(def.id)?.value)).join("|");return recordByKey(exactKey)||null}
syncAvailableParams=syncAvailableParamsSelectableFinal;
selKey=function(){syncAvailableParamsSelectableFinal();return detailParamDefs().map(def=>keyOf($(def.id)?.value)).join("|")}
const loadSummarySelectableFinal=loadSummary;loadSummary=async function(file){const result=await loadSummarySelectableFinal(file);syncAvailableParamsSelectableFinal();return result}
bootstrapPreset();
</script>'''
)

PAGE = PAGE.replace(
    'bootstrapPreset();\n</script>',
    '''function currentProgramTag(){return String(state.programTag||"").trim()}
function isShockProgramTag(tag){return String(tag||"").trim().toLowerCase()==="long_momentum_arch_shock"}
function isGarchProgramTag(tag){return String(tag||"").trim().toLowerCase()==="long_momentum_garch"}
function currentParamLabelMap(){if(isShockProgramTag(currentProgramTag())){return{open_bar:"骞充粨绐楀彛",open_threshold:"shock 寮€浠撳€嶆暟",open_continous_threshold:"速度平仓倍数",withdrawal_limit:"回撤平仓倍数"}}if(isGarchProgramTag(currentProgramTag())){return{open_bar:"鏃堕棿绐楀彛锛堝垎閽燂級",open_threshold:"速度限制倍数",open_continous_threshold:"寮€浠撻棬妲涘€嶆暟",withdrawal_limit:"回撤限制倍数"}}return{open_bar:"鏃堕棿绐楀彛",open_threshold:"閫熷害闄愬埗",open_continous_threshold:"寮€浠撻棬妲?,withdrawal_limit:"回撤限制"}}
function syncProgramSpecificLabels(){const labels=currentParamLabelMap();const barLabel=$("inputBar")?.parentElement?.querySelector(".label");const thresholdLabel=$("inputThreshold")?.parentElement?.querySelector(".label");const contLabel=$("inputCont")?.parentElement?.querySelector(".label");const wdLabel=$("inputWd")?.parentElement?.querySelector(".label");if(barLabel){barLabel.textContent=labels.open_bar}if(thresholdLabel){thresholdLabel.textContent=labels.open_threshold}if(contLabel){contLabel.textContent=labels.open_continous_threshold}if(wdLabel){wdLabel.textContent=labels.withdrawal_limit}}
const refreshWorkspaceSelectorsProgramTag=refreshWorkspaceSelectors;refreshWorkspaceSelectors=function(autoLoad){const result=refreshWorkspaceSelectorsProgramTag(autoLoad);const programSelect=$("programSelect");state.programTag=programSelect?.value||state.programTag||"";syncProgramSpecificLabels();return result}
const loadSummaryProgramTag=loadSummary;loadSummary=async function(file){if(typeof summaryFileMeta==="function"){const meta=summaryFileMeta(file);state.programTag=meta?.programTag||state.programTag||""}const result=await loadSummaryProgramTag(file);syncProgramSpecificLabels();return result}
const resetUiProgramTag=resetUi;resetUi=function(){resetUiProgramTag();state.programTag="";syncProgramSpecificLabels()}
const folderChangedProgramTag=folderChanged;folderChanged=function(files,folderLabel){const result=folderChangedProgramTag(files,folderLabel);const programSelect=$("programSelect");state.programTag=programSelect?.value||state.programTag||"";syncProgramSpecificLabels();return result}
setupLayout();
syncProgramSpecificLabels();
bootstrapPreset();
</script>'''
)

PAGE = PAGE.replace(
    'bootstrapPreset();\n</script>',
    '''function currentProgramTagFinal(){return String(state.programTag||"").trim().toLowerCase()}
function isShockProgramFinal(){const tag=currentProgramTagFinal();return tag==="long_momentum_arch_shock"||tag==="long_momentum_arch_shock_multi"}
function isGarchProgramFinal(){return currentProgramTagFinal()==="long_momentum_garch"}
function currentProgramLabelMapFinal(){if(isShockProgramFinal()){return{open_bar:"骞充粨绐楀彛",open_threshold:"shock 寮€浠撳€嶆暟",open_continous_threshold:"速度平仓倍数",withdrawal_limit:"回撤平仓倍数"}}if(isGarchProgramFinal()){return{open_bar:"鏃堕棿绐楀彛锛堝垎閽燂級",open_threshold:"速度限制倍数",open_continous_threshold:"寮€浠撻棬妲涘€嶆暟",withdrawal_limit:"回撤限制倍数"}}return{open_bar:"鏃堕棿绐楀彛",open_threshold:"閫熷害闄愬埗",open_continous_threshold:"寮€浠撻棬妲?,withdrawal_limit:"回撤限制"}}
function labelTextFinal(field){const labels=currentProgramLabelMapFinal();return labels[field]||field}
const overviewProgramSpecificFinal=overview;overview=function(){if(!state.summary||!state.summary.records.length){return overviewProgramSpecificFinal()}const sortMode=$("overviewSortMode")?.value||"current";const numOrInf=value=>{const n=Number(value);return Number.isFinite(n)?n:Infinity};const numOrNegInf=value=>{const n=Number(value);return Number.isFinite(n)?n:-Infinity};const visibleRows=state.summary.records.filter(row=>{const capital=Number(row?.capital);return !Number.isFinite(capital)||Math.abs(capital-100)>1e-9});if(!visibleRows.length){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";$("overviewEmpty").innerHTML="当前批次过滤掉 capital=100 之后没有可显示的数据。";return}const baseRows=[...visibleRows].sort((a,b)=>numOrInf(a.open_bar)-numOrInf(b.open_bar)||numOrInf(a.open_threshold)-numOrInf(b.open_threshold)||numOrInf(a.open_continous_threshold)-numOrInf(b.open_continous_threshold)||String(a.param_tag||"").localeCompare(String(b.param_tag||""))).map((row,index)=>({...row,__currentOrder:index+1}));const rows=sortMode==="capital"?[...baseRows].sort((a,b)=>numOrNegInf(b.capital)-numOrNegInf(a.capital)||a.__currentOrder-b.__currentOrder):baseRows;const groupTicks=[];const separators=[];if(sortMode==="current"){let groupStart=0;for(let i=1;i<=rows.length;i+=1){const sameGroup=i<rows.length&&keyOf(rows[i]?.open_bar)===keyOf(rows[groupStart]?.open_bar);if(sameGroup)continue;const left=groupStart+1;const right=i;groupTicks.push({value:(left+right)/2,label:formatParamDisplay(rows[groupStart]?.open_bar)});if(i<rows.length){separators.push({type:"line",xref:"x",yref:"paper",x0:i+0.5,x1:i+0.5,y0:0,y1:1,line:{color:"rgba(120,120,120,0.1)",width:1}})}groupStart=i}}$("overviewEmpty").style.display="none";$("overviewChart").style.display="block";Plotly.newPlot($("overviewChart"),[{x:rows.map((_,i)=>i+1),y:rows.map(r=>r.capital),mode:"markers",marker:{size:6,color:"#2f6bff"},customdata:rows.map(r=>r.selection_key),text:rows.map((r,index)=>["param_tag: "+esc(r.param_tag),"当前序号: "+(r.__currentOrder??"-"),"收益排序: "+(sortMode==="capital"?index+1:"-"),labelTextFinal("open_bar")+": "+formatParamDisplay(r.open_bar),labelTextFinal("open_threshold")+": "+formatParamDisplay(r.open_threshold),labelTextFinal("open_continous_threshold")+": "+formatParamDisplay(r.open_continous_threshold),"capital: "+(r.capital??"-"),"trade_num: "+(r.trade_num??"-")].join("<br>")),hovertemplate:"%{text}<extra></extra>",name:"capital"}],{margin:{l:44,r:18,t:16,b:40},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:sortMode==="capital"?{title:"收益排名",showgrid:false,zeroline:false}:{title:labelTextFinal("open_bar"),showgrid:false,zeroline:false,tickmode:"array",tickvals:groupTicks.map(item=>item.value),ticktext:groupTicks.map(item=>item.label)},yaxis:{title:"capital",showgrid:false,zeroline:false},shapes:separators,showlegend:false},{responsive:true,displayModeBar:false});if($("overviewChart").removeAllListeners){$("overviewChart").removeAllListeners("plotly_click")}$("overviewChart").on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record)setRecord(record,true)})}
renderOverview3d=function(){const chart=$("overview3dChart");const empty=$("overview3dEmpty");if(!chart||!empty){return}if(!state.summary||!state.summary.records.length){setOverview3dEmpty("璇烽€夋嫨涓€涓洖娴嬫壒娆°€?);return}const rows=state.summary.records.filter(row=>{const capital=Number(row?.capital);return row.capital!==null&&row.capital!==undefined&&row.open_bar!==null&&row.open_bar!==undefined&&row.open_threshold!==null&&row.open_threshold!==undefined&&(!Number.isFinite(capital)||Math.abs(capital-100)>1e-9)});if(!rows.length){setOverview3dEmpty("褰撳墠鎵规缂哄皯鍙敤浜庝笁缁村浘鐨勫弬鏁版暟鎹€?);return}empty.style.display="none";chart.style.display="block";const text=rows.map(row=>["param_tag: "+esc(row.param_tag),labelTextFinal("open_bar")+": "+formatParamDisplay(row.open_bar),labelTextFinal("open_threshold")+": "+formatParamDisplay(row.open_threshold),labelTextFinal("open_continous_threshold")+": "+formatParamDisplay(row.open_continous_threshold),"capital: "+formatMetric3(row.capital),"trade_num: "+(row.trade_num??"-")].join("<br>"));Plotly.newPlot(chart,[{type:"scatter3d",mode:"markers",x:rows.map(row=>row.open_threshold),y:rows.map(row=>row.capital),z:rows.map(row=>row.open_bar),customdata:rows.map(row=>row.selection_key),text:text,hovertemplate:"%{text}<extra></extra>",marker:{size:5,color:rows.map(row=>row.capital),colorscale:"Viridis",opacity:0.95,colorbar:{title:"capital"}}}],{margin:{l:0,r:0,t:10,b:0},paper_bgcolor:"rgba(0,0,0,0)",scene:{xaxis:{title:labelTextFinal("open_threshold"),gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},yaxis:{title:"capital",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},zaxis:{title:labelTextFinal("open_bar"),gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"}},showlegend:false},{responsive:true,displayModeBar:false});if(chart.removeAllListeners){chart.removeAllListeners("plotly_click")}chart.on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record){setRecord(record,true)}})}
const syncProgramSpecificLabelsFinal=syncProgramSpecificLabels;syncProgramSpecificLabels=function(){syncProgramSpecificLabelsFinal();if($("overviewChart")?.style.display==="block"){overview()}if($("overview3dChart")?.style.display==="block"){renderOverview3d()}}
setupLayout();
syncProgramSpecificLabels();
bootstrapPreset();
</script>'''
)

PAGE = PAGE.replace(
    'bootstrapPreset();\n</script>',
    '''function cleanLabelFinal(field){if(field==="open_bar"){return "时间窗口"}if(field==="open_threshold"){return "速度限制"}if(field==="open_continous_threshold"){return "开仓门槛"}if(field==="withdrawal_limit"){return "回撤限制"}return field||""}
function cleanOverviewRowsFinal(){const rows=state.summary?.records||[];return rows.filter(row=>{const capital=Number(row?.capital);return !Number.isFinite(capital)||Math.abs(capital-100)>1e-9})}
syncProgramSpecificLabels=function(){const barLabel=$("inputBar")?.parentElement?.querySelector(".label");const thresholdLabel=$("inputThreshold")?.parentElement?.querySelector(".label");const contLabel=$("inputCont")?.parentElement?.querySelector(".label");const wdLabel=$("inputWd")?.parentElement?.querySelector(".label");if(barLabel){barLabel.textContent=cleanLabelFinal("open_bar")}if(thresholdLabel){thresholdLabel.textContent=cleanLabelFinal("open_threshold")}if(contLabel){contLabel.textContent=cleanLabelFinal("open_continous_threshold")}if(wdLabel){wdLabel.textContent=cleanLabelFinal("withdrawal_limit")}}
overview=function(){if(!state.summary||!state.summary.records.length){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";return}const sortMode=$("overviewSortMode")?.value||"current";const numOrInf=value=>{const n=Number(value);return Number.isFinite(n)?n:Infinity};const numOrNegInf=value=>{const n=Number(value);return Number.isFinite(n)?n:-Infinity};const visibleRows=cleanOverviewRowsFinal();if(!visibleRows.length){$("overviewChart").style.display="none";$("overviewEmpty").style.display="grid";$("overviewEmpty").innerHTML="当前批次过滤掉 capital=100 之后没有可显示的数据。";return}const baseRows=[...visibleRows].sort((a,b)=>numOrInf(a.open_bar)-numOrInf(b.open_bar)||numOrInf(a.open_threshold)-numOrInf(b.open_threshold)||numOrInf(a.open_continous_threshold)-numOrInf(b.open_continous_threshold)||String(a.param_tag||"").localeCompare(String(b.param_tag||""))).map((row,index)=>({...row,__currentOrder:index+1}));const rows=sortMode==="capital"?[...baseRows].sort((a,b)=>numOrNegInf(b.capital)-numOrNegInf(a.capital)||a.__currentOrder-b.__currentOrder):sortMode==="drawdown"?[...baseRows].sort((a,b)=>numOrInf(a.biggest_wd)-numOrInf(b.biggest_wd)||a.__currentOrder-b.__currentOrder):baseRows;const groupTicks=[];const separators=[];if(sortMode==="current"){let groupStart=0;for(let i=1;i<=rows.length;i+=1){const sameGroup=i<rows.length&&keyOf(rows[i]?.open_bar)===keyOf(rows[groupStart]?.open_bar);if(sameGroup)continue;const left=groupStart+1;const right=i;groupTicks.push({value:(left+right)/2,label:formatParamDisplay(rows[groupStart]?.open_bar)});if(i<rows.length){separators.push({type:"line",xref:"x",yref:"paper",x0:i+0.5,x1:i+0.5,y0:0,y1:1,line:{color:"rgba(120,120,120,0.1)",width:1}})}groupStart=i}}$("overviewEmpty").style.display="none";$("overviewChart").style.display="block";Plotly.newPlot($("overviewChart"),[{x:rows.map((_,i)=>i+1),y:rows.map(r=>r.capital),mode:"markers",marker:{size:6,color:"#2f6bff"},customdata:rows.map(r=>r.selection_key),text:rows.map((r,index)=>["param_tag: "+esc(r.param_tag),"当前序号: "+(r.__currentOrder??"-"),"收益排序: "+(sortMode==="capital"?index+1:"-"),"回撤排序: "+(sortMode==="drawdown"?index+1:"-"),cleanLabelFinal("open_bar")+": "+formatParamDisplay(r.open_bar),cleanLabelFinal("open_threshold")+": "+formatParamDisplay(r.open_threshold),cleanLabelFinal("open_continous_threshold")+": "+formatParamDisplay(r.open_continous_threshold),"capital: "+(r.capital??"-"),"trade_num: "+(r.trade_num??"-")].join("<br>")),hovertemplate:"%{text}<extra></extra>",name:"capital"}],{margin:{l:44,r:18,t:16,b:40},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"#fff",xaxis:sortMode==="capital"?{title:"收益排名",showgrid:false,zeroline:false}:sortMode==="drawdown"?{title:"最小回撤排名",showgrid:false,zeroline:false}:{title:cleanLabelFinal("open_bar"),showgrid:false,zeroline:false,tickmode:"array",tickvals:groupTicks.map(item=>item.value),ticktext:groupTicks.map(item=>item.label)},yaxis:{title:"capital",showgrid:false,zeroline:false},shapes:separators,showlegend:false},{responsive:true,displayModeBar:false});if($("overviewChart").removeAllListeners){$("overviewChart").removeAllListeners("plotly_click")}$("overviewChart").on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record)setRecord(record,true)})}
renderOverview3d=function(){const chart=$("overview3dChart");const empty=$("overview3dEmpty");if(!chart||!empty){return}if(!state.summary||!state.summary.records.length){setOverview3dEmpty("请选择一个回测批次。");return}const rows=cleanOverviewRowsFinal().filter(row=>row.capital!==null&&row.capital!==undefined&&row.open_bar!==null&&row.open_bar!==undefined&&row.open_threshold!==null&&row.open_threshold!==undefined);if(!rows.length){setOverview3dEmpty("当前批次过滤掉 capital=100 之后没有可显示的数据。");return}empty.style.display="none";chart.style.display="block";const text=rows.map(row=>["param_tag: "+esc(row.param_tag),cleanLabelFinal("open_bar")+": "+formatParamDisplay(row.open_bar),cleanLabelFinal("open_threshold")+": "+formatParamDisplay(row.open_threshold),cleanLabelFinal("open_continous_threshold")+": "+formatParamDisplay(row.open_continous_threshold),"capital: "+formatMetric3(row.capital),"trade_num: "+(row.trade_num??"-")].join("<br>"));Plotly.newPlot(chart,[{type:"scatter3d",mode:"markers",x:rows.map(row=>row.open_threshold),y:rows.map(row=>row.capital),z:rows.map(row=>row.open_bar),customdata:rows.map(row=>row.selection_key),text:text,hovertemplate:"%{text}<extra></extra>",marker:{size:5,color:rows.map(row=>row.capital),colorscale:"Viridis",opacity:0.95,colorbar:{title:"capital"}}}],{margin:{l:0,r:0,t:10,b:0},paper_bgcolor:"rgba(0,0,0,0)",scene:{xaxis:{title:cleanLabelFinal("open_threshold"),gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},yaxis:{title:"capital",gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"},zaxis:{title:cleanLabelFinal("open_bar"),gridcolor:"#e8eef8",zerolinecolor:"#e8eef8",backgroundcolor:"#fff"}},showlegend:false},{responsive:true,displayModeBar:false});if(chart.removeAllListeners){chart.removeAllListeners("plotly_click")}chart.on("plotly_click",evt=>{const key=evt?.points?.[0]?.customdata;const record=recordByKey(key);if(record){setRecord(record,true)}})}
function repairChineseLabelsFinal(){const sortNode=$("overviewSortMode");if(sortNode&&sortNode.options.length>=3){sortNode.options[0].text="当前序号";sortNode.options[1].text="收益排序";sortNode.options[2].text="最小回撤排序"}const sortLabel=document.querySelector(".overview-sort-label");if(sortLabel){sortLabel.textContent="排序"}const overviewTitle=document.querySelector("#pageOverview .charttitle");if(overviewTitle){overviewTitle.textContent="收益总览"}const overview3dTitle=document.querySelector("#overview3dChart")?.closest(".chartbox")?.querySelector(".charttitle");if(overview3dTitle){overview3dTitle.textContent="参数三维图"}syncProgramSpecificLabels()}
repairChineseLabelsFinal();
bootstrapPreset();
</script>'''
)


def run_dashboard_server() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    preset_root = _preset_root()
    print(f"[Dashboard] project root: {PROJECT_ROOT}")
    if preset_root is not None:
        print(f"[Dashboard] preset result dir: {preset_root}")
    print(f"[Dashboard] open: {url}")
    print("[Dashboard] stop: Ctrl+C")
    if AUTO_OPEN_BROWSER:
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"[Dashboard] browser open failed: {exc}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard] stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_dashboard_server()


