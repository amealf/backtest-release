# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:
    raise SystemExit("plotly is required. Install it with: pip install plotly") from exc

try:
    from arch import arch_model
except ImportError as exc:
    raise SystemExit("arch is required. Install it with: pip install arch") from exc

import os as _os

sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from backtest_main import load_data


start_time = time.time()

# ============================================================
# User Config
# ============================================================
data_folder_path = r"D:\Code\ibkr\data\SImain\\"
data_file_name = "SImain"
source_timezone = "Asia/Shanghai"
exchange_timezone = "America/Chicago"
direct_period_timezone_mode = "exchange_local"

warmup_start_date = ""
export_start_date = ""
export_end_date = ""

resample_rules = ["day", "5min", "15min", "30min"]

garch_window_bars = 10000
garch_window_bars_by_period = {"day": 500}
refit_every_bars = 100
garch_p = 1
garch_q = 1
garch_dist = "t"
return_scale = 100.0
validation_quantiles = 10


MODEL_SPEC = "garch11_t"
GAP_MULTIPLIER = 1.5
OVERVIEW_MAX_POINTS = 3000
trade_calendar_path = Path(r"D:\Code\data\trade_day\cme_comex_trade_calendar_2020_2026.xlsx")
EXPECTED_SLOT_FREQ_THRESHOLD = 0.55
SUSPICIOUS_GAP_COLOR = "rgba(220, 60, 60, 0.12)"
CANDLE_UP_EDGE = "rgba(185, 185, 185, 0.9)"
CANDLE_DOWN_EDGE = "rgba(85, 85, 85, 0.9)"
CANDLE_UP_FILL = "rgba(245, 245, 245, 0.9)"
CANDLE_DOWN_FILL = "rgba(120, 120, 120, 0.9)"
shock_score_green_threshold = 3.0
shock_score_blue_threshold = 4.0
DIRECT_PERIOD_FILE_SPECS = {
    "1min": {"suffix": "1_min", "bar_seconds": 60, "period_label": "1min"},
    "5min": {"suffix": "5_mins", "bar_seconds": 300, "period_label": "5min"},
    "15min": {"suffix": "15_mins", "bar_seconds": 900, "period_label": "15min"},
    "30min": {"suffix": "30_mins", "bar_seconds": 1800, "period_label": "30min"},
    "day": {"suffix": "1_day", "bar_seconds": 86400, "period_label": "day"},
}


_trade_calendar_cache: pd.DataFrame | None = None


class ProgressBar:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.width = 28
        self.last_percent = -1
        self.last_current = -1

    def update(self, current: int) -> None:
        current_value = min(max(0, int(current)), self.total)
        percent = int((current_value * 100) / self.total)
        if percent == self.last_percent and current_value != self.total:
            return

        filled = int(round((current_value / self.total) * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        print(
            f"\r[GARCH] {self.label} [{bar}] {percent:3d}% "
            f"({current_value}/{self.total})",
            end="",
            flush=True,
        )
        self.last_percent = percent
        self.last_current = current_value

    def close(self) -> None:
        self.update(self.total)
        print("")


def compress_overview_frame(frame: pd.DataFrame, max_points: int = OVERVIEW_MAX_POINTS) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame.reset_index(drop=True).copy()

    bucket_size = int(math.ceil(len(frame) / float(max_points)))
    bucket_id = np.arange(len(frame)) // bucket_size
    agg_map = {
        "Date": ("Date", "first"),
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
    }
    if "garch_sigma_return" in frame.columns:
        agg_map["garch_sigma_return"] = ("garch_sigma_return", "mean")
    if "actual_abs_next_log_return" in frame.columns:
        agg_map["actual_abs_next_log_return"] = ("actual_abs_next_log_return", "mean")
    if "shock_score" in frame.columns:
        agg_map["shock_score"] = ("shock_score", "mean")

    compressed = (
        frame.assign(_bucket_id=bucket_id)
        .groupby("_bucket_id", as_index=False)
        .agg(**agg_map)
    )
    return compressed.reset_index(drop=True)


def load_trade_calendar_df() -> pd.DataFrame:
    global _trade_calendar_cache
    if _trade_calendar_cache is not None:
        return _trade_calendar_cache.copy()

    if not trade_calendar_path.exists():
        raise FileNotFoundError(
            "Trade calendar file not found: "
            + str(trade_calendar_path)
        )

    calendar_df = pd.read_excel(trade_calendar_path, sheet_name="calendar")
    required_columns = {
        "calendar_date",
        "weekday_name",
        "is_weekend",
        "is_holiday",
        "is_trade_day",
        "session_status",
        "holiday_name",
    }
    missing_columns = sorted(required_columns.difference(calendar_df.columns))
    if missing_columns:
        raise ValueError(
            "Trade calendar file missing columns: "
            + ", ".join(missing_columns)
        )

    calendar_df = calendar_df.copy()
    calendar_df["calendar_date"] = pd.to_datetime(
        calendar_df["calendar_date"],
        errors="coerce",
    )
    calendar_df = calendar_df.dropna(subset=["calendar_date"]).reset_index(drop=True)
    _trade_calendar_cache = calendar_df.copy()
    return calendar_df.copy()


def build_slot_expectation_map(date_series: pd.Series) -> dict[int, set[int]]:
    frame = pd.DataFrame({"Date": pd.to_datetime(date_series, errors="coerce")})
    frame = frame.dropna(subset=["Date"]).copy()
    if frame.empty:
        return {}

    frame["calendar_date"] = frame["Date"].dt.normalize()
    frame["weekday"] = frame["Date"].dt.weekday
    frame["slot_minute"] = frame["Date"].dt.hour * 60 + frame["Date"].dt.minute
    active_dates = frame[["calendar_date", "weekday"]].drop_duplicates()
    weekday_date_count = active_dates.groupby("weekday")["calendar_date"].nunique()
    slot_counts = (
        frame[["calendar_date", "weekday", "slot_minute"]]
        .drop_duplicates()
        .groupby(["weekday", "slot_minute"])
        .size()
    )

    expected_slots: dict[int, set[int]] = {}
    for (weekday, slot_minute), slot_count in slot_counts.items():
        total_dates = int(weekday_date_count.get(weekday, 0))
        if total_dates <= 0:
            continue
        if float(slot_count) / float(total_dates) >= EXPECTED_SLOT_FREQ_THRESHOLD:
            expected_slots.setdefault(int(weekday), set()).add(int(slot_minute))
    return expected_slots


def build_date_envelopes(date_series: pd.Series) -> dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]]:
    frame = pd.DataFrame({"Date": pd.to_datetime(date_series, errors="coerce")})
    frame = frame.dropna(subset=["Date"]).copy()
    if frame.empty:
        return {}

    frame["calendar_date"] = frame["Date"].dt.normalize()
    grouped = frame.groupby("calendar_date")["Date"].agg(["min", "max"]).reset_index()
    return {
        pd.Timestamp(row["calendar_date"]): (
            pd.Timestamp(row["min"]),
            pd.Timestamp(row["max"]),
        )
        for _, row in grouped.iterrows()
    }


def build_calendar_trade_day_map() -> dict[pd.Timestamp, int]:
    calendar_df = load_trade_calendar_df()
    calendar_df = calendar_df.copy()
    calendar_df["calendar_date"] = pd.to_datetime(calendar_df["calendar_date"], errors="coerce")
    calendar_df["is_trade_day"] = (
        pd.to_numeric(calendar_df["is_trade_day"], errors="coerce").fillna(1).astype(int)
    )
    return {
        pd.Timestamp(row["calendar_date"]).normalize(): int(row["is_trade_day"])
        for _, row in calendar_df.iterrows()
        if pd.notna(row["calendar_date"])
    }


def aggregate_gap_intervals(
    gap_values: list[pd.Timestamp],
    bar_seconds: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not gap_values:
        return []

    sorted_values = sorted(set(pd.Timestamp(value) for value in gap_values))
    step = pd.Timedelta(seconds=int(bar_seconds))
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    block_start = sorted_values[0]
    previous_value = sorted_values[0]

    for current_value in sorted_values[1:]:
        if current_value - previous_value != step:
            intervals.append((block_start, previous_value + step))
            block_start = current_value
        previous_value = current_value

    intervals.append((block_start, previous_value + step))
    return intervals


def build_gap_controls(
    full_frame: pd.DataFrame,
    displayed_dates: pd.Series,
    bar_seconds: int,
) -> tuple[list[dict], list[tuple[pd.Timestamp, pd.Timestamp]]]:
    full_dates = pd.to_datetime(full_frame["Date"], errors="coerce").dropna().drop_duplicates().sort_values()
    plotted_dates = pd.to_datetime(displayed_dates, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(plotted_dates) <= 1:
        return [], []

    all_available_ts = set(pd.Timestamp(value) for value in full_dates.tolist())
    date_envelopes = build_date_envelopes(full_dates)
    expected_slots_by_weekday = build_slot_expectation_map(full_dates)
    trade_day_map = build_calendar_trade_day_map()
    step = pd.Timedelta(seconds=int(bar_seconds))

    normal_gap_values: list[pd.Timestamp] = []
    suspicious_gap_values: list[pd.Timestamp] = []

    plotted_values = [pd.Timestamp(value) for value in plotted_dates.tolist()]
    for prev_dt, next_dt in zip(plotted_values[:-1], plotted_values[1:]):
        if next_dt - prev_dt <= step:
            continue

        missing_values = pd.date_range(
            start=prev_dt + step,
            end=next_dt - step,
            freq=step,
        )
        for missing_dt in missing_values:
            missing_ts = pd.Timestamp(missing_dt)
            if missing_ts in all_available_ts:
                normal_gap_values.append(missing_ts)
                continue

            calendar_date = missing_ts.normalize()
            weekday = int(missing_ts.weekday())
            slot_minute = int(missing_ts.hour * 60 + missing_ts.minute)
            expected_slots = expected_slots_by_weekday.get(weekday, set())
            is_expected_slot = slot_minute in expected_slots

            if not is_expected_slot:
                normal_gap_values.append(missing_ts)
                continue

            is_trade_day = int(trade_day_map.get(calendar_date, 1))
            prev_trade_flag = int(trade_day_map.get(calendar_date - pd.Timedelta(days=1), 1))
            next_trade_flag = int(trade_day_map.get(calendar_date + pd.Timedelta(days=1), 1))
            day_envelope = date_envelopes.get(calendar_date)

            if is_trade_day == 0:
                normal_gap_values.append(missing_ts)
                continue

            if day_envelope is not None and prev_trade_flag == 0 and missing_ts < day_envelope[0]:
                normal_gap_values.append(missing_ts)
                continue

            if day_envelope is not None and next_trade_flag == 0 and missing_ts > day_envelope[1]:
                normal_gap_values.append(missing_ts)
                continue

            suspicious_gap_values.append(missing_ts)

    rangebreaks = []
    normal_values_sorted = sorted(set(normal_gap_values))
    if normal_values_sorted:
        rangebreaks.append(
            dict(
                values=normal_values_sorted,
                dvalue=int(bar_seconds) * 1000,
            )
        )

    suspicious_intervals = aggregate_gap_intervals(
        suspicious_gap_values,
        bar_seconds=bar_seconds,
    )
    return rangebreaks, suspicious_intervals


def build_standard_kline_trace(
    x_values: pd.Series | np.ndarray,
    plot_df: pd.DataFrame,
    hover_text: pd.Series,
) -> go.Candlestick:
    return go.Candlestick(
        x=x_values,
        open=plot_df["open"],
        high=plot_df["high"],
        low=plot_df["low"],
        close=plot_df["close"],
        name="Kline",
        increasing=dict(
            line=dict(color=CANDLE_UP_EDGE, width=0.8),
            fillcolor=CANDLE_UP_FILL,
        ),
        decreasing=dict(
            line=dict(color=CANDLE_DOWN_EDGE, width=0.8),
            fillcolor=CANDLE_DOWN_FILL,
        ),
        hovertext=hover_text,
        hoverinfo="text",
    )


def build_kline_hover_text(plot_df: pd.DataFrame, extra_lines: list[pd.Series] | None = None) -> pd.Series:
    open_values = pd.to_numeric(plot_df["open"], errors="coerce")
    high_values = pd.to_numeric(plot_df["high"], errors="coerce")
    low_values = pd.to_numeric(plot_df["low"], errors="coerce")
    close_values = pd.to_numeric(plot_df["close"], errors="coerce")

    return_pct = np.where(open_values > 0, (close_values / open_values - 1.0) * 100.0, np.nan)
    amplitude_pct = np.where(open_values > 0, (high_values - low_values) / open_values * 100.0, np.nan)

    text = (
        "bar_return_pct="
        + pd.Series(return_pct, index=plot_df.index).map(
            lambda x: "nan" if pd.isna(x) else f"{float(x):.4f}%"
        )
        + "<br>bar_range_pct="
        + pd.Series(amplitude_pct, index=plot_df.index).map(
            lambda x: "nan" if pd.isna(x) else f"{float(x):.4f}%"
        )
        + "<br>Date="
        + plot_df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        + "<br>open="
        + open_values.map(lambda x: f"{float(x):.6f}")
        + "<br>high="
        + high_values.map(lambda x: f"{float(x):.6f}")
        + "<br>low="
        + low_values.map(lambda x: f"{float(x):.6f}")
        + "<br>close="
        + close_values.map(lambda x: f"{float(x):.6f}")
    )

    if extra_lines:
        for line in extra_lines:
            text = text + line
    return text


def build_numeric_time_axis(date_series: pd.Series, tick_count: int = 8) -> dict:
    labels = pd.to_datetime(date_series, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    values = np.arange(len(labels), dtype=float)
    if len(labels) == 0:
        return {
            "type": "linear",
            "tickmode": "array",
            "tickvals": [],
            "ticktext": [],
            "showgrid": False,
            "zeroline": False,
            "rangeslider": dict(visible=False),
        }

    if len(labels) <= tick_count:
        tick_index = list(range(len(labels)))
    else:
        step = max(1, int(math.ceil(len(labels) / float(tick_count))))
        tick_index = list(range(0, len(labels), step))
        if tick_index[-1] != len(labels) - 1:
            tick_index.append(len(labels) - 1)

    tick_vals = [float(idx) for idx in tick_index]
    tick_text = [labels.iloc[idx] for idx in tick_index]
    return {
        "type": "linear",
        "tickmode": "array",
        "tickvals": tick_vals,
        "ticktext": tick_text,
        "showgrid": False,
        "zeroline": False,
        "rangeslider": dict(visible=False),
    }


def build_suspicious_gap_positions(
    displayed_dates: pd.Series,
    suspicious_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[float]:
    if not suspicious_intervals:
        return []

    displayed_index = pd.Index(pd.to_datetime(displayed_dates, errors="coerce"))
    positions: list[float] = []
    for _, suspicious_end in suspicious_intervals:
        insert_pos = int(displayed_index.searchsorted(pd.Timestamp(suspicious_end), side="left"))
        if 0 < insert_pos < len(displayed_index):
            positions.append(float(insert_pos) - 0.5)
    return sorted(set(positions))


def normalize_rule_token(rule: str) -> str:
    text = str(rule or "").strip().lower()
    compact = text.replace(" ", "").replace("_", "")
    alias_map = {
        "d": "day",
        "1d": "day",
        "1day": "day",
        "day": "day",
        "1m": "1min",
        "1min": "1min",
        "1mins": "1min",
        "5m": "5min",
        "5min": "5min",
        "5mins": "5min",
        "15m": "15min",
        "15min": "15min",
        "15mins": "15min",
        "30m": "30min",
        "30min": "30min",
        "30mins": "30min",
        "1h": "1h",
        "1hour": "1h",
        "30s": "30s",
    }
    return alias_map.get(compact, text)


def normalize_resample_rule_for_pandas(rule: str) -> str:
    token = normalize_rule_token(rule)
    pandas_map = {
        "day": "1D",
        "1min": "1min",
        "5min": "5min",
        "15min": "15min",
        "30min": "30min",
        "1h": "1H",
        "30s": "30s",
    }
    return pandas_map.get(token, str(rule or "").strip())


def detect_bar_seconds_from_df(df: pd.DataFrame) -> int:
    dates = pd.to_datetime(df["Date"], errors="coerce")
    diffs = dates.diff().dropna()
    if len(diffs) > 50:
        diffs = diffs.iloc[:50]
    median_delta = diffs.median()
    if pd.isna(median_delta):
        raise ValueError("Cannot detect bar period from Date column.")
    total_seconds = int(median_delta.total_seconds())
    if total_seconds <= 0:
        raise ValueError(f"Invalid detected bar period: {total_seconds}")
    return total_seconds


def resample_ohlc_df(df: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, int]:
    resample_rule = normalize_resample_rule_for_pandas(rule)
    if not resample_rule:
        return df.copy(), detect_bar_seconds_from_df(df)

    temp = df.copy()
    temp["Date"] = pd.to_datetime(temp["Date"], errors="coerce")
    temp = temp.dropna(subset=["Date"]).sort_values("Date")
    temp = temp.set_index("Date")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "vol" in temp.columns:
        agg["vol"] = "sum"
    temp = temp.resample(resample_rule).agg(agg)
    temp = temp.dropna(subset=["open", "high", "low", "close"]).reset_index()
    temp["Date"] = pd.to_datetime(temp["Date"], errors="coerce")
    if "vol" not in temp.columns:
        temp["vol"] = 0.0
    return temp, detect_bar_seconds_from_df(temp)


def format_period_label(resample_rule: str, bar_seconds: int) -> str:
    token = normalize_rule_token(resample_rule)
    if token in DIRECT_PERIOD_FILE_SPECS:
        return DIRECT_PERIOD_FILE_SPECS[token]["period_label"]

    rule = normalize_resample_rule_for_pandas(resample_rule)
    if rule:
        return rule.replace(" ", "")
    if bar_seconds % 3600 == 0:
        hours = bar_seconds // 3600
        return f"{hours}h"
    if bar_seconds % 60 == 0:
        minutes = bar_seconds // 60
        return f"{minutes}min"
    return f"{bar_seconds}s"


def parse_selection_datetime(value, is_end: bool = False) -> pd.Timestamp:
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        ts = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Invalid date value: {value}")
        if is_end:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return ts

    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date value: {value}")
    return ts


def should_drop_incomplete_initial_resampled_bar(
    raw_df: pd.DataFrame,
    resample_rule: str,
) -> bool:
    rule = normalize_resample_rule_for_pandas(resample_rule)
    if not rule or raw_df.empty:
        return False

    first_ts = pd.to_datetime(raw_df["Date"].iloc[0], errors="coerce")
    if pd.isna(first_ts):
        return False

    try:
        return first_ts != first_ts.floor(rule)
    except Exception:
        return False


def build_output_dir(file_name: str) -> Path:
    out_dir = Path("./result") / f"{file_name} long outcome" / "garch forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalize_config_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def convert_series_to_exchange_timezone(
    series: pd.Series,
    from_timezone: str,
    to_timezone: str,
) -> pd.Series:
    dt_series = pd.to_datetime(series, errors="coerce")
    if normalize_config_text(to_timezone) == "":
        return dt_series
    if getattr(dt_series.dt, "tz", None) is None:
        if normalize_config_text(from_timezone) == "":
            return dt_series
        dt_series = dt_series.dt.tz_localize(
            from_timezone,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    return dt_series.dt.tz_convert(to_timezone).dt.tz_localize(None)


def parse_direct_period_datetime_series(
    series: pd.Series,
    timezone_mode: str,
) -> pd.Series:
    mode = normalize_config_text(timezone_mode).lower()
    dt_series = pd.to_datetime(series, errors="coerce")
    if mode in {"", "exchange_local", "exchange"}:
        return dt_series
    if mode in {"source", "source_timezone"}:
        return convert_series_to_exchange_timezone(
            series,
            from_timezone=source_timezone,
            to_timezone=exchange_timezone,
        )
    raise ValueError("Unsupported direct_period_timezone_mode: " + str(timezone_mode))


def resolve_garch_window_bars(period_label: str) -> int:
    token = normalize_rule_token(period_label)
    return int(garch_window_bars_by_period.get(token, garch_window_bars))


def build_run_config(period_label: str) -> dict:
    return {
        "data_folder_path": normalize_config_text(data_folder_path),
        "data_file_name": normalize_config_text(data_file_name),
        "source_timezone": normalize_config_text(source_timezone),
        "exchange_timezone": normalize_config_text(exchange_timezone),
        "direct_period_timezone_mode": normalize_config_text(direct_period_timezone_mode),
        "warmup_start_date": normalize_config_text(warmup_start_date),
        "export_start_date": normalize_config_text(export_start_date),
        "export_end_date": normalize_config_text(export_end_date),
        "period_label": normalize_config_text(period_label),
        "garch_window_bars": str(int(resolve_garch_window_bars(period_label))),
        "garch_window_bars_base": str(int(garch_window_bars)),
        "garch_window_bars_by_period": normalize_config_text(garch_window_bars_by_period),
        "refit_every_bars": str(int(refit_every_bars)),
        "garch_p": str(int(garch_p)),
        "garch_q": str(int(garch_q)),
        "garch_dist": normalize_config_text(garch_dist),
        "return_scale": normalize_config_text(return_scale),
        "validation_quantiles": str(int(validation_quantiles)),
        "shock_score_green_threshold": normalize_config_text(shock_score_green_threshold),
        "shock_score_blue_threshold": normalize_config_text(shock_score_blue_threshold),
        "model_spec": MODEL_SPEC,
    }


def config_matches_file(config_df: pd.DataFrame, expected_config: dict) -> bool:
    if config_df.empty or "key" not in config_df.columns or "value" not in config_df.columns:
        return False

    config_map = {
        normalize_config_text(row["key"]): normalize_config_text(row["value"])
        for _, row in config_df.iterrows()
    }
    for key, expected_value in expected_config.items():
        if config_map.get(key, "") != normalize_config_text(expected_value):
            return False
    return True


def resolve_range_boundaries(
    date_series: pd.Series,
    start_value,
    end_value,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    valid_dates = pd.to_datetime(date_series, errors="coerce").dropna()
    if valid_dates.empty:
        raise RuntimeError("Date series is empty.")

    start_ts = (
        parse_selection_datetime(start_value, is_end=False)
        if normalize_config_text(start_value)
        else pd.Timestamp(valid_dates.min())
    )
    end_ts = (
        parse_selection_datetime(end_value, is_end=True)
        if normalize_config_text(end_value)
        else pd.Timestamp(valid_dates.max())
    )
    return start_ts, end_ts


def build_period_file_stem(period_label: str) -> str:
    return f"period_{period_label} garch forecast"


def build_period_parquet_name(period_label: str) -> str:
    return f"{build_period_file_stem(period_label)}.parquet"


def build_period_summary_excel_name(period_label: str) -> str:
    return f"period_{period_label} garch summary.xlsx"


def build_period_overview_html_name(period_label: str) -> str:
    return f"period_{period_label} garch overview interactive.html"


def build_period_vol_surprise_html_name(period_label: str) -> str:
    return f"period_{period_label} garch vol_surprise interactive.html"


def normalize_forecast_frame(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    frame = df.copy()
    if "Date" not in frame.columns:
        return pd.DataFrame()

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).copy()
    frame["period_label"] = period_label
    frame = frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return frame.reset_index(drop=True)


def normalize_refit_log_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    frame = df.copy()
    if "refit_date" not in frame.columns:
        return pd.DataFrame()

    for col in ["refit_date", "fit_start_date", "fit_end_date"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    frame = frame.dropna(subset=["refit_date"]).copy()
    frame = frame.sort_values("refit_date").drop_duplicates(subset=["refit_date"], keep="last")
    return frame.reset_index(drop=True)


def cleanup_stale_period_artifacts(out_dir: Path, period_label: str) -> None:
    stale_names = [
        f"period_{period_label} garch validation interactive.html",
        f"2025 period_{period_label} garch validation interactive.html",
        f"period_{period_label} garch overview.png",
        f"period_{period_label} garch overview.html",
    ]
    for file_name in stale_names:
        path = out_dir / file_name
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass


def cleanup_all_stale_html_artifacts(out_dir: Path) -> None:
    stale_patterns = [
        "*garch validation interactive.html",
        "*garch overview.png",
        "period_* garch overview.html",
    ]
    for pattern in stale_patterns:
        for path in out_dir.glob(pattern):
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass


def read_existing_period_outputs(
    out_dir: Path,
    period_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    expected_config = build_run_config(period_label)
    forecast_frames = []
    refit_frames = []
    source_names = []

    summary_excel_path = out_dir / build_period_summary_excel_name(period_label)
    parquet_path = out_dir / build_period_parquet_name(period_label)
    legacy_excel_path = out_dir / f"{build_period_file_stem(period_label)}.xlsx"

    summary_path_for_config = None
    if summary_excel_path.exists():
        summary_path_for_config = summary_excel_path
    elif legacy_excel_path.exists():
        summary_path_for_config = legacy_excel_path

    config_ok = False
    if summary_path_for_config is not None:
        try:
            config_df = pd.read_excel(summary_path_for_config, sheet_name="run_config")
            config_ok = config_matches_file(config_df, expected_config)
        except Exception:
            config_ok = False

    if config_ok and parquet_path.exists():
        try:
            forecast_df = pd.read_parquet(parquet_path)
            forecast_df = normalize_forecast_frame(forecast_df, period_label)
            if not forecast_df.empty:
                forecast_frames.append(forecast_df)
                source_names.append(parquet_path.name)
        except Exception:
            pass

    if config_ok and not forecast_frames and legacy_excel_path.exists():
        try:
            legacy_forecast_df = pd.read_excel(legacy_excel_path, sheet_name="forecast")
            legacy_forecast_df = normalize_forecast_frame(legacy_forecast_df, period_label)
            if not legacy_forecast_df.empty:
                forecast_frames.append(legacy_forecast_df)
                source_names.append(legacy_excel_path.name)
        except Exception:
            pass

    if config_ok and summary_path_for_config is not None:
        try:
            refit_df = pd.read_excel(summary_path_for_config, sheet_name="refit_log")
            refit_df = normalize_refit_log_frame(refit_df)
            if not refit_df.empty:
                refit_frames.append(refit_df)
                if summary_path_for_config.name not in source_names:
                    source_names.append(summary_path_for_config.name)
        except Exception:
            pass

    if forecast_frames:
        existing_forecast_df = pd.concat(forecast_frames, ignore_index=True)
        existing_forecast_df = normalize_forecast_frame(existing_forecast_df, period_label)
    else:
        existing_forecast_df = pd.DataFrame()

    if refit_frames:
        existing_refit_log_df = pd.concat(refit_frames, ignore_index=True)
        existing_refit_log_df = normalize_refit_log_frame(existing_refit_log_df)
    else:
        existing_refit_log_df = pd.DataFrame()

    return existing_forecast_df, existing_refit_log_df, source_names


def load_legacy_raw_data() -> pd.DataFrame:
    raw_df, _, _ = load_data(data_folder_path, data_file_name)
    raw_df["Date"] = convert_series_to_exchange_timezone(
        raw_df["Date"],
        from_timezone=source_timezone,
        to_timezone=exchange_timezone,
    )
    raw_df = raw_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    warmup_start_ts, export_end_ts = resolve_range_boundaries(
        raw_df["Date"],
        warmup_start_date,
        export_end_date,
    )
    raw_df = raw_df[
        (raw_df["Date"] >= warmup_start_ts) & (raw_df["Date"] <= export_end_ts)
    ].reset_index(drop=True)
    if raw_df.empty:
        raise RuntimeError("No source rows remain after warmup/export date filtering.")
    return raw_df


def try_load_direct_period_data(
    period_rule: str,
) -> tuple[pd.DataFrame, int, str, Path] | None:
    token = normalize_rule_token(period_rule)
    spec = DIRECT_PERIOD_FILE_SPECS.get(token)
    if spec is None:
        return None

    csv_path = Path(data_folder_path) / f"{data_file_name}_{spec['suffix']}.csv"
    if not csv_path.exists():
        return None

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise RuntimeError("Direct period file is empty: " + str(csv_path))

    lower_map = {str(col).strip().lower(): col for col in frame.columns}
    date_col = None
    for name in ("datetime", "date", "time"):
        if name in lower_map:
            date_col = lower_map[name]
            break
    required_cols = {}
    for target in ("open", "high", "low", "close"):
        source_col = lower_map.get(target)
        if source_col is None:
            raise ValueError(
                "Direct period file missing column: "
                + target
                + " | "
                + str(csv_path)
            )
        required_cols[target] = source_col
    volume_col = lower_map.get("volume") or lower_map.get("vol")
    if date_col is None:
        raise ValueError("Direct period file missing datetime column: " + str(csv_path))

    normalized = pd.DataFrame({
        "Date": parse_direct_period_datetime_series(
            frame[date_col],
            timezone_mode=direct_period_timezone_mode,
        ),
        "open": pd.to_numeric(frame[required_cols["open"]], errors="coerce"),
        "high": pd.to_numeric(frame[required_cols["high"]], errors="coerce"),
        "low": pd.to_numeric(frame[required_cols["low"]], errors="coerce"),
        "close": pd.to_numeric(frame[required_cols["close"]], errors="coerce"),
    })
    if volume_col is not None:
        normalized["vol"] = pd.to_numeric(frame[volume_col], errors="coerce")
    else:
        normalized["vol"] = 0.0

    normalized = normalized.dropna(
        subset=["Date", "open", "high", "low", "close"]
    ).sort_values("Date").reset_index(drop=True)
    if normalized.empty:
        raise RuntimeError("No valid rows remain in direct period file: " + str(csv_path))

    warmup_start_ts, export_end_ts = resolve_range_boundaries(
        normalized["Date"],
        warmup_start_date,
        export_end_date,
    )
    normalized = normalized[
        (normalized["Date"] >= warmup_start_ts)
        & (normalized["Date"] <= export_end_ts)
    ].reset_index(drop=True)
    if normalized.empty:
        raise RuntimeError("No direct period rows remain after date filtering: " + str(csv_path))

    return normalized, int(spec["bar_seconds"]), str(spec["period_label"]), csv_path


def finalize_period_bars(
    preview_df: pd.DataFrame,
    bar_seconds: int,
    period_label: str,
) -> tuple[pd.DataFrame, int, str]:
    preview_df = preview_df.copy()
    preview_df["Date"] = pd.to_datetime(preview_df["Date"], errors="coerce")
    preview_df = preview_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    preview_df["prev_close"] = pd.to_numeric(preview_df["close"], errors="coerce").shift(1)
    preview_df["log_return"] = np.log(preview_df["close"] / preview_df["prev_close"])

    gap_seconds = preview_df["Date"].diff().dt.total_seconds()
    preview_df.loc[gap_seconds > (bar_seconds * GAP_MULTIPLIER), "log_return"] = np.nan

    next_gap_seconds = preview_df["Date"].shift(-1).sub(preview_df["Date"]).dt.total_seconds()
    next_gap_ok = next_gap_seconds.le(bar_seconds * GAP_MULTIPLIER)
    preview_df["actual_abs_next_log_return"] = preview_df["log_return"].shift(-1).abs()
    preview_df.loc[~next_gap_ok.fillna(False), "actual_abs_next_log_return"] = np.nan
    return preview_df, int(bar_seconds), str(period_label)


def load_period_bars(
    period_rule: str,
    legacy_raw_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, int, str, str, pd.DataFrame | None]:
    direct_loaded = try_load_direct_period_data(period_rule)
    if direct_loaded is not None:
        direct_df, bar_seconds, period_label, csv_path = direct_loaded
        bars_df, bar_seconds, period_label = finalize_period_bars(
            direct_df,
            bar_seconds=bar_seconds,
            period_label=period_label,
        )
        return bars_df, bar_seconds, period_label, csv_path.name, legacy_raw_df

    if legacy_raw_df is None:
        legacy_raw_df = load_legacy_raw_data()
    bars_df, bar_seconds, period_label = prepare_period_bars(
        legacy_raw_df,
        period_rule,
    )
    return bars_df, bar_seconds, period_label, "legacy_single_csv", legacy_raw_df


def prepare_period_bars(raw_df: pd.DataFrame, resample_rule: str) -> tuple[pd.DataFrame, int, str]:
    native_preview_df = raw_df.copy().reset_index(drop=True)
    if (resample_rule or "").strip():
        preview_df, bar_seconds = resample_ohlc_df(native_preview_df, resample_rule)
        if (
            len(preview_df) > 0
            and should_drop_incomplete_initial_resampled_bar(native_preview_df, resample_rule)
        ):
            preview_df = preview_df.iloc[1:].reset_index(drop=True)
    else:
        preview_df = native_preview_df.copy()
        bar_seconds = detect_bar_seconds_from_df(preview_df)

    period_label = format_period_label(resample_rule, bar_seconds)
    return finalize_period_bars(
        preview_df,
        bar_seconds=bar_seconds,
        period_label=period_label,
    )


def extract_float_param(params: pd.Series, *keys: str, default=np.nan) -> float:
    for key in keys:
        if key in params.index:
            return float(params[key])
    return float(default)


def fit_garch_window(window_returns_scaled: pd.Series) -> dict:
    model = arch_model(
        window_returns_scaled,
        mean="Zero",
        vol="GARCH",
        p=garch_p,
        q=garch_q,
        dist=garch_dist,
        rescale=False,
    )
    result = model.fit(disp="off", show_warning=False)
    convergence_flag = int(getattr(result, "convergence_flag", 0))
    if convergence_flag != 0:
        raise RuntimeError(f"GARCH fit did not converge. convergence_flag={convergence_flag}")

    params = result.params
    omega = extract_float_param(params, "omega")
    alpha_1 = extract_float_param(params, "alpha[1]", "alpha1")
    beta_1 = extract_float_param(params, "beta[1]", "beta1")
    nu = extract_float_param(params, "nu")

    if not np.isfinite(omega) or not np.isfinite(alpha_1) or not np.isfinite(beta_1):
        raise RuntimeError("GARCH parameters contain non-finite values.")

    sigma2_current = float(result.conditional_volatility.iloc[-1]) ** 2
    if not np.isfinite(sigma2_current):
        raise RuntimeError("Current conditional variance is not finite.")

    return {
        "omega": float(omega),
        "alpha_1": float(alpha_1),
        "beta_1": float(beta_1),
        "nu": float(nu),
        "sigma2_current": float(sigma2_current),
    }


def assign_quantile_bucket(series: pd.Series, quantiles: int) -> pd.Series:
    frame = series.dropna()
    if frame.empty:
        return pd.Series(index=series.index, dtype="float64")

    q_value = int(min(max(1, quantiles), len(frame)))
    ranked = frame.rank(method="first")
    bucket = pd.qcut(ranked, q=q_value, labels=False) + 1
    out = pd.Series(index=series.index, dtype="float64")
    out.loc[frame.index] = bucket.astype(float)
    return out


def build_forecast_df(
    bars_df: pd.DataFrame,
    period_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    work = bars_df.copy().reset_index(drop=True)
    row_count = len(work)
    effective_window_bars = int(resolve_garch_window_bars(period_label))

    garch_sigma_return = np.full(row_count, np.nan, dtype=float)
    garch_sigma_return_pct = np.full(row_count, np.nan, dtype=float)
    garch_sigma_price = np.full(row_count, np.nan, dtype=float)
    refit_executed = np.zeros(row_count, dtype=int)
    converged_flag = np.zeros(row_count, dtype=int)
    model_spec_values = np.full(row_count, MODEL_SPEC, dtype=object)

    export_start_ts, export_end_ts = resolve_range_boundaries(
        work["Date"],
        export_start_date,
        export_end_date,
    )
    export_mask = (work["Date"] >= export_start_ts) & (work["Date"] <= export_end_ts)

    out_dir = build_output_dir(data_file_name)
    existing_forecast_df, existing_refit_log_df, existing_sources = read_existing_period_outputs(
        out_dir=out_dir,
        period_label=period_label,
    )
    existing_export_dates = set()
    if not existing_forecast_df.empty:
        existing_range_df = existing_forecast_df[
            (existing_forecast_df["Date"] >= export_start_ts)
            & (existing_forecast_df["Date"] <= export_end_ts)
        ].copy()
        existing_export_dates = set(existing_range_df["Date"].tolist())

    missing_export_mask = export_mask & ~work["Date"].isin(existing_export_dates)
    missing_export_positions = np.flatnonzero(missing_export_mask.to_numpy())

    if len(missing_export_positions) == 0:
        skip_meta = {
            "existing_forecast_df": existing_forecast_df,
            "existing_refit_log_df": existing_refit_log_df,
            "existing_sources": existing_sources,
            "skipped": True,
            "export_row_count": int(export_mask.sum()),
            "new_row_count": 0,
            "existing_row_count": int(len(existing_forecast_df)),
        }
        return pd.DataFrame(), pd.DataFrame(), skip_meta

    valid_positions = np.flatnonzero(work["log_return"].notna().to_numpy())
    active_params: dict | None = None
    sigma2_current = np.nan
    refit_rows: list[dict] = []
    total_refits = 0
    compute_end_pos = int(missing_export_positions.max())
    compute_positions = valid_positions[valid_positions <= compute_end_pos]
    progress_bar = ProgressBar(period_label, len(compute_positions))
    processed_count = 0
    progress_bar.update(0)

    for valid_idx, row_pos in enumerate(valid_positions):
        if row_pos > compute_end_pos:
            break

        row_date = work.at[row_pos, "Date"]
        row_return = float(work.at[row_pos, "log_return"])
        row_return_scaled = row_return * float(return_scale)

        should_refit = (
            (valid_idx + 1) >= effective_window_bars
            and ((valid_idx + 1 - effective_window_bars) % int(refit_every_bars) == 0)
        )

        if should_refit:
            refit_executed[row_pos] = 1
            total_refits += 1
            window_positions = valid_positions[
                valid_idx - effective_window_bars + 1: valid_idx + 1
            ]
            fit_start_date = work.at[int(window_positions[0]), "Date"]
            fit_end_date = work.at[int(window_positions[-1]), "Date"]
            window_returns_scaled = (
                work.loc[window_positions, "log_return"]
                .astype(float)
                .reset_index(drop=True)
                * float(return_scale)
            )

            refit_row = {
                "refit_date": row_date,
                "fit_start_date": fit_start_date,
                "fit_end_date": fit_end_date,
                "fit_window_bars": effective_window_bars,
                "omega": np.nan,
                "alpha_1": np.nan,
                "beta_1": np.nan,
                "nu": np.nan,
                "converged": 0,
            }

            try:
                fit_result = fit_garch_window(window_returns_scaled)
                active_params = {
                    "omega": fit_result["omega"],
                    "alpha_1": fit_result["alpha_1"],
                    "beta_1": fit_result["beta_1"],
                    "nu": fit_result["nu"],
                }
                sigma2_current = float(fit_result["sigma2_current"])
                refit_row["omega"] = active_params["omega"]
                refit_row["alpha_1"] = active_params["alpha_1"]
                refit_row["beta_1"] = active_params["beta_1"]
                refit_row["nu"] = active_params["nu"]
                refit_row["converged"] = 1
            except Exception as exc:
                refit_row["error"] = str(exc)
            refit_rows.append(refit_row)

        if active_params is None or not np.isfinite(sigma2_current):
            processed_count += 1
            progress_bar.update(processed_count)
            continue

        omega = float(active_params["omega"])
        alpha_1 = float(active_params["alpha_1"])
        beta_1 = float(active_params["beta_1"])

        sigma2_next = omega + alpha_1 * (row_return_scaled ** 2) + beta_1 * sigma2_current
        if np.isfinite(sigma2_next) and sigma2_next >= 0.0:
            sigma_next_return = math.sqrt(float(sigma2_next)) / float(return_scale)
            garch_sigma_return[row_pos] = sigma_next_return
            garch_sigma_return_pct[row_pos] = sigma_next_return * 100.0

            prev_close = pd.to_numeric(work.at[row_pos, "prev_close"], errors="coerce")
            if np.isfinite(prev_close):
                garch_sigma_price[row_pos] = sigma_next_return * float(prev_close)
            converged_flag[row_pos] = 1
            sigma2_current = float(sigma2_next)
        else:
            sigma2_current = np.nan

        processed_count += 1
        progress_bar.update(processed_count)

    progress_bar.close()

    work["garch_sigma_return"] = garch_sigma_return
    work["garch_sigma_return_pct"] = garch_sigma_return_pct
    work["garch_sigma_price"] = garch_sigma_price
    work["refit_executed"] = refit_executed
    work["converged"] = converged_flag
    work["model_spec"] = model_spec_values
    work["period_label"] = period_label

    refit_log_df = pd.DataFrame(refit_rows)
    if not refit_log_df.empty:
        ordered_columns = [
            "refit_date",
            "fit_start_date",
            "fit_end_date",
            "fit_window_bars",
            "omega",
            "alpha_1",
            "beta_1",
            "nu",
            "converged",
        ]
        if "error" in refit_log_df.columns:
            ordered_columns.append("error")
        refit_log_df = refit_log_df[ordered_columns]
    else:
        refit_log_df = pd.DataFrame(
            columns=[
                "refit_date",
                "fit_start_date",
                "fit_end_date",
                "fit_window_bars",
                "omega",
                "alpha_1",
                "beta_1",
                "nu",
                "converged",
            ]
        )

    forecast_df = work[
        missing_export_mask
    ].reset_index(drop=True)

    run_meta = {
        "existing_forecast_df": existing_forecast_df,
        "existing_refit_log_df": existing_refit_log_df,
        "existing_sources": existing_sources,
        "skipped": False,
        "export_row_count": int(export_mask.sum()),
        "new_row_count": int(len(forecast_df)),
        "existing_row_count": int(len(existing_forecast_df)),
        "effective_window_bars": effective_window_bars,
    }

    return forecast_df, refit_log_df, run_meta


def build_validation_tables(
    forecast_df: pd.DataFrame,
    period_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = forecast_df[
        ["Date", "garch_sigma_return", "actual_abs_next_log_return"]
    ].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if frame.empty:
        raw_forecast_count = int(forecast_df["garch_sigma_return"].notna().sum())
        raw_actual_count = int(forecast_df["actual_abs_next_log_return"].notna().sum())
        raise RuntimeError(
            f"No validation rows remain for period {period_label}. "
            f"non_null_forecast={raw_forecast_count}, "
            f"non_null_actual={raw_actual_count}, "
            f"forecast_rows={len(forecast_df)}"
        )

    frame["bucket_id"] = assign_quantile_bucket(frame["garch_sigma_return"], validation_quantiles)
    frame = frame.dropna(subset=["bucket_id"]).copy()
    frame["bucket_id"] = frame["bucket_id"].astype(int)

    validation_decile = (
        frame.groupby("bucket_id", as_index=False)
        .agg(
            sample_count=("bucket_id", "size"),
            forecast_mean=("garch_sigma_return", "mean"),
            forecast_median=("garch_sigma_return", "median"),
            actual_mean=("actual_abs_next_log_return", "mean"),
            actual_median=("actual_abs_next_log_return", "median"),
            actual_std=("actual_abs_next_log_return", "std"),
        )
        .sort_values("bucket_id")
        .reset_index(drop=True)
    )

    bucket_min = int(validation_decile["bucket_id"].min())
    bucket_max = int(validation_decile["bucket_id"].max())
    top_actual_mean = float(
        validation_decile.loc[validation_decile["bucket_id"] == bucket_max, "actual_mean"].iloc[0]
    )
    bottom_actual_mean = float(
        validation_decile.loc[validation_decile["bucket_id"] == bucket_min, "actual_mean"].iloc[0]
    )
    top_bottom_ratio = np.nan
    if np.isfinite(bottom_actual_mean) and abs(bottom_actual_mean) > 1e-18:
        top_bottom_ratio = top_actual_mean / bottom_actual_mean

    pearson_corr = float(
        frame["garch_sigma_return"].corr(frame["actual_abs_next_log_return"], method="pearson")
    )
    spearman_corr = float(
        frame["garch_sigma_return"].corr(frame["actual_abs_next_log_return"], method="spearman")
    )

    validation_summary = pd.DataFrame(
        [
            {
                "period_label": period_label,
                "model_spec": MODEL_SPEC,
                "garch_window_bars": int(resolve_garch_window_bars(period_label)),
                "refit_every_bars": int(refit_every_bars),
                "forecast_rows": int(len(forecast_df)),
                "valid_sample_count": int(len(frame)),
                "pearson_corr": pearson_corr,
                "spearman_corr": spearman_corr,
                "forecast_mean": float(frame["garch_sigma_return"].mean()),
                "actual_mean": float(frame["actual_abs_next_log_return"].mean()),
                "top_quantile_actual_mean": top_actual_mean,
                "bottom_quantile_actual_mean": bottom_actual_mean,
                "top_bottom_actual_ratio": float(top_bottom_ratio)
                if np.isfinite(top_bottom_ratio)
                else np.nan,
            }
        ]
    )

    return validation_summary, validation_decile


def build_validation_html(
    period_label: str,
    validation_summary: pd.DataFrame,
    validation_decile: pd.DataFrame,
    html_path: Path,
) -> None:
    summary_row = validation_summary.iloc[0]
    custom_data = np.column_stack(
        [
            validation_decile["sample_count"].to_numpy(),
            validation_decile["forecast_mean"].to_numpy(),
            validation_decile["forecast_median"].to_numpy(),
            validation_decile["actual_mean"].to_numpy(),
            validation_decile["actual_median"].to_numpy(),
            validation_decile["actual_std"].fillna(np.nan).to_numpy(),
        ]
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=validation_decile["bucket_id"].astype(str),
            y=validation_decile["actual_mean"],
            name="actual_mean",
            marker=dict(color="rgba(31, 119, 180, 0.85)"),
            customdata=custom_data,
            hovertemplate=(
                "bucket=%{x}<br>"
                "sample_count=%{customdata[0]:.0f}<br>"
                "actual_mean=%{customdata[3]:.8f}<br>"
                "actual_median=%{customdata[4]:.8f}<br>"
                "actual_std=%{customdata[5]:.8f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=validation_decile["bucket_id"].astype(str),
            y=validation_decile["forecast_mean"],
            name="forecast_mean",
            mode="lines+markers",
            line=dict(color="rgba(255, 127, 14, 0.95)", width=2),
            marker=dict(size=7),
            customdata=custom_data,
            hovertemplate=(
                "bucket=%{x}<br>"
                "sample_count=%{customdata[0]:.0f}<br>"
                "forecast_mean=%{customdata[1]:.8f}<br>"
                "forecast_median=%{customdata[2]:.8f}<br>"
                "actual_mean=%{customdata[3]:.8f}<br>"
                "actual_median=%{customdata[4]:.8f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=(
            f"GARCH validation | {period_label} | "
            f"valid={int(summary_row['valid_sample_count'])} | "
            f"spearman={summary_row['spearman_corr']:.4f} | "
            f"top/bottom={summary_row['top_bottom_actual_ratio']:.4f}"
        ),
        xaxis=dict(title="forecast quantile bucket"),
        yaxis=dict(title="absolute next log return"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=55, r=30, t=60, b=45),
    )

    html_text = fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        default_width="100vw",
        default_height="100vh",
        config={
            "responsive": True,
            "displayModeBar": False,
            "displaylogo": False,
        },
    )
    html_text = html_text.replace(
        "<head>",
        (
            "<head><style>"
            "html,body{width:100%;height:100%;margin:0;padding:0;overflow:hidden;}"
            ".plotly-graph-div{width:100vw !important;height:100vh !important;}"
            "</style>"
        ),
        1,
    )
    html_text = html_text.replace("<body>", '<body style="margin:0;overflow:hidden;">', 1)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(html_text)


def build_overview_interactive_html(
    forecast_df: pd.DataFrame,
    period_label: str,
    validation_summary: pd.DataFrame,
    out_dir: Path,
) -> Path:
    full_frame = forecast_df[
        [
            "Date",
            "open",
            "high",
            "low",
            "close",
            "garch_sigma_return",
            "actual_abs_next_log_return",
        ]
    ].copy()
    full_frame["Date"] = pd.to_datetime(full_frame["Date"], errors="coerce")
    full_frame = full_frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    frame = full_frame.loc[full_frame["garch_sigma_return"].notna()].copy().reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No rows available for overview plot: {period_label}")

    bar_seconds = detect_bar_seconds_from_df(full_frame)
    plot_df = frame.reset_index(drop=True).copy()
    _, suspicious_intervals = build_gap_controls(
        full_frame=full_frame,
        displayed_dates=plot_df["Date"],
        bar_seconds=bar_seconds,
    )
    plot_x = np.arange(len(plot_df), dtype=float)
    suspicious_positions = build_suspicious_gap_positions(
        displayed_dates=plot_df["Date"],
        suspicious_intervals=suspicious_intervals,
    )
    summary_row = validation_summary.iloc[0]

    hover_text = build_kline_hover_text(
        plot_df=plot_df,
        extra_lines=[
            "<br>pred_vol="
            + plot_df["garch_sigma_return"].map(
                lambda x: "nan" if pd.isna(x) else f"{float(x):.8f}"
            ),
            "<br>real_vol="
            + plot_df["actual_abs_next_log_return"].map(
                lambda x: "nan" if pd.isna(x) else f"{float(x):.8f}"
            ),
        ],
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("Price", "Volatility"),
    )
    fig.add_trace(
        build_standard_kline_trace(
            x_values=plot_x,
            plot_df=plot_df,
            hover_text=hover_text,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_x,
            y=plot_df["garch_sigma_return"],
            mode="lines",
            name="Predicted Vol",
            line=dict(color="rgba(31, 119, 180, 0.50)", width=1.4),
            customdata=plot_df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            hovertemplate=(
                "Date=%{customdata}<br>"
                "pred_vol=%{y:.8f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_x,
            y=plot_df["actual_abs_next_log_return"],
            mode="lines",
            name="Realized Vol",
            line=dict(color="#ff7f0e", width=1.2),
            customdata=plot_df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            hovertemplate=(
                "Date=%{customdata}<br>"
                "real_vol=%{y:.8f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    for suspicious_x in suspicious_positions:
        fig.add_vline(
            x=suspicious_x,
            line=dict(color="rgba(220, 60, 60, 0.45)", width=1, dash="dash"),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=suspicious_x,
            line=dict(color="rgba(220, 60, 60, 0.45)", width=1, dash="dash"),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        title=(
            f"GARCH Overview | {period_label} | "
            f"valid={int(summary_row['valid_sample_count'])} | "
            f"spearman={summary_row['spearman_corr']:.4f} | "
            f"top/bottom={summary_row['top_bottom_actual_ratio']:.4f}"
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.32)",
            bordercolor="rgba(70, 70, 70, 0.18)",
            font=dict(color="rgba(20, 50, 95, 0.98)"),
        ),
        xaxis=build_numeric_time_axis(plot_df["Date"]),
        xaxis2=build_numeric_time_axis(plot_df["Date"]),
        yaxis=dict(
            title="Price",
            showgrid=False,
        ),
        yaxis2=dict(
            title="Volatility",
            showgrid=False,
            rangemode="tozero",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=55, r=60, t=60, b=45),
    )

    html_path = out_dir / build_period_overview_html_name(period_label)
    html_text = fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        default_width="100vw",
        default_height="100vh",
        config={
            "responsive": True,
            "displayModeBar": False,
            "displaylogo": False,
        },
    )
    html_text = html_text.replace(
        "<head>",
        (
            "<head><style>"
            "html,body{width:100%;height:100%;margin:0;padding:0;overflow:hidden;}"
            ".plotly-graph-div{width:100vw !important;height:100vh !important;}"
            "</style>"
        ),
        1,
    )
    html_text = html_text.replace("<body>", '<body style="margin:0;overflow:hidden;">', 1)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(html_text)
    return html_path


def build_vol_surprise_interactive_html(
    forecast_df: pd.DataFrame,
    period_label: str,
    out_dir: Path,
) -> Path:
    full_frame = forecast_df[
        [
            "Date",
            "open",
            "high",
            "low",
            "close",
            "garch_sigma_return",
            "actual_abs_next_log_return",
    ]
    ].copy()
    full_frame["Date"] = pd.to_datetime(full_frame["Date"], errors="coerce")
    full_frame = full_frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    predicted_vol = pd.to_numeric(full_frame["garch_sigma_return"], errors="coerce")
    realized_vol = pd.to_numeric(full_frame["actual_abs_next_log_return"], errors="coerce")
    full_frame["shock_score"] = np.where(
        predicted_vol > 0,
        realized_vol / predicted_vol,
        np.nan,
    )
    full_frame["event_pred_vol"] = full_frame["garch_sigma_return"].shift(1)
    full_frame["event_real_vol"] = full_frame["actual_abs_next_log_return"].shift(1)
    full_frame["event_shock_score"] = full_frame["shock_score"].shift(1)

    frame = full_frame.loc[full_frame["event_pred_vol"].notna()].copy().reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No rows available for vol surprise plot: {period_label}")

    score_source = frame["event_shock_score"].replace([np.inf, -np.inf], np.nan).dropna()
    if score_source.empty:
        raise RuntimeError(f"No shock score rows available for period {period_label}.")

    threshold_green = float(shock_score_green_threshold)
    threshold_blue = float(shock_score_blue_threshold)
    if threshold_blue < threshold_green:
        threshold_green, threshold_blue = threshold_blue, threshold_green

    bar_seconds = detect_bar_seconds_from_df(full_frame)
    plot_df = frame.reset_index(drop=True).copy()
    _, suspicious_intervals = build_gap_controls(
        full_frame=full_frame,
        displayed_dates=plot_df["Date"],
        bar_seconds=bar_seconds,
    )
    plot_x = np.arange(len(plot_df), dtype=float)
    suspicious_positions = build_suspicious_gap_positions(
        displayed_dates=plot_df["Date"],
        suspicious_intervals=suspicious_intervals,
    )

    hover_text = build_kline_hover_text(
        plot_df=plot_df,
        extra_lines=[
            "<br>pred_vol="
            + plot_df["event_pred_vol"].map(
                lambda x: "nan" if pd.isna(x) else f"{float(x):.8f}"
            ),
            "<br>real_vol="
            + plot_df["event_real_vol"].map(
                lambda x: "nan" if pd.isna(x) else f"{float(x):.8f}"
            ),
            "<br>shock_score="
            + plot_df["event_shock_score"].map(
                lambda x: "nan" if pd.isna(x) else f"{float(x):.8f}"
            ),
        ],
    )

    marker_green_df = plot_df[
        (plot_df["event_shock_score"] > threshold_green)
        & (plot_df["event_shock_score"] <= threshold_blue)
    ].copy()
    marker_green_df["marker_y"] = (
        pd.to_numeric(marker_green_df["high"], errors="coerce")
        + pd.to_numeric(marker_green_df["low"], errors="coerce")
    ) / 2.0
    marker_blue_df = plot_df[plot_df["event_shock_score"] > threshold_blue].copy()
    marker_blue_df["marker_y"] = (
        pd.to_numeric(marker_blue_df["high"], errors="coerce")
        + pd.to_numeric(marker_blue_df["low"], errors="coerce")
    ) / 2.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("Price", "Realized / Predicted"),
    )
    fig.add_trace(
        build_standard_kline_trace(
            x_values=plot_x,
            plot_df=plot_df,
            hover_text=hover_text,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=marker_green_df.index.to_numpy(dtype=float),
            y=marker_green_df["marker_y"],
            mode="markers",
            name=f"shock > {threshold_green:.2f}x",
            marker=dict(color="rgba(60, 180, 75, 0.95)", size=3.4),
            hoverinfo="skip",
            customdata=np.column_stack(
                [
                    marker_green_df["event_shock_score"].to_numpy(),
                    np.full(len(marker_green_df), threshold_green),
                ]
            ) if len(marker_green_df) else None,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=marker_blue_df.index.to_numpy(dtype=float),
            y=marker_blue_df["marker_y"],
            mode="markers",
            name=f"shock > {threshold_blue:.2f}x",
            marker=dict(color="rgba(47, 107, 255, 0.95)", size=3.4),
            hoverinfo="skip",
            customdata=np.column_stack(
                [
                    marker_blue_df["event_shock_score"].to_numpy(),
                    np.full(len(marker_blue_df), threshold_blue),
                ]
            ) if len(marker_blue_df) else None,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_x,
            y=plot_df["event_shock_score"],
            mode="lines",
            name="Shock Score",
            line=dict(color="rgba(70, 70, 70, 0.30)", width=1.3),
            customdata=plot_df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            hovertemplate=(
                "Date=%{customdata}<br>"
                "shock_score=%{y:.8f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_x,
            y=plot_df["event_shock_score"].where(plot_df["event_shock_score"] > 1.0, np.nan),
            mode="lines",
            name="Shock Score >= 1",
            showlegend=False,
            line=dict(color="rgba(70, 70, 70, 0.85)", width=1.3),
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_x,
            y=plot_df["event_shock_score"].where(plot_df["event_shock_score"] > 1.0, np.nan),
            mode="markers",
            name="Shock Score > 1",
            showlegend=False,
            marker=dict(color="rgba(20, 20, 20, 0.62)", size=3),
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=threshold_green,
        line=dict(color="rgba(60, 180, 75, 0.85)", width=1.2, dash="dot"),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=threshold_blue,
        line=dict(color="rgba(47, 107, 255, 0.85)", width=1.2, dash="dot"),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=1.0,
        line=dict(color="rgba(90, 90, 90, 0.85)", width=1.0),
        row=2,
        col=1,
    )

    for suspicious_x in suspicious_positions:
        fig.add_vline(
            x=suspicious_x,
            line=dict(color="rgba(220, 60, 60, 0.45)", width=1, dash="dash"),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=suspicious_x,
            line=dict(color="rgba(220, 60, 60, 0.45)", width=1, dash="dash"),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        title=(
            f"Shock Score Overview | {period_label} | "
            f"green>{threshold_green:.2f}x | blue>{threshold_blue:.2f}x"
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.32)",
            bordercolor="rgba(70, 70, 70, 0.18)",
            font=dict(color="rgba(20, 50, 95, 0.98)"),
        ),
        xaxis=build_numeric_time_axis(plot_df["Date"]),
        xaxis2=build_numeric_time_axis(plot_df["Date"]),
        yaxis=dict(
            title="Price",
            showgrid=False,
        ),
        yaxis2=dict(
            title="Realized / Predicted",
            showgrid=False,
            zeroline=True,
            zerolinecolor="rgba(90, 90, 90, 0.25)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=55, r=60, t=60, b=45),
    )

    html_path = out_dir / build_period_vol_surprise_html_name(period_label)
    html_text = fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        default_width="100vw",
        default_height="100vh",
        config={
            "responsive": True,
            "displayModeBar": False,
            "displaylogo": False,
        },
    )
    html_text = html_text.replace(
        "<head>",
        (
            "<head><style>"
            "html,body{width:100%;height:100%;margin:0;padding:0;overflow:hidden;}"
            ".plotly-graph-div{width:100vw !important;height:100vh !important;}"
            "</style>"
        ),
        1,
    )
    html_text = html_text.replace("<body>", '<body style="margin:0;overflow:hidden;">', 1)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(html_text)
    return html_path


def export_period_outputs(
    forecast_df: pd.DataFrame,
    refit_log_df: pd.DataFrame,
    validation_summary: pd.DataFrame,
    validation_decile: pd.DataFrame,
    period_label: str,
    out_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    parquet_path = out_dir / build_period_parquet_name(period_label)
    excel_path = out_dir / build_period_summary_excel_name(period_label)
    cleanup_stale_period_artifacts(out_dir, period_label)
    overview_html_path = build_overview_interactive_html(
        forecast_df=forecast_df,
        period_label=period_label,
        validation_summary=validation_summary,
        out_dir=out_dir,
    )
    vol_surprise_html_path = build_vol_surprise_interactive_html(
        forecast_df=forecast_df,
        period_label=period_label,
        out_dir=out_dir,
    )

    forecast_df.to_parquet(parquet_path, index=False)

    with pd.ExcelWriter(excel_path) as writer:
        refit_log_df.to_excel(writer, sheet_name="refit_log", index=False)
        validation_summary.to_excel(writer, sheet_name="validation_summary", index=False)
        validation_decile.to_excel(writer, sheet_name="validation_decile", index=False)
        pd.DataFrame(
            [{"key": key, "value": value} for key, value in build_run_config(period_label).items()]
        ).to_excel(writer, sheet_name="run_config", index=False)

    return parquet_path, excel_path, overview_html_path, vol_surprise_html_path


def run_period(
    resample_rule: str,
    out_dir: Path,
    legacy_raw_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    bars_df, bar_seconds, period_label, source_name, legacy_raw_df = load_period_bars(
        period_rule=resample_rule,
        legacy_raw_df=legacy_raw_df,
    )
    if bars_df.empty:
        raise RuntimeError(f"No bars available for period {resample_rule}.")

    print(
        f"[GARCH] start period={period_label} | rows={len(bars_df)} | "
        f"bar_seconds={bar_seconds} | source={source_name} | "
        f"window={resolve_garch_window_bars(period_label)}"
    )

    forecast_df, refit_log_df, run_meta = build_forecast_df(bars_df, period_label)
    existing_forecast_df = run_meta["existing_forecast_df"]
    existing_refit_log_df = run_meta["existing_refit_log_df"]

    if run_meta["skipped"]:
        merged_forecast_df = existing_forecast_df.copy()
        merged_refit_log_df = existing_refit_log_df.copy()
        print(
            f"[GARCH] period={period_label} | "
            f"skip duplicated rows | export_rows={run_meta['export_row_count']}"
        )
    else:
        merged_forecast_df = pd.concat(
            [existing_forecast_df, forecast_df],
            ignore_index=True,
        )
        merged_forecast_df = normalize_forecast_frame(merged_forecast_df, period_label)

        merged_refit_log_df = pd.concat(
            [existing_refit_log_df, refit_log_df],
            ignore_index=True,
        )
        merged_refit_log_df = normalize_refit_log_frame(merged_refit_log_df)
        print(
            f"[GARCH] period={period_label} | "
            f"new_rows={run_meta['new_row_count']} | "
            f"existing_rows={run_meta['existing_row_count']}"
        )

    validation_summary, validation_decile = build_validation_tables(merged_forecast_df, period_label)
    parquet_path, excel_path, overview_html_path, vol_surprise_html_path = export_period_outputs(
        forecast_df=merged_forecast_df,
        refit_log_df=merged_refit_log_df,
        validation_summary=validation_summary,
        validation_decile=validation_decile,
        period_label=period_label,
        out_dir=out_dir,
    )

    summary_row = validation_summary.iloc[0]
    print(
        f"[GARCH] period={period_label} | "
        f"valid={int(summary_row['valid_sample_count'])} | "
        f"spearman={summary_row['spearman_corr']:.6f} | "
        f"top_bottom={summary_row['top_bottom_actual_ratio']:.6f}"
    )
    print(f"[GARCH] saved parquet: {parquet_path}")
    print(f"[GARCH] saved excel: {excel_path}")
    print(f"[GARCH] saved overview html: {overview_html_path}")
    print(f"[GARCH] saved vol surprise html: {vol_surprise_html_path}")
    return legacy_raw_df


def main() -> None:
    out_dir = build_output_dir(data_file_name)
    cleanup_all_stale_html_artifacts(out_dir)
    legacy_raw_df = None
    for rule in resample_rules:
        print("")
        print("=" * 88)
        legacy_raw_df = run_period(
            resample_rule=rule,
            out_dir=out_dir,
            legacy_raw_df=legacy_raw_df,
        )
        print("=" * 88)

    elapsed = time.time() - start_time
    print(f"[GARCH] all done. elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
