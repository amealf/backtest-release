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
data_folder_path = r"D:\Code\data\20260326\\"
data_file_name = "xagusd_30s_all"

warmup_start_date = ""
export_start_date = ""
export_end_date = ""

resample_rules = ["30s", "1min", "5min", "1H"]

garch_window_bars = 10000
refit_every_bars = 100
garch_p = 1
garch_q = 1
garch_dist = "t"
return_scale = 100.0
validation_quantiles = 10


MODEL_SPEC = "garch11_t"
GAP_MULTIPLIER = 1.5
OVERVIEW_MAX_POINTS = 3000


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

    compressed = (
        frame.assign(_bucket_id=bucket_id)
        .groupby("_bucket_id", as_index=False)
        .agg(**agg_map)
    )
    return compressed.reset_index(drop=True)


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
    resample_rule = (rule or "").strip()
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
    rule = (resample_rule or "").strip()
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
    rule = (resample_rule or "").strip()
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


def build_run_config(period_label: str) -> dict:
    return {
        "data_folder_path": normalize_config_text(data_folder_path),
        "data_file_name": normalize_config_text(data_file_name),
        "warmup_start_date": normalize_config_text(warmup_start_date),
        "export_start_date": normalize_config_text(export_start_date),
        "export_end_date": normalize_config_text(export_end_date),
        "period_label": normalize_config_text(period_label),
        "garch_window_bars": str(int(garch_window_bars)),
        "refit_every_bars": str(int(refit_every_bars)),
        "garch_p": str(int(garch_p)),
        "garch_q": str(int(garch_q)),
        "garch_dist": normalize_config_text(garch_dist),
        "return_scale": normalize_config_text(return_scale),
        "validation_quantiles": str(int(validation_quantiles)),
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


def load_raw_data() -> pd.DataFrame:
    raw_df, _, _ = load_data(data_folder_path, data_file_name)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], errors="coerce")
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

    period_label = format_period_label(resample_rule, bar_seconds)
    return preview_df, bar_seconds, period_label


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
            (valid_idx + 1) >= int(garch_window_bars)
            and ((valid_idx + 1 - int(garch_window_bars)) % int(refit_every_bars) == 0)
        )

        if should_refit:
            refit_executed[row_pos] = 1
            total_refits += 1
            window_positions = valid_positions[valid_idx - int(garch_window_bars) + 1: valid_idx + 1]
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
                "fit_window_bars": int(garch_window_bars),
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
                "garch_window_bars": int(garch_window_bars),
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
    frame = forecast_df[
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
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    frame = frame.loc[frame["garch_sigma_return"].notna()].copy()
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No rows available for overview plot: {period_label}")

    plot_df = compress_overview_frame(frame)
    summary_row = validation_summary.iloc[0]

    hover_text = (
        "Date="
        + plot_df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        + "<br>open="
        + plot_df["open"].map(lambda x: f"{float(x):.6f}")
        + "<br>high="
        + plot_df["high"].map(lambda x: f"{float(x):.6f}")
        + "<br>low="
        + plot_df["low"].map(lambda x: f"{float(x):.6f}")
        + "<br>close="
        + plot_df["close"].map(lambda x: f"{float(x):.6f}")
        + "<br>pred_vol="
        + plot_df["garch_sigma_return"].map(
            lambda x: "nan" if pd.isna(x) else f"{float(x):.8f}"
        )
        + "<br>real_vol="
        + plot_df["actual_abs_next_log_return"].map(
            lambda x: "nan" if pd.isna(x) else f"{float(x):.8f}"
        )
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
        go.Candlestick(
            x=plot_df["Date"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Kline",
            increasing=dict(
                line=dict(color="#111111", width=0.8),
                fillcolor="rgba(245, 245, 245, 0.9)",
            ),
            decreasing=dict(
                line=dict(color="#111111", width=0.8),
                fillcolor="rgba(120, 120, 120, 0.9)",
            ),
            hovertext=hover_text,
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["garch_sigma_return"],
            mode="lines",
            name="Predicted Vol",
            line=dict(color="rgba(31, 119, 180, 0.50)", width=1.4),
            hovertemplate=(
                "Date=%{x|%Y-%m-%d %H:%M:%S}<br>"
                "pred_vol=%{y:.8f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["actual_abs_next_log_return"],
            mode="lines",
            name="Realized Vol",
            line=dict(color="#ff7f0e", width=1.2),
            hovertemplate=(
                "Date=%{x|%Y-%m-%d %H:%M:%S}<br>"
                "real_vol=%{y:.8f}<extra></extra>"
            ),
        ),
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
        xaxis=dict(
            title=None,
            rangeslider=dict(visible=False),
            showgrid=False,
        ),
        xaxis2=dict(
            title=None,
            showgrid=False,
        ),
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


def export_period_outputs(
    forecast_df: pd.DataFrame,
    refit_log_df: pd.DataFrame,
    validation_summary: pd.DataFrame,
    validation_decile: pd.DataFrame,
    period_label: str,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    parquet_path = out_dir / build_period_parquet_name(period_label)
    excel_path = out_dir / build_period_summary_excel_name(period_label)
    cleanup_stale_period_artifacts(out_dir, period_label)
    overview_html_path = build_overview_interactive_html(
        forecast_df=forecast_df,
        period_label=period_label,
        validation_summary=validation_summary,
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

    return parquet_path, excel_path, overview_html_path


def run_period(raw_df: pd.DataFrame, resample_rule: str, out_dir: Path) -> None:
    bars_df, bar_seconds, period_label = prepare_period_bars(raw_df, resample_rule)
    if bars_df.empty:
        raise RuntimeError(f"No bars available for period {resample_rule}.")

    print(
        f"[GARCH] start period={period_label} | rows={len(bars_df)} | "
        f"bar_seconds={bar_seconds}"
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
    parquet_path, excel_path, overview_html_path = export_period_outputs(
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


def main() -> None:
    out_dir = build_output_dir(data_file_name)
    cleanup_all_stale_html_artifacts(out_dir)
    raw_df = load_raw_data()
    for rule in resample_rules:
        print("")
        print("=" * 88)
        run_period(raw_df=raw_df, resample_rule=rule, out_dir=out_dir)
        print("=" * 88)

    elapsed = time.time() - start_time
    print(f"[GARCH] all done. elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
