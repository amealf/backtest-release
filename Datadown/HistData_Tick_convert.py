# ============================================================
# 白银 (XAGUSD) 专用 Tick -> K 线转换脚本
# - 自动填充无成交的时间窗口（OHLC = 前一根 close，volume = 0）
# - 跳过每日 17:00-18:00 收盘时段（CME COMEX 白银规则）
# - 通过已有 tick 数据推断交易日，不会在非交易日填充数据
# ============================================================

import os
import re
import csv
import zipfile
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm


# ========= 路径设置 =========
base_dir = r"D:\Code\data"
zip_dir = os.path.join(base_dir, "archive")

# K 线周期（秒），例如 5 / 15 / 30 / 300
bar_interval_seconds = 30

# Zip range examples: all | 2023-latest | 2023-2024 | 202301-202403
zip_selection = "2023-latest"

_label = f"{bar_interval_seconds}s"
run_date = datetime.now().strftime("%Y%m%d")
run_dir = os.path.join(base_dir, run_date)
extract_dir = os.path.join(run_dir, "extracted_tick")
convert_dir = os.path.join(run_dir, f"converted_{_label}")
yearly_dir = os.path.join(run_dir, f"yearly_{_label}")
out_file = os.path.join(run_dir, f"xagusd_{_label}_all.csv")

# 价格来源：bid / ask / mid
price_source = "bid"

# HistData timestamps are interpreted in New York local time.
# America/New_York will apply DST automatically.
source_timezone = "America/New_York"
convert_new_york_to_utc = False
trading_timezone = "America/New_York"

# Silver maintenance break uses New York local time.
# 17:00-18:00 stays on the New York clock and DST is automatic.

# ========= 白银交易时间 =========
# CME COMEX 白银每日 17:00-18:00 (EST) 休市
SILVER_CLOSE_HOUR_START = 17   # 收盘开始 (含)
SILVER_CLOSE_HOUR_END   = 18   # 收盘结束 (不含，即 18:00 恢复交易)
# ================================


def detect_delimiter(file_path: str) -> str:
    """
    根据首行判断分隔符，优先 tab，其次逗号。
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if first_line.count("\t") >= 3:
        return "\t"
    return ","


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    处理例如 "14\t\t\t" 这类字符串，只保留首个数值。
    """
    x = s.astype(str).str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(x, errors="coerce")


def parse_hist_ts(dt_series: pd.Series) -> pd.Series:
    """
    支持两类时间：
    1) HistData: YYYYMMDD HHMMSSNNN
    2) 常规文本时间: 1/1/2025 18:00 或 2025-01-01 18:00:30
    """
    dt_series = dt_series.astype(str).str.strip()

    out = pd.Series(pd.NaT, index=dt_series.index, dtype="datetime64[ns]")

    # A: HistData 样式
    mask_hist = dt_series.str.match(r"^\d{8}\s\d{6,9}$")
    if mask_hist.any():
        part = dt_series[mask_hist].str.split(" ", n=1, expand=True)
        d = part[0]
        t = part[1].str.zfill(9)  # HHMMSSNNN
        out.loc[mask_hist] = pd.to_datetime(
            d + " " + t,
            format="%Y%m%d %H%M%S%f",
            errors="coerce",
        )

    # B: 常规文本时间
    mask_other = ~mask_hist
    if mask_other.any():
        out.loc[mask_other] = pd.to_datetime(
            dt_series[mask_other],
            errors="coerce",
        )

    return out


def localize_source_timestamps(ts: pd.Series) -> pd.Series:
    localized = pd.DatetimeIndex(ts).tz_localize(
        source_timezone,
        ambiguous="infer",
        nonexistent="shift_forward",
    )
    return pd.Series(localized, index=ts.index)


def to_trading_clock(ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        idx = idx.tz_localize(
            source_timezone,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    return idx.tz_convert(trading_timezone)


def normalize_tick_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    统一成四列：dt, bid, ask, vol
    支持：
    - 4列: dt,bid,ask,vol
    - 5列: date,time,bid,ask,vol
    """
    df_raw = df_raw.dropna(how="all")
    ncol = df_raw.shape[1]

    if ncol >= 5:
        c0 = df_raw.iloc[:, 0].astype(str).str.strip()
        c1 = df_raw.iloc[:, 1].astype(str).str.strip()
        is_date = c0.str.match(r"^\d{8}$")
        is_time = c1.str.match(r"^\d{6,9}$")
        if (is_date & is_time).mean() > 0.8:
            out = pd.DataFrame({
                "dt": c0 + " " + c1,
                "bid": df_raw.iloc[:, 2],
                "ask": df_raw.iloc[:, 3],
                "vol": df_raw.iloc[:, 4],
            })
            return out

    if ncol >= 4:
        out = pd.DataFrame({
            "dt": df_raw.iloc[:, 0],
            "bid": df_raw.iloc[:, 1],
            "ask": df_raw.iloc[:, 2],
            "vol": df_raw.iloc[:, 3],
        })
        return out

    raise ValueError("输入列数不足，无法识别为 tick 数据。")


def _is_in_close_period(ts: pd.DatetimeIndex) -> pd.Series:
    """判断时间戳是否落在白银收盘时段 [17:00, 18:00)"""
    local_ts = to_trading_clock(ts)
    return (local_ts.hour >= SILVER_CLOSE_HOUR_START) & (local_ts.hour < SILVER_CLOSE_HOUR_END)


def _get_trading_session_date(ts: pd.DatetimeIndex) -> pd.Series:
    """
    将时间戳映射到所属的交易日期。
    白银交易日从 18:00 开始到次日 17:00 结束，
    因此 18:00 之后的交易归属到「下一个日历日」。
    """
    local_ts = to_trading_clock(ts)
    dates = local_ts.normalize()  # 日历日 00:00
    # 18:00 及之后的交易归属到下一个日历日
    mask_next_day = local_ts.hour >= SILVER_CLOSE_HOUR_END
    adj = pd.Series(dates)
    adj[mask_next_day] = adj[mask_next_day] + pd.Timedelta(days=1)
    return adj


def _build_full_index(trading_dates: set, interval_seconds: int,
                      ts_min: pd.Timestamp, ts_max: pd.Timestamp) -> pd.DatetimeIndex:
    """
    为所有交易日生成完整的时间索引，跳过 17:00-18:00 收盘时段。
    trading_dates: 从已有 tick 数据推断出的交易日集合 (日历日级别)
    """
    freq = f"{interval_seconds}s"
    # 生成覆盖完整时间范围的索引
    full_range = pd.date_range(start=ts_min.floor(freq), end=ts_max.ceil(freq), freq=freq)

    # 过滤：只保留交易日 + 排除收盘时段
    session_dates = _get_trading_session_date(full_range)
    # 将 session_dates 归一化到 date 用于集合查找
    session_date_values = session_dates.dt.date

    mask_trading_day = session_date_values.isin(trading_dates)
    mask_not_closed = ~_is_in_close_period(full_range)

    return full_range[np.asarray(mask_trading_day) & np.asarray(mask_not_closed)]


def ticks_to_bars(df_tick: pd.DataFrame, interval_seconds: int = 30) -> pd.DataFrame:
    df_tick["ts"] = parse_hist_ts(df_tick["dt"])
    df_tick["bid"] = pd.to_numeric(df_tick["bid"], errors="coerce")
    df_tick["ask"] = pd.to_numeric(df_tick["ask"], errors="coerce")
    df_tick["vol_num"] = clean_numeric_series(df_tick["vol"]).fillna(0.0)

    df_tick = df_tick.dropna(subset=["ts", "bid", "ask"]).sort_values("ts")
    df_tick["ts"] = localize_source_timestamps(df_tick["ts"])

    if convert_new_york_to_utc:
        df_tick["ts"] = df_tick["ts"].dt.tz_convert("UTC")

    if price_source == "bid":
        df_tick["px"] = df_tick["bid"]
    elif price_source == "ask":
        df_tick["px"] = df_tick["ask"]
    elif price_source == "mid":
        df_tick["px"] = (df_tick["bid"] + df_tick["ask"]) / 2.0
    else:
        raise ValueError('price_source 只能是 "bid" / "ask" / "mid"')

    df_tick = df_tick.set_index("ts")

    # --- 第1步：从已有数据推断交易日 ---
    session_dates = _get_trading_session_date(df_tick.index)
    trading_dates = set(session_dates.dt.date.unique())

    # --- 第2步：正常 resample 聚合 ---
    freq = f"{interval_seconds}s"
    bars = df_tick.resample(freq, label="left", closed="left").agg(
        low=("px", "min"),
        high=("px", "max"),
        open=("px", "first"),
        close=("px", "last"),
        vol_sum=("vol_num", "sum"),
        tick_count=("px", "size"),
    )

    # --- 第3步：在 reindex 之前决定使用 vol_sum 还是 tick_count ---
    has_real_vol = bars["vol_sum"].sum() > 0
    if has_real_vol:
        bars["volume"] = bars["vol_sum"].astype(float)
    else:
        bars["volume"] = bars["tick_count"].astype(float)

    bars = bars[["open", "high", "low", "close", "volume"]]

    # 去掉收盘时段中有零星 tick 的 bar（不应出现）
    mask_close = _is_in_close_period(bars.index)
    bars = bars[~mask_close]

    # --- 第4步：生成完整时间索引并 reindex ---
    full_idx = _build_full_index(
        trading_dates, interval_seconds,
        ts_min=df_tick.index.min(),
        ts_max=df_tick.index.max(),
    )
    bars = bars.reindex(full_idx)

    # --- 第5步：前向填充空窗口 ---
    # 用前一根 bar 的 close 填充 OHLC
    bars["close"] = bars["close"].ffill()
    for col in ["open", "high", "low"]:
        bars[col] = bars[col].fillna(bars["close"])

    # volume 填 0
    bars["volume"] = bars["volume"].fillna(0.0)

    # 丢弃最初仍无数据的行（第一根 bar 之前无法前向填充）
    bars = bars.dropna(subset=["close"])

    # --- 第6步：组装输出 ---
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    out.insert(0, "time", out.index.strftime("%Y-%m-%d %H:%M:%S"))  # 强制保留秒
    out = out.reset_index(drop=True)
    return out


def extract_one_zip(zip_path: str, target_root: str) -> str:
    """
    解压 zip 到 extracted_tick 目录，返回解压出的数据文件路径。
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        data_files = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not data_files:
            raise ValueError(f"压缩包内未找到 csv/txt: {zip_path}")
        data_files.sort(key=lambda x: ("dat_ascii" not in x.lower(), x))
        chosen = data_files[0]
        zf.extract(chosen, path=target_root)
        return os.path.join(target_root, chosen)


def extract_zip_period(zip_name: str) -> str:
    match = re.search(r"_T(\d{6})\.zip$", zip_name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"unsupported zip name: {zip_name}")
    return match.group(1)


def normalize_period_token(token: str, is_start: bool) -> str:
    token = token.strip()
    if re.fullmatch(r"\d{4}", token):
        return f"{token}{'01' if is_start else '12'}"
    if re.fullmatch(r"\d{6}", token):
        return token
    raise ValueError(f"invalid period token: {token}")


def parse_selected_period(selection: str, available_periods: list[str]) -> tuple[str, str]:
    choice = selection.strip().lower()
    if not choice or choice == "all":
        return available_periods[0], available_periods[-1]

    if "-" not in choice:
        raise ValueError("selection must look like all, YYYY-latest, YYYY-YYYY, or YYYYMM-YYYYMM")

    start_token, end_token = [part.strip() for part in choice.split("-", 1)]
    start_period = normalize_period_token(start_token, is_start=True)
    end_period = (
        available_periods[-1]
        if end_token == "latest"
        else normalize_period_token(end_token, is_start=False)
    )

    if start_period > end_period:
        raise ValueError(f"invalid selection: {selection}")
    return start_period, end_period


def select_zip_files(zips: list[str], selection: str) -> tuple[list[str], str, str]:
    zip_items = [(name, extract_zip_period(name)) for name in zips]
    available_periods = sorted(period for _, period in zip_items)
    start_period, end_period = parse_selected_period(selection, available_periods)

    selected = [
        name for name, period in zip_items
        if start_period <= period <= end_period
    ]
    if not selected:
        raise ValueError(f"no zip files matched selection: {selection}")

    return selected, start_period, end_period


def save_yearly_outputs(result: pd.DataFrame) -> None:
    os.makedirs(yearly_dir, exist_ok=True)

    for year, year_df in result.groupby(result["time"].dt.year, sort=True):
        year_out = year_df.copy()
        year_out["time"] = year_out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        year_file = os.path.join(yearly_dir, f"xagusd_{_label}_{year}.csv")
        year_out.to_csv(year_file, index=False, header=False)
        print(f"Year output: {year_file}  rows={len(year_out)}")


def main():
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(convert_dir, exist_ok=True)
    os.makedirs(yearly_dir, exist_ok=True)

    zips = [f for f in os.listdir(zip_dir) if f.lower().endswith(".zip")]
    zips.sort()
    if not zips:
        raise ValueError(f"目录中没有 zip 文件: {zip_dir}")

    selection = zip_selection.strip() or "all"
    zips, start_period, end_period = select_zip_files(zips, selection)
    print(f"Selected period: {start_period}-{end_period}  zips={len(zips)}")

    all_bars = []

    for idx, zname in enumerate(zips, start=1):
        zpath = os.path.join(zip_dir, zname)
        print(f"\n[{idx}/{len(zips)}] {zname}")

        data_path = extract_one_zip(zpath, extract_dir)
        delim = detect_delimiter(data_path)

        # 统计总行数用于进度条
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            total_lines = sum(1 for _ in f)

        # 分块读取并显示进度
        chunks = []
        chunk_size = 100_000
        reader = pd.read_csv(
            data_path,
            header=None,
            sep=delim,
            engine="python",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            dtype=str,
            on_bad_lines="skip",
            chunksize=chunk_size,
        )
        with tqdm(total=total_lines, desc="  读取 tick", unit="行") as pbar:
            for chunk in reader:
                chunks.append(chunk)
                pbar.update(len(chunk))

        df_raw = pd.concat(chunks, ignore_index=True)
        df_tick = normalize_tick_columns(df_raw)

        bars = ticks_to_bars(df_tick, bar_interval_seconds)
        all_bars.append(bars)

        # 保存每个月的转换结果
        month_name = os.path.splitext(zname)[0]  # 去掉 .zip
        month_file = os.path.join(convert_dir, f"{month_name}_{_label}.csv")
        bars.to_csv(month_file, index=False, header=False)
        print(f"  生成 {_label} K线: {len(bars)} 条 -> {month_file}")

    # 合并所有月份
    result = pd.concat(all_bars, ignore_index=True)
    result["time"] = pd.to_datetime(result["time"], errors="coerce")
    result = result.dropna(subset=["time"]).sort_values("time")
    save_yearly_outputs(result)
    result["time"] = result["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    result.to_csv(out_file, index=False, header=False)
    print(f"\n完成输出: {out_file}  总行数: {len(result)}")


if __name__ == "__main__":
    main()
