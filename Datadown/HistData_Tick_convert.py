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
import numpy as np
import pandas as pd
from tqdm import tqdm


# ========= 路径设置 =========
base_dir = r"D:\Code\data"
zip_dir = os.path.join(base_dir, "")

# K 线周期（秒），例如 5 / 15 / 30 / 300
bar_interval_seconds = 15

_label = f"{bar_interval_seconds}s"
extract_dir = os.path.join(base_dir, "extracted_tick")
convert_dir = os.path.join(base_dir, f"converted_{_label}")
out_file = os.path.join(base_dir, f"xagusd_{_label}_all.csv")

# 价格来源：bid / ask / mid
price_source = "bid"

# HistData 时间若需转 UTC，可设为 True（原始常用 EST 固定时区）
convert_est_to_utc = False

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
    return (ts.hour >= SILVER_CLOSE_HOUR_START) & (ts.hour < SILVER_CLOSE_HOUR_END)


def _get_trading_session_date(ts: pd.DatetimeIndex) -> pd.Series:
    """
    将时间戳映射到所属的交易日期。
    白银交易日从 18:00 开始到次日 17:00 结束，
    因此 18:00 之后的交易归属到「下一个日历日」。
    """
    dates = ts.normalize()  # 日历日 00:00
    # 18:00 及之后的交易归属到下一个日历日
    mask_next_day = ts.hour >= SILVER_CLOSE_HOUR_END
    adj = pd.Series(dates, index=ts)
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

    if convert_est_to_utc:
        df_tick["ts"] = df_tick["ts"] + pd.Timedelta(hours=5)

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


def main():
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(convert_dir, exist_ok=True)

    zips = [f for f in os.listdir(zip_dir) if f.lower().endswith(".zip")]
    zips.sort()
    if not zips:
        raise ValueError(f"目录中没有 zip 文件: {zip_dir}")

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
    result["time"] = result["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    result.to_csv(out_file, index=False, header=False)
    print(f"\n完成输出: {out_file}  总行数: {len(result)}")


if __name__ == "__main__":
    main()