# -*- coding: utf-8 -*-
"""
GARCH(1,1) 模型验证程序（XAGUSD 15秒K线）

功能：
1) 读取K线CSV：datetime, open, high, low, close, volume（自动兼容逗号/Tab分隔）
2) 计算对数收益率
3) 剔除隔夜跳空：跨交易日第一根bar的收益率置NaN并删除
4) 按时间顺序 70/30 划分训练/测试
5) 训练集拟合 GARCH(1,1)（arch库），输出参数及显著性
6) 测试集做滚动一步预测（固定参数，递推更新条件方差）
7) 计算验证指标：
   - 实现方差 RV：后续240根bar(1小时)的平方收益率之和
   - Mincer-Zarnowitz：RV = a + b*sigma_hat^2（报告a,b,联合F检验p值、R²）
   - QLIKE：mean(log(sigma_hat^2) + RV/sigma_hat^2)
8) 可视化：4张图，中文标题，字体大，配色清晰，保存PNG(dpi=150)

依赖：
pip install arch pandas numpy matplotlib scipy statsmodels
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import scipy.stats as st
import statsmodels.api as sm

try:
    from arch import arch_model
except ImportError as e:
    raise SystemExit(
        "未找到 arch 库。请先安装：pip install arch"
    ) from e


try:
    from mplfinance.original_flavor import candlestick2_ohlc
except Exception:
    candlestick2_ohlc = None

import openpyxl  # xlsx 输出依赖

# ========= 路径设置 =========
CSV_PATH = r"D:\Code\data\converted_300s\HISTDATA_COM_ASCII_XAGUSD_T202512_300s.csv"
# ============================

# ----------------------------
# 画图：中文字体与全局样式
# ----------------------------
def setup_matplotlib_style():
    # 尽量在 Windows 下自动找到可用中文字体；找不到也不崩，只是中文可能显示为方块
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 字体整体放大、线条清晰
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 20,
        "lines.linewidth": 1.3,
    })


# ----------------------------
# 读CSV（自动识别分隔符）
# ----------------------------
def read_kline_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    # 优先用“自动分隔符嗅探”
    # sep=None + engine='python' 会尝试识别逗号/Tab/分号等（速度略慢，但最稳）
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # 兜底再试 tab
        df = pd.read_csv(path, sep="\t")

    # 如果读出来只有一列，通常是分隔符没识别出来，再尝试 tab
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep="\t")

    # 统一列名
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["datetime", "open", "high", "low", "close", "volume"]

    if "datetime" not in df.columns:
        # 可能没有表头
        df = pd.read_csv(
            path, sep=None, engine="python", header=None,
            names=required
        )
        df.columns = [c.lower() for c in df.columns]

    # 只保留需要的列（如果文件里更多列，不影响）
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}；当前列：{list(df.columns)}")

    df = df[required].copy()

    # datetime 解析
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    # 数值列转换
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"]).copy()
    df = df[df["close"] > 0].copy()

    # OHLC 合理性检查：high >= max(open,close), low <= min(open,close)
    if len(df) > 10:
        oc_max = df[["open", "close"]].max(axis=1)
        oc_min = df[["open", "close"]].min(axis=1)
        high_ok = (df["high"] >= oc_max - 1e-8).mean()
        low_ok  = (df["low"]  <= oc_min + 1e-8).mean()
        if high_ok < 0.9 or low_ok < 0.9:
            # 尝试检测 open/low 互换（常见的列顺序错误）
            if (df["low"] >= oc_max - 1e-8).mean() > 0.5:
                print("警告：检测到 open 和 low 列可能互换，自动修正。")
                df["open"], df["low"] = df["low"].copy(), df["open"].copy()
            elif (df["high"] <= oc_min + 1e-8).mean() > 0.5:
                print("警告：检测到 open 和 high 列可能互换，自动修正。")
                df["open"], df["high"] = df["high"].copy(), df["open"].copy()
            else:
                print(f"警告：OHLC 数据可能有列顺序问题 (high合规率={high_ok:.1%}, low合规率={low_ok:.1%})")

    # 排序、去重
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)

    return df


# ----------------------------
# 跳过长时间零成交段（停盘检测）
# ----------------------------
ZERO_VOL_GAP_THRESHOLD_MINUTES = 3  # 连续0成交超过此分钟数视为停盘


def _mask_zero_volume_runs(df: pd.DataFrame, threshold_minutes: float = ZERO_VOL_GAP_THRESHOLD_MINUTES) -> pd.Series:
    """
    检测连续 volume==0 的区段。
    如果一段连续0成交的总时间跨度 >= threshold_minutes 分钟，
    则该段所有行标记为 True（应跳过）。
    """
    is_zero = (df["volume"] == 0).values
    dt = df["datetime"].values  # datetime64
    n = len(df)
    mask = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if is_zero[i]:
            j = i
            while j < n and is_zero[j]:
                j += 1
            # [i, j) 是连续0成交段
            span_minutes = (dt[j - 1] - dt[i]) / np.timedelta64(1, "m")
            if span_minutes >= threshold_minutes:
                mask[i:j] = True
            i = j
        else:
            i += 1

    return pd.Series(mask, index=df.index, name="zero_vol_gap")


# ----------------------------
# 预处理：对数收益率 + 剔除隔夜跳空 + 跳过停盘段
# ----------------------------
def preprocess_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = np.log(out["close"]).diff()

    # 剔除隔夜跳空：跨交易日第一根bar（date变化处）收益率设为NaN
    date = out["datetime"].dt.date
    is_first_bar_of_day = date.ne(date.shift(1))
    out.loc[is_first_bar_of_day, "ret"] = np.nan

    # 跳过停盘段：连续 >= 3分钟 volume==0 的区段
    gap_mask = _mask_zero_volume_runs(out)
    n_gap = gap_mask.sum()
    if n_gap > 0:
        print(f"  停盘检测：跳过 {n_gap} 根连续零成交bar（阈值={ZERO_VOL_GAP_THRESHOLD_MINUTES}分钟）")
        out.loc[gap_mask, "ret"] = np.nan
        # 停盘段之后的第一根有成交bar的收益率也应设为NaN（避免跳空）
        gap_end = gap_mask & ~gap_mask.shift(-1, fill_value=False)
        for idx_pos in np.where(gap_end.values)[0]:
            next_pos = idx_pos + 1
            if next_pos < len(out):
                out.iloc[next_pos, out.columns.get_loc("ret")] = np.nan

    # 删除 NaN（包含首行diff产生的NaN、跨日第一根bar、停盘段）
    out = out.dropna(subset=["ret"]).copy()

    # 使用 datetime 作为索引（后续时序/回归更方便）
    out = out.set_index("datetime", drop=True)

    return out


# ----------------------------
# 滚动一步预测（固定参数、递推更新）
# 说明：这里不“每一步重估参数”，只用训练集估计的参数滚动更新条件方差
# ----------------------------
def garch11_roll_conditional_variance(
    r_test: pd.Series,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    h_last_train: float,
    eps_last_train: float,
) -> pd.Series:
    """
    返回：测试集每个时点 t 的条件方差 h_t（即 Var(r_t | t-1)）
    递推：
      eps_t = r_t - mu
      h_t   = omega + alpha*eps_{t-1}^2 + beta*h_{t-1}
    """
    h_list = []
    h_prev = float(h_last_train)
    eps_prev = float(eps_last_train)

    for rt in r_test.values:
        h_t = omega + alpha * (eps_prev ** 2) + beta * h_prev
        h_list.append(h_t)

        eps_t = rt - mu
        h_prev = h_t
        eps_prev = eps_t

    return pd.Series(h_list, index=r_test.index, name="h_t")


def aggregate_garch_variance_1h_from_h1(
    h1: pd.Series,
    omega: float,
    g: float,
    horizon: int,
) -> pd.Series:
    """
    将 1-step ahead 方差 h_{t+1|t} 聚合成未来 horizon 步的“总方差”：
      sigma_hat^2(1h) = sum_{k=1..H} h_{t+k|t}

    对 GARCH(1,1)，k>=2 时：
      h_{t+k|t} = omega + g * h_{t+k-1|t}, 其中 g = alpha+beta

    Closed-form：
      A = sum_{k=1..H} g^{k-1} = (1 - g^H)/(1-g)  (g!=1)
      sum = omega/(1-g) * (H - A) + h1 * A

    g≈1 用极限形式：
      h_{t+k|t} = h1 + (k-1)*omega
      sum = H*h1 + omega*H*(H-1)/2
    """
    H = int(horizon)
    g = float(g)
    omega = float(omega)

    if H <= 0:
        raise ValueError("horizon 必须为正整数")

    if abs(1.0 - g) < 1e-10:
        # g -> 1 的极限
        out = H * h1 + omega * (H * (H - 1) / 2.0)
        out.name = f"sigma2_hat_{H}"
        return out

    A = (1.0 - (g ** H)) / (1.0 - g)
    out = (omega / (1.0 - g)) * (H - A) + h1 * A
    out.name = f"sigma2_hat_{H}"
    return out


# ----------------------------
# RV：后续240根bar平方收益率之和
# RV_t = sum_{i=1..H} r_{t+i}^2
# ----------------------------
def realized_variance_forward(r: pd.Series, horizon: int) -> pd.Series:
    r2 = r ** 2

    # forward rolling sum：用反转技巧实现“向前窗口”
    # sum_{i=0..H-1} r2_{t+i}
    rv_including_t = r2.iloc[::-1].rolling(window=horizon, min_periods=horizon).sum().iloc[::-1]

    # 我们要 t+1..t+H，所以整体 shift(-1)
    rv = rv_including_t.shift(-1)
    rv.name = f"RV_{horizon}"
    return rv


# ----------------------------
# 评估口径与结果有效性检查
# ----------------------------
def infer_bar_seconds_from_datetime(dt: pd.Series | pd.DatetimeIndex) -> float | None:
    if isinstance(dt, pd.Series):
        ts = pd.to_datetime(dt, errors="coerce").dropna()
        idx = pd.DatetimeIndex(ts)
    else:
        idx = pd.DatetimeIndex(dt)
        idx = idx[~idx.isna()]

    if len(idx) < 2:
        return None

    diffs = pd.Series(idx).diff().dt.total_seconds().dropna()
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None

    return float(np.median(diffs.values))


def evaluate_result_validity(
    *,
    horizon: int,
    inferred_bar_seconds: float | None,
    eval_df: pd.DataFrame,
    qlike: float,
    omega: float,
    alpha: float,
    beta: float,
    g: float,
    mean_pred: float,
    mean_rv: float,
) -> tuple[bool, list[str], int | None, float]:
    reasons: list[str] = []
    expected_horizon_1h: int | None = None

    # 口径检查：如果目标是 1 小时评估，horizon 应等于 3600/bar_seconds
    if inferred_bar_seconds is not None and inferred_bar_seconds > 0:
        expected_horizon_1h = int(round(3600.0 / inferred_bar_seconds))
        if abs(horizon - expected_horizon_1h) > 1:
            reasons.append(
                f"horizon 与 1 小时口径不匹配（当前={horizon}, 期望≈{expected_horizon_1h}）。"
            )

    # 样本量与数值稳定性检查
    min_eval_n = max(300, horizon * 2)
    if len(eval_df) < min_eval_n:
        reasons.append(f"可评估样本过少（当前={len(eval_df)}, 建议至少={min_eval_n}）。")

    if not np.isfinite(qlike):
        reasons.append("QLIKE 非有限值（NaN/Inf），结果不可用。")

    # 参数合理性（GARCH(1,1)）
    if not (omega > 0):
        reasons.append("omega <= 0，条件方差递推不合理。")
    if alpha < 0 or beta < 0:
        reasons.append("alpha 或 beta 为负，违反常见 GARCH 约束。")
    if g >= 1:
        reasons.append("alpha+beta >= 1，模型接近/进入非平稳区间。")

    # 预测与实现的量级检查
    ratio = np.nan
    if mean_rv > 0 and np.isfinite(mean_rv) and np.isfinite(mean_pred):
        ratio = mean_pred / mean_rv
        if ratio < 0.2 or ratio > 5.0:
            reasons.append(
                f"预测方差均值与 RV 均值量级严重失配（mean_pred/mean_rv={ratio:.3f}）。"
            )
    else:
        reasons.append("mean(RV) 或 mean(pred) 非法，无法比较量级。")

    is_valid = len(reasons) == 0
    return is_valid, reasons, expected_horizon_1h, float(ratio) if np.isfinite(ratio) else np.nan


# ----------------------------
# MZ 回归 + 联合F检验
# ----------------------------
def mincer_zarnowitz_regression(rv: pd.Series, sigma2_hat: pd.Series):
    df = pd.DataFrame({"RV": rv, "sigma2_hat": sigma2_hat}).dropna()
    y = df["RV"]
    X = sm.add_constant(df["sigma2_hat"])
    ols = sm.OLS(y, X).fit()

    a = float(ols.params["const"])
    b = float(ols.params["sigma2_hat"])
    r2 = float(ols.rsquared)

    # 联合检验：a=0 且 b=1
    # beta = [const, slope]
    R = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    q = np.array([0.0, 1.0])
    ftest = ols.f_test((R, q))
    pval = float(ftest.pvalue)

    return ols, a, b, r2, pval, df


# ----------------------------
# 图表输出
# ----------------------------
def save_fig(fig, out_path: Path, dpi: int = 150, close: bool = True):
    fig.tight_layout()
    # 按需求：不保存图片，仅保留图窗显示
    if close:
        plt.close(fig)


def _build_time_formatter(time_index: pd.DatetimeIndex):
    n = len(time_index)

    def _fmt(x, _pos):
        i = int(round(x))
        if i < 0 or i >= n:
            return ""
        return pd.Timestamp(time_index[i]).strftime("%m-%d %H:%M")

    return FuncFormatter(_fmt)


def format_period_label(bar_seconds: float | int | None) -> str:
    if bar_seconds is None or not np.isfinite(bar_seconds):
        return "unknown"
    sec = int(round(float(bar_seconds)))
    if sec < 60:
        return f"{sec}s"
    if sec % 3600 == 0:
        return f"{sec // 3600}h"
    if sec % 60 == 0:
        return f"{sec // 60}m"
    return f"{sec}s"


def infer_symbol_from_csv_path(csv_path: str) -> str:
    stem_upper = Path(csv_path).stem.upper()
    m = re.search(r"([A-Z]{6})", stem_upper)
    if m:
        return m.group(1)[:3]
    m = re.search(r"(XAG|XAU|BTC|ETH|EUR|GBP|JPY|AUD|CAD|CHF|NZD)", stem_upper)
    if m:
        return m.group(1)[:3]
    return "UNK"


def _insert_gap_nans(df: pd.DataFrame, gap_factor: float = 2.0) -> pd.DataFrame:
    """
    在时间索引的缺口处插入 NaN 行，使 matplotlib 断开连线。
    gap_factor: 时间间隔超过中位数的多少倍时视为缺口。
    """
    if len(df) < 2:
        return df
    diffs = np.diff(df.index.values).astype("timedelta64[s]").astype(float)
    median_diff = np.median(diffs[diffs > 0])
    threshold = median_diff * gap_factor

    gap_positions = np.where(diffs > threshold)[0]
    if len(gap_positions) == 0:
        return df

    # 在每个缺口处插入一行 NaN
    insert_rows = []
    for pos in gap_positions:
        gap_time = df.index[pos] + pd.Timedelta(seconds=median_diff)
        insert_rows.append(pd.DataFrame(
            {col: [np.nan] for col in df.columns},
            index=[gap_time]
        ))

    result = pd.concat([df] + insert_rows).sort_index()
    return result


def plot_fig1_overview(
    ohlc_all: pd.DataFrame,
    train_end_time: pd.Timestamp,
    eval_df: pd.DataFrame,
    out_path: Path,
):
    if eval_df.empty:
        raise ValueError("eval_df is empty, cannot plot fig1.")
    # Plot only the evaluation window so all three subplots are aligned.
    t0 = eval_df.index.min()
    t1 = eval_df.index.max()
    ohlc = ohlc_all.loc[(ohlc_all.index >= t0) & (ohlc_all.index <= t1)].copy()
    if ohlc.empty:
        raise ValueError("OHLC window is empty; check datetime alignment between eval_df and ohlc_all.")
    ohlc = ohlc[["open", "high", "low", "close"]].sort_index()
    vol_hat_aligned = pd.Series(np.nan, index=ohlc.index, dtype=float, name="vol_hat_1h")
    vol_real_aligned = pd.Series(np.nan, index=ohlc.index, dtype=float, name="vol_real_1h")
    common_idx = eval_df.index.intersection(ohlc.index)
    vol_hat_aligned.loc[common_idx] = eval_df.loc[common_idx, "vol_hat_1h"].to_numpy()
    vol_real_aligned.loc[common_idx] = eval_df.loc[common_idx, "vol_real_1h"].to_numpy()
    x = np.arange(len(ohlc))
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(26, 15), sharex=True,
        gridspec_kw={"height_ratios": [4, 1.6, 1.6]}
    )
    if candlestick2_ohlc is not None:
        candlestick2_ohlc(
            ax1,
            ohlc["open"].to_numpy(),
            ohlc["high"].to_numpy(),
            ohlc["low"].to_numpy(),
            ohlc["close"].to_numpy(),
            width=0.68,
            colorup="salmon",
            colordown="#2ca02c",
        )
    else:
        ax1.plot(x, ohlc["close"].to_numpy(), color="tab:blue", linewidth=1.1, label="close")
        ax1.legend(loc="upper left")
        ax1.text(
            0.01, 0.97,
            "mplfinance not installed; fallback to close line.",
            transform=ax1.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    split_pos = int(np.searchsorted(ohlc.index.values, np.datetime64(train_end_time), side="left"))
    if 0 <= split_pos < len(ohlc):
        for _ax in (ax1, ax2, ax3):
            _ax.axvline(split_pos, color="red", linewidth=1.6, alpha=0.9)
    ax1.set_title("Fig1 (Top): Candlestick")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.18)
    ax2.plot(x, vol_hat_aligned.to_numpy(), color="tab:blue", linewidth=1.2, label="Predicted Vol")
    ax2.set_title("Fig1 (Middle): Predicted Volatility")
    ax2.set_ylabel("Pred Vol")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right")
    ax3.plot(x, vol_real_aligned.to_numpy(), color="tab:orange", linewidth=1.2, label="Realized Vol")
    ax3.set_title("Fig1 (Bottom): Realized Volatility")
    ax3.set_ylabel("Real Vol")
    ax3.set_xlabel("Time")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper right")
    ax3.xaxis.set_major_locator(MaxNLocator(10))
    ax3.xaxis.set_major_formatter(_build_time_formatter(ohlc.index))
    plt.setp(ax3.get_xticklabels(), rotation=25, ha="right")
    fig.suptitle("GARCH Overview: Candlestick + Predicted Vol + Realized Vol", y=1.01)
    save_fig(fig, out_path, close=False)


def plot_fig2_scatter_mz(
    eval_df: pd.DataFrame,
    a: float,
    b: float,
    r2: float,
    pval: float,
    qlike: float,
    out_path: Path,
):
    x = eval_df["sigma2_hat_1h"].values
    y = eval_df["RV_1h"].values

    fig, ax = plt.subplots(figsize=(18, 13))

    ax.scatter(x, y, s=18, alpha=0.25, color="tab:blue", edgecolors="none")

    # 45度线
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.6, label="45°参考线（y=x）")

    # 回归线
    ax.plot([lo, hi], [a + b * lo, a + b * hi], color="red", linewidth=2.2, label="MZ回归线（红）")

    ax.set_title("图2：预测方差 vs 实现方差（散点图，核心图）")
    ax.set_xlabel("GARCH 预测方差 σ̂²（未来1小时，240根bar）")
    ax.set_ylabel("实现方差 RV（未来1小时，240根bar）")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    # 标注：回归方程、R²、F检验p值、QLIKE
    text = (
        "Mincer-Zarnowitz 回归：RV = a + b·σ̂²\n"
        f"a = {a:.6g}\n"
        f"b = {b:.6g}\n"
        f"R² = {r2:.4f}\n"
        f"联合F检验（a=0,b=1）p值 = {pval:.3g}\n"
        f"QLIKE = {qlike:.6g}"
    )
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        va="top",
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    save_fig(fig, out_path, close=False)


def plot_fig3_vol_ts(
    eval_df: pd.DataFrame,
    out_path: Path,
):
    # 插入 NaN 断开停盘缺口的连线
    plot_df = _insert_gap_nans(eval_df[["vol_hat_1h", "vol_real_1h"]])

    fig, ax = plt.subplots(figsize=(26, 11))

    ax.plot(plot_df.index, plot_df["vol_hat_1h"].values, label="预测波动率（蓝）", color="tab:blue")
    ax.plot(plot_df.index, plot_df["vol_real_1h"].values, label="实现波动率（橙）", color="tab:orange")

    ax.set_title("图3：预测波动率 vs 实现波动率（未来1小时）时序对比")
    ax.set_xlabel("时间")
    ax.set_ylabel("波动率（sqrt(方差)）")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    save_fig(fig, out_path, close=False)


def plot_fig4_qq(std_resid: pd.Series, out_path: Path):
    z = std_resid.dropna().values
    z = z[np.isfinite(z)]
    z_sorted = np.sort(z)

    n = len(z_sorted)
    if n < 50:
        warnings.warn(f"QQ图样本量偏少（n={n}），图形解释需谨慎。")

    # 理论正态分位数
    p = (np.arange(1, n + 1) - 0.5) / n
    theo = st.norm.ppf(p)

    fig, ax = plt.subplots(figsize=(16, 13))
    ax.scatter(theo, z_sorted, s=18, alpha=0.35, color="tab:blue", edgecolors="none")

    lo = float(min(theo.min(), z_sorted.min()))
    hi = float(max(theo.max(), z_sorted.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.6)

    ax.set_title("图4：标准化残差 QQ图（对正态分布）")
    ax.set_xlabel("理论分位数 N(0,1)")
    ax.set_ylabel("样本分位数（标准化残差 r_t / σ_t）")
    ax.grid(True, alpha=0.25)

    save_fig(fig, out_path, close=False)


# ----------------------------
# 主流程
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="GARCH(1,1) 模型验证（XAGUSD 15秒K线）")
    parser.add_argument(
        "--csv",
        type=str,
        default=CSV_PATH,
        help="K线CSV路径"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="garch_validate_output",
        help="输出目录（保存PNG与结果）"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="训练集比例（按时间顺序）"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=240,
        help="RV/预测聚合窗口（240根bar=1小时）"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="收益率缩放因子：ret_scaled = logret * scale。默认100（近似百分比），更稳定更直观。"
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    setup_matplotlib_style()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("1) 读取数据")
    df_raw = read_kline_csv(args.csv)
    print(f"原始行数: {len(df_raw):,}")
    print(f"时间范围: {df_raw['datetime'].min()}  ->  {df_raw['datetime'].max()}")
    inferred_bar_seconds = infer_bar_seconds_from_datetime(df_raw["datetime"])
    if inferred_bar_seconds is not None:
        expected_horizon_1h = int(round(3600.0 / inferred_bar_seconds))
        print(
            f"检测到bar周期约 {inferred_bar_seconds:.2f} 秒；"
            f"1小时对应约 {expected_horizon_1h} 根bar"
        )
        if abs(int(args.horizon) - expected_horizon_1h) > 1:
            print(f"自动修正 horizon: {args.horizon} -> {expected_horizon_1h}")
            args.horizon = expected_horizon_1h
        else:
            print(f"当前 horizon={args.horizon}，与检测结果一致")
    else:
        print("提示：未能自动识别bar周期，使用默认 horizon={args.horizon}")

    print("=" * 80)
    print("2) 预处理：对数收益率 + 剔除隔夜跳空")
    df = preprocess_returns(df_raw)
    print(f"预处理后行数(=收益率样本数): {len(df):,}")
    print(f"预处理后时间范围: {df.index.min()}  ->  {df.index.max()}")

    # 收益率（缩放）
    r = (df["ret"] * float(args.scale)).rename("r")
    r = r.replace([np.inf, -np.inf], np.nan).dropna()

    # 70/30 切分（按时间顺序）
    n = len(r)
    split_idx = int(np.floor(n * float(args.train_ratio)))
    split_idx = max(10, min(split_idx, n - 10))  # 防止极端比例导致没法建模
    r_train = r.iloc[:split_idx].copy()
    r_test = r.iloc[split_idx:].copy()

    train_end_time = r_train.index[-1]
    test_start_time = r_test.index[0]

    print(f"训练集样本: {len(r_train):,}  ({r_train.index.min()} -> {r_train.index.max()})")
    print(f"测试集样本: {len(r_test):,}  ({r_test.index.min()} -> {r_test.index.max()})")
    print(f"训练/测试分界(红线): {train_end_time}")

    print("=" * 80)
    print("3) 训练集拟合 GARCH(1,1)（arch库）")
    am = arch_model(
        r_train,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False
    )
    res = am.fit(disp="off")

    # 参数表
    params = res.params
    tvals = res.tvalues
    pvals = res.pvalues
    param_table = pd.DataFrame({
        "param": params,
        "t值": tvals,
        "p值": pvals
    })
    print("\n--- GARCH(1,1) 参数与显著性（训练集） ---")
    print(param_table.to_string(float_format=lambda x: f"{x: .6g}"))

    # 提取关键参数
    # mean 常数项可能叫 mu（常见），但不同版本也可能是 'Const'，做兼容
    if "mu" in params.index:
        mu = float(params["mu"])
    elif "const" in params.index:
        mu = float(params["const"])
    elif "Const" in params.index:
        mu = float(params["Const"])
    else:
        # 极少数情况：如果你改成 mean='Zero' 则没有
        mu = 0.0

    # arch 对 GARCH(1,1) 参数命名通常是 omega, alpha[1], beta[1]
    omega = float(params.get("omega"))
    # 兼容不同命名
    alpha = float(params.get("alpha[1]", params.get("alpha1")))
    beta = float(params.get("beta[1]", params.get("beta1")))
    g = alpha + beta

    print("\n--- 关键参数 ---")
    print(f"mu={mu:.6g}, omega={omega:.6g}, alpha={alpha:.6g}, beta={beta:.6g}, alpha+beta={g:.6g}")

    print("=" * 80)
    print("4) 测试集滚动一步预测（递推更新条件方差）")

    # 训练集最后一个条件方差与残差，用于衔接到测试集
    # res.conditional_volatility 是训练集内的 sigma_t（对 r_train）
    sigma_train = pd.Series(res.conditional_volatility, index=r_train.index, name="sigma_train")
    h_train = sigma_train ** 2

    resid_train = pd.Series(res.resid, index=r_train.index, name="resid_train")

    h_last_train = float(h_train.iloc[-1])
    eps_last_train = float(resid_train.iloc[-1])

    # 测试集每个时点 t 的条件方差 h_t（Var(r_t | t-1)）
    h_test = garch11_roll_conditional_variance(
        r_test=r_test,
        mu=mu,
        omega=omega,
        alpha=alpha,
        beta=beta,
        h_last_train=h_last_train,
        eps_last_train=eps_last_train,
    )
    sigma_test = np.sqrt(h_test)

    # 计算每个“测试集 origin=t” 的一步预测方差 h_{t+1|t}
    eps_test = (r_test - mu).rename("eps_test")
    h1_test_origin = (omega + alpha * (eps_test ** 2) + beta * h_test).rename("h1_next")

    # 聚合成 1小时预测方差（未来240根bar方差之和）
    sigma2_hat_1h = aggregate_garch_variance_1h_from_h1(
        h1=h1_test_origin,
        omega=omega,
        g=g,
        horizon=int(args.horizon)
    ).rename("sigma2_hat_1h")

    # 实现方差 RV（在“全样本”上算未来窗，然后取测试集部分）
    rv_all = realized_variance_forward(r, horizon=int(args.horizon))
    rv_test_origin = rv_all.reindex(r_test.index).rename("RV_1h")

    # 评估样本（测试集 origin）需要：sigma2_hat_1h 与 RV_1h 同时非空
    eval_df = pd.DataFrame({
        "sigma2_hat_1h": sigma2_hat_1h,
        "RV_1h": rv_test_origin,
    }).dropna()

    # 波动率形式（用于图3）
    eval_df["vol_hat_1h"] = np.sqrt(eval_df["sigma2_hat_1h"])
    eval_df["vol_real_1h"] = np.sqrt(eval_df["RV_1h"])

    print(f"可评估样本数（测试集且未来有{args.horizon}根bar）: {len(eval_df):,}")

    print("=" * 80)
    print("5) Mincer-Zarnowitz 回归 + QLIKE")

    ols, a, b, r2_mz, pval_joint, mz_df = mincer_zarnowitz_regression(
        rv=eval_df["RV_1h"],
        sigma2_hat=eval_df["sigma2_hat_1h"],
    )

    # QLIKE
    sig2 = eval_df["sigma2_hat_1h"].values
    rvv = eval_df["RV_1h"].values
    sig2 = np.clip(sig2, 1e-18, None)
    qlike = float(np.mean(np.log(sig2) + rvv / sig2))
    mean_pred = float(np.mean(sig2))
    mean_rv = float(np.mean(rvv))
    ratio_pred_to_rv = mean_pred / mean_rv if mean_rv > 0 else np.nan

    print("\n--- Mincer-Zarnowitz 回归结果（RV = a + b*sigma_hat^2）---")
    print(f"a = {a:.6g}")
    print(f"b = {b:.6g}")
    print(f"R² = {r2_mz:.4f}")
    print(f"联合F检验（a=0,b=1）p值 = {pval_joint:.3g}")
    print(f"QLIKE = {qlike:.6g}")
    print(f"mean(sigma2_hat)/mean(RV) = {ratio_pred_to_rv:.4f}")

    # 可选：保存评估数据
    eval_csv = out_dir / "eval_forecast_vs_rv.csv"
    eval_df.to_csv(eval_csv, encoding="utf-8-sig")
    print(f"\n评估明细已保存：{eval_csv}")

    print("=" * 80)
    print("6) 生成图表（仅显示，不保存）")

    # 图1：上方K线，下方两栏分别显示预测波动/实现波动
    ohlc_all = (
        df_raw.set_index("datetime")[["open", "high", "low", "close"]]
        .sort_index()
        .copy()
    )

    fig1_path = out_dir / "图1_全局概览.png"
    plot_fig1_overview(
        ohlc_all=ohlc_all,
        train_end_time=train_end_time,
        eval_df=eval_df,
        out_path=fig1_path
    )
    print("图1已生成（不保存）")

    fig2_path = out_dir / "图2_预测vs实现_散点_MZ.png"
    plot_fig2_scatter_mz(
        eval_df=eval_df,
        a=a, b=b, r2=r2_mz, pval=pval_joint, qlike=qlike,
        out_path=fig2_path
    )
    print("图2已生成（不保存）")

    fig3_path = out_dir / "图3_波动率时序对比.png"
    plot_fig3_vol_ts(eval_df=eval_df, out_path=fig3_path)
    print("图3已生成（不保存）")

    # QQ图：用训练集标准化残差（诊断模型）
    std_resid_train = (resid_train / sigma_train).rename("std_resid")
    fig4_path = out_dir / "图4_标准化残差QQ图.png"
    plot_fig4_qq(std_resid=std_resid_train, out_path=fig4_path)
    print("图4已生成（不保存）")

    print("=" * 80)
    print("全部完成。输出目录：", out_dir.resolve())
    print("注意：收益率使用 ret_scaled = logret * scale；当前 scale =", args.scale)

    # ----------------------------
    # 输出回归与QLIKE的最终结果
    # ----------------------------
    def print_results(a, b, r2_mz, pval_joint, qlike):
        print("=" * 80)
        print("6) 最终验证指标")
    
        # 打印 Mincer-Zarnowitz 回归结果
        print(f"Mincer-Zarnowitz 回归：RV = a + b·σ̂²")
        print(f"a = {a:.6g}")
        print(f"b = {b:.6g}")
        print(f"R² = {r2_mz:.4f}")
        print(f"联合F检验（a=0,b=1）p值 = {pval_joint:.3g}")
    
        # 打印 QLIKE 指标
        print(f"QLIKE = {qlike:.6g}")
        print("=" * 80)
    
    # 在主流程中调用
    print_results(a, b, r2_mz, pval_joint, qlike)

    print("=" * 80)
    print("7) 结果有效性判断")
    is_valid, invalid_reasons, expected_h_1h, checked_ratio = evaluate_result_validity(
        horizon=int(args.horizon),
        inferred_bar_seconds=inferred_bar_seconds,
        eval_df=eval_df,
        qlike=qlike,
        omega=omega,
        alpha=alpha,
        beta=beta,
        g=g,
        mean_pred=mean_pred,
        mean_rv=mean_rv,
    )
    if expected_h_1h is not None:
        print(f"口径检查：当前 horizon={args.horizon}, 1小时期望≈{expected_h_1h}")
    print(f"量级检查：mean_pred/mean_rv = {checked_ratio:.4f}")
    print(f"有效性结论：{'有效' if is_valid else '无效'}")
    if is_valid:
        print("结果通过基础有效性检查（口径、样本量、数值稳定性）。")
    else:
        print("无效原因：")
        for reason in invalid_reasons:
            print(f"- {reason}")

    # ----------------------------
    # 8) 导出 GARCH 参数到 xlsx
    # ----------------------------
    print("="* 80)
    print("8) 导出 GARCH 参数到 xlsx")
    param_export = pd.DataFrame([{
        "csv_path": str(args.csv),
        "train_ratio": float(args.train_ratio),
        "horizon": int(args.horizon),
        "scale": float(args.scale),
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "alpha_plus_beta": g,
    }])
    symbol = infer_symbol_from_csv_path(str(args.csv))
    period_label = format_period_label(inferred_bar_seconds)
    xlsx_path = out_dir / f"garch_params_{symbol}_{period_label}.xlsx"
    param_export.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"参数已保存：{xlsx_path}")

    print("9) 显示全部图窗 ...")
    plt.show()
        
if __name__ == "__main__":
    main()
    


