from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

try:
    from arch import arch_model
except ImportError as exc:
    raise SystemExit("arch is required. Install it with: pip install arch") from exc


DEFAULT_CSV_PATH = r"D:\Code\data\20260326\yearly_30s\xagusd_30s_2025.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "garch_validate_output"
DEFAULT_START_DATE = "20250601"
DEFAULT_END_DATE = "20250615"
DEFAULT_TRAIN_DAYS = 120
DEFAULT_RESAMPLE_RULE = "5min"
DEFAULT_SCALE = 100.0
DEFAULT_FUTURE_MINUTES = 60
ZERO_VOLUME_GAP_MINUTES = 15


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
        }
    )


def read_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    column_names = ["datetime", "open", "high", "low", "close", "volume"]
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")

    df.columns = [str(col).strip().lower() for col in df.columns]
    if "datetime" not in df.columns:
        df = pd.read_csv(path, sep=None, engine="python", header=None, names=column_names)
        df.columns = column_names

    missing = [name for name in column_names if name not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[column_names].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for name in ["open", "high", "low", "close", "volume"]:
        df[name] = pd.to_numeric(df[name], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    df = df[df["close"] > 0].copy()
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    temp = df.set_index("datetime").sort_index()
    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = temp.resample(rule, label="left", closed="left").agg(agg_map)
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def infer_bar_seconds(dt: pd.Series | pd.DatetimeIndex) -> float | None:
    idx = pd.DatetimeIndex(pd.to_datetime(dt, errors="coerce")).dropna()
    if len(idx) < 2:
        return None

    diffs = pd.Series(idx).diff().dt.total_seconds().dropna()
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.empty:
        return None
    return float(np.median(diffs.to_numpy()))


def zero_volume_gap_mask(index: pd.DatetimeIndex, volume: pd.Series, threshold_minutes: float) -> pd.Series:
    is_zero = volume.fillna(0).eq(0).to_numpy()
    dt_values = index.values
    mask = np.zeros(len(index), dtype=bool)

    i = 0
    while i < len(index):
        if not is_zero[i]:
            i += 1
            continue

        j = i
        while j < len(index) and is_zero[j]:
            j += 1

        span_minutes = (dt_values[j - 1] - dt_values[i]) / np.timedelta64(1, "m")
        if span_minutes >= threshold_minutes:
            mask[i:j] = True
        i = j

    return pd.Series(mask, index=index)


def preprocess_returns(df: pd.DataFrame, zero_gap_minutes: float = ZERO_VOLUME_GAP_MINUTES) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("datetime").set_index("datetime")
    out["ret"] = np.log(out["close"]).diff()

    bar_seconds = infer_bar_seconds(out.index)
    if bar_seconds is not None:
        gap_seconds = pd.Series(out.index, index=out.index).diff().dt.total_seconds()
        out.loc[gap_seconds > bar_seconds * 1.5, "ret"] = np.nan

    gap_mask = zero_volume_gap_mask(out.index, out["volume"], threshold_minutes=zero_gap_minutes)
    if gap_mask.any():
        out.loc[gap_mask, "ret"] = np.nan
        gap_end = gap_mask & ~gap_mask.shift(-1, fill_value=False)
        gap_end_positions = np.where(gap_end.to_numpy())[0]
        for pos in gap_end_positions:
            next_pos = pos + 1
            if next_pos < len(out):
                out.iloc[next_pos, out.columns.get_loc("ret")] = np.nan

    out = out.dropna(subset=["ret"]).copy()
    return out


def contiguous_forward_window_mask(index: pd.DatetimeIndex, horizon: int, bar_seconds: float) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    idx_series = pd.Series(idx, index=idx)
    actual_span = idx_series.shift(-horizon) - idx_series
    expected_span = pd.to_timedelta(horizon * bar_seconds, unit="s")
    tolerance = pd.to_timedelta(max(1.0, bar_seconds * 0.1), unit="s")
    mask = (actual_span >= expected_span - tolerance) & (actual_span <= expected_span + tolerance)
    return mask.fillna(False)


def parse_date_boundary(value: str, bar_delta: pd.Timedelta, as_end: bool) -> pd.Timestamp:
    text = str(value).strip()
    ts = pd.to_datetime(text)
    if re.fullmatch(r"\d{8}", text) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        if as_end:
            return ts + pd.Timedelta(days=1) - bar_delta
        return ts
    return ts


def rolling_conditional_variance(
    returns: pd.Series,
    *,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    h_last: float,
    eps_last: float,
) -> pd.Series:
    values: list[float] = []
    h_prev = float(h_last)
    eps_prev = float(eps_last)

    for rt in returns.to_numpy():
        h_t = omega + alpha * (eps_prev**2) + beta * h_prev
        values.append(h_t)
        eps_prev = float(rt - mu)
        h_prev = float(h_t)

    return pd.Series(values, index=returns.index, name="h_t")


def aggregate_future_variance(h1: pd.Series, omega: float, g: float, horizon: int) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if abs(1.0 - g) < 1e-10:
        out = horizon * h1 + omega * (horizon * (horizon - 1) / 2.0)
        return out.rename("sigma2_hat")

    a_term = (1.0 - g**horizon) / (1.0 - g)
    out = (omega / (1.0 - g)) * (horizon - a_term) + h1 * a_term
    return out.rename("sigma2_hat")


def realized_variance_forward(returns: pd.Series, horizon: int) -> pd.Series:
    ret2 = returns**2
    forward_sum = ret2.iloc[::-1].rolling(window=horizon, min_periods=horizon).sum().iloc[::-1]
    return forward_sum.shift(-1).rename("rv")


def mincer_zarnowitz(
    rv: pd.Series, sigma2_hat: pd.Series
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, float, float, float, float]:
    frame = pd.DataFrame({"rv": rv, "sigma2_hat": sigma2_hat}).dropna()
    x = sm.add_constant(frame["sigma2_hat"])
    model = sm.OLS(frame["rv"], x).fit()
    a_value = float(model.params["const"])
    b_value = float(model.params["sigma2_hat"])
    r2_value = float(model.rsquared)
    restriction = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array([0.0, 1.0])
    f_test = model.f_test((restriction, target))
    p_value = float(f_test.pvalue)
    return model, a_value, b_value, r2_value, p_value


def qlike_loss(rv: pd.Series, sigma2_hat: pd.Series) -> float:
    frame = pd.DataFrame({"rv": rv, "sigma2_hat": sigma2_hat}).dropna()
    sig2 = np.clip(frame["sigma2_hat"].to_numpy(), 1e-18, None)
    rv_values = np.clip(frame["rv"].to_numpy(), 1e-18, None)
    return float(np.mean(np.log(sig2) + rv_values / sig2))


def safe_correlation(a: Iterable[float], b: Iterable[float]) -> float:
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(arr_a[mask], arr_b[mask])[0, 1])


def compute_rmse(a: Iterable[float], b: Iterable[float]) -> float:
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((arr_a[mask] - arr_b[mask]) ** 2)))


def build_time_formatter(index: pd.DatetimeIndex) -> FuncFormatter:
    def formatter(x: float, _pos: int) -> str:
        loc = int(round(x))
        if loc < 0 or loc >= len(index):
            return ""
        return index[loc].strftime("%m-%d %H:%M")

    return FuncFormatter(formatter)


def draw_bw_candles(ax: plt.Axes, ohlc: pd.DataFrame, width: float = 0.86) -> None:
    price_span = float(ohlc["high"].max() - ohlc["low"].min())
    min_body = max(price_span * 0.00035, 1e-6)

    for i, row in enumerate(ohlc.itertuples(index=False)):
        open_price = float(row.open)
        high_price = float(row.high)
        low_price = float(row.low)
        close_price = float(row.close)

        ax.vlines(i, low_price, high_price, color="black", linewidth=0.85, zorder=2)
        lower = min(open_price, close_price)
        height = max(abs(close_price - open_price), min_body)
        face = "white" if close_price >= open_price else "black"
        body = Rectangle(
            (i - width / 2.0, lower),
            width,
            height,
            facecolor=face,
            edgecolor="black",
            linewidth=0.85,
            zorder=3,
        )
        ax.add_patch(body)

    ax.set_xlim(-1, len(ohlc))


def plot_prediction_panels(
    ohlc_window: pd.DataFrame,
    eval_plot_df: pd.DataFrame,
    out_path: Path,
    title: str,
    horizon_minutes: int,
) -> None:
    aligned = pd.DataFrame(index=ohlc_window.index)
    aligned["pred_vol"] = eval_plot_df["pred_vol"].reindex(ohlc_window.index)
    aligned["real_vol"] = eval_plot_df["real_vol"].reindex(ohlc_window.index)

    x = np.arange(len(ohlc_window))
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(36, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [4.2, 1.8, 1.8]},
    )

    draw_bw_candles(ax1, ohlc_window.reset_index()[["open", "high", "low", "close"]])
    ax1.set_title(title)
    ax1.set_ylabel("Price")
    ax1.grid(True, axis="y", alpha=0.16)

    ax2.plot(x, aligned["pred_vol"].to_numpy(), color="black", linewidth=1.2)
    ax2.set_ylabel(f"Pred {horizon_minutes}m vol")
    ax2.set_title("GARCH forecast volatility")
    ax2.grid(True, alpha=0.18)

    ax3.plot(x, aligned["real_vol"].to_numpy(), color="dimgray", linewidth=1.2)
    ax3.set_ylabel(f"Real {horizon_minutes}m vol")
    ax3.set_title("Realized volatility")
    ax3.grid(True, alpha=0.18)
    ax3.set_xlabel("Time")

    ax3.xaxis.set_major_locator(MaxNLocator(10))
    ax3.xaxis.set_major_formatter(build_time_formatter(ohlc_window.index))
    plt.setp(ax3.get_xticklabels(), rotation=25, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def interpretation_lines(metrics: dict[str, float | int | str]) -> list[str]:
    ratio = (
        float(metrics["mean_pred_vol"] / metrics["mean_real_vol"])
        if float(metrics["mean_real_vol"]) != 0.0
        else float("nan")
    )
    corr_value = float(metrics["vol_corr"])
    slope = float(metrics["mz_b"])
    qlike = float(metrics["qlike"])

    lines: list[str] = []
    if np.isfinite(ratio):
        if ratio > 1.1:
            lines.append("The forecast volatility level stays above the realized level for most of the window.")
        elif ratio < 0.9:
            lines.append("The forecast volatility level stays below the realized level for most of the window.")
        else:
            lines.append("The forecast volatility level is close to the realized level on average.")

    if np.isfinite(corr_value):
        if corr_value >= 0.6:
            lines.append("The model tracks volatility swings reasonably well in this window.")
        elif corr_value >= 0.3:
            lines.append("The model captures part of the volatility swings, though the fit is still loose.")
        else:
            lines.append("The model reacts to volatility swings weakly in this window.")

    if np.isfinite(slope):
        if 0.8 <= slope <= 1.2:
            lines.append("The Mincer-Zarnowitz slope stays near 1, so the calibration is acceptable.")
        else:
            lines.append("The Mincer-Zarnowitz slope is far from 1, so the calibration needs caution.")

    if np.isfinite(qlike):
        lines.append(f"The QLIKE value is {qlike:.6f}. Lower values indicate a tighter forecast.")

    return lines


def write_report(report_path: Path, metrics: dict[str, float | int | str], params: dict[str, float], paths: dict[str, str]) -> None:
    lines: list[str] = []
    lines.append("# XAGUSD GARCH forecast report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Source CSV: `{paths['csv']}`")
    lines.append(f"- Resampled bar: `{paths['bar_label']}`")
    lines.append(f"- Training window: `{paths['train_window']}`")
    lines.append(f"- Forecast window: `{paths['predict_window']}`")
    lines.append(f"- Forecast horizon: `{paths['horizon_label']}`")
    lines.append(f"- Figure: `{paths['figure']}`")
    lines.append(f"- Evaluation CSV: `{paths['eval_csv']}`")
    lines.append("")
    lines.append("## Model")
    lines.append(f"- mu: `{params['mu']:.8f}`")
    lines.append(f"- omega: `{params['omega']:.8f}`")
    lines.append(f"- alpha: `{params['alpha']:.8f}`")
    lines.append(f"- beta: `{params['beta']:.8f}`")
    lines.append(f"- alpha + beta: `{params['alpha_beta']:.8f}`")
    lines.append("")
    lines.append("## Evaluation")
    lines.append(f"- Train samples: `{metrics['train_samples']}`")
    lines.append(f"- Forecast samples: `{metrics['forecast_samples']}`")
    lines.append(f"- Eval samples: `{metrics['eval_samples']}`")
    lines.append(f"- Mean predicted volatility: `{metrics['mean_pred_vol']:.6f}`")
    lines.append(f"- Mean realized volatility: `{metrics['mean_real_vol']:.6f}`")
    lines.append(f"- Volatility correlation: `{metrics['vol_corr']:.6f}`")
    lines.append(f"- Volatility RMSE: `{metrics['vol_rmse']:.6f}`")
    lines.append(f"- MZ intercept a: `{metrics['mz_a']:.6f}`")
    lines.append(f"- MZ slope b: `{metrics['mz_b']:.6f}`")
    lines.append(f"- MZ R^2: `{metrics['mz_r2']:.6f}`")
    lines.append(f"- MZ joint F-test p-value: `{metrics['mz_p']:.6f}`")
    lines.append(f"- QLIKE: `{metrics['qlike']:.6f}`")
    lines.append(f"- Peak predicted volatility time: `{metrics['peak_pred_time']}`")
    lines.append(f"- Peak predicted volatility: `{metrics['peak_pred_vol']:.6f}`")
    lines.append(f"- Peak realized volatility time: `{metrics['peak_real_time']}`")
    lines.append(f"- Peak realized volatility: `{metrics['peak_real_vol']:.6f}`")
    lines.append("")
    lines.append("## Reading")
    for text in interpretation_lines(metrics):
        lines.append(f"- {text}")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="XAGUSD GARCH forecast on 5-minute bars")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Source CSV path")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--start_date", type=str, default=DEFAULT_START_DATE, help="Forecast start date or timestamp")
    parser.add_argument("--end_date", type=str, default=DEFAULT_END_DATE, help="Forecast end date or timestamp")
    parser.add_argument("--train_days", type=int, default=DEFAULT_TRAIN_DAYS, help="Calendar days used for fitting")
    parser.add_argument("--resample_rule", type=str, default=DEFAULT_RESAMPLE_RULE, help="Resample rule")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Return scale factor")
    parser.add_argument(
        "--future_minutes",
        type=int,
        default=DEFAULT_FUTURE_MINUTES,
        help="Volatility forecast horizon in minutes",
    )
    args = parser.parse_args()

    setup_plot_style()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_ohlcv_csv(args.csv)
    bar_df = resample_ohlcv(raw_df, args.resample_rule)
    processed = preprocess_returns(bar_df)

    bar_seconds = infer_bar_seconds(bar_df["datetime"])
    if bar_seconds is None:
        raise RuntimeError("Unable to infer the bar period after resampling.")
    bar_minutes = int(round(bar_seconds / 60.0))
    if bar_minutes <= 0:
        raise RuntimeError("Invalid bar period detected.")

    horizon_bars = max(1, int(round(args.future_minutes * 60.0 / bar_seconds)))
    horizon_minutes = horizon_bars * bar_minutes
    bar_delta = pd.Timedelta(seconds=bar_seconds)

    predict_start = parse_date_boundary(args.start_date, bar_delta=bar_delta, as_end=False)
    predict_end = parse_date_boundary(args.end_date, bar_delta=bar_delta, as_end=True)
    train_start = predict_start - pd.Timedelta(days=int(args.train_days))
    train_end = predict_start - bar_delta

    returns = (processed["ret"] * float(args.scale)).rename("r").replace([np.inf, -np.inf], np.nan).dropna()
    r_train = returns.loc[(returns.index >= train_start) & (returns.index <= train_end)].copy()
    r_predict = returns.loc[(returns.index >= predict_start) & (returns.index <= predict_end)].copy()

    if len(r_train) < 500:
        raise RuntimeError(f"Training samples are too small: {len(r_train)}")
    if len(r_predict) < 50:
        raise RuntimeError(f"Forecast samples are too small: {len(r_predict)}")

    model = arch_model(
        r_train,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    )
    result = model.fit(disp="off")

    params = result.params
    mu = float(params["mu"] if "mu" in params.index else params.get("Const", params.get("const", 0.0)))
    omega = float(params["omega"])
    alpha = float(params.get("alpha[1]", params.get("alpha1")))
    beta = float(params.get("beta[1]", params.get("beta1")))
    alpha_beta = alpha + beta

    sigma_train = pd.Series(result.conditional_volatility, index=r_train.index)
    resid_train = pd.Series(result.resid, index=r_train.index)
    h_last_train = float((sigma_train.iloc[-1]) ** 2)
    eps_last_train = float(resid_train.iloc[-1])

    h_predict = rolling_conditional_variance(
        r_predict,
        mu=mu,
        omega=omega,
        alpha=alpha,
        beta=beta,
        h_last=h_last_train,
        eps_last=eps_last_train,
    )
    eps_predict = (r_predict - mu).rename("eps")
    h1_next = (omega + alpha * (eps_predict**2) + beta * h_predict).rename("h1_next")
    sigma2_hat = aggregate_future_variance(h1_next, omega=omega, g=alpha_beta, horizon=horizon_bars)
    rv_all = realized_variance_forward(returns, horizon=horizon_bars)
    rv_predict = rv_all.reindex(r_predict.index)
    valid_forward_mask = contiguous_forward_window_mask(returns.index, horizon=horizon_bars, bar_seconds=bar_seconds)
    valid_predict_mask = valid_forward_mask.reindex(r_predict.index).fillna(False)

    eval_df = pd.DataFrame(index=r_predict.index)
    eval_df["sigma2_hat"] = sigma2_hat.where(valid_predict_mask)
    eval_df["rv"] = rv_predict.where(valid_predict_mask)
    eval_df["pred_vol"] = np.sqrt(np.clip(eval_df["sigma2_hat"], 0.0, None))
    eval_df["real_vol"] = np.sqrt(np.clip(eval_df["rv"], 0.0, None))
    eval_valid = eval_df.dropna().copy()

    if eval_valid.empty:
        raise RuntimeError("No valid forecast rows remain after aligning realized variance.")

    _, mz_a, mz_b, mz_r2, mz_p = mincer_zarnowitz(eval_valid["rv"], eval_valid["sigma2_hat"])
    qlike = qlike_loss(eval_valid["rv"], eval_valid["sigma2_hat"])
    vol_corr = safe_correlation(eval_valid["pred_vol"], eval_valid["real_vol"])
    vol_rmse = compute_rmse(eval_valid["pred_vol"], eval_valid["real_vol"])

    peak_pred_time = eval_valid["pred_vol"].idxmax()
    peak_real_time = eval_valid["real_vol"].idxmax()

    plot_window = bar_df.set_index("datetime").sort_index().loc[predict_start:predict_end].copy()
    if plot_window.empty:
        raise RuntimeError("The forecast window has no OHLC bars to plot.")

    prefix = f"xagusd_garch_5m_{predict_start.strftime('%Y%m%d')}_{predict_end.strftime('%Y%m%d')}"
    figure_path = out_dir / f"{prefix}.png"
    eval_csv_path = out_dir / f"{prefix}_eval.csv"
    params_csv_path = out_dir / f"{prefix}_params.csv"
    report_path = out_dir / f"{prefix}_report.md"

    plot_prediction_panels(
        ohlc_window=plot_window,
        eval_plot_df=eval_df,
        out_path=figure_path,
        title=f"XAGUSD 5m GARCH(1,1) forecast | {predict_start:%Y-%m-%d} to {predict_end:%Y-%m-%d}",
        horizon_minutes=horizon_minutes,
    )

    eval_df.to_csv(eval_csv_path, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "mu": mu,
                "omega": omega,
                "alpha": alpha,
                "beta": beta,
                "alpha_plus_beta": alpha_beta,
                "train_start": train_start,
                "train_end": train_end,
                "predict_start": predict_start,
                "predict_end": predict_end,
                "bar_minutes": bar_minutes,
                "horizon_bars": horizon_bars,
                "horizon_minutes": horizon_minutes,
                "scale": args.scale,
            }
        ]
    ).to_csv(params_csv_path, index=False, encoding="utf-8-sig")

    metrics: dict[str, float | int | str] = {
        "train_samples": int(len(r_train)),
        "forecast_samples": int(len(r_predict)),
        "eval_samples": int(len(eval_valid)),
        "mean_pred_vol": float(eval_valid["pred_vol"].mean()),
        "mean_real_vol": float(eval_valid["real_vol"].mean()),
        "vol_corr": float(vol_corr),
        "vol_rmse": float(vol_rmse),
        "mz_a": float(mz_a),
        "mz_b": float(mz_b),
        "mz_r2": float(mz_r2),
        "mz_p": float(mz_p),
        "qlike": float(qlike),
        "peak_pred_time": str(peak_pred_time),
        "peak_pred_vol": float(eval_valid.loc[peak_pred_time, "pred_vol"]),
        "peak_real_time": str(peak_real_time),
        "peak_real_vol": float(eval_valid.loc[peak_real_time, "real_vol"]),
    }
    write_report(
        report_path=report_path,
        metrics=metrics,
        params={
            "mu": mu,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "alpha_beta": alpha_beta,
        },
        paths={
            "csv": str(Path(args.csv).resolve()),
            "bar_label": f"{bar_minutes}m",
            "train_window": f"{train_start} -> {train_end}",
            "predict_window": f"{predict_start} -> {predict_end}",
            "horizon_label": f"{horizon_bars} bars / {horizon_minutes} minutes",
            "figure": str(figure_path),
            "eval_csv": str(eval_csv_path),
        },
    )

    print("=" * 72)
    print("XAGUSD GARCH forecast finished")
    print(f"source_csv      : {Path(args.csv).resolve()}")
    print(f"bar_period      : {bar_minutes} minutes")
    print(f"train_window    : {train_start} -> {train_end}")
    print(f"forecast_window : {predict_start} -> {predict_end}")
    print(f"horizon         : {horizon_bars} bars / {horizon_minutes} minutes")
    print(f"train_samples   : {len(r_train)}")
    print(f"forecast_samples: {len(r_predict)}")
    print(f"eval_samples    : {len(eval_valid)}")
    print(f"mu              : {mu:.8f}")
    print(f"omega           : {omega:.8f}")
    print(f"alpha           : {alpha:.8f}")
    print(f"beta            : {beta:.8f}")
    print(f"alpha+beta      : {alpha_beta:.8f}")
    print(f"mean_pred_vol   : {metrics['mean_pred_vol']:.6f}")
    print(f"mean_real_vol   : {metrics['mean_real_vol']:.6f}")
    print(f"vol_corr        : {metrics['vol_corr']:.6f}")
    print(f"vol_rmse        : {metrics['vol_rmse']:.6f}")
    print(f"mz_a            : {metrics['mz_a']:.6f}")
    print(f"mz_b            : {metrics['mz_b']:.6f}")
    print(f"mz_r2           : {metrics['mz_r2']:.6f}")
    print(f"mz_p            : {metrics['mz_p']:.6f}")
    print(f"qlike           : {metrics['qlike']:.6f}")
    print(f"figure          : {figure_path}")
    print(f"eval_csv        : {eval_csv_path}")
    print(f"params_csv      : {params_csv_path}")
    print(f"report          : {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
