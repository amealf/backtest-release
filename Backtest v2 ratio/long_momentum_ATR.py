# -*- coding: utf-8 -*-
"""
Long Momentum Strategy - 动量做多策略
=====================================
策略入口脚本：包含 MomentumStrategy 类、参数循环、绘图、Excel 输出。
依赖 backtest_main.py 中的通用框架。
"""
import pandas as pd
import numpy as np
import matplotlib
try:
    matplotlib.use('qtagg', force=True)
except Exception:
    try:
        matplotlib.use('tkagg', force=True)
    except Exception:
        pass
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker
from mplfinance.original_flavor import candlestick2_ohlc
import time, os
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None
import sys, os as _os
sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
from backtest_main import (
    BacktestEngine, BaseStrategy,
    BarContext, OpenResult, CloseResult,
    generate_performance, load_data,
    plot_backtest_chart,
)
start_time = time.time()
# ============================================================
# User Config
# ============================================================
# 数据
data_folder_path = r"D:\Code\data\20260324\\"
data_file_name = "xagusd_30s_all"

# 回测区间
# data_selection_mode:
# 'index' = 使用 start_index / end_index，在原始数据上按切片语义 [start_index, end_index) 取数
# 'date' = 使用 start_date / end_date，在原始数据上按时间 between 取数
data_selection_mode = 'date'
start_index = 400000
end_index = 500000  # 或 'latest'
start_date = '20250602'
end_date = '20250610'  # 或 '2024-12-31 23:59:59'
only_close = False

# 重采样设置：设为 '' 表示直接使用原始周期
# 例如 '1min' / '5min' / '15min' / '1H'
resample_rule = '1min'

# 运行模式：
# 'manual' = 使用当前参数直接回测，并弹出 K 线买卖点图
# 'grid' = 执行网格搜索，并输出参数结果图
run_mode = 'manual'

# 兼容保留：当前版本未启用第二套开仓参数。
open_bar2 = np.nan

# 策略参数（直接使用 bar 数，单位是当前实际回测周期）
open_bar = 30
close_bar = open_bar

atr_period = 70
open_atr_multiplier = 2.5
open_continous_atr_multiplier = open_atr_multiplier
# 连续多少次上调 ATR 倍数后，总交易数保持不变，就停止当前 bar 的搜索。
open_atr_stop_flat_rounds = 5
# 安全上限，防止极端数据导致 ATR 搜索循环过长。
open_atr_max_iterations = 100
open_wd_atr_multiplier = open_atr_multiplier
close_speed_atr_multiplier = open_atr_multiplier

# Grid search
bar_step = 3
bar_end = 99
atr_step = 0.1
for_num_3 = open_atr_max_iterations
step3 = 0.1

# 当前不使用固定 close_wd_atr_multiplier，持仓止损改成动态阈值。
close_wd_atr_multiplier = 1.0
# 每根持仓 bar 都更新一次，阈值等于当前最大涨幅的 close_wd_max_inc_ratio。
close_wd_max_inc_ratio = 2.0 / 3.0

commision_percent = 0.000
capital = 100.0
# 网格搜索时建议关闭逐次图表与明细导出。
EXPORT_INTERACTIVE_HTML = False
EXPORT_STATS = False
ACCENT_BLUE = '#1F77B4'
SELL_WD_COLOR = 'green'
SELL_SPEED_COLOR = '#D4AA00'
HTML_CROSSHAIR_ENABLED = False
HTML_CROSSHAIR_COLOR = 'rgba(255, 120, 120, 0.45)'
HTML_SHOW_TRADE_COUNT_BADGE = True
# 静态图保存开关：默认不保存 PDF/PNG（保留 HTML 导出）
SAVE_STATIC_PLOT = False
# 当 SAVE_STATIC_PLOT=True 时决定保存为 PDF 或 PNG
SAVE_PLOT_AS_PDF = False


def detect_bar_seconds_from_df(df: pd.DataFrame) -> int:
    dates = pd.to_datetime(df['Date'], errors='coerce')
    diffs = dates.diff().dropna()
    if len(diffs) > 50:
        diffs = diffs.iloc[:50]
    median_delta = diffs.median()
    if pd.isna(median_delta):
        raise ValueError('Cannot detect bar period from Date column.')
    total_seconds = int(median_delta.total_seconds())
    if total_seconds <= 0:
        raise ValueError(f'Invalid detected bar period: {total_seconds}')
    return total_seconds


def get_long_gap_marker_positions(
        df: pd.DataFrame,
        min_gap: pd.Timedelta = pd.Timedelta(days=1),
        use_actual_index: bool = False) -> list[float]:
    if df.empty or 'Date' not in df.columns or len(df) < 2:
        return []

    dates = pd.to_datetime(df['Date'], errors='coerce')
    positions = []
    for pos in range(1, len(dates)):
        prev_dt = dates.iloc[pos - 1]
        curr_dt = dates.iloc[pos]
        if pd.isna(prev_dt) or pd.isna(curr_dt):
            continue
        if curr_dt - prev_dt <= min_gap:
            continue
        if use_actual_index:
            try:
                prev_x = float(df.index[pos - 1])
                curr_x = float(df.index[pos])
                gap_x = (prev_x + curr_x) / 2.0
            except Exception:
                gap_x = float(pos) - 0.5
        else:
            gap_x = float(pos) - 0.5
        positions.append(gap_x)
    return positions


def add_long_gap_shapes(fig, df: pd.DataFrame) -> None:
    for gap_x in get_long_gap_marker_positions(df, use_actual_index=True):
        fig.add_shape(
            type='line',
            x0=gap_x,
            x1=gap_x,
            y0=0,
            y1=1,
            xref='x',
            yref='paper',
            layer='above',
            line=dict(color='rgba(128, 128, 128, 0.30)', width=1.0, dash='dash'),
        )


def draw_long_gap_lines(ax, df: pd.DataFrame) -> None:
    for gap_x in get_long_gap_marker_positions(df):
        ax.axvline(
            x=gap_x,
            color='gray',
            alpha=0.30,
            linestyle='--',
            linewidth=1.0,
            zorder=0,
        )


def resample_ohlc_df(df: pd.DataFrame, rule: str):
    resample_rule = (rule or '').strip()
    if not resample_rule:
        return df.copy(), detect_bar_seconds_from_df(df)

    temp = df.copy()
    temp['Date'] = pd.to_datetime(temp['Date'], errors='coerce')
    temp = temp.dropna(subset=['Date']).sort_values('Date')
    temp = temp.set_index('Date')
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }
    if 'vol' in temp.columns:
        agg['vol'] = 'sum'
    temp = temp.resample(resample_rule).agg(agg)
    temp = temp.dropna(subset=['open', 'high', 'low', 'close']).reset_index()
    temp['Date'] = temp['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    if 'vol' not in temp.columns:
        temp['vol'] = 0.0
    return temp, detect_bar_seconds_from_df(temp)


def format_period_label(resample_rule: str, bar_seconds: int) -> str:
    rule = (resample_rule or '').strip()
    if rule:
        return rule.replace(' ', '')
    if bar_seconds % 3600 == 0:
        hours = bar_seconds // 3600
        return f'{hours}h'
    if bar_seconds % 60 == 0:
        minutes = bar_seconds // 60
        return f'{minutes}min'
    return f'{bar_seconds}s'


def make_safe_range_token(value) -> str:
    text = str(value).strip()
    return (
        text.replace(':', '-')
        .replace(' ', '_')
        .replace('/', '-')
        .replace('\\', '-')
    )


def parse_selection_datetime(value, is_end: bool = False) -> pd.Timestamp:
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        ts = pd.to_datetime(text, format='%Y%m%d', errors='coerce')
        if pd.isna(ts):
            raise ValueError(f'Invalid date value: {value}')
        if is_end:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return ts

    ts = pd.to_datetime(text, errors='coerce')
    if pd.isna(ts):
        raise ValueError(f'Invalid date value: {value}')
    return ts


def should_drop_incomplete_initial_resampled_bar(
        raw_df: pd.DataFrame,
        resample_rule: str) -> bool:
    rule = (resample_rule or '').strip()
    if not rule or raw_df.empty:
        return False

    first_ts = pd.to_datetime(raw_df['Date'].iloc[0], errors='coerce')
    if pd.isna(first_ts):
        return False

    try:
        return first_ts != first_ts.floor(rule)
    except Exception:
        return False


def precompute_atr(quote: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    if atr_period <= 0:
        raise ValueError('atr_period must be positive.')

    quote = quote.copy()
    prev_close = quote['close'].shift(1)
    tr_df = pd.DataFrame({
        'hl': quote['high'] - quote['low'],
        'hc': (quote['high'] - prev_close).abs(),
        'lc': (quote['low'] - prev_close).abs(),
    })
    true_range = tr_df.max(axis=1, skipna=True).astype(float)

    atr = pd.Series(np.nan, index=quote.index, dtype=float)
    if len(true_range) >= atr_period:
        atr.iloc[atr_period - 1] = true_range.iloc[:atr_period].mean()
        for ii in range(atr_period, len(true_range)):
            atr.iloc[ii] = (
                (atr.iloc[ii - 1] * (atr_period - 1)) + true_range.iloc[ii]
            ) / atr_period

    quote['atr'] = atr
    quote['atr_ready'] = quote['atr'].notna().astype(int)
    quote['true_range'] = true_range
    return quote


# ============================================================
# Utility Functions
# ============================================================

def get_increase(df):
    if df.empty:
        print('received empty dataframe at get_increase function.')
        return np.nan
    need_cols = ['open', 'high', 'low', 'close']
    if any(c not in df.columns for c in need_cols):
        return np.nan
    if df[need_cols].isna().any().any():
        return np.nan
    if df.iloc[0]['open'] >= df.iloc[0]['close']:
        low = df.iloc[0]['low']
        high = df.iloc[0]['high']
    else:
        low = df.iloc[0]['low']
        high = df.iloc[0]['close']
    increase = 0
    for index, row in df.iterrows():
        if row['low'] <= low:
            high = row['close']
            low = row['low']
        elif row['high'] > high:
            high = row['high']
        increase = high - low
    return increase


def get_increase_with_base(df):
    """
    返回涨幅及其对应的真实基准 low。
    注意：基准 low 可能不是窗口第一根 bar 的 low。
    """
    if df.empty:
        print('received empty dataframe at get_increase function.')
        return np.nan, np.nan
    need_cols = ['open', 'high', 'low', 'close']
    if any(c not in df.columns for c in need_cols):
        return np.nan, np.nan
    if df[need_cols].isna().any().any():
        return np.nan, np.nan
    if df.iloc[0]['open'] >= df.iloc[0]['close']:
        low = df.iloc[0]['low']
        high = df.iloc[0]['high']
    else:
        low = df.iloc[0]['low']
        high = df.iloc[0]['close']
    increase = 0
    for index, row in df.iterrows():
        if row['low'] <= low:
            high = row['close']
            low = row['low']
        elif row['high'] > high:
            high = row['high']
        increase = high - low
    return increase, low


def get_analysis_increase(df):
    if df.empty:
        print('received empty dataframe at get_increase function.')
        return np.nan
    if len(df) == 1:
        return 0.0
    need_cols = ['open', 'high', 'low', 'close']
    if any(c not in df.columns for c in need_cols):
        return np.nan
    if df[need_cols].isna().any().any():
        return np.nan
    low = df.iloc[0]['close']
    high = df.iloc[1:].high.max()
    analysis_increase = high - low
    return analysis_increase


def get_withdrawal(df, close_withdrawal_threshold0,
                   index0, assumebarwithdrawal=True,
                   switch0=False):
    if df.empty:
        print('received empty dataframe at get_increase function.')
        return np.nan
    need_cols = ['open', 'high', 'low', 'close']
    if any(c not in df.columns for c in need_cols):
        return np.nan
    if df[need_cols].isna().any().any():
        return np.nan
    initialized = False
    with_high = 0
    with_low = 0
    withdrawal = 0
    for index, row in df.iterrows():
        if not initialized:
            with_high = row['close']
            with_low = row['close']
            withdrawal = with_high - with_low
            initialized = True
        else:
            if row['high'] > with_high:
                with_high = row['high']
                with_low = row['close']
            elif row['low'] < with_low:
                with_low = row['low']
            withdrawal = with_high - with_low
    return with_high, withdrawal


def get_max_wd(df, assumebarwithdrawal=True):
    if df.empty:
        print('received empty dataframe at get_increase function.')
        return np.nan
    need_cols = ['open', 'high', 'low', 'close']
    if any(c not in df.columns for c in need_cols):
        return np.nan
    if df[need_cols].isna().any().any():
        return np.nan
    initialized = False
    with_high = 0
    with_low = 0
    withdrawal = 0
    max_wd = 0
    for index, row in df.iterrows():
        if not initialized:
            with_high = row['high']
            with_low = row['close']
            initialized = True
        else:
            if row['high'] > with_high:
                with_high = row['high']
                with_low = row['close']
            elif row['low'] < with_low:
                with_low = row['low']
            withdrawal = (with_high - with_low) / with_high
        if withdrawal > max_wd:
            max_wd = withdrawal
    return max_wd


def get_outcome_withdrawal(sers):
    initialized = False
    with_high = 0
    with_low = 0
    withdrawal = 0
    for row in sers:
        if not initialized:
            with_high = row
            with_low = row
            withdrawal = with_high - with_low
            initialized = True
        else:
            if row > with_high:
                with_high = row
                with_low = row
            elif row < with_low:
                with_low = row
            withdrawal = with_high - with_low
    return with_high, withdrawal


def build_summary_metrics(
        perf_outcome: pd.DataFrame,
        transactions_df: pd.DataFrame,
        initial_capital: float) -> dict:
    capital_curve = perf_outcome['capital'].astype(float)
    final_capital = (
        float(capital_curve.iloc[-1])
        if len(capital_curve) else float(initial_capital)
    )
    total_return_pct = (
        (final_capital / float(initial_capital) - 1.0) * 100.0
        if initial_capital != 0 else np.nan
    )

    outcome_high, biggest_wd_abs = get_outcome_withdrawal(capital_curve)
    biggest_wd_pct = (
        (biggest_wd_abs / outcome_high) * 100.0
        if outcome_high not in (0, np.nan) and not pd.isna(outcome_high)
        else np.nan
    )

    closed_trades = transactions_df[transactions_df['Type'] != 'long'].copy()
    trade_num = int(len(closed_trades))
    trade_returns = pd.to_numeric(
        closed_trades['Percent'],
        errors='coerce',
    ) - 1.0
    trade_returns = trade_returns.dropna()

    win_trades = trade_returns[trade_returns > 0]
    loss_trades = trade_returns[trade_returns < 0]
    win_rate_pct = (
        float((trade_returns > 0).mean() * 100.0)
        if len(trade_returns) > 0 else np.nan
    )
    avg_trade_return_pct = (
        float(trade_returns.mean() * 100.0)
        if len(trade_returns) > 0 else np.nan
    )
    median_trade_return_pct = (
        float(trade_returns.median() * 100.0)
        if len(trade_returns) > 0 else np.nan
    )

    payoff_ratio = np.nan
    if len(win_trades) > 0 and len(loss_trades) > 0:
        avg_win = float(win_trades.mean())
        avg_loss = float(loss_trades.mean())
        if avg_loss != 0:
            payoff_ratio = avg_win / abs(avg_loss)

    profit_factor = np.nan
    gross_profit = float(win_trades.sum()) if len(win_trades) > 0 else 0.0
    gross_loss = float(abs(loss_trades.sum())) if len(loss_trades) > 0 else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss

    bar_returns = capital_curve.pct_change().dropna()
    sharpe_ratio = np.nan
    if len(bar_returns) > 1:
        std = float(bar_returns.std(ddof=1))
        if std > 0:
            sharpe_ratio = float(
                bar_returns.mean() / std * np.sqrt(len(bar_returns))
            )

    return {
        'final_capital': final_capital,
        'total_return_pct': total_return_pct,
        'outcome_high': float(outcome_high),
        'biggest_wd_abs': float(biggest_wd_abs),
        'biggest_wd_pct': biggest_wd_pct,
        'trade_num': trade_num,
        'win_rate_pct': win_rate_pct,
        'avg_trade_return_pct': avg_trade_return_pct,
        'median_trade_return_pct': median_trade_return_pct,
        'payoff_ratio': payoff_ratio,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
    }


def load_existing_outcome_stats(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path, index_col=0)
    return df[~df.index.duplicated(keep='last')]


def compute_intrabar_withdrawal_metrics(
        raw_df: pd.DataFrame,
        resampled_df: pd.DataFrame,
        bar_seconds: int,
        metric_kind: str = 'ratio') -> pd.DataFrame:
    columns = ['Date', 'intrabar_wd', 'intrabar_raw_count', 'intrabar_wd_max20']
    if raw_df.empty or resampled_df.empty or int(bar_seconds) <= 0:
        return pd.DataFrame(columns=columns)

    raw = raw_df[['Date', 'open', 'high', 'low', 'close']].copy()
    res = resampled_df[['Date']].copy()
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    res['Date'] = pd.to_datetime(res['Date'], errors='coerce')
    raw = raw.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    res = res.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    if raw.empty or res.empty:
        return pd.DataFrame(columns=columns)

    bar_ends = res['Date'].shift(-1)
    bar_ends.iloc[-1] = res.iloc[-1]['Date'] + pd.Timedelta(seconds=int(bar_seconds))

    raw_times = raw['Date'].to_numpy()
    res_times = res['Date'].to_numpy()
    end_times = bar_ends.to_numpy()
    left_index = raw_times.searchsorted(res_times, side='left')
    right_index = raw_times.searchsorted(end_times, side='left')

    metrics = []
    raw_counts = []
    for left, right in zip(left_index, right_index):
        left = int(left)
        right = int(right)
        raw_counts.append(max(right - left, 0))
        intrabar_slice = raw.iloc[left:right]
        if len(intrabar_slice) <= 1:
            metrics.append(0.0)
            continue

        with_high, withdrawal = get_withdrawal(
            intrabar_slice,
            0,
            0,
            switch0=True,
        )
        if pd.isna(with_high) or pd.isna(withdrawal):
            metrics.append(np.nan)
        elif metric_kind == 'absolute':
            metrics.append(float(withdrawal))
        else:
            metrics.append(float(withdrawal / with_high if with_high != 0 else 0.0))

    metric_df = pd.DataFrame({
        'Date': res['Date'],
        'intrabar_wd': metrics,
        'intrabar_raw_count': raw_counts,
    })
    metric_df['intrabar_wd_max20'] = (
        metric_df['intrabar_wd'].rolling(20, min_periods=1).max()
    )
    return metric_df


def prompt_manual_intrabar_precheck(
        raw_df: pd.DataFrame,
        resampled_df: pd.DataFrame,
        bar_seconds: int,
        open_threshold_values,
        open_cont_threshold_values,
        metric_kind: str = 'ratio') -> None:
    metric_df = compute_intrabar_withdrawal_metrics(
        raw_df,
        resampled_df,
        bar_seconds,
        metric_kind=metric_kind,
    )
    if metric_df.empty:
        return

    open_series = pd.Series(open_threshold_values, index=metric_df.index, dtype='float64')
    open_cont_series = pd.Series(
        open_cont_threshold_values,
        index=metric_df.index,
        dtype='float64',
    )
    compare_df = metric_df.copy()
    compare_df['open_threshold'] = open_series
    compare_df['open_cont_threshold'] = open_cont_series

    valid_mask = compare_df[
        ['intrabar_wd_max20', 'open_threshold', 'open_cont_threshold']
    ].notna().all(axis=1)
    risk_mask = valid_mask & (
        (compare_df['intrabar_wd_max20'] >= compare_df['open_threshold'])
        | (compare_df['intrabar_wd_max20'] >= compare_df['open_cont_threshold'])
    )
    risk_df = compare_df.loc[risk_mask].copy()
    if len(risk_df) == 0:
        print('[Precheck] no obvious intrabar reversal conflict found.')
        return

    valid_count = int(valid_mask.sum())
    risk_ratio = (len(risk_df) / valid_count) if valid_count > 0 else 0.0
    print('[Precheck] intrabar reversal risk detected in manual mode.')
    print(
        '[Precheck] risky bars: '
        + f'{len(risk_df)} / {valid_count} ({risk_ratio * 100:.2f}%)'
    )

    sample_df = risk_df.head(10)
    for idx, row in sample_df.iterrows():
        if metric_kind == 'absolute':
            print(
                '  idx=' + str(idx)
                + ' date=' + str(row['Date'])
                + f' intrabar_max20={row["intrabar_wd_max20"]:.6f}'
                + f' open={row["open_threshold"]:.6f}'
                + f' open_cont={row["open_cont_threshold"]:.6f}'
            )
        else:
            print(
                '  idx=' + str(idx)
                + ' date=' + str(row['Date'])
                + f' intrabar_max20={row["intrabar_wd_max20"] * 100:.4f}%'
                + f' open={row["open_threshold"] * 100:.4f}%'
                + f' open_cont={row["open_cont_threshold"] * 100:.4f}%'
            )

    print(
        '[Precheck] suggestion: raise open_threshold / open_continous_threshold, '
        + 'or use a finer resample_rule.'
    )
    answer = input('Continue backtest? [y/N]: ').strip().lower()
    if answer not in ('y', 'yes'):
        raise SystemExit('Stopped by user after intrabar precheck.')


def build_long_param_tag(
        open_bar: int,
        open_threshold: float,
        open_continous_threshold: float,
        open_withdrawal_threshold: float,
        close_bar: int,
        close_threshold: float,
        close_wd_max_inc_ratio: float,
        atr_period: int) -> str:
    return (
        f'atr{atr_period}'
        + ' oa' + str(round(open_threshold, 4))
        + ' oca' + str(round(open_continous_threshold, 4))
        + ' owa' + str(round(open_withdrawal_threshold, 4))
        + ' ca' + str(round(close_threshold, 4))
        + ' cwdm' + str(round(close_wd_max_inc_ratio, 4))
        + ' ob' + str(round(open_bar, 4))
        + ' cb' + str(round(close_bar, 4))
    )


def build_progress_marks(total_count: int) -> dict[int, int]:
    if total_count <= 0:
        return {}
    return {
        max(1, int(np.ceil(total_count * pct / 100.0))): pct
        for pct in (20, 40, 60, 80, 100)
    }


def print_search_progress(
        completed_count: int,
        total_count: int,
        progress_marks: dict[int, int],
        printed_marks: set[int]) -> None:
    if total_count <= 0:
        return
    for mark in sorted(progress_marks):
        pct = progress_marks[mark]
        if completed_count >= mark and pct not in printed_marks:
            print(f'[Grid] progress: {pct}% ({completed_count}/{total_count})')
            printed_marks.add(pct)


def build_planned_param_tags_long_atr(
        open_bar_values: list[int],
        open_atr_multiplier_start: float,
        open_atr_iterations: int,
        atr_step_value: float,
        open_cont_iterations: int,
        step3_value: float,
        close_wd_max_inc_ratio_value: float,
        atr_period_value: int) -> set[str]:
    planned_tags = set()
    for open_bar_value in open_bar_values:
        close_bar_value = int(open_bar_value)
        for atr_iter in range(int(open_atr_iterations)):
            open_threshold_value = round(
                open_atr_multiplier_start + (atr_iter * atr_step_value),
                10,
            )
            open_withdrawal_threshold_value = open_threshold_value
            close_threshold_value = open_threshold_value
            if min(
                    open_threshold_value,
                    open_withdrawal_threshold_value,
                    close_threshold_value) < 0:
                continue
            for open_cont_iter in range(int(open_cont_iterations)):
                open_continous_threshold_value = round(
                    open_threshold_value + (open_cont_iter * step3_value),
                    10,
                )
                if open_continous_threshold_value < open_threshold_value:
                    continue
                planned_tags.add(build_long_param_tag(
                    open_bar_value,
                    open_threshold_value,
                    open_continous_threshold_value,
                    open_withdrawal_threshold_value,
                    close_bar_value,
                    close_threshold_value,
                    close_wd_max_inc_ratio_value,
                    atr_period=atr_period_value,
                ))
    return planned_tags


def format_hover_value(value, digits: int = 6) -> str:
    if pd.isna(value):
        return 'nan'
    try:
        return str(round(float(value), digits))
    except Exception:
        return str(value)


def draw_monochrome_candles(
        ax,
        ohlc_df: pd.DataFrame,
        width: float = 0.68) -> None:
    if ohlc_df.empty:
        return

    typical_range = (ohlc_df['high'] - ohlc_df['low']).median()
    if pd.isna(typical_range) or typical_range <= 0:
        min_body = 0.0001
    else:
        min_body = max(float(typical_range) * 0.06, 0.0001)

    for idx, row in enumerate(ohlc_df.itertuples(index=False)):
        open_price = float(row.open)
        high_price = float(row.high)
        low_price = float(row.low)
        close_price = float(row.close)
        ax.vlines(
            idx,
            low_price,
            high_price,
            color='black',
            linewidth=0.7,
            zorder=1,
        )
        body_low = min(open_price, close_price)
        body_height = abs(close_price - open_price)
        if body_height < min_body:
            mid = (open_price + close_price) / 2
            body_low = mid - (min_body / 2)
            body_height = min_body
        facecolor = 'white' if close_price >= open_price else 'black'
        ax.add_patch(Rectangle(
            (idx - width / 2, body_low),
            width,
            body_height,
            facecolor=facecolor,
            edgecolor='black',
            linewidth=0.8,
            zorder=2,
        ))


def build_int_search_values(start: int, end: int, step: int) -> list[int]:
    if step == 0:
        raise ValueError('step cannot be 0.')
    if step > 0 and start > end:
        raise ValueError('positive step requires start <= end.')
    if step < 0 and start < end:
        raise ValueError('negative step requires start >= end.')
    stop = end + (1 if step > 0 else -1)
    return list(range(start, stop, step))


def export_interactive_html_long(
        file_name: str,
        save_name: str,
        title: str,
        underlying1: pd.DataFrame,
        detail_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        factor: float):
    if go is None:
        print('[HTML] plotly is not installed, skip html export.')
        return

    def _safe_val(pref_data, key, digits=None):
        if isinstance(pref_data, pd.Series) and key in pref_data.index:
            val = pref_data[key]
        else:
            return 'nan'
        if pd.isna(val):
            return 'nan'
        if digits is not None:
            try:
                return str(round(float(val), digits))
            except Exception:
                return str(val)
        return str(val)

    def _date_text(raw):
        dt = str(raw)[:-3]
        if len(dt) > 5:
            return dt[:-5] + ' ' + dt[-5:]
        return dt

    fig_html = go.Figure()
    x_index = underlying1.index.to_numpy()
    x_min = int(x_index[0]) if len(x_index) > 0 else 0
    x_max = int(x_index[-1]) if len(x_index) > 0 else 1
    x_span = max(1, x_max - x_min + 1)
    x_left_pad = max(1, int(round(x_span * 0.006)))
    x_right_pad = max(1, int(round(x_span * 0.010)))

    x_spike_cfg = {'showspikes': False}
    y_spike_cfg = {'showspikes': False}
    if HTML_CROSSHAIR_ENABLED:
        x_spike_cfg = {
            'showspikes': True,
            'spikemode': 'across',
            'spikesnap': 'cursor',
            'spikecolor': HTML_CROSSHAIR_COLOR,
            'spikethickness': 1,
            'spikedash': 'solid'
        }
        y_spike_cfg = {
            'showspikes': True,
            'spikemode': 'across',
            'spikesnap': 'cursor',
            'spikecolor': HTML_CROSSHAIR_COLOR,
            'spikethickness': 1,
            'spikedash': 'solid'
        }

    fig_html.add_trace(go.Scatter(
        x=detail_df.index,
        y=detail_df.capital,
        mode='lines',
        line=dict(width=1.2, color=ACCENT_BLUE),
        name='capital',
        hovertemplate='index: %{x}<br>capital: %{y:.4f}<extra></extra>'
    ))

    fig_html.add_trace(go.Candlestick(
        x=x_index,
        open=underlying1['open'] / factor * 100,
        high=underlying1['high'] / factor * 100,
        low=underlying1['low'] / factor * 100,
        close=underlying1['close'] / factor * 100,
        name='price',
        increasing=dict(
            line=dict(color='salmon', width=0.8),
            fillcolor='rgba(250, 128, 114, 0.28)'
        ),
        decreasing=dict(
            line=dict(color='#2ca02c', width=0.8),
            fillcolor='rgba(44, 160, 44, 0.28)'
        )
    ))
    add_long_gap_shapes(fig_html, underlying1)

    long_record = transactions_df.copy()
    long_record['target'] = long_record['Price'] / factor * 100
    long_record = long_record[long_record.Type == 'long']
    if len(long_record) != 0:
        long_texts = []
        for idx, row in long_record.iterrows():
            pref_data = detail_df.loc[idx] if idx in detail_df.index else pd.Series(dtype='object')
            long_texts.append(
                _date_text(row['Date']) + '<br>'
                + 'high: ' + _safe_val(pref_data, 'high') + '<br>'
                + 'atr: ' + _safe_val(pref_data, 'atr', 6) + '<br>'
                + 'frozen_atr: ' + _safe_val(pref_data, 'frozen_open_atr', 6) + '<br>'
                + 'oa_mult: ' + _safe_val(pref_data, 'open_atr_multiplier_runtime', 4) + '<br>'
                + 'oca_mult: ' + _safe_val(pref_data, 'open_cont_atr_multiplier_runtime', 4) + '<br>'
                + 'owa_mult: ' + _safe_val(pref_data, 'open_wd_atr_multiplier_runtime', 4) + '<br>'
                + 'trigger: ' + _safe_val(pref_data, 'open_trigger_price', 6) + '<br>'
                + 'inc_abs: ' + _safe_val(pref_data, 'increase', 6) + '<br>'
                + 'wd_abs: ' + _safe_val(pref_data, 'withdrawal', 6) + '<br>'
                + 'total_inc: ' + _safe_val(pref_data, 't_inc_per', 2) + '%' + '<br>'
                + 'execution: ' + _safe_val(pref_data, 'execution') + '<br>'
                + 'low_date: ' + _safe_val(pref_data, 'low_date') + '<br>'
                + 'low_price: ' + _safe_val(pref_data, 'low_price') + '<br>'
                + 'new_opening_count: ' + _safe_val(pref_data, 'new_opening_count') + '<br>'
                + 'index: ' + str(idx)
            )
        fig_html.add_trace(go.Scatter(
            x=long_record.index,
            y=long_record['target'],
            mode='markers',
            marker=dict(color='red', size=4),
            name='long',
            text=long_texts,
            hovertemplate='%{text}<extra></extra>'
        ))

    sell_record = transactions_df.copy()
    sell_record['target'] = sell_record['Price'] / factor * 100
    sell_record = sell_record[sell_record.Type == 'sell']
    sell_1_count = 0
    sell_2_count = 0
    if len(sell_record) != 0:
        close_type_1_df = sell_record[sell_record['Close_type'] == 1]
        sell_1_count = int(len(close_type_1_df))
        if len(close_type_1_df) != 0:
            sell_1_texts = []
            for idx, row in close_type_1_df.iterrows():
                pref_data = detail_df.loc[idx] if idx in detail_df.index else pd.Series(dtype='object')
                sell_1_texts.append(
                    _date_text(row['Date']) + '<br>'
                    + 'atr: ' + _safe_val(pref_data, 'atr', 6) + '<br>'
                    + 'ca_mult: ' + _safe_val(pref_data, 'close_speed_atr_multiplier_runtime', 4) + '<br>'
                    + 'close_th: ' + _safe_val(pref_data, 'active_close_threshold', 6) + '<br>'
                    + 'close_wd_th: ' + _safe_val(pref_data, 'active_close_wd_threshold', 6) + '<br>'
                    + 'cwd_ratio: ' + _safe_val(pref_data, 'close_wd_ratio_runtime', 4) + '<br>'
                    + 'low: ' + _safe_val(pref_data, 'low') + '<br>'
                    + 'hld_wd_per: ' + _safe_val(pref_data, 'hld_wd_per', 2) + '%' + '<br>'
                    + 'holding_inc: ' + _safe_val(pref_data, 'holding_inc', 2) + '<br>'
                    + 'max_inc: ' + _safe_val(pref_data, 'max_inc', 2) + '%' + '<br>'
                    + 'max_wd: ' + _safe_val(pref_data, 'max_wd', 2) + '%' + '<br>'
                    + 'execution2: ' + _safe_val(pref_data, 'execution') + '<br>'
                    + 'period: ' + _safe_val(pref_data, 'period') + '<br>'
                    + 'low_date: ' + _safe_val(pref_data, 'low_date') + '<br>'
                    + 'high_date: ' + _safe_val(pref_data, 'high_date') + '<br>'
                    + 'high_price: ' + _safe_val(pref_data, 'high_price') + '<br>'
                    + 'index: ' + str(idx)
                )
            fig_html.add_trace(go.Scatter(
                x=close_type_1_df.index,
                y=close_type_1_df['target'],
                mode='markers',
                marker=dict(color=SELL_WD_COLOR, size=4),
                name='sell_1',
                text=sell_1_texts,
                hovertemplate='%{text}<extra></extra>'
            ))

        close_type_2_df = sell_record[sell_record['Close_type'] == 2]
        sell_2_count = int(len(close_type_2_df))
        if len(close_type_2_df) != 0:
            sell_2_texts = []
            for idx, row in close_type_2_df.iterrows():
                pref_data = detail_df.loc[idx] if idx in detail_df.index else pd.Series(dtype='object')
                sell_2_texts.append(
                    _date_text(row['Date']) + '<br>'
                    + 'atr: ' + _safe_val(pref_data, 'atr', 6) + '<br>'
                    + 'ca_mult: ' + _safe_val(pref_data, 'close_speed_atr_multiplier_runtime', 4) + '<br>'
                    + 'close_th: ' + _safe_val(pref_data, 'active_close_threshold', 6) + '<br>'
                    + 'close_wd_th: ' + _safe_val(pref_data, 'active_close_wd_threshold', 6) + '<br>'
                    + 'cwd_ratio: ' + _safe_val(pref_data, 'close_wd_ratio_runtime', 4) + '<br>'
                    + 'low: ' + _safe_val(pref_data, 'low') + '<br>'
                    + 'hld_wd_per: ' + _safe_val(pref_data, 'hld_wd_per', 2) + '%' + '<br>'
                    + 'max_inc: ' + _safe_val(pref_data, 'max_inc', 2) + '%' + '<br>'
                    + 'max_wd: ' + _safe_val(pref_data, 'max_wd', 2) + '%' + '<br>'
                    + 'execution2: ' + _safe_val(pref_data, 'execution') + '<br>'
                    + 'period: ' + _safe_val(pref_data, 'period') + '<br>'
                    + 'low_date: ' + _safe_val(pref_data, 'low_date') + '<br>'
                    + 'high_date: ' + _safe_val(pref_data, 'high_date') + '<br>'
                    + 'high_price: ' + _safe_val(pref_data, 'high_price') + '<br>'
                    + 'index: ' + str(idx)
                )
            fig_html.add_trace(go.Scatter(
                x=close_type_2_df.index,
                y=close_type_2_df['target'],
                mode='markers',
                marker=dict(color=SELL_SPEED_COLOR, size=4),
                name='sell_2',
                text=sell_2_texts,
                hovertemplate='%{text}<extra></extra>'
            ))

    trade_seq = transactions_df[
        transactions_df['Type'].isin(['long', 'sell'])].copy()
    trade_seq = trade_seq.sort_index()
    trade_seq['target'] = trade_seq['Price'] / factor * 100
    line_x = []
    line_y = []
    buy_idx = None
    buy_y = None
    for idx, row in trade_seq.iterrows():
        if row['Type'] == 'long':
            buy_idx = idx
            buy_y = row['target']
        elif row['Type'] == 'sell' and buy_idx is not None:
            line_x.extend([buy_idx, idx, None])
            line_y.extend([buy_y, row['target'], None])
            buy_idx = None
            buy_y = None
    if len(line_x) > 0:
        fig_html.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            line=dict(color=ACCENT_BLUE, width=2),
            name='trade_link',
            hoverinfo='skip'
        ))

    trade_count_annotation = []
    if HTML_SHOW_TRADE_COUNT_BADGE:
        total_trade_count = sell_1_count + sell_2_count
        trade_count_annotation = [dict(
            x=0.995, y=0.995,
            xref='paper', yref='paper',
            xanchor='right', yanchor='top',
            align='right',
            showarrow=False,
            text=(
                f"trades: {total_trade_count}"
                + "<br>"
                + f"sell_1: {sell_1_count}"
                + "<br>"
                + f"sell_2: {sell_2_count}"
            ),
            font=dict(size=11, color='black')
        )]

    fig_html.update_layout(
        title=None,
        template='plotly_white',
        autosize=True,
        hovermode='closest',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
        xaxis=dict(
            title=None,
            tickfont=dict(size=10),
            showgrid=False,
            rangeslider=dict(visible=False),
            range=[x_min - x_left_pad, x_max + x_right_pad],
            autorange=False,
            **x_spike_cfg
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(size=10),
            showgrid=False,
            **y_spike_cfg
        ),
        margin=dict(l=42, r=25, t=38, b=45, pad=0),
        annotations=trade_count_annotation,
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.35)',
            bordercolor='rgba(0, 0, 0, 0.45)',
            font=dict(color='black')
        )
    )

    html_dir = './result/%s long_momentum_ATR outcome/html' % file_name
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(html_dir, save_name + ' Long interactive.html')
    html_text = fig_html.to_html(
        include_plotlyjs=True,
        full_html=True,
        default_width='100vw',
        default_height='100vh',
        config={
            'responsive': True,
            'displayModeBar': False,
            'displaylogo': False
        }
    )
    html_text = html_text.replace(
        '<head>',
        '<head><style>'
        'html,body{width:100%;height:100%;margin:0;padding:0;overflow:hidden;}'
        '.plotly-graph-div{width:100vw !important;height:100vh !important;}'
        '.hoverlayer .hovertext .bg,'
        '.hoverlayer .hovertext rect,'
        '.hoverlayer .hovertext path{'
        'fill:rgba(255,255,255,0.35) !important;'
        'fill-opacity:0.35 !important;'
        'stroke:rgba(0,0,0,0.45) !important;'
        'stroke-opacity:0.45 !important;}'
        '.hoverlayer .hovertext{opacity:1 !important;}'
        '.hoverlayer .hovertext text{fill:#000 !important;}'
        '</style>',
        1
    )
    html_text = html_text.replace('<body>', '<body style="margin:0;overflow:hidden;">', 1)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print('\n')
    print(f'[HTML] saved interactive chart: {html_path}')


# ============================================================
# Momentum Strategy
# ============================================================

class MomentumStrategy(BaseStrategy):
    """
    动量策略实现。
    从原 generate_signals 中的策略逻辑提取而来。
    """

    def __init__(self, params: dict):
        super().__init__(params)
        # 策略内部状态
        self.var0 = 0
        self.new_opening = False
        self.new_opening_count = 0
        self.last_index = 0
        self.low_index = 0
        self.start_index = 0
        self.recent_low_index = 0
        self.first_cond1_price = 0
        self.analysis_increase = 0
        self.holding_start_index = 0
        self.increase_start_index = 0
        self.open_base_price = np.nan
        self.frozen_open_atr = np.nan
        self.frozen_open_threshold = np.nan
        self.frozen_open_cont_threshold = np.nan
        self.frozen_open_wd_threshold = np.nan
        self.holding_increase_percent = np.nan
        self.HIGH_MATCH_EPS = 1e-10

    def get_extra_columns(self) -> list:
        return [
            'withdrawal', 'wd_per', 'wd_signal',
            'increase', 'inc_per', 'inc_signal',
            'ana_inc', 'a_inc_per',
            'total_inc', 't_inc_per', 'total_inc_signal',
            'max_inc', 'max_wd',
            'holding_wd', 'hld_wd_per', 'holding_wd_signal',
            'holding_inc', 'speed_close_signal',
            'var0', 'period',
            'low_index', 'high_index',
            'low_date', 'low_price',
            'high_date', 'high_price',
            'last_index', 'new_opening_count',
            'basis_ready', 'pos_basis', 'neg_basis',
            'active_open_threshold',
            'active_open_cont_threshold',
            'active_open_wd_threshold',
            'active_close_threshold',
            'active_close_wd_threshold',
            'frozen_open_atr',
            'frozen_open_threshold',
            'frozen_open_cont_threshold',
            'frozen_open_wd_threshold',
        ]

    def get_default_columns(self) -> dict:
        return {
            'wd_signal': 0.0,
            'inc_signal': 0.0,
            'total_inc_signal': 0.0,
            'holding_wd_signal': 0.0,
            'speed_close_signal': 0.0,
        }

    def on_bar_record(self, ctx: BarContext):
        """每根K线记录策略状态到signal"""
        ctx.signal.at[ctx.index, 'last_index'] = self.last_index
        ctx.signal.at[ctx.index, 'new_opening_count'] = self.new_opening_count

    def _clear_frozen_open_thresholds(self):
        self.frozen_open_atr = np.nan
        self.frozen_open_threshold = np.nan
        self.frozen_open_cont_threshold = np.nan
        self.frozen_open_wd_threshold = np.nan

    def _has_frozen_open_thresholds(self) -> bool:
        return not any(pd.isna(v) for v in (
            self.frozen_open_atr,
            self.frozen_open_threshold,
            self.frozen_open_cont_threshold,
            self.frozen_open_wd_threshold,
        ))

    def _build_frozen_open_thresholds(
            self,
            quote: pd.DataFrame,
            low_index: int) -> dict | None:
        if low_index <= 0 or 'atr' not in quote.columns:
            return None

        frozen_open_atr = quote.iloc[low_index - 1]['atr']
        if pd.isna(frozen_open_atr):
            return None

        return {
            'frozen_open_atr': float(frozen_open_atr),
            'open_threshold': float(frozen_open_atr) * self.params['open_threshold'],
            'open_continous_threshold': (
                float(frozen_open_atr) * self.params['open_continous_threshold']
            ),
            'open_withdrawal_threshold': (
                float(frozen_open_atr) * self.params['open_withdrawal_threshold']
            ),
        }

    def _apply_frozen_open_thresholds(self, frozen_thresholds: dict):
        self.frozen_open_atr = frozen_thresholds['frozen_open_atr']
        self.frozen_open_threshold = frozen_thresholds['open_threshold']
        self.frozen_open_cont_threshold = (
            frozen_thresholds['open_continous_threshold']
        )
        self.frozen_open_wd_threshold = (
            frozen_thresholds['open_withdrawal_threshold']
        )

    def _write_open_threshold_state(
            self,
            signal: pd.DataFrame,
            index,
            open_threshold,
            open_continous_threshold,
            open_withdrawal_threshold,
            frozen_open_atr,
            frozen_open_threshold,
            frozen_open_cont_threshold,
            frozen_open_wd_threshold):
        signal.at[index, 'active_open_threshold'] = open_threshold
        signal.at[index, 'active_open_cont_threshold'] = open_continous_threshold
        signal.at[index, 'active_open_wd_threshold'] = open_withdrawal_threshold
        signal.at[index, 'frozen_open_atr'] = frozen_open_atr
        signal.at[index, 'frozen_open_threshold'] = frozen_open_threshold
        signal.at[index, 'frozen_open_cont_threshold'] = frozen_open_cont_threshold
        signal.at[index, 'frozen_open_wd_threshold'] = frozen_open_wd_threshold

    def _resolve_thresholds(self, ctx: BarContext) -> dict:
        quote = ctx.quote
        signal = ctx.signal
        index = ctx.index
        p = self.params

        atr_basis = (
            quote.at[index, 'atr']
            if 'atr' in quote.columns else np.nan
        )
        pos_basis = atr_basis
        neg_basis = atr_basis
        basis_ready = int(not pd.isna(atr_basis))

        open_threshold = atr_basis * p['open_threshold']
        open_continous_threshold = atr_basis * p['open_continous_threshold']
        open_withdrawal_threshold = atr_basis * p['open_withdrawal_threshold']
        close_threshold = atr_basis * p['close_threshold']
        close_withdrawal_threshold = np.nan

        signal.at[index, 'basis_ready'] = basis_ready
        signal.at[index, 'pos_basis'] = pos_basis
        signal.at[index, 'neg_basis'] = neg_basis
        signal.at[index, 'active_close_threshold'] = close_threshold
        signal.at[index, 'active_close_wd_threshold'] = close_withdrawal_threshold

        return {
            'open_threshold': open_threshold,
            'open_continous_threshold': open_continous_threshold,
            'open_withdrawal_threshold': open_withdrawal_threshold,
            'close_threshold': close_threshold,
            'close_withdrawal_threshold': close_withdrawal_threshold,
        }

    def on_bar_idle(self, ctx: BarContext) -> OpenResult | None:
        quote = ctx.quote
        signal = ctx.signal
        index = ctx.index
        ii = ctx.integer_index
        p = self.params

        open_bar = p['open_bar']
        close_bar = p['close_bar']
        thresholds = self._resolve_thresholds(ctx)
        current_open_threshold = thresholds['open_threshold']
        current_open_continous_threshold = thresholds['open_continous_threshold']
        current_open_withdrawal_threshold = thresholds['open_withdrawal_threshold']
        close_threshold = thresholds['close_threshold']
        close_withdrawal_threshold = thresholds['close_withdrawal_threshold']
        open_threshold = current_open_threshold
        open_continous_threshold = current_open_continous_threshold
        open_withdrawal_threshold = current_open_withdrawal_threshold
        row_frozen_open_atr = np.nan
        row_frozen_open_threshold = np.nan
        row_frozen_open_cont_threshold = np.nan
        row_frozen_open_wd_threshold = np.nan

        if self._has_frozen_open_thresholds():
            open_threshold = self.frozen_open_threshold
            open_continous_threshold = self.frozen_open_cont_threshold
            open_withdrawal_threshold = self.frozen_open_wd_threshold
            row_frozen_open_atr = self.frozen_open_atr
            row_frozen_open_threshold = self.frozen_open_threshold
            row_frozen_open_cont_threshold = self.frozen_open_cont_threshold
            row_frozen_open_wd_threshold = self.frozen_open_wd_threshold

        # 1. last_index 赋值
        if self.new_opening:
            self.last_index = ii - 1
            self.new_opening = False

        # 2. 窗口移动
        if self.new_opening_count >= open_bar:
            self.last_index = ii - open_bar + 1
        self.new_opening_count += 1

        analysis_slice = quote.iloc[self.last_index:ii + 1]

        # --- 阶段 0: 检查速度 + 回撤 ---
        if self.var0 == 0:
            increase, inc_base = get_increase_with_base(analysis_slice)
            inc_percent = increase / inc_base if inc_base != 0 else 0
            with_high, withdrawal = get_withdrawal(
                analysis_slice, close_withdrawal_threshold, ii)
            wd_percent = withdrawal / with_high if with_high != 0 else 0

            signal.at[index, 'withdrawal'] = withdrawal
            signal.at[index, 'wd_per'] = round(wd_percent * 100, 4)
            signal.at[index, 'increase'] = increase
            signal.at[index, 'inc_per'] = round(inc_percent * 100, 4)
            candidate_low_index = None
            for i in reversed(range(self.last_index, ii + 1)):
                low_index_slice = quote.iloc[i:ii + 1]
                increase2 = get_increase(low_index_slice)
                if np.isclose(increase2, increase,
                              rtol=0.0, atol=self.HIGH_MATCH_EPS):
                    candidate_low_index = i
                    break

            candidate_frozen_thresholds = None
            if candidate_low_index is not None:
                candidate_frozen_thresholds = self._build_frozen_open_thresholds(
                    quote, candidate_low_index
                )

            if candidate_frozen_thresholds is not None:
                open_threshold = candidate_frozen_thresholds['open_threshold']
                open_continous_threshold = (
                    candidate_frozen_thresholds['open_continous_threshold']
                )
                open_withdrawal_threshold = (
                    candidate_frozen_thresholds['open_withdrawal_threshold']
                )
                row_frozen_open_atr = candidate_frozen_thresholds['frozen_open_atr']
                row_frozen_open_threshold = candidate_frozen_thresholds['open_threshold']
                row_frozen_open_cont_threshold = (
                    candidate_frozen_thresholds['open_continous_threshold']
                )
                row_frozen_open_wd_threshold = (
                    candidate_frozen_thresholds['open_withdrawal_threshold']
                )

            cond1 = (
                candidate_frozen_thresholds is not None
                and increase >= open_threshold
            )
            signal.at[index, 'inc_signal'] = 1 if cond1 else 0

            cond2 = (
                candidate_frozen_thresholds is not None
                and withdrawal < open_withdrawal_threshold
            )
            signal.at[index, 'wd_signal'] = 1 if cond2 else 0

            if (
                candidate_low_index is not None
                and candidate_frozen_thresholds is None
            ):
                self._clear_frozen_open_thresholds()
                self.new_opening = True
                self.new_opening_count = 1
            elif signal.at[index, 'wd_signal']:
                if signal.at[index, 'inc_signal']:
                    self.low_index = candidate_low_index
                    self._apply_frozen_open_thresholds(
                        candidate_frozen_thresholds
                    )
                    row_frozen_open_atr = self.frozen_open_atr
                    row_frozen_open_threshold = self.frozen_open_threshold
                    row_frozen_open_cont_threshold = self.frozen_open_cont_threshold
                    row_frozen_open_wd_threshold = self.frozen_open_wd_threshold
                    signal.at[index, 'low_index'] = self.low_index
                    signal.at[index, 'low_date'] = str(
                        signal.at[self.low_index, 'date'])
                    self.last_index = self.low_index
                    self.start_index = self.last_index
                    self.var0 = 1
            else:
                if (
                    candidate_frozen_thresholds is not None
                    and increase > open_continous_threshold
                ):
                    print(str(index) + '满足开仓和满足回撤reset同时发生')
                self._clear_frozen_open_thresholds()
                self.new_opening = True
                self.new_opening_count = 1

            self._write_open_threshold_state(
                signal,
                index,
                open_threshold,
                open_continous_threshold,
                open_withdrawal_threshold,
                row_frozen_open_atr,
                row_frozen_open_threshold,
                row_frozen_open_cont_threshold,
                row_frozen_open_wd_threshold,
            )

        # --- 阶段 1: 赋值 new_opening_count ---
        if self.var0 == 1:
            if not self._has_frozen_open_thresholds():
                self.var0 = 0
                self.new_opening = True
                self.new_opening_count = 1
                self._write_open_threshold_state(
                    signal,
                    index,
                    current_open_threshold,
                    current_open_continous_threshold,
                    current_open_withdrawal_threshold,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
                return None
            self.new_opening_count = ii - self.low_index + 1
            signal.at[index, 'low_index'] = self.low_index
            signal.at[index, 'low_date'] = str(
                signal.at[self.low_index, 'date']).removesuffix('.0')
            signal.at[index, 'period'] = self.new_opening_count
            self.var0 = 2
            open_threshold = self.frozen_open_threshold
            open_continous_threshold = self.frozen_open_cont_threshold
            open_withdrawal_threshold = self.frozen_open_wd_threshold
            row_frozen_open_atr = self.frozen_open_atr
            row_frozen_open_threshold = self.frozen_open_threshold
            row_frozen_open_cont_threshold = self.frozen_open_cont_threshold
            row_frozen_open_wd_threshold = self.frozen_open_wd_threshold

        # --- 阶段 2: 判断持续涨幅 ---
        if self.var0 == 2:
            if not self._has_frozen_open_thresholds():
                self.var0 = 0
                self.new_opening = True
                self.new_opening_count = 1
                self._write_open_threshold_state(
                    signal,
                    index,
                    current_open_threshold,
                    current_open_continous_threshold,
                    current_open_withdrawal_threshold,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
                return None
            open_threshold = self.frozen_open_threshold
            open_continous_threshold = self.frozen_open_cont_threshold
            open_withdrawal_threshold = self.frozen_open_wd_threshold
            row_frozen_open_atr = self.frozen_open_atr
            row_frozen_open_threshold = self.frozen_open_threshold
            row_frozen_open_cont_threshold = self.frozen_open_cont_threshold
            row_frozen_open_wd_threshold = self.frozen_open_wd_threshold
            cond3_analysis_slice = quote.iloc[self.low_index:ii + 1]
            with_high, withdrawal = get_withdrawal(
                cond3_analysis_slice, close_withdrawal_threshold, ii)
            signal.at[index, 'withdrawal'] = withdrawal
            withdrawal_percent = withdrawal / with_high if with_high != 0 else 0
            total_increase, inc_base = get_increase_with_base(cond3_analysis_slice)

            cond3 = withdrawal < open_withdrawal_threshold
            signal.at[index, 'wd_signal'] = 1 if cond3 else 0

            if signal.at[index, 'wd_signal']:
                if self.new_opening_count >= open_bar:
                    ana_inc_slice_1 = quote.iloc[self.low_index:ii + 1]
                    ana_inc_slice_2 = quote.iloc[
                        self.low_index:ii + 1 - open_bar]
                    analysis_increase = (ana_inc_slice_1.high.max()
                                         - ana_inc_slice_2.high.max())
                    ana_inc_base = ana_inc_slice_1['low'].iloc[0]
                    analysis_increase_percent = analysis_increase / ana_inc_base if ana_inc_base != 0 else 0
                    signal.at[index, 'ana_inc'] = analysis_increase
                    signal.at[index, 'a_inc_per'] = round(
                        analysis_increase_percent * 100, 4)
                    if analysis_increase < close_threshold:
                        self.var0 = 4

                total_increase_percent = (
                    total_increase / inc_base if inc_base != 0 else 0
                )
                signal.at[index, 'total_inc'] = total_increase
                signal.at[index, 't_inc_per'] = round(
                    total_increase_percent * 100, 4)
                self.first_cond1_price = inc_base

                if total_increase >= open_continous_threshold:
                    signal.at[index, 'total_inc_signal'] = 1
            else:
                self.var0 = 3

            # var0=3: 回撤 reset
            if self.var0 == 3:
                self._do_idle_stats(quote, signal, index, ii, 'open withdraw')
                self.new_opening = True
                self.var0 = 0
                self.new_opening_count = 0
                self.first_cond1_price = 0
                self.analysis_increase = 0
                self._clear_frozen_open_thresholds()
                row_frozen_open_atr = np.nan
                row_frozen_open_threshold = np.nan
                row_frozen_open_cont_threshold = np.nan
                row_frozen_open_wd_threshold = np.nan

            # var0=4: 涨速不够 reset
            if self.var0 == 4:
                self._do_idle_stats(quote, signal, index, ii, 'open speed')
                # reset with recalculated low_index
                increase1_slice = quote.iloc[self.last_index:ii + 1]
                increase1 = get_increase(increase1_slice)
                for i in range(self.last_index, ii + 1):
                    low_index_slice = quote.iloc[i:ii + 1]
                    increase2 = get_increase(low_index_slice)
                    if np.isclose(increase2, increase1,
                                  rtol=0.0, atol=self.HIGH_MATCH_EPS):
                        self.recent_low_index = i
                self.last_index = self.recent_low_index
                self.var0 = 0
                self.new_opening_count = ii - self.recent_low_index + 1
                self.first_cond1_price = 0
                self.analysis_increase = 0
                self._clear_frozen_open_thresholds()
                row_frozen_open_atr = np.nan
                row_frozen_open_threshold = np.nan
                row_frozen_open_cont_threshold = np.nan
                row_frozen_open_wd_threshold = np.nan

            self._write_open_threshold_state(
                signal,
                index,
                open_threshold,
                open_continous_threshold,
                open_withdrawal_threshold,
                row_frozen_open_atr,
                row_frozen_open_threshold,
                row_frozen_open_cont_threshold,
                row_frozen_open_wd_threshold,
            )

            # 开仓信号
            if signal.at[index, 'total_inc_signal'] == 1:
                self._write_open_threshold_state(
                    signal,
                    index,
                    open_threshold,
                    open_continous_threshold,
                    open_withdrawal_threshold,
                    row_frozen_open_atr,
                    row_frozen_open_threshold,
                    row_frozen_open_cont_threshold,
                    row_frozen_open_wd_threshold,
                )
                return OpenResult(
                    execution_price=round(
                        self.first_cond1_price + open_continous_threshold,
                        self.params['round_precision']),
                    low_index=self.low_index,
                    low_price=self.first_cond1_price,
                    start_index=self.start_index,
                )

        if self.var0 not in (0, 2):
            self._write_open_threshold_state(
                signal,
                index,
                open_threshold,
                open_continous_threshold,
                open_withdrawal_threshold,
                row_frozen_open_atr,
                row_frozen_open_threshold,
                row_frozen_open_cont_threshold,
                row_frozen_open_wd_threshold,
            )
        return None

    def on_position_opened(self, ctx: BarContext, result):
        """开仓后记录开仓信息并重置策略状态"""
        signal = ctx.signal
        index = ctx.index
        # 记录开仓相关字段
        signal.at[index, 'low_price'] = result.low_price
        signal.at[index, 'low_index'] = result.low_index
        signal.at[index, 'low_date'] = str(
            signal.at[result.low_index, 'date']).removesuffix('.0')
        # 重置策略状态
        self._write_open_threshold_state(
            signal,
            index,
            self.frozen_open_threshold,
            self.frozen_open_cont_threshold,
            self.frozen_open_wd_threshold,
            self.frozen_open_atr,
            self.frozen_open_threshold,
            self.frozen_open_cont_threshold,
            self.frozen_open_wd_threshold,
        )
        self.open_base_price = result.low_price
        self._clear_frozen_open_thresholds()
        self.new_opening_count = ctx.integer_index - result.low_index
        self.var0 = 0
        self.new_opening = True

    def adjust_and_validate_open_execution(self, ctx: BarContext,
                                           result: OpenResult,
                                           execution_price: float) -> float:
        """做多开仓执行价校正与校验。"""
        quote = ctx.quote
        index = ctx.index
        exec_price = execution_price

        # 跳空上开: 若期望买价低于开盘，按开盘成交
        if exec_price < quote.loc[index, 'open']:
            exec_price = quote.loc[index, 'open']

        if exec_price > quote.loc[index, 'high']:
            print('long open execution price > high, plz check.')
            print(f'idx={index}, low_price={result.low_price}, '
                  f'exec={exec_price}, '
                  f'open={quote.loc[index, "open"]}, '
                  f'high={quote.loc[index, "high"]}, '
                  f'low={quote.loc[index, "low"]}')
            print('error index', index)
            print('\n')
        if exec_price < quote.loc[index, 'low']:
            print('long open execution price < low, plz check.')
            print(result.low_price, exec_price)
            print('error index', index)

        return exec_price

    def on_bar_holding(self, ctx: BarContext) -> CloseResult | None:
        quote = ctx.quote
        signal = ctx.signal
        index = ctx.index
        ii = ctx.integer_index
        p = self.params

        close_bar = p['close_bar']
        thresholds = self._resolve_thresholds(ctx)
        close_threshold = thresholds['close_threshold']
        self._write_open_threshold_state(
            signal,
            index,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

        # 初始化
        if self.new_opening:
            self.last_index = self.low_index
            self.increase_start_index = self.low_index
            self.holding_start_index = ii
            self.new_opening = False

        window_ready = (self.new_opening_count >= close_bar)
        if window_ready:
            self.last_index = ii - close_bar
        self.new_opening_count += 1

        analysis_slice = quote.iloc[self.last_index + 1:ii + 1]
        holding_slice = quote.iloc[self.increase_start_index:ii + 1]
        current_high = float(holding_slice['high'].max())
        base_price = self.open_base_price
        if pd.isna(base_price):
            base_price = float(holding_slice['low'].iloc[0])
        current_max_increase = max(current_high - base_price, 0.0)
        close_withdrawal_threshold = (
            current_max_increase * p['close_wd_max_inc_ratio']
        )
        signal.at[index, 'active_close_wd_threshold'] = close_withdrawal_threshold

        # 速度条件
        if window_ready:
            ana_inc_slice_1 = quote.iloc[self.low_index:ii + 1]
            ana_inc_slice_2 = quote.iloc[
                self.low_index:ii + 1 - close_bar]
            holding_increase = (
                ana_inc_slice_1.high.max() - ana_inc_slice_2.high.max())
            holding_base = analysis_slice['low'].iloc[0]
            self.holding_increase_percent = holding_increase / holding_base if holding_base != 0 else 0
            signal.at[index, 'holding_inc'] = holding_increase
            if holding_increase < close_threshold:
                signal.at[index, 'speed_close_signal'] = 1

        # 回撤条件
        with_high, holding_withdrawal = get_withdrawal(
            holding_slice, close_withdrawal_threshold, ii, switch0=True)
        holding_withdrawal_percent = (
            holding_withdrawal / with_high if with_high != 0 else 0)
        signal.at[index, 'holding_wd'] = holding_withdrawal
        signal.at[index, 'hld_wd_per'] = round(
            holding_withdrawal_percent * 100, 4)

        if holding_withdrawal > close_withdrawal_threshold:
            signal.at[index, 'holding_wd_signal'] = 1

        period = ii - self.holding_start_index + 1
        signal.at[index, 'high_price'] = max(holding_slice['high'])

        # 回撤平仓
        if signal.at[index, 'holding_wd_signal'] == 1:
            exec_price = max(holding_slice['high']) - close_withdrawal_threshold
            if exec_price > quote.loc[index, 'open']:
                exec_price = quote.loc[index, 'open']
            return CloseResult(
                close_type=1,
                execution_price=round(
                    exec_price, self.params['round_precision']),
                start_index=self.start_index,
                low_index=self.low_index,
                period=period,
            )

        # 速度平仓
        if signal.at[index, 'speed_close_signal'] == 1:
            return CloseResult(
                close_type=2,
                execution_price=round(
                    quote.loc[index]['close'],
                    self.params['round_precision']),
                start_index=self.start_index,
                low_index=self.low_index,
                period=period,
            )

        return None

    def on_position_closed(self, ctx: BarContext, result):
        """平仓后记录平仓信息并重置策略状态"""
        signal = ctx.signal
        index = ctx.index
        # 记录平仓相关字段
        signal.at[index, 'period'] = result.period
        signal.at[index, 'type'] = result.close_type
        # 重置策略状态
        self.open_base_price = np.nan
        self._clear_frozen_open_thresholds()
        self.new_opening = True
        self.new_opening_count = 0

    def on_trade_stats(self, ctx: BarContext,
                        start_index: int, low_index: int):
        """平仓后的交易统计: high_index, max_wd, max_inc"""
        quote = ctx.quote
        signal = ctx.signal
        index = ctx.index
        ii = ctx.integer_index

        increase3_slice = quote.iloc[start_index:ii + 1]
        increase3 = get_analysis_increase(increase3_slice)
        high_index = start_index
        for i in range(start_index + 1, ii + 2):
            high_index_slice = quote.iloc[start_index:i]
            increase4 = get_analysis_increase(high_index_slice)
            if np.isclose(increase4, increase3,
                          rtol=0.0, atol=self.HIGH_MATCH_EPS):
                high_index = i - 1
                break

        max_slice = quote.iloc[low_index:high_index + 1]
        max_wd = get_max_wd(max_slice)
        max_inc, inc_base = get_increase_with_base(max_slice)
        max_inc_percent = max_inc / inc_base if inc_base != 0 else 0
        signal.at[index, 'max_inc'] = round(max_inc_percent * 100, 4)
        signal.at[index, 'max_wd'] = round(max_wd * 100, 4)
        signal.at[index, 'high_index'] = high_index
        signal.at[index, 'high_date'] = str(
            signal.at[high_index, 'date']).removesuffix('.0')
        # 持仓平仓时也记录 high_price
        holding_slice = quote.iloc[
            self.increase_start_index:ii + 1]
        signal.at[index, 'high_price'] = max(holding_slice['high'])
        signal.at[index, 'low_index'] = low_index
        signal.at[index, 'low_date'] = str(
            signal.at[low_index, 'date']).removesuffix('.0')

    def _do_idle_stats(self, quote, signal, index, ii, stat_type):
        """未开仓时的 reset 统计（var0=3 或 var0=4）"""
        increase3_slice = quote.iloc[self.start_index:ii + 1]
        increase3 = get_analysis_increase(increase3_slice)
        high_index = self.start_index
        for i in range(self.start_index + 1, ii + 2):
            high_index_slice = quote.iloc[self.start_index:i]
            increase4 = get_analysis_increase(high_index_slice)
            if np.isclose(increase4, increase3,
                          rtol=0.0, atol=self.HIGH_MATCH_EPS):
                high_index = i - 1
                break
        max_slice = quote.iloc[self.low_index:high_index + 1]
        max_wd = get_max_wd(max_slice)
        max_inc, inc_base = get_increase_with_base(max_slice)
        max_inc_percent = max_inc / inc_base if inc_base != 0 else 0
        signal.at[index, 'max_inc'] = round(max_inc_percent * 100, 4)
        signal.at[index, 'max_wd'] = round(max_wd * 100, 4)
        signal.at[index, 'high_index'] = high_index
        signal.at[index, 'high_date'] = str(
            signal.at[high_index, 'date']).removesuffix('.0')
        signal.at[index, 'low_index'] = self.low_index
        signal.at[index, 'low_date'] = str(
            signal.at[self.low_index, 'date']).removesuffix('.0')
        signal.at[index, 'period'] = self.new_opening_count
        signal.at[index, 'type'] = stat_type


# ============================================================
# Main Script
# ============================================================

if __name__ == '__main__':

    # --- 数据加载 ---
    folder_path = data_folder_path
    file_name = data_file_name

    native_df, ROUND_PRECISION, NATIVE_BAR_SECONDS = load_data(folder_path, file_name)
    run_mode = str(run_mode).strip().lower()
    if run_mode not in ('manual', 'grid'):
        raise ValueError("run_mode must be 'manual' or 'grid'.")
    data_selection_mode = str(data_selection_mode).strip().lower()
    if data_selection_mode not in ('index', 'date'):
        raise ValueError("data_selection_mode must be 'index' or 'date'.")
    if int(atr_period) <= 0:
        raise ValueError('atr_period must be positive.')
    if close_wd_max_inc_ratio < 0:
        raise ValueError('close_wd_max_inc_ratio must be >= 0.')

    if run_mode == 'manual':
        if min(int(open_bar), int(close_bar)) <= 0:
            raise ValueError('open_bar and close_bar must be positive in manual mode.')
        open_bar_values = [int(open_bar)]
    else:
        if bar_step == 0:
            raise ValueError('bar_step cannot be 0.')
        if atr_step == 0:
            raise ValueError('atr_step cannot be 0.')
        if step3 <= 0:
            raise ValueError('step3 must be positive.')
        if int(for_num_3) <= 0:
            raise ValueError('for_num_3 must be positive.')
        if open_atr_stop_flat_rounds <= 0:
            raise ValueError('open_atr_stop_flat_rounds must be positive.')
        if open_atr_max_iterations <= 0:
            raise ValueError('open_atr_max_iterations must be positive.')
        open_bar_values = build_int_search_values(
            int(open_bar),
            int(bar_end),
            int(bar_step),
        )
        if len(open_bar_values) == 0:
            raise ValueError('open_bar search range is empty.')

    # 创建输出文件夹
    os.makedirs('./result', exist_ok=True)
    os.makedirs(f'./result/{file_name} long_momentum_ATR outcome/perf', exist_ok=True)
    os.makedirs(f'./result/{file_name} long_momentum_ATR outcome/trans', exist_ok=True)

    outcome_stats = pd.DataFrame()

    # 选择回测时间区间
    if data_selection_mode == 'index':
        startdate = start_index
        enddate = end_index

        if enddate == 'latest':
            native_preview_df = native_df.iloc[int(startdate):].copy()
        else:
            native_preview_df = native_df.iloc[int(startdate):int(enddate)].copy()
        range_start_label = startdate
        range_end_label = enddate
        print(f'[Main] native index range: ({startdate}, {enddate})')
    else:
        native_dates = pd.to_datetime(native_df['Date'], errors='coerce')
        if native_dates.isna().all():
            raise ValueError('Date column cannot be parsed for date selection.')
        startdate = str(start_date).strip()
        enddate = end_date
        start_ts = parse_selection_datetime(startdate, is_end=False)
        if str(enddate).strip().lower() == 'latest':
            date_mask = native_dates >= start_ts
        else:
            end_ts = parse_selection_datetime(str(enddate).strip(), is_end=True)
            if end_ts < start_ts:
                raise ValueError('end_date must be >= start_date.')
            date_mask = native_dates.between(start_ts, end_ts, inclusive='both')
        native_preview_df = native_df.loc[date_mask].copy()
        range_start_label = startdate
        range_end_label = enddate
        print(f'[Main] native date range: {startdate} -> {enddate}')

    if len(native_preview_df) == 0:
        raise ValueError(
            f'No data in selected range: {range_start_label} -> {range_end_label}'
        )
    print(
        f'[Main] native time range: '
        + f'{native_preview_df.iloc[0]["Date"]} -> {native_preview_df.iloc[-1]["Date"]}'
    )

    if (resample_rule or '').strip():
        preview_df, BAR_SECONDS = resample_ohlc_df(native_preview_df, resample_rule)
        print(f'[Data] resampled to {resample_rule}  |  bar period: {BAR_SECONDS}s')
        if (
            len(preview_df) > 0
            and should_drop_incomplete_initial_resampled_bar(
                native_preview_df, resample_rule)
        ):
            dropped_bar_date = preview_df.iloc[0]['Date']
            preview_df = preview_df.iloc[1:].reset_index(drop=True)
            print(
                '[Data] dropped incomplete initial resampled bar: '
                + str(dropped_bar_date)
            )
    else:
        preview_df = native_preview_df.copy()
        BAR_SECONDS = NATIVE_BAR_SECONDS
        print(f'[Data] using native period  |  bar period: {BAR_SECONDS}s')
    if len(preview_df) == 0:
        raise ValueError('No data remains after resampling the selected native range.')

    period_label = format_period_label(resample_rule, BAR_SECONDS)
    print(f'[Main] backtest time range: {preview_df.iloc[0]["Date"]} -> {preview_df.iloc[-1]["Date"]}')

    df5 = preview_df.reset_index(drop=True).copy()
    underlying = df5.copy()
    run_name = (
        f'period_{period_label} '
        + f'{make_safe_range_token(range_start_label)}-'
        + f'{make_safe_range_token(range_end_label)}'
    )
    dashboard_outcome_stats_path = (
        './result/%s long_momentum_ATR outcome/outcome stats/' % file_name
        + 'long_momentum_ATR ' + run_name + ' outcome_stats.xlsx'
    )

    only_close_cfg = only_close
    if only_close_cfg:
        underlying.open = underlying.low = underlying.high = underlying.close

    export_stats_enabled = EXPORT_STATS or (run_mode == 'manual')
    export_interactive_html_enabled = (
        EXPORT_INTERACTIVE_HTML or (run_mode == 'manual')
    )

    print(f'[Main] atr period: {atr_period}')
    underlying = precompute_atr(underlying, atr_period)

    if run_mode == 'manual' and (resample_rule or '').strip():
        prompt_manual_intrabar_precheck(
            native_preview_df,
            preview_df,
            BAR_SECONDS,
            underlying['atr'].astype(float) * float(open_atr_multiplier),
            underlying['atr'].astype(float) * float(open_continous_atr_multiplier),
            metric_kind='absolute',
        )

    # --- 参数循环 ---
    if run_mode == 'grid':
        print(
            '[Grid] open_bar: '
            + f'{open_bar_values[0]} -> {open_bar_values[-1]} step {bar_step}'
        )
        print(
            '[Grid] open_atr_multiplier: '
            + f'{open_atr_multiplier} step {atr_step}'
        )
        print(
            '[Grid] open_continous_atr_multiplier: start from open_atr_multiplier '
            + f'step {step3} max {for_num_3}'
        )
        print(
            '[Grid] stop each atr loop after '
            + f'{open_atr_stop_flat_rounds} unchanged trade-count steps'
        )
    else:
        print(
            '[Manual] open_bar=' + str(open_bar)
            + ' close_bar=' + str(close_bar)
        )
        print(
            '[Manual] open_atr_multiplier=' + str(open_atr_multiplier)
            + ' open_wd_atr_multiplier=' + str(open_wd_atr_multiplier)
            + ' open_continous_atr_multiplier=' + str(open_continous_atr_multiplier)
            + ' close_speed_atr_multiplier=' + str(close_speed_atr_multiplier)
        )
    print(
        '[Main] close_wd threshold = current max increase * '
        + f'{round(close_wd_max_inc_ratio, 4)}'
    )

    executed_run_count = 0
    if run_mode == 'grid':
        outcome_stats = load_existing_outcome_stats(dashboard_outcome_stats_path)
        existing_param_tags = set(outcome_stats.index.astype(str).tolist())
        if len(existing_param_tags) > 0:
            print(
                '[Grid] loaded existing stats rows: '
                + str(len(existing_param_tags))
            )
        planned_param_tags = build_planned_param_tags_long_atr(
            open_bar_values,
            float(open_atr_multiplier),
            int(open_atr_max_iterations),
            float(atr_step),
            int(for_num_3),
            float(step3),
            float(close_wd_max_inc_ratio),
            int(atr_period),
        )
        completed_param_tags = planned_param_tags.intersection(existing_param_tags)
        total_search_space = len(planned_param_tags)
        progress_marks = build_progress_marks(total_search_space)
        printed_progress_marks = set()
        print_search_progress(
            len(completed_param_tags),
            total_search_space,
            progress_marks,
            printed_progress_marks,
        )
    else:
        existing_param_tags = set()
        completed_param_tags = set()
        total_search_space = 0
        progress_marks = {}
        printed_progress_marks = set()

    for open_bar_runtime in open_bar_values:
        if run_mode == 'grid':
            close_bar_runtime = open_bar_runtime
        else:
            close_bar_runtime = int(close_bar)
        last_open_atr_trade_count = None
        unchanged_open_atr_steps = 0

        if run_mode == 'grid':
            atr_iterations = range(int(open_atr_max_iterations))
        else:
            atr_iterations = [0]

        for atr_iter in atr_iterations:
            if run_mode == 'grid':
                open_atr_multiplier_runtime = round(
                    open_atr_multiplier + (atr_iter * atr_step),
                    10,
                )
                open_wd_atr_multiplier_runtime = open_atr_multiplier_runtime
                close_speed_atr_multiplier_runtime = open_atr_multiplier_runtime
                print(
                    f'\n[Grid] open_bar={open_bar_runtime} '
                    + f'open_atr_multiplier={open_atr_multiplier_runtime}'
                )
                open_cont_iterations = range(int(for_num_3))
                last_open_cont_trade_count = None
                unchanged_open_cont_steps = 0
                outer_reference_trade_count = None
            else:
                print(
                    f'\n[Manual] open_bar={open_bar_runtime} '
                    + f'close_bar={close_bar_runtime}'
                )
                open_atr_multiplier_runtime = float(open_atr_multiplier)
                open_wd_atr_multiplier_runtime = float(open_wd_atr_multiplier)
                close_speed_atr_multiplier_runtime = float(close_speed_atr_multiplier)
                open_cont_iterations = [0]

            for open_cont_iter in open_cont_iterations:
                if run_mode == 'grid':
                    open_continous_atr_multiplier_runtime = round(
                        open_atr_multiplier_runtime + (open_cont_iter * step3),
                        10,
                    )
                    print(
                        f'[Grid]   open_continous_atr_multiplier='
                        + f'{open_continous_atr_multiplier_runtime}'
                    )
                else:
                    open_continous_atr_multiplier_runtime = float(
                        open_continous_atr_multiplier
                    )
                open_threshold = open_atr_multiplier_runtime
                open_withdrawal_threshold = open_wd_atr_multiplier_runtime
                open_continous_threshold = open_continous_atr_multiplier_runtime
                close_threshold = close_speed_atr_multiplier_runtime
                commision_percent_cfg = commision_percent
                capital_cfg = capital

                if min(
                    open_threshold,
                    open_withdrawal_threshold,
                    close_threshold,
                    open_continous_threshold,
                ) < 0:
                    print('atr multiplier不可为负数')
                    continue
                if open_continous_threshold < open_threshold:
                    print(
                        'open_continous_atr_multiplier不可小于'
                        + 'open_atr_multiplier'
                    )
                    continue

                param_tag = build_long_param_tag(
                    open_bar_runtime,
                    open_threshold,
                    open_continous_threshold,
                    open_withdrawal_threshold,
                    close_bar_runtime,
                    close_threshold,
                    close_wd_max_inc_ratio,
                    atr_period=atr_period,
                )
                if run_mode == 'grid' and param_tag in existing_param_tags:
                    print('[Grid] skip existing param: ' + param_tag)
                    continue

                params = {
                    'open_bar': open_bar_runtime,
                    'open_threshold': open_threshold,
                    'open_continous_threshold': open_continous_threshold,
                    'open_withdrawal_threshold': open_withdrawal_threshold,
                    'close_bar': close_bar_runtime,
                    'close_threshold': close_threshold,
                    'close_withdrawal_threshold': np.nan,
                    'close_wd_max_inc_ratio': close_wd_max_inc_ratio,
                    'round_precision': ROUND_PRECISION,
                }

                strategy = MomentumStrategy(params)
                engine = BacktestEngine(
                    underlying, strategy, capital_cfg,
                    ROUND_PRECISION, commision_percent_cfg,
                    show_progress=(run_mode != 'grid'))
                (df_signal, signal, close_counts) = engine.run()
                withdrawal_close_count = close_counts.get(1, 0)
                speed_close_count = close_counts.get(2, 0)
                total_trade_count = withdrawal_close_count + speed_close_count

                performance, transactions_df = generate_performance(
                    underlying, df_signal, capital_cfg, commision_percent_cfg)

                if len(transactions_df) > 1:
                    Capital_outcome = round(
                        transactions_df[
                            transactions_df.Type != 'long'].Capital.iloc[-1], 2)
                else:
                    Capital_outcome = capital_cfg
                perf_outcome = performance.reset_index(
                    drop=True)[['date', 'capital']]
                count_tag = (
                    str(round(withdrawal_close_count, 4))
                    + '+' + str(round(speed_close_count, 4))
                )
                result_tag = param_tag + ' ' + count_tag

                print(str(startdate) + '-' + str(enddate))
                print('total trade count = ' + str(total_trade_count))
                print('withdrawal close count = '
                      + str(round(withdrawal_close_count, 4)))
                print('speed close count = '
                      + str(round(speed_close_count, 4)))
                print(result_tag)
                print('profit: ' + str(round(performance.capital.iloc[-1], 2)))

                save_name = run_name + ' ' + result_tag
                fig1_title = str(Capital_outcome) + ' ' + save_name
                if SAVE_STATIC_PLOT:
                    plot_ext = 'pdf' if SAVE_PLOT_AS_PDF else 'png'
                    fig1_path = ('./result/%s long_momentum_ATR outcome/' % file_name
                                 + ' ' + str(Capital_outcome)
                                 + save_name + f' Long.{plot_ext}')
                    plot_backtest_chart(
                        underlying, transactions_df, perf_outcome,
                        title=fig1_title,
                        save_path=fig1_path,
                        close_fig=True)

                detail_df = pd.concat([signal, df5], axis=1, join='inner')
                detail_df = pd.concat(
                    [detail_df, perf_outcome.capital], axis=1, join='inner')
                if 'atr' in underlying.columns:
                    detail_df['atr'] = underlying['atr'].to_numpy()
                if 'atr_ready' in underlying.columns:
                    detail_df['atr_ready'] = underlying['atr_ready'].to_numpy()
                if 'true_range' in underlying.columns:
                    detail_df['true_range'] = underlying['true_range'].to_numpy()
                detail_df['open_trigger_price'] = (
                    detail_df['low_price'] + detail_df['frozen_open_cont_threshold']
                )
                frozen_basis = detail_df['frozen_open_atr']
                current_atr = detail_df['atr'] if 'atr' in detail_df.columns else np.nan
                detail_df['open_atr_multiplier_runtime'] = np.where(
                    frozen_basis.notna() & (frozen_basis != 0),
                    detail_df['frozen_open_threshold'] / frozen_basis,
                    np.nan,
                )
                detail_df['open_cont_atr_multiplier_runtime'] = np.where(
                    frozen_basis.notna() & (frozen_basis != 0),
                    detail_df['frozen_open_cont_threshold'] / frozen_basis,
                    np.nan,
                )
                detail_df['open_wd_atr_multiplier_runtime'] = np.where(
                    frozen_basis.notna() & (frozen_basis != 0),
                    detail_df['frozen_open_wd_threshold'] / frozen_basis,
                    np.nan,
                )
                if isinstance(current_atr, pd.Series):
                    detail_df['close_speed_atr_multiplier_runtime'] = np.where(
                        current_atr.notna() & (current_atr != 0),
                        detail_df['active_close_threshold'] / current_atr,
                        np.nan,
                    )
                else:
                    detail_df['close_speed_atr_multiplier_runtime'] = np.nan
                detail_df['close_wd_ratio_runtime'] = close_wd_max_inc_ratio
                detail_df.drop(
                    ['holding_signal', 'inc_signal', 'wd_signal',
                     'holding_wd_signal', 'total_inc_signal',
                     'speed_close_signal', 'have_holding'],
                    axis=1, inplace=True)
                detail_df.drop(
                    ['var0', 'low_index', 'high_index'],
                    axis=1, inplace=True)
                if len(detail_df) == 0:
                    detail_df.drop(
                        ['holding_wd', 'holding_inc', 'execution'],
                        axis=1, inplace=True)

                perf_name = (
                    param_tag + ' ' + count_tag
                    + ' ' + run_name + ' Long'
                    + ' ' + str(Capital_outcome)
                    + ' ' + 'perf.xlsx'
                )
                if export_stats_enabled:
                    writer1 = pd.ExcelWriter(
                    './result/%s long_momentum_ATR outcome/perf/' % file_name + perf_name,
                        engine='xlsxwriter')
                    detail_df.to_excel(writer1, sheet_name='stats')
                    workbook = writer1.book
                    worksheet = writer1.sheets['stats']
                    worksheet.set_default_row(15)
                    fmt = workbook.add_format()
                    fmt.set_font_name('Microsoft YaHei UI Light')
                    fmt.set_align('justify')
                    fmt.set_align('center')
                    fmt.set_align('vjustify')
                    fmt.set_align('vcenter')
                    fmt.set_font_size(12)
                    fmt1 = workbook.add_format({'num_format': '0'})
                    fmt1.set_font_name('Microsoft YaHei UI Light')
                    fmt1.set_align('justify')
                    fmt1.set_align('center')
                    fmt1.set_align('vjustify')
                    fmt1.set_align('vcenter')
                    worksheet.set_column('A:A', 7, fmt1)
                    worksheet.set_column('B:B', 18.5, fmt1)
                    worksheet.set_column('C:C', 12, fmt)
                    worksheet.set_column('D:D', 10, fmt)
                    worksheet.set_column('E:E', 9, fmt)
                    worksheet.set_column('F:F', 12, fmt)
                    worksheet.set_column('G:G', 11, fmt)
                    worksheet.set_column('H:H', 11, fmt)
                    worksheet.set_column('I:I', 11, fmt)
                    worksheet.set_column('J:J', 13, fmt)
                    worksheet.set_column('K:K', 9, fmt1)
                    worksheet.set_column('L:L', 8, fmt1)
                    worksheet.set_column('M:O', 8, fmt)
                    worksheet.set_column('P:P', 7.8, fmt1)
                    worksheet.set_column('Q:R', 10, fmt)
                    worksheet.set_column('S:S', 11.8, fmt)
                    worksheet.set_column('T:Y', 10.4, fmt)
                    worksheet.set_column('Z:Z', 22, fmt)
                    worksheet.freeze_panes(1, 2)
                    writer1.close()

                    if len(transactions_df) != 0:
                        writer2 = pd.ExcelWriter(
                        './result/%s long_momentum_ATR outcome/trans/' % file_name
                            + param_tag + ' ' + count_tag + ' '
                            + run_name + ' Long'
                            + ' ' + str(Capital_outcome)
                            + ' ' + 'trans.xlsx', engine='xlsxwriter')
                        transactions_df.reset_index(
                            drop=False).to_excel(writer2, sheet_name='stats')
                        workbook2 = writer2.book
                        worksheet2 = writer2.sheets['stats']
                        worksheet2.set_default_row(21)
                        fmt3 = workbook2.add_format()
                        fmt3.set_num_format('0')
                        fmt3.set_font_name('Microsoft YaHei UI Light')
                        fmt3.set_align('justify')
                        fmt3.set_align('center')
                        fmt3.set_align('vjustify')
                        fmt3.set_align('vcenter')
                        worksheet2.set_column('B:B', 17, fmt3)
                        fmt2 = workbook2.add_format()
                        fmt2.set_font_name('Microsoft YaHei UI Light')
                        fmt2.set_align('justify')
                        fmt2.set_align('center')
                        fmt2.set_align('vjustify')
                        fmt2.set_align('vcenter')
                        fmt2.set_font_size(12)
                        worksheet2.set_column('A:A', 11, fmt2)
                        worksheet2.set_column('C:D', 11, fmt2)
                        worksheet2.set_column('E:E', 14, fmt2)
                        worksheet2.set_column('F:G', 13, fmt2)
                        writer2.close()

                outcome_index = param_tag
                summary_metrics = build_summary_metrics(
                    perf_outcome,
                    transactions_df,
                    initial_capital=capital_cfg,
                )
                summary_row = {
                    'atr_period': atr_period,
                    'open_bar': open_bar_runtime,
                    'close_bar': close_bar_runtime,
                    'open_threshold': open_threshold,
                    'open_continous_threshold': open_continous_threshold,
                    'open_withdrawal_threshold': open_withdrawal_threshold,
                    'close_threshold': close_threshold,
                    'close_wd_max_inc_ratio': close_wd_max_inc_ratio,
                    'withdrawal_close_count': withdrawal_close_count,
                    'speed_close_count': speed_close_count,
                }
                for metric_name, metric_value in summary_metrics.items():
                    summary_row[metric_name] = metric_value
                summary_row['capital'] = summary_metrics['final_capital']
                summary_row['trade_num'] = summary_metrics['trade_num']
                summary_row['outcome_high'] = summary_metrics['outcome_high']
                summary_row['biggest_wd'] = summary_metrics['biggest_wd_abs']
                for metric_name, metric_value in summary_row.items():
                    outcome_stats.at[outcome_index, metric_name] = metric_value
                existing_param_tags.add(str(outcome_index))
                if run_mode == 'grid':
                    completed_param_tags.add(str(outcome_index))
                    print_search_progress(
                        len(completed_param_tags),
                        total_search_space,
                        progress_marks,
                        printed_progress_marks,
                    )

                executed_run_count += 1
                if run_mode == 'grid':
                    if outer_reference_trade_count is None:
                        outer_reference_trade_count = total_trade_count
                    if (
                        last_open_cont_trade_count is None
                        or total_trade_count != last_open_cont_trade_count
                    ):
                        unchanged_open_cont_steps = 0
                    else:
                        unchanged_open_cont_steps += 1
                    last_open_cont_trade_count = total_trade_count

                    if unchanged_open_cont_steps >= open_atr_stop_flat_rounds:
                        for remain_open_cont_iter in range(
                                open_cont_iter + 1,
                                int(for_num_3)):
                            remain_open_cont = round(
                                open_atr_multiplier_runtime
                                + (remain_open_cont_iter * step3),
                                10,
                            )
                            completed_param_tags.add(build_long_param_tag(
                                open_bar_runtime,
                                open_atr_multiplier_runtime,
                                remain_open_cont,
                                open_atr_multiplier_runtime,
                                close_bar_runtime,
                                open_atr_multiplier_runtime,
                                close_wd_max_inc_ratio,
                                atr_period=atr_period,
                            ))
                        print_search_progress(
                            len(completed_param_tags),
                            total_search_space,
                            progress_marks,
                            printed_progress_marks,
                        )
                        print(
                            '[Grid] stop open_cont atr loop at '
                            + f'open_bar={open_bar_runtime} '
                            + f'open_atr_multiplier={open_atr_multiplier_runtime}: '
                            + 'total trade count unchanged for '
                            + f'{open_atr_stop_flat_rounds} steps.'
                        )
                        break
            else:
                if run_mode == 'grid':
                    print(
                        '[Grid] reached for_num_3='
                        + str(for_num_3)
                        + f' at open_bar={open_bar_runtime} '
                        + f'open_atr_multiplier={open_atr_multiplier_runtime}.'
                    )

            if run_mode == 'grid':
                if outer_reference_trade_count is None:
                    continue
                if (
                    last_open_atr_trade_count is None
                    or outer_reference_trade_count != last_open_atr_trade_count
                ):
                    unchanged_open_atr_steps = 0
                else:
                    unchanged_open_atr_steps += 1
                last_open_atr_trade_count = outer_reference_trade_count

                if unchanged_open_atr_steps >= open_atr_stop_flat_rounds:
                    for remain_atr_iter in range(
                            atr_iter + 1,
                            int(open_atr_max_iterations)):
                        remain_open_atr = round(
                            open_atr_multiplier + (remain_atr_iter * atr_step),
                            10,
                        )
                        if remain_open_atr < 0:
                            continue
                        for remain_open_cont_iter in range(int(for_num_3)):
                            remain_open_cont = round(
                                remain_open_atr
                                + (remain_open_cont_iter * step3),
                                10,
                            )
                            if remain_open_cont < remain_open_atr:
                                continue
                            completed_param_tags.add(build_long_param_tag(
                                open_bar_runtime,
                                remain_open_atr,
                                remain_open_cont,
                                remain_open_atr,
                                close_bar_runtime,
                                remain_open_atr,
                                close_wd_max_inc_ratio,
                                atr_period=atr_period,
                            ))
                    print_search_progress(
                        len(completed_param_tags),
                        total_search_space,
                        progress_marks,
                        printed_progress_marks,
                    )
                    print(
                        '[Grid] stop open_atr loop at '
                        + f'open_bar={open_bar_runtime}: '
                        + 'base total trade count unchanged for '
                        + f'{open_atr_stop_flat_rounds} steps.'
                    )
                    break
        else:
            if run_mode == 'grid':
                print(
                    '[Grid] reached open_atr_max_iterations='
                    + str(open_atr_max_iterations)
                    + f' at open_bar={open_bar_runtime}.'
                )

    if len(outcome_stats) == 0:
        raise ValueError('No parameter combination was executed.')
    if run_mode == 'grid' and executed_run_count == 0:
        print('[Grid] no new parameter executed in this run.')

    print("\ntime = --- %s seconds ---" % (time.time() - start_time))
    outcome_stats.index.name = 'param_tag'

    # 多参数对比图
    if run_mode == 'grid' and len(outcome_stats) > 1:
        fig_stat_1 = plt.figure('stats', figsize=(18, 9))
        left = 0.033
        width = 0.943
        bottom = 0.055
        height = 0.9
        rect_line = [left, bottom, width, height]
        ax_stat_1 = fig_stat_1.add_axes(rect_line)
        stat_labels = outcome_stats.index.astype(str).tolist()
        stat_x = np.arange(len(stat_labels))
        stat_capital = outcome_stats['capital'].astype(float).to_numpy()
        stat_biggest_wd = outcome_stats['biggest_wd'].astype(float).to_numpy()
        stat_trade_num = outcome_stats['trade_num'].astype(float).to_numpy()

        line_capital, = ax_stat_1.plot(stat_x, stat_capital, label='capital')
        ax_stat_2 = ax_stat_1.twinx()
        line_biggest_wd, = ax_stat_2.plot(
            stat_x, stat_biggest_wd, color='orange', label='biggest wd')
        ax_stat_3 = ax_stat_1.twinx()
        line_trade_num, = ax_stat_3.plot(
            stat_x, stat_trade_num, color='salmon', label='trade num')
        ax_stat_3.tick_params(axis='y', colors='red')
        if len(stat_x) > 0:
            tick_count = min(12, len(stat_x))
            tick_positions = np.unique(
                np.linspace(0, len(stat_x) - 1, num=tick_count, dtype=int)
            )
            ax_stat_1.set_xticks(tick_positions)
            ax_stat_1.set_xticklabels(
                [stat_labels[i] for i in tick_positions],
                rotation=70
            )
            ax_stat_1.set_xlim(-0.5, len(stat_x) - 0.5)

        stat_cursor = Cursor(
            ax_stat_1, useblit=True, color='gray', linewidth=0.8)
        stat_cursor.visible = True
        stat_hover = ax_stat_1.annotate(
            "",
            xy=(0, 0),
            xytext=(18, 18),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"))
        stat_hover.set_visible(False)

        def update_stat_hover(stat_idx):
            stat_hover.xy = (stat_x[stat_idx], stat_capital[stat_idx])
            stat_hover.set_text(
                stat_labels[stat_idx] + '\n'
                + 'capital: ' + str(round(stat_capital[stat_idx], 4)) + '\n'
                + 'biggest wd: ' + str(round(stat_biggest_wd[stat_idx], 4)) + '\n'
                + 'trade num: ' + str(round(stat_trade_num[stat_idx], 4))
            )
            stat_hover.get_bbox_patch().set_alpha(0.82)

        def hover_stat(event):
            if event.inaxes not in (ax_stat_1, ax_stat_2, ax_stat_3):
                if stat_hover.get_visible():
                    stat_hover.set_visible(False)
                    fig_stat_1.canvas.draw_idle()
                return
            if event.xdata is None or len(stat_x) == 0:
                if stat_hover.get_visible():
                    stat_hover.set_visible(False)
                    fig_stat_1.canvas.draw_idle()
                return

            stat_idx = int(np.clip(round(event.xdata), 0, len(stat_x) - 1))
            update_stat_hover(stat_idx)
            if not stat_hover.get_visible():
                stat_hover.set_visible(True)
            fig_stat_1.canvas.draw_idle()

        fig_stat_1.canvas.mpl_connect("motion_notify_event", hover_stat)
        fig_stat_1.show()
        fig_stat_1.legend(
            handles=[line_capital, line_biggest_wd, line_trade_num],
            labels=['capital', 'biggest wd', 'trade num']
        )
        plt.title('stats ' + run_name)
        os.makedirs('./result/stats %s long_momentum_ATR outcome/' % file_name, exist_ok=True)
        stats_plot_ext = 'pdf' if SAVE_PLOT_AS_PDF else 'png'
        plt.savefig('./result/stats %s long_momentum_ATR outcome/' % file_name
                    + ' ' + run_name + ' '
                    + str(executed_run_count) + ' '
                    + f'all outcome.{stats_plot_ext}', dpi=1000)
        outcome_stats.to_excel('./result/stats %s long_momentum_ATR outcome/' % file_name
                               + ' ' + run_name + ' '
                               + str(executed_run_count) + ' '
                               + 'all outcome.xlsx')
    else:
        disk_path = './result/'
        open_excel = False
        if open_excel:
            os.startfile(
            disk_path + '%s long_momentum_ATR outcome/perf/' % file_name + perf_name)

    # ====== 交互式图 (fig2) ======
    if run_mode == 'manual' and executed_run_count == 1:
        fig2 = plt.figure(figsize=(18, 9))
        left = 0.043
        width = 0.943
        bottom = 0.055
        height = 0.9
        rect_line = [left, bottom, width, height]
        ax2 = fig2.add_axes(rect_line)

        underlying1 = underlying.reset_index(drop=True)
        factor = underlying1['open'][0]
        underlying_ratio = pd.DataFrame()
        underlying_ratio['Date'] = underlying1['Date']
        underlying_ratio[['open', 'high', 'low', 'close']] = underlying1[
            ['open', 'high', 'low', 'close']] / factor * 100
        date_list_0 = underlying1.Date.to_list()
        date_list = [str(ii) for ii in date_list_0]
        underlying_ratio.index = date_list

        long_record = transactions_df.copy()
        long_record['target'] = long_record['Price'] / factor * 100
        long_record = long_record[long_record.Type == 'long']
        long_record['date'] = long_record['Date'].astype(str).str[:-3]
        if len(long_record) != 0:
            scatter_r = ax2.scatter(
                long_record.index, long_record['target'], c='red', s=10)

        sell_record = transactions_df.copy()
        sell_record['target'] = sell_record['Price'] / factor * 100
        sell_record = sell_record[sell_record.Type == 'sell']
        sell_record['date'] = sell_record['Date'].astype(str).str[:-3]
        trade_seq = transactions_df[
            transactions_df['Type'].isin(['long', 'sell'])].copy()
        trade_seq = trade_seq.sort_index()
        trade_seq['target'] = trade_seq['Price'] / factor * 100

        def draw_trade_links(ax_obj):
            buy_idx = None
            buy_y = None
            for idx, row in trade_seq.iterrows():
                if row['Type'] == 'long':
                    buy_idx = idx
                    buy_y = row['target']
                elif row['Type'] == 'sell' and buy_idx is not None:
                    sell_idx = idx
                    sell_y = row['target']
                    ax_obj.plot(
                        [buy_idx, sell_idx], [buy_y, sell_y],
                        color=ACCENT_BLUE, linewidth=2.0, alpha=0.8, zorder=1)
                    buy_idx = None
                    buy_y = None

        def attach_trade_hover(fig_obj, ax_obj, extra_hover_axes=None):
            hover_axes = {ax_obj}
            if extra_hover_axes is not None:
                hover_axes.update(extra_hover_axes)

            if len(long_record) != 0:
                scatter_r_local = ax_obj.scatter(
                    long_record.index, long_record['target'], c='red', s=10)
            else:
                scatter_r_local = None

            close_type_1_df_local = pd.DataFrame()
            close_type_2_df_local = pd.DataFrame()
            scatter_g_local = None
            scatter_b_local = None
            if len(sell_record) != 0:
                close_type_1_df_local = sell_record[sell_record['Close_type'] == 1]
                close_type_2_df_local = sell_record[sell_record['Close_type'] == 2]
                if len(close_type_1_df_local) != 0:
                    scatter_g_local = ax_obj.scatter(
                        close_type_1_df_local.index,
                        close_type_1_df_local['target'],
                        c=SELL_WD_COLOR,
                        s=10,
                    )
                if len(close_type_2_df_local) != 0:
                    scatter_b_local = ax_obj.scatter(
                        close_type_2_df_local.index,
                        close_type_2_df_local['target'],
                        c=SELL_SPEED_COLOR,
                        s=10,
                    )

            if scatter_r_local is not None:
                annot_r = ax_obj.annotate(
                    "", xy=(0, 0), xytext=(20, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
                annot_r.set_visible(False)

                def update_annot_r(ind):
                    index_num = ind["ind"][0]
                    pos = scatter_r_local.get_offsets()[index_num]
                    annot_r.xy = pos
                    trade_data = long_record.iloc[index_num]
                    index0 = trade_data.name
                    date = str(trade_data['Date'])[:-3]
                    pref_data = detail_df.loc[index0]
                    high = format_hover_value(pref_data.high, 6)
                    atr_now = format_hover_value(pref_data.get('atr', np.nan), 6)
                    frozen_atr = format_hover_value(
                        pref_data.get('frozen_open_atr', np.nan), 6)
                    oa_mult = format_hover_value(
                        pref_data.get('open_atr_multiplier_runtime', np.nan), 4)
                    oca_mult = format_hover_value(
                        pref_data.get('open_cont_atr_multiplier_runtime', np.nan), 4)
                    owa_mult = format_hover_value(
                        pref_data.get('open_wd_atr_multiplier_runtime', np.nan), 4)
                    trigger_price = format_hover_value(
                        pref_data.get('open_trigger_price', np.nan), 6)
                    increase = format_hover_value(
                        pref_data.get('increase', np.nan), 6)
                    withdrawal = format_hover_value(
                        pref_data.get('withdrawal', np.nan), 6)
                    t_inc_per = format_hover_value(
                        pref_data.get('t_inc_per', np.nan), 2)
                    execution = format_hover_value(
                        pref_data.get('execution', np.nan), 6)
                    low_date = str(pref_data.get('low_date', 'nan'))
                    new_opening_count = format_hover_value(
                        pref_data.get('new_opening_count', np.nan), 0)
                    low_price = format_hover_value(
                        pref_data.get('low_price', np.nan), 6)
                    text = (date[:-5] + ' ' + date[-5:] + '\n'
                            + 'high: ' + high + '\n'
                            + 'atr: ' + atr_now + '\n'
                            + 'frozen_atr: ' + frozen_atr + '\n'
                            + 'oa_mult: ' + oa_mult + '\n'
                            + 'oca_mult: ' + oca_mult + '\n'
                            + 'owa_mult: ' + owa_mult + '\n'
                            + 'trigger: ' + trigger_price + '\n'
                            + 'inc_abs: ' + increase + '\n'
                            + 'wd_abs: ' + withdrawal + '\n'
                            + 'total_inc: ' + t_inc_per + '%' + '\n'
                            + 'execution: ' + execution + '\n'
                            + 'low_date: ' + low_date + '\n'
                            + 'low_price: ' + low_price + '\n'
                            + 'new_opening_count: ' + new_opening_count + '\n'
                            + 'index: ' + str(index0) + '\n')
                    annot_r.set_text(text)
                    annot_r.get_bbox_patch().set_alpha(0.4)

                def hover_r(event):
                    vis = annot_r.get_visible()
                    if event.inaxes in hover_axes:
                        cont, ind = scatter_r_local.contains(event)
                        if cont:
                            update_annot_r(ind)
                            annot_r.set_visible(True)
                            fig_obj.canvas.draw_idle()
                        elif vis:
                            annot_r.set_visible(False)
                            fig_obj.canvas.draw_idle()

                fig_obj.canvas.mpl_connect("motion_notify_event", hover_r)

            if scatter_g_local is not None:
                annot_g = ax_obj.annotate(
                    "", xy=(0, 0), xytext=(20, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
                annot_g.set_visible(False)

                def update_annot_g(ind):
                    index_num = ind["ind"][0]
                    pos = scatter_g_local.get_offsets()[index_num]
                    annot_g.xy = pos
                    trade_data = close_type_1_df_local.iloc[index_num]
                    index0 = trade_data.name
                    date = str(trade_data['Date'])[:-3]
                    pref_data = detail_df.loc[index0]
                    low = format_hover_value(pref_data.low, 6)
                    atr_now = format_hover_value(pref_data.get('atr', np.nan), 6)
                    close_mult = format_hover_value(
                        pref_data.get('close_speed_atr_multiplier_runtime', np.nan), 4)
                    close_threshold = format_hover_value(
                        pref_data.get('active_close_threshold', np.nan), 6)
                    close_wd_threshold = format_hover_value(
                        pref_data.get('active_close_wd_threshold', np.nan), 6)
                    close_wd_ratio = format_hover_value(
                        pref_data.get('close_wd_ratio_runtime', np.nan), 4)
                    hld_wd_per = format_hover_value(
                        pref_data.get('hld_wd_per', np.nan), 2)
                    holding_inc = format_hover_value(
                        pref_data.get('holding_inc', np.nan), 2)
                    max_inc = format_hover_value(
                        pref_data.get('max_inc', np.nan), 2)
                    max_wd = format_hover_value(
                        pref_data.get('max_wd', np.nan), 2)
                    execution = format_hover_value(
                        pref_data.get('execution', np.nan), 6)
                    low_date = str(pref_data.get('low_date', 'nan'))
                    high_date = str(pref_data.get('high_date', 'nan'))
                    high_price = format_hover_value(
                        pref_data.get('high_price', np.nan), 6)
                    period = format_hover_value(pref_data.get('period', np.nan), 0)
                    text = (date[:-5] + ' ' + date[-5:] + '\n'
                            + 'low: ' + low + '\n'
                            + 'atr: ' + atr_now + '\n'
                            + 'ca_mult: ' + close_mult + '\n'
                            + 'close_th: ' + close_threshold + '\n'
                            + 'close_wd_th: ' + close_wd_threshold + '\n'
                            + 'cwd_ratio: ' + close_wd_ratio + '\n'
                            + 'hld_wd_per: ' + hld_wd_per + '%' + '\n'
                            + 'holding_inc: ' + holding_inc + '\n'
                            + 'max_inc: ' + max_inc + '%' + '\n'
                            + 'max_wd: ' + max_wd + '%' + '\n'
                            + 'execution2: ' + execution + '\n'
                            + 'period: ' + period + '\n'
                            + 'low_date: ' + low_date + '\n'
                            + 'high_date: ' + high_date + '\n'
                            + 'high_price: ' + high_price + '\n'
                            + 'index: ' + str(index0))
                    annot_g.set_text(text)
                    annot_g.get_bbox_patch().set_alpha(0.4)

                def hover_g(event):
                    vis = annot_g.get_visible()
                    if event.inaxes in hover_axes:
                        cont, ind = scatter_g_local.contains(event)
                        if cont:
                            update_annot_g(ind)
                            annot_g.set_visible(True)
                            fig_obj.canvas.draw_idle()
                        elif vis:
                            annot_g.set_visible(False)
                            fig_obj.canvas.draw_idle()

                fig_obj.canvas.mpl_connect("motion_notify_event", hover_g)

            if scatter_b_local is not None:
                annot_b = ax_obj.annotate(
                    "", xy=(0, 0), xytext=(20, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
                annot_b.set_visible(False)

                def update_annot_b(ind):
                    index_num = ind["ind"][0]
                    pos = scatter_b_local.get_offsets()[index_num]
                    annot_b.xy = pos
                    trade_data = close_type_2_df_local.iloc[index_num]
                    index0 = trade_data.name
                    date = str(trade_data['Date'])[:-3]
                    pref_data = detail_df.loc[index0]
                    low = format_hover_value(pref_data.low, 6)
                    atr_now = format_hover_value(pref_data.get('atr', np.nan), 6)
                    close_mult = format_hover_value(
                        pref_data.get('close_speed_atr_multiplier_runtime', np.nan), 4)
                    close_threshold = format_hover_value(
                        pref_data.get('active_close_threshold', np.nan), 6)
                    close_wd_threshold = format_hover_value(
                        pref_data.get('active_close_wd_threshold', np.nan), 6)
                    close_wd_ratio = format_hover_value(
                        pref_data.get('close_wd_ratio_runtime', np.nan), 4)
                    hld_wd_per = format_hover_value(
                        pref_data.get('hld_wd_per', np.nan), 2)
                    max_inc = format_hover_value(
                        pref_data.get('max_inc', np.nan), 2)
                    max_wd = format_hover_value(
                        pref_data.get('max_wd', np.nan), 2)
                    execution = format_hover_value(
                        pref_data.get('execution', np.nan), 6)
                    low_date = str(pref_data.get('low_date', 'nan'))
                    high_date = str(pref_data.get('high_date', 'nan'))
                    high_price = format_hover_value(
                        pref_data.get('high_price', np.nan), 6)
                    period = format_hover_value(pref_data.get('period', np.nan), 0)
                    text = (date[:-5] + ' ' + date[-5:] + '\n'
                            + 'low: ' + low + '\n'
                            + 'atr: ' + atr_now + '\n'
                            + 'ca_mult: ' + close_mult + '\n'
                            + 'close_th: ' + close_threshold + '\n'
                            + 'close_wd_th: ' + close_wd_threshold + '\n'
                            + 'cwd_ratio: ' + close_wd_ratio + '\n'
                            + 'hld_wd_per: ' + hld_wd_per + '%' + '\n'
                            + 'max_inc: ' + max_inc + '%' + '\n'
                            + 'max_wd: ' + max_wd + '%' + '\n'
                            + 'execution2: ' + execution + '\n'
                            + 'period: ' + period + '\n'
                            + 'low_date: ' + low_date + '\n'
                            + 'high_date: ' + high_date + '\n'
                            + 'high_price: ' + high_price + '\n'
                            + 'index: ' + str(index0))
                    annot_b.set_text(text)
                    annot_b.get_bbox_patch().set_alpha(0.4)

                def hover_b(event):
                    vis = annot_b.get_visible()
                    if event.inaxes in hover_axes:
                        cont, ind = scatter_b_local.contains(event)
                        if cont:
                            update_annot_b(ind)
                            annot_b.set_visible(True)
                            fig_obj.canvas.draw_idle()
                        elif vis:
                            annot_b.set_visible(False)
                            fig_obj.canvas.draw_idle()

                fig_obj.canvas.mpl_connect("motion_notify_event", hover_b)

        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.xticks(rotation=0)
        fig2_title = (
            ' ' + str(round(Capital_outcome, 2))
            + ' ' + result_tag
            + ' ' + run_name
        )
        plt.title('%s' % fig2_title)

        ax2.plot(detail_df.index, detail_df.capital,
                 linewidth=1.2, color=ACCENT_BLUE)
        draw_monochrome_candles(
            ax2,
            underlying_ratio[['open', 'high', 'low', 'close']],
            width=0.68,
        )
        draw_long_gap_lines(ax2, underlying1)
        draw_trade_links(ax2)
        attach_trade_hover(fig2, ax2)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(12))

        fig3 = plt.figure(figsize=(18, 9))
        rect_line_diag = [left, bottom, 0.915, height]
        ax3 = fig3.add_axes(rect_line_diag)
        ax3_atr = ax3.twinx()
        ax3.set_zorder(2)
        ax3_atr.set_zorder(1)
        ax3_atr.patch.set_visible(False)
        ax3.set_facecolor('white')
        ax3.patch.set_alpha(0.0)
        ax3.grid(
            True, which='major',
            color='#d2d2d2', linestyle='--', linewidth=0.6, alpha=0.95)
        ax3.grid(
            True, which='minor',
            color='#ececec', linestyle='--', linewidth=0.5, alpha=0.9)
        ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        draw_monochrome_candles(
            ax3,
            underlying_ratio[['open', 'high', 'low', 'close']],
            width=0.68,
        )
        draw_long_gap_lines(ax3, underlying1)
        draw_trade_links(ax3)
        attach_trade_hover(fig3, ax3, extra_hover_axes=[ax3_atr])
        ax3_atr.plot(
            underlying1.index,
            underlying1['atr'],
            color='#6b7280',
            linewidth=1.15,
            alpha=0.95,
        )
        ax3.set_xlim(-0.7, len(underlying_ratio) - 0.3)
        ax3.xaxis.set_major_locator(plt.MaxNLocator(12))
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3_atr.spines['top'].set_visible(False)
        ax3_atr.spines['left'].set_visible(False)
        ax3_atr.grid(False)
        ax3_atr.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        ax3_atr.set_ylabel('ATR', color='#6b7280')
        ax3_atr.tick_params(axis='y', colors='#6b7280', pad=6)
        ax3.set_title(
            'ATR view ' + str(round(Capital_outcome, 2))
            + ' ' + result_tag + ' ' + run_name
        )

        if export_interactive_html_enabled:
            export_interactive_html_long(
                file_name=file_name,
                save_name=save_name,
                title=fig2_title,
                underlying1=underlying1,
                detail_df=detail_df,
                transactions_df=transactions_df,
                factor=factor
            )
        plt.show()

    export_outcome_stats = outcome_stats.sort_index()
    export_outcome_stats.index.name = 'param_tag'
    os.makedirs('./result/%s long_momentum_ATR outcome/outcome stats/' % file_name,
                exist_ok=True)
    export_outcome_stats.to_excel(dashboard_outcome_stats_path)
