# -*- coding: utf-8 -*-
"""
Long Momentum Strategy - 动量做多策略
=====================================
策略入口脚本：包含 MomentumStrategy 类、参数循环、绘图、Excel 输出。
依赖 backtest_main.py 中的通用框架。
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
import matplotlib.ticker as ticker
from mplfinance.original_flavor import candlestick2_ohlc
import time, os, json
from contextlib import contextmanager
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
data_folder_path = r"D:\Code\data\20260326\\"
data_file_name = "xagusd_30s_all"

# 回测区间
# data_selection_mode:
# 'index' = 使用 start_index / end_index，在原始数据上按切片语义 [start_index, end_index) 取数
# 'date' = 使用 start_date / end_date，在原始数据上按时间 between 取数
data_selection_mode = 'date'
start_index = 2000
end_index = 5000  # 或 'latest'
start_date = '20250601'
end_date = '20250701'  # 或 'latest'
only_close = False

# 重采样设置：设为 '' 表示直接使用原始周期
# 例如 '1min' / '5min' / '15min' / '1H'
resample_rule = '1min'

# 运行模式：
# 'manual' = 单周期单参数验证
# 'grid' = 执行多周期参数搜索
run_mode = 'manual'

# 多周期参数
PERIOD_LIST = (
    list(range(1, 31))
    + list(range(33, 61, 3))
    + list(range(65, 121, 5))
)
MANUAL_PERIOD = 30

# 动态阈值参数
DAILY_ATR_LOOKBACK_DAYS = 10
DAILY_ATR_MULTIPLIER = 0.25
RECENT_MEAN_FACTOR = 10
RECENT_MEAN_MULTIPLIER = 2.0

# 平仓参数
SPEED_EXIT_FACTOR = 2
SPEED_EXIT_THRESHOLD = 0.0
ENABLE_WD_EXIT = False
WD_EXIT_MULTIPLIER = 1.0

# Case HTML
EXPORT_CASE_HTML = True
CASE_CONTEXT_BARS = 20
SHOW_MANUAL_PLOT = False

# Grid search
# for_num_1: 搜索 DAILY_ATR_MULTIPLIER
for_num_1 = 1
step1 = 0.05
# for_num_2: 搜索 RECENT_MEAN_MULTIPLIER
for_num_2 = 1
step2 = 0.5
for_num_3 = 1
step3 = 0.0
for_num_4 = 1
step4 = 0.0
open_threshold_stop_flat_rounds = 5

# 兼容原有字段
open_bar = MANUAL_PERIOD
open_threshold = DAILY_ATR_MULTIPLIER
open_withdrawal_threshold = 999.0
close_bar = SPEED_EXIT_FACTOR * MANUAL_PERIOD
close_threshold = SPEED_EXIT_THRESHOLD
open_continous_threshold = DAILY_ATR_MULTIPLIER
close_withdrawal_threshold = 999.0
open_bar2 = np.nan
open_threshold2 = np.nan
open_continous_threshold2 = 0.0
close_withdrawal_threshold2 = 999.0
close_withdrawal_mode = 'fixed_high_pct'

commision_percent = 0.000
capital = 100.0
export_interactive_html = True
accent_blue = '#1F77B4'
sell_wd_color = 'green'
sell_speed_color = 'black'
html_crosshair_enabled = False
html_crosshair_color = 'rgba(255, 120, 120, 0.45)'
html_show_trade_count_badge = True
# 两种模式都会保存结果图，默认使用较清晰且体积适中的 JPG。
result_image_ext = 'jpg'
result_image_dpi = 220
# 静态图保存开关：默认不保存 PDF/PNG（保留 HTML 导出）
save_static_plot = False
# 当 save_static_plot=True 时决定保存为 PDF 或 PNG
save_plot_as_pdf = False
grid_outcome_stats_flush_every = 10

# 并发 grid 可通过环境变量覆盖这几个值
grid_shard_tag = os.environ.get('LM_GRID_SHARD_TAG', '').strip()
grid_open_bar_values_env = os.environ.get('LM_GRID_OPEN_BAR_VALUES', '').strip()
run_mode = os.environ.get('LM_RUN_MODE', run_mode).strip()
if os.environ.get('LMP_MANUAL_PERIOD', '').strip():
    MANUAL_PERIOD = int(os.environ['LMP_MANUAL_PERIOD'])
if os.environ.get('LMP_DAILY_ATR_MULTIPLIER', '').strip():
    DAILY_ATR_MULTIPLIER = float(os.environ['LMP_DAILY_ATR_MULTIPLIER'])
if os.environ.get('LMP_RECENT_MEAN_MULTIPLIER', '').strip():
    RECENT_MEAN_MULTIPLIER = float(os.environ['LMP_RECENT_MEAN_MULTIPLIER'])
if os.environ.get('LMP_RECENT_MEAN_FACTOR', '').strip():
    RECENT_MEAN_FACTOR = float(os.environ['LMP_RECENT_MEAN_FACTOR'])
if os.environ.get('LMP_EXPORT_CASE_HTML', '').strip():
    EXPORT_CASE_HTML = os.environ['LMP_EXPORT_CASE_HTML'].strip() not in (
        '0', 'false', 'False')

open_bar = MANUAL_PERIOD
open_threshold = DAILY_ATR_MULTIPLIER
open_continous_threshold = DAILY_ATR_MULTIPLIER
close_bar = SPEED_EXIT_FACTOR * MANUAL_PERIOD
close_threshold = SPEED_EXIT_THRESHOLD
outcome_dir_name = f'{data_file_name} multi_period outcome'


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
    normalized_rule = (rule or '').strip()
    if not normalized_rule:
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
    temp = temp.resample(normalized_rule).agg(agg)
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


def build_summary_metrics(
        perf_outcome: pd.DataFrame,
        transactions_df: pd.DataFrame,
        initial_capital: float) -> dict:
    capital_curve = perf_outcome['capital'].astype(float)
    final_capital = float(capital_curve.iloc[-1]) if len(capital_curve) else float(initial_capital)
    total_return_pct = (
        (final_capital / float(initial_capital) - 1.0) * 100.0
        if initial_capital != 0 else np.nan
    )

    outcome_high, biggest_wd_abs = get_outcome_withdrawal(capital_curve)
    biggest_wd_pct = (
        (biggest_wd_abs / outcome_high) * 100.0
        if outcome_high not in (0, np.nan) and not pd.isna(outcome_high) else np.nan
    )

    closed_trades = transactions_df[transactions_df['Type'] != 'long'].copy()
    trade_num = int(len(closed_trades))
    trade_returns = pd.to_numeric(closed_trades['Percent'], errors='coerce') - 1.0
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
            sharpe_ratio = float(bar_returns.mean() / std * np.sqrt(len(bar_returns)))

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


def build_long_param_tag(
        period_n: int,
        daily_atr_multiplier: float,
        recent_mean_multiplier: float,
        include_thresholds: bool = True) -> str:
    tag = 'p' + f'{int(period_n):03d}'
    if include_thresholds:
        tag += ' dv' + str(round(float(daily_atr_multiplier), 6))
        tag += ' rm' + str(round(float(recent_mean_multiplier), 6))
    return tag


def build_int_search_values(start: int, end: int, step: int) -> list[int]:
    if step == 0:
        raise ValueError('step cannot be 0.')
    if step > 0 and start > end:
        raise ValueError('positive step requires start <= end.')
    if step < 0 and start < end:
        raise ValueError('negative step requires start >= end.')
    stop = end + (1 if step > 0 else -1)
    return list(range(start, stop, step))


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


def build_planned_param_tags_long(
        period_values: list[int],
        daily_atr_multiplier_start: float,
        for_num_1_runtime: int,
        step1_value: float,
        recent_mean_multiplier_start: float,
        for_num_2_runtime: int,
        step2_value: float) -> set[str]:
    planned_tags = set()
    include_thresholds = (
        int(for_num_1_runtime) > 1
        or int(for_num_2_runtime) > 1
        or float(step1_value) != 0
        or float(step2_value) != 0
    )
    for period_n in period_values:
        for atr_iter in range(int(for_num_1_runtime)):
            atr_value = round(
                daily_atr_multiplier_start + (atr_iter * step1_value),
                10,
            )
            for recent_iter in range(int(for_num_2_runtime)):
                recent_value = round(
                    recent_mean_multiplier_start + (recent_iter * step2_value),
                    10,
                )
                planned_tags.add(build_long_param_tag(
                    period_n,
                    atr_value,
                    recent_value,
                    include_thresholds=include_thresholds,
                ))
    return planned_tags


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    if pd.isna(value) if not isinstance(value, str) else False:
        return None
    return value


def load_progress_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def save_progress_json(path: str, payload: dict):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(make_json_safe(payload), fh, ensure_ascii=False, indent=2)


def load_progress_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    return df[~df.index.duplicated(keep='last')]


def load_existing_outcome_stats(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path, index_col=0)
    return df[~df.index.duplicated(keep='last')]


def merge_outcome_stats(base_df: pd.DataFrame, extra_df: pd.DataFrame) -> pd.DataFrame:
    if len(base_df) == 0:
        return extra_df.copy()
    if len(extra_df) == 0:
        return base_df.copy()
    merged_df = pd.concat([base_df, extra_df], axis=0)
    return merged_df[~merged_df.index.duplicated(keep='last')]


def sort_outcome_stats(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    ordered_df = df.copy()
    ordered_df['_param_tag_sort'] = ordered_df.index.astype(str)
    sort_cols = [
        col for col in (
            'period',
            'daily_atr_multiplier',
            'recent_mean_multiplier',
            'open_bar',
            'open_threshold',
            'open_continous_threshold',
            'close_bar',
            'close_threshold',
            'close_withdrawal_threshold',
            '_param_tag_sort',
        ) if col in ordered_df.columns
    ]
    if not sort_cols:
        return ordered_df.sort_index()
    ordered_df = ordered_df.sort_values(sort_cols, kind='mergesort')
    ordered_df = ordered_df.drop(columns=['_param_tag_sort'], errors='ignore')
    return ordered_df


def parse_env_open_bar_values(default_values: list[int]) -> list[int]:
    text = grid_open_bar_values_env
    if not text:
        return default_values
    values = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if len(values) == 0:
        raise ValueError('LM_GRID_OPEN_BAR_VALUES is empty.')
    if min(values) <= 0:
        raise ValueError('LM_GRID_OPEN_BAR_VALUES must contain positive integers.')
    return values


@contextmanager
def file_lock(lock_path: str, timeout_seconds: float = 120.0):
    start_at = time.time()
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode('utf-8'))
            break
        except FileExistsError:
            if time.time() - start_at > timeout_seconds:
                raise TimeoutError('Lock wait timeout: ' + lock_path)
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


def append_progress_summary(path: str, row_index: str, row_data: dict):
    row_df = pd.DataFrame([row_data], index=[row_index])
    row_df.index.name = 'param_tag'
    row_df.to_csv(
        path,
        mode='a',
        header=not os.path.exists(path),
        encoding='utf-8-sig',
    )


def flush_dashboard_outcome_stats(path: str, outcome_stats_df: pd.DataFrame) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    lock_path = path + '.lock'
    with file_lock(lock_path):
        export_df = sort_outcome_stats(merge_outcome_stats(
            load_existing_outcome_stats(path),
            outcome_stats_df,
        ))
        drop_param_cols = [
            col for col in export_df.columns
            if str(col).strip().lower().startswith('param_tag')
        ]
        if drop_param_cols:
            export_df = export_df.drop(columns=drop_param_cols, errors='ignore')
        export_df.index.name = 'param_tag'
        export_df.to_excel(path)


def save_result_figure(fig, path: str, dpi: int):
    save_kwargs = {
        'dpi': dpi,
        'bbox_inches': 'tight',
    }
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        save_kwargs['pil_kwargs'] = {
            'quality': 92,
            'optimize': True,
        }
    fig.savefig(path, **save_kwargs)


def _increase_with_base_from_arrays(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray) -> tuple[float, float]:
    if len(opens) == 0:
        return np.nan, np.nan
    if opens[0] >= closes[0]:
        low = lows[0]
        high = highs[0]
    else:
        low = lows[0]
        high = closes[0]
    increase = 0.0
    for pos in range(len(opens)):
        if lows[pos] <= low:
            high = closes[pos]
            low = lows[pos]
        elif highs[pos] > high:
            high = highs[pos]
        increase = high - low
    return increase, low


def precompute_return_pct_matrix(
        underlying: pd.DataFrame,
        period_list: list[int]) -> np.ndarray:
    n_bars = len(underlying)
    matrix = np.full((n_bars, len(period_list)), np.nan, dtype='float64')
    opens = underlying['open'].to_numpy(dtype='float64')
    highs = underlying['high'].to_numpy(dtype='float64')
    lows = underlying['low'].to_numpy(dtype='float64')
    closes = underlying['close'].to_numpy(dtype='float64')
    for col, period_n in enumerate(period_list):
        period_n = int(period_n)
        for end_pos in range(period_n - 1, n_bars):
            start_pos = end_pos - period_n + 1
            increase, base = _increase_with_base_from_arrays(
                opens[start_pos:end_pos + 1],
                highs[start_pos:end_pos + 1],
                lows[start_pos:end_pos + 1],
                closes[start_pos:end_pos + 1],
            )
            if base and pd.notna(base):
                matrix[end_pos, col] = increase / base
    return matrix


def precompute_peak_mask(
        return_pct_matrix: np.ndarray,
        skip_bars: int) -> np.ndarray:
    n_bars, n_periods = return_pct_matrix.shape
    peak_mask = np.zeros((n_bars, n_periods), dtype=bool)
    for row in range(int(skip_bars), n_bars):
        values = return_pct_matrix[row]
        for col in range(n_periods):
            value = values[col]
            if not np.isfinite(value):
                continue
            left_ok = True
            right_ok = True
            if col > 0:
                left = values[col - 1]
                left_ok = np.isfinite(left) and value >= left
            if col < n_periods - 1:
                right = values[col + 1]
                right_ok = np.isfinite(right) and value >= right
            peak_mask[row, col] = bool(left_ok and right_ok)
    return peak_mask


def precompute_daily_atr_pct(
        underlying: pd.DataFrame,
        lookback_days: int) -> np.ndarray:
    dates = pd.to_datetime(underlying['Date'], errors='coerce')
    if dates.isna().any():
        raise ValueError('Date column cannot be parsed for daily ATR.')
    work = underlying[['high', 'low', 'close']].copy()
    work['day'] = dates.dt.normalize()
    daily = work.groupby('day', sort=True).agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
    })
    prev_close = daily['close'].shift(1)
    tr_parts = pd.concat([
        daily['high'] - daily['low'],
        (daily['high'] - prev_close).abs(),
        (daily['low'] - prev_close).abs(),
    ], axis=1)
    daily['tr'] = tr_parts.max(axis=1)
    daily['atr_abs'] = (
        daily['tr']
        .rolling(int(lookback_days), min_periods=int(lookback_days))
        .mean()
        .shift(1)
    )
    daily['atr_pct'] = daily['atr_abs'] / prev_close
    day_to_atr = daily['atr_pct'].to_dict()
    return dates.dt.normalize().map(day_to_atr).to_numpy(dtype='float64')


def precompute_recent_mean_1d(
        return_pct_1d: np.ndarray,
        lookback: int) -> np.ndarray:
    series = pd.Series(return_pct_1d, dtype='float64')
    return (
        series.shift(1)
        .rolling(int(lookback), min_periods=int(lookback))
        .mean()
        .to_numpy(dtype='float64')
    )


def export_case_html_multi_period(
        underlying: pd.DataFrame,
        transactions_df: pd.DataFrame,
        result_root: str,
        param_tag: str,
        context_bars: int,
        round_precision: int) -> list[str]:
    if go is None or make_subplots is None:
        print('[HTML] plotly is not installed, skip case html export.')
        return []
    if transactions_df is None or len(transactions_df) == 0:
        return []
    trade_rows = transactions_df[
        transactions_df['Type'].isin(['long', 'sell'])
    ].sort_index()
    if len(trade_rows) < 2:
        return []
    cases_dir = os.path.join(result_root, 'cases')
    os.makedirs(cases_dir, exist_ok=True)
    paths = []
    open_row = None
    case_id = 0
    for entry_index, row in trade_rows.iterrows():
        if row['Type'] == 'long':
            open_row = (entry_index, row)
            continue
        if row['Type'] != 'sell' or open_row is None:
            continue
        open_index, long_row = open_row
        close_index = entry_index
        case_id += 1
        left = max(0, int(open_index) - int(context_bars))
        right = min(len(underlying) - 1, int(close_index) + int(context_bars))
        case_df = underlying.iloc[left:right + 1].copy()
        x_values = list(range(left, right + 1))
        fig_html = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.78, 0.22],
        )
        fig_html.add_trace(go.Candlestick(
            x=x_values,
            open=case_df['open'],
            high=case_df['high'],
            low=case_df['low'],
            close=case_df['close'],
            increasing_line_color='red',
            decreasing_line_color='green',
            name='K',
        ), row=1, col=1)
        fig_html.add_trace(go.Scatter(
            x=[int(open_index)],
            y=[float(long_row['Price'])],
            mode='markers',
            marker=dict(color='red', size=10),
            name='open',
            hovertemplate='open<br>bar=%{x}<br>price=%{y}<extra></extra>',
        ), row=1, col=1)
        close_color = sell_wd_color if int(row.get('Close_type', 0) or 0) == 1 else sell_speed_color
        fig_html.add_trace(go.Scatter(
            x=[int(close_index)],
            y=[float(row['Price'])],
            mode='markers',
            marker=dict(color=close_color, size=10),
            name='close',
            hovertemplate='close<br>bar=%{x}<br>price=%{y}<extra></extra>',
        ), row=1, col=1)
        fig_html.add_trace(go.Scatter(
            x=[int(open_index), int(close_index)],
            y=[float(long_row['Price']), float(row['Price'])],
            mode='lines',
            line=dict(color=accent_blue, width=2),
            name='trade',
            hoverinfo='skip',
        ), row=1, col=1)
        fig_html.update_layout(
            title=(
                f'{param_tag} case {case_id:03d} '
                + f'bar {int(open_index)} -> {int(close_index)}'
            ),
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            height=760,
            margin=dict(l=48, r=24, t=56, b=36),
        )
        fig_html.update_yaxes(
            title='price',
            tickformat=f'.{int(round_precision)}f',
            row=1,
            col=1,
        )
        fig_html.update_yaxes(visible=False, row=2, col=1)
        html_path = os.path.join(
            cases_dir,
            f'{param_tag}_case{case_id:03d}_bar{int(open_index)}.html',
        )
        html_text = fig_html.to_html(
            include_plotlyjs=True,
            full_html=True,
            config={'scrollZoom': True},
        )
        with open(html_path, 'w', encoding='utf-8') as fh:
            fh.write(html_text)
        paths.append(html_path)
        open_row = None
    return paths


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
    if html_crosshair_enabled:
        x_spike_cfg = {
            'showspikes': True,
            'spikemode': 'across',
            'spikesnap': 'cursor',
            'spikecolor': html_crosshair_color,
            'spikethickness': 1,
            'spikedash': 'solid'
        }
        y_spike_cfg = {
            'showspikes': True,
            'spikemode': 'across',
            'spikesnap': 'cursor',
            'spikecolor': html_crosshair_color,
            'spikethickness': 1,
            'spikedash': 'solid'
        }

    fig_html.add_trace(go.Scatter(
        x=detail_df.index,
        y=detail_df.capital,
        mode='lines',
        line=dict(width=1.2, color=accent_blue),
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
                    + 'low: ' + _safe_val(pref_data, 'low') + '<br>'
                    + 'hld_wd_per: ' + _safe_val(pref_data, 'hld_wd_per', 2) + '%' + '<br>'
                    + 'holding_inc: ' + _safe_val(pref_data, 'holding_inc', 2) + '<br>'
                    + 'max_inc: ' + _safe_val(pref_data, 'max_inc', 2) + '%' + '<br>'
                    + 'max_wd: ' + _safe_val(pref_data, 'max_wd', 2) + '%' + '<br>'
                    + 'close_wd_floor: ' + _safe_val(pref_data, 'close_wd_floor_per', 2) + '%' + '<br>'
                    + 'close_wd_dyn: ' + _safe_val(pref_data, 'close_wd_dyn_per', 2) + '%' + '<br>'
                    + 'close_wd_th: ' + _safe_val(pref_data, 'close_wd_th_per', 2) + '%' + '<br>'
                    + 'close_wd_th_abs: ' + _safe_val(pref_data, 'close_wd_th_abs', 4) + '<br>'
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
                marker=dict(color=sell_wd_color, size=4),
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
                    + 'low: ' + _safe_val(pref_data, 'low') + '<br>'
                    + 'hld_wd_per: ' + _safe_val(pref_data, 'hld_wd_per', 2) + '%' + '<br>'
                    + 'max_inc: ' + _safe_val(pref_data, 'max_inc', 2) + '%' + '<br>'
                    + 'max_wd: ' + _safe_val(pref_data, 'max_wd', 2) + '%' + '<br>'
                    + 'close_wd_floor: ' + _safe_val(pref_data, 'close_wd_floor_per', 2) + '%' + '<br>'
                    + 'close_wd_dyn: ' + _safe_val(pref_data, 'close_wd_dyn_per', 2) + '%' + '<br>'
                    + 'close_wd_th: ' + _safe_val(pref_data, 'close_wd_th_per', 2) + '%' + '<br>'
                    + 'close_wd_th_abs: ' + _safe_val(pref_data, 'close_wd_th_abs', 4) + '<br>'
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
                marker=dict(color=sell_speed_color, size=4),
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
            line=dict(color=accent_blue, width=2),
            name='trade_link',
            hoverinfo='skip'
        ))

    trade_count_annotation = []
    if html_show_trade_count_badge:
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

    html_dir = f'./result/{outcome_dir_name}/html'
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
        self.holding_increase_percent = np.nan
        self.HIGH_MATCH_EPS = 1e-10
        self.open_withdraw_reset_same_bar_count = 0
        self.period_n = int(params.get('period_n', params.get('open_bar', 1)))
        self.period_idx = int(params.get('period_idx', 0))
        self.daily_atr_array = params.get('daily_atr_array')
        self.recent_mean_array = params.get('recent_mean_array')
        self.return_pct_matrix = params.get('return_pct_matrix')
        self.peak_mask_matrix = params.get('peak_mask_matrix')

    def get_extra_columns(self) -> list:
        return [
            'withdrawal', 'wd_per', 'wd_signal',
            'increase', 'inc_per', 'inc_signal',
            'ana_inc', 'a_inc_per',
            'total_inc', 't_inc_per', 'total_inc_signal',
            'max_inc', 'max_wd',
            'holding_wd', 'hld_wd_per', 'holding_wd_signal',
            'close_wd_floor_per', 'close_wd_dyn_per',
            'close_wd_th_per', 'close_wd_th_abs',
            'holding_inc', 'speed_close_signal',
            'var0', 'period',
            'low_index', 'high_index',
            'low_date', 'low_price',
            'high_date', 'high_price',
            'last_index', 'new_opening_count',
            'period_n', 'dynamic_threshold',
            'daily_atr_pct', 'recent_mean',
            'return_pct', 'is_peak', 'is_new_high',
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

    def on_bar_idle(self, ctx: BarContext) -> OpenResult | None:
        quote = ctx.quote
        signal = ctx.signal
        index = ctx.index
        ii = ctx.integer_index
        p = self.params

        open_bar = int(p['open_bar'])
        close_bar = int(p['close_bar'])
        close_threshold = float(p['close_threshold'])
        open_continous_threshold2 = p['open_continous_threshold2']
        daily_atr_value = (
            float(self.daily_atr_array[ii])
            if self.daily_atr_array is not None and ii < len(self.daily_atr_array)
            else np.nan
        )
        recent_mean_value = (
            float(self.recent_mean_array[ii])
            if self.recent_mean_array is not None and ii < len(self.recent_mean_array)
            else np.nan
        )
        return_pct_value = (
            float(self.return_pct_matrix[ii, self.period_idx])
            if self.return_pct_matrix is not None
            and ii < self.return_pct_matrix.shape[0]
            else np.nan
        )
        is_peak = (
            bool(self.peak_mask_matrix[ii, self.period_idx])
            if self.peak_mask_matrix is not None
            and ii < self.peak_mask_matrix.shape[0]
            else False
        )
        if np.isfinite(daily_atr_value) and np.isfinite(recent_mean_value):
            dynamic_threshold = max(
                daily_atr_value * float(p['daily_atr_multiplier']),
                recent_mean_value * float(p['recent_mean_multiplier']),
            )
        else:
            dynamic_threshold = np.nan

        signal.at[index, 'period_n'] = self.period_n
        signal.at[index, 'daily_atr_pct'] = daily_atr_value
        signal.at[index, 'recent_mean'] = recent_mean_value
        signal.at[index, 'return_pct'] = return_pct_value
        signal.at[index, 'is_peak'] = 1 if is_peak else 0
        signal.at[index, 'dynamic_threshold'] = dynamic_threshold

        if not np.isfinite(dynamic_threshold):
            self.var0 = 0
            self.new_opening = True
            self.new_opening_count = 0
            self.first_cond1_price = 0
            self.analysis_increase = 0
            return None

        open_threshold = dynamic_threshold
        open_continous_threshold = dynamic_threshold
        if p.get('enable_wd_exit', False):
            open_withdrawal_threshold = dynamic_threshold * float(
                p.get('wd_exit_multiplier', 1.0))
            close_withdrawal_threshold = open_withdrawal_threshold
        else:
            open_withdrawal_threshold = 999.0
            close_withdrawal_threshold = 999.0

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

            current_high = float(quote.iloc[ii]['high'])
            is_new_high = current_high >= float(analysis_slice['high'].max())
            signal.at[index, 'is_new_high'] = 1 if is_new_high else 0

            cond1 = (
                is_new_high
                and is_peak
                and (inc_percent >= open_threshold)
            )
            signal.at[index, 'inc_signal'] = 1 if cond1 else 0

            if open_withdrawal_threshold == 0:
                cond2 = True
            else:
                cond2 = wd_percent < open_withdrawal_threshold
            signal.at[index, 'wd_signal'] = 1 if cond2 else 0

            if signal.at[index, 'wd_signal']:
                if signal.at[index, 'inc_signal']:
                    for i in reversed(range(self.last_index, ii + 1)):
                        low_index_slice = quote.iloc[i:ii + 1]
                        increase2 = get_increase(low_index_slice)
                        if np.isclose(increase2, increase,
                                      rtol=0.0, atol=self.HIGH_MATCH_EPS):
                            self.low_index = i
                            break
                    signal.at[index, 'low_index'] = self.low_index
                    signal.at[index, 'low_date'] = str(
                        signal.at[self.low_index, 'date'])
                    self.last_index = self.low_index
                    self.start_index = self.last_index
                    self.var0 = 1
            else:
                if inc_percent > open_continous_threshold:
                    self.open_withdraw_reset_same_bar_count += 1
                    print(str(index) + '满足开仓和满足回撤reset同时发生')
                self.new_opening = True
                self.new_opening_count = 1

        # --- 阶段 1: 赋值 new_opening_count ---
        if self.var0 == 1:
            self.new_opening_count = ii - self.low_index + 1
            signal.at[index, 'low_index'] = self.low_index
            signal.at[index, 'low_date'] = str(
                signal.at[self.low_index, 'date']).removesuffix('.0')
            signal.at[index, 'period'] = self.new_opening_count
            self.var0 = 2

        # --- 阶段 2: 判断持续涨幅 ---
        if self.var0 == 2:
            cond3_analysis_slice = quote.iloc[self.low_index:ii + 1]
            with_high, withdrawal = get_withdrawal(
                cond3_analysis_slice, close_withdrawal_threshold, ii)
            signal.at[index, 'withdrawal'] = withdrawal
            withdrawal_percent = withdrawal / with_high if with_high != 0 else 0
            total_increase, inc_base = get_increase_with_base(cond3_analysis_slice)

            if open_withdrawal_threshold == 0:
                low_price = float(quote.iloc[self.low_index]['low'])
                current_low = float(quote.iloc[ii]['low'])
                cond3 = (ii == self.low_index) or (current_low > low_price)
            else:
                cond3 = withdrawal_percent < open_withdrawal_threshold
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
                    if analysis_increase_percent < close_threshold:
                        self.var0 = 4

                total_increase_percent = (
                    total_increase / inc_base if inc_base != 0 else 0
                )
                signal.at[index, 'total_inc'] = total_increase
                signal.at[index, 't_inc_per'] = round(
                    total_increase_percent * 100, 4)
                self.first_cond1_price = inc_base

                if total_increase_percent >= open_continous_threshold:
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

            # 开仓信号
            if signal.at[index, 'total_inc_signal'] == 1:
                return OpenResult(
                    execution_price=round(
                        self.first_cond1_price * (1 + open_continous_threshold),
                        self.params['round_precision']),
                    low_index=self.low_index,
                    low_price=self.first_cond1_price,
                    start_index=self.start_index,
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

        close_bar = int(p['close_bar'])
        close_threshold = float(p['close_threshold'])
        if p.get('enable_wd_exit', False):
            close_withdrawal_threshold_floor = float(p['close_withdrawal_threshold'])
        else:
            close_withdrawal_threshold_floor = 999.0
        open_continous_threshold2 = p['open_continous_threshold2']

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
            if self.holding_increase_percent <= close_threshold:
                signal.at[index, 'speed_close_signal'] = 1

        # 回撤条件
        holding_high = float(holding_slice['high'].max())
        dynamic_close_withdrawal_threshold = np.nan
        if close_withdrawal_mode == 'legacy_low_to_high_2over3':
            low_price = float(quote.iloc[self.low_index]['low'])
            max_increase_abs = max(holding_high - low_price, 0.0)
            dynamic_withdrawal_abs = max_increase_abs * (2.0 / 3.0)
            dynamic_close_withdrawal_threshold = (
                dynamic_withdrawal_abs / holding_high if holding_high != 0 else 0.0
            )
            close_withdrawal_threshold = dynamic_close_withdrawal_threshold
        else:
            close_withdrawal_threshold = close_withdrawal_threshold_floor
        signal.at[index, 'close_wd_floor_per'] = round(
            close_withdrawal_threshold_floor * 100, 4)
        signal.at[index, 'close_wd_dyn_per'] = round(
            close_withdrawal_threshold * 100, 4)
        signal.at[index, 'close_wd_th_per'] = round(
            close_withdrawal_threshold * 100, 4)
        signal.at[index, 'close_wd_th_abs'] = round(
            holding_high * close_withdrawal_threshold,
            self.params['round_precision'],
        )

        with_high, holding_withdrawal = get_withdrawal(
            holding_slice, close_withdrawal_threshold, ii, switch0=True)
        holding_withdrawal_percent = (
            holding_withdrawal / with_high if with_high != 0 else 0)
        signal.at[index, 'holding_wd'] = holding_withdrawal
        signal.at[index, 'hld_wd_per'] = round(
            holding_withdrawal_percent * 100, 4)

        if (open_continous_threshold2 == 0
            or (window_ready
                and (self.holding_increase_percent
                     < open_continous_threshold2))):
            if holding_withdrawal_percent > close_withdrawal_threshold:
                signal.at[index, 'holding_wd_signal'] = 1
        else:
            if holding_withdrawal_percent > close_withdrawal_threshold:
                signal.at[index, 'holding_wd_signal'] = 1

        period = ii - self.holding_start_index + 1
        signal.at[index, 'high_price'] = holding_high

        # 回撤平仓
        if signal.at[index, 'holding_wd_signal'] == 1:
            exec_price = (holding_high
                          * (1 - close_withdrawal_threshold))
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

    folder_path = data_folder_path
    file_name = data_file_name

    native_df, ROUND_PRECISION, NATIVE_BAR_SECONDS = load_data(folder_path, file_name)
    run_mode = str(run_mode).strip().lower()
    if run_mode not in ('manual', 'grid'):
        raise ValueError("run_mode must be 'manual' or 'grid'.")
    data_selection_mode = str(data_selection_mode).strip().lower()
    if data_selection_mode not in ('index', 'date'):
        raise ValueError("data_selection_mode must be 'index' or 'date'.")

    if data_selection_mode == 'index':
        range_start_label = start_index
        range_end_label = end_index
        if end_index == 'latest':
            native_preview_df = native_df.iloc[int(start_index):].copy()
        else:
            native_preview_df = native_df.iloc[int(start_index):int(end_index)].copy()
        print(f'[Main] native index range: ({start_index}, {end_index})')
    else:
        native_dates = pd.to_datetime(native_df['Date'], errors='coerce')
        if native_dates.isna().all():
            raise ValueError('Date column cannot be parsed for date selection.')
        start_ts = parse_selection_datetime(start_date, is_end=False)
        end_text = str(end_date).strip()
        if end_text.lower() == 'latest':
            date_mask = native_dates >= start_ts
        else:
            end_ts = parse_selection_datetime(end_text, is_end=True)
            if end_ts < start_ts:
                raise ValueError('end_date must be >= start_date.')
            date_mask = native_dates.between(start_ts, end_ts, inclusive='both')
        native_preview_df = native_df.loc[date_mask].copy()
        range_start_label = start_date
        range_end_label = end_date
        print(f'[Main] native date range: {start_date} -> {end_date}')

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
            print('[Data] dropped incomplete initial resampled bar: ' + str(dropped_bar_date))
    else:
        preview_df = native_preview_df.copy()
        BAR_SECONDS = NATIVE_BAR_SECONDS
        print(f'[Data] using native period  |  bar period: {BAR_SECONDS}s')
    if len(preview_df) == 0:
        raise ValueError('No data remains after resampling the selected native range.')

    period_label = format_period_label(resample_rule, BAR_SECONDS)
    run_name = (
        f'period_{period_label} '
        + f'{make_safe_range_token(range_start_label)}-'
        + f'{make_safe_range_token(range_end_label)}'
    )

    result_root = os.path.abspath(os.path.join('.', 'result', outcome_dir_name))
    os.makedirs(result_root, exist_ok=True)
    for subdir in ('image', 'perf', 'trans', 'html', 'cases', 'outcome stats'):
        os.makedirs(os.path.join(result_root, subdir), exist_ok=True)

    summary_dir = os.path.abspath(os.path.join('.', 'result', f'stats {outcome_dir_name}'))
    os.makedirs(summary_dir, exist_ok=True)
    dashboard_outcome_stats_path = os.path.join(
        result_root,
        'outcome stats',
        'long_multi_period ' + run_name + ' outcome_stats.xlsx',
    )

    print(f'[Main] backtest time range: {preview_df.iloc[0]["Date"]} -> {preview_df.iloc[-1]["Date"]}')
    print(f'[Main] result root: {result_root}')

    df5 = preview_df.reset_index(drop=True).copy()
    underlying = df5.copy()
    if only_close:
        underlying.open = underlying.low = underlying.high = underlying.close

    if run_mode == 'manual':
        period_values = [int(MANUAL_PERIOD)]
        daily_atr_multiplier_values = [float(DAILY_ATR_MULTIPLIER)]
        recent_mean_multiplier_values = [float(RECENT_MEAN_MULTIPLIER)]
    else:
        period_values = [int(value) for value in PERIOD_LIST]
        if int(for_num_1) <= 0 or int(for_num_2) <= 0:
            raise ValueError('for_num_1 and for_num_2 must be positive.')
        daily_atr_multiplier_values = [
            round(float(DAILY_ATR_MULTIPLIER) + i * float(step1), 10)
            for i in range(int(for_num_1))
        ]
        recent_mean_multiplier_values = [
            round(float(RECENT_MEAN_MULTIPLIER) + i * float(step2), 10)
            for i in range(int(for_num_2))
        ]

    if min(period_values) <= 0:
        raise ValueError('All periods must be positive.')
    matrix_periods = sorted(set(int(value) for value in PERIOD_LIST).union(period_values))
    period_index_map = {period_n: idx for idx, period_n in enumerate(matrix_periods)}
    include_thresholds = (
        run_mode == 'grid'
        and (len(daily_atr_multiplier_values) > 1 or len(recent_mean_multiplier_values) > 1)
    )

    print('[Precompute] return_pct_matrix periods=' + str(matrix_periods))
    return_pct_matrix = precompute_return_pct_matrix(underlying, matrix_periods)
    peak_mask_matrix = precompute_peak_mask(return_pct_matrix, max(matrix_periods))
    daily_atr_array = precompute_daily_atr_pct(
        underlying,
        int(DAILY_ATR_LOOKBACK_DAYS),
    )
    print('[Precompute] done.')

    outcome_stats = pd.DataFrame()
    existing_param_tags = set()
    if run_mode == 'grid':
        outcome_stats = load_existing_outcome_stats(dashboard_outcome_stats_path)
        existing_param_tags = set(outcome_stats.index.astype(str).tolist())

    total_search_space = (
        len(period_values)
        * len(daily_atr_multiplier_values)
        * len(recent_mean_multiplier_values)
    )
    progress_marks = build_progress_marks(total_search_space)
    printed_progress_marks = set()
    executed_run_count = 0
    last_perf_name = ''
    last_detail_df = None
    last_transactions_df = None
    last_capital_outcome = capital
    last_save_name = ''
    all_case_paths = []

    for period_n in period_values:
        period_idx = period_index_map[int(period_n)]
        recent_mean_lookback = max(1, int(round(float(RECENT_MEAN_FACTOR) * int(period_n))))
        recent_mean_array = precompute_recent_mean_1d(
            return_pct_matrix[:, period_idx],
            recent_mean_lookback,
        )
        close_bar_value = max(1, int(round(float(SPEED_EXIT_FACTOR) * int(period_n))))

        for daily_atr_multiplier_value in daily_atr_multiplier_values:
            for recent_mean_multiplier_value in recent_mean_multiplier_values:
                param_tag = build_long_param_tag(
                    period_n,
                    daily_atr_multiplier_value,
                    recent_mean_multiplier_value,
                    include_thresholds=include_thresholds,
                )
                if run_mode == 'grid' and param_tag in existing_param_tags:
                    print('[Grid] skip existing param: ' + param_tag)
                    continue

                print(
                    f'[{run_mode.title()}] period={period_n} '
                    + f'daily_atr_multiplier={daily_atr_multiplier_value} '
                    + f'recent_mean_multiplier={recent_mean_multiplier_value}'
                )

                close_withdrawal_threshold_value = (
                    float(WD_EXIT_MULTIPLIER)
                    if ENABLE_WD_EXIT else 999.0
                )
                params = {
                    'period_n': int(period_n),
                    'period_idx': int(period_idx),
                    'open_bar': int(period_n),
                    'open_threshold': 0.0,
                    'open_continous_threshold': 0.0,
                    'open_withdrawal_threshold': 999.0,
                    'close_bar': int(close_bar_value),
                    'close_threshold': float(SPEED_EXIT_THRESHOLD),
                    'close_withdrawal_threshold': close_withdrawal_threshold_value,
                    'close_withdrawal_mode': close_withdrawal_mode,
                    'open_continous_threshold2': 0.0,
                    'close_withdrawal_threshold2': close_withdrawal_threshold_value,
                    'round_precision': ROUND_PRECISION,
                    'daily_atr_array': daily_atr_array,
                    'recent_mean_array': recent_mean_array,
                    'return_pct_matrix': return_pct_matrix,
                    'peak_mask_matrix': peak_mask_matrix,
                    'daily_atr_multiplier': float(daily_atr_multiplier_value),
                    'recent_mean_multiplier': float(recent_mean_multiplier_value),
                    'enable_wd_exit': bool(ENABLE_WD_EXIT),
                    'wd_exit_multiplier': float(WD_EXIT_MULTIPLIER),
                }

                strategy = MomentumStrategy(params)
                engine = BacktestEngine(
                    underlying,
                    strategy,
                    capital,
                    ROUND_PRECISION,
                    commision_percent,
                    show_progress=(run_mode != 'grid'),
                )
                df_signal, signal, close_counts = engine.run()
                withdrawal_close_count = int(close_counts.get(1, 0))
                speed_close_count = int(close_counts.get(2, 0))
                total_trade_count = withdrawal_close_count + speed_close_count

                performance, transactions_df = generate_performance(
                    underlying,
                    df_signal,
                    capital,
                    commision_percent,
                )
                if len(transactions_df) > 1:
                    Capital_outcome = round(
                        transactions_df[
                            transactions_df.Type != 'long'
                        ].Capital.iloc[-1],
                        2,
                    )
                else:
                    Capital_outcome = capital
                perf_outcome = performance.reset_index(drop=True)[['date', 'capital']]

                count_tag = str(withdrawal_close_count) + '+' + str(speed_close_count)
                result_tag = param_tag + ' ' + count_tag
                save_name = run_name + ' ' + result_tag
                print('total close count = ' + str(total_trade_count))
                print('withdrawal close count = ' + str(withdrawal_close_count))
                print('speed close count = ' + str(speed_close_count))
                print('profit: ' + str(round(performance.capital.iloc[-1], 2)))

                detail_df = pd.concat([signal, df5], axis=1, join='inner')
                detail_df = pd.concat(
                    [detail_df, perf_outcome.capital], axis=1, join='inner')
                detail_df.drop(
                    ['holding_signal', 'inc_signal', 'wd_signal',
                     'holding_wd_signal', 'total_inc_signal',
                     'speed_close_signal', 'have_holding'],
                    axis=1,
                    inplace=True,
                    errors='ignore',
                )
                detail_df.drop(
                    ['var0', 'low_index', 'high_index'],
                    axis=1,
                    inplace=True,
                    errors='ignore',
                )

                if export_interactive_html or run_mode == 'manual':
                    html_title = str(round(Capital_outcome, 2)) + ' ' + save_name
                    export_interactive_html_long(
                        file_name=file_name,
                        save_name=save_name,
                        title=html_title,
                        underlying1=underlying.reset_index(drop=True).copy(),
                        detail_df=detail_df,
                        transactions_df=transactions_df,
                        factor=underlying['open'].iloc[0],
                    )

                if EXPORT_CASE_HTML:
                    safe_case_tag = param_tag.replace(' ', '_')
                    case_paths = export_case_html_multi_period(
                        underlying=underlying,
                        transactions_df=transactions_df,
                        result_root=result_root,
                        param_tag=safe_case_tag,
                        context_bars=int(CASE_CONTEXT_BARS),
                        round_precision=ROUND_PRECISION,
                    )
                    all_case_paths.extend(case_paths)
                    if case_paths:
                        print('[HTML] saved case count: ' + str(len(case_paths)))

                perf_name = (
                    param_tag + ' ' + count_tag
                    + ' Long ' + run_name
                    + ' ' + str(Capital_outcome)
                    + ' perf.xlsx'
                )
                perf_path = os.path.join(result_root, 'perf', perf_name)
                with pd.ExcelWriter(perf_path, engine='xlsxwriter') as writer1:
                    detail_df.to_excel(writer1, sheet_name='stats')

                if len(transactions_df) != 0:
                    trans_path = os.path.join(
                        result_root,
                        'trans',
                        param_tag + ' ' + count_tag
                        + ' Long ' + run_name
                        + ' ' + str(Capital_outcome)
                        + ' trans.xlsx',
                    )
                    with pd.ExcelWriter(trans_path, engine='xlsxwriter') as writer2:
                        transactions_df.reset_index(drop=False).to_excel(
                            writer2,
                            sheet_name='stats',
                        )

                summary_metrics = build_summary_metrics(
                    perf_outcome,
                    transactions_df,
                    initial_capital=capital,
                )
                summary_row = {
                    'period': int(period_n),
                    'open_bar': int(period_n),
                    'close_bar': int(close_bar_value),
                    'daily_atr_lookback_days': int(DAILY_ATR_LOOKBACK_DAYS),
                    'daily_atr_multiplier': float(daily_atr_multiplier_value),
                    'recent_mean_factor': float(RECENT_MEAN_FACTOR),
                    'recent_mean_lookback': int(recent_mean_lookback),
                    'recent_mean_multiplier': float(recent_mean_multiplier_value),
                    'speed_exit_factor': float(SPEED_EXIT_FACTOR),
                    'speed_exit_threshold': float(SPEED_EXIT_THRESHOLD),
                    'enable_wd_exit': bool(ENABLE_WD_EXIT),
                    'wd_exit_multiplier': float(WD_EXIT_MULTIPLIER),
                    'withdrawal_close_count': withdrawal_close_count,
                    'wd_close_count': withdrawal_close_count,
                    'speed_close_count': speed_close_count,
                    'open_threshold': float(daily_atr_multiplier_value),
                    'open_continous_threshold': float(recent_mean_multiplier_value),
                    'close_threshold': float(SPEED_EXIT_THRESHOLD),
                    'close_withdrawal_threshold': close_withdrawal_threshold_value,
                    'open_withdraw_reset_same_bar_count': int(
                        getattr(strategy, 'open_withdraw_reset_same_bar_count', 0)
                    ),
                }
                for metric_name, metric_value in summary_metrics.items():
                    summary_row[metric_name] = metric_value
                summary_row['capital'] = summary_metrics['final_capital']
                summary_row['trade_num'] = summary_metrics['trade_num']
                summary_row['outcome_high'] = summary_metrics['outcome_high']
                summary_row['biggest_wd'] = summary_metrics['biggest_wd_abs']
                for metric_name, metric_value in summary_row.items():
                    outcome_stats.at[param_tag, metric_name] = metric_value

                existing_param_tags.add(str(param_tag))
                executed_run_count += 1
                print_search_progress(
                    executed_run_count,
                    total_search_space,
                    progress_marks,
                    printed_progress_marks,
                )
                flush_dashboard_outcome_stats(
                    dashboard_outcome_stats_path,
                    outcome_stats,
                )
                last_perf_name = perf_name
                last_detail_df = detail_df
                last_transactions_df = transactions_df
                last_capital_outcome = Capital_outcome
                last_save_name = save_name

    print("\ntime = --- %s seconds ---" % (time.time() - start_time))
    if len(outcome_stats) == 0:
        raise ValueError('No parameter combination was executed.')

    flush_dashboard_outcome_stats(
        dashboard_outcome_stats_path,
        outcome_stats,
    )
    export_outcome_stats = load_existing_outcome_stats(dashboard_outcome_stats_path)
    summary_base = os.path.join(
        summary_dir,
        ' ' + run_name + ' ' + str(len(export_outcome_stats)) + ' all outcome',
    )
    export_outcome_stats.to_excel(summary_base + '.xlsx')

    if (
        run_mode == 'manual'
        and SHOW_MANUAL_PLOT
        and last_detail_df is not None
        and last_transactions_df is not None
    ):
        fig2 = plt.figure(figsize=(18, 9))
        ax2 = fig2.add_axes([0.043, 0.055, 0.943, 0.9])
        underlying1 = underlying.reset_index(drop=True)
        factor = underlying1['open'][0]
        underlying_ratio = pd.DataFrame()
        underlying_ratio['Date'] = underlying1['Date']
        underlying_ratio[['open', 'high', 'low', 'close']] = underlying1[
            ['open', 'high', 'low', 'close']] / factor * 100
        candlestick2_ohlc(
            ax2,
            underlying_ratio.open,
            underlying_ratio.high,
            underlying_ratio.low,
            underlying_ratio.close,
            width=0.7,
            colorup='salmon',
            colordown='#2ca02c',
        )
        draw_long_gap_lines(ax2, underlying1)
        plt.title(str(round(last_capital_outcome, 2)) + ' ' + last_save_name)
        manual_image_path = os.path.join(
            result_root,
            'image',
            str(round(last_capital_outcome, 2))
            + ' ' + last_save_name
            + f' Long result.{result_image_ext}',
        )
        save_result_figure(fig2, manual_image_path, result_image_dpi)
        plt.show()

    print('[Output] outcome_stats=' + dashboard_outcome_stats_path)
    print('[Output] result_root=' + result_root)
    if all_case_paths:
        print('[Output] sample_cases=' + ' | '.join(all_case_paths[:3]))
