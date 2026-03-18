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
DATA_FOLDER_PATH = r"D:\Code\data\converted_15s\\"
DATA_FILE_NAME = "HISTDATA_COM_ASCII_XAGUSD_T202512_15s"

# 回测区间
start_index = 1000
end_index = 5000  # 或 'latest'
only_close = False

# 参数循环
for_num_1 = 1
for_num_2 = 1
for_num_3 = 1
step1 = 0.001
step2 = 0.001
step3 = 0.01

# 策略参数（分钟输入，自动换算 bars）
open_bar_minutes = 50.0
open_threshold_cfg = 0.0020
open_withdrawal_threshold_cfg = 0.0013
close_bar_minutes = open_bar_minutes
close_threshold_cfg = 0.001
open_continous_threshold_cfg = open_threshold_cfg
close_withdrawal_threshold_cfg = open_withdrawal_threshold_cfg

# 双策略参数（保留）
open_bar2_minutes = np.nan  # np.nan 表示不启用
open_threshold2_cfg = np.nan
open_continous_threshold2_cfg = 0.003
close_withdrawal_threshold2_cfg = 0.003

# 阈值模式:
# - 'fixed': 使用原固定比例
# - 'adaptive_directional_test': 使用测试版自适应方向波动基准
THRESHOLD_MODE = 'adaptive_directional_test'

# 测试版自适应方向波动基准:
# 当前 bar 为 t 时，只使用 (t - basis_span - basis_subwindow, t) 的历史 bar。
# 基准按每根 15s bar 独立重算，和是否持仓无关。
basis_span_minutes = 5 * open_bar_minutes
basis_subwindow_minutes = open_bar_minutes
BASIS_AGGREGATION_METHOD = 'mean'

# 当 THRESHOLD_MODE='adaptive_directional_test' 时，
# 以下参数不再表示固定阈值，而是各阈值对应的倍率。
OPEN_POS_MULTIPLIER = 1.0
OPEN_CONTINOUS_POS_MULTIPLIER = 1.0
CLOSE_SPEED_POS_MULTIPLIER = 1.0
OPEN_WD_NEG_MULTIPLIER = 1.0
CLOSE_WD_NEG_MULTIPLIER = 1.0

# 第一版仅保留最小阈值。默认值为 0，表示事实上不起作用。
OPEN_POS_MIN_THRESHOLD = 0.0
OPEN_CONTINOUS_POS_MIN_THRESHOLD = 0.0
CLOSE_SPEED_POS_MIN_THRESHOLD = 0.0
OPEN_WD_NEG_MIN_THRESHOLD = 0.0
CLOSE_WD_NEG_MIN_THRESHOLD = 0.0

COMMISION_PERCENT = 0.000
CAPITAL = 100.0
EXPORT_INTERACTIVE_HTML = True
EXPORT_STATS = True
ACCENT_BLUE = '#1F77B4'
SELL_WD_COLOR = 'green'
SELL_SPEED_COLOR = 'black'
HTML_CROSSHAIR_ENABLED = False
HTML_CROSSHAIR_COLOR = 'rgba(255, 120, 120, 0.45)'
HTML_SHOW_TRADE_COUNT_BADGE = True
# 静态图保存开关：默认不保存 PDF/PNG（保留 HTML 导出）
SAVE_STATIC_PLOT = False
# 当 SAVE_STATIC_PLOT=True 时决定保存为 PDF 或 PNG
SAVE_PLOT_AS_PDF = False


def minutes_to_bars(minutes: float, bar_seconds: int, name: str) -> int:
    raw = (minutes * 60.0) / bar_seconds
    bars = max(1, int(round(raw)))
    if not np.isclose(raw, bars):
        print(f'[Config] {name}: {minutes}min -> {raw:.4f} bars, rounded to {bars}')
    return bars


def minutes_to_bars_optional(minutes: float, bar_seconds: int, name: str):
    if pd.isna(minutes):
        return np.nan
    return minutes_to_bars(minutes, bar_seconds, name)


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


def get_decrease_with_base(df):
    """
    返回跌幅及其对应的真实基准 high。
    这是做多策略里反向基准的测试版本，用于估算回撤类阈值。
    """
    if df.empty:
        print('received empty dataframe at get_decrease function.')
        return np.nan, np.nan
    need_cols = ['open', 'high', 'low', 'close']
    if any(c not in df.columns for c in need_cols):
        return np.nan, np.nan
    if df[need_cols].isna().any().any():
        return np.nan, np.nan
    if df.iloc[0]['open'] <= df.iloc[0]['close']:
        high = df.iloc[0]['high']
        low = df.iloc[0]['low']
    else:
        high = df.iloc[0]['high']
        low = df.iloc[0]['close']
    decrease = 0
    for index, row in df.iterrows():
        if row['high'] >= high:
            low = row['close']
            high = row['high']
        elif row['low'] < low:
            low = row['low']
        decrease = high - low
    return decrease, high


def _get_increase_with_base_array(window):
    first_low, first_high, first_open, first_close = (
        window[0, 0], window[0, 1], window[0, 2], window[0, 3]
    )
    if first_open >= first_close:
        low = first_low
        high = first_high
    else:
        low = first_low
        high = first_close

    for j in range(1, len(window)):
        low_j, high_j, close_j = window[j, 0], window[j, 1], window[j, 3]
        if low_j <= low:
            high = close_j
            low = low_j
        elif high_j > high:
            high = high_j

    return high - low, low


def _get_increase_with_base_array_detail(window):
    first_low, first_high, first_open, first_close = (
        window[0, 0], window[0, 1], window[0, 2], window[0, 3]
    )
    if first_open >= first_close:
        low = first_low
        high = first_high
    else:
        low = first_low
        high = first_close
    low_idx = 0
    high_idx = 0

    for j in range(1, len(window)):
        low_j, high_j, close_j = window[j, 0], window[j, 1], window[j, 3]
        if low_j <= low:
            high = close_j
            low = low_j
            low_idx = j
            high_idx = j
        elif high_j > high:
            high = high_j
            high_idx = j

    return {
        'move': high - low,
        'base': low,
        'low_idx': low_idx,
        'high_idx': high_idx,
    }


def _get_decrease_with_base_array(window):
    first_low, first_high, first_open, first_close = (
        window[0, 0], window[0, 1], window[0, 2], window[0, 3]
    )
    if first_open <= first_close:
        high = first_high
        low = first_low
    else:
        high = first_high
        low = first_close

    for j in range(1, len(window)):
        low_j, high_j, close_j = window[j, 0], window[j, 1], window[j, 3]
        if high_j >= high:
            low = close_j
            high = high_j
        elif low_j < low:
            low = low_j

    return high - low, high


def _get_decrease_with_base_array_detail(window):
    first_low, first_high, first_open, first_close = (
        window[0, 0], window[0, 1], window[0, 2], window[0, 3]
    )
    if first_open <= first_close:
        high = first_high
        low = first_low
    else:
        high = first_high
        low = first_close
    high_idx = 0
    low_idx = 0

    for j in range(1, len(window)):
        low_j, high_j, close_j = window[j, 0], window[j, 1], window[j, 3]
        if high_j >= high:
            low = close_j
            high = high_j
            high_idx = j
            low_idx = j
        elif low_j < low:
            low = low_j
            low_idx = j

    return {
        'move': high - low,
        'base': high,
        'high_idx': high_idx,
        'low_idx': low_idx,
    }


def _aggregate_directional_basis(values: np.ndarray, method: str) -> float:
    if len(values) == 0:
        return np.nan
    if method == 'mean':
        return float(np.nanmean(values))
    if method == 'ewma':
        # Placeholder for future versions.
        pass
    if method == 'semivariance_rms':
        # Placeholder for future versions.
        pass
    raise NotImplementedError(
        f'Unsupported BASIS_AGGREGATION_METHOD={method!r}.'
    )


def precompute_long_adaptive_bases(
        quote: pd.DataFrame,
        bar_seconds: int,
        basis_span_minutes: float,
        basis_subwindow_minutes: float,
        aggregation_method: str = 'mean') -> pd.DataFrame:
    """
    预计算做多策略的测试版自适应方向波动基准。

    定义:
    - 当前 bar 起点为 t
    - 只使用 (t - basis_span - basis_subwindow, t) 的历史 bar
    - 在这段历史里按 1 个 bar 滑动子窗口
    - 正向基准: mean(get_increase_with_base(window) / base_low)
    - 反向基准: mean(get_decrease_with_base(window) / base_high)
    """
    quote = quote.copy()
    span_bars = minutes_to_bars(
        basis_span_minutes, bar_seconds, 'basis_span'
    )
    subwindow_bars = minutes_to_bars(
        basis_subwindow_minutes, bar_seconds, 'basis_subwindow'
    )
    required_history_bars = span_bars + subwindow_bars - 1

    dates = pd.to_datetime(quote['Date'], errors='coerce')
    if dates.notna().sum() >= 2:
        diffs = dates.diff().dt.total_seconds()
        irregular_mask = diffs.notna() & (~np.isclose(diffs, bar_seconds))
        irregular_count = int(irregular_mask.sum())
        if irregular_count > 0:
            print(
                '[AdaptiveBasis] warning: detected '
                f'{irregular_count} irregular bar gaps; '
                'test version continues by row order.'
            )

    arr = quote[['low', 'high', 'open', 'close']].to_numpy(dtype=float)
    n = arr.shape[0]
    window_count = n - subwindow_bars + 1
    pos_window_ratio = np.full(window_count, np.nan, dtype=float)
    neg_window_ratio = np.full(window_count, np.nan, dtype=float)

    for start in range(window_count):
        window = arr[start:start + subwindow_bars]
        increase, low_base = _get_increase_with_base_array(window)
        decrease, high_base = _get_decrease_with_base_array(window)
        if low_base != 0:
            pos_window_ratio[start] = increase / low_base
        if high_base != 0:
            neg_window_ratio[start] = decrease / high_base

    if aggregation_method == 'mean':
        pos_valid = ~np.isnan(pos_window_ratio)
        neg_valid = ~np.isnan(neg_window_ratio)
        pos_cumsum = np.concatenate((
            [0.0], np.cumsum(np.where(pos_valid, pos_window_ratio, 0.0))
        ))
        neg_cumsum = np.concatenate((
            [0.0], np.cumsum(np.where(neg_valid, neg_window_ratio, 0.0))
        ))
        pos_count_cumsum = np.concatenate(([0], np.cumsum(pos_valid.astype(int))))
        neg_count_cumsum = np.concatenate(([0], np.cumsum(neg_valid.astype(int))))
    else:
        pos_cumsum = neg_cumsum = pos_count_cumsum = neg_count_cumsum = None

    adaptive_pos_basis = np.full(n, np.nan, dtype=float)
    adaptive_neg_basis = np.full(n, np.nan, dtype=float)

    for ii in range(required_history_bars, n):
        start_lo = ii - required_history_bars
        start_hi = ii - subwindow_bars

        if aggregation_method == 'mean':
            pos_sum = pos_cumsum[start_hi + 1] - pos_cumsum[start_lo]
            pos_count = pos_count_cumsum[start_hi + 1] - pos_count_cumsum[start_lo]
            neg_sum = neg_cumsum[start_hi + 1] - neg_cumsum[start_lo]
            neg_count = neg_count_cumsum[start_hi + 1] - neg_count_cumsum[start_lo]

            if pos_count > 0:
                adaptive_pos_basis[ii] = pos_sum / pos_count
            if neg_count > 0:
                adaptive_neg_basis[ii] = neg_sum / neg_count
        else:
            adaptive_pos_basis[ii] = _aggregate_directional_basis(
                pos_window_ratio[start_lo:start_hi + 1], aggregation_method
            )
            adaptive_neg_basis[ii] = _aggregate_directional_basis(
                neg_window_ratio[start_lo:start_hi + 1], aggregation_method
            )

    quote['adaptive_pos_basis'] = adaptive_pos_basis
    quote['adaptive_neg_basis'] = adaptive_neg_basis
    quote['adaptive_basis_ready'] = (
        quote['adaptive_pos_basis'].notna() & quote['adaptive_neg_basis'].notna()
    ).astype(int)

    return quote


def export_first_long_basis_snapshot_excel(
        file_name: str,
        save_name: str,
        quote: pd.DataFrame,
        open_bar: int,
        bar_seconds: int,
        basis_span_minutes_cfg: float,
        basis_subwindow_minutes_cfg: float):
    """
    导出首个 basis 有效时刻的调试快照。

    sheets:
    - summary: 首个 basis 时刻与参数摘要
    - ohlc_open_bar: 当时过去 open_bar_minutes 的全部 OHLC
    - basis_windows: 当时 basis_span_minutes 内所有子窗口的正/反向波动明细
    """
    if 'adaptive_basis_ready' not in quote.columns:
        return

    ready_index = quote.index[quote['adaptive_basis_ready'] == 1]
    if len(ready_index) == 0:
        print('[BasisDebug] no ready basis row found, skip snapshot export.')
        return

    current_idx = int(ready_index[0])
    span_bars = minutes_to_bars(
        basis_span_minutes_cfg, bar_seconds, 'basis_span'
    )
    subwindow_bars = minutes_to_bars(
        basis_subwindow_minutes_cfg, bar_seconds, 'basis_subwindow'
    )
    required_history_bars = span_bars + subwindow_bars - 1

    open_bar_start = max(0, current_idx - open_bar)
    ohlc_slice = quote.iloc[open_bar_start:current_idx].copy().reset_index(drop=False)
    ohlc_slice = ohlc_slice.rename(columns={'index': 'source_index'})

    basis_start_lo = current_idx - required_history_bars
    basis_start_hi = current_idx - subwindow_bars
    arr = quote[['low', 'high', 'open', 'close']].to_numpy(dtype=float)
    basis_rows = []
    for start in range(basis_start_lo, basis_start_hi + 1):
        window = arr[start:start + subwindow_bars]
        pos_detail = _get_increase_with_base_array_detail(window)
        neg_detail = _get_decrease_with_base_array_detail(window)

        pos_ratio = (
            pos_detail['move'] / pos_detail['base']
            if pos_detail['base'] != 0 else np.nan
        )
        neg_ratio = (
            neg_detail['move'] / neg_detail['base']
            if neg_detail['base'] != 0 else np.nan
        )

        pos_low_abs = start + pos_detail['low_idx']
        pos_high_abs = start + pos_detail['high_idx']
        neg_high_abs = start + neg_detail['high_idx']
        neg_low_abs = start + neg_detail['low_idx']

        basis_rows.append({
            'window_start_index': start,
            'start_date': quote.at[start, 'Date'],
            'pos_low_index': pos_low_abs,
            'pos_low': quote.at[pos_low_abs, 'low'],
            'pos_high_index': pos_high_abs,
            'pos_high': quote.at[pos_high_abs, 'high'],
            'pos_move': pos_detail['move'],
            'pos_ratio': pos_ratio,
            'neg_high_index': neg_high_abs,
            'neg_high': quote.at[neg_high_abs, 'high'],
            'neg_low_index': neg_low_abs,
            'neg_low': quote.at[neg_low_abs, 'low'],
            'neg_move': neg_detail['move'],
            'neg_ratio': neg_ratio,
        })

    basis_windows_df = pd.DataFrame(basis_rows, columns=[
        'window_start_index',
        'start_date',
        'pos_low_index',
        'pos_low',
        'pos_high_index',
        'pos_high',
        'pos_move',
        'pos_ratio',
        'neg_high_index',
        'neg_high',
        'neg_low_index',
        'neg_low',
        'neg_move',
        'neg_ratio',
    ])
    summary_df = pd.DataFrame([{
        'first_basis_index': current_idx,
        'first_basis_date': quote.at[current_idx, 'Date'],
        'adaptive_pos_basis': quote.at[current_idx, 'adaptive_pos_basis'],
        'adaptive_neg_basis': quote.at[current_idx, 'adaptive_neg_basis'],
        'open_bar_bars': open_bar,
        'open_bar_minutes': open_bar * bar_seconds / 60.0,
        'basis_span_minutes': basis_span_minutes_cfg,
        'basis_subwindow_minutes': basis_subwindow_minutes_cfg,
        'basis_span_window_count': span_bars,
        'basis_subwindow_bars': subwindow_bars,
        'basis_required_history_bars': required_history_bars,
        'ohlc_open_bar_start_index': open_bar_start,
        'ohlc_open_bar_end_index': current_idx - 1,
        'basis_window_start_lo': basis_start_lo,
        'basis_window_start_hi': basis_start_hi,
        'basis_window_count': len(basis_windows_df),
    }])

    debug_name = 'stats ' + save_name + ' first basis snapshot.xlsx'
    debug_dir = './result/%s long outcome/stats excel/' % file_name
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, debug_name)

    def _cell_text(value) -> str:
        if pd.isna(value):
            return ''
        if isinstance(value, pd.Timestamp):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        return str(value)

    with pd.ExcelWriter(debug_path, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        ohlc_slice.to_excel(writer, sheet_name='ohlc_open_bar', index=False)
        basis_windows_df.to_excel(writer, sheet_name='basis_windows', index=False)

        workbook = writer.book
        center_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
        })
        header_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'bold': True,
        })

        for sheet_name, df in {
                'summary': summary_df,
                'ohlc_open_bar': ohlc_slice,
                'basis_windows': basis_windows_df,
        }.items():
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.set_default_row(20)

            for col_idx, column in enumerate(df.columns):
                worksheet.write(0, col_idx, column, header_format)
                max_len = len(str(column))
                if not df.empty:
                    max_len = max(
                        max_len,
                        max(len(_cell_text(value)) for value in df[column])
                    )
                worksheet.set_column(
                    col_idx,
                    col_idx,
                    min(max(max_len + 2, 12), 32),
                    center_format
                )

    print(f'[BasisDebug] saved first basis snapshot: {debug_path}')
    return debug_path


def build_long_param_tag(
        threshold_mode: str,
        open_bar: int,
        open_threshold: float,
        open_continous_threshold: float,
        open_withdrawal_threshold: float,
        close_bar: int,
        close_threshold: float,
        close_withdrawal_threshold: float,
        basis_span_minutes: float | None = None,
        basis_subwindow_minutes: float | None = None) -> str:
    if threshold_mode == 'adaptive_directional_test':
        return (
            f'adt bs{basis_span_minutes:g} bw{basis_subwindow_minutes:g}'
            + ' om' + str(round(open_bar, 4))
            + ' opm' + str(round(open_threshold, 4))
            + ' ocpm' + str(round(open_continous_threshold, 4))
            + ' owm' + str(round(open_withdrawal_threshold, 4))
            + ' cm' + str(round(close_bar, 4))
            + ' cpm' + str(round(close_threshold, 4))
            + ' cwm' + str(round(close_withdrawal_threshold, 4))
        )

    return (
        'om' + str(round(open_bar, 4))
        + ' o' + str(round(open_threshold, 4))
        + ' oc' + str(round(open_continous_threshold, 4))
        + ' ow' + str(round(open_withdrawal_threshold, 4))
        + ' cm' + str(round(close_bar, 4))
        + ' c' + str(round(close_threshold, 4))
        + ' cw' + str(round(close_withdrawal_threshold, 4))
    )


def plot_long_adaptive_threshold_chart(
        underlying1: pd.DataFrame,
        detail_df: pd.DataFrame,
        title: str):
    fig = plt.figure('adaptive_thresholds', figsize=(18, 9))
    left = 0.043
    width = 0.943
    bottom = 0.055
    height = 0.9
    rect_line = [left, bottom, width, height]
    ax_price = fig.add_axes(rect_line)
    ax_threshold = ax_price.twinx()

    factor = underlying1['open'].iloc[0]
    underlying_ratio = pd.DataFrame(index=underlying1.index)
    underlying_ratio[['open', 'high', 'low', 'close']] = (
        underlying1[['open', 'high', 'low', 'close']] / factor * 100
    )

    candlestick2_ohlc(
        ax_price,
        underlying_ratio.open,
        underlying_ratio.high,
        underlying_ratio.low,
        underlying_ratio.close,
        width=0.7,
        colorup='salmon',
        colordown='#2ca02c'
    )

    basis_series = [
        ('pos_basis', 'pos_basis_t', '#d62728'),
        ('neg_basis', 'neg_basis_t', '#1f77b4'),
    ]

    for col, label, color in basis_series:
        if col not in detail_df.columns:
            continue
        series = pd.to_numeric(detail_df[col], errors='coerce') * 100
        if series.notna().any():
            ax_threshold.plot(
                series.index,
                series,
                label=label,
                linewidth=1.15,
                color=color,
                alpha=0.9,
            )

    ax_price.set_title(title)
    ax_price.xaxis.set_major_locator(plt.MaxNLocator(12))
    ax_price.spines['top'].set_visible(False)
    ax_price.spines['right'].set_visible(False)
    ax_threshold.spines['top'].set_visible(False)
    ax_threshold.set_ylabel('directional basis mean (%)')
    ax_threshold.tick_params(axis='y', colors='black')

    handles_1, labels_1 = ax_price.get_legend_handles_labels()
    handles_2, labels_2 = ax_threshold.get_legend_handles_labels()
    if labels_1 or labels_2:
        fig.legend(handles_1 + handles_2, labels_1 + labels_2, loc='upper left')

    return fig, ax_price, ax_threshold


def export_interactive_html_long_basis(
        file_name: str,
        save_name: str,
        title: str,
        underlying1: pd.DataFrame,
        detail_df: pd.DataFrame,
        factor: float):
    if go is None or make_subplots is None:
        print('[HTML] plotly is not installed, skip basis html export.')
        return

    def _safe_pct(pref_data, key, digits=4):
        if isinstance(pref_data, pd.Series) and key in pref_data.index:
            val = pref_data[key]
        else:
            return 'nan'
        if pd.isna(val):
            return 'nan'
        try:
            return str(round(float(val) * 100, digits))
        except Exception:
            return str(val)

    def _safe_val(pref_data, key, digits=4):
        if isinstance(pref_data, pd.Series) and key in pref_data.index:
            val = pref_data[key]
        else:
            return 'nan'
        if pd.isna(val):
            return 'nan'
        try:
            return str(round(float(val), digits))
        except Exception:
            return str(val)

    def _date_text(raw):
        dt = str(raw)[:-3]
        if len(dt) > 5:
            return dt[:-5] + ' ' + dt[-5:]
        return dt

    def _price_text(value, digits=6):
        if pd.isna(value):
            return 'nan'
        try:
            return str(round(float(value), digits))
        except Exception:
            return str(value)

    fig_html = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.67, 0.33],
    )
    x_index = underlying1.index.to_numpy()
    x_min = int(x_index[0]) if len(x_index) > 0 else 0
    x_max = int(x_index[-1]) if len(x_index) > 0 else 1
    x_span = max(1, x_max - x_min + 1)
    x_left_pad = max(1, int(round(x_span * 0.006)))
    x_right_pad = max(1, int(round(x_span * 0.010)))

    # stats basis html 固定开启十字线，便于逐 bar 对照 basis 数值。
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
    y2_spike_cfg = y_spike_cfg.copy()

    candle_texts = []
    for idx in underlying1.index:
        row = underlying1.loc[idx]
        pref_data = detail_df.loc[idx] if idx in detail_df.index else pd.Series(dtype='object')
        candle_texts.append(
            _date_text(row['Date']) + '<br>'
            + 'open: ' + _price_text(row['open']) + '<br>'
            + 'high: ' + _price_text(row['high']) + '<br>'
            + 'low: ' + _price_text(row['low']) + '<br>'
            + 'close: ' + _price_text(row['close']) + '<br>'
            + 'pos_basis: ' + _safe_pct(pref_data, 'pos_basis') + '%' + '<br>'
            + 'neg_basis: ' + _safe_pct(pref_data, 'neg_basis') + '%' + '<br>'
            + 'basis_ready: ' + _safe_val(pref_data, 'basis_ready', 0) + '<br>'
            + 'index: ' + str(idx)
        )

    fig_html.add_trace(go.Candlestick(
        x=x_index,
        open=underlying1['open'] / factor * 100,
        high=underlying1['high'] / factor * 100,
        low=underlying1['low'] / factor * 100,
        close=underlying1['close'] / factor * 100,
        name='price',
        hovertext=candle_texts,
        hoverinfo='text',
        increasing=dict(
            line=dict(color='salmon', width=0.8),
            fillcolor='rgba(250, 128, 114, 0.28)'
        ),
        decreasing=dict(
            line=dict(color='#2ca02c', width=0.8),
            fillcolor='rgba(44, 160, 44, 0.28)'
        )
    ), row=1, col=1)

    basis_configs = [
        ('pos_basis', 'pos_basis_t', '#d62728'),
        ('neg_basis', 'neg_basis_t', '#1f77b4'),
    ]
    for col, label, color in basis_configs:
        if col not in detail_df.columns:
            continue
        series = pd.to_numeric(detail_df[col], errors='coerce') * 100
        line_texts = []
        for idx in detail_df.index:
            pref_data = detail_df.loc[idx]
            row = underlying1.loc[idx]
            line_texts.append(
                _date_text(row['Date']) + '<br>'
                + 'open: ' + _price_text(row['open']) + '<br>'
                + 'high: ' + _price_text(row['high']) + '<br>'
                + 'low: ' + _price_text(row['low']) + '<br>'
                + 'close: ' + _price_text(row['close']) + '<br>'
                + 'pos_basis: ' + _safe_pct(pref_data, 'pos_basis') + '%' + '<br>'
                + 'neg_basis: ' + _safe_pct(pref_data, 'neg_basis') + '%' + '<br>'
                + 'basis_ready: ' + _safe_val(pref_data, 'basis_ready', 0) + '<br>'
                + 'index: ' + str(idx)
            )
        fig_html.add_trace(go.Scatter(
            x=detail_df.index,
            y=series,
            mode='lines',
            name=label,
            line=dict(color=color, width=1.3),
            text=line_texts,
            hovertemplate='%{text}<extra></extra>'
        ), row=2, col=1)

    fig_html.update_layout(
        title=title,
        template='plotly_white',
        autosize=True,
        hovermode='closest',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
        margin=dict(l=42, r=42, t=45, b=45, pad=0),
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.70)',
            bordercolor='rgba(0, 0, 0, 0.45)',
            font=dict(color='black')
        )
    )
    fig_html.update_xaxes(
        title=None,
        tickfont=dict(size=10),
        showgrid=False,
        rangeslider=dict(visible=False),
        range=[x_min - x_left_pad, x_max + x_right_pad],
        autorange=False,
        showticklabels=False,
        row=1,
        col=1,
        **x_spike_cfg
    )
    fig_html.update_xaxes(
        title=None,
        tickfont=dict(size=10),
        showgrid=False,
        range=[x_min - x_left_pad, x_max + x_right_pad],
        autorange=False,
        row=2,
        col=1,
        **x_spike_cfg
    )
    fig_html.update_yaxes(
        title='price (base=100)',
        tickfont=dict(size=10),
        showgrid=False,
        row=1,
        col=1,
        **y_spike_cfg
    )
    fig_html.update_yaxes(
        title='basis mean (%)',
        tickfont=dict(size=10),
        showgrid=False,
        row=2,
        col=1,
        **y2_spike_cfg
    )

    html_dir = './result/%s long outcome/html' % file_name
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(
        html_dir,
        'stats ' + save_name + ' basis interactive.html'
    )
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
        'fill:rgba(255,255,255,0.70) !important;'
        'fill-opacity:0.70 !important;'
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
    print(f'[HTML] saved interactive basis chart: {html_path}')
    return html_path


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

    html_dir = './result/%s long outcome/html' % file_name
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

    @staticmethod
    def _apply_min_threshold(value, min_threshold):
        if pd.isna(value):
            return np.nan
        return max(float(value), float(min_threshold))

    def _resolve_thresholds(self, ctx: BarContext) -> dict:
        quote = ctx.quote
        signal = ctx.signal
        index = ctx.index
        p = self.params

        threshold_mode = p.get('threshold_mode', 'fixed')
        pos_basis = np.nan
        neg_basis = np.nan
        basis_ready = 1

        if threshold_mode == 'fixed':
            open_threshold = p['open_threshold']
            open_continous_threshold = p['open_continous_threshold']
            open_withdrawal_threshold = p['open_withdrawal_threshold']
            close_threshold = p['close_threshold']
            close_withdrawal_threshold = p['close_withdrawal_threshold']
        elif threshold_mode == 'adaptive_directional_test':
            pos_basis = (
                quote.at[index, 'adaptive_pos_basis']
                if 'adaptive_pos_basis' in quote.columns else np.nan
            )
            neg_basis = (
                quote.at[index, 'adaptive_neg_basis']
                if 'adaptive_neg_basis' in quote.columns else np.nan
            )
            basis_ready = int((not pd.isna(pos_basis)) and (not pd.isna(neg_basis)))

            open_threshold = self._apply_min_threshold(
                pos_basis * p['open_threshold'],
                p['open_min_threshold'],
            )
            open_continous_threshold = self._apply_min_threshold(
                pos_basis * p['open_continous_threshold'],
                p['open_cont_min_threshold'],
            )
            open_withdrawal_threshold = self._apply_min_threshold(
                neg_basis * p['open_withdrawal_threshold'],
                p['open_wd_min_threshold'],
            )
            close_threshold = self._apply_min_threshold(
                pos_basis * p['close_threshold'],
                p['close_min_threshold'],
            )
            close_withdrawal_threshold = self._apply_min_threshold(
                neg_basis * p['close_withdrawal_threshold'],
                p['close_wd_min_threshold'],
            )
        else:
            raise ValueError(f'Unsupported threshold_mode={threshold_mode!r}')

        signal.at[index, 'basis_ready'] = basis_ready
        signal.at[index, 'pos_basis'] = pos_basis
        signal.at[index, 'neg_basis'] = neg_basis
        signal.at[index, 'active_open_threshold'] = open_threshold
        signal.at[index, 'active_open_cont_threshold'] = open_continous_threshold
        signal.at[index, 'active_open_wd_threshold'] = open_withdrawal_threshold
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
        open_continous_threshold2 = p['open_continous_threshold2']
        thresholds = self._resolve_thresholds(ctx)
        open_threshold = thresholds['open_threshold']
        open_continous_threshold = thresholds['open_continous_threshold']
        open_withdrawal_threshold = thresholds['open_withdrawal_threshold']
        close_threshold = thresholds['close_threshold']
        close_withdrawal_threshold = thresholds['close_withdrawal_threshold']

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

            cond1 = (inc_percent >= open_threshold)
            signal.at[index, 'inc_signal'] = 1 if cond1 else 0

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

        close_bar = p['close_bar']
        open_continous_threshold2 = p['open_continous_threshold2']
        thresholds = self._resolve_thresholds(ctx)
        close_threshold = thresholds['close_threshold']
        close_withdrawal_threshold = thresholds['close_withdrawal_threshold']

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
            if self.holding_increase_percent < close_threshold:
                signal.at[index, 'speed_close_signal'] = 1

        # 回撤条件
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
        signal.at[index, 'high_price'] = max(holding_slice['high'])

        # 回撤平仓
        if signal.at[index, 'holding_wd_signal'] == 1:
            exec_price = (max(holding_slice['high'])
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

    # --- 数据加载 ---
    folder_path = DATA_FOLDER_PATH
    file_name = DATA_FILE_NAME

    df, ROUND_PRECISION, BAR_SECONDS = load_data(folder_path, file_name)
    open_bar_cfg = minutes_to_bars(open_bar_minutes, BAR_SECONDS, 'open_bar')
    close_bar_cfg = minutes_to_bars(close_bar_minutes, BAR_SECONDS, 'close_bar')
    open_bar2_cfg = minutes_to_bars_optional(
        open_bar2_minutes, BAR_SECONDS, 'open_bar2'
    )

    # 创建输出文件夹
    os.makedirs('./result', exist_ok=True)
    os.makedirs(f'./result/{file_name} long outcome/perf', exist_ok=True)
    os.makedirs(f'./result/{file_name} long outcome/trans', exist_ok=True)

    outcome_stats = pd.DataFrame()
    last_stats_html_path = None

    # 选择回测时间区间
    startdate = start_index
    enddate = end_index

    preview_df = df[df.index > startdate]
    if enddate != 'latest':
        preview_df = preview_df[preview_df.index < enddate]
    if len(preview_df) == 0:
        raise ValueError(
            f'No data in selected range: START_INDEX={startdate}, END_INDEX={enddate}'
        )
    print(f'[Main] backtest index range: ({startdate}, {enddate})')
    print(f'[Main] backtest time range: {preview_df.iloc[0]["Date"]} -> {preview_df.iloc[-1]["Date"]}')

    df5 = df[df.index > startdate]
    if enddate != 'latest':
        df5 = df5[df5.index < enddate].reset_index(drop=True)
    underlying = df5.copy()

    only_close_cfg = only_close
    if only_close_cfg:
        underlying.open = underlying.low = underlying.high = underlying.close

    threshold_mode = THRESHOLD_MODE
    basis_span_minutes_cfg = basis_span_minutes
    basis_subwindow_minutes_cfg = basis_subwindow_minutes
    print(f'[Main] threshold mode: {threshold_mode}')

    if threshold_mode == 'adaptive_directional_test':
        underlying = precompute_long_adaptive_bases(
            underlying,
            BAR_SECONDS,
            basis_span_minutes_cfg,
            basis_subwindow_minutes_cfg,
            aggregation_method=BASIS_AGGREGATION_METHOD,
        )

    # --- 参数循环 ---
    for_num_1_cfg = for_num_1
    for_num_2_cfg = for_num_2
    for_num_3_cfg = for_num_3
    print(for_num_1_cfg, for_num_2_cfg, for_num_3_cfg)
    step1_cfg = step1
    step2_cfg = step2
    step3_cfg = step3

    for num in range(for_num_1_cfg):
        for i in range(for_num_2_cfg):
            print(f'{str(num)} {str(i)}\n')

            # 策略参数
            open_bar = open_bar_cfg
            close_bar = close_bar_cfg
            if threshold_mode == 'adaptive_directional_test':
                open_threshold = OPEN_POS_MULTIPLIER
                open_withdrawal_threshold = OPEN_WD_NEG_MULTIPLIER
                close_threshold = CLOSE_SPEED_POS_MULTIPLIER
                open_continous_threshold = (
                    OPEN_CONTINOUS_POS_MULTIPLIER + (i * step1_cfg)
                )
                close_withdrawal_threshold = (
                    CLOSE_WD_NEG_MULTIPLIER + (num * step2_cfg)
                )
            else:
                open_threshold = open_threshold_cfg
                open_withdrawal_threshold = open_withdrawal_threshold_cfg
                close_threshold = close_threshold_cfg
                open_continous_threshold = open_continous_threshold_cfg + (i * step1_cfg)
                close_withdrawal_threshold = (
                    close_withdrawal_threshold_cfg + (num * step2_cfg)
                )
            # 双策略
            open_bar2 = open_bar2_cfg
            open_threshold2_runtime = open_threshold2_cfg
            open_continous_threshold2_runtime = open_continous_threshold2_cfg
            close_withdrawal_threshold2_runtime = close_withdrawal_threshold2_cfg + (num * step3_cfg)
            commision_percent = COMMISION_PERCENT
            capital = CAPITAL

            # 参数校验
            if threshold_mode == 'adaptive_directional_test':
                if min(
                    open_threshold,
                    open_withdrawal_threshold,
                    close_threshold,
                    open_continous_threshold,
                    close_withdrawal_threshold,
                    OPEN_POS_MIN_THRESHOLD,
                    OPEN_CONTINOUS_POS_MIN_THRESHOLD,
                    CLOSE_SPEED_POS_MIN_THRESHOLD,
                    OPEN_WD_NEG_MIN_THRESHOLD,
                    CLOSE_WD_NEG_MIN_THRESHOLD,
                ) < 0:
                    print('adaptive multiplier/min threshold不可为负数')
                    continue
                if open_continous_threshold < open_threshold:
                    print('open_continous_threshold multiplier不可小于open_threshold multiplier')
                    continue
            else:
                if open_threshold < open_withdrawal_threshold:
                    print('open_threshold不可小于open_withdrawal_threshold')
                    continue
                if open_continous_threshold < open_threshold:
                    print('open_continous_threshold不可小于open_threshold')
                    continue
                if open_continous_threshold < close_withdrawal_threshold:
                    print('open_continous_threshold不可小于close_withdrawal_threshold')
                    continue

            # Window_Increase 预计算
            arr = underlying[['low', 'high', 'open', 'close']].to_numpy(dtype=float)
            n = arr.shape[0]
            win = open_bar
            window_increase = np.full(n, np.nan, dtype=float)
            for end in range(win - 1, n):
                start = end - win + 1
                w = arr[start:end + 1]
                win_low = w[0, 0]
                win_high = w[0, 1] if w[0, 2] >= w[0, 3] else w[0, 3]
                for j in range(1, win):
                    low_j, high_j, close_j = w[j, 0], w[j, 1], w[j, 3]
                    if low_j <= win_low:
                        win_low = low_j
                        win_high = close_j
                    elif high_j > win_high:
                        win_high = high_j
                window_increase[end] = win_high - win_low
            underlying['Window_Increase'] = window_increase

            # ====== 使用引擎运行回测 ======
            params = {
                'open_bar': open_bar,
                'threshold_mode': threshold_mode,
                'open_threshold': open_threshold,
                'open_continous_threshold': open_continous_threshold,
                'open_withdrawal_threshold': open_withdrawal_threshold,
                'open_min_threshold': OPEN_POS_MIN_THRESHOLD,
                'open_cont_min_threshold': OPEN_CONTINOUS_POS_MIN_THRESHOLD,
                'open_wd_min_threshold': OPEN_WD_NEG_MIN_THRESHOLD,
                'close_bar': close_bar,
                'close_threshold': close_threshold,
                'close_withdrawal_threshold': close_withdrawal_threshold,
                'close_min_threshold': CLOSE_SPEED_POS_MIN_THRESHOLD,
                'close_wd_min_threshold': CLOSE_WD_NEG_MIN_THRESHOLD,
                'open_continous_threshold2': open_continous_threshold2_runtime,
                'close_withdrawal_threshold2': close_withdrawal_threshold2_runtime,
                'round_precision': ROUND_PRECISION,
            }

            strategy = MomentumStrategy(params)
            engine = BacktestEngine(
                underlying, strategy, capital,
                ROUND_PRECISION, commision_percent)
            (df_signal, signal, close_counts) = engine.run()
            withdrawal_close_count = close_counts.get(1, 0)
            speed_close_count = close_counts.get(2, 0)

            performance, transactions_df = generate_performance(
                underlying, df_signal, capital, commision_percent)

            if len(transactions_df) > 1:
                Capital_outcome = round(
                    transactions_df[
                        transactions_df.Type != 'long'].Capital.iloc[-1], 2)
            else:
                Capital_outcome = 100
            perf_outcome = performance.reset_index(
                drop=True)[['date', 'capital']]
            param_tag = build_long_param_tag(
                threshold_mode,
                open_bar,
                open_threshold,
                open_continous_threshold,
                open_withdrawal_threshold,
                close_bar,
                close_threshold,
                close_withdrawal_threshold,
                basis_span_minutes=(
                    basis_span_minutes_cfg
                    if threshold_mode == 'adaptive_directional_test' else None
                ),
                basis_subwindow_minutes=(
                    basis_subwindow_minutes_cfg
                    if threshold_mode == 'adaptive_directional_test' else None
                ),
            )
            count_tag = (
                str(round(withdrawal_close_count, 4))
                + '+' + str(round(speed_close_count, 4))
            )
            result_tag = param_tag + ' ' + count_tag

            # 打印结果
            print(str(startdate) + '-' + str(enddate))
            print('total close count = '
                  + str(withdrawal_close_count + speed_close_count))
            print('withdrawal close count = '
                  + str(round(withdrawal_close_count, 4)))
            print('speed close count = '
                  + str(round(speed_close_count, 4)))
            print(result_tag)
            print('profit: ' + str(round(performance.capital.iloc[-1], 2)))

            # ====== Plot (fig1) ======
            save_name = str(startdate) + '-' + str(enddate) + ' ' + result_tag

            if (
                    threshold_mode == 'adaptive_directional_test'
                    and EXPORT_STATS
            ):
                export_first_long_basis_snapshot_excel(
                    file_name=file_name,
                    save_name=save_name,
                    quote=underlying,
                    open_bar=open_bar,
                    bar_seconds=BAR_SECONDS,
                    basis_span_minutes_cfg=basis_span_minutes_cfg,
                    basis_subwindow_minutes_cfg=basis_subwindow_minutes_cfg,
                )

            fig1_title = str(Capital_outcome) + ' ' + save_name
            if SAVE_STATIC_PLOT:
                plot_ext = 'pdf' if SAVE_PLOT_AS_PDF else 'png'
                fig1_path = ('./result/%s long outcome/' % file_name
                             + ' ' + str(Capital_outcome)
                             + save_name + f' Long.{plot_ext}')
                close_fig = (for_num_2_cfg != 1) or (len(transactions_df) == 0)
                plot_backtest_chart(
                    underlying, transactions_df, perf_outcome,
                    title=fig1_title,
                    save_path=fig1_path,
                    close_fig=close_fig)

            # ====== Perf & Excel ======
            detail_df = pd.concat([signal, df5], axis=1, join='inner')
            detail_df = pd.concat(
                [detail_df, perf_outcome.capital], axis=1, join='inner')
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

            if threshold_mode == 'adaptive_directional_test' and EXPORT_STATS:
                underlying1_stats = underlying.reset_index(drop=True)
                factor_stats = underlying1_stats['open'].iloc[0]
                last_stats_html_path = export_interactive_html_long_basis(
                    file_name=file_name,
                    save_name=save_name,
                    title='adaptive thresholds ' + str(Capital_outcome) + ' ' + save_name,
                    underlying1=underlying1_stats,
                    detail_df=detail_df,
                    factor=factor_stats,
                )

            perf_name = (
                param_tag + ' ' + count_tag
                + ' ' + 'Long ' + str(startdate) + '-' + str(enddate)
                + ' ' + str(Capital_outcome)
                + ' ' + 'perf.xlsx'
            )
            writer1 = pd.ExcelWriter(
                './result/%s long outcome/perf/' % file_name + perf_name,
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
                    './result/%s long outcome/trans/' % file_name
                    + param_tag + ' ' + count_tag + ' '
                    + 'Long ' + str(startdate) + '-' + str(enddate)
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

            # Stats
            outcome_index = param_tag
            perf_temp = perf_outcome[-1:].capital.iloc[0] - 100
            outcome_stats.at[outcome_index, 'capital'] = perf_temp + 100
            trade_num = len(transactions_df) / 2
            outcome_stats.at[outcome_index, 'trade_num'] = trade_num
            outcome_high, outcome_wd = get_outcome_withdrawal(
                perf_outcome.capital)
            outcome_stats.at[outcome_index, 'outcome_high'] = outcome_high
            outcome_stats.at[outcome_index, 'biggest_wd'] = outcome_wd

    print("\ntime = --- %s seconds ---" % (time.time() - start_time))

    # 多参数对比图
    if for_num_2_cfg > 1:
        fig_stat_1 = plt.figure('stats', figsize=(18, 9))
        left = 0.033
        width = 0.943
        bottom = 0.055
        height = 0.9
        rect_line = [left, bottom, width, height]
        ax_stat_1 = fig_stat_1.add_axes(rect_line)
        ax_stat_1.plot(outcome_stats.capital, label='capital')
        ax_stat_2 = ax_stat_1.twinx()
        ax_stat_2.plot(outcome_stats.biggest_wd, color='orange',
                       label='biggest wd')
        ax_stat_3 = ax_stat_1.twinx()
        ax_stat_3.plot(outcome_stats.trade_num, color='salmon',
                       label='trade num')
        ax_stat_3.tick_params(axis='y', colors='red')
        fig_stat_1.show()
        ax_stat_1.xaxis.set_major_locator(plt.MaxNLocator(12))
        plt.xticks(rotation=70)
        fig_stat_1.legend()
        plt.title('stats ' + str(startdate) + '-' + str(enddate))
        os.makedirs('./result/stats %s long outcome/' % file_name, exist_ok=True)
        stats_plot_ext = 'pdf' if SAVE_PLOT_AS_PDF else 'png'
        plt.savefig('./result/stats %s long outcome/' % file_name
                    + ' ' + save_name + ' '
                    + str(for_num_1_cfg) + ' '
                    + str(for_num_2_cfg) + ' '
                    + f'all outcome.{stats_plot_ext}', dpi=1000)
        outcome_stats.to_excel('./result/stats %s long outcome/' % file_name
                               + ' ' + save_name + ' '
                               + str(for_num_1_cfg) + ' '
                               + str(for_num_2_cfg) + ' '
                               + 'all outcome.xlsx')
    else:
        disk_path = './result/'
        open_excel = False
        if open_excel:
            os.startfile(
                disk_path + '%s long outcome/perf/' % file_name + perf_name)

    # ====== 交互式图 (fig2) ======
    if for_num_2_cfg == 1:
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
        x = underlying_ratio['close']
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
        if len(sell_record) != 0:
            close_type_1_df = sell_record[sell_record['Close_type'] == 1]
            scatter_g = ax2.scatter(
                close_type_1_df.index,
                close_type_1_df['target'], c=SELL_WD_COLOR, s=10)
            close_type_2_df = sell_record[sell_record['Close_type'] == 2]
            scatter_b = ax2.scatter(
                close_type_2_df.index,
                close_type_2_df['target'], c=SELL_SPEED_COLOR, s=10)

        # 交互: 买点 hover
        if len(long_record) != 0:
            annot_r = ax2.annotate(
                "", xy=(0, 0), xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"))
            annot_r.set_visible(False)

            def update_annot_r(ind):
                index_num = ind["ind"][0]
                pos = scatter_r.get_offsets()[index_num]
                annot_r.xy = pos
                trade_data = long_record.iloc[index_num]
                index0 = trade_data.name
                date = str(trade_data['Date'])[:-3]
                pref_data = detail_df.loc[index0]
                high = pref_data.high
                t_inc_per = round(pref_data['t_inc_per'], 2)
                execution = pref_data['execution']
                low_date = pref_data['low_date']
                new_opening_count = pref_data['new_opening_count']
                low_price = pref_data['low_price']
                text = (date[:-5] + ' ' + date[-5:] + '\n'
                        + 'high: ' + str(high) + '\n'
                        + 'total_inc: ' + str(t_inc_per) + '%' + '\n'
                        + 'execution: ' + str(execution) + '\n'
                        + 'low_date: ' + str(low_date) + '\n'
                        + 'low_price: ' + str(low_price) + '\n'
                        + 'new_opening_count: '
                        + str(new_opening_count)[:-2] + '\n'
                        + 'index: ' + str(index0) + '\n')
                annot_r.set_text(text)
                annot_r.get_bbox_patch().set_alpha(0.4)

            def hover_r(event):
                vis = annot_r.get_visible()
                if event.inaxes == ax2:
                    cont, ind = scatter_r.contains(event)
                    if cont:
                        update_annot_r(ind)
                        annot_r.set_visible(True)
                        fig2.canvas.draw_idle()
                    else:
                        if vis:
                            annot_r.set_visible(False)
                            fig2.canvas.draw_idle()
            fig2.canvas.mpl_connect("motion_notify_event", hover_r)

            annot_g = ax2.annotate(
                "", xy=(0, 0), xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"))
            annot_g.set_visible(False)

        # 交互: 回撤平仓点 hover
        if len(sell_record) != 0:
            def update_annot_g(ind):
                index_num = ind["ind"][0]
                pos = scatter_g.get_offsets()[index_num]
                annot_g.xy = pos
                trade_data = close_type_1_df.iloc[index_num]
                index0 = trade_data.name
                date = str(trade_data['Date'])[:-3]
                pref_data = detail_df.loc[index0]
                low = pref_data.low
                hld_wd_per = round(pref_data['hld_wd_per'], 2)
                holding_inc = round(pref_data['holding_inc'], 2)
                max_inc = round(pref_data['max_inc'], 2)
                max_wd = round(pref_data['max_wd'], 2)
                execution = pref_data['execution']
                low_date = pref_data['low_date']
                high_date = pref_data['high_date']
                high_price = pref_data['high_price']
                period = pref_data['period']
                text = (date[:-5] + ' ' + date[-5:] + '\n'
                        + 'low: ' + str(low) + '\n'
                        + 'hld_wd_per: ' + str(hld_wd_per) + '%' + '\n'
                        + 'holding_inc: ' + str(holding_inc) + '\n'
                        + 'max_inc: ' + str(max_inc) + '%' + '\n'
                        + 'max_wd: ' + str(max_wd) + '%' + '\n'
                        + 'execution2: ' + str(execution) + '\n'
                        + 'period: ' + str(period) + '\n'
                        + 'low_date: ' + str(low_date) + '\n'
                        + 'high_date: ' + str(high_date) + '\n'
                        + 'high_price: ' + str(high_price) + '\n'
                        + 'index: ' + str(index0))
                annot_g.set_text(text)
                annot_g.get_bbox_patch().set_alpha(0.4)

        if len(sell_record) != 0:
            def hover_g(event):
                vis = annot_g.get_visible()
                if event.inaxes == ax2:
                    cont, ind = scatter_g.contains(event)
                    if cont:
                        update_annot_g(ind)
                        annot_g.set_visible(True)
                        fig2.canvas.draw_idle()
                    else:
                        if vis:
                            annot_g.set_visible(False)
                            fig2.canvas.draw_idle()
            fig2.canvas.mpl_connect("motion_notify_event", hover_g)

            annot_b = ax2.annotate(
                "", xy=(0, 0), xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"))
            annot_b.set_visible(False)

            # 交互: 速度平仓点 hover
            def update_annot_b(ind):
                index_num = ind["ind"][0]
                pos = scatter_b.get_offsets()[index_num]
                annot_b.xy = pos
                trade_data = close_type_2_df.iloc[index_num]
                index0 = trade_data.name
                date = str(trade_data['Date'])[:-3]
                pref_data = detail_df.loc[index0]
                low = pref_data.low
                hld_wd_per = round(pref_data['hld_wd_per'], 2)
                max_inc = round(pref_data['max_inc'], 2)
                max_wd = round(pref_data['max_wd'], 2)
                execution = pref_data['execution']
                low_date = pref_data['low_date']
                high_date = pref_data['high_date']
                high_price = pref_data['high_price']
                period = pref_data['period']
                text = (date[:-5] + ' ' + date[-5:] + '\n'
                        + 'low: ' + str(low) + '\n'
                        + 'hld_wd_per: ' + str(hld_wd_per) + '%' + '\n'
                        + 'max_inc: ' + str(max_inc) + '%' + '\n'
                        + 'max_wd: ' + str(max_wd) + '%' + '\n'
                        + 'execution2: ' + str(execution) + '\n'
                        + 'period: ' + str(period) + '\n'
                        + 'low_date: ' + str(low_date) + '\n'
                        + 'high_date: ' + str(high_date) + '\n'
                        + 'high_price: ' + str(high_price) + '\n'
                        + 'index: ' + str(index0))
                annot_b.set_text(text)
                annot_b.get_bbox_patch().set_alpha(0.4)

            def hover_b(event):
                vis = annot_b.get_visible()
                if event.inaxes == ax2:
                    cont, ind = scatter_b.contains(event)
                    if cont:
                        update_annot_b(ind)
                        annot_b.set_visible(True)
                        fig2.canvas.draw_idle()
                    else:
                        if vis:
                            annot_b.set_visible(False)
                            fig2.canvas.draw_idle()
            fig2.canvas.mpl_connect("motion_notify_event", hover_b)

        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.xticks(rotation=0)
        fig2_title = (
            ' ' + str(round(Capital_outcome, 2))
            + ' ' + result_tag
            + ' ' + str(startdate) + '-' + str(enddate)
        )
        plt.title('%s' % fig2_title)

        xaxis1 = detail_df.index
        yaxis1 = detail_df.capital
        xaxis2 = x.index
        yaxis2 = x
        plt.plot(xaxis1, yaxis1, linewidth=1.2, color=ACCENT_BLUE)
        candlestick2_ohlc(ax2, underlying_ratio.open, underlying_ratio.high,
                          underlying_ratio.low, underlying_ratio.close,
                          width=0.7,
                          colorup='salmon', colordown='#2ca02c')

        # 蓝线连接买卖点 (fig2)
        trade_seq = transactions_df[
            transactions_df['Type'].isin(['long', 'sell'])].copy()
        trade_seq = trade_seq.sort_index()
        trade_seq['target'] = trade_seq['Price'] / factor * 100
        buy_idx = None
        buy_y = None
        for idx, row in trade_seq.iterrows():
            if row['Type'] == 'long':
                buy_idx = idx
                buy_y = row['target']
            elif row['Type'] == 'sell' and buy_idx is not None:
                sell_idx = idx
                sell_y = row['target']
                ax2.plot(
                    [buy_idx, sell_idx], [buy_y, sell_y],
                    color=ACCENT_BLUE, linewidth=2.0, alpha=0.8, zorder=1)
                buy_idx = None
                buy_y = None
        ax2.xaxis.set_major_locator(plt.MaxNLocator(12))
        if EXPORT_INTERACTIVE_HTML:
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

    if EXPORT_STATS and last_stats_html_path:
        try:
            if hasattr(os, 'startfile'):
                os.startfile(last_stats_html_path)
            else:
                print(f'[Stats] generated stats html: {last_stats_html_path}')
        except OSError as exc:
            print(f'[Stats] failed to open stats html: {exc}')
