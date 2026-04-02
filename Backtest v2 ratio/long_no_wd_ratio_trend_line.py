# -*- coding: utf-8 -*-
"""
Long No-WD Strategy - 无回撤做多策略
=====================================
策略入口脚本：包含 LongNoWDStrategy 类、参数循环、绘图、Excel 输出。
依赖 backtest_main.py 中的通用框架。
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
import matplotlib.ticker as ticker
from mplfinance.original_flavor import candlestick2_ohlc
import time, os
import json
import socket
import subprocess
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

import sys, os as _os
from urllib.parse import urlparse
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
DATA_FOLDER_PATH = r"F:\Data\XAGUSD\\"
DATA_FILE_NAME = "xagusd_30s_all"

# 回测区间：
# START_INDEX / END_INDEX 按当前实际回测周期解释。
# 采用切片语义 [START_INDEX, END_INDEX)。
# 例如 RESAMPLE_RULE = '30min' 时，0~150 表示前 150 根 30 分钟 bar。
START_INDEX = 5000
END_INDEX = 10000  # 或 'latest'
ONLY_CLOSE = False

# 重采样设置：设为 '' 表示直接使用当前原始周期
# 例如 '1min' / '5min' / '15min' / '1H'
RESAMPLE_RULE = '1H'
TREND_W_MIN_BARS = 1
TREND_W_MAX_BARS = 10
DEBUG_TREND_SEARCH = False
DEBUG_RECORD_FROM_INDEX = None
# DEBUG_TREND_SEARCH = True
# DEBUG_RECORD_FROM_INDEX = 6865
DEBUG_RECORD_SEARCH_START = None

# 参数循环
FOR_NUM_1 = 1
FOR_NUM_2 = 1
FOR_NUM_3 = 1
STEP1 = 0.001
STEP3 = 0.01

# 策略参数直接使用 bar 数，单位是当前实际回测周期。
# 例如：
# 1. RESAMPLE_RULE = '' 时，OPEN_BAR = 60 表示原始数据的 60 根 bar
# 2. RESAMPLE_RULE = '5min' 时，OPEN_BAR = 6 表示 6 根 5 分钟 bar
OPEN_BAR = TREND_W_MAX_BARS
OPEN_THRESHOLD = 0.0001
CLOSE_BAR = OPEN_BAR
CLOSE_THRESHOLD = 0.001
OPEN_CONTINOUS_THRESHOLD = 0.0

# 双策略参数（保留）
OPEN_BAR2 = np.nan  # np.nan 表示不启用
OPEN_THRESHOLD2 = np.nan
OPEN_CONTINOUS_THRESHOLD2 = 0.003


COMMISION_PERCENT = 0.000
CAPITAL = 100.0

TREND_MIN_PROGRESS_SAMPLES = 3
TREND_IMPROVEMENT_CAPTURE_TARGET = 0.85
TREND_TEST_MODE = True
TREND_TEST_CASE_INDEX = 0
TREND_TEST_CASE_COUNT = 10
TREND_TEST_WINDOWS = [5, 7, 15]
TREND_TEST_NEAR_BAND = 1
TREND_TEST_BIG_DROP_RATIO = 2.0
# 仅用于柱图可视化：把 segment_withdrawal=0 的柱子显示为最小高度（不改原始数据）
ZERO_BAR_VISUAL_FLOOR_PCT = 0.0001
EXPORT_INTERACTIVE_HTML = True
ACCENT_BLUE = '#1F77B4'
SELL_WD_COLOR = 'green'
SELL_SPEED_COLOR = 'black'
HTML_CROSSHAIR_ENABLED = False
HTML_CROSSHAIR_COLOR = 'rgba(255, 120, 120, 0.45)'
HTML_SHOW_TRADE_COUNT_BADGE = False
SAVE_STATIC_PLOT = False
# 当 SAVE_STATIC_PLOT=True 时决定保存为 PDF 或 PNG
SAVE_PLOT_AS_PDF = False
SHOW_MATPLOTLIB_PLOTS = False
AUTO_OPEN_TREND_HTML = True
AUTO_OPEN_DASHBOARD = True
DASHBOARD_URL = 'http://127.0.0.1:8765'
BACKTEST_HTML_FOLDER = 'backtest html'
TREND_ANALYSIS_HTML_FOLDER = 'trend analysis html'
TREND_TEST_CASE_HTML_FOLDER = 'trend test case html'
TREND_MULTIPLE_FOLDER = 'trend multiple'
TREND_HTML_DEFAULT_MULTIPLE = 5.0
TREND_HTML_MIN_VISIBLE_MULTIPLE = 1.5
def detect_bar_seconds_from_df(df: pd.DataFrame) -> int:
    dates = pd.to_datetime(df['Date'], errors='coerce')
    diffs = dates.diff().dropna()
    if len(diffs) > 50:
        diffs = diffs.iloc[:50]
    median_delta = diffs.median()
    if pd.isna(median_delta):
        raise ValueError('无法识别当前数据周期：Date 列无法计算相邻时间差。')
    total_seconds = int(median_delta.total_seconds())
    if total_seconds <= 0:
        raise ValueError(f'识别到非法周期秒数：{total_seconds}')
    return total_seconds


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


def build_candlestick_hovertext(df: pd.DataFrame, factor: float):
    dates = df['Date'].astype(str).to_numpy()
    opens = (df['open'] / factor * 100).astype(float).to_numpy()
    highs = (df['high'] / factor * 100).astype(float).to_numpy()
    lows = (df['low'] / factor * 100).astype(float).to_numpy()
    closes = (df['close'] / factor * 100).astype(float).to_numpy()
    index_values = df.index.to_numpy()
    texts = []
    for idx, date_text, open_v, high_v, low_v, close_v in zip(
        index_values, dates, opens, highs, lows, closes
    ):
        texts.append(
            f'bar_index: {int(idx)}<br>'
            f'time: {date_text}<br>'
            f'open: {open_v:.4f}<br>'
            f'high: {high_v:.4f}<br>'
            f'low: {low_v:.4f}<br>'
            f'close: {close_v:.4f}'
        )
    return texts


def get_html_output_dir(file_name: str, folder_name: str) -> str:
    return './result/%s long no wd outcome/%s' % (file_name, folder_name)


def should_record_trend_debug(search_start: int, end_idx: int) -> bool:
    if not DEBUG_TREND_SEARCH:
        return False
    if DEBUG_RECORD_SEARCH_START is not None and int(search_start) != int(DEBUG_RECORD_SEARCH_START):
        return False
    if DEBUG_RECORD_FROM_INDEX is not None and int(end_idx) < int(DEBUG_RECORD_FROM_INDEX):
        return False
    return True


def is_tcp_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_dashboard_server_running(url: str,
                                    wait_timeout: float = 6.0) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or '127.0.0.1'
    port = parsed.port or 80
    if is_tcp_port_open(host, port):
        return True

    dashboard_script = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), '..', 'dashboard', 'dashboard.py')
    )
    if not _os.path.exists(dashboard_script):
        print(f'[Dashboard] script not found: {dashboard_script}')
        return False

    creationflags = 0
    if os.name == 'nt':
        creationflags = getattr(subprocess, 'DETACHED_PROCESS', 0) | getattr(
            subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)

    try:
        subprocess.Popen(
            [sys.executable, dashboard_script],
            cwd=_os.path.dirname(dashboard_script),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception as exc:
        print(f'[Dashboard] failed to start server: {exc}')
        return False

    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        if is_tcp_port_open(host, port):
            return True
        time.sleep(0.2)

    print(f'[Dashboard] server did not become ready: {url}')
    return False


def build_trend_atr_multiple_df(quote: pd.DataFrame,
                                trend_df: pd.DataFrame):
    if len(trend_df) == 0:
        return pd.DataFrame(columns=[
            'trade_id',
            'pre_atr',
            'pre_atr_pct',
            'trend_atr_multiple',
        ])

    prev_close = quote['close'].shift(1)
    true_range = np.maximum.reduce([
        (quote['high'] - quote['low']).to_numpy(dtype=float),
        (quote['high'] - prev_close).abs().to_numpy(dtype=float),
        (quote['low'] - prev_close).abs().to_numpy(dtype=float),
    ])
    tr_series = pd.Series(true_range, index=quote.index, dtype=float)

    records = []
    for _, row in trend_df.iterrows():
        low_index = int(row['low_index'])
        atr_window = (
            int(row['constraint_w_bars'])
            if 'constraint_w_bars' in row.index and pd.notna(row['constraint_w_bars'])
            else int(TREND_W_MAX_BARS)
        )
        pre_end = low_index - 1
        pre_start = low_index - atr_window

        pre_atr = np.nan
        pre_atr_pct = np.nan
        if pre_start >= 0 and pre_end >= pre_start:
            pre_tr = tr_series.iloc[pre_start:pre_end + 1]
            pre_close = quote['close'].iloc[pre_start:pre_end + 1]
            if len(pre_tr) == atr_window and len(pre_close) == atr_window:
                pre_atr = float(pre_tr.mean())
                pre_close_mean = float(pre_close.mean())
                if np.isfinite(pre_close_mean) and pre_close_mean > 0:
                    pre_atr_pct = pre_atr / pre_close_mean * 100.0

        trend_return_pct = (
            float(row['total_return_pct'])
            if pd.notna(row['total_return_pct']) else np.nan
        )
        trend_atr_multiple = (
            trend_return_pct / pre_atr_pct
            if np.isfinite(pre_atr_pct) and pre_atr_pct > 0 and np.isfinite(trend_return_pct)
            else np.nan
        )
        records.append({
            'trade_id': int(row['trade_id']),
            'pre_atr': pre_atr,
            'pre_atr_pct': pre_atr_pct,
            'trend_atr_multiple': trend_atr_multiple,
        })

    return pd.DataFrame(records)


def build_filtered_trend_case_df(quote: pd.DataFrame,
                                 trend_df: pd.DataFrame,
                                 multiple_threshold: float):
    if len(trend_df) == 0:
        return trend_df.copy()

    case_df = trend_df.copy()
    atr_df = build_trend_atr_multiple_df(quote, case_df)
    if len(atr_df) > 0:
        case_df = case_df.merge(
            atr_df[['trade_id', 'trend_atr_multiple']],
            on='trade_id',
            how='left'
        )
    else:
        case_df['trend_atr_multiple'] = np.nan

    threshold = float(multiple_threshold)
    multiple_series = pd.to_numeric(
        case_df['trend_atr_multiple'],
        errors='coerce'
    )
    case_df = case_df[multiple_series >= threshold].copy()
    return case_df.reset_index(drop=True)


def build_trend_multiple_summary_df(quote: pd.DataFrame,
                                    trend_df: pd.DataFrame,
                                    bar_seconds: int):
    if len(trend_df) == 0:
        return pd.DataFrame(columns=[
            'multiple_rank',
            'trade_id',
            'trend_atr_multiple',
            'duration_bars',
            'duration_minutes',
            'segment_max_drawdown_pct',
            'low_date',
            'high_date',
            'end_date',
            'total_return_pct',
            'pre_atr_pct',
        ])

    summary_df = trend_df.copy()
    atr_df = build_trend_atr_multiple_df(quote, summary_df)
    if len(atr_df) > 0:
        summary_df = summary_df.merge(
            atr_df,
            on='trade_id',
            how='left'
        )
    else:
        summary_df['pre_atr'] = np.nan
        summary_df['pre_atr_pct'] = np.nan
        summary_df['trend_atr_multiple'] = np.nan

    summary_df['trend_atr_multiple'] = pd.to_numeric(
        summary_df['trend_atr_multiple'],
        errors='coerce'
    )
    summary_df = summary_df[summary_df['trend_atr_multiple'].notna()].copy()
    if len(summary_df) == 0:
        return pd.DataFrame(columns=[
            'multiple_rank',
            'trade_id',
            'trend_atr_multiple',
            'duration_bars',
            'duration_minutes',
            'segment_max_drawdown_pct',
            'low_date',
            'high_date',
            'end_date',
            'total_return_pct',
            'pre_atr_pct',
        ])

    summary_df['duration_bars'] = pd.to_numeric(
        summary_df['duration_bars'],
        errors='coerce'
    )
    if int(bar_seconds) > 0:
        summary_df['duration_minutes'] = (
            summary_df['duration_bars'] * float(bar_seconds) / 60.0
        )
    else:
        summary_df['duration_minutes'] = np.nan

    segment_max_drawdown_pct_values = []
    for _, row in summary_df.iterrows():
        low_index = int(row['low_index'])
        high_index = int(row['high_index'])
        seg = quote.iloc[low_index:high_index + 1]
        if len(seg) == 0:
            segment_max_drawdown_pct_values.append(np.nan)
            continue
        max_wd = get_max_wd(seg)
        if pd.notna(max_wd):
            segment_max_drawdown_pct_values.append(float(max_wd) * 100.0)
        else:
            segment_max_drawdown_pct_values.append(np.nan)
    summary_df['segment_max_drawdown_pct'] = segment_max_drawdown_pct_values

    summary_df = summary_df.sort_values(
        ['trend_atr_multiple', 'duration_bars', 'trade_id'],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    summary_df.insert(0, 'multiple_rank', np.arange(1, len(summary_df) + 1))

    selected_cols = [
        'multiple_rank',
        'trade_id',
        'trend_atr_multiple',
        'duration_bars',
        'duration_minutes',
        'segment_max_drawdown_pct',
        'low_date',
        'high_date',
        'end_date',
        'total_return_pct',
        'pre_atr_pct',
        'low_index',
        'high_index',
        'end_index',
    ]
    return summary_df[selected_cols].copy()


def export_trend_multiple_ranked_html(file_name: str,
                                      save_name: str,
                                      trend_multiple_df: pd.DataFrame):
    if go is None:
        print('[HTML] plotly is not installed, skip trend multiple html export.')
        return
    if len(trend_multiple_df) == 0:
        print('[HTML] no trend multiple data, skip trend multiple html export.')
        return

    plot_df = trend_multiple_df.copy()
    hover_text = []
    drawdown_hover_text = []
    for _, row in plot_df.iterrows():
        duration_minutes = (
            f"{float(row['duration_minutes']):.2f}"
            if pd.notna(row['duration_minutes']) else 'nan'
        )
        hover_text.append(
            f"rank: {int(row['multiple_rank'])}<br>"
            f"segment: {int(row['trade_id'])}<br>"
            f"trend_multiple: {float(row['trend_atr_multiple']):.4f}<br>"
            f"duration_bars: {int(row['duration_bars'])}<br>"
            f"duration_minutes: {duration_minutes}<br>"
            f"segment_max_drawdown_pct: {float(row['segment_max_drawdown_pct']):.4f}%<br>"
            f"low_time: {row['low_date']}<br>"
            f"high_time: {row['high_date']}<br>"
            f"end_time: {row['end_date']}<br>"
            f"total_return_pct: {float(row['total_return_pct']):.4f}%"
        )
        drawdown_hover_text.append(
            f"rank: {int(row['multiple_rank'])}<br>"
            f"segment: {int(row['trade_id'])}<br>"
            f"segment_max_drawdown_pct: {float(row['segment_max_drawdown_pct']):.4f}%<br>"
            f"trend_multiple: {float(row['trend_atr_multiple']):.4f}<br>"
            f"duration_bars: {int(row['duration_bars'])}<br>"
            f"low_time: {row['low_date']}<br>"
            f"high_time: {row['high_date']}"
        )

    fig_html = go.Figure()
    fig_html.add_trace(go.Bar(
        x=plot_df['multiple_rank'].astype(int).tolist(),
        y=plot_df['trend_atr_multiple'].astype(float).tolist(),
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        marker=dict(
            color=ACCENT_BLUE,
        ),
        name='trend_multiple',
    ))
    fig_html.add_trace(go.Bar(
        x=plot_df['multiple_rank'].astype(int).tolist(),
        y=plot_df['segment_max_drawdown_pct'].astype(float).tolist(),
        hovertext=drawdown_hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        marker=dict(
            color='rgba(255, 99, 71, 0.24)',
        ),
        name='segment_max_drawdown_pct',
    ))

    fig_html.update_layout(
        title='Trend Multiple Ranked Segments',
        template='plotly_white',
        autosize=True,
        hovermode='closest',
        barmode='overlay',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.01,
            xanchor='left',
            x=0,
        ),
        xaxis=dict(
            title='rank by trend_multiple',
            tickfont=dict(size=10),
            showgrid=False,
        ),
        yaxis=dict(
            title='trend_multiple',
            tickfont=dict(size=10),
            showgrid=False,
        ),
        margin=dict(l=42, r=25, t=58, b=45, pad=0),
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.50)',
            bordercolor='rgba(0, 0, 0, 0.45)',
            font=dict(color='black')
        )
    )

    html_dir = get_html_output_dir(file_name, TREND_MULTIPLE_FOLDER)
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(
        html_dir, save_name + ' trend_multiple_ranked interactive.html')
    html_text = fig_html.to_html(
        include_plotlyjs=True, full_html=True,
        default_width='100vw', default_height='100vh',
        config={'responsive': True, 'displayModeBar': False,
                'displaylogo': False}
    )
    html_text = html_text.replace(
        '<head>',
        '<head><style>'
        'html,body{width:100%;height:100%;margin:0;padding:0;overflow:hidden;}'
        '.plotly-graph-div{width:100vw !important;height:100vh !important;}'
        '.hoverlayer .hovertext .bg,'
        '.hoverlayer .hovertext rect,'
        '.hoverlayer .hovertext path{'
        'fill:rgba(255,255,255,0.50) !important;'
        'fill-opacity:0.50 !important;'
        'stroke:rgba(0,0,0,0.45) !important;'
        'stroke-opacity:0.45 !important;}'
        '.hoverlayer .hovertext{opacity:1 !important;}'
        '.hoverlayer .hovertext text{fill:#000 !important;}'
        '</style>',
        1
    )
    html_text = html_text.replace(
        '<body>', '<body style="margin:0;overflow:hidden;">', 1)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print('[HTML] saved trend multiple chart.')


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


def get_withdrawal(df):
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


# 每笔交易统计：最大收益、最大收益前最大亏损
def build_trade_extreme_stats_long(quote: pd.DataFrame,
                                   transactions_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    tr = transactions_df[transactions_df['Type'].isin(['long', 'sell'])].sort_index()
    current_entry = None

    for idx, row in tr.iterrows():
        if row['Type'] == 'long':
            current_entry = (int(idx), row)
            continue

        if row['Type'] == 'sell' and current_entry is not None:
            entry_idx, entry_row = current_entry
            exit_idx = int(idx)
            if exit_idx < entry_idx:
                current_entry = None
                continue

            entry_price = float(entry_row['Price'])
            exit_price = float(row['Price'])
            trade_slice = quote.iloc[entry_idx:exit_idx + 1].copy()
            if len(trade_slice) == 0:
                current_entry = None
                continue

            max_profit_bar_idx = int(trade_slice['high'].idxmax())
            max_profit_price = float(quote.loc[max_profit_bar_idx, 'high'])
            max_profit_pct = (max_profit_price / entry_price - 1.0) * 100.0

            pre_slice = quote.iloc[entry_idx:max_profit_bar_idx + 1].copy()
            max_loss_bar_idx = int(pre_slice['low'].idxmin())
            max_loss_price = float(quote.loc[max_loss_bar_idx, 'low'])
            max_loss_before_max_profit_pct = (max_loss_price / entry_price - 1.0) * 100.0

            realized_pct = (exit_price / entry_price - 1.0) * 100.0
            holding_bars = exit_idx - entry_idx + 1

            records.append({
                'entry_index': entry_idx,
                'entry_date': quote.loc[entry_idx, 'Date'],
                'entry_price': entry_price,
                'exit_index': exit_idx,
                'exit_date': quote.loc[exit_idx, 'Date'],
                'exit_price': exit_price,
                'holding_bars': holding_bars,
                'realized_pct': realized_pct,
                'max_profit_pct': max_profit_pct,
                'max_profit_index': max_profit_bar_idx,
                'max_profit_date': quote.loc[max_profit_bar_idx, 'Date'],
                'max_profit_price': max_profit_price,
                'max_loss_before_max_profit_pct': max_loss_before_max_profit_pct,
                'max_loss_before_max_profit_index': max_loss_bar_idx,
                'max_loss_before_max_profit_date': quote.loc[max_loss_bar_idx, 'Date'],
                'max_loss_before_max_profit_price': max_loss_price,
            })

            current_entry = None

    return pd.DataFrame(records)


def build_entry_to_max_profit_withdrawal_df(
        quote: pd.DataFrame,
        transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    逐笔统计：从开仓到「该笔最大盈利点」这段区间的最大回撤比例。
    回撤比例口径 = withdrawal / with_high（%）。
    """
    records = []
    tr = transactions_df[transactions_df['Type'].isin(['long', 'sell'])].sort_index()
    current_entry = None
    trade_id = 0

    for idx, row in tr.iterrows():
        if row['Type'] == 'long':
            current_entry = (int(idx), row)
            continue

        if row['Type'] == 'sell' and current_entry is not None:
            trade_id += 1
            entry_idx, entry_row = current_entry
            exit_idx = int(idx)
            if exit_idx < entry_idx:
                current_entry = None
                continue

            trade_slice = quote.iloc[entry_idx:exit_idx + 1].copy()
            if len(trade_slice) == 0:
                current_entry = None
                continue

            # 该笔交易的最大盈利点（long口径：最高点）
            max_profit_idx = int(trade_slice['high'].idxmax())
            entry_price = float(entry_row['Price'])
            max_profit_price = float(quote.loc[max_profit_idx, 'high'])
            max_profit_pct = (max_profit_price / entry_price - 1.0) * 100.0

            # 开仓 -> 最大盈利点
            seg_slice = quote.iloc[entry_idx:max_profit_idx + 1].copy()
            if len(seg_slice) == 0:
                current_entry = None
                continue

            with_high, withdrawal = get_withdrawal(seg_slice)
            max_withdrawal_to_max_profit_pct = (
                (withdrawal / with_high) * 100.0
                if (pd.notna(with_high) and with_high != 0)
                else np.nan
            )

            records.append({
                'trade_id': trade_id,
                'entry_index': entry_idx,
                'entry_date': quote.loc[entry_idx, 'Date'],
                'entry_price': entry_price,
                'max_profit_index': max_profit_idx,
                'max_profit_date': quote.loc[max_profit_idx, 'Date'],
                'max_profit_pct': max_profit_pct,
                'segment_bars': max_profit_idx - entry_idx + 1,
                'segment_with_high': with_high,
                'segment_withdrawal': withdrawal,
                'max_withdrawal_to_max_profit_pct': max_withdrawal_to_max_profit_pct,
            })

            current_entry = None

    return pd.DataFrame(records)


def legacy_compute_trend_stats(prices: np.ndarray):
    """
    对一段价格序列做 OLS 拟合 + ACF 最优时间窗口搜索。

    Parameters
    ----------
    prices : np.ndarray
        close 价格序列（从 low 到 high 的区间）。

    Returns
    -------
    dict with keys:
        ols_slope, ols_intercept, ols_r_squared,
        optimal_w_bars, optimal_w_cv
    """
    n = len(prices)
    result = {
        'ols_slope': np.nan,
        'ols_intercept': np.nan,
        'ols_r_squared': np.nan,
        'optimal_w_bars': np.nan,
        'optimal_w_cv': np.nan,
    }
    if n < 4:
        return result

    # OLS: P(t) = a*t + b
    t = np.arange(n, dtype=float)
    coeffs = np.polyfit(t, prices, 1)
    slope, intercept = coeffs[0], coeffs[1]
    fitted = slope * t + intercept
    ss_res = np.sum((prices - fitted) ** 2)
    ss_tot = np.sum((prices - np.mean(prices)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    result['ols_slope'] = slope
    result['ols_intercept'] = intercept
    result['ols_r_squared'] = r_squared

    # 残差
    residuals = prices - fitted

    # ACF 搜索范围: [2, n//2]
    w_min = 2
    w_max = n // 2
    if w_max < w_min:
        return result

    # 计算残差 ACF
    r_var = np.var(residuals)
    if r_var < 1e-20:
        # 残差几乎为零 -> 完美线性趋势，任何 W 都可以
        result['optimal_w_bars'] = w_min
        result['optimal_w_cv'] = 0.0
        return result

    r_mean = np.mean(residuals)
    acf_values = np.zeros(w_max + 1)
    for lag in range(w_min, w_max + 1):
        cov = np.mean(
            (residuals[:n - lag] - r_mean) * (residuals[lag:] - r_mean)
        )
        acf_values[lag] = cov / r_var

    # 找 ACF 在 [w_min, w_max] 内的第一个局部峰值
    best_w = np.nan
    for lag in range(w_min + 1, w_max):
        if (acf_values[lag] > acf_values[lag - 1]
                and acf_values[lag] >= acf_values[lag + 1]):
            best_w = lag
            break

    # 如果没有局部峰值，选 ACF 最大的 lag
    if np.isnan(best_w):
        best_w = w_min + int(np.argmax(acf_values[w_min:w_max + 1]))

    result['optimal_w_bars'] = best_w

    # 计算 optimal_w_bars 下的 CV
    w = int(best_w)
    drops = np.array([
        (prices[i] - prices[i + w]) / prices[i]
        if prices[i] != 0 else 0.0
        for i in range(n - w)
    ])
    if len(drops) > 0 and np.abs(np.mean(drops)) > 1e-12:
        result['optimal_w_cv'] = float(np.std(drops) / np.abs(np.mean(drops)))
    else:
        result['optimal_w_cv'] = np.nan

    return result


def legacy_build_trend_analysis_df(quote: pd.DataFrame,
                                   trade_extreme_df: pd.DataFrame) -> pd.DataFrame:
    """
    遍历每笔交易，对 low -> high 区间做趋势分析。
    low: 在 [prev_max_profit, curr_max_profit] 范围内搜索真正的最低点
    high: trade[i].max_profit（真正的最高点）
    """
    records = []

    for idx, row in trade_extreme_df.iterrows():
        high_idx = int(row['max_profit_index'])

        # 确定 low 搜索起点
        search_start = int(row['entry_index'])

        # 在 [search_start, high_idx] 范围内找最低 low
        search_slice = quote.iloc[search_start:high_idx + 1]
        if len(search_slice) == 0:
            continue

        low_bar_idx = int(search_slice['low'].idxmin())
        low_price = float(quote.loc[low_bar_idx, 'low'])
        high_price = float(row['max_profit_price'])

        # 更新 prev_max_profit_idx（无论是否写入 records）
        if high_idx <= low_bar_idx:
            continue

        seg = quote.iloc[low_bar_idx:high_idx + 1]
        prices = seg['close'].to_numpy(dtype=float)
        stats = legacy_compute_trend_stats(prices)

        duration = high_idx - low_bar_idx
        # OLS 拟合线两端的值
        ols_fitted_start = stats['ols_intercept']
        ols_fitted_end = (stats['ols_slope'] * duration + stats['ols_intercept']
                          if not np.isnan(stats['ols_slope']) else np.nan)

        records.append({
            'trade_id': idx + 1,
            'entry_index': int(row['entry_index']),
            'entry_date': str(row.get('entry_date', '')),
            'exit_index': int(row['exit_index']),
            'exit_date': str(row.get('exit_date', '')),
            'search_start': search_start,
            'low_index': low_bar_idx,
            'low_date': str(quote.loc[low_bar_idx, 'Date']),
            'low_price': low_price,
            'high_index': high_idx,
            'high_date': str(row.get('max_profit_date', '')),
            'high_price': high_price,
            'duration_bars': duration,
            'total_return_pct': round(
                (high_price / low_price - 1.0) * 100.0, 4)
                if low_price != 0 else np.nan,
            'ols_slope': stats['ols_slope'],
            'ols_intercept': stats['ols_intercept'],
            'ols_r_squared': round(stats['ols_r_squared'], 6)
                if not np.isnan(stats['ols_r_squared']) else np.nan,
            'optimal_w_bars': stats['optimal_w_bars'],
            'optimal_w_cv': round(stats['optimal_w_cv'], 6)
                if not np.isnan(stats['optimal_w_cv']) else np.nan,
            'ols_fitted_start': ols_fitted_start,
            'ols_fitted_end': ols_fitted_end,
        })

    return pd.DataFrame(records)


def round_or_nan(value, digits: int = 6):
    if value is None or pd.isna(value) or not np.isfinite(value):
        return np.nan
    return round(float(value), digits)


def legacy_compute_optimal_window_stats_long(seg: pd.DataFrame,
                                             w_min: int,
                                             w_max: int,
                                             min_samples: int,
                                             stability_tol_ratio: float):
    result = {
        'optimal_w_bars': np.nan,
        'min_progress_pct': np.nan,
        'mean_progress_pct': np.nan,
        'max_progress_pct': np.nan,
        'std_progress_pct': np.nan,
        'std_to_min_ratio': np.nan,
        'range_to_min_ratio': np.nan,
        'progress_count': np.nan,
        'optimal_window_valid': 0,
        'optimal_window_invalid_reason': '',
    }
    scan_records = []

    if w_min > w_max:
        result['optimal_window_invalid_reason'] = 'invalid_window_range'
        return result, scan_records

    n = len(seg)
    highs = seg['high'].to_numpy(dtype=float)
    lows = seg['low'].to_numpy(dtype=float)
    cummax = np.maximum.accumulate(highs) if n > 0 else np.array([])

    has_candidate = False
    valid_rows = []

    for w in range(int(w_min), int(w_max) + 1):
        scan = {
            'w_bars': int(w),
            'window_valid': 0,
            'invalid_reason': '',
            'min_progress_pct': np.nan,
            'mean_progress_pct': np.nan,
            'max_progress_pct': np.nan,
            'std_progress_pct': np.nan,
            'range_above_min_pct': np.nan,
            'std_to_min_ratio': np.nan,
            'range_to_min_ratio': np.nan,
            'progress_count': 0,
        }

        if w >= n:
            scan['invalid_reason'] = 'segment_too_short'
            scan_records.append(scan)
            continue

        if (n - w) < min_samples:
            scan['invalid_reason'] = 'insufficient_samples'
            scan_records.append(scan)
            continue

        has_candidate = True
        progress_values = []
        invalid_reason = ''

        for t in range(w, n):
            base_price = lows[t - w + 1]
            if (not np.isfinite(base_price)) or base_price <= 0:
                invalid_reason = 'non_positive_base'
                break

            prev_max = cummax[t - w]
            curr_max = cummax[t]
            progress_pct = 100.0 * (curr_max - prev_max) / base_price

            if not np.isfinite(progress_pct):
                invalid_reason = 'invalid_progress'
                break
            if progress_pct <= 0:
                invalid_reason = 'non_positive_progress'
                break
            progress_values.append(progress_pct)

        if invalid_reason:
            scan['invalid_reason'] = invalid_reason
            scan_records.append(scan)
            continue

        arr = np.asarray(progress_values, dtype=float)
        min_progress = float(arr.min())
        mean_progress = float(arr.mean())
        max_progress = float(arr.max())
        std_progress = float(arr.std(ddof=0))
        range_above_min = max_progress - min_progress
        std_to_min = std_progress / min_progress
        range_to_min = range_above_min / min_progress

        scan.update({
            'window_valid': 1,
            'min_progress_pct': min_progress,
            'mean_progress_pct': mean_progress,
            'max_progress_pct': max_progress,
            'std_progress_pct': std_progress,
            'range_above_min_pct': range_above_min,
            'std_to_min_ratio': std_to_min,
            'range_to_min_ratio': range_to_min,
            'progress_count': int(len(arr)),
        })
        scan_records.append(scan)
        valid_rows.append(scan)

    if not has_candidate:
        result['optimal_window_invalid_reason'] = 'segment_too_short'
        return result, scan_records

    if len(valid_rows) == 0:
        result['optimal_window_invalid_reason'] = 'no_valid_window'
        return result, scan_records

    best_std_to_min = min(row['std_to_min_ratio'] for row in valid_rows)
    tol_limit = best_std_to_min * (1.0 + stability_tol_ratio)
    stable_rows = [
        row for row in valid_rows
        if row['std_to_min_ratio'] <= tol_limit + 1e-12
    ]
    stable_rows.sort(key=lambda row: (
        row['w_bars'],
        row['range_to_min_ratio'],
        -row['min_progress_pct'],
    ))
    best_row = stable_rows[0]

    result.update({
        'optimal_w_bars': int(best_row['w_bars']),
        'min_progress_pct': float(best_row['min_progress_pct']),
        'mean_progress_pct': float(best_row['mean_progress_pct']),
        'max_progress_pct': float(best_row['max_progress_pct']),
        'std_progress_pct': float(best_row['std_progress_pct']),
        'std_to_min_ratio': float(best_row['std_to_min_ratio']),
        'range_to_min_ratio': float(best_row['range_to_min_ratio']),
        'progress_count': int(best_row['progress_count']),
        'optimal_window_valid': 1,
        'optimal_window_invalid_reason': '',
    })
    return result, scan_records


def compute_progress_scan_long(seg: pd.DataFrame, w: int):
    n = len(seg)
    if w <= 0 or w >= n:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
            'segment_too_short',
        )

    highs = seg['high'].to_numpy(dtype=float)
    lows = seg['low'].to_numpy(dtype=float)
    x_values = seg.index.to_numpy(dtype=int)
    cummax = np.maximum.accumulate(highs)
    progress_offsets = []
    progress_x = []
    progress_y = []

    for t in range(w, n):
        base_price = lows[t - w + 1]
        if (not np.isfinite(base_price)) or base_price <= 0:
            return (
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=float),
                'non_positive_base',
            )

        progress_pct = 100.0 * (cummax[t] - cummax[t - w]) / base_price
        if not np.isfinite(progress_pct):
            return (
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=float),
                'invalid_progress',
            )

        progress_offsets.append(int(t))
        progress_x.append(int(x_values[t]))
        progress_y.append(float(progress_pct))

    return (
        np.asarray(progress_offsets, dtype=int),
        np.asarray(progress_x, dtype=int),
        np.asarray(progress_y, dtype=float),
        '',
    )


def compute_progress_pct_series_long(seg: pd.DataFrame,
                                     w: int,
                                     require_positive: bool = False):
    _, progress_x, progress_y, invalid_reason = compute_progress_scan_long(seg, w)
    if invalid_reason:
        return np.array([], dtype=int), np.array([], dtype=float), invalid_reason
    if require_positive and len(progress_y) > 0 and np.any(progress_y <= 0):
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            'non_positive_progress',
        )
    return progress_x, progress_y, ''


def legacy_build_trend_analysis_df_v2(quote: pd.DataFrame,
                                      trade_extreme_df: pd.DataFrame,
                                      w_min: int,
                                      w_max: int,
                                      min_samples: int,
                                      stability_tol_ratio: float):
    records = []
    window_scan_records = []

    for idx, row in trade_extreme_df.iterrows():
        high_idx = int(row['max_profit_index'])

        search_start = int(row['entry_index'])

        search_slice = quote.iloc[search_start:high_idx + 1]
        if len(search_slice) == 0:
            continue

        low_bar_idx = int(search_slice['low'].idxmin())
        low_price = float(quote.loc[low_bar_idx, 'low'])
        high_price = float(row['max_profit_price'])

        if high_idx <= low_bar_idx:
            continue

        seg = quote.iloc[low_bar_idx:high_idx + 1]
        stats, scan_rows = legacy_compute_optimal_window_stats_long(
            seg=seg,
            w_min=w_min,
            w_max=w_max,
            min_samples=min_samples,
            stability_tol_ratio=stability_tol_ratio,
        )

        trade_id = idx + 1
        duration = high_idx - low_bar_idx

        for scan_row in scan_rows:
            window_scan_records.append({
                'trade_id': trade_id,
                'entry_index': int(row['entry_index']),
                'entry_date': str(row.get('entry_date', '')),
                'low_index': low_bar_idx,
                'high_index': high_idx,
                'duration_bars': duration,
                'w_bars': int(scan_row['w_bars']),
                'window_valid': int(scan_row['window_valid']),
                'invalid_reason': scan_row['invalid_reason'],
                'min_progress_pct': round_or_nan(scan_row['min_progress_pct']),
                'mean_progress_pct': round_or_nan(scan_row['mean_progress_pct']),
                'max_progress_pct': round_or_nan(scan_row['max_progress_pct']),
                'std_progress_pct': round_or_nan(scan_row['std_progress_pct']),
                'range_above_min_pct': round_or_nan(scan_row['range_above_min_pct']),
                'std_to_min_ratio': round_or_nan(scan_row['std_to_min_ratio']),
                'range_to_min_ratio': round_or_nan(scan_row['range_to_min_ratio']),
                'progress_count': int(scan_row['progress_count']),
            })

        records.append({
            'trade_id': trade_id,
            'entry_index': int(row['entry_index']),
            'entry_date': str(row.get('entry_date', '')),
            'exit_index': int(row['exit_index']),
            'exit_date': str(row.get('exit_date', '')),
            'search_start': search_start,
            'low_index': low_bar_idx,
            'low_date': str(quote.loc[low_bar_idx, 'Date']),
            'low_price': low_price,
            'high_index': high_idx,
            'high_date': str(row.get('max_profit_date', '')),
            'high_price': high_price,
            'duration_bars': duration,
            'total_return_pct': round(
                (high_price / low_price - 1.0) * 100.0, 4)
                if low_price != 0 else np.nan,
            'optimal_w_bars': (
                int(stats['optimal_w_bars'])
                if pd.notna(stats['optimal_w_bars']) else np.nan
            ),
            'min_progress_pct': round_or_nan(stats['min_progress_pct']),
            'mean_progress_pct': round_or_nan(stats['mean_progress_pct']),
            'max_progress_pct': round_or_nan(stats['max_progress_pct']),
            'std_progress_pct': round_or_nan(stats['std_progress_pct']),
            'std_to_min_ratio': round_or_nan(stats['std_to_min_ratio']),
            'range_to_min_ratio': round_or_nan(stats['range_to_min_ratio']),
            'progress_count': (
                int(stats['progress_count'])
                if pd.notna(stats['progress_count']) else np.nan
            ),
            'optimal_window_valid': int(stats['optimal_window_valid']),
            'optimal_window_invalid_reason': (
                stats['optimal_window_invalid_reason']
            ),
        })

    return pd.DataFrame(records), pd.DataFrame(window_scan_records)


def get_reference_speed_from_row(row) -> float:
    if isinstance(row, pd.Series):
        ref = row.get('reference_speed_pct_per_bar', np.nan)
        trend_speed = row.get('trend_line_speed_pct_per_bar', np.nan)
        segment_speed = row.get('segment_speed_pct_per_bar', np.nan)
    else:
        ref = getattr(row, 'reference_speed_pct_per_bar', np.nan)
        trend_speed = getattr(row, 'trend_line_speed_pct_per_bar', np.nan)
        segment_speed = getattr(row, 'segment_speed_pct_per_bar', np.nan)
    if pd.notna(ref):
        return float(ref)
    if pd.notna(trend_speed):
        return float(trend_speed)
    if pd.notna(segment_speed):
        return float(segment_speed)
    return np.nan


def get_trend_test_case_data(trend_df: pd.DataFrame,
                             window_scan_df: pd.DataFrame,
                             test_case_index: int,
                             requested_windows: list[int]):
    if len(trend_df) == 0:
        return None, pd.DataFrame(), []
    safe_index = min(max(int(test_case_index), 0), len(trend_df) - 1)
    case_row = trend_df.iloc[safe_index].copy()
    trade_id = int(case_row['trade_id'])
    case_scan = window_scan_df[
        window_scan_df['trade_id'] == trade_id
    ].copy().sort_values('w_bars').reset_index(drop=True)
    windows = []
    optimal_w = (
        int(case_row['optimal_w_bars'])
        if pd.notna(case_row['optimal_w_bars']) else None
    )
    for w in list(requested_windows) + ([optimal_w] if optimal_w is not None else []):
        if w is None or pd.isna(w):
            continue
        w_int = int(w)
        if w_int not in windows:
            windows.append(w_int)
    return case_row, case_scan, windows


def print_trend_test_case_summary(trend_df: pd.DataFrame,
                                  window_scan_df: pd.DataFrame,
                                  test_case_index: int,
                                  test_windows: list[int],
                                  improvement_capture_target: float,
                                  near_band: int,
                                  big_drop_ratio: float):
    case_row, case_scan, display_windows = get_trend_test_case_data(
        trend_df, window_scan_df, test_case_index, test_windows)
    if case_row is None or len(case_scan) == 0:
        print('[TestMode] no trend test case available.')
        return

    lookup = {
        int(row['w_bars']): row
        for _, row in case_scan.iterrows()
    }

    print('\n[TestMode] ===== Case Summary =====')
    print(
        f"[TestMode] trade_id={int(case_row['trade_id'])}, "
        f"low={int(case_row['low_index'])} ({case_row['low_date']}) @ {case_row['low_price']}, "
        f"high={int(case_row['high_index'])} ({case_row['high_date']}) @ {case_row['high_price']}"
    )
    print(
        f"[TestMode] duration_bars={int(case_row['duration_bars'])}, "
        f"speed_reference_source={case_row['speed_reference_source']}, "
        f"reference_speed_pct_per_bar={round_or_nan(case_row['reference_speed_pct_per_bar'])}"
    )
    print(
        f"[TestMode] first_feasible_w={case_row['first_feasible_w_bars']}, "
        f"best_offset_w={case_row['best_offset_w_bars']}, "
        f"optimal_w={case_row['optimal_w_bars']}"
    )

    candidate_rows = []
    for w in display_windows:
        scan_row = lookup.get(int(w))
        if scan_row is None:
            candidate_rows.append({
                'w_bars': int(w),
                'window_valid': 0,
                'invalid_reason': 'not_scanned',
                'offset_score': np.nan,
                'improvement_capture': np.nan,
                'fit_y_pct': np.nan,
                'y_speed_pct_per_bar': np.nan,
                'fit_entry_index': np.nan,
                'entry_delay_bars': np.nan,
                'fitted_return_pct': np.nan,
                'capture_ratio': np.nan,
                'selected_flag': 0,
            })
            continue
        candidate_rows.append({
            'w_bars': int(scan_row['w_bars']),
            'window_valid': int(scan_row['window_valid']),
            'invalid_reason': scan_row['invalid_reason'],
            'offset_score': round_or_nan(scan_row['offset_score']),
            'improvement_capture': round_or_nan(scan_row['improvement_capture']),
            'fit_y_pct': round_or_nan(scan_row['fit_y_pct']),
            'y_speed_pct_per_bar': round_or_nan(
                scan_row['y_speed_pct_per_bar']),
            'fit_entry_index': (
                int(scan_row['fit_entry_index'])
                if pd.notna(scan_row['fit_entry_index']) else np.nan
            ),
            'entry_delay_bars': (
                int(scan_row['entry_delay_bars'])
                if pd.notna(scan_row['entry_delay_bars']) else np.nan
            ),
            'fitted_return_pct': round_or_nan(scan_row['fitted_return_pct']),
            'capture_ratio': round_or_nan(scan_row['capture_ratio']),
            'selected_flag': int(scan_row['selected_flag']),
        })

    print('\n[TestMode] ===== Candidate Table =====')
    print(pd.DataFrame(candidate_rows).to_string(index=False))

    sampled_cols = [
        'w_bars', 'window_valid', 'offset_score', 'improvement_capture',
        'fit_y_pct', 'y_speed_pct_per_bar', 'entry_delay_bars',
        'fitted_return_pct', 'capture_ratio', 'selected_flag'
    ]
    sampled_scan = case_scan[sampled_cols].copy()
    print('\n[TestMode] ===== Sampled Window Scan =====')
    print(sampled_scan.to_string(index=False))

    def _valid_row(w: int):
        row = lookup.get(int(w))
        if row is None or int(row['window_valid']) != 1:
            return None
        return row

    row5 = _valid_row(5)
    row7 = _valid_row(7)
    row15 = _valid_row(15)
    optimal_w = int(case_row['optimal_w_bars']) if pd.notna(case_row['optimal_w_bars']) else None

    print('\n[TestMode] ===== Explicit Checks =====')

    if row5 is None or row7 is None:
        print('[TestMode][Check-1] w=5 or w=7 invalid -> SKIP')
    else:
        delta = float(row5['offset_score']) - float(row7['offset_score'])
        status = 'PASS' if float(row7['offset_score']) < float(row5['offset_score']) else 'FAIL'
        print(
            f"[TestMode][Check-1] offset_5={float(row5['offset_score']):.6f}, "
            f"offset_7={float(row7['offset_score']):.6f}, "
            f"delta_5_7={delta:.6f} -> {status}"
        )

    if row5 is None or row7 is None or row15 is None:
        print('[TestMode][Check-2] w=5 or w=7 or w=15 invalid -> SKIP')
    else:
        delta_5_7 = float(row5['offset_score']) - float(row7['offset_score'])
        delta_7_15 = float(row7['offset_score']) - float(row15['offset_score'])
        if delta_7_15 <= 0:
            ratio_text = 'inf'
            status = 'PASS'
        else:
            ratio = delta_5_7 / delta_7_15
            ratio_text = f'{ratio:.6f}'
            status = 'PASS' if ratio >= big_drop_ratio else 'FAIL'
        print(
            f"[TestMode][Check-2] delta_5_7={delta_5_7:.6f}, "
            f"delta_7_15={delta_7_15:.6f}, ratio={ratio_text}, "
            f"threshold={big_drop_ratio:.2f} -> {status}"
        )

    if row7 is None:
        print('[TestMode][Check-3] w=7 invalid -> SKIP')
    else:
        capture = float(row7['improvement_capture'])
        status = 'PASS' if capture >= improvement_capture_target else 'FAIL'
        print(
            f"[TestMode][Check-3] improvement_capture_7={capture:.6f}, "
            f"threshold={improvement_capture_target:.2f} -> {status}"
        )

    if optimal_w is None:
        print('[TestMode][Check-4] optimal_w unavailable -> SKIP')
    else:
        lower = 7 - int(near_band)
        upper = 7 + int(near_band)
        status = 'PASS' if lower <= optimal_w <= upper else 'FAIL'
        print(
            f"[TestMode][Check-4] optimal_w={optimal_w}, "
            f"acceptable_range=[{lower}, {upper}] -> {status}"
        )


def export_trend_test_case_html(file_name: str,
                                save_name: str,
                                underlying1: pd.DataFrame,
                                trend_df: pd.DataFrame,
                                window_scan_df: pd.DataFrame,
                                factor: float,
                                test_case_index: int,
                                test_windows: list[int],
                                improvement_capture_target: float):
    if go is None or make_subplots is None:
        print('[HTML] plotly is not installed, skip trend test html export.')
        return

    case_row, case_scan, display_windows = get_trend_test_case_data(
        trend_df, window_scan_df, test_case_index, test_windows)
    if case_row is None or len(case_scan) == 0:
        print('[HTML] no trend test case data, skip test html export.')
        return

    lookup = {
        int(row['w_bars']): row
        for _, row in case_scan.iterrows()
    }
    optimal_w = (
        int(case_row['optimal_w_bars'])
        if pd.notna(case_row['optimal_w_bars']) else None
    )
    optimal_row = lookup.get(optimal_w) if optimal_w is not None else None
    low_idx = int(case_row['low_index'])
    high_idx = int(case_row['high_index'])
    seg = underlying1.iloc[low_idx:high_idx + 1].copy()
    ref_speed = get_reference_speed_from_row(case_row)
    x_left = low_idx
    x_right = high_idx
    view = underlying1.iloc[x_left:x_right + 1]

    fig_html = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        specs=[
            [{}],
            [{'secondary_y': True}],
            [{}],
            [{}],
        ],
        vertical_spacing=0.06,
        row_heights=[0.42, 0.22, 0.18, 0.18],
        subplot_titles=(
            'Case Price',
            'Offset Score And Improvement Capture',
            'Mean Speed By Window',
            'Minimum Speed By Window',
        ),
    )

    fig_html.add_trace(go.Candlestick(
        x=view.index.to_numpy(),
        open=view['open'] / factor * 100,
        high=view['high'] / factor * 100,
        low=view['low'] / factor * 100,
        close=view['close'] / factor * 100,
        text=build_candlestick_hovertext(view, factor),
        name='candles',
        showlegend=False,
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

    if len(seg) >= 2:
        t = np.arange(len(seg), dtype=float)
        fitted = case_row['ols_slope'] * t + case_row['ols_intercept']
        fig_html.add_trace(go.Scatter(
            x=seg.index.to_list(),
            y=(fitted / factor * 100).tolist(),
            mode='lines',
            line=dict(color=ACCENT_BLUE, width=2),
            name='trend_line',
            hoverinfo='skip',
        ), row=1, col=1)

    fig_html.add_trace(go.Scatter(
        x=[low_idx],
        y=[float(case_row['low_price']) / factor * 100],
        mode='markers',
        marker=dict(color='#1F77B4', size=6),
        name='low',
        hovertemplate=(
            'low point<br>'
            f"index: {low_idx}<br>"
            f"date: {case_row['low_date']}<br>"
            f"price: {case_row['low_price']}<extra></extra>"
        ),
    ), row=1, col=1)
    fig_html.add_trace(go.Scatter(
        x=[high_idx],
        y=[float(case_row['high_price']) / factor * 100],
        mode='markers',
        marker=dict(color='orange', size=6),
        name='high',
        hovertemplate=(
            'high point<br>'
            f"index: {high_idx}<br>"
            f"date: {case_row['high_date']}<br>"
            f"price: {case_row['high_price']}<extra></extra>"
        ),
    ), row=1, col=1)

    optimal_color = '#2ca02c'
    if optimal_row is not None and int(optimal_row['window_valid']) == 1:
        hover_text = (
            f"optimal_w: {int(optimal_w)}<br>"
            "fit entry marker<br>"
            f"fit_entry_index: {int(optimal_row['fit_entry_index'])}<br>"
            f"fit_entry_price: {float(optimal_row['fit_entry_price']):.4f}<br>"
            f"offset_score: {float(optimal_row['offset_score']):.6f}<br>"
            f"improvement_capture: {float(optimal_row['improvement_capture']):.6f}<br>"
            f"fitted_return_pct: {float(optimal_row['fitted_return_pct']):.4f}<br>"
            f"capture_ratio: {float(optimal_row['capture_ratio']):.4f}"
        )
        fig_html.add_trace(go.Scatter(
            x=[int(optimal_row['fit_entry_index'])],
            y=[float(optimal_row['fit_entry_price']) / factor * 100],
            mode='markers',
            marker=dict(color=optimal_color, size=8, symbol='triangle-up'),
            name=f'optimal_w={int(optimal_w)}',
            legendgroup='optimal_w',
            showlegend=True,
            text=[hover_text],
            hovertemplate='%{text}<extra></extra>',
        ), row=1, col=1)

    valid_scan = case_scan[case_scan['window_valid'] == 1].copy()
    if len(valid_scan) > 0:
        valid_scan['mean_speed_pct_per_bar'] = (
            valid_scan['mean_progress_pct'].astype(float)
            / valid_scan['w_bars'].astype(float)
        )
        valid_scan['min_speed_pct_per_bar'] = (
            valid_scan['fit_y_pct'].astype(float)
            / valid_scan['w_bars'].astype(float)
        )
        fig_html.add_trace(go.Scatter(
            x=valid_scan['w_bars'].astype(int).tolist(),
            y=valid_scan['offset_score'].astype(float).tolist(),
            mode='lines+markers',
            line=dict(color='rgba(60,60,60,0.85)', width=2),
            marker=dict(size=5),
            name='offset_score',
            hovertemplate=(
                'offset_score: mean abs relative speed deviation<br>'
                'w: %{x}<br>'
                'offset_score: %{y:.6f}<extra></extra>'
            ),
        ), row=2, col=1, secondary_y=False)

        fig_html.add_trace(go.Scatter(
            x=valid_scan['w_bars'].astype(int).tolist(),
            y=valid_scan['improvement_capture'].astype(float).tolist(),
            mode='lines+markers',
            line=dict(color='rgba(20,120,80,0.85)', width=2),
            marker=dict(size=5),
            name='improvement_capture',
            hovertemplate=(
                'improvement_capture: captured share of total offset improvement<br>'
                'w: %{x}<br>'
                'improvement_capture: %{y:.6f}<extra></extra>'
            ),
        ), row=2, col=1, secondary_y=True)

        fig_html.add_trace(go.Scatter(
            x=valid_scan['w_bars'].astype(int).tolist(),
            y=valid_scan['mean_speed_pct_per_bar'].astype(float).tolist(),
            mode='lines+markers',
            line=dict(color='rgba(31,119,180,0.85)', width=2),
            marker=dict(size=5),
            name='mean_speed_pct_per_bar',
            hovertemplate=(
                'mean_speed_pct_per_bar: mean progress divided by window size<br>'
                'w: %{x}<br>'
                'mean_speed_pct_per_bar: %{y:.6f}<extra></extra>'
            ),
        ), row=3, col=1)

        fig_html.add_trace(go.Scatter(
            x=valid_scan['w_bars'].astype(int).tolist(),
            y=valid_scan['min_speed_pct_per_bar'].astype(float).tolist(),
            mode='lines+markers',
            line=dict(color='rgba(255,140,0,0.88)', width=2),
            marker=dict(size=5),
            name='min_speed_pct_per_bar',
            hovertemplate=(
                'min_speed_pct_per_bar: minimum progress divided by window size<br>'
                'w: %{x}<br>'
                'min_speed_pct_per_bar: %{y:.6f}<extra></extra>'
            ),
        ), row=4, col=1)

        if np.isfinite(ref_speed):
            fig_html.add_trace(go.Scatter(
                x=valid_scan['w_bars'].astype(int).tolist(),
                y=[float(ref_speed)] * len(valid_scan),
                mode='lines',
                line=dict(color='rgba(0,0,0,0.35)', width=1.2, dash='dash'),
                name='reference_speed',
                hovertemplate=(
                    'reference_speed_pct_per_bar: trend-line reference speed<br>'
                    f'value: {float(ref_speed):.6f}<extra></extra>'
                ),
            ), row=3, col=1)
            fig_html.add_trace(go.Scatter(
                x=valid_scan['w_bars'].astype(int).tolist(),
                y=[float(ref_speed)] * len(valid_scan),
                mode='lines',
                line=dict(color='rgba(0,0,0,0.35)', width=1.2, dash='dash'),
                name='reference_speed',
                showlegend=False,
                hovertemplate=(
                    'reference_speed_pct_per_bar: trend-line reference speed<br>'
                    f'value: {float(ref_speed):.6f}<extra></extra>'
                ),
            ), row=4, col=1)

    if optimal_row is not None and int(optimal_row['window_valid']) == 1:
        marker_text = (
            f"optimal_w: {int(optimal_w)}<br>"
            f"offset_score: {float(optimal_row['offset_score']):.6f}<br>"
            f"improvement_capture: {float(optimal_row['improvement_capture']):.6f}<br>"
            f"fit_y_pct: {float(optimal_row['fit_y_pct']):.4f}<br>"
            f"y_speed_pct_per_bar: {float(optimal_row['y_speed_pct_per_bar']):.4f}<br>"
            f"entry_delay_bars: {int(optimal_row['entry_delay_bars'])}<br>"
            f"fitted_return_pct: {float(optimal_row['fitted_return_pct']):.4f}"
        )
        for field in ['offset_score', 'improvement_capture']:
            fig_html.add_trace(go.Scatter(
                x=[int(optimal_row['w_bars'])],
                y=[float(optimal_row[field])],
                mode='markers',
                marker=dict(color=optimal_color, size=9),
                name=f'optimal_w={int(optimal_w)}',
                legendgroup='optimal_w',
                showlegend=False,
                text=[marker_text],
                hovertemplate='%{text}<extra></extra>',
            ), row=2, col=1, secondary_y=(field == 'improvement_capture'))

        mean_speed_value = (
            float(optimal_row['mean_progress_pct']) / float(optimal_row['w_bars'])
        )
        min_speed_value = float(optimal_row['y_speed_pct_per_bar'])
        mean_speed_text = (
            f"optimal_w: {int(optimal_w)}<br>"
            f"mean_speed_pct_per_bar: {mean_speed_value:.6f}<br>"
            f"min_speed_pct_per_bar: {min_speed_value:.6f}<br>"
            f"reference_speed_pct_per_bar: {float(ref_speed):.6f}<br>"
            f"fit_y_pct: {float(optimal_row['fit_y_pct']):.4f}<br>"
            f"y_speed_pct_per_bar: {float(optimal_row['y_speed_pct_per_bar']):.6f}"
        )
        fig_html.add_trace(go.Scatter(
            x=[int(optimal_row['w_bars'])],
            y=[mean_speed_value],
            mode='markers',
            marker=dict(color=optimal_color, size=9),
            name=f'optimal_w={int(optimal_w)}',
            legendgroup='optimal_w',
            showlegend=False,
            text=[mean_speed_text],
            hovertemplate='%{text}<extra></extra>',
        ), row=3, col=1)

        min_speed_text = (
            f"optimal_w: {int(optimal_w)}<br>"
            f"min_speed_pct_per_bar: {min_speed_value:.6f}<br>"
            f"mean_speed_pct_per_bar: {mean_speed_value:.6f}<br>"
            f"reference_speed_pct_per_bar: {float(ref_speed):.6f}<br>"
            f"fit_y_pct: {float(optimal_row['fit_y_pct']):.4f}"
        )
        fig_html.add_trace(go.Scatter(
            x=[int(optimal_row['w_bars'])],
            y=[min_speed_value],
            mode='markers',
            marker=dict(color='#ff8c00', size=9, symbol='diamond'),
            name=f'optimal_w={int(optimal_w)}',
            legendgroup='optimal_w',
            showlegend=False,
            text=[min_speed_text],
            hovertemplate='%{text}<extra></extra>',
        ), row=4, col=1)

    if len(valid_scan) > 0:
        fig_html.add_trace(go.Scatter(
            x=valid_scan['w_bars'].astype(int).tolist(),
            y=[improvement_capture_target] * len(valid_scan),
            mode='lines',
            line=dict(color='rgba(0,0,0,0.35)', width=1.2, dash='dash'),
            name='capture_target',
            hovertemplate=(
                'improvement_capture target<br>'
                f"value: {improvement_capture_target:.2f}<extra></extra>"
            ),
        ), row=2, col=1, secondary_y=True)

    fig_html.update_layout(
        title=(
            f"Trend Test Case trade {int(case_row['trade_id'])}: "
            f"offset / capture / speed sampling"
        ),
        template='plotly_white',
        autosize=True,
        hovermode='closest',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='left', x=0),
        margin=dict(l=42, r=25, t=90, b=45, pad=0),
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.50)',
            bordercolor='rgba(0, 0, 0, 0.45)',
            font=dict(color='black')
        )
    )
    for row_no in [1, 2, 3, 4]:
        fig_html.update_xaxes(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.35)',
            tickfont=dict(size=10),
            row=row_no,
            col=1,
        )
        fig_html.update_yaxes(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.35)',
            tickfont=dict(size=10),
            row=row_no,
            col=1,
        )
    fig_html.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    fig_html.update_xaxes(
        range=[x_left, x_right],
        autorange=False,
        row=1,
        col=1,
    )
    fig_html.update_yaxes(title='price %', row=1, col=1)
    fig_html.update_yaxes(title='offset_score', row=2, col=1, secondary_y=False)
    fig_html.update_yaxes(
        title='improvement_capture', row=2, col=1, secondary_y=True)
    fig_html.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.35)',
        tickfont=dict(size=10),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig_html.update_yaxes(title='mean_speed_pct_per_bar', row=3, col=1)
    fig_html.update_yaxes(title='min_speed_pct_per_bar', row=4, col=1)

    html_dir = get_html_output_dir(file_name, TREND_TEST_CASE_HTML_FOLDER)
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(
        html_dir, save_name + ' trend_test_case interactive.html')
    html_text = fig_html.to_html(
        include_plotlyjs=True, full_html=True,
        default_width='100vw', default_height='100vh',
        config={'responsive': True, 'displayModeBar': False,
                'displaylogo': False}
    )
    html_text = html_text.replace(
        '<head>',
        '<head><style>'
        'html,body{width:100%;height:100%;margin:0;padding:0;overflow:hidden;}'
        '.plotly-graph-div{width:100vw !important;height:100vh !important;}'
        '.hoverlayer .hovertext .bg,'
        '.hoverlayer .hovertext rect,'
        '.hoverlayer .hovertext path{'
        'fill:rgba(255,255,255,0.50) !important;'
        'fill-opacity:0.50 !important;'
        'stroke:rgba(0,0,0,0.45) !important;'
        'stroke-opacity:0.45 !important;}'
        '.hoverlayer .hovertext{opacity:1 !important;}'
        '.hoverlayer .hovertext text{fill:#000 !important;}'
        '</style>',
        1
    )
    html_text = html_text.replace(
        '<body>', '<body style="margin:0;overflow:hidden;">', 1)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print('[HTML] saved trend test chart.')


def compute_ols_trend_line_stats_long(seg: pd.DataFrame):
    result = {
        'ols_slope': np.nan,
        'ols_intercept': np.nan,
        'ols_r_squared': np.nan,
        'trend_line_speed_pct_per_bar': np.nan,
        'segment_speed_pct_per_bar': np.nan,
        'trend_total_return_pct': np.nan,
    }
    n = len(seg)
    if n < 2:
        return result

    close_prices = seg['close'].to_numpy(dtype=float)
    low_price = float(seg['low'].iloc[0])
    high_price = float(seg['high'].iloc[-1])
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, close_prices, 1)
    fitted = slope * t + intercept
    ss_res = np.sum((close_prices - fitted) ** 2)
    ss_tot = np.sum((close_prices - np.mean(close_prices)) ** 2)
    fitted_start = float(fitted[0])
    result.update({
        'ols_slope': float(slope),
        'ols_intercept': float(intercept),
        'ols_r_squared': (
            float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        ),
        'trend_line_speed_pct_per_bar': (
            float(100.0 * slope / fitted_start)
            if np.isfinite(fitted_start) and abs(fitted_start) > 1e-12
            else np.nan
        ),
        'trend_total_return_pct': (
            float((high_price / low_price - 1.0) * 100.0)
            if low_price != 0 else np.nan
        ),
        'segment_speed_pct_per_bar': (
            float(((high_price / low_price - 1.0) * 100.0) / (n - 1))
            if low_price != 0 and n > 1 else np.nan
        ),
    })
    return result


def compute_optimal_window_stats_long(seg: pd.DataFrame,
                                      w_min: int,
                                      w_max: int,
                                      min_samples: int,
                                      improvement_capture_target: float):
    ols_stats = compute_ols_trend_line_stats_long(seg)
    reference_speed = ols_stats['trend_line_speed_pct_per_bar']
    speed_reference_source = 'ols'
    if (not np.isfinite(reference_speed)) or reference_speed <= 0:
        reference_speed = ols_stats['segment_speed_pct_per_bar']
        speed_reference_source = 'segment_avg'

    result = {
        **ols_stats,
        'speed_reference_source': speed_reference_source,
        'reference_speed_pct_per_bar': (
            float(reference_speed) if np.isfinite(reference_speed) else np.nan
        ),
        'first_feasible_w_bars': np.nan,
        'best_offset_w_bars': np.nan,
        'best_offset_score': np.nan,
        'optimal_w_bars': np.nan,
        'offset_score': np.nan,
        'improvement_capture': np.nan,
        'fit_y_pct': np.nan,
        'y_speed_pct_per_bar': np.nan,
        'fit_entry_index': np.nan,
        'fit_entry_date': '',
        'fit_entry_price': np.nan,
        'entry_delay_bars': np.nan,
        'fitted_return_pct': np.nan,
        'capture_ratio': np.nan,
        'progress_count': np.nan,
        'mean_progress_pct': np.nan,
        'max_progress_pct': np.nan,
        'std_progress_pct': np.nan,
        'std_to_y_ratio': np.nan,
        'range_to_y_ratio': np.nan,
        'optimal_window_valid': 0,
        'optimal_window_invalid_reason': '',
    }
    scan_records = []

    if w_min > w_max:
        result['optimal_window_invalid_reason'] = 'invalid_window_range'
        return result, scan_records

    n = len(seg)
    if n < 2:
        result['optimal_window_invalid_reason'] = 'segment_too_short'
        return result, scan_records

    if (not np.isfinite(reference_speed)) or reference_speed <= 0:
        result['optimal_window_invalid_reason'] = 'invalid_reference_speed'
        return result, scan_records

    seg_dates = seg['Date'].to_numpy()
    seg_close = seg['close'].to_numpy(dtype=float)
    low_price = float(seg['low'].iloc[0])
    high_price = float(seg['high'].iloc[-1])
    price_span = high_price - low_price

    has_candidate = False
    valid_rows = []

    for w in range(int(w_min), int(w_max) + 1):
        scan = {
            **ols_stats,
            'speed_reference_source': speed_reference_source,
            'reference_speed_pct_per_bar': float(reference_speed),
            'w_bars': int(w),
            'window_valid': 0,
            'invalid_reason': '',
            'offset_score': np.nan,
            'improvement_capture': np.nan,
            'first_feasible_flag': 0,
            'best_offset_flag': 0,
            'selected_flag': 0,
            'fit_y_pct': np.nan,
            'y_speed_pct_per_bar': np.nan,
            'fit_entry_index': np.nan,
            'fit_entry_date': '',
            'fit_entry_price': np.nan,
            'entry_delay_bars': np.nan,
            'fitted_return_pct': np.nan,
            'capture_ratio': np.nan,
            'progress_count': 0,
            'mean_progress_pct': np.nan,
            'max_progress_pct': np.nan,
            'std_progress_pct': np.nan,
            'range_above_y_pct': np.nan,
            'std_to_y_ratio': np.nan,
            'range_to_y_ratio': np.nan,
        }

        if w >= n:
            scan['invalid_reason'] = 'segment_too_short'
            scan_records.append(scan)
            continue
        if (n - w) < min_samples:
            scan['invalid_reason'] = 'insufficient_samples'
            scan_records.append(scan)
            continue

        has_candidate = True
        progress_offsets, progress_x, progress_values, invalid_reason = (
            compute_progress_scan_long(seg, w)
        )
        if invalid_reason:
            scan['invalid_reason'] = invalid_reason
            scan_records.append(scan)
            continue
        if len(progress_values) < min_samples:
            scan['invalid_reason'] = 'insufficient_samples'
            scan_records.append(scan)
            continue
        if np.any(progress_values <= 0):
            scan['invalid_reason'] = 'non_positive_progress'
            scan_records.append(scan)
            continue

        fit_y_pct = float(progress_values.min())
        if (not np.isfinite(fit_y_pct)) or fit_y_pct <= 0:
            scan['invalid_reason'] = 'non_positive_fit_y'
            scan_records.append(scan)
            continue

        fit_entry_offset = int(progress_offsets[0])
        fit_entry_index = int(progress_x[0])
        fit_entry_price = float(seg_close[fit_entry_offset])
        fitted_return_pct = (
            (high_price / fit_entry_price - 1.0) * 100.0
            if fit_entry_price != 0 else np.nan
        )
        capture_ratio = (
            (high_price - fit_entry_price) / price_span
            if price_span > 0 else np.nan
        )
        if not np.isfinite(capture_ratio):
            scan['invalid_reason'] = 'invalid_capture_ratio'
            scan_records.append(scan)
            continue

        mean_progress = float(progress_values.mean())
        max_progress = float(progress_values.max())
        std_progress = float(progress_values.std(ddof=0))
        range_above_y = max_progress - fit_y_pct
        std_to_y = std_progress / fit_y_pct
        range_to_y = range_above_y / fit_y_pct
        y_speed_pct_per_bar = fit_y_pct / float(w)
        offset_values = np.abs((progress_values / float(w)) / reference_speed - 1.0)
        offset_score = float(offset_values.mean())

        scan.update({
            'window_valid': 1,
            'offset_score': offset_score,
            'fit_y_pct': fit_y_pct,
            'y_speed_pct_per_bar': y_speed_pct_per_bar,
            'fit_entry_index': fit_entry_index,
            'fit_entry_date': str(seg_dates[fit_entry_offset]),
            'fit_entry_price': fit_entry_price,
            'entry_delay_bars': int(fit_entry_offset),
            'fitted_return_pct': fitted_return_pct,
            'capture_ratio': capture_ratio,
            'progress_count': int(len(progress_values)),
            'mean_progress_pct': mean_progress,
            'max_progress_pct': max_progress,
            'std_progress_pct': std_progress,
            'range_above_y_pct': range_above_y,
            'std_to_y_ratio': std_to_y,
            'range_to_y_ratio': range_to_y,
        })
        scan_records.append(scan)
        valid_rows.append(scan)

    if not has_candidate:
        result['optimal_window_invalid_reason'] = 'segment_too_short'
        return result, scan_records

    if len(valid_rows) == 0:
        result['optimal_window_invalid_reason'] = 'no_valid_window'
        return result, scan_records

    first_feasible_row = min(valid_rows, key=lambda row: row['w_bars'])
    best_offset_row = min(
        valid_rows,
        key=lambda row: (row['offset_score'], row['w_bars'])
    )
    total_improvement = (
        first_feasible_row['offset_score'] - best_offset_row['offset_score']
    )
    if total_improvement <= 1e-12:
        for row in valid_rows:
            row['improvement_capture'] = 1.0
    else:
        for row in valid_rows:
            row['improvement_capture'] = (
                (first_feasible_row['offset_score'] - row['offset_score'])
                / total_improvement
            )

    for row in valid_rows:
        row['first_feasible_flag'] = int(
            row['w_bars'] == first_feasible_row['w_bars']
        )
        row['best_offset_flag'] = int(
            row['w_bars'] == best_offset_row['w_bars']
        )

    candidate_rows = [
        row for row in valid_rows
        if row['improvement_capture'] >= improvement_capture_target - 1e-12
    ]
    if len(candidate_rows) == 0:
        candidate_rows = [best_offset_row]

    candidate_rows.sort(key=lambda row: (
        row['w_bars'],
        row['offset_score'],
        -row['fit_y_pct'],
    ))
    best_row = candidate_rows[0]
    best_row['selected_flag'] = 1

    result.update({
        'speed_reference_source': speed_reference_source,
        'reference_speed_pct_per_bar': float(reference_speed),
        'first_feasible_w_bars': int(first_feasible_row['w_bars']),
        'best_offset_w_bars': int(best_offset_row['w_bars']),
        'best_offset_score': float(best_offset_row['offset_score']),
        'optimal_w_bars': int(best_row['w_bars']),
        'offset_score': float(best_row['offset_score']),
        'improvement_capture': float(best_row['improvement_capture']),
        'fit_y_pct': float(best_row['fit_y_pct']),
        'y_speed_pct_per_bar': float(best_row['y_speed_pct_per_bar']),
        'fit_entry_index': int(best_row['fit_entry_index']),
        'fit_entry_date': best_row['fit_entry_date'],
        'fit_entry_price': float(best_row['fit_entry_price']),
        'entry_delay_bars': int(best_row['entry_delay_bars']),
        'fitted_return_pct': float(best_row['fitted_return_pct']),
        'capture_ratio': float(best_row['capture_ratio']),
        'progress_count': int(best_row['progress_count']),
        'mean_progress_pct': float(best_row['mean_progress_pct']),
        'max_progress_pct': float(best_row['max_progress_pct']),
        'std_progress_pct': float(best_row['std_progress_pct']),
        'std_to_y_ratio': float(best_row['std_to_y_ratio']),
        'range_to_y_ratio': float(best_row['range_to_y_ratio']),
        'optimal_window_valid': 1,
        'optimal_window_invalid_reason': '',
    })
    return result, scan_records


def find_next_trend_segment_by_constraint_window(
        quote: pd.DataFrame,
        search_start: int,
        constraint_w: int,
        debug_records=None):
    n = len(quote)
    if constraint_w <= 0 or search_start < 0 or search_start >= n - 1:
        return None

    start_idx = int(search_start)
    start_low = float(quote.loc[start_idx, 'low'])

    for end_idx in range(start_idx + 1, n):
        record_debug = should_record_trend_debug(start_idx, end_idx)
        current_low = float(quote.loc[int(end_idx), 'low'])
        if current_low < start_low:
            valid_end_idx = int(end_idx - 1)
            valid_slice = quote.iloc[start_idx:valid_end_idx + 1]
            if len(valid_slice) == 0:
                return None
            high_idx = int(valid_slice['high'].idxmax())
            if record_debug and debug_records is not None:
                debug_records.append({
                    'search_start': int(start_idx),
                    'search_start_date': str(quote.loc[int(start_idx), 'Date']),
                    'end_idx': int(end_idx),
                    'end_date': str(quote.loc[int(end_idx), 'Date']),
                    'low_idx': int(start_idx),
                    'low_date': str(quote.loc[int(start_idx), 'Date']),
                    'seg_bars': int(valid_end_idx - start_idx + 1),
                    'progress_count': 0,
                    'progress_min': np.nan,
                    'progress_min_abs_index': np.nan,
                    'non_positive_count': 0,
                    'invalid_reason': 'new_lower_low_reset',
                    'activated_before': 1 if end_idx >= start_idx + constraint_w else 0,
                    'state': 'reset_by_new_low',
                })
            return {
                'search_start': int(start_idx),
                'low_index': int(start_idx),
                'high_index': int(high_idx),
                'valid_end_index': int(valid_end_idx),
                'end_index': int(end_idx),
            }

        if end_idx < start_idx + constraint_w:
            if record_debug and debug_records is not None:
                debug_records.append({
                    'search_start': int(start_idx),
                    'search_start_date': str(quote.loc[int(start_idx), 'Date']),
                    'end_idx': int(end_idx),
                    'end_date': str(quote.loc[int(end_idx), 'Date']),
                    'low_idx': int(start_idx),
                    'low_date': str(quote.loc[int(start_idx), 'Date']),
                    'seg_bars': int(end_idx - start_idx + 1),
                    'progress_count': 0,
                    'progress_min': np.nan,
                    'progress_min_abs_index': np.nan,
                    'non_positive_count': 0,
                    'invalid_reason': 'segment_too_short',
                    'activated_before': 0,
                    'state': 'segment_too_short',
                })
            continue

        seg = quote.iloc[start_idx:end_idx + 1]
        progress_offsets, progress_x, progress_values, invalid_reason = compute_progress_scan_long(
            seg, constraint_w
        )
        progress_count = int(len(progress_values))
        progress_min = np.nan
        progress_min_abs_index = np.nan
        non_positive_count = 0
        state = 'window_check'

        if progress_count > 0:
            min_pos = int(np.argmin(progress_values))
            progress_min = float(progress_values[min_pos])
            progress_min_abs_index = int(progress_x[min_pos])
            non_positive_count = int(np.sum(progress_values <= 0))

        if invalid_reason or progress_count == 0:
            if record_debug and debug_records is not None:
                debug_records.append({
                    'search_start': int(start_idx),
                    'search_start_date': str(quote.loc[int(start_idx), 'Date']),
                    'end_idx': int(end_idx),
                    'end_date': str(quote.loc[int(end_idx), 'Date']),
                    'low_idx': int(start_idx),
                    'low_date': str(quote.loc[int(start_idx), 'Date']),
                    'seg_bars': int(len(seg)),
                    'progress_count': progress_count,
                    'progress_min': progress_min,
                    'progress_min_abs_index': progress_min_abs_index,
                    'non_positive_count': non_positive_count,
                    'invalid_reason': invalid_reason or 'progress_unavailable',
                    'activated_before': 0,
                    'state': 'progress_unavailable',
                })
            continue

        current_progress = float(progress_values[-1])
        if current_progress <= 0:
            valid_end_idx = int(end_idx - 1)
            valid_slice = quote.iloc[start_idx:valid_end_idx + 1]
            if len(valid_slice) == 0:
                return None
            high_idx = int(valid_slice['high'].idxmax())
            if record_debug and debug_records is not None:
                debug_records.append({
                    'search_start': int(start_idx),
                    'search_start_date': str(quote.loc[int(start_idx), 'Date']),
                    'end_idx': int(end_idx),
                    'end_date': str(quote.loc[int(end_idx), 'Date']),
                    'low_idx': int(start_idx),
                    'low_date': str(quote.loc[int(start_idx), 'Date']),
                    'seg_bars': int(valid_end_idx - start_idx + 1),
                    'progress_count': progress_count,
                    'progress_min': progress_min,
                    'progress_min_abs_index': progress_min_abs_index,
                    'non_positive_count': non_positive_count,
                    'invalid_reason': 'non_positive_progress',
                    'activated_before': 1,
                    'state': 'fail_on_latest_window',
                })
            return {
                'search_start': int(start_idx),
                'low_index': int(start_idx),
                'high_index': int(high_idx),
                'valid_end_index': int(valid_end_idx),
                'end_index': int(end_idx),
            }

        if record_debug and debug_records is not None:
            debug_records.append({
                'search_start': int(start_idx),
                'search_start_date': str(quote.loc[int(start_idx), 'Date']),
                'end_idx': int(end_idx),
                'end_date': str(quote.loc[int(end_idx), 'Date']),
                'low_idx': int(start_idx),
                'low_date': str(quote.loc[int(start_idx), 'Date']),
                'seg_bars': int(len(seg)),
                'progress_count': progress_count,
                'progress_min': progress_min,
                'progress_min_abs_index': progress_min_abs_index,
                'non_positive_count': non_positive_count,
                'invalid_reason': '',
                'activated_before': 1,
                'state': 'valid_active',
            })

    valid_end_idx = int(n - 1)
    valid_slice = quote.iloc[start_idx:valid_end_idx + 1]
    if len(valid_slice) == 0:
        return None
    high_idx = int(valid_slice['high'].idxmax())
    return {
        'search_start': int(start_idx),
        'low_index': int(start_idx),
        'high_index': int(high_idx),
        'valid_end_index': int(valid_end_idx),
        'end_index': int(valid_end_idx),
    }


def build_trend_analysis_df_from_constraint_window(
        quote: pd.DataFrame,
        constraint_w: int):
    records = []
    debug_records = []
    n = len(quote)
    segment_id = 1
    search_start = 0

    while search_start < n - 1:
        segment_meta = find_next_trend_segment_by_constraint_window(
            quote=quote,
            search_start=search_start,
            constraint_w=constraint_w,
            debug_records=debug_records,
        )
        if segment_meta is None:
            break

        low_idx = int(segment_meta['low_index'])
        high_idx = int(segment_meta['high_index'])
        valid_end_idx = int(segment_meta['valid_end_index'])
        end_idx = int(segment_meta['end_index'])
        seg = quote.iloc[low_idx:high_idx + 1].copy()

        ols_stats = compute_ols_trend_line_stats_long(seg)
        reference_speed = ols_stats['trend_line_speed_pct_per_bar']
        speed_reference_source = 'ols'
        if (not np.isfinite(reference_speed)) or reference_speed <= 0:
            reference_speed = ols_stats['segment_speed_pct_per_bar']
            speed_reference_source = 'segment_avg'

        progress_offsets, progress_x, progress_values, invalid_reason = (
            compute_progress_scan_long(seg, constraint_w)
        )
        has_valid_progress = (
            invalid_reason == ''
            and len(progress_values) > 0
            and np.all(progress_values > 0)
        )

        fit_y_pct = np.nan
        y_speed_pct_per_bar = np.nan
        fit_entry_index = np.nan
        fit_entry_date = ''
        fit_entry_price = np.nan
        entry_delay_bars = np.nan
        fitted_return_pct = np.nan
        capture_ratio = np.nan
        mean_progress_pct = np.nan
        max_progress_pct = np.nan
        std_progress_pct = np.nan
        std_to_y_ratio = np.nan
        range_to_y_ratio = np.nan
        offset_score = np.nan
        best_offset_score = np.nan
        optimal_window_valid = 0
        optimal_window_invalid_reason = invalid_reason or 'constraint_progress_unavailable'

        low_price = float(quote.loc[low_idx, 'low'])
        high_price = float(quote.loc[high_idx, 'high'])
        price_span = high_price - low_price

        if has_valid_progress:
            fit_y_pct = float(progress_values.min())
            y_speed_pct_per_bar = fit_y_pct / float(constraint_w)
            first_offset = int(progress_offsets[0])
            fit_entry_index = int(progress_x[0])
            fit_entry_date = str(seg['Date'].iloc[first_offset])
            fit_entry_price = float(seg['close'].iloc[first_offset])
            entry_delay_bars = int(first_offset)
            fitted_return_pct = (
                (high_price / fit_entry_price - 1.0) * 100.0
                if fit_entry_price != 0 else np.nan
            )
            capture_ratio = (
                (high_price - fit_entry_price) / price_span
                if price_span > 0 else np.nan
            )
            mean_progress_pct = float(progress_values.mean())
            max_progress_pct = float(progress_values.max())
            std_progress_pct = float(progress_values.std(ddof=0))
            if fit_y_pct > 0:
                std_to_y_ratio = std_progress_pct / fit_y_pct
                range_to_y_ratio = (max_progress_pct - fit_y_pct) / fit_y_pct
            if np.isfinite(reference_speed) and reference_speed > 0:
                offset_values = np.abs(
                    (progress_values / float(constraint_w)) / reference_speed - 1.0
                )
                offset_score = float(offset_values.mean())
                best_offset_score = offset_score
            optimal_window_valid = 1
            optimal_window_invalid_reason = ''

        records.append({
            'trade_id': int(segment_id),
            'entry_index': int(segment_meta['search_start']),
            'entry_date': str(quote.loc[int(segment_meta['search_start']), 'Date']),
            'exit_index': valid_end_idx,
            'exit_date': str(quote.loc[valid_end_idx, 'Date']),
            'end_index': end_idx,
            'end_date': str(quote.loc[end_idx, 'Date']),
            'end_price': float(quote.loc[end_idx, 'close']),
            'search_start': int(segment_meta['search_start']),
            'low_index': low_idx,
            'low_date': str(quote.loc[low_idx, 'Date']),
            'low_price': low_price,
            'high_index': high_idx,
            'high_date': str(quote.loc[high_idx, 'Date']),
            'high_price': high_price,
            'trend_end_index': valid_end_idx,
            'trend_end_date': str(quote.loc[valid_end_idx, 'Date']),
            'duration_bars': int(high_idx - low_idx),
            'fit_segment_bars': int(high_idx - low_idx + 1),
            'total_return_pct': round(
                (high_price / low_price - 1.0) * 100.0, 4
            ) if low_price != 0 else np.nan,
            'constraint_w_bars': int(constraint_w),
            'optimal_w_bars': int(constraint_w),
            'ols_slope': round_or_nan(ols_stats['ols_slope']),
            'ols_intercept': round_or_nan(ols_stats['ols_intercept']),
            'ols_r_squared': round_or_nan(ols_stats['ols_r_squared']),
            'speed_reference_source': speed_reference_source,
            'reference_speed_pct_per_bar': round_or_nan(reference_speed),
            'trend_line_speed_pct_per_bar': round_or_nan(
                ols_stats['trend_line_speed_pct_per_bar']
            ),
            'segment_speed_pct_per_bar': round_or_nan(
                ols_stats['segment_speed_pct_per_bar']
            ),
            'first_feasible_w_bars': int(constraint_w),
            'best_offset_w_bars': int(constraint_w),
            'best_offset_score': round_or_nan(best_offset_score),
            'offset_score': round_or_nan(offset_score),
            'improvement_capture': np.nan,
            'fit_y_pct': round_or_nan(fit_y_pct),
            'y_speed_pct_per_bar': round_or_nan(y_speed_pct_per_bar),
            'fit_entry_index': (
                int(fit_entry_index) if pd.notna(fit_entry_index) else np.nan
            ),
            'fit_entry_date': fit_entry_date,
            'fit_entry_price': round_or_nan(fit_entry_price),
            'entry_delay_bars': (
                int(entry_delay_bars) if pd.notna(entry_delay_bars) else np.nan
            ),
            'fitted_return_pct': round_or_nan(fitted_return_pct),
            'capture_ratio': round_or_nan(capture_ratio),
            'progress_count': int(len(progress_values)) if has_valid_progress else 0,
            'mean_progress_pct': round_or_nan(mean_progress_pct),
            'max_progress_pct': round_or_nan(max_progress_pct),
            'std_progress_pct': round_or_nan(std_progress_pct),
            'std_to_y_ratio': round_or_nan(std_to_y_ratio),
            'range_to_y_ratio': round_or_nan(range_to_y_ratio),
            'optimal_window_valid': int(optimal_window_valid),
            'optimal_window_invalid_reason': optimal_window_invalid_reason,
        })

        segment_id += 1
        next_search_start = int(high_idx) + 1
        if next_search_start <= search_start:
            next_search_start = search_start + 1
        search_start = next_search_start

    return pd.DataFrame(records), pd.DataFrame(), pd.DataFrame(debug_records)


def build_debug_reset_segments(debug_df: pd.DataFrame,
                               quote: pd.DataFrame):
    if debug_df is None or len(debug_df) == 0:
        return []

    debug_df = debug_df.sort_values(['search_start', 'end_idx']).reset_index(drop=True)
    segments = []

    for search_start, group in debug_df.groupby('search_start', sort=True):
        active_low_idx = None
        active_last_end_idx = None

        for _, row in group.iterrows():
            state = str(row.get('state', ''))
            low_idx = int(row['low_idx']) if pd.notna(row['low_idx']) else None
            end_idx = int(row['end_idx']) if pd.notna(row['end_idx']) else None

            if state == 'valid_active':
                if active_low_idx is None:
                    active_low_idx = low_idx
                active_last_end_idx = end_idx
                continue

            if (
                active_low_idx is not None
                and active_last_end_idx is not None
                and state in ('segment_too_short', 'fail_before_activation', 'reset_by_new_low')
                and low_idx is not None
                and low_idx > active_low_idx
            ):
                valid_slice = quote.iloc[active_low_idx:active_last_end_idx + 1]
                if len(valid_slice) >= 2:
                    high_idx = int(valid_slice['high'].idxmax())
                    if high_idx > active_low_idx:
                        segments.append({
                            'search_start': int(search_start),
                            'low_index': int(active_low_idx),
                            'high_index': int(high_idx),
                            'valid_end_index': int(active_last_end_idx),
                            'reset_end_index': int(end_idx),
                            'reset_low_index': int(low_idx),
                            'state': 'reset_by_new_low',
                        })
                active_low_idx = None
                active_last_end_idx = None

    return segments


def build_trend_analysis_df(quote: pd.DataFrame,
                            trade_extreme_df: pd.DataFrame,
                            w_min: int,
                            w_max: int,
                            min_samples: int,
                            improvement_capture_target: float):
    records = []
    window_scan_records = []

    for idx, row in trade_extreme_df.iterrows():
        high_idx = int(row['max_profit_index'])

        search_start = int(row['entry_index'])

        search_slice = quote.iloc[search_start:high_idx + 1]
        if len(search_slice) == 0:
            continue

        low_bar_idx = int(search_slice['low'].idxmin())
        low_price = float(quote.loc[low_bar_idx, 'low'])
        high_price = float(row['max_profit_price'])

        if high_idx <= low_bar_idx:
            continue

        fit_seg = quote.iloc[low_bar_idx:high_idx + 1]
        stats, scan_rows = compute_optimal_window_stats_long(
            seg=fit_seg,
            w_min=w_min,
            w_max=w_max,
            min_samples=min_samples,
            improvement_capture_target=improvement_capture_target,
        )

        trade_id = idx + 1
        duration = high_idx - low_bar_idx
        fit_segment_bars = high_idx - low_bar_idx + 1

        for scan_row in scan_rows:
            window_scan_records.append({
                'trade_id': trade_id,
                'entry_index': int(row['entry_index']),
                'entry_date': str(row.get('entry_date', '')),
                'low_index': low_bar_idx,
                'high_index': high_idx,
                'exit_index': int(row['exit_index']),
                'duration_bars': duration,
                'fit_segment_bars': fit_segment_bars,
                'ols_slope': round_or_nan(scan_row['ols_slope']),
                'ols_intercept': round_or_nan(scan_row['ols_intercept']),
                'ols_r_squared': round_or_nan(scan_row['ols_r_squared']),
                'speed_reference_source': scan_row['speed_reference_source'],
                'reference_speed_pct_per_bar': round_or_nan(
                    scan_row['reference_speed_pct_per_bar']),
                'trend_line_speed_pct_per_bar': round_or_nan(
                    scan_row['trend_line_speed_pct_per_bar']),
                'segment_speed_pct_per_bar': round_or_nan(
                    scan_row['segment_speed_pct_per_bar']),
                'w_bars': int(scan_row['w_bars']),
                'window_valid': int(scan_row['window_valid']),
                'invalid_reason': scan_row['invalid_reason'],
                'offset_score': round_or_nan(scan_row['offset_score']),
                'improvement_capture': round_or_nan(
                    scan_row['improvement_capture']),
                'first_feasible_flag': int(scan_row['first_feasible_flag']),
                'best_offset_flag': int(scan_row['best_offset_flag']),
                'selected_flag': int(scan_row['selected_flag']),
                'fit_y_pct': round_or_nan(scan_row['fit_y_pct']),
                'y_speed_pct_per_bar': round_or_nan(
                    scan_row['y_speed_pct_per_bar']),
                'fit_entry_index': (
                    int(scan_row['fit_entry_index'])
                    if pd.notna(scan_row['fit_entry_index']) else np.nan
                ),
                'fit_entry_date': scan_row['fit_entry_date'],
                'fit_entry_price': round_or_nan(scan_row['fit_entry_price']),
                'entry_delay_bars': (
                    int(scan_row['entry_delay_bars'])
                    if pd.notna(scan_row['entry_delay_bars']) else np.nan
                ),
                'fitted_return_pct': round_or_nan(scan_row['fitted_return_pct']),
                'capture_ratio': round_or_nan(scan_row['capture_ratio']),
                'progress_count': int(scan_row['progress_count']),
                'mean_progress_pct': round_or_nan(scan_row['mean_progress_pct']),
                'max_progress_pct': round_or_nan(scan_row['max_progress_pct']),
                'std_progress_pct': round_or_nan(scan_row['std_progress_pct']),
                'range_above_y_pct': round_or_nan(scan_row['range_above_y_pct']),
                'std_to_y_ratio': round_or_nan(scan_row['std_to_y_ratio']),
                'range_to_y_ratio': round_or_nan(scan_row['range_to_y_ratio']),
            })

        records.append({
            'trade_id': trade_id,
            'entry_index': int(row['entry_index']),
            'entry_date': str(row.get('entry_date', '')),
            'exit_index': int(row['exit_index']),
            'exit_date': str(row.get('exit_date', '')),
            'search_start': search_start,
            'low_index': low_bar_idx,
            'low_date': str(quote.loc[low_bar_idx, 'Date']),
            'low_price': low_price,
            'high_index': high_idx,
            'high_date': str(row.get('max_profit_date', '')),
            'high_price': high_price,
            'duration_bars': duration,
            'fit_segment_bars': fit_segment_bars,
            'total_return_pct': round(
                (high_price / low_price - 1.0) * 100.0, 4)
                if low_price != 0 else np.nan,
            'optimal_w_bars': (
                int(stats['optimal_w_bars'])
                if pd.notna(stats['optimal_w_bars']) else np.nan
            ),
            'ols_slope': round_or_nan(stats['ols_slope']),
            'ols_intercept': round_or_nan(stats['ols_intercept']),
            'ols_r_squared': round_or_nan(stats['ols_r_squared']),
            'speed_reference_source': stats['speed_reference_source'],
            'reference_speed_pct_per_bar': round_or_nan(
                stats['reference_speed_pct_per_bar']),
            'trend_line_speed_pct_per_bar': round_or_nan(
                stats['trend_line_speed_pct_per_bar']),
            'segment_speed_pct_per_bar': round_or_nan(
                stats['segment_speed_pct_per_bar']),
            'first_feasible_w_bars': (
                int(stats['first_feasible_w_bars'])
                if pd.notna(stats['first_feasible_w_bars']) else np.nan
            ),
            'best_offset_w_bars': (
                int(stats['best_offset_w_bars'])
                if pd.notna(stats['best_offset_w_bars']) else np.nan
            ),
            'best_offset_score': round_or_nan(stats['best_offset_score']),
            'offset_score': round_or_nan(stats['offset_score']),
            'improvement_capture': round_or_nan(stats['improvement_capture']),
            'fit_y_pct': round_or_nan(stats['fit_y_pct']),
            'y_speed_pct_per_bar': round_or_nan(
                stats['y_speed_pct_per_bar']),
            'fit_entry_index': (
                int(stats['fit_entry_index'])
                if pd.notna(stats['fit_entry_index']) else np.nan
            ),
            'fit_entry_date': stats['fit_entry_date'],
            'fit_entry_price': round_or_nan(stats['fit_entry_price']),
            'entry_delay_bars': (
                int(stats['entry_delay_bars'])
                if pd.notna(stats['entry_delay_bars']) else np.nan
            ),
            'fitted_return_pct': round_or_nan(stats['fitted_return_pct']),
            'capture_ratio': round_or_nan(stats['capture_ratio']),
            'progress_count': (
                int(stats['progress_count'])
                if pd.notna(stats['progress_count']) else np.nan
            ),
            'mean_progress_pct': round_or_nan(stats['mean_progress_pct']),
            'max_progress_pct': round_or_nan(stats['max_progress_pct']),
            'std_progress_pct': round_or_nan(stats['std_progress_pct']),
            'std_to_y_ratio': round_or_nan(stats['std_to_y_ratio']),
            'range_to_y_ratio': round_or_nan(stats['range_to_y_ratio']),
            'optimal_window_valid': int(stats['optimal_window_valid']),
            'optimal_window_invalid_reason': (
                stats['optimal_window_invalid_reason']
            ),
        })

    return pd.DataFrame(records), pd.DataFrame(window_scan_records)


def export_interactive_html_long_no_wd(
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

    fig_html.add_trace(go.Candlestick(
        x=x_index,
        open=underlying1['open'] / factor * 100,
        high=underlying1['high'] / factor * 100,
        low=underlying1['low'] / factor * 100,
        close=underlying1['close'] / factor * 100,
        text=build_candlestick_hovertext(underlying1, factor),
        name='candles',
        showlegend=False,
        hoverinfo='text',
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
            bgcolor='rgba(255, 255, 255, 0.50)',
            bordercolor='rgba(0, 0, 0, 0.45)',
            font=dict(color='black')
        )
    )

    html_dir = get_html_output_dir(file_name, BACKTEST_HTML_FOLDER)
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(html_dir, save_name + ' LongNoWD interactive.html')
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
        'fill:rgba(255,255,255,0.50) !important;'
        'fill-opacity:0.50 !important;'
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


def legacy_export_trend_analysis_html(
        file_name: str,
        save_name: str,
        underlying1: pd.DataFrame,
        trend_df: pd.DataFrame,
        factor: float):
    """趋势分析专用 HTML：hover 显示 high/low 点，连线 low->high。"""
    if go is None:
        print('[HTML] plotly is not installed, skip trend html export.')
        return
    if len(trend_df) == 0:
        print('[HTML] no trend data, skip trend html export.')
        return

    fig_html = go.Figure()
    x_index = underlying1.index.to_numpy()
    x_min = int(x_index[0]) if len(x_index) > 0 else 0
    x_max = int(x_index[-1]) if len(x_index) > 0 else 1
    x_span = max(1, x_max - x_min + 1)
    x_left_pad = max(1, int(round(x_span * 0.006)))
    x_right_pad = max(1, int(round(x_span * 0.010)))

    # K 线
    fig_html.add_trace(go.Candlestick(
        x=x_index,
        open=underlying1['open'] / factor * 100,
        high=underlying1['high'] / factor * 100,
        low=underlying1['low'] / factor * 100,
        close=underlying1['close'] / factor * 100,
        text=build_candlestick_hovertext(underlying1, factor),
        name='candles',
        showlegend=False,
        hoverinfo='text',
        increasing=dict(
            line=dict(color='salmon', width=0.8),
            fillcolor='rgba(250, 128, 114, 0.28)'
        ),
        decreasing=dict(
            line=dict(color='#2ca02c', width=0.8),
            fillcolor='rgba(44, 160, 44, 0.28)'
        )
    ))

    # Low 点 (蓝色)
    low_texts = []
    for _, row in trend_df.iterrows():
        low_texts.append(
            f"trade: {int(row['trade_id'])}<br>"
            f"low_date: {row['low_date']}<br>"
            f"low_price: {row['low_price']}<br>"
            f"low_index: {int(row['low_index'])}<br>"
            f"---<br>"
            f"search_start: {int(row['search_start'])}<br>"
            f"trade_entry: {int(row['entry_index'])} ({row['entry_date'][:16]})<br>"
            f"trade_exit: {int(row['exit_index'])} ({row['exit_date'][:16]})"
        )
    fig_html.add_trace(go.Scatter(
        x=trend_df['low_index'].astype(int).tolist(),
        y=(trend_df['low_price'] / factor * 100).tolist(),
        mode='markers',
        marker=dict(color='#1F77B4', size=5),
        name='low',
        text=low_texts,
        hovertemplate='%{text}<extra></extra>'
    ))

    # High 点 (橙色)
    high_texts = []
    for _, row in trend_df.iterrows():
        opt_w = (
            f"{int(row['optimal_w_bars'])}"
            if pd.notna(row['optimal_w_bars']) else 'nan'
        )
        min_progress = (
            f"{row['min_progress_pct']:.4f}"
            if pd.notna(row['min_progress_pct']) else 'nan'
        )
        mean_progress = (
            f"{row['mean_progress_pct']:.4f}"
            if pd.notna(row['mean_progress_pct']) else 'nan'
        )
        max_progress = (
            f"{row['max_progress_pct']:.4f}"
            if pd.notna(row['max_progress_pct']) else 'nan'
        )
        std_progress = (
            f"{row['std_progress_pct']:.4f}"
            if pd.notna(row['std_progress_pct']) else 'nan'
        )
        std_to_min = (
            f"{row['std_to_min_ratio']:.4f}"
            if pd.notna(row['std_to_min_ratio']) else 'nan'
        )
        range_to_min = (
            f"{row['range_to_min_ratio']:.4f}"
            if pd.notna(row['range_to_min_ratio']) else 'nan'
        )
        progress_count = (
            f"{int(row['progress_count'])}"
            if pd.notna(row['progress_count']) else 'nan'
        )
        valid_flag = int(row['optimal_window_valid'])
        invalid_reason = row['optimal_window_invalid_reason'] or ''
        high_texts.append(
            f"trade: {int(row['trade_id'])}<br>"
            f"high_date: {row['high_date']}<br>"
            f"high_price: {row['high_price']}<br>"
            f"high_index: {int(row['high_index'])}<br>"
            f"duration: {int(row['duration_bars'])} bars<br>"
            f"return: {row['total_return_pct']:.4f}%<br>"
            f"optimal_w_bars: {opt_w}<br>"
            f"min_progress_pct: {min_progress}<br>"
            f"mean_progress_pct: {mean_progress}<br>"
            f"max_progress_pct: {max_progress}<br>"
            f"std_progress_pct: {std_progress}<br>"
            f"std_to_min_ratio: {std_to_min}<br>"
            f"range_to_min_ratio: {range_to_min}<br>"
            f"progress_count: {progress_count}<br>"
            f"window_valid: {valid_flag}<br>"
            f"invalid_reason: {invalid_reason}<br>"
            f"---<br>"
            f"trade_entry: {int(row['entry_index'])} ({row['entry_date'][:16]})<br>"
            f"trade_exit: {int(row['exit_index'])} ({row['exit_date'][:16]})"
        )
    fig_html.add_trace(go.Scatter(
        x=trend_df['high_index'].astype(int).tolist(),
        y=(trend_df['high_price'] / factor * 100).tolist(),
        mode='markers',
        marker=dict(color='orange', size=5),
        name='high',
        text=high_texts,
        hovertemplate='%{text}<extra></extra>'
    ))

    # Entry/Exit 点 (50% 透明黑色小点)
    entry_x = trend_df['entry_index'].astype(int).tolist()
    exit_x = trend_df['exit_index'].astype(int).tolist()
    # entry 价格: quote 在 entry_index 的 open
    entry_y = []
    exit_y = []
    for _, row in trend_df.iterrows():
        ei = int(row['entry_index'])
        xi = int(row['exit_index'])
        entry_y.append(float(underlying1.loc[ei, 'open']) / factor * 100)
        exit_y.append(float(underlying1.loc[xi, 'close']) / factor * 100)
    fig_html.add_trace(go.Scatter(
        x=entry_x, y=entry_y,
        mode='markers',
        marker=dict(color='rgba(0,0,0,0.5)', size=4),
        name='entry',
        hoverinfo='skip'
    ))
    fig_html.add_trace(go.Scatter(
        x=exit_x, y=exit_y,
        mode='markers',
        marker=dict(color='rgba(0,0,0,0.5)', size=4,
                    symbol='x'),
        name='exit',
        hoverinfo='skip'
    ))

    # Entry 点 (蓝色小点 50% 透明)
    entry_x = trend_df['entry_index'].astype(int).tolist()
    entry_y = [float(underlying1.loc[int(r['entry_index']), 'open']) / factor * 100
               for _, r in trend_df.iterrows()]
    fig_html.add_trace(go.Scatter(
        x=entry_x, y=entry_y,
        mode='markers',
        marker=dict(color='rgba(31,119,180,0.5)', size=4),
        name='entry',
        hoverinfo='skip'
    ))

    # Exit 点 (黑色小点 50% 透明)
    exit_x = trend_df['exit_index'].astype(int).tolist()
    exit_y = [float(underlying1.loc[int(r['exit_index']), 'close']) / factor * 100
              for _, r in trend_df.iterrows()]
    fig_html.add_trace(go.Scatter(
        x=exit_x, y=exit_y,
        mode='markers',
        marker=dict(color='rgba(0,0,0,0.5)', size=4, symbol='x'),
        name='exit',
        hoverinfo='skip'
    ))

    # OLS 拟合线
    fig_html.update_layout(
        title=None,
        template='plotly_white',
        autosize=True,
        hovermode='closest',
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    xanchor='left', x=0),
        xaxis=dict(
            title=None, tickfont=dict(size=10), showgrid=False,
            rangeslider=dict(visible=False),
            range=[x_min - x_left_pad, x_max + x_right_pad],
            autorange=False, showspikes=False
        ),
        yaxis=dict(
            title=None, tickfont=dict(size=10), showgrid=False,
            showspikes=False
        ),
        margin=dict(l=42, r=25, t=38, b=45, pad=0),
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.50)',
            bordercolor='rgba(0, 0, 0, 0.45)',
            font=dict(color='black')
        )
    )

    html_dir = get_html_output_dir(file_name, TREND_ANALYSIS_HTML_FOLDER)
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(
        html_dir, save_name + ' trend_analysis interactive.html')
    html_text = fig_html.to_html(
        include_plotlyjs=True, full_html=True,
        default_width='100vw', default_height='100vh',
        config={'responsive': True, 'displayModeBar': False,
                'displaylogo': False}
    )
    html_text = html_text.replace(
        '<head>',
        '<head><style>'
        'html,body{width:100%;height:100%;margin:0;padding:0;overflow:hidden;}'
        '.plotly-graph-div{width:100vw !important;height:100vh !important;}'
        '.hoverlayer .hovertext .bg,'
        '.hoverlayer .hovertext rect,'
        '.hoverlayer .hovertext path{'
        'fill:rgba(255,255,255,0.50) !important;'
        'fill-opacity:0.50 !important;'
        'stroke:rgba(0,0,0,0.45) !important;'
        'stroke-opacity:0.45 !important;}'
        '.hoverlayer .hovertext{opacity:1 !important;}'
        '.hoverlayer .hovertext text{fill:#000 !important;}'
        '</style>',
        1
    )
    html_text = html_text.replace(
        '<body>', '<body style="margin:0;overflow:hidden;">', 1)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print('[HTML] saved trend analysis chart.')


def export_trend_analysis_html(
        file_name: str,
        save_name: str,
        underlying1: pd.DataFrame,
        trend_df: pd.DataFrame,
        factor: float,
        debug_df: pd.DataFrame | None = None):
    if len(trend_df) == 0:
        print('[HTML] no trend data, skip trend html export.')
        return

    trend_display_df = trend_df.copy()
    atr_df = build_trend_atr_multiple_df(underlying1, trend_display_df)
    if len(atr_df) > 0:
        trend_display_df = trend_display_df.merge(
            atr_df[['trade_id', 'trend_atr_multiple']],
            on='trade_id',
            how='left'
        )
    else:
        trend_display_df['trend_atr_multiple'] = np.nan

    x_index = underlying1.index.to_numpy()
    x_min = int(x_index[0]) if len(x_index) > 0 else 0
    x_max = int(x_index[-1]) if len(x_index) > 0 else 1
    x_span = max(1, x_max - x_min + 1)
    x_left_pad = max(1, int(round(x_span * 0.006)))
    x_right_pad = max(1, int(round(x_span * 0.010)))

    segment_records = []
    for _, row in trend_display_df.iterrows():
        trend_atr_multiple = (
            float(row['trend_atr_multiple'])
            if 'trend_atr_multiple' in row.index and pd.notna(row['trend_atr_multiple'])
            else np.nan
        )
        if (not np.isfinite(trend_atr_multiple)
                or trend_atr_multiple < TREND_HTML_MIN_VISIBLE_MULTIPLE):
            continue

        trade_id = int(row['trade_id'])
        search_start = int(row['search_start'])
        low_index = int(row['low_index'])
        high_index = int(row['high_index'])
        end_index = int(row['end_index']) if pd.notna(row['end_index']) else high_index
        duration_bars = int(row['duration_bars'])
        low_price = float(row['low_price'])
        high_price = float(row['high_price'])
        end_price = (
            float(row['end_price']) if pd.notna(row['end_price'])
            else float(underlying1.loc[end_index, 'close'])
        )
        total_return_pct = float(row['total_return_pct'])
        reference_speed = (
            f"{float(row['reference_speed_pct_per_bar']):.4f}"
            if pd.notna(row['reference_speed_pct_per_bar']) else 'nan'
        )
        trend_line_speed = (
            f"{float(row['trend_line_speed_pct_per_bar']):.4f}"
            if pd.notna(row['trend_line_speed_pct_per_bar']) else 'nan'
        )
        segment_speed = (
            f"{float(row['segment_speed_pct_per_bar']):.4f}"
            if pd.notna(row['segment_speed_pct_per_bar']) else 'nan'
        )
        ols_r2 = (
            f"{float(row['ols_r_squared']):.4f}"
            if pd.notna(row['ols_r_squared']) else 'nan'
        )
        low_text = (
            f"segment: {trade_id}<br>"
            f"search_start: {search_start}<br>"
            f"low_date: {row['low_date']}<br>"
            f"low_price: {low_price:.4f}<br>"
            f"low_index: {low_index}"
        )
        high_text = (
            f"multiple: {trend_atr_multiple:.4f}<br>"
            f"segment: {trade_id}<br>"
            f"high_date: {row['high_date']}<br>"
            f"high_price: {high_price:.4f}<br>"
            f"high_index: {high_index}<br>"
            f"duration_bars: {duration_bars}<br>"
            f"total_return_pct: {total_return_pct:.4f}%<br>"
            f"speed_reference_source: {row['speed_reference_source']}<br>"
            f"reference_speed_pct_per_bar: {reference_speed}<br>"
            f"trend_line_speed_pct_per_bar: {trend_line_speed}<br>"
            f"segment_speed_pct_per_bar: {segment_speed}<br>"
            f"ols_r_squared: {ols_r2}"
        )
        end_text = (
            f"segment: {trade_id}<br>"
            f"end_date: {row['end_date']}<br>"
            f"end_price: {end_price:.4f}<br>"
            f"end_index: {end_index}"
        )

        trend_line_y0 = None
        trend_line_y1 = None
        trend_seg = underlying1.iloc[low_index:high_index + 1]
        if len(trend_seg) >= 2 and pd.notna(row['ols_slope']) and pd.notna(row['ols_intercept']):
            t = np.arange(len(trend_seg), dtype=float)
            fitted = row['ols_slope'] * t + row['ols_intercept']
            trend_line_y0 = float(fitted[0] / factor * 100)
            trend_line_y1 = float(fitted[-1] / factor * 100)

        end_link_y0 = None
        end_link_y1 = None
        if end_index > high_index:
            end_link_y0 = float(high_price / factor * 100)
            end_link_y1 = float(end_price / factor * 100)

        segment_records.append({
            'trade_id': trade_id,
            'trend_atr_multiple': trend_atr_multiple,
            'low_index': low_index,
            'low_y': float(low_price / factor * 100),
            'low_text': low_text,
            'high_index': high_index,
            'high_y': float(high_price / factor * 100),
            'high_text': high_text,
            'end_index': end_index,
            'end_y': float(end_price / factor * 100),
            'end_text': end_text,
            'trend_line_y0': trend_line_y0,
            'trend_line_y1': trend_line_y1,
            'end_link_y0': end_link_y0,
            'end_link_y1': end_link_y1,
        })

    debug_line_x = []
    debug_line_y = []
    debug_reset_x = []
    debug_reset_y = []
    debug_reset_texts = []
    if DEBUG_TREND_SEARCH and debug_df is not None:
        debug_segments = build_debug_reset_segments(debug_df, underlying1)
        for seg_meta in debug_segments:
            low_index = int(seg_meta['low_index'])
            high_index = int(seg_meta['high_index'])
            valid_slice = underlying1.iloc[low_index:high_index + 1]
            if len(valid_slice) < 2:
                continue
            ols_stats = compute_ols_trend_line_stats_long(valid_slice)
            if pd.isna(ols_stats['ols_slope']) or pd.isna(ols_stats['ols_intercept']):
                continue
            t = np.arange(len(valid_slice), dtype=float)
            fitted = ols_stats['ols_slope'] * t + ols_stats['ols_intercept']
            debug_line_x.extend([low_index, high_index, None])
            debug_line_y.extend([
                float(fitted[0] / factor * 100),
                float(fitted[-1] / factor * 100),
                None,
            ])
            debug_reset_x.append(int(seg_meta['reset_end_index']))
            debug_reset_y.append(
                float(underlying1.loc[int(seg_meta['reset_end_index']), 'close']) / factor * 100
            )
            debug_reset_texts.append(
                f"debug_state: reset_by_new_low<br>"
                f"search_start: {int(seg_meta['search_start'])}<br>"
                f"candidate_low_index: {low_index}<br>"
                f"candidate_high_index: {high_index}<br>"
                f"candidate_valid_end_index: {int(seg_meta['valid_end_index'])}<br>"
                f"reset_end_index: {int(seg_meta['reset_end_index'])}<br>"
                f"reset_low_index: {int(seg_meta['reset_low_index'])}"
            )

    candle_data = {
        'x': x_index.astype(int).tolist(),
        'open': (underlying1['open'] / factor * 100).astype(float).tolist(),
        'high': (underlying1['high'] / factor * 100).astype(float).tolist(),
        'low': (underlying1['low'] / factor * 100).astype(float).tolist(),
        'close': (underlying1['close'] / factor * 100).astype(float).tolist(),
        'text': build_candlestick_hovertext(underlying1, factor),
    }
    debug_data = {
        'line_x': debug_line_x,
        'line_y': debug_line_y,
        'reset_x': debug_reset_x,
        'reset_y': debug_reset_y,
        'reset_text': debug_reset_texts,
    }

    source_json = json.dumps(segment_records, ensure_ascii=False)
    candle_json = json.dumps(candle_data, ensure_ascii=False)
    debug_json = json.dumps(debug_data, ensure_ascii=False)

    html_dir = get_html_output_dir(file_name, TREND_ANALYSIS_HTML_FOLDER)
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(
        html_dir, save_name + ' trend_analysis interactive.html')
    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>trend analysis</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
html, body {{
    width: 100%;
    height: 100%;
    margin: 0;
    padding: 0;
    overflow: hidden;
    background: #ffffff;
}}
#app {{
    width: 100%;
    height: 100%;
    position: relative;
}}
#sidebar {{
    width: 286px;
    max-height: calc(100% - 28px);
    box-sizing: border-box;
    padding: 18px 16px 16px 16px;
    border: 1px solid rgba(0,0,0,0.10);
    border-radius: 16px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.14);
    background: rgba(248,249,250,0.97);
    backdrop-filter: blur(10px);
    transform: translateX(calc(-100% - 18px));
    transition: transform 0.22s ease;
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 6;
    overflow-y: auto;
    pointer-events: none;
}}
#sidebar.open {{
    transform: translateX(0);
    pointer-events: auto;
}}
#sidebar h2 {{
    margin: 0 0 14px 0;
    font-size: 18px;
    font-weight: 600;
    color: #20324d;
}}
#sidebar .field {{
    margin-bottom: 14px;
}}
#sidebar label {{
    display: block;
    margin-bottom: 6px;
    font-size: 13px;
    color: #22344f;
}}
#sidebar input,
#sidebar select {{
    width: 100%;
    box-sizing: border-box;
    padding: 8px 10px;
    border: 1px solid rgba(0,0,0,0.20);
    border-radius: 8px;
    font-size: 14px;
    background: rgba(255,255,255,0.96);
    color: #22344f;
}}
#sidebar input[type=number] {{
    appearance: textfield;
    -moz-appearance: textfield;
}}
#sidebar input[type=number]::-webkit-outer-spin-button,
#sidebar input[type=number]::-webkit-inner-spin-button {{
    -webkit-appearance: none;
    margin: 0;
}}
#sidebar .number-wrap {{
    display: flex;
    align-items: stretch;
    gap: 10px;
}}
#sidebar .number-wrap input {{
    flex: 1 1 auto;
}}
#sidebar .stepper {{
    flex: 0 0 40px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}}
#sidebar .step-btn {{
    flex: 1 1 auto;
    padding: 0;
    border: 1px solid rgba(0,0,0,0.16);
    border-radius: 8px;
    background: rgba(255,255,255,0.96);
    color: #20324d;
    font-size: 12px;
    cursor: pointer;
}}
#sidebar .step-btn:hover {{
    background: rgba(236,242,248,0.96);
}}
#sidebar .hint {{
    margin-top: 10px;
    font-size: 12px;
    line-height: 1.5;
    color: rgba(0,0,0,0.62);
}}
#toggle-btn {{
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 7;
    padding: 8px 12px;
    border: 1px solid rgba(0,0,0,0.18);
    border-radius: 10px;
    background: rgba(255,255,255,0.96);
    color: #20324d;
    font-size: 14px;
    cursor: pointer;
    box-shadow: 0 8px 22px rgba(0,0,0,0.12);
    transition: left 0.22s ease;
}}
#toggle-btn.sidebar-open {{
    left: 314px;
}}
#main {{
    width: 100%;
    height: 100%;
    position: relative;
}}
#summary {{
    margin-top: 14px;
    padding: 10px 12px;
    border-radius: 10px;
    background: rgba(255,255,255,0.92);
    color: #22344f;
    font-size: 13px;
    border: 1px solid rgba(0,0,0,0.10);
    line-height: 1.5;
}}
#chart {{
    width: 100%;
    height: 100%;
}}
.hoverlayer .hovertext .bg,
.hoverlayer .hovertext rect,
.hoverlayer .hovertext path {{
    fill: rgba(255,255,255,0.50) !important;
    fill-opacity: 0.50 !important;
    stroke: rgba(0,0,0,0.45) !important;
    stroke-opacity: 0.45 !important;
}}
.hoverlayer .hovertext {{
    opacity: 1 !important;
}}
.hoverlayer .hovertext text {{
    fill: #000 !important;
}}
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <h2>筛选设置</h2>
    <div class="field">
      <label for="multiple-input">最小 multiple</label>
      <input id="multiple-input" type="number" min="{TREND_HTML_MIN_VISIBLE_MULTIPLE:.1f}" step="0.1" value="{TREND_HTML_DEFAULT_MULTIPLE:.1f}">
    </div>
    <div class="hint">
      默认阈值为 {TREND_HTML_DEFAULT_MULTIPLE:.1f}。<br>
      小于 {TREND_HTML_MIN_VISIBLE_MULTIPLE:.1f} 的样本不会显示。<br>
      high point 的 hover 第一行显示 multiple。
    </div>
    <div id="summary"></div>
  </div>
  <button id="toggle-btn">筛选</button>
  <div id="main">
    <div id="chart"></div>
  </div>
</div>
<script>
const candleData = {candle_json};
const sourceSegments = {source_json};
const debugData = {debug_json};
const defaultMultiple = {TREND_HTML_DEFAULT_MULTIPLE:.1f};
const minVisibleMultiple = {TREND_HTML_MIN_VISIBLE_MULTIPLE:.1f};
const xMin = {x_min};
const xMax = {x_max};
const xLeftPad = {x_left_pad};
const xRightPad = {x_right_pad};
const sidebar = document.getElementById('sidebar');
const toggleBtn = document.getElementById('toggle-btn');
sidebar.innerHTML = ''
    + '<h2>筛选设置</h2>'
    + '<div class="field">'
    + '  <label for="multiple-input">最小 multiple</label>'
    + '  <div class="number-wrap">'
    + '    <input id="multiple-input" type="number" min="' + minVisibleMultiple.toFixed(1) + '" step="0.1" value="' + defaultMultiple.toFixed(1) + '">'
    + '    <div class="stepper">'
    + '      <button id="multiple-step-up" class="step-btn" type="button" aria-label="increase">▲</button>'
    + '      <button id="multiple-step-down" class="step-btn" type="button" aria-label="decrease">▼</button>'
    + '    </div>'
    + '  </div>'
    + '</div>'
    + '<div class="field">'
    + '  <label for="multiple-select">可用阈值</label>'
    + '  <select id="multiple-select"></select>'
    + '</div>'
    + '<div class="hint">'
    + '  默认阈值为 ' + defaultMultiple.toFixed(1) + '。<br>'
    + '  小于 ' + minVisibleMultiple.toFixed(1) + ' 的样本不会显示。<br>'
    + '  high point 的 hover 第一行显示 multiple。'
    + '</div>'
    + '<div id="summary"></div>';
toggleBtn.textContent = '筛选';
const multipleInput = document.getElementById('multiple-input');
const multipleSelect = document.getElementById('multiple-select');
const multipleStepUp = document.getElementById('multiple-step-up');
const multipleStepDown = document.getElementById('multiple-step-down');
const summaryBox = document.getElementById('summary');
const chartDiv = document.getElementById('chart');

function buildAvailableThresholds() {{
    const values = [];
    const seen = new Set();

    function pushValue(rawValue) {{
        const num = Number(rawValue);
        if (!Number.isFinite(num)) {{
            return;
        }}
        const rounded = Number(Math.max(minVisibleMultiple, num).toFixed(1));
        const key = rounded.toFixed(1);
        if (seen.has(key)) {{
            return;
        }}
        seen.add(key);
        values.push(rounded);
    }}

    pushValue(minVisibleMultiple);
    pushValue(defaultMultiple);
    sourceSegments.forEach(row => {{
        pushValue(row.trend_atr_multiple);
    }});
    values.sort((a, b) => a - b);
    return values;
}}

const availableThresholds = buildAvailableThresholds();

function syncMultipleSelect(threshold) {{
    const targetValue = threshold.toFixed(1);
    const existing = Array.from(multipleSelect.options).find(
        option => option.value === targetValue
    );
    const customOption = multipleSelect.querySelector('option[data-custom="1"]');

    if (existing) {{
        if (customOption) {{
            customOption.remove();
        }}
        multipleSelect.value = targetValue;
        return;
    }}

    if (customOption) {{
        customOption.value = targetValue;
        customOption.textContent = '当前值 ' + targetValue;
    }} else {{
        const option = document.createElement('option');
        option.value = targetValue;
        option.textContent = '当前值 ' + targetValue;
        option.setAttribute('data-custom', '1');
        multipleSelect.insertBefore(option, multipleSelect.firstChild);
    }}
    multipleSelect.value = targetValue;
}}

function populateMultipleSelect() {{
    multipleSelect.innerHTML = '';
    availableThresholds.forEach(value => {{
        const option = document.createElement('option');
        option.value = value.toFixed(1);
        option.textContent = value.toFixed(1);
        multipleSelect.appendChild(option);
    }});
}}

function normalizeThreshold(rawValue) {{
    let threshold = parseFloat(rawValue);
    if (!Number.isFinite(threshold)) {{
        threshold = defaultMultiple;
    }}
    threshold = Math.max(minVisibleMultiple, threshold);
    threshold = Number(threshold.toFixed(1));
    multipleInput.value = threshold.toFixed(1);
    syncMultipleSelect(threshold);
    return threshold;
}}

function buildSegmentTraces(filtered) {{
    const traces = [{{
        type: 'candlestick',
        x: candleData.x,
        open: candleData.open,
        high: candleData.high,
        low: candleData.low,
        close: candleData.close,
        text: candleData.text,
        name: 'candles',
        showlegend: false,
        hoverinfo: 'text',
        increasing: {{
            line: {{color: 'salmon', width: 0.8}},
            fillcolor: 'rgba(250, 128, 114, 0.28)'
        }},
        decreasing: {{
            line: {{color: '#2ca02c', width: 0.8}},
            fillcolor: 'rgba(44, 160, 44, 0.28)'
        }}
    }}];

    const lowX = [];
    const lowY = [];
    const lowText = [];
    const highX = [];
    const highY = [];
    const highText = [];
    const endX = [];
    const endY = [];
    const endText = [];
    const trendLineX = [];
    const trendLineY = [];
    const endLinkX = [];
    const endLinkY = [];

    filtered.forEach(row => {{
        lowX.push(row.low_index);
        lowY.push(row.low_y);
        lowText.push(row.low_text);
        highX.push(row.high_index);
        highY.push(row.high_y);
        highText.push(row.high_text);
        endX.push(row.end_index);
        endY.push(row.end_y);
        endText.push(row.end_text);

        if (row.trend_line_y0 !== null && row.trend_line_y1 !== null) {{
            trendLineX.push(row.low_index, row.high_index, null);
            trendLineY.push(row.trend_line_y0, row.trend_line_y1, null);
        }}
        if (row.end_link_y0 !== null && row.end_link_y1 !== null) {{
            endLinkX.push(row.high_index, row.end_index, null);
            endLinkY.push(row.end_link_y0, row.end_link_y1, null);
        }}
    }});

    if (lowX.length > 0) {{
        traces.push({{
            type: 'scatter',
            x: lowX,
            y: lowY,
            mode: 'markers',
            marker: {{color: '#1F77B4', size: 5}},
            name: 'low',
            text: lowText,
            hovertemplate: '%{{text}}<extra></extra>'
        }});
        traces.push({{
            type: 'scatter',
            x: highX,
            y: highY,
            mode: 'markers',
            marker: {{color: 'orange', size: 5}},
            name: 'high',
            text: highText,
            hovertemplate: '%{{text}}<extra></extra>'
        }});
        traces.push({{
            type: 'scatter',
            x: endX,
            y: endY,
            mode: 'markers',
            marker: {{color: 'rgba(255,99,71,0.60)', size: 6}},
            name: 'end',
            text: endText,
            hovertemplate: '%{{text}}<extra></extra>'
        }});
    }}

    if (endLinkX.length > 0) {{
        traces.push({{
            type: 'scatter',
            x: endLinkX,
            y: endLinkY,
            mode: 'lines',
            line: {{color: 'rgba(255,99,71,0.60)', width: 1.4, dash: 'dash'}},
            name: 'high_to_end',
            hoverinfo: 'skip'
        }});
    }}

    if (trendLineX.length > 0) {{
        traces.push({{
            type: 'scatter',
            x: trendLineX,
            y: trendLineY,
            mode: 'lines',
            line: {{color: '{ACCENT_BLUE}', width: 2}},
            name: 'trend_line',
            hoverinfo: 'skip'
        }});
    }}

    if (debugData.line_x.length > 0) {{
        traces.push({{
            type: 'scatter',
            x: debugData.line_x,
            y: debugData.line_y,
            mode: 'lines',
            line: {{color: 'rgba(148,103,189,0.85)', width: 2, dash: 'dot'}},
            name: 'debug_line',
            hoverinfo: 'skip'
        }});
    }}

    if (debugData.reset_x.length > 0) {{
        traces.push({{
            type: 'scatter',
            x: debugData.reset_x,
            y: debugData.reset_y,
            mode: 'markers',
            marker: {{color: 'rgba(148,103,189,0.90)', size: 6, symbol: 'x'}},
            name: 'debug_reset',
            text: debugData.reset_text,
            hovertemplate: '%{{text}}<extra></extra>'
        }});
    }}

    return traces;
}}

function buildLayout(filteredCount) {{
    const layout = {{
        title: null,
        template: 'plotly_white',
        autosize: true,
        hovermode: 'closest',
        legend: {{
            orientation: 'h',
            yanchor: 'bottom',
            y: 1.01,
            xanchor: 'left',
            x: 0
        }},
        margin: {{l: 42, r: 25, t: 72, b: 45, pad: 0}},
        hoverlabel: {{
            bgcolor: 'rgba(255, 255, 255, 0.50)',
            bordercolor: 'rgba(0, 0, 0, 0.45)',
            font: {{color: 'black'}}
        }},
        xaxis: {{
            title: null,
            tickfont: {{size: 10}},
            showgrid: false,
            range: [xMin - xLeftPad, xMax + xRightPad],
            autorange: false,
            showspikes: false,
            showline: true,
            linewidth: 1,
            linecolor: 'rgba(0,0,0,0.35)',
            rangeslider: {{visible: false}},
        }},
        yaxis: {{
            title: null,
            tickfont: {{size: 10}},
            showgrid: false,
            showspikes: false,
            showline: true,
            linewidth: 1,
            linecolor: 'rgba(0,0,0,0.35)',
        }},
        annotations: []
    }};

    if (filteredCount === 0) {{
        layout.annotations.push({{
            x: 0.5,
            y: 0.97,
            xref: 'paper',
            yref: 'paper',
            text: '当前阈值下没有显示样本',
            showarrow: false,
            font: {{size: 14, color: 'rgba(0,0,0,0.60)'}}
        }});
    }}
    return layout;
}}

function renderTrendChart() {{
    const threshold = normalizeThreshold(
        multipleInput.value || String(defaultMultiple)
    );

    const filtered = sourceSegments.filter(
        row => row.trend_atr_multiple >= threshold
    );
    summaryBox.textContent = 'multiple >= ' + threshold.toFixed(1)
        + '，显示 ' + filtered.length + ' / ' + sourceSegments.length + ' 段';

    Plotly.react(
        chartDiv,
        buildSegmentTraces(filtered),
        buildLayout(filtered.length),
        {{
            responsive: true,
            displayModeBar: false,
            displaylogo: false
        }}
    );
}}

toggleBtn.addEventListener('click', () => {{
    sidebar.classList.toggle('open');
    toggleBtn.classList.toggle('sidebar-open', sidebar.classList.contains('open'));
}});
multipleInput.addEventListener('input', renderTrendChart);
multipleSelect.addEventListener('change', () => {{
    multipleInput.value = multipleSelect.value;
    renderTrendChart();
}});
multipleStepUp.addEventListener('click', () => {{
    const current = normalizeThreshold(multipleInput.value);
    multipleInput.value = (current + 0.1).toFixed(1);
    renderTrendChart();
}});
multipleStepDown.addEventListener('click', () => {{
    const current = normalizeThreshold(multipleInput.value);
    multipleInput.value = Math.max(minVisibleMultiple, current - 0.1).toFixed(1);
    renderTrendChart();
}});
populateMultipleSelect();
renderTrendChart();
</script>
</body>
</html>"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print('[HTML] saved trend analysis chart.')


def compute_running_ols_diagnostics(seg: pd.DataFrame,
                                    start_bars: int,
                                    constraint_w: int):
    x_values = []
    slope_pct_values = []
    max_downward_deviation_pct_values = []
    residual_std_pct_values = []

    if len(seg) < max(2, start_bars):
        return (
            x_values,
            slope_pct_values,
            max_downward_deviation_pct_values,
            residual_std_pct_values,
        )

    for end_offset in range(max(start_bars - 1, 1), len(seg)):
        current_seg = seg.iloc[:end_offset + 1]
        _, _, progress_values, invalid_reason = compute_progress_scan_long(
            current_seg, constraint_w
        )
        if invalid_reason:
            continue
        if len(progress_values) == 0:
            continue
        if np.any(progress_values <= 0):
            continue

        stats = compute_ols_trend_line_stats_long(current_seg)
        close_prices = current_seg['close'].to_numpy(dtype=float)
        t = np.arange(len(current_seg), dtype=float)
        slope, intercept = np.polyfit(t, close_prices, 1)
        fitted = slope * t + intercept
        valid_mask = np.abs(fitted) > 1e-12
        if np.any(valid_mask):
            residual_pct = (
                (close_prices[valid_mask] - fitted[valid_mask])
                / np.abs(fitted[valid_mask]) * 100.0
            )
            downward_gap = (
                fitted[valid_mask] - close_prices[valid_mask]
            )
            downward_gap = np.maximum(downward_gap, 0.0)
            downward_deviation_pct = (
                downward_gap / np.abs(fitted[valid_mask]) * 100.0
            )
            max_downward_deviation_pct = float(np.max(downward_deviation_pct))
            residual_std_pct = float(np.std(residual_pct, ddof=0))
        else:
            max_downward_deviation_pct = np.nan
            residual_std_pct = np.nan

        x_values.append(int(end_offset))
        slope_pct_values.append(stats['trend_line_speed_pct_per_bar'])
        max_downward_deviation_pct_values.append(max_downward_deviation_pct)
        residual_std_pct_values.append(residual_std_pct)

    return (
        x_values,
        slope_pct_values,
        max_downward_deviation_pct_values,
        residual_std_pct_values,
    )


def get_final_feasible_windows(seg: pd.DataFrame,
                               w_min: int,
                               w_max: int):
    feasible_windows = []
    for w in range(int(w_min), int(w_max) + 1):
        _, _, progress_values, invalid_reason = compute_progress_scan_long(seg, w)
        if invalid_reason:
            continue
        if len(progress_values) == 0:
            continue
        if np.any(progress_values <= 0):
            continue
        feasible_windows.append(int(w))
    return feasible_windows


def compute_running_fit_y_series(seg: pd.DataFrame,
                                 w: int):
    x_values = []
    fit_y_values = []

    if len(seg) <= w:
        return x_values, fit_y_values

    for end_offset in range(w, len(seg)):
        current_seg = seg.iloc[:end_offset + 1]
        _, _, progress_values, invalid_reason = compute_progress_scan_long(
            current_seg, w
        )
        if invalid_reason:
            continue
        if len(progress_values) == 0:
            continue
        if np.any(progress_values <= 0):
            continue
        x_values.append(int(end_offset))
        fit_y_values.append(float(progress_values.min()))

    return x_values, fit_y_values


def align_series_to_view(x_left: int,
                         x_right: int,
                         x_values: list[int],
                         y_values: list[float]):
    full_x = list(range(int(x_left), int(x_right) + 1))
    value_map = {
        int(x): float(y)
        for x, y in zip(x_values, y_values)
    }
    full_y = [value_map.get(int(x), np.nan) for x in full_x]
    return full_x, full_y


def export_constraint_trend_case_html(
        file_name: str,
        save_name: str,
        underlying1: pd.DataFrame,
        trend_df: pd.DataFrame,
        factor: float,
        case_index: int,
        w_min: int,
        w_max: int):
    if go is None or make_subplots is None:
        print('[HTML] plotly is not installed, skip trend test html export.')
        return
    if case_index < 0 or case_index >= len(trend_df):
        return

    case_row = trend_df.iloc[case_index]
    segment_id = int(case_row['trade_id'])
    low_idx = int(case_row['low_index'])
    high_idx = int(case_row['high_index'])
    end_idx = int(case_row['end_index']) if pd.notna(case_row['end_index']) else high_idx
    constraint_w = int(case_row['constraint_w_bars'])
    context_bars = max(0, int(w_max))
    view_start_idx = max(0, low_idx - context_bars)
    view = underlying1.iloc[view_start_idx:end_idx + 1].copy()
    seg = underlying1.iloc[low_idx:high_idx + 1].copy()
    x_left = 0
    x_right = len(view) - 1
    x_left_pad = max(1.0, round(len(view) * 0.03, 2))
    x_right_pad = x_left_pad
    x_view_left = x_left - x_left_pad
    x_view_right = x_right + x_right_pad
    low_pos = low_idx - view_start_idx
    high_pos = high_idx - view_start_idx
    end_pos = end_idx - view_start_idx
    plot_x = list(range(len(view)))
    seg_x = list(range(low_pos, low_pos + len(seg)))
    x_span = max(1, x_right - x_left)
    x_dtick = max(1, int(round(x_span / 8.0)))
    case_hovertext = []
    for bar_no, (abs_idx, row_view) in enumerate(view.iterrows(), start=0):
        bars_from_low = bar_no - low_pos
        case_hovertext.append(
            f'bar_no: {bar_no}<br>'
            f'bar_index: {int(abs_idx)}<br>'
            f'bars_from_low: {bars_from_low}<br>'
            f'time: {row_view["Date"]}<br>'
            f'open: {float(row_view["open"] / factor * 100):.4f}<br>'
            f'high: {float(row_view["high"] / factor * 100):.4f}<br>'
            f'low: {float(row_view["low"] / factor * 100):.4f}<br>'
            f'close: {float(row_view["close"] / factor * 100):.4f}'
        )

    fig_html = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        specs=[
            [{}],
            [{'secondary_y': True}],
            [{}],
        ],
        vertical_spacing=0.06,
        row_heights=[0.45, 0.24, 0.31],
        subplot_titles=(
            '主图',
            '运行中的 OLS 斜率、最大向下偏离与下方残差标准差',
            '可行时间窗口的最低涨幅',
        ),
    )

    fig_html = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        specs=[
            [{}],
            [{'secondary_y': True}],
            [{}],
        ],
        vertical_spacing=0.06,
        row_heights=[0.45, 0.24, 0.31],
        subplot_titles=(
            '主图',
            '运行中的 OLS 斜率、最大向下偏离与残差离散度',
            '可行时间窗口的最低涨幅',
        ),
    )

    fig_html.add_trace(go.Candlestick(
        x=plot_x,
        open=view['open'] / factor * 100,
        high=view['high'] / factor * 100,
        low=view['low'] / factor * 100,
        close=view['close'] / factor * 100,
        text=case_hovertext,
        name='price',
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

    if len(seg) >= 2 and pd.notna(case_row['ols_slope']) and pd.notna(case_row['ols_intercept']):
        t = np.arange(len(seg), dtype=float)
        fitted = case_row['ols_slope'] * t + case_row['ols_intercept']
        fig_html.add_trace(go.Scatter(
            x=seg_x,
            y=(fitted / factor * 100).tolist(),
            mode='lines',
            line=dict(color=ACCENT_BLUE, width=2),
            name='trend_line',
            hoverinfo='skip',
        ), row=1, col=1)

    fig_html.add_trace(go.Scatter(
        x=[low_pos],
        y=[float(case_row['low_price']) / factor * 100],
        mode='markers',
        marker=dict(color='#1F77B4', size=6),
        name='low',
        hovertemplate=(
            f'segment: {segment_id}<br>'
            f'bar_no: {low_pos}<br>'
            f'low_index: {low_idx}<br>'
            f'low_date: {case_row["low_date"]}<br>'
            f'low_price: {float(case_row["low_price"]):.4f}<extra></extra>'
        ),
    ), row=1, col=1)

    fig_html.add_trace(go.Scatter(
        x=[high_pos],
        y=[float(case_row['high_price']) / factor * 100],
        mode='markers',
        marker=dict(color='orange', size=6),
        name='high',
        hovertemplate=(
            f'segment: {segment_id}<br>'
            f'bar_no: {high_pos}<br>'
            f'high_index: {high_idx}<br>'
            f'high_date: {case_row["high_date"]}<br>'
            f'high_price: {float(case_row["high_price"]):.4f}<extra></extra>'
        ),
    ), row=1, col=1)

    fig_html.add_trace(go.Scatter(
        x=[end_pos],
        y=[float(case_row['end_price']) / factor * 100],
        mode='markers',
        marker=dict(color='rgba(255,99,71,0.60)', size=7),
        name='end',
        hovertemplate=(
            f'segment: {segment_id}<br>'
            f'bar_no: {end_pos}<br>'
            f'end_index: {end_idx}<br>'
            f'end_date: {case_row["end_date"]}<br>'
            f'end_price: {float(case_row["end_price"]):.4f}<extra></extra>'
        ),
    ), row=1, col=1)

    if end_idx > high_idx:
        fig_html.add_trace(go.Scatter(
            x=[high_pos, end_pos],
            y=[
                float(case_row['high_price']) / factor * 100,
                float(case_row['end_price']) / factor * 100,
            ],
            mode='lines',
            line=dict(color='rgba(255,99,71,0.35)', width=1.4, dash='dash'),
            name='high_to_end',
            hoverinfo='skip',
        ), row=1, col=1)

    feasible_windows = get_final_feasible_windows(seg, w_min=w_min, w_max=w_max)
    min_feasible_w = min(feasible_windows) if len(feasible_windows) > 0 else None
    diag_x, diag_slope_pct, diag_max_downward_dev_pct, diag_residual_std_pct = compute_running_ols_diagnostics(
        seg,
        start_bars=(min_feasible_w + 1) if min_feasible_w is not None else (w_min + 1),
        constraint_w=min_feasible_w if min_feasible_w is not None else w_min,
    )
    has_row2_data = False
    if len(diag_x) > 0:
        has_row2_data = True
        shifted_diag_x = [low_pos + int(x) for x in diag_x]
        aligned_diag_x, aligned_slope_y = align_series_to_view(
            x_left, x_right, shifted_diag_x, diag_slope_pct
        )
        _, aligned_downward_y = align_series_to_view(
            x_left, x_right, shifted_diag_x, diag_max_downward_dev_pct
        )
        _, aligned_residual_std_y = align_series_to_view(
            x_left, x_right, shifted_diag_x, diag_residual_std_pct
        )
        fig_html.add_trace(go.Scatter(
            x=aligned_diag_x,
            y=aligned_slope_y,
            mode='lines+markers',
            line=dict(color='rgba(31,119,180,0.90)', width=2),
            marker=dict(size=3),
            name='ols_slope_pct_per_bar',
            connectgaps=False,
            hovertemplate=(
                'running OLS slope<br>'
                'bar_no: %{x}<br>'
                'slope_pct_per_bar: %{y:.6f}<extra></extra>'
            ),
        ), row=2, col=1, secondary_y=False)

        fig_html.add_trace(go.Scatter(
            x=aligned_diag_x,
            y=aligned_downward_y,
            mode='lines+markers',
            line=dict(color='rgba(255,140,0,0.55)', width=2),
            marker=dict(size=3),
            name='max_downward_deviation_pct',
            connectgaps=False,
            hovertemplate=(
                'maximum downward deviation from current OLS<br>'
                'bar_no: %{x}<br>'
                'max_downward_deviation_pct: %{y:.4f}<extra></extra>'
            ),
        ), row=2, col=1, secondary_y=True)

        fig_html.add_trace(go.Scatter(
            x=aligned_diag_x,
            y=aligned_residual_std_y,
            mode='lines+markers',
            line=dict(color='rgba(148,103,189,0.88)', width=2),
            marker=dict(size=3),
            name='residual_std_pct',
            connectgaps=False,
            hovertemplate=(
                'neutral residual std under current OLS<br>'
                'bar_no: %{x}<br>'
                'residual_std_pct: %{y:.6f}<extra></extra>'
            ),
        ), row=2, col=1, secondary_y=True)
    else:
        fig_html.add_trace(go.Scatter(
            x=[x_left, x_right],
            y=[0.0, 0.0],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.0)', width=1),
            hoverinfo='skip',
            showlegend=False,
        ), row=2, col=1, secondary_y=False)
        fig_html.add_trace(go.Scatter(
            x=[x_left, x_right],
            y=[0.0, 0.0],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.0)', width=1),
            hoverinfo='skip',
            showlegend=False,
        ), row=2, col=1, secondary_y=True)

    has_row3_data = False
    row3_windows = []
    row3_fit_y_values = []
    row3_min_bar_no = []
    row3_min_bar_index = []
    for w in feasible_windows:
        progress_offsets, progress_x, progress_values, invalid_reason = (
            compute_progress_scan_long(seg, w)
        )
        if invalid_reason or len(progress_values) == 0:
            continue
        if np.any(progress_values <= 0):
            continue
        min_pos = int(np.argmin(progress_values))
        has_row3_data = True
        row3_windows.append(int(w))
        row3_fit_y_values.append(float(progress_values.min()))
        row3_min_bar_no.append(int(progress_offsets[min_pos]))
        row3_min_bar_index.append(int(progress_x[min_pos]))

    w_left = int(w_min)
    w_right = int(w_max)
    if len(row3_windows) > 0:
        w_left = int(min(row3_windows))
        w_right = int(max(row3_windows))
    w_span = max(1, w_right - w_left)
    w_pad = max(0.5, round(max(1, len(feasible_windows)) * 0.08, 2))
    w_view_left = w_left - w_pad
    w_view_right = w_right + w_pad
    w_dtick = max(1, int(round(w_span / 8.0)))

    if has_row3_data:
        fig_html.add_trace(go.Scatter(
            x=row3_windows,
            y=row3_fit_y_values,
            customdata=np.column_stack([
                np.asarray(row3_min_bar_no, dtype=int),
                np.asarray(row3_min_bar_index, dtype=int),
            ]),
            mode='lines+markers',
            line=dict(color='rgba(31,119,180,0.90)', width=2),
            marker=dict(size=3),
            name='fit_y_pct_by_window',
            hovertemplate=(
                'minimum progress by feasible window<br>'
                'w: %{x}<br>'
                'fit_y_pct: %{y:.4f}%<br>'
                'min_progress_bar_no: %{customdata[0]}<br>'
                'min_progress_bar_index: %{customdata[1]}<extra></extra>'
            ),
        ), row=3, col=1)
    if not has_row3_data:
        fig_html.add_trace(go.Scatter(
            x=[w_left, w_right],
            y=[0.0, 0.0],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.0)', width=1),
            hoverinfo='skip',
            showlegend=False,
        ), row=3, col=1)
        fig_html.add_annotation(
            xref='paper',
            yref='paper',
            x=0.5,
            y=0.06,
            text='no feasible windows in this segment',
            showarrow=False,
            font=dict(size=12, color='rgba(60,60,60,0.90)'),
        )

    fig_html.update_layout(
        title=(
            f'Trend Test Case segment {segment_id}: '
            f'constraint_w={constraint_w}'
        ),
        template='plotly_white',
        autosize=True,
        hovermode='closest',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
        ),
        margin=dict(l=42, r=25, t=90, b=45, pad=0),
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.50)',
            bordercolor='rgba(0, 0, 0, 0.45)',
            font=dict(color='black')
        )
    )
    fig_html.add_annotation(
        xref='paper',
        yref='paper',
        x=0.01,
        y=0.40,
        text='图注：蓝线是 OLS 斜率，橙线是最大向下偏离，紫线是残差离散度。',
        showarrow=False,
        align='left',
        font=dict(size=11, color='rgba(60,60,60,0.92)'),
        bgcolor='rgba(255,255,255,0.65)',
        bordercolor='rgba(0,0,0,0.12)',
        borderpad=3,
    )
    fig_html.add_annotation(
        xref='paper',
        yref='paper',
        x=0.01,
        y=0.10,
        text='图注：横轴是窗口 w，纵轴是该窗口的最低涨幅 fit_y_pct，hover 会显示最低涨幅出现的 bar。',
        showarrow=False,
        align='left',
        font=dict(size=11, color='rgba(60,60,60,0.92)'),
        bgcolor='rgba(255,255,255,0.65)',
        bordercolor='rgba(0,0,0,0.12)',
        borderpad=3,
    )

    for row_no in [1, 2]:
        fig_html.update_xaxes(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.35)',
            tickfont=dict(size=10),
            tickmode='linear',
            tick0=x_left,
            dtick=x_dtick,
            row=row_no,
            col=1,
        )
        fig_html.update_yaxes(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.35)',
            tickfont=dict(size=10),
            row=row_no,
            col=1,
        )

    fig_html.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.35)',
        tickfont=dict(size=10),
        tickmode='linear',
        tick0=w_left,
        dtick=w_dtick,
        row=3,
        col=1,
    )
    fig_html.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.35)',
        tickfont=dict(size=10),
        row=3,
        col=1,
    )

    for row_no in [1, 2]:
        fig_html.update_xaxes(
            range=[x_view_left, x_view_right],
            autorange=False,
            row=row_no,
            col=1,
        )
    fig_html.update_xaxes(
        range=[w_view_left, w_view_right],
        autorange=False,
        row=3,
        col=1,
    )
    fig_html.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    fig_html.update_yaxes(title='price %', row=1, col=1)
    fig_html.update_yaxes(title='ols_slope_pct_per_bar', row=2, col=1, secondary_y=False)
    fig_html.update_yaxes(
        title='max_downward_deviation_pct / residual_std_pct',
        row=2, col=1, secondary_y=True
    )
    fig_html.update_yaxes(title='fit_y_pct', row=3, col=1)
    if not has_row2_data:
        fig_html.update_yaxes(range=[0.0, 1.0], row=2, col=1, secondary_y=False)
        fig_html.update_yaxes(range=[0.0, 1.0], row=2, col=1, secondary_y=True)
    if not has_row3_data:
        fig_html.update_yaxes(range=[0.0, 1.0], row=3, col=1)

    html_dir = get_html_output_dir(file_name, TREND_TEST_CASE_HTML_FOLDER)
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(
        html_dir,
        save_name + f' case_{case_index + 1:02d} trend_test_case interactive.html'
    )
    html_text = fig_html.to_html(
        include_plotlyjs=True, full_html=True,
        default_width='100vw', default_height='100vh',
        config={'responsive': True, 'displayModeBar': False,
                'displaylogo': False}
    )
    html_text = html_text.replace(
        '<head>',
        '<head><style>'
        'html,body{width:100%;height:100%;margin:0;padding:0;overflow:hidden;}'
        '.plotly-graph-div{width:100vw !important;height:100vh !important;}'
        '.hoverlayer .hovertext .bg,'
        '.hoverlayer .hovertext rect,'
        '.hoverlayer .hovertext path{'
        'fill:rgba(255,255,255,0.50) !important;'
        'fill-opacity:0.50 !important;'
        'stroke:rgba(0,0,0,0.45) !important;'
        'stroke-opacity:0.45 !important;}'
        '.hoverlayer .hovertext{opacity:1 !important;}'
        '.hoverlayer .hovertext text{fill:#000 !important;}'
        '</style>',
        1
    )
    html_text = html_text.replace(
        '<body>', '<body style="margin:0;overflow:hidden;">', 1)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print('[HTML] saved trend test chart.')


# ============================================================
# Momentum Strategy
# ============================================================

class LongNoWDStrategy(BaseStrategy):
    """无回撤版本：空仓即开仓，持仓仅速度平仓。"""

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
        open_bar = p['open_bar']

        # 空仓时每根bar直接开仓（当根开盘价）
        if open_bar > 1 and ii + 1 >= open_bar:
            open_slice = quote.iloc[ii + 1 - open_bar:ii + 1]
            open_increase, inc_base = get_increase_with_base(open_slice)
            t_inc_per = (open_increase / inc_base * 100) if inc_base != 0 else 0.0
            signal.at[index, 'total_inc'] = open_increase
            signal.at[index, 't_inc_per'] = round(t_inc_per, 4)
        else:
            signal.at[index, 'total_inc'] = 0.0
            signal.at[index, 't_inc_per'] = 0.0

        signal.at[index, 'total_inc_signal'] = 1.0
        signal.at[index, 'inc_signal'] = 1.0
        signal.at[index, 'wd_signal'] = 1.0

        self.low_index = ii
        self.start_index = ii
        self.first_cond1_price = float(quote.loc[index, 'open'])
        self.new_opening_count = 1
        self.new_opening = True

        signal.at[index, 'low_index'] = self.low_index
        signal.at[index, 'period'] = 1

        return OpenResult(
            execution_price=round(self.first_cond1_price, self.params['round_precision']),
            low_index=self.low_index,
            low_price=self.first_cond1_price,
            start_index=self.start_index,
        )

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
        self.new_opening_count = 1
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
        close_threshold = p['close_threshold']

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

        # 速度条件（唯一平仓条件）
        if window_ready:
            ana_inc_slice_1 = quote.iloc[self.low_index:ii + 1]
            ana_inc_slice_2 = quote.iloc[
                self.low_index:ii + 1 - close_bar]
            holding_increase = (
                ana_inc_slice_1.high.max() - ana_inc_slice_2.high.max())
            holding_base = analysis_slice['low'].iloc[0]
            self.holding_increase_percent = (
                holding_increase / holding_base if holding_base != 0 else 0.0)
            signal.at[index, 'holding_inc'] = holding_increase
            if self.holding_increase_percent < close_threshold:
                signal.at[index, 'speed_close_signal'] = 1

        # 回撤仅记录，不参与平仓
        with_high, holding_withdrawal = get_withdrawal(holding_slice)
        holding_withdrawal_percent = (
            holding_withdrawal / with_high if with_high != 0 else 0)
        signal.at[index, 'holding_wd'] = holding_withdrawal
        signal.at[index, 'hld_wd_per'] = round(
            holding_withdrawal_percent * 100, 4)
        signal.at[index, 'holding_wd_signal'] = 0.0

        period = ii - self.holding_start_index + 1
        signal.at[index, 'high_price'] = max(holding_slice['high'])

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

    native_df, ROUND_PRECISION, NATIVE_BAR_SECONDS = load_data(folder_path, file_name)
    if (RESAMPLE_RULE or '').strip():
        df, BAR_SECONDS = resample_ohlc_df(native_df, RESAMPLE_RULE)
        print(f'[Data] resampled to {RESAMPLE_RULE}  |  bar period: {BAR_SECONDS}s')
    else:
        df = native_df.copy()
        BAR_SECONDS = NATIVE_BAR_SECONDS
        print(f'[Data] using native period  |  bar period: {BAR_SECONDS}s')

    # 创建输出文件夹
    os.makedirs('./result', exist_ok=True)
    os.makedirs(f'./result/{file_name} long no wd outcome/perf', exist_ok=True)
    os.makedirs(f'./result/{file_name} long no wd outcome/trans', exist_ok=True)
    os.makedirs(f'./result/{file_name} long no wd outcome/trade_stats', exist_ok=True)
    os.makedirs(f'./result/{file_name} long no wd outcome/trend_stats', exist_ok=True)

    outcome_stats = pd.DataFrame()

    # 选择回测时间区间
    startdate = START_INDEX
    enddate = END_INDEX

    if enddate == 'latest':
        preview_df = df.iloc[int(startdate):].copy()
    else:
        preview_df = df.iloc[int(startdate):int(enddate)].copy()
    if len(preview_df) == 0:
        raise ValueError(
            f'No data in selected range: START_INDEX={startdate}, END_INDEX={enddate}'
        )
    print(f'[Main] backtest index range: ({startdate}, {enddate})')
    print(f'[Main] backtest time range: {preview_df.iloc[0]["Date"]} -> {preview_df.iloc[-1]["Date"]}')

    open_bar_cfg = int(OPEN_BAR)
    close_bar_cfg = int(CLOSE_BAR)
    open_bar2_cfg = np.nan if pd.isna(OPEN_BAR2) else int(OPEN_BAR2)

    df5 = preview_df.reset_index(drop=True).copy()
    underlying = df5.copy()

    only_close = ONLY_CLOSE
    if only_close:
        underlying.open = underlying.low = underlying.high = underlying.close

    trend_analysis_df, window_scan_df, trend_debug_df = build_trend_analysis_df_from_constraint_window(
        underlying,
        constraint_w=TREND_W_MAX_BARS,
    )
    print(
        f'[TrendLine] constraint_w={TREND_W_MAX_BARS}, '
        f'segments={len(trend_analysis_df)}'
    )

    period_label = format_period_label(RESAMPLE_RULE, BAR_SECONDS)
    run_name = (
        f'w{TREND_W_MIN_BARS}-{TREND_W_MAX_BARS} '
        f'period_{period_label} '
        f'{startdate}-{enddate}'
    )
    save_name = run_name
    trend_multiple_df = build_trend_multiple_summary_df(
        underlying.reset_index(drop=True),
        trend_analysis_df,
        BAR_SECONDS,
    )

    if len(trend_analysis_df) > 0:
        trend_stats_name = f'{save_name} trend_line_segments.xlsx'
        writer = pd.ExcelWriter(
            './result/%s long no wd outcome/trend_stats/' % file_name + trend_stats_name,
            engine='xlsxwriter'
        )
        trend_analysis_df.to_excel(writer, sheet_name='trend_analysis', index=False)
        writer.close()
        print(
            './result/%s long no wd outcome/trend_stats/' % file_name
            + trend_stats_name
        )

    if len(trend_multiple_df) > 0:
        trend_multiple_dir = get_html_output_dir(file_name, TREND_MULTIPLE_FOLDER)
        os.makedirs(trend_multiple_dir, exist_ok=True)
        trend_multiple_name = f'{save_name} trend_multiple_summary.xlsx'
        trend_multiple_path = os.path.join(trend_multiple_dir, trend_multiple_name)
        writer = pd.ExcelWriter(trend_multiple_path, engine='xlsxwriter')
        trend_multiple_df.to_excel(writer, sheet_name='trend_multiple', index=False)
        writer.close()
        print(trend_multiple_path)

    if DEBUG_TREND_SEARCH and len(trend_debug_df) > 0:
        debug_name = f'{save_name} trend_search_debug.csv'
        debug_path = (
            './result/%s long no wd outcome/trend_stats/' % file_name + debug_name
        )
        trend_debug_df.to_csv(debug_path, index=False, encoding='utf-8-sig')
        print(debug_path)
        fail_rows = trend_debug_df[
            trend_debug_df['state'].isin(['fail_before_activation', 'fail_after_activation'])
        ]
        if len(fail_rows) > 0:
            first_fail = fail_rows.iloc[0]
            print(
                '[Debug] first fail: '
                f"search_start={int(first_fail['search_start'])}, "
                f"end_idx={int(first_fail['end_idx'])}, "
                f"low_idx={int(first_fail['low_idx'])}, "
                f"seg_bars={int(first_fail['seg_bars'])}, "
                f"progress_min={first_fail['progress_min']}, "
                f"non_positive_count={int(first_fail['non_positive_count'])}, "
                f"state={first_fail['state']}"
            )

    if EXPORT_INTERACTIVE_HTML and len(trend_analysis_df) > 0:
        factor = underlying['open'].iloc[0]
        case_trend_df = build_filtered_trend_case_df(
            underlying.reset_index(drop=True),
            trend_analysis_df,
            TREND_HTML_DEFAULT_MULTIPLE,
        )
        export_trend_analysis_html(
            file_name=file_name,
            save_name=save_name,
            underlying1=underlying.reset_index(drop=True),
            trend_df=trend_analysis_df,
            factor=factor,
            debug_df=trend_debug_df,
        )
        export_trend_multiple_ranked_html(
            file_name=file_name,
            save_name=save_name,
            trend_multiple_df=trend_multiple_df,
        )

        case_count = min(TREND_TEST_CASE_COUNT, len(case_trend_df))
        for case_idx in range(case_count):
            export_constraint_trend_case_html(
                file_name=file_name,
                save_name=save_name,
                underlying1=underlying.reset_index(drop=True),
                trend_df=case_trend_df,
                factor=factor,
                case_index=case_idx,
                w_min=TREND_W_MIN_BARS,
                w_max=TREND_W_MAX_BARS,
            )

    print("\ntime = --- %s seconds ---" % (time.time() - start_time))

    if AUTO_OPEN_TREND_HTML and len(trend_analysis_df) > 0:
        import webbrowser
        trend_html_dir = get_html_output_dir(file_name, TREND_ANALYSIS_HTML_FOLDER)
        trend_html_path = os.path.join(
            trend_html_dir,
            save_name + ' trend_analysis interactive.html'
        )
        webbrowser.open(os.path.abspath(trend_html_path))
        if len(trend_multiple_df) > 0:
            trend_multiple_dir = get_html_output_dir(file_name, TREND_MULTIPLE_FOLDER)
            trend_multiple_path = os.path.join(
                trend_multiple_dir,
                save_name + ' trend_multiple_ranked interactive.html'
            )
            webbrowser.open(os.path.abspath(trend_multiple_path))

    if AUTO_OPEN_DASHBOARD and len(trend_analysis_df) > 0:
        import webbrowser
        if ensure_dashboard_server_running(DASHBOARD_URL):
            webbrowser.open(DASHBOARD_URL)
        else:
            print(f'[Dashboard] open failed: {DASHBOARD_URL}')

    raise SystemExit(0)

    # --- 参数循环 ---
    for_num_1 = FOR_NUM_1
    for_num_2 = FOR_NUM_2
    for_num_3 = FOR_NUM_3
    print(for_num_1, for_num_2, for_num_3)
    step1 = STEP1
    step3 = STEP3

    for num in range(for_num_1):
        for i in range(for_num_2):
            print(f'{str(num)} {str(i)}\n')

            # 策略参数
            open_bar = open_bar_cfg
            open_threshold = OPEN_THRESHOLD
            close_bar = close_bar_cfg
            close_threshold = CLOSE_THRESHOLD
            open_continous_threshold = OPEN_CONTINOUS_THRESHOLD + (i * step1)
            open_bar2 = open_bar2_cfg
            open_threshold2 = OPEN_THRESHOLD2
            open_continous_threshold2 = OPEN_CONTINOUS_THRESHOLD2
            # 双策略
            commision_percent = COMMISION_PERCENT
            capital = CAPITAL

            # 无回撤策略下，回撤阈值参数已移除

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
                'open_threshold': open_threshold,
                'open_continous_threshold': open_continous_threshold,
                'close_bar': close_bar,
                'close_threshold': close_threshold,
                'open_continous_threshold2': open_continous_threshold2,
                'round_precision': ROUND_PRECISION,
            }

            strategy = LongNoWDStrategy(params)
            engine = BacktestEngine(
                underlying, strategy, capital,
                ROUND_PRECISION, commision_percent)
            (df_signal, signal, close_counts) = engine.run()
            withdrawal_close_count = close_counts.get(1, 0)
            speed_close_count = close_counts.get(2, 0)

            performance, transactions_df = generate_performance(
                underlying, df_signal, capital, commision_percent)
            trade_extreme_df = build_trade_extreme_stats_long(
                underlying, transactions_df)
            trend_analysis_df, window_scan_df = build_trend_analysis_df(
                underlying,
                trade_extreme_df,
                w_min=TREND_W_MIN_BARS,
                w_max=TREND_W_MAX_BARS,
                min_samples=TREND_MIN_PROGRESS_SAMPLES,
                improvement_capture_target=TREND_IMPROVEMENT_CAPTURE_TARGET,
            )
            filtered_case_df = build_filtered_trend_case_df(
                underlying,
                trend_analysis_df,
                TREND_HTML_DEFAULT_MULTIPLE,
            )
            print(f'[Trend] trade_extreme_df rows: {len(trade_extreme_df)}, '
                  f'trend_analysis_df rows: {len(trend_analysis_df)}, '
                  f'window_scan_df rows: {len(window_scan_df)}')
            if TREND_TEST_MODE and len(filtered_case_df) > 0:
                print_trend_test_case_summary(
                    trend_df=filtered_case_df,
                    window_scan_df=window_scan_df,
                    test_case_index=TREND_TEST_CASE_INDEX,
                    test_windows=TREND_TEST_WINDOWS,
                    improvement_capture_target=(
                        TREND_IMPROVEMENT_CAPTURE_TARGET
                    ),
                    near_band=TREND_TEST_NEAR_BAND,
                    big_drop_ratio=TREND_TEST_BIG_DROP_RATIO,
                )
            entry_to_max_profit_wd_df = build_entry_to_max_profit_withdrawal_df(
                underlying, transactions_df)

            open_count = int((df_signal['signal'] == 1.0).sum())
            idle_count = int((signal['have_holding'] == 0).sum())

            if len(transactions_df) > 1:
                Capital_outcome = round(
                    transactions_df[
                        transactions_df.Type != 'long'].Capital.iloc[-1], 2)
            else:
                Capital_outcome = 100
            perf_outcome = performance.reset_index(
                drop=True)[['date', 'capital']]

            # 打印结果
            print(str(startdate) + '-' + str(enddate))
            print('total close count = '
                  + str(withdrawal_close_count + speed_close_count))
            print('withdrawal close count = '
                  + str(round(withdrawal_close_count, 4)))
            print('speed close count = '
                  + str(round(speed_close_count, 4)))
            print(
                f'{startdate}-{enddate} '
                f'close_{int(withdrawal_close_count)}+{int(speed_close_count)}'
            )
            print('profit: ' + str(round(performance.capital.iloc[-1], 2)))
            print(f'[Check-1] withdrawal close count (should be 0): {withdrawal_close_count}')
            print(f'[Check-2] open-on-idle count: open={open_count}, idle={idle_count}')
            print(f'[Check-3] trade extreme stats rows: {len(trade_extreme_df)}')
            print(f'[Check-4] entry->max-profit wd rows: {len(entry_to_max_profit_wd_df)}')
            if len(entry_to_max_profit_wd_df) > 0:
                print(entry_to_max_profit_wd_df.head(5))

            # ====== 命名（fig1 已移除） ======
            period_label = format_period_label(RESAMPLE_RULE, BAR_SECONDS)
            run_name = (
                f'w{TREND_W_MIN_BARS}-{TREND_W_MAX_BARS} '
                f'period_{period_label} '
                f'{startdate}-{enddate} '
                f'close_{int(withdrawal_close_count)}+{int(speed_close_count)}'
            )
            report_name = f'LongNoWD {run_name}'
            save_name = run_name

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

            perf_name = f'{report_name} capital_{Capital_outcome} perf.xlsx'
            writer1 = pd.ExcelWriter(
                './result/%s long no wd outcome/perf/' % file_name + perf_name,
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
                    './result/%s long no wd outcome/trans/' % file_name
                    + f'{report_name} capital_{Capital_outcome} trans.xlsx',
                    engine='xlsxwriter')
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

            trade_stats_name = (
                f'{report_name} capital_{Capital_outcome} trade_stats.xlsx'
            )
            writer3 = pd.ExcelWriter(
                './result/%s long no wd outcome/trade_stats/' % file_name + trade_stats_name,
                engine='xlsxwriter')
            trade_extreme_df.to_excel(writer3, sheet_name='trade_extremes', index=False)
            entry_to_max_profit_wd_df.to_excel(
                writer3, sheet_name='entry_to_max_profit_wd', index=False)
            writer3.close()

            # Trend analysis Excel
            if len(trend_analysis_df) > 0 or len(window_scan_df) > 0:
                trend_stats_name = (
                    f'{report_name} capital_{Capital_outcome} trend_stats.xlsx'
                )
                writer_trend = pd.ExcelWriter(
                    './result/%s long no wd outcome/trend_stats/' % file_name
                    + trend_stats_name,
                    engine='xlsxwriter')
                trend_analysis_df.to_excel(
                    writer_trend, sheet_name='trend_analysis', index=False)
                window_scan_df.to_excel(
                    writer_trend, sheet_name='window_scan', index=False)
                wb_t = writer_trend.book
                ws_t = writer_trend.sheets['trend_analysis']
                ws_scan = writer_trend.sheets['window_scan']
                ws_t.set_default_row(15)
                ws_scan.set_default_row(15)
                fmt_t = wb_t.add_format()
                fmt_t.set_font_name('Microsoft YaHei UI Light')
                fmt_t.set_align('center')
                fmt_t.set_align('vcenter')
                fmt_t.set_font_size(12)
                fmt_t_int = wb_t.add_format({'num_format': '0'})
                fmt_t_int.set_font_name('Microsoft YaHei UI Light')
                fmt_t_int.set_align('center')
                fmt_t_int.set_align('vcenter')
                for col_idx, col_name in enumerate(trend_analysis_df.columns):
                    col_width = 14
                    col_fmt = fmt_t
                    if col_name in {'optimal_window_invalid_reason'}:
                        col_width = 24
                    elif 'date' in col_name:
                        col_width = 19
                    elif (col_name.endswith('_index')
                          or col_name.endswith('_bars')
                          or col_name in {
                              'trade_id', 'search_start',
                              'optimal_window_valid'}):
                        col_width = 12
                        col_fmt = fmt_t_int
                    elif col_name.endswith('_price'):
                        col_width = 12
                    elif ('speed' in col_name or 'ratio' in col_name
                          or col_name.endswith('_pct')
                          or col_name.startswith('ols_')):
                        col_width = 16
                    ws_t.set_column(col_idx, col_idx, col_width, col_fmt)
                ws_t.freeze_panes(1, 0)

                for col_idx, col_name in enumerate(window_scan_df.columns):
                    col_width = 14
                    col_fmt = fmt_t
                    if col_name in {'invalid_reason'}:
                        col_width = 24
                    elif 'date' in col_name:
                        col_width = 19
                    elif (col_name.endswith('_index')
                          or col_name.endswith('_bars')
                          or col_name in {
                              'trade_id', 'window_valid',
                              'first_feasible_flag',
                              'best_offset_flag',
                              'selected_flag'}):
                        col_width = 12
                        col_fmt = fmt_t_int
                    elif col_name.endswith('_price'):
                        col_width = 12
                    elif ('speed' in col_name or 'ratio' in col_name
                          or col_name.endswith('_pct')
                          or col_name.startswith('ols_')):
                        col_width = 16
                    ws_scan.set_column(col_idx, col_idx, col_width, col_fmt)
                ws_scan.freeze_panes(1, 0)
                writer_trend.close()
                print(f'[Excel] saved trend stats: {trend_stats_name}')

            # 排序柱状图：每笔交易一根柱（按回撤比例从大到小）
            if len(entry_to_max_profit_wd_df) > 0:
                wd_plot_df = entry_to_max_profit_wd_df.copy()
                wd_plot_df['max_withdrawal_to_max_profit_pct'] = pd.to_numeric(
                    wd_plot_df['max_withdrawal_to_max_profit_pct'],
                    errors='coerce'
                )
                wd_plot_df = wd_plot_df.dropna(
                    subset=['max_withdrawal_to_max_profit_pct']
                ).sort_values(
                    by='max_withdrawal_to_max_profit_pct',
                    ascending=False
                ).reset_index(drop=True)
                if len(wd_plot_df) > 0:
                    fig_wd = plt.figure(figsize=(20, 11))
                    fig_wd.clf()
                    if hasattr(fig_wd.canvas, 'manager') and fig_wd.canvas.manager is not None:
                        fig_wd.canvas.manager.set_window_title('profit_withdrawal')
                    ax_wd = fig_wd.add_subplot(111)
                    x_rank = np.arange(1, len(wd_plot_df) + 1)
                    wd_values = wd_plot_df['max_withdrawal_to_max_profit_pct'].to_numpy(
                        dtype=float
                    )
                    max_profit_values = wd_plot_df['max_profit_pct'].to_numpy(dtype=float)
                    wd_visual_values = wd_values.copy()
                    zero_mask = np.isclose(wd_visual_values, 0.0, atol=1e-12)
                    if zero_mask.any():
                        wd_visual_values[zero_mask] = ZERO_BAR_VISUAL_FLOOR_PCT

                    bars = ax_wd.bar(
                        x_rank,
                        wd_visual_values,
                        width=0.9,
                        label='segment_withdrawal_pct'
                    )
                    # 叠加每笔交易最大盈利幅度柱（宽度=withdrawal柱的1/3）
                    ax_wd.bar(
                        x_rank,
                        max_profit_values,
                        width=0.3,
                        color='tab:orange',
                        alpha=0.8,
                        label='max_profit_pct'
                    )
                    # x 轴只覆盖实际柱子数量，避免右侧留白
                    ax_wd.set_xlim(0.5, len(wd_plot_df) + 0.5)
                    ax_wd.set_title('profit_withdrawal')
                    ax_wd.set_xlabel('Trade Rank (Descending)')
                    ax_wd.set_ylabel('Max Withdrawal to Max Profit (%)')
                    ax_wd.grid(alpha=0.25)
                    ax_wd.legend()
                    if zero_mask.any():
                        ax_wd.text(
                            0.99, 0.98,
                            f'zero-value bars shown as {ZERO_BAR_VISUAL_FLOOR_PCT:.4f}: {int(zero_mask.sum())}',
                            transform=ax_wd.transAxes,
                            ha='right', va='top',
                            bbox=dict(boxstyle='round', fc='white', alpha=0.6)
                        )

                    # 交互：鼠标移到柱子上显示该笔交易时间区间
                    annot_wd = ax_wd.annotate(
                        "", xy=(0, 0), xytext=(18, 18),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->")
                    )
                    annot_wd.set_visible(False)

                    def update_wd_annot(rect, idx):
                        x = rect.get_x() + rect.get_width() / 2.0
                        y = rect.get_height()
                        annot_wd.xy = (x, y)
                        row = wd_plot_df.iloc[idx]
                        text = (
                            f"rank: {idx + 1}\n"
                            f"wd: {row['max_withdrawal_to_max_profit_pct']:.4f}%\n"
                            f"max_profit_pct: {row['max_profit_pct']:.4f}%\n"
                            f"entry: {row['entry_date']}\n"
                            f"max_profit: {row['max_profit_date']}\n"
                            f"bars: {int(row['segment_bars'])}"
                        )
                        annot_wd.set_text(text)
                        annot_wd.get_bbox_patch().set_alpha(0.4)

                    def hover_wd_bar(event):
                        vis = annot_wd.get_visible()
                        if event.inaxes == ax_wd:
                            for idx, rect in enumerate(bars):
                                contains, _ = rect.contains(event)
                                if contains:
                                    update_wd_annot(rect, idx)
                                    annot_wd.set_visible(True)
                                    fig_wd.canvas.draw_idle()
                                    return
                            if vis:
                                annot_wd.set_visible(False)
                                fig_wd.canvas.draw_idle()

                    fig_wd.canvas.mpl_connect("motion_notify_event", hover_wd_bar)

                    wd_hist_name = f'{report_name} profit_withdrawal'
                    if SAVE_STATIC_PLOT:
                        wd_plot_ext = 'pdf' if SAVE_PLOT_AS_PDF else 'png'
                        fig_wd.savefig(
                            './result/%s long no wd outcome/trade_stats/' % file_name
                            + wd_hist_name + f'.{wd_plot_ext}',
                            dpi=300, bbox_inches='tight')
                    if for_num_2 == 1 and SHOW_MATPLOTLIB_PLOTS:
                        fig_wd.show()
                    else:
                        plt.close(fig_wd)

                    # 新柱状图：按最大盈利从大到小排序，叠加最大回撤（宽度=1/3）
                    profit_plot_df = entry_to_max_profit_wd_df.copy()
                    profit_plot_df['max_profit_pct'] = pd.to_numeric(
                        profit_plot_df['max_profit_pct'],
                        errors='coerce'
                    )
                    profit_plot_df['max_withdrawal_to_max_profit_pct'] = pd.to_numeric(
                        profit_plot_df['max_withdrawal_to_max_profit_pct'],
                        errors='coerce'
                    )
                    profit_plot_df = profit_plot_df.dropna(
                        subset=['max_profit_pct']
                    ).sort_values(
                        by='max_profit_pct',
                        ascending=False
                    ).reset_index(drop=True)

                    if len(profit_plot_df) > 0:
                        fig_profit = plt.figure(figsize=(20, 11))
                        fig_profit.clf()
                        if hasattr(fig_profit.canvas, 'manager') and fig_profit.canvas.manager is not None:
                            fig_profit.canvas.manager.set_window_title('profit_sorted_withdrawal')
                        ax_profit = fig_profit.add_subplot(111)
                        x_rank_profit = np.arange(1, len(profit_plot_df) + 1)
                        max_profit_values_sorted = profit_plot_df['max_profit_pct'].to_numpy(dtype=float)
                        wd_values_sorted = np.nan_to_num(
                            profit_plot_df['max_withdrawal_to_max_profit_pct'].to_numpy(dtype=float),
                            nan=0.0
                        )
                        bars_profit_main = ax_profit.bar(
                            x_rank_profit,
                            max_profit_values_sorted,
                            width=0.9,
                            label='max_profit_pct'
                        )
                        ax_profit.bar(
                            x_rank_profit,
                            wd_values_sorted,
                            width=0.3,
                            color='tab:orange',
                            alpha=0.8,
                            label='segment_withdrawal_pct'
                        )
                        ax_profit.set_xlim(0.5, len(profit_plot_df) + 0.5)
                        ax_profit.set_title('profit_sorted_withdrawal')
                        ax_profit.set_xlabel('Trade Rank By Max Profit (Descending)')
                        ax_profit.set_ylabel('Percent (%)')
                        ax_profit.grid(alpha=0.25)
                        ax_profit.legend()

                        # 交互：鼠标移到柱子上显示该笔交易时间区间
                        annot_profit = ax_profit.annotate(
                            "", xy=(0, 0), xytext=(18, 18),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->")
                        )
                        annot_profit.set_visible(False)

                        def update_profit_annot(rect, idx):
                            x = rect.get_x() + rect.get_width() / 2.0
                            y = rect.get_height()
                            annot_profit.xy = (x, y)
                            row = profit_plot_df.iloc[idx]
                            text = (
                                f"rank: {idx + 1}\n"
                                f"max_profit_pct: {row['max_profit_pct']:.4f}%\n"
                                f"wd: {row['max_withdrawal_to_max_profit_pct']:.4f}%\n"
                                f"entry: {row['entry_date']}\n"
                                f"max_profit: {row['max_profit_date']}\n"
                                f"bars: {int(row['segment_bars'])}"
                            )
                            annot_profit.set_text(text)
                            annot_profit.get_bbox_patch().set_alpha(0.4)

                        def hover_profit_bar(event):
                            vis = annot_profit.get_visible()
                            if event.inaxes == ax_profit:
                                for idx, rect in enumerate(bars_profit_main):
                                    contains, _ = rect.contains(event)
                                    if contains:
                                        update_profit_annot(rect, idx)
                                        annot_profit.set_visible(True)
                                        fig_profit.canvas.draw_idle()
                                        return
                                if vis:
                                    annot_profit.set_visible(False)
                                    fig_profit.canvas.draw_idle()

                        fig_profit.canvas.mpl_connect("motion_notify_event", hover_profit_bar)

                        profit_hist_name = (
                            f'{report_name} profit_sorted_withdrawal'
                        )
                        if SAVE_STATIC_PLOT:
                            profit_plot_ext = 'pdf' if SAVE_PLOT_AS_PDF else 'png'
                            fig_profit.savefig(
                                './result/%s long no wd outcome/trade_stats/' % file_name
                                + profit_hist_name + f'.{profit_plot_ext}',
                                dpi=300, bbox_inches='tight')
                        if for_num_2 == 1 and SHOW_MATPLOTLIB_PLOTS:
                            fig_profit.show()
                        else:
                            plt.close(fig_profit)

            # Stats
            outcome_index = str(round(open_continous_threshold, 4))
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
    if for_num_2 > 1:
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
        os.makedirs('./result/stats %s long no wd outcome/' % file_name, exist_ok=True)
        if SAVE_STATIC_PLOT:
            stats_plot_ext = 'pdf' if SAVE_PLOT_AS_PDF else 'png'
            plt.savefig('./result/stats %s long no wd outcome/' % file_name
                        + ' ' + save_name + ' '
                        + str(for_num_1) + ' '
                        + str(for_num_2) + ' '
                        + f'all outcome.{stats_plot_ext}', dpi=1000)
        outcome_stats.to_excel('./result/stats %s long no wd outcome/' % file_name
                               + ' ' + save_name + ' '
                               + str(for_num_1) + ' '
                               + str(for_num_2) + ' '
                               + 'all outcome.xlsx')
    else:
        disk_path = './result/'
        open_excel = False
        if open_excel:
            os.startfile(
                disk_path + '%s long no wd outcome/perf/' % file_name + perf_name)

    # ====== Trend HTML + Trade HTML ======
    if for_num_2 == 1 and EXPORT_INTERACTIVE_HTML:
        underlying1 = underlying.reset_index(drop=True)
        factor = underlying1['open'][0]
        fig2_title = (
            f' capital_{round(Capital_outcome, 2)} '
            f'{report_name}'
        )
        export_interactive_html_long_no_wd(
            file_name=file_name,
            save_name=save_name,
            title=fig2_title,
            underlying1=underlying1,
            detail_df=detail_df,
            transactions_df=transactions_df,
            factor=factor
        )
        if len(trend_analysis_df) > 0:
            trend_multiple_df = build_trend_multiple_summary_df(
                underlying1,
                trend_analysis_df,
                BAR_SECONDS,
            )
            filtered_case_df = build_filtered_trend_case_df(
                underlying1,
                trend_analysis_df,
                TREND_HTML_DEFAULT_MULTIPLE,
            )
            export_trend_analysis_html(
                file_name=file_name,
                save_name=save_name,
                underlying1=underlying1,
                trend_df=trend_analysis_df,
                factor=factor
            )
            export_trend_multiple_ranked_html(
                file_name=file_name,
                save_name=save_name,
                trend_multiple_df=trend_multiple_df,
            )
            if TREND_TEST_MODE and len(filtered_case_df) > 0:
                case_count = min(TREND_TEST_CASE_COUNT, len(filtered_case_df))
                print(
                    f'[HTML] exporting first {case_count} trend test cases '
                    f'(multiple >= {TREND_HTML_DEFAULT_MULTIPLE:.1f})...'
                )
                for case_idx in range(case_count):
                    case_save_name = f'{save_name} case_{case_idx + 1:02d}'
                    export_trend_test_case_html(
                        file_name=file_name,
                        save_name=case_save_name,
                        underlying1=underlying1,
                        trend_df=filtered_case_df,
                        window_scan_df=window_scan_df,
                        factor=factor,
                        test_case_index=case_idx,
                        test_windows=TREND_TEST_WINDOWS,
                        improvement_capture_target=(
                            TREND_IMPROVEMENT_CAPTURE_TARGET
                        ),
                    )
            if AUTO_OPEN_DASHBOARD:
                import webbrowser
                if ensure_dashboard_server_running(DASHBOARD_URL):
                    webbrowser.open(DASHBOARD_URL)
                else:
                    print(f'[Dashboard] open failed: {DASHBOARD_URL}')
            if AUTO_OPEN_TREND_HTML:
                import webbrowser
                html_dir = get_html_output_dir(
                    file_name, TREND_ANALYSIS_HTML_FOLDER)
                html_path = os.path.join(
                    html_dir,
                    save_name + ' trend_analysis interactive.html')
                webbrowser.open(os.path.abspath(html_path))
                if len(trend_multiple_df) > 0:
                    trend_multiple_dir = get_html_output_dir(
                        file_name, TREND_MULTIPLE_FOLDER)
                    trend_multiple_path = os.path.join(
                        trend_multiple_dir,
                        save_name + ' trend_multiple_ranked interactive.html')
                    webbrowser.open(os.path.abspath(trend_multiple_path))

    # plt.show()

