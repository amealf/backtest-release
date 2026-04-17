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
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
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
end_date = '20250610'  # 或 'latest'
only_close = False

# 重采样设置：设为 '' 表示直接使用原始周期
# 例如 '1min' / '5min' / '15min' / '1H'
resample_rule = ''

# 运行模式：
# 'manual' = 使用当前参数直接回测，并弹出 K 线买卖点图
# 'grid' = 执行网格搜索，并输出参数结果图
run_mode = 'grid'
run_mode = 'manual'

# 策略参数（直接使用 bar 数，单位是当前实际回测周期）
open_bar = 10
open_threshold = 0.0020
open_withdrawal_threshold = open_threshold  # 暂时不需要
close_bar = open_bar
close_threshold = open_threshold
open_continous_threshold = open_threshold
close_withdrawal_threshold = open_withdrawal_threshold

# Grid search
bar_end = 20
bar_step = 1
threshold_step = 0.001
open_threshold_stop_flat_rounds = 5
open_threshold_max_iterations = 30
for_num_3 = open_threshold_max_iterations
step3 = 0.001

# 双策略参数（保留）
open_bar2 = np.nan  # np.nan 表示不启用
open_threshold2 = np.nan
open_continous_threshold2 = 0.003
close_withdrawal_threshold2 = 0.003

commision_percent = 0.000
capital = 100.0
export_interactive_html = True
accent_blue = '#1F77B4'
sell_wd_color = 'green'
sell_speed_color = 'black'
html_crosshair_enabled = False
html_crosshair_color = 'rgba(255, 120, 120, 0.45)'
html_show_trade_count_badge = True
# 静态图保存开关：默认不保存 PDF/PNG（保留 HTML 导出）
save_static_plot = False
# 当 save_static_plot=True 时决定保存为 PDF 或 PNG
save_plot_as_pdf = False


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


def build_int_search_values(start: int, end: int, step: int) -> list[int]:
    if step == 0:
        raise ValueError('step cannot be 0.')
    if step > 0 and start > end:
        raise ValueError('positive step requires start <= end.')
    if step < 0 and start < end:
        raise ValueError('negative step requires start >= end.')
    stop = end + (1 if step > 0 else -1)
    return list(range(start, stop, step))


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


def append_progress_summary(path: str, row_index: str, row_data: dict):
    row_df = pd.DataFrame([row_data], index=[row_index])
    row_df.index.name = 'param_tag'
    row_df.to_csv(
        path,
        mode='a',
        header=not os.path.exists(path),
        encoding='utf-8-sig',
    )


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
        self.open_withdraw_reset_same_bar_count = 0

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
        open_threshold = p['open_threshold']
        open_continous_threshold = p['open_continous_threshold']
        open_withdrawal_threshold = p['open_withdrawal_threshold']
        close_bar = p['close_bar']
        close_threshold = p['close_threshold']
        close_withdrawal_threshold = p['close_withdrawal_threshold']
        open_continous_threshold2 = p['open_continous_threshold2']

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
        close_threshold = p['close_threshold']
        close_withdrawal_threshold = p['close_withdrawal_threshold']
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

    if run_mode == 'manual' and (resample_rule or '').strip():
        prompt_manual_intrabar_precheck(
            native_preview_df,
            preview_df,
            BAR_SECONDS,
            open_threshold,
            open_continous_threshold,
            metric_kind='ratio',
        )

    period_label = format_period_label(resample_rule, BAR_SECONDS)
    open_bar_cfg = int(open_bar)
    close_bar_cfg = int(close_bar)
    open_bar2_cfg = np.nan if pd.isna(open_bar2) else int(open_bar2)

    # 创建输出文件夹
    os.makedirs('./result', exist_ok=True)
    os.makedirs(f'./result/{file_name} long outcome/perf', exist_ok=True)
    os.makedirs(f'./result/{file_name} long outcome/trans', exist_ok=True)

    outcome_stats = pd.DataFrame()

    print(f'[Main] backtest time range: {preview_df.iloc[0]["Date"]} -> {preview_df.iloc[-1]["Date"]}')

    df5 = preview_df.reset_index(drop=True).copy()
    underlying = df5.copy()
    run_name = (
        f'period_{period_label} '
        + f'{make_safe_range_token(range_start_label)}-'
        + f'{make_safe_range_token(range_end_label)}'
    )
    summary_dir = './result/stats %s long outcome/' % file_name
    os.makedirs(summary_dir, exist_ok=True)

    progress_token = make_safe_range_token(
        'ob' + str(open_bar_cfg)
        + '_be' + str(bar_end)
        + '_bs' + str(bar_step)
        + '_ot' + str(open_threshold)
        + '_ts' + str(threshold_step)
        + '_omax' + str(open_threshold_max_iterations)
        + '_f3' + str(for_num_3)
        + '_s3' + str(step3)
    )
    progress_json_path = (
        summary_dir + ' ' + run_name + ' ' + progress_token + ' progress.json'
    )
    progress_summary_path = (
        summary_dir + ' ' + run_name + ' ' + progress_token + ' progress.csv'
    )
    progress_signature = {
        'run_mode': run_mode,
        'data_selection_mode': data_selection_mode,
        'range_start_label': str(range_start_label),
        'range_end_label': str(range_end_label),
        'resample_rule': str(resample_rule),
        'period_label': str(period_label),
        'open_bar_start': int(open_bar_cfg),
        'bar_end': int(bar_end),
        'bar_step': int(bar_step),
        'open_threshold_start': float(open_threshold),
        'threshold_step': float(threshold_step),
        'open_threshold_max_iterations': int(open_threshold_max_iterations),
        'for_num_3': int(for_num_3),
        'step3': float(step3),
        'open_threshold_stop_flat_rounds': int(open_threshold_stop_flat_rounds),
    }
    resume_progress = None
    resume_open_bar_index = 0
    resume_threshold_iter = 0
    resume_open_cont_iter = 0
    resume_outer_state = {
        'last_open_threshold_trade_count': None,
        'unchanged_open_threshold_steps': 0,
    }
    resume_inner_state = {
        'last_open_cont_trade_count': None,
        'unchanged_open_cont_steps': 0,
        'outer_reference_trade_count': None,
    }

    only_close_cfg = only_close
    if only_close_cfg:
        underlying.open = underlying.low = underlying.high = underlying.close

    export_interactive_html_enabled = (
        export_interactive_html or (run_mode == 'manual')
    )

    # --- 参数循环 ---
    if run_mode == 'manual':
        if min(int(open_bar_cfg), int(close_bar_cfg)) <= 0:
            raise ValueError('open_bar and close_bar must be positive in manual mode.')
        open_bar_values = [int(open_bar_cfg)]
        for_num_3 = 1
    else:
        if bar_step == 0:
            raise ValueError('bar_step cannot be 0.')
        if threshold_step == 0:
            raise ValueError('threshold_step cannot be 0.')
        if step3 <= 0:
            raise ValueError('step3 must be positive.')
        if int(for_num_3) <= 0:
            raise ValueError('for_num_3 must be positive.')
        if open_threshold_stop_flat_rounds <= 0:
            raise ValueError('open_threshold_stop_flat_rounds must be positive.')
        if open_threshold_max_iterations <= 0:
            raise ValueError('open_threshold_max_iterations must be positive.')
        open_bar_values = build_int_search_values(
            int(open_bar_cfg),
            int(bar_end),
            int(bar_step),
        )
        if len(open_bar_values) == 0:
            raise ValueError('open_bar search range is empty.')
        resume_progress = load_progress_json(progress_json_path)
        if (
            resume_progress is not None
            and resume_progress.get('signature') == progress_signature
            and resume_progress.get('status') in ('running', 'interrupted')
        ):
            outcome_stats = load_progress_summary(progress_summary_path)
            executed_run_count = int(
                resume_progress.get('executed_run_count', len(outcome_stats))
            )
            next_cursor = resume_progress.get('next_cursor', {})
            resume_open_bar_index = int(next_cursor.get('open_bar_index', 0))
            resume_threshold_iter = int(next_cursor.get('threshold_iter', 0))
            resume_open_cont_iter = int(next_cursor.get('open_cont_iter', 0))
            resume_outer_state = resume_progress.get(
                'outer_state', resume_outer_state)
            resume_inner_state = resume_progress.get(
                'inner_state', resume_inner_state)
            print(
                '[Grid] resume from progress json: '
                + progress_json_path
            )
            print(
                '[Grid] next cursor: '
                + f'open_bar_index={resume_open_bar_index}, '
                + f'threshold_iter={resume_threshold_iter}, '
                + f'open_cont_iter={resume_open_cont_iter}'
            )
        else:
            if resume_progress is not None:
                print('[Grid] progress json exists but signature does not match, start fresh.')
            executed_run_count = 0

    print(
        '[Main] open_bar start=' + str(open_bar_cfg)
        + ' end=' + str(bar_end)
        + ' step=' + str(bar_step)
    )
    if run_mode == 'grid':
        print(
            '[Grid] open_threshold start=' + str(open_threshold)
            + ' step=' + str(threshold_step)
        )
        print(
            '[Grid] open_continous_threshold starts from open_threshold'
            + ' step=' + str(step3)
            + ' max=' + str(for_num_3)
        )
        print(
            '[Grid] stop threshold loops after '
            + str(open_threshold_stop_flat_rounds)
            + ' unchanged trade-count steps'
        )
    else:
        print(
            '[Manual] open_bar=' + str(open_bar_cfg)
            + ' close_bar=' + str(close_bar_cfg)
        )
        print(
            '[Manual] open_threshold=' + str(open_threshold)
            + ' open_continous_threshold=' + str(open_continous_threshold)
            + ' open_withdrawal_threshold=' + str(open_withdrawal_threshold)
        )

    if run_mode == 'manual':
        executed_run_count = 0

    open_bar_pairs = list(enumerate(open_bar_values))
    if run_mode == 'grid':
        open_bar_pairs = open_bar_pairs[resume_open_bar_index:]

    for open_bar_index, open_bar_runtime in open_bar_pairs:
        if run_mode == 'grid':
            close_bar_runtime = open_bar_runtime
        else:
            close_bar_runtime = int(close_bar_cfg)
        if run_mode == 'grid' and open_bar_index == resume_open_bar_index:
            last_open_threshold_trade_count = resume_outer_state.get(
                'last_open_threshold_trade_count')
            unchanged_open_threshold_steps = int(
                resume_outer_state.get('unchanged_open_threshold_steps', 0)
            )
            threshold_start_iter = resume_threshold_iter
        else:
            last_open_threshold_trade_count = None
            unchanged_open_threshold_steps = 0
            threshold_start_iter = 0

        if run_mode == 'grid':
            threshold_iterations = range(
                int(threshold_start_iter),
                int(open_threshold_max_iterations)
            )
        else:
            threshold_iterations = [0]

        for threshold_iter in threshold_iterations:
            if run_mode == 'grid':
                open_threshold_runtime = round(
                    open_threshold + (threshold_iter * threshold_step),
                    10,
                )
                print(
                    f'\n[Grid] open_bar={open_bar_runtime} '
                    + f'open_threshold={open_threshold_runtime}'
                )
                if (
                    open_bar_index == resume_open_bar_index
                    and threshold_iter == resume_threshold_iter
                ):
                    open_cont_start_iter = resume_open_cont_iter
                    last_open_cont_trade_count = resume_inner_state.get(
                        'last_open_cont_trade_count')
                    unchanged_open_cont_steps = int(
                        resume_inner_state.get('unchanged_open_cont_steps', 0)
                    )
                    outer_reference_trade_count = resume_inner_state.get(
                        'outer_reference_trade_count')
                else:
                    open_cont_start_iter = 0
                    last_open_cont_trade_count = None
                    unchanged_open_cont_steps = 0
                    outer_reference_trade_count = None
                open_cont_iterations = range(
                    int(open_cont_start_iter),
                    int(for_num_3)
                )
            else:
                print(
                    f'\n[Manual] open_bar={open_bar_runtime} '
                    + f'close_bar={close_bar_runtime}'
                )
                open_threshold_runtime = float(open_threshold)
                open_cont_iterations = [0]

            for open_cont_iter in open_cont_iterations:
                if run_mode == 'grid':
                    open_continous_threshold_runtime = round(
                        open_threshold_runtime + (open_cont_iter * step3),
                        10,
                    )
                    print(
                        '[Grid]   open_continous_threshold='
                        + str(open_continous_threshold_runtime)
                    )
                else:
                    open_continous_threshold_runtime = float(
                        open_continous_threshold
                    )

                open_bar_value = open_bar_runtime
                open_threshold_value = open_threshold_runtime
                if run_mode == 'grid':
                    open_withdrawal_threshold_value = open_threshold_runtime
                    close_threshold_value = open_threshold_runtime
                else:
                    open_withdrawal_threshold_value = open_withdrawal_threshold
                    close_threshold_value = close_threshold
                close_bar_value = close_bar_runtime
                open_continous_threshold_value = open_continous_threshold_runtime
                close_withdrawal_threshold_value = close_withdrawal_threshold
                open_bar2_runtime = open_bar2_cfg
                open_threshold2_runtime = open_threshold2
                open_continous_threshold2_runtime = open_continous_threshold2
                close_withdrawal_threshold2_runtime = close_withdrawal_threshold2
                commision_percent_cfg = commision_percent
                capital_cfg = capital

                if open_threshold_value < open_withdrawal_threshold_value:
                    print('open_threshold不可小于open_withdrawal_threshold')
                    continue
                if open_continous_threshold_value < open_threshold_value:
                    print('open_continous_threshold不可小于open_threshold')
                    continue
                if open_continous_threshold_value < close_withdrawal_threshold_value:
                    print('open_continous_threshold不可小于close_withdrawal_threshold')
                    continue

                params = {
                    'open_bar': open_bar_value,
                    'open_threshold': open_threshold_value,
                    'open_continous_threshold': open_continous_threshold_value,
                    'open_withdrawal_threshold': open_withdrawal_threshold_value,
                    'close_bar': close_bar_value,
                    'close_threshold': close_threshold_value,
                    'close_withdrawal_threshold': close_withdrawal_threshold_value,
                    'open_continous_threshold2': open_continous_threshold2_runtime,
                    'close_withdrawal_threshold2': close_withdrawal_threshold2_runtime,
                    'round_precision': ROUND_PRECISION,
                }

                strategy = MomentumStrategy(params)
                engine = BacktestEngine(
                    underlying, strategy, capital_cfg,
                    ROUND_PRECISION, commision_percent_cfg)
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
                    Capital_outcome = 100
                perf_outcome = performance.reset_index(
                    drop=True)[['date', 'capital']]

                count_tag = (
                    str(round(withdrawal_close_count, 4))
                    + '+' + str(round(speed_close_count, 4))
                )
                param_tag = (
                    'om' + str(round(open_bar_value, 4))
                    + ' o' + str(round(open_threshold_value, 4))
                    + ' oc' + str(round(open_continous_threshold_value, 4))
                    + ' cm' + str(round(close_bar_value, 4))
                    + ' c' + str(round(close_threshold_value, 4))
                    + ' ow' + str(round(open_withdrawal_threshold_value, 4))
                    + ' cw' + str(round(close_withdrawal_threshold_value, 4))
                )
                result_tag = param_tag + ' ' + count_tag

                print(str(range_start_label) + '-' + str(range_end_label))
                print('total close count = ' + str(total_trade_count))
                print('withdrawal close count = '
                      + str(round(withdrawal_close_count, 4)))
                print('speed close count = '
                      + str(round(speed_close_count, 4)))
                print(result_tag)
                print('profit: ' + str(round(performance.capital.iloc[-1], 2)))

                save_name = run_name + ' ' + result_tag
                fig1_title = str(Capital_outcome) + ' ' + save_name
                if save_static_plot:
                    plot_ext = 'pdf' if save_plot_as_pdf else 'png'
                    fig1_path = ('./result/%s long outcome/' % file_name
                                 + ' ' + str(Capital_outcome)
                                 + save_name + f' Long.{plot_ext}')
                    close_fig = (run_mode != 'manual') or (len(transactions_df) == 0)
                    plot_backtest_chart(
                        underlying, transactions_df, perf_outcome,
                        title=fig1_title,
                        save_path=fig1_path,
                        close_fig=close_fig)

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

                perf_name = (
                    param_tag + ' ' + count_tag
                    + ' Long ' + run_name
                    + ' ' + str(Capital_outcome)
                    + ' perf.xlsx'
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
                        + param_tag + ' ' + count_tag
                        + ' Long ' + run_name
                        + ' ' + str(Capital_outcome)
                        + ' trans.xlsx', engine='xlsxwriter')
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
                    'open_bar': open_bar_value,
                    'close_bar': close_bar_value,
                    'open_threshold': open_threshold_value,
                    'open_continous_threshold': open_continous_threshold_value,
                    'open_withdrawal_threshold': open_withdrawal_threshold_value,
                    'close_threshold': close_threshold_value,
                    'close_withdrawal_threshold': close_withdrawal_threshold_value,
                    'withdrawal_close_count': withdrawal_close_count,
                    'speed_close_count': speed_close_count,
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
                    outcome_stats.at[outcome_index, metric_name] = metric_value
                if run_mode == 'grid':
                    append_progress_summary(
                        progress_summary_path,
                        outcome_index,
                        summary_row,
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

                    inner_stop_now = (
                        unchanged_open_cont_steps
                        >= open_threshold_stop_flat_rounds
                    )
                    inner_exhausted_now = (
                        open_cont_iter + 1 >= int(for_num_3)
                    )

                    next_cursor = None
                    next_outer_state = {
                        'last_open_threshold_trade_count': last_open_threshold_trade_count,
                        'unchanged_open_threshold_steps': unchanged_open_threshold_steps,
                    }
                    next_inner_state = {
                        'last_open_cont_trade_count': last_open_cont_trade_count,
                        'unchanged_open_cont_steps': unchanged_open_cont_steps,
                        'outer_reference_trade_count': outer_reference_trade_count,
                    }

                    if not inner_stop_now and not inner_exhausted_now:
                        next_cursor = {
                            'open_bar_index': open_bar_index,
                            'threshold_iter': threshold_iter,
                            'open_cont_iter': open_cont_iter + 1,
                        }
                    else:
                        if outer_reference_trade_count is None:
                            next_open_threshold_trade_count = last_open_threshold_trade_count
                            next_unchanged_open_threshold_steps = unchanged_open_threshold_steps
                        elif (
                            last_open_threshold_trade_count is None
                            or outer_reference_trade_count
                            != last_open_threshold_trade_count
                        ):
                            next_open_threshold_trade_count = outer_reference_trade_count
                            next_unchanged_open_threshold_steps = 0
                        else:
                            next_open_threshold_trade_count = outer_reference_trade_count
                            next_unchanged_open_threshold_steps = (
                                unchanged_open_threshold_steps + 1
                            )

                        outer_stop_now = (
                            next_unchanged_open_threshold_steps
                            >= open_threshold_stop_flat_rounds
                        )
                        threshold_exhausted_now = (
                            threshold_iter + 1
                            >= int(open_threshold_max_iterations)
                        )

                        if (
                            not outer_stop_now
                            and not threshold_exhausted_now
                        ):
                            next_cursor = {
                                'open_bar_index': open_bar_index,
                                'threshold_iter': threshold_iter + 1,
                                'open_cont_iter': 0,
                            }
                            next_outer_state = {
                                'last_open_threshold_trade_count': next_open_threshold_trade_count,
                                'unchanged_open_threshold_steps': next_unchanged_open_threshold_steps,
                            }
                            next_inner_state = {
                                'last_open_cont_trade_count': None,
                                'unchanged_open_cont_steps': 0,
                                'outer_reference_trade_count': None,
                            }
                        else:
                            next_cursor = {
                                'open_bar_index': open_bar_index + 1,
                                'threshold_iter': 0,
                                'open_cont_iter': 0,
                            }
                            next_outer_state = {
                                'last_open_threshold_trade_count': None,
                                'unchanged_open_threshold_steps': 0,
                            }
                            next_inner_state = {
                                'last_open_cont_trade_count': None,
                                'unchanged_open_cont_steps': 0,
                                'outer_reference_trade_count': None,
                            }

                    save_progress_json(
                        progress_json_path,
                        {
                            'status': 'running',
                            'signature': progress_signature,
                            'executed_run_count': executed_run_count,
                            'next_cursor': next_cursor,
                            'outer_state': next_outer_state,
                            'inner_state': next_inner_state,
                            'last_completed': {
                                'param_tag': outcome_index,
                                'open_bar': open_bar_value,
                                'open_threshold': open_threshold_value,
                                'open_continous_threshold': open_continous_threshold_value,
                                'total_trade_count': total_trade_count,
                                'capital': summary_metrics['final_capital'],
                            },
                            'progress_summary_path': progress_summary_path,
                        }
                    )

                    if unchanged_open_cont_steps >= open_threshold_stop_flat_rounds:
                        print(
                            '[Grid] stop open_cont loop at '
                            + f'open_bar={open_bar_runtime} '
                            + f'open_threshold={open_threshold_runtime}: '
                            + 'total trade count unchanged for '
                            + f'{open_threshold_stop_flat_rounds} steps.'
                        )
                        break
            else:
                if run_mode == 'grid':
                    print(
                        '[Grid] reached for_num_3='
                        + str(for_num_3)
                        + f' at open_bar={open_bar_runtime} '
                        + f'open_threshold={open_threshold_runtime}.'
                    )

            if run_mode == 'grid':
                if outer_reference_trade_count is None:
                    continue
                if (
                    last_open_threshold_trade_count is None
                    or outer_reference_trade_count != last_open_threshold_trade_count
                ):
                    unchanged_open_threshold_steps = 0
                else:
                    unchanged_open_threshold_steps += 1
                last_open_threshold_trade_count = outer_reference_trade_count

                if unchanged_open_threshold_steps >= open_threshold_stop_flat_rounds:
                    print(
                        '[Grid] stop open_threshold loop at '
                        + f'open_bar={open_bar_runtime}: '
                        + 'base total trade count unchanged for '
                        + f'{open_threshold_stop_flat_rounds} steps.'
                    )
                    break
        else:
            if run_mode == 'grid':
                print(
                    '[Grid] reached open_threshold_max_iterations='
                    + str(open_threshold_max_iterations)
                    + f' at open_bar={open_bar_runtime}.'
                )

    if run_mode == 'grid':
        save_progress_json(
            progress_json_path,
            {
                'status': 'completed',
                'signature': progress_signature,
                'executed_run_count': executed_run_count,
                'next_cursor': {
                    'open_bar_index': len(open_bar_values),
                    'threshold_iter': 0,
                    'open_cont_iter': 0,
                },
                'outer_state': {
                    'last_open_threshold_trade_count': None,
                    'unchanged_open_threshold_steps': 0,
                },
                'inner_state': {
                    'last_open_cont_trade_count': None,
                    'unchanged_open_cont_steps': 0,
                    'outer_reference_trade_count': None,
                },
                'progress_summary_path': progress_summary_path,
            }
        )

    print("\ntime = --- %s seconds ---" % (time.time() - start_time))

    # 多参数对比图
    if executed_run_count == 0:
        raise ValueError('No parameter combination was executed.')

    if run_mode == 'grid' and executed_run_count > 1:
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
        plt.title('stats ' + run_name)
        os.makedirs('./result/stats %s long outcome/' % file_name, exist_ok=True)
        stats_plot_ext = 'pdf' if save_plot_as_pdf else 'png'
        plt.savefig('./result/stats %s long outcome/' % file_name
                    + ' ' + run_name + ' '
                    + str(executed_run_count) + ' '
                    + f'all outcome.{stats_plot_ext}', dpi=1000)
    else:
        disk_path = './result/'
        open_excel = False
        if open_excel:
            os.startfile(
                disk_path + '%s long outcome/perf/' % file_name + perf_name)

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
                close_type_1_df['target'], c=sell_wd_color, s=10)
            close_type_2_df = sell_record[sell_record['Close_type'] == 2]
            scatter_b = ax2.scatter(
                close_type_2_df.index,
                close_type_2_df['target'], c=sell_speed_color, s=10)

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
            + ' ' + run_name
        )
        plt.title('%s' % fig2_title)

        xaxis1 = detail_df.index
        yaxis1 = detail_df.capital
        xaxis2 = x.index
        yaxis2 = x
        plt.plot(xaxis1, yaxis1, linewidth=1.2, color=accent_blue)
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
                    color=accent_blue, linewidth=2.0, alpha=0.8, zorder=1)
                buy_idx = None
                buy_y = None
        ax2.xaxis.set_major_locator(plt.MaxNLocator(12))
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

    summary_base = (
        summary_dir
        + ' ' + run_name + ' '
        + str(executed_run_count)
        + ' all outcome'
    )
    outcome_stats.sort_index().to_excel(summary_base + '.xlsx')
