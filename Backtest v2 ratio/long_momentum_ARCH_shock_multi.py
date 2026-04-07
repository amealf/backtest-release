# -*- coding: utf-8 -*-
"""
Long Momentum Strategy - 动量做多策略
=====================================
策略入口脚本：包含 MomentumStrategy 类、参数循环、绘图、Excel 输出。
依赖 backtest_main.py 中的通用框架。
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
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
data_folder_path = r"D:\Code\data\20260326\\"
data_file_name = "xagusd_30s_all"

# 回测区间
# data_selection_mode:
# 'index' = 使用 start_index / end_index，在原始数据上按切片语义 [start_index, end_index) 取数
# 'date' = 使用 start_date / end_date，在原始数据上按时间 between 取数
data_selection_mode = 'date'
start_index = 0
end_index = 'latest'  # 或 'latest'
start_date = '20250601'
end_date = '20250615'  # 或 '2024-12-31 23:59:59'
only_close = False

# 重采样设置：设为 '' 表示直接使用原始周期
# 例如 '1min' / '5min' / '15min' / '1H'
resample_rule = '5min'

# 运行模式：
# 'manual' = 使用当前参数直接回测，并弹出 K 线买卖点图
# 'grid' = 执行网格搜索，并输出参数结果图
run_mode = 'manual'

# shock 策略参数
volatility_method = 'garch'
shock_open_multiplier = 2.4
close_bar = 7
close_speed_sigma_multiplier = 1.2
close_wd_sigma_multiplier = 1.6

MULTI_SHOCK_CONFIG = {
    'config_key': 'vote_2of3_core',
    'label': '5min+30min+1H vote 2 of 3',
    'signal_mode': 'vote',
    'periods': ['5min', '30min', '1H'],
    'agreement_required': 2,
    'min_ready_count': 3,
    'require_base_period': True,
    'score_weights': {},
}

RUN_LABEL_SUFFIX = ''

# Grid search
shock_open_multiplier_values = [1.8, 2.1, 2.4, 2.7, 3.0, 3.3]
close_bar_values = [3, 5, 7, 9, 12, 15]
close_speed_sigma_multiplier_values = [0.6, 0.9, 1.2, 1.5]
close_wd_sigma_multiplier_values = [1.0, 1.4, 1.8, 2.2, 2.6]

commision_percent = 0.000
capital = 100.0
# 网格搜索时建议关闭逐次图表与明细导出。
EXPORT_INTERACTIVE_HTML = False
EXPORT_STATS = False
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

VOLATILITY_METHOD_SPECS = {
    'garch': {
        'subdir': 'garch forecast',
        'file_template': 'period_{period_label} garch forecast.parquet',
        'price_col': 'garch_sigma_price_trade_bar',
        'return_col': 'garch_sigma_return_trade_bar',
    }
}

DIRECT_PERIOD_SOURCE_SPECS = {
    '1min': {'suffix': '1_min', 'bar_seconds': 60},
    '5min': {'suffix': '5_mins', 'bar_seconds': 300},
    '15min': {'suffix': '15_mins', 'bar_seconds': 900},
    '30min': {'suffix': '30_mins', 'bar_seconds': 1800},
    'day': {'suffix': '1_day', 'bar_seconds': 86400},
}

PERIOD_TOKEN_ALIASES = {
    '1min': 'p1min',
    '5min': 'p5min',
    '10min': 'p10min',
    '15min': 'p15min',
    '30min': 'p30min',
    '1h': 'p1h',
    '2h': 'p2h',
    '4h': 'p4h',
    'day': 'pday',
}


def build_result_root(file_name: str) -> Path:
    return Path('./result') / f'{file_name} long shock multi outcome'


def build_stats_result_root(file_name: str) -> Path:
    return Path('./result') / f'stats {file_name} long shock multi outcome'


def normalize_rule_token(rule: str) -> str:
    return str(rule or '').strip().lower().replace(' ', '')


def normalize_period_label(period_label: str) -> str:
    token = normalize_rule_token(period_label)
    if token in ('1hour', '60min'):
        return '1H'
    if token in ('2hour', '120min'):
        return '2H'
    if token in ('4hour', '240min'):
        return '4H'
    if token in ('1day', 'd', '1d'):
        return 'day'
    if token.endswith('h') and token[:-1].isdigit():
        return f"{int(token[:-1])}H"
    if token.endswith('min') and token[:-3].isdigit():
        return f"{int(token[:-3])}min"
    return str(period_label).strip()


def period_label_to_token(period_label: str) -> str:
    normalized = normalize_rule_token(normalize_period_label(period_label))
    if normalized not in PERIOD_TOKEN_ALIASES:
        raise ValueError('Unsupported period label for multi shock: ' + str(period_label))
    return PERIOD_TOKEN_ALIASES[normalized]


def normalize_multi_shock_config(config: dict | None, base_period_label: str) -> dict:
    raw = dict(config or {})
    periods = raw.get('periods') or [base_period_label]
    normalized_periods = [normalize_period_label(value) for value in periods]
    base_period = normalize_period_label(base_period_label)
    if base_period not in normalized_periods:
        normalized_periods = [base_period] + normalized_periods

    normalized_weights = {}
    for key, value in dict(raw.get('score_weights') or {}).items():
        normalized_weights[normalize_period_label(key)] = float(value)

    signal_mode = str(raw.get('signal_mode', 'single')).strip().lower()
    if signal_mode not in ('single', 'vote', 'blend', 'max_sigma'):
        raise ValueError('Unsupported multi shock signal_mode: ' + signal_mode)

    agreement_required = int(raw.get('agreement_required', 1))
    min_ready_count = int(raw.get('min_ready_count', len(normalized_periods)))

    return {
        'config_key': str(raw.get('config_key') or signal_mode).strip(),
        'label': str(raw.get('label') or signal_mode).strip(),
        'signal_mode': signal_mode,
        'periods': normalized_periods,
        'base_period_label': base_period,
        'agreement_required': max(1, agreement_required),
        'min_ready_count': max(1, min_ready_count),
        'require_base_period': bool(raw.get('require_base_period', True)),
        'score_weights': normalized_weights,
    }


def _count_decimal_places(text) -> int:
    value = str(text).strip()
    if value == '' or value.lower() == 'nan':
        return 0
    if '.' not in value:
        return 0
    return len(value.split('.')[-1].rstrip('0'))


def try_load_direct_period_source(
        folder_path: str,
        file_name: str,
        period_rule: str):
    spec = DIRECT_PERIOD_SOURCE_SPECS.get(normalize_rule_token(period_rule))
    if spec is None:
        return None

    csv_path = Path(folder_path) / f"{file_name}_{spec['suffix']}.csv"
    if not csv_path.exists():
        return None

    raw_df = pd.read_csv(csv_path)
    if raw_df.empty:
        raise ValueError('Direct period file is empty: ' + str(csv_path))

    lower_map = {str(col).strip().lower(): col for col in raw_df.columns}
    date_col = None
    for candidate in ('datetime', 'date', 'time'):
        if candidate in lower_map:
            date_col = lower_map[candidate]
            break
    if date_col is None:
        raise ValueError('Direct period file missing datetime column: ' + str(csv_path))

    source_cols = {}
    for target in ('open', 'high', 'low', 'close'):
        if target not in lower_map:
            raise ValueError(
                'Direct period file missing column: '
                + target
                + ' | '
                + str(csv_path)
            )
        source_cols[target] = lower_map[target]

    volume_col = lower_map.get('volume') or lower_map.get('vol')
    normalized = pd.DataFrame({
        'Date': pd.to_datetime(raw_df[date_col], errors='coerce'),
        'open': pd.to_numeric(raw_df[source_cols['open']], errors='coerce'),
        'high': pd.to_numeric(raw_df[source_cols['high']], errors='coerce'),
        'low': pd.to_numeric(raw_df[source_cols['low']], errors='coerce'),
        'close': pd.to_numeric(raw_df[source_cols['close']], errors='coerce'),
    })
    if volume_col is not None:
        normalized['vol'] = pd.to_numeric(raw_df[volume_col], errors='coerce')
    else:
        normalized['vol'] = 0.0

    normalized = normalized.dropna(
        subset=['Date', 'open', 'high', 'low', 'close']
    ).sort_values('Date').reset_index(drop=True)
    if normalized.empty:
        raise ValueError('No valid rows remain in direct period file: ' + str(csv_path))

    round_precision = 0
    for col in source_cols.values():
        round_precision = max(
            round_precision,
            int(raw_df[col].astype(str).map(_count_decimal_places).max()),
        )

    return normalized, round_precision, int(spec['bar_seconds']), csv_path


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


def get_volatility_method_spec(vol_method: str) -> dict:
    method = str(vol_method).strip().lower()
    if method not in VOLATILITY_METHOD_SPECS:
        raise ValueError(
            'Unsupported volatility_method: '
            + str(vol_method)
            + '. Available: '
            + ', '.join(sorted(VOLATILITY_METHOD_SPECS))
        )
    return VOLATILITY_METHOD_SPECS[method]


def build_volatility_parquet_path(
        file_name: str,
        period_label: str,
        vol_method: str) -> Path:
    spec = get_volatility_method_spec(vol_method)
    return (
        Path('./result')
        / spec['subdir']
        / spec['file_template'].format(period_label=period_label)
    )


def load_shifted_volatility_forecast(
        file_name: str,
        period_label: str,
        vol_method: str) -> tuple[pd.DataFrame, Path]:
    spec = get_volatility_method_spec(vol_method)
    parquet_path = build_volatility_parquet_path(
        file_name=file_name,
        period_label=period_label,
        vol_method=vol_method,
    )
    if not parquet_path.exists():
        raise FileNotFoundError(
            'Volatility forecast file not found: ' + str(parquet_path)
        )

    forecast_df = pd.read_parquet(
        parquet_path,
        columns=[
            'Date',
            spec['price_col'],
            spec['return_col'],
        ],
    ).copy()
    period_token = period_label_to_token(period_label)
    forecast_df['merge_date'] = pd.to_datetime(forecast_df['Date'], errors='coerce')
    forecast_df = forecast_df.dropna(subset=['merge_date']).sort_values('merge_date')
    forecast_df = forecast_df.rename(columns={
        spec['price_col']: f'vol_forecast_price_{period_token}',
        spec['return_col']: f'vol_forecast_return_{period_token}',
    })
    forecast_df = forecast_df.drop_duplicates(
        subset=['merge_date'], keep='last'
    )
    forecast_df[f'entry_sigma_price_{period_token}'] = (
        forecast_df[f'vol_forecast_price_{period_token}'].shift(1)
    )
    forecast_df[f'entry_sigma_return_{period_token}'] = (
        forecast_df[f'vol_forecast_return_{period_token}'].shift(1)
    )
    keep_cols = [
        'merge_date',
        f'vol_forecast_price_{period_token}',
        f'vol_forecast_return_{period_token}',
        f'entry_sigma_price_{period_token}',
        f'entry_sigma_return_{period_token}',
    ]
    return forecast_df[keep_cols].copy(), parquet_path


def merge_volatility_forecast(
        quote: pd.DataFrame,
        file_name: str,
        period_label: str,
        vol_method: str) -> tuple[pd.DataFrame, Path]:
    forecast_df, parquet_path = load_shifted_volatility_forecast(
        file_name=file_name,
        period_label=period_label,
        vol_method=vol_method,
    )
    period_token = period_label_to_token(period_label)

    merged = quote.copy()
    merged['merge_date'] = pd.to_datetime(merged['Date'], errors='coerce')
    merged = merged.sort_values('merge_date').reset_index(drop=True)
    merged = pd.merge_asof(
        merged,
        forecast_df.sort_values('merge_date'),
        on='merge_date',
        direction='backward',
        allow_exact_matches=True,
    )
    merged['vol_forecast_price'] = merged[f'vol_forecast_price_{period_token}']
    merged['vol_forecast_return'] = merged[f'vol_forecast_return_{period_token}']
    merged['entry_sigma_source'] = merged[f'entry_sigma_price_{period_token}']
    merged['entry_sigma_ready'] = merged['entry_sigma_source'].notna().astype(int)
    merged = merged.drop(columns=['merge_date'])
    merged['vol_ready'] = merged['vol_forecast_price'].notna().astype(int)
    return merged, parquet_path


def build_shock_signal_for_period(
        quote: pd.DataFrame,
        file_name: str,
        period_label: str,
        vol_method: str,
        multi_config: dict | None = None) -> tuple[pd.DataFrame, dict[str, str], dict]:
    config = normalize_multi_shock_config(multi_config, period_label)
    merged = quote.copy()
    merged['merge_date'] = pd.to_datetime(merged['Date'], errors='coerce')
    merged = merged.sort_values('merge_date').reset_index(drop=True)

    forecast_path_map: dict[str, str] = {}
    for period in config['periods']:
        forecast_df, parquet_path = load_shifted_volatility_forecast(
            file_name=file_name,
            period_label=period,
            vol_method=vol_method,
        )
        merged = pd.merge_asof(
            merged,
            forecast_df.sort_values('merge_date'),
            on='merge_date',
            direction='backward',
            allow_exact_matches=True,
        )
        forecast_path_map[period] = str(parquet_path)

    base_token = period_label_to_token(config['base_period_label'])
    merged['vol_forecast_price'] = merged[f'vol_forecast_price_{base_token}']
    merged['vol_forecast_return'] = merged[f'vol_forecast_return_{base_token}']
    merged['entry_sigma_source'] = merged[f'entry_sigma_price_{base_token}']
    merged['entry_sigma_ready'] = merged['entry_sigma_source'].notna().astype(int)
    merged['vol_ready'] = merged['vol_forecast_price'].notna().astype(int)
    merged = merged.drop(columns=['merge_date'])
    return merged, forecast_path_map, config


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


def build_shock_param_tag(
        shock_open_multiplier: float,
        close_bar: int,
        close_speed_sigma_multiplier: float,
        close_wd_sigma_multiplier: float,
        volatility_method: str) -> str:
    return (
        'vm' + str(volatility_method).strip().lower()
        + ' sha' + str(round(shock_open_multiplier, 4))
        + ' cb' + str(round(close_bar, 4))
        + ' csm' + str(round(close_speed_sigma_multiplier, 4))
        + ' cwd' + str(round(close_wd_sigma_multiplier, 4))
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


def build_planned_param_tags_long_shock(
        shock_open_multiplier_values: list[float],
        close_bar_values: list[int],
        close_speed_sigma_multiplier_values: list[float],
        close_wd_sigma_multiplier_values: list[float],
        volatility_method: str) -> set[str]:
    planned_tags = set()
    for shock_open_multiplier_value in shock_open_multiplier_values:
        for close_bar_value in close_bar_values:
            for close_speed_sigma_multiplier_value in close_speed_sigma_multiplier_values:
                for close_wd_sigma_multiplier_value in close_wd_sigma_multiplier_values:
                    planned_tags.add(build_shock_param_tag(
                        shock_open_multiplier_value,
                        close_bar_value,
                        close_speed_sigma_multiplier_value,
                        close_wd_sigma_multiplier_value,
                        volatility_method=volatility_method,
                    ))
    return planned_tags


def load_existing_outcome_stats(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path, index_col=0)
    return df[~df.index.duplicated(keep='last')]


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
    if go is None or make_subplots is None:
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

    fig_html = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.76, 0.24],
        subplot_titles=('Price', 'Realized / Predicted'),
    )
    subplot_title_annotations = list(fig_html.layout.annotations)
    x_index = underlying1.index.to_numpy()
    x_min = int(x_index[0]) if len(x_index) > 0 else 0
    x_max = int(x_index[-1]) if len(x_index) > 0 else 1
    x_span = max(1, x_max - x_min + 1)
    x_left_pad = max(1, int(round(x_span * 0.006)))
    x_right_pad = max(1, int(round(x_span * 0.010)))
    plot_x = detail_df.index.to_numpy()
    open_values = pd.to_numeric(underlying1.get('open', np.nan), errors='coerce')
    high_values = pd.to_numeric(underlying1.get('high', np.nan), errors='coerce')
    low_values = pd.to_numeric(underlying1.get('low', np.nan), errors='coerce')
    close_values = pd.to_numeric(underlying1.get('close', np.nan), errors='coerce')
    date_values = pd.to_datetime(underlying1.get('Date'), errors='coerce')
    bar_return_pct = np.where(
        open_values > 0,
        (close_values / open_values - 1.0) * 100.0,
        np.nan,
    )
    bar_range_pct = np.where(
        open_values > 0,
        (high_values - low_values) / open_values * 100.0,
        np.nan,
    )
    kline_hover_text = (
        'bar_return_pct='
        + pd.Series(bar_return_pct, index=underlying1.index).map(
            lambda x: 'nan' if pd.isna(x) else f'{float(x):.4f}%'
        )
        + '<br>bar_range_pct='
        + pd.Series(bar_range_pct, index=underlying1.index).map(
            lambda x: 'nan' if pd.isna(x) else f'{float(x):.4f}%'
        )
        + '<br>Date='
        + pd.Series(date_values, index=underlying1.index).map(
            lambda x: 'nan' if pd.isna(x) else x.strftime('%Y-%m-%d %H:%M:%S')
        )
        + '<br>open='
        + open_values.map(lambda x: 'nan' if pd.isna(x) else f'{float(x):.6f}')
        + '<br>high='
        + high_values.map(lambda x: 'nan' if pd.isna(x) else f'{float(x):.6f}')
        + '<br>low='
        + low_values.map(lambda x: 'nan' if pd.isna(x) else f'{float(x):.6f}')
        + '<br>close='
        + close_values.map(lambda x: 'nan' if pd.isna(x) else f'{float(x):.6f}')
    )

    entry_sigma_source_series = pd.to_numeric(
        detail_df.get('entry_sigma_source', np.nan),
        errors='coerce',
    )
    open_series = pd.to_numeric(underlying1.get('open', np.nan), errors='coerce')
    high_series = pd.to_numeric(underlying1.get('high', np.nan), errors='coerce')
    realized_predicted = pd.Series(np.nan, index=detail_df.index, dtype='float64')
    valid_ratio_mask = (
        entry_sigma_source_series.notna()
        & (entry_sigma_source_series > 0)
        & open_series.notna()
        & high_series.notna()
    )
    realized_predicted.loc[valid_ratio_mask] = (
        (high_series.loc[valid_ratio_mask] - open_series.loc[valid_ratio_mask])
        / entry_sigma_source_series.loc[valid_ratio_mask]
    )
    threshold_series = pd.to_numeric(
        detail_df.get('shock_open_multiplier_runtime', np.nan),
        errors='coerce',
    )
    valid_threshold_series = threshold_series.dropna()
    entry_threshold = (
        float(valid_threshold_series.iloc[0])
        if len(valid_threshold_series) > 0 else np.nan
    )
    ratio_dates = pd.to_datetime(detail_df.get('Date'), errors='coerce')
    ratio_customdata = np.array([
        dt.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(dt) else 'nan'
        for dt in ratio_dates
    ], dtype=object)

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
    ), row=1, col=1)

    fig_html.add_trace(go.Candlestick(
        x=x_index,
        open=underlying1['open'] / factor * 100,
        high=underlying1['high'] / factor * 100,
        low=underlying1['low'] / factor * 100,
        close=underlying1['close'] / factor * 100,
        name='price',
        increasing=dict(
            line=dict(color='rgba(185, 185, 185, 0.9)', width=0.8),
            fillcolor='rgba(245, 245, 245, 0.9)'
        ),
        decreasing=dict(
            line=dict(color='rgba(85, 85, 85, 0.9)', width=0.8),
            fillcolor='rgba(120, 120, 120, 0.9)'
        ),
        hovertext=kline_hover_text,
        hoverinfo='text',
    ), row=1, col=1)
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
                + 'entry_sigma: ' + _safe_val(pref_data, 'entry_sigma', 6) + '<br>'
                + 'shock_mult: ' + _safe_val(pref_data, 'shock_open_multiplier_runtime', 4) + '<br>'
                + 'multi_mode: ' + _safe_val(pref_data, 'multi_trigger_mode') + '<br>'
                + 'multi_cfg: ' + _safe_val(pref_data, 'multi_config_key') + '<br>'
                + 'ready/pass: '
                + _safe_val(pref_data, 'multi_ready_count', 0)
                + '/'
                + _safe_val(pref_data, 'multi_pass_count', 0) + '<br>'
                + 'selected: ' + _safe_val(pref_data, 'multi_selected_periods') + '<br>'
                + 'scores: ' + _safe_val(pref_data, 'multi_period_scores') + '<br>'
                + 'sigmas: ' + _safe_val(pref_data, 'multi_period_sigmas') + '<br>'
                + 'trigger: ' + _safe_val(pref_data, 'shock_trigger_price', 6) + '<br>'
                + 'shock_move: ' + _safe_val(pref_data, 'shock_move_abs', 6) + '<br>'
                + 'shock_score: ' + _safe_val(pref_data, 'shock_score_high', 4) + '<br>'
                + 'execution: ' + _safe_val(pref_data, 'execution') + '<br>'
                + 'entry_date: ' + _safe_val(pref_data, 'shock_trigger_bar_date') + '<br>'
                + 'entry_price: ' + _safe_val(pref_data, 'low_price') + '<br>'
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
        ), row=1, col=1)

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
                    + 'entry_sigma: ' + _safe_val(pref_data, 'entry_sigma', 6) + '<br>'
                    + 'speed_mult: ' + _safe_val(pref_data, 'close_speed_sigma_multiplier_runtime', 4) + '<br>'
                    + 'wd_mult: ' + _safe_val(pref_data, 'close_wd_sigma_multiplier_runtime', 4) + '<br>'
                    + 'close_th: ' + _safe_val(pref_data, 'active_close_threshold', 6) + '<br>'
                    + 'close_wd_th: ' + _safe_val(pref_data, 'active_close_wd_threshold', 6) + '<br>'
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
            ), row=1, col=1)

        close_type_2_df = sell_record[sell_record['Close_type'] == 2]
        sell_2_count = int(len(close_type_2_df))
        if len(close_type_2_df) != 0:
            sell_2_texts = []
            for idx, row in close_type_2_df.iterrows():
                pref_data = detail_df.loc[idx] if idx in detail_df.index else pd.Series(dtype='object')
                sell_2_texts.append(
                    _date_text(row['Date']) + '<br>'
                    + 'entry_sigma: ' + _safe_val(pref_data, 'entry_sigma', 6) + '<br>'
                    + 'speed_mult: ' + _safe_val(pref_data, 'close_speed_sigma_multiplier_runtime', 4) + '<br>'
                    + 'wd_mult: ' + _safe_val(pref_data, 'close_wd_sigma_multiplier_runtime', 4) + '<br>'
                    + 'close_th: ' + _safe_val(pref_data, 'active_close_threshold', 6) + '<br>'
                    + 'close_wd_th: ' + _safe_val(pref_data, 'active_close_wd_threshold', 6) + '<br>'
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
            ), row=1, col=1)

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
        ), row=1, col=1)

    fig_html.add_trace(go.Scatter(
        x=plot_x,
        y=realized_predicted,
        mode='lines',
        line=dict(color='rgba(70, 70, 70, 0.55)', width=1.25),
        name='Realized / Predicted',
        connectgaps=True,
        customdata=ratio_customdata,
        hovertemplate=(
            'Date=%{customdata}<br>'
            + 'realized_predicted=%{y:.8f}<extra></extra>'
        )
    ), row=2, col=1)

    if len(long_record) != 0:
        long_ratio_y = pd.to_numeric(
            realized_predicted.reindex(long_record.index),
            errors='coerce',
        )
        fig_html.add_trace(go.Scatter(
            x=long_record.index,
            y=long_ratio_y,
            mode='markers',
            marker=dict(color='rgba(220, 40, 40, 0.95)', size=5),
            name='long_signal',
            hoverinfo='skip',
            showlegend=True,
        ), row=2, col=1)

    if pd.notna(entry_threshold):
        fig_html.add_trace(go.Scatter(
            x=plot_x,
            y=np.full(len(plot_x), entry_threshold, dtype=float),
            mode='lines',
            line=dict(color='rgba(60, 180, 75, 0.88)', width=1.2, dash='dash'),
            name=f'entry >= {entry_threshold:.2f}x',
            hoverinfo='skip',
        ), row=2, col=1)

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
        margin=dict(l=42, r=25, t=38, b=45, pad=0),
        annotations=subplot_title_annotations + trade_count_annotation,
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.35)',
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
        row=1,
        col=1,
        **x_spike_cfg,
    )
    fig_html.update_xaxes(
        title=None,
        tickfont=dict(size=10),
        showgrid=False,
        rangeslider=dict(visible=False),
        range=[x_min - x_left_pad, x_max + x_right_pad],
        autorange=False,
        row=2,
        col=1,
        **x_spike_cfg,
    )
    fig_html.update_yaxes(
        title=None,
        tickfont=dict(size=10),
        showgrid=False,
        row=1,
        col=1,
        **y_spike_cfg,
    )
    fig_html.update_yaxes(
        title='Realized / Predicted',
        tickfont=dict(size=10),
        showgrid=False,
        row=2,
        col=1,
        **y_spike_cfg,
    )

    html_dir = str(build_result_root(file_name) / 'html')
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
    """Multi-period GARCH shock 做多策略。"""

    def __init__(self, params: dict):
        super().__init__(params)
        self.start_index = 0
        self.low_index = 0
        self.holding_start_index = 0
        self.entry_execution_price = np.nan
        self.entry_sigma = np.nan
        self.entry_trigger_price = np.nan
        self.entry_bar_index = None
        self.entry_bar_date = ''
        self.pending_entry_sigma = np.nan
        self.pending_trigger_price = np.nan
        self.pending_entry_bar_index = None
        self.pending_entry_bar_date = ''
        raw_multi_config = dict(self.params.get('multi_config') or {})
        self.multi_config = normalize_multi_shock_config(
            raw_multi_config,
            raw_multi_config.get('base_period_label') or self.params.get('base_period_label'),
        )
        self.config_key = self.multi_config['config_key']
        self.signal_mode = self.multi_config['signal_mode']
        self.period_labels = list(self.multi_config['periods'])
        self.base_period_label = self.multi_config['base_period_label']
        self.agreement_required = int(self.multi_config['agreement_required'])
        self.min_ready_count = int(self.multi_config['min_ready_count'])
        self.require_base_period = bool(self.multi_config['require_base_period'])
        self.score_weights = dict(self.multi_config['score_weights'])
        self.period_specs = []
        for label in self.period_labels:
            token = period_label_to_token(label)
            self.period_specs.append({
                'label': label,
                'token': token,
                'entry_sigma_col': f'entry_sigma_price_{token}',
                'threshold_col': f'shock_threshold_abs_{token}',
                'score_col': f'shock_score_{token}',
                'pass_col': f'shock_pass_{token}',
            })

    def get_extra_columns(self) -> list:
        columns = [
            'max_inc', 'max_wd',
            'holding_wd', 'hld_wd_per', 'holding_wd_signal',
            'holding_inc', 'speed_close_signal',
            'period',
            'low_index', 'high_index',
            'low_date', 'low_price',
            'high_date', 'high_price',
            'basis_ready', 'pos_basis', 'neg_basis',
            'active_close_threshold',
            'active_close_wd_threshold',
            'entry_sigma_source',
            'entry_sigma',
            'shock_threshold_abs',
            'shock_trigger_price',
            'shock_trigger_bar_date',
            'shock_move_abs',
            'shock_score_high',
            'shock_signal',
            'shock_open_multiplier_runtime',
            'close_speed_sigma_multiplier_runtime',
            'close_wd_sigma_multiplier_runtime',
            'current_max_increase',
            'multi_ready_count',
            'multi_pass_count',
            'multi_period_count',
            'multi_required_count',
            'multi_trigger_sigma',
            'multi_trigger_mode',
            'multi_config_key',
            'multi_selected_periods',
            'multi_period_scores',
            'multi_period_sigmas',
            'multi_score_mean',
            'multi_score_min',
            'multi_score_max',
        ]
        for spec in self.period_specs:
            columns.extend([
                spec['entry_sigma_col'],
                spec['threshold_col'],
                spec['score_col'],
                spec['pass_col'],
            ])
        return columns

    def get_default_columns(self) -> dict:
        defaults = {
            'holding_wd_signal': 0.0,
            'speed_close_signal': 0.0,
            'shock_signal': 0.0,
            'multi_ready_count': 0.0,
            'multi_pass_count': 0.0,
            'multi_period_count': float(len(self.period_specs)),
            'multi_required_count': float(self.agreement_required),
        }
        for spec in self.period_specs:
            defaults[spec['pass_col']] = 0.0
        return defaults

    def on_bar_record(self, ctx: BarContext):
        signal = ctx.signal
        index = ctx.index
        signal.at[index, 'multi_config_key'] = self.config_key
        signal.at[index, 'multi_trigger_mode'] = self.signal_mode
        signal.at[index, 'multi_period_count'] = len(self.period_specs)
        signal.at[index, 'multi_required_count'] = self.agreement_required
        if self.entry_bar_index is not None:
            signal.at[index, 'entry_sigma'] = self.entry_sigma
            signal.at[index, 'shock_trigger_price'] = self.entry_trigger_price
            signal.at[index, 'shock_trigger_bar_date'] = self.entry_bar_date
            signal.at[index, 'shock_open_multiplier_runtime'] = (
                self.params['shock_open_multiplier']
            )
            signal.at[index, 'close_speed_sigma_multiplier_runtime'] = (
                self.params['close_speed_sigma_multiplier']
            )
            signal.at[index, 'close_wd_sigma_multiplier_runtime'] = (
                self.params['close_wd_sigma_multiplier']
            )

    def _clear_pending_entry(self):
        self.pending_entry_sigma = np.nan
        self.pending_trigger_price = np.nan
        self.pending_entry_bar_index = None
        self.pending_entry_bar_date = ''

    def _clear_position_state(self):
        self.start_index = 0
        self.low_index = 0
        self.holding_start_index = 0
        self.entry_execution_price = np.nan
        self.entry_sigma = np.nan
        self.entry_trigger_price = np.nan
        self.entry_bar_index = None
        self.entry_bar_date = ''
        self._clear_pending_entry()

    @staticmethod
    def _format_date_value(value) -> str:
        return str(value).removesuffix('.0')

    def _format_period_value_text(
            self,
            rows: list[dict],
            key: str,
            digits: int) -> str:
        texts = []
        for row in rows:
            value = row.get(key, np.nan)
            if pd.isna(value):
                continue
            texts.append(f"{row['label']}={float(value):.{digits}f}")
        return ' | '.join(texts)

    def _resolve_vote_trigger(
            self,
            ready_rows: list[dict]) -> tuple[list[dict], float] | tuple[None, None]:
        required = min(max(1, self.agreement_required), len(self.period_specs))
        if len(ready_rows) < max(required, self.min_ready_count):
            return None, None

        if self.require_base_period:
            base_row = next(
                (row for row in ready_rows if row['label'] == self.base_period_label),
                None,
            )
            if base_row is None:
                return None, None
            if required == 1:
                selected_rows = [base_row]
            else:
                other_rows = [
                    row for row in ready_rows
                    if row['label'] != self.base_period_label
                ]
                if len(other_rows) < required - 1:
                    return None, None
                selected_rows = [base_row] + sorted(
                    other_rows,
                    key=lambda row: row['threshold_abs'],
                )[:required - 1]
        else:
            if len(ready_rows) < required:
                return None, None
            selected_rows = sorted(
                ready_rows,
                key=lambda row: row['threshold_abs'],
            )[:required]

        trigger_abs = max(row['threshold_abs'] for row in selected_rows)
        return selected_rows, float(trigger_abs)

    def _resolve_blend_trigger(
            self,
            ready_rows: list[dict],
            p: dict) -> tuple[list[dict], float, float] | tuple[None, None, None]:
        if len(ready_rows) < max(1, self.min_ready_count):
            return None, None, None
        if self.require_base_period and not any(
                row['label'] == self.base_period_label for row in ready_rows):
            return None, None, None

        weight_sum = 0.0
        inv_sigma_sum = 0.0
        for row in ready_rows:
            sigma = float(row['sigma'])
            if sigma <= 0:
                continue
            weight = float(self.score_weights.get(row['label'], 1.0))
            if weight <= 0:
                continue
            weight_sum += weight
            inv_sigma_sum += weight / sigma

        if weight_sum <= 0 or inv_sigma_sum <= 0:
            return None, None, None

        effective_sigma = weight_sum / inv_sigma_sum
        trigger_abs = effective_sigma * float(p['shock_open_multiplier'])
        return ready_rows, float(trigger_abs), float(effective_sigma)

    def _resolve_entry_thresholds(self, ctx: BarContext) -> dict:
        quote = ctx.quote
        signal = ctx.signal
        index = ctx.index
        p = self.params
        current_open = float(quote.at[index, 'open'])
        current_high = float(quote.at[index, 'high'])
        shock_move_abs = current_high - current_open

        period_rows = []
        for spec in self.period_specs:
            sigma = (
                quote.at[index, spec['entry_sigma_col']]
                if spec['entry_sigma_col'] in quote.columns else np.nan
            )
            sigma = float(sigma) if pd.notna(sigma) else np.nan
            ready = bool(pd.notna(sigma) and sigma > 0)
            threshold_abs = (
                sigma * float(p['shock_open_multiplier'])
                if ready else np.nan
            )
            score = (shock_move_abs / sigma) if ready else np.nan
            passed = bool(
                ready
                and pd.notna(threshold_abs)
                and shock_move_abs >= threshold_abs
            )
            signal.at[index, spec['entry_sigma_col']] = sigma
            signal.at[index, spec['threshold_col']] = threshold_abs
            signal.at[index, spec['score_col']] = score
            signal.at[index, spec['pass_col']] = int(passed)
            period_rows.append({
                'label': spec['label'],
                'sigma': sigma,
                'threshold_abs': threshold_abs,
                'score': score,
                'passed': passed,
                'ready': ready,
            })

        ready_rows = [row for row in period_rows if row['ready']]
        pass_rows = [row for row in period_rows if row['passed']]
        ready_count = len(ready_rows)
        pass_count = len(pass_rows)

        selected_rows = []
        basis_ready = False
        shock_threshold_abs = np.nan
        entry_sigma_source = np.nan

        if self.signal_mode == 'single':
            base_row = next(
                (row for row in ready_rows if row['label'] == self.base_period_label),
                None,
            )
            if base_row is not None and ready_count >= max(1, self.min_ready_count):
                selected_rows = [base_row]
                shock_threshold_abs = float(base_row['threshold_abs'])
                entry_sigma_source = float(base_row['sigma'])
                basis_ready = True
        elif self.signal_mode == 'vote':
            selected_rows, shock_threshold_abs = self._resolve_vote_trigger(ready_rows)
            if selected_rows is None:
                selected_rows = []
            basis_ready = bool(selected_rows)
            if basis_ready and float(p['shock_open_multiplier']) > 0:
                entry_sigma_source = shock_threshold_abs / float(p['shock_open_multiplier'])
        elif self.signal_mode == 'blend':
            selected_rows, shock_threshold_abs, entry_sigma_source = self._resolve_blend_trigger(
                ready_rows,
                p,
            )
            if selected_rows is None:
                selected_rows = []
            basis_ready = bool(selected_rows)
        elif self.signal_mode == 'max_sigma':
            if ready_count >= max(1, self.min_ready_count):
                if self.require_base_period and not any(
                        row['label'] == self.base_period_label for row in ready_rows):
                    basis_ready = False
                else:
                    selected_rows = ready_rows
                    entry_sigma_source = max(float(row['sigma']) for row in ready_rows)
                    shock_threshold_abs = (
                        entry_sigma_source * float(p['shock_open_multiplier'])
                    )
                    basis_ready = True

        if pd.isna(entry_sigma_source) and selected_rows:
            if float(p['shock_open_multiplier']) > 0 and pd.notna(shock_threshold_abs):
                entry_sigma_source = shock_threshold_abs / float(p['shock_open_multiplier'])
            else:
                entry_sigma_source = max(float(row['sigma']) for row in selected_rows)

        shock_trigger_price = (
            current_open + float(shock_threshold_abs)
            if basis_ready and pd.notna(shock_threshold_abs)
            else np.nan
        )
        shock_score_high = (
            shock_move_abs / float(entry_sigma_source)
            if basis_ready and pd.notna(entry_sigma_source) and float(entry_sigma_source) > 0
            else np.nan
        )
        signal_pass = bool(
            basis_ready
            and pd.notna(shock_trigger_price)
            and current_high >= float(shock_trigger_price)
        )
        score_values = [
            float(row['score']) for row in ready_rows
            if pd.notna(row['score'])
        ]

        signal.at[index, 'basis_ready'] = int(basis_ready)
        signal.at[index, 'pos_basis'] = entry_sigma_source
        signal.at[index, 'neg_basis'] = entry_sigma_source
        signal.at[index, 'entry_sigma_source'] = entry_sigma_source
        signal.at[index, 'shock_threshold_abs'] = shock_threshold_abs
        signal.at[index, 'shock_trigger_price'] = shock_trigger_price
        signal.at[index, 'shock_move_abs'] = shock_move_abs
        signal.at[index, 'shock_score_high'] = shock_score_high
        signal.at[index, 'shock_open_multiplier_runtime'] = p['shock_open_multiplier']
        signal.at[index, 'close_speed_sigma_multiplier_runtime'] = (
            p['close_speed_sigma_multiplier']
        )
        signal.at[index, 'close_wd_sigma_multiplier_runtime'] = (
            p['close_wd_sigma_multiplier']
        )
        signal.at[index, 'active_close_threshold'] = np.nan
        signal.at[index, 'active_close_wd_threshold'] = np.nan
        signal.at[index, 'multi_ready_count'] = ready_count
        signal.at[index, 'multi_pass_count'] = pass_count
        signal.at[index, 'multi_period_count'] = len(self.period_specs)
        signal.at[index, 'multi_required_count'] = self.agreement_required
        signal.at[index, 'multi_trigger_sigma'] = entry_sigma_source
        signal.at[index, 'multi_trigger_mode'] = self.signal_mode
        signal.at[index, 'multi_config_key'] = self.config_key
        signal.at[index, 'multi_selected_periods'] = ' | '.join(
            row['label'] for row in selected_rows
        )
        signal.at[index, 'multi_period_scores'] = self._format_period_value_text(
            period_rows,
            'score',
            3,
        )
        signal.at[index, 'multi_period_sigmas'] = self._format_period_value_text(
            period_rows,
            'sigma',
            6,
        )
        signal.at[index, 'multi_score_mean'] = (
            float(np.mean(score_values)) if len(score_values) > 0 else np.nan
        )
        signal.at[index, 'multi_score_min'] = (
            float(np.min(score_values)) if len(score_values) > 0 else np.nan
        )
        signal.at[index, 'multi_score_max'] = (
            float(np.max(score_values)) if len(score_values) > 0 else np.nan
        )

        return {
            'basis_ready': bool(basis_ready),
            'signal_pass': signal_pass,
            'entry_sigma_source': entry_sigma_source,
            'shock_threshold_abs': shock_threshold_abs,
            'shock_trigger_price': shock_trigger_price,
            'shock_move_abs': shock_move_abs,
            'shock_score_high': shock_score_high,
        }

    def on_bar_idle(self, ctx: BarContext) -> OpenResult | None:
        quote = ctx.quote
        signal = ctx.signal
        index = ctx.index
        ii = ctx.integer_index
        thresholds = self._resolve_entry_thresholds(ctx)
        if not thresholds['basis_ready']:
            return None
        if not thresholds['signal_pass']:
            return None
        if pd.isna(thresholds['entry_sigma_source']):
            return None

        current_open = float(quote.at[index, 'open'])
        trigger_price = float(thresholds['shock_trigger_price'])
        if trigger_price <= current_open:
            return None

        self.pending_entry_sigma = float(thresholds['entry_sigma_source'])
        self.pending_trigger_price = trigger_price
        self.pending_entry_bar_index = ii
        self.pending_entry_bar_date = self._format_date_value(quote.iat[ii, 0])
        signal.at[index, 'shock_signal'] = 1
        return OpenResult(
            execution_price=round(trigger_price, self.params['round_precision']),
            low_index=ii,
            low_price=trigger_price,
            start_index=ii,
        )

    def on_position_opened(self, ctx: BarContext, result):
        signal = ctx.signal
        quote = ctx.quote
        index = ctx.index
        ii = ctx.integer_index
        execution_price = float(signal.at[index, 'execution'])
        entry_bar_date = self._format_date_value(quote.iat[ii, 0])

        self.entry_sigma = float(self.pending_entry_sigma)
        self.entry_trigger_price = float(self.pending_trigger_price)
        self.entry_bar_index = ii
        self.entry_bar_date = entry_bar_date
        self.entry_execution_price = execution_price
        self.start_index = ii
        self.low_index = ii
        self.holding_start_index = ii

        signal.at[index, 'entry_sigma'] = self.entry_sigma
        signal.at[index, 'shock_trigger_price'] = self.entry_trigger_price
        signal.at[index, 'shock_trigger_bar_date'] = entry_bar_date
        signal.at[index, 'low_price'] = execution_price
        signal.at[index, 'low_index'] = ii
        signal.at[index, 'low_date'] = entry_bar_date
        self._clear_pending_entry()

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

        if self.entry_bar_index is None or pd.isna(self.entry_sigma):
            return None

        close_threshold = self.entry_sigma * p['close_speed_sigma_multiplier']
        close_withdrawal_threshold = (
            self.entry_sigma * p['close_wd_sigma_multiplier']
        )
        holding_slice = quote.iloc[self.low_index:ii + 1]
        current_high = float(holding_slice['high'].max())
        current_max_increase = max(
            current_high - float(self.entry_execution_price),
            0.0,
        )
        with_high, holding_withdrawal = get_withdrawal(
            holding_slice,
            close_withdrawal_threshold,
            ii,
            switch0=True,
        )
        holding_withdrawal_percent = (
            holding_withdrawal / with_high if with_high != 0 else 0
        )
        period = ii - self.holding_start_index + 1

        signal.at[index, 'entry_sigma'] = self.entry_sigma
        signal.at[index, 'shock_trigger_price'] = self.entry_trigger_price
        signal.at[index, 'shock_trigger_bar_date'] = self.entry_bar_date
        signal.at[index, 'active_close_threshold'] = close_threshold
        signal.at[index, 'active_close_wd_threshold'] = close_withdrawal_threshold
        signal.at[index, 'holding_wd'] = holding_withdrawal
        signal.at[index, 'hld_wd_per'] = round(holding_withdrawal_percent * 100, 4)
        signal.at[index, 'period'] = period
        signal.at[index, 'high_price'] = current_high
        signal.at[index, 'current_max_increase'] = current_max_increase
        signal.at[index, 'close_speed_sigma_multiplier_runtime'] = (
            p['close_speed_sigma_multiplier']
        )
        signal.at[index, 'close_wd_sigma_multiplier_runtime'] = (
            p['close_wd_sigma_multiplier']
        )

        if period >= int(p['close_bar']):
            window_start = ii - int(p['close_bar']) + 1
            if window_start <= self.low_index:
                prior_high = float(self.entry_execution_price)
            else:
                prior_slice = quote.iloc[self.low_index:window_start]
                prior_high = (
                    float(prior_slice['high'].max())
                    if len(prior_slice) > 0 else float(self.entry_execution_price)
                )
            holding_increase = max(current_high - prior_high, 0.0)
            signal.at[index, 'holding_inc'] = holding_increase
            if holding_increase < close_threshold:
                signal.at[index, 'speed_close_signal'] = 1

        if holding_withdrawal > close_withdrawal_threshold:
            signal.at[index, 'holding_wd_signal'] = 1
            execution_price = current_high - close_withdrawal_threshold
            if execution_price > float(quote.loc[index, 'open']):
                execution_price = float(quote.loc[index, 'open'])
            return CloseResult(
                close_type=1,
                execution_price=round(
                    execution_price,
                    self.params['round_precision'],
                ),
                start_index=self.start_index,
                low_index=self.low_index,
                period=period,
            )

        if signal.at[index, 'speed_close_signal'] == 1:
            return CloseResult(
                close_type=2,
                execution_price=round(
                    float(quote.loc[index, 'close']),
                    self.params['round_precision']),
                start_index=self.start_index,
                low_index=self.low_index,
                period=period,
            )

        return None

    def on_position_closed(self, ctx: BarContext, result):
        signal = ctx.signal
        index = ctx.index
        signal.at[index, 'period'] = result.period
        signal.at[index, 'type'] = result.close_type
        self._clear_position_state()

    def on_trade_stats(self, ctx: BarContext,
                        start_index: int, low_index: int):
        quote = ctx.quote
        signal = ctx.signal
        index = ctx.index
        holding_slice = quote.iloc[start_index:ctx.integer_index + 1]
        if len(holding_slice) == 0:
            return

        high_index = int(holding_slice['high'].astype(float).idxmax())
        max_high = float(holding_slice['high'].max())
        entry_price = (
            float(self.entry_execution_price)
            if pd.notna(self.entry_execution_price)
            else float(holding_slice['open'].iloc[0])
        )
        max_wd = get_max_wd(holding_slice)
        max_inc_percent = (
            (max_high - entry_price) / entry_price * 100
            if entry_price != 0 else np.nan
        )

        signal.at[index, 'max_inc'] = round(max_inc_percent, 4)
        signal.at[index, 'max_wd'] = round(max_wd * 100, 4)
        signal.at[index, 'high_index'] = high_index
        signal.at[index, 'high_date'] = self._format_date_value(
            quote.iat[high_index, 0]
        )
        signal.at[index, 'high_price'] = max_high
        signal.at[index, 'low_index'] = low_index
        signal.at[index, 'low_date'] = self._format_date_value(
            quote.iat[low_index, 0]
        )
        signal.at[index, 'low_price'] = entry_price


# ============================================================
# Main Script
# ============================================================

if False and __name__ == '__main__':

    # --- 数据加载 ---
    folder_path = data_folder_path
    file_name = data_file_name

    native_df, ROUND_PRECISION, NATIVE_BAR_SECONDS = load_data(folder_path, file_name)
    run_mode = str(run_mode).strip().lower()
    volatility_method = str(volatility_method).strip().lower()
    if run_mode not in ('manual', 'grid'):
        raise ValueError("run_mode must be 'manual' or 'grid'.")
    get_volatility_method_spec(volatility_method)
    data_selection_mode = str(data_selection_mode).strip().lower()
    if data_selection_mode not in ('index', 'date'):
        raise ValueError("data_selection_mode must be 'index' or 'date'.")
    if close_wd_max_inc_ratio < 0:
        raise ValueError('close_wd_max_inc_ratio must be >= 0.')

    if run_mode == 'manual':
        if min(int(open_bar), int(close_bar)) <= 0:
            raise ValueError('open_bar and close_bar must be positive in manual mode.')
        open_bar_values = [int(open_bar)]
    else:
        if bar_step == 0:
            raise ValueError('bar_step cannot be 0.')
        if open_vol_step == 0:
            raise ValueError('open_vol_step cannot be 0.')
        if open_cont_vol_step <= 0:
            raise ValueError('open_cont_vol_step must be positive.')
        if int(open_cont_vol_max_iterations) <= 0:
            raise ValueError('open_cont_vol_max_iterations must be positive.')
        if open_vol_stop_flat_rounds <= 0:
            raise ValueError('open_vol_stop_flat_rounds must be positive.')
        if open_vol_max_iterations <= 0:
            raise ValueError('open_vol_max_iterations must be positive.')
        open_bar_values = build_int_search_values(
            int(open_bar),
            int(bar_end),
            int(bar_step),
        )
        if len(open_bar_values) == 0:
            raise ValueError('open_bar search range is empty.')

    # 创建输出文件夹
    os.makedirs('./result', exist_ok=True)
    os.makedirs(f'./result/{file_name} long outcome/perf', exist_ok=True)
    os.makedirs(f'./result/{file_name} long outcome/trans', exist_ok=True)

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
        './result/%s long outcome/outcome stats/' % file_name
        + 'long_momentum_ARCH ' + run_name + ' outcome_stats.xlsx'
    )

    only_close_cfg = only_close
    if only_close_cfg:
        underlying.open = underlying.low = underlying.high = underlying.close

    export_stats_enabled = EXPORT_STATS or (run_mode == 'manual')
    export_interactive_html_enabled = (
        EXPORT_INTERACTIVE_HTML or (run_mode == 'manual')
    )

    print(f'[Main] volatility_method: {volatility_method}')
    underlying, volatility_forecast_path = merge_volatility_forecast(
        underlying,
        file_name=file_name,
        period_label=period_label,
        vol_method=volatility_method,
    )
    print('[Main] volatility forecast: ' + volatility_forecast_path.name)

    if run_mode == 'manual' and (resample_rule or '').strip():
        prompt_manual_intrabar_precheck(
            native_preview_df,
            preview_df,
            BAR_SECONDS,
            underlying['vol_forecast_price'].astype(float) * float(open_vol_multiplier),
            underlying['vol_forecast_price'].astype(float) * float(open_continous_vol_multiplier),
            metric_kind='absolute',
        )

    # --- 参数循环 ---
    if run_mode == 'grid':
        print(
            '[Grid] open_bar: '
            + f'{open_bar_values[0]} -> {open_bar_values[-1]} step {bar_step}'
        )
        print(
            '[Grid] open_vol_multiplier: '
            + f'{open_vol_multiplier} step {open_vol_step}'
        )
        print(
            '[Grid] open_continous_vol_multiplier: start from open_vol_multiplier '
            + f'step {open_cont_vol_step} max {open_cont_vol_max_iterations}'
        )
        print(
            '[Grid] stop each vol loop after '
            + f'{open_vol_stop_flat_rounds} unchanged trade-count steps'
        )
    else:
        print(
            '[Manual] open_bar=' + str(open_bar)
            + ' close_bar=' + str(close_bar)
        )
        print(
            '[Manual] open_vol_multiplier=' + str(open_vol_multiplier)
            + ' open_wd_vol_multiplier=' + str(open_wd_vol_multiplier)
            + ' open_continous_vol_multiplier=' + str(open_continous_vol_multiplier)
            + ' close_speed_vol_multiplier=' + str(close_speed_vol_multiplier)
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
        planned_param_tags = build_planned_param_tags_long_vol(
            open_bar_values,
            float(open_vol_multiplier),
            int(open_vol_max_iterations),
            float(open_vol_step),
            int(open_cont_vol_max_iterations),
            float(open_cont_vol_step),
            float(close_wd_max_inc_ratio),
            volatility_method=volatility_method,
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
        last_open_vol_trade_count = None
        unchanged_open_vol_steps = 0

        if run_mode == 'grid':
            open_vol_iterations = range(int(open_vol_max_iterations))
        else:
            open_vol_iterations = [0]

        for open_vol_iter in open_vol_iterations:
            if run_mode == 'grid':
                open_vol_multiplier_runtime = round(
                    open_vol_multiplier + (open_vol_iter * open_vol_step),
                    10,
                )
                open_wd_vol_multiplier_runtime = open_vol_multiplier_runtime
                close_speed_vol_multiplier_runtime = open_vol_multiplier_runtime
                print(
                    f'\n[Grid] open_bar={open_bar_runtime} '
                    + f'open_vol_multiplier={open_vol_multiplier_runtime}'
                )
                open_cont_iterations = range(int(open_cont_vol_max_iterations))
                last_open_cont_trade_count = None
                unchanged_open_cont_steps = 0
                outer_reference_trade_count = None
            else:
                print(
                    f'\n[Manual] open_bar={open_bar_runtime} '
                    + f'close_bar={close_bar_runtime}'
                )
                open_vol_multiplier_runtime = float(open_vol_multiplier)
                open_wd_vol_multiplier_runtime = float(open_wd_vol_multiplier)
                close_speed_vol_multiplier_runtime = float(close_speed_vol_multiplier)
                open_cont_iterations = [0]

            for open_cont_iter in open_cont_iterations:
                if run_mode == 'grid':
                    open_continous_vol_multiplier_runtime = round(
                        open_vol_multiplier_runtime
                        + (open_cont_iter * open_cont_vol_step),
                        10,
                    )
                    print(
                        f'[Grid]   open_continous_vol_multiplier='
                        + f'{open_continous_vol_multiplier_runtime}'
                    )
                else:
                    open_continous_vol_multiplier_runtime = float(
                        open_continous_vol_multiplier
                    )
                open_threshold = open_vol_multiplier_runtime
                open_withdrawal_threshold = open_wd_vol_multiplier_runtime
                open_continous_threshold = open_continous_vol_multiplier_runtime
                close_threshold = close_speed_vol_multiplier_runtime
                commision_percent_cfg = commision_percent
                capital_cfg = capital

                if min(
                    open_threshold,
                    open_withdrawal_threshold,
                    close_threshold,
                    open_continous_threshold,
                ) < 0:
                    print('vol multiplier不可为负数')
                    continue
                if open_continous_threshold < open_threshold:
                    print(
                        'open_continous_vol_multiplier不可小于'
                        + 'open_vol_multiplier'
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
                    volatility_method=volatility_method,
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
                    fig1_path = ('./result/%s long outcome/' % file_name
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
                if 'vol_forecast_price' in underlying.columns:
                    detail_df['vol_forecast_price'] = (
                        underlying['vol_forecast_price'].to_numpy()
                    )
                if 'vol_forecast_return' in underlying.columns:
                    detail_df['vol_forecast_return'] = (
                        underlying['vol_forecast_return'].to_numpy()
                    )
                if 'vol_ready' in underlying.columns:
                    detail_df['vol_ready'] = underlying['vol_ready'].to_numpy()
                detail_df['open_trigger_price'] = (
                    detail_df['low_price'] + detail_df['frozen_open_cont_threshold']
                )
                frozen_basis = detail_df['frozen_open_vol']
                current_vol = (
                    detail_df['vol_forecast_price']
                    if 'vol_forecast_price' in detail_df.columns else np.nan
                )
                detail_df['open_vol_multiplier_runtime'] = np.where(
                    frozen_basis.notna() & (frozen_basis != 0),
                    detail_df['frozen_open_threshold'] / frozen_basis,
                    np.nan,
                )
                detail_df['open_cont_vol_multiplier_runtime'] = np.where(
                    frozen_basis.notna() & (frozen_basis != 0),
                    detail_df['frozen_open_cont_threshold'] / frozen_basis,
                    np.nan,
                )
                detail_df['open_wd_vol_multiplier_runtime'] = np.where(
                    frozen_basis.notna() & (frozen_basis != 0),
                    detail_df['frozen_open_wd_threshold'] / frozen_basis,
                    np.nan,
                )
                if isinstance(current_vol, pd.Series):
                    detail_df['close_speed_vol_multiplier_runtime'] = np.where(
                        current_vol.notna() & (current_vol != 0),
                        detail_df['active_close_threshold'] / current_vol,
                        np.nan,
                    )
                else:
                    detail_df['close_speed_vol_multiplier_runtime'] = np.nan
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
                outcome_stats.at[outcome_index, 'capital'] = perf_outcome[-1:].capital.iloc[0]
                trade_num = len(transactions_df) / 2
                outcome_stats.at[outcome_index, 'trade_num'] = trade_num
                outcome_high, outcome_wd = get_outcome_withdrawal(
                    perf_outcome.capital)
                outcome_stats.at[outcome_index, 'outcome_high'] = outcome_high
                outcome_stats.at[outcome_index, 'biggest_wd'] = outcome_wd
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

                    if unchanged_open_cont_steps >= open_vol_stop_flat_rounds:
                        for remain_open_cont_iter in range(
                                open_cont_iter + 1,
                                int(open_cont_vol_max_iterations)):
                            remain_open_cont = round(
                                open_vol_multiplier_runtime
                                + (remain_open_cont_iter * open_cont_vol_step),
                                10,
                            )
                            completed_param_tags.add(build_long_param_tag(
                                open_bar_runtime,
                                open_vol_multiplier_runtime,
                                remain_open_cont,
                                open_vol_multiplier_runtime,
                                close_bar_runtime,
                                open_vol_multiplier_runtime,
                                close_wd_max_inc_ratio,
                                volatility_method=volatility_method,
                            ))
                        print_search_progress(
                            len(completed_param_tags),
                            total_search_space,
                            progress_marks,
                            printed_progress_marks,
                        )
                        print(
                            '[Grid] stop open_cont vol loop at '
                            + f'open_bar={open_bar_runtime} '
                            + f'open_vol_multiplier={open_vol_multiplier_runtime}: '
                            + 'total trade count unchanged for '
                            + f'{open_vol_stop_flat_rounds} steps.'
                        )
                        break
            else:
                if run_mode == 'grid':
                    print(
                        '[Grid] reached open_cont_vol_max_iterations='
                        + str(open_cont_vol_max_iterations)
                        + f' at open_bar={open_bar_runtime} '
                        + f'open_vol_multiplier={open_vol_multiplier_runtime}.'
                    )

            if run_mode == 'grid':
                if outer_reference_trade_count is None:
                    continue
                if (
                    last_open_vol_trade_count is None
                    or outer_reference_trade_count != last_open_vol_trade_count
                ):
                    unchanged_open_vol_steps = 0
                else:
                    unchanged_open_vol_steps += 1
                last_open_vol_trade_count = outer_reference_trade_count

                if unchanged_open_vol_steps >= open_vol_stop_flat_rounds:
                    for remain_vol_iter in range(
                            open_vol_iter + 1,
                            int(open_vol_max_iterations)):
                        remain_open_vol = round(
                            open_vol_multiplier
                            + (remain_vol_iter * open_vol_step),
                            10,
                        )
                        if remain_open_vol < 0:
                            continue
                        for remain_open_cont_iter in range(
                                int(open_cont_vol_max_iterations)):
                            remain_open_cont = round(
                                remain_open_vol
                                + (remain_open_cont_iter * open_cont_vol_step),
                                10,
                            )
                            if remain_open_cont < remain_open_vol:
                                continue
                            completed_param_tags.add(build_long_param_tag(
                                open_bar_runtime,
                                remain_open_vol,
                                remain_open_cont,
                                remain_open_vol,
                                close_bar_runtime,
                                remain_open_vol,
                                close_wd_max_inc_ratio,
                                volatility_method=volatility_method,
                            ))
                    print_search_progress(
                        len(completed_param_tags),
                        total_search_space,
                        progress_marks,
                        printed_progress_marks,
                    )
                    print(
                        '[Grid] stop open_vol loop at '
                        + f'open_bar={open_bar_runtime}: '
                        + 'base total trade count unchanged for '
                        + f'{open_vol_stop_flat_rounds} steps.'
                    )
                    break
        else:
            if run_mode == 'grid':
                print(
                    '[Grid] reached open_vol_max_iterations='
                    + str(open_vol_max_iterations)
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
        os.makedirs('./result/stats %s long outcome/' % file_name, exist_ok=True)
        stats_plot_ext = 'pdf' if SAVE_PLOT_AS_PDF else 'png'
        plt.savefig('./result/stats %s long outcome/' % file_name
                    + ' ' + run_name + ' '
                    + str(len(outcome_stats)) + ' '
                    + f'all outcome.{stats_plot_ext}', dpi=1000)
        outcome_stats.to_excel('./result/stats %s long outcome/' % file_name
                               + ' ' + run_name + ' '
                               + str(len(outcome_stats)) + ' '
                               + 'all outcome.xlsx')
    else:
        disk_path = './result/'
        open_excel = False
        if open_excel:
            os.startfile(
                disk_path + '%s long outcome/perf/' % file_name + perf_name)

    export_outcome_stats = outcome_stats.sort_index()
    export_outcome_stats.index.name = 'param_tag'
    os.makedirs('./result/%s long outcome/outcome stats/' % file_name,
                exist_ok=True)
    export_outcome_stats.to_excel(dashboard_outcome_stats_path)

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
                    vol_now = format_hover_value(
                        pref_data.get('vol_forecast_price', np.nan), 6)
                    frozen_vol = format_hover_value(
                        pref_data.get('frozen_open_vol', np.nan), 6)
                    ov_mult = format_hover_value(
                        pref_data.get('open_vol_multiplier_runtime', np.nan), 4)
                    ocv_mult = format_hover_value(
                        pref_data.get('open_cont_vol_multiplier_runtime', np.nan), 4)
                    owv_mult = format_hover_value(
                        pref_data.get('open_wd_vol_multiplier_runtime', np.nan), 4)
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
                            + 'vol: ' + vol_now + '\n'
                            + 'frozen_vol: ' + frozen_vol + '\n'
                            + 'ov_mult: ' + ov_mult + '\n'
                            + 'ocv_mult: ' + ocv_mult + '\n'
                            + 'owv_mult: ' + owv_mult + '\n'
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
                    vol_now = format_hover_value(
                        pref_data.get('vol_forecast_price', np.nan), 6)
                    close_mult = format_hover_value(
                        pref_data.get('close_speed_vol_multiplier_runtime', np.nan), 4)
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
                            + 'vol: ' + vol_now + '\n'
                            + 'cv_mult: ' + close_mult + '\n'
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
                    vol_now = format_hover_value(
                        pref_data.get('vol_forecast_price', np.nan), 6)
                    close_mult = format_hover_value(
                        pref_data.get('close_speed_vol_multiplier_runtime', np.nan), 4)
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
                            + 'vol: ' + vol_now + '\n'
                            + 'cv_mult: ' + close_mult + '\n'
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
        ax3_vol = ax3.twinx()
        ax3.set_zorder(2)
        ax3_vol.set_zorder(1)
        ax3_vol.patch.set_visible(False)
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
        attach_trade_hover(fig3, ax3, extra_hover_axes=[ax3_vol])
        ax3_vol.plot(
            underlying1.index,
            underlying1['vol_forecast_price'],
            color='#6b7280',
            linewidth=1.15,
            alpha=0.95,
        )
        ax3.set_xlim(-0.7, len(underlying_ratio) - 0.3)
        ax3.xaxis.set_major_locator(plt.MaxNLocator(12))
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3_vol.spines['top'].set_visible(False)
        ax3_vol.spines['left'].set_visible(False)
        ax3_vol.grid(False)
        ax3_vol.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        ax3_vol.set_ylabel('Vol Forecast', color='#6b7280')
        ax3_vol.tick_params(axis='y', colors='#6b7280', pad=6)
        ax3.set_title(
            'Volatility view ' + str(round(Capital_outcome, 2))
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


def run_shock_backtest():
    folder_path = data_folder_path
    file_name = data_file_name

    direct_period_loaded = try_load_direct_period_source(
        folder_path,
        file_name,
        resample_rule,
    )
    if direct_period_loaded is None:
        native_df, ROUND_PRECISION, NATIVE_BAR_SECONDS = load_data(folder_path, file_name)
        source_data_path = Path(folder_path) / f'{file_name}.csv'
        used_direct_period_source = False
    else:
        native_df, ROUND_PRECISION, NATIVE_BAR_SECONDS, source_data_path = direct_period_loaded
        used_direct_period_source = True
    run_mode_value = str(run_mode).strip().lower()
    volatility_method_value = str(volatility_method).strip().lower()
    selection_mode = str(data_selection_mode).strip().lower()

    if run_mode_value not in ('manual', 'grid'):
        raise ValueError("run_mode must be 'manual' or 'grid'.")
    if selection_mode not in ('index', 'date'):
        raise ValueError("data_selection_mode must be 'index' or 'date'.")
    get_volatility_method_spec(volatility_method_value)
    multi_config = normalize_multi_shock_config(
        MULTI_SHOCK_CONFIG,
        resample_rule or '5min',
    )

    if run_mode_value == 'manual':
        shock_open_values = [float(shock_open_multiplier)]
        close_bar_values_runtime = [int(close_bar)]
        close_speed_values = [float(close_speed_sigma_multiplier)]
        close_wd_values = [float(close_wd_sigma_multiplier)]
    else:
        shock_open_values = list(dict.fromkeys(
            float(value) for value in shock_open_multiplier_values
        ))
        close_bar_values_runtime = list(dict.fromkeys(
            int(value) for value in close_bar_values
        ))
        close_speed_values = list(dict.fromkeys(
            float(value) for value in close_speed_sigma_multiplier_values
        ))
        close_wd_values = list(dict.fromkeys(
            float(value) for value in close_wd_sigma_multiplier_values
        ))

    if len(close_bar_values_runtime) == 0:
        raise ValueError('close_bar search range is empty.')
    if min(close_bar_values_runtime) <= 0:
        raise ValueError('close_bar must be positive.')
    if min(shock_open_values) < 0:
        raise ValueError('shock_open_multiplier must be >= 0.')
    if min(close_speed_values) < 0:
        raise ValueError('close_speed_sigma_multiplier must be >= 0.')
    if min(close_wd_values) < 0:
        raise ValueError('close_wd_sigma_multiplier must be >= 0.')

    result_root = build_result_root(file_name)
    stats_root = build_stats_result_root(file_name)
    os.makedirs(result_root, exist_ok=True)
    os.makedirs(result_root / 'perf', exist_ok=True)
    os.makedirs(result_root / 'trans', exist_ok=True)
    os.makedirs(result_root / 'html', exist_ok=True)
    os.makedirs(result_root / 'outcome stats', exist_ok=True)
    os.makedirs(stats_root, exist_ok=True)

    if selection_mode == 'index':
        start_label = start_index
        end_label = end_index
        if str(end_index).strip().lower() == 'latest':
            native_preview_df = native_df.iloc[int(start_index):].copy()
        else:
            native_preview_df = native_df.iloc[int(start_index):int(end_index)].copy()
    else:
        native_dates = pd.to_datetime(native_df['Date'], errors='coerce')
        if native_dates.isna().all():
            raise ValueError('Date column cannot be parsed for date selection.')
        start_label = str(start_date).strip()
        end_label = end_date
        start_ts = parse_selection_datetime(start_label, is_end=False)
        if str(end_label).strip().lower() == 'latest':
            date_mask = native_dates >= start_ts
        else:
            end_ts = parse_selection_datetime(str(end_label).strip(), is_end=True)
            if end_ts < start_ts:
                raise ValueError('end_date must be >= start_date.')
            date_mask = native_dates.between(end_ts, end_ts, inclusive='both')
            date_mask = native_dates.between(start_ts, end_ts, inclusive='both')
        native_preview_df = native_df.loc[date_mask].copy()

    if len(native_preview_df) == 0:
        raise ValueError(
            'No data in selected range: '
            + str(start_label)
            + ' -> '
            + str(end_label)
        )

    if used_direct_period_source:
        preview_df = native_preview_df.reset_index(drop=True).copy()
        BAR_SECONDS = NATIVE_BAR_SECONDS
    elif (resample_rule or '').strip():
        preview_df, BAR_SECONDS = resample_ohlc_df(native_preview_df, resample_rule)
        if (
            len(preview_df) > 0
            and should_drop_incomplete_initial_resampled_bar(
                native_preview_df,
                resample_rule,
            )
        ):
            preview_df = preview_df.iloc[1:].reset_index(drop=True)
    else:
        preview_df = native_preview_df.reset_index(drop=True).copy()
        BAR_SECONDS = NATIVE_BAR_SECONDS

    if len(preview_df) == 0:
        raise ValueError('No data remains after resampling the selected native range.')

    period_label = normalize_period_label(
        format_period_label(resample_rule, BAR_SECONDS)
    )
    range_token = (
        make_safe_range_token(start_label)
        + '-'
        + make_safe_range_token(end_label)
    )
    batch_name = (
        f"shock_multi {multi_config['config_key']} "
        + f"period_{period_label} {range_token}"
    )
    if str(RUN_LABEL_SUFFIX).strip():
        batch_name += ' ' + make_safe_range_token(RUN_LABEL_SUFFIX)
    dashboard_outcome_stats_path = (
        result_root
        / 'outcome stats'
        / f'long_momentum_ARCH_shock_multi {batch_name} outcome_stats.xlsx'
    )

    underlying = preview_df.reset_index(drop=True).copy()
    if only_close:
        for col in ['open', 'high', 'low', 'close']:
            underlying[col] = underlying['close']

    export_stats_enabled = EXPORT_STATS or (run_mode_value == 'manual')
    export_interactive_html_enabled = (
        EXPORT_INTERACTIVE_HTML or (run_mode_value == 'manual')
    )

    underlying, volatility_forecast_paths, multi_config = build_shock_signal_for_period(
        quote=underlying,
        file_name=file_name,
        period_label=period_label,
        vol_method=volatility_method_value,
        multi_config=multi_config,
    )

    outcome_stats = pd.DataFrame()
    executed_run_count = 0
    if run_mode_value == 'grid':
        outcome_stats = load_existing_outcome_stats(str(dashboard_outcome_stats_path))
        existing_param_tags = set(outcome_stats.index.astype(str).tolist())
        planned_param_tags = build_planned_param_tags_long_shock(
            shock_open_values,
            close_bar_values_runtime,
            close_speed_values,
            close_wd_values,
            volatility_method=volatility_method_value,
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
        total_search_space = 1
        progress_marks = {}
        printed_progress_marks = set()

    last_run = None
    print('[Main] shock test range: ' + str(preview_df.iloc[0]['Date']) + ' -> ' + str(preview_df.iloc[-1]['Date']))
    print('[Main] source file: ' + str(source_data_path))
    print('[Main] forecast files: ' + str(volatility_forecast_paths))

    for shock_open_runtime in shock_open_values:
        for close_bar_runtime in close_bar_values_runtime:
            for close_speed_runtime in close_speed_values:
                for close_wd_runtime in close_wd_values:
                    param_tag = build_shock_param_tag(
                        shock_open_runtime,
                        close_bar_runtime,
                        close_speed_runtime,
                        close_wd_runtime,
                        volatility_method=volatility_method_value,
                    )
                    if run_mode_value == 'grid' and param_tag in existing_param_tags:
                        continue

                    params = {
                        'shock_open_multiplier': float(shock_open_runtime),
                        'close_bar': int(close_bar_runtime),
                        'close_speed_sigma_multiplier': float(close_speed_runtime),
                        'close_wd_sigma_multiplier': float(close_wd_runtime),
                        'round_precision': ROUND_PRECISION,
                        'multi_config': multi_config,
                        'base_period_label': period_label,
                    }

                    strategy = MomentumStrategy(params)
                    engine = BacktestEngine(
                        underlying,
                        strategy,
                        capital,
                        ROUND_PRECISION,
                        commision_percent,
                        show_progress=(run_mode_value != 'grid'),
                    )
                    df_signal, signal, close_counts = engine.run()
                    performance, transactions_df = generate_performance(
                        underlying,
                        df_signal,
                        capital,
                        commision_percent,
                    )

                    perf_outcome = performance.reset_index(drop=True)[['date', 'capital']]
                    capital_outcome = (
                        float(perf_outcome['capital'].iloc[-1])
                        if len(perf_outcome) > 0 else float(capital)
                    )
                    withdrawal_close_count = int(close_counts.get(1, 0))
                    speed_close_count = int(close_counts.get(2, 0))
                    trade_num = (
                        int((transactions_df['Type'] == 'sell').sum())
                        if 'Type' in transactions_df.columns else 0
                    )
                    outcome_high, outcome_wd = get_outcome_withdrawal(
                        perf_outcome['capital']
                    )
                    count_tag = f'{withdrawal_close_count}+{speed_close_count}'

                    detail_df = pd.concat(
                        [
                            signal.reset_index(drop=True),
                            preview_df.reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    detail_df = pd.concat(
                        [
                            detail_df,
                            perf_outcome['capital'].reset_index(drop=True).rename('capital'),
                        ],
                        axis=1,
                    )
                    for col in [
                            'vol_forecast_price',
                            'vol_forecast_return',
                            'vol_ready',
                            'entry_sigma_source',
                            'entry_sigma_ready',
                    ]:
                        if col in underlying.columns:
                            detail_df[col] = underlying[col].to_numpy()
                    for period in multi_config['periods']:
                        token = period_label_to_token(period)
                        for col in [
                                f'vol_forecast_price_{token}',
                                f'vol_forecast_return_{token}',
                                f'entry_sigma_price_{token}',
                                f'entry_sigma_return_{token}',
                        ]:
                            if col in underlying.columns:
                                detail_df[col] = underlying[col].to_numpy()

                    file_stem = f'{param_tag} {count_tag} {batch_name}'
                    if export_stats_enabled:
                        detail_df.reset_index(drop=False).to_excel(
                            result_root / 'perf' / f'{file_stem} perf.xlsx',
                            sheet_name='stats',
                            index=False,
                        )
                        transactions_df.reset_index(drop=False).to_excel(
                            result_root / 'trans' / f'{file_stem} trans.xlsx',
                            sheet_name='stats',
                            index=False,
                        )

                    if export_interactive_html_enabled:
                        factor = float(underlying['open'].iloc[0])
                        if factor == 0:
                            factor = 1.0
                        export_interactive_html_long(
                            file_name=file_name,
                            save_name=file_stem,
                            title=str(round(capital_outcome, 2)) + ' ' + file_stem,
                            underlying1=underlying.reset_index(drop=True),
                            detail_df=detail_df,
                            transactions_df=transactions_df,
                            factor=factor,
                        )

                    outcome_stats.at[param_tag, 'capital'] = capital_outcome
                    outcome_stats.at[param_tag, 'trade_num'] = trade_num
                    outcome_stats.at[param_tag, 'outcome_high'] = outcome_high
                    outcome_stats.at[param_tag, 'biggest_wd'] = outcome_wd
                    outcome_stats.at[param_tag, 'open_bar'] = int(close_bar_runtime)
                    outcome_stats.at[param_tag, 'open_threshold'] = float(shock_open_runtime)
                    outcome_stats.at[param_tag, 'open_continous_threshold'] = float(close_speed_runtime)
                    outcome_stats.at[param_tag, 'open_withdrawal_threshold'] = float(close_wd_runtime)
                    outcome_stats.at[param_tag, 'close_withdrawal_threshold'] = float(close_wd_runtime)
                    outcome_stats.at[param_tag, 'withdrawal_limit'] = float(close_wd_runtime)
                    outcome_stats.at[param_tag, 'close_bar'] = int(close_bar_runtime)
                    outcome_stats.at[param_tag, 'shock_open_multiplier'] = float(shock_open_runtime)
                    outcome_stats.at[param_tag, 'close_speed_sigma_multiplier'] = float(close_speed_runtime)
                    outcome_stats.at[param_tag, 'close_wd_sigma_multiplier'] = float(close_wd_runtime)
                    outcome_stats.at[param_tag, 'volatility_method'] = volatility_method_value
                    outcome_stats.at[param_tag, 'period_label'] = period_label
                    outcome_stats.at[param_tag, 'forecast_file'] = json.dumps(
                        volatility_forecast_paths,
                        ensure_ascii=False,
                    )
                    outcome_stats.at[param_tag, 'range_start'] = str(start_label)
                    outcome_stats.at[param_tag, 'range_end'] = str(end_label)
                    outcome_stats.at[param_tag, 'speed_close_count'] = speed_close_count
                    outcome_stats.at[param_tag, 'withdrawal_close_count'] = withdrawal_close_count
                    outcome_stats.at[param_tag, 'multi_config_key'] = multi_config['config_key']
                    outcome_stats.at[param_tag, 'multi_signal_mode'] = multi_config['signal_mode']
                    outcome_stats.at[param_tag, 'multi_periods'] = ', '.join(multi_config['periods'])
                    outcome_stats.at[param_tag, 'multi_agreement_required'] = int(
                        multi_config['agreement_required']
                    )
                    outcome_stats.at[param_tag, 'multi_min_ready_count'] = int(
                        multi_config['min_ready_count']
                    )
                    outcome_stats.at[param_tag, 'multi_require_base_period'] = int(
                        bool(multi_config['require_base_period'])
                    )

                    existing_param_tags.add(param_tag)
                    if run_mode_value == 'grid':
                        completed_param_tags.add(param_tag)
                        print_search_progress(
                            len(completed_param_tags),
                            total_search_space,
                            progress_marks,
                            printed_progress_marks,
                        )

                    last_run = {
                        'param_tag': param_tag,
                        'count_tag': count_tag,
                        'capital_outcome': capital_outcome,
                        'close_counts': close_counts.copy(),
                    }
                    executed_run_count += 1

    if len(outcome_stats) == 0:
        raise ValueError('No parameter combination is available in outcome_stats.')

    export_outcome_stats = outcome_stats.sort_index()
    export_outcome_stats.index.name = 'param_tag'
    export_outcome_stats.to_excel(dashboard_outcome_stats_path)

    if run_mode_value == 'grid':
        if executed_run_count == 0:
            print('[Grid] no new parameter executed in this run.')
        if len(outcome_stats) > 1:
            outcome_stats.sort_values('capital', ascending=False).to_excel(
                stats_root / f'{batch_name} {len(outcome_stats)} all outcome.xlsx'
            )

    if run_mode_value == 'manual' and last_run is not None:
        print('[Manual] param_tag: ' + str(last_run['param_tag']))
        print('[Manual] close counts: ' + str(last_run['close_counts']))
        print('[Manual] capital: ' + str(round(last_run['capital_outcome'], 4)))

    return {
        'result_root': result_root,
        'stats_root': stats_root,
        'dashboard_outcome_stats_path': dashboard_outcome_stats_path,
        'batch_name': batch_name,
        'period_label': period_label,
        'range_token': range_token,
        'multi_config': multi_config,
        'forecast_paths': volatility_forecast_paths,
        'outcome_stats': export_outcome_stats.copy(),
        'executed_run_count': executed_run_count,
    }


if __name__ == '__main__':
    run_shock_backtest()
