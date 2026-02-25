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

# 回测区间
START_INDEX = 35001
END_INDEX = 40000  # 或 'lastest'
ONLY_CLOSE = False

# 参数循环
FOR_NUM_1 = 1
FOR_NUM_2 = 1
FOR_NUM_3 = 1
STEP1 = 0.001
STEP3 = 0.01

# 策略参数（分钟输入，自动换算 bars）
OPEN_BAR_MINUTES = 10.0
OPEN_THRESHOLD = 0.0001
CLOSE_BAR_MINUTES = 10.0
CLOSE_THRESHOLD = 0.001
OPEN_CONTINOUS_THRESHOLD = 0.0013

# 双策略参数（保留）
OPEN_BAR2_MINUTES = np.nan  # np.nan 表示不启用
OPEN_THRESHOLD2 = np.nan
OPEN_CONTINOUS_THRESHOLD2 = 0.003

COMMISION_PERCENT = 0.000
CAPITAL = 100.0
# 仅用于柱图可视化：把 segment_withdrawal=0 的柱子显示为最小高度（不改原始数据）
ZERO_BAR_VISUAL_FLOOR_PCT = 0.0001


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
            open_increase = get_increase(open_slice)
            inc_base = open_slice['low'].iloc[0]
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
        max_inc = get_increase(max_slice)
        inc_base = max_slice['low'].iloc[0]
        max_inc_percent = max_inc / inc_base
        signal.at[index, 'max_inc'] = max_inc_percent * 100
        signal.at[index, 'max_wd'] = max_wd * 100
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
        max_inc = get_increase(max_slice)
        inc_base = max_slice['low'].iloc[0]
        max_inc_percent = max_inc / inc_base
        signal.at[index, 'max_inc'] = max_inc_percent * 100
        signal.at[index, 'max_wd'] = max_wd * 100
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
    open_bar_cfg = minutes_to_bars(OPEN_BAR_MINUTES, BAR_SECONDS, 'open_bar')
    close_bar_cfg = minutes_to_bars(CLOSE_BAR_MINUTES, BAR_SECONDS, 'close_bar')
    open_bar2_cfg = minutes_to_bars_optional(
        OPEN_BAR2_MINUTES, BAR_SECONDS, 'open_bar2'
    )

    # 创建输出文件夹
    os.makedirs(f'./{file_name} long no wd outcome/perf', exist_ok=True)
    os.makedirs(f'./{file_name} long no wd outcome/trans', exist_ok=True)
    os.makedirs(f'./{file_name} long no wd outcome/trade_stats', exist_ok=True)

    outcome_stats = pd.DataFrame()

    # 选择回测时间区间
    startdate = START_INDEX
    enddate = END_INDEX

    preview_df = df[df.index > startdate]
    if enddate != 'lastest':
        preview_df = preview_df[preview_df.index < enddate]
    if len(preview_df) == 0:
        raise ValueError(
            f'No data in selected range: START_INDEX={startdate}, END_INDEX={enddate}'
        )
    print(f'[Main] backtest index range: ({startdate}, {enddate})')
    print(f'[Main] backtest time range: {preview_df.iloc[0]["Date"]} -> {preview_df.iloc[-1]["Date"]}')

    df5 = df[df.index > startdate]
    if enddate != 'lastest':
        df5 = df5[df5.index < enddate].reset_index(drop=True)
    underlying = df5.copy()

    only_close = ONLY_CLOSE
    if only_close:
        underlying.open = underlying.low = underlying.high = underlying.close

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
            # 双策略
            open_bar2 = open_bar2_cfg
            open_threshold2 = OPEN_THRESHOLD2
            open_continous_threshold2 = OPEN_CONTINOUS_THRESHOLD2
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
            print('om' + str(round(open_bar, 4))
                  + ' o' + str(round(open_threshold, 4))
                  + ' oc' + str(round(open_continous_threshold, 4))
                  + ' cm' + str(round(close_bar, 4))
                  + ' c' + str(round(close_threshold, 4))
                  + ' ' + str(round(withdrawal_close_count, 4))
                  + '+' + str(round(speed_close_count, 4)))
            print('profit: ' + str(round(performance.capital.iloc[-1], 2)))
            print(f'[Check-1] withdrawal close count (should be 0): {withdrawal_close_count}')
            print(f'[Check-2] open-on-idle count: open={open_count}, idle={idle_count}')
            print(f'[Check-3] trade extreme stats rows: {len(trade_extreme_df)}')
            print(f'[Check-4] entry->max-profit wd rows: {len(entry_to_max_profit_wd_df)}')
            if len(entry_to_max_profit_wd_df) > 0:
                print(entry_to_max_profit_wd_df.head(5))

            # ====== 命名（fig1 已移除） ======
            save_name = (str(startdate) + '-' + str(enddate)
                         + ' om' + str(round(open_bar, 4))
                         + ' o' + str(round(open_threshold, 4))
                         + ' oc' + str(round(open_continous_threshold, 4))
                         + ' cm' + str(round(close_bar, 4))
                         + ' c' + str(round(close_threshold, 4))
                         + ' ' + str(round(withdrawal_close_count, 4))
                         + '+' + str(round(speed_close_count, 4)))

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

            perf_name = ('om' + str(round(open_bar, 4))
                         + ' o' + str(round(open_threshold, 4))
                         + ' oc' + str(round(open_continous_threshold, 4))
                         + ' cm' + str(round(close_bar, 4))
                         + ' c' + str(round(close_threshold, 4))
                         + ' ' + str(round(withdrawal_close_count, 4))
                         + '+' + str(round(speed_close_count, 4))
                         + ' ' + 'Long ' + str(startdate) + '-' + str(enddate)
                         + ' ' + str(Capital_outcome)
                         + ' ' + 'perf.xlsx')
            writer1 = pd.ExcelWriter(
                '%s long no wd outcome/perf/' % file_name + perf_name,
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
                    '%s long no wd outcome/trans/' % file_name
                    + 'om' + str(round(open_bar, 4))
                    + ' o' + str(round(open_threshold, 4))
                    + ' oc' + str(round(open_continous_threshold, 4))
                    + ' cm' + str(round(close_bar, 4))
                    + ' c' + str(round(close_threshold, 4))
                    + ' ' + str(round(withdrawal_close_count, 4))
                    + '+' + str(round(speed_close_count, 4)) + ' '
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

            trade_stats_name = ('om' + str(round(open_bar, 4))
                                + ' o' + str(round(open_threshold, 4))
                                + ' oc' + str(round(open_continous_threshold, 4))
                                + ' cm' + str(round(close_bar, 4))
                                + ' c' + str(round(close_threshold, 4))
                                + ' ' + str(round(withdrawal_close_count, 4))
                                + '+' + str(round(speed_close_count, 4))
                                + ' ' + 'Long ' + str(startdate) + '-' + str(enddate)
                                + ' ' + str(Capital_outcome)
                                + ' ' + 'trade_stats.xlsx')
            writer3 = pd.ExcelWriter(
                '%s long no wd outcome/trade_stats/' % file_name + trade_stats_name,
                engine='xlsxwriter')
            trade_extreme_df.to_excel(writer3, sheet_name='trade_extremes', index=False)
            entry_to_max_profit_wd_df.to_excel(
                writer3, sheet_name='entry_to_max_profit_wd', index=False)
            writer3.close()

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

                    wd_hist_name = ('om' + str(round(open_bar, 4))
                                    + ' o' + str(round(open_threshold, 4))
                                    + ' oc' + str(round(open_continous_threshold, 4))
                                    + ' cm' + str(round(close_bar, 4))
                                    + ' c' + str(round(close_threshold, 4))
                                    + ' ' + str(round(withdrawal_close_count, 4))
                                    + '+' + str(round(speed_close_count, 4))
                                    + ' ' + str(startdate) + '-' + str(enddate)
                                    + ' profit_withdrawal.png')
                    fig_wd.savefig(
                        '%s long no wd outcome/trade_stats/' % file_name + wd_hist_name,
                        dpi=300, bbox_inches='tight')
                    if for_num_2 == 1:
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

                        profit_hist_name = ('om' + str(round(open_bar, 4))
                                            + ' o' + str(round(open_threshold, 4))
                                            + ' oc' + str(round(open_continous_threshold, 4))
                                            + ' cm' + str(round(close_bar, 4))
                                            + ' c' + str(round(close_threshold, 4))
                                            + ' ' + str(round(withdrawal_close_count, 4))
                                            + '+' + str(round(speed_close_count, 4))
                                            + ' ' + str(startdate) + '-' + str(enddate)
                                            + ' profit_sorted_withdrawal.png')
                        fig_profit.savefig(
                            '%s long no wd outcome/trade_stats/' % file_name + profit_hist_name,
                            dpi=300, bbox_inches='tight')
                        if for_num_2 == 1:
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
        os.makedirs('stats %s long no wd outcome/' % file_name, exist_ok=True)
        plt.savefig('stats %s long no wd outcome/' % file_name
                    + ' ' + save_name + ' '
                    + str(for_num_1) + ' '
                    + str(for_num_2) + ' '
                    + 'all outcome.pdf', dpi=1000)
        outcome_stats.to_excel('stats %s long no wd outcome/' % file_name
                               + ' ' + save_name + ' '
                               + str(for_num_1) + ' '
                               + str(for_num_2) + ' '
                               + 'all outcome.xlsx')
    else:
        disk_path = 'C:/Users/lenovo/Desktop/backtest/'
        open_excel = False
        if open_excel:
            os.startfile(
                disk_path + '%s long no wd outcome/perf/' % file_name + perf_name)

    # ====== 交互式图 (fig2) ======
    if for_num_2 == 1:
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
                close_type_1_df['target'], c='green', s=10)
            close_type_2_df = sell_record[sell_record['Close_type'] == 2]
            scatter_b = ax2.scatter(
                close_type_2_df.index,
                close_type_2_df['target'], c='black', s=10)

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
        plt.title('%s' % (' ' + str(round(Capital_outcome, 2))
                          + ' om' + str(round(open_bar, 4))
                          + ' o' + str(round(open_threshold, 4))
                          + ' oc' + str(round(open_continous_threshold, 4))
                          + ' cm' + str(round(close_bar, 4))
                          + ' c' + str(round(close_threshold, 4))
                          + ' ' + str(round(withdrawal_close_count, 4))
                          + '+' + str(round(speed_close_count, 4))
                          + ' ' + str(startdate) + '-' + str(enddate)))

        # fig2 不显示资金曲线
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
                    color='tab:blue', linewidth=2.0, alpha=0.8, zorder=1)
                buy_idx = None
                buy_y = None
        ax2.xaxis.set_major_locator(plt.MaxNLocator(12))
        plt.show()
