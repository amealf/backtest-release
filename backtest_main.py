# -*- coding: utf-8 -*-
"""
Backtest Main - 通用回测框架
============================
包含：引擎、策略基类、数据类、数据加载、性能计算。
所有策略共享此模块。
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
import matplotlib.ticker as ticker
from mplfinance.original_flavor import candlestick2_ohlc
import time, os, sys
import warnings
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Optional
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# ============================================================
# Data Loading & Validation
# ============================================================

def _ohlc_validity_rate(df, o, h, l, c):
    """计算 OHLC 自洽性比率：high>=max(o,c) 且 low<=min(o,c) 且 low<=high 的行占比"""
    valid = (
        (df[h] >= df[o] - 1e-8) &
        (df[h] >= df[c] - 1e-8) &
        (df[l] <= df[o] + 1e-8) &
        (df[l] <= df[c] + 1e-8) &
        (df[l] <= df[h] + 1e-8)
    )
    return valid.mean()


def _detect_ohlc_columns(df, candidate_cols):
    """
    自动检测4列中哪列是 open/high/low/close。
    策略：遍历所有排列，选 OHLC 自洽率最高的。
    """
    from itertools import permutations
    best_rate = -1
    best_mapping = None
    for perm in permutations(candidate_cols):
        o, h, l, c = perm
        rate = _ohlc_validity_rate(df, o, h, l, c)
        if rate > best_rate:
            best_rate = rate
            best_mapping = (o, h, l, c)
    return best_mapping, best_rate


def load_data(folder_path, file_name):
    """
    读取 CSV 数据，自动检测 OHLC 列顺序，执行自洽性校验，检测价格精度。

    Parameters
    ----------
    folder_path : str
        数据文件夹路径，末尾需含分隔符。
    file_name : str
        文件名（不含 .csv 后缀）。

    Returns
    -------
    df : pd.DataFrame
        加载后的数据，列名固定为 Date/open/high/low/close/vol。
    round_precision : int
        价格小数位数。
    bar_seconds : int
        自动识别的周期（秒）。
    """
    path = folder_path + file_name + ".csv"

    # 先用临时列名读入
    temp_names = ['Date', 'c1', 'c2', 'c3', 'c4', 'vol']
    df = pd.read_csv(path, names=temp_names)

    # 数值化价格列
    for c in ['c1', 'c2', 'c3', 'c4', 'vol']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 自动检测 OHLC 列顺序
    price_cols = ['c1', 'c2', 'c3', 'c4']
    mapping, validity = _detect_ohlc_columns(df.dropna(subset=price_cols), price_cols)

    if mapping is None or validity < 0.8:
        raise ValueError(
            f"无法自动识别 OHLC 列顺序（最佳自洽率={validity:.1%}）。\n"
            "请检查 CSV 的列顺序是否为: datetime, [4个价格列], volume"
        )

    o_col, h_col, l_col, c_col = mapping
    detected_order = [price_cols.index(x) + 1 for x in [o_col, h_col, l_col, c_col]]
    standard_order = [1, 2, 3, 4]
    if detected_order != standard_order:
        print(f"[Data] 自动检测到 OHLC 列顺序: 第{detected_order}列 -> open/high/low/close (自洽率={validity:.1%})")
    else:
        print(f"[Data] OHLC 列顺序正常 (自洽率={validity:.1%})")

    # 重命名为标准列名
    df = df.rename(columns={o_col: 'open', h_col: 'high', l_col: 'low', c_col: 'close'})

    # Bar period 检测
    dates = pd.to_datetime(df['Date'], errors='coerce')
    diffs = dates.diff().dropna()
    if len(diffs) > 50:
        diffs = diffs.iloc[:50]
    median_delta = diffs.median()
    if pd.isna(median_delta):
        raise ValueError(
            "无法识别数据周期：Date 列可能为空或格式异常，无法计算相邻时间差。"
        )
    total_seconds = int(median_delta.total_seconds())
    if total_seconds <= 0:
        raise ValueError(
            f"识别到非法周期秒数：{total_seconds}。请检查 Date 列排序与数据质量。"
        )

    if total_seconds < 60:
        bar_period = f'{total_seconds}s'
    elif total_seconds < 3600:
        m = total_seconds // 60
        bar_period = f'{m}min'
    elif total_seconds < 86400:
        h = total_seconds // 3600
        bar_period = f'{h}h'
    else:
        d = total_seconds // 86400
        bar_period = f'{d}d'

    print(f'[Data] {file_name}  |  bar period: {bar_period} ({total_seconds}s)')

    # OHLC 自洽性校验（用已正确映射的列）
    bad = (
        df['Date'].isna() |
        df[['open', 'high', 'low', 'close']].isna().any(axis=1) |
        (df['low'] > df['high']) |
        (df['open'] < df['low']) | (df['open'] > df['high']) |
        (df['close'] < df['low']) | (df['close'] > df['high'])
    )
    bad_cnt = int(bad.sum())
    if bad_cnt > 0:
        ratio = bad_cnt / len(df)
        if ratio > 0.05:
            sample = df.loc[bad, ['Date', 'open', 'high', 'low', 'close', 'vol']].head(8)
            raise ValueError(
                "OHLC 校验失败，已停止程序。\n"
                f"失败行数：{bad_cnt} / {len(df)} ({ratio:.1%})\n"
                "示例（前 8 行）：\n"
                f"{sample.to_string(index=False)}"
            )
        else:
            print(f"[Data] OHLC 校验：{bad_cnt} 行异常 ({ratio:.1%})，已忽略")

    # 精度检测
    price_cols_final = ['low', 'high', 'open', 'close']
    sample_raw = pd.read_csv(path, names=temp_names,
                             nrows=50, dtype=str, keep_default_na=False)
    # 映射回标准列名用于精度检测
    sample_raw = sample_raw.rename(columns={o_col: 'open', h_col: 'high', l_col: 'low', c_col: 'close'})

    def decimal_places_from_str(x: str) -> int:
        s = str(x).strip()
        if s == '':
            return 0
        try:
            d = Decimal(s)
        except InvalidOperation:
            return 0
        return max(0, -d.as_tuple().exponent)

    col_precision = {}
    for c in price_cols_final:
        p = sample_raw[c].map(decimal_places_from_str).max()
        col_precision[c] = int(p) if p == p else 0
    round_precision = max(col_precision.values()) if col_precision else 0

    return df, round_precision, total_seconds


# ============================================================
# Data Classes
# ============================================================

@dataclass
class BarContext:
    """引擎传给策略的当前状态快照"""
    quote: pd.DataFrame          # 完整行情数据
    signal: pd.DataFrame         # signal DataFrame
    index: Any                   # 当前 DataFrame index
    integer_index: int           # 当前整数位置


@dataclass
class OpenResult:
    """策略返回的开仓决策"""
    execution_price: float       # 期望开仓执行价
    low_index: int               # 统计用: 涨幅起算位置
    low_price: float             # first_cond1_price
    start_index: int             # 统计用: high_index 查找起点


@dataclass
class CloseResult:
    """策略返回的平仓决策"""
    close_type: int              # 平仓类型（含义由策略定义）
    execution_price: float       # 期望平仓执行价
    start_index: int             # 统计用
    low_index: int               # 统计用
    period: int                  # 持仓时间


# ============================================================
# Base Strategy
# ============================================================

class BaseStrategy(ABC):
    """策略基类，子类需要实现 on_bar_idle 和 on_bar_holding"""

    def __init__(self, params: dict):
        self.params = params

    def get_extra_columns(self) -> list:
        """策略需要在 signal DataFrame 中额外添加的列"""
        return []

    def get_default_columns(self) -> dict:
        """返回需要初始化为非 NaN 默认值的列。格式: {列名: 默认值}"""
        return {}

    @abstractmethod
    def on_bar_idle(self, ctx: BarContext) -> Optional[OpenResult]:
        """无持仓时每根 K 线调用。返回 OpenResult 表示开仓，返回 None 继续等待。"""
        ...

    @abstractmethod
    def on_bar_holding(self, ctx: BarContext) -> Optional[CloseResult]:
        """持仓时每根 K 线调用。返回 CloseResult 表示平仓，返回 None 继续持仓。"""
        ...

    def on_position_opened(self, ctx: BarContext, result: 'OpenResult'):
        """开仓成功后调用（可选覆盖）。策略在此记录开仓相关字段到 signal。"""
        pass

    def adjust_and_validate_open_execution(self, ctx: BarContext,
                                           result: 'OpenResult',
                                           execution_price: float) -> float:
        """
        开仓执行价校正与校验（可选覆盖）。
        默认不做处理，返回原 execution_price。
        """
        return execution_price

    def on_position_closed(self, ctx: BarContext, result: 'CloseResult'):
        """平仓成功后调用（可选覆盖）。策略在此记录平仓相关字段到 signal。"""
        pass

    def on_trade_stats(self, ctx: BarContext,
                       start_index: int, low_index: int):
        """平仓后的交易统计（可选覆盖）"""
        pass

    def on_bar_record(self, ctx: BarContext):
        """每根 K 线结束时记录策略特有状态到 signal（可选覆盖）"""
        pass


# ============================================================
# Backtest Engine
# ============================================================

class BacktestEngine:
    """
    通用回测引擎。
    负责：信号 DataFrame 管理、状态机、执行价格校验。
    """

    def __init__(self, quote, strategy: BaseStrategy, capital: float,
                 round_precision: int, commision_percent: float):
        self.quote = quote
        self.strategy = strategy
        self.capital = capital
        self.round_precision = round_precision
        self.commision_percent = commision_percent

    def run(self):
        signal = self._init_signal_df()
        have_holdings = False
        integer_index = 0
        close_counts = defaultdict(int)
        total_rows = len(signal)
        progress_marks = {
            max(1, int(np.ceil(total_rows * p / 100.0))): p
            for p in (20, 40, 60, 80, 100)
        }
        printed_marks = set()

        for index, row in signal.iterrows():
            ctx = BarContext(
                quote=self.quote,
                signal=signal,
                index=index,
                integer_index=integer_index,
            )

            if not have_holdings:
                signal.at[index, 'have_holding'] = 0
                result = self.strategy.on_bar_idle(ctx)

                if result is not None:
                    # 执行价校正与校验由策略实现
                    exec_price = self.strategy.adjust_and_validate_open_execution(
                        ctx, result, result.execution_price)

                    # 记录开仓信号
                    signal.at[index, 'execution'] = exec_price
                    signal.at[index, 'holding_signal'] = 1
                    have_holdings = True
                    self.strategy.on_position_opened(ctx, result)

            elif have_holdings:
                signal.at[index, 'have_holding'] = 1
                result = self.strategy.on_bar_holding(ctx)

                if result is not None:
                    # 记录平仓信号
                    signal.at[index, 'holding_signal'] = 0
                    signal.at[index, 'execution'] = result.execution_price

                    close_counts[result.close_type] += 1

                    # 交易统计（策略实现）
                    self.strategy.on_trade_stats(
                        ctx, result.start_index, result.low_index)

                    have_holdings = False
                    self.strategy.on_position_closed(ctx, result)
                else:
                    signal.at[index, 'holding_signal'] = 3

            # 通用: 记录日期和位置
            signal.at[index, 'integer_index'] = integer_index
            todaysdate = self.quote.iat[integer_index, 0]
            signal.at[index, 'date'] = todaysdate

            # 策略特有状态记录
            self.strategy.on_bar_record(ctx)

            integer_index += 1
            if integer_index in progress_marks:
                pct = progress_marks[integer_index]
                if pct not in printed_marks:
                    print(f'[Engine] progress: {pct}% ({integer_index}/{total_rows})')
                    printed_marks.add(pct)

        # 构建 df_signal
        df_signal = pd.DataFrame({
            'date': signal.date,
            'signal': signal.holding_signal,
            'execution': signal.execution,
            'type': signal.type
        })

        return df_signal, signal, dict(close_counts)

    def _init_signal_df(self):
        """初始化 signal DataFrame"""
        signal = pd.DataFrame(index=self.quote.index)
        # 通用列
        common_cols = [
            'date', 'have_holding', 'holding_signal',
            'execution', 'type',
            'integer_index',
        ]
        for col in common_cols:
            signal[col] = np.nan
        signal['holding_signal'] = 0.0

        # 策略额外列
        for col in self.strategy.get_extra_columns():
            signal[col] = np.nan

        # 策略指定的默认值
        for col, default in self.strategy.get_default_columns().items():
            if col in signal.columns:
                signal[col] = default

        return signal


# ============================================================
# Chart: K-line + Buy/Sell + Capital
# ============================================================

def plot_backtest_chart(underlying, transactions_df, perf_outcome,
                        title, save_path, close_fig=True,
                        direction='long'):
    """
    通用回测结果图：K 线 + 买卖标记 + 资金曲线叠加。

    Parameters
    ----------
    underlying : pd.DataFrame
        行情数据，包含 Date/open/high/low/close。
    transactions_df : pd.DataFrame
        交易记录，包含 Date/Type/Price/Capital 列。
    perf_outcome : pd.DataFrame
        包含 date 和 capital 列的绩效序列。
    title : str
        图表标题。
    save_path : str
        PDF 保存路径。
    close_fig : bool
        是否在保存后关闭图表，默认 True。
    direction : str
        'long' 或 'short'，决定开仓/平仓类型名称。

    Returns
    -------
    fig, ax : matplotlib Figure 和 Axes 对象。
    """
    open_type = direction                          # 'long' or 'short'
    close_type = 'sell' if direction == 'long' else 'buy'

    fig = plt.figure(figsize=(19, 9.8))
    rect_line = [0.043, 0.055, 0.943, 0.9]
    ax = fig.add_axes(rect_line)
    ax.xaxis.set_major_locator(ticker.LinearLocator(12))

    # 价格归一化
    underlying1 = underlying.reset_index(drop=True)
    factor = underlying1['open'].iloc[0]
    underlying_ratio = pd.DataFrame()
    underlying_ratio['Date'] = underlying1['Date']
    underlying_ratio[['open', 'high', 'low', 'close']] = underlying1[
        ['open', 'high', 'low', 'close']] / factor * 100

    # 资金锚点
    cap_series = pd.to_numeric(perf_outcome['capital'], errors='coerce')
    tr = transactions_df[
        transactions_df['Type'].isin([open_type, close_type])].copy()
    tr = tr.sort_index()
    tr['Capital'] = pd.to_numeric(tr.get('Capital'), errors='coerce')
    cap_at_bar = cap_series.reindex(tr.index)
    tr['cap_point'] = np.where(
        tr['Type'].eq(close_type) & tr['Capital'].notna(),
        tr['Capital'], cap_at_bar)
    cap0 = float(cap_series.iloc[0]) if len(cap_series) else np.nan
    tr['cap_point'] = tr['cap_point'].ffill().fillna(cap0)
    pos_map = pd.Series(underlying_ratio.index,
                        index=pd.to_datetime(underlying_ratio['Date']))
    tr['pos'] = pd.to_datetime(tr['Date']).map(pos_map)
    tr = tr.dropna(subset=['pos'])
    tr['pos'] = tr['pos'].astype(int)

    # K 线
    candlestick2_ohlc(ax, underlying_ratio.open,
                      underlying_ratio.high,
                      underlying_ratio.low,
                      underlying_ratio.close,
                      width=0.7,
                      colorup='salmon',
                      colordown='#2ca02c')

    # 买卖标记
    open_record = transactions_df[transactions_df.Type == open_type]
    if len(open_record) != 0:
        for idx, row in open_record.iterrows():
            plt.scatter(idx, row['Price'] / factor * 100, c='red', s=10)
    close_record = transactions_df[transactions_df.Type == close_type]
    if len(close_record) != 0:
        for idx, row in close_record.iterrows():
            plt.scatter(idx, row['Price'] / factor * 100, c='green', s=10)

    # 蓝线连接买卖点
    buy_idx = None
    buy_y = None
    trade_seq = transactions_df[
        transactions_df.Type.isin([open_type, close_type])].sort_index()
    for idx, row in trade_seq.iterrows():
        if row['Type'] == open_type:
            buy_idx = idx
            buy_y = row['Price'] / factor * 100
        elif row['Type'] == close_type and buy_idx is not None:
            sell_idx = idx
            sell_y = row['Price'] / factor * 100
            ax.plot([buy_idx, sell_idx], [buy_y, sell_y],
                    color='#1F77B4', linewidth=2.0, alpha=0.8)
            buy_idx = None
            buy_y = None

    # 资金曲线叠加
    ax_outcome = ax.twinx()
    ax_outcome.plot(tr['pos'], tr['cap_point'],
                    linewidth=1.2, color='orange', alpha=0.3)

    cursor = Cursor(ax, useblit=True, color='red', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=0)
    plt.title(title)
    plt.savefig(save_path, dpi=1000)

    if close_fig or len(transactions_df) == 0:
        plt.close()

    return fig, ax


# ============================================================
# Performance Calculator
# ============================================================

def generate_performance(quote, signal, capital, commision_percent,
                         direction='long'):
    """direction: 'long' or 'short'"""
    starting_capital = capital
    signal['capital'] = 0.0
    transactions_df = pd.DataFrame(columns=[
        'Date', 'Type', 'Price',
        'Close_type', 'Capital', 'Percent'])
    state = None
    cost = None
    open_type = direction                          # 'long' or 'short'
    close_type = 'sell' if direction == 'long' else 'buy'
    for index, row in signal.iterrows():
        if row['signal'] == 0.0:
            if state is None:
                signal.at[index, 'capital'] = starting_capital
            elif state == open_type:
                if direction == 'long':
                    percent = row['execution'] / cost
                else:
                    percent = cost / row['execution']
                starting_capital = starting_capital * percent
                signal.at[index, 'capital'] = starting_capital
                state = None
                transactions_df.loc[index] = [
                    row['date'], close_type,
                    row['execution'], row['type'],
                    starting_capital, percent]
        elif row['signal'] == 1.0:
            starting_capital = starting_capital * (1 - commision_percent)
            signal.at[index, 'capital'] = starting_capital
            cost = row['execution']
            state = open_type
            transactions_df.loc[index] = [
                row['date'], state, cost, "", "", ""]
        elif row['signal'] == 3.0:
            if state == open_type:
                if direction == 'long':
                    percent = quote.close[index] / cost
                else:
                    percent = cost / quote.close[index]
                temp = starting_capital * percent
                signal.at[index, 'capital'] = temp
    return signal, transactions_df
