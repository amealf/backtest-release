import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor  #十字光标
import matplotlib.ticker as ticker  # 限制坐标数
from mplfinance.original_flavor import candlestick2_ohlc  # k线图
import time, os, sys
import warnings
# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_rows', None)  # Display all rows in console
pd.set_option('display.max_columns', None)  # Display all columns in console
start_time = time.time() # Record start time.

def get_increase(df):
    # input: df(a slice of minute stock data df)
    # output: the increase price from the lowest point to highest point
    # during the interval of df.
    if df.empty:
        print('received empty dataframe at get_increase function.')
        pass
    # 加入这个判断是因为有可能出现阴线，
    # 但还是将最高价作为high，导致incr_diff计算不符预期
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
    # 此函数用于分析涨幅
    if df.empty:
        print('received empty dataframe at get_analysis_increase function.')
        pass
    low = 0
    high = 0
    low = df.iloc[0]['close']
    high = df[1:].high.max()
    analysis_increase = high - low
    return analysis_increase

def get_withdrawal(df, assumebarwithdrawal=True):
    # 此函数用于计算回撤幅度
    with_high = 0
    with_low = 0
    withdrawal = 0
    for index, row in df.iterrows():
        if with_high == 0:
            with_high = row['close']
            with_low = row['close']
            withdrawal = with_high - with_low
        else:
            if row['high'] > with_high:
                with_high = row['high']
                with_low = row['close']
            elif row['low'] < with_low:
                with_low = row['low']

            withdrawal = with_high - with_low
    return with_high, withdrawal

def get_max_wd(df, assumebarwithdrawal=True):
    # 此算法会将回撤平仓的那一根bar的跌幅全部计算进去
    # 因此得数可能会比回撤平仓的设定值高
    with_high = 0
    with_low = 0
    withdrawal = 0
    max_wd = 0
    for index, row in df.iterrows():
        if with_high == 0:
            with_high = row['high']
            with_low = row['close']
        else:
            if row['high'] > with_high:
                with_high = row['high']
                with_low = row['close']
            elif row['low'] < with_low:
                with_low = row['low']
            withdrawal = round((with_high - with_low) / with_high, 5)  # percent
        if withdrawal > max_wd:
            max_wd = withdrawal
    # print(max_wd)
    return max_wd

## 统计函数
def get_outcome_withdrawal(sers):
    with_high = 0
    with_low = 0
    withdrawal = 0
    for row in sers:
        if with_high == 0:
            with_high = row
            with_low = row
            withdrawal = with_high - with_low
        else:
            if row > with_high:
                with_high = row
                with_low = row
            elif row < with_low:
                with_low = row
            withdrawal = with_high - with_low
    return with_high, withdrawal


def generate_signals(quote, capital, open_minute, open_threshold, open_continous_threshold, 
                     open_withdrawal_threshold, close_minute, close_threshold, 
                     close_withdrawal_threshold, open_continous_threshold2, 
                     close_withdrawal_threshold2, commision_percent):
    # Initialize the dataframe
    # index = self.quote.index即跟underlying一样数量的index
    open_withdrawal_threshold_ori = open_withdrawal_threshold
    signal = pd.DataFrame(index=quote.index)
    signal['date'] = np.nan
    signal['withdrawal'] = np.nan  # withdrawal data
    signal['wd_per'] = np.nan  # withdrawal data
    signal['wd_signal'] = 0.0  # withdrawal signal
    signal['increase'] = np.nan  # increase_data
    signal['inc_per'] = np.nan  # increase_data percent
    signal['inc_signal'] = 0.0
    signal['ana_inc'] = np.nan
    signal['a_inc_per'] = np.nan
    signal['total_inc'] = np.nan
    signal['t_inc_per'] = np.nan
    signal['total_inc_signal'] = 0.0
    signal['max_inc'] = np.nan  # 不开仓的情况下，False部分一次计算的全部涨幅
    signal['max_wd'] = np.nan  # low_index到high_index之间出现过的最大回撤
    
    signal['have_holding'] = np.nan
    signal['holding_wd'] = np.nan
    signal['hld_wd_per'] = np.nan
    signal['holding_wd_signal'] = 0.0
    signal['holding_inc'] = np.nan
    signal['speed_close_signal'] = 0.0
    signal['var0'] = np.nan
    signal['low_index'] = np.nan  # 开始计算涨幅的那根k线的位置
    signal['high_index'] = np.nan  # 最高价出现的那根k线的位置
    signal['low_date'] = np.nan
    signal['low_price'] = np.nan  # 开始计算的价格
    signal['high_date'] = np.nan
    signal['high_price'] = np.nan  # 开始计算的价格
    signal['integer_index'] = np.nan  # 当前k线的位置
    signal['last_index'] = np.nan
    signal['execution'] = np.nan  # 执行交易的价格
    signal['period'] = np.nan  # 用于统计「持仓时间」
    signal['new_opening_count'] = np.nan  # 用于记录「开始」后已经过了多长时间
    signal['holding_signal'] = 0.0  # 标记资金账户持仓状态的信号
    signal['type'] = np.nan  # 回撤平仓时value=1，速度平仓时value=2
    # signal['test1'] = 0.0 # 测试参数
    
    ## Initialize loop parameters
    have_holdings = False  # 初始状态为未持仓状态
    new_opening = False
    # new_opening2 = False
    new_opening_count = 0  # 初始状态为未持仓状态*
    var0 = 0  # 初始状态为未持仓状态
    integer_index = 0  # 从第一根k线开始
    withdrawal_close_count = 0  # 回撤平仓次数，从0开始计算
    speed_close_count = 0  # 速度平仓次数，从0开始计算
    last_index = 0
    recent_low_index = 0
    holding_increase_percent = 0
    low_index = 0
    
    for index, row in signal.iterrows():  # 对每一根k线进行运算
        
        ## 0 if have no holdings, see if can open a position
        ## 开仓模块
        
        if have_holdings == False:
            signal.at[index, 'have_holding'] = 0
            # 1 给last_index 赋值 这一段好像可以放在最后
            if new_opening:
                # -1是为了从「判断未不合条件从而reset的那根k线」开始计算
                last_index = integer_index - 1
                # 避免last_index = 0-1 = -1
                new_opening = False
            # 2 如果new opening count大于open minute，将计算区域设为为 [last index:last index+open minute]。我哦们计算计算涨幅是否
            if new_opening_count >= open_minute:
                last_index = integer_index - open_minute + 1  # +1
            new_opening_count += 1
            analysis_slice = quote[last_index:integer_index + 1]
            if var0 == 0:  # 0是默认值，第一个阶段
                # 3 definitions of increase & withdrawal
                increase = get_increase(analysis_slice)
                # print(increase)
                # print(analysis_slice)
                inc_base = analysis_slice['low'].iloc[0]
                # 以策略开始计算的价位为基准的涨幅
                inc_percent = increase / inc_base
                # print(analysis_slice)
                with_high, withdrawal = get_withdrawal(analysis_slice)
                # 以最高价为基准的回撤幅度
                wd_percent = withdrawal / with_high  
                
                signal.at[index, 'withdrawal'] = withdrawal
                signal.at[index, 'wd_per'] = round(wd_percent * 100, 4)
                signal.at[index, 'increase'] = increase
                signal.at[index, 'inc_per'] = round(inc_percent * 100, 4)
                
                # 4 定义cond1(increaser cond)和cond2(withdrawal cond)
                # 满足条件时在该条件的signal的column上标记为1，不满足时标记为0
                # 条件1：increase 大于等于 open_threshold ，即满足开仓的速度
                cond1 = (inc_percent >= open_threshold)
                if cond1:
                    signal.at[index, 'inc_signal'] = 1
                else:
                    signal.at[index, 'inc_signal'] = 0
                # 条件2: 
                # if withdrawal is less than open_withdrawal_threshold
                
                # 回测的上限是increase的值。
                # 显然，如果跌破了increase的值， 应该重新算low index了

                # cond2 = wd_percent < open_withdrawal_threshold
                # if cond2:
                #     signal.at[index, 'wd_signal'] = 1
                # else:
                #     signal.at[index, 'wd_signal'] = 0
                
                signal.at[index, 'wd_signal'] = 1
                # 5 如果cond2满足（回测没有超过设定值），
                # 那么判断一次cond1是否满足。
                # 两者都满足时，计算low_index位置，然后令var0 = 1，进入「准开仓」状态
                # 之后不再判断cond1。
                if signal.at[index, 'wd_signal']:
                    if signal.at[index, 'inc_signal']:
                        # 满足cond1和cond2时，计算行情开始的位置，即low_index
                        # 然后将v0设为1
                        for i in reversed(
                                range(last_index, integer_index + 1)
                                ):  # 循环会在integer_index处停止
                            low_index_slice = quote[
                                i: integer_index + 1]
                            increase2 = get_increase(low_index_slice)
                            if increase2 == increase:
                                low_index = i
                                break
                        signal.at[index, 'low_index'] = low_index
                        # print(index, low_index)
                        signal.at[index, 'low_date'] = str(
                            signal.at[low_index, 'date'])
                        last_index = low_index
                        # start_index会一直等于last_index不会reset
                        # 用于计算high_index
                        start_index = last_index  
                        # print(start_index)
                        var0 = 1
                        
                # 6 如果cond2不满足，那么reset
                else:
                    if inc_percent > open_continous_threshold:
                        # 实际上应开仓但回测程序上没开
                        print(str(index) + '满足开仓和满足回撤reset同时发生')
                    new_opening = True  # 移动last_index和holding_start_index的位置
                    new_opening_count = 1

            # 6 满足cond1和cond2后, var0 = 1, 对new_opening_count进行一次赋值
            # 使其在当前k线等于integer_index - low_index +1
            if var0 == 1:
                new_opening_count = integer_index - low_index + 1
                signal.at[index, 'low_index'] = low_index
                signal.at[index, 'low_date'] = str(
                    signal.at[low_index, 'date']).removesuffix('.0')
                signal.at[index, 'period'] = new_opening_count
                var0 = 2
                # 通过将var0的值设为2来进入「判断涨幅是否足以开仓」的阶段

            # 7 开仓前的阶段2
            # 判断是否开仓的标准是
            # 「计算总涨幅是否大于open_continous_threshold」
            if var0 == 2:
                cond3_analysis_slice = quote[low_index: integer_index + 1]
                with_high, withdrawal = get_withdrawal(cond3_analysis_slice)
                signal.at[index, 'withdrawal'] = withdrawal
                withdrawal_percent = withdrawal/with_high
                total_increase = get_increase(cond3_analysis_slice)  
                # if open_withdrawal_threshold_ori > total_increase: # 取二者小值
                #     open_withdrawal_threshold = total_increase
                #     print(index, total_increase, open_withdrawal_threshold)
                # else:
                #     open_withdrawal_threshold = open_withdrawal_threshold_ori
                cond3 = withdrawal_percent < open_withdrawal_threshold
                
                # 条件3: 
                # if withdrawal is less than open_withdrawal_threshold
                # 'wd_signal' = 0
                if cond3:
                    signal.at[index, 'wd_signal'] = 1
                else:
                    signal.at[index, 'wd_signal'] = 0
                    
                if signal.at[index, 'wd_signal']:  
                    # 仍要继续判断回撤条件是否满足
                    # 如果不满足x时间内涨y的条件，平仓
                    if new_opening_count >= open_minute:  
                        ana_inc_slice_1 = quote[low_index: integer_index + 1]
                        ana_inc_slice_2 = quote[
                            low_index: integer_index + 1 - open_minute]
                        analysis_increase = ana_inc_slice_1.high.max(
                            ) - ana_inc_slice_2.high.max()  # 这个区间的涨幅
                        inc_base = ana_inc_slice_1['low'].iloc[0]
                        analysis_increase_percent = analysis_increase/inc_base
                        signal.at[index, 'ana_inc'] = analysis_increase
                        signal.at[index, 'a_inc_per'
                                  ] = analysis_increase_percent * 100
                        
                        # 令var0 = 4，进入统计模块后reset
                        # if analysis_increase_percent != 0:  # 可能等于0吗
                        if analysis_increase_percent < close_threshold:
                            var0 = 4
                    # 最低价到最高价的长度
                    
                    # print(cond3_analysis_slice)
                    inc_base = cond3_analysis_slice['low'].iloc[0]
                    total_increase_percent = total_increase/inc_base
                    signal.at[index, 'total_inc'] = total_increase
                    signal.at[index, 't_inc_per'] = round(
                        total_increase_percent * 100, 4)
                    first_cond1_price = cond3_analysis_slice.iloc[0]['low']
                    # 如果涨幅高于限制，就发出开仓信号
                    if total_increase_percent >= open_continous_threshold:
                        signal.at[index, 'total_inc_signal'] = 1
                        # 如果不满足回撤限制，进入统计模块后reset
                else:
                    var0 = 3 # var0 = 3 未开仓时回测reset
                    
                # 9 处于阶段二（满足了速度要求）,由于回撤超过设定值而reset
                # v0=3是在reset前得到统计模块。
                if var0 == 3:
                    increase3_slice = quote[start_index: integer_index + 1]
                    increase3 = get_analysis_increase(increase3_slice)
                    for i in range(start_index + 1, integer_index + 2):  
                        # last_index+1是为了
                        # 在第一次计算时值为last_index所在位置的值
                        high_index_slice = quote[start_index: i]
                        increase4 = get_analysis_increase(high_index_slice)
                        if increase4 == increase3:
                            high_index = i - 1
                            break
                    max_slice = quote[low_index: high_index + 1]
                    max_wd = get_max_wd(max_slice)
                    max_inc = get_increase(max_slice)
                    inc_base = max_slice['low'].iloc[0]
                    max_inc_percent = round(max_inc/inc_base, 5)
                    
                    signal.at[index, 'max_inc'] = float(max_inc_percent) * 100
                    
                    
                    signal.at[index, 'max_wd'] = max_wd * 100
                    signal.at[index, 'high_index'] = high_index
                    signal.at[index, 'high_date'] = str(
                        signal.at[high_index, 'date']).removesuffix('.0')
                    # 此时沿用了之前开仓时的low_index，也许应该重新计算
                    signal.at[index, 'low_index'] = low_index  
                    signal.at[index, 'low_date'] = str(
                        signal.at[low_index, 'date']).removesuffix('.0')
                    signal.at[index, 'period'] = new_opening_count
                    signal.at[index, 'type'] = 'open withdraw'
                    # End
                    # 统计完成后reset变量
                    new_opening = True
                    var0 = 0
                    new_opening_count = 0
                    first_cond1_price = 0
                    analysis_increase = 0
                # 处于阶段二（满足了速度要求）,由于涨速不够而reset
                if var0 == 4:  
                    increase3_slice = quote[start_index: integer_index+1]
                    increase3 = get_analysis_increase(increase3_slice)
                    for i in range(start_index+1, integer_index+2):  # last_index+1是为了在第一次计算时值为last_index所在位置的值
                        high_index_slice = quote[start_index: i]
                        increase4 = get_analysis_increase(high_index_slice)
                        if increase4 == increase3:
                            high_index = i - 1
                            break
                    max_slice = quote[low_index: high_index+1]
                    max_wd = get_max_wd(max_slice)
                    max_inc = get_increase(max_slice)
                    inc_base = max_slice['low'].iloc[0]
                    max_inc_percent = round(max_inc/inc_base, 5)              
                    
                    signal.at[index, 'max_inc'] = float(max_inc_percent) * 100
                    
                    
                    signal.at[index, 'max_wd'] = max_wd * 100
                    signal.at[index, 'high_index'] = high_index
                    signal.at[index, 'high_date'] = str(
                        signal.at[high_index, 'date']).removesuffix('.0')
                    signal.at[index, 'low_index'] = low_index
                    signal.at[index, 'low_date'] = str(
                        signal.at[low_index, 'date']).removesuffix('.0')
                    signal.at[index, 'period'] = new_opening_count
                    signal.at[index, 'type'] = 'open speed'
                    # End
                    # 统计完成后reset模块
                    increase1_slice = quote[last_index: integer_index+1]
                    increase1 = get_increase(increase1_slice)
                    # print(increase1)
                    for i in range(last_index, integer_index+1):  # 循环会在integer_index处停止
                        low_index_slice = quote[i: integer_index+1]
                        increase2 = get_increase(low_index_slice)
                        if increase2 == increase1:
                            recent_low_index = i
                    last_index = recent_low_index
                    # print(last_index)
                    var0 = 0
                    new_opening_count = integer_index - recent_low_index + 1
                    first_cond1_price = 0
                    analysis_increase = 0

                # 10 如果得到的开仓信号：开仓，然后转入持仓模块，不再继续检察是否能开仓
                if signal.at[index, 'total_inc_signal']:
                    open_execution_price = round(
                        (first_cond1_price * (1+open_continous_threshold)), 2)
                    # 跳空处理：跳空时在开盘价位置开仓
                    if open_execution_price < quote.loc[index, 'open']:
                        open_execution_price = quote.loc[index, 'open']
                    
                    if open_execution_price > quote.loc[index, 'high']:
                        print('open execution price > high, plz check.')
                        print(first_cond1_price, open_continous_threshold,
                              open_execution_price)
                        print('error index' , index)
                        # os.exit(0)
                        
                    if open_execution_price < quote.loc[index, 'low']:
                        print('open execution price < low, plz check.')
                        print(first_cond1_price, open_continous_threshold,
                              open_execution_price)
                        print('error index' , index)
                        
                        print("loop index(label) =", index,
                              " integer_index(pos) =", integer_index,
                              " quote.index[pos] =", quote.index[integer_index])
                        print(quote.loc[index].to_dict())
                        print("OHLC by loc(label):", quote.loc[index, ['open','low','high']].to_dict())
                        print("OHLC by iloc(pos):", quote.iloc[integer_index][['open','low','high']].to_dict())

                        
                    # End
                    signal.at[index, 'low_price'] = first_cond1_price
                    signal.at[index, 'execution'] = open_execution_price
                    new_opening_count = integer_index - low_index
                    signal.at[index, 'holding_signal'] = 1 # open signal
                    signal.at[index, 'low_index'] = low_index
                    signal.at[index, 'low_date'] = str(
                        signal.at[low_index, 'date']).removesuffix('.0')
                    var0 = 0
                    have_holdings = True  # 开仓
                    new_opening = True

        ## 持仓模块
        
        # # 强制开仓x分钟以上
        # if integer_index < open_minute:
        #     continue
        
        elif have_holdings == True:
            signal.at[index, 'have_holding'] = 1
            # 1 初始化赋值
            if new_opening:
                last_index = low_index
                increase_start_index = low_index
                holding_start_index = integer_index
                new_opening = False
            if new_opening_count >= close_minute:
                last_index = integer_index - close_minute
            new_opening_count += 1
            # 不包括last_index当根bar的 分析区间(持仓中的一段时间)
            analysis_slice = quote[last_index + 1:integer_index + 1]
            # 包括开仓位当根bar的 持仓区间（从持仓开始到当前的时间）
            holding_slice = quote[
                increase_start_index:integer_index + 1]
            # if new_opening_count >= self.close_minute:
            # holding_increase = self.get_increase(analysis_slice)
            # 不能这么算
            
            # 2 条件1
            if new_opening_count >= close_minute:

                ana_inc_slice_1 = quote[low_index: integer_index + 1]
                # warning 在开头，ana_inc_slice_2会是空值
                # 是否未来函数了？
                ana_inc_slice_2 = quote[
                    low_index: integer_index + 1 - close_minute]
                
                holding_increase = ana_inc_slice_1.high.max(
                    ) - ana_inc_slice_2.high.max()
                holding_base = analysis_slice['low'].iloc[0]
                holding_increase_percent = holding_increase / holding_base
                
                signal.at[index, 'holding_inc'] = holding_increase

                if holding_increase_percent < close_threshold:
                    signal.at[index, 'speed_close_signal'] = 1
            # 3 condition2: withdrawal exceeds close_withdrawal_threshold
            with_high, holding_withdrawal = get_withdrawal(
                holding_slice)
            holding_withdrawal_percent = holding_withdrawal / with_high
            signal.at[index, 'holding_wd'] = holding_withdrawal
            signal.at[index, 'hld_wd_per'] = round(
                holding_withdrawal_percent * 100, 4)
            if (open_continous_threshold2 == 0 
                or 
                (new_opening_count >= close_minute
                 and (holding_increase_percent <
                 open_continous_threshold2))
                ):
                if holding_withdrawal_percent > close_withdrawal_threshold:
                    pass
                    # signal.at[index, 'holding_wd_signal'] = 1
            else:
                if holding_withdrawal_percent > close_withdrawal_threshold:
                    pass
                    # signal.at[index, 'holding_wd_signal'] = 1
                    # print(str(index) + ' secondly close')
            # 4 计算持仓时间period
            period = integer_index - holding_start_index + 1
           
            signal.at[index, 'high_price'] = max(holding_slice['high'])

            # 平仓
            # 回撤平仓
            if signal.at[index, 'holding_wd_signal'] == 1:
                # 执行平仓
                have_holdings = False
                signal.at[index, 'holding_signal'] = 0
                cond2_execution_price = max(
                    holding_slice['high']
                    ) * (1 - holding_slice['close'][-1])
                
                # 跳空处理：
                # 如果执行价大1于此刻的开盘价，实际上是成交不了的
                # 所以应该在此刻开盘价成交
                if cond2_execution_price > quote.loc[index, 'open']:
                    cond2_execution_price = quote.loc[index, 'open']
                    
                signal.at[index, 'execution'] = round(
                    cond2_execution_price, 2)
                signal.at[index, 'period'] = period
                withdrawal_close_count += 1
                new_opening = True
                new_opening_count = 0
                # End
                # 执行统计
                increase3_slice = quote[start_index: integer_index + 1]
                increase3 = get_analysis_increase(increase3_slice)
                for i in range(start_index + 1, integer_index + 2):
                    high_index_slice = quote[start_index: i]
                    increase4 = get_analysis_increase(high_index_slice)
                    if increase4 == increase3:
                        high_index = i - 1
                        # print('high_index', high_index)
                        break
                # if integer_index == 92: # for test
                #     sys.exit(0)
                max_slice = quote[low_index: high_index + 1]
                max_wd = get_max_wd(max_slice)
                max_inc = get_increase(max_slice)
                inc_base = max_slice['low'].iloc[0]
                max_inc_percent = round(max_inc/inc_base, 5)
                
                signal.at[index, 'max_inc'] = float(max_inc_percent) * 100
                
                
                signal.at[index, 'max_wd'] = max_wd * 100
                signal.at[index, 'high_index'] = high_index
                signal.at[index, 'high_date'] = str(signal.at[high_index, 'date']).removesuffix('.0')
                signal.at[index, 'high_price'] = max(holding_slice['high'])
                signal.at[index, 'low_index'] = low_index
                signal.at[index, 'low_date'] = str(signal.at[low_index, 'date']).removesuffix('.0')
                signal.at[index, 'type'] = 1
                # End
            # End

            # 速度平仓
            elif signal.at[index, 'speed_close_signal'] == 1:
                # 执行平仓
                have_holdings = False
                signal.at[index, 'holding_signal'] = 0
                signal.at[index, 'execution'] = quote.loc[index]['close']
                signal.at[index, 'period'] = period
                speed_close_count += 1
                new_opening = True
                new_opening_count = 0
                # End
                # 执行统计
                increase3_slice = quote[start_index: integer_index + 1]
                increase3 = get_analysis_increase(increase3_slice)
                for i in range(start_index + 1, integer_index + 2):
                    # index+1是为了在第一次计算时值为last_index所在位置的值
                    high_index_slice = quote[start_index: i]
                    increase4 = get_analysis_increase(high_index_slice)
                    if increase4 == increase3:
                        high_index = i - 1
                        break
                max_slice = quote[low_index: high_index + 1]
                max_wd = get_max_wd(max_slice)
                max_inc = get_increase(max_slice)
                inc_base = max_slice['low'].iloc[0]
                max_inc_percent = round(max_inc/inc_base, 5)
                signal.at[index, 'max_inc'] = max_inc_percent * 100
                signal.at[index, 'max_wd'] = max_wd * 100
                signal.at[index, 'high_index'] = high_index
                signal.at[index, 'high_date'] = str(signal.at[high_index, 'date']).removesuffix('.0')
                signal.at[index, 'low_index'] = low_index
                signal.at[index, 'low_date'] = str(signal.at[low_index, 'date']).removesuffix('.0')
                signal.at[index, 'type'] = 2
                # End
            # End

            else:
                signal.at[index, 'holding_signal'] = 3 # holding signal
                
        # 6 不管有没有持仓都set_value和values的variable
        signal.at[index, 'integer_index'] = integer_index
        signal.at[index, 'last_index'] = last_index
        signal.at[index, 'new_opening_count'] = new_opening_count
        
        # 7 全部结束后，标注当前日期，进入下一根index的计算。
        # todays_dateperiod = self.quote[integer_index:integer_index + 1]
        todaysdate = quote.iat[integer_index, 0]
        signal.at[index, 'date'] = todaysdate
        
        integer_index += 1
        
        # print(integer_index) # 进度
        
    # 建立一个名叫df_signal的DataFrame，return给self.signal
    df_signal = pd.DataFrame({
        'date': signal.date, 
        'signal': signal.holding_signal, 
        'execution': signal.execution,
         'type': signal.type
         })
    # 46行 self.signal, self.signal_history, self.speed_close_count, self.withdrawal_close_count  = self.generate_signals()
    
    return df_signal, signal, speed_close_count, withdrawal_close_count

def generate_performance(quote, signal, capital, commision_percent):  
    """
    when signal = 0.0, it means no holdings
    when signal = 1.0, it means a longing signal
    when signal = 2.0, it means a shorting signal
    when signal = 3.0, it means a holding signal
    calculate performance based on signals
    exposure is used for calculating performance when 
    we hold a shorting position
    """
    starting_capital = capital
    signal['capital'] = 0.0  # signal就是df_signal
    # print(signal)
    transactions_df = pd.DataFrame(columns=[
        'Date', 'Type', 'Price', 
        'Close_type', 'Capital', 'Percent'])  # 最后会输出为trans
    state = None
    cost = None
    for index, row in signal.iterrows():  # signal就是df_signal
        # print(row)
        # if have no holdings
        if row['signal'] == 0.0:
            # two situations: 
            # 1. when you don't have positions at all.  
            # 2. when you are closing a position (from 3 to 0)
            if state == None:
                # row['capital'] = starting_capital
                signal.at[index,'capital'] = starting_capital
            elif state == 'long':  # 持仓中，检测到平仓信号
                # close a long position
                percent = row['execution'] / cost
                starting_capital = starting_capital * percent  
                # * (1 - commision_percent) 买入不用手续费
                # row['capital'] = starting_capital
                signal.at[index,'capital'] = starting_capital
                # reset type
                state = None
                transactions_df.loc[index] = [
                    row['date'], 'sell', 
                    row['execution'], row['type'], 
                    starting_capital, percent]

        # if you have longing signal
        elif row['signal'] == 1.0:
            # assume that there is no bid-ask spread cost
            # record cost and types of transaction
            starting_capital = starting_capital * (1 - commision_percent)
            signal.at[index,'capital'] = starting_capital
            cost = row['execution']
            # cost = quote.close[index]
            state = 'long'
            
            transactions_df.loc[index] = [row['date'], state, cost, "", "", "" ]
        
        # if have a holding
        elif row['signal'] == 3.0:
            # if it is a long holding, update capital by multiply capital
            # with (1 + percent change) in price
            if state == 'long':
                percent = quote.close[index] / cost
                temp = starting_capital * percent
                signal.at[index,'capital'] = temp
            '''
            # opposite direction (1 - percent change)
            elif state == 'short':
                percent = 1 - ((quote.close[index] - cost) / cost)
                temp = starting_capital * percent
                row['capital'] = temp
            '''
    return signal, transactions_df



def backtest(quote, capital, open_minute, open_threshold, open_continous_threshold, 
             open_withdrawal_threshold, close_minute, close_threshold, 
             close_withdrawal_threshold, open_continous_threshold2, 
             close_withdrawal_threshold2, commision_percent):
    
    # 生成信号
    (df_signal, signal,
     speed_close_count, withdrawal_close_count) = generate_signals(
        quote, capital, open_minute, open_threshold, open_continous_threshold, 
        open_withdrawal_threshold, close_minute, close_threshold, 
        close_withdrawal_threshold, open_continous_threshold2, 
        close_withdrawal_threshold2, commision_percent
    )
    
    # print(signal)
    
    # 计算表现
    performance, transactions_df = generate_performance(
        quote, df_signal, capital, commision_percent)
    
    # 打印结果
    # print('om' + str(round(open_minute, 4))
    #       + ' o' + str(round(open_threshold, 4))
    #       + ' oc' + str(round(open_continous_threshold, 4))
    #       + ' cm' + str(round(close_minute, 4))
    #       + ' c' + str(round(close_threshold, 4))
    #       + ' ow' + str(round(open_withdrawal_threshold, 4))
    #       + ' cw' + str(round(close_withdrawal_threshold, 4))
    #       + ' ' + str(round(withdrawal_close_count, 4))
    #       + '+' + str(round(speed_close_count, 4)))
    # print('profit: ' + str(round(performance.capital.iloc[-1], 2)))
    
    return df_signal, signal, speed_close_count, withdrawal_close_count, performance, transactions_df

# 参数设置&初始化
folder_path = r"F:\My strategy\\"
file_name = "MSTR 2024 1min"
path = folder_path + file_name + ".xlsx"

# 读数据文件
df = pd.read_excel(folder_path + '%s.xlsx' %file_name, 
                   names=['Date', 'open', 'high', 'low', 'close', 'vol'])

# OHLC 自洽性校验：任何一条失败都视为输入不可信，直接停止
bad = (
    df['Date'].isna() |
    df[['open','high','low','close']].isna().any(axis=1) |
    (df['low'] > df['high']) |
    (df['open'] < df['low']) | (df['open'] > df['high']) |
    (df['close'] < df['low']) | (df['close'] > df['high'])
)

bad_cnt = int(bad.sum())
if bad_cnt > 0:
    sample = df.loc[bad, ['Date','open','high','low','close','vol']].head(8)

    raise ValueError(
        "OHLC 校验失败，已停止程序。\n"
        f"失败行数：{bad_cnt} / {len(df)}\n"
        "示例（前 8 行）：\n"
        f"{sample.to_string(index=False)}\n\n"
        "常见原因：\n"
        "1) 读入时 names 的列顺序与文件真实顺序不一致（例如文件是 Date,open,high,low,close,vol）。\n"
        "2) Excel 第一行是表头，但你用 names 覆盖后把表头当数据读进来了。"
    )

# 创建文件夹（如果不存在的话）来保存回测结果的excel文件
os.makedirs(f'./{file_name} long stats/perf', exist_ok=True)
os.makedirs(f'./{file_name} long stats/trans', exist_ok=True)

outcome_stats = pd.DataFrame()  # 用于存放每一次运算最后的结果，进行分析

# 选择回测的时间区间
startdate = 20001  # 起始时间
enddate = 35000    # 结束时间

# 截取指定区间的数据
df5 = df[df.index > startdate]
if enddate != 'lastest':
    df5 = df5[df5.index < enddate].reset_index(drop=True)
underlying = df5.copy()  # 用于存储选定时间区间的数据


# 如果只使用收盘价的情况
only_close = False
if only_close:
    underlying.open = underlying.low = underlying.high = underlying.close

# # 统计函数
# def get_withdrawal(sers):
#     # print(sers)
#     with_high = 0
#     with_low = 0
#     withdrawal = 0
#     for row in sers:
#         if with_high == 0:
#             # print(row)
#             with_high = row['high']
#             with_low = row['low']
#             withdrawal = with_high - with_low
#         else:
#             if row > with_high:
#                 with_high = row['high']
#                 with_low = row['low']
#             elif row < with_low:
#                 with_low = row['low']
#             withdrawal = with_high - with_low
#     return with_high, withdrawal

# 设置回测参数并初始化
for_num_1 = 1  # open_withdrawal_threshold 运行次数
for_num_2 = 1  # open_continous_threshold 运行次数
for_num_3 = 80  # open_withdrawal_threshold运行次数
print(for_num_1, for_num_2, for_num_3)
step1 = 0.001  # open_continous_threshold的步长
step2 = 0.001  # close_withdrawal_threshold的步长
step3 = 2  # 开仓速度的时间的步长

for num in range(for_num_1):
    for i in range(for_num_2):
        for e in range(for_num_3):
            print(f'{str(num)} {str(i)}\n')
    
            # strategy parameters
            open_minute = 8 +  (e * step3)
            open_threshold = 0.000001
            open_withdrawal_threshold = 0.001 # 如果ow大于
            # open_withdrawal_threshold = 0.011
            close_minute = 8 +  (e * step3)
            close_threshold = 0.000001
            open_continous_threshold = 0.000001 + (i * step1)
            close_withdrawal_threshold = 0.001 + (num * step2)
            
            # open_continous_threshold2 = 0.015 + (i*step1)
            
            # om2的原理是：同时运行多种速度的策略。当满足其他策略的速度时，
            # 平仓当前策略并开启新的策略
            open_minute2 = np.nan
            open_threshold2 = np.nan
            
            open_continous_threshold2 = 0.003  # disable when it = 0
            close_withdrawal_threshold2 =  0.003  + (num * step3)
            commision_percent = 0.000
            capital = 100.0  # 初始资本
    
            # if open_threshold < close_withdrawal_threshold:
            #     print('open_threshold不可小于close_withdrawal_threshold')
            #     continue
            # if open_continous_threshold < open_threshold:
            #     print('open_continous_threshold不可小于open_threshold')
            #     continue
            # if open_continous_threshold < close_withdrawal_threshold:
            #     print('open_continous_threshold不可小于close_withdrawal_threshold')
            #     continue
            
            arr = underlying[['low', 'high', 'open', 'close']].to_numpy(dtype=float)
            n = arr.shape[0]
            win = open_minute
            
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
            
            
            
            # 运行backtest函数
            (df_signal, signal, speed_close_count, withdrawal_close_count, performance, transactions_df) = backtest(
                underlying, capital, open_minute, open_threshold, open_continous_threshold, 
                     open_withdrawal_threshold, close_minute, close_threshold, 
                     close_withdrawal_threshold, open_continous_threshold2, 
                     close_withdrawal_threshold2, commision_percent)
            
            
            
            if len(transactions_df)>1:
                Capital_outcome = round(transactions_df[
                    transactions_df.Type != 'long'].Capital.iloc[-1],2)
            else:
                Capital_outcome = 100
            perf_outcome = performance.reset_index(
                drop=True)[['date','capital']]
    
            print(str(startdate) + '-' + str(enddate))
            print('total close count = '
                  + str(withdrawal_close_count
                        + speed_close_count)
                  )
            print('withdrawal close count = '
                  + str(round(withdrawal_close_count, 4))
                  )
            print('speed close count = '
                  + str(round(speed_close_count, 4))
                  )
            print('om' + str(round(open_minute, 4))
                    + ' o' + str(round(open_threshold, 4))
                    + ' oc' + str(round(open_continous_threshold, 4))
                    + ' cm' +str(round(close_minute, 4))
                    + ' c' +str(round(close_threshold, 4))
                    + ' ow' + str(round(open_withdrawal_threshold, 4))
                    + ' cw' + str(round(close_withdrawal_threshold, 4))
                    + ' ' + str(round(withdrawal_close_count, 4))
                    + '+' + str(round(speed_close_count, 4)))
            print('profit: ' + str(round(performance.capital.iloc[-1],2)))
            
            ## Plot
            save_name = (str(startdate) + '-' + str(enddate) 
                    +' om'+ str(round(open_minute, 4))
                    + ' o' + str(round(open_threshold, 4))
                    + ' oc' + str(round(open_continous_threshold, 4))
                    + ' cm' + str(round(close_minute, 4))
                    + ' c' +str(round(close_threshold, 4))
                    + ' ow' + str(round(open_withdrawal_threshold, 4))
                    + ' cw' + str(round(close_withdrawal_threshold, 4))
                    + ' ' + str(round(withdrawal_close_count, 4))
                    + '+' + str(round(speed_close_count, 4))
                    )
            
            fig = plt.figure(figsize=(19, 9.8))
            left = 0.043
            width = 0.943
            bottom = 0.055
            height = 0.9
            rect_line = [left, bottom, width, height] # below parameter
            ax = fig.add_axes(rect_line)
            ax.xaxis.set_major_locator(ticker.LinearLocator(12))  # 限制坐标数
            underlying1 = underlying.reset_index(drop=True)  # reorder
            factor = underlying1['open'][0] # 以第一个open为基准
            underlying_ratio = pd.DataFrame() # 行情的变动比例
            underlying_ratio['Date'] = underlying1['Date']
            underlying_ratio[['open','high','low','close']] = underlying1[
                ['open','high','low','close']] / factor * 100
            x = underlying_ratio['close'] # 行情的变动比例
            
            date_list_0 = underlying1.Date.to_list()
            date_list = [str(i) for i in date_list_0]
            # underlying_ratio.index = date_list
            # xaxis1 = date_list
            xaxis1 = perf_outcome.index
            yaxis1 = perf_outcome.capital
            xaxis2 = x.index
            yaxis2 = x
            # plt.semilogy(xaxis1, yaxis1, linewidth=1.2)  # semilogy 对数坐标
            # plt.semilogy(xaxis2, yaxis2, linewidth=1.2)
            # plt.plot(yaxis1.index, yaxis1, linewidth=1.2)  # 普通坐标
            # --- 用交易点构造“买-卖-买-卖”资金锚点 ---
            cap_series = pd.to_numeric(perf_outcome['capital'], errors='coerce')
            
            tr = transactions_df[transactions_df['Type'].isin(['long', 'sell'])].copy()
            tr = tr.sort_index()
            
            # sell 行里的 Capital 可能是字符串，转成数值；buy 行通常为空
            tr['Capital'] = pd.to_numeric(tr.get('Capital'), errors='coerce')
            
            # buy 点的资金：用该bar的 perf_outcome.capital
            cap_at_bar = cap_series.reindex(tr.index)
            
            # 资金锚点：sell 用交易结果 Capital；buy 用该bar的 capital
            tr['cap_point'] = np.where(
                tr['Type'].eq('sell') & tr['Capital'].notna(),
                tr['Capital'],
                cap_at_bar
            )
            
            # 如果出现缺失（索引对不齐或首笔buy没有capital），用前值/初始值补齐
            cap0 = float(cap_series.iloc[0]) if len(cap_series) else np.nan
            tr['cap_point'] = tr['cap_point'].ffill().fillna(cap0)
            
            pos_map = pd.Series(underlying_ratio.index, index=pd.to_datetime(underlying_ratio['Date']))
            tr['pos'] = pd.to_datetime(tr['Date']).map(pos_map)
            
            # 去掉映射失败的行
            tr = tr.dropna(subset=['pos'])
            tr['pos'] = tr['pos'].astype(int)

            candlestick2_ohlc(ax, underlying_ratio.open,
                              underlying_ratio.high, 
                              underlying_ratio.low,
                              underlying_ratio.close,
                              width=0.7,
                              colorup='salmon',
                              colordown='#2ca02c')  # 绘制K线走势
    
            long_record = transactions_df[
                transactions_df.Type == 'long']
            if len(long_record) != 0:
                for index, row in long_record.iterrows():
                    d = str(row['Date'])
                    scatter_r = plt.scatter(index, x[index], c ='red', s = 10)
            sell_record = transactions_df[
                transactions_df.Type == 'sell']
            if len(sell_record) != 0:
                for index,row in sell_record.iterrows():
                    # print(index,row)
                    d = str(row['Date'])
                    scatter_g = plt.scatter(index, x[index], c = 'green', s = 10)
                    
                    
            # --- 仅连接 买点 -> 卖点 的分段蓝线（复用现有 transactions_df 的 index/x） ---
            buy_idx = None
            buy_y = None
            
            # 按时间顺序遍历 long/sell（用 index 排序即可）
            trade_seq = transactions_df[transactions_df.Type.isin(['long', 'sell'])].sort_index()
            
            for idx, row in trade_seq.iterrows():
                if row['Type'] == 'long':
                    buy_idx = idx
                    buy_y = x[idx]          # 买点对应的行情y（你现在红点用的就是它）
                elif row['Type'] == 'sell' and buy_idx is not None:
                    sell_idx = idx
                    sell_y = x[idx]         # 卖点对应的行情y（你现在绿点用的就是它）
            
                    # 画这一笔交易的线段：买 -> 卖
                    ax.plot([buy_idx, sell_idx], [buy_y, sell_y],
                            color='tab:blue', linewidth=2.0, alpha=0.8)
            
                    # 清空，确保不会连到下一次买点
                    buy_idx = None
                    buy_y = None
            # ax_price = ax.twinx()
            # ax.plot(tr.index, tr['Price']/factor*100, linewidth=1.2)  # 默认蓝色，不指定颜色
            
            ax_outcome =  ax.twinx()
            # 画“买-卖-买-卖”折线（建议用 ax.plot，保证画在同一个 axes 上）
            ax_outcome.plot(tr['pos'], tr['cap_point'], 
                            linewidth=1.2, color='orange', alpha=0.5)
            
            # Plot1 = ax.plot(xaxis1, yaxis1, linewidth=1.2, color = '#1f77b4')
            # Plot2 = ax.plot(xaxis2, yaxis2, linewidth=1.2, color = '#ff7f0e')
            
            cursor = Cursor(
                ax, useblit=True, color='red', linewidth=0.7)  # 十字光标
            # remove spot's frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            degrees = 0  # x轴日期偏斜的角度
            plt.xticks(rotation=degrees)
            
            plt.title('%s'%  (
                        ' ' + str(Capital_outcome)) + ' '
                        + save_name
                      )
            #     plt.show()
            plt.savefig('%s long stats/'%file_name
                        + ' ' + str(Capital_outcome) + ' '
                        + save_name
                        + ' Long.pdf', dpi=1000)
            
            plt.savefig('%s long stats/'%file_name
                        + ' ' + str(Capital_outcome) + ' '
                        + save_name
                        + ' Long.png', dpi=1000)
            if len(transactions_df) == 0: #  如果没有成交就不用show了
                plt.close()
                
            if for_num_2 != 1:
                plt.close()
                
            # Perf
            # 建立一个新的dataframe，用于最后输出为perf.xlsx(统计文档)
            detail_df = pd.concat([
                signal, df5], axis=1, join='inner'
                )  # signal_history在25行
            
            detail_df = pd.concat([
                detail_df, perf_outcome.capital], axis=1, join='inner'
                ) # 最后一行加上回测结果
            
            detail_df.drop(
                ['holding_signal', 'inc_signal', 'wd_signal',
                 'holding_wd_signal', 'total_inc_signal', 
                 'speed_close_signal','have_holding'],
                axis=1, inplace=True) # signal不用时可以drop掉
            # 在使用不开仓的统计时,drop掉开仓才能用上的column
            # 现在使用low_date，所以不需要index了 
            detail_df.drop(['var0','low_index','high_index'], axis=1, inplace=True)
            # 在使用不开仓的统计时,drop掉开仓才能用上的column
            if len(detail_df) == 0:
                detail_df.drop(['holding_wd','holding_inc','execution'],
                               axis =1, inplace = True)
    
            # 将detail_df输出perf
            perf_name = ('om' + str(round(open_minute, 4))
                    + ' o' + str(round(open_threshold, 4))
                    + ' oc' + str(round(open_continous_threshold, 4))
                    + ' ow' + str(round(open_withdrawal_threshold, 4))
                    + ' cm' +str(round(close_minute, 4))
                    + ' c' +str(round(close_threshold, 4))
                    + ' cw' + str(round(close_withdrawal_threshold, 4))
                    + ' ' + str(round(withdrawal_close_count, 4)) 
                    + '+' + str(round(speed_close_count, 4)) 
                    + ' ' + 'Long ' + str(startdate) + '-' + str(enddate)
                    + ' ' + str(Capital_outcome)
                    + ' ' + 'perf.xlsx')
            writer1 = pd.ExcelWriter(
                '%s long stats/perf/'%file_name + perf_name,
                engine='xlsxwriter')
            detail_df.to_excel(writer1, sheet_name='stats')
            
            ## improving the appearence of perf_stats.xlsx
            workbook = writer1.book
            worksheet = writer1.sheets['stats']
            worksheet.set_default_row(15)
            # worksheet.autofilter('A1:Z1')
            # format 居中 12号微软雅黑
            format = workbook.add_format()
            format.set_font_name('Microsoft YaHei UI Light')
            format.set_align('justify')
            format.set_align('center')
            format.set_align('vjustify')
            format.set_align('vcenter')
            format.set_font_size(12)
            # format1 数字不显示小数点，否则在excel中会以科学计数法的形式显示
            format1 = workbook.add_format({'num_format': '0.00'})
            format1.set_font_name('Microsoft YaHei UI Light')
            format1.set_align('justify')
            format1.set_align('center')
            format1.set_align('vjustify')
            format1.set_align('vcenter')
            worksheet.set_column('A:A', 7, format1)
            worksheet.set_column('B:B', 19, format1)
            worksheet.set_column('C:C', 12, format)
            worksheet.set_column('D:D', 10, format)
            worksheet.set_column('E:E', 9, format)
            worksheet.set_column('F:F', 12, format)
            worksheet.set_column('G:G', 11, format)
            worksheet.set_column('H:H', 11, format)
            worksheet.set_column('I:I', 11, format)
            worksheet.set_column('J:J', 13, format)
            worksheet.set_column('K:K', 9, format1)
            worksheet.set_column('L:L', 8, format1)
            worksheet.set_column('M:O', 8, format)
            worksheet.set_column('P:P', 7.8, format1)
            worksheet.set_column('Q:R', 10, format)
            worksheet.set_column('S:S', 11.8, format)
            worksheet.set_column('T:Y', 10.4, format)
            worksheet.set_column('Z:Z', 22, format)
            worksheet.freeze_panes(1, 2)
            writer1.close()
            # End
    
            if len(transactions_df) != 0:  # 如果有交易，保存交易记录
                ## trans.xlsx
                writer2 = pd.ExcelWriter(
                    '%s long stats/trans/'%file_name
                    + 'om' + str(round(open_minute, 4))
                    + ' o' + str(round(open_threshold, 4))
                    + ' oc' + str(round(open_continous_threshold, 4))
                    + ' ow' + str(round(open_withdrawal_threshold, 4))
                    + ' cm' +str(round(close_minute, 4))
                    + ' c' +str(round(close_threshold, 4))
                    + ' cw' + str(round(close_withdrawal_threshold, 4))
                    + ' ' + str(round(withdrawal_close_count, 4)) 
                    + '+' + str(round(speed_close_count, 4)) + ' '
                    + 'Long ' + str(startdate) + '-' + str(enddate)
                    + ' ' + str(Capital_outcome)
                    + ' ' + 'trans.xlsx', engine='xlsxwriter'
                    )
                transactions_df.reset_index(drop=False).to_excel(writer2, sheet_name='stats')
    
                ## improving the appearence of perf_stats.xlsx
                workbook2 = writer2.book
                worksheet2 = writer2.sheets['stats']
                worksheet2.set_default_row(21)
                
                format3 = workbook2.add_format()
                format3.set_num_format('0')
                format3.set_font_name('Microsoft YaHei UI Light')
                format3.set_align('justify')
                format3.set_align('center')
                format3.set_align('vjustify')
                format3.set_align('vcenter')
                worksheet2.set_column('B:B', 17, format3)
                
                format2 = workbook2.add_format()
                format2.set_font_name('Microsoft YaHei UI Light')
                format2.set_align('justify')
                format2.set_align('center')
                format2.set_align('vjustify')
                format2.set_align('vcenter')
                format2.set_font_size(12)
                worksheet2.set_column('A:A', 11, format2)
                worksheet2.set_column('C:D', 11, format2)
                worksheet2.set_column('E:E', 14, format2)
                worksheet2.set_column('F:G', 13, format2)
                # worksheet2.freeze_panes(1, 1)
                ## Save excel
                writer2.close()
                
            ## stats df
            outcome_index = (str(open_minute)  
                             + ' ' 
                             + str(round(open_continous_threshold, 4)) 
                             + ' ' 
                             + str(round(close_withdrawal_threshold, 4))
                             )
            perf_temp = perf_outcome[-1:].capital.iloc[0] - 100
            outcome_stats.at[outcome_index, 'capital'] = perf_temp + 100
            trade_num = len(transactions_df) / 2
            outcome_stats.at[outcome_index,'trade_num'] = trade_num
            # average_profit = perf_temp / trade_num
            # outcome_stats.at[outcome_index,'average_profit'] = trade_num
            outcome_high, outcome_wd = get_outcome_withdrawal(perf_outcome.capital)
            outcome_stats.at[outcome_index, 'outcome_high'] = outcome_high
            outcome_stats.at[outcome_index, 'biggest_wd'] = outcome_wd
            max_inc_s = pd.to_numeric(detail_df['max_inc'], errors='coerce')
            ave_max_inc = max_inc_s.dropna().mean()   # 如果全是 NaN，这里会得到 NaN
            outcome_stats.at[outcome_index, 'ave_max_inc%'] = ave_max_inc
        
        

print("\ntime = --- %s seconds ---" % (time.time() - start_time))  # 总运算时间

## 如果是计算多个结果，那么将每个结果都plot到同一个折线图上
if len(outcome_stats) > 1:
    fig_stat_1 = plt.figure('stats', figsize=(18, 9))
    left = 0.033
    width = 0.943
    bottom = 0.055
    height = 0.9
    rect_line = [left, bottom, width, height] # below parameter
    ax_stat_1 = fig_stat_1.add_axes(rect_line)
    ax_stat_1.plot(outcome_stats.capital, label='capital')

    ax_stat_3 = ax_stat_1.twinx()
    ax_stat_3.plot(outcome_stats.trade_num, color = 'salmon',
                   label='trade num')
    ax_stat_3.tick_params(axis='y', colors='red')
    fig_stat_1.show()
    ax_stat_1.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=70)
    fig_stat_1.legend()
    plt.title('stats ' + str(startdate) + '-' + str(enddate))
    os.makedirs('stats %s long stats/'%file_name, exist_ok=True)
    plt.savefig('stats %s long stats/'%file_name
                + ' ' + save_name + ' '
                + str(for_num_1) + ' '
                + str(for_num_1) + ' '
                + 'all outcome.pdf', dpi=1000)
    outcome_stats.to_excel('stats %s long stats/'%file_name
                + ' ' + save_name + ' '
                + str(for_num_1) + ' '
                + str(for_num_1) + ' '
                + 'all outcome.xlsx')

    fig_ave = plt.figure('2', figsize=(18, 9))
    left, width, bottom, height = 0.033, 0.943, 0.055, 0.9
    ax_ave = fig_ave.add_axes([left, bottom, width, height])

    # 只画有效值（避免中间 NaN 让线断掉太多；也可不 dropna 直接画）
    s = pd.to_numeric(outcome_stats['ave_max_inc%'], errors='coerce')
    ax_ave.plot(s, label='ave_max_inc%')

    ax_stat_2 = ax_ave.twinx()
    ax_stat_2.plot(outcome_stats.biggest_wd, color = 'orange', 
                   label='biggest wd')
    ax_ave.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=70)
    fig_ave.legend()
    plt.title('ave_max_inc% ' + str(startdate) + '-' + str(enddate))

    os.makedirs('stats %s long stats/' % file_name, exist_ok=True)
    plt.savefig('stats %s long stats/' % file_name
                + ' ' + save_name + ' '
                + str(for_num_1) + ' '
                + str(for_num_1) + ' '
                + 'ave_max_inc.pdf', dpi=1000)

    
else:
    disk_path = 'C:/Users/lenovo/Desktop/backtest/'
    open_excel = False
    if open_excel == True:
        os.startfile(disk_path + '%s long stats/perf/'%file_name + perf_name)

# fig1 = plt.figure('BSA', figsize=(18, 9))
# fig1.set_figheight(9)
# fig1.set_figwidth(18)
# left = 0.04
# width = 0.94
# # bottom = 0.055
# bottom = 0.083
# height = 0.87

## 交互版本的图
if len(outcome_stats) == 1: 
    fig2 = plt.figure(figsize=(18, 9))
    left = 0.043
    width = 0.943
    bottom = 0.055
    height = 0.9
    rect_line = [left, bottom, width, height] # below parameter
    ax2 = fig2.add_axes(rect_line)
    
    underlying1 = underlying.reset_index(drop=True)  # reorder
    factor = underlying1['open'][0] # 以第一个open为基准
    underlying_ratio = pd.DataFrame() # 行情的变动比例
    underlying_ratio['Date'] = underlying1['Date']
    underlying_ratio[['open','high','low','close']] = underlying1[
        ['open','high','low','close']] / factor * 100
    x = underlying_ratio['close'] # 行情的变动比例
    
    date_list_0 = underlying1.Date.to_list()
    date_list = [str(i) for i in date_list_0]
    underlying_ratio.index = date_list
    # xaxis1 = date_list
    xaxis1 = perf_outcome.index
    yaxis1 = perf_outcome
    xaxis2 = x.index
    yaxis2 = x
    
    long_record = transactions_df.copy()
    long_record['target'] = long_record['Price']/factor * 100
    long_record = long_record[long_record.Type == 'long']
    long_record['date'] = long_record['Date'].astype(str).str[:-3]
    if len(long_record) != 0:
        scatter_r = ax2.scatter(
            long_record.index, long_record['target'], c ='red', s = 10)
    
    sell_record = transactions_df.copy()
    sell_record['target'] = sell_record['Price']/factor * 100
    sell_record = sell_record[sell_record.Type == 'sell']
    sell_record['date'] = sell_record['Date'].astype(str).str[:-3]
    if len(sell_record) != 0:
        close_type_1_df = sell_record[sell_record['Close_type']==1]
        scatter_g = ax2.scatter(
            close_type_1_df.index,
            close_type_1_df['target'], c ='green', s = 10)
        close_type_2_df = sell_record[sell_record['Close_type']==2]
        scatter_b = ax2.scatter(
            close_type_2_df.index,
            close_type_2_df['target'], c ='black', s = 10)
    
    # pd.set_option('display.float_format', lambda x: '%.2f' % x)
    trade_rc_df = transactions_df.reset_index().copy()
    trade_rc_df['target'] = x
    trade_rc_df['date'] = trade_rc_df['Date'].astype(str).str[:-2]
    if len(long_record) != 0:
        ## 交互
        annot_r = ax2.annotate(
                            "", xy=(0,0), xytext=(20,20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->")
                            )
        annot_r.set_visible(False)
        def update_annot_r(ind):
            index_num = ind["ind"][0]
            pos = scatter_r.get_offsets()[index_num]
            annot_r.xy = pos
            # Trade data
            trade_data = long_record.iloc[index_num]
            index0 = trade_data.name  # 在trade_data中的index
            date = str(trade_data['Date'])[:-3]
            # print(trade_data['Date'], date, date[-4:])
            pref_data = detail_df.loc[index0]
            high = pref_data.high
            # increase = pref_data['increase'] 
            # wd_per = round(pref_data['wd_per'] *100, 2)
            t_inc_per = round(pref_data['t_inc_per'], 2)
            execution = pref_data['execution'] 
            low_date = pref_data['low_date'] 
            new_opening_count = pref_data['new_opening_count'] 
            low_price = pref_data['low_price'] 
    
            
            text = (date[:-5] + ' ' + date[-5:]  + '\n'
                    + 'high: ' + str(high) + '\n'
                    + 'total_inc: ' + str(t_inc_per) + '%' + '\n'
                    + 'execution: ' + str(execution)  + '\n'
                    + 'low_date: ' + str(low_date)  + '\n'
                    + 'low_price: ' + str(low_price)  + '\n'
                    + 'new_opening_count: '  + str(new_opening_count)[:-2] + '\n'
                    + 'index: ' + str(index0)  + '\n'
                    )
            annot_r.set_text(text)
            # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
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
        
        annot_g = ax2.annotate("", xy=(0,0), xytext=(20,20),
                               textcoords="offset points",
                               bbox=dict(boxstyle="round", fc="w"),
                               arrowprops=dict(arrowstyle="->"))
        annot_g.set_visible(False)
    
    if len(sell_record) != 0:
        def update_annot_g(ind):
            index_num = ind["ind"][0]
            pos = scatter_g.get_offsets()[index_num]
            annot_g.xy = pos
            # Trade data
            trade_data = sell_record.iloc[index_num]
            index0 = trade_data.name  # 在trade_data中的index
            date = str(trade_data['Date'])[:-3]
            pref_data = detail_df.loc[index0]  # 数据集
            low = pref_data.low
            hld_wd_per = round(pref_data['hld_wd_per'], 2)
            holding_inc = round(pref_data['holding_inc'], 2)
            max_inc = round(pref_data['max_inc'], 2)
            max_wd = round(pref_data['max_wd'], 2)
            execution = pref_data['execution']
            # t_inc_per = round(pref_data['t_inc_per']*100, 2)
            low_date = pref_data['low_date'] 
            high_date = pref_data['high_date']
            high_price = pref_data['high_price'] 
            period = pref_data['period'] 
            text = (date[:-5] + ' ' + date[-5:]  + '\n'
                    + 'low: ' + str(low) + '\n'
                    + 'hld_wd_per: ' + str(hld_wd_per) + '%' + '\n'
                    + 'holding_inc: ' + str(holding_inc) + '\n'
                    + 'max_inc: ' +  str(max_inc) + '%'   + '\n'
                    + 'max_wd: ' + str(max_wd) + '%'   + '\n'
                    + 'execution2: ' + str(execution)  + '\n'
                    + 'period: ' + str(period)  + '\n'
                    + 'low_date: ' + str(low_date)  + '\n'
                    + 'high_date: ' + str(high_date)  + '\n'
                    + 'high_price: ' + str(high_price)  + '\n'
                    + 'index: ' + str(index0))
            
            annot_g.set_text(text)
            # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            annot_g.get_bbox_patch().set_alpha(0.4)
        
    if len(sell_record) != 0:
        def hover_g(event):
            vis = annot_g.get_visible()
            if event.inaxes == ax2:
                cont, ind = scatter_g.contains(event)
                if cont:
                    # print(ind)
                    update_annot_g(ind)
                    annot_g.set_visible(True)
                    fig2.canvas.draw_idle()
                else:
                    if vis:
                        annot_g.set_visible(False)
                        fig2.canvas.draw_idle()
        fig2.canvas.mpl_connect("motion_notify_event", hover_g)
        
        annot_b = ax2.annotate("", xy=(0,0),
                               xytext=(20,20),textcoords="offset points",
                               bbox=dict(boxstyle="round", fc="w"),
                               arrowprops=dict(arrowstyle="->"))
        annot_b.set_visible(False)
        
        # blue for speed close
        def update_annot_b(ind):
            index_num = ind["ind"][0]
            pos = scatter_b.get_offsets()[index_num]
            annot_b.xy = pos
            # Trade data
            trade_data = sell_record.iloc[index_num]
            index0 = trade_data.name  # 在trade_data中的index
            date = str(trade_data['Date'])[:-3]
            pref_data = detail_df.loc[index0]  # 数据集
            low = pref_data.low
            hld_wd_per = round(pref_data['hld_wd_per'], 2)
            # holding_inc = pref_data['holding_inc']
            max_inc = round(pref_data['max_inc'], 2)
            max_wd = round(pref_data['max_wd'], 2)
            execution = pref_data['execution']
            # t_inc_per = round(pref_data['t_inc_per']*100, 2)
            low_date = pref_data['low_date'] 
            high_date = pref_data['high_date']
            high_price = pref_data['high_price'] 
            period = pref_data['period'] 
            text = (date[:-5] + ' ' + date[-5:]  + '\n'
                    + 'low: ' + str(low) + '\n'
                    + 'hld_wd_per: ' + str(hld_wd_per) + '%'  + '\n'
                    + 'max_inc: ' +  str(max_inc) + '%'   + '\n'
                    + 'max_wd: ' + str(max_wd) + '%'   + '\n'
                    + 'execution2: ' + str(execution)  + '\n'
                    + 'period: ' + str(period)  + '\n'
                    + 'low_date: ' + str(low_date)  + '\n'
                    + 'high_date: ' + str(high_date)  + '\n'
                    + 'high_price: ' + str(high_price)  + '\n'
                    + 'index: ' + str(index0)
                    )
            
            annot_b.set_text(text)
            # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            annot_b.get_bbox_patch().set_alpha(0.4)
        def hover_b(event):
            vis = annot_b.get_visible()
            if event.inaxes == ax2:
                cont, ind = scatter_b.contains(event)
                if cont:
                    # print(ind)
                    update_annot_b(ind)
                    annot_b.set_visible(True)
                    fig2.canvas.draw_idle()
                else:
                    if vis:
                        annot_b.set_visible(False)
                        fig2.canvas.draw_idle()
        fig2.canvas.mpl_connect("motion_notify_event", hover_b)
        ## 交互End
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    degrees = 0  # x轴日期偏斜的角度
    plt.xticks(rotation=degrees)
    plt.title('%s'%  (' ' + str(round(Capital_outcome, 2))
                + ' om'+ str(round(open_minute, 4))
                + ' o' + str(round(open_threshold, 4))
                + ' oc' + str(round(open_continous_threshold, 4))
                + ' cm' +str(round(close_minute, 4))
                + ' c' +str(round(close_threshold, 4))
                + ' ow' + str(round(open_withdrawal_threshold, 4))
                + ' cw' + str(round(close_withdrawal_threshold, 4))
                + ' ' + str(round(withdrawal_close_count, 4))
                + '+' + str(round(speed_close_count, 4))
                + ' ' + str(startdate) + '-' + str(enddate)
              ))
    
    underlying1 = underlying.reset_index(drop=True)  # reorder
    factor = underlying1['open'][0] # 以第一个open为基准
    underlying_ratio = pd.DataFrame() # 行情的变动比例
    underlying_ratio['Date'] = underlying1['Date']
    underlying_ratio[['open','high','low','close']] = underlying1[
        ['open','high','low','close']] / factor * 100
    x = underlying_ratio['close'] # 行情的变动比例
    
    date_list_0 = underlying1.Date.to_list()
    date_list = [str(i) for i in date_list_0]
    xaxis1 = detail_df.index
    yaxis1 = detail_df.capital
    xaxis2 = x.index
    yaxis2 = x
    plt.plot(xaxis1, yaxis1, linewidth=1.2)  # 普通坐标
    candlestick2_ohlc(ax2, underlying_ratio.open, underlying_ratio.high, 
                      underlying_ratio.low, underlying_ratio.close,
                      width=0.7,
                      colorup='salmon', colordown='#2ca02c')  # 绘制K线走势
    # ax2.set_xticks(underlying_ratio.index, underlying_ratio['Date'])
    ax2.xaxis.set_major_locator(plt.MaxNLocator(12))
    # ax2.set_xticklabels(underlying_ratio['Date'], rotation = 0)
    # ax2.xaxis.set_major_locator(ticker.LinearLocator(12))  # 限制坐标数
