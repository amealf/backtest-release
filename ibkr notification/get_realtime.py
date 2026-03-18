import argparse
import time
import threading
import sys
import winsound
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum
from ibkr_base import IBKRBase
from twilio_alert import TwilioAlert

# ==========================================
# 报警系统全局配置区 (Alert Configurations)
# ==========================================
# 合约参数
SYMBOL = "SI"
CONTRACT_MONTH = "202605"
MULTIPLIER = "1000"
EXCHANGE = "COMEX"

# 价格跌破预警配置
ENABLE_PRICE_THRESHOLD_ALERT = False
# 跌破此价格时触发预警
ALERT_PRICE_THRESHOLD = 87.8
# 是否启用 Twilio 电话预警 (默认关闭)
ENABLE_TWILIO_ALERT = False
# 是否启用 Windows 系统提示音预警
ENABLE_WINSOUND_ALERT = True
# 滚动窗口跌幅预警配置 (Rolling Window Drop Alerts)
ENABLE_ROLLING_DROP_ALERT = True
ENABLE_DROP_1M = False
DROP_1M_THRESHOLD = 0.0055   # 1分钟跌幅
ENABLE_DROP_3M = False
DROP_3M_THRESHOLD = 0.007   # 3分钟跌幅
ENABLE_DROP_5M = True
DROP_5M_THRESHOLD = 0.0082   # 5分钟跌幅
ENABLE_DROP_10M = False
DROP_10M_THRESHOLD = 0.013   # 10分钟跌幅

# 持续下跌预警 (Continuous Decline Alert)
# 条件: 每个 200 秒子窗口的跌幅都 > 0，且总跌幅 >= CONTINUOUS_DECLINE_THRESHOLD
ENABLE_CONTINUOUS_DECLINE_ALERT = True
CONTINUOUS_DECLINE_WINDOW = 200   # 子窗口秒数
CONTINUOUS_DECLINE_THRESHOLD = 0.011  # 总跌幅 1.2%
# ==========================================

from collections import deque

class RealtimeClient(IBKRBase):
    """
    具体的客户端，用于从 IBKR 获取期货的实时数据并打印。
    """
    def __init__(self):
        super().__init__()
        self.alerter = TwilioAlert()
        self.winsound_triggered = False
        
        # 用于滚动计算跌幅的历史数据队列: (timestamp, price)
        self.tick_history = deque()
        self.last_alert_time_1m = 0
        self.last_alert_time_3m = 0
        self.last_alert_time_5m = 0
        self.last_alert_time_10m = 0
        self.last_alert_time_continuous = 0
        
        # 用于中断提示音的标志位
        self.stop_alarm_flag = threading.Event()
        self._stop_btn_window = None
        
    def play_alarm(self, freq, duration_ms):
        """在后台线程播放声音，避免阻塞 IBKR 行情接收，并支持随时中断"""
        if not ENABLE_WINSOUND_ALERT:
            return
        
        # 先停掉上一个还在响的声音，避免两个声音同时播放
        self.stop_alarm_flag.set()
        time.sleep(0.05)  # 等待旧线程检测到 flag
        self.stop_alarm_flag.clear()
        
        def beep_loop():
            # 将总时长拆分成每次 500 毫秒的短鸣，以便随时响应打断
            loops = duration_ms // 500
            for _ in range(loops):
                if self.stop_alarm_flag.is_set():
                    break
                winsound.Beep(freq, 500)
            # 声音结束后自动关闭按钮窗口
            self._close_stop_btn()
                
        # 启动后台发声线程
        threading.Thread(target=beep_loop, daemon=True).start()
        # 弹出停止按钮
        self._show_stop_btn()
        
    def stop_alarm(self):
        """中断当前正在播放的提示音"""
        if not self.stop_alarm_flag.is_set():
            self.stop_alarm_flag.set()
            print("\n[系统] 提示音已手动中断!")
        self._close_stop_btn()
            
    def _show_stop_btn(self):
        """弹出一个置顶的小窗口，点击按钮即可停止声音"""
        def _create_window():
            try:
                import tkinter as tk
                root = tk.Tk()
                root.title("警报")
                root.attributes('-topmost', True)
                root.geometry('280x80+100+100')
                root.configure(bg='#1a1a2e')
                root.resizable(False, False)
                btn = tk.Button(
                    root, text="停止报警声音",
                    command=lambda: [self.stop_alarm(), root.destroy()],
                    font=('Microsoft YaHei UI', 14, 'bold'),
                    bg='#e94560', fg='white',
                    activebackground='#c23152', activeforeground='white',
                    relief='flat', cursor='hand2',
                    padx=20, pady=8
                )
                btn.pack(expand=True)
                self._stop_btn_window = root
                root.protocol('WM_DELETE_WINDOW', lambda: [self.stop_alarm(), root.destroy()])
                root.mainloop()
            except Exception:
                pass
        threading.Thread(target=_create_window, daemon=True).start()
        
    def _close_stop_btn(self):
        """安全关闭按钮窗口"""
        if self._stop_btn_window:
            try:
                self._stop_btn_window.after(0, self._stop_btn_window.destroy)
            except Exception:
                pass
            self._stop_btn_window = None

    def tickPrice(self, reqId, tickType, price, attrib):
        """接收实时价格变动的回调"""
        tick_name = TickTypeEnum.to_str(tickType)
        
        # 我们现在只关心 最新成交价 (LAST, Type 4)
        if tickType == 4:
            print(f"[{time.strftime('%H:%M:%S')}] 最新成交价 (LAST, Type {tickType}): {price}")
            
            if ENABLE_PRICE_THRESHOLD_ALERT and price < ALERT_PRICE_THRESHOLD:
                # 1. 本地声音预警
                if ENABLE_WINSOUND_ALERT and not self.winsound_triggered:
                    print(f"\n!!! 触发本地声音预警：当前价格 {price} 已跌破总设定阈值 {ALERT_PRICE_THRESHOLD} !!!")
                    print("--> 按键盘任意键可立即关闭提示音 <--")
                    # 使用多线程播放，时长 15000 毫秒（15秒）
                    self.play_alarm(1000, 15000)
                    self.winsound_triggered = True
                
                # 2. Twilio 电话预警
                self.alerter.check_and_trigger(price, ALERT_PRICE_THRESHOLD, ENABLE_TWILIO_ALERT)

            # --- 滚动窗口跌幅计算 ---
            if ENABLE_ROLLING_DROP_ALERT:
                now = time.time()
                self.tick_history.append((now, price))
                
                # 移除超过 10 分钟 (600秒) 的历史价格
                while self.tick_history and now - self.tick_history[0][0] > 600:
                    self.tick_history.popleft()
                    
                self._check_rolling_drops(now, price)
                self._check_continuous_decline(now, price)

    def _calc_decrease_from_ticks(self, prices):
        """完全按照 short_momentum.py 中 get_decrease 的逻辑计算实时 tick 序列跌幅"""
        if not prices:
            return 0.0, 0.0
            
        high = prices[0]
        low = prices[0]
        decrease = 0.0
        
        for price in prices:
            if price >= high:
                low = price
                high = price
            elif price < low:
                low = price
            decrease = high - low
            
        return decrease, high

    def _check_rolling_drops(self, now, current_price):
        prices_1m = []
        prices_3m = []
        prices_5m = []
        prices_10m = []
        
        # 顺序遍历收集各个窗口的价格 (tick_history 越靠左越旧)
        for t, p in self.tick_history:
            dt = now - t
            if dt <= 60:
                prices_1m.append(p)
            if dt <= 180:
                prices_3m.append(p)
            if dt <= 300:
                prices_5m.append(p)
            if dt <= 600:
                prices_10m.append(p)
                
        dec_1m, base_1m = self._calc_decrease_from_ticks(prices_1m)
        dec_3m, base_3m = self._calc_decrease_from_ticks(prices_3m)
        dec_5m, base_5m = self._calc_decrease_from_ticks(prices_5m)
        dec_10m, base_10m = self._calc_decrease_from_ticks(prices_10m)
        
        drop_1m = dec_1m / base_1m if base_1m > 0 else 0
        drop_3m = dec_3m / base_3m if base_3m > 0 else 0
        drop_5m = dec_5m / base_5m if base_5m > 0 else 0
        drop_10m = dec_10m / base_10m if base_10m > 0 else 0
        
        # 只 print 开启了的窗口
        parts = []
        if ENABLE_DROP_1M:
            parts.append(f"1m: -{drop_1m*100:.2f}%")
        if ENABLE_DROP_3M:
            parts.append(f"3m: -{drop_3m*100:.2f}%")
        if ENABLE_DROP_5M:
            parts.append(f"5m: -{drop_5m*100:.2f}%")
        if ENABLE_DROP_10M:
            parts.append(f"10m: -{drop_10m*100:.2f}%")
        if parts:
            print(f"[{time.strftime('%H:%M:%S')}] 跌幅监测 - {', '.join(parts)}")
        
        # 1分钟窗口预警 (冷却 60 秒)
        if ENABLE_DROP_1M and drop_1m > DROP_1M_THRESHOLD and now - self.last_alert_time_1m > 60:
            print(f"\n!!! [1分钟窗口] 跌幅预警: 最新价 {current_price} 较基准 {base_1m} 下跌 {drop_1m*100:.2f}% (>{DROP_1M_THRESHOLD*100}%) !!!")
            self.play_alarm(1200, 30000)
            self.last_alert_time_1m = now
            
        # 3分钟窗口预警 (冷却 180 秒)
        if ENABLE_DROP_3M and drop_3m > DROP_3M_THRESHOLD and now - self.last_alert_time_3m > 180:
            print(f"\n!!! [3分钟窗口] 跌幅预警: 最新价 {current_price} 较基准 {base_3m} 下跌 {drop_3m*100:.2f}% (>{DROP_3M_THRESHOLD*100}%) !!!")
            self.play_alarm(1000, 30000)
            self.last_alert_time_3m = now
            
        # 5分钟窗口预警 (冷却 300 秒)
        if ENABLE_DROP_5M and drop_5m > DROP_5M_THRESHOLD and now - self.last_alert_time_5m > 300:
            print(f"\n!!! [5分钟窗口] 跌幅预警: 最新价 {current_price} 较基准 {base_5m} 下跌 {drop_5m*100:.2f}% (>{DROP_5M_THRESHOLD*100}%) !!!")
            self.play_alarm(800, 30000)
            self.last_alert_time_5m = now
            
        # 10分钟窗口预警 (冷却 600 秒)
        if ENABLE_DROP_10M and drop_10m > DROP_10M_THRESHOLD and now - self.last_alert_time_10m > 600:
            print(f"\n!!! [10分钟窗口] 跌幅预警: 最新价 {current_price} 较基准 {base_10m} 下跌 {drop_10m*100:.2f}% (>{DROP_10M_THRESHOLD*100}%) !!!")
            self.play_alarm(600, 30000)
            self.last_alert_time_10m = now

    def _check_continuous_decline(self, now, current_price):
        """持续下跌预警: 将历史数据按 CONTINUOUS_DECLINE_WINDOW 秒切分子窗口,
        从最新往前找连续每段跌幅都 > 0 的「连续下跌链」,
        一旦某段跌幅为 0 就 reset, 只看 reset 之后的连续部分。
        连续链的总跌幅 >= CONTINUOUS_DECLINE_THRESHOLD 时触发报警。"""
        if not ENABLE_CONTINUOUS_DECLINE_ALERT:
            return
            
        win = CONTINUOUS_DECLINE_WINDOW
        oldest_t = self.tick_history[0][0] if self.tick_history else now
        total_span = now - oldest_t
        if total_span < win:
            return
            
        # 按 win 秒切分子窗口
        n_windows = int(total_span // win)
        if n_windows < 2:
            return
            
        # 子窗口起点: now - n_windows * win
        start_time = now - n_windows * win
        
        # 收集每个子窗口的价格序列
        window_prices = [[] for _ in range(n_windows)]
        for t, p in self.tick_history:
            if t < start_time:
                continue
            w_idx = int((t - start_time) // win)
            if w_idx >= n_windows:
                w_idx = n_windows - 1
            window_prices[w_idx].append(p)
        
        # 从最后一个窗口往前找: 连续跌幅 > 0 的最长「尾部链」
        # 一旦碰到跌幅 <= 0 的窗口就 reset (停止往前延伸)
        streak_start = n_windows  # 连续链的起始窗口 index
        for i in range(n_windows - 1, -1, -1):
            wp = window_prices[i]
            if len(wp) < 2:
                break
            dec, base = self._calc_decrease_from_ticks(wp)
            if dec <= 0:
                break
            streak_start = i
        
        streak_len = n_windows - streak_start
        
        if streak_len < 2:
            # 连续下跌链不足 2 段, 不打印也不报警
            return
            
        # 计算连续链的总跌幅
        chain_start_time = start_time + streak_start * win
        chain_prices = []
        for t, p in self.tick_history:
            if t >= chain_start_time:
                chain_prices.append(p)
        
        total_dec, total_base = self._calc_decrease_from_ticks(chain_prices)
        total_drop = total_dec / total_base if total_base > 0 else 0
        
        # 每次都 print 当前持续下跌状态
        print(f"[{time.strftime('%H:%M:%S')}] 持续下跌 - 连续 {streak_len}x{win}s 均在跌, 总跌幅: -{total_drop*100:.2f}%")
        
        # 冷却 600 秒
        if total_drop >= CONTINUOUS_DECLINE_THRESHOLD and now - self.last_alert_time_continuous > 600:
            print(f"\n!!! [持续下跌] 预警: 连续 {streak_len}x{win}s 每段均在下跌, "
                  f"总跌幅 {total_drop*100:.2f}% (>={CONTINUOUS_DECLINE_THRESHOLD*100}%) !!!")
            self.play_alarm(900, 30000)
            self.last_alert_time_continuous = now
            
    def tickSize(self, reqId, tickType, size):
        """接收实时数量变动的回调（暂时忽略不打印）"""
        pass
            
    def tickString(self, reqId, tickType, value):
        """字符串类型数据的回调（暂时忽略不打印）"""
        pass

    def historicalData(self, reqId, bar):
        """接收历史数据 K 线的返回"""
        print(f"历史数据 Bar: Time: {bar.date}, Open: {bar.open}, High: {bar.high}, Low: {bar.low}, Close: {bar.close}, Volume: {bar.volume}")
        
    def historicalDataEnd(self, reqId, start, end):
        """标明历史数据获取完毕"""
        print(f"已完成历史数据请求区间获取: {start} -> {end}")

def main():
    # 使用 argparse 在程序开头解析命令行参数
    parser = argparse.ArgumentParser(description=f"获取 IBKR {SYMBOL} 期货的实时数据。")
    parser.add_argument(
        "--live", 
        action="store_true", 
        help="使用真实的交易端口 (Live Trading, 7496)。如果不指定该选项，则默认使用纸交易端 (Paper Trading, 7497)。"
    )
    args = parser.parse_args()
    
    # 根据命令行参数设置所使用的连接端口，根据截图目前的设置是 4002 
    port = 7496 if args.live else 4002
    env_name = "Live Trading" if args.live else "Paper Trading"
    print(f"[{env_name}] 正在尝试连接至 IBKR TWS/Gateway (端口 {port})...")
    
    import random
    app = RealtimeClient()
    # 使用随机一个 client_id 连接，避免客户号码已被使用的错误
    client_id = random.randint(100, 9999)
    connected = app.connect_and_run(host='127.0.0.1', port=port, client_id=client_id)
    
    if not connected:
        return
    
    # 定义期货合约
    contract = Contract()
    contract.symbol = SYMBOL
    contract.secType = "FUT"
    contract.exchange = EXCHANGE
    contract.lastTradeDateOrContractMonth = CONTRACT_MONTH
    contract.multiplier = MULTIPLIER
    
    print(f"\n正在请求 {contract.symbol} {contract.lastTradeDateOrContractMonth} 的实时行情数据流...")
    
    # 强制请求实时的市场数据 (Type 1)
    # 1: 实时流 (Live), 3: 延迟流 (Delayed), 4: 延迟冻结 (Delayed Frozen)
    app.reqMarketDataType(1) 

    # 调起行情数据请求
    # reqMktData(reqId, contract, genericTickList, snapshot, regulatorySnapshot, mktDataOptions)
    app.reqMktData(reqId=1, 
                   contract=contract, 
                   genericTickList="", 
                   snapshot=False, 
                   regulatorySnapshot=False, 
                   mktDataOptions=[])
    
    print("正在接收数据推送 (按 Ctrl+C 退出程序，报警时会弹出按钮可停止声音)...\n")
    
    try:
        # 当连接存在且没收到异常时，阻塞主线程以保持后台监听线程存活
        while app.isConnected():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n收到用户中断指令。正在断开连接...")
    finally:
        # 退出前优雅清理：取消订阅跟断开连接
        app.cancelMktData(reqId=1)
        app.disconnect()
        print("已断开连接。")

if __name__ == "__main__":
    main()
