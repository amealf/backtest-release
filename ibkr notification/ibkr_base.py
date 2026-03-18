import threading
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class IBKRBase(EWrapper, EClient):
    """
    基础类，用于与 IBKR API 交互。
    提供底层连接管理，错误处理，以及用于 API 事件循环的后台线程。
    """
    def __init__(self):
        EClient.__init__(self, self)
        self.next_valid_order_id = None
        self._connected_event = threading.Event()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """处理来自 IBKR API 的消息与错误。"""
        if not hasattr(self, '_error_counts'):
            self._error_counts = {}
            self._error_last_times = {}

        # 通常 errorCode 2104, 2106, 2158 为数据农场连接成功等提示信息，而非真正的交易错误
        if errorCode in [2104, 2106, 2158]:
            pass # print(f"Info/Warning [{errorCode}]: {errorString}")
        elif errorCode in [2119, 10197]:
            now = time.time()
            last_time = self._error_last_times.get(errorCode, 0)
            
            # 如果距离上次相同错误超过60秒，重置计数
            if now - last_time > 60:
                self._error_counts[errorCode] = 0
                
            self._error_counts[errorCode] += 1
            self._error_last_times[errorCode] = now
            
            count = self._error_counts[errorCode]
            # 连续失败时每隔几次打印一次，避免刷屏
            if count >= 3 and count % 3 == 0:
                print(f"Error. Id: {reqId}, Code: {errorCode}, Msg: {errorString} (连续发生 {count} 次)")
        else:
            print(f"Error. Id: {reqId}, Code: {errorCode}, Msg: {errorString}")

    def nextValidId(self, orderId: int):
        """
        来自 IBKR API 的回调，提供下一个有效的订单 ID（Order ID）。
        在接收到该回调时，代表 API 已成功连接并准备好接受请求。
        """
        super().nextValidId(orderId)
        self.next_valid_order_id = orderId
        self._connected_event.set()

    def connect_and_run(self, host='127.0.0.1', port=7497, client_id=1, timeout=10.0):
        """连接到 API 并启动后台事件循环线程。"""
        self.connect(host, port, client_id)

        # 在独立线程启动 API 消息循环以免阻塞主线程
        self.api_thread = threading.Thread(target=self.run, daemon=True)
        self.api_thread.start()

        # 等待直到成功连接并且获取到了合法的 nextValidId
        is_connected = self._connected_event.wait(timeout)
        
        if not is_connected:
            print(f"在 {timeout} 秒内未能成功连接到 {host}:{port}。")
            self.disconnect()
            return False
        
        print(f"已成功连接至 IBKR 网关/TWS (端口: {port})。下一个有效的 Order ID: {self.next_valid_order_id}")
        return True
