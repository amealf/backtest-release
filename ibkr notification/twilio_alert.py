import os
from twilio.rest import Client

# ==========================================
# Twilio 账号配置区 (Twilio Credentials)
# ==========================================
# Twilio 设置 (请确 Auth Token 已正确填写)
TWILIO_ACCOUNT_SID = 'AC838b8ce0b72a995b4623bede8fa996d7'
TWILIO_AUTH_TOKEN = '6f9af1d2fe8c17a599f427a139a44009'
TWILIO_TO_PHONE = '+85246130160'
TWILIO_FROM_PHONE = '+19789559770'
# ==========================================

class TwilioAlert:
    """Twilio 预警电话触发器"""
    def __init__(self):
        self.alert_triggered = False
        
    def check_and_trigger(self, current_price, threshold, is_enabled):
        """
        检查当前价格是否触发预警，若触发且未拨打过，则拨打电话。
        此函数设计为可以在每次价格刷新时调用。
        """
        if not is_enabled:
            return
            
        if current_price < threshold and not self.alert_triggered:
            print(f"\n!!! 触发 Twilio 电话预警：当前价格 {current_price} 已跌破设定阈值 {threshold} !!!")
            self._trigger_phone_alert()
            self.alert_triggered = True # 标记为已触发，避免重复拨打
            
    def _trigger_phone_alert(self):
        """内部函数：调用 Twilio Programmable Voice 发起通话"""
        try:
            print("正在调用 Twilio API 拨打电话...")
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            # 使用底层的 Programmable Voice API 拨打电话
            call = client.calls.create(
                twiml='<Response><Say>Warning! Silver price drops below your threshold.</Say></Response>',
                to=TWILIO_TO_PHONE,
                from_=TWILIO_FROM_PHONE
            )
            print(f"电话拨打请求成功提交，Call SID: {call.sid}")
            
        except Exception as e:
            # 对于诸如 20500 之类的服务器错误，直接打印而不中断主程序
            print(f"电话拨打请求失败，请检查您的 Twilio 账户权限或记录: {e}")
