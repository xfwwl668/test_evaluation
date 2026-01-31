# ============================================================================
# 文件: utils/trading_calendar.py
# ============================================================================
"""
交易日历 - A股市场交易日处理
"""
import pandas as pd
from datetime import datetime, date
from typing import List, Set

class TradingCalendar:
    """
    中国股市交易日历 (简化版，实际应使用 exchange_calendars 或 tushare 数据)
    """
    
    def __init__(self):
        # 这里应该加载实际的节假日数据
        self.holidays = set() 
        
    def is_trading_day(self, dt: datetime) -> bool:
        """
        判断是否为交易日
        """
        # 1. 检查周末
        if dt.weekday() >= 5: # 5=Saturday, 6=Sunday
            return False
            
        # 2. 检查节假日 (简化: 这里需要真实数据)
        date_str = dt.strftime('%Y-%m-%d')
        if date_str in self.holidays:
            return False
            
        return True

    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """
        获取区间内所有交易日
        """
        all_days = pd.date_range(start=start_date, end=end_date)
        trading_days = [d.strftime('%Y-%m-%d') for d in all_days if self.is_trading_day(d)]
        return trading_days

    def offset_trading_day(self, current_date: str, offset: int) -> str:
        """
        交易日偏移
        """
        # 简化实现
        dt = pd.to_datetime(current_date)
        count = 0
        direction = 1 if offset > 0 else -1
        abs_offset = abs(offset)
        
        curr_dt = dt
        while count < abs_offset:
            curr_dt += pd.Timedelta(days=direction)
            if self.is_trading_day(curr_dt):
                count += 1
                
        return curr_dt.strftime('%Y-%m-%d')
