# ============================================================================
# 文件: utils/trading_calendar.py
# ============================================================================
"""
交易日历 - A股市场交易日处理

功能:
1. 中国A股交易日历管理
2. 节假日自动过滤
3. 交易日偏移计算
4. 调仓日校验
"""
import pandas as pd
from datetime import datetime, date
from typing import List, Set, Optional, Tuple
import logging


class TradingCalendar:
    """
    中国股市交易日历
    
    支持的交易所:
    - SSE (上海证券交易所)
    - SZSE (深圳证券交易所)
    
    节假日数据来源:
    - 内置节假日数据库 (每年更新)
    - 可从 exchange_calendars 库导入
    """
    
    # 内置节假日数据 (2020-2025年主要节假日)
    # 格式: YYYY-MM-DD
    HOLIDAYS = {
        # 2020年
        '2020-01-01', '2020-01-24', '2020-01-27', '2020-01-28', '2020-01-29', '2020-01-30', '2020-01-31',
        '2020-04-04', '2020-04-05', '2020-04-06',
        '2020-05-01', '2020-05-04', '2020-05-05',
        '2020-06-25', '2020-06-26', '2020-06-27',
        '2020-10-01', '2020-10-02', '2020-10-05', '2020-10-06', '2020-10-07', '2020-10-08',
        # 2021年
        '2021-01-01', '2021-02-11', '2021-02-12', '2021-02-15', '2021-02-16', '2021-02-17',
        '2021-04-03', '2021-04-04', '2021-04-05',
        '2021-05-03', '2021-05-04', '2021-05-05',
        '2021-06-12', '2021-06-13', '2021-06-14',
        '2021-09-19', '2021-09-20', '2021-09-21',
        '2021-10-01', '2021-10-02', '2021-10-04', '2021-10-05', '2021-10-06', '2021-10-07',
        # 2022年
        '2022-01-01', '2022-01-31', '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-04',
        '2022-04-02', '2022-04-03', '2022-04-04', '2022-04-05',
        '2022-04-30', '2022-05-02', '2022-05-03', '2022-05-04',
        '2022-06-03', '2022-06-04', '2022-06-05',
        '2022-09-10', '2022-09-11', '2022-09-12',
        '2022-10-01', '2022-10-02', '2022-10-03', '2022-10-04', '2022-10-05', '2022-10-06', '2022-10-07',
        # 2023年
        '2023-01-01', '2023-01-02', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27',
        '2023-04-05', '2023-04-29', '2023-04-30', '2023-05-01', '2023-05-02', '2023-05-03',
        '2023-06-22', '2023-06-23', '2023-06-24', '2023-06-25',
        '2023-09-29', '2023-09-30',
        '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-06',
        # 2024年
        '2024-01-01', '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14', '2024-02-15', '2024-02-16', '2024-02-17',
        '2024-04-04', '2024-04-05', '2024-04-06',
        '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05',
        '2024-06-10',
        '2024-09-15', '2024-09-16', '2024-09-17',
        '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-07',
        # 2025年
        '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02', '2025-02-03', '2025-02-04',
    }
    
    def __init__(self, holidays: Optional[Set[str]] = None):
        """
        初始化交易日历
        
        Args:
            holidays: 自定义节假日集合 (默认使用内置数据)
        """
        self.holidays = holidays or self.HOLIDAYS
        self.logger = logging.getLogger("TradingCalendar")
        
    def is_trading_day(self, dt: datetime) -> bool:
        """
        判断是否为交易日
        
        Args:
            dt: 日期时间对象
        
        Returns:
            True if trading day, False otherwise
        """
        # 1. 检查周末 (周六周日)
        if dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
        
        # 2. 检查节假日
        date_str = dt.strftime('%Y-%m-%d')
        if date_str in self.holidays:
            return False
        
        return True
    
    def is_trading_day_str(self, date_str: str) -> bool:
        """
        判断日期字符串是否为交易日
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD)
        
        Returns:
            True if trading day, False otherwise
        """
        dt = pd.to_datetime(date_str).to_pydatetime()
        return self.is_trading_day(dt)
    
    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """
        获取区间内所有交易日
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        Returns:
            交易日列表 (按时间排序)
        """
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = []
        
        for d in all_days:
            dt = d.to_pydatetime()
            if self.is_trading_day(dt):
                trading_days.append(d.strftime('%Y-%m-%d'))
        
        return trading_days
    
    def offset_trading_day(self, current_date: str, offset: int) -> str:
        """
        交易日偏移 (正数向前，负数向后)
        
        Args:
            current_date: 当前日期 (YYYY-MM-DD)
            offset: 偏移天数
        
        Returns:
            偏移后的交易日 (YYYY-MM-DD)
        """
        if offset == 0:
            return current_date
        
        dt = pd.to_datetime(current_date).to_pydatetime()
        count = 0
        direction = 1 if offset > 0 else -1
        abs_offset = abs(offset)
        
        curr_dt = dt
        while count < abs_offset:
            curr_dt += pd.Timedelta(days=direction)
            if self.is_trading_day(curr_dt):
                count += 1
        
        return curr_dt.strftime('%Y-%m-%d')
    
    def next_trading_day(self, current_date: str) -> str:
        """
        获取下一个交易日
        
        Args:
            current_date: 当前日期 (YYYY-MM-DD)
        
        Returns:
            下一个交易日 (YYYY-MM-DD)
        """
        return self.offset_trading_day(current_date, 1)
    
    def prev_trading_day(self, current_date: str) -> str:
        """
        获取上一个交易日
        
        Args:
            current_date: 当前日期 (YYYY-MM-DD)
        
        Returns:
            上一个交易日 (YYYY-MM-DD)
        """
        return self.offset_trading_day(current_date, -1)
    
    def filter_trading_dates(self, dates: List[str]) -> List[str]:
        """
        过滤非交易日
        
        Args:
            dates: 日期列表
        
        Returns:
            过滤后的交易日列表
        """
        return [d for d in dates if self.is_trading_day_str(d)]
    
    def validate_rebalance_dates(self, dates: List[str]) -> Tuple[List[str], List[str]]:
        """
        验证调仓日期，过滤非交易日
        
        Args:
            dates: 待验证日期列表
        
        Returns:
            (trading_dates, non_trading_dates)
        """
        trading = []
        non_trading = []
        
        for d in dates:
            if self.is_trading_day_str(d):
                trading.append(d)
            else:
                non_trading.append(d)
                self.logger.warning(f"Date {d} is not a trading day, removing from rebalance schedule")
        
        return trading, non_trading
    
    def get_settlement_date(self, trade_date: str) -> str:
        """
        获取T+1结算日期
        
        Args:
            trade_date: 交易日期 (YYYY-MM-DD)
        
        Returns:
            结算日期 (YYYY-MM-DD)
        """
        return self.next_trading_day(trade_date)
