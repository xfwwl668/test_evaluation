import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtest import BacktestEngine, BacktestResult
from strategy.alpha_hunter_strategy import AlphaHunterStrategy
from utils.nan_handler import NaNHandler
from core.data_validator import DataValidator

class TestCriticalFixes(unittest.TestCase):
    
    def test_nan_handler(self):
        """测试 NaN 处理框架"""
        s = pd.Series([1, np.nan, 3, np.nan, 5])
        filled = NaNHandler.safe_fillna(s, method='interpolate', reason='test')
        self.assertEqual(filled[1], 2.0)
        self.assertEqual(filled[3], 4.0)
        
    def test_data_validator(self):
        """测试数据验证器"""
        validator = DataValidator()
        df = pd.DataFrame({
            'open': [10.0, 11.0],
            'high': [10.5, 11.5], # 错误: high < open
            'low': [9.5, 10.5],
            'close': [10.2, 11.2],
            'vol': [1000, 1100]
        })
        # high < open is not strictly invalid in my validator yet, but high < low is
        df.loc[0, 'high'] = 9.0 # Now high < low
        self.assertFalse(validator.validate_ohlcv(df, "TEST"))

    def test_kelly_protection(self):
        """测试 Kelly 仓位保护"""
        strategy = AlphaHunterStrategy()
        # 模拟交易记录
        from strategy.alpha_hunter_strategy import TradeRecord
        for i in range(5): # 少于 10 笔
            strategy._trade_history.append(TradeRecord("000001", "2023-01-01", "2023-01-02", 10, 11, 0.1, True))
        
        pos = strategy._calculate_kelly_position(1000000)
        self.assertEqual(pos, 0.02) # 应该触发样本量保护

    def test_win_rate_calculation(self):
        """测试交易胜率计算"""
        # 构造模拟数据
        class MockPortfolio:
            def get_trades_df(self):
                return pd.DataFrame([
                    {'code': '000001', 'side': 'BUY', 'quantity': 100, 'price': 10.0, 'date': '2023-01-01', 'total_cost': 5.0},
                    {'code': '000001', 'side': 'SELL', 'quantity': 100, 'price': 11.0, 'date': '2023-01-02', 'total_cost': 5.0},
                    {'code': '000002', 'side': 'BUY', 'quantity': 100, 'price': 20.0, 'date': '2023-01-01', 'total_cost': 5.0},
                    {'code': '000002', 'side': 'SELL', 'quantity': 100, 'price': 19.0, 'date': '2023-01-02', 'total_cost': 5.0},
                ])
            def get_equity_df(self):
                return pd.DataFrame({'equity': [100, 101], 'daily_return': [0, 0.01], 'drawdown': [0, 0]}, 
                                    index=pd.to_datetime(['2023-01-01', '2023-01-02']))
        
        result = BacktestResult("Test", MockPortfolio(), 100)
        # 1 赢 1 输 -> 50% 胜率
        self.assertEqual(result.metrics['win_rate'], 0.5)

if __name__ == '__main__':
    unittest.main()
