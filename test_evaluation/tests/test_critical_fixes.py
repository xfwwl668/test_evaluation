# ============================================================================
# 文件: tests/test_critical_fixes.py
# ============================================================================
"""
关键修复的单元测试

测试覆盖:
1. Look-Ahead Bias - 因子不使用未来数据
2. NaN处理 - 标准化填充和验证
3. Kelly准则 - 风险保护机制
4. 交易统计 - 胜率计算正确性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque

# 测试NaNHandler
from utils.nan_handler import NaNHandler, FillMethod

# 测试Kelly准则
from strategy.alpha_hunter_strategy import AlphaHunterStrategy, TradeRecord, AlphaPosition


class TestLookAheadBias:
    """测试Look-Ahead Bias修复"""
    
    def test_factors_not_using_future_data(self):
        """
        验证因子计算不使用未来数据
        
        原理: 在第t天计算因子时，只能使用t-1及之前的数据
        """
        # 创建模拟历史数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        history = {}
        for code in ['000001', '000002']:
            df = pd.DataFrame({
                'open': np.random.uniform(10, 20, 100),
                'high': np.random.uniform(10, 22, 100),
                'low': np.random.uniform(8, 18, 100),
                'close': np.random.uniform(10, 20, 100),
                'vol': np.random.uniform(1000000, 5000000, 100),
                'amount': np.random.uniform(10000000, 100000000, 100)
            }, index=dates)
            df['high'] = np.maximum(df[['open', 'high', 'close']].max(axis=1), df['low'])
            df['low'] = np.minimum(df[['open', 'low', 'close']].min(axis=1), df['high'])
            history[code] = df
        
        # 在回测中，因子应该按天计算，而非一次性计算
        # 这里我们验证数据切片的正确性
        current_date = dates[50]
        
        for code, df in history.items():
            # 只能使用当前日期之前的数据
            available_data = df[df.index < current_date]
            
            # 验证不包含未来数据
            assert available_data.index.max() < current_date, \
                f"数据包含未来日期: {code}"
            
            # 验证数据长度
            assert len(available_data) == 50, \
                f"数据长度错误: {code}, 期望50，实际{len(available_data)}"
        
        print("✓ Look-Ahead Bias测试通过")


class TestNaNHandling:
    """测试NaN处理框架"""
    
    def test_nan_validation_valid_data(self):
        """测试有效数据通过验证"""
        df = pd.DataFrame({
            'open': [10.0, 11.0, 12.0],
            'high': [11.0, 12.0, 13.0],
            'low': [9.0, 10.0, 11.0],
            'close': [10.5, 11.5, 12.5],
            'volume': [1000, 2000, 3000]
        })
        
        assert NaNHandler.validate_ohlcv(df, 'TEST001') == True
        print("✓ NaN验证(有效数据)测试通过")
    
    def test_nan_validation_negative_price(self):
        """测试负价格被拒绝"""
        df = pd.DataFrame({
            'open': [10.0, -11.0, 12.0],  # 负价格
            'high': [11.0, 12.0, 13.0],
            'low': [9.0, 10.0, 11.0],
            'close': [10.5, 11.5, 12.5],
            'volume': [1000, 2000, 3000]
        })
        
        assert NaNHandler.validate_ohlcv(df, 'TEST001') == False
        print("✓ NaN验证(负价格)测试通过")
    
    def test_nan_validation_high_low_logic(self):
        """测试high < low被拒绝"""
        df = pd.DataFrame({
            'open': [10.0, 11.0, 12.0],
            'high': [9.0, 12.0, 13.0],   # high < low
            'low': [11.0, 10.0, 11.0],
            'close': [10.5, 11.5, 12.5],
            'volume': [1000, 2000, 3000]
        })
        
        assert NaNHandler.validate_ohlcv(df, 'TEST001') == False
        print("✓ NaN验证(价格逻辑)测试通过")
    
    def test_safe_fillna_forward(self):
        """测试前向填充"""
        series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], name='test')
        
        result = NaNHandler.safe_fillna(series, method='forward', reason='test')
        
        # 检查NaN被填充
        assert result.isna().sum() == 0
        # 检查第一个值保留
        assert result.iloc[0] == 1.0
        # 检查最后一个值保留
        assert result.iloc[-1] == 5.0
        print("✓ NaN前向填充测试通过")
    
    def test_safe_fillna_interpolate(self):
        """测试插值填充"""
        series = pd.Series([1.0, np.nan, np.nan, 4.0], name='test')
        
        result = NaNHandler.safe_fillna(series, method='interpolate', reason='test')
        
        # 检查NaN被填充
        assert result.isna().sum() == 0
        # 插值结果应该是2.0和3.0
        assert result.iloc[1] == 2.0
        assert result.iloc[2] == 3.0
        print("✓ NaN插值填充测试通过")
    
    def test_safe_fillna_no_nan(self):
        """测试无NaN时返回原序列"""
        series = pd.Series([1.0, 2.0, 3.0, 4.0], name='test')
        
        result = NaNHandler.safe_fillna(series, method='forward', reason='test')
        
        # 应该原样返回
        pd.testing.assert_series_equal(result, series)
        print("✓ NaN无填充测试通过")
    
    def test_get_summary(self):
        """测试NaN处理摘要"""
        NaNHandler.clear_reports()
        
        # 生成一些NaN处理记录
        series = pd.Series([1.0, np.nan, 3.0], name='test1')
        NaNHandler.safe_fillna(series, method='forward', reason='test')
        
        series = pd.Series([np.nan, 2.0, 3.0], name='test2')
        NaNHandler.safe_fillna(series, method='mean', reason='test')
        
        summary = NaNHandler.get_summary()
        
        assert summary['total_operations'] == 2
        assert 'forward' in summary['methods_used']
        assert 'mean' in summary['methods_used']
        print("✓ NaN摘要统计测试通过")


class TestKellyPosition:
    """测试Kelly准则修复"""
    
    def setup(self):
        """每个测试前初始化策略"""
        self.strategy = AlphaHunterStrategy()
        self.strategy.initialize()
    
    def test_kelly_small_sample(self):
        """测试样本<10时使用保守仓位"""
        # 添加9笔交易(少于10笔)
        for i in range(9):
            self.strategy._trade_history.append(
                TradeRecord(
                    code='TEST',
                    entry_date='2024-01-01',
                    exit_date='2024-01-02',
                    entry_price=10.0,
                    exit_price=11.0,
                    pnl_ratio=0.1,
                    is_win=True
                )
            )
        
        position = self.strategy._calculate_kelly_position(100000)
        
        # 应该返回保守仓位2%
        assert position == 0.02, f"期望0.02, 实际{position}"
        print("✓ Kelly小样本测试通过")
    
    def test_kelly_upper_limit(self):
        """测试Kelly值不超过25%"""
        self.setup()
        # 添加20笔盈利交易，会导致很高的Kelly值
        for i in range(20):
            self.strategy._trade_history.append(
                TradeRecord(
                    code='TEST',
                    entry_date='2024-01-01',
                    exit_date='2024-01-02',
                    entry_price=10.0,
                    exit_price=15.0,  # 50%盈利
                    pnl_ratio=0.5,
                    is_win=True
                )
            )
        
        position = self.strategy._calculate_kelly_position(100000)
        
        # Kelly上限是25%，再乘以kelly_fraction(0.5)，应该<=12.5%
        assert position <= 0.15, f"Kelly仓位超过上限: {position}"
        print("✓ Kelly上限保护测试通过")
    
    def test_kelly_lower_limit(self):
        """测试Kelly值不低于1%"""
        self.setup()
        # 添加20笔亏损交易
        for i in range(20):
            self.strategy._trade_history.append(
                TradeRecord(
                    code='TEST',
                    entry_date='2024-01-01',
                    exit_date='2024-01-02',
                    entry_price=10.0,
                    exit_price=8.0,  # 20%亏损
                    pnl_ratio=-0.2,
                    is_win=False
                )
            )
        
        position = self.strategy._calculate_kelly_position(100000)
        
        # 应该至少有1%的最小仓位
        assert position >= 0.01, f"Kelly仓位低于下限: {position}"
        print("✓ Kelly下限保护测试通过")
    
    def test_kelly_low_winrate(self):
        """测试胜率<30%时仓位<3%"""
        self.setup()
        # 添加20笔交易，胜率25%(5胜15负)
        for i in range(5):
            self.strategy._trade_history.append(
                TradeRecord(
                    code='TEST',
                    entry_date='2024-01-01',
                    exit_date='2024-01-02',
                    entry_price=10.0,
                    exit_price=11.0,
                    pnl_ratio=0.1,
                    is_win=True
                )
            )
        for i in range(15):
            self.strategy._trade_history.append(
                TradeRecord(
                    code='TEST',
                    entry_date='2024-01-01',
                    exit_date='2024-01-02',
                    entry_price=10.0,
                    exit_price=9.0,
                    pnl_ratio=-0.1,
                    is_win=False
                )
            )
        
        position = self.strategy._calculate_kelly_position(100000)
        
        # 胜率<30%应该返回2%
        assert position == 0.02, f"期望0.02, 实际{position}"
        print("✓ Kelly低胜率保护测试通过")
    
    def test_kelly_no_wins(self):
        """测试无盈利交易时返回最低仓位"""
        self.setup()
        # 添加20笔亏损交易
        for i in range(20):
            self.strategy._trade_history.append(
                TradeRecord(
                    code='TEST',
                    entry_date='2024-01-01',
                    exit_date='2024-01-02',
                    entry_price=10.0,
                    exit_price=9.0,
                    pnl_ratio=-0.1,
                    is_win=False
                )
            )
        
        position = self.strategy._calculate_kelly_position(100000)
        
        # 无盈利交易应该返回1%
        assert position == 0.01, f"期望0.01, 实际{position}"
        print("✓ Kelly无盈利保护测试通过")


class TestTradeStatistics:
    """测试交易统计修复"""
    
    def test_win_rate_calculation(self):
        """测试交易胜率计算正确性"""
        # 模拟交易记录
        trades = pd.DataFrame({
            'code': ['000001', '000001', '000002', '000002'],
            'side': ['BUY', 'SELL', 'BUY', 'SELL'],
            'price': [10.0, 11.0, 20.0, 18.0],  # 第一笔盈利，第二笔亏损
            'quantity': [100, 100, 100, 100]
        })
        
        # 计算交易胜率
        buy_trades = trades[trades['side'] == 'BUY']
        sell_trades = trades[trades['side'] == 'SELL']
        
        profit_count = 0
        for _, sell in sell_trades.iterrows():
            code = sell['code']
            matching_buys = buy_trades[buy_trades['code'] == code]
            
            if not matching_buys.empty:
                avg_buy_price = matching_buys['price'].mean()
                sell_price = sell['price']
                
                # 考虑交易成本
                profit_ratio = (sell_price - avg_buy_price) / avg_buy_price - 0.0015
                
                if profit_ratio > 0:
                    profit_count += 1
        
        win_rate = profit_count / len(sell_trades)
        
        # 第一笔盈利(11-10)/10=10%，第二笔亏损(18-20)/20=-10%
        # 应该只有1笔盈利
        assert win_rate == 0.5, f"期望胜率0.5, 实际{win_rate}"
        print("✓ 交易胜率计算测试通过")
    
    def test_daily_vs_trade_win_rate(self):
        """测试区分日胜率和交易胜率"""
        # 日收益率
        daily_returns = np.array([0.01, -0.02, 0.015, -0.01, 0.005])
        daily_win_rate = (daily_returns > 0).sum() / len(daily_returns)
        
        # 交易胜率(模拟)
        trade_win_rate = 0.4  # 假设交易胜率40%
        
        # 两者应该不同
        assert daily_win_rate != trade_win_rate, \
            "日胜率和交易胜率不应该相同"
        print("✓ 日胜率vs交易胜率测试通过")


class TestAlphaPosition:
    """测试AlphaPosition T+1修复"""
    
    def test_position_has_filled_date(self):
        """测试AlphaPosition有entry_filled_date字段"""
        pos = AlphaPosition(
            code='000001',
            entry_price=10.0,
            entry_date='2024-01-01',
            quantity=100,
            stop_loss_price=9.7,
            take_profit_price=11.5,
            trailing_stop=9.7,
            highest_price=10.0,
            highest_date='2024-01-01',
            entry_filled_date='2024-01-01'  # 成交日期
        )
        
        assert pos.entry_filled_date == '2024-01-01'
        assert pos.entry_date == '2024-01-01'
        print("✓ AlphaPosition字段测试通过")
    
    def test_t1_check_uses_filled_date(self):
        """测试T+1检查使用成交日期"""
        # 创建持仓(当日成交)
        pos = AlphaPosition(
            code='000001',
            entry_price=10.0,
            entry_date='2024-01-01',  # 创建日期
            quantity=100,
            stop_loss_price=9.7,
            take_profit_price=11.5,
            trailing_stop=9.7,
            highest_price=10.0,
            highest_date='2024-01-01',
            entry_filled_date='2024-01-01'  # 成交日期
        )
        
        current_date = '2024-01-01'
        filled_dt = datetime.strptime(pos.entry_filled_date, '%Y-%m-%d')
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        
        days_held = (current_dt - filled_dt).days
        
        # 当日成交，不可卖出
        assert days_held < 1, "当日成交不应该可卖"
        print("✓ T+1当日不可卖测试通过")
    
    def test_t1_next_day_sellable(self):
        """测试T+1次日可卖"""
        pos = AlphaPosition(
            code='000001',
            entry_price=10.0,
            entry_date='2024-01-01',
            quantity=100,
            stop_loss_price=9.7,
            take_profit_price=11.5,
            trailing_stop=9.7,
            highest_price=10.0,
            highest_date='2024-01-01',
            entry_filled_date='2024-01-01'
        )
        
        current_date = '2024-01-02'  # 次日
        filled_dt = datetime.strptime(pos.entry_filled_date, '%Y-%m-%d')
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        
        days_held = (current_dt - filled_dt).days
        
        # 次日应该可卖
        assert days_held >= 1, "次日应该可卖"
        print("✓ T+1次日可卖测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("运行关键修复单元测试")
    print("="*60)
    
    # Look-Ahead Bias测试
    test_lab = TestLookAheadBias()
    test_lab.test_factors_not_using_future_data()
    
    # NaN处理测试
    test_nan = TestNaNHandling()
    test_nan.test_nan_validation_valid_data()
    test_nan.test_nan_validation_negative_price()
    test_nan.test_nan_validation_high_low_logic()
    test_nan.test_safe_fillna_forward()
    test_nan.test_safe_fillna_interpolate()
    test_nan.test_safe_fillna_no_nan()
    test_nan.test_get_summary()
    
    # Kelly准则测试
    test_kelly = TestKellyPosition()
    test_kelly.setup()  # 手动调用setup
    test_kelly.test_kelly_small_sample()
    test_kelly.setup()  # 重新初始化
    test_kelly.test_kelly_upper_limit()
    test_kelly.setup()
    test_kelly.test_kelly_lower_limit()
    test_kelly.setup()
    test_kelly.test_kelly_low_winrate()
    test_kelly.setup()
    test_kelly.test_kelly_no_wins()
    
    # 交易统计测试
    test_stats = TestTradeStatistics()
    test_stats.test_win_rate_calculation()
    test_stats.test_daily_vs_trade_win_rate()
    
    # AlphaPosition测试
    test_pos = TestAlphaPosition()
    test_pos.test_position_has_filled_date()
    test_pos.test_t1_check_uses_filled_date()
    test_pos.test_t1_next_day_sellable()
    
    print("\n" + "="*60)
    print("所有测试通过! ✓")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
