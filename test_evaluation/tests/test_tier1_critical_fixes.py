# ============================================================================
# File: tests/test_tier1_critical_fixes.py
# ============================================================================
"""
Tier 1 致命问题修复测试
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.trading_calendar import TradingCalendar
from engine.slippage_model import AdvancedSlippageModel, AlmgrenChrissModel, LiquidityProfile
from engine.matcher import MatchEngine, Order, OrderStatus
from engine.portfolio import PortfolioManager, Position


class TestTradingCalendar:
    """交易日历测试"""
    
    def test_is_trading_day_weekend(self):
        """测试周末过滤"""
        calendar = TradingCalendar()
        
        # 周六
        assert not calendar.is_trading_day_str('2024-01-06')
        # 周日
        assert not calendar.is_trading_day_str('2024-01-07')
        # 周一
        assert calendar.is_trading_day_str('2024-01-08')
    
    def test_is_trading_day_holiday(self):
        """测试节假日过滤"""
        calendar = TradingCalendar()
        
        # 2024年春节假期
        assert not calendar.is_trading_day_str('2024-02-10')  # 春节初一
        assert not calendar.is_trading_day_str('2024-02-11')
        assert calendar.is_trading_day_str('2024-02-08')  # 节前
    
    def test_get_trading_days(self):
        """测试获取交易日列表"""
        calendar = TradingCalendar()
        
        trading_days = calendar.get_trading_days('2024-01-01', '2024-01-31')
        
        # 应该过滤掉周末
        assert all(calendar.is_trading_day_str(d) for d in trading_days)
        assert len(trading_days) > 15  # 1月应该有至少15个交易日
    
    def test_offset_trading_day(self):
        """测试交易日偏移"""
        calendar = TradingCalendar()
        
        # 正向偏移
        next_day = calendar.offset_trading_day('2024-01-08', 1)
        assert calendar.is_trading_day_str(next_day)
        
        # 负向偏移
        prev_day = calendar.offset_trading_day('2024-01-08', -1)
        assert calendar.is_trading_day_str(prev_day)
    
    def test_validate_rebalance_dates(self):
        """测试调仓日验证"""
        calendar = TradingCalendar()
        
        # 混合交易日和非交易日
        test_dates = ['2024-01-08', '2024-01-06', '2024-01-09']
        valid, invalid = calendar.validate_rebalance_dates(test_dates)
        
        assert '2024-01-06' not in valid  # 周六
        assert '2024-01-08' in valid
        assert '2024-01-09' in valid
        assert len(invalid) == 1


class TestAdvancedSlippageModel:
    """高级滑点模型测试"""
    
    def test_almgren_chriss_impact(self):
        """测试 Almgren-Chriss 市场冲击"""
        model = AlmgrenChrissModel(gamma=0.0001, eta=0.0005)
        
        impact = model.calculate_market_impact(
            order_quantity=10000,
            avg_daily_volume=1_000_000,
            price=10.0,
            side='BUY'
        )
        
        assert impact['permanent_impact'] > 0
        assert impact['temporary_impact'] > 0
        assert impact['total_impact'] > 0
        assert impact['impact_bps'] > 0
    
    def test_large_order_impact(self):
        """测试大单冲击"""
        model = AlmgrenChrissModel()
        
        # 小单
        small_impact = model.calculate_market_impact(
            order_quantity=1000,
            avg_daily_volume=1_000_000,
            price=10.0,
            side='BUY'
        )
        
        # 大单
        large_impact = model.calculate_market_impact(
            order_quantity=100000,
            avg_daily_volume=1_000_000,
            price=10.0,
            side='BUY'
        )
        
        # 大单冲击应该更大
        assert large_impact['impact_bps'] > small_impact['impact_bps'] * 10
    
    def test_liquidity_profile(self):
        """测试流动性特征估算"""
        model = AdvancedSlippageModel()
        
        # 创建模拟数据
        historical_data = pd.DataFrame({
            'open': [10.0, 10.1, 10.2],
            'high': [10.5, 10.6, 10.7],
            'low': [9.5, 9.6, 9.7],
            'close': [10.2, 10.3, 10.4],
            'volume': [1_000_000, 1_100_000, 1_200_000]
        })
        
        profile = model.estimate_liquidity_profile(historical_data)
        
        assert profile.avg_daily_volume > 0
        assert profile.volatility > 0
        assert profile.liquidity_score > 0
        assert profile.liquidity_score <= 100
    
    def test_slippage_calculation(self):
        """测试滑点计算"""
        model = AdvancedSlippageModel(base_slippage_rate=0.0001)
        
        slippage = model.calculate_slippage(
            order_quantity=10000,
            price=10.0,
            side='BUY',
            daily_volume=1_000_000
        )
        
        assert isinstance(slippage, float)
        assert slippage > 0  # 买入滑点为正
    
    def test_slippage_sell(self):
        """测试卖出滑点"""
        model = AdvancedSlippageModel()
        
        buy_slippage = model.calculate_slippage(10000, 10.0, 'BUY', 1_000_000)
        sell_slippage = model.calculate_slippage(10000, 10.0, 'SELL', 1_000_000)
        
        # 买入滑点为正，卖出滑点为负
        assert buy_slippage > 0
        assert sell_slippage < 0


class TestPartialFill:
    """部分成交测试"""
    
    def test_order_partial_fill_update(self):
        """测试订单部分成交更新"""
        order = Order(
            order_id='001',
            code='000001',
            side='BUY',
            price=10.0,
            quantity=10000,
            create_date='2024-01-01'
        )
        
        # 第一次部分成交
        order.update_partial_fill(5000, 10.01)
        
        assert order.filled_quantity == 5000
        assert order.status == OrderStatus.PARTIAL
        assert order.is_partial_fill
        assert order.fill_ratio == 0.5
        
        # 第二次部分成交 (完成)
        order.update_partial_fill(5000, 10.02)
        
        assert order.filled_quantity == 10000
        assert order.status == OrderStatus.FILLED
    
    def test_matcher_partial_fill_large_order(self):
        """测试大单部分成交"""
        engine = MatchEngine()
        
        market_data = pd.Series({
            'open': 10.0,
            'high': 10.5,
            'low': 9.5,
            'close': 10.2,
            'volume': 100000,  # 日成交10万
            'is_limit_up': False,
            'is_limit_down': False
        })
        
        # 大单 (超过5%日成交)
        order = engine.create_order('000001', 'BUY', 10.0, 10000, '2024-01-01')
        matched = engine.match(order, market_data, None, '2024-01-02')
        
        # 应该部分成交
        assert matched.filled_quantity > 0
        assert matched.filled_quantity < order.quantity or matched.status == OrderStatus.PARTIAL


class TestTPlusOneSettlement:
    """T+1 清算测试"""
    
    def test_t_plus1_sell_restriction(self):
        """测试T+1卖出限制"""
        engine = MatchEngine()
        
        market_data = pd.Series({
            'open': 10.0,
            'high': 10.5,
            'low': 9.5,
            'close': 10.2,
            'is_limit_up': False,
            'is_limit_down': False
        })
        
        # 持仓 (今天买入)
        position = Position(
            code='000001',
            quantity=1000,
            avg_cost=10.0,
            buy_date='2024-01-02'  # 同一天
        )
        
        order = engine.create_order('000001', 'SELL', 10.0, 100, '2024-01-02')
        matched = engine.match(order, market_data, position, '2024-01-02')
        
        # 应该被拒绝
        assert matched.status == OrderStatus.REJECTED
        assert 'T+1' in matched.reject_reason
    
    def test_t_plus1_sell_next_day(self):
        """测试T+1次日可卖"""
        engine = MatchEngine()
        
        market_data = pd.Series({
            'open': 10.0,
            'high': 10.5,
            'low': 9.5,
            'close': 10.2,
            'is_limit_up': False,
            'is_limit_down': False
        })
        
        # 持仓 (昨天买入)
        position = Position(
            code='000001',
            quantity=1000,
            avg_cost=10.0,
            buy_date='2024-01-01'  # 前一天
        )
        
        order = engine.create_order('000001', 'SELL', 10.0, 100, '2024-01-02')
        matched = engine.match(order, market_data, position, '2024-01-02')
        
        # 应该成交
        assert matched.status == OrderStatus.FILLED
    
    def test_portfolio_apply_partial_fill(self):
        """测试持仓管理器处理部分成交"""
        portfolio = PortfolioManager(initial_capital=1_000_000)
        
        # 部分成交的订单
        order = Order(
            order_id='001',
            code='000001',
            side='BUY',
            price=10.0,
            quantity=10000,
            create_date='2024-01-01',
            status=OrderStatus.PARTIAL,
            filled_quantity=5000,
            filled_price=10.01,
            commission=15.0,
            total_cost=25.0
        )
        
        portfolio.apply_order(order, '2024-01-01')
        
        assert '000001' in portfolio.positions
        assert portfolio.positions['000001'].quantity == 5000
        assert portfolio.cash < 1_000_000  # 现金减少


class TestLimitUpDown:
    """涨跌停测试"""
    
    def test_limit_up_reject_buy(self):
        """测试涨停无法买入"""
        engine = MatchEngine()
        
        market_data = pd.Series({
            'open': 10.0,
            'is_limit_up': True,
            'is_limit_down': False
        })
        
        order = engine.create_order('000001', 'BUY', 10.0, 100, '2024-01-01')
        matched = engine.match(order, market_data, None, '2024-01-02')
        
        assert matched.status == OrderStatus.REJECTED
        assert '涨停' in matched.reject_reason
    
    def test_limit_down_reject_sell(self):
        """测试跌停无法卖出"""
        engine = MatchEngine()
        
        market_data = pd.Series({
            'open': 10.0,
            'is_limit_up': False,
            'is_limit_down': True
        })
        
        position = Position(
            code='000001',
            quantity=1000,
            avg_cost=10.0,
            buy_date='2024-01-01'
        )
        
        order = engine.create_order('000001', 'SELL', 10.0, 100, '2024-01-02')
        matched = engine.match(order, market_data, position, '2024-01-02')
        
        assert matched.status == OrderStatus.REJECTED
        assert '跌停' in matched.reject_reason


class TestIntegration:
    """集成测试"""
    
    def test_backtest_with_calendar(self):
        """测试回测引擎集成交易日历"""
        from engine.backtest import BacktestEngine
        
        engine = BacktestEngine()
        
        # 验证交易日历已初始化
        assert hasattr(engine, 'calendar')
        assert isinstance(engine.calendar, TradingCalendar)
    
    def test_matcher_with_advanced_slippage(self):
        """测试撮合引擎集成高级滑点"""
        engine = MatchEngine(use_advanced_slippage=True)
        
        # 验证高级滑点模型已初始化
        assert hasattr(engine, 'advanced_slippage')
        assert isinstance(engine.advanced_slippage, AdvancedSlippageModel)
    
    def test_portfolio_with_settlement(self):
        """测试持仓管理器支持T+1清算"""
        portfolio = PortfolioManager()
        
        # 验证T+1数据结构已初始化
        assert hasattr(portfolio, 'pending_cash')
        assert hasattr(portfolio, 'pending_positions')
        assert hasattr(portfolio, 'process_settlement')
