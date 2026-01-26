# ============================================================================
# 文件: tests/test_engine.py
# ============================================================================
"""
引擎模块测试
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.matcher import MatchEngine, Order, OrderStatus
from engine.portfolio import PortfolioManager, Position


class TestMatchEngine:
    """撮合引擎测试"""
    
    @pytest.fixture
    def engine(self):
        return MatchEngine()
    
    @pytest.fixture
    def market_data(self):
        return pd.Series({
            'open': 10.0,
            'high': 10.5,
            'low': 9.5,
            'close': 10.2,
            'is_limit_up': False,
            'is_limit_down': False
        })
    
    def test_create_order(self, engine):
        """测试创建订单"""
        order = engine.create_order(
            code='000001',
            side='BUY',
            price=10.0,
            quantity=100,
            create_date='2024-01-01'
        )
        
        assert order.code == '000001'
        assert order.side == 'BUY'
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
    
    def test_match_buy_order(self, engine, market_data):
        """测试买入撮合"""
        order = engine.create_order('000001', 'BUY', 10.0, 100, '2024-01-01')
        matched = engine.match(order, market_data, None, '2024-01-02')
        
        assert matched.status == OrderStatus.FILLED
        assert matched.filled_quantity == 100
        assert matched.filled_price > 0
        assert matched.commission > 0
    
    def test_match_limit_up_rejected(self, engine):
        """测试涨停无法买入"""
        market_data = pd.Series({
            'open': 10.0,
            'is_limit_up': True,
            'is_limit_down': False
        })
        
        order = engine.create_order('000001', 'BUY', 10.0, 100, '2024-01-01')
        matched = engine.match(order, market_data, None, '2024-01-02')
        
        assert matched.status == OrderStatus.REJECTED
        assert '涨停' in matched.reject_reason
    
    def test_match_t1_rejected(self, engine, market_data):
        """测试 T+1 限制"""
        position = Position(
            code='000001',
            quantity=100,
            avg_cost=10.0,
            buy_date='2024-01-02'  # 同一天买入
        )
        
        order = engine.create_order('000001', 'SELL', 10.0, 100, '2024-01-02')
        matched = engine.match(order, market_data, position, '2024-01-02')
        
        assert matched.status == OrderStatus.REJECTED
        assert 'T+1' in matched.reject_reason


class TestPortfolioManager:
    """持仓管理器测试"""
    
    @pytest.fixture
    def portfolio(self):
        return PortfolioManager(initial_capital=1_000_000)
    
    def test_initial_state(self, portfolio):
        """测试初始状态"""
        assert portfolio.cash == 1_000_000
        assert portfolio.total_equity == 1_000_000
        assert portfolio.positions_count == 0
    
    def test_apply_buy_order(self, portfolio):
        """测试应用买入订单"""
        order = Order(
            order_id='001',
            code='000001',
            side='BUY',
            price=10.0,
            quantity=100,
            create_date='2024-01-01',
            status=OrderStatus.FILLED,
            filled_price=10.01,
            filled_quantity=100,
            commission=5.0
        )
        
        portfolio.apply_order(order, '2024-01-01')
        
        assert '000001' in portfolio.positions
        assert portfolio.positions['000001'].quantity == 100
        assert portfolio.cash < 1_000_000
    
    def test_apply_sell_order(self, portfolio):
        """测试应用卖出订单"""
        # 先买入
        buy_order = Order(
            order_id='001', code='000001', side='BUY', price=10.0,
            quantity=100, create_date='2024-01-01',
            status=OrderStatus.FILLED, filled_price=10.0,
            filled_quantity=100, commission=5.0
        )
        portfolio.apply_order(buy_order, '2024-01-01')
        
        # 再卖出
        sell_order = Order(
            order_id='002', code='000001', side='SELL', price=11.0,
            quantity=100, create_date='2024-01-02',
            status=OrderStatus.FILLED, filled_price=11.0,
            filled_quantity=100, commission=5.0, stamp_duty=11.0
        )
        portfolio.apply_order(sell_order, '2024-01-02')
        
        assert '000001' not in portfolio.positions
    
    def test_get_weight(self, portfolio):
        """测试权重计算"""
        # 买入后检查权重
        order = Order(
            order_id='001', code='000001', side='BUY', price=10.0,
            quantity=1000, create_date='2024-01-01',
            status=OrderStatus.FILLED, filled_price=10.0,
            filled_quantity=1000, commission=5.0
        )
        portfolio.apply_order(order, '2024-01-01')
        
        # 更新市值
        market_data = pd.DataFrame({'code': ['000001'], 'close': [10.0]})
        portfolio.update_market_value(market_data)
        
        weight = portfolio.get_weight('000001')
        assert 0 < weight < 1


class TestPosition:
    """持仓对象测试"""
    
    def test_update_market_value(self):
        """测试市值更新"""
        pos = Position(
            code='000001',
            quantity=100,
            avg_cost=10.0,
            buy_date='2024-01-01'
        )
        
        pos.update_market_value(11.0)
        
        assert pos.market_value == 1100
        assert pos.unrealized_pnl == 100
    
    def test_pnl_ratio(self):
        """测试盈亏比例"""
        pos = Position(code='000001', quantity=100, avg_cost=10.0, buy_date='2024-01-01')
        pos.update_market_value(12.0)
        
        assert pos.pnl_ratio == pytest.approx(0.2, rel=0.01)