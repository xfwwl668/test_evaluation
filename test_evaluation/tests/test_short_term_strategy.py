# ============================================================================
# 文件: tests/test_short_term_strategy.py
# ============================================================================
"""
短线策略单元测试
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from factors.technical.rsrs_advanced import RSRSAdvancedFactor, RSRSMultiPeriodFactor
from strategy.short_term_strategy import ShortTermRSRSStrategy, PositionState
from engine.risk import PositionSizer, RiskManager, DrawdownProtector


class TestRSRSAdvancedFactor:
    """修正版 RSRS 因子测试"""
    
    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        np.random.seed(42)
        n = 700
        dates = pd.bdate_range('2020-01-01', periods=n)
        
        # 模拟趋势
        trend = np.cumsum(np.random.randn(n) * 0.01)
        price = 20 * np.exp(trend)
        
        return pd.DataFrame({
            'open': price * (1 + np.random.randn(n) * 0.005),
            'high': price * (1 + np.abs(np.random.randn(n)) * 0.015),
            'low': price * (1 - np.abs(np.random.randn(n)) * 0.015),
            'close': price,
            'vol': np.random.randint(1e5, 1e7, n).astype(float),
            'amount': (price * np.random.randint(1e5, 1e7, n)).astype(float)
        }, index=dates)
    
    def test_compute_all(self, sample_data):
        """测试完整计算"""
        factor = RSRSAdvancedFactor(window=18, std_window=200)
        result = factor.compute_all(sample_data)
        
        assert 'rsrs_slope' in result.columns
        assert 'rsrs_r2' in result.columns
        assert 'rsrs_zscore' in result.columns
        assert 'rsrs_score' in result.columns
        assert 'rsrs_valid' in result.columns
        
        # 最后应有有效值
        assert not pd.isna(result['rsrs_score'].iloc[-1])
    
    def test_r2_range(self, sample_data):
        """测试 R² 范围"""
        factor = RSRSAdvancedFactor()
        result = factor.compute_all(sample_data)
        
        r2 = result['rsrs_r2'].dropna()
        assert (r2 >= 0).all()
        assert (r2 <= 1).all()
    
    def test_skew_correction(self, sample_data):
        """测试偏度修正"""
        # 添加极端值
        sample_data.loc[sample_data.index[-10:], 'high'] *= 1.5
        
        factor_with_skew = RSRSAdvancedFactor(skew_penalty_factor=0.1)
        factor_no_skew = RSRSAdvancedFactor(skew_penalty_factor=0.0)
        
        result_with = factor_with_skew.compute_all(sample_data)
        result_without = factor_no_skew.compute_all(sample_data)
        
        # 有偏度惩罚的分数应该更低
        # (在正分数区域)
        last_with = result_with['rsrs_score'].iloc[-1]
        last_without = result_without['rsrs_score'].iloc[-1]
        
        if last_without > 0:
            assert last_with <= last_without


class TestShortTermStrategy:
    """短线策略测试"""
    
    @pytest.fixture
    def strategy(self):
        return ShortTermRSRSStrategy(params={
            'rsrs_entry_threshold': 0.5,  # 降低阈值便于测试
            'r2_threshold': 0.7,
            'volume_multiplier': 1.2,
            'max_positions': 5
        })
    
    def test_initialization(self, strategy):
        """测试初始化"""
        assert strategy.name == "short_term_rsrs"
        assert strategy.get_param('rsrs_entry_threshold') == 0.5
        assert strategy.get_param('max_positions') == 5
    
    def test_position_state(self):
        """测试持仓状态"""
        state = PositionState(
            code='000001',
            entry_price=10.0,
            entry_date='2024-01-01',
            quantity=100,
            highest_price=10.0,
            stop_loss_price=9.7,
            trailing_stop_price=9.7,
            atr_at_entry=0.2
        )
        
        # 更新移动止盈
        state.update_trailing_stop(11.0, 0.2, 2.0)
        
        assert state.highest_price == 11.0
        assert state.trailing_stop_price > 9.7  # 止盈价上移
    
    def test_trailing_stop_only_moves_up(self):
        """测试移动止盈只能上移"""
        state = PositionState(
            code='000001',
            entry_price=10.0,
            entry_date='2024-01-01',
            quantity=100,
            highest_price=12.0,
            stop_loss_price=9.7,
            trailing_stop_price=11.0,
            atr_at_entry=0.2
        )
        
        # 价格回落
        state.update_trailing_stop(11.5, 0.2, 2.0)
        
        # 止盈价不应下移
        assert state.trailing_stop_price >= 11.0


class TestPositionSizer:
    """仓位计算器测试"""
    
    def test_basic_calculation(self):
        """测试基础计算"""
        sizer = PositionSizer(risk_per_trade=0.005)
        
        weight, shares = sizer.calculate(
            total_equity=1_000_000,
            entry_price=10.0,
            atr=0.5  # 5% ATR
        )
        
        assert 0 < weight <= 0.10
        assert shares > 0
        assert shares % 100 == 0  # 整百股
    
    def test_high_volatility_reduces_position(self):
        """测试高波动率降低仓位"""
        sizer = PositionSizer()
        
        weight_low_vol, _ = sizer.calculate(1e6, 10.0, 0.02, volatility=0.15)
        weight_high_vol, _ = sizer.calculate(1e6, 10.0, 0.02, volatility=0.50)
        
        assert weight_high_vol < weight_low_vol
    
    def test_position_limits(self):
        """测试仓位限制"""
        sizer = PositionSizer(
            risk_per_trade=0.05,  # 高风险
            max_position_pct=0.10
        )
        
        weight, _ = sizer.calculate(1e6, 10.0, 0.01)
        
        assert weight <= 0.10  # 不超过最大限制


class TestRiskManager:
    """风险管理器测试"""
    
    def test_position_limits_check(self):
        """测试仓位限制检查"""
        rm = RiskManager(max_single_weight=0.10, max_total_weight=0.80)
        
        current = {'A': 0.10, 'B': 0.10}
        
        # 正常情况
        allowed, _ = rm.check_position_limits(0.05, current)
        assert allowed
        
        # 超过单只
        allowed, reason = rm.check_position_limits(0.15, current)
        assert not allowed
        assert '单只' in reason
        
        # 超过总仓位
        current = {'A': 0.20, 'B': 0.20, 'C': 0.20, 'D': 0.15}
        allowed, reason = rm.check_position_limits(0.10, current)
        assert not allowed
        assert '总仓位' in reason
    
    def test_risk_metrics(self):
        """测试风险指标计算"""
        rm = RiskManager()
        
        positions = {
            'A': {'weight': 0.1, 'volatility': 0.3},
            'B': {'weight': 0.1, 'volatility': 0.2},
            'C': {'weight': 0.1, 'volatility': 0.25}
        }
        
        metrics = rm.calculate_risk_metrics(positions)
        
        assert metrics.position_count == 3
        assert 0 < metrics.total_exposure <= 1
        assert metrics.var_1d > 0
    
    def test_emergency_stop(self):
        """测试紧急熔断"""
        rm = RiskManager()
        
        # 正常情况
        stop, _ = rm.emergency_stop(-0.02, 0.05, 0.2)
        assert not stop
        
        # 暴跌
        stop, reason = rm.emergency_stop(-0.06, 0.05, 0.2)
        assert stop
        assert '暴跌' in reason
        
        # 回撤过大
        stop, reason = rm.emergency_stop(-0.01, 0.20, 0.2)
        assert stop
        assert '回撤' in reason


class TestDrawdownProtector:
    """回撤保护器测试"""
    
    def test_position_multiplier(self):
        """测试仓位乘数"""
        protector = DrawdownProtector(
            warning_level=0.05,
            reduce_level=0.08,
            stop_level=0.12
        )
        
        # 无回撤
        mult = protector.get_position_multiplier(0.02)
        assert mult == 1.0
        
        # 警告区间
        mult = protector.get_position_multiplier(0.06)
        assert 0.5 < mult < 1.0
        
        # 减仓区间
        mult = protector.get_position_multiplier(0.10)
        assert mult == 0.5
        
        # 停止区间
        mult = protector.get_position_multiplier(0.15)
        assert mult == 0.0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])