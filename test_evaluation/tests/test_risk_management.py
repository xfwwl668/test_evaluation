# ============================================================================
# File: tests/test_risk_management.py
# ============================================================================
"""
风险管理测试
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.risk import RiskManager, PositionSizer, DrawdownProtector


class TestPositionSizer:
    """仓位计算器测试"""
    
    def test_calculate_position_size(self):
        """测试仓位计算"""
        sizer = PositionSizer(
            risk_per_trade=0.01,
            atr_multiplier=2.0
        )
        
        weight, shares = sizer.calculate(
            total_equity=1_000_000,
            entry_price=10.0,
            atr=0.5
        )
        
        assert 0 < weight <= 1
        assert shares > 0
        assert shares % 100 == 0  # 整百股
    
    def test_volatility_adjustment(self):
        """测试波动率调整"""
        sizer = PositionSizer()
        
        # 低波动
        weight_low, _ = sizer.calculate(
            total_equity=1_000_000,
            entry_price=10.0,
            atr=0.5,
            volatility=0.1
        )
        
        # 高波动
        weight_high, _ = sizer.calculate(
            total_equity=1_000_000,
            entry_price=10.0,
            atr=0.5,
            volatility=0.5
        )
        
        # 高波动时仓位应该更小
        assert weight_high < weight_low
    
    def test_max_position_limit(self):
        """测试最大仓位限制"""
        sizer = PositionSizer(max_position_pct=0.10)
        
        weight, _ = sizer.calculate(
            total_equity=1_000_000,
            entry_price=1.0,
            atr=0.01,
            volatility=0.01  # 极低波动
        )
        
        # 不应超过最大仓位
        assert weight <= 0.10
    
    def test_batch_calculate(self):
        """测试批量计算"""
        sizer = PositionSizer()
        
        candidates = [
            {'code': '000001', 'price': 10.0, 'atr': 0.5, 'volatility': 0.2},
            {'code': '000002', 'price': 20.0, 'atr': 1.0, 'volatility': 0.3},
        ]
        
        results = sizer.calculate_batch(1_000_000, candidates)
        
        assert len(results) == 2
        assert '000001' in results
        assert '000002' in results


class TestRiskManager:
    """风险管理器测试"""
    
    def test_check_position_limits(self):
        """测试仓位限制检查"""
        manager = RiskManager(max_single_weight=0.10, max_total_weight=0.80)
        
        # 正常仓位
        allowed, reason = manager.check_position_limits(0.08, {})
        assert allowed
        
        # 超过单只限制
        allowed, reason = manager.check_position_limits(0.15, {})
        assert not allowed
        assert '单只限制' in reason
    
    def test_check_total_weight_limit(self):
        """测试总仓位限制"""
        manager = RiskManager(max_total_weight=0.70)
        
        # 接近上限
        allowed, reason = manager.check_position_limits(0.10, {'000001': 0.65})
        assert not allowed
        assert '总仓位限制' in reason
    
    def test_calculate_risk_metrics(self):
        """测试风险指标计算"""
        manager = RiskManager()
        
        positions = {
            '000001': {'weight': 0.10, 'volatility': 0.25},
            '000002': {'weight': 0.15, 'volatility': 0.30},
            '000003': {'weight': 0.20, 'volatility': 0.20},
        }
        
        metrics = manager.calculate_risk_metrics(positions)
        
        assert metrics.total_exposure == pytest.approx(0.45, rel=0.01)
        assert metrics.single_max_exposure == 0.20
        assert metrics.var_1d > 0
        assert metrics.position_count == 3
    
    def test_emergency_stop(self):
        """测试极端行情保护"""
        manager = RiskManager()
        
        # 正常情况
        stopped, reason = manager.emergency_stop(0.01, 0.05, 0.15)
        assert not stopped
        
        # 单日暴跌
        stopped, reason = manager.emergency_stop(-0.06, 0.05, 0.15)
        assert stopped
        assert '暴跌' in reason
        
        # 回撤过大
        stopped, reason = manager.emergency_stop(0.01, 0.16, 0.15)
        assert stopped
        assert '回撤' in reason
        
        # 波动率异常
        stopped, reason = manager.emergency_stop(0.01, 0.05, 0.6)
        assert stopped
        assert '波动率' in reason
    
    def test_adjust_for_correlation(self):
        """测试相关性调整"""
        manager = RiskManager(max_correlation=0.7)
        
        weights = {
            '000001': 0.15,
            '000002': 0.15,
            '000003': 0.15
        }
        
        # 创建相关性矩阵 (高相关)
        corr_matrix = pd.DataFrame(
            [[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]],
            index=['000001', '000002', '000003'],
            columns=['000001', '000002', '000003']
        )
        
        adjusted = manager.adjust_for_correlation(weights, corr_matrix)
        
        # 高相关的应该被降低
        assert adjusted['000001'] < weights['000001'] or \
               adjusted['000002'] < weights['000002'] or \
               adjusted['000003'] < weights['000003']


class TestDrawdownProtector:
    """回撤保护器测试"""
    
    def test_normal_state(self):
        """测试正常状态"""
        protector = DrawdownProtector(
            warning_level=0.05,
            reduce_level=0.08,
            stop_level=0.12
        )
        
        multiplier = protector.get_position_multiplier(0.02)
        assert multiplier == pytest.approx(1.0, rel=0.1)
    
    def test_warning_level(self):
        """测试警告级别"""
        protector = DrawdownProtector(
            warning_level=0.05,
            reduce_level=0.08
        )
        
        # 警告级别 (5%-8%)
        multiplier = protector.get_position_multiplier(0.06)
        assert 0.5 < multiplier < 1.0
    
    def test_reduce_level(self):
        """测试减仓级别"""
        protector = DrawdownProtector(
            reduce_level=0.08,
            stop_level=0.12
        )
        
        # 减仓级别 (8%-12%)
        multiplier = protector.get_position_multiplier(0.10)
        assert multiplier == pytest.approx(0.5, rel=0.1)
    
    def test_stop_level(self):
        """测试停止级别"""
        protector = DrawdownProtector(stop_level=0.12)
        
        # 停止级别 (>12%)
        multiplier = protector.get_position_multiplier(0.15)
        assert multiplier == 0.0
    
    def test_recovery(self):
        """测试恢复"""
        protector = DrawdownProtector(
            warning_level=0.05,
            reduce_level=0.08,
            stop_level=0.12,
            recovery_level=0.03
        )
        
        # 触发保护
        multiplier = protector.get_position_multiplier(0.10)
        assert protector.is_protecting
        
        # 恢复
        multiplier = protector.get_position_multiplier(0.02)
        assert not protector.is_protecting
        assert multiplier == 1.0
