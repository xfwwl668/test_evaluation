# ============================================================================
# 文件: tests/test_factors.py
# ============================================================================
"""
因子模块测试
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from factors import FactorRegistry, BaseFactor
from factors.technical.rsrs import RSRSFactor, RSRSZScoreFactor
from factors.technical.momentum import MomentumFactor, OBVTrendFactor


class TestFactorRegistry:
    """因子注册表测试"""
    
    def test_list_all(self):
        """测试列出所有因子"""
        factors = FactorRegistry.list_all()
        assert len(factors) > 0
        assert 'rsrs_slope' in factors
        assert 'momentum' in factors
    
    def test_get_factor(self):
        """测试获取因子"""
        factor = FactorRegistry.get('rsrs_slope')
        assert isinstance(factor, BaseFactor)
        assert factor.name == 'rsrs_slope'
    
    def test_get_nonexistent(self):
        """测试获取不存在的因子"""
        with pytest.raises(KeyError):
            FactorRegistry.get('nonexistent_factor')


class TestRSRSFactor:
    """RSRS因子测试"""
    
    def test_compute(self, sample_ohlcv):
        """测试计算"""
        factor = RSRSFactor()
        result = factor.compute(sample_ohlcv)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        assert result.name == 'rsrs_slope'
    
    def test_lookback(self):
        """测试回溯期"""
        factor = RSRSFactor(window=18)
        assert factor.lookback >= 18
    
    def test_short_data(self, sample_ohlcv):
        """测试短数据"""
        short_df = sample_ohlcv.head(10)
        factor = RSRSFactor()
        result = factor.compute(short_df)
        
        assert result.isna().all()  # 数据不足应全为 NaN


class TestRSRSZScoreFactor:
    """RSRS Z-Score 因子测试"""
    
    def test_compute(self, sample_ohlcv):
        """测试计算"""
        factor = RSRSZScoreFactor()
        result = factor.compute(sample_ohlcv)
        
        assert isinstance(result, pd.Series)
        assert not result.iloc[-1:].isna().all()  # 最后应有值
    
    def test_zscore_range(self, sample_ohlcv):
        """测试 Z-Score 范围"""
        factor = RSRSZScoreFactor()
        result = factor.compute(sample_ohlcv)
        
        valid = result.dropna()
        # Z-Score 应在合理范围内
        assert valid.abs().max() < 10


class TestMomentumFactor:
    """动量因子测试"""
    
    def test_compute(self, sample_ohlcv):
        """测试计算"""
        factor = MomentumFactor(window=20)
        result = factor.compute(sample_ohlcv)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
    
    def test_different_windows(self, sample_ohlcv):
        """测试不同窗口"""
        factor_10 = MomentumFactor(window=10)
        factor_30 = MomentumFactor(window=30)
        
        result_10 = factor_10.compute(sample_ohlcv)
        result_30 = factor_30.compute(sample_ohlcv)
        
        # 不同窗口应产生不同结果
        assert not result_10.equals(result_30)


class TestOBVTrendFactor:
    """OBV趋势因子测试"""
    
    def test_compute(self, sample_ohlcv):
        """测试计算"""
        factor = OBVTrendFactor()
        result = factor.compute(sample_ohlcv)
        
        assert isinstance(result, pd.Series)
        assert set(result.dropna().unique()).issubset({0, 1})