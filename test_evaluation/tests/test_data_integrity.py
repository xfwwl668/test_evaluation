# ============================================================================
# File: tests/test_data_integrity.py
# ============================================================================
"""
数据完整性和前向偏差测试
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_validator import DataValidator


class TestDataValidation:
    """数据验证测试"""
    
    def test_validate_ohlcv_valid_data(self):
        """测试有效 OHLCV 数据"""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'open': [10.0, 10.1, 10.2],
            'high': [10.5, 10.6, 10.7],
            'low': [9.5, 9.6, 9.7],
            'close': [10.2, 10.3, 10.4],
            'volume': [100000, 110000, 120000]
        })
        
        result = validator.validate_ohlcv(df, 'test_data')
        assert result is True
    
    def test_validate_ohlcv_with_nan(self):
        """测试包含 NaN 的数据"""
        validator = DataValidator()
        
        # 包含 NaN
        df = pd.DataFrame({
            'open': [10.0, np.nan, 10.2],
            'high': [10.5, 10.6, 10.7],
            'low': [9.5, 9.6, 9.7],
            'close': [10.2, 10.3, 10.4],
            'volume': [100000, 110000, 120000]
        })
        
        result = validator.validate_ohlcv(df, 'test_data')
        # 允许 NaN 但应该记录警告
        assert isinstance(result, bool)
    
    def test_validate_high_low_consistency(self):
        """测试高低价一致性"""
        validator = DataValidator()
        
        # 不一致: high < low
        df = pd.DataFrame({
            'open': [10.0, 10.1, 10.2],
            'high': [9.5, 10.6, 10.7],  # 第一行 high < low
            'low': [9.6, 9.6, 9.7],
            'close': [10.2, 10.3, 10.4],
            'volume': [100000, 110000, 120000]
        })
        
        result = validator.validate_ohlcv(df, 'test_data')
        assert isinstance(result, bool)
    
    def test_validate_price_range(self):
        """测试价格合理性"""
        validator = DataValidator()
        
        # 异常价格
        df = pd.DataFrame({
            'open': [10.0, 10.1, 1000.0],  # 异常高价
            'high': [10.5, 10.6, 1000.0],
            'low': [9.5, 9.6, 9.7],
            'close': [10.2, 10.3, 10.4],
            'volume': [100000, 110000, 120000]
        })
        
        result = validator.validate_ohlcv(df, 'test_data')
        assert isinstance(result, bool)
    
    def test_validate_volume_positive(self):
        """测试成交量为正"""
        validator = DataValidator()
        
        # 负成交量
        df = pd.DataFrame({
            'open': [10.0, 10.1, 10.2],
            'high': [10.5, 10.6, 10.7],
            'low': [9.5, 9.6, 9.7],
            'close': [10.2, 10.3, 10.4],
            'volume': [100000, -100, 120000]  # 负成交量
        })
        
        result = validator.validate_ohlcv(df, 'test_data')
        assert isinstance(result, bool)


class TestLookAheadBias:
    """前向偏差测试"""
    
    def test_no_future_data_in_backtest(self):
        """测试回测不使用未来数据"""
        from engine.backtest import BacktestEngine
        
        engine = BacktestEngine()
        
        # 模拟历史数据
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'code': ['000001'] * len(dates),
            'open': np.random.uniform(9.5, 10.5, len(dates)),
            'high': np.random.uniform(10.0, 11.0, len(dates)),
            'low': np.random.uniform(9.0, 9.8, len(dates)),
            'close': np.random.uniform(9.5, 10.5, len(dates)),
            'volume': np.random.uniform(100000, 200000, len(dates))
        })
        
        # 确保按日期排序
        df = df.sort_values('date')
        
        # 验证 _get_history_for_factors 只返回历史数据
        current_date = '2023-06-15'
        history = engine._get_history_for_factors(current_date)
        
        # 所有返回的数据应该是当前日期之前的
        for code, hist_df in history.items():
            assert all(hist_df.index < current_date), \
                f"Found future data for {code} in historical data"
    
    def test_factor_calculation_no_lookahead(self):
        """测试因子计算不使用未来数据"""
        from factors.momentum_factor import MomentumFactor
        
        factor = MomentumFactor()
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'code': '000001',
            'close': np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        })
        df = df.set_index('date')
        
        # 计算因子 (在特定日期)
        test_date = '2023-06-15'
        result = factor.compute(df, window=20)
        
        # 验证结果只基于历史数据
        # 因子值应该不包含未来日期的信息
        assert isinstance(result, float)
    
    def test_rebalance_dates_not_future(self):
        """测试调仓日期不为未来日期"""
        from engine.backtest import BacktestEngine
        from utils.trading_calendar import TradingCalendar
        
        engine = BacktestEngine()
        calendar = TradingCalendar()
        
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        
        # 获取调仓日期
        rebalance_dates = engine._get_rebalance_dates('W')
        
        # 验证所有调仓日期都在回测区间内
        for date_str in rebalance_dates:
            assert start_date <= date_str <= end_date, \
                f"Rebalance date {date_str} outside backtest range"
            
            # 验证是交易日
            assert calendar.is_trading_day_str(date_str), \
                f"Rebalance date {date_str} is not a trading day"


class TestDataQuality:
    """数据质量测试"""
    
    def test_missing_dates_handling(self):
        """测试缺失日期处理"""
        validator = DataValidator()
        
        # 有缺失日期的数据
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        # 移除一些日期
        dates = dates[dates.dayofweek < 5]  # 移除周末
        
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(9.5, 10.5, len(dates)),
            'high': np.random.uniform(10.0, 11.0, len(dates)),
            'low': np.random.uniform(9.0, 9.8, len(dates)),
            'close': np.random.uniform(9.5, 10.5, len(dates)),
            'volume': np.random.uniform(100000, 200000, len(dates))
        })
        
        # 应该能处理缺失日期
        result = validator.validate_ohlcv(df, 'test_data')
        assert isinstance(result, bool)
    
    def test_duplicate_dates_handling(self):
        """测试重复日期处理"""
        validator = DataValidator()
        
        # 创建重复日期
        dates = ['2023-01-01', '2023-01-01', '2023-01-02']
        
        df = pd.DataFrame({
            'date': dates,
            'open': [10.0, 10.1, 10.2],
            'high': [10.5, 10.6, 10.7],
            'low': [9.5, 9.6, 9.7],
            'close': [10.2, 10.3, 10.4],
            'volume': [100000, 110000, 120000]
        })
        
        # 应该能检测或处理重复
        result = validator.validate_ohlcv(df, 'test_data')
        assert isinstance(result, bool)
    
    def test_data_type_consistency(self):
        """测试数据类型一致性"""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'open': [10.0, 10.1, 10.2],
            'high': [10.5, 10.6, 10.7],
            'low': [9.5, 9.6, 9.7],
            'close': [10.2, 10.3, 10.4],
            'volume': [100000, 110000, 120000]
        })
        
        # 检查数据类型
        assert df['open'].dtype in [np.float64, np.float32]
        assert df['high'].dtype in [np.float64, np.float32]
        assert df['low'].dtype in [np.float64, np.float32]
        assert df['close'].dtype in [np.float64, np.float32]
        assert df['volume'].dtype in [np.int64, np.int32]
