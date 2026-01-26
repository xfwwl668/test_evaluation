# ============================================================================
# 文件: factors/technical/rsrs.py
# ============================================================================
"""
RSRS 因子 - 阻力支撑相对强度
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple

from ..base import BaseFactor, FactorMeta, FactorRegistry
from config import settings


@FactorRegistry.register
class RSRSFactor(BaseFactor):
    """
    RSRS 斜率因子
    
    原理:
        对滚动窗口做回归: High = α + β × Low
        β (斜率) 反映支撑/阻力相对强度
    """
    
    meta = FactorMeta(
        name="rsrs_slope",
        category="technical",
        description="RSRS 回归斜率",
        lookback=20
    )
    
    def __init__(self, window: int = None, **kwargs):
        super().__init__(**kwargs)
        self.window = window or settings.factor.RSRS_WINDOW
        self.meta.lookback = self.window + 10
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算 RSRS 斜率"""
        high = df['high'].to_numpy(dtype=np.float64)
        low = df['low'].to_numpy(dtype=np.float64)
        
        n = len(df)
        if n < self.window:
            return pd.Series(np.nan, index=df.index)
        
        # 滑动窗口回归
        slope, r2 = self._rolling_regression(high, low, self.window)
        
        # 填充前 window-1 个 NaN
        pad = np.full(self.window - 1, np.nan)
        slope_full = np.concatenate([pad, slope])
        
        return pd.Series(slope_full, index=df.index, name=self.name)
    
    @staticmethod
    def _rolling_regression(
        high: np.ndarray, 
        low: np.ndarray, 
        window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        向量化滚动OLS回归
        
        Returns:
            (slope, r2) 数组
        """
        low_win = sliding_window_view(low, window)
        high_win = sliding_window_view(high, window)
        
        x_mean = low_win.mean(axis=1, keepdims=True)
        y_mean = high_win.mean(axis=1, keepdims=True)
        
        x_dev = low_win - x_mean
        y_dev = high_win - y_mean
        
        cov_xy = (x_dev * y_dev).sum(axis=1)
        var_x = (x_dev ** 2).sum(axis=1)
        var_y = (y_dev ** 2).sum(axis=1)
        
        # 斜率
        slope = np.divide(cov_xy, var_x, out=np.zeros_like(cov_xy), where=var_x > 1e-10)
        
        # R²
        denom = var_x * var_y
        r2 = np.divide(cov_xy ** 2, denom, out=np.zeros_like(cov_xy), where=denom > 1e-10)
        
        return slope, r2


@FactorRegistry.register
class RSRSZScoreFactor(BaseFactor):
    """
    RSRS 标准化因子 (修正版)
    
    计算:
        1. 计算原始斜率
        2. 滚动标准化 (Z-Score)
        3. R² 加权修正
    """
    
    meta = FactorMeta(
        name="rsrs_zscore",
        category="technical",
        description="RSRS Z-Score (R²加权)",
        lookback=650
    )
    
    def __init__(
        self, 
        window: int = None, 
        std_window: int = None,
        r2_threshold: float = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window = window or settings.factor.RSRS_WINDOW
        self.std_window = std_window or settings.factor.RSRS_STD_WINDOW
        self.r2_threshold = r2_threshold or settings.factor.RSRS_R2_THRESHOLD
        self.meta.lookback = self.std_window + self.window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算 RSRS Z-Score"""
        high = df['high'].to_numpy(dtype=np.float64)
        low = df['low'].to_numpy(dtype=np.float64)
        
        n = len(df)
        if n < self.window + 60:
            return pd.Series(np.nan, index=df.index, name=self.name)
        
        # 1. 计算斜率和 R²
        slope, r2 = RSRSFactor._rolling_regression(high, low, self.window)
        
        pad = np.full(self.window - 1, np.nan)
        slope_full = np.concatenate([pad, slope])
        r2_full = np.concatenate([pad, r2])
        
        # 2. 滚动标准化
        slope_s = pd.Series(slope_full, index=df.index)
        roll_mean = slope_s.rolling(self.std_window, min_periods=60).mean()
        roll_std = slope_s.rolling(self.std_window, min_periods=60).std()
        
        zscore = (slope_s - roll_mean) / roll_std.clip(lower=1e-10)
        
        # 3. R² 加权修正
        zscore_weighted = zscore * r2_full
        
        # 4. R² 过滤 (可选: 低于阈值时信号衰减)
        valid = r2_full >= self.r2_threshold
        zscore_filtered = np.where(valid, zscore_weighted, zscore_weighted * 0.5)
        
        return pd.Series(zscore_filtered, index=df.index, name=self.name)
    
    def compute_with_r2(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回 zscore 和 r2 两列"""
        high = df['high'].to_numpy(dtype=np.float64)
        low = df['low'].to_numpy(dtype=np.float64)
        
        n = len(df)
        if n < self.window + 60:
            return pd.DataFrame({
                'rsrs_zscore': np.nan,
                'rsrs_r2': np.nan
            }, index=df.index)
        
        slope, r2 = RSRSFactor._rolling_regression(high, low, self.window)
        
        pad = np.full(self.window - 1, np.nan)
        slope_full = np.concatenate([pad, slope])
        r2_full = np.concatenate([pad, r2])
        
        slope_s = pd.Series(slope_full, index=df.index)
        roll_mean = slope_s.rolling(self.std_window, min_periods=60).mean()
        roll_std = slope_s.rolling(self.std_window, min_periods=60).std()
        
        zscore = (slope_s - roll_mean) / roll_std.clip(lower=1e-10)
        zscore_weighted = zscore * r2_full
        
        return pd.DataFrame({
            'rsrs_zscore': zscore_weighted,
            'rsrs_r2': r2_full
        }, index=df.index)


@FactorRegistry.register  
class RSRSValidFactor(BaseFactor):
    """RSRS 有效性标记 (R² > 阈值)"""
    
    meta = FactorMeta(
        name="rsrs_valid",
        category="technical",
        description="RSRS R²有效性标记",
        lookback=20
    )
    
    def __init__(self, window: int = None, r2_threshold: float = None, **kwargs):
        super().__init__(**kwargs)
        self.window = window or settings.factor.RSRS_WINDOW
        self.r2_threshold = r2_threshold or settings.factor.RSRS_R2_THRESHOLD
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """返回 0/1 有效性标记"""
        high = df['high'].to_numpy(dtype=np.float64)
        low = df['low'].to_numpy(dtype=np.float64)
        
        if len(df) < self.window:
            return pd.Series(0, index=df.index, name=self.name)
        
        _, r2 = RSRSFactor._rolling_regression(high, low, self.window)
        
        pad = np.zeros(self.window - 1)
        r2_full = np.concatenate([pad, r2])
        
        valid = (r2_full >= self.r2_threshold).astype(np.int8)
        
        return pd.Series(valid, index=df.index, name=self.name)