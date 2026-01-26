# ============================================================================
# 文件: factors/technical/volatility.py
# ============================================================================
"""
波动率因子
"""
import numpy as np
import pandas as pd

from ..base import BaseFactor, FactorMeta, FactorRegistry
from config import settings


@FactorRegistry.register
class ATRFactor(BaseFactor):
    """
    ATR (Average True Range) 因子
    """
    
    meta = FactorMeta(
        name="atr",
        category="technical",
        description="平均真实波幅",
        lookback=20
    )
    
    def __init__(self, period: int = None, **kwargs):
        super().__init__(**kwargs)
        self.period = period or settings.factor.ATR_PERIOD
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算 ATR"""
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()
        
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        # True Range
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        ])
        
        # ATR (EMA)
        tr_s = pd.Series(tr, index=df.index)
        atr = tr_s.ewm(span=self.period, adjust=False).mean()
        
        return atr.rename(self.name)


@FactorRegistry.register
class ATRPercentFactor(BaseFactor):
    """
    ATR 百分比因子 (波动率)
    
    ATR / Close，衡量相对波动率
    """
    
    meta = FactorMeta(
        name="atr_pct",
        category="technical",
        description="ATR占价格百分比",
        lookback=20
    )
    
    def __init__(self, period: int = None, **kwargs):
        super().__init__(**kwargs)
        self.period = period or settings.factor.ATR_PERIOD
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算 ATR%"""
        atr_factor = ATRFactor(period=self.period)
        atr = atr_factor.compute(df)
        
        atr_pct = atr / df['close']
        
        return atr_pct.rename(self.name)


@FactorRegistry.register
class VolatilityFactor(BaseFactor):
    """
    历史波动率因子
    
    N日收益率标准差 × √252 (年化)
    """
    
    meta = FactorMeta(
        name="volatility",
        category="technical",
        description="N日年化波动率",
        lookback=25
    )
    
    def __init__(self, window: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.window = window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算历史波动率"""
        returns = df['close'].pct_change()
        vol = returns.rolling(self.window).std() * np.sqrt(252)
        
        return vol.rename(self.name)


@FactorRegistry.register
class ChandelierStopFactor(BaseFactor):
    """
    吊灯止损因子
    
    Stop = 最高价回溯 - N × ATR
    """
    
    meta = FactorMeta(
        name="chandelier_stop",
        category="technical",
        description="吊灯止损线",
        lookback=20
    )
    
    def __init__(
        self, 
        atr_period: int = None, 
        multiplier: float = None, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.atr_period = atr_period or settings.factor.ATR_PERIOD
        self.multiplier = multiplier or settings.strategy.CHANDELIER_MULT
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算吊灯止损线"""
        atr_factor = ATRFactor(period=self.atr_period)
        atr = atr_factor.compute(df)
        
        rolling_high = df['high'].rolling(self.atr_period, min_periods=1).max()
        
        stop = rolling_high - self.multiplier * atr
        
        return stop.rename(self.name)


@FactorRegistry.register
class VolatilityRegimeFactor(BaseFactor):
    """
    波动率状态因子
    
    判断当前处于高波动/低波动环境
    """
    
    meta = FactorMeta(
        name="vol_regime",
        category="technical",
        description="波动率状态 (0=低, 1=中, 2=高)",
        lookback=65
    )
    
    def __init__(self, window: int = 60, **kwargs):
        super().__init__(**kwargs)
        self.window = window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算波动率状态"""
        vol_factor = VolatilityFactor(window=20)
        vol = vol_factor.compute(df)
        
        # 分位数
        vol_rank = vol.rolling(self.window, min_periods=20).apply(
            lambda x: (x[-1] > x[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
            raw=True
        )
        
        # 状态划分
        regime = pd.Series(1, index=df.index)  # 默认中等
        regime[vol_rank < 0.33] = 0  # 低波动
        regime[vol_rank > 0.67] = 2  # 高波动
        
        return regime.astype(np.int8).rename(self.name)