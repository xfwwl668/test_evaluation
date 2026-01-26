# ============================================================================
# 文件: factors/technical/momentum.py
# ============================================================================
"""
动量因子
"""
import numpy as np
import pandas as pd

from ..base import BaseFactor, FactorMeta, FactorRegistry
from config import settings


@FactorRegistry.register
class MomentumFactor(BaseFactor):
    """
    价格动量因子
    
    计算: (P_t - P_{t-n}) / P_{t-n}
    """
    
    meta = FactorMeta(
        name="momentum",
        category="technical",
        description="N日价格动量",
        lookback=25
    )
    
    def __init__(self, window: int = None, **kwargs):
        super().__init__(**kwargs)
        self.window = window or settings.factor.MOMENTUM_WINDOW
        self.meta.lookback = self.window + 5
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算动量"""
        return df['close'].pct_change(self.window).rename(self.name)


@FactorRegistry.register
class ROCFactor(BaseFactor):
    """
    ROC (Rate of Change) 因子
    
    与 Momentum 类似，但使用对数收益
    """
    
    meta = FactorMeta(
        name="roc",
        category="technical",
        description="N日对数收益",
        lookback=25
    )
    
    def __init__(self, window: int = None, **kwargs):
        super().__init__(**kwargs)
        self.window = window or settings.factor.MOMENTUM_WINDOW
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算 ROC"""
        log_ret = np.log(df['close'] / df['close'].shift(self.window))
        return log_ret.rename(self.name)


@FactorRegistry.register
class OBVTrendFactor(BaseFactor):
    """
    OBV 趋势因子
    
    判断资金流入/流出趋势
    """
    
    meta = FactorMeta(
        name="obv_trend",
        category="technical",
        description="OBV趋势方向 (1=多, 0=空)",
        lookback=25
    )
    
    def __init__(self, ma_period: int = None, **kwargs):
        super().__init__(**kwargs)
        self.ma_period = ma_period or settings.factor.OBV_MA_PERIOD
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算 OBV 趋势"""
        close = df['close'].to_numpy()
        vol = df['vol'].to_numpy().astype(np.float64)
        
        # OBV 计算
        price_change = np.diff(close, prepend=close[0])
        obv_sign = np.sign(price_change)
        obv_sign[price_change == 0] = 0
        obv = np.cumsum(obv_sign * vol)
        
        # OBV 均线
        obv_s = pd.Series(obv, index=df.index)
        obv_ma = obv_s.rolling(self.ma_period, min_periods=1).mean()
        
        # 趋势判断
        trend = (obv > obv_ma.to_numpy()).astype(np.int8)
        
        return pd.Series(trend, index=df.index, name=self.name)


@FactorRegistry.register
class VWAPBiasFactor(BaseFactor):
    """
    VWAP 乖离率因子
    
    衡量当前价格偏离成交量加权均价的程度
    """
    
    meta = FactorMeta(
        name="vwap_bias",
        category="technical",
        description="价格对VWAP的乖离率",
        lookback=25
    )
    
    def __init__(self, period: int = None, **kwargs):
        super().__init__(**kwargs)
        self.period = period or settings.factor.VWAP_PERIOD
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算 VWAP 乖离率"""
        close = df['close']
        vol = df['vol'].astype(np.float64)
        amount = df['amount'].astype(np.float64)
        
        # 滚动 VWAP
        cum_amt = amount.rolling(self.period, min_periods=1).sum()
        cum_vol = vol.rolling(self.period, min_periods=1).sum()
        vwap = cum_amt / cum_vol.clip(lower=1)
        
        # 乖离率
        bias = (close - vwap) / vwap
        
        return bias.rename(self.name)


@FactorRegistry.register
class VolumeRankFactor(BaseFactor):
    """
    成交量分位因子
    
    当前成交量在历史中的相对位置
    """
    
    meta = FactorMeta(
        name="vol_rank",
        category="technical", 
        description="成交量历史分位 (0-1)",
        lookback=65
    )
    
    def __init__(self, window: int = 60, **kwargs):
        super().__init__(**kwargs)
        self.window = window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算成交量分位"""
        vol = df['vol']
        
        def rank_pct(x):
            if len(x) < 2:
                return 0.5
            return (x[-1] > x[:-1]).sum() / (len(x) - 1)
        
        vol_rank = vol.rolling(self.window, min_periods=20).apply(rank_pct, raw=True)
        
        return vol_rank.rename(self.name)