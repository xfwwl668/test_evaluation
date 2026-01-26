# ============================================================================
# 文件: factors/composite/alpha_score.py
# ============================================================================
"""
复合因子 - Alpha 评分
"""
import numpy as np
import pandas as pd
from typing import Dict, List

from ..base import BaseFactor, FactorMeta, FactorRegistry, FactorPipeline
from config import settings


@FactorRegistry.register
class AlphaScoreFactor(BaseFactor):
    """
    多因子综合评分
    
    Score = w1×RSRS + w2×Volume + w3×Price
    """
    
    meta = FactorMeta(
        name="alpha_score",
        category="composite",
        description="多因子综合评分",
        lookback=650,
        dependencies=["rsrs_zscore", "vol_rank", "vwap_bias"]
    )
    
    def __init__(
        self,
        w_rsrs: float = 0.5,
        w_volume: float = 0.3,
        w_price: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.w_rsrs = w_rsrs
        self.w_volume = w_volume
        self.w_price = w_price
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算综合评分"""
        # 获取子因子
        rsrs = FactorRegistry.get("rsrs_zscore").compute(df)
        vol_rank = FactorRegistry.get("vol_rank").compute(df)
        vwap_bias = FactorRegistry.get("vwap_bias").compute(df)
        
        # 标准化到 [0, 1]
        def minmax(s):
            return (s - s.min()) / (s.max() - s.min() + 1e-10)
        
        rsrs_norm = minmax(rsrs.clip(-3, 3))
        
        # 成交量: 倒U型评分 (0.5 最优)
        vol_score = 1 - 4 * (vol_rank - 0.5) ** 2
        vol_score = vol_score.clip(0, 1)
        
        price_norm = minmax(vwap_bias.clip(-0.1, 0.1))
        
        # 加权求和
        score = (
            self.w_rsrs * rsrs_norm +
            self.w_volume * vol_score +
            self.w_price * price_norm
        )
        
        return score.rename(self.name)


class MultiFactorScorer:
    """
    通用多因子评分器
    
    支持自定义因子权重和评分方式
    """
    
    def __init__(self):
        self.factors: List[tuple] = []  # [(factor_name, weight, transform)]
    
    def add_factor(
        self, 
        name: str, 
        weight: float = 1.0,
        transform: str = "rank"  # rank, zscore, minmax, raw
    ) -> 'MultiFactorScorer':
        """添加因子"""
        self.factors.append((name, weight, transform))
        return self
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        """计算综合评分"""
        scores = []
        total_weight = sum(w for _, w, _ in self.factors)
        
        for name, weight, transform in self.factors:
            factor = FactorRegistry.get(name)
            values = factor.compute(df)
            
            # 变换
            if transform == "rank":
                transformed = values.rank(pct=True)
            elif transform == "zscore":
                transformed = (values - values.mean()) / (values.std() + 1e-10)
            elif transform == "minmax":
                transformed = (values - values.min()) / (values.max() - values.min() + 1e-10)
            else:
                transformed = values
            
            scores.append(transformed * weight / total_weight)
        
        return sum(scores)
    
    def score_batch(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """批量评分"""
        results = {}
        
        for code, df in data.items():
            try:
                results[code] = self.score(df)
            except Exception:
                continue
        
        return pd.DataFrame(results)