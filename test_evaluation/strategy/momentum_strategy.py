# ============================================================================
# 文件: strategy/momentum_strategy.py
# ============================================================================
"""
动量策略实现 - 基于价格动量和趋势跟随
"""
import numpy as np
import pandas as pd
from typing import Dict, List

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry
from factors import FactorRegistry
from config import settings


@StrategyRegistry.register
class MomentumStrategy(BaseStrategy):
    """
    动量策略
    
    核心逻辑:
    - 选择过去 N 日涨幅最大的股票
    - 结合波动率进行仓位调整
    - 使用均值回归作为离场信号
    
    参数:
    - lookback: 动量回溯周期
    - top_n: 选股数量
    - rebalance_threshold: 调仓阈值
    """
    
    name = "momentum"
    version = "1.0.0"
    
    def __init__(self, params: Dict = None):
        default_params = {
            'lookback': settings.factor.MOMENTUM_WINDOW,
            'top_n': 20,
            'min_momentum': 0.05,      # 最低动量 5%
            'max_volatility': 0.5,     # 最大波动率 50%
            'use_volatility_weight': True,
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算因子"""
        lookback = self.get_param('lookback')
        
        momentum_factor = FactorRegistry.get("momentum", window=lookback)
        volatility_factor = FactorRegistry.get("volatility", window=lookback)
        atr_pct_factor = FactorRegistry.get("atr_pct")
        
        factors = {
            'momentum': momentum_factor.compute_batch(history),
            'volatility': volatility_factor.compute_batch(history),
            'atr_pct': atr_pct_factor.compute_batch(history),
        }
        
        self.logger.info(f"Computed momentum factors for {len(history)} stocks")
        
        return factors
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """生成交易信号"""
        signals = []
        
        top_n = self.get_param('top_n')
        min_mom = self.get_param('min_momentum')
        max_vol = self.get_param('max_volatility')
        use_vol_weight = self.get_param('use_volatility_weight')
        
        # 收集候选
        candidates = []
        
        for _, row in context.current_data.iterrows():
            code = row['code']
            
            mom = context.get_factor('momentum', code)
            vol = context.get_factor('volatility', code)
            atr_pct = context.get_factor('atr_pct', code)
            
            if mom is None or pd.isna(mom):
                continue
            
            # 过滤条件
            if mom < min_mom:
                continue
            
            if vol is not None and vol > max_vol:
                continue
            
            candidates.append({
                'code': code,
                'close': row['close'],
                'momentum': mom,
                'volatility': vol or 0.3,
                'atr_pct': atr_pct or 0.02
            })
        
        # 排序选 Top N
        candidates.sort(key=lambda x: x['momentum'], reverse=True)
        selected = candidates[:top_n]
        
        # 计算权重
        if use_vol_weight and selected:
            # 波动率倒数加权
            inv_vol = [1 / max(c['volatility'], 0.1) for c in selected]
            total_inv_vol = sum(inv_vol)
            weights = [v / total_inv_vol for v in inv_vol]
        else:
            weights = [1.0 / len(selected)] * len(selected) if selected else []
        
        # 生成卖出信号 (清仓不在 top_n 中的)
        for code in list(context.positions.keys()):
            if code not in [c['code'] for c in selected]:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    reason="动量排名下降，清仓"
                ))
        
        # 生成买入信号
        for c, w in zip(selected, weights):
            current_weight = context.positions.get(c['code'], 0) / context.total_equity if context.total_equity > 0 else 0
            
            # 只在权重变化较大时调整
            if abs(w - current_weight) > 0.02:
                signals.append(Signal(
                    code=c['code'],
                    side=OrderSide.BUY,
                    weight=w * 0.95,  # 保留现金
                    price=c['close'],
                    reason=f"MOM={c['momentum']:.1%} VOL={c['volatility']:.1%}"
                ))
        
        return signals