# ============================================================================
# 文件: strategy/rsrs_strategy.py
# ============================================================================
"""
RSRS 策略实现 - 基于阻力支撑相对强度
"""
import numpy as np
import pandas as pd
from typing import Dict, List

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry
from factors import FactorRegistry
from config import settings


@StrategyRegistry.register
class RSRSStrategy(BaseStrategy):
    """
    RSRS 策略
    
    核心逻辑:
    - 使用 RSRS Z-Score 判断趋势强度
    - R² 过滤确保信号有效性
    - 量价共振增强信号质量
    - 吊灯止损保护利润
    
    参数:
    - top_n: 选股数量
    - entry_threshold: 入场阈值
    - exit_threshold: 离场阈值
    - use_volume_filter: 是否使用量价过滤
    """
    
    name = "rsrs"
    version = "2.0.0"
    
    def __init__(self, params: Dict = None):
        default_params = {
            'top_n': settings.strategy.TOP_N_STOCKS,
            'entry_threshold': settings.strategy.ENTRY_THRESHOLD,
            'exit_threshold': settings.strategy.EXIT_THRESHOLD,
            'r2_threshold': settings.factor.RSRS_R2_THRESHOLD,
            'use_volume_filter': True,
            'use_chandelier_stop': True,
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)
        
        # 内部状态
        self._entry_prices: Dict[str, float] = {}  # 记录入场价格
        self._stop_prices: Dict[str, float] = {}   # 动态止损价
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算所需因子"""
        factors = {}
        
        # 获取因子实例
        rsrs_factor = FactorRegistry.get("rsrs_zscore")
        r2_factor = FactorRegistry.get("rsrs_valid")
        vol_rank_factor = FactorRegistry.get("vol_rank")
        obv_trend_factor = FactorRegistry.get("obv_trend")
        vwap_bias_factor = FactorRegistry.get("vwap_bias")
        chandelier_factor = FactorRegistry.get("chandelier_stop")
        
        # 批量计算
        factors['rsrs_zscore'] = rsrs_factor.compute_batch(history)
        factors['rsrs_valid'] = r2_factor.compute_batch(history)
        factors['vol_rank'] = vol_rank_factor.compute_batch(history)
        factors['obv_trend'] = obv_trend_factor.compute_batch(history)
        factors['vwap_bias'] = vwap_bias_factor.compute_batch(history)
        factors['chandelier_stop'] = chandelier_factor.compute_batch(history)
        
        self.logger.info(f"Computed {len(factors)} factors for {len(history)} stocks")
        
        return factors
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """生成交易信号"""
        signals = []
        current_date = context.current_date
        
        # 1. 收集候选股票及其因子值
        candidates = []
        
        for _, row in context.current_data.iterrows():
            code = row['code']
            close = row['close']
            
            # 获取因子值
            rsrs = context.get_factor('rsrs_zscore', code)
            rsrs_valid = context.get_factor('rsrs_valid', code)
            vol_rank = context.get_factor('vol_rank', code)
            obv_trend = context.get_factor('obv_trend', code)
            vwap_bias = context.get_factor('vwap_bias', code)
            chandelier = context.get_factor('chandelier_stop', code)
            
            # 跳过无效数据
            if rsrs is None or pd.isna(rsrs):
                continue
            
            candidates.append({
                'code': code,
                'close': close,
                'rsrs': rsrs,
                'rsrs_valid': rsrs_valid or 0,
                'vol_rank': vol_rank or 0.5,
                'obv_trend': obv_trend or 0,
                'vwap_bias': vwap_bias or 0,
                'chandelier': chandelier or 0
            })
        
        if not candidates:
            return signals
        
        # 2. 生成卖出信号 (先卖后买)
        signals.extend(self._generate_exit_signals(context, candidates))
        
        # 3. 生成买入信号
        signals.extend(self._generate_entry_signals(context, candidates))
        
        return signals
    
    def _generate_entry_signals(
        self, 
        context: StrategyContext, 
        candidates: List[Dict]
    ) -> List[Signal]:
        """生成入场信号"""
        signals = []
        
        entry_th = self.get_param('entry_threshold')
        r2_th = self.get_param('r2_threshold')
        use_vol_filter = self.get_param('use_volume_filter')
        top_n = self.get_param('top_n')
        
        # 过滤条件
        qualified = []
        
        for c in candidates:
            # 基础条件: RSRS > 阈值
            if c['rsrs'] <= entry_th:
                continue
            
            # R² 有效性
            if c['rsrs_valid'] < 1:
                continue
            
            # 量价过滤 (可选)
            if use_vol_filter:
                # 成交量温和 (30%-75% 分位)
                if not (0.3 <= c['vol_rank'] <= 0.75):
                    continue
                
                # OBV 上升趋势
                if c['obv_trend'] < 1:
                    continue
                
                # 价格在 VWAP 上方
                if c['vwap_bias'] <= 0:
                    continue
            
            qualified.append(c)
        
        # 按 RSRS 排序选 Top N
        qualified.sort(key=lambda x: x['rsrs'], reverse=True)
        selected = qualified[:top_n]
        
        # 生成信号
        for c in selected:
            # 跳过已持仓
            if c['code'] in context.positions:
                continue
            
            reason = f"RSRS={c['rsrs']:.2f} R²有效 Vol={c['vol_rank']:.0%}"
            
            signals.append(Signal(
                code=c['code'],
                side=OrderSide.BUY,
                weight=1.0 / top_n,  # 等权重
                price=c['close'],
                reason=reason
            ))
        
        return signals
    
    def _generate_exit_signals(
        self, 
        context: StrategyContext, 
        candidates: List[Dict]
    ) -> List[Signal]:
        """生成离场信号"""
        signals = []
        
        exit_th = self.get_param('exit_threshold')
        use_stop = self.get_param('use_chandelier_stop')
        
        # 遍历持仓
        for code, qty in context.positions.items():
            if qty <= 0:
                continue
            
            # 查找该股票的因子
            cand = next((c for c in candidates if c['code'] == code), None)
            
            if cand is None:
                continue
            
            should_exit = False
            reason = ""
            
            # 条件1: RSRS 低于离场阈值
            if cand['rsrs'] < exit_th:
                should_exit = True
                reason = f"RSRS={cand['rsrs']:.2f} < {exit_th}"
            
            # 条件2: 吊灯止损
            if use_stop and cand['chandelier'] > 0:
                if cand['close'] < cand['chandelier']:
                    should_exit = True
                    reason = f"触发吊灯止损 {cand['chandelier']:.2f}"
            
            # 条件3: 巨量下跌 (警示信号)
            if cand['vol_rank'] > 0.85 and cand['rsrs'] < 0:
                should_exit = True
                reason = f"巨量下跌 Vol={cand['vol_rank']:.0%}"
            
            if should_exit:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    price=cand['close'],
                    reason=reason
                ))
        
        return signals
    
    def on_order_filled(self, order: 'Order') -> None:
        """订单成交回调"""
        if order.side == OrderSide.BUY:
            self._entry_prices[order.code] = order.filled_price
            self.logger.info(f"[BUY] {order.code} @ {order.filled_price:.2f} | {order.signal_reason}")
        else:
            if order.code in self._entry_prices:
                entry = self._entry_prices.pop(order.code)
                pnl = (order.filled_price - entry) / entry
                self.logger.info(f"[SELL] {order.code} @ {order.filled_price:.2f} | PnL={pnl:.2%} | {order.signal_reason}")