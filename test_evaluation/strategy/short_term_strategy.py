# ============================================================================
# 文件: strategy/short_term_strategy.py
# ============================================================================
"""
高胜率短线 RSRS 策略

策略逻辑:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ShortTermRSRSStrategy                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【入场条件】 ALL 必须满足:                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. RSRS Score > 0.7 (修正版，含 R² 加权)                            │   │
│  │ 2. R² > 0.8 (信号有效性)                                            │   │
│  │ 3. Price > MA5 AND Price > MA20 (趋势共振)                          │   │
│  │ 4. Volume > MA5_Vol × 1.5 (放量突破)                                │   │
│  │ 5. 非 T+1 锁定期                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【离场条件】 ANY 触发:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 固定止损: 亏损 > 3%                                              │   │
│  │ 2. 趋势止损: Price < MA5                                            │   │
│  │ 3. ATR 移动止盈: Price < (最高点 - 2×ATR)                           │   │
│  │ 4. 时间止损: 持仓 > 5 天且无盈利                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【仓位管理】                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Position = (Total × 0.5%) / (ATR × Price)                           │   │
│  │ 单只最大 10%，总仓位最大 80%                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry
from factors import FactorRegistry
from factors.technical.rsrs_advanced import RSRSAdvancedFactor, RSRSMultiPeriodFactor
from engine.risk import RiskManager, PositionSizer
from config import settings


@dataclass
class PositionState:
    """持仓状态跟踪"""
    code: str
    entry_price: float
    entry_date: str
    quantity: int
    highest_price: float          # 持仓期间最高价 (用于移动止盈)
    stop_loss_price: float        # 止损价
    trailing_stop_price: float    # 移动止盈价
    atr_at_entry: float           # 入场时的 ATR
    
    def update_trailing_stop(self, current_price: float, atr: float, multiplier: float = 2.0):
        """更新移动止盈价"""
        if current_price > self.highest_price:
            self.highest_price = current_price
            # 新止盈价 = 最高价 - ATR × 倍数
            new_stop = current_price - atr * multiplier
            # 只能上移，不能下移
            self.trailing_stop_price = max(self.trailing_stop_price, new_stop)


@StrategyRegistry.register
class ShortTermRSRSStrategy(BaseStrategy):
    """
    高胜率短线 RSRS 策略
    
    特点:
    - 严格的入场过滤 (高胜率)
    - 动态止盈止损 (保护利润)
    - 波动率仓位管理 (风险可控)
    - T+1 合规处理
    """
    
    name = "short_term_rsrs"
    version = "2.0.0"
    
    def __init__(self, params: Dict = None):
        # 默认参数
        default_params = {
            # 入场参数
            'rsrs_entry_threshold': 0.7,      # RSRS 入场阈值
            'r2_threshold': 0.8,              # R² 有效阈值
            'volume_multiplier': 1.5,         # 放量倍数
            
            # 离场参数
            'fixed_stop_loss': 0.03,          # 固定止损 3%
            'trailing_atr_mult': 2.0,         # ATR 移动止盈倍数
            'max_holding_days': 5,            # 最大持仓天数
            
            # 仓位参数
            'risk_per_trade': 0.005,          # 单笔风险 0.5%
            'max_single_weight': 0.10,        # 单只最大 10%
            'max_total_weight': 0.80,         # 总仓位最大 80%
            'max_positions': 10,              # 最大持仓数
            
            # 过滤参数
            'min_price': 3.0,                 # 最低价格
            'max_price': 100.0,               # 最高价格
            'min_volume': 1000000,            # 最低成交量
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)
        
        # 持仓状态跟踪
        self._position_states: Dict[str, PositionState] = {}
        
        # 风控模块
        self.risk_manager = RiskManager(
            risk_per_trade=merged_params['risk_per_trade'],
            max_single_weight=merged_params['max_single_weight'],
            max_total_weight=merged_params['max_total_weight']
        )
        
        self.position_sizer = PositionSizer(
            risk_per_trade=merged_params['risk_per_trade']
        )
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """预计算因子"""
        self.logger.info("Computing advanced RSRS factors...")
        
        factors = {}
        
        # 修正版 RSRS
        rsrs_factor = RSRSAdvancedFactor(
            window=18,
            std_window=600,
            r2_threshold=self.get_param('r2_threshold')
        )
        
        # 多周期共振
        resonance_factor = RSRSMultiPeriodFactor(short_window=18, mid_window=60)
        
        # ATR
        atr_factor = FactorRegistry.get('atr_pct')
        
        # 批量计算
        rsrs_results = {}
        resonance_results = {}
        atr_results = {}
        
        for code, df in history.items():
            if len(df) < 250:
                continue
            
            try:
                # RSRS 详细数据
                rsrs_data = rsrs_factor.compute_all(df)
                rsrs_results[code] = rsrs_data
                
                # 共振
                res = resonance_factor.compute(df)
                resonance_results[code] = res
                
                # ATR
                atr = atr_factor.compute(df)
                atr_results[code] = atr
                
            except Exception as e:
                self.logger.warning(f"Factor compute failed for {code}: {e}")
                continue
        
        # 转换为宽表
        if rsrs_results:
            # RSRS Score
            factors['rsrs_score'] = pd.DataFrame({
                code: data['rsrs_score'] for code, data in rsrs_results.items()
            })
            
            # RSRS R²
            factors['rsrs_r2'] = pd.DataFrame({
                code: data['rsrs_r2'] for code, data in rsrs_results.items()
            })
            
            # RSRS Valid
            factors['rsrs_valid'] = pd.DataFrame({
                code: data['rsrs_valid'] for code, data in rsrs_results.items()
            })
        
        if resonance_results:
            factors['rsrs_resonance'] = pd.DataFrame(resonance_results)
        
        if atr_results:
            factors['atr_pct'] = pd.DataFrame(atr_results)
        
        self.logger.info(f"Computed factors for {len(rsrs_results)} stocks")
        
        return factors
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """生成交易信号"""
        signals = []
        current_date = context.current_date
        
        # 1. 更新持仓状态 & 生成离场信号
        exit_signals = self._generate_exit_signals(context)
        signals.extend(exit_signals)
        
        # 2. 检查是否可以开新仓
        current_positions = len(context.positions)
        max_positions = self.get_param('max_positions')
        
        if current_positions >= max_positions:
            self.logger.debug(f"Max positions reached: {current_positions}/{max_positions}")
            return signals
        
        # 3. 筛选入场候选
        candidates = self._screen_entry_candidates(context)
        
        if not candidates:
            return signals
        
        # 4. 排序并选择
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        slots_available = max_positions - current_positions
        selected = candidates[:slots_available]
        
        # 5. 计算仓位并生成信号
        for cand in selected:
            # 仓位计算
            weight = self._calculate_position_weight(
                cand, context.total_equity
            )
            
            if weight < 0.01:  # 权重太小跳过
                continue
            
            signal = Signal(
                code=cand['code'],
                side=OrderSide.BUY,
                weight=weight,
                price=cand['close'],
                reason=self._format_entry_reason(cand)
            )
            signals.append(signal)
            
            self.logger.info(
                f"[ENTRY] {cand['code']} | RSRS={cand['rsrs_score']:.2f} "
                f"R²={cand['rsrs_r2']:.2f} | Vol×{cand['vol_ratio']:.1f}"
            )
        
        return signals
    
    def _screen_entry_candidates(self, context: StrategyContext) -> List[Dict]:
        """筛选入场候选股 (向量化优化)"""
        # 获取所有因子值 (向量化)
        rsrs_score_series = context.get_all_factors('rsrs_score')
        rsrs_r2_series = context.get_all_factors('rsrs_r2')
        atr_pct_series = context.get_all_factors('atr_pct')
        
        # 参数
        rsrs_threshold = self.get_param('rsrs_entry_threshold')
        r2_threshold = self.get_param('r2_threshold')
        vol_mult = self.get_param('volume_multiplier')
        min_price = self.get_param('min_price')
        max_price = self.get_param('max_price')
        min_volume = self.get_param('min_volume')
        
        # 复制数据并添加因子
        current_data = context.current_data.copy()
        
        if rsrs_score_series is not None:
            current_data['rsrs_score'] = rsrs_score_series
            current_data['rsrs_r2'] = rsrs_r2_series if rsrs_r2_series is not None else 0
            current_data['atr_pct'] = atr_pct_series if atr_pct_series is not None else 0.02
        
        # 统一NaN处理 (向量化)
        current_data = current_data.fillna({
            'rsrs_r2': 0,
            'atr_pct': 0.02
        })
        
        # 计算均线和成交量指标 (向量化)
        def calculate_ma_vol(group):
            if len(group) < 20:
                return pd.Series({
                    'ma5': 0,
                    'ma20': 0,
                    'vol_ratio': 0,
                    'vol_ma5': 0
                })
            
            ma5 = group['close'].tail(5).mean()
            ma20 = group['close'].tail(20).mean()
            vol_ma5 = group['vol'].tail(5).mean() if 'vol' in group.columns else 0
            
            return pd.Series({
                'ma5': ma5,
                'ma20': ma20,
                'vol_ratio': group['vol'].iloc[-1] / vol_ma5 if vol_ma5 > 0 else 0,
                'vol_ma5': vol_ma5
            })
        
        # 按股票计算指标
        indicators = current_data.groupby('code').apply(calculate_ma_vol)
        current_data = current_data.merge(indicators, left_on='code', right_index=True)
        
        # 过滤条件 (向量化)
        mask = (
            (~current_data['code'].isin(context.positions.keys())) &  # 不在持仓中
            (current_data['close'] >= min_price) &  # 价格过滤
            (current_data['close'] <= max_price) &
            (current_data['vol'] >= min_volume) &  # 成交量过滤
            (current_data['rsrs_score'].notna()) &  # RSRS 过滤
            (current_data['rsrs_score'] > rsrs_threshold) &
            (current_data['rsrs_r2'] >= r2_threshold) &
            (current_data['close'] > current_data['ma5']) &  # 均线趋势
            (current_data['close'] > current_data['ma20']) &
            (current_data['vol_ratio'] >= vol_mult)  # 放量突破
        )
        
        filtered_data = current_data[mask].copy()
        
        # 计算综合评分
        filtered_data['score'] = filtered_data['rsrs_score'] * filtered_data['rsrs_r2']
        
        # 转换为字典列表
        candidates = filtered_data.to_dict('records')
        
        return candidates
    
    def _generate_exit_signals(self, context: StrategyContext) -> List[Signal]:
        """生成离场信号"""
        signals = []
        current_date = context.current_date
        
        fixed_stop = self.get_param('fixed_stop_loss')
        atr_mult = self.get_param('trailing_atr_mult')
        max_days = self.get_param('max_holding_days')
        
        for code, quantity in list(context.positions.items()):
            if quantity <= 0:
                continue
            
            # 获取当前数据
            current_row = context.current_data[context.current_data['code'] == code]
            if current_row.empty:
                continue
            
            current_price = current_row['close'].iloc[0]
            
            # 获取持仓状态
            state = self._position_states.get(code)
            
            if state is None:
                # 新持仓，初始化状态
                atr_pct = context.get_factor('atr_pct', code) or 0.02
                history = context.get_history(code, 5)
                ma5 = history['close'].mean() if not history.empty else current_price * 0.98
                
                state = PositionState(
                    code=code,
                    entry_price=current_price,
                    entry_date=current_date,
                    quantity=quantity,
                    highest_price=current_price,
                    stop_loss_price=current_price * (1 - fixed_stop),
                    trailing_stop_price=current_price * (1 - fixed_stop),
                    atr_at_entry=atr_pct * current_price
                )
                self._position_states[code] = state
                continue  # 入场当天不检查离场 (T+1)
            
            # ===== T+1 检查 =====
            if state.entry_date == current_date:
                continue  # 当天买入不可卖出
            
            should_exit = False
            exit_reason = ""
            
            # ===== 检查1: 固定止损 =====
            pnl_ratio = (current_price - state.entry_price) / state.entry_price
            
            if pnl_ratio <= -fixed_stop:
                should_exit = True
                exit_reason = f"固定止损 {pnl_ratio:.1%}"
            
            # ===== 检查2: 跌破 MA5 =====
            if not should_exit:
                history = context.get_history(code, 5)
                if not history.empty:
                    ma5 = history['close'].mean()
                    if current_price < ma5:
                        should_exit = True
                        exit_reason = f"跌破MA5 ({ma5:.2f})"
            
            # ===== 检查3: ATR 移动止盈 =====
            if not should_exit:
                atr = (context.get_factor('atr_pct', code) or 0.02) * current_price
                state.update_trailing_stop(current_price, atr, atr_mult)
                
                if current_price < state.trailing_stop_price:
                    should_exit = True
                    exit_reason = f"移动止盈 ({state.trailing_stop_price:.2f})"
            
            # ===== 检查4: 时间止损 =====
            if not should_exit:
                try:
                    entry_dt = datetime.strptime(state.entry_date, '%Y-%m-%d')
                    current_dt = datetime.strptime(current_date, '%Y-%m-%d')
                    holding_days = (current_dt - entry_dt).days
                    
                    if holding_days >= max_days and pnl_ratio <= 0:
                        should_exit = True
                        exit_reason = f"时间止损 ({holding_days}天)"
                except:
                    pass
            
            # ===== 生成离场信号 =====
            if should_exit:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    price=current_price,
                    reason=exit_reason
                ))
                
                # 清理状态
                if code in self._position_states:
                    del self._position_states[code]
                
                self.logger.info(
                    f"[EXIT] {code} | {exit_reason} | "
                    f"PnL={pnl_ratio:.1%} Entry={state.entry_price:.2f}"
                )
        
        return signals
    
    def _calculate_position_weight(self, candidate: Dict, total_equity: float) -> float:
        """
        计算仓位权重 (波动率头寸管理)
        
        公式: Position = (Total × Risk%) / (ATR × Price)
        """
        risk_per_trade = self.get_param('risk_per_trade')
        max_single = self.get_param('max_single_weight')
        
        price = candidate['close']
        atr_pct = candidate['atr_pct']
        
        # ATR 金额
        atr_amount = atr_pct * price
        
        if atr_amount <= 0:
            return 0.0
        
        # 风险预算
        risk_budget = total_equity * risk_per_trade
        
        # 仓位金额
        position_value = risk_budget / atr_pct
        
        # 权重
        weight = position_value / total_equity
        
        # 限制
        weight = min(weight, max_single)
        
        return round(weight, 4)
    
    def _format_entry_reason(self, candidate: Dict) -> str:
        """格式化入场原因"""
        return (
            f"RSRS={candidate['rsrs_score']:.2f} "
            f"R²={candidate['rsrs_r2']:.2f} "
            f"Vol×{candidate['vol_ratio']:.1f} "
            f">MA5/20"
        )
    
    def on_order_filled(self, order) -> None:
        """订单成交回调"""
        if order.side == OrderSide.BUY:
            # 初始化持仓状态
            atr = order.price * 0.02  # 默认 ATR
            fixed_stop = self.get_param('fixed_stop_loss')
            
            self._position_states[order.code] = PositionState(
                code=order.code,
                entry_price=order.filled_price,
                entry_date=order.create_date,
                quantity=order.filled_quantity,
                highest_price=order.filled_price,
                stop_loss_price=order.filled_price * (1 - fixed_stop),
                trailing_stop_price=order.filled_price * (1 - fixed_stop),
                atr_at_entry=atr
            )
            
            self.logger.info(
                f"[FILLED BUY] {order.code} @ {order.filled_price:.2f} "
                f"qty={order.filled_quantity} | {order.signal_reason}"
            )
        
        else:
            # 清理持仓状态
            if order.code in self._position_states:
                state = self._position_states.pop(order.code)
                pnl = (order.filled_price - state.entry_price) / state.entry_price
                self.logger.info(
                    f"[FILLED SELL] {order.code} @ {order.filled_price:.2f} "
                    f"PnL={pnl:.1%} | {order.signal_reason}"
                )
    
    def on_order_rejected(self, order, reason: str) -> None:
        """订单拒绝回调"""
        self.logger.warning(f"[REJECTED] {order.code} {order.side}: {reason}")