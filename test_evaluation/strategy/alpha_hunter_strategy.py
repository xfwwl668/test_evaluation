# ============================================================================
# 文件: strategy/alpha_hunter_strategy.py
# ============================================================================
"""
Alpha-Hunter-V1 私募级超短线策略

目标:
- 年化收益 > 30%
- 最大回撤 < 10%
- 持仓周期 T+1 到 T+2

核心逻辑:
1. 极致胜率过滤 (5重条件)
2. T+1 必杀卖出
3. 动态移动锁利
4. Kelly 仓位管理
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry
from factors.alpha_hunter_factors import (
    AlphaHunterFactorEngine, AlphaSignal, SignalStrength,
    AdvancedRSRSFactor, PressureLevelFactor, MarketBreadthFactor
)
from engine.high_freq_matcher import HighFreqMatcher, MarketMicrostructure
from engine.risk import RiskManager


@dataclass
class TradeRecord:
    """交易记录 (用于 Kelly 计算)"""
    code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_ratio: float
    is_win: bool


@dataclass
class AlphaPosition:
    """Alpha 策略持仓状态"""
    code: str
    entry_price: float
    entry_date: str
    quantity: int
    
    # 动态止损止盈
    stop_loss_price: float
    take_profit_price: float
    trailing_stop: float
    
    # 历史高点 (用于移动锁利)
    highest_price: float
    highest_date: str
    
    # 累计利润阈值 (每 +3% 触发一次锁利)
    lock_profit_thresholds: List[float] = field(default_factory=lambda: [0.03, 0.06, 0.09, 0.12])
    current_lock_level: int = 0
    
    def update_trailing_stop(self, current_price: float, current_date: str):
        """
        更新移动锁利
        
        规则: 每增加 3% 利润，止损上移 2%
        """
        if current_price > self.highest_price:
            self.highest_price = current_price
            self.highest_date = current_date
        
        current_pnl = (current_price - self.entry_price) / self.entry_price
        
        # 检查是否触发新的锁利阈值
        while (self.current_lock_level < len(self.lock_profit_thresholds) and
               current_pnl >= self.lock_profit_thresholds[self.current_lock_level]):
            
            # 止损上移 2%
            new_stop = self.entry_price * (1 + 0.02 * (self.current_lock_level + 1))
            
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
                logging.getLogger("AlphaPosition").info(
                    f"[LOCK-PROFIT] {self.code} 锁利触发 L{self.current_lock_level+1} "
                    f"止损上移至 {new_stop:.2f}"
                )
            
            self.current_lock_level += 1
        
        # 硬止损不动
        self.trailing_stop = max(self.trailing_stop, self.stop_loss_price)


@StrategyRegistry.register
class AlphaHunterStrategy(BaseStrategy):
    """
    Alpha-Hunter-V1 策略
    
    买入准则 (ALL 条件):
    1. 修正 RSRS > 0.8 且 R² > 0.85
    2. 价格 > MA5 且 MA5 斜率向上
    3. 换手率 < 25%
    4. 全市场上涨家数 > 40%
    5. 距离压力位 > 5%
    
    卖出准则 (ANY 条件):
    1. 开盘强卖: 15分钟未涨2% 且 跌破昨收
    2. 移动锁利: 每+3%利润 → 止损上移2%
    3. 硬止损: -3%
    4. 跌破 MA5
    5. 最大持仓 2 天
    """
    
    name = "alpha_hunter_v1"
    version = "1.0.0"
    
    # ===== 策略参数 =====
    DEFAULT_PARAMS = {
        # 入场参数
        'rsrs_threshold': 0.8,
        'r2_threshold': 0.85,
        'max_turnover': 0.25,           # 最大换手率 25%
        'market_breadth_threshold': 0.40,  # 上涨家数 40%
        'min_pressure_distance': 0.05,  # 压力距离 5%
        'ma5_slope_threshold': 0.001,   # MA5 斜率阈值
        
        # 离场参数
        'opening_check_gain': 0.02,     # 开盘检查涨幅阈值
        'hard_stop_loss': 0.03,         # 硬止损 3%
        'profit_lock_step': 0.03,       # 每 3% 锁利一次
        'stop_raise_step': 0.02,        # 止损上移 2%
        'max_holding_days': 2,          # 最大持仓天数
        
        # 仓位参数
        'kelly_lookback': 20,           # Kelly 回溯交易数
        'kelly_fraction': 0.5,          # Kelly 保守系数
        'max_single_position': 0.08,    # 单只最大 8%
        'max_total_position': 0.70,     # 总仓位最大 70%
        'max_positions': 8,             # 最大持仓数
        
        # 行业限制
        'max_sector_exposure': 0.20,    # 单行业最大 20%
        
        # 涨停限制
        'allow_limit_up_chase': False,  # 不追涨停
        
        # 价格过滤
        'min_price': 5.0,
        'max_price': 80.0,
        'min_volume': 2000000,          # 最低成交额 200万
    }
    
    def __init__(self, params: Dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        
        # 因子引擎
        self.factor_engine = AlphaHunterFactorEngine()
        
        # 高频撮合器
        self.hf_matcher = HighFreqMatcher()
        
        # 持仓状态
        self._positions: Dict[str, AlphaPosition] = {}
        
        # 交易记录 (Kelly 计算用)
        self._trade_history: deque = deque(maxlen=100)
        
        # 市场情绪缓存
        self._market_breadth_cache: Dict = {}
        
        # 行业敞口
        self._sector_exposure: Dict[str, float] = {}
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """预计算因子"""
        self.logger.info("Computing Alpha-Hunter factors...")
        
        factors = {}
        rsrs_factor = AdvancedRSRSFactor()
        pressure_factor = PressureLevelFactor()
        
        rsrs_results = {}
        pressure_results = {}
        ma5_results = {}
        
        for code, df in history.items():
            if len(df) < 250:
                continue
            
            try:
                # RSRS
                rsrs_data = rsrs_factor.compute_full(df)
                rsrs_results[code] = rsrs_data
                
                # 压力位
                pressure_data = pressure_factor.compute_full(df)
                pressure_results[code] = pressure_data
                
                # MA5 及斜率
                ma5 = df['close'].rolling(5).mean()
                ma5_slope = ma5.diff(3) / ma5.shift(3)
                ma5_results[code] = pd.DataFrame({
                    'ma5': ma5,
                    'ma5_slope': ma5_slope
                }, index=df.index)
                
            except Exception as e:
                self.logger.warning(f"Factor error for {code}: {e}")
        
        # 转换为宽表
        if rsrs_results:
            factors['rsrs_score'] = pd.DataFrame({
                code: data['rsrs_final_score'] for code, data in rsrs_results.items()
            })
            factors['rsrs_r2'] = pd.DataFrame({
                code: data['rsrs_r2'] for code, data in rsrs_results.items()
            })
            factors['rsrs_valid'] = pd.DataFrame({
                code: data['rsrs_valid'] for code, data in rsrs_results.items()
            })
        
        if pressure_results:
            factors['pressure_distance'] = pd.DataFrame({
                code: data['distance_to_pressure'] for code, data in pressure_results.items()
            })
        
        if ma5_results:
            factors['ma5'] = pd.DataFrame({
                code: data['ma5'] for code, data in ma5_results.items()
            })
            factors['ma5_slope'] = pd.DataFrame({
                code: data['ma5_slope'] for code, data in ma5_results.items()
            })
        
        self.logger.info(f"Computed factors for {len(rsrs_results)} stocks")
        return factors
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """生成交易信号"""
        signals = []
        current_date = context.current_date
        
        # 1. 计算市场情绪
        breadth = self._calculate_market_breadth(context)
        self._market_breadth_cache[current_date] = breadth
        
        # 2. 开盘强卖检查 (优先处理)
        sell_signals = self._generate_opening_force_sell(context, breadth)
        signals.extend(sell_signals)
        
        # 3. 常规离场检查
        exit_signals = self._generate_exit_signals(context)
        signals.extend(exit_signals)
        
        # 4. 市场情绪过滤
        if not breadth.get('is_bullish', False):
            self.logger.info(f"市场情绪偏空 ({breadth.get('advance_ratio', 0):.0%})，暂停入场")
            return signals
        
        # 5. 入场信号
        entry_signals = self._generate_entry_signals(context, breadth)
        signals.extend(entry_signals)
        
        return signals
    
    def _calculate_market_breadth(self, context: StrategyContext) -> Dict:
        """计算市场广度"""
        breadth_factor = MarketBreadthFactor()
        return breadth_factor.compute_market_breadth(
            context.current_data,
            context.current_date
        )
    
    def _generate_opening_force_sell(
        self,
        context: StrategyContext,
        breadth: Dict
    ) -> List[Signal]:
        """
        生成开盘强卖信号
        
        条件:
        1. T+1 可卖
        2. 15分钟未涨超 2%
        3. 跌破昨日收盘价
        """
        signals = []
        
        opening_threshold = self.get_param('opening_check_gain')
        
        for code, pos in list(self._positions.items()):
            # T+1 检查
            if pos.entry_date == context.current_date:
                continue
            
            # 获取数据
            row = context.current_data[context.current_data['code'] == code]
            if row.empty:
                continue
            
            current_price = row['close'].iloc[0]
            open_price = row['open'].iloc[0]
            
            # 昨日收盘价
            history = context.get_history(code, 2)
            if len(history) < 2:
                continue
            prev_close = history['close'].iloc[-2]
            
            # 涨幅检查
            change_from_prev = (current_price - prev_close) / prev_close
            
            # 开盘强卖条件
            if change_from_prev < opening_threshold and current_price < prev_close:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    price=current_price,
                    priority=100,  # 最高优先级
                    reason=f"开盘强卖: 涨幅{change_from_prev:.1%} < {opening_threshold:.0%}, 跌破昨收"
                ))
                
                self.logger.warning(
                    f"[FORCE-SELL] {code} 开盘强卖触发 | "
                    f"现价={current_price:.2f} 昨收={prev_close:.2f}"
                )
        
        return signals
    
    def _generate_exit_signals(self, context: StrategyContext) -> List[Signal]:
        """生成常规离场信号"""
        signals = []
        
        hard_stop = self.get_param('hard_stop_loss')
        max_days = self.get_param('max_holding_days')
        
        for code, pos in list(self._positions.items()):
            # T+1 检查
            if pos.entry_date == context.current_date:
                continue
            
            row = context.current_data[context.current_data['code'] == code]
            if row.empty:
                continue
            
            current_price = row['close'].iloc[0]
            
            # 更新移动锁利
            pos.update_trailing_stop(current_price, context.current_date)
            
            should_exit = False
            reason = ""
            
            # ===== 条件1: 硬止损 =====
            pnl = (current_price - pos.entry_price) / pos.entry_price
            if pnl <= -hard_stop:
                should_exit = True
                reason = f"硬止损 {pnl:.1%}"
            
            # ===== 条件2: 移动止损 =====
            if not should_exit and current_price < pos.trailing_stop:
                should_exit = True
                reason = f"移动止损触发 ({pos.trailing_stop:.2f})"
            
            # ===== 条件3: 跌破 MA5 =====
            if not should_exit:
                ma5 = context.get_factor('ma5', code)
                if ma5 is not None and current_price < ma5:
                    should_exit = True
                    reason = f"跌破 MA5 ({ma5:.2f})"
            
            # ===== 条件4: 最大持仓天数 =====
            if not should_exit:
                try:
                    entry_dt = datetime.strptime(pos.entry_date, '%Y-%m-%d')
                    current_dt = datetime.strptime(context.current_date, '%Y-%m-%d')
                    holding_days = (current_dt - entry_dt).days
                    
                    if holding_days >= max_days:
                        should_exit = True
                        reason = f"持仓{holding_days}天，强制离场"
                except:
                    pass
            
            if should_exit:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    price=current_price,
                    reason=reason
                ))
                
                self.logger.info(f"[EXIT] {code} | {reason} | PnL={pnl:.1%}")
        
        return signals
    
    def _generate_entry_signals(
        self,
        context: StrategyContext,
        breadth: Dict
    ) -> List[Signal]:
        """生成入场信号"""
        signals = []
        
        # 参数
        rsrs_th = self.get_param('rsrs_threshold')
        r2_th = self.get_param('r2_threshold')
        max_turnover = self.get_param('max_turnover')
        min_pressure = self.get_param('min_pressure_distance')
        ma5_slope_th = self.get_param('ma5_slope_threshold')
        min_price = self.get_param('min_price')
        max_price = self.get_param('max_price')
        min_volume = self.get_param('min_volume')
        max_positions = self.get_param('max_positions')
        
        # 检查持仓数
        if len(self._positions) >= max_positions:
            return signals
        
        candidates = []
        
        for _, row in context.current_data.iterrows():
            code = row['code']
            close = row['close']
            volume = row.get('vol', 0)
            amount = row.get('amount', close * volume)
            
            # ===== 基础过滤 =====
            if code in self._positions or code in context.positions:
                continue
            
            if close < min_price or close > max_price:
                continue
            
            if amount < min_volume:
                continue
            
            # ===== 条件1: RSRS =====
            rsrs = context.get_factor('rsrs_score', code)
            r2 = context.get_factor('rsrs_r2', code)
            
            if rsrs is None or pd.isna(rsrs) or rsrs <= rsrs_th:
                continue
            
            if r2 is None or pd.isna(r2) or r2 < r2_th:
                continue
            
            # ===== 条件2: MA5 趋势 =====
            ma5 = context.get_factor('ma5', code)
            ma5_slope = context.get_factor('ma5_slope', code)
            
            if ma5 is None or close <= ma5:
                continue
            
            if ma5_slope is None or ma5_slope < ma5_slope_th:
                continue
            
            # ===== 条件3: 换手率 =====
            # 估算换手率 (成交额/市值，简化)
            history = context.get_history(code, 5)
            if not history.empty:
                avg_amount = history['amount'].mean() if 'amount' in history.columns else 0
                est_market_cap = close * history['vol'].mean() * 100 if 'vol' in history.columns else 1e10
                turnover = avg_amount / est_market_cap if est_market_cap > 0 else 0
                
                if turnover > max_turnover:
                    continue
            
            # ===== 条件4: 压力距离 =====
            pressure = context.get_factor('pressure_distance', code)
            if pressure is not None and pressure < min_pressure:
                continue
            
            # ===== 条件5: 非涨停 (默认不追) =====
            if not self.get_param('allow_limit_up_chase'):
                prev_close = history['close'].iloc[-1] if not history.empty else close * 0.95
                if close >= prev_close * 1.095:
                    continue
            
            # 通过所有条件
            score = rsrs * r2
            candidates.append({
                'code': code,
                'close': close,
                'rsrs': rsrs,
                'r2': r2,
                'pressure': pressure or 0.1,
                'score': score
            })
        
        # 排序选最强
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        slots = max_positions - len(self._positions)
        selected = candidates[:slots]
        
        # 计算仓位
        for cand in selected:
            weight = self._calculate_kelly_position(context.total_equity)
            weight = min(weight, self.get_param('max_single_position'))
            
            if weight < 0.02:
                continue
            
            signals.append(Signal(
                code=cand['code'],
                side=OrderSide.BUY,
                weight=weight,
                price=cand['close'],
                reason=f"RSRS={cand['rsrs']:.2f} R²={cand['r2']:.2f} 压力距={cand['pressure']:.1%}"
            ))
            
            self.logger.info(
                f"[ENTRY] {cand['code']} | RSRS={cand['rsrs']:.2f} R²={cand['r2']:.2f} | "
                f"Weight={weight:.1%}"
            )
        
        return signals
    
    def _calculate_kelly_position(self, total_equity: float) -> float:
        """
        Kelly 准则计算仓位
        
        公式: f = (p × b - q) / b
        其中: p=胜率, q=败率, b=盈亏比
        
        实际使用: f × kelly_fraction (保守系数)
        """
        if len(self._trade_history) < 5:
            # 历史不足，使用默认仓位
            return 0.05
        
        # 计算胜率和盈亏比
        wins = [t for t in self._trade_history if t.is_win]
        losses = [t for t in self._trade_history if not t.is_win]
        
        if not wins or not losses:
            return 0.05
        
        p = len(wins) / len(self._trade_history)
        q = 1 - p
        
        avg_win = np.mean([t.pnl_ratio for t in wins])
        avg_loss = abs(np.mean([t.pnl_ratio for t in losses]))
        
        if avg_loss <= 0:
            return 0.05
        
        b = avg_win / avg_loss
        
        # Kelly 公式
        kelly = (p * b - q) / b if b > 0 else 0
        
        # 保守调整
        kelly_fraction = self.get_param('kelly_fraction')
        position = kelly * kelly_fraction
        
        # 限制范围
        position = np.clip(position, 0.02, 0.10)
        
        self.logger.debug(
            f"[KELLY] 胜率={p:.0%} 盈亏比={b:.2f} Kelly={kelly:.2%} → 仓位={position:.2%}"
        )
        
        return position
    
    def on_order_filled(self, order) -> None:
        """订单成交回调"""
        if order.side == OrderSide.BUY:
            # 初始化持仓状态
            hard_stop = self.get_param('hard_stop_loss')
            
            self._positions[order.code] = AlphaPosition(
                code=order.code,
                entry_price=order.filled_price,
                entry_date=order.create_date,
                quantity=order.filled_quantity,
                stop_loss_price=order.filled_price * (1 - hard_stop),
                take_profit_price=order.filled_price * 1.15,
                trailing_stop=order.filled_price * (1 - hard_stop),
                highest_price=order.filled_price,
                highest_date=order.create_date
            )
            
            self.logger.info(
                f"[FILLED-BUY] {order.code} @ {order.filled_price:.2f} "
                f"止损={order.filled_price * (1 - hard_stop):.2f}"
            )
        
        else:
            # 记录交易
            if order.code in self._positions:
                pos = self._positions.pop(order.code)
                pnl = (order.filled_price - pos.entry_price) / pos.entry_price
                
                trade = TradeRecord(
                    code=order.code,
                    entry_date=pos.entry_date,
                    exit_date=order.create_date,
                    entry_price=pos.entry_price,
                    exit_price=order.filled_price,
                    pnl_ratio=pnl,
                    is_win=(pnl > 0)
                )
                self._trade_history.append(trade)
                
                self.logger.info(
                    f"[FILLED-SELL] {order.code} @ {order.filled_price:.2f} "
                    f"PnL={pnl:.1%} | {order.signal_reason}"
                )
    
    def get_performance_summary(self) -> Dict:
        """获取绩效摘要"""
        if not self._trade_history:
            return {'trades': 0, 'win_rate': 0, 'avg_pnl': 0}
        
        wins = [t for t in self._trade_history if t.is_win]
        
        return {
            'trades': len(self._trade_history),
            'win_rate': len(wins) / len(self._trade_history),
            'avg_pnl': np.mean([t.pnl_ratio for t in self._trade_history]),
            'avg_win': np.mean([t.pnl_ratio for t in wins]) if wins else 0,
            'avg_loss': np.mean([t.pnl_ratio for t in self._trade_history if not t.is_win]) if len(wins) < len(self._trade_history) else 0,
            'max_win': max([t.pnl_ratio for t in self._trade_history]),
            'max_loss': min([t.pnl_ratio for t in self._trade_history])
        }