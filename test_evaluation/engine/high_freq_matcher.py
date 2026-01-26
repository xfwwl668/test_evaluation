# ============================================================================
# 文件: engine/high_freq_matcher.py
# ============================================================================
"""
高频撮合引擎 - 私募级别

特性:
1. 涨停板检测与风险规避
2. 分时成交模拟
3. 极速挂单逻辑
4. 冲击成本计算
5. T+1 锁定处理

涨停板敢死队规避策略:
┌─────────────────────────────────────────────────────────────────────────────┐
│                        涨停板风险处理流程                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  买入时:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 检测是否涨停封板 (is_limit_up && 封单量/流通盘 > 5%)             │   │
│  │ 2. 若封死 → 放弃追涨停，等回封                                      │   │
│  │ 3. 若开板 → 判断开板次数，首次开板可买，多次开板风险大              │   │
│  │ 4. 挂单价格 = 涨停价 - 0.01 (防止追高买不到)                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  卖出时:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 检测是否跌停封板                                                 │   │
│  │ 2. 若跌停封死 → 挂跌停价排队，记录队列位置                          │   │
│  │ 3. 若开板 → 立即市价卖出                                            │   │
│  │ 4. 预估成交概率，若概率 < 30% → 考虑次日集合竞价卖                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, time
import logging

from .matcher import Order, OrderStatus, MatchEngine
from config import settings


class OrderType(Enum):
    """订单类型"""
    LIMIT = "LIMIT"           # 限价单
    MARKET = "MARKET"         # 市价单
    LIMIT_UP_CHASE = "LIMIT_UP_CHASE"  # 追涨停
    FORCE_SELL = "FORCE_SELL"  # 强制卖出


class LimitStatus(Enum):
    """涨跌停状态"""
    NORMAL = "NORMAL"
    LIMIT_UP_SEALED = "LIMIT_UP_SEALED"      # 涨停封死
    LIMIT_UP_OPEN = "LIMIT_UP_OPEN"          # 涨停开板
    LIMIT_DOWN_SEALED = "LIMIT_DOWN_SEALED"  # 跌停封死
    LIMIT_DOWN_OPEN = "LIMIT_DOWN_OPEN"      # 跌停开板


@dataclass
class HighFreqOrder(Order):
    """高频订单扩展"""
    order_type: OrderType = OrderType.LIMIT
    priority: int = 0                    # 优先级 (越高越先处理)
    time_in_force: str = "DAY"           # GTC / DAY / IOC
    allow_partial: bool = True           # 允许部分成交
    
    # 涨跌停相关
    is_limit_chase: bool = False         # 是否追涨停
    limit_status: LimitStatus = LimitStatus.NORMAL
    queue_position: int = 0              # 排队位置
    estimated_fill_prob: float = 1.0     # 预估成交概率
    
    # 执行时间
    submit_time: str = ""
    fill_time: str = ""


@dataclass
class MarketMicrostructure:
    """市场微观结构数据"""
    code: str
    current_price: float
    
    # 涨跌停
    limit_up_price: float
    limit_down_price: float
    is_limit_up: bool
    is_limit_down: bool
    
    # 封单数据
    limit_up_volume: float = 0      # 涨停封单量
    limit_down_volume: float = 0    # 跌停封单量
    open_times: int = 0             # 开板次数
    
    # 流动性
    bid_volume: float = 0           # 买一量
    ask_volume: float = 0           # 卖一量
    spread: float = 0               # 买卖价差
    
    # 成交
    total_volume: float = 0
    avg_trade_size: float = 0
    
    @property
    def seal_strength(self) -> float:
        """封单强度 (封单量/流通盘估计)"""
        if self.total_volume <= 0:
            return 0
        return self.limit_up_volume / self.total_volume
    
    @property
    def is_hard_sealed(self) -> bool:
        """是否封死 (封单强度 > 5%)"""
        return self.seal_strength > 0.05


class HighFreqMatcher:
    """
    高频撮合引擎
    
    核心功能:
    1. 涨跌停智能处理
    2. 冲击成本计算
    3. 分时成交模拟
    4. 极速开盘卖出
    """
    
    def __init__(
        self,
        base_commission: float = 0.0003,
        base_slippage: float = 0.001,
        impact_factor: float = 0.1,      # 冲击成本系数
        limit_chase_premium: float = 0.002  # 追涨停溢价
    ):
        self.base_commission = base_commission
        self.base_slippage = base_slippage
        self.impact_factor = impact_factor
        self.limit_chase_premium = limit_chase_premium
        
        self._order_id = 0
        self.logger = logging.getLogger("HighFreqMatcher")
    
    def analyze_market_structure(
        self,
        market_data: pd.Series,
        historical_vol: float = None
    ) -> MarketMicrostructure:
        """
        分析市场微观结构
        
        Args:
            market_data: 当日行情 (open, high, low, close, vol, ...)
            historical_vol: 历史日均成交量
        """
        code = market_data.get('code', 'unknown')
        close = market_data['close']
        open_price = market_data['open']
        high = market_data['high']
        low = market_data['low']
        volume = market_data.get('vol', 0)
        
        prev_close = market_data.get('prev_close', close / 1.05)
        
        # 涨跌停价格
        limit_up = round(prev_close * 1.1, 2)
        limit_down = round(prev_close * 0.9, 2)
        
        # 判断涨跌停
        is_limit_up = close >= limit_up - 0.01
        is_limit_down = close <= limit_down + 0.01
        
        # 封单量估计 (用最高/最低价判断)
        limit_up_vol = 0
        limit_down_vol = 0
        open_times = 0
        
        if is_limit_up:
            # 涨停: high == limit_up 且 close == limit_up
            if high == close:
                # 一字板，封单强
                limit_up_vol = volume * 0.3  # 估计
            else:
                # 开过板
                open_times = 1
                limit_up_vol = volume * 0.1
        
        if is_limit_down:
            if low == close:
                limit_down_vol = volume * 0.3
            else:
                open_times = 1
                limit_down_vol = volume * 0.1
        
        return MarketMicrostructure(
            code=code,
            current_price=close,
            limit_up_price=limit_up,
            limit_down_price=limit_down,
            is_limit_up=is_limit_up,
            is_limit_down=is_limit_down,
            limit_up_volume=limit_up_vol,
            limit_down_volume=limit_down_vol,
            open_times=open_times,
            total_volume=volume,
            avg_trade_size=volume / 240 if volume > 0 else 0
        )
    
    def create_high_freq_order(
        self,
        code: str,
        side: str,
        price: float,
        quantity: int,
        create_date: str,
        order_type: OrderType = OrderType.LIMIT,
        signal_reason: str = ""
    ) -> HighFreqOrder:
        """创建高频订单"""
        self._order_id += 1
        
        return HighFreqOrder(
            order_id=f"HF-{self._order_id:08d}",
            code=code,
            side=side,
            price=price,
            quantity=quantity,
            create_date=create_date,
            order_type=order_type,
            signal_reason=signal_reason,
            submit_time=datetime.now().strftime('%H:%M:%S.%f')
        )
    
    def match_order(
        self,
        order: HighFreqOrder,
        market_data: pd.Series,
        micro: MarketMicrostructure,
        position: Optional['Position'],
        current_date: str,
        is_opening: bool = False
    ) -> HighFreqOrder:
        """
        高频订单撮合
        
        Args:
            order: 订单
            market_data: 当日行情
            micro: 市场微观结构
            position: 持仓
            current_date: 当前日期
            is_opening: 是否为开盘时段 (用于开盘强卖)
        """
        # 基础检查
        if market_data.empty or pd.isna(market_data.get('open')):
            return self._reject(order, "停牌或无数据")
        
        # ===== 涨跌停风险处理 =====
        if order.side == "BUY":
            result = self._handle_buy_limit_risk(order, micro)
            if result is not None:
                return result
        else:
            result = self._handle_sell_limit_risk(order, micro, position, current_date)
            if result is not None:
                return result
        
        # ===== T+1 检查 =====
        if order.side == "SELL" and position is not None:
            if position.buy_date == current_date:
                return self._reject(order, "T+1 限制")
        
        # ===== 成交价格计算 =====
        fill_price, impact_cost = self._calculate_fill_price(
            order, market_data, micro, is_opening
        )
        
        # ===== 成交数量 (考虑流动性) =====
        fill_qty, fill_prob = self._calculate_fill_quantity(
            order, micro
        )
        
        if fill_qty <= 0:
            return self._reject(order, "流动性不足")
        
        # ===== 更新订单 =====
        order.status = OrderStatus.FILLED
        order.filled_price = round(fill_price, 4)
        order.filled_quantity = fill_qty
        order.slippage = round((fill_price - order.price) * fill_qty, 2)
        order.commission = round(
            fill_price * fill_qty * self.base_commission + impact_cost, 2
        )
        order.estimated_fill_prob = fill_prob
        order.fill_time = datetime.now().strftime('%H:%M:%S.%f')
        
        self.logger.info(
            f"[HF-MATCH] {order.side} {order.code} "
            f"qty={fill_qty} @ {fill_price:.3f} "
            f"impact={impact_cost:.2f} prob={fill_prob:.0%}"
        )
        
        return order
    
    def _handle_buy_limit_risk(
        self,
        order: HighFreqOrder,
        micro: MarketMicrostructure
    ) -> Optional[HighFreqOrder]:
        """
        处理买入时的涨停风险
        
        策略:
        1. 封死涨停 → 拒绝买入 (无法成交)
        2. 开板涨停 → 降低成交概率
        3. 追涨停订单 → 特殊处理
        """
        if micro.is_limit_up:
            order.limit_status = (
                LimitStatus.LIMIT_UP_SEALED if micro.is_hard_sealed 
                else LimitStatus.LIMIT_UP_OPEN
            )
            
            if micro.is_hard_sealed:
                # 涨停封死，无法买入
                self.logger.warning(f"[LIMIT-UP] {order.code} 涨停封死，拒绝买入")
                return self._reject(order, "涨停封死无法买入")
            
            if micro.open_times >= 3:
                # 多次开板，风险大
                self.logger.warning(f"[LIMIT-UP] {order.code} 开板{micro.open_times}次，风险高")
                return self._reject(order, f"涨停开板{micro.open_times}次，风险过高")
            
            # 首次开板可买，但降低成交概率
            order.estimated_fill_prob = 0.6
        
        return None
    
    def _handle_sell_limit_risk(
        self,
        order: HighFreqOrder,
        micro: MarketMicrostructure,
        position: Optional['Position'],
        current_date: str
    ) -> Optional[HighFreqOrder]:
        """
        处理卖出时的跌停风险
        
        策略:
        1. 跌停封死 → 挂跌停价排队，估算成交概率
        2. 跌停开板 → 立即市价卖出
        3. 若成交概率 < 30% → 建议次日集合竞价
        """
        if micro.is_limit_down:
            order.limit_status = (
                LimitStatus.LIMIT_DOWN_SEALED if micro.is_hard_sealed
                else LimitStatus.LIMIT_DOWN_OPEN
            )
            
            if micro.is_hard_sealed:
                # 跌停封死
                # 估算排队位置和成交概率
                if position is not None:
                    queue_position = int(micro.limit_down_volume * 0.5)  # 假设在队列中间
                    total_queue = micro.limit_down_volume
                    
                    # 预估成交量 (通常只有 10-20% 能成交)
                    estimated_tradeable = micro.total_volume * 0.15
                    fill_prob = min(estimated_tradeable / (queue_position + 1), 1.0)
                    
                    order.queue_position = queue_position
                    order.estimated_fill_prob = fill_prob
                    
                    if fill_prob < 0.3:
                        self.logger.warning(
                            f"[LIMIT-DOWN] {order.code} 跌停封死，成交概率 {fill_prob:.0%}，"
                            f"建议次日集合竞价"
                        )
                        # 不拒绝，但标记低概率
                        order.signal_reason += " [低成交概率]"
                
                # 挂跌停价
                order.price = micro.limit_down_price
            
            else:
                # 跌停开板，立即卖出
                order.order_type = OrderType.FORCE_SELL
                self.logger.info(f"[LIMIT-DOWN] {order.code} 跌停开板，强制卖出")
        
        return None
    
    def _calculate_fill_price(
        self,
        order: HighFreqOrder,
        market_data: pd.Series,
        micro: MarketMicrostructure,
        is_opening: bool
    ) -> Tuple[float, float]:
        """
        计算成交价格和冲击成本
        
        Returns:
            (fill_price, impact_cost)
        """
        open_price = market_data['open']
        close = market_data['close']
        volume = market_data.get('vol', 1e6)
        
        base_price = open_price if is_opening else close
        
        # ===== 基础滑点 =====
        if order.side == "BUY":
            slippage = self.base_slippage
        else:
            slippage = -self.base_slippage
        
        # ===== 冲击成本 (大单额外成本) =====
        order_value = order.price * order.quantity
        avg_trade_value = micro.avg_trade_size * base_price if micro.avg_trade_size > 0 else 50000
        
        # 订单相对于平均交易的倍数
        size_ratio = order_value / avg_trade_value if avg_trade_value > 0 else 1
        
        # 冲击成本 = 系数 × 订单倍数 × 基础价格
        impact = self.impact_factor * np.sqrt(size_ratio) * base_price * 0.001
        
        if order.side == "BUY":
            impact = abs(impact)
        else:
            impact = -abs(impact)
        
        # ===== 涨跌停特殊处理 =====
        if order.is_limit_chase:
            # 追涨停溢价
            slippage += self.limit_chase_premium
        
        if micro.is_limit_up and order.side == "BUY":
            # 涨停买入，成交价只能是涨停价
            fill_price = micro.limit_up_price
            impact = 0
        elif micro.is_limit_down and order.side == "SELL":
            # 跌停卖出，成交价只能是跌停价
            fill_price = micro.limit_down_price
            impact = 0
        else:
            fill_price = base_price * (1 + slippage) + impact
        
        impact_cost = abs(impact * order.quantity)
        
        return fill_price, impact_cost
    
    def _calculate_fill_quantity(
        self,
        order: HighFreqOrder,
        micro: MarketMicrostructure
    ) -> Tuple[int, float]:
        """
        计算成交数量和概率
        
        考虑:
        1. 流动性限制
        2. 涨跌停排队
        """
        # 流动性检查: 单笔不超过日成交量的 5%
        max_fillable = int(micro.total_volume * 0.05 / 100) * 100
        
        if max_fillable <= 0:
            max_fillable = order.quantity
        
        fill_qty = min(order.quantity, max_fillable)
        
        # 成交概率
        fill_prob = order.estimated_fill_prob
        
        if micro.is_limit_up and order.side == "BUY":
            fill_prob *= 0.5  # 涨停追买概率降低
        
        if micro.is_limit_down and order.side == "SELL":
            if micro.is_hard_sealed:
                fill_prob *= 0.3  # 跌停封死概率大降
        
        # 整百股
        fill_qty = int(fill_qty / 100) * 100
        
        return fill_qty, fill_prob
    
    def execute_opening_force_sell(
        self,
        code: str,
        position: 'Position',
        market_data: pd.Series,
        prev_close: float,
        current_date: str
    ) -> Optional[HighFreqOrder]:
        """
        开盘强卖逻辑
        
        条件:
        1. 昨日买入 (今日可卖 T+1)
        2. 开盘 15 分钟未涨超 2%
        3. 当前价跌破昨日收盘价
        
        Returns:
            卖出订单或 None
        """
        # T+1 检查
        if position.buy_date == current_date:
            return None
        
        open_price = market_data['open']
        current_price = market_data.get('close', open_price)  # 模拟 15 分钟价格
        
        # 计算涨幅
        change_from_prev = (current_price - prev_close) / prev_close
        change_from_open = (current_price - open_price) / open_price
        
        # 强卖条件
        should_force_sell = (
            change_from_prev < 0.02 and    # 未涨超 2%
            current_price < prev_close      # 跌破昨收
        )
        
        if should_force_sell:
            order = self.create_high_freq_order(
                code=code,
                side="SELL",
                price=current_price,
                quantity=position.quantity,
                create_date=current_date,
                order_type=OrderType.FORCE_SELL,
                signal_reason=f"开盘强卖: 涨幅{change_from_prev:.1%}, 跌破昨收"
            )
            order.priority = 100  # 最高优先级
            
            self.logger.warning(
                f"[FORCE-SELL] {code} 触发开盘强卖 | "
                f"开盘={open_price:.2f} 现价={current_price:.2f} 昨收={prev_close:.2f}"
            )
            
            return order
        
        return None
    
    def _reject(self, order: HighFreqOrder, reason: str) -> HighFreqOrder:
        """拒绝订单"""
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason
        self.logger.warning(f"[HF-REJECT] {order.code}: {reason}")
        return order


class LimitUpRiskManager:
    """
    涨停板风险管理器
    
    专门处理"涨停板敢死队"策略的风险
    """
    
    def __init__(
        self,
        max_limit_chase_ratio: float = 0.1,    # 追涨停资金占比上限
        min_seal_strength: float = 0.03,        # 最低封单强度
        max_open_times: int = 2,                # 最大开板次数
        limit_down_stop_loss: float = 0.07      # 跌停止损线
    ):
        self.max_limit_chase_ratio = max_limit_chase_ratio
        self.min_seal_strength = min_seal_strength
        self.max_open_times = max_open_times
        self.limit_down_stop_loss = limit_down_stop_loss
        
        self.logger = logging.getLogger("LimitUpRisk")
    
    def evaluate_limit_up_trade(
        self,
        micro: MarketMicrostructure,
        total_equity: float,
        current_limit_up_exposure: float
    ) -> Tuple[bool, str, float]:
        """
        评估涨停板交易
        
        Args:
            micro: 市场微观结构
            total_equity: 总资金
            current_limit_up_exposure: 当前涨停股敞口
        
        Returns:
            (is_allowed, reason, max_position_value)
        """
        if not micro.is_limit_up:
            return True, "非涨停", total_equity * 0.1
        
        # 检查敞口限制
        if current_limit_up_exposure >= total_equity * self.max_limit_chase_ratio:
            return False, f"涨停敞口超限 ({current_limit_up_exposure/total_equity:.0%})", 0
        
        # 检查封单强度
        if micro.seal_strength < self.min_seal_strength:
            return False, f"封单强度不足 ({micro.seal_strength:.1%})", 0
        
        # 检查开板次数
        if micro.open_times > self.max_open_times:
            return False, f"开板次数过多 ({micro.open_times}次)", 0
        
        # 计算最大仓位
        remaining_quota = total_equity * self.max_limit_chase_ratio - current_limit_up_exposure
        max_position = min(remaining_quota, total_equity * 0.05)  # 单只最多 5%
        
        return True, "允许追涨停", max_position
    
    def handle_limit_down_position(
        self,
        micro: MarketMicrostructure,
        position: 'Position',
        entry_price: float
    ) -> Dict:
        """
        处理跌停持仓
        
        Returns:
            {'action': 'hold'/'sell'/'queue', 'reason': str, 'urgency': int}
        """
        current_pnl = (micro.current_price - entry_price) / entry_price
        
        result = {
            'action': 'hold',
            'reason': '',
            'urgency': 0,
            'estimated_fill_prob': 1.0
        }
        
        if not micro.is_limit_down:
            return result
        
        # 跌停中
        if micro.is_hard_sealed:
            # 封死
            if current_pnl <= -self.limit_down_stop_loss:
                # 亏损过大，无论如何要卖
                result['action'] = 'queue'
                result['reason'] = f'跌停封死+亏损{current_pnl:.1%}，排队卖出'
                result['urgency'] = 10
                result['estimated_fill_prob'] = 0.2
            else:
                result['action'] = 'hold'
                result['reason'] = '跌停封死但亏损可控，暂持'
                result['urgency'] = 5
        else:
            # 开板
            result['action'] = 'sell'
            result['reason'] = '跌停开板，立即卖出'
            result['urgency'] = 10
            result['estimated_fill_prob'] = 0.8
        
        return result