# ============================================================================
# 文件: engine/matcher.py
# ============================================================================
"""
撮合引擎 - 模拟真实交易规则
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict
import pandas as pd
import logging

from config import settings


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    """
    订单对象
    
    记录订单全生命周期
    
    改进:
    - 添加filled_date字段用于T+1检查
    """
    order_id: str
    code: str
    side: str                       # BUY / SELL
    price: float                    # 委托价
    quantity: int                   # 委托量
    create_date: str                # 创建日期
    
    # 成交信息
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_quantity: int = 0
    filled_date: str = ""          # 实际成交日期(用于T+1检查)
    
    # 成本
    commission: float = 0.0
    slippage: float = 0.0
    stamp_duty: float = 0.0
    
    # 拒绝原因
    reject_reason: str = ""
    
    # 信号来源
    signal_reason: str = ""
    
    @property
    def total_cost(self) -> float:
        """总交易成本"""
        return self.commission + self.slippage + self.stamp_duty
    
    @property
    def trade_value(self) -> float:
        """成交金额"""
        return self.filled_price * self.filled_quantity
    
    def __repr__(self) -> str:
        return (f"Order({self.order_id} {self.side} {self.code} "
                f"qty={self.quantity} @ {self.price:.2f} [{self.status.value}])")


class MatchEngine:
    """
    撮合引擎
    
    核心规则:
    1. T+1: 当日买入不可卖出
    2. 涨停: 无法买入 (封板)
    3. 跌停: 无法卖出 (封板)
    4. 滑点: 按开盘价 + 滑点成交
    5. 手续费: 佣金 + 印花税 (卖出)
    
    架构:
    ┌─────────────────────────────────────────────────────────┐
    │                    MatchEngine                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
    │  │ 规则检查    │→ │ 价格撮合    │→ │ 成本计算        │ │
    │  │ T+1/涨跌停  │  │ 开盘+滑点   │  │ 佣金+印花税     │ │
    │  └─────────────┘  └─────────────┘  └─────────────────┘ │
    └─────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        commission_rate: float = None,
        min_commission: float = None,
        stamp_duty: float = None,
        slippage_rate: float = None
    ):
        self.commission_rate = commission_rate or settings.backtest.COMMISSION_RATE
        self.min_commission = min_commission or settings.backtest.MIN_COMMISSION
        self.stamp_duty = stamp_duty or settings.backtest.STAMP_DUTY
        self.slippage_rate = slippage_rate or settings.backtest.SLIPPAGE_RATE
        
        self._order_id_counter = 0
        self.logger = logging.getLogger("MatchEngine")
    
    def create_order(
        self,
        code: str,
        side: str,
        price: float,
        quantity: int,
        create_date: str,
        signal_reason: str = ""
    ) -> Order:
        """创建订单"""
        self._order_id_counter += 1
        
        return Order(
            order_id=f"ORD-{self._order_id_counter:08d}",
            code=code,
            side=side,
            price=price,
            quantity=quantity,
            create_date=create_date,
            signal_reason=signal_reason
        )
    
    def match(
        self,
        order: Order,
        market_data: pd.Series,
        position: Optional['Position'],
        current_date: str
    ) -> Order:
        """
        撮合订单
        
        Args:
            order: 订单
            market_data: 当日行情 (Series: open, high, low, close, is_limit_up, is_limit_down)
            position: 当前持仓 (卖出时需要)
            current_date: 当前日期
        
        Returns:
            更新后的订单
        """
        # 1. 数据检查
        if market_data.empty or pd.isna(market_data.get('open')):
            return self._reject(order, "停牌或无行情数据")
        
        open_price = market_data['open']
        is_limit_up = market_data.get('is_limit_up', False)
        is_limit_down = market_data.get('is_limit_down', False)
        
        # 2. 涨跌停检查
        if order.side == "BUY" and is_limit_up:
            return self._reject(order, "涨停封板，无法买入")
        
        if order.side == "SELL" and is_limit_down:
            return self._reject(order, "跌停封板，无法卖出")
        
        # 3. T+1 检查 (卖出)
        if order.side == "SELL":
            if position is None:
                return self._reject(order, "无持仓")
            
            if position.quantity < order.quantity:
                return self._reject(order, f"持仓不足: 持有{position.quantity}，卖出{order.quantity}")
            
            # 🔴 修复 Problem 16: 使用成交日期进行T+1检查
            if position.buy_date == current_date:
                return self._reject(order, "T+1限制: 当日买入不可卖出")
        
        # 4. 计算成交价 (开盘价 + 滑点)
        # 🔴 修复 Problem 7: 改进滑点模型
        daily_vol = market_data.get('vol', 1000000)
        slippage_rate = self.calculate_slippage_rate(order.quantity, daily_vol)
        
        if order.side == "BUY":
            slippage = open_price * slippage_rate
            filled_price = open_price + slippage
        else:
            slippage = open_price * slippage_rate
            filled_price = open_price - slippage
        
        # 5. 计算手续费
        trade_value = filled_price * order.quantity
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # 印花税 (仅卖出)
        stamp = trade_value * self.stamp_duty if order.side == "SELL" else 0.0
        
        # 6. 更新订单
        order.status = OrderStatus.FILLED
        order.filled_price = round(filled_price, 4)
        order.filled_quantity = order.quantity
        order.filled_date = current_date  # 设置实际成交日期
        order.slippage = round(slippage * order.quantity, 2)
        order.commission = round(commission, 2)
        order.stamp_duty = round(stamp, 2)
        
        self.logger.debug(
            f"[MATCH] {order.side} {order.code} qty={order.quantity} "
            f"@ {order.filled_price:.3f} cost={order.total_cost:.2f} "
            f"date={current_date}"
        )
        
        return order
    
    def _reject(self, order: Order, reason: str) -> Order:
        """拒绝订单"""
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason
        self.logger.warning(f"[REJECT] {order.code} {order.side}: {reason}")
        return order
    
    def calculate_slippage_rate(self, order_qty: int, daily_vol: float) -> float:
        """
        计算动态滑点率
        """
        if daily_vol <= 0:
            return self.slippage_rate
            
        ratio = order_qty / (daily_vol + 1e-9)
        
        if ratio < 0.01:
            rate = 0.0001 # 1bp
        elif ratio < 0.05:
            rate = 0.0003 # 3bp
        else:
            rate = 0.0005 + (ratio - 0.05) * 0.1 # 大单惩罚
            
        return max(rate, self.slippage_rate)

    def calculate_slippage(self, price: float, side: str) -> float:
        """计算滑点"""
        if side == "BUY":
            return price * self.slippage_rate
        else:
            return -price * self.slippage_rate
    
    def calculate_commission(self, trade_value: float, side: str) -> float:
        """计算手续费"""
        comm = max(trade_value * self.commission_rate, self.min_commission)
        if side == "SELL":
            comm += trade_value * self.stamp_duty
        return comm