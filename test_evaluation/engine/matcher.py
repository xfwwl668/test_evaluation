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
from .slippage_model import AdvancedSlippageModel


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
    - 支持部分成交
    - 添加 unfilled_quantity
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
    
    # 🔴 修复 Problem 4: 部分成交支持
    unfilled_quantity: int = 0     # 未成交量
    is_partial_fill: bool = False   # 是否部分成交
    
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
    
    @property
    def fill_ratio(self) -> float:
        """成交比例"""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity
    
    def update_partial_fill(self, filled_qty: int, price: float) -> None:
        """
        更新部分成交
        
        Args:
            filled_qty: 新增成交量
            price: 成交价格
        """
        if filled_qty <= 0:
            return
        
        # 更新成交量
        old_filled = self.filled_quantity
        self.filled_quantity += filled_qty
        self.unfilled_quantity = self.quantity - self.filled_quantity
        
        # 更新成交价 (加权平均)
        if old_filled > 0:
            total_value = self.filled_price * old_filled + price * filled_qty
            self.filled_price = total_value / self.filled_quantity
        else:
            self.filled_price = price
        
        # 更新状态
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.is_partial_fill = True
        else:
            self.status = OrderStatus.PARTIAL
            self.is_partial_fill = True
    
    def __repr__(self) -> str:
        fill_info = f"{self.filled_quantity}/{self.quantity}" if self.is_partial_fill else str(self.quantity)
        return (f"Order({self.order_id} {self.side} {self.code} "
                f"qty={fill_info} @ {self.price:.2f} [{self.status.value}])")


class MatchEngine:
    """
    撮合引擎
    
    核心规则:
    1. T+1: 当日买入不可卖出
    2. 涨停: 无法买入 (封板)
    3. 跌停: 无法卖出 (封板)
    4. 滑点: 按开盘价 + 滑点成交
    5. 手续费: 佣金 + 印花税 (卖出)
    6. 部分成交: 大单可能部分成交 (成交量限制)
    
    架构:
    ┌─────────────────────────────────────────────────────────┐
    │                    MatchEngine                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
    │  │ 规则检查    │→ │ 价格撮合    │→ │ 成本计算        │ │
    │  │ T+1/涨跌停  │  │ 开盘+滑点   │  │ 佣金+印花税     │ │
    │  │ 成交量限制  │  │ 部分成交    │  │                  │ │
    │  └─────────────┘  └─────────────┘  └─────────────────┘ │
    └─────────────────────────────────────────────────────────┘
    """
    
    # 🔴 修复 Problem 4: 成交量限制
    MAX_PARTICIPATION_RATE = 0.05  # 最大成交占比 5% (避免大单冲击)
    MIN_PARTIAL_FILL_QTY = 100     # 最小部分成交股数
    
    def __init__(
        self,
        commission_rate: float = None,
        min_commission: float = None,
        stamp_duty: float = None,
        slippage_rate: float = None,
        use_advanced_slippage: bool = True
    ):
        self.commission_rate = commission_rate or settings.backtest.COMMISSION_RATE
        self.min_commission = min_commission or settings.backtest.MIN_COMMISSION
        self.stamp_duty = stamp_duty or settings.backtest.STAMP_DUTY
        self.slippage_rate = slippage_rate or settings.backtest.SLIPPAGE_RATE
        
        # 🔴 修复 Problem 2: 高级滑点模型
        self.use_advanced_slippage = use_advanced_slippage
        if use_advanced_slippage:
            self.advanced_slippage = AdvancedSlippageModel(
                base_slippage_rate=self.slippage_rate
            )
        
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
        daily_vol = market_data.get('volume', market_data.get('vol', 1000000))
        
        # 🔴 修复 Problem 6: 停牌检查
        is_suspended = (daily_vol == 0 and not pd.isna(market_data.get('close')))
        if is_suspended:
            return self._reject(order, "停牌，无法交易")
        
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
        
        # 🔴 修复 Problem 4: 成交量限制检查
        actual_quantity = order.quantity
        participation_rate = order.quantity / (daily_vol + 1e-9)
        
        if participation_rate > self.MAX_PARTICIPATION_RATE:
            # 部分成交
            actual_quantity = int(daily_vol * self.MAX_PARTICIPATION_RATE / 100) * 100
            
            if actual_quantity < self.MIN_PARTIAL_FILL_QTY:
                return self._reject(order, f"成交量过小: 订单{order.quantity} > 日成交量{int(daily_vol)}*5%")
            
            self.logger.warning(
                f"[PARTIAL_FILL] {order.code} 订单{order.quantity}过大，"
                f"部分成交{actual_quantity} (占比{participation_rate:.2%})"
            )
        
        # 4. 计算成交价 (开盘价 + 滑点)
        # 🔴 修复 Problem 2: 高级滑点模型
        volatility = market_data.get('volatility', 0.2)
        
        if self.use_advanced_slippage:
            # 使用高级滑点模型
            slippage = self.advanced_slippage.calculate_slippage(
                order_quantity=actual_quantity,
                price=open_price,
                side=order.side,
                daily_volume=daily_vol,
                volatility=volatility
            )
            
            filled_price = open_price + slippage
            slippage_rate = abs(slippage / open_price)
        else:
            # 使用简单滑点模型
            slippage_rate = self.calculate_slippage_rate(actual_quantity, daily_vol)
            
            if order.side == "BUY":
                slippage = open_price * slippage_rate
                filled_price = open_price + slippage
            else:
                slippage = open_price * slippage_rate
                filled_price = open_price - slippage
        
        # 5. 计算手续费
        trade_value = filled_price * actual_quantity
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # 印花税 (仅卖出)
        stamp = trade_value * self.stamp_duty if order.side == "SELL" else 0.0
        
        # 6. 更新订单
        if actual_quantity < order.quantity:
            # 部分成交
            order.update_partial_fill(actual_quantity, filled_price)
            order.unfilled_quantity = order.quantity - actual_quantity
        else:
            # 完全成交
            order.status = OrderStatus.FILLED
            order.filled_price = round(filled_price, 4)
            order.filled_quantity = actual_quantity
            order.filled_date = current_date
        
        order.slippage = round(slippage * actual_quantity, 2)
        order.commission = round(commission, 2)
        order.stamp_duty = round(stamp, 2)
        
        fill_status = f"PARTIAL ({actual_quantity}/{order.quantity})" if order.is_partial_fill else "FILLED"
        self.logger.debug(
            f"[MATCH] {order.side} {order.code} qty={actual_quantity} "
            f"@ {order.filled_price:.3f} cost={order.total_cost:.2f} "
            f"status={fill_status} date={current_date}"
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