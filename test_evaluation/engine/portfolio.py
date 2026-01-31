# ============================================================================
# 文件: engine/portfolio.py
# ============================================================================
"""
持仓管理器 - 资金与持仓管理
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from .matcher import Order, OrderStatus, MatchEngine
from config import settings


@dataclass
class Position:
    """
    持仓对象
    
    记录单只股票的持仓信息
    """
    code: str
    quantity: int
    avg_cost: float                 # 平均成本 (含手续费)
    buy_date: str                   # 最近买入日期 (T+1用)
    
    market_value: float = 0.0       # 当前市值
    unrealized_pnl: float = 0.0     # 未实现盈亏
    
    # 🔴 修复 Problem 20: 停牌追踪
    suspension_days: int = 0         # 连续停牌天数
    
    @property
    def cost_value(self) -> float:
        """成本价值"""
        return self.avg_cost * self.quantity
    
    @property
    def pnl_ratio(self) -> float:
        """盈亏比例"""
        if self.cost_value <= 0:
            return 0.0
        return self.unrealized_pnl / self.cost_value
    
    def update_market_value(self, price: float) -> None:
        """更新市值"""
        self.market_value = price * self.quantity
        self.unrealized_pnl = self.market_value - self.cost_value


@dataclass
class EquitySnapshot:
    """权益快照"""
    date: str
    total_equity: float
    cash: float
    market_value: float
    positions_count: int
    daily_return: float = 0.0
    drawdown: float = 0.0


class PortfolioManager:
    """
    持仓管理器
    
    功能:
    - 持仓跟踪与更新
    - 资金管理
    - 权益计算
    - 目标权重调仓
    
    架构:
    ┌─────────────────────────────────────────────────────────────┐
    │                   PortfolioManager                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
    │  │ Positions   │  │ Cash        │  │ Equity Curve        │ │
    │  │ {code: pos} │  │ 可用资金    │  │ [snapshots]         │ │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘ │
    │                         │                                   │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │              Target Weight Rebalancer                   ││
    │  │   {code: weight} → 计算买卖订单 → 执行调仓              ││
    │  └─────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or settings.backtest.INITIAL_CAPITAL
        self.cash = self.initial_capital
        
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[EquitySnapshot] = []
        self.trade_history: List[Order] = []
        
        # 🔴 修复 Problem 13: 舍入误差补偿
        self.rounding_errors: Dict[str, float] = {} # {code: shares_diff}
        
        self._peak_equity = self.initial_capital
        self._last_equity = self.initial_capital  # 缓存上次权益值
        self._cached_drawdown = 0.0  # 缓存回撤值
        
        self.logger = logging.getLogger("Portfolio")
    
    # ==================== 属性 ====================
    
    @property
    def total_market_value(self) -> float:
        """总市值"""
        return sum(p.market_value for p in self.positions.values())
    
    @property
    def total_equity(self) -> float:
        """总权益"""
        return self.cash + self.total_market_value
    
    @property
    def position_codes(self) -> List[str]:
        """持仓股票代码"""
        return list(self.positions.keys())
    
    @property
    def positions_count(self) -> int:
        """持仓数量"""
        return len(self.positions)
    
    @property
    def current_drawdown(self) -> float:
        """当前回撤 (优化: 缓存计算)"""
        current_equity = self.total_equity
        
        # 只有在权益变化时重新计算
        if current_equity != self._last_equity:
            if self._peak_equity <= 0:
                self._cached_drawdown = 0.0
            else:
                self._cached_drawdown = (self._peak_equity - current_equity) / self._peak_equity
            self._last_equity = current_equity
        
        return self._cached_drawdown
    
    # ==================== 查询方法 ====================
    
    def get_position(self, code: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(code)
    
    def get_weight(self, code: str) -> float:
        """获取持仓权重"""
        if code not in self.positions or self.total_equity <= 0:
            return 0.0
        return self.positions[code].market_value / self.total_equity
    
    def get_all_weights(self) -> Dict[str, float]:
        """获取所有持仓权重"""
        if self.total_equity <= 0:
            return {}
        return {
            code: pos.market_value / self.total_equity
            for code, pos in self.positions.items()
        }
    
    # ==================== 更新方法 ====================
    
    def update_market_value(self, market_data: pd.DataFrame) -> None:
        """更新持仓市值 (向量化)"""
        # 创建价格映射 (向量化)
        price_map = dict(zip(market_data['code'], market_data['close']))
        
        # 批量更新持仓市值
        for code, pos in self.positions.items():
            if code in price_map:
                pos.update_market_value(price_map[code])
                pos.suspension_days = 0
            else:
                # 🔴 修复 Problem 20: 记录停牌天数
                pos.suspension_days += 1
                # 市值保持不变 (或者根据市场大盘波动调整，这里简单处理保持不变)
    
    def apply_order(self, order: Order, current_date: str) -> None:
        """应用已成交订单"""
        if order.status != OrderStatus.FILLED:
            return
        
        code = order.code
        
        if order.side == "BUY":
            self._apply_buy(order, current_date)
        else:
            self._apply_sell(order)
        
        self.trade_history.append(order)
    
    def _apply_buy(self, order: Order, current_date: str) -> None:
        """应用买入订单"""
        code = order.code
        cost = order.trade_value + order.total_cost
        
        if code in self.positions:
            pos = self.positions[code]
            total_cost = pos.avg_cost * pos.quantity + cost
            total_qty = pos.quantity + order.filled_quantity
            pos.avg_cost = total_cost / total_qty
            pos.quantity = total_qty
            pos.buy_date = current_date
        else:
            self.positions[code] = Position(
                code=code,
                quantity=order.filled_quantity,
                avg_cost=cost / order.filled_quantity,
                buy_date=current_date
            )
        
        self.cash -= cost
    
    def _apply_sell(self, order: Order) -> None:
        """应用卖出订单"""
        code = order.code
        revenue = order.trade_value - order.total_cost
        
        self.cash += revenue
        
        pos = self.positions[code]
        pos.quantity -= order.filled_quantity
        
        if pos.quantity <= 0:
            del self.positions[code]
    
    def record_snapshot(self, current_date: str) -> None:
        """记录权益快照 (优化: 更新缓存)"""
        equity = self.total_equity
        
        # 更新峰值
        if equity > self._peak_equity:
            self._peak_equity = equity
        
        # 计算日收益
        daily_return = 0.0
        if self.equity_curve:
            prev_equity = self.equity_curve[-1].total_equity
            if prev_equity > 0:
                daily_return = (equity - prev_equity) / prev_equity
        
        # 更新缓存
        self._last_equity = equity
        
        snapshot = EquitySnapshot(
            date=current_date,
            total_equity=equity,
            cash=self.cash,
            market_value=self.total_market_value,
            positions_count=self.positions_count,
            daily_return=daily_return,
            drawdown=self.current_drawdown
        )
        
        self.equity_curve.append(snapshot)
    
    # ==================== 调仓计算 ====================
    
    def calculate_rebalance_orders(
        self,
        target_weights: Dict[str, float],
        market_data: pd.DataFrame,
        current_date: str,
        match_engine: MatchEngine
    ) -> List[Order]:
        """
        计算调仓订单
        
        Args:
            target_weights: {code: weight} 目标权重
            market_data: 当日行情
            current_date: 当前日期
            match_engine: 撮合引擎
        
        Returns:
            订单列表 (先卖后买)
        """
        orders = []
        price_map = dict(zip(market_data['code'], market_data['close']))
        
        total_equity = self.total_equity
        max_weight = settings.backtest.MAX_POSITION_WEIGHT
        cash_reserve = settings.backtest.CASH_RESERVE
        
        # 1. 计算卖出订单 (减仓/清仓)
        for code in list(self.positions.keys()):
            current_weight = self.get_weight(code)
            target_weight = target_weights.get(code, 0.0)
            
            if target_weight < current_weight * 0.95:  # 需要减仓
                pos = self.positions[code]
                
                if code not in price_map:
                    continue
                
                price = price_map[code]
                
                if target_weight == 0:
                    # 清仓
                    sell_qty = pos.quantity
                    reason = "清仓"
                    self.rounding_errors[code] = 0
                else:
                    # 减仓
                    target_value = total_equity * target_weight
                    target_qty = target_value / price
                    
                    # 🔴 补偿之前的误差
                    target_qty += self.rounding_errors.get(code, 0)
                    
                    diff_qty = pos.quantity - target_qty
                    sell_qty = int(diff_qty / 100) * 100
                    
                    # 记录新误差
                    self.rounding_errors[code] = diff_qty - sell_qty
                    reason = f"减仓 {current_weight:.1%}→{target_weight:.1%}"
                
                if sell_qty >= 100:
                    order = match_engine.create_order(
                        code=code,
                        side="SELL",
                        price=price,
                        quantity=sell_qty,
                        create_date=current_date,
                        signal_reason=reason
                    )
                    orders.append(order)
        
        # 2. 计算买入订单 (加仓/建仓)
        available_cash = self.cash * (1 - cash_reserve)
        
        for code, target_weight in target_weights.items():
            if target_weight <= 0:
                continue
            
            # 限制单只权重
            target_weight = min(target_weight, max_weight)
            
            current_weight = self.get_weight(code)
            
            if target_weight > current_weight * 1.05:  # 需要加仓
                if code not in price_map:
                    continue
                
                price = price_map[code]
                
                target_value = total_equity * target_weight
                target_qty = target_value / price
                
                # 🔴 补偿之前的误差
                target_qty -= self.rounding_errors.get(code, 0)
                
                current_qty = self.positions[code].quantity if code in self.positions else 0
                buy_qty = int((target_qty - current_qty) / 100) * 100
                
                # 检查现金
                if buy_qty * price > available_cash:
                    buy_qty = int(available_cash / price / 100) * 100
                
                if buy_qty >= 100:
                    available_cash -= buy_qty * price
                    # 记录新误差
                    self.rounding_errors[code] = buy_qty - (target_qty - current_qty)
                    
                    reason = f"{'加仓' if code in self.positions else '建仓'} →{target_weight:.1%}"
                    
                    order = match_engine.create_order(
                        code=code,
                        side="BUY",
                        price=price,
                        quantity=buy_qty,
                        create_date=current_date,
                        signal_reason=reason
                    )
                    orders.append(order)
        
        return orders
    
    # ==================== 数据导出 ====================
    
    def get_equity_df(self) -> pd.DataFrame:
        """获取权益曲线 DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'date': s.date,
                'equity': s.total_equity,
                'cash': s.cash,
                'market_value': s.market_value,
                'positions': s.positions_count,
                'daily_return': s.daily_return,
                'drawdown': s.drawdown
            }
            for s in self.equity_curve
        ])
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录 DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'order_id': o.order_id,
                'date': o.create_date,
                'code': o.code,
                'side': o.side,
                'quantity': o.filled_quantity,
                'price': o.filled_price,
                'value': o.trade_value,
                'commission': o.commission,
                'slippage': o.slippage,
                'stamp_duty': o.stamp_duty,
                'total_cost': o.total_cost,
                'reason': o.signal_reason
            }
            for o in self.trade_history
            if o.status == OrderStatus.FILLED
        ])