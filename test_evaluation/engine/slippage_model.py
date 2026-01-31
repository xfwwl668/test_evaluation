# ============================================================================
# File: engine/slippage_model.py
# ============================================================================
"""
高级滑点模型 - Almgren-Chriss 市场冲击模型

基于 Almgren-Chriss 模型实现动态滑点计算:
参考: https://arxiv.org/pdf/math/0305152.pdf

核心因素:
1. 日均成交量占比 (VWAP)
2. 时间加权平均价 (TWAP)
3. 日内成交时间段差异 (早盘高流动性)
4. 个股流动性评分 (根据历史波动)

实现:
- Permanent market impact: γ * X / V
- Temporary market impact: η * ν
- VWAP/TWAP 择时优化
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict
from dataclasses import dataclass
import logging


@dataclass
class LiquidityProfile:
    """流动性特征"""
    avg_daily_volume: float       # 日均成交量
    volatility: float             # 波动率
    spread_bps: float             # 买卖价差 (bp)
    price_impact_factor: float    # 价格冲击系数
    
    @property
    def liquidity_score(self) -> float:
        """
        流动性评分 (0-100)
        """
        score = 50.0
        
        # 成交量评分 (越高越好)
        if self.avg_daily_volume > 10_000_000:
            score += 20
        elif self.avg_daily_volume > 5_000_000:
            score += 10
        elif self.avg_daily_volume < 1_000_000:
            score -= 20
        
        # 波动率评分 (越低越好)
        if self.volatility < 0.15:
            score += 15
        elif self.volatility > 0.35:
            score -= 15
        
        # 价差评分 (越小越好)
        if self.spread_bps < 5:
            score += 15
        elif self.spread_bps > 20:
            score -= 15
        
        return max(0, min(100, score))


class AlmgrenChrissModel:
    """
    Almgren-Chriss 市场冲击模型
    
    滑点 = permanent_impact + temporary_impact
    
    where:
    - permanent_impact = γ * (X / V) * price
    - temporary_impact = η * (ν / V) * price
    
    Parameters:
    - γ: permanent impact coefficient (永久冲击系数)
    - η: temporary impact coefficient (临时冲击系数)
    - X: total order size (总订单量)
    - V: average daily volume (日均成交量)
    - ν: trading rate (交易速率)
    """
    
    def __init__(
        self,
        gamma: float = 0.0001,      # 永久冲击系数
        eta: float = 0.0005,        # 临时冲击系数
        time_window: float = 0.1    # 执行时间窗口 (0-1, 1=全天)
    ):
        self.gamma = gamma
        self.eta = eta
        self.time_window = time_window
        self.logger = logging.getLogger("AlmgrenChrissModel")
    
    def calculate_market_impact(
        self,
        order_quantity: int,
        avg_daily_volume: float,
        price: float,
        side: str
    ) -> Dict[str, float]:
        """
        计算市场冲击
        
        Args:
            order_quantity: 订单数量
            avg_daily_volume: 日均成交量
            price: 当前价格
            side: BUY/SELL
        
        Returns:
            {
                'permanent_impact': 永久冲击 (price)
                'temporary_impact': 临时冲击 (price)
                'total_impact': 总冲击 (price)
                'impact_bps': 冲击 (bp)
            }
        """
        if avg_daily_volume <= 0:
            return {
                'permanent_impact': 0.0,
                'temporary_impact': 0.0,
                'total_impact': 0.0,
                'impact_bps': 0.0
            }
        
        # 订单占比
        participation_rate = order_quantity / avg_daily_volume
        
        # 永久冲击 (随持仓规模线性增长)
        permanent_impact = self.gamma * participation_rate * price
        
        # 临时冲击 (随执行速率增长)
        execution_rate = participation_rate / self.time_window
        temporary_impact = self.eta * execution_rate * price
        
        total_impact = permanent_impact + temporary_impact
        
        # 买入冲击为正，卖出冲击为负
        if side.upper() == 'SELL':
            total_impact = -total_impact
        
        impact_bps = abs(total_impact / price) * 10000
        
        self.logger.debug(
            f"Market Impact: {side} {order_quantity} @ {price:.2f} "
            f"part_rate={participation_rate:.4f} "
            f"imp={impact_bps:.2f}bp"
        )
        
        return {
            'permanent_impact': permanent_impact,
            'temporary_impact': temporary_impact,
            'total_impact': total_impact,
            'impact_bps': impact_bps
        }


class AdvancedSlippageModel:
    """
    高级滑点模型
    
    整合多种因素:
    1. Almgren-Chriss 市场冲击
    2. 日内时间段调整
    3. 流动性评分
    4. VWAP/TWAP 优化
    """
    
    # 日内时间段权重 (相对开盘价的倍数)
    # A股: 早盘9:30-10:30流动性最好
    INTRADAY_WEIGHTS = {
        '09:30-10:30': 1.0,   # 开盘后1小时 (流动性最佳)
        '10:30-11:30': 0.8,   # 上午后段
        '13:00-14:00': 0.7,   # 开盘后
        '14:00-15:00': 0.9,   # 收盘前1小时 (流动性回升)
    }
    
    def __init__(
        self,
        base_slippage_rate: float = 0.0001,
        use_almgren_chriss: bool = True,
        max_slippage_bps: float = 20.0  # 最大滑点 20bp
    ):
        self.base_slippage_rate = base_slippage_rate
        self.use_almgren_chriss = use_almgren_chriss
        self.max_slippage_bps = max_slippage_bps
        
        self.ac_model = AlmgrenChrissModel()
        self.logger = logging.getLogger("AdvancedSlippageModel")
    
    def calculate_slippage(
        self,
        order_quantity: int,
        price: float,
        side: str,
        daily_volume: Optional[float] = None,
        volatility: Optional[float] = None,
        execution_time: Optional[str] = None,
        liquidity_profile: Optional[LiquidityProfile] = None
    ) -> float:
        """
        计算滑点 (综合考虑多种因素)
        
        Args:
            order_quantity: 订单数量
            price: 当前价格
            side: BUY/SELL
            daily_volume: 日成交量 (可选)
            volatility: 波动率 (可选)
            execution_time: 执行时间段 (可选, 格式: "HH:MM")
            liquidity_profile: 流动性特征 (可选)
        
        Returns:
            滑点金额 (price * slippage_rate)
        """
        slippage_rate = self.base_slippage_rate
        
        # 1. 基础滑点 (买入加, 卖出减)
        base_slippage = price * slippage_rate
        if side.upper() == 'SELL':
            base_slippage = -base_slippage
        
        # 2. 市场冲击模型
        if self.use_almgren_chriss and daily_volume:
            impact = self.ac_model.calculate_market_impact(
                order_quantity=order_quantity,
                avg_daily_volume=daily_volume,
                price=price,
                side=side
            )
            market_impact = impact['total_impact']
            
            # 检查是否超过最大滑点
            impact_bps = impact['impact_bps']
            if impact_bps > self.max_slippage_bps:
                self.logger.warning(
                    f"Market impact too high: {impact_bps:.2f}bp > {self.max_slippage_bps:.2f}bp"
                )
                # 限制到最大值
                market_impact = (self.max_slippage_bps / 10000) * price
                if side.upper() == 'SELL':
                    market_impact = -market_impact
            
            base_slippage += market_impact
        
        # 3. 流动性调整
        if liquidity_profile:
            score = liquidity_profile.liquidity_score
            # 低流动性增加滑点
            if score < 40:
                liquidity_factor = 1.5
            elif score < 60:
                liquidity_factor = 1.2
            else:
                liquidity_factor = 1.0
            
            base_slippage *= liquidity_factor
        
        # 4. 波动率调整
        if volatility:
            # 高波动增加滑点
            vol_factor = min(2.0, 1.0 + volatility * 2.0)
            base_slippage *= vol_factor
        
        # 5. 日内时间段调整
        if execution_time:
            time_weight = self._get_intraday_weight(execution_time)
            base_slippage *= time_weight
        
        return round(base_slippage, 4)
    
    def _get_intraday_weight(self, execution_time: str) -> float:
        """
        获取日内时间段权重
        
        Args:
            execution_time: 执行时间 (HH:MM)
        
        Returns:
            权重因子 (0.7-1.0)
        """
        if not execution_time or ':' not in execution_time:
            return 1.0
        
        try:
            hour_min = execution_time.split(':')
            hour = int(hour_min[0])
            minute = int(hour_min[1])
            
            # 简化判断
            if hour == 9 and minute >= 30:  # 9:30-10:30
                return 1.0
            elif hour == 10:  # 10:30-11:30
                return 0.8
            elif hour == 13:  # 13:00-14:00
                return 0.7
            elif hour == 14:  # 14:00-15:00
                return 0.9
            else:
                return 0.85
        except:
            return 1.0
    
    def calculate_slippage_rate(
        self,
        order_quantity: int,
        daily_volume: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        计算滑点率 (简化版，用于 matcher)
        
        Args:
            order_quantity: 订单数量
            daily_volume: 日成交量
            volatility: 波动率 (可选)
        
        Returns:
            滑点率 (0-1)
        """
        if daily_volume <= 0:
            return self.base_slippage_rate
        
        ratio = order_quantity / daily_volume
        
        # 基础滑点率
        if ratio < 0.01:
            rate = 0.0001  # 1bp
        elif ratio < 0.05:
            rate = 0.0003  # 3bp
        else:
            rate = 0.0005 + (ratio - 0.05) * 0.1  # 大单惩罚
        
        # 波动率调整
        if volatility:
            vol_factor = min(2.0, 1.0 + volatility * 2.0)
            rate *= vol_factor
        
        return max(rate, self.base_slippage_rate)
    
    def estimate_liquidity_profile(
        self,
        historical_data: pd.DataFrame
    ) -> LiquidityProfile:
        """
        估算流动性特征
        
        Args:
            historical_data: 历史行情数据 (OHLCV)
        
        Returns:
            LiquidityProfile 对象
        """
        if historical_data.empty:
            return LiquidityProfile(
                avg_daily_volume=1_000_000,
                volatility=0.2,
                spread_bps=10.0,
                price_impact_factor=0.0001
            )
        
        # 日均成交量
        avg_volume = historical_data['volume'].mean()
        
        # 波动率 (年化)
        returns = historical_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2
        
        # 买卖价差 (简化: (high - low) / close)
        spread = ((historical_data['high'] - historical_data['low']) / historical_data['close']).mean() * 10000
        spread_bps = max(1.0, min(100.0, spread))
        
        # 价格冲击系数 (基于波动率)
        price_impact_factor = 0.0001 * (1 + volatility * 5)
        
        return LiquidityProfile(
            avg_daily_volume=avg_volume,
            volatility=volatility,
            spread_bps=spread_bps,
            price_impact_factor=price_impact_factor
        )
