# ============================================================================
# 文件: strategy/base.py
# ============================================================================
"""
策略抽象基类 - 定义策略接口规范
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd
import logging


class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Signal:
    """
    交易信号
    
    策略产生的买卖信号
    """
    code: str                      # 股票代码
    side: OrderSide                # 方向
    weight: float = 1.0            # 目标权重 (0-1)
    price: float = 0.0             # 参考价格
    reason: str = ""               # 信号原因 (用于日志)
    priority: int = 0              # 优先级 (同时多信号时)
    
    def __repr__(self) -> str:
        return f"Signal({self.side.value} {self.code} w={self.weight:.2f} | {self.reason})"


@dataclass
class StrategyContext:
    """
    策略运行上下文
    
    由回测引擎构建，传递给策略
    """
    current_date: str                          # 当前日期
    current_data: pd.DataFrame                 # 当日全市场行情
    history_data: Dict[str, pd.DataFrame]      # 历史数据 {code: df}
    factors: Dict[str, pd.DataFrame]           # 预计算因子 {name: df}
    
    # 持仓信息
    positions: Dict[str, float]                # 当前持仓 {code: quantity}
    cash: float                                # 可用现金
    total_equity: float                        # 总权益
    
    # 配置
    config: Dict[str, Any] = field(default_factory=dict)
    
    def get_factor(self, name: str, code: str) -> Optional[float]:
        """获取指定股票的因子值"""
        if name not in self.factors:
            return None
        factor_df = self.factors[name]
        if code not in factor_df.columns:
            return None
        if self.current_date not in factor_df.index:
            return None
        return factor_df.loc[self.current_date, code]
    
    def get_history(self, code: str, lookback: int = 250) -> pd.DataFrame:
        """获取股票历史数据"""
        if code not in self.history_data:
            return pd.DataFrame()
        return self.history_data[code].tail(lookback)


class BaseStrategy(ABC):
    """
    策略抽象基类
    
    所有策略必须继承此类并实现:
    - name: 策略名称 (类属性)
    - initialize(): 初始化逻辑
    - compute_factors(): 因子计算
    - generate_signals(): 信号生成
    
    使用示例:
    ```python
    class MyStrategy(BaseStrategy):
        name = "my_strategy"
        
        def compute_factors(self, history):
            # 计算因子
            return {"factor1": df1, "factor2": df2}
        
        def generate_signals(self, context):
            # 生成信号
            signals = []
            for code in context.current_data['code']:
                if some_condition:
                    signals.append(Signal(code, OrderSide.BUY, reason="..."))
            return signals
    ```
    """
    
    name: str = "base_strategy"
    version: str = "1.0.0"
    
    def __init__(self, params: Dict = None):
        """
        初始化策略
        
        Args:
            params: 策略参数
        """
        self.params = params or {}
        self.logger = logging.getLogger(f"Strategy.{self.name}")
        self._is_initialized = False
    
    def initialize(self) -> None:
        """
        初始化 (在回测开始前调用)
        
        可重写，用于:
        - 加载外部数据
        - 初始化模型
        - 设置状态变量
        """
        self._is_initialized = True
        self.logger.info(f"Strategy '{self.name}' initialized with params: {self.params}")
    
    @abstractmethod
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        计算因子 (在回测开始时或定期调用)
        
        Args:
            history: {code: DataFrame} 历史数据
        
        Returns:
            {factor_name: DataFrame} 因子数据
            DataFrame: index=date, columns=code
        """
        pass
    
    @abstractmethod
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """
        生成交易信号 (每个交易日调用)
        
        Args:
            context: 策略上下文
        
        Returns:
            信号列表
        """
        pass
    
    def on_order_filled(self, order: 'Order') -> None:
        """
        订单成交回调
        
        可重写，用于:
        - 记录交易
        - 更新内部状态
        - 调整后续逻辑
        """
        self.logger.debug(f"Order filled: {order}")
    
    def on_order_rejected(self, order: 'Order', reason: str) -> None:
        """订单拒绝回调"""
        self.logger.warning(f"Order rejected: {order} | Reason: {reason}")
    
    def on_day_end(self, context: StrategyContext) -> None:
        """
        日终回调
        
        可重写，用于:
        - 记录每日状态
        - 日志输出
        """
        pass
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """获取策略参数"""
        return self.params.get(key, default)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, params={self.params})"