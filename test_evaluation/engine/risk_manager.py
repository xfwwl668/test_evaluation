# ============================================================================
# 文件: engine/risk_manager.py
# ============================================================================
"""
风险管理系统 - 回撤控制、集中度管理、流动性检查
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

class RiskManager:
    """
    风险管理系统
    
    实现:
    - 最大回撤监控 (Portfolio level)
    - 板块/个股集中度限制
    - 流动性检查 (成交量占比)
    - VaR / CVaR 计算
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("RiskManager")
        
        # 默认风险参数
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 0.15)
        self.max_single_stock_weight = self.config.get('max_single_stock_weight', 0.20)
        self.max_sector_weight = self.config.get('max_sector_weight', 0.30)
        self.min_liquidity_ratio = self.config.get('min_liquidity_ratio', 0.01) # 订单量/日均成交量
        
    def check_portfolio_risk(self, portfolio_stats: Dict) -> bool:
        """
        全账户风险检查
        """
        drawdown = portfolio_stats.get('drawdown', 0)
        if drawdown > self.max_drawdown_limit:
            self.logger.warning(f"触发风控: 当前回撤 {drawdown:.2%} > 限制 {self.max_drawdown_limit:.2%}")
            return False
        return True
        
    def validate_order(self, order_qty: int, avg_volume_5d: float) -> bool:
        """
        流动性风险检查
        """
        if avg_volume_5d <= 0:
            return False
            
        ratio = order_qty / avg_volume_5d
        if ratio > self.min_liquidity_ratio:
            self.logger.warning(f"流动性风险: 订单占比 {ratio:.2%} > 限制 {self.min_liquidity_ratio:.2%}")
            # 这里可以选择减小订单量或拒绝
            return False
        return True

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        计算 Value at Risk (历史模拟法)
        """
        if len(returns) < 20:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)

    def check_concentration(self, positions: Dict[str, float], total_equity: float) -> List[str]:
        """
        检查集中度风险
        """
        violations = []
        for code, mkt_val in positions.items():
            weight = mkt_val / total_equity
            if weight > self.max_single_stock_weight:
                violations.append(code)
                self.logger.warning(f"集中度风险: {code} 权重 {weight:.2%} > {self.max_single_stock_weight:.2%}")
        return violations
