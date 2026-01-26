# ============================================================================
# 文件: engine/risk.py
# ============================================================================
"""
风控模块 - 波动率头寸管理 & 风险控制

核心功能:
1. 波动率仓位计算
2. 风险敞口监控
3. 相关性检查
4. 极端行情保护
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from config import settings


@dataclass
class RiskMetrics:
    """风险指标"""
    total_exposure: float           # 总风险敞口
    single_max_exposure: float      # 单只最大敞口
    var_1d: float                   # 1日VaR
    expected_shortfall: float       # 预期亏损
    position_count: int             # 持仓数量
    correlation_risk: float         # 相关性风险


class PositionSizer:
    """
    波动率仓位计算器
    
    核心公式:
        Position = (Total × Risk%) / (ATR × Entry_Price)
    
    确保每笔交易的潜在亏损不超过总资产的 Risk%
    """
    
    def __init__(
        self,
        risk_per_trade: float = 0.005,    # 单笔风险 0.5%
        atr_multiplier: float = 2.0,       # ATR 倍数 (作为止损距离)
        min_position_pct: float = 0.01,    # 最小仓位 1%
        max_position_pct: float = 0.10     # 最大仓位 10%
    ):
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct
        
        self.logger = logging.getLogger("PositionSizer")
    
    def calculate(
        self,
        total_equity: float,
        entry_price: float,
        atr: float,
        volatility: float = None
    ) -> Tuple[float, int]:
        """
        计算仓位
        
        Args:
            total_equity: 总权益
            entry_price: 入场价格
            atr: ATR 值 (绝对值或百分比)
            volatility: 波动率 (可选，用于调整)
        
        Returns:
            (position_weight, share_count)
        """
        # ATR 处理 (如果是百分比，转为金额)
        if atr < 1:
            atr_amount = atr * entry_price
        else:
            atr_amount = atr
        
        # 止损距离 = ATR × 倍数
        stop_distance = atr_amount * self.atr_multiplier
        
        if stop_distance <= 0:
            return 0.0, 0
        
        # 风险预算
        risk_budget = total_equity * self.risk_per_trade
        
        # 可承受的股数
        shares = int(risk_budget / stop_distance)
        
        # 仓位金额
        position_value = shares * entry_price
        
        # 权重
        weight = position_value / total_equity
        
        # 波动率调整 (高波动降低仓位)
        if volatility is not None and volatility > 0:
            vol_adj = min(1.0, 0.15 / volatility)  # 目标波动率 15%
            weight *= vol_adj
        
        # 限制范围
        weight = max(self.min_position_pct, min(weight, self.max_position_pct))
        
        # 重新计算股数 (整百股)
        shares = int(total_equity * weight / entry_price / 100) * 100
        
        final_weight = (shares * entry_price) / total_equity if total_equity > 0 else 0
        
        self.logger.debug(
            f"Position calc: equity={total_equity:.0f} price={entry_price:.2f} "
            f"ATR={atr_amount:.2f} → weight={final_weight:.2%} shares={shares}"
        )
        
        return final_weight, shares
    
    def calculate_batch(
        self,
        total_equity: float,
        candidates: List[Dict]
    ) -> Dict[str, Tuple[float, int]]:
        """
        批量计算仓位
        
        Args:
            total_equity: 总权益
            candidates: [{'code': str, 'price': float, 'atr': float, 'volatility': float}, ...]
        
        Returns:
            {code: (weight, shares)}
        """
        results = {}
        
        for cand in candidates:
            code = cand['code']
            weight, shares = self.calculate(
                total_equity,
                cand['price'],
                cand.get('atr', cand['price'] * 0.02),
                cand.get('volatility')
            )
            results[code] = (weight, shares)
        
        return results


class RiskManager:
    """
    风险管理器
    
    功能:
    1. 仓位限制检查
    2. 风险敞口监控
    3. 相关性风险
    4. 极端行情保护
    """
    
    def __init__(
        self,
        risk_per_trade: float = 0.005,
        max_single_weight: float = 0.10,
        max_total_weight: float = 0.80,
        max_sector_weight: float = 0.30,
        max_correlation: float = 0.7,
        var_confidence: float = 0.95
    ):
        self.risk_per_trade = risk_per_trade
        self.max_single_weight = max_single_weight
        self.max_total_weight = max_total_weight
        self.max_sector_weight = max_sector_weight
        self.max_correlation = max_correlation
        self.var_confidence = var_confidence
        
        self.position_sizer = PositionSizer(
            risk_per_trade=risk_per_trade,
            max_position_pct=max_single_weight
        )
        
        self.logger = logging.getLogger("RiskManager")
    
    def check_position_limits(
        self,
        new_weight: float,
        current_weights: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        检查仓位限制
        
        Returns:
            (is_allowed, reason)
        """
        # 单只限制
        if new_weight > self.max_single_weight:
            return False, f"超过单只限制 ({new_weight:.1%} > {self.max_single_weight:.1%})"
        
        # 总仓位限制
        total_weight = sum(current_weights.values()) + new_weight
        if total_weight > self.max_total_weight:
            return False, f"超过总仓位限制 ({total_weight:.1%} > {self.max_total_weight:.1%})"
        
        return True, ""
    
    def calculate_risk_metrics(
        self,
        positions: Dict[str, Dict],
        returns_data: pd.DataFrame = None
    ) -> RiskMetrics:
        """
        计算风险指标
        
        Args:
            positions: {code: {'weight': float, 'volatility': float}}
            returns_data: 历史收益率数据 (用于 VaR 计算)
        """
        if not positions:
            return RiskMetrics(0, 0, 0, 0, 0, 0)
        
        weights = [p['weight'] for p in positions.values()]
        vols = [p.get('volatility', 0.3) for p in positions.values()]
        
        total_exposure = sum(weights)
        single_max = max(weights) if weights else 0
        
        # 简化 VaR 计算 (正态分布假设)
        avg_vol = np.average(vols, weights=weights) if weights else 0
        z_score = 1.645  # 95% 置信度
        var_1d = total_exposure * avg_vol / np.sqrt(252) * z_score
        
        # 预期亏损 (简化)
        expected_shortfall = var_1d * 1.25
        
        # 相关性风险 (简化: 假设全相关)
        correlation_risk = total_exposure * avg_vol
        
        return RiskMetrics(
            total_exposure=round(total_exposure, 4),
            single_max_exposure=round(single_max, 4),
            var_1d=round(var_1d, 4),
            expected_shortfall=round(expected_shortfall, 4),
            position_count=len(positions),
            correlation_risk=round(correlation_risk, 4)
        )
    
    def adjust_for_correlation(
        self,
        weights: Dict[str, float],
        correlation_matrix: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        相关性调整仓位
        
        高相关的股票降低仓位
        """
        if correlation_matrix is None or len(weights) <= 1:
            return weights
        
        adjusted = weights.copy()
        
        for code in weights:
            if code not in correlation_matrix.columns:
                continue
            
            # 计算与其他持仓的平均相关性
            other_codes = [c for c in weights if c != code and c in correlation_matrix.columns]
            
            if not other_codes:
                continue
            
            avg_corr = correlation_matrix.loc[code, other_codes].abs().mean()
            
            # 高相关惩罚
            if avg_corr > self.max_correlation:
                penalty = 1 - (avg_corr - self.max_correlation)
                adjusted[code] = weights[code] * penalty
                self.logger.debug(f"{code}: correlation penalty {penalty:.2f}")
        
        return adjusted
    
    def emergency_stop(
        self,
        daily_return: float,
        drawdown: float,
        volatility: float
    ) -> Tuple[bool, str]:
        """
        极端行情保护
        
        检查是否触发熔断
        """
        # 单日暴跌
        if daily_return < -0.05:
            return True, f"单日暴跌 {daily_return:.1%}"
        
        # 回撤过大
        if drawdown > 0.15:
            return True, f"回撤超限 {drawdown:.1%}"
        
        # 波动率异常
        if volatility > 0.5:
            return True, f"波动率异常 {volatility:.1%}"
        
        return False, ""
    
    def generate_risk_report(
        self,
        positions: Dict[str, Dict],
        equity_curve: pd.DataFrame = None
    ) -> str:
        """生成风险报告"""
        metrics = self.calculate_risk_metrics(positions)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                        📊 风险报告                               ║
╠══════════════════════════════════════════════════════════════════╣
║  持仓数量:      {metrics.position_count:>10d}                              ║
║  总敞口:        {metrics.total_exposure:>10.2%}                              ║
║  最大单只:      {metrics.single_max_exposure:>10.2%}                              ║
║  1日 VaR (95%): {metrics.var_1d:>10.2%}                              ║
║  预期亏损:      {metrics.expected_shortfall:>10.2%}                              ║
╚══════════════════════════════════════════════════════════════════╝
"""
        return report


class DrawdownProtector:
    """
    回撤保护器
    
    动态调整仓位应对回撤
    """
    
    def __init__(
        self,
        warning_level: float = 0.05,    # 警告回撤
        reduce_level: float = 0.08,     # 减仓回撤
        stop_level: float = 0.12,       # 停止回撤
        recovery_level: float = 0.03    # 恢复回撤
    ):
        self.warning_level = warning_level
        self.reduce_level = reduce_level
        self.stop_level = stop_level
        self.recovery_level = recovery_level
        
        self.is_protecting = False
        self.logger = logging.getLogger("DrawdownProtector")
    
    def get_position_multiplier(self, current_drawdown: float) -> float:
        """
        根据回撤获取仓位乘数
        
        Returns:
            0.0 - 1.0 的乘数
        """
        if current_drawdown >= self.stop_level:
            self.is_protecting = True
            return 0.0  # 完全停止
        
        elif current_drawdown >= self.reduce_level:
            self.is_protecting = True
            return 0.5  # 半仓
        
        elif current_drawdown >= self.warning_level:
            # 线性减仓
            reduction = (current_drawdown - self.warning_level) / (self.reduce_level - self.warning_level)
            return 1.0 - 0.5 * reduction
        
        else:
            # 检查是否从保护状态恢复
            if self.is_protecting and current_drawdown <= self.recovery_level:
                self.is_protecting = False
                self.logger.info("Drawdown protection lifted")
            
            return 1.0 if not self.is_protecting else 0.7