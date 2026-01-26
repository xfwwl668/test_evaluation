# ============================================================================
# 文件: analysis/performance.py
# ============================================================================
"""
绩效分析器 - 策略评估指标计算
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    # 收益
    total_return: float
    annual_return: float
    
    # 风险
    max_drawdown: float
    volatility: float
    
    # 比率
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # 交易统计
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    avg_holding_days: float
    
    # 其他
    best_day: float
    worst_day: float
    positive_days: int
    negative_days: int


class PerformanceAnalyzer:
    """
    绩效分析器
    
    计算策略回测的各项指标
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        """
        Args:
            risk_free_rate: 无风险利率 (年化)
        """
        self.rf = risk_free_rate
        self.logger = logging.getLogger("Performance")
    
    def analyze(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame = None,
        initial_capital: float = 1_000_000
    ) -> PerformanceMetrics:
        """
        计算绩效指标
        
        Args:
            equity_curve: 权益曲线 (需含 equity 列)
            trades: 交易记录
            initial_capital: 初始资金
        """
        if equity_curve.empty:
            return self._empty_metrics()
        
        equity = equity_curve['equity'].values
        
        # 日收益率
        if 'daily_return' in equity_curve.columns:
            returns = equity_curve['daily_return'].values
        else:
            returns = np.diff(equity) / equity[:-1]
            returns = np.concatenate([[0], returns])
        
        # 交易日数
        n_days = len(equity)
        n_years = n_days / 252
        
        # ====== 收益指标 ======
        total_return = (equity[-1] / initial_capital) - 1
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        
        # ====== 风险指标 ======
        # 最大回撤
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()
        
        # 波动率
        volatility = returns.std() * np.sqrt(252)
        
        # ====== 风险调整收益 ======
        # Sharpe
        excess_return = returns.mean() * 252 - self.rf
        sharpe_ratio = excess_return / (volatility + 1e-10)
        
        # Sortino
        neg_returns = returns[returns < 0]
        downside_std = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else volatility
        sortino_ratio = excess_return / (downside_std + 1e-10)
        
        # Calmar
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # ====== 日统计 ======
        positive_days = (returns > 0).sum()
        negative_days = (returns < 0).sum()
        best_day = returns.max()
        worst_day = returns.min()
        
        # ====== 交易统计 ======
        win_rate = 0.0
        profit_loss_ratio = 0.0
        total_trades = 0
        avg_holding_days = 0.0
        
        if trades is not None and not trades.empty:
            total_trades = len(trades)
            
            # 简化胜率计算 (日度)
            win_rate = positive_days / max(positive_days + negative_days, 1)
            
            # 盈亏比
            avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
            avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        return PerformanceMetrics(
            total_return=round(total_return, 4),
            annual_return=round(annual_return, 4),
            max_drawdown=round(max_drawdown, 4),
            volatility=round(volatility, 4),
            sharpe_ratio=round(sharpe_ratio, 3),
            sortino_ratio=round(sortino_ratio, 3),
            calmar_ratio=round(calmar_ratio, 3),
            win_rate=round(win_rate, 4),
            profit_loss_ratio=round(profit_loss_ratio, 3),
            total_trades=total_trades,
            avg_holding_days=round(avg_holding_days, 1),
            best_day=round(best_day, 4),
            worst_day=round(worst_day, 4),
            positive_days=positive_days,
            negative_days=negative_days
        )
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """空指标"""
        return PerformanceMetrics(
            total_return=0, annual_return=0, max_drawdown=0, volatility=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            win_rate=0, profit_loss_ratio=0, total_trades=0, avg_holding_days=0,
            best_day=0, worst_day=0, positive_days=0, negative_days=0
        )
    
    def compare(
        self,
        strategies: Dict[str, pd.DataFrame],
        initial_capital: float = 1_000_000
    ) -> pd.DataFrame:
        """
        对比多策略绩效
        
        Args:
            strategies: {name: equity_curve}
        """
        rows = []
        
        for name, curve in strategies.items():
            metrics = self.analyze(curve, initial_capital=initial_capital)
            rows.append({
                'Strategy': name,
                'Total Return': f"{metrics.total_return:.2%}",
                'Annual Return': f"{metrics.annual_return:.2%}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Sharpe': f"{metrics.sharpe_ratio:.2f}",
                'Sortino': f"{metrics.sortino_ratio:.2f}",
                'Calmar': f"{metrics.calmar_ratio:.2f}",
                'Win Rate': f"{metrics.win_rate:.2%}"
            })
        
        return pd.DataFrame(rows).set_index('Strategy')
    
    def monthly_returns(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """计算月度收益表"""
        if 'equity' not in equity_curve.columns:
            return pd.DataFrame()
        
        df = equity_curve.copy()
        df['month'] = df.index.to_period('M')
        
        monthly = df.groupby('month')['equity'].agg(['first', 'last'])
        monthly['return'] = (monthly['last'] / monthly['first']) - 1
        
        # 转为年月矩阵
        monthly = monthly.reset_index()
        monthly['year'] = monthly['month'].dt.year
        monthly['mon'] = monthly['month'].dt.month
        
        pivot = monthly.pivot(index='year', columns='mon', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
        
        return pivot