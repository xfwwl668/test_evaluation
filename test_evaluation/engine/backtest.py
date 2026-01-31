# ============================================================================
# 文件: engine/backtest.py
# ============================================================================
"""
回测引擎 - 核心调度器
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Type
from datetime import datetime
import logging

from .matcher import MatchEngine, Order, OrderStatus
from .portfolio import PortfolioManager, Position
from .risk_manager import RiskManager
from strategy import BaseStrategy, StrategyContext, Signal, OrderSide
from core.database import StockDatabase
from core.data_validator import DataValidator
from config import settings


class BacktestEngine:
    """
    回测引擎 - 策略评测核心
    """
    
    def __init__(
        self,
        db_path: str = None,
        initial_capital: float = None,
        commission_rate: float = None,
        slippage_rate: float = None
    ):
        self.db_path = db_path or str(settings.path.DB_PATH)
        self.initial_capital = initial_capital or settings.backtest.INITIAL_CAPITAL
        
        # 组件
        self.db = StockDatabase(self.db_path)
        self.validator = DataValidator()
        self.risk_manager = RiskManager()
        self.match_engine = MatchEngine(
            commission_rate=commission_rate,
            slippage_rate=slippage_rate
        )
        
        # 策略容器 (支持多策略对比)
        self.strategies: Dict[str, Tuple[BaseStrategy, PortfolioManager]] = {}
        
        # 数据
        self.trading_dates: List[str] = []
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._market_data: pd.DataFrame = None
        
        self.logger = logging.getLogger("BacktestEngine")
    
    def add_strategy(self, strategy: BaseStrategy) -> 'BacktestEngine':
        """添加策略"""
        portfolio = PortfolioManager(self.initial_capital)
        self.strategies[strategy.name] = (strategy, portfolio)
        self.logger.info(f"Added strategy: {strategy.name}")
        return self
    
    def run(
        self,
        start_date: str,
        end_date: str,
        codes: List[str] = None,
        rebalance_freq: str = None
    ) -> Dict[str, 'BacktestResult']:
        """
        运行回测
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期
            codes: 股票池 (None=全市场)
            rebalance_freq: 调仓频率 (D/W/M)
        
        Returns:
            {strategy_name: BacktestResult}
        """
        rebalance_freq = rebalance_freq or settings.backtest.REBALANCE_FREQ
        
        self.logger.info(f"Starting backtest: {start_date} to {end_date}")
        self.logger.info(f"Strategies: {list(self.strategies.keys())}")
        
        # 1. 加载数据
        self._load_data(start_date, end_date, codes)
        
        # 2. 初始化策略 & 预计算因子
        self._initialize_strategies()
        
        # 3. 获取调仓日期
        rebalance_dates = self._get_rebalance_dates(rebalance_freq)
        
        # 4. 逐日回测
        self.logger.info(f"Running {len(self.trading_dates)} trading days...")
        
        for i, current_date in enumerate(self.trading_dates):
            current_data = self._get_daily_data(current_date)
            
            # 数据质量检查
            if i % 100 == 0:
                self.validator.generate_quality_report(current_data)
                
            is_rebalance = current_date in rebalance_dates
            
            for name, (strategy, portfolio) in self.strategies.items():
                # 更新市值
                portfolio.update_market_value(current_data)
                
                # 🔴 修复 Problem 20: 长期停牌强制卖出
                for code, pos in list(portfolio.positions.items()):
                    if pos.suspension_days > 30:
                        self.logger.warning(f"[SUSPEND-EXIT] {code} 停牌超过30天，强制清仓")
                        # 模拟强制成交
                        exit_price = pos.market_value / pos.quantity if pos.quantity > 0 else 0
                        order = self.match_engine.create_order(
                            code=code, side="SELL", price=exit_price,
                            quantity=pos.quantity, create_date=current_date, signal_reason="长期停牌强制卖出"
                        )
                        order.status = OrderStatus.FILLED
                        order.filled_price = exit_price
                        order.filled_quantity = pos.quantity
                        order.filled_date = current_date
                        portfolio.apply_order(order, current_date)
                
                # 🔴 修复 Problem 10: 风险管理
                if not self.risk_manager.check_portfolio_risk({
                    'drawdown': portfolio.current_drawdown,
                    'total_equity': portfolio.total_equity
                }):
                    self.logger.critical(f"[{name}] 触发全局风控，停止该策略调仓")
                    continue
                
                # 调仓日生成信号
                if is_rebalance:
                    # 🔴 修复 Problem 1: 逐日动态计算因子，确保无前向偏差
                    history_for_factors = self._get_history_for_factors(current_date)
                    strategy._factors = strategy.compute_factors(history_for_factors)
                    
                    context = self._build_context(current_date, current_data, portfolio, strategy)
                    signals = strategy.generate_signals(context)
                    
                    if signals:
                        self._execute_signals(signals, current_data, current_date, portfolio, strategy)
                
                # 记录权益
                portfolio.record_snapshot(current_date)
                
                # 日终回调
                strategy.on_day_end(context if is_rebalance else None)
            
            # 进度
            if (i + 1) % 50 == 0:
                self.logger.info(f"  Processed {i+1}/{len(self.trading_dates)} days")
        
        # 5. 生成结果
        results = {}
        for name, (strategy, portfolio) in self.strategies.items():
            results[name] = BacktestResult(
                strategy_name=name,
                portfolio=portfolio,
                initial_capital=self.initial_capital
            )
            results[name].print_summary()
        
        return results
    
    def _load_data(self, start_date: str, end_date: str, codes: List[str]) -> None:
        """加载数据"""
        self.logger.info("Loading market data...")
        
        # 扩展开始日期 (需要历史数据计算因子)
        extended_start = pd.to_datetime(start_date) - pd.DateOffset(years=3)
        extended_start_str = extended_start.strftime('%Y-%m-%d')
        
        # 从数据库加载
        if codes:
            self._market_data = self.db.get_multi_stock_panel(codes, extended_start_str, end_date)
        else:
            self._market_data = self.db.get_market_snapshot(end_date)  # 简化处理
        
        # 🔴 修复 Problem 5: 数据质量验证
        if not self.validator.validate_ohlcv(self._market_data, "MarketData"):
            self.logger.warning("Market data validation failed, but proceeding with caution...")
            
        # 计算涨跌停
        self._market_data = self._add_limit_flags(self._market_data)
        
        # 交易日列表 (只取回测区间)
        all_dates = self._market_data['date'].unique()
        self.trading_dates = sorted([
            d for d in all_dates 
            if start_date <= str(d) <= end_date
        ])
        
        # 按股票缓存历史数据
        for code in self._market_data['code'].unique():
            self._data_cache[code] = self._market_data[
                self._market_data['code'] == code
            ].copy().set_index('date')
        
        self.logger.info(f"Loaded {len(self._data_cache)} stocks, {len(self.trading_dates)} trading days")

    def _get_history_for_factors(self, current_date: str) -> Dict[str, pd.DataFrame]:
        """获取用于因子计算的历史数据 (确保无前向偏差)"""
        history = {}
        for code, cache in self._data_cache.items():
            # 只取 current_date 之前的数据
            # 严格排除当日数据，因为因子计算通常基于历史
            mask = cache.index < current_date
            if mask.any():
                history[code] = cache[mask].tail(250)
        return history
    
    def _add_limit_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加涨跌停标记"""
        df = df.copy()
        df['prev_close'] = df.groupby('code')['close'].shift(1)
        df['limit_up'] = (df['prev_close'] * 1.1).round(2)
        df['limit_down'] = (df['prev_close'] * 0.9).round(2)
        df['is_limit_up'] = df['close'] >= df['limit_up'] - 0.01
        df['is_limit_down'] = df['close'] <= df['limit_down'] + 0.01
        return df
    
    def _initialize_strategies(self) -> None:
        """
        初始化策略
        
        注意: 因子不再一次性预计算，而是逐日动态计算以避免前向偏差
        """
        self.logger.info("Initializing strategies...")
        
        for name, (strategy, _) in self.strategies.items():
            strategy.initialize()
            # 因子计算移到逐日回测中，确保不使用未来数据
            self.logger.info(f"  {name}: initialized (factors will be computed daily)")
    
    def _get_daily_data(self, date: str) -> pd.DataFrame:
        """获取当日数据"""
        return self._market_data[self._market_data['date'] == date].copy()
    
    def _get_rebalance_dates(self, freq: str) -> set:
        """获取调仓日期"""
        dates = pd.to_datetime(self.trading_dates)
        
        if freq == 'D':
            return set(self.trading_dates)
        
        df = pd.DataFrame({'date': dates})
        
        if freq == 'W':
            df['period'] = df['date'].dt.isocalendar().week
        elif freq == 'M':
            df['period'] = df['date'].dt.to_period('M')
        else:
            return set(self.trading_dates)
        
        last_dates = df.groupby('period')['date'].last()
        return set(last_dates.dt.strftime('%Y-%m-%d').tolist())
    
    def _build_context(
        self,
        current_date: str,
        current_data: pd.DataFrame,
        portfolio: PortfolioManager,
        strategy: BaseStrategy
    ) -> StrategyContext:
        """构建策略上下文 (优化历史数据获取)"""
        # 获取历史数据 (优化: 只获取当前日期之前的数据)
        history = {}
        codes_to_fetch = current_data['code'].unique()
        
        for code in codes_to_fetch:
            if code in self._data_cache:
                hist = self._data_cache[code]
                # 使用索引切片直接获取最近250个交易日数据
                history[code] = hist[hist.index <= current_date].tail(250)
        
        # 持仓转换
        positions = {
            code: pos.quantity
            for code, pos in portfolio.positions.items()
        }
        
        return StrategyContext(
            current_date=current_date,
            current_data=current_data,
            history_data=history,
            factors=getattr(strategy, '_factors', {}),
            positions=positions,
            cash=portfolio.cash,
            total_equity=portfolio.total_equity
        )
    
    def _execute_signals(
        self,
        signals: List[Signal],
        market_data: pd.DataFrame,
        current_date: str,
        portfolio: PortfolioManager,
        strategy: BaseStrategy
    ) -> None:
        """执行信号"""
        # 转换为目标权重
        target_weights = self._signals_to_weights(signals)
        
        # 计算订单
        orders = portfolio.calculate_rebalance_orders(
            target_weights, market_data, current_date, self.match_engine
        )
        
        # 撮合执行
        for order in orders:
            code_data = market_data[market_data['code'] == order.code]
            
            if code_data.empty:
                continue
            
            bar = code_data.iloc[0][['open', 'high', 'low', 'close', 'is_limit_up', 'is_limit_down']].copy()
            if bar.isna().any():
                bar = code_data.iloc[0]
            position = portfolio.get_position(order.code)
            
            matched = self.match_engine.match(
                order, bar, position, current_date
            )
            
            if matched.status == OrderStatus.FILLED:
                portfolio.apply_order(matched, current_date)
                strategy.on_order_filled(matched)
            else:
                strategy.on_order_rejected(matched, matched.reject_reason)
    
    def _signals_to_weights(self, signals: List[Signal]) -> Dict[str, float]:
        """信号转权重"""
        weights = {}
        
        # 分离买卖信号
        buy_signals = [s for s in signals if s.side == OrderSide.BUY]
        sell_signals = [s for s in signals if s.side == OrderSide.SELL]
        
        # 卖出信号: 权重=0
        for s in sell_signals:
            weights[s.code] = 0.0
        
        # 买入信号: 归一化权重
        if buy_signals:
            total = sum(s.weight for s in buy_signals)
            reserve = settings.backtest.CASH_RESERVE
            
            for s in buy_signals:
                weights[s.code] = (s.weight / total) * (1 - reserve)
        
        return weights
    
    def compare_strategies(self) -> pd.DataFrame:
        """对比策略净值"""
        curves = {}
        
        for name, (_, portfolio) in self.strategies.items():
            df = portfolio.get_equity_df()
            if not df.empty:
                curves[name] = df['equity'] / self.initial_capital
        
        return pd.DataFrame(curves)


class BacktestResult:
    """回测结果"""
    
    def __init__(
        self,
        strategy_name: str,
        portfolio: PortfolioManager,
        initial_capital: float
    ):
        self.strategy_name = strategy_name
        self.portfolio = portfolio
        self.initial_capital = initial_capital
        
        self._compute_metrics()
    
    def _compute_metrics(self) -> None:
        """计算绩效指标"""
        df = self.portfolio.get_equity_df()
        
        if df.empty:
            self.metrics = {}
            return
        
        equity = df['equity'].values
        returns = df['daily_return'].values
        
        # 收益
        total_return = (equity[-1] / self.initial_capital) - 1
        n_years = len(equity) / 252
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        
        # 风险
        max_drawdown = df['drawdown'].max()
        volatility = returns.std() * np.sqrt(252)
        
        # 比率
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        neg_returns = returns[returns < 0]
        sortino = np.sqrt(252) * returns.mean() / (neg_returns.std() + 1e-10) if len(neg_returns) > 0 else sharpe
        
        # 交易统计 - 修复: 正确计算交易胜率(而非日胜率)
        trades = self.portfolio.get_trades_df()
        win_rate = 0.0
        profit_trades = 0
        loss_trades = 0
        profit_factor = 0.0
        total_profit = 0.0
        total_loss = 0.0
        
        if not trades.empty and 'side' in trades.columns:
            # 🔴 修复 Problem 4: 配对交易计算
            # 记录每只股票的买入队列
            buy_queues = {} # {code: [(qty, price)]}
            
            # 按时间排序
            trades_sorted = trades.sort_values('date')
            
            for _, trade in trades_sorted.iterrows():
                code = trade['code']
                qty = trade['quantity']
                price = trade['price']
                
                if trade['side'] == 'BUY':
                    if code not in buy_queues:
                        buy_queues[code] = []
                    buy_queues[code].append({'qty': qty, 'price': price})
                else:
                    # 卖出，配对买入
                    if code in buy_queues and buy_queues[code]:
                        matched_qty = 0
                        matched_cost = 0.0
                        
                        to_sell = qty
                        while to_sell > 0 and buy_queues[code]:
                            buy = buy_queues[code][0]
                            if buy['qty'] <= to_sell:
                                # 全部吃掉这笔买入
                                matched_qty += buy['qty']
                                matched_cost += buy['qty'] * buy['price']
                                to_sell -= buy['qty']
                                buy_queues[code].pop(0)
                            else:
                                # 部分吃掉
                                matched_qty += to_sell
                                matched_cost += to_sell * buy['price']
                                buy['qty'] -= to_sell
                                to_sell = 0
                        
                        if matched_qty > 0:
                            avg_buy_price = matched_cost / matched_qty
                            sell_price = price
                            
                            # 考虑手续费 (大概 0.15%)
                            pnl = (sell_price - avg_buy_price) * matched_qty - trade['total_cost']
                            
                            if pnl > 0:
                                profit_trades += 1
                                total_profit += pnl
                            else:
                                loss_trades += 1
                                total_loss += abs(pnl)
            
            total_closed = profit_trades + loss_trades
            win_rate = profit_trades / total_closed if total_closed > 0 else 0.0
            profit_factor = total_profit / total_loss if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)
        
        # 日胜率 (用于对比)
        daily_win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
        
        self.metrics = {
            'total_return': round(total_return, 4),
            'annual_return': round(annual_return, 4),
            'max_drawdown': round(max_drawdown, 4),
            'volatility': round(volatility, 4),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'calmar': round(calmar, 3),
            'win_rate': round(win_rate, 4),
            'daily_win_rate': round(daily_win_rate, 4),
            'profit_trades': profit_trades,
            'loss_trades': loss_trades,
            'total_trades': len(trades) if not trades.empty else 0,
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else "inf"
        }
    
    def print_summary(self) -> None:
        """打印摘要"""
        m = self.metrics
        # 修复: 区分交易胜率和日胜率
        win_rate = m.get('win_rate', 0)
        daily_win_rate = m.get('daily_win_rate', 0)
        profit_trades = m.get('profit_trades', 0)
        loss_trades = m.get('loss_trades', 0)
        profit_factor = m.get('profit_factor', 0)
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    📊 回测结果: {self.strategy_name:<30}║
╠══════════════════════════════════════════════════════════════════╣
║  总收益:      {m.get('total_return', 0):>10.2%}    年化收益:    {m.get('annual_return', 0):>10.2%}   ║
║  最大回撤:    {m.get('max_drawdown', 0):>10.2%}    波动率:      {m.get('volatility', 0):>10.2%}   ║
║  夏普比率:    {m.get('sharpe', 0):>10.3f}    卡玛比率:    {m.get('calmar', 0):>10.3f}   ║
║  索提诺:      {m.get('sortino', 0):>10.3f}    交易胜率:    {win_rate:>10.2%}   ║
║  日胜率:      {daily_win_rate:>10.2%}    盈亏次数:    {profit_trades}/{loss_trades}               ║
║  利润因子:    {str(profit_factor):>10}    交易次数:    {m.get('total_trades', 0):>10d}   ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    def get_equity_curve(self) -> pd.DataFrame:
        return self.portfolio.get_equity_df()
    
    def get_trades(self) -> pd.DataFrame:
        return self.portfolio.get_trades_df()