# ============================================================================
# 文件: scripts/run_backtest.py
# ============================================================================
#!/usr/bin/env python
"""
运行回测
"""
import click
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logging
from engine.backtest import BacktestEngine
from strategy import StrategyRegistry
from strategy.rsrs_strategy import RSRSStrategy
from strategy.momentum_strategy import MomentumStrategy
from config import settings


@click.command()
@click.option('--strategy', '-s', default='rsrs', help='策略名称 (rsrs/momentum)')
@click.option('--start', default='2020-01-01', help='开始日期')
@click.option('--end', default='2023-12-31', help='结束日期')
@click.option('--capital', default=1000000, type=float, help='初始资金')
@click.option('--freq', default='W', help='调仓频率 (D/W/M)')
@click.option('--compare', is_flag=True, help='多策略对比模式')
@click.option('--verbose', '-v', is_flag=True, help='详细日志')
def main(strategy: str, start: str, end: str, capital: float, freq: str, compare: bool, verbose: bool):
    """运行策略回测"""
    
    setup_logging(level='DEBUG' if verbose else 'INFO')
    
    click.echo("=" * 60)
    click.echo("🚀 量化回测引擎")
    click.echo("=" * 60)
    click.echo(f"策略: {strategy}")
    click.echo(f"区间: {start} ~ {end}")
    click.echo(f"资金: {capital:,.0f}")
    click.echo(f"频率: {freq}")
    click.echo("=" * 60)
    
    # 创建引擎
    engine = BacktestEngine(initial_capital=capital)
    
    if compare:
        # 多策略对比
        engine.add_strategy(RSRSStrategy())
        engine.add_strategy(MomentumStrategy())
    else:
        # 单策略
        if strategy == 'rsrs':
            engine.add_strategy(RSRSStrategy())
        elif strategy == 'momentum':
            engine.add_strategy(MomentumStrategy())
        else:
            try:
                strat_cls = StrategyRegistry.get(strategy)
                engine.add_strategy(strat_cls())
            except KeyError:
                click.echo(f"❌ 未知策略: {strategy}")
                click.echo(f"可用策略: {StrategyRegistry.list_all()}")
                return
    
    # 运行回测
    results = engine.run(start, end, rebalance_freq=freq)
    
    # 对比输出
    if compare:
        click.echo("\n📊 策略对比:")
        comparison = engine.compare_strategies()
        click.echo(comparison.tail(10).to_string())


if __name__ == "__main__":
    main()