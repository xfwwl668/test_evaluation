# ============================================================================
# 文件: main.py
# ============================================================================
#!/usr/bin/env python
"""
量化引擎主入口
"""
import click
import sys
from pathlib import Path

# 确保模块可导入
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from utils.logger import setup_logging, get_logger
from config import settings


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='详细日志')
@click.pass_context
def cli(ctx, verbose: bool):
    """
    🚀 量化交易引擎
    
    使用示例:
    
    \b
    # 初始化数据库
    python main.py init
    
    \b
    # 每日更新
    python main.py update
    
    \b
    # 运行回测
    python main.py backtest --strategy rsrs --start 2020-01-01
    
    \b
    # 市场扫描
    python main.py scan --top 30
    
    \b
    # 单股诊断
    python main.py diagnose 000001
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(level='DEBUG' if verbose else 'INFO')


@cli.command()
@click.option('--workers', '-w', default=None, type=int, help='并行进程数')
@click.pass_context
def init(ctx, workers: int):
    """初始化数据库 - 全量下载"""
    from core.updater import DataUpdater
    
    click.echo("📦 初始化数据库...")
    updater = DataUpdater()
    stats = updater.full_update(n_workers=workers)
    click.echo(f"✅ 完成! 下载 {stats['downloaded']} 只股票")


@cli.command()
@click.option('--full', is_flag=True, help='全量更新')
@click.pass_context
def update(ctx, full: bool):
    """更新数据"""
    from core.updater import DataUpdater
    
    click.echo(f"📈 {'全量' if full else '增量'}更新...")
    updater = DataUpdater()
    
    if full:
        stats = updater.full_update()
    else:
        stats = updater.incremental_update()
    
    click.echo(f"✅ 完成! 更新 {stats.get('updated', stats.get('written', 0))} 条")


@cli.command()
@click.option('--strategy', '-s', default='rsrs', help='策略名称')
@click.option('--start', default='2020-01-01', help='开始日期')
@click.option('--end', default='2023-12-31', help='结束日期')
@click.option('--capital', default=1000000, type=float, help='初始资金')
@click.option('--freq', default='W', help='调仓频率')
@click.pass_context
def backtest(ctx, strategy: str, start: str, end: str, capital: float, freq: str):
    """运行回测"""
    from engine.backtest import BacktestEngine
    from strategy.rsrs_strategy import RSRSStrategy
    from strategy.momentum_strategy import MomentumStrategy
    
    click.echo(f"🚀 运行回测: {strategy}")
    
    engine = BacktestEngine(initial_capital=capital)
    
    if strategy == 'rsrs':
        engine.add_strategy(RSRSStrategy())
    elif strategy == 'momentum':
        engine.add_strategy(MomentumStrategy())
    else:
        click.echo(f"❌ 未知策略: {strategy}")
        return
    
    results = engine.run(start, end, rebalance_freq=freq)


@cli.command()
@click.option('--date', '-d', default=None, help='扫描日期')
@click.option('--top', '-n', default=50, type=int, help='输出数量')
@click.pass_context
def scan(ctx, date: str, top: int):
    """全市场扫描"""
    from analysis.scanner import MarketScanner
    from analysis.report import ReportGenerator
    
    click.echo("🔍 扫描市场...")
    
    scanner = MarketScanner()
    result = scanner.scan(target_date=date, top_n=top)
    
    if not result.empty:
        ReportGenerator.print_golden_stocks(result)
    else:
        click.echo("未找到符合条件的股票")


@cli.command()
@click.argument('code')
@click.pass_context
def diagnose(ctx, code: str):
    """单股诊断"""
    from analysis.stock_doctor import StockDoctor
    
    click.echo(f"🔬 诊断 {code}...")
    
    doctor = StockDoctor()
    result = doctor.diagnose(code)
    report = doctor.generate_report(result)
    click.echo(report)


@cli.command()
@click.pass_context
def info(ctx):
    """显示系统信息"""
    click.echo("=" * 60)
    click.echo("📊 量化引擎信息")
    click.echo("=" * 60)
    click.echo(f"数据库: {settings.path.DB_PATH}")
    click.echo(f"日志目录: {settings.path.LOG_DIR}")
    click.echo(f"初始资金: {settings.backtest.INITIAL_CAPITAL:,.0f}")
    click.echo(f"RSRS窗口: {settings.factor.RSRS_WINDOW}")
    click.echo("=" * 60)


if __name__ == "__main__":
    cli()