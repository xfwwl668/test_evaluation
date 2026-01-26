# ============================================================================
# 文件: scripts/scan_market.py
# ============================================================================
#!/usr/bin/env python
"""
全市场扫描
"""
import click
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logging
from analysis.scanner import MarketScanner
from analysis.report import ReportGenerator


@click.command()
@click.option('--date', '-d', default=None, help='扫描日期 (默认最新)')
@click.option('--top', '-n', default=50, type=int, help='输出数量')
@click.option('--r2', default=0.8, type=float, help='R²阈值')
@click.option('--output', '-o', default=None, help='输出文件路径')
@click.option('--verbose', '-v', is_flag=True, help='详细日志')
def main(date: str, top: int, r2: float, output: str, verbose: bool):
    """全市场扫描 - 寻找金股"""
    
    setup_logging(level='DEBUG' if verbose else 'INFO')
    
    click.echo("=" * 60)
    click.echo("🔍 全市场扫描")
    click.echo("=" * 60)
    
    scanner = MarketScanner()
    
    result = scanner.scan(
        target_date=date,
        top_n=top,
        filters={'r2_min': r2}
    )
    
    if result.empty:
        click.echo("❌ 未找到符合条件的股票")
        return
    
    # 输出
    ReportGenerator.print_golden_stocks(result)
    
    # 保存文件
    if output:
        result.to_csv(output, encoding='utf-8-sig')
        click.echo(f"\n✅ 已保存至: {output}")


if __name__ == "__main__":
    main()