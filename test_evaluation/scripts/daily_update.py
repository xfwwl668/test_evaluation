# ============================================================================
# 文件: scripts/daily_update.py
# ============================================================================
#!/usr/bin/env python
"""
每日数据更新
"""
import click
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logging
from core.updater import DataUpdater
from config import settings


@click.command()
@click.option('--full', is_flag=True, help='全量更新 (否则增量)')
@click.option('--db-path', default=None, help='数据库路径')
@click.option('--verbose', '-v', is_flag=True, help='详细日志')
def main(full: bool, db_path: str, verbose: bool):
    """每日数据更新"""
    
    setup_logging(level='DEBUG' if verbose else 'INFO')
    
    click.echo("=" * 60)
    click.echo(f"📈 数据更新 - {'全量' if full else '增量'}模式")
    click.echo("=" * 60)
    
    db_path = db_path or str(settings.path.DB_PATH)
    updater = DataUpdater(db_path)
    
    if full:
        stats = updater.full_update()
    else:
        stats = updater.incremental_update()
    
    click.echo("\n" + "=" * 60)
    click.echo("✅ 更新完成!")
    click.echo(f"   更新数量: {stats.get('updated', stats.get('written', 0))}")
    click.echo(f"   耗时: {stats['elapsed_seconds']:.1f}s")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()