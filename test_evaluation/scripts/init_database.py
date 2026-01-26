# ============================================================================
# 文件: scripts/init_database.py
# ============================================================================
#!/usr/bin/env python
"""
初始化数据库 - 全量数据下载
"""
import click
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logging
from core.updater import DataUpdater
from config import settings


@click.command()
@click.option('--workers', '-w', default=None, type=int, help='并行进程数')
@click.option('--db-path', default=None, help='数据库路径')
@click.option('--verbose', '-v', is_flag=True, help='详细日志')
def main(workers: int, db_path: str, verbose: bool):
    """初始化数据库 - 下载全量历史数据"""
    
    # 配置日志
    setup_logging(level='DEBUG' if verbose else 'INFO')
    
    click.echo("=" * 60)
    click.echo("📦 量化引擎 - 数据库初始化")
    click.echo("=" * 60)
    
    db_path = db_path or str(settings.path.DB_PATH)
    click.echo(f"数据库路径: {db_path}")
    
    if Path(db_path).exists():
        if not click.confirm("数据库已存在，是否覆盖?"):
            click.echo("取消操作")
            return
    
    # 执行全量更新
    updater = DataUpdater(db_path)
    
    def progress(current, total):
        click.echo(f"\r下载进度: {current}/{total} ({current/total*100:.1f}%)", nl=False)
    
    stats = updater.full_update(n_workers=workers, progress_callback=progress)
    
    click.echo("\n" + "=" * 60)
    click.echo("✅ 初始化完成!")
    click.echo(f"   股票数量: {stats['downloaded']}")
    click.echo(f"   耗时: {stats['elapsed_seconds']:.1f}s")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()