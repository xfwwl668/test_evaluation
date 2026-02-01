# ============================================================================
# 文件: core/updater.py
# ============================================================================
"""
数据更新调度器 - 增量/全量更新
"""
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

from .node_scanner import TDXNodeScanner as NodeScanner
from .downloader import StockDownloader
from .database import StockDatabase
from config import settings


class DataUpdater:
    """
    数据更新调度器
    
    功能:
    - 全量更新: 下载全部历史数据
    - 增量更新: 只下载最新数据
    - 自动节点选择
    - 断点续传
    
    使用:
    ```python
    updater = DataUpdater()
    
    # 全量更新
    updater.full_update()
    
    # 增量更新 (每日调用)
    updater.incremental_update()
    ```
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(settings.path.DB_PATH)
        self.db = StockDatabase(self.db_path)
        
        self.logger = logging.getLogger("DataUpdater")
    
    def full_update(
        self,
        n_workers: int = None,
        progress_callback: callable = None
    ) -> Dict:
        """
        全量更新
        
        Args:
            n_workers: 并行进程数
            progress_callback: 进度回调 fn(current, total)
        
        Returns:
            更新统计
        """
        t0 = time.perf_counter()
        self.logger.info("=" * 60)
        self.logger.info("Starting FULL update...")
        
        # 1. 扫描节点
        self.logger.info("[1/4] Scanning TDX nodes...")
        scanner = NodeScanner()
        top_nodes = scanner.scan_fastest(top_n=5)
        self.logger.info(f"       Best nodes: {[n['name'] for n in top_nodes[:3]]}")
        
        # 2. 获取股票列表
        self.logger.info("[2/4] Fetching stock list...")
        downloader = StockDownloader(top_nodes)
        stock_list = downloader.get_all_stocks()
        self.logger.info(f"       Found {len(stock_list)} stocks")
        
        # 3. 并行下载
        self.logger.info("[3/4] Downloading daily bars...")
        results = downloader.download_all(
            stock_list,
            n_workers=n_workers,
            progress_callback=progress_callback
        )
        
        # 4. 写入数据库
        self.logger.info("[4/4] Writing to database...")
        written = 0
        for code, df in results.items():
            if df is not None and not df.empty:
                self.db.upsert(df)
                written += 1
        
        elapsed = time.perf_counter() - t0
        stats = {
            'mode': 'full',
            'total_stocks': len(stock_list),
            'downloaded': len(results),
            'written': written,
            'elapsed_seconds': round(elapsed, 2)
        }
        
        self.logger.info("=" * 60)
        self.logger.info(f"Full update completed: {stats}")
        
        return stats
    
    def incremental_update(
        self,
        progress_callback: callable = None
    ) -> Dict:
        """
        增量更新 - 只下载最新数据
        
        逻辑:
        1. 查询数据库中每只股票的最新日期
        2. 只下载 latest_date+1 到 today 的数据
        3. UPSERT 写入
        """
        t0 = time.perf_counter()
        self.logger.info("=" * 60)
        self.logger.info("Starting INCREMENTAL update...")
        
        # 1. 获取数据库状态
        db_stats = self.db.get_stats()
        self.logger.info(f"Database: {db_stats['stocks']} stocks, latest={db_stats['max_date']}")
        
        # 2. 扫描节点
        scanner = NodeScanner()
        top_nodes = scanner.scan_fastest(top_n=3)
        
        # 3. 获取需要更新的股票
        latest_dates = self._get_latest_dates()
        today = datetime.now().strftime('%Y-%m-%d')
        
        stocks_to_update = []
        for code, market, latest in latest_dates:
            if latest < today:
                stocks_to_update.append((code, market, latest))
        
        self.logger.info(f"Stocks to update: {len(stocks_to_update)}")
        
        if not stocks_to_update:
            self.logger.info("Already up to date!")
            return {'mode': 'incremental', 'updated': 0}
        
        # 4. 下载增量数据
        downloader = StockDownloader(top_nodes)
        updated_count = 0
        
        for i, (code, market, latest) in enumerate(stocks_to_update):
            try:
                # 计算需要下载的天数
                days_gap = (datetime.now() - datetime.strptime(latest, '%Y-%m-%d')).days
                
                if days_gap <= 0:
                    continue
                
                # 下载
                df = downloader.download_single(code, market, bars=min(days_gap + 10, 100))
                
                if df is not None and not df.empty:
                    # 过滤只保留新数据
                    df = df[df['date'] > latest]
                    
                    if not df.empty:
                        self.db.upsert(df)
                        updated_count += 1
                
                if progress_callback:
                    progress_callback(i + 1, len(stocks_to_update))
                
                # 防封
                if (i + 1) % 50 == 0:
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.warning(f"Failed to update {code}: {e}")
                continue
        
        elapsed = time.perf_counter() - t0
        stats = {
            'mode': 'incremental',
            'checked': len(stocks_to_update),
            'updated': updated_count,
            'elapsed_seconds': round(elapsed, 2)
        }
        
        self.logger.info("=" * 60)
        self.logger.info(f"Incremental update completed: {stats}")
        
        return stats
    
    def _get_latest_dates(self) -> List[Tuple[str, int, str]]:
        """获取每只股票的最新日期"""
        with self.db.connect() as conn:
            result = conn.execute("""
                SELECT code, market, MAX(date) as latest
                FROM daily_bars
                GROUP BY code, market
            """).fetchall()
            
            return [(r[0], r[1], str(r[2])) for r in result]
    
    def update_single(self, code: str, market: int = None) -> Dict:
        """更新单只股票"""
        scanner = NodeScanner()
        top_nodes = scanner.scan_fastest(top_n=3)
        
        downloader = StockDownloader(top_nodes)
        df = downloader.download_single(code, market or 0)
        
        if df is not None and not df.empty:
            rows = self.db.upsert(df)
            return {'code': code, 'rows': rows, 'success': True}
        
        return {'code': code, 'rows': 0, 'success': False}
    
    def check_integrity(self) -> Dict:
        """检查数据完整性"""
        stats = self.db.get_stats()
        
        # 检查缺失
        with self.db.connect() as conn:
            # 获取所有交易日
            trading_days = conn.execute("""
                SELECT DISTINCT date FROM daily_bars ORDER BY date
            """).fetchdf()
            
            # 检查每只股票的数据连续性
            gaps = conn.execute("""
                SELECT code, COUNT(*) as days
                FROM daily_bars
                GROUP BY code
                HAVING days < (SELECT COUNT(DISTINCT date) FROM daily_bars) * 0.9
            """).fetchdf()
        
        return {
            'total_rows': stats['total_rows'],
            'stocks': stats['stocks'],
            'trading_days': len(trading_days),
            'incomplete_stocks': len(gaps) if not gaps.empty else 0
        }


class ScheduledUpdater:
    """
    定时更新器 - 用于每日自动更新
    
    使用:
    ```python
    scheduler = ScheduledUpdater()
    scheduler.run_daily(hour=18, minute=0)
    ```
    """
    
    def __init__(self):
        self.updater = DataUpdater()
        self.logger = logging.getLogger("Scheduler")
    
    def run_once(self) -> Dict:
        """执行一次更新"""
        now = datetime.now()
        
        # 判断是否交易日
        if now.weekday() >= 5:  # 周末
            self.logger.info("Weekend, skip update")
            return {'skipped': True, 'reason': 'weekend'}
        
        # 判断时间
        if now.hour < 15:  # 15:00 前
            self.logger.info("Market not closed yet")
            return {'skipped': True, 'reason': 'market_open'}
        
        # 执行增量更新
        return self.updater.incremental_update()
    
    def run_daily(self, hour: int = 18, minute: int = 0) -> None:
        """
        每日定时运行
        
        Args:
            hour: 执行时间 (小时)
            minute: 执行时间 (分钟)
        """
        import schedule
        
        run_time = f"{hour:02d}:{minute:02d}"
        self.logger.info(f"Scheduled daily update at {run_time}")
        
        schedule.every().day.at(run_time).do(self.run_once)
        
        while True:
            schedule.run_pending()
            time.sleep(60)