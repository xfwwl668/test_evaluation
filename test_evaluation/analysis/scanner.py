# ============================================================================
# 文件: analysis/scanner.py
# ============================================================================
"""
全市场扫描器 - 高性能因子筛选
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import logging

from core.database import StockDatabase
from factors import FactorRegistry, FactorPipeline
from config import settings


@dataclass
class ScanResult:
    """扫描结果"""
    code: str
    close: float
    rsrs_zscore: float
    rsrs_r2: float
    vol_rank: float
    vwap_bias: float
    obv_trend: int
    atr_pct: float
    alpha_score: float


class MarketScanner:
    """
    全市场扫描器
    
    功能:
    - 批量加载数据
    - 并行计算因子
    - 多因子评分排序
    - 输出金股表
    
    架构:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      MarketScanner                              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
    │  │ DataLoader  │─►│ FactorCalc  │─►│ Scorer      │─► 金股表   │
    │  │ 批量读取    │  │ 并行计算    │  │ 评分排序    │            │
    │  └─────────────┘  └─────────────┘  └─────────────┘            │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(settings.path.DB_PATH)
        self.db = StockDatabase(self.db_path)
        self.logger = logging.getLogger("MarketScanner")
        
        # 配置
        self.lookback = settings.strategy.MIN_TRADING_DAYS
        self.r2_threshold = settings.factor.RSRS_R2_THRESHOLD
        
        # 评分权重
        self.weights = {
            'rsrs': settings.strategy.w_rsrs if hasattr(settings.strategy, 'w_rsrs') else 0.5,
            'volume': 0.3,
            'price': 0.2
        }
    
    def scan(
        self,
        target_date: str = None,
        top_n: int = 50,
        filters: Dict = None
    ) -> pd.DataFrame:
        """
        执行全市场扫描
        
        Args:
            target_date: 目标日期 (None=最新)
            top_n: 返回数量
            filters: 过滤条件
        
        Returns:
            金股表 DataFrame
        """
        t0 = time.perf_counter()
        
        # 1. 确定日期
        if target_date is None:
            target_date = self._get_latest_date()
        self.logger.info(f"Scanning market for {target_date}")
        
        # 2. 加载数据
        self.logger.info("[1/4] Loading data...")
        stock_data = self._load_data(target_date)
        self.logger.info(f"       Loaded {len(stock_data)} stocks")
        
        # 3. 并行计算因子
        self.logger.info("[2/4] Computing factors...")
        factor_results = self._compute_factors_parallel(stock_data, target_date)
        self.logger.info(f"       Computed {len(factor_results)} valid stocks")
        
        # 4. 过滤 + 评分
        self.logger.info("[3/4] Filtering and scoring...")
        scored = self._filter_and_score(factor_results, filters)
        self.logger.info(f"       Qualified {len(scored)} stocks")
        
        # 5. 格式化输出
        self.logger.info("[4/4] Generating output...")
        result = self._format_output(scored, top_n)
        
        elapsed = time.perf_counter() - t0
        self.logger.info(f"Scan completed in {elapsed:.2f}s ({len(stock_data)/elapsed:.0f} stocks/sec)")
        
        return result
    
    def _get_latest_date(self) -> str:
        """获取最新日期"""
        stats = self.db.get_stats()
        return str(stats.get('max_date', ''))
    
    def _load_data(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """批量加载数据"""
        # 计算起始日期
        end = pd.to_datetime(target_date)
        start = end - pd.DateOffset(days=self.lookback + 100)
        start_str = start.strftime('%Y-%m-%d')
        
        # 加载全市场数据
        with self.db.connect() as conn:
            df = conn.execute(f"""
                SELECT code, market, date, open, high, low, close, vol, amount
                FROM daily_bars
                WHERE date BETWEEN '{start_str}' AND '{target_date}'
                ORDER BY code, date
            """).fetchdf()
        
        # 按股票分组
        stock_data = {}
        for code, group in df.groupby('code'):
            if len(group) >= self.lookback:
                group = group.copy()
                group['date'] = pd.to_datetime(group['date'])
                group.set_index('date', inplace=True)
                stock_data[code] = group
        
        return stock_data
    
    def _compute_factors_parallel(
        self,
        stock_data: Dict[str, pd.DataFrame],
        target_date: str
    ) -> List[Dict]:
        """并行计算因子"""
        n_workers = mp.cpu_count()
        codes = list(stock_data.keys())
        
        # 准备任务
        tasks = [(code, stock_data[code], target_date) for code in codes]
        
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for result in executor.map(_compute_single_stock, tasks):
                if result is not None:
                    results.append(result)
        
        return results
    
    def _filter_and_score(
        self,
        results: List[Dict],
        filters: Dict = None
    ) -> List[Dict]:
        """过滤和评分"""
        filters = filters or {}
        
        # 默认过滤条件
        r2_min = filters.get('r2_min', self.r2_threshold)
        vwap_above = filters.get('vwap_above', True)
        vol_min = filters.get('vol_min', 0)
        
        qualified = []
        
        for r in results:
            # R² 过滤
            if r['rsrs_r2'] < r2_min:
                continue
            
            # VWAP 过滤
            if vwap_above and r['vwap_bias'] <= 0:
                continue
            
            # 成交量过滤
            if r['vol'] <= vol_min:
                continue
            
            # RSRS 有效
            if pd.isna(r['rsrs_zscore']):
                continue
            
            qualified.append(r)
        
        if not qualified:
            return []
        
        # 评分
        df = pd.DataFrame(qualified)
        
        # 标准化
        def minmax(s):
            return (s - s.min()) / (s.max() - s.min() + 1e-10)
        
        rsrs_score = minmax(df['rsrs_zscore'].clip(-3, 3))
        
        # 成交量倒U型评分
        vol_rank = df['vol_rank']
        vol_score = 1 - 4 * (vol_rank - 0.5) ** 2
        vol_score = vol_score.clip(0, 1)
        
        price_score = minmax(df['vwap_bias'].clip(-0.1, 0.1))
        
        # 综合评分
        df['alpha_score'] = (
            self.weights['rsrs'] * rsrs_score +
            self.weights['volume'] * vol_score +
            self.weights['price'] * price_score
        )
        
        # 排序
        df = df.sort_values('alpha_score', ascending=False)
        
        return df.to_dict('records')
    
    def _format_output(self, results: List[Dict], top_n: int) -> pd.DataFrame:
        """格式化输出"""
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results[:top_n])
        
        # 列映射
        col_map = {
            'code': '代码',
            'close': '收盘价',
            'rsrs_zscore': 'RSRS分数',
            'rsrs_r2': 'R²',
            'vol_rank': '量能分位',
            'vwap_bias': 'VWAP偏离',
            'obv_trend': 'OBV趋势',
            'atr_pct': '波动率',
            'alpha_score': '综合评分'
        }
        
        output = df[[c for c in col_map.keys() if c in df.columns]].copy()
        output.columns = [col_map.get(c, c) for c in output.columns]
        
        # 格式化
        if '收盘价' in output.columns:
            output['收盘价'] = output['收盘价'].round(2)
        if 'RSRS分数' in output.columns:
            output['RSRS分数'] = output['RSRS分数'].round(3)
        if 'R²' in output.columns:
            output['R²'] = output['R²'].round(3)
        if '量能分位' in output.columns:
            output['量能分位'] = (output['量能分位'] * 100).round(1).astype(str) + '%'
        if 'VWAP偏离' in output.columns:
            output['VWAP偏离'] = (output['VWAP偏离'] * 100).round(2).astype(str) + '%'
        if 'OBV趋势' in output.columns:
            output['OBV趋势'] = output['OBV趋势'].map({1: '↑', 0: '↓'})
        if '波动率' in output.columns:
            output['波动率'] = (output['波动率'] * 100).round(2).astype(str) + '%'
        if '综合评分' in output.columns:
            output['综合评分'] = output['综合评分'].round(4)
        
        output.index = range(1, len(output) + 1)
        output.index.name = '排名'
        
        return output


def _compute_single_stock(args: Tuple) -> Optional[Dict]:
    """单股因子计算 (用于多进程)"""
    code, df, target_date = args
    
    try:
        n = len(df)
        if n < 60:
            return None
        
        close = df['close'].to_numpy(dtype=np.float64)
        high = df['high'].to_numpy(dtype=np.float64)
        low = df['low'].to_numpy(dtype=np.float64)
        vol = df['vol'].to_numpy(dtype=np.float64)
        amount = df['amount'].to_numpy(dtype=np.float64)
        
        # RSRS
        window = settings.factor.RSRS_WINDOW
        if n < window + 60:
            return None
        
        from numpy.lib.stride_tricks import sliding_window_view
        
        low_win = sliding_window_view(low, window)
        high_win = sliding_window_view(high, window)
        
        x_mean = low_win.mean(axis=1, keepdims=True)
        y_mean = high_win.mean(axis=1, keepdims=True)
        
        cov = ((low_win - x_mean) * (high_win - y_mean)).sum(axis=1)
        var_x = ((low_win - x_mean) ** 2).sum(axis=1)
        var_y = ((high_win - y_mean) ** 2).sum(axis=1)
        
        slope = np.divide(cov, var_x, out=np.zeros_like(cov), where=var_x > 1e-10)
        r2 = np.divide(cov**2, var_x * var_y, out=np.zeros_like(cov), where=(var_x * var_y) > 1e-10)
        
        # 标准化
        std_win = min(200, len(slope))
        recent = slope[-std_win:]
        zscore = (slope[-1] - recent.mean()) / (recent.std() + 1e-10)
        rsrs_zscore = zscore * r2[-1]
        
        # OBV
        price_change = np.diff(close, prepend=close[0])
        obv = np.cumsum(np.sign(price_change) * vol)
        obv_ma = np.convolve(obv, np.ones(20)/20, mode='valid')
        obv_trend = 1 if obv[-1] > obv_ma[-1] else 0
        
        # VWAP
        vwap = amount[-20:].sum() / (vol[-20:].sum() + 1e-10)
        vwap_bias = (close[-1] - vwap) / (vwap + 1e-10)
        
        # 成交量分位
        vol_60 = vol[-60:]
        vol_rank = (vol[-1] > vol_60[:-1]).sum() / (len(vol_60) - 1)
        
        # ATR
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
        atr = tr[-14:].mean()
        atr_pct = atr / close[-1]
        
        return {
            'code': code,
            'close': close[-1],
            'vol': vol[-1],
            'rsrs_zscore': rsrs_zscore,
            'rsrs_r2': r2[-1],
            'obv_trend': obv_trend,
            'vwap_bias': vwap_bias,
            'vol_rank': vol_rank,
            'atr_pct': atr_pct
        }
        
    except Exception:
        return None


def scan_market(
    db_path: str = None,
    target_date: str = None,
    top_n: int = 50,
    **kwargs
) -> pd.DataFrame:
    """
    快捷扫描接口
    
    Usage:
        df = scan_market(top_n=30)
    """
    scanner = MarketScanner(db_path)
    return scanner.scan(target_date, top_n, kwargs)