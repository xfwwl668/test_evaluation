"""TDX 高性能多进程日线下载引擎"""
import os
import time
import random
import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd

from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams

# ==================== 配置常量 ====================
BATCH_SIZE: int = 50           # 防封批次大小
SLEEP_RANGE: Tuple[float, float] = (1.0, 3.0)
BARS_PER_REQ: int = 800        # TDX单次请求上限
TOTAL_BARS: int = 2500         # ~10年日线

# ==================== 进程级全局变量 ====================
_api: Optional[TdxHq_API] = None
_node_info: str = ""

def _worker_init(top_nodes: List[Dict]) -> None:
    """子进程初始化: 从Top3随机选节点建立持久连接"""
    global _api, _node_info
    
    node = random.choice(top_nodes[:3])  # 防封: 分散到不同节点
    _node_info = f"{node['name']}({node['host']})"
    _api = TdxHq_API(heartbeat=True, auto_retry=True)
    
    try:
        _api.connect(node["host"], node["port"], time_out=10)
    except Exception:
        _api = None


def fetch_worker(task: Tuple[str, int, int, str]) -> Tuple[str, int, Optional[pd.DataFrame]]:
    """
    子进程工作函数 - 下载单只股票全量日线
    """
    code, market, idx, name = task
    
    if idx > 0 and idx % BATCH_SIZE == 0:
        time.sleep(random.uniform(*SLEEP_RANGE))
    
    if _api is None:
        return (code, market, None)
    
    all_bars: List[dict] = []
    offset = 0
    
    try:
        while offset < TOTAL_BARS:
            bars = _api.get_security_bars(
                category=TDXParams.KLINE_TYPE_DAILY,
                market=market,
                code=code,
                start=offset,
                count=min(BARS_PER_REQ, TOTAL_BARS - offset)
            )
            if not bars:
                break
            all_bars.extend(bars)
            if len(bars) < BARS_PER_REQ:
                break
            offset += BARS_PER_REQ
        
        if not all_bars:
            return (code, market, None)
        
        df = pd.DataFrame(all_bars)
        df = _vectorized_clean(df, code, market, name)
        return (code, market, df)
        
    except Exception:
        return (code, market, None)


def _vectorized_clean(df: pd.DataFrame, code: str, market: int, name: str = "") -> pd.DataFrame:
    """向量化清洗"""
    col_map = {'datetime': 'date', 'vol': 'vol', 'amount': 'amount'}
    df = df.rename(columns=col_map)
    
    std_cols = ['date', 'open', 'high', 'low', 'close', 'vol', 'amount']
    df = df[[c for c in std_cols if c in df.columns]].copy()
    
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype('float32')
    df[['vol', 'amount']] = df[['vol', 'amount']].astype('int64')
    
    df.insert(0, 'code', code)
    df.insert(1, 'market', market)
    df.insert(3, 'name', name)
    
    return df.sort_values('date').reset_index(drop=True)


class StockDownloader:
    """
    股票数据下载器
    """
    
    def __init__(self, top_nodes: List[Dict]):
        self.top_nodes = top_nodes
        self.stock_names: Dict[str, str] = {}

    def get_all_stocks(self) -> List[Tuple[str, int]]:
        """获取所有 A 股列表及名称"""
        stocks = []
        node = self.top_nodes[0]
        api = TdxHq_API()
        
        try:
            api.connect(node['host'], node['port'])
            for market in [0, 1]:
                count = api.get_security_count(market)
                for start in range(0, count, 1000):
                    batch = api.get_security_list(market, start)
                    if not batch:
                        continue
                    for item in batch:
                        code = item['code']
                        name = item['name']
                        if market == 0 and code.startswith(('00', '30')):
                            stocks.append((code, market))
                            self.stock_names[code] = name
                        elif market == 1 and code.startswith(('60', '68')):
                            stocks.append((code, market))
                            self.stock_names[code] = name
            api.disconnect()
        except Exception as e:
            print(f"Error fetching stock list: {e}")
            
        return list(set(stocks))

    def download_all(
        self,
        stock_list: List[Tuple[str, int]],
        n_workers: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, pd.DataFrame]:
        """并行下载所有股票"""
        n_workers = n_workers or mp.cpu_count()
        tasks = []
        for i, (code, mkt) in enumerate(stock_list):
            name = self.stock_names.get(code, "")
            tasks.append((code, mkt, i, name))
            
        total = len(tasks)
        results: Dict[str, pd.DataFrame] = {}
        
        with Pool(processes=n_workers, initializer=_worker_init, initargs=(self.top_nodes,)) as pool:
            for i, (code, market, df) in enumerate(pool.imap_unordered(fetch_worker, tasks), 1):
                if df is not None:
                    results[code] = df
                if progress_callback:
                    progress_callback(i, total)
                    
        return results

    def download_single(self, code: str, market: int, bars: int = 800) -> Optional[pd.DataFrame]:
        """下载单只股票"""
        node = random.choice(self.top_nodes[:3])
        api = TdxHq_API()
        try:
            api.connect(node['host'], node['port'])
            raw_bars = api.get_security_bars(TDXParams.KLINE_TYPE_DAILY, market, code, 0, bars)
            api.disconnect()
            if not raw_bars:
                return None
            df = pd.DataFrame(raw_bars)
            name = self.stock_names.get(code, "")
            return _vectorized_clean(df, code, market, name)
        except Exception:
            return None
