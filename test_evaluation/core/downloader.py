"""TDX 高性能多进程日线下载引擎"""
import os
import time
import random
import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Dict, Optional, Tuple
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
        print(f"[PID-{os.getpid()}] ✓ Connected → {_node_info}")
    except Exception as e:
        print(f"[PID-{os.getpid()}] ✗ Failed: {e}")
        _api = None


# ==================== 核心函数1: fetch_worker ====================
def fetch_worker(task: Tuple[str, int, int]) -> Tuple[str, int, Optional[pd.DataFrame]]:
    """
    子进程工作函数 - 下载单只股票全量日线
    
    Args:
        task: (stock_code, market, batch_index)
              market: 0=深圳, 1=上海
    
    Returns:
        (code, market, DataFrame | None)
    """
    code, market, idx = task
    
    # ===== 防封策略: 每50只股票休眠 =====
    if idx > 0 and idx % BATCH_SIZE == 0:
        time.sleep(random.uniform(*SLEEP_RANGE))
    
    if _api is None:
        return (code, market, None)
    
    # ===== 分页拉取全量日线 =====
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
            if len(bars) < BARS_PER_REQ:  # 已到最早数据
                break
            offset += BARS_PER_REQ
        
        if not all_bars:
            return (code, market, None)
        
        # ===== 向量化数据清洗 =====
        df = pd.DataFrame(all_bars)
        df = _vectorized_clean(df, code, market)
        return (code, market, df)
        
    except Exception as e:
        print(f"[PID-{os.getpid()}] Error {code}: {type(e).__name__}")
        return (code, market, None)


def _vectorized_clean(df: pd.DataFrame, code: str, market: int) -> pd.DataFrame:
    """向量化清洗: 列名标准化 + 类型转换"""
    # 列名映射
    col_map = {'datetime': 'date', 'vol': 'vol', 'amount': 'amount'}
    df = df.rename(columns=col_map)
    
    # 选取标准列
    std_cols = ['date', 'open', 'high', 'low', 'close', 'vol', 'amount']
    df = df[[c for c in std_cols if c in df.columns]].copy()
    
    # 向量化类型转换
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype('float32')
    df[['vol', 'amount']] = df[['vol', 'amount']].astype('int64')
    
    # 元数据
    df.insert(0, 'code', code)
    df.insert(1, 'market', market)
    
    return df.sort_values('date').reset_index(drop=True)


# ==================== 核心函数2: parallel_runner ====================
def parallel_runner(
    stock_list: List[Tuple[str, int]],
    top_nodes: List[Dict],
    n_workers: Optional[int] = None,
    callback: Optional[callable] = None
) -> Dict[str, pd.DataFrame]:
    """
    多进程并行下载调度器
    
    Args:
        stock_list: [(code, market), ...] 待下载股票列表
        top_nodes:  最优节点列表 (来自节点筛选模块)
        n_workers:  进程数, 默认=CPU核心数
        callback:   可选回调 fn(code, df) 用于流式存储
    
    Returns:
        {code: DataFrame} 下载结果字典
    
    Architecture:
        ┌─────────────┐
        │ Main Process│──分发任务──►┌──────────────────────────┐
        └─────────────┘              │   multiprocessing.Pool   │
              ▲                      │  ┌─────┐ ┌─────┐ ┌─────┐│
              │                      │  │ W1  │ │ W2  │ │ W3  ││
              │ 聚合结果             │  │Node1│ │Node2│ │Node3││
              │                      │  └─────┘ └─────┘ └─────┘│
              └──────────────────────└──────────────────────────┘
    """
    n_workers = n_workers or mp.cpu_count()
    
    # 构建带索引的任务 (索引用于批次计数防封)
    tasks = [(code, mkt, i) for i, (code, mkt) in enumerate(stock_list)]
    total = len(tasks)
    
    print(f"{'='*55}")
    print(f"[Engine] Stocks: {total} | Workers: {n_workers} | Batch: {BATCH_SIZE}")
    print(f"[Engine] Nodes: {[n['name'] for n in top_nodes[:3]]}")
    print(f"{'='*55}")
    
    results: Dict[str, pd.DataFrame] = {}
    stats = {"success": 0, "failed": 0}
    t0 = time.perf_counter()
    
    # ===== 进程池执行 =====
    with Pool(processes=n_workers, initializer=_worker_init, initargs=(top_nodes,)) as pool:
        
        # imap_unordered: 流式获取结果, 内存友好
        for i, (code, market, df) in enumerate(pool.imap_unordered(fetch_worker, tasks), 1):
            
            if df is not None:
                results[code] = df
                stats["success"] += 1
                if callback:
                    callback(code, df)  # 流式存储回调
            else:
                stats["failed"] += 1
            
            # 进度报告 (每200只或完成时)
            if i % 200 == 0 or i == total:
                elapsed = time.perf_counter() - t0
                speed = i / elapsed
                eta = (total - i) / speed if speed > 0 else 0
                print(f"[Progress] {i:>5}/{total} │ ✓{stats['success']:>4} ✗{stats['failed']:>3} │ "
                      f"{speed:>5.1f}/s │ ETA {eta:>4.0f}s")
    
    elapsed = time.perf_counter() - t0
    print(f"{'='*55}")
    print(f"[Done] {stats['success']}/{total} downloaded in {elapsed:.1f}s "
          f"({stats['success']/elapsed:.1f} stocks/sec)")
    
    return results


# ==================== 辅助函数 ====================
def get_all_stock_codes(api: TdxHq_API) -> List[Tuple[str, int]]:
    """硬核全量 A 股抓取逻辑"""
    stocks = []
    # 市场 0: 深市 (含创业板), 1: 沪市 (含科创板)
    for market in [0, 1]:
        # 先获取该市场股票总数，确保翻页不遗漏
        count = api.get_security_count(market)
        print(f"[Market {market}] 探测到总记录数: {count}")

        for start in range(0, count, 1000):
            batch = api.get_security_list(market, start)
            if not batch:
                continue

            for item in batch:
                code = item['code']
                # 严格匹配 A 股代码逻辑
                # 深市 A 股：000, 001, 002, 003(中小板合并), 300-301(创业板)
                if market == 0:
                    if code.startswith(('00', '30')):
                        stocks.append((code, market))

                # 沪市 A 股：600, 601, 603, 605(主板), 688, 689(科创板)
                elif market == 1:
                    if code.startswith(('60', '68')):
                        stocks.append((code, market))

    # 去重，防止 TDX 列表重复返回
    stocks = list(set(stocks))
    print(f">>> 过滤后纯 A 股总数: {len(stocks)}")
    return stocks

def save_to_parquet(output_dir: str):
    """返回流式存储回调函数"""
    os.makedirs(output_dir, exist_ok=True)
    def _callback(code: str, df: pd.DataFrame):
        df.to_parquet(f"{output_dir}/{code}.parquet", index=False, compression='zstd')
    return _callback


# ==================== 完整使用示例 ====================
if __name__ == "__main__":
    from tdx_node_scanner import get_fastest_nodes  # 第一步模块
    
    # Step 1: 获取最优节点
    print("[Step 1] Scanning TDX nodes...")
    top_nodes = get_fastest_nodes(top_n=5)
    
    # Step 2: 获取股票列表 (或使用预定义列表)
    print("[Step 2] Fetching stock list...")
    with TdxHq_API() as api:
        api.connect(top_nodes[0]['host'], top_nodes[0]['port'])
        stock_list = get_all_stock_codes(api)
    print(f"         Found {len(stock_list)} stocks")
    
    # Step 3: 并行下载
    print("[Step 3] Downloading daily bars...")
    data = parallel_runner(
        stock_list=stock_list,
        top_nodes=top_nodes,
        n_workers=mp.cpu_count(),
        callback=save_to_parquet("./tdx_daily_data")  # 边下载边存储
    )
    
    # 验证结果
    if data:
        sample = list(data.values())[0]
        print(f"\n[Sample Data]\n{sample.tail(5)}")