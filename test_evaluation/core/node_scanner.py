"""TDX极速节点筛选模块 - 异步/多线程双模式"""
import asyncio
import socket
import time
import unittest
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

# ==================== 节点配置 (>10个) ====================
TDX_NODES: List[Dict] = [
    {"name": "深圳双线1", "host": "119.147.212.81", "port": 7709},
    {"name": "深圳双线2", "host": "112.91.112.219", "port": 7709},
    {"name": "上海双线1", "host": "101.227.73.20", "port": 7709},
    {"name": "上海双线2", "host": "101.227.77.254", "port": 7709},
    {"name": "北京双线", "host": "218.108.98.244", "port": 7709},
    {"name": "广州双线", "host": "113.105.142.162", "port": 7709},
    {"name": "杭州双线", "host": "121.14.110.194", "port": 7709},
    {"name": "武汉双线", "host": "59.173.18.69", "port": 7709},
    {"name": "南京双线", "host": "180.153.18.170", "port": 7709},
    {"name": "成都双线", "host": "61.135.142.73", "port": 7709},
    {"name": "重庆双线", "host": "124.161.145.45", "port": 7709},
    {"name": "天津双线", "host": "60.28.23.80", "port": 7709},
]

# ==================== 核心扫描类 ====================
class TDXNodeScanner:
    """TDX节点延迟探测器 - 支持异步/多线程"""
    
    def __init__(self, timeout: float = 3.0) -> None:
        self.timeout = timeout

    async def _probe_async(self, node: Dict) -> Dict:
        """异步探测单节点"""
        start = time.perf_counter()
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(node["host"], node["port"]),
                timeout=self.timeout
            )
            latency_ms = (time.perf_counter() - start) * 1000
            writer.close()
            await writer.wait_closed()
            status = "ok"
        except (asyncio.TimeoutError, OSError) as e:
            latency_ms, status = -1.0, f"fail:{type(e).__name__}"
        
        return {**node, "latency_ms": round(latency_ms, 2), "status": status}

    def _probe_sync(self, node: Dict) -> Dict:
        """同步探测单节点 (线程池用)"""
        start = time.perf_counter()
        try:
            with socket.create_connection(
                (node["host"], node["port"]), timeout=self.timeout
            ) as sock:
                latency_ms = (time.perf_counter() - start) * 1000
                status = "ok"
        except (socket.timeout, OSError) as e:
            latency_ms, status = -1.0, f"fail:{type(e).__name__}"
        
        return {**node, "latency_ms": round(latency_ms, 2), "status": status}

    @staticmethod
    def _sort_by_latency(results: List[Dict]) -> List[Dict]:
        """按延迟升序排序，失败节点置后"""
        return sorted(results, key=lambda x: (x["latency_ms"] < 0, x["latency_ms"]))

    async def scan_async(self, nodes: Optional[List[Dict]] = None) -> List[Dict]:
        """异步并发扫描"""
        nodes = nodes or TDX_NODES
        tasks = [self._probe_async(n) for n in nodes]
        results = await asyncio.gather(*tasks)
        return self._sort_by_latency(list(results))

    def scan_threaded(self, nodes: Optional[List[Dict]] = None, workers: int = 16) -> List[Dict]:
        """多线程并行扫描"""
        nodes = nodes or TDX_NODES
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(self._probe_sync, nodes))
        return self._sort_by_latency(results)


def get_fastest_nodes(top_n: int = 5, async_mode: bool = True) -> List[Dict]:
    """快捷接口: 获取最快的N个节点"""
    scanner = TDXNodeScanner(timeout=2.0)
    if async_mode:
        results = asyncio.run(scanner.scan_async())
    else:
        results = scanner.scan_threaded()
    return [r for r in results if r["status"] == "ok"][:top_n]


# ==================== 单元测试 ====================
class TestTDXScanner(unittest.TestCase):
    """节点扫描器单元测试"""
    
    def test_node_count_gte_10(self) -> None:
        """验证节点数量>=10"""
        self.assertGreaterEqual(len(TDX_NODES), 10)
    
    def test_result_schema(self) -> None:
        """验证返回结构完整性"""
        scanner = TDXNodeScanner(timeout=0.5)
        mock = [{"name": "mock", "host": "127.0.0.1", "port": 9999}]
        results = scanner.scan_threaded(mock)
        self.assertEqual(len(results), 1)
        self.assertIn("latency_ms", results[0])
        self.assertIn("status", results[0])
    
    def test_sort_logic(self) -> None:
        """验证排序逻辑: 有效延迟升序, 失败置后"""
        data = [{"latency_ms": 100}, {"latency_ms": -1}, {"latency_ms": 30}]
        sorted_data = TDXNodeScanner._sort_by_latency(data)
        self.assertEqual([d["latency_ms"] for d in sorted_data], [30, 100, -1])
    
    def test_async_sync_parity(self) -> None:
        """验证异步/同步模式结果一致性"""
        scanner = TDXNodeScanner(timeout=0.3)
        mock = [{"name": "x", "host": "10.255.255.1", "port": 1}]  # 不可达
        r1 = asyncio.run(scanner.scan_async(mock))
        r2 = scanner.scan_threaded(mock)
        self.assertEqual(r1[0]["status"][:4], r2[0]["status"][:4])  # 都应失败


# ==================== 执行入口 ====================
if __name__ == "__main__":
    # 运行单元测试
    print("=" * 50)
    print("Running Unit Tests...")
    unittest.main(verbosity=2, exit=False)
    
    # 实际扫描演示
    print("\n" + "=" * 50)
    print("TDX Node Scan Results (Top 5):\n")
    for i, node in enumerate(get_fastest_nodes(5), 1):
        print(f"  {i}. {node['name']:10} │ {node['host']:>16}:{node['port']} │ {node['latency_ms']:>7.2f} ms")