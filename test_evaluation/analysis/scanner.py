# ============================================================================
# 文件: analysis/scanner.py
# ============================================================================
"""全市场扫描器 - 高性能因子筛选 + 策略模式选股"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from core.database import StockDatabase
from .strategy_scorers import StrategyScorer, get_scorer


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
    """全市场扫描器

    支持 6 种选股模式:
    - factor (默认): 当前因子融合逻辑
    - rsrs: RSRS 策略入场规则
    - momentum: 动量策略入场规则
    - short_term: ShortTermRSRS 入场规则
    - alpha_hunter: AlphaHunter 入场规则
    - ensemble: 多策略投票融合
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(settings.path.DB_PATH)
        self.db = StockDatabase(self.db_path)
        self.logger = logging.getLogger("MarketScanner")

        self.lookback = settings.strategy.MIN_TRADING_DAYS
        self.r2_threshold = settings.factor.RSRS_R2_THRESHOLD

        self.weights = {
            "rsrs": settings.strategy.w_rsrs if hasattr(settings.strategy, "w_rsrs") else 0.5,
            "volume": 0.3,
            "price": 0.2,
        }

    def scan(
        self,
        target_date: str | None = None,
        top_n: int = 50,
        mode: str | Dict | None = "factor",
        filters: Dict | None = None,
        strategy_params: Dict | None = None,
    ) -> pd.DataFrame:
        """执行全市场扫描

        Args:
            target_date: 目标日期
            top_n: 返回数量
            mode: 选股模式 ('factor'|'rsrs'|'momentum'|'short_term'|'alpha_hunter'|'ensemble')
            filters: 过滤条件
            strategy_params: 策略参数覆盖
        """

        # 兼容旧签名: scan(target_date, top_n, filters)
        if isinstance(mode, dict) and filters is None and strategy_params is None:
            filters = mode
            mode = "factor"

        mode = mode or "factor"
        filters = filters or {}
        strategy_params = strategy_params or {}

        t0 = time.perf_counter()

        if target_date is None:
            target_date = self._get_latest_date()
        self.logger.info(f"Scanning market for {target_date} (mode={mode})")

        self.logger.info("[1/4] Loading data...")
        stock_data = self._load_data(target_date)
        self.logger.info(f"       Loaded {len(stock_data)} stocks")

        self.logger.info("[2/4] Computing factors...")
        factor_results = self._compute_factors_parallel(stock_data, target_date, mode, strategy_params)
        self.logger.info(f"       Computed {len(factor_results)} valid stocks")

        # market breadth for alpha_hunter / ensemble
        if mode in {"alpha_hunter", "ensemble"}:
            breadth = self._compute_market_breadth(factor_results)
            filters = {**filters, "market_breadth": breadth}

        self.logger.info("[3/4] Filtering and scoring...")
        scored = self._filter_and_score(factor_results, filters, mode, strategy_params)
        self.logger.info(f"       Qualified {len(scored)} stocks")

        self.logger.info("[4/4] Generating output...")
        result = self._format_output(scored, top_n)

        elapsed = time.perf_counter() - t0
        if elapsed > 0:
            self.logger.info(
                f"Scan completed in {elapsed:.2f}s ({len(stock_data) / elapsed:.0f} stocks/sec)"
            )

        return result

    def _get_latest_date(self) -> str:
        """获取最新日期"""

        stats = self.db.get_stats()
        return str(stats.get("max_date", ""))

    def _load_data(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """批量加载数据"""

        end = pd.to_datetime(target_date)
        start = end - pd.DateOffset(days=self.lookback + 100)
        start_str = start.strftime("%Y-%m-%d")

        with self.db.connect() as conn:
            df = conn.execute(
                f"""
                SELECT code, market, date, open, high, low, close, vol, amount
                FROM daily_bars
                WHERE date BETWEEN '{start_str}' AND '{target_date}'
                ORDER BY code, date
            """
            ).fetchdf()

        stock_data: Dict[str, pd.DataFrame] = {}
        for code, group in df.groupby("code"):
            if len(group) >= self.lookback:
                group = group.copy()
                group["date"] = pd.to_datetime(group["date"])
                group.set_index("date", inplace=True)
                stock_data[code] = group

        return stock_data

    def _build_compute_config(self, mode: str, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """构建因子计算配置，避免无意义的重复计算"""

        momentum_window = int(strategy_params.get("lookback", settings.factor.MOMENTUM_WINDOW))

        if mode == "ensemble":
            nested = strategy_params.get("momentum_params", {})
            if isinstance(nested, dict) and "lookback" in nested:
                momentum_window = int(nested["lookback"])

        compute_config: Dict[str, Any] = {
            "mode": mode,
            "momentum_window": momentum_window,
            "need_momentum": mode in {"momentum", "ensemble"},
            "need_ma": mode in {"short_term", "alpha_hunter", "ensemble"},
            "need_alpha_extras": mode in {"alpha_hunter", "ensemble"},
        }

        return compute_config

    def _compute_factors_parallel(
        self,
        stock_data: Dict[str, pd.DataFrame],
        target_date: str,
        mode: str = "factor",
        strategy_params: Dict[str, Any] | None = None,
    ) -> List[Dict]:
        """并行计算因子"""

        strategy_params = strategy_params or {}
        compute_config = self._build_compute_config(mode, strategy_params)

        codes = list(stock_data.keys())
        if not codes:
            return []

        n_workers = min(mp.cpu_count(), len(codes))
        tasks = [(code, stock_data[code], target_date, compute_config) for code in codes]

        results: List[Dict] = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for result in executor.map(_compute_single_stock, tasks):
                if result is not None:
                    results.append(result)

        return results

    def _get_scorer(self, mode: str, strategy_params: Dict[str, Any]) -> StrategyScorer:
        """根据 mode 获取评分器，并合并默认参数与覆盖参数"""

        default_params: Dict[str, Any] = {}
        if mode == "rsrs" and hasattr(settings.strategy, "SCAN_RSRS_PARAMS"):
            default_params = dict(settings.strategy.SCAN_RSRS_PARAMS)
        elif mode == "momentum" and hasattr(settings.strategy, "SCAN_MOMENTUM_PARAMS"):
            default_params = dict(settings.strategy.SCAN_MOMENTUM_PARAMS)
        elif mode == "short_term" and hasattr(settings.strategy, "SCAN_SHORT_TERM_PARAMS"):
            default_params = dict(settings.strategy.SCAN_SHORT_TERM_PARAMS)
        elif mode == "alpha_hunter" and hasattr(settings.strategy, "SCAN_ALPHA_HUNTER_PARAMS"):
            default_params = dict(settings.strategy.SCAN_ALPHA_HUNTER_PARAMS)
        elif mode == "ensemble" and hasattr(settings.strategy, "SCAN_ENSEMBLE_PARAMS"):
            default_params = dict(settings.strategy.SCAN_ENSEMBLE_PARAMS)

        merged = {**default_params, **(strategy_params or {})}
        return get_scorer(mode, merged)

    def _filter_and_score(
        self,
        results: List[Dict],
        filters: Dict | None = None,
        mode: str = "factor",
        strategy_params: Dict | None = None,
    ) -> List[Dict]:
        """过滤和评分: factor 模式走原逻辑，其余模式走策略评分器"""

        filters = filters or {}
        strategy_params = strategy_params or {}

        if mode == "factor":
            return self._filter_and_score_factor(results, filters)

        scorer = self._get_scorer(mode, strategy_params)
        return scorer.score(results, filters)

    def _filter_and_score_factor(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """原有的因子融合评分逻辑（必须保持行为不变）"""

        r2_min = filters.get("r2_min", self.r2_threshold)
        vwap_above = filters.get("vwap_above", True)
        vol_min = filters.get("vol_min", 0)

        qualified = []

        for r in results:
            if r["rsrs_r2"] < r2_min:
                continue

            if vwap_above and r["vwap_bias"] <= 0:
                continue

            if r["vol"] <= vol_min:
                continue

            if pd.isna(r["rsrs_zscore"]):
                continue

            qualified.append(r)

        if not qualified:
            return []

        df = pd.DataFrame(qualified)

        def minmax(s: pd.Series) -> pd.Series:
            return (s - s.min()) / (s.max() - s.min() + 1e-10)

        rsrs_score = minmax(df["rsrs_zscore"].clip(-3, 3))

        vol_rank = df["vol_rank"]
        vol_score = 1 - 4 * (vol_rank - 0.5) ** 2
        vol_score = vol_score.clip(0, 1)

        price_score = minmax(df["vwap_bias"].clip(-0.1, 0.1))

        df["alpha_score"] = (
            self.weights["rsrs"] * rsrs_score
            + self.weights["volume"] * vol_score
            + self.weights["price"] * price_score
        )

        df = df.sort_values("alpha_score", ascending=False)
        return df.to_dict("records")

    def _format_output(self, results: List[Dict], top_n: int) -> pd.DataFrame:
        """格式化输出（保持原有列格式）"""

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results[:top_n])

        col_map = {
            "code": "代码",
            "close": "收盘价",
            "rsrs_zscore": "RSRS分数",
            "rsrs_r2": "R²",
            "vol_rank": "量能分位",
            "vwap_bias": "VWAP偏离",
            "obv_trend": "OBV趋势",
            "atr_pct": "波动率",
            "alpha_score": "综合评分",
        }

        output = df[[c for c in col_map.keys() if c in df.columns]].copy()
        output.columns = [col_map.get(c, c) for c in output.columns]

        if "收盘价" in output.columns:
            output["收盘价"] = output["收盘价"].round(2)
        if "RSRS分数" in output.columns:
            output["RSRS分数"] = output["RSRS分数"].round(3)
        if "R²" in output.columns:
            output["R²"] = output["R²"].round(3)
        if "量能分位" in output.columns:
            output["量能分位"] = (output["量能分位"] * 100).round(1).astype(str) + "%"
        if "VWAP偏离" in output.columns:
            output["VWAP偏离"] = (output["VWAP偏离"] * 100).round(2).astype(str) + "%"
        if "OBV趋势" in output.columns:
            output["OBV趋势"] = output["OBV趋势"].map({1: "↑", 0: "↓"})
        if "波动率" in output.columns:
            output["波动率"] = (output["波动率"] * 100).round(2).astype(str) + "%"
        if "综合评分" in output.columns:
            output["综合评分"] = output["综合评分"].round(4)

        output.index = range(1, len(output) + 1)
        output.index.name = "排名"

        return output

    def _compute_market_breadth(self, results: List[Dict]) -> Dict[str, Any]:
        """计算市场广度 (简化版)"""

        df = pd.DataFrame(results)
        if df.empty or "prev_close" not in df.columns:
            return {"advance_ratio": 0.0, "is_bullish": False}

        valid = df.dropna(subset=["close", "prev_close"])
        if valid.empty:
            return {"advance_ratio": 0.0, "is_bullish": False}

        advance_ratio = float((valid["close"] > valid["prev_close"]).mean())

        threshold = getattr(settings.strategy, "SCAN_ALPHA_HUNTER_PARAMS", {}).get(
            "market_breadth_threshold", 0.40
        )
        return {"advance_ratio": advance_ratio, "is_bullish": advance_ratio >= float(threshold)}


def _compute_single_stock(args: Tuple) -> Optional[Dict]:
    """单股因子计算 (用于多进程)"""

    code, df, _target_date, compute_config = args

    try:
        n = len(df)
        if n < 60:
            return None

        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        vol = df["vol"].to_numpy(dtype=np.float64)
        amount = df["amount"].to_numpy(dtype=np.float64)

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

        std_win = min(200, len(slope))
        recent = slope[-std_win:]
        zscore = (slope[-1] - recent.mean()) / (recent.std() + 1e-10)
        rsrs_zscore = float(zscore * r2[-1])

        price_change = np.diff(close, prepend=close[0])
        obv = np.cumsum(np.sign(price_change) * vol)
        obv_ma = np.convolve(obv, np.ones(20) / 20, mode="valid")
        obv_trend = 1 if obv[-1] > obv_ma[-1] else 0

        vwap = amount[-20:].sum() / (vol[-20:].sum() + 1e-10)
        vwap_bias = float((close[-1] - vwap) / (vwap + 1e-10))

        vol_60 = vol[-60:]
        vol_rank = float((vol[-1] > vol_60[:-1]).sum() / (len(vol_60) - 1))

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
        atr = tr[-14:].mean()
        atr_pct = float(atr / (close[-1] + 1e-10))

        result: Dict[str, Any] = {
            "code": code,
            "close": float(close[-1]),
            "vol": float(vol[-1]),
            "amount": float(amount[-1]),
            "rsrs_zscore": rsrs_zscore,
            "rsrs_r2": float(r2[-1]),
            "obv_trend": int(obv_trend),
            "vwap_bias": vwap_bias,
            "vol_rank": vol_rank,
            "atr_pct": atr_pct,
        }

        if compute_config.get("need_ma"):
            if n >= 5:
                result["ma5"] = float(close[-5:].mean())
                result["vol_ma5"] = float(vol[-5:].mean())
            if n >= 20:
                result["ma20"] = float(close[-20:].mean())

            if n >= 8:
                ma5_series = np.convolve(close, np.ones(5) / 5, mode="valid")
                if len(ma5_series) >= 4:
                    result["ma5_slope"] = float(
                        (ma5_series[-1] - ma5_series[-4]) / (ma5_series[-4] + 1e-10)
                    )

        if compute_config.get("need_momentum"):
            m_win = int(compute_config.get("momentum_window", settings.factor.MOMENTUM_WINDOW))
            if n > m_win:
                result["momentum"] = float((close[-1] - close[-m_win - 1]) / (close[-m_win - 1] + 1e-10))

                rets = np.diff(np.log(close[-m_win - 1 :]))
                result["volatility"] = float(rets.std() * np.sqrt(252))

        if compute_config.get("need_alpha_extras"):
            if n >= 2:
                result["prev_close"] = float(close[-2])

            # 压力位距离 (简化): 最近 60 日最高价相对距离
            if n >= 60:
                pressure_level = float(np.max(high[-60:]))
            else:
                pressure_level = float(np.max(high))

            result["pressure_distance"] = float((pressure_level - close[-1]) / (close[-1] + 1e-10))

            # 换手率估算 (与策略中简化版本一致)
            if n >= 5:
                avg_amount = float(np.mean(amount[-5:]))
                avg_vol = float(np.mean(vol[-5:]))
                est_market_cap = float(close[-1] * avg_vol * 100)
                turnover = avg_amount / est_market_cap if est_market_cap > 0 else 0.0
                result["turnover"] = float(turnover)

        return result

    except Exception:
        return None


def scan_market(
    db_path: str | None = None,
    target_date: str | None = None,
    top_n: int = 50,
    mode: str = "factor",
    filters: Dict | None = None,
    strategy_params: Dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    """快捷扫描接口"""

    scanner = MarketScanner(db_path)

    merged_filters = {**(filters or {}), **kwargs}

    return scanner.scan(
        target_date=target_date,
        top_n=top_n,
        mode=mode,
        filters=merged_filters,
        strategy_params=strategy_params,
    )
