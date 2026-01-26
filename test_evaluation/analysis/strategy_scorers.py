# ============================================================================
# 文件: analysis/strategy_scorers.py
# ============================================================================
"""
策略评分器 - 将各策略的选股逻辑提取为独立评分器

功能:
- 提供统一的评分器接口
- 实现各策略的入场规则
- 支持多策略融合
"""
import pandas as pd
from typing import Any, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import logging


@dataclass
class ScoredStock:
    """评分后的股票"""
    code: str
    close: float
    score: float
    details: Dict  # 详细信息


class StrategyScorer(ABC):
    """
    策略评分器抽象基类
    
    所有评分器必须实现:
    - score(): 对候选股票进行评分和过滤
    """
    
    def __init__(self, params: Dict = None):
        self.params = params or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_param(self, key: str, default=None):
        """获取参数"""
        return self.params.get(key, default)
    
    @abstractmethod
    def score(
        self,
        results: List[Dict],
        filters: Dict = None
    ) -> List[Dict]:
        """
        对候选股票进行评分和过滤
        
        Args:
            results: 原始因子结果列表
            filters: 额外的过滤条件
            
        Returns:
            评分后的股票列表，包含 'score' 和 'alpha_score' 字段
        """
        pass


class RSRSScorer(StrategyScorer):
    """
    RSRS 策略评分器
    
    选股逻辑:
    - RSRS Z-Score > entry_threshold (默认 0.7)
    - R² >= r2_threshold (默认 0.8)
    - 可选量价过滤:
      * 成交量温和 (30%-75% 分位)
      * OBV 上升趋势
      * 价格在 VWAP 上方
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'entry_threshold': 0.7,
            'r2_threshold': 0.8,
            'use_volume_filter': True,
            'vol_warm_low': 0.3,
            'vol_warm_high': 0.75,
        }
        merged = {**default_params, **(params or {})}
        super().__init__(merged)
    
    def score(
        self,
        results: List[Dict],
        filters: Dict = None
    ) -> List[Dict]:
        """RSRS 策略评分"""
        filters = filters or {}
        
        entry_th = self.get_param('entry_threshold')
        r2_th = self.get_param('r2_threshold')
        use_vol_filter = self.get_param('use_volume_filter')
        vol_low = self.get_param('vol_warm_low')
        vol_high = self.get_param('vol_warm_high')
        
        qualified = []
        
        for r in results:
            # 基础条件: RSRS > 阈值
            rsrs = r.get('rsrs_zscore', 0)
            if rsrs <= entry_th:
                continue
            
            # R² 有效性
            r2 = r.get('rsrs_r2', 0)
            if r2 < r2_th:
                continue
            
            # 量价过滤 (可选)
            if use_vol_filter:
                vol_rank = r.get('vol_rank', 0.5)
                if not (vol_low <= vol_rank <= vol_high):
                    continue
                
                obv_trend = r.get('obv_trend', 0)
                if obv_trend < 1:
                    continue
                
                vwap_bias = r.get('vwap_bias', 0)
                if vwap_bias <= 0:
                    continue
            
            # 评分 = RSRS Z-Score
            score = rsrs
            
            # 添加 alpha_score 用于排序
            qualified.append({
                **r,
                'score': score,
                'alpha_score': score
            })
        
        # 按 RSRS 分数排序
        qualified.sort(key=lambda x: x['score'], reverse=True)
        
        return qualified


class MomentumScorer(StrategyScorer):
    """
    动量策略评分器
    
    选股逻辑:
    - 动量 >= min_momentum (默认 0.05)
    - 波动率 <= max_volatility (默认 0.5)
    - 按动量排序
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'min_momentum': 0.05,
            'max_volatility': 0.5,
        }
        merged = {**default_params, **(params or {})}
        super().__init__(merged)
    
    def score(
        self,
        results: List[Dict],
        filters: Dict = None
    ) -> List[Dict]:
        """动量策略评分"""
        filters = filters or {}
        
        min_mom = self.get_param('min_momentum')
        max_vol = self.get_param('max_volatility')
        
        qualified = []
        
        for r in results:
            # 获取动量因子 (需要提前计算)
            momentum = r.get('momentum', 0)
            volatility = r.get('volatility', 0.3)
            
            # 过滤条件
            if momentum < min_mom:
                continue
            
            if volatility > max_vol:
                continue
            
            # 评分 = 动量值
            score = momentum
            
            qualified.append({
                **r,
                'score': score,
                'alpha_score': score
            })
        
        # 按动量排序
        qualified.sort(key=lambda x: x['score'], reverse=True)
        
        return qualified


class ShortTermScorer(StrategyScorer):
    """
    短线 RSRS 策略评分器
    
    选股逻辑:
    - RSRS Score > rsrs_entry_threshold (默认 0.7)
    - R² >= r2_threshold (默认 0.8)
    - Price > MA5 AND Price > MA20
    - Volume > MA5_Vol * volume_multiplier (默认 1.5)
    - 价格和成交量基础过滤
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'rsrs_entry_threshold': 0.7,
            'r2_threshold': 0.8,
            'volume_multiplier': 1.5,
            'min_price': 3.0,
            'max_price': 100.0,
            'min_volume': 1000000,
        }
        merged = {**default_params, **(params or {})}
        super().__init__(merged)
    
    def score(
        self,
        results: List[Dict],
        filters: Dict = None
    ) -> List[Dict]:
        """短线 RSRS 策略评分"""
        filters = filters or {}
        
        rsrs_th = self.get_param('rsrs_entry_threshold')
        r2_th = self.get_param('r2_threshold')
        vol_mult = self.get_param('volume_multiplier')
        min_price = self.get_param('min_price')
        max_price = self.get_param('max_price')
        min_volume = self.get_param('min_volume')
        
        qualified = []
        
        for r in results:
            close = r.get('close', 0)
            volume = r.get('vol', 0)
            
            # 基础过滤
            if close < min_price or close > max_price:
                continue
            
            if volume < min_volume:
                continue
            
            # RSRS 过滤
            rsrs_score = r.get('rsrs_zscore', 0)
            if rsrs_score <= rsrs_th:
                continue
            
            rsrs_r2 = r.get('rsrs_r2', 0)
            if rsrs_r2 < r2_th:
                continue
            
            # 均线趋势过滤
            ma5 = r.get('ma5', 0)
            ma20 = r.get('ma20', 0)
            if ma5 > 0 and close <= ma5:
                continue
            if ma20 > 0 and close <= ma20:
                continue
            
            # 放量突破过滤
            vol_ma5 = r.get('vol_ma5', 0)
            if vol_ma5 > 0:
                vol_ratio = volume / vol_ma5
                if vol_ratio < vol_mult:
                    continue
            else:
                vol_ratio = 1.0
            
            # 综合评分
            score = rsrs_score * rsrs_r2
            
            qualified.append({
                **r,
                'score': score,
                'alpha_score': score,
                'vol_ratio': vol_ratio
            })
        
        # 按综合评分排序
        qualified.sort(key=lambda x: x['score'], reverse=True)
        
        return qualified


class AlphaHunterScorer(StrategyScorer):
    """
    Alpha Hunter 策略评分器
    
    选股逻辑:
    - RSRS >= rsrs_threshold (默认 0.8)
    - R² >= r2_threshold (默认 0.85)
    - Price > MA5 AND MA5_slope > slope_threshold
    - 换手率 <= max_turnover (默认 0.25)
    - 距离压力位 >= min_pressure_distance (默认 0.05)
    - 非涨停板
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'rsrs_threshold': 0.8,
            'r2_threshold': 0.85,
            'ma5_slope_threshold': 0.001,
            'max_turnover': 0.25,
            'min_pressure_distance': 0.05,
            'min_price': 5.0,
            'max_price': 80.0,
            'min_volume': 2000000,
            'allow_limit_up_chase': False,
        }
        merged = {**default_params, **(params or {})}
        super().__init__(merged)
    
    def score(
        self,
        results: List[Dict],
        filters: Dict = None
    ) -> List[Dict]:
        """Alpha Hunter 策略评分"""
        filters = filters or {}
        
        # 市场广度过滤（AlphaHunter 策略核心逻辑之一）
        breadth = filters.get('market_breadth')
        if isinstance(breadth, dict) and not breadth.get('is_bullish', True):
            return []
        
        rsrs_th = self.get_param('rsrs_threshold')
        r2_th = self.get_param('r2_threshold')
        slope_th = self.get_param('ma5_slope_threshold')
        max_to = self.get_param('max_turnover')
        min_pressure = self.get_param('min_pressure_distance')
        min_price = self.get_param('min_price')
        max_price = self.get_param('max_price')
        min_volume = self.get_param('min_volume')
        allow_limit = self.get_param('allow_limit_up_chase')
        
        qualified = []
        
        for r in results:
            close = r.get('close', 0)
            amount = r.get('amount', 0)
            
            # 基础过滤
            if close < min_price or close > max_price:
                continue
            
            if amount < min_volume:
                continue
            
            # RSRS 过滤
            rsrs = r.get('rsrs_zscore', 0)
            if rsrs < rsrs_th:
                continue
            
            r2 = r.get('rsrs_r2', 0)
            if r2 < r2_th:
                continue
            
            # MA5 趋势过滤
            ma5 = r.get('ma5', 0)
            if ma5 > 0 and close <= ma5:
                continue
            
            ma5_slope = r.get('ma5_slope', 0)
            if ma5_slope < slope_th:
                continue
            
            # 换手率过滤
            turnover = r.get('turnover', 0)
            if turnover > max_to:
                continue
            
            # 压力位过滤
            pressure_dist = r.get('pressure_distance', 0.1)
            if pressure_dist < min_pressure:
                continue
            
            # 涨停过滤
            if not allow_limit:
                prev_close = r.get('prev_close', close * 0.95)
                if prev_close > 0 and close >= prev_close * 1.095:
                    continue
            
            # 综合评分
            score = rsrs * r2
            
            qualified.append({
                **r,
                'score': score,
                'alpha_score': score
            })
        
        # 按综合评分排序
        qualified.sort(key=lambda x: x['score'], reverse=True)
        
        return qualified


class EnsembleScorer(StrategyScorer):
    """
    多策略融合评分器
    
    融合方式:
    - 使用各策略评分器独立打分
    - 投票融合: 股票在多个策略中出现的次数
    - 加权评分: 结合各策略的评分和权重
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'weights': {
                'rsrs': 0.3,
                'momentum': 0.2,
                'short_term': 0.3,
                'alpha_hunter': 0.2,
            },
            'min_votes': 2,  # 至少出现在 2 个策略中
            'voting_mode': 'weighted',  # 'count' or 'weighted'
        }
        merged = {**default_params, **(params or {})}
        super().__init__(merged)
        
        # 创建各策略评分器
        self.scorers = {
            'rsrs': RSRSScorer(params.get('rsrs_params', {}) if params else {}),
            'momentum': MomentumScorer(params.get('momentum_params', {}) if params else {}),
            'short_term': ShortTermScorer(params.get('short_term_params', {}) if params else {}),
            'alpha_hunter': AlphaHunterScorer(params.get('alpha_hunter_params', {}) if params else {}),
        }
    
    def score(
        self,
        results: List[Dict],
        filters: Dict = None
    ) -> List[Dict]:
        """多策略融合评分"""
        filters = filters or {}
        
        weights = self.get_param('weights')
        min_votes = self.get_param('min_votes')
        voting_mode = self.get_param('voting_mode')
        
        # 各策略独立评分（大样本时可并行）
        strategy_results: Dict[str, Dict[str, Dict]] = {}
        
        parallel_threshold = self.get_param('parallel_threshold', 2000)
        use_parallel = len(results) >= parallel_threshold
        
        if use_parallel:
            tasks = []
            for name in self.scorers.keys():
                params_key = f"{name}_params"
                params = self.params.get(params_key, {})
                tasks.append((name, params, results, filters))
            
            n_workers = min(mp.cpu_count(), len(tasks))
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for name, mapping in executor.map(_ensemble_score_worker, tasks):
                    strategy_results[name] = mapping
        else:
            for strategy_name, scorer in self.scorers.items():
                try:
                    scored = scorer.score(results, filters)
                    strategy_results[strategy_name] = {r['code']: r for r in scored}
                    self.logger.debug(f"{strategy_name}: {len(scored)} candidates")
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy_name} scoring failed: {e}")
                    strategy_results[strategy_name] = {}
        
        # 收集所有出现的股票代码
        all_codes = set()
        for results_dict in strategy_results.values():
            all_codes.update(results_dict.keys())
        
        # 融合评分
        ensemble_scores = []
        
        for code in all_codes:
            # 统计投票
            votes = 0
            weighted_score = 0.0
            details = {}
            
            for strategy_name, results_dict in strategy_results.items():
                if code in results_dict:
                    votes += 1
                    strategy_score = results_dict[code].get('score', 0)
                    weight = weights.get(strategy_name, 0)
                    weighted_score += strategy_score * weight
                    details[f'{strategy_name}_score'] = strategy_score
            
            # 过滤: 至少出现在 min_votes 个策略中
            if votes < min_votes:
                continue
            
            # 获取股票基础信息 (从任一策略结果中)
            stock_info = None
            for results_dict in strategy_results.values():
                if code in results_dict:
                    stock_info = results_dict[code]
                    break
            
            if stock_info is None:
                continue
            
            # 最终评分
            if voting_mode == 'count':
                final_score = votes
            else:  # weighted
                final_score = weighted_score
            
            ensemble_scores.append({
                **stock_info,
                'score': final_score,
                'alpha_score': final_score,
                'votes': votes,
                'ensemble_details': details
            })
        
        # 排序
        ensemble_scores.sort(key=lambda x: x['score'], reverse=True)
        
        self.logger.info(f"Ensemble fusion: {len(ensemble_scores)} candidates from {len(all_codes)} total")
        
        return ensemble_scores


def _ensemble_score_worker(args) -> tuple:
    """Ensemble 并行评分任务（用于 ProcessPoolExecutor）"""
    strategy_name, params, results, filters = args
    scorer = get_scorer(strategy_name, params)
    scored = scorer.score(results, filters)
    return strategy_name, {r['code']: r for r in scored}


def get_scorer(mode: str, params: Dict = None) -> StrategyScorer:
    """
    工厂方法: 根据模式获取评分器
    
    Args:
        mode: 选股模式 ('factor'|'rsrs'|'momentum'|'short_term'|'alpha_hunter'|'ensemble')
        params: 策略参数
        
    Returns:
        对应的评分器实例
    """
    scorers = {
        'rsrs': RSRSScorer,
        'momentum': MomentumScorer,
        'short_term': ShortTermScorer,
        'alpha_hunter': AlphaHunterScorer,
        'ensemble': EnsembleScorer,
    }
    
    if mode not in scorers:
        raise ValueError(f"Unknown scorer mode: {mode}. Available: {list(scorers.keys())}")
    
    scorer_class = scorers[mode]
    return scorer_class(params)
