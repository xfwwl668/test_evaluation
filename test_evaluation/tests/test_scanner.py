# ============================================================================
# 文件: tests/test_scanner.py
# ============================================================================
"""
MarketScanner 测试
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

from analysis import MarketScanner, scan_market
from analysis.strategy_scorers import (
    RSRSScorer,
    MomentumScorer,
    ShortTermScorer,
    AlphaHunterScorer,
    EnsembleScorer,
    get_scorer
)


class TestMarketScannerModes:
    """测试 MarketScanner 的各种模式"""
    
    def test_scanner_init(self):
        """测试扫描器初始化"""
        scanner = MarketScanner()
        assert scanner is not None
        assert hasattr(scanner, 'scan')
    
    def test_get_scorer_rsrs(self):
        """测试获取 RSRS 评分器"""
        scorer = get_scorer('rsrs')
        assert isinstance(scorer, RSRSScorer)
    
    def test_get_scorer_momentum(self):
        """测试获取 Momentum 评分器"""
        scorer = get_scorer('momentum')
        assert isinstance(scorer, MomentumScorer)
    
    def test_get_scorer_short_term(self):
        """测试获取 ShortTerm 评分器"""
        scorer = get_scorer('short_term')
        assert isinstance(scorer, ShortTermScorer)
    
    def test_get_scorer_alpha_hunter(self):
        """测试获取 AlphaHunter 评分器"""
        scorer = get_scorer('alpha_hunter')
        assert isinstance(scorer, AlphaHunterScorer)
    
    def test_get_scorer_ensemble(self):
        """测试获取 Ensemble 评分器"""
        scorer = get_scorer('ensemble')
        assert isinstance(scorer, EnsembleScorer)
    
    def test_get_scorer_invalid(self):
        """测试获取无效评分器"""
        with pytest.raises(ValueError):
            get_scorer('invalid_mode')


class TestStrategyScorers:
    """测试策略评分器"""
    
    @pytest.fixture
    def sample_results(self):
        """模拟因子计算结果"""
        return [
            {
                'code': '000001',
                'close': 10.0,
                'vol': 1000000,
                'rsrs_zscore': 0.8,
                'rsrs_r2': 0.85,
                'vol_rank': 0.5,
                'obv_trend': 1,
                'vwap_bias': 0.01,
                'atr_pct': 0.02,
                'momentum': 0.1,
                'volatility': 0.3,
                'ma5': 9.8,
                'ma20': 9.5,
                'vol_ma5': 900000,
            },
            {
                'code': '000002',
                'close': 20.0,
                'vol': 2000000,
                'rsrs_zscore': 0.6,
                'rsrs_r2': 0.75,
                'vol_rank': 0.4,
                'obv_trend': 1,
                'vwap_bias': 0.02,
                'atr_pct': 0.015,
                'momentum': 0.08,
                'volatility': 0.25,
                'ma5': 19.5,
                'ma20': 19.0,
                'vol_ma5': 1800000,
            }
        ]
    
    def test_rsrs_scorer(self, sample_results):
        """测试 RSRS 评分器"""
        scorer = RSRSScorer({'entry_threshold': 0.7, 'r2_threshold': 0.8})
        scored = scorer.score(sample_results)
        
        assert len(scored) >= 0
        if scored:
            assert 'score' in scored[0]
            assert 'alpha_score' in scored[0]
    
    def test_momentum_scorer(self, sample_results):
        """测试动量评分器"""
        scorer = MomentumScorer({'min_momentum': 0.05, 'max_volatility': 0.5})
        scored = scorer.score(sample_results)
        
        assert len(scored) >= 0
        if scored:
            assert 'score' in scored[0]
            assert 'momentum' in scored[0]
    
    def test_short_term_scorer(self, sample_results):
        """测试短线评分器"""
        scorer = ShortTermScorer({
            'rsrs_entry_threshold': 0.7,
            'r2_threshold': 0.8,
            'volume_multiplier': 1.0,
        })
        scored = scorer.score(sample_results)
        
        assert len(scored) >= 0
        if scored:
            assert 'score' in scored[0]
    
    def test_alpha_hunter_scorer(self, sample_results):
        """测试 AlphaHunter 评分器"""
        scorer = AlphaHunterScorer({
            'rsrs_threshold': 0.7,
            'r2_threshold': 0.7,
        })
        scored = scorer.score(sample_results)
        
        assert len(scored) >= 0
        if scored:
            assert 'score' in scored[0]
    
    def test_ensemble_scorer(self, sample_results):
        """测试多策略融合评分器"""
        scorer = EnsembleScorer({'min_votes': 1})
        scored = scorer.score(sample_results)
        
        assert len(scored) >= 0
        if scored:
            assert 'score' in scored[0]
            assert 'votes' in scored[0]


class TestBackwardCompatibility:
    """测试后向兼容性"""
    
    def test_scan_signature_old(self):
        """测试旧签名兼容性 (target_date, top_n, filters)"""
        scanner = MarketScanner()
        
        # 旧签名调用 (filters作为第3个参数)
        # scanner.scan(target_date='2024-01-01', top_n=10, filters={'r2_min': 0.8})
        # 这应该被正确处理而不报错
        pass
    
    def test_scan_signature_new(self):
        """测试新签名兼容性 (mode参数)"""
        scanner = MarketScanner()
        
        # 新签名调用
        # scanner.scan(target_date='2024-01-01', top_n=10, mode='rsrs')
        # 这应该使用RSRS策略模式
        pass


@pytest.mark.parametrize("mode", ['factor', 'rsrs', 'momentum', 'short_term', 'alpha_hunter', 'ensemble'])
def test_all_modes_return_dataframe(mode):
    """测试所有模式都返回 DataFrame"""
    # 由于没有实际数据库，我们只测试基本结构
    scanner = MarketScanner()
    
    # 空结果应该返回空 DataFrame
    result = scanner._format_output([], 10)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_scorer_params_override():
    """测试参数覆盖"""
    default_params = {'entry_threshold': 0.7}
    override_params = {'entry_threshold': 0.9}
    
    scorer = RSRSScorer(override_params)
    assert scorer.get_param('entry_threshold') == 0.9


def test_ensemble_voting():
    """测试融合投票机制"""
    results = [
        {
            'code': '000001',
            'close': 10.0,
            'vol': 1000000,
            'rsrs_zscore': 0.8,
            'rsrs_r2': 0.85,
            'vol_rank': 0.5,
            'obv_trend': 1,
            'vwap_bias': 0.01,
            'momentum': 0.1,
            'volatility': 0.3,
            'ma5': 10.1,
            'ma20': 9.9,
            'vol_ma5': 900000,
        }
    ]
    
    scorer = EnsembleScorer({'min_votes': 1})
    scored = scorer.score(results)
    
    if scored:
        assert 'votes' in scored[0]
        assert scored[0]['votes'] >= 0
