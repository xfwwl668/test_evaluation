# ============================================================================
# 文件: analysis/stock_doctor.py
# ============================================================================
"""
单股深度诊断 - 多维度分析
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from core.database import StockDatabase
from factors import FactorRegistry
from config import settings


class MarketRegime(Enum):
    """市场状态"""
    EXTREME_STRONG = "🔥 极度强势"
    STRONG_TREND = "📈 强势趋势"
    HEALTHY_PULLBACK = "💚 健康回调"
    CONSOLIDATION = "📊 横盘整理"
    WEAK_REBOUND = "⚠️ 弱势反弹"
    VOLUME_DIVERGE = "🚨 缩量诱多"
    BREAKDOWN_WARN = "🔻 破位预警"
    CAPITULATION = "💀 恐慌杀跌"


@dataclass
class DiagnosisResult:
    """诊断结果"""
    code: str
    date: str
    close: float
    
    # 多周期 RSRS
    rsrs_short: float
    rsrs_mid: float
    rsrs_resonance: str
    
    # 量价分析
    price_vol_corr: float
    vol_pattern: str
    turnover_regime: str
    
    # 筹码分布
    profit_ratio: float
    chip_zone: str
    avg_cost: float
    
    # 压力支撑
    resistance_1: float
    resistance_2: float
    support_1: float
    support_2: float
    price_position: float
    
    # 综合诊断
    regime: MarketRegime
    score: float
    signals: List[str]
    
    # 原始数据
    df: pd.DataFrame = field(repr=False, default=None)


class StockDoctor:
    """
    单股诊断引擎
    
    分析维度:
    1. 多周期趋势 (RSRS 18日/60日)
    2. 量价关系
    3. 筹码分布
    4. 压力支撑位
    5. 综合诊断
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(settings.path.DB_PATH)
        self.db = StockDatabase(self.db_path)
        self.logger = logging.getLogger("StockDoctor")
    
    def diagnose(self, code: str) -> DiagnosisResult:
        """执行诊断"""
        # 加载数据
        df = self._load_data(code)
        if len(df) < 250:
            raise ValueError(f"数据不足: {len(df)} 天 (需要 250+)")
        
        latest_date = str(df.index[-1].date())
        close = df['close'].iloc[-1]
        
        # 多维度分析
        rsrs_result = self._analyze_multi_period_rsrs(df)
        vol_result = self._analyze_volume_price(df)
        chip_result = self._analyze_chip_distribution(df)
        level_result = self._analyze_support_resistance(df)
        
        # 综合诊断
        regime, score, signals = self._comprehensive_diagnosis(
            rsrs_result, vol_result, chip_result, level_result, df
        )
        
        return DiagnosisResult(
            code=code,
            date=latest_date,
            close=close,
            **rsrs_result,
            **vol_result,
            **chip_result,
            **level_result,
            regime=regime,
            score=score,
            signals=signals,
            df=df
        )
    
    def _load_data(self, code: str) -> pd.DataFrame:
        """加载数据"""
        df = self.db.get_stock_history(code)
        if df.empty:
            raise ValueError(f"股票 {code} 未找到")
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df.tail(500)
    
    def _analyze_multi_period_rsrs(self, df: pd.DataFrame) -> Dict:
        """多周期RSRS分析"""
        rsrs_short = self._calc_rsrs(df, window=18)
        rsrs_mid = self._calc_rsrs(df, window=60)
        
        # 判断共振
        if rsrs_short > 0.7 and rsrs_mid > 0.5:
            resonance = "✅ 多头共振 (短中期同步走强)"
        elif rsrs_short < -0.7 and rsrs_mid < -0.5:
            resonance = "🔻 空头共振 (短中期同步走弱)"
        elif rsrs_short > 0.5 and rsrs_mid < -0.3:
            resonance = "⚠️ 短强中弱 (反弹待确认)"
        elif rsrs_short < -0.5 and rsrs_mid > 0.3:
            resonance = "📉 短弱中强 (回调蓄势)"
        else:
            resonance = "📊 周期背离 (震荡整理)"
        
        return {
            'rsrs_short': round(rsrs_short, 3),
            'rsrs_mid': round(rsrs_mid, 3),
            'rsrs_resonance': resonance
        }
    
    def _calc_rsrs(self, df: pd.DataFrame, window: int) -> float:
        """计算RSRS"""
        from numpy.lib.stride_tricks import sliding_window_view
        
        high = df['high'].to_numpy(dtype=np.float64)
        low = df['low'].to_numpy(dtype=np.float64)
        
        if len(high) < window + 60:
            return 0.0
        
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
        
        return zscore * r2[-1]
    
    def _analyze_volume_price(self, df: pd.DataFrame) -> Dict:
        """量价分析"""
        close = df['close'].to_numpy()
        vol = df['vol'].to_numpy().astype(np.float64)
        
        # 相关系数
        price_ret = np.diff(close[-60:]) / close[-60:-1]
        vol_ret = np.diff(vol[-60:]) / (vol[-60:-1] + 1)
        corr = np.corrcoef(price_ret, vol_ret)[0, 1]
        
        # 模式判断
        if corr > 0.4:
            pattern = "📈 量价齐升 (健康上涨)"
        elif corr > 0.1:
            pattern = "📊 量价温和 (稳步推进)"
        elif corr > -0.1:
            pattern = "⚖️ 量价背离 (方向待定)"
        elif corr > -0.4:
            pattern = "⚠️ 量增价滞 (上行乏力)"
        else:
            pattern = "🚨 量缩价涨 (警惕诱多)"
        
        # 换手率状态
        vol_5 = vol[-5:].mean()
        vol_20 = vol[-20:].mean()
        
        if vol_5 > vol_20 * 1.5:
            turnover = "🔥 异常放量"
        elif vol_5 > vol_20 * 1.1:
            turnover = "📈 温和放量"
        elif vol_5 > vol_20 * 0.7:
            turnover = "📊 正常换手"
        else:
            turnover = "📉 明显缩量"
        
        return {
            'price_vol_corr': round(corr, 3),
            'vol_pattern': pattern,
            'turnover_regime': turnover
        }
    
    def _analyze_chip_distribution(self, df: pd.DataFrame) -> Dict:
        """筹码分布分析"""
        close = df['close'].to_numpy()
        vol = df['vol'].to_numpy().astype(np.float64)
        amount = df['amount'].to_numpy().astype(np.float64)
        
        current_price = close[-1]
        
        # 多周期VWAP
        periods = [5, 10, 20, 40, 60, 120]
        vwaps = []
        weights = []
        
        for p in periods:
            if len(close) >= p:
                v = amount[-p:].sum() / (vol[-p:].sum() + 1e-10)
                w = vol[-p:].sum()
                vwaps.append(v)
                weights.append(w)
        
        if not vwaps:
            return {'profit_ratio': 0.5, 'chip_zone': "无法计算", 'avg_cost': current_price}
        
        weights = np.array(weights) / sum(weights)
        avg_cost = sum(v * w for v, w in zip(vwaps, weights))
        
        profit_count = sum(1 for v in vwaps if current_price > v)
        profit_ratio = profit_count / len(vwaps)
        
        cost_ratio = current_price / avg_cost
        
        if cost_ratio > 1.15:
            chip_zone = "🟢 深度获利区"
        elif cost_ratio > 1.05:
            chip_zone = "💚 获利区"
        elif cost_ratio > 0.98:
            chip_zone = "⚖️ 成本区"
        elif cost_ratio > 0.90:
            chip_zone = "🟡 浅套区"
        else:
            chip_zone = "🔴 深套区"
        
        return {
            'profit_ratio': round(profit_ratio, 2),
            'chip_zone': chip_zone,
            'avg_cost': round(avg_cost, 2)
        }
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict:
        """压力支撑分析"""
        close = df['close'].to_numpy()[-250:]
        high = df['high'].to_numpy()[-250:]
        low = df['low'].to_numpy()[-250:]
        vol = df['vol'].to_numpy()[-250:].astype(np.float64)
        
        current = close[-1]
        
        # 成交量加权价格分布
        price_bins = np.linspace(low.min(), high.max(), 50)
        vol_profile = np.zeros(len(price_bins) - 1)
        
        for i in range(len(close)):
            idx = np.searchsorted(price_bins, close[i]) - 1
            idx = max(0, min(idx, len(vol_profile) - 1))
            vol_profile[idx] += vol[i]
        
        # 找成交密集区
        peak_idx = np.argsort(vol_profile)[-5:]
        peak_prices = [(price_bins[i] + price_bins[i+1]) / 2 for i in peak_idx]
        
        resistances = sorted([p for p in peak_prices if p > current])[:2]
        supports = sorted([p for p in peak_prices if p < current], reverse=True)[:2]
        
        # 补充历史高低点
        high_250 = high.max()
        low_250 = low.min()
        
        r1 = resistances[0] if resistances else current * 1.05
        r2 = resistances[1] if len(resistances) > 1 else min(high_250, current * 1.10)
        s1 = supports[0] if supports else current * 0.95
        s2 = supports[1] if len(supports) > 1 else max(low_250, current * 0.90)
        
        price_position = (current - low_250) / (high_250 - low_250 + 1e-10)
        
        return {
            'resistance_1': round(r1, 2),
            'resistance_2': round(r2, 2),
            'support_1': round(s1, 2),
            'support_2': round(s2, 2),
            'price_position': round(price_position, 3)
        }
    
    def _comprehensive_diagnosis(
        self,
        rsrs: Dict,
        vol: Dict,
        chip: Dict,
        level: Dict,
        df: pd.DataFrame
    ) -> Tuple[MarketRegime, float, List[str]]:
        """综合诊断"""
        signals = []
        score = 0.0
        
        # 趋势得分 (40%)
        rsrs_avg = (rsrs['rsrs_short'] + rsrs['rsrs_mid']) / 2
        if rsrs_avg > 1.0:
            score += 0.4
            signals.append("📈 强势多头趋势")
        elif rsrs_avg > 0.3:
            score += 0.2
            signals.append("📊 温和上行")
        elif rsrs_avg < -1.0:
            score -= 0.4
            signals.append("🔻 强势空头趋势")
        elif rsrs_avg < -0.3:
            score -= 0.2
            signals.append("📉 弱势下行")
        
        # 量价得分 (25%)
        corr = vol['price_vol_corr']
        if corr > 0.3:
            score += 0.25
            signals.append("✅ 量价配合良好")
        elif corr < -0.3:
            score -= 0.25
            signals.append("🚨 量价严重背离")
        
        # 筹码得分 (20%)
        profit = chip['profit_ratio']
        if profit > 0.8:
            score += 0.2
            signals.append("💰 筹码获利充分")
        elif profit < 0.3:
            score -= 0.2
            signals.append("🔴 筹码深度套牢")
        
        # 位置得分 (15%)
        pos = level['price_position']
        if pos > 0.9:
            score -= 0.1
            signals.append("⚠️ 接近历史高位")
        elif pos < 0.2:
            score += 0.1
            signals.append("💡 接近历史低位")
        
        # 判断状态
        is_volume_diverge = (rsrs['rsrs_short'] > 0.5 and corr < -0.3)
        is_breakdown = (rsrs['rsrs_short'] < -0.7 and df['close'].iloc[-1] < level['support_1'])
        
        if is_volume_diverge:
            regime = MarketRegime.VOLUME_DIVERGE
            signals.insert(0, "🚨 缩量诱多信号!")
        elif is_breakdown:
            regime = MarketRegime.BREAKDOWN_WARN
            signals.insert(0, "🔻 破位预警!")
        elif score > 0.6:
            regime = MarketRegime.EXTREME_STRONG
        elif score > 0.3:
            regime = MarketRegime.STRONG_TREND
        elif score > 0.1:
            regime = MarketRegime.HEALTHY_PULLBACK
        elif score > -0.1:
            regime = MarketRegime.CONSOLIDATION
        elif score > -0.3:
            regime = MarketRegime.WEAK_REBOUND
        else:
            regime = MarketRegime.CAPITULATION
        
        return regime, round(score, 3), signals
    
    def generate_report(self, result: DiagnosisResult) -> str:
        """生成诊断报告"""
        pos = result.price_position
        bar_len = 40
        pos_idx = int(pos * bar_len)
        price_bar = "─" * pos_idx + "◆" + "─" * (bar_len - pos_idx - 1)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🔬 单股深度诊断报告                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  股票代码: {result.code:<10}  诊断日期: {result.date}  收盘: {result.close:<8.2f}  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  【综合诊断】 {result.regime.value:<40}                    ║
║   综合评分: {result.score:+.3f}                                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  📊 多周期 RSRS                                                          ║
║   18日: {result.rsrs_short:+.3f}   60日: {result.rsrs_mid:+.3f}                                  ║
║   {result.rsrs_resonance:<60}║
╠══════════════════════════════════════════════════════════════════════════╣
║  📈 量价关系                                                             ║
║   相关系数: {result.price_vol_corr:+.3f}  {result.vol_pattern:<40}║
║   换手状态: {result.turnover_regime:<50}║
╠══════════════════════════════════════════════════════════════════════════╣
║  💰 筹码分布                                                             ║
║   平均成本: ¥{result.avg_cost:<8.2f}  获利比例: {result.profit_ratio*100:>5.1f}%                     ║
║   {result.chip_zone:<60}║
╠══════════════════════════════════════════════════════════════════════════╣
║  🎯 压力/支撑位                                                          ║
║   压力: ¥{result.resistance_1:<8.2f} / ¥{result.resistance_2:<8.2f}                                ║
║   支撑: ¥{result.support_1:<8.2f} / ¥{result.support_2:<8.2f}                                ║
║   位置: [{price_bar}] {result.price_position*100:.0f}%   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  🔔 关键信号                                                             ║"""
        
        for i, signal in enumerate(result.signals[:5], 1):
            report += f"\n║   {i}. {signal:<65}║"
        
        report += """
╚══════════════════════════════════════════════════════════════════════════╝
"""
        return report


def analyze_stock(code: str, db_path: str = None) -> DiagnosisResult:
    """
    快捷诊断接口
    
    Usage:
        result = analyze_stock('000001')
        print(StockDoctor().generate_report(result))
    """
    doctor = StockDoctor(db_path)
    return doctor.diagnose(code)