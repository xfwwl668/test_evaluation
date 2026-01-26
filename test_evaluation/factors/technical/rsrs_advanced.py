# ============================================================================
# 文件: factors/technical/rsrs_advanced.py
# ============================================================================
"""
修正版 RSRS 因子 - 高胜率短线专用

核心改进:
1. R² 加权修正
2. 右偏标准分 (偏度修正)
3. 斜率加速度 (二阶导)
4. 多周期共振
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from typing import Tuple, Dict

from ..base import BaseFactor, FactorMeta, FactorRegistry
from config import settings


@FactorRegistry.register
class RSRSAdvancedFactor(BaseFactor):
    """
    修正版 RSRS 因子
    
    计算公式:
        1. 基础斜率: High = α + β × Low (OLS回归)
        2. Z-Score 标准化: z = (β - μ) / σ
        3. R² 加权: score = z × R²
        4. 右偏修正: score_adj = score × (1 - skew_penalty)
    
    输出:
        - rsrs_score: 最终修正得分
        - rsrs_r2: 拟合优度
        - rsrs_slope: 原始斜率
        - rsrs_zscore: 标准化分数
    """
    
    meta = FactorMeta(
        name="rsrs_advanced",
        category="technical",
        description="修正版 RSRS (R²加权 + 偏度修正)",
        lookback=650
    )
    
    def __init__(
        self,
        window: int = 18,
        std_window: int = 600,
        r2_threshold: float = 0.8,
        skew_penalty_factor: float = 0.1,
        **kwargs
    ):
        """
        Args:
            window: RSRS 回归窗口 (默认18日)
            std_window: 标准化滚动窗口 (默认600日)
            r2_threshold: R² 有效阈值
            skew_penalty_factor: 偏度惩罚系数
        """
        super().__init__(**kwargs)
        self.window = window
        self.std_window = std_window
        self.r2_threshold = r2_threshold
        self.skew_penalty_factor = skew_penalty_factor
        self.meta.lookback = std_window + window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算修正版 RSRS 得分"""
        result = self.compute_all(df)
        return result['rsrs_score']
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有 RSRS 相关指标
        
        Returns:
            DataFrame with columns: rsrs_slope, rsrs_r2, rsrs_zscore, rsrs_score
        """
        n = len(df)
        
        if n < self.window + 60:
            return pd.DataFrame({
                'rsrs_slope': np.nan,
                'rsrs_r2': np.nan,
                'rsrs_zscore': np.nan,
                'rsrs_score': np.nan,
                'rsrs_valid': 0
            }, index=df.index)
        
        high = df['high'].to_numpy(dtype=np.float64)
        low = df['low'].to_numpy(dtype=np.float64)
        
        # ===== Step 1: 滚动 OLS 回归 =====
        slope, r2, residual_std = self._rolling_ols(high, low)
        
        # 填充前 window-1 个 NaN
        pad = np.full(self.window - 1, np.nan)
        slope_full = np.concatenate([pad, slope])
        r2_full = np.concatenate([pad, r2])
        
        # ===== Step 2: 滚动标准化 (Z-Score) =====
        slope_series = pd.Series(slope_full, index=df.index)
        roll_mean = slope_series.rolling(self.std_window, min_periods=60).mean()
        roll_std = slope_series.rolling(self.std_window, min_periods=60).std()
        
        zscore = (slope_series - roll_mean) / roll_std.clip(lower=1e-10)
        
        # ===== Step 3: R² 加权 =====
        score_r2_weighted = zscore.values * r2_full
        
        # ===== Step 4: 右偏修正 (偏度惩罚) =====
        # 计算滚动偏度
        skewness = slope_series.rolling(self.std_window, min_periods=60).apply(
            lambda x: stats.skew(x, nan_policy='omit'), raw=False
        )
        
        # 右偏惩罚: 偏度越大，惩罚越重
        # 当偏度 > 0 (右偏)，说明有极端高值，需要对高分信号打折
        skew_penalty = np.clip(skewness.values * self.skew_penalty_factor, 0, 0.5)
        
        # 只对正分数施加惩罚 (负分数不惩罚)
        score_adjusted = np.where(
            score_r2_weighted > 0,
            score_r2_weighted * (1 - skew_penalty),
            score_r2_weighted
        )
        
        # ===== Step 5: 有效性标记 =====
        valid = (r2_full >= self.r2_threshold).astype(np.int8)
        
        # 无效信号衰减
        score_final = np.where(valid, score_adjusted, score_adjusted * 0.5)
        
        return pd.DataFrame({
            'rsrs_slope': slope_full,
            'rsrs_r2': r2_full,
            'rsrs_zscore': zscore.values,
            'rsrs_score': score_final,
            'rsrs_valid': valid
        }, index=df.index)
    
    def _rolling_ols(
        self,
        high: np.ndarray,
        low: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        向量化滚动 OLS 回归
        
        Model: High = α + β × Low
        
        Returns:
            (slope, r2, residual_std)
        """
        window = self.window
        n = len(high)
        
        # 滑动窗口视图
        low_win = sliding_window_view(low, window)
        high_win = sliding_window_view(high, window)
        
        # 均值
        x_mean = low_win.mean(axis=1, keepdims=True)
        y_mean = high_win.mean(axis=1, keepdims=True)
        
        # 偏差
        x_dev = low_win - x_mean
        y_dev = high_win - y_mean
        
        # 协方差和方差
        cov_xy = (x_dev * y_dev).sum(axis=1)
        var_x = (x_dev ** 2).sum(axis=1)
        var_y = (y_dev ** 2).sum(axis=1)
        
        # 斜率 β = Cov(X,Y) / Var(X)
        slope = np.divide(cov_xy, var_x, out=np.zeros_like(cov_xy), where=var_x > 1e-10)
        
        # 截距 α = ȳ - β × x̄
        intercept = y_mean.flatten() - slope * x_mean.flatten()
        
        # R² = Corr(X,Y)²
        denom = var_x * var_y
        r2 = np.divide(cov_xy ** 2, denom, out=np.zeros_like(cov_xy), where=denom > 1e-10)
        
        # 残差标准差 (用于异常检测)
        y_pred = slope.reshape(-1, 1) * low_win + intercept.reshape(-1, 1)
        residuals = high_win - y_pred
        residual_std = residuals.std(axis=1)
        
        return slope, r2, residual_std
    
    def get_signal_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        获取信号质量评估
        
        Returns:
            DataFrame with quality metrics
        """
        result = self.compute_all(df)
        
        # 信号强度分级
        score = result['rsrs_score']
        r2 = result['rsrs_r2']
        
        quality = pd.Series('neutral', index=df.index)
        quality[(score > 0.7) & (r2 > 0.85)] = 'strong_buy'
        quality[(score > 0.5) & (r2 > 0.8)] = 'buy'
        quality[(score < -0.7) & (r2 > 0.85)] = 'strong_sell'
        quality[(score < -0.5) & (r2 > 0.8)] = 'sell'
        
        result['signal_quality'] = quality
        
        return result


@FactorRegistry.register
class RSRSMomentumFactor(BaseFactor):
    """
    RSRS 动量因子 - 斜率变化速度
    
    捕捉 RSRS 斜率的加速/减速
    """
    
    meta = FactorMeta(
        name="rsrs_momentum",
        category="technical",
        description="RSRS斜率变化动量",
        lookback=30
    )
    
    def __init__(self, momentum_window: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.momentum_window = momentum_window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算 RSRS 动量"""
        rsrs_factor = RSRSAdvancedFactor()
        rsrs_data = rsrs_factor.compute_all(df)
        
        slope = rsrs_data['rsrs_slope']
        
        # 斜率变化率 (一阶导)
        slope_change = slope.diff(self.momentum_window)
        
        # 加速度 (二阶导)
        acceleration = slope_change.diff(self.momentum_window)
        
        # 综合动量 = 变化率 + 0.5 × 加速度
        momentum = slope_change + 0.5 * acceleration
        
        return momentum.rename(self.name)


@FactorRegistry.register  
class RSRSMultiPeriodFactor(BaseFactor):
    """
    多周期 RSRS 共振因子
    
    同时计算短期(18日)和中期(60日) RSRS，判断共振
    """
    
    meta = FactorMeta(
        name="rsrs_resonance",
        category="technical",
        description="多周期RSRS共振得分",
        lookback=700
    )
    
    def __init__(
        self,
        short_window: int = 18,
        mid_window: int = 60,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.short_window = short_window
        self.mid_window = mid_window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算共振得分"""
        # 短周期
        short_factor = RSRSAdvancedFactor(window=self.short_window)
        short_score = short_factor.compute(df)
        
        # 中周期
        mid_factor = RSRSAdvancedFactor(window=self.mid_window)
        mid_score = mid_factor.compute(df)
        
        # 共振评分
        # 同向加成，背离惩罚
        same_direction = (short_score * mid_score > 0).astype(float)
        
        # 共振得分 = 短期 × 0.6 + 中期 × 0.4 × 同向系数
        resonance = short_score * 0.6 + mid_score * 0.4 * (same_direction * 1.5 + (1 - same_direction) * 0.5)
        
        return resonance.rename(self.name)
    
    def compute_detail(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算详细共振数据"""
        short_factor = RSRSAdvancedFactor(window=self.short_window)
        mid_factor = RSRSAdvancedFactor(window=self.mid_window)
        
        short_data = short_factor.compute_all(df)
        mid_data = mid_factor.compute_all(df)
        
        return pd.DataFrame({
            'rsrs_short': short_data['rsrs_score'],
            'rsrs_mid': mid_data['rsrs_score'],
            'rsrs_short_r2': short_data['rsrs_r2'],
            'rsrs_mid_r2': mid_data['rsrs_r2'],
            'resonance': self.compute(df)
        }, index=df.index)