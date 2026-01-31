# ============================================================================
# 文件: factors/alpha_hunter_factors.py
# ============================================================================
"""
Alpha-Hunter-V1 综合因子模块

核心因子:
1. 进阶 RSRS (Z-Score × R² × 偏度修正)
2. 开盘异动因子 (早盘放量捕捉)
3. 压力位因子 (筹码密集区 + 历史高点)
4. 市场情绪因子 (涨跌家数比)

改进:
- 集成NaNHandler进行数据验证
- 安全的NaN处理
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from factors.base import BaseFactor, FactorMeta, FactorRegistry
from config import settings


class SignalStrength(Enum):
    """信号强度"""
    NONE = 0
    WEAK = 1
    MEDIUM = 2
    STRONG = 3
    EXTREME = 4


@dataclass
class AlphaSignal:
    """Alpha 信号"""
    code: str
    strength: SignalStrength
    rsrs_score: float
    pressure_distance: float
    momentum_score: float
    composite_score: float
    risk_level: float
    reason: str


@FactorRegistry.register
class AdvancedRSRSFactor(BaseFactor):
    """
    进阶 RSRS 因子 - 私募级别
    
    核心改进:
    1. R² 拟合优度加权
    2. 右偏分布修正 (剔除虚假暴涨)
    3. 斜率加速度 (二阶导)
    4. 异常值检测与处理
    
    公式:
        Score = Z_score × R² × (1 - skew_penalty) × validity_mask
    """
    
    meta = FactorMeta(
        name="rsrs_alpha",
        category="technical",
        description="私募级进阶 RSRS",
        lookback=650
    )
    
    def __init__(
        self,
        window: int = 18,
        std_window: int = 600,
        r2_threshold: float = 0.85,
        skew_penalty_factor: float = 0.15,
        outlier_std: float = 3.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window = window
        self.std_window = std_window
        self.r2_threshold = r2_threshold
        self.skew_penalty_factor = skew_penalty_factor
        self.outlier_std = outlier_std
        self._logger = logging.getLogger("AdvancedRSRSFactor")
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算最终修正 RSRS 得分"""
        result = self.compute_full(df)
        return result['rsrs_final_score']
    
    def compute_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算完整 RSRS 数据 - 修复: 添加数据验证和NaN处理
        
        Returns:
            DataFrame with: rsrs_slope, rsrs_r2, rsrs_zscore, 
                           rsrs_skew_adj, rsrs_final_score, rsrs_valid
        """
        n = len(df)
        
        if n < self.window + 60:
            return self._empty_result(df.index)
        
        # 数据验证
        from utils.nan_handler import NaNHandler
        code = getattr(df, 'name', '')
        if not NaNHandler.validate_ohlcv(df, code):
            self._logger.warning(f"[{code}] RSRS数据验证失败")
            return self._empty_result(df.index)
        
        # 安全的NaN处理
        high = NaNHandler.safe_fillna(
            df['high'], method='interpolate', reason='RSRS-high', code=code
        ).to_numpy(dtype=np.float64)
        
        low = NaNHandler.safe_fillna(
            df['low'], method='interpolate', reason='RSRS-low', code=code
        ).to_numpy(dtype=np.float64)
        
        # ===== Step 1: 滚动 OLS 回归 =====
        slope, r2, residual = self._vectorized_ols(high, low)
        
        # 填充
        pad = np.full(self.window - 1, np.nan)
        slope_full = np.concatenate([pad, slope])
        r2_full = np.concatenate([pad, r2])
        
        # ===== Step 2: 异常值处理 =====
        slope_cleaned = self._handle_outliers(slope_full)
        
        # ===== Step 3: 滚动标准化 =====
        slope_s = pd.Series(slope_cleaned, index=df.index)
        roll_mean = slope_s.rolling(self.std_window, min_periods=60).mean()
        roll_std = slope_s.rolling(self.std_window, min_periods=60).std()
        
        zscore = (slope_s - roll_mean) / roll_std.clip(lower=1e-10)
        
        # ===== Step 4: R² 加权 =====
        score_r2 = zscore.values * r2_full
        
        # ===== Step 5: 右偏修正 =====
        # 计算滚动偏度
        skewness = slope_s.rolling(self.std_window, min_periods=60).apply(
            lambda x: stats.skew(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 10 else 0,
            raw=True
        )
        
        # 右偏惩罚 (只惩罚正分数)
        skew_penalty = np.clip(skewness.values * self.skew_penalty_factor, 0, 0.5)
        
        score_skew_adj = np.where(
            score_r2 > 0,
            score_r2 * (1 - skew_penalty),
            score_r2
        )
        
        # ===== Step 6: 有效性过滤 =====
        valid = (r2_full >= self.r2_threshold).astype(np.int8)
        
        # 无效信号大幅衰减
        final_score = np.where(valid, score_skew_adj, score_skew_adj * 0.3)
        
        # ===== Step 7: 斜率动量 (加速度) =====
        # 🔴 修复 Problem 12: 避免 np.nan_to_num(0) 带来的虚假动量
        slope_momentum = np.full_like(slope_full, np.nan)
        valid_mask = ~np.isnan(slope_full)
        if valid_mask.sum() > 2:
            # 只在有效数据上计算梯度
            slope_momentum[valid_mask] = np.gradient(slope_full[valid_mask])
        
        return pd.DataFrame({
            'rsrs_slope': slope_full,
            'rsrs_r2': r2_full,
            'rsrs_zscore': zscore.values,
            'rsrs_skew_adj': score_skew_adj,
            'rsrs_final_score': final_score,
            'rsrs_valid': valid,
            'rsrs_momentum': slope_momentum
        }, index=df.index)
    
    def _vectorized_ols(self, high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """向量化 OLS"""
        low_win = sliding_window_view(low, self.window)
        high_win = sliding_window_view(high, self.window)
        
        x_mean = low_win.mean(axis=1, keepdims=True)
        y_mean = high_win.mean(axis=1, keepdims=True)
        
        x_dev = low_win - x_mean
        y_dev = high_win - y_mean
        
        cov_xy = (x_dev * y_dev).sum(axis=1)
        var_x = (x_dev ** 2).sum(axis=1)
        var_y = (y_dev ** 2).sum(axis=1)
        
        slope = np.divide(cov_xy, var_x, out=np.zeros_like(cov_xy), where=var_x > 1e-10)
        
        denom = var_x * var_y
        r2 = np.divide(cov_xy ** 2, denom, out=np.zeros_like(cov_xy), where=denom > 1e-10)
        
        # 残差
        intercept = y_mean.flatten() - slope * x_mean.flatten()
        y_pred = slope.reshape(-1, 1) * low_win + intercept.reshape(-1, 1)
        residual = (high_win - y_pred).std(axis=1)
        
        return slope, r2, residual
    
    def _handle_outliers(self, arr: np.ndarray) -> np.ndarray:
        """异常值处理 (Winsorize)"""
        result = arr.copy()
        valid = ~np.isnan(arr)
        
        if valid.sum() < 10:
            return result
        
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        
        upper = mean + self.outlier_std * std
        lower = mean - self.outlier_std * std
        
        result[valid & (arr > upper)] = upper
        result[valid & (arr < lower)] = lower
        
        return result
    
    def _empty_result(self, index) -> pd.DataFrame:
        """空结果"""
        return pd.DataFrame({
            'rsrs_slope': np.nan,
            'rsrs_r2': np.nan,
            'rsrs_zscore': np.nan,
            'rsrs_skew_adj': np.nan,
            'rsrs_final_score': np.nan,
            'rsrs_valid': 0,
            'rsrs_momentum': np.nan
        }, index=index)


@FactorRegistry.register
class OpeningMomentumFactor(BaseFactor):
    """
    开盘异动因子
    
    捕捉:
    1. 早盘 15 分钟成交量占比 (需分钟数据)
    2. 开盘跳空幅度
    3. 首笔成交异常
    
    注: 日线级别用竞价数据模拟
    """
    
    meta = FactorMeta(
        name="opening_momentum",
        category="technical",
        description="开盘异动因子",
        lookback=10
    )
    
    def __init__(
        self,
        volume_ratio_threshold: float = 0.10,  # 15分钟量占比阈值
        gap_threshold: float = 0.02,            # 跳空阈值 2%
        **kwargs
    ):
        super().__init__(**kwargs)
        self.volume_ratio_threshold = volume_ratio_threshold
        self.gap_threshold = gap_threshold
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算开盘异动得分"""
        open_price = df['open'].to_numpy()
        close = df['close'].to_numpy()
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        volume = df['vol'].to_numpy().astype(np.float64)
        
        n = len(df)
        
        # 昨收
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        prev_volume = np.roll(volume, 1)
        prev_volume[0] = volume[0]
        
        # ===== 1. 跳空幅度 =====
        gap = (open_price - prev_close) / prev_close
        gap_score = np.clip(gap / self.gap_threshold, -2, 2)
        
        # ===== 2. 开盘位置 (在日内区间的位置) =====
        intraday_range = high - low
        open_position = np.where(
            intraday_range > 1e-10,
            (open_price - low) / intraday_range,
            0.5
        )
        # 开盘在高位更强势
        open_pos_score = (open_position - 0.5) * 2
        
        # ===== 3. 成交量异动 (模拟早盘放量) =====
        # 假设开盘量 ≈ 全天量 × (high/close 越高越集中)
        vol_concentration = (high / np.clip(close, 1e-10, None))
        vol_ratio_estimate = np.clip(vol_concentration - 1, 0, 0.5)
        
        vol_score = np.where(
            vol_ratio_estimate > self.volume_ratio_threshold,
            vol_ratio_estimate / self.volume_ratio_threshold,
            vol_ratio_estimate / self.volume_ratio_threshold * 0.5
        )
        
        # ===== 4. 综合开盘异动得分 =====
        # 权重: 跳空 40%, 开盘位置 30%, 量能 30%
        momentum_score = (
            0.4 * gap_score +
            0.3 * open_pos_score +
            0.3 * vol_score
        )
        
        return pd.Series(momentum_score, index=df.index, name=self.name)
    
    def compute_with_minute_data(
        self,
        daily_df: pd.DataFrame,
        minute_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """
        使用分钟数据计算 (更精确)
        
        Args:
            daily_df: 日线数据
            minute_data: {date: minute_df} 分钟数据
        """
        scores = []
        
        for date in daily_df.index:
            date_str = str(date)[:10]
            
            if date_str not in minute_data:
                scores.append(np.nan)
                continue
            
            min_df = minute_data[date_str]
            
            if len(min_df) < 15:
                scores.append(np.nan)
                continue
            
            # 前 15 分钟数据
            first_15 = min_df.head(15)
            
            # 成交量占比
            vol_15 = first_15['vol'].sum()
            vol_total = min_df['vol'].sum()
            vol_ratio = vol_15 / vol_total if vol_total > 0 else 0
            
            # 价格走势
            price_change_15 = (first_15['close'].iloc[-1] - first_15['open'].iloc[0]) / first_15['open'].iloc[0]
            
            # 综合得分
            score = vol_ratio * 5 + price_change_15 * 10
            scores.append(score)
        
        return pd.Series(scores, index=daily_df.index, name=self.name)


@FactorRegistry.register
class PressureLevelFactor(BaseFactor):
    """
    压力位因子
    
    计算:
    1. 20日最高价压力
    2. 成交量密集区压力
    3. 整数关口压力
    4. 历史套牢盘压力
    """
    
    meta = FactorMeta(
        name="pressure_level",
        category="technical",
        description="压力位因子",
        lookback=250
    )
    
    def __init__(
        self,
        high_window: int = 20,
        volume_profile_bins: int = 50,
        safe_distance: float = 0.05,  # 5% 安全距离
        **kwargs
    ):
        super().__init__(**kwargs)
        self.high_window = high_window
        self.volume_profile_bins = volume_profile_bins
        self.safe_distance = safe_distance
        self._logger = logging.getLogger("PressureLevelFactor")
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算距离压力位的百分比"""
        result = self.compute_full(df)
        return result['distance_to_pressure']
    
    def compute_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算完整压力位数据 - 修复: 添加数据验证和NaN处理
        
        Returns:
            DataFrame with: pressure_20d, pressure_volume, 
                           pressure_round, distance_to_pressure, safe_score
        """
        # 数据验证
        from utils.nan_handler import NaNHandler
        code = getattr(df, 'name', '')
        if not NaNHandler.validate_ohlcv(df, code):
            self._logger.warning(f"[{code}] 压力位数据验证失败")
            return self._empty_result(df.index)
        
        # 安全的NaN处理
        close = NaNHandler.safe_fillna(
            df['close'], method='interpolate', reason='Pressure-close', code=code
        ).to_numpy()
        high = NaNHandler.safe_fillna(
            df['high'], method='interpolate', reason='Pressure-high', code=code
        ).to_numpy()
        low = NaNHandler.safe_fillna(
            df['low'], method='interpolate', reason='Pressure-low', code=code
        ).to_numpy()
        
        volume_col = 'vol' if 'vol' in df.columns else 'volume'
        volume = NaNHandler.safe_fillna(
            df[volume_col], method='zero', reason='Pressure-volume', code=code
        ).to_numpy().astype(np.float64)
        
        n = len(close)
        
        # ===== 1. 20日最高价压力 =====
        high_s = pd.Series(high, index=df.index)
        pressure_20d = high_s.rolling(self.high_window, min_periods=5).max().to_numpy()
        
        # ===== 2. 成交量密集区压力 =====
        pressure_volume = self._calc_volume_pressure(close, high, low, volume)
        
        # ===== 3. 整数关口压力 =====
        pressure_round = self._calc_round_pressure(close)
        
        # ===== 4. 综合压力位 (取最近的) =====
        combined_pressure = np.minimum.reduce([
            np.where(pressure_20d > close, pressure_20d, np.inf),
            np.where(pressure_volume > close, pressure_volume, np.inf),
            np.where(pressure_round > close, pressure_round, np.inf)
        ])
        
        # 无穷大替换为 20日高点
        combined_pressure = np.where(
            np.isinf(combined_pressure),
            pressure_20d,
            combined_pressure
        )
        
        # ===== 5. 距离压力位百分比 =====
        distance = (combined_pressure - close) / close
        distance = np.clip(distance, -0.5, 0.5)
        
        # ===== 6. 安全评分 (距离越远越安全) =====
        safe_score = np.where(
            distance >= self.safe_distance,
            1.0,
            distance / self.safe_distance
        )
        
        return pd.DataFrame({
            'pressure_20d': pressure_20d,
            'pressure_volume': pressure_volume,
            'pressure_round': pressure_round,
            'combined_pressure': combined_pressure,
            'distance_to_pressure': distance,
            'safe_score': safe_score
        }, index=df.index)
    
    def _calc_volume_pressure(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """计算成交量密集区压力"""
        n = len(close)
        pressure = np.full(n, np.nan)
        
        lookback = min(250, n)
        
        for i in range(lookback, n):
            # 回溯窗口
            window_close = close[i-lookback:i]
            window_vol = volume[i-lookback:i]
            window_high = high[i-lookback:i]
            
            current = close[i]
            
            # 成交量分布
            price_bins = np.linspace(
                window_close.min() * 0.9,
                window_high.max() * 1.1,
                self.volume_profile_bins
            )
            
            vol_profile = np.zeros(len(price_bins) - 1)
            
            for j in range(len(window_close)):
                idx = np.searchsorted(price_bins, window_close[j]) - 1
                idx = max(0, min(idx, len(vol_profile) - 1))
                vol_profile[idx] += window_vol[j]
            
            # 找到当前价上方的密集区
            bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
            above_mask = bin_centers > current
            
            if above_mask.any():
                above_vol = vol_profile[above_mask]
                above_prices = bin_centers[above_mask]
                
                # 成交量最大的上方区域
                if len(above_vol) > 0:
                    max_idx = np.argmax(above_vol)
                    pressure[i] = above_prices[max_idx]
        
        # 填充 NaN
        pressure = pd.Series(pressure).fillna(method='bfill').fillna(close.max() * 1.1).to_numpy()
        
        return pressure
    
    def _calc_round_pressure(self, close: np.ndarray) -> np.ndarray:
        """计算整数关口压力"""
        # 找到上方最近的整数/半整数关口
        round_levels = []
        
        for price in close:
            if price < 10:
                step = 0.5
            elif price < 50:
                step = 1.0
            elif price < 100:
                step = 5.0
            else:
                step = 10.0
            
            # 上方最近的整数位
            next_round = np.ceil(price / step) * step
            if next_round == price:
                next_round += step
            
            round_levels.append(next_round)
        
        return np.array(round_levels)


@FactorRegistry.register
class MarketBreadthFactor(BaseFactor):
    """
    市场广度因子
    
    计算:
    1. 涨跌家数比
    2. 涨停/跌停家数
    3. 成交额集中度
    4. 板块轮动强度
    """
    
    meta = FactorMeta(
        name="market_breadth",
        category="market",
        description="市场情绪广度因子",
        lookback=5
    )
    
    def __init__(
        self,
        advance_threshold: float = 0.40,  # 上涨家数阈值 40%
        **kwargs
    ):
        super().__init__(**kwargs)
        self.advance_threshold = advance_threshold
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """实现抽象方法 - 返回空序列(此因子用于市场广度计算，非单股票)"""
        return pd.Series(np.nan, index=df.index, name=self.name)
    
    def compute_market_breadth(
        self,
        market_data: pd.DataFrame,
        current_date: str
    ) -> Dict:
        """
        计算市场广度指标
        
        Args:
            market_data: 全市场当日数据
            current_date: 当前日期
        
        Returns:
            {'advance_ratio': float, 'is_bullish': bool, 'score': float, ...}
        """
        if market_data.empty:
            return self._empty_breadth()
        
        # 涨跌判断
        changes = market_data['close'] / market_data['open'] - 1
        
        advancing = (changes > 0.001).sum()
        declining = (changes < -0.001).sum()
        unchanged = len(market_data) - advancing - declining
        
        total = len(market_data)
        
        # 涨跌比
        advance_ratio = advancing / total if total > 0 else 0.5
        decline_ratio = declining / total if total > 0 else 0.5
        
        # 涨停/跌停
        limit_up = (changes >= 0.095).sum()  # 近似涨停
        limit_down = (changes <= -0.095).sum()
        
        # 市场情绪评分
        breadth_score = (advance_ratio - 0.5) * 2  # [-1, 1]
        
        # 是否适合做多
        is_bullish = advance_ratio >= self.advance_threshold
        
        return {
            'advancing': int(advancing),
            'declining': int(declining),
            'unchanged': int(unchanged),
            'advance_ratio': round(advance_ratio, 4),
            'decline_ratio': round(decline_ratio, 4),
            'limit_up': int(limit_up),
            'limit_down': int(limit_down),
            'breadth_score': round(breadth_score, 4),
            'is_bullish': is_bullish
        }
    
    def _empty_breadth(self) -> Dict:
        return {
            'advancing': 0, 'declining': 0, 'unchanged': 0,
            'advance_ratio': 0.5, 'decline_ratio': 0.5,
            'limit_up': 0, 'limit_down': 0,
            'breadth_score': 0, 'is_bullish': False
        }


class AlphaHunterFactorEngine:
    """
    Alpha-Hunter 综合因子引擎
    
    整合所有因子计算，生成最终 Alpha 信号
    """
    
    def __init__(self):
        self.rsrs_factor = AdvancedRSRSFactor()
        self.opening_factor = OpeningMomentumFactor()
        self.pressure_factor = PressureLevelFactor()
        self.breadth_factor = MarketBreadthFactor()
    
    def compute_alpha_signal(
        self,
        df: pd.DataFrame,
        market_data: pd.DataFrame = None,
        current_date: str = None
    ) -> AlphaSignal:
        """
        计算综合 Alpha 信号
        
        Args:
            df: 单只股票历史数据
            market_data: 全市场当日数据 (用于市场情绪)
            current_date: 当前日期
        
        Returns:
            AlphaSignal 对象
        """
        if len(df) < 100:
            return self._empty_signal(df.name if hasattr(df, 'name') else 'unknown')
        
        code = df.name if hasattr(df, 'name') else 'unknown'
        
        # 1. RSRS
        rsrs_data = self.rsrs_factor.compute_full(df)
        rsrs_score = rsrs_data['rsrs_final_score'].iloc[-1]
        rsrs_r2 = rsrs_data['rsrs_r2'].iloc[-1]
        rsrs_valid = rsrs_data['rsrs_valid'].iloc[-1]
        
        # 2. 压力位
        pressure_data = self.pressure_factor.compute_full(df)
        pressure_distance = pressure_data['distance_to_pressure'].iloc[-1]
        pressure_safe = pressure_data['safe_score'].iloc[-1]
        
        # 3. 开盘动量
        momentum = self.opening_factor.compute(df).iloc[-1]
        
        # 4. 市场广度
        breadth = {'is_bullish': True, 'breadth_score': 0}
        if market_data is not None:
            breadth = self.breadth_factor.compute_market_breadth(market_data, current_date)
        
        # 5. 综合评分
        composite = self._calculate_composite_score(
            rsrs_score, rsrs_r2, rsrs_valid,
            pressure_distance, pressure_safe,
            momentum, breadth
        )
        
        # 6. 信号强度判定
        strength = self._determine_signal_strength(composite, rsrs_valid, breadth['is_bullish'])
        
        # 7. 风险评估
        risk_level = self._assess_risk_level(pressure_distance, rsrs_r2)
        
        return AlphaSignal(
            code=code,
            strength=strength,
            rsrs_score=float(rsrs_score) if not np.isnan(rsrs_score) else 0,
            pressure_distance=float(pressure_distance) if not np.isnan(pressure_distance) else 0,
            momentum_score=float(momentum) if not np.isnan(momentum) else 0,
            composite_score=composite,
            risk_level=risk_level,
            reason=self._generate_reason(rsrs_score, pressure_distance, momentum)
        )
    
    def _calculate_composite_score(
        self,
        rsrs_score: float,
        rsrs_r2: float,
        rsrs_valid: int,
        pressure_distance: float,
        pressure_safe: float,
        momentum: float,
        breadth: Dict
    ) -> float:
        """计算综合评分"""
        # 处理 NaN
        rsrs = rsrs_score if not np.isnan(rsrs_score) else 0
        pressure = pressure_distance if not np.isnan(pressure_distance) else 0
        mom = momentum if not np.isnan(momentum) else 0
        safe = pressure_safe if not np.isnan(pressure_safe) else 0
        breadth_score = breadth.get('breadth_score', 0)
        
        # 加权计算
        # RSRS 40%, 压力距离 20%, 动量 20%, 市场情绪 20%
        composite = (
            0.40 * np.clip(rsrs, -2, 2) +
            0.20 * np.clip(pressure * 10, -2, 2) +
            0.20 * np.clip(mom, -2, 2) +
            0.20 * np.clip(breadth_score, -1, 1)
        )
        
        # R² 有效性惩罚
        if rsrs_valid == 0:
            composite *= 0.5
        
        # 安全距离加成
        composite *= (0.8 + 0.2 * safe)
        
        return round(float(composite), 4)
    
    def _determine_signal_strength(
        self,
        composite: float,
        rsrs_valid: int,
        is_bullish: bool
    ) -> SignalStrength:
        """判定信号强度"""
        if not is_bullish:
            return SignalStrength.NONE
        
        if rsrs_valid == 0:
            return SignalStrength.NONE
        
        if composite >= 1.5:
            return SignalStrength.EXTREME
        elif composite >= 1.0:
            return SignalStrength.STRONG
        elif composite >= 0.6:
            return SignalStrength.MEDIUM
        elif composite >= 0.3:
            return SignalStrength.WEAK
        else:
            return SignalStrength.NONE
    
    def _assess_risk_level(self, pressure_distance: float, rsrs_r2: float) -> float:
        """评估风险水平 (0-1, 越高越危险)"""
        pressure = pressure_distance if not np.isnan(pressure_distance) else 0.05
        r2 = rsrs_r2 if not np.isnan(rsrs_r2) else 0.5
        
        # 距离压力位近 + R² 低 = 高风险
        risk = (1 - np.clip(pressure * 10, 0, 1)) * 0.5 + (1 - r2) * 0.5
        
        return round(float(np.clip(risk, 0, 1)), 4)
    
    def _generate_reason(self, rsrs: float, pressure: float, momentum: float) -> str:
        """生成信号原因"""
        parts = []
        
        if not np.isnan(rsrs):
            parts.append(f"RSRS={rsrs:.2f}")
        if not np.isnan(pressure):
            parts.append(f"压力距离={pressure:.1%}")
        if not np.isnan(momentum):
            parts.append(f"动量={momentum:.2f}")
        
        return " | ".join(parts)
    
    def _empty_signal(self, code: str) -> AlphaSignal:
        return AlphaSignal(
            code=code,
            strength=SignalStrength.NONE,
            rsrs_score=0,
            pressure_distance=0,
            momentum_score=0,
            composite_score=0,
            risk_level=1.0,
            reason="数据不足"
        )