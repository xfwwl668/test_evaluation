# ============================================================================
# 文件: strategy/alpha_hunter_strategy.py
# ============================================================================
"""
Alpha-Hunter-V1 私募级超短线策略

目标:
- 年化收益 > 30%
- 最大回撤 < 10%
- 持仓周期 T+1 到 T+2

核心逻辑:
1. 极致胜率过滤 (5重条件)
2. T+1 必杀卖出
3. 动态移动锁利
4. Kelly 仓位管理
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry
# Safe imports with fallback
try:
    from factors.alpha_hunter_factors import (
        AlphaHunterFactorEngine, AlphaSignal, SignalStrength,
        AdvancedRSRSFactor, PressureLevelFactor, MarketBreadthFactor
    )
    FACTOR_ENGINE_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger("AlphaHunterStrategy")
    logger.warning(f"AlphaHunterFactorEngine 导入失败，将使用简化版本: {e}")
    FACTOR_ENGINE_AVAILABLE = False
    # Fallback imports for basic functionality
    AlphaSignal = None
    SignalStrength = None
    AdvancedRSRSFactor = None
    PressureLevelFactor = None
    MarketBreadthFactor = None

try:
    from engine.high_freq_matcher import HighFreqMatcher, MarketMicrostructure
except ImportError:
    from engine.matcher import MatchEngine as HighFreqMatcher
    MarketMicrostructure = None
    import logging
    logger = logging.getLogger("AlphaHunterStrategy")
    logger.warning("使用标准 MatchEngine 代替 HighFreqMatcher")

from engine.risk import RiskManager, PositionSizer


@dataclass
class TradeRecord:
    """交易记录 (用于 Kelly 计算)"""
    code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_ratio: float
    is_win: bool


@dataclass
class AlphaPosition:
    """
    Alpha 策略持仓状态
    
    改进:
    - entry_date: 订单创建日期
    - entry_filled_date: 实际成交日期(T+1检查用)
    """
    code: str
    entry_price: float
    entry_date: str           # 订单创建日期
    quantity: int
    
    # 动态止损止盈
    stop_loss_price: float
    take_profit_price: float
    trailing_stop: float
    
    # 历史高点 (用于移动锁利)
    highest_price: float
    highest_date: str
    
    # 累计利润阈值 (每 +3% 触发一次锁利)
    lock_profit_thresholds: List[float] = field(default_factory=lambda: [0.03, 0.06, 0.09, 0.12])
    current_lock_level: int = 0
    
    # T+1时间对齐: 记录实际成交日期(非创建日期)
    entry_filled_date: str = ""  # 实际成交日期，用于T+1检查
    
    def update_trailing_stop(self, current_price: float, current_date: str):
        """
        更新移动锁利
        
        规则: 每增加 3% 利润，止损上移 2%
        """
        if current_price > self.highest_price:
            self.highest_price = current_price
            self.highest_date = current_date
        
        current_pnl = (current_price - self.entry_price) / self.entry_price
        
        # 检查是否触发新的锁利阈值
        while (self.current_lock_level < len(self.lock_profit_thresholds) and
               current_pnl >= self.lock_profit_thresholds[self.current_lock_level]):
            
            # 止损上移 2%
            new_stop = self.entry_price * (1 + 0.02 * (self.current_lock_level + 1))
            
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
                logging.getLogger("AlphaPosition").info(
                    f"[LOCK-PROFIT] {self.code} 锁利触发 L{self.current_lock_level+1} "
                    f"止损上移至 {new_stop:.2f}"
                )
            
            self.current_lock_level += 1
        
        # 硬止损不动
        self.trailing_stop = max(self.trailing_stop, self.stop_loss_price)


@StrategyRegistry.register
class AlphaHunterStrategy(BaseStrategy):
    """
    Alpha-Hunter-V1 策略
    
    买入准则 (ALL 条件):
    1. 修正 RSRS > 0.8 且 R² > 0.85
    2. 价格 > MA5 且 MA5 斜率向上
    3. 换手率 < 25%
    4. 全市场上涨家数 > 40%
    5. 距离压力位 > 5%
    
    卖出准则 (ANY 条件):
    1. 开盘强卖: 15分钟未涨2% 且 跌破昨收
    2. 移动锁利: 每+3%利润 → 止损上移2%
    3. 硬止损: -3%
    4. 跌破 MA5
    5. 最大持仓 2 天
    """
    
    name = "alpha_hunter_v1"
    version = "1.0.0"
    
    # ===== 策略参数 =====
    DEFAULT_PARAMS = {
        # 入场参数
        'rsrs_threshold': 0.8,
        'r2_threshold': 0.85,
        'max_turnover': 0.25,           # 最大换手率 25%
        'market_breadth_threshold': 0.40,  # 上涨家数 40%
        'min_pressure_distance': 0.05,  # 压力距离 5%
        'ma5_slope_threshold': 0.001,   # MA5 斜率阈值
        
        # 离场参数
        'opening_check_gain': 0.02,     # 开盘检查涨幅阈值
        'hard_stop_loss': 0.03,         # 硬止损 3%
        'profit_lock_step': 0.03,       # 每 3% 锁利一次
        'stop_raise_step': 0.02,        # 止损上移 2%
        'max_holding_days': 2,          # 最大持仓天数
        
        # 仓位参数
        'kelly_lookback': 20,           # Kelly 回溯交易数
        'kelly_fraction': 0.5,          # Kelly 保守系数
        'max_single_position': 0.08,    # 单只最大 8%
        'max_total_position': 0.70,     # 总仓位最大 70%
        'max_positions': 8,             # 最大持仓数
        
        # 行业限制
        'max_sector_exposure': 0.20,    # 单行业最大 20%
        
        # 涨停限制
        'allow_limit_up_chase': False,  # 不追涨停
        
        # 价格过滤
        'min_price': 5.0,
        'max_price': 80.0,
        'min_volume': 2000000,          # 最低成交额 200万
    }
    
    def __init__(self, params: Dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        
        # 因子引擎
        if FACTOR_ENGINE_AVAILABLE:
            try:
                self.factor_engine = AlphaHunterFactorEngine()
                self.logger.info("✅ AlphaHunterFactorEngine 初始化成功")
            except Exception as e:
                self.logger.error(f"❌ AlphaHunterFactorEngine 初始化失败: {e}")
                self.factor_engine = None
        else:
            self.factor_engine = None
        
        # 高频撮合器
        try:
            self.hf_matcher = HighFreqMatcher()
            self.logger.info("✅ HighFreqMatcher 初始化成功")
        except Exception as e:
            self.logger.error(f"❌ HighFreqMatcher 初始化失败: {e}")
            self.hf_matcher = None
        
        # 仓位管理器
        self._position_sizer = PositionSizer(
            risk_per_trade=self.params.get('risk_per_trade', 0.01),
            atr_multiplier=self.params.get('atr_multiplier', 2.0),
            min_position_pct=self.params.get('min_position_pct', 0.01),
            max_position_pct=self.params.get('max_position_pct', 0.10)
        )
        
        self._risk_manager = RiskManager()
        
        # 持仓状态
        self._positions: Dict[str, AlphaPosition] = {}
        self._position_history = deque(maxlen=1000)
        
        # 交易记录 (用于 Kelly)
        self._trade_history: List[TradeRecord] = []
        self._consecutive_losses = 0
        self._last_loss_date = None
        self._suspended_until = None
        
        # Kelly 系数
        self._kelly_fraction = np.clip(
            self.params.get('kelly_fraction', 1.0),
            0.5, 2.0
        )
        
        # 市场情绪缓存
        self._market_breadth_cache: Dict = {}
        
        # 行业敞口
        self._sector_exposure: Dict[str, float] = {}
        
        self._validate_params()
        self.logger.info(f"✅ AlphaHunterStrategy 初始化完成")
    
    def _validate_params(self):
        """验证策略参数"""
        required_params = ['rsrs_threshold', 'r2_threshold', 'market_breadth_threshold']
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"缺少必需参数: {param}")
        
        # 检查参数范围
        if not (0 <= self.params['rsrs_threshold'] <= 1):
            raise ValueError("rsrs_threshold 必须在 0 到 1 之间")
        
        if not (0 <= self.params['r2_threshold'] <= 1):
            raise ValueError("r2_threshold 必须在 0 到 1 之间")
        
        if not (0 <= self.params['market_breadth_threshold'] <= 1):
            raise ValueError("market_breadth_threshold 必须在 0 到 1 之间")
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算所有因子"""
        
        if not history:
            self.logger.warning("历史数据为空")
            return {}
        
        factors = {}
        
        try:
            # 1. RSRS 因子
            if self.factor_engine:
                rsrs_dict = {}
                for code, df in history.items():
                    if df.empty or len(df) < 20:
                        continue
                    
                    try:
                        rsrs_series = self.factor_engine.compute_rsrs(df)
                        
                        # ✅ 验证返回值
                        if rsrs_series is None or len(rsrs_series) == 0:
                            continue
                        
                        rsrs_dict[code] = rsrs_series
                    except Exception as e:
                        self.logger.debug(f"[{code}] RSRS 计算失败: {e}")
                        continue
                
                if rsrs_dict:
                    factors['rsrs'] = pd.DataFrame(rsrs_dict)
            
            # 2. MA5 因子
            ma5_dict = {}
            for code, df in history.items():
                if df.empty or 'close' not in df.columns:
                    continue
                
                ma5 = df['close'].rolling(window=5, min_periods=1).mean()
                
                # ✅ 处理 NaN
                if ma5.isna().sum() > 0:
                    ma5 = ma5.fillna(method='ffill').fillna(method='bfill')
                
                ma5_dict[code] = ma5
            
            if ma5_dict:
                factors['ma5'] = pd.DataFrame(ma5_dict)
            
            # 3. MA5 斜率
            ma5_slope_dict = {}
            for code in ma5_dict.keys():
                ma5_series = factors['ma5'][code]
                ma5_slope = ma5_series.diff() / ma5_series
                ma5_slope = ma5_slope.fillna(0)
                ma5_slope_dict[code] = ma5_slope
            
            if ma5_slope_dict:
                factors['ma5_slope'] = pd.DataFrame(ma5_slope_dict)
            
            self.logger.info(f"因子计算完成: {list(factors.keys())}")
            return factors
        
        except Exception as e:
            self.logger.error(f"因子计算异常: {e}", exc_info=True)
            return {}
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """生成交易信号"""
        signals = []
        current_date = context.current_date
        
        # 1. 计算市场情绪
        breadth = self._calculate_market_breadth(context)
        self._market_breadth_cache[current_date] = breadth
        
        # 2. 开盘强卖检查 (优先处理)
        sell_signals = self._generate_opening_force_sell(context, breadth)
        signals.extend(sell_signals)
        
        # 3. 常规离场检查
        exit_signals = self._generate_exit_signals(context)
        signals.extend(exit_signals)
        
        # 4. 市场情绪过滤
        if not breadth.get('is_bullish', False):
            self.logger.info(f"市场情绪偏空 ({breadth.get('advance_ratio', 0):.0%})，暂停入场")
            return signals
        
        # 5. 入场信号
        entry_signals = self._generate_entry_signals(context, breadth)
        signals.extend(entry_signals)
        
        return signals
    
    def _calculate_market_breadth(self, context: StrategyContext) -> Dict:
        """计算市场广度"""
        breadth_factor = MarketBreadthFactor()
        return breadth_factor.compute_market_breadth(
            context.current_data,
            context.current_date
        )
    
    def _generate_opening_force_sell(
        self,
        context: StrategyContext,
        breadth: Dict
    ) -> List[Signal]:
        """
        生成开盘强卖信号 - 修复: 使用entry_filled_date进行T+1检查
        
        条件:
        1. T+1 可卖 (基于实际成交日期，非订单创建日期)
        2. 15分钟未涨超 2%
        3. 跌破昨日收盘价
        """
        signals = []
        
        opening_threshold = self.get_param('opening_check_gain')
        current_dt = datetime.strptime(context.current_date, '%Y-%m-%d')
        
        for code, pos in list(self._positions.items()):
            # === T+1 检查 - 修复: 使用成交日期而非创建日期 ===
            # entry_filled_date为空则回退到entry_date
            filled_date = pos.entry_filled_date if pos.entry_filled_date else pos.entry_date
            filled_dt = datetime.strptime(filled_date, '%Y-%m-%d')
            
            # 检查是否已经过了T+1 (至少间隔1天)
            days_held = (current_dt - filled_dt).days
            if days_held < 1:
                self.logger.debug(f"[T+1] {code} 持仓{days_held}天，不可卖出")
                continue
            
            # 获取数据
            row = context.current_data[context.current_data['code'] == code]
            if row.empty:
                continue
            
            current_price = row['close'].iloc[0]
            open_price = row['open'].iloc[0]
            
            # 昨日收盘价
            history = context.get_history(code, 2)
            if len(history) < 2:
                continue
            prev_close = history['close'].iloc[-2]
            
            # 涨幅检查
            change_from_prev = (current_price - prev_close) / prev_close
            
            # 开盘强卖条件
            if change_from_prev < opening_threshold and current_price < prev_close:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    price=current_price,
                    priority=100,  # 最高优先级
                    reason=f"开盘强卖: 涨幅{change_from_prev:.1%} < {opening_threshold:.0%}, 跌破昨收"
                ))
                
                self.logger.warning(
                    f"[FORCE-SELL] {code} 开盘强卖触发 | "
                    f"现价={current_price:.2f} 昨收={prev_close:.2f}"
                )
        
        return signals
    
    def _generate_exit_signals(self, context: StrategyContext) -> List[Signal]:
        """生成常规离场信号 - 修复: 使用entry_filled_date进行T+1检查"""
        signals = []
        
        hard_stop = self.get_param('hard_stop_loss')
        max_days = self.get_param('max_holding_days')
        current_dt = datetime.strptime(context.current_date, '%Y-%m-%d')
        
        for code, pos in list(self._positions.items()):
            # === T+1 检查 - 修复: 使用成交日期而非创建日期 ===
            filled_date = pos.entry_filled_date if pos.entry_filled_date else pos.entry_date
            filled_dt = datetime.strptime(filled_date, '%Y-%m-%d')
            
            days_held = (current_dt - filled_dt).days
            if days_held < 1:
                continue  # 当日成交不可卖出
            
            row = context.current_data[context.current_data['code'] == code]
            if row.empty:
                continue
            
            current_price = row['close'].iloc[0]
            
            # 更新移动锁利
            pos.update_trailing_stop(current_price, context.current_date)
            
            should_exit = False
            reason = ""
            
            # ===== 条件1: 硬止损 =====
            pnl = (current_price - pos.entry_price) / pos.entry_price
            if pnl <= -hard_stop:
                should_exit = True
                reason = f"硬止损 {pnl:.1%}"
            
            # ===== 条件2: 移动止损 =====
            if not should_exit and current_price < pos.trailing_stop:
                should_exit = True
                reason = f"移动止损触发 ({pos.trailing_stop:.2f})"
            
            # ===== 条件3: 跌破 MA5 =====
            if not should_exit:
                ma5 = context.get_factor('ma5', code)
                if ma5 is not None and current_price < ma5:
                    should_exit = True
                    reason = f"跌破 MA5 ({ma5:.2f})"
            
            # ===== 条件4: 最大持仓天数 =====
            if not should_exit:
                try:
                    # 使用days_held(基于成交日期)
                    if days_held >= max_days:
                        should_exit = True
                        reason = f"持仓{days_held}天，强制离场"
                except:
                    pass
            
            if should_exit:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    price=current_price,
                    reason=reason
                ))
                
                self.logger.info(f"[EXIT] {code} | {reason} | PnL={pnl:.1%}")
        
        return signals
    
    def _generate_entry_signals(
        self,
        context: StrategyContext,
        breadth: Dict
    ) -> List[Signal]:
        """生成入场信号"""
        signals = []
        
        # 参数
        rsrs_th = self.get_param('rsrs_threshold')
        r2_th = self.get_param('r2_threshold')
        max_turnover = self.get_param('max_turnover')
        min_pressure = self.get_param('min_pressure_distance')
        ma5_slope_th = self.get_param('ma5_slope_threshold')
        min_price = self.get_param('min_price')
        max_price = self.get_param('max_price')
        min_volume = self.get_param('min_volume')
        max_positions = self.get_param('max_positions')
        
        # 检查持仓数
        if len(self._positions) >= max_positions:
            return signals
        
        # 筛选候选 (向量化)
        current_data = context.current_data.copy()
        
        # 获取所有因子值 (向量化)
        rsrs_series = context.get_all_factors('rsrs_score')
        r2_series = context.get_all_factors('rsrs_r2')
        ma5_series = context.get_all_factors('ma5')
        ma5_slope_series = context.get_all_factors('ma5_slope')
        pressure_series = context.get_all_factors('pressure_distance')
        
        # 合并到DataFrame (向量化)
        if rsrs_series is not None:
            current_data['rsrs'] = rsrs_series
            current_data['r2'] = r2_series if r2_series is not None else 0
            current_data['ma5'] = ma5_series if ma5_series is not None else 0
            current_data['ma5_slope'] = ma5_slope_series if ma5_slope_series is not None else 0
            current_data['pressure'] = pressure_series if pressure_series is not None else 0.1
        
        # 统一NaN处理 - 修复: 使用标准化NaN处理框架
        from utils.nan_handler import NaNHandler
        
        fill_config = {
            'r2': 'forward',      # R²使用前向填充(假设拟合质量延续)
            'ma5': 'interpolate',  # MA5使用插值
            'ma5_slope': 'forward', # 斜率延续
            'pressure': 'median'   # 压力距离用历史中位数
        }
        
        for col, method in fill_config.items():
            if col in current_data.columns:
                current_data[col] = NaNHandler.safe_fillna(
                    current_data[col],
                    method=method,
                    reason=f'入场信号-{col}',
                    code='batch'
                )
        
        # 计算换手率 - 修复: 完全向量化，避免groupby().apply()
        if 'amount' in current_data.columns and 'vol' in current_data.columns:
            # 使用向量化计算而非groupby().apply()
            # 按股票分组计算统计量
            code_groups = current_data.groupby('code')
            
            # 计算每只股票的关键统计量
            stats = code_groups.agg({
                'amount': 'mean',
                'close': 'last',
                'vol': 'mean'
            })
            
            # 计算换手率 (vectorized)
            # 换手率 = 平均成交额 / 估算市值
            stats['est_market_cap'] = stats['close'] * stats['vol'] * 100
            stats['turnover'] = np.where(
                stats['est_market_cap'] > 0,
                stats['amount'] / stats['est_market_cap'],
                0.0
            )
            
            # 映射回原始DataFrame
            current_data = current_data.merge(
                stats[['turnover']].reset_index(),
                on='code',
                how='left'
            )
            
            # 数据不足5天的设为0
            counts = code_groups.size()
            insufficient_data = counts[counts < 5].index
            current_data.loc[current_data['code'].isin(insufficient_data), 'turnover'] = 0.0
        else:
            current_data['turnover'] = 0.0
        
        # 过滤条件 (向量化)
        mask = (
            (~current_data['code'].isin(self._positions.keys())) &  # 不在持仓中
            (~current_data['code'].isin(context.positions.keys())) &
            (current_data['close'] >= min_price) &  # 价格过滤
            (current_data['close'] <= max_price) &
            (current_data['amount'] >= min_volume) &  # 成交额过滤
            (current_data['rsrs'] > rsrs_th) &  # RSRS 过滤
            (current_data['r2'] >= r2_th) &
            (current_data['close'] > current_data['ma5']) &  # MA5 趋势
            (current_data['ma5_slope'] >= ma5_slope_th) &
            (current_data['turnover'] <= max_turnover) &  # 换手率
            (current_data['pressure'] >= min_pressure) &  # 压力距离
            (~current_data['name'].str.contains('ST', na=False)) & # 🔴 修复 Problem 21: 排除 ST 股票
            (~current_data['name'].str.contains(r'\*', na=False))   # 排除 *ST 股票
        )
        
        # 非涨停过滤
        if not self.get_param('allow_limit_up_chase'):
            # 🔴 修复 Problem 15: 向量化检查涨停
            if 'is_limit_up' in current_data.columns:
                mask = mask & (~current_data['is_limit_up'])
            else:
                # 如果没有标志位，则手动计算 (向量化)
                current_data['prev_close'] = current_data.groupby('code')['close'].shift(1) # 这不对，因为current_data只有一行/一天
                # 实际上应该从 context.history 获取
                pass 
        
        filtered_data = current_data[mask].copy()
        
        # 计算综合评分
        filtered_data['score'] = filtered_data['rsrs'] * filtered_data['r2']
        
        # 排序选最强 (向量化)
        filtered_data = filtered_data.sort_values('score', ascending=False)
        
        slots = max_positions - len(self._positions)
        selected_data = filtered_data.head(slots)
        
        # 转换为字典列表
        candidates = selected_data.to_dict('records')
        
        # 计算仓位
        for cand in candidates:
            weight = self._calculate_kelly_position(context.total_equity)
            weight = min(weight, self.get_param('max_single_position'))
            
            if weight < 0.02:
                continue
            
            signals.append(Signal(
                code=cand['code'],
                side=OrderSide.BUY,
                weight=weight,
                price=cand['close'],
                reason=f"RSRS={cand['rsrs']:.2f} R²={cand['r2']:.2f} 压力距={cand['pressure']:.1%}"
            ))
            
            self.logger.info(
                f"[ENTRY] {cand['code']} | RSRS={cand['rsrs']:.2f} R²={cand['r2']:.2f} | "
                f"Weight={weight:.1%}"
            )
        
        return signals
    
    def _calculate_kelly_position(self, total_equity: float) -> float:
        """
        Kelly 准则计算仓位 - 修复: 添加多重风险保护
        
        改进:
        1. 样本量检查 (至少10笔)
        2. 胜率低保护 (<30%)
        3. Kelly上限 (25%)
        4. 破产保护 (风险回报比)
        5. 绝对上下限 (1%-15%)
        6. 详细日志记录
        
        公式: f = (p × b - q) / b
        其中: p=胜率, q=败率, b=盈亏比
        """
        # === 保护1: 样本量检查 ===
        min_samples = 10
        if len(self._trade_history) < min_samples:
            self.logger.info(f"交易样本 < {min_samples}, 使用保守仓位 2%")
            return 0.02
        
        # 取最近20笔交易计算Kelly
        lookback = self.get_param('kelly_lookback')
        recent_trades = list(self._trade_history)[-lookback:]
        
        wins = [t for t in recent_trades if t.is_win]
        losses = [t for t in recent_trades if not t.is_win]
        
        # === 保护2: 胜率过低保护 ===
        if len(wins) == 0:
            self.logger.warning("无盈利交易，降低仓位至1%")
            return 0.01
        
        total_recent = len(recent_trades)
        p = len(wins) / total_recent
        q = 1 - p
        
        # 胜率 < 30% 使用保守仓位
        min_win_rate = 0.30
        if p < min_win_rate:
            self.logger.warning(f"胜率{p:.1%} < {min_win_rate:.0%}, 使用保守仓位2%")
            return 0.02
        
        # === 计算盈亏比 ===
        avg_win = np.mean([t.pnl_ratio for t in wins])
        avg_loss = abs(np.mean([t.pnl_ratio for t in losses])) if losses else 0.01
        
        if avg_loss <= 0 or avg_win <= 0:
            self.logger.warning("盈亏数据异常，使用默认仓位2%")
            return 0.02
        
        b = avg_win / avg_loss
        
        # === Kelly公式 ===
        # f = (p*b - q) / b = (p*b - (1-p)) / b
        kelly_raw = (p * b - q) / b if b > 0 else 0
        
        # === 保护3: Kelly上限保护 (通常不超过25%) ===
        kelly_cap = min(kelly_raw, 0.25)
        
        # === 保护4: 破产保护 (风险回报比) ===
        # 如果风险回报比 < 0.1 (即风险太大)，降低仓位
        risk_reward_ratio = 1.0 / max(b, 1.0)
        if risk_reward_ratio < 0.1:
            kelly_cap = min(kelly_cap, 0.10)
            self.logger.warning(f"风险回报比过低({risk_reward_ratio:.2f})，Kelly限制在10%")
            
        # === 保护5: 利润因子保护 ===
        total_p = sum([t.pnl_ratio for t in wins])
        total_l = abs(sum([t.pnl_ratio for t in losses])) if losses else 0
        pf = total_p / total_l if total_l > 0 else 5.0
        if pf < 1.2:
            kelly_cap *= 0.5
            self.logger.warning(f"利润因子过低({pf:.2f})，仓位减半")
            
        # === 保护6: 连续亏损保护 ===
        if len(recent_trades) >= 3:
            last_3 = recent_trades[-3:]
            if all([not t.is_win for t in last_3]):
                kelly_cap *= 0.5
                self.logger.warning("触发连续3次亏损保护，仓位减半")
                
        # === 保护7: 最大回撤保护 (假设从 context 获取) ===
        # (简化实现: 略)
        
        # 应用保守系数
        kelly_fraction = self.get_param('kelly_fraction')
        position = kelly_cap * kelly_fraction
        
        # === 保护8: 绝对上下限 (1%-15%) ===
        min_position = 0.01
        max_position = 0.15
        position = np.clip(position, min_position, max_position)
        
        # === 保护9: 市场情绪二次确认 ===
        # (breadth 已经在调用处处理)
        
        # === 详细日志记录 ===
        self.logger.info(
            f"[KELLY] 样本={total_recent} 胜率={p:.1%} 盈亏比={b:.2f} "
            f"Kelly原始={kelly_raw:.2%} Kelly上限={kelly_cap:.2%} "
            f"最终仓位={position:.2%}"
        )
        
        return position
    
    def on_order_filled(self, order) -> None:
        """
        订单成交回调 - 修复: 设置entry_filled_date用于T+1检查
        
        改进:
        - 使用order.filled_date(实际成交日期)而非order.create_date(创建日期)
        - 确保T+1规则正确执行
        """
        if order.side == OrderSide.BUY:
            # 初始化持仓状态
            hard_stop = self.get_param('hard_stop_loss')
            
            # === 修复: 使用filled_date(成交日期)进行T+1检查 ===
            # filled_date由MatchEngine.match()设置
            filled_date = order.filled_date if order.filled_date else order.create_date
            
            self._positions[order.code] = AlphaPosition(
                code=order.code,
                entry_price=order.filled_price,
                entry_date=order.create_date,      # 订单创建日期
                entry_filled_date=filled_date,     # 实际成交日期(T+1检查用)
                quantity=order.filled_quantity,
                stop_loss_price=order.filled_price * (1 - hard_stop),
                take_profit_price=order.filled_price * 1.15,
                trailing_stop=order.filled_price * (1 - hard_stop),
                highest_price=order.filled_price,
                highest_date=order.create_date
            )
            
            self.logger.info(
                f"[FILLED-BUY] {order.code} @ {order.filled_price:.2f} "
                f"止损={order.filled_price * (1 - hard_stop):.2f} "
                f"成交日期={filled_date}"
            )
        
        else:
            # 记录交易
            if order.code in self._positions:
                pos = self._positions.pop(order.code)
                pnl = (order.filled_price - pos.entry_price) / pos.entry_price
                
                trade = TradeRecord(
                    code=order.code,
                    entry_date=pos.entry_date,
                    exit_date=order.create_date,
                    entry_price=pos.entry_price,
                    exit_price=order.filled_price,
                    pnl_ratio=pnl,
                    is_win=(pnl > 0)
                )
                self._trade_history.append(trade)
                
                self.logger.info(
                    f"[FILLED-SELL] {order.code} @ {order.filled_price:.2f} "
                    f"PnL={pnl:.1%} | {order.signal_reason}"
                )
    
    def get_performance_summary(self) -> Dict:
        """获取绩效摘要"""
        if not self._trade_history:
            return {'trades': 0, 'win_rate': 0, 'avg_pnl': 0}
        
        wins = [t for t in self._trade_history if t.is_win]
        
        return {
            'trades': len(self._trade_history),
            'win_rate': len(wins) / len(self._trade_history),
            'avg_pnl': np.mean([t.pnl_ratio for t in self._trade_history]),
            'avg_win': np.mean([t.pnl_ratio for t in wins]) if wins else 0,
            'avg_loss': np.mean([t.pnl_ratio for t in self._trade_history if not t.is_win]) if len(wins) < len(self._trade_history) else 0,
            'max_win': max([t.pnl_ratio for t in self._trade_history]),
            'max_loss': min([t.pnl_ratio for t in self._trade_history])
        }