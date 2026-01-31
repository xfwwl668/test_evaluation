# ============================================================================
# 文件: utils/nan_handler.py
# ============================================================================
"""
NaN处理框架 - 标准化数据质量检查和NaN填充

设计原则:
1. 前置验证: 在因子计算前检查数据质量
2. 安全填充: 使用合理的填充方法，避免0值污染
3. 追踪日志: 所有NaN操作都记录日志
4. 统计报告: 提供NaN率统计
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class FillMethod(Enum):
    """NaN填充方法"""
    FORWARD = "forward"      # 前向填充
    BACKWARD = "backward"    # 后向填充
    INTERPOLATE = "interpolate"  # 线性插值
    MEAN = "mean"            # 均值填充
    MEDIAN = "median"        # 中位数填充
    ZERO = "zero"            # 零填充(谨慎使用)


@dataclass
class NaNReport:
    """NaN处理报告"""
    column: str
    original_nan_count: int
    original_nan_ratio: float
    fill_method: str
    filled_count: int
    remaining_nan: int
    reason: str
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))


class NaNHandler:
    """
    标准化NaN处理工具类
    
    使用示例:
    ```python
    # 数据验证
    if NaNHandler.validate_ohlcv(df):
        factors = compute_factors(df)
    
    # 安全填充
    clean_series = NaNHandler.safe_fillna(series, method='forward', reason='RSRS计算')
    
    # 批量处理
    df_clean = NaNHandler.fill_dataframe(df, {
        'close': 'forward',
        'volume': 'zero'
    })
    ```
    """
    
    _reports: List[NaNReport] = []
    _logger = logging.getLogger("NaNHandler")
    
    @classmethod
    def validate_ohlcv(cls, df: pd.DataFrame, code: str = "") -> bool:
        """
        前置数据质量检查 - OHLCV数据验证
        
        Args:
            df: OHLCV数据
            code: 股票代码(用于日志)
        
        Returns:
            bool: 数据是否有效
        """
        prefix = f"[{code}] " if code else ""
        
        # 检查必要列
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            cls._logger.error(f"{prefix}缺少必要列: {missing_cols}")
            return False
        
        # 检查价格有效性 (必须为正)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                invalid = (df[col] <= 0).sum()
                if invalid > 0:
                    cls._logger.error(f"{prefix}{col}列有{invalid}个非正值")
                    return False
        
        # 检查价格逻辑 (high >= low)
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = (df['high'] < df['low']).sum()
            if invalid_hl > 0:
                cls._logger.error(f"{prefix}有{invalid_hl}条记录high < low")
                return False
        
        # 检查收盘价在高低点之间
        if all(c in df.columns for c in ['high', 'low', 'close']):
            out_of_range = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()
            if out_of_range > 0:
                cls._logger.warning(f"{prefix}有{out_of_range}条记录close不在[low,high]范围内")
        
        # 检查成交量非负
        if 'volume' in df.columns or 'vol' in df.columns:
            vol_col = 'volume' if 'volume' in df.columns else 'vol'
            invalid_vol = (df[vol_col] < 0).sum()
            if invalid_vol > 0:
                cls._logger.error(f"{prefix}{vol_col}列有{invalid_vol}个负值")
                return False
        
        # 检查NaN率
        nan_stats = df.isna().sum() / len(df)
        high_nan_cols = nan_stats[nan_stats > 0.05]
        if not high_nan_cols.empty:
            cls._logger.warning(f"{prefix}NaN率过高(>5%):\n{high_nan_cols}")
            # 不返回False，因为可能还可以处理
        
        critical_nan = nan_stats[nan_stats > 0.30]
        if not critical_nan.empty:
            cls._logger.error(f"{prefix}关键列NaN率过高(>30%): {list(critical_nan.index)}")
            return False
        
        return True
    
    @classmethod
    def safe_fillna(
        cls,
        series: pd.Series,
        method: str = 'forward',
        reason: str = '',
        code: str = ""
    ) -> pd.Series:
        """
        安全的NaN填充，带追踪日志
        
        Args:
            series: 输入序列
            method: 填充方法 ('forward', 'backward', 'interpolate', 'mean', 'median', 'zero')
            reason: 填充原因(用于日志)
            code: 股票代码
        
        Returns:
            填充后的序列
        """
        prefix = f"[{code}] " if code else ""
        
        # 检查是否需要填充
        nan_count = series.isna().sum()
        if nan_count == 0:
            return series
        
        nan_ratio = nan_count / len(series)
        
        # 记录日志
        cls._logger.warning(
            f"{prefix}NaN填充 [{reason}] 列={series.name}: "
            f"数量={nan_count}, 比例={nan_ratio:.2%}, 方法={method}"
        )
        
        # 执行填充
        result = series.copy()
        filled_count = nan_count
        
        if method == 'forward' or method == 'ffill':
            result = result.ffill().bfill()
        elif method == 'backward' or method == 'bfill':
            result = result.bfill().ffill()
        elif method == 'interpolate':
            result = result.interpolate(method='linear', limit_direction='both')
            # 边界可能还有NaN
            result = result.ffill().bfill()
        elif method == 'mean':
            mean_val = result.mean()
            result = result.fillna(mean_val)
        elif method == 'median':
            median_val = result.median()
            result = result.fillna(median_val)
        elif method == 'zero':
            result = result.fillna(0)
            cls._logger.warning(f"{prefix}使用0填充，请确认这是预期行为")
        else:
            raise ValueError(f"未知的填充方法: {method}")
        
        # 检查剩余NaN
        remaining = result.isna().sum()
        if remaining > 0:
            cls._logger.warning(f"{prefix}填充后仍有{remaining}个NaN，使用0填充")
            result = result.fillna(0)
        
        # 记录报告
        report = NaNReport(
            column=str(series.name),
            original_nan_count=nan_count,
            original_nan_ratio=nan_ratio,
            fill_method=method,
            filled_count=filled_count - remaining,
            remaining_nan=remaining,
            reason=reason
        )
        cls._reports.append(report)
        
        return result
    
    @classmethod
    def fill_dataframe(
        cls,
        df: pd.DataFrame,
        fill_config: Dict[str, str],
        reason: str = "",
        code: str = ""
    ) -> pd.DataFrame:
        """
        批量填充DataFrame
        
        Args:
            df: 输入DataFrame
            fill_config: {列名: 填充方法}
            reason: 填充原因
            code: 股票代码
        
        Returns:
            填充后的DataFrame
        """
        result = df.copy()
        
        for col, method in fill_config.items():
            if col in result.columns:
                result[col] = cls.safe_fillna(
                    result[col],
                    method=method,
                    reason=f"{reason}-{col}" if reason else col,
                    code=code
                )
        
        return result
    
    @classmethod
    def get_reports(cls) -> List[NaNReport]:
        """获取所有NaN处理报告"""
        return cls._reports.copy()
    
    @classmethod
    def clear_reports(cls):
        """清空报告缓存"""
        cls._reports.clear()
    
    @classmethod
    def get_summary(cls) -> Dict:
        """获取NaN处理摘要统计"""
        if not cls._reports:
            return {"total_operations": 0}
        
        total_filled = sum(r.filled_count for r in cls._reports)
        methods_used = {}
        for r in cls._reports:
            methods_used[r.fill_method] = methods_used.get(r.fill_method, 0) + 1
        
        return {
            "total_operations": len(cls._reports),
            "total_filled": total_filled,
            "methods_used": methods_used,
            "columns_affected": list(set(r.column for r in cls._reports)),
            "avg_nan_ratio": np.mean([r.original_nan_ratio for r in cls._reports])
        }
    
    @classmethod
    def print_summary(cls):
        """打印NaN处理摘要"""
        summary = cls.get_summary()
        print("\n" + "="*60)
        print("NaN处理统计报告")
        print("="*60)
        print(f"总操作次数: {summary['total_operations']}")
        print(f"总填充数: {summary['total_filled']}")
        print(f"使用方法: {summary['methods_used']}")
        print(f"影响列: {summary['columns_affected']}")
        print(f"平均NaN率: {summary.get('avg_nan_ratio', 0):.2%}")
        print("="*60)


def validate_factor_input(func):
    """
    装饰器: 在因子计算前进行数据验证
    
    使用:
    ```python
    @validate_factor_input
    def compute(self, df):
        ...
    ```
    """
    def wrapper(self, df: pd.DataFrame, *args, **kwargs):
        code = getattr(df, 'name', '')
        if not NaNHandler.validate_ohlcv(df, code):
            cls._logger.error(f"数据验证失败: {code}")
            return pd.Series(np.nan, index=df.index, name=self.name)
        return func(self, df, *args, **kwargs)
    return wrapper
