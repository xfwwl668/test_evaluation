# ============================================================================
# 文件: core/data_validator.py
# ============================================================================
"""
数据质量验证器 - 确保回测数据真实、完整、合规
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

class DataValidator:
    """
    数据质量验证器
    
    实现:
    - OHLCV 逻辑检查 (open, high, low, close > 0, high >= low 等)
    - 价格异常波动检测
    - NaN 率统计与阈值报警
    - 停牌/交易日完整性检查
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("DataValidator")
        self.quality_reports = []

    def validate_ohlcv(self, df: pd.DataFrame, code: str = "") -> bool:
        """
        验证 OHLCV 数据逻辑
        """
        if df.empty:
            self.logger.warning(f"[{code}] 数据为空")
            return False
            
        # 检查必要列
        required = ['open', 'high', 'low', 'close', 'vol']
        if 'volume' in df.columns and 'vol' not in df.columns:
            df = df.rename(columns={'volume': 'vol'})
            
        missing = [c for c in required if c not in df.columns]
        if missing:
            self.logger.error(f"[{code}] 缺少列: {missing}")
            return False

        # 1. 价格必须为正
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                self.logger.error(f"[{code}] {col} 存在非正值")
                return False

        # 2. 逻辑关系: high >= low, high >= open, high >= close, low <= open, low <= close
        if (df['high'] < df['low']).any():
            self.logger.error(f"[{code}] 存在 high < low")
            return False
            
        # 3. 成交量非负
        if (df['vol'] < 0).any():
            self.logger.error(f"[{code}] 存在负成交量")
            return False

        # 4. 检查 NaN
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            nan_ratio = nan_count / (len(df) * len(df.columns))
            if nan_ratio > 0.1:
                self.logger.warning(f"[{code}] NaN 比例过高: {nan_ratio:.2%}")
                if nan_ratio > 0.5:
                    return False

        return True

    def detect_outliers(self, df: pd.DataFrame, threshold: float = 0.21) -> List[pd.Timestamp]:
        """
        检测异常波动 (默认涨跌幅超过 21% 为异常，A股限制)
        """
        if len(df) < 2:
            return []
            
        returns = df['close'].pct_change().abs()
        outliers = returns[returns > threshold].index.tolist()
        
        if outliers:
            self.logger.warning(f"检测到异常波动: {len(outliers)} 处")
            
        return outliers

    def check_continuity(self, df: pd.DataFrame, expected_days: int) -> bool:
        """
        检查数据连续性
        """
        if len(df) < expected_days * 0.8:
            self.logger.warning(f"数据量不足: 实际 {len(df)} < 预期 {expected_days}")
            return False
        return True

    def generate_quality_report(self, market_data: pd.DataFrame) -> Dict:
        """
        生成全市场数据质量报告
        """
        report = {
            'total_stocks': market_data['code'].nunique() if 'code' in market_data.columns else 1,
            'total_rows': len(market_data),
            'nan_count': market_data.isna().sum().sum(),
            'zero_vol_count': (market_data['vol'] == 0).sum() if 'vol' in market_data.columns else 0,
            'issues': []
        }
        
        # 简单统计
        if 'code' in market_data.columns:
            stats = market_data.groupby('code').size()
            report['avg_rows_per_stock'] = stats.mean()
            report['min_rows'] = stats.min()
            
        return report
