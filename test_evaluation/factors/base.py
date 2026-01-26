# ============================================================================
# 文件: factors/base.py
# ============================================================================
"""
因子基类 - 所有因子的抽象接口
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Type
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class FactorMeta:
    """因子元信息"""
    name: str                           # 因子名称
    category: str                       # 类别: technical/fundamental/composite
    description: str = ""               # 描述
    lookback: int = 0                   # 所需历史数据长度
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他因子


class BaseFactor(ABC):
    """
    因子抽象基类
    
    所有因子必须继承此类并实现:
    - meta: 因子元信息
    - compute(): 计算因子值
    
    使用示例:
    ```python
    class MyFactor(BaseFactor):
        meta = FactorMeta(
            name="my_factor",
            category="technical",
            lookback=20
        )
        
        def compute(self, df: pd.DataFrame) -> pd.Series:
            return df['close'].pct_change(20)
    ```
    """
    
    meta: FactorMeta  # 子类必须定义
    
    def __init__(self, **params):
        """
        初始化因子
        
        Args:
            **params: 因子参数，覆盖默认值
        """
        self.params = params
        self._cache: Dict[str, pd.Series] = {}
    
    @property
    def name(self) -> str:
        return self.meta.name
    
    @property
    def lookback(self) -> int:
        return self.meta.lookback
    
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算因子值 (核心方法)
        
        Args:
            df: OHLCV 数据，columns=['open','high','low','close','vol','amount']
        
        Returns:
            pd.Series: 因子值序列，index 与 df 一致
        """
        pass
    
    def compute_batch(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        批量计算多只股票的因子
        
        Args:
            data: {code: DataFrame} 字典
        
        Returns:
            DataFrame: index=date, columns=code
        """
        results = {}
        
        for code, df in data.items():
            if len(df) >= self.lookback:
                try:
                    results[code] = self.compute(df)
                except Exception:
                    continue
        
        if not results:
            return pd.DataFrame()
        
        # 合并为宽表
        combined = pd.DataFrame(results)
        return combined
    
    def get_latest(self, df: pd.DataFrame) -> float:
        """获取最新因子值"""
        series = self.compute(df)
        return series.iloc[-1] if len(series) > 0 else np.nan
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, params={self.params})"


class FactorRegistry:
    """
    因子注册表 - 管理所有可用因子
    
    使用方式:
    ```python
    # 注册
    @FactorRegistry.register
    class RSRSFactor(BaseFactor):
        ...
    
    # 获取
    rsrs = FactorRegistry.get("rsrs")
    
    # 列表
    all_factors = FactorRegistry.list_all()
    ```
    """
    
    _registry: Dict[str, Type[BaseFactor]] = {}
    
    @classmethod
    def register(cls, factor_class: Type[BaseFactor]) -> Type[BaseFactor]:
        """注册因子 (装饰器)"""
        name = factor_class.meta.name
        cls._registry[name] = factor_class
        return factor_class
    
    @classmethod
    def get(cls, name: str, **params) -> BaseFactor:
        """获取因子实例"""
        if name not in cls._registry:
            raise KeyError(f"Factor '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**params)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有因子名称"""
        return list(cls._registry.keys())
    
    @classmethod
    def get_meta(cls, name: str) -> FactorMeta:
        """获取因子元信息"""
        return cls._registry[name].meta
    
    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """按类别列出因子"""
        return [
            name for name, factor_cls in cls._registry.items()
            if factor_cls.meta.category == category
        ]


class FactorPipeline:
    """
    因子计算流水线 - 批量计算多个因子
    
    使用方式:
    ```python
    pipeline = FactorPipeline()
    pipeline.add("rsrs")
    pipeline.add("momentum", window=20)
    
    results = pipeline.run(stock_data)
    # results = {
    #     "rsrs": DataFrame,
    #     "momentum": DataFrame
    # }
    ```
    """
    
    def __init__(self):
        self.factors: List[BaseFactor] = []
    
    def add(self, factor_name: str, **params) -> 'FactorPipeline':
        """添加因子"""
        factor = FactorRegistry.get(factor_name, **params)
        self.factors.append(factor)
        return self
    
    def add_instance(self, factor: BaseFactor) -> 'FactorPipeline':
        """直接添加因子实例"""
        self.factors.append(factor)
        return self
    
    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        运行流水线
        
        Args:
            data: {code: DataFrame}
        
        Returns:
            {factor_name: DataFrame}
        """
        results = {}
        
        for factor in self.factors:
            results[factor.name] = factor.compute_batch(data)
        
        return results
    
    def run_single(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """单股票计算所有因子"""
        return {
            factor.name: factor.compute(df)
            for factor in self.factors
        }