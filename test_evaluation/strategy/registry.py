# ============================================================================
# 文件: strategy/registry.py
# ============================================================================
"""
策略注册表 - 插件化管理
"""
from typing import Dict, Type, List, Optional
from .base import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    策略注册表
    
    功能:
    - 策略自动发现和注册
    - 按名称获取策略类
    - 列出所有可用策略
    
    使用方式:
    ```python
    # 注册 (装饰器方式)
    @StrategyRegistry.register
    class MyStrategy(BaseStrategy):
        name = "my_strategy"
        ...
    
    # 或手动注册
    StrategyRegistry.register_class(MyStrategy)
    
    # 获取策略
    strategy_cls = StrategyRegistry.get("my_strategy")
    strategy = strategy_cls(params={...})
    
    # 列表
    all_strategies = StrategyRegistry.list_all()
    ```
    """
    
    _registry: Dict[str, Type[BaseStrategy]] = {}
    
    @classmethod
    def register(cls, strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
        """
        注册策略 (装饰器)
        
        使用:
            @StrategyRegistry.register
            class MyStrategy(BaseStrategy):
                name = "my_strategy"
        """
        name = strategy_class.name
        
        if name in cls._registry:
            logger.warning(f"Strategy '{name}' already registered, overwriting...")
        
        cls._registry[name] = strategy_class
        logger.info(f"Registered strategy: {name}")
        
        return strategy_class
    
    @classmethod
    def register_class(cls, strategy_class: Type[BaseStrategy]) -> None:
        """手动注册策略类"""
        cls.register(strategy_class)
    
    @classmethod
    def get(cls, name: str) -> Type[BaseStrategy]:
        """
        获取策略类
        
        Args:
            name: 策略名称
        
        Returns:
            策略类 (未实例化)
        
        Raises:
            KeyError: 策略不存在
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"Strategy '{name}' not found. Available: {available}")
        
        return cls._registry[name]
    
    @classmethod
    def create(cls, name: str, params: Dict = None) -> BaseStrategy:
        """
        创建策略实例
        
        Args:
            name: 策略名称
            params: 策略参数
        
        Returns:
            策略实例
        """
        strategy_cls = cls.get(name)
        return strategy_cls(params=params)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册策略"""
        return list(cls._registry.keys())
    
    @classmethod
    def get_info(cls, name: str) -> Dict:
        """获取策略信息"""
        strategy_cls = cls.get(name)
        return {
            "name": strategy_cls.name,
            "version": getattr(strategy_cls, 'version', '1.0.0'),
            "docstring": strategy_cls.__doc__,
            "class": strategy_cls.__name__
        }
    
    @classmethod
    def clear(cls) -> None:
        """清空注册表 (测试用)"""
        cls._registry.clear()


def auto_discover_strategies(package_path: str = "strategy") -> None:
    """
    自动发现并导入策略模块
    
    扫描指定目录下所有 *_strategy.py 文件
    """
    import importlib
    import pkgutil
    from pathlib import Path
    
    package = importlib.import_module(package_path)
    package_dir = Path(package.__file__).parent
    
    for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        if module_name.endswith('_strategy'):
            full_module = f"{package_path}.{module_name}"
            try:
                importlib.import_module(full_module)
                logger.info(f"Auto-discovered strategy module: {full_module}")
            except Exception as e:
                logger.error(f"Failed to import {full_module}: {e}")