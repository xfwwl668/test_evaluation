# ============================================================================
# 文件: config/settings.py
# ============================================================================
"""
全局配置中心 - 集中管理所有参数
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import os


@dataclass
class PathConfig:
    """路径配置"""
    ROOT_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    
    @property
    def DB_PATH(self) -> Path:
        return self.DATA_DIR / "stocks_daily.db"
    
    @property
    def CACHE_DIR(self) -> Path:
        return self.DATA_DIR / "cache"
    
    @property
    def LOG_DIR(self) -> Path:
        return self.DATA_DIR / "logs"
    
    def ensure_dirs(self):
        """确保目录存在"""
        for d in [self.DATA_DIR, self.CACHE_DIR, self.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class TDXConfig:
    """通达信配置"""
    NODES: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "深圳双线1", "host": "119.147.212.81", "port": 7709},
        {"name": "深圳双线2", "host": "112.91.112.219", "port": 7709},
        {"name": "上海双线1", "host": "101.227.73.20", "port": 7709},
        {"name": "上海双线2", "host": "101.227.77.254", "port": 7709},
        {"name": "北京双线", "host": "218.108.98.244", "port": 7709},
        {"name": "广州双线", "host": "113.105.142.162", "port": 7709},
        {"name": "杭州双线", "host": "121.14.110.194", "port": 7709},
        {"name": "武汉双线", "host": "59.173.18.69", "port": 7709},
        {"name": "南京双线", "host": "180.153.18.170", "port": 7709},
        {"name": "成都双线", "host": "61.135.142.73", "port": 7709},
        {"name": "重庆双线", "host": "124.161.145.45", "port": 7709},
        {"name": "天津双线", "host": "60.28.23.80", "port": 7709},
    ])
    
    CONNECT_TIMEOUT: float = 3.0
    HEARTBEAT: bool = True
    AUTO_RETRY: bool = True
    
    # 下载配置
    BARS_PER_REQUEST: int = 800
    TOTAL_BARS: int = 2500          # ~10年日线
    BATCH_SIZE: int = 50            # 防封批次
    SLEEP_RANGE: tuple = (1.0, 3.0) # 休眠区间


@dataclass
class FactorConfig:
    """因子配置"""
    # RSRS
    RSRS_WINDOW: int = 18
    RSRS_STD_WINDOW: int = 600
    RSRS_R2_THRESHOLD: float = 0.8
    
    # 动量
    MOMENTUM_WINDOW: int = 20
    
    # 波动率
    ATR_PERIOD: int = 14
    VOLATILITY_TARGET: float = 0.02
    
    # OBV
    OBV_MA_PERIOD: int = 20
    
    # VWAP
    VWAP_PERIOD: int = 20


@dataclass
class StrategyConfig:
    """策略配置"""
    # 信号阈值
    ENTRY_THRESHOLD: float = 0.7
    EXIT_THRESHOLD: float = -0.5
    
    # 量价条件
    VOL_WARM_LOW: float = 0.3
    VOL_WARM_HIGH: float = 0.75
    VWAP_BIAS_THRESHOLD: float = 0.02
    
    # 吊灯止损
    CHANDELIER_MULT: float = 3.0
    
    # 信号衰减
    SIGNAL_DECAY_PERIOD: int = 5
    COOLDOWN_AFTER_STOP: int = 3
    
    # 选股
    TOP_N_STOCKS: int = 30
    MIN_TRADING_DAYS: int = 250


@dataclass 
class BacktestConfig:
    """回测配置"""
    INITIAL_CAPITAL: float = 1_000_000.0
    
    # 交易成本
    COMMISSION_RATE: float = 0.0003      # 万三
    MIN_COMMISSION: float = 5.0           # 最低5元
    STAMP_DUTY: float = 0.001             # 印花税千一
    SLIPPAGE_RATE: float = 0.001          # 滑点千一
    
    # 仓位
    MAX_POSITION_WEIGHT: float = 0.1      # 单只最大10%
    CASH_RESERVE: float = 0.05            # 保留5%现金
    
    # 调仓
    REBALANCE_FREQ: str = "W"             # D/W/M


@dataclass
class LogConfig:
    """日志配置"""
    LEVEL: str = "INFO"
    FORMAT: str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    FILE_ENABLED: bool = True
    CONSOLE_ENABLED: bool = True


@dataclass
class Settings:
    """
    全局配置聚合
    
    使用方式:
        from config import settings
        
        db_path = settings.path.DB_PATH
        rsrs_window = settings.factor.RSRS_WINDOW
    """
    path: PathConfig = field(default_factory=PathConfig)
    tdx: TDXConfig = field(default_factory=TDXConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    log: LogConfig = field(default_factory=LogConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        self.path.ensure_dirs()
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """从环境变量加载配置"""
        settings = cls()
        
        # 示例: 从环境变量覆盖
        if os.getenv('QUANT_DATA_DIR'):
            settings.path.DATA_DIR = Path(os.getenv('QUANT_DATA_DIR'))
        
        if os.getenv('QUANT_INITIAL_CAPITAL'):
            settings.backtest.INITIAL_CAPITAL = float(os.getenv('QUANT_INITIAL_CAPITAL'))
        
        return settings


# 全局单例
settings = Settings()