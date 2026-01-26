# ============================================================================
# 文件: utils/logger.py
# ============================================================================
"""
日志工具 - 统一日志配置
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import settings


def setup_logging(
    level: str = None,
    log_file: bool = None,
    log_dir: Path = None
) -> None:
    """
    配置全局日志
    
    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 是否输出到文件
        log_dir: 日志目录
    """
    level = level or settings.log.LEVEL
    log_file = log_file if log_file is not None else settings.log.FILE_ENABLED
    log_dir = log_dir or settings.path.LOG_DIR
    
    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 格式
    formatter = logging.Formatter(
        fmt=settings.log.FORMAT,
        datefmt=settings.log.DATE_FORMAT
    )
    
    # 控制台处理器
    if settings.log.CONSOLE_ENABLED:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        today = datetime.now().strftime('%Y-%m-%d')
        log_path = log_dir / f"quant_{today}.log"
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # 文件记录更详细
        root_logger.addHandler(file_handler)
    
    # 减少第三方库日志
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)


class LoggerMixin:
    """日志混入类"""
    
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger