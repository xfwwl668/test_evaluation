# ============================================================================
# 文件: utils/monitoring.py
# ============================================================================
"""
监控告警机制
"""
import logging
from typing import Dict, Any

class Monitor:
    """
    监控系统
    """
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Monitor.{name}")
        
    def alert(self, message: str, level: str = "WARNING"):
        """
        发送告警 (目前仅记录日志，可扩展为钉钉/邮件)
        """
        log_msg = f"🔔 [ALERT] [{self.name}] {message}"
        if level == "INFO":
            self.logger.info(log_msg)
        elif level == "WARNING":
            self.logger.warning(log_msg)
        elif level == "ERROR":
            self.logger.error(log_msg)
        elif level == "CRITICAL":
            self.logger.critical(log_msg)

    def log_trade(self, order_info: Dict[str, Any]):
        """
        记录重要交易
        """
        self.logger.info(f"📈 [TRADE] {order_info.get('code')} {order_info.get('side')} "
                         f"qty={order_info.get('quantity')} @ {order_info.get('price')}")

    def log_performance(self, metrics: Dict[str, Any]):
        """
        记录绩效指标
        """
        self.logger.info(f"📊 [PERF] Equity: {metrics.get('equity')} | Drawdown: {metrics.get('drawdown'):.2%}")
