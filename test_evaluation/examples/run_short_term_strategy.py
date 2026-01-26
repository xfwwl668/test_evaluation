# ============================================================================
# 文件: examples/run_short_term_strategy.py
# ============================================================================
"""
短线策略回测示例
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import BacktestEngine
from strategy.short_term_strategy import ShortTermRSRSStrategy
from analysis import PerformanceAnalyzer, ReportGenerator
from utils.logger import setup_logging


def main():
    """运行短线策略回测"""
    
    # 配置日志
    setup_logging(level='INFO')
    
    print("=" * 70)
    print("🚀 高胜率短线 RSRS 策略回测")
    print("=" * 70)
    
    # 创建引擎
    engine = BacktestEngine(
        db_path='stocks_daily.db',
        initial_capital=1_000_000,
        commission_rate=0.0003,
        slippage_rate=0.001
    )
    
    # 创建策略 (可调参数)
    strategy = ShortTermRSRSStrategy(params={
        # 入场
        'rsrs_entry_threshold': 0.7,
        'r2_threshold': 0.8,
        'volume_multiplier': 1.5,
        
        # 离场
        'fixed_stop_loss': 0.03,
        'trailing_atr_mult': 2.0,
        'max_holding_days': 5,
        
        # 仓位
        'risk_per_trade': 0.005,
        'max_single_weight': 0.10,
        'max_total_weight': 0.80,
        'max_positions': 10,
    })
    
    engine.add_strategy(strategy)
    
    # 运行回测
    results = engine.run(
        start_date='2020-01-01',
        end_date='2023-12-31',
        rebalance_freq='D'  # 日度检查
    )
    
    # 分析结果
    result = results['short_term_rsrs']
    
    # 打印报告
    ReportGenerator.print_backtest_summary(result.metrics, "短线RSRS策略")
    
    # 导出数据
    equity = result.get_equity_curve()
    trades = result.get_trades()
    
    print(f"\n📈 权益曲线 (最近10天):")
    print(equity.tail(10))
    
    print(f"\n📋 交易记录 (最近10笔):")
    print(trades.tail(10) if not trades.empty else "无交易")
    
    # 胜率统计
    if not trades.empty:
        buy_trades = trades[trades['side'] == 'BUY']
        sell_trades = trades[trades['side'] == 'SELL']
        
        print(f"\n📊 交易统计:")
        print(f"   买入次数: {len(buy_trades)}")
        print(f"   卖出次数: {len(sell_trades)}")
    
    return results


if __name__ == "__main__":
    main()