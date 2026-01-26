# ============================================================================
# 文件: examples/run_alpha_hunter.py
# ============================================================================
"""
Alpha-Hunter-V1 策略回测
"""
from engine import BacktestEngine
from strategy.alpha_hunter_strategy import AlphaHunterStrategy
from analysis import PerformanceAnalyzer, ReportGenerator

def main():
    print("=" * 70)
    print("🎯 Alpha-Hunter-V1 私募级超短线策略")
    print("=" * 70)
    
    engine = BacktestEngine(
        db_path='stocks_daily.db',
        initial_capital=1_000_000,
        commission_rate=0.0003,
        slippage_rate=0.001
    )
    
    strategy = AlphaHunterStrategy(params={
        'rsrs_threshold': 0.8,
        'r2_threshold': 0.85,
        'hard_stop_loss': 0.03,
        'max_holding_days': 2,
        'kelly_fraction': 0.5,
        'max_positions': 8,
    })
    
    engine.add_strategy(strategy)
    
    results = engine.run(
        start_date='2020-01-01',
        end_date='2023-12-31',
        rebalance_freq='D'
    )
    
    # 绩效统计
    result = results['alpha_hunter_v1']
    ReportGenerator.print_backtest_summary(result.metrics, "Alpha-Hunter-V1")
    
    # Kelly 统计
    perf = strategy.get_performance_summary()
    print(f"\n📊 交易统计:")
    print(f"   总交易: {perf['trades']}")
    print(f"   胜率: {perf['win_rate']:.1%}")
    print(f"   平均盈利: {perf['avg_win']:.1%}")
    print(f"   平均亏损: {perf['avg_loss']:.1%}")

if __name__ == "__main__":
    main()