#!/usr/bin/env python3
# ============================================================================
# 文件: examples/scan_modes_demo.py
# ============================================================================
"""
MarketScanner 多模式选股演示

演示如何使用不同的选股模式进行市场扫描
"""

from analysis import MarketScanner, scan_market
from config import settings

def demo_factor_mode():
    """演示因子融合模式（默认）"""
    print("=" * 80)
    print("模式 1: 因子融合模式 (factor)")
    print("=" * 80)
    
    scanner = MarketScanner()
    result = scanner.scan(mode='factor', top_n=20)
    
    print(f"\n选出 {len(result)} 只股票")
    if not result.empty:
        print(result.head())
    print()


def demo_rsrs_mode():
    """演示 RSRS 策略模式"""
    print("=" * 80)
    print("模式 2: RSRS 策略模式 (rsrs)")
    print("=" * 80)
    
    scanner = MarketScanner()
    result = scanner.scan(mode='rsrs', top_n=20)
    
    print(f"\n选出 {len(result)} 只股票")
    if not result.empty:
        print(result.head())
    print()


def demo_momentum_mode():
    """演示动量策略模式"""
    print("=" * 80)
    print("模式 3: 动量策略模式 (momentum)")
    print("=" * 80)
    
    scanner = MarketScanner()
    result = scanner.scan(mode='momentum', top_n=20)
    
    print(f"\n选出 {len(result)} 只股票")
    if not result.empty:
        print(result.head())
    print()


def demo_short_term_mode():
    """演示短线策略模式"""
    print("=" * 80)
    print("模式 4: 短线 RSRS 策略模式 (short_term)")
    print("=" * 80)
    
    scanner = MarketScanner()
    result = scanner.scan(mode='short_term', top_n=15)
    
    print(f"\n选出 {len(result)} 只股票")
    if not result.empty:
        print(result.head())
    print()


def demo_alpha_hunter_mode():
    """演示 Alpha Hunter 策略模式"""
    print("=" * 80)
    print("模式 5: Alpha Hunter 策略模式 (alpha_hunter)")
    print("=" * 80)
    
    scanner = MarketScanner()
    result = scanner.scan(mode='alpha_hunter', top_n=15)
    
    print(f"\n选出 {len(result)} 只股票")
    if not result.empty:
        print(result.head())
    print()


def demo_ensemble_mode():
    """演示多策略融合模式"""
    print("=" * 80)
    print("模式 6: 多策略融合模式 (ensemble)")
    print("=" * 80)
    
    scanner = MarketScanner()
    result = scanner.scan(mode='ensemble', top_n=30)
    
    print(f"\n选出 {len(result)} 只股票")
    if not result.empty:
        print(result.head())
    print()


def demo_custom_params():
    """演示自定义参数"""
    print("=" * 80)
    print("模式 7: 自定义参数的 RSRS 模式")
    print("=" * 80)
    
    scanner = MarketScanner()
    
    # 覆盖默认参数
    custom_params = {
        'entry_threshold': 0.9,     # 更严格的入场阈值
        'r2_threshold': 0.9,         # 更高的 R² 要求
        'use_volume_filter': False,  # 关闭量价过滤
    }
    
    result = scanner.scan(
        mode='rsrs',
        top_n=10,
        strategy_params=custom_params
    )
    
    print(f"\n选出 {len(result)} 只股票 (使用严格过滤)")
    if not result.empty:
        print(result.head())
    print()


def demo_ensemble_custom():
    """演示自定义融合权重"""
    print("=" * 80)
    print("模式 8: 自定义权重的多策略融合")
    print("=" * 80)
    
    scanner = MarketScanner()
    
    # 自定义融合参数
    ensemble_params = {
        'weights': {
            'rsrs': 0.4,           # RSRS 权重 40%
            'momentum': 0.1,       # 动量权重 10%
            'short_term': 0.4,     # 短线权重 40%
            'alpha_hunter': 0.1,   # AlphaHunter 权重 10%
        },
        'min_votes': 3,             # 至少 3 个策略选中
        'voting_mode': 'weighted',  # 加权模式
    }
    
    result = scanner.scan(
        mode='ensemble',
        top_n=20,
        strategy_params=ensemble_params
    )
    
    print(f"\n选出 {len(result)} 只股票 (自定义权重)")
    if not result.empty:
        print(result.head())
    print()


def demo_quick_scan():
    """演示快捷接口"""
    print("=" * 80)
    print("快捷接口演示")
    print("=" * 80)
    
    # 使用快捷接口
    result = scan_market(mode='rsrs', top_n=10)
    
    print(f"\n使用快捷接口选出 {len(result)} 只股票")
    if not result.empty:
        print(result.head())
    print()


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MarketScanner 多模式选股演示" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        # 1. 因子融合模式
        demo_factor_mode()
        
        # 2. RSRS 策略模式
        demo_rsrs_mode()
        
        # 3. 动量策略模式
        demo_momentum_mode()
        
        # 4. 短线策略模式
        demo_short_term_mode()
        
        # 5. Alpha Hunter 策略模式
        demo_alpha_hunter_mode()
        
        # 6. 多策略融合模式
        demo_ensemble_mode()
        
        # 7. 自定义参数
        demo_custom_params()
        
        # 8. 自定义融合权重
        demo_ensemble_custom()
        
        # 9. 快捷接口
        demo_quick_scan()
        
        print("=" * 80)
        print("演示完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
