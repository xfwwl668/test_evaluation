# ============================================================================
# 文件: examples/interactive_demo.py
# ============================================================================
"""
交互式菜单系统使用示例
展示如何使用新的交互式功能
"""

from interactive_main import InteractiveEngine
from quick_launcher import QuickLauncher
from session_manager import SessionManager


def demo_basic_usage():
    """基础使用演示"""
    print("=" * 60)
    print("🚀 交互式菜单系统 - 基础使用演示")
    print("=" * 60)
    print()
    
    print("1. 启动交互式模式:")
    print("   python main.py start -i")
    print()
    
    print("2. 菜单导航:")
    print("   - 输入数字 (0-18) 执行对应功能")
    print("   - 输入 'h' 查看帮助")
    print("   - 输入 'q' 退出程序")
    print("   - 输入 'c' 清屏")
    print()
    
    print("3. 操作流程示例:")
    print("   a) 初始化数据: 选择 1")
    print("   b) 运行回测: 选择 3-6")
    print("   c) 实盘选股: 选择 7-12")
    print("   d) 查看历史: 选择 15")
    print("   e) 导出报告: 选择 17")
    print()


def demo_quick_presets():
    """快速预设演示"""
    print("=" * 60)
    print("⚡ 快速预设系统")
    print("=" * 60)
    print()
    
    launcher = QuickLauncher()
    presets = launcher.list_presets()
    
    print("可用预设:")
    for name, desc in presets.items():
        print(f"  {name:30s} - {desc}")
    print()
    
    print("使用示例:")
    print("  1. 在交互模式中: 选择 18 (加载快速预设)")
    print("  2. 选择预设名称 (如: backtest_rsrs_stable)")
    print("  3. 系统自动加载参数并执行")
    print()


def demo_session_management():
    """会话管理演示"""
    print("=" * 60)
    print("💾 会话管理系统")
    print("=" * 60)
    print()
    
    manager = SessionManager()
    
    print("功能特性:")
    print("  ✓ 自动记录所有操作")
    print("  ✓ 支持历史查看 (选项15)")
    print("  ✓ 支持结果对比 (选项16)")
    print("  ✓ 支持导出报告 (选项17)")
    print("  ✓ 多格式导出 (JSON/CSV/Excel)")
    print()
    
    print("导出示例:")
    print("  1. 执行若干操作（回测、扫描等）")
    print("  2. 选择 17 (导出会话报告)")
    print("  3. 选择格式 (json/csv/xlsx)")
    print("  4. 查看导出文件: sessions/session_export_*.json")
    print()


def demo_workflow():
    """典型工作流程演示"""
    print("=" * 60)
    print("📈 典型工作流程示例")
    print("=" * 60)
    print()
    
    print("场景1: 策略回测对比")
    print("  1. 选择 3 - RSRS策略回测")
    print("  2. 选择 4 - Momentum策略回测")
    print("  3. 选择 15 - 查看会话历史")
    print("  4. 选择 16 - 对比历史结果")
    print("  5. 输入要对比的序号: 1,2")
    print("  6. 查看对比表格")
    print()
    
    print("场景2: 实盘选股")
    print("  1. 选择 2 - 增量更新数据")
    print("  2. 选择 18 - 加载快速预设")
    print("  3. 选择 scan_ensemble")
    print("  4. 查看选股结果")
    print("  5. 可选: 选择 13 - 对感兴趣的股票进行诊断")
    print()
    
    print("场景3: 单股深度分析")
    print("  1. 选择 13 - 单股诊断分析")
    print("  2. 输入股票代码 (如: 000001)")
    print("  3. 查看诊断报告")
    print("  4. 选择 17 - 导出诊断结果")
    print()


def demo_programmatic_usage():
    """编程式使用演示"""
    print("=" * 60)
    print("💻 编程式使用示例")
    print("=" * 60)
    print()
    
    print("直接调用交互式引擎:")
    print("""
    from interactive_main import InteractiveEngine
    
    # 创建引擎实例
    engine = InteractiveEngine()
    
    # 启动交互模式
    engine.run()
    """)
    print()
    
    print("使用会话管理器:")
    print("""
    from session_manager import SessionManager
    
    # 创建管理器
    manager = SessionManager()
    
    # 保存操作结果
    manager.save_result(
        operation='backtest_rsrs',
        parameters={'start': '2020-01-01', 'end': '2023-12-31'},
        result=backtest_metrics,
        exec_time=125.5
    )
    
    # 查看历史
    history = manager.view_history()
    print(history)
    
    # 导出报告
    filepath = manager.export_session(format='json')
    """)
    print()


def demo_all_features():
    """完整功能演示"""
    print("=" * 60)
    print("✨ 交互式菜单系统功能清单")
    print("=" * 60)
    print()
    
    features = {
        "数据管理": [
            "全量数据初始化 (选项1)",
            "增量数据更新 (选项2)"
        ],
        "策略回测": [
            "RSRS中期策略 (选项3)",
            "Momentum动量策略 (选项4)",
            "ShortTermRSRS短线策略 (选项5)",
            "AlphaHunter超短策略 (选项6)"
        ],
        "实盘选股": [
            "因子融合选股 (选项7)",
            "RSRS规则选股 (选项8)",
            "Momentum规则选股 (选项9)",
            "ShortTermRSRS选股 (选项10)",
            "AlphaHunter选股 (选项11)",
            "多策略融合选股 (选项12)"
        ],
        "市场分析": [
            "单股深度诊断 (选项13)",
            "系统信息查看 (选项14)"
        ],
        "会话管理": [
            "历史记录查看 (选项15)",
            "结果对比分析 (选项16)",
            "导出完整报告 (选项17)",
            "快速预设加载 (选项18)"
        ]
    }
    
    for category, items in features.items():
        print(f"[bold]{category}: [/bold]")
        for item in items:
            print(f"  • {item}")
        print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("📚 交互式菜单系统 - 完整演示")
    print("=" * 60 + "\n")
    
    demo_basic_usage()
    demo_quick_presets()
    demo_session_management()
    demo_workflow()
    demo_programmatic_usage()
    demo_all_features()
    
    print("=" * 60)
    print("🎉 演示完成!")
    print("=" * 60)
    print()
    print("开始使用:")
    print("  python main.py start -i")
    print()