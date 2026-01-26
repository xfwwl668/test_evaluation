# ============================================================================
# 文件: interactive_main.py
# ============================================================================
"""
交互式引擎 - 主程序
"""
from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

from utils.logger import setup_logging, get_logger
from config import settings
from session_manager import SessionManager
from quick_launcher import QuickLauncher
from ui_helpers import (
    print_result_table, 
    show_progress_bar, 
    format_backtest_result, 
    format_scan_result,
    print_status_message,
    Colors
)

# 确保模块可导入
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))


class InteractiveEngine:
    """
    交互式量化引擎主类
    提供完整的交互式菜单系统，支持18个功能选项
    """

    def __init__(self):
        self.console = Console()
        self.logger = get_logger("InteractiveEngine")
        self.session_manager = SessionManager()
        self.launcher = QuickLauncher()
        self.supported_strategies = ['rsrs', 'momentum', 'short_term', 'alpha_hunter']
        self.supported_scan_modes = ['factor', 'rsrs', 'momentum', 'short_term', 'alpha_hunter', 'ensemble']
        
    def run(self):
        """主循环，处理用户交互"""
        setup_logging(level='INFO')
        
        welcome_panel = Panel(
            Group(
                Text("🚀 量化交易引擎 - 交互式主程序 v2.0", style="bold cyan"),
                Text("", style=""),
                Text("输入 'h' 查看帮助 | 'q' 退出程序", style="dim blue")
            ),
            border_style="cyan",
            box=box.DOUBLE
        )
        self.console.print(welcome_panel)
        
        try:
            while True:
                self.display_menu()
                choice = self.get_user_input()
                
                if not self.handle_input(choice):
                    break
        except KeyboardInterrupt:
            self.console.print("\n[yellow]检测到中断信号，正在退出...[/yellow]")
        except Exception as e:
            self.logger.error(f"主循环出错: {e}")
            self.console.print(f"[bold red]❌ 系统错误: {str(e)}[/bold red]")
        finally:
            self.exit_program()

    def handle_input(self, choice: str) -> bool:
        """处理用户输入，返回是否继续运行"""
        if not choice:
            return True
            
        if choice.lower() in ['q', 'quit', 'exit', '0']:
            return False
        elif choice.lower() in ['h', 'help']:
            self.show_help()
        elif choice.lower() == 'c':
            self.clear_screen()
        else:
            self.execute_option(choice)
        
        return True

    def clear_screen(self):
        """清屏"""
        self.console.clear()

    def display_menu(self):
        """显示完整菜单"""
        menu_content = (
            "[bold cyan]╔════════════════════════════════════════════════╗[/bold cyan]\n"
            "[bold cyan]║   🚀 量化交易引擎 - 交互式主程序 v2.0         ║[/bold cyan]\n"
            "[bold cyan]╚════════════════════════════════════════════════╝[/bold cyan]\n\n"
            "[bold blue]┌────────────────────────────────────────────────┐[/bold blue]\n"
            "[bold blue]│ 【数据管理】                                    │[/bold blue]\n"
            "[bold blue]│  [/bold blue]1. 初始化数据库         (full download)       \n"
            "[bold blue]│  [/bold blue]2. 增量更新数据         (daily update)        \n"
            "[bold blue]│                                                 │[/bold blue]\n"
            "[bold blue]│ 【策略回测】                                    │[/bold blue]\n"
            "[bold blue]│  [/bold blue]3. RSRS策略回测         (中期趋势)            \n"
            "[bold blue]│  [/bold blue]4. Momentum策略回测     (动量)                \n"
            "[bold blue]│  [/bold blue]5. ShortTermRSRS回测    (短线高胜率)  ⚡      \n"
            "[bold blue]│  [/bold blue]6. AlphaHunter回测      (超短私募级)  🎯      \n"
            "[bold blue]│                                                 │[/bold blue]\n"
            "[bold blue]│ 【实盘选股】                                    │[/bold blue]\n"
            "[bold blue]│  [/bold blue]7. 因子融合选股         (classical)           \n"
            "[bold blue]│  [/bold blue]8. RSRS规则选股                               \n"
            "[bold blue]│  [/bold blue]9. Momentum规则选股                           \n"
            "[bold blue]│  [/bold blue]10. ShortTermRSRS选股                         \n"
            "[bold blue]│  [/bold blue]11. AlphaHunter选股                           \n"
            "[bold blue]│  [/bold blue]12. 多策略融合选股       (ensemble)           \n"
            "[bold blue]│                                                 │[/bold blue]\n"
            "[bold blue]│ 【市场分析】                                    │[/bold blue]\n"
            "[bold blue]│  [/bold blue]13. 单股诊断分析                               \n"
            "[bold blue]│  [/bold blue]14. 查看系统信息                               \n"
            "[bold blue]│                                                 │[/bold blue]\n"
            "[bold blue]│ 【会话管理】                                    │[/bold blue]\n"
            "[bold blue]│  [/bold blue]15. 查看会话历史                               \n"
            "[bold blue]│  [/bold blue]16. 对比历史结果                               \n"
            "[bold blue]│  [/bold blue]17. 导出会话报告                               \n"
            "[bold blue]│  [/bold blue]18. 加载快速预设                               \n"
            "[bold blue]│                                                 │[/bold blue]\n"
            "[bold blue]│  [/bold blue]0. 退出                                       \n"
            "[bold blue]└────────────────────────────────────────────────┘[/bold blue]"
        )
        
        self.console.print(menu_content)

    def get_user_input(self) -> str:
        """获取用户输入并验证"""
        return self.console.input("\n[bold cyan]请选择功能 (0-18): [/bold cyan]").strip()

    def execute_option(self, choice: str):
        """根据选择执行对应功能"""
        handlers = {
            '1': self.handle_init_db,
            '2': self.handle_update_db,
            '3': lambda: self.handle_backtest('rsrs'),
            '4': lambda: self.handle_backtest('momentum'),
            '5': lambda: self.handle_backtest('short_term'),
            '6': lambda: self.handle_backtest('alpha_hunter'),
            '7': lambda: self.handle_scan('factor'),
            '8': lambda: self.handle_scan('rsrs'),
            '9': lambda: self.handle_scan('momentum'),
            '10': lambda: self.handle_scan('short_term'),
            '11': lambda: self.handle_scan('alpha_hunter'),
            '12': lambda: self.handle_scan('ensemble'),
            '13': self.handle_diagnose,
            '14': self.handle_info,
            '15': self.handle_view_history,
            '16': self.handle_compare,
            '17': self.handle_export,
            '18': self.handle_quick_preset,
        }
        
        handler = handlers.get(choice)
        if handler:
            self.execute_with_error_handling(handler, choice)
        else:
            print_status_message(self.console, "无效选项，请重试", "error")

    def execute_with_error_handling(self, handler, choice: str):
        """带错误处理的执行函数"""
        start_time = time.time()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                progress.add_task("正在执行...", total=None)
                
                handler()
                
            exec_time = time.time() - start_time
            print_status_message(self.console, f"操作完成 (耗时: {exec_time:.2f}s)", "success")
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]操作被用户中断[/yellow]")
        except Exception as e:
            self.logger.error(f"选项 {choice} 执行出错: {e}")
            print_status_message(self.console, f"执行错误: {str(e)}", "error")

    def collect_backtest_params(self, strategy_name: str) -> Dict[str, Any]:
        """收集回测参数"""
        print_status_message(
            self.console, 
            f"🚀 运行 {strategy_name} 策略回测", 
            "info"
        )
        
        try:
            startup_preset = self.launcher.load_preset(f'backtest_{strategy_name}')
            default_start = startup_preset.get('start', '2020-01-01')
            default_end = startup_preset.get('end', '2023-12-31')
            default_capital = startup_preset.get('capital', 1000000)
            default_freq = startup_preset.get('freq', 'W')
        except:
            default_start = '2020-01-01'
            default_end = '2023-12-31'
            default_capital = 1000000
            default_freq = 'W'
        
        use_preset = self.console.input(
            f"[cyan]是否使用快速预设参数? (y/n) [默认: y]: [/cyan]"
        ).lower() or 'y'
        
        if use_preset == 'y':
            preset_name = f'backtest_{strategy_name}'
            try:
                params = self.launcher.load_preset(preset_name)
                print_status_message(
                    self.console, 
                    f"已加载预设: {params.get('description', preset_name)}", 
                    "info"
                )
                return params
            except Exception as e:
                print_status_message(self.console, f"加载预设失败: {e}", "warning")
        
        return {
            'start': self.console.input(f"[cyan]开始日期 [默认: {default_start}]: [/cyan]") or default_start,
            'end': self.console.input(f"[cyan]结束日期 [默认: {default_end}]: [/cyan]") or default_end,
            'capital': float(self.console.input(f"[cyan]初始资金 [默认: {default_capital}]: [/cyan]") or default_capital),
            'freq': self.console.input(f"[cyan]调仓频率 [默认: {default_freq}]: [/cyan]") or default_freq
        }

    def handle_backtest(self, strategy_name: str):
        """处理策略回测"""
        if strategy_name not in self.supported_strategies:
            print_status_message(self.console, f"不支持的策略: {strategy_name}", "error")
            return
        
        params = self.collect_backtest_params(strategy_name)
        
        from strategy.registry import StrategyRegistry
        from engine.backtest import BacktestEngine
        
        if strategy_name not in StrategyRegistry.list_all():
            print_status_message(self.console, f"策略 {strategy_name} 未注册", "error")
            return
        
        try:
            engine = BacktestEngine(initial_capital=params['capital'])
            strategy = StrategyRegistry.create(strategy_name, params={})
            engine.add_strategy(strategy)
            
            results = engine.run(params['start'], params['end'], rebalance_freq=params['freq'])
            
            self.console.print(format_backtest_result({'metrics': results}))
            
            self.session_manager.save_result(
                operation=f'backtest_{strategy_name}',
                parameters=params,
                result=results,
                exec_time=0
            )
            
        except Exception as e:
            self.logger.error(f"回测失败: {e}")
            print_status_message(self.console, f"回测失败: {str(e)}", "error")

    def collect_scan_params(self) -> Dict[str, Any]:
        """收集扫描参数"""
        target_date = self.console.input("[cyan]目标日期 (YYYY-MM-DD) [默认: 最新]: [/cyan]") or None
        top_n = int(self.console.input("[cyan]返回数量 [默认: 30]: [/cyan]") or 30)
        
        return {
            'target_date': target_date,
            'top_n': top_n
        }

    def handle_scan(self, mode: str):
        """处理市场扫描（集成6种mode）"""
        if mode not in self.supported_scan_modes:
            print_status_message(self.console, f"不支持的模式: {mode}", "error")
            return
        
        params = self.collect_scan_params()
        
        from analysis.scanner import MarketScanner
        scanner = MarketScanner()
        
        try:
            result = scanner.scan(
                target_date=params['target_date'],
                top_n=params['top_n'],
                mode=mode
            )
            
            if not result.empty:
                print_result_table(self.console, result, f"{mode} 选股结果")
                stats = self._calculate_scan_stats(result)
                print_status_message(self.console, f"选股统计: {stats}", "info")
            else:
                print_status_message(self.console, "未找到符合条件的股票", "warning")
            
            self.session_manager.save_result(
                operation=f'scan_{mode}',
                parameters=params,
                result=result.to_dict('records'),
                exec_time=0
            )
            
        except Exception as e:
            self.logger.error(f"扫描失败: {e}")
            print_status_message(self.console, f"扫描失败: {str(e)}", "error")

    def _calculate_scan_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算选股统计信息"""
        return {
            'total': len(df),
            'avg_score': df.get('综合评分', [0]).mean() if '综合评分' in df.columns else 0,
            'top10_avg': df.head(10).get('综合评分', [0]).mean() if '综合评分' in df.columns else 0
        }

    def handle_init_db(self):
        """初始化数据库"""
        print_status_message(self.console, "初始化数据库", "info")
        
        workers = self.console.input("[cyan]并行进程数 (回车使用默认值): [/cyan]").strip()
        workers = int(workers) if workers else None
        
        from core.updater import DataUpdater
        updater = DataUpdater()
        stats = updater.full_update(n_workers=workers)
        
        print_status_message(self.console, f"完成! 下载 {stats['downloaded']} 只股票", "success")

    def handle_update_db(self):
        """增量更新数据"""
        print_status_message(self.console, "增量更新数据", "info")
        
        is_full = self.console.input("[cyan]是否全量更新? (y/N): [/cyan]").lower() == 'y'
        
        from core.updater import DataUpdater
        updater = DataUpdater()
        
        stats = updater.full_update() if is_full else updater.incremental_update()
        updated = stats.get('updated', stats.get('written', 0))
        
        print_status_message(self.console, f"完成! 更新 {updated} 条", "success")

    def handle_diagnose(self):
        """单股诊断分析"""
        print_status_message(self.console, "单股诊断", "info")
        
        code = self.console.input("[cyan]请输入股票代码: [/cyan]").strip()
        if not code:
            print_status_message(self.console, "代码不能为空", "error")
            return
        
        from analysis.stock_doctor import StockDoctor
        doctor = StockDoctor()
        
        try:
            result = doctor.diagnose(code)
            report = doctor.generate_report(result)
            
            self.console.print(report)
            
            self.session_manager.save_result(
                operation='diagnose',
                parameters={'code': code},
                result=result,
                exec_time=0
            )
            
        except Exception as e:
            self.logger.error(f"诊断失败: {e}")
            print_status_message(self.console, f"诊断失败: {str(e)}", "error")

    def handle_info(self):
        """查看系统信息"""
        print_status_message(self.console, "系统信息", "info")
        
        info_data = self._gather_system_info()
        info_table = self._create_info_table(info_data)
        self.console.print(info_table)

    def _gather_system_info(self) -> Dict[str, str]:
        """收集系统信息"""
        return {
            "数据库路径": str(settings.path.DB_PATH),
            "日志目录": str(settings.path.LOG_DIR),
            "初始资金": f"{settings.backtest.INITIAL_CAPITAL:,.0f}",
            "RSRS窗口": str(settings.factor.RSRS_WINDOW),
            "动量窗口": str(settings.factor.MOMENTUM_WINDOW),
            "交易费率": f"{settings.backtest.COMMISSION_RATE:.1%}",
            "最大仓位": f"{settings.backtest.MAX_POSITION_WEIGHT:.1%}",
        }

    def _create_info_table(self, info_data: Dict[str, str]) -> Table:
        """创建系统信息表格"""
        table = Table(title="系统配置", show_lines=True, box=box.ROUNDED)
        table.add_column("参数", style="cyan", no_wrap=True)
        table.add_column("值", style="green")
        
        for key, value in info_data.items():
            table.add_row(key, value)
        
        return table

    def handle_view_history(self):
        """查看会话历史"""
        history = self.session_manager.view_history()
        if history.empty:
            print_status_message(self.console, "暂无历史记录", "warning")
        else:
            print_result_table(self.console, history, "会话历史")

    def handle_compare(self):
        """对比历史结果"""
        print_status_message(self.console, "对比历史结果", "info")
        
        indices_input = self.console.input("[cyan]输入序号 (用逗号分隔): [/cyan]").strip()
        if not indices_input:
            print_status_message(self.console, "未输入序号", "warning")
            return
        
        try:
            indices = [int(i.strip()) for i in indices_input.split(',') if i.strip()]
            if not indices:
                print_status_message(self.console, "无效序号", "error")
                return
            
            comparison = self.session_manager.compare_results(indices)
            if not comparison.empty:
                print_result_table(self.console, comparison, "对比结果")
            else:
                print_status_message(self.console, "无法对比这些结果", "warning")
                
        except ValueError:
            print_status_message(self.console, "请输入有效数字序号", "error")

    def handle_export(self):
        """导出会话报告"""
        print_status_message(self.console, "导出会话报告", "info")
        
        fmt = self.console.input("[cyan]导出格式 (json/csv/xlsx) [默认: json]: [/cyan]") or 'json'
        
        try:
            filepath = self.session_manager.export_session(format=fmt)
            print_status_message(self.console, f"报告已导出: {filepath}", "success")
        except Exception as e:
            self.logger.error(f"导出失败: {e}")
            print_status_message(self.console, f"导出失败: {str(e)}", "error")

    def handle_quick_preset(self):
        """加载快速预设"""
        print_status_message(self.console, "快速预设", "info")
        
        # 显示可用预设
        presets = self.launcher.list_presets()
        preset_table = Table(title="可用预设", show_lines=True)
        preset_table.add_column("名称", style="cyan")
        preset_table.add_column("描述", style="white")
        
        for name, desc in presets.items():
            preset_table.add_row(name, desc)
        
        self.console.print(preset_table)
        
        choice = self.console.input("[cyan]选择预设: [/cyan]").strip()
        if choice not in presets:
            print_status_message(self.console, "预设不存在", "error")
            return
        
        params = self.launcher.load_preset(choice)
        print_status_message(
            self.console, 
            f"已加载: {params.get('description', choice)}", 
            "info"
        )
        
        # 根据预设类型执行对应操作
        if choice.startswith('backtest_'):
            strategy = params['strategy']
            self.handle_backtest(strategy)
        elif choice.startswith('scan_'):
            mode = params['mode']
            self.handle_scan(mode)

    def show_help(self):
        """显示帮助信息"""
        help_panel = Panel(
            Group(
                Text("使用说明", style="bold cyan"),
                Text("", style=""),
                Text("• 输入数字执行对应功能", style="white"),
                Text("• 按 Enter 使用默认值", style="white"),
                Text("• 支持快速预设加速操作", style="white"),
                Text("• 所有操作自动记录到历史", style="white"),
                Text("", style=""),
                Text("快捷命令:", style="bold yellow"),
                Text("• h - 显示帮助", style="yellow"),
                Text("• q - 退出程序", style="yellow"),
                Text("• c - 清屏", style="yellow"),
                Text("", style=""),
                Text("按 Enter 继续...", style="dim")
            ),
            border_style="blue"
        )
        self.console.print(help_panel)
        input()

    def exit_program(self):
        """退出前的清理工作"""
        self.console.print("\n[bold green]✅ 会话结束，感谢使用量化交易引擎![/bold green]")
        
        # 保存会话统计
        stats = self.session_manager.get_session_stats()
        if stats['total_operations'] > 0:
            self.console.print(f"[dim]本次会话共执行 {stats['total_operations']} 次操作[/dim]")


if __name__ == "__main__":
    engine = InteractiveEngine()
    engine.run()