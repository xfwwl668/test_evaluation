# ============================================================================
# 文件: ui_helpers.py
# ============================================================================
"""
UI辅助工具 - 彩色输出、进度条、格式化
"""
from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text


def print_result_table(console: Console, data: pd.DataFrame, title: str) -> None:
    """打印彩色表格结果"""
    if data.empty:
        console.print("[yellow]暂无数据[/yellow]")
        return

    table = Table(title=title, show_lines=True)
    
    for col in data.columns:
        if col in ['代码', '操作', '序号', '操作类型']:
            style = "cyan"
        elif any(x in col.lower() for x in ['收益', '中率', '胜率', '盈利', '分', 'ratio', 'return', 'rate']):
            style = "green"
        elif any(x in col.lower() for x in ['风险', '回撤', '波动', '亏损', 'loss', 'drawdown', 'risk']):
            style = "red"
        elif any(x in col.lower() for x in ['时间', '日期', 'date', 'time']):
            style = "magenta"
        else:
            style = "white"
        
        table.add_column(col, style=style, max_width=30)
    
    for _, row in data.iterrows():
        row_values = []
        for col in data.columns:
            value = row[col]
            if pd.isna(value):
                row_values.append("-")
            elif isinstance(value, (int, float)):
                row_values.append(f"{value:g}")
            else:
                row_values.append(str(value)[:100])
        
        table.add_row(*row_values)
    
    console.print(table)


def show_progress_bar(total: int, task_name: str, console: Console = None) -> Progress:
    """显示进度条"""
    if console is None:
        console = Console()
    
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None, complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True
    )
    
    task = progress.add_task(task_name, total=total)
    return progress, task


def format_backtest_result(metrics: Dict[str, Any]) -> Panel:
    """格式化回测结果为彩色面板"""
    if not metrics or 'metrics' not in metrics:
        return Panel(Text("无有效结果", style="red"), title="回测结果")
    
    perf = metrics['metrics']
    
    text = Text()
    text.append("📈 绩效指标\n", style="bold cyan")
    text.append(f"总收益: {perf.get('total_return', 0):.2%}", style="green" if perf.get('total_return', 0) > 0 else "red")
    text.append(f" | 年化收益: {perf.get('annual_return', 0):.2%}\n", style="green" if perf.get('annual_return', 0) > 0 else "red")
    text.append(f"最大回撤: {perf.get('max_drawdown', 0):.2%}", style="red")
    text.append(f" | 夏普比率: {perf.get('sharpe_ratio', 0):.2f}\n", style="white")
    text.append(f"交易次数: {perf.get('num_trades', 0)}", style="white")
    text.append(f" | 胜率: {perf.get('win_rate', 0):.1%}", style="green" if perf.get('win_rate', 0) > 0.5 else "red")
    
    if 'portfolio_values' in metrics:
        start_value = metrics['portfolio_values'][0]
        end_value = metrics['portfolio_values'][-1]
        text.append(f"\n💰 资金变化: {start_value:,.0f} → {end_value:,.0f}", style="cyan")
    
    return Panel(text, title="回测结果", border_style="cyan")


def format_scan_result(result_df: pd.DataFrame) -> Panel:
    """格式化选股结果为彩色面板"""
    if result_df.empty:
        return Panel(Text("未找到符合条件的股票", style="yellow"), title="选股结果")
    
    text = Text()
    text.append(f"🔍 选股结果 (共 {len(result_df)} 只)\n", style="bold cyan")
    text.append(f"前5名:\n", style="dim")
    
    top5 = result_df.head(5)
    for i, (idx, row) in enumerate(top5.iterrows(), 1):
        code = row.get('代码', row.get('code', '-'))
        score = row.get('综合评分', row.get('alpha_score', '-'))
        text.append(f"{i}. {code}: {score}\n", style="white")
    
    return Panel(text, title="选股结果", border_style="cyan")


def format_diagnose_result(diagnosis: Dict[str, Any]) -> Panel:
    """格式化诊断结果为彩色面板"""
    text = Text()
    text.append(f"🔬 股票诊断: {diagnosis.get('code', '-')}\n\n", style="bold cyan")
    
    metrics = diagnosis.get('metrics', {})
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if 'change' in key.lower() or 'bias' in key.lower():
                style = "green" if float(value) > 0 else "red"
            else:
                style = "white"
            text.append(f"{key}: {value:.3f}\n", style=style)
        else:
            text.append(f"{key}: {value}\n", style="white")
    
    recommendations = diagnosis.get('recommendations', [])
    if recommendations:
        text.append(f"\n💡 建议:\n", style="bold yellow")
        for rec in recommendations:
            text.append(f"• {rec}\n", style="yellow")
    
    return Panel(text, title="诊断报告", border_style="cyan")


def create_menu_panel(title: str, options: List[str]) -> Panel:
    """创建菜单面板"""
    text = Text()
    
    for i, option in enumerate(options, 1):
        text.append(f"  {i:2d}. {option}\n", style="white")
    
    return Panel(text, title=title, border_style="blue")


def print_status_message(console: Console, message: str, status: str = "info") -> None:
    """打印状态消息"""
    status_styles = {
        'success': ('✓', 'green'),
        'error': ('✗', 'red'),
        'warning': ('⚠', 'yellow'),
        'info': ('ℹ', 'blue')
    }
    
    symbol, color = status_styles.get(status, ('•', 'white'))
    console.print(f"[{color}]{symbol} {message}[/{color}]")


def display_comparison_table(console: Console, data: List[Dict[str, Any]], title: str) -> None:
    """显示对比表格"""
    if not data:
        return
    
    table = Table(title=title)
    
    # 添加列
    columns = data[0].keys()
    for col in columns:
        table.add_column(col, style="white")
    
    # 添加行
    for row in data:
        table.add_row(*[str(value) for value in row.values()])
    
    console.print(table)


def create_parameter_input_prompt(param_name: str, default_value: Any, description: str = "") -> str:
    """创建参数输入提示"""
    prompt = f"{param_name}"
    if description:
        prompt += f" ({description})"
    
    if default_value is not None:
        prompt += f" [默认: {default_value}]: "
    else:
        prompt += ": "
    
    return prompt


# 颜色常量定义
class Colors:
    """常用颜色定义"""
    SUCCESS = "green"
    ERROR = "red"
    WARNING = "yellow"
    INFO = "blue"
    TITLE = "cyan"
    HIGHLIGHT = "magenta"
    DIM = "dim"
    WHITE = "white"


def truncate_text(text: str, max_length: int = 50) -> str:
    """截断文本并添加省略号"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_number(value: Any, decimals: int = 2) -> str:
    """格式化数字"""
    if value is None:
        return "-"
    
    try:
        if isinstance(value, (int, float)):
            if abs(value) >= 1e6:
                return f"{value / 1e6:.1f}M"
            elif abs(value) >= 1e3:
                return f"{value / 1e3:.1f}K"
            else:
                return f"{value:.{decimals}f}"
        return str(value)
    except (ValueError, TypeError):
        return str(value)[:20]