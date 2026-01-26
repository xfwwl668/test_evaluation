# ============================================================================
# 文件: session_manager.py
# ============================================================================
"""
会话管理器 - 保存操作历史和结果
支持多格式导出和结果对比
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import pandas as pd


@dataclass
class SessionResult:
    """单个会话结果"""
    operation: str  # 操作类型 ('backtest', 'scan', 'diagnose')
    timestamp: str  # 执行时间
    parameters: Dict  # 输入参数
    result: Any  # 执行结果
    execution_time: float  # 耗时(秒)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'operation': self.operation,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'execution_time': self.execution_time,
            'result_preview': self._get_result_preview()
        }

    def _get_result_preview(self) -> str:
        """获取结果预览"""
        try:
            if self.operation.startswith('backtest'):
                if isinstance(self.result, dict) and 'metrics' in self.result:
                    metrics = self.result['metrics']
                    return f"总收益: {metrics.get('total_return', 0):.2%}, 夏普: {metrics.get('sharpe_ratio', 0):.2f}"
            elif self.operation.startswith('scan'):
                if hasattr(self.result, '__len__'):
                    return f"选股数量: {len(self.result)}"
            elif self.operation == 'diagnose':
                return "诊断完成"
            
            return str(self.result)[:100]
        except Exception:
            return "预览不可用"


class SessionManager:
    """会话管理器 - 管理操作历史和结果"""

    def __init__(self, session_dir: str = 'sessions'):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[SessionResult] = []
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_result(self, operation: str, parameters: Dict, result: Any, exec_time: float) -> str:
        """保存单个操作结果"""
        result_obj = SessionResult(
            operation=operation,
            timestamp=datetime.now().isoformat(),
            parameters=parameters,
            result=result,
            execution_time=exec_time
        )

        self.results.append(result_obj)
        self._save_to_file(result_obj)

        return result_obj.timestamp

    def _save_to_file(self, result: SessionResult) -> None:
        """保存到文件"""
        filename = f"{self.current_session}_{len(self.results):03d}_{result.operation}.json"
        filepath = self.session_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    def get_result(self, index: int) -> Optional[SessionResult]:
        """获取指定索引的结果 (1-based)"""
        if 1 <= index <= len(self.results):
            return self.results[index - 1]
        return None

    def compare_results(self, indices: List[int]) -> pd.DataFrame:
        """对比多个结果"""
        if not indices:
            return pd.DataFrame()

        data = []
        for idx in indices:
            result = self.get_result(idx)
            if result:
                base_info = {
                    '序号': idx,
                    '操作': result.operation,
                    '时间': result.timestamp[:19],
                    '耗时(s)': f"{result.execution_time:.2f}",
                }

                # 添加策略特定的指标
                if result.operation.startswith('backtest'):
                    metrics = result.result.get('metrics', {}) if isinstance(result.result, dict) else {}
                    base_info.update({
                        '总收益': f"{metrics.get('total_return', 0):.2%}",
                        '年化收益': f"{metrics.get('annual_return', 0):.2%}",
                        '最大回撤': f"{metrics.get('max_drawdown', 0):.2%}",
                        '夏普比率': f"{metrics.get('sharpe_ratio', 0):.2f}",
                        '胜率': f"{metrics.get('win_rate', 0):.1%}",
                        '交易次数': metrics.get('num_trades', 0)
                    })

                elif result.operation.startswith('scan'):
                    result_data = result.result if isinstance(result.result, list) else []
                    base_info['选股数量'] = len(result_data)
                    if result_data and isinstance(result_data[0], dict):
                        # 尝试获取平均分
                        scores = [r.get('alpha_score', 0) for r in result_data if 'alpha_score' in r]
                        if scores:
                            base_info['平均分'] = f"{sum(scores) / len(scores):.3f}"

                data.append(base_info)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.index = range(1, len(df) + 1)
        return df

    def export_session(self, format: str = 'json') -> str:
        """导出整个会话"""
        if format not in ['json', 'csv', 'xlsx']:
            format = 'json'

        filename = f"session_export_{self.current_session}.{format}"
        filepath = self.session_dir / filename

        if format == 'json':
            export_data = {
                'session_id': self.current_session,
                'export_time': datetime.now().isoformat(),
                'total_operations': len(self.results),
                'operations': [result.to_dict() for result in self.results]
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

        elif format in ['csv', 'xlsx']:
            export_records = []
            for i, result in enumerate(self.results):
                record = {
                    '序号': i + 1,
                    '操作': result.operation,
                    '时间': result.timestamp,
                    '耗时(秒)': result.execution_time,
                    '结果预览': result._get_result_preview()
                }

                # 添加参数
                for key, value in result.parameters.items():
                    param_key = f'参数_{key}'
                    record[param_key] = str(value)

                export_records.append(record)

            df = pd.DataFrame(export_records)

            if format == 'csv':
                df.to_csv(filepath, index=False, encoding='utf_8_sig')
            else:
                try:
                    df.to_excel(filepath, index=False, engine='openpyxl')
                except ImportError:
                    raise ImportError("导出Excel需要安装 openpyxl: pip install openpyxl")

        return str(filepath)

    def view_history(self) -> pd.DataFrame:
        """查看会话历史"""
        if not self.results:
            self._load_from_files()

        if not self.results:
            return pd.DataFrame(columns=['序号', '操作', '时间', '耗时(s)', '结果预览'])

        data = []
        for i, result in enumerate(self.results, 1):
            data.append({
                '序号': i,
                '操作': result.operation,
                '时间': result.timestamp[:19],
                '耗时(s)': f"{result.execution_time:.2f}",
                '结果预览': result._get_result_preview()
            })

        df = pd.DataFrame(data)
        df.index = range(1, len(df) + 1)
        return df

    def _load_from_files(self) -> None:
        """从文件加载历史会话"""
        json_files = list(self.session_dir.glob("*.json"))
        if not json_files:
            return

        loaded = 0
        for file in sorted(json_files):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 处理不同的文件格式
                if 'operations' in data:
                    # 导出的会话文件
                    for op in data['operations']:
                        result = SessionResult(
                            operation=op.get('operation', 'unknown'),
                            timestamp=op.get('timestamp', ''),
                            parameters=op.get('parameters', {}),
                            result=f"加载自文件: {file.name}",
                            execution_time=op.get('execution_time', 0)
                        )
                        self.results.append(result)
                        loaded += 1
                else:
                    # 单个操作文件
                    result = SessionResult(
                        operation=data.get('operation', 'unknown'),
                        timestamp=data.get('timestamp', ''),
                        parameters=data.get('parameters', {}),
                        result=f"加载自文件: {file.name}",
                        execution_time=data.get('execution_time', 0)
                    )
                    self.results.append(result)
                    loaded += 1

            except Exception as e:
                print(f"加载文件失败 {file}: {e}")

        if loaded > 0:
            print(f"已加载 {loaded} 个历史操作")

    def clear_session(self) -> None:
        """清理当前会话"""
        self.results = []
        print("当前会话已清空")

    def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计"""
        return {
            'total_operations': len(self.results),
            'session_id': self.current_session,
            'operations_by_type': pd.Series([r.operation for r in self.results]).value_counts().to_dict()
        }

    def get_operations_by_type(self, operation_type: str) -> List[SessionResult]:
        """获取特定类型的操作"""
        return [r for r in self.results if r.operation.startswith(operation_type)]

    def get_last_n_operations(self, n: int = 10) -> List[SessionResult]:
        """获取最近的N个操作"""
        return self.results[-n:]