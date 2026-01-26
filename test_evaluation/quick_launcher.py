# ============================================================================
# 文件: quick_launcher.py
# ============================================================================
"""
快速启动预设管理器 - 保存常用操作参数
"""
from __future__ import annotations

from typing import Dict, Any, List
import json
from pathlib import Path


class QuickLauncher:
    """快速启动预设管理"""

    # 内置预设定义
    PRESETS = {
        'backtest_rsrs_stable': {
            'strategy': 'rsrs',
            'start': '2020-01-01',
            'end': '2023-12-31',
            'capital': 1000000,
            'freq': 'W',
            'description': 'RSRS中期稳健策略'
        },
        'backtest_momentum_trend': {
            'strategy': 'momentum',
            'start': '2021-01-01',
            'end': '2023-12-31',
            'capital': 1000000,
            'freq': 'W',
            'description': '动量趋势跟踪策略'
        },
        'backtest_short_term': {
            'strategy': 'short_term',
            'start': '2021-01-01',
            'end': '2023-12-31',
            'capital': 500000,
            'freq': 'D',
            'description': '短线高胜率策略'
        },
        'backtest_alpha_aggressive': {
            'strategy': 'alpha_hunter',
            'start': '2021-01-01',
            'end': '2023-12-31',
            'capital': 500000,
            'freq': 'D',
            'description': 'AlphaHunter超短线激进策略'
        },
        'scan_short_term': {
            'mode': 'short_term',
            'top_n': 20,
            'description': '短线高胜率选股'
        },
        'scan_ensemble': {
            'mode': 'ensemble',
            'top_n': 30,
            'description': '多策略融合选股'
        },
        'scan_factor_classical': {
            'mode': 'factor',
            'top_n': 50,
            'description': '因子融合经典选股'
        },
        'scan_rsrs_rules': {
            'mode': 'rsrs',
            'top_n': 30,
            'description': 'RSRS规则选股'
        },
        'scan_momentum_rules': {
            'mode': 'momentum',
            'top_n': 25,
            'description': '动量规则选股'
        },
        'scan_alpha_hunter': {
            'mode': 'alpha_hunter',
            'top_n': 20,
            'description': 'AlphaHunter私募级选股'
        }
    }

    def __init__(self, preset_file: str = None):
        """初始化快速启动器"""
        if preset_file is None:
            preset_file = "user_presets.json"
        self.preset_file = Path(preset_file)
        self.user_presets: Dict[str, Dict[str, Any]] = self._load_user_presets()

    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """列出所有预设 (内置+用户)"""
        all_presets = {}
        
        # 内置预设
        for name, preset in cls.PRESETS.items():
            all_presets[name] = f"[系统] {preset['description']}"
        
        return all_presets

    @classmethod
    def load_preset(cls, name: str) -> Dict[str, Any]:
        """加载指定预设"""
        if name in cls.PRESETS:
            return cls.PRESETS[name].copy()
        else:
            raise ValueError(f"预设 '{name}' 不存在. 可用预设: {list(cls.PRESETS.keys())}")

    def _load_user_presets(self) -> Dict[str, Dict[str, Any]]:
        """从文件加载用户预设"""
        if not self.preset_file.exists():
            return {}
        
        try:
            with open(self.preset_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载用户预设失败: {e}")
            return {}

    def save_preset(self, name: str, preset_data: Dict[str, Any]) -> None:
        """保存自定义预设
        
        Args:
            name: 预设名称
            preset_data: 预设数据，必须包含 'description' 字段
        """
        if 'description' not in preset_data:
            raise ValueError("预设数据必须包含 'description' 字段")
        
        if name in self.PRESETS:
            raise ValueError(f"无法覆盖系统预设 '{name}'")
        
        self.user_presets[name] = preset_data
        
        try:
            with open(self.preset_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_presets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存用户预设失败: {e}")

    def delete_preset(self, name: str) -> bool:
        """删除用户预设"""
        if name in self.PRESETS:
            print(f"无法删除系统预设 '{name}'")
            return False
        
        if name not in self.user_presets:
            print(f"用户预设 '{name}' 不存在")
            return False
        
        del self.user_presets[name]
        
        try:
            with open(self.preset_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_presets, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"删除用户预设失败: {e}")
            return False

    def get_preset_categories(self) -> Dict[str, List[str]]:
        """按类别获取预设列表"""
        categories = {
            'backtest': [],
            'scan': [],
            'other': []
        }
        
        for name in self.PRESETS.keys():
            if name.startswith('backtest_'):
                categories['backtest'].append(name)
            elif name.startswith('scan_'):
                categories['scan'].append(name)
            else:
                categories['other'].append(name)
        
        return categories

    def get_preset_details(self, name: str) -> Dict[str, Any]:
        """获取预设详细信息"""
        preset = self.load_preset(name)
        
        details = {
            'name': name,
            'category': '系统预设' if name in self.PRESETS else '用户预设',
            'description': preset.get('description', ''),
            'parameters': {k: v for k, v in preset.items() if k != 'description'}
        }
        
        return details

    def validate_preset_data(self, preset_data: Dict[str, Any]) -> tuple[bool, list[str]]:
        """验证预设数据的有效性
        
        Returns:
            tuple: (是否有效, 错误信息列表)
        """
        errors = []
        
        if 'description' not in preset_data:
            errors.append("缺少 'description' 字段")
        
        if not isinstance(preset_data.get('description', ''), str):
            errors.append("'description' 必须是字符串")
        
        # 检查策略回测预设
        if 'strategy' in preset_data:
            if preset_data['strategy'] not in ['rsrs', 'momentum', 'short_term', 'alpha_hunter']:
                errors.append(f"无效的策略名称: {preset_data['strategy']}")
            
            required_fields = ['start', 'end', 'capital']
            for field in required_fields:
                if field not in preset_data:
                    errors.append(f"策略回测预设缺少 '{field}' 字段")
        
        # 检查扫描预设
        if 'mode' in preset_data:
            if preset_data['mode'] not in ['factor', 'rsrs', 'momentum', 'short_term', 'alpha_hunter', 'ensemble']:
                errors.append(f"无效的扫描模式: {preset_data['mode']}")
            
            if 'top_n' not in preset_data:
                errors.append("扫描预设缺少 'top_n' 字段")
            elif not isinstance(preset_data['top_n'], int) or preset_data['top_n'] <= 0:
                errors.append("'top_n' 必须是正整数")
        
        return len(errors) == 0, errors