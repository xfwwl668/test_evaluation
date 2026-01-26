# MarketScanner 多模式选股功能

## 概述

MarketScanner 已增强为支持 6 种选股模式，将策略的入场规则集成到市场扫描中，实现实盘选股与回测逻辑的统一。

## 支持的模式

### 1. factor - 因子融合模式（默认）

**描述**: 保持原有的多因子评分逻辑，后向兼容。

**选股逻辑**:
- R² >= 阈值 (默认 0.8)
- VWAP 偏离 > 0
- RSRS Z-Score 有效
- 综合评分 = RSRS权重 * RSRS评分 + 量能权重 * 量能评分 + 价格权重 * 价格评分

**使用示例**:
```python
from analysis import MarketScanner

scanner = MarketScanner()
result = scanner.scan(mode='factor', top_n=50)
```

### 2. rsrs - RSRS策略模式

**描述**: 使用 RSRS 策略的入场规则。

**选股逻辑**:
- RSRS Z-Score > entry_threshold (默认 0.7)
- R² >= r2_threshold (默认 0.8)
- 可选量价过滤:
  - 成交量分位 30%-75%
  - OBV 上升趋势
  - 价格在 VWAP 上方

**默认参数**:
```python
{
    'entry_threshold': 0.7,
    'r2_threshold': 0.8,
    'use_volume_filter': True,
    'vol_warm_low': 0.3,
    'vol_warm_high': 0.75,
}
```

**使用示例**:
```python
scanner = MarketScanner()

# 使用默认参数
result = scanner.scan(mode='rsrs', top_n=30)

# 自定义参数
result = scanner.scan(
    mode='rsrs',
    top_n=30,
    strategy_params={
        'entry_threshold': 0.9,
        'use_volume_filter': False
    }
)
```

### 3. momentum - 动量策略模式

**描述**: 选择过去 N 日涨幅最大的股票。

**选股逻辑**:
- 动量 >= min_momentum (默认 5%)
- 波动率 <= max_volatility (默认 50%)
- 按动量排序

**默认参数**:
```python
{
    'min_momentum': 0.05,
    'max_volatility': 0.5,
    'lookback': 20,
}
```

**使用示例**:
```python
result = scanner.scan(mode='momentum', top_n=20)
```

### 4. short_term - 短线RSRS模式

**描述**: 使用 ShortTermRSRS 策略的严格入场过滤。

**选股逻辑**:
- RSRS Score > rsrs_entry_threshold (默认 0.7)
- R² >= r2_threshold (默认 0.8)
- Price > MA5 AND Price > MA20
- Volume > MA5_Vol × volume_multiplier (默认 1.5)
- 价格和成交量基础过滤

**默认参数**:
```python
{
    'rsrs_entry_threshold': 0.7,
    'r2_threshold': 0.8,
    'volume_multiplier': 1.5,
    'min_price': 3.0,
    'max_price': 100.0,
    'min_volume': 1000000,
}
```

**使用示例**:
```python
result = scanner.scan(mode='short_term', top_n=15)
```

### 5. alpha_hunter - AlphaHunter模式

**描述**: 使用 AlphaHunter 策略的极致过滤条件。

**选股逻辑**:
- RSRS >= rsrs_threshold (默认 0.8)
- R² >= r2_threshold (默认 0.85)
- Price > MA5 AND MA5_slope > slope_threshold
- 换手率 <= max_turnover (默认 25%)
- 距离压力位 >= min_pressure_distance (默认 5%)
- 非涨停板（可选）
- 市场广度 >= threshold (默认 40%)

**默认参数**:
```python
{
    'rsrs_threshold': 0.8,
    'r2_threshold': 0.85,
    'ma5_slope_threshold': 0.001,
    'max_turnover': 0.25,
    'min_pressure_distance': 0.05,
    'min_price': 5.0,
    'max_price': 80.0,
    'min_volume': 2000000,
    'market_breadth_threshold': 0.40,
    'allow_limit_up_chase': False,
}
```

**使用示例**:
```python
result = scanner.scan(mode='alpha_hunter', top_n=15)
```

### 6. ensemble - 多策略融合模式

**描述**: 投票融合多个策略的选股结果。

**融合方式**:
- 各策略独立评分
- 投票统计: 股票在多少个策略中出现
- 加权融合: 结合各策略评分和权重
- 过滤: 至少出现在 min_votes 个策略中

**默认参数**:
```python
{
    'weights': {
        'rsrs': 0.3,
        'momentum': 0.2,
        'short_term': 0.3,
        'alpha_hunter': 0.2,
    },
    'min_votes': 2,
    'voting_mode': 'weighted',  # 'count' or 'weighted'
}
```

**使用示例**:
```python
# 使用默认权重
result = scanner.scan(mode='ensemble', top_n=30)

# 自定义权重
result = scanner.scan(
    mode='ensemble',
    top_n=30,
    strategy_params={
        'weights': {
            'rsrs': 0.4,
            'momentum': 0.1,
            'short_term': 0.4,
            'alpha_hunter': 0.1,
        },
        'min_votes': 3,  # 至少 3 个策略选中
    }
)
```

## API 参考

### MarketScanner.scan()

**完整签名**:
```python
def scan(
    self,
    target_date: str = None,      # 目标日期 (None=最新)
    top_n: int = 50,               # 返回数量
    mode: str = 'factor',          # 选股模式
    filters: Dict = None,          # 过滤条件
    strategy_params: Dict = None   # 策略参数覆盖
) -> pd.DataFrame:
```

**参数说明**:
- `target_date`: 扫描日期，默认为最新日期
- `top_n`: 返回的股票数量
- `mode`: 选股模式，可选值：
  - `'factor'` - 因子融合模式（默认）
  - `'rsrs'` - RSRS策略模式
  - `'momentum'` - 动量策略模式
  - `'short_term'` - 短线RSRS模式
  - `'alpha_hunter'` - AlphaHunter模式
  - `'ensemble'` - 多策略融合模式
- `filters`: 额外的过滤条件（字典）
- `strategy_params`: 覆盖默认策略参数（字典）

**返回值**:
- `pd.DataFrame`: 金股表，包含以下列（根据模式不同可能有差异）:
  - 代码
  - 收盘价
  - RSRS分数
  - R²
  - 量能分位
  - VWAP偏离
  - OBV趋势
  - 波动率
  - 综合评分

### 快捷函数

```python
from analysis import scan_market

result = scan_market(
    db_path=None,
    target_date=None,
    top_n=50,
    mode='factor',
    filters=None,
    strategy_params=None,
    **kwargs
)
```

## 配置管理

在 `config/settings.py` 中可以配置各模式的默认参数：

```python
@dataclass
class StrategyConfig:
    # RSRS 扫描参数
    SCAN_RSRS_PARAMS: Dict = field(default_factory=lambda: {
        'entry_threshold': 0.7,
        'r2_threshold': 0.8,
        ...
    })
    
    # 动量扫描参数
    SCAN_MOMENTUM_PARAMS: Dict = field(default_factory=lambda: {
        'min_momentum': 0.05,
        ...
    })
    
    # 短线扫描参数
    SCAN_SHORT_TERM_PARAMS: Dict = field(default_factory=lambda: {
        'rsrs_entry_threshold': 0.7,
        ...
    })
    
    # AlphaHunter 扫描参数
    SCAN_ALPHA_HUNTER_PARAMS: Dict = field(default_factory=lambda: {
        'rsrs_threshold': 0.8,
        ...
    })
    
    # 融合扫描参数
    SCAN_ENSEMBLE_PARAMS: Dict = field(default_factory=lambda: {
        'weights': {...},
        ...
    })
```

## 后向兼容性

原有代码无需修改，默认使用 `factor` 模式：

```python
# 这些调用保持原有行为
scanner = MarketScanner()
result = scanner.scan(target_date='2024-01-01', top_n=50)

# 或者
result = scanner.scan('2024-01-01', 50, {'r2_min': 0.8})
```

## 使用建议

### 1. 日常选股

推荐使用 **ensemble** 模式，综合多个策略的优势：

```python
result = scanner.scan(mode='ensemble', top_n=30)
```

### 2. 激进短线

推荐使用 **short_term** 或 **alpha_hunter** 模式：

```python
result = scanner.scan(mode='short_term', top_n=15)
```

### 3. 趋势跟踪

推荐使用 **rsrs** 或 **momentum** 模式：

```python
result = scanner.scan(mode='rsrs', top_n=30)
```

### 4. 参数调优

通过 `strategy_params` 参数调整策略行为：

```python
# 更严格的 RSRS 过滤
result = scanner.scan(
    mode='rsrs',
    strategy_params={
        'entry_threshold': 0.9,
        'r2_threshold': 0.9
    }
)

# 更宽松的动量过滤
result = scanner.scan(
    mode='momentum',
    strategy_params={
        'min_momentum': 0.02,
        'max_volatility': 0.8
    }
)
```

## 性能特性

- **并行计算**: 使用多进程并行计算因子
- **智能缓存**: 根据模式按需计算因子，避免无意义的重复计算
- **高效融合**: ensemble 模式并发执行各策略评分

## 测试

运行测试：

```bash
cd test_evaluation
pytest tests/test_scanner.py -v
```

## 示例脚本

完整示例参见 `examples/scan_modes_demo.py`：

```bash
cd test_evaluation
python examples/scan_modes_demo.py
```

## 架构说明

### 组件关系

```
MarketScanner
    ├─ _load_data()           # 批量加载数据
    ├─ _compute_factors()     # 并行计算因子
    ├─ _filter_and_score()    # 过滤和评分
    │   ├─ mode='factor' → _filter_and_score_factor()  # 原逻辑
    │   └─ mode=other    → StrategyScorer.score()      # 策略评分器
    └─ _format_output()       # 格式化输出

StrategyScorer (抽象基类)
    ├─ RSRSScorer             # RSRS 策略评分器
    ├─ MomentumScorer         # 动量策略评分器
    ├─ ShortTermScorer        # 短线策略评分器
    ├─ AlphaHunterScorer      # AlphaHunter 策略评分器
    └─ EnsembleScorer         # 多策略融合评分器
```

### 数据流

```
1. 加载数据
   ↓
2. 并行计算因子 (根据 mode 选择性计算)
   ↓
3. 选择评分器
   ├─ mode='factor' → 原逻辑
   └─ mode=other    → 策略评分器
   ↓
4. 评分和排序
   ↓
5. 格式化输出
```

## 常见问题

### Q1: 如何选择合适的模式？

A: 根据交易风格选择：
- 稳健长线 → `factor` 或 `rsrs`
- 趋势跟踪 → `momentum`
- 短线交易 → `short_term`
- 极致短线 → `alpha_hunter`
- 综合考虑 → `ensemble`

### Q2: ensemble 模式的投票数如何设置？

A: 
- `min_votes=1`: 任一策略选中即可（宽松）
- `min_votes=2`: 至少 2 个策略选中（平衡）
- `min_votes=3`: 至少 3 个策略选中（严格）
- `min_votes=4`: 4 个策略都选中（极严格）

### Q3: 策略参数如何调优？

A: 建议流程：
1. 使用默认参数扫描
2. 观察结果质量和数量
3. 根据需求调整参数
4. 回测验证效果

### Q4: 是否会影响现有代码？

A: 完全后向兼容，现有代码无需修改。

## 更新日志

### v2.0.0 (2024-01-26)
- ✅ 新增 6 种选股模式支持
- ✅ 创建策略评分器框架
- ✅ 支持参数自定义覆盖
- ✅ 实现多策略融合
- ✅ 保持后向兼容性
- ✅ 添加完整测试

## 相关文档

- [策略评分器实现](./strategy_scorers_design.md)
- [配置参数详解](./scanner_config.md)
- [性能优化指南](./scanner_performance.md)

## 开发团队

- 架构设计: AI Assistant
- 实现开发: AI Assistant
- 测试验证: AI Assistant
