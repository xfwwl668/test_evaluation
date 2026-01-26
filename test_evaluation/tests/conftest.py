# ============================================================================
# 文件: tests/conftest.py
# ============================================================================
"""
Pytest 配置和 Fixtures
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# 添加项目根目录
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def sample_ohlcv():
    """生成示例 OHLCV 数据"""
    np.random.seed(42)
    n = 500
    
    dates = pd.bdate_range('2020-01-01', periods=n)
    
    returns = np.random.randn(n) * 0.02
    price = 20 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(n) * 0.005),
        'high': price * (1 + np.abs(np.random.randn(n)) * 0.015),
        'low': price * (1 - np.abs(np.random.randn(n)) * 0.015),
        'close': price,
        'vol': np.random.randint(1e5, 1e7, n).astype(np.float64),
        'amount': (price * np.random.randint(1e5, 1e7, n)).astype(np.float64)
    }, index=dates)
    
    return df


@pytest.fixture
def temp_db():
    """临时数据库"""
    import duckdb
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # 创建表
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE daily_bars (
            code VARCHAR, market INT, date DATE,
            open FLOAT, high FLOAT, low FLOAT, close FLOAT,
            vol BIGINT, amount BIGINT,
            PRIMARY KEY (code, date)
        )
    """)
    conn.close()
    
    yield db_path
    
    # 清理
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_stock_data(temp_db, sample_ohlcv):
    """填充模拟数据的数据库"""
    import duckdb
    
    df = sample_ohlcv.copy()
    df['code'] = '000001'
    df['market'] = 0
    df['date'] = df.index
    df = df.reset_index(drop=True)
    
    conn = duckdb.connect(temp_db)
    conn.execute("INSERT INTO daily_bars SELECT * FROM df")
    conn.close()
    
    return temp_db