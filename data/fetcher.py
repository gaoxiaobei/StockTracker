import akshare as ak
import pandas as pd
from typing import Optional
import hashlib
import pickle
import os
from datetime import datetime, timedelta


# Cache directory
CACHE_DIR = ".data_cache"

def _get_cache_key(symbol: str, period: str, start_date: Optional[str], end_date: Optional[str], adjust: str) -> str:
    """Generate a cache key based on parameters."""
    cache_str = f"{symbol}_{period}_{start_date}_{end_date}_{adjust}"
    return hashlib.md5(cache_str.encode()).hexdigest()

def _get_cache_file_path(cache_key: str) -> str:
    """Get the full path for the cache file."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"{cache_key}.pkl")

def _is_cache_valid(file_path: str, max_age_hours: int = 24) -> bool:
    """Check if cache file exists and is not older than max_age_hours."""
    if not os.path.exists(file_path):
        return False

    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    return (datetime.now() - file_time).total_seconds() < max_age_hours * 3600

def get_stock_data(symbol: str, period: str = "daily", start_date: Optional[str] = None,
                   end_date: Optional[str] = None, adjust: str = "qfq") -> pd.DataFrame:
    """
    获取股票数据

    Args:
        symbol: 股票代码 (例如: "002607")
        period: 数据周期 ("daily", "weekly", "monthly")
        start_date: 开始日期 (格式: "YYYYMMDD")
        end_date: 结束日期 (格式: "YYYYMMDD")
        adjust: 复权类型 ("qfq": 前复权, "hfq": 后复权, "": 不复权)

    Returns:
        pd.DataFrame: 股票数据
    """
    # Generate cache key
    cache_key = _get_cache_key(symbol, period, start_date, end_date, adjust)
    cache_file_path = _get_cache_file_path(cache_key)

    # Check if cached data exists and is still valid
    if _is_cache_valid(cache_file_path):
        try:
            with open(cache_file_path, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"从缓存加载股票 {symbol} 数据")
            return cached_data
        except Exception:
            # If cache loading fails, continue to fetch fresh data
            pass

    print(f"从网络获取股票 {symbol} 数据")
    try:
        # 验证股票代码格式
        if not symbol or not symbol.strip():
            print("错误：股票代码不能为空")
            return pd.DataFrame()

        symbol = symbol.strip()

        # 获取前复权数据
        params = {}
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date

        stock_df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            adjust=adjust,
            timeout=30,  # 设置超时时间
            **params
        )

        # 检查返回数据是否为空
        if stock_df is None or stock_df.empty:
            print(f"警告：股票 {symbol} 没有返回任何数据")
            return pd.DataFrame()

        # 确保列名是英文的
        if '日期' in stock_df.columns:
            stock_df = stock_df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            })

        # 检查必要的列是否存在
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in stock_df.columns]
        if missing_columns:
            print(f"警告：股票 {symbol} 数据缺少列: {missing_columns}")
            return pd.DataFrame()

        # 将日期列转换为datetime类型
        if 'date' in stock_df.columns:
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            stock_df.set_index('date', inplace=True)

        # 确保数据按日期排序
        stock_df = stock_df.sort_index()

        # 移除任何包含NaN的行
        stock_df = stock_df.dropna()

        # 验证数据完整性
        if len(stock_df) < 10:  # 至少需要10条记录
            print(f"警告：股票 {symbol} 数据记录太少 ({len(stock_df)} 条)")
            return pd.DataFrame()

        # Cache the successful result
        try:
            with open(cache_file_path, 'wb') as f:
                pickle.dump(stock_df, f)
            print(f"股票 {symbol} 数据已缓存")
        except Exception as e:
            print(f"缓存数据时出错: {str(e)}")

        return stock_df
    except Exception as e:
        print(f"获取股票 {symbol} 数据时出错: {str(e)}")
        return pd.DataFrame()


def get_stock_info(symbol: str) -> dict:
    """
    获取股票基本信息

    Args:
        symbol: 股票代码

    Returns:
        dict: 股票基本信息
    """
    # Generate cache key for stock info
    cache_key = _get_cache_key(symbol, "info", None, None, "")
    cache_file_path = _get_cache_file_path(cache_key)

    # Check if cached data exists and is still valid (with shorter expiry for info)
    if _is_cache_valid(cache_file_path, max_age_hours=6):  # 6 hours for info
        try:
            with open(cache_file_path, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"从缓存加载股票 {symbol} 信息")
            return cached_data
        except Exception:
            # If cache loading fails, continue to fetch fresh data
            pass

    try:
        # 验证股票代码
        if not symbol or not symbol.strip():
            print("错误：股票代码不能为空")
            return {}

        symbol = symbol.strip()

        # 获取股票信息
        stock_info = ak.stock_individual_info_em(symbol=symbol)

        if stock_info is None or stock_info.empty:
            print(f"警告：无法获取股票 {symbol} 的基本信息")
            return {}

        # 转换为字典
        info_dict = dict(zip(stock_info['item'], stock_info['value']))

        # 确保返回的是基本Python类型
        result = {}
        for key, value in info_dict.items():
            if isinstance(value, (pd.Timestamp, pd.DatetimeIndex)):
                result[key] = str(value)
            elif pd.isna(value):
                result[key] = None
            else:
                result[key] = str(value)

        # Cache the successful result
        try:
            with open(cache_file_path, 'wb') as f:
                pickle.dump(result, f)
            print(f"股票 {symbol} 信息已缓存")
        except Exception as e:
            print(f"缓存信息时出错: {str(e)}")

        return result
    except Exception as e:
        print(f"获取股票 {symbol} 信息时出错: {str(e)}")
        return {}


if __name__ == "__main__":
    # 测试代码
    symbol = "002607"
    print(f"获取股票 {symbol} 的数据...")
    
    # 获取股票基本信息
    info = get_stock_info(symbol)
    print("股票基本信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 获取股票历史数据
    data = get_stock_data(symbol, start_date="20230101", end_date="20241231", adjust="qfq")
    print(f"\n股票历史数据 (最近5行):")
    print(data.tail())