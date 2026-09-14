import akshare as ak
import pandas as pd
from typing import Optional, Callable, Any, List, Tuple
import hashlib
import joblib
import os
import random
import threading
import time
from datetime import datetime, timedelta


# Cache directory
CACHE_DIR = ".data_cache"
MAX_RETRIES = 4
RETRY_BASE_DELAY_SECONDS = 1.5
MIN_REQUEST_INTERVAL_SECONDS = 1.2

# Serialize outbound network calls and keep a minimum gap between them.
# East Money (and similar) endpoints frequently drop connections when hit too fast.
_request_lock = threading.Lock()
_last_request_at = 0.0


def _get_cache_key(symbol: str, period: str, start_date: Optional[str], end_date: Optional[str], adjust: str) -> str:
    """Generate a cache key based on parameters."""
    cache_str = f"{symbol}_{period}_{start_date}_{end_date}_{adjust}"
    return hashlib.md5(cache_str.encode()).hexdigest()


def _get_cache_file_path(cache_key: str) -> str:
    """Get the full path for the cache file."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"{cache_key}.joblib")


def _is_cache_valid(file_path: str, max_age_hours: int = 24) -> bool:
    """Check if cache file exists and is not older than max_age_hours."""
    if not os.path.exists(file_path):
        return False

    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    return (datetime.now() - file_time).total_seconds() < max_age_hours * 3600


def _to_market_symbol(symbol: str) -> str:
    """Convert bare A-share codes to sina/tencent style market symbols (sh/sz)."""
    symbol = symbol.strip().lower()
    if symbol.startswith(("sh", "sz")):
        return symbol
    # Shanghai: 60xxxx / 68xxxx (STAR) / 90xxxx B-shares; funds often 5xxxxx
    if symbol.startswith(("5", "6", "9")):
        return f"sh{symbol}"
    return f"sz{symbol}"


def _throttle_requests() -> None:
    """Ensure a minimum interval between outbound data-source requests."""
    global _last_request_at
    with _request_lock:
        now = time.monotonic()
        wait = MIN_REQUEST_INTERVAL_SECONDS - (now - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _retry_request(fetch_fn: Callable[[], Any], label: str, max_retries: int = MAX_RETRIES):
    """
    Execute a network fetch with throttling and exponential backoff.

    Retries on exceptions and empty DataFrame responses (transient upstream failures).
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            _throttle_requests()
            result = fetch_fn()
            if isinstance(result, pd.DataFrame) and result.empty:
                raise ValueError("empty dataframe response")
            return result
        except Exception as e:
            last_error = e
            if attempt >= max_retries:
                break
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.6)
            print(f"{label} 第 {attempt} 次请求失败 ({e})，{delay:.1f}s 后重试...")
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    return pd.DataFrame()


def _normalize_ohlcv(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Chinese/English OHLCV frames to a dated index with English columns."""
    if stock_df is None or stock_df.empty:
        return pd.DataFrame()

    stock_df = stock_df.copy()

    if "日期" in stock_df.columns:
        stock_df = stock_df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        })

    # Tencent hist has amount but no volume — approximate share volume from amount/close.
    if "volume" not in stock_df.columns and "amount" in stock_df.columns and "close" in stock_df.columns:
        close = stock_df["close"].replace(0, pd.NA)
        stock_df["volume"] = (stock_df["amount"] / close).fillna(0)

    required_columns = ["close", "open", "high", "low", "volume"]
    missing_columns = [col for col in required_columns if col not in stock_df.columns]
    if missing_columns:
        raise ValueError(f"missing columns: {missing_columns}")

    if "date" in stock_df.columns:
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        stock_df.set_index("date", inplace=True)
    elif not isinstance(stock_df.index, pd.DatetimeIndex):
        stock_df.index = pd.to_datetime(stock_df.index)

    stock_df = stock_df.sort_index().dropna(subset=required_columns)
    return stock_df


def _stock_data_sources(
    symbol: str,
    period: str,
    start_date: Optional[str],
    end_date: Optional[str],
    adjust: str,
) -> List[Tuple[str, Callable[[], pd.DataFrame]]]:
    """
    Ordered data-source fallbacks.

    East Money is preferred when available, but it frequently rate-limits
    (RemoteDisconnected after the first successful call). Sina / Tencent are
    used as backups so repeated fetches keep working.
    """
    market_symbol = _to_market_symbol(symbol)
    resolved_end = end_date or datetime.now().strftime("%Y%m%d")
    resolved_start = start_date or "19700101"

    def _fetch_eastmoney():
        params = {}
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        return ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            adjust=adjust,
            timeout=30,
            **params,
        )

    def _fetch_sina():
        # Sina daily API uses market-prefixed symbols and YYYYMMDD dates.
        return ak.stock_zh_a_daily(
            symbol=market_symbol,
            start_date=resolved_start,
            end_date=resolved_end,
            adjust=adjust,
        )

    def _fetch_tencent():
        return ak.stock_zh_a_hist_tx(
            symbol=market_symbol,
            start_date=resolved_start,
            end_date=resolved_end,
            adjust=adjust,
            timeout=30,
        )

    sources: List[Tuple[str, Callable[[], pd.DataFrame]]] = [
        ("eastmoney", _fetch_eastmoney),
    ]
    # Sina / Tencent daily APIs are only valid fallbacks for daily bars.
    # Weekly/monthly still rely on East Money (with retry/throttle).
    if period == "daily":
        sources.append(("sina", _fetch_sina))
        sources.append(("tencent", _fetch_tencent))
    return sources


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
                cached_data = joblib.load(f)
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
        last_error = None
        stock_df = None

        for source_name, fetch_fn in _stock_data_sources(symbol, period, start_date, end_date, adjust):
            try:
                # Fewer retries per source; multi-source fallback recovers faster than
                # hammering a single rate-limited endpoint.
                raw = _retry_request(fetch_fn, f"股票 {symbol}/{source_name}", max_retries=2)
                stock_df = _normalize_ohlcv(raw)
                if stock_df.empty:
                    raise ValueError("normalized dataframe is empty")
                print(f"股票 {symbol} 数据来源: {source_name}")
                break
            except Exception as e:
                last_error = e
                print(f"股票 {symbol} 数据源 {source_name} 失败: {e}")
                stock_df = None
                continue

        if stock_df is None or stock_df.empty:
            if last_error:
                raise last_error
            print(f"警告：股票 {symbol} 没有返回任何数据")
            return pd.DataFrame()

        # 验证数据完整性
        if len(stock_df) < 10:  # 至少需要10条记录
            print(f"警告：股票 {symbol} 数据记录太少 ({len(stock_df)} 条)")
            return pd.DataFrame()

        # Cache the successful result
        try:
            with open(cache_file_path, 'wb') as f:
                joblib.dump(stock_df, f)
            print(f"股票 {symbol} 数据已缓存")
        except Exception as e:
            print(f"缓存数据时出错: {str(e)}")

        return stock_df
    except Exception as e:
        print(f"获取股票 {symbol} 数据时出错: {str(e)}")
        return pd.DataFrame()


def get_index_data(symbol: str = "000001", period: str = "daily",
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
    """
    获取A股指数历史数据（使用指数专用接口，避免与个股代码混淆）。

    Args:
        symbol: 指数代码，上证指数为 "000001" 或 "sh000001"
        period: 数据周期 ("daily", "weekly", "monthly")
        start_date: 开始日期 (格式: "YYYYMMDD")
        end_date: 结束日期 (格式: "YYYYMMDD")

    Returns:
        pd.DataFrame: 指数数据，索引为日期，含 close/open/high/low/volume
    """
    cache_key = _get_cache_key(f"index_{symbol}", period, start_date, end_date, "index")
    cache_file_path = _get_cache_file_path(cache_key)

    if _is_cache_valid(cache_file_path):
        try:
            with open(cache_file_path, 'rb') as f:
                return joblib.load(f)
        except Exception:
            pass

    # Normalize aliases. Prefer sina-style codes for stock_zh_index_daily.
    symbol_aliases = {
        "000001": "sh000001",
        "szzs": "sh000001",
        "上证指数": "sh000001",
        "sh000001": "sh000001",
    }
    sina_symbol = symbol_aliases.get(symbol, symbol)
    if not sina_symbol.startswith(("sh", "sz")):
        # Default unknown numeric codes to Shanghai prefix for index fetch
        sina_symbol = f"sh{sina_symbol}" if sina_symbol.isdigit() else sina_symbol

    em_symbol = sina_symbol[2:] if sina_symbol.startswith(("sh", "sz")) else sina_symbol

    def _normalize_index_frame(index_df: pd.DataFrame) -> pd.DataFrame:
        if '日期' in index_df.columns:
            index_df = index_df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
            })
        if 'date' in index_df.columns:
            index_df['date'] = pd.to_datetime(index_df['date'])
            index_df.set_index('date', inplace=True)
        elif not isinstance(index_df.index, pd.DatetimeIndex):
            index_df.index = pd.to_datetime(index_df.index)

        required_columns = ['close', 'open', 'high', 'low']
        missing = [c for c in required_columns if c not in index_df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")

        index_df = index_df.sort_index().dropna(subset=['close'])

        if start_date:
            index_df = index_df[index_df.index >= pd.to_datetime(start_date)]
        if end_date:
            index_df = index_df[index_df.index <= pd.to_datetime(end_date)]
        return index_df

    def _fetch_sina():
        return ak.stock_zh_index_daily(symbol=sina_symbol)

    def _fetch_em():
        params = {"symbol": em_symbol, "period": period}
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        return ak.index_zh_a_hist(**params)

    try:
        index_df = None
        last_error = None
        for fetch_fn, label in ((_fetch_sina, "sina"), (_fetch_em, "eastmoney")):
            try:
                raw = _retry_request(fetch_fn, f"指数 {symbol}/{label}", max_retries=2)
                index_df = _normalize_index_frame(raw)
                if not index_df.empty:
                    break
            except Exception as e:
                last_error = e
                continue

        if index_df is None or index_df.empty:
            if last_error:
                raise last_error
            print(f"警告：指数 {symbol} 没有返回任何数据")
            return pd.DataFrame()

        try:
            with open(cache_file_path, 'wb') as f:
                joblib.dump(index_df, f)
            print(f"指数 {symbol} 数据已缓存")
        except Exception as e:
            print(f"缓存指数数据时出错: {str(e)}")

        return index_df
    except Exception as e:
        print(f"获取指数 {symbol} 数据时出错: {str(e)}")
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
                cached_data = joblib.load(f)
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
        def _fetch_info_once():
            return ak.stock_individual_info_em(symbol=symbol)

        stock_info = _retry_request(_fetch_info_once, f"股票信息 {symbol}", max_retries=3)

        if stock_info is None or (isinstance(stock_info, pd.DataFrame) and stock_info.empty):
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
                joblib.dump(result, f)
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
