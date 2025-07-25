import akshare as ak
import pandas as pd
from typing import Optional


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
    try:
        # 验证股票代码格式
        if not symbol or not symbol.strip():
            print("错误：股票代码不能为空")
            return pd.DataFrame()
        
        symbol = symbol.strip()
        
        # 获取前复权数据
        stock_df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            adjust=adjust,
            timeout=30,  # 设置超时时间
            **{k: v for k, v in {"start_date": start_date, "end_date": end_date}.items() if v is not None}
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