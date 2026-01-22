#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标计算模块

本模块提供了常用的技术指标计算函数，用于股票技术分析。
所有函数都接受pandas DataFrame作为输入，并返回计算结果。
"""

import pandas as pd
import numpy as np


def simple_moving_average(data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
    """
    计算简单移动平均线 (SMA)
    
    Args:
        data: 股票数据DataFrame，应包含指定的列
        period: 计算周期，默认为20
        column: 计算移动平均线的列名，默认为'close'
        
    Returns:
        pd.Series: 简单移动平均线值
        
    Raises:
        ValueError: 当输入数据不合法时抛出
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("数据必须是pandas DataFrame类型")
    
    if column not in data.columns:
        raise ValueError(f"列 '{column}' 不存在于数据中")
    
    if period <= 0 or period > len(data):
        raise ValueError("周期必须大于0且小于等于数据长度")
    
    return data[column].rolling(window=period).mean()


def exponential_moving_average(data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
    """
    计算指数移动平均线 (EMA)
    
    Args:
        data: 股票数据DataFrame，应包含指定的列
        period: 计算周期，默认为20
        column: 计算移动平均线的列名，默认为'close'
        
    Returns:
        pd.Series: 指数移动平均线值
        
    Raises:
        ValueError: 当输入数据不合法时抛出
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("数据必须是pandas DataFrame类型")
    
    if column not in data.columns:
        raise ValueError(f"列 '{column}' 不存在于数据中")
    
    if period <= 0 or period > len(data):
        raise ValueError("周期必须大于0且小于等于数据长度")
    
    return data[column].ewm(span=period, adjust=False).mean()


def relative_strength_index(data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    计算相对强弱指数 (RSI)
    
    Args:
        data: 股票数据DataFrame，应包含指定的列
        period: 计算周期，默认为14
        column: 计算RSI的列名，默认为'close'
        
    Returns:
        pd.Series: RSI值 (0-100)
        
    Raises:
        ValueError: 当输入数据不合法时抛出
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("数据必须是pandas DataFrame类型")
    
    if column not in data.columns:
        raise ValueError(f"列 '{column}' 不存在于数据中")
    
    if period <= 0 or period > len(data):
        raise ValueError("周期必须大于0且小于等于数据长度")
    
    # 计算价格变化
    delta = data[column].diff()
    
    # 分离上涨和下跌
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta).clip(lower=0).rolling(window=period).mean()
    
    # 计算相对强度RS
    rs = gain / loss
    
    # 计算RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def moving_average_convergence_divergence(
    data: pd.DataFrame, 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9,
    column: str = 'close'
) -> pd.DataFrame:
    """
    计算异同移动平均线 (MACD)
    
    Args:
        data: 股票数据DataFrame，应包含指定的列
        fast_period: 快速EMA周期，默认为12
        slow_period: 慢速EMA周期，默认为26
        signal_period: 信号线周期，默认为9
        column: 计算MACD的列名，默认为'close'
        
    Returns:
        pd.DataFrame: 包含MACD线、信号线和柱状图的DataFrame
        
    Raises:
        ValueError: 当输入数据不合法时抛出
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("数据必须是pandas DataFrame类型")
    
    if column not in data.columns:
        raise ValueError(f"列 '{column}' 不存在于数据中")
    
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("所有周期参数必须大于0")
    
    if fast_period >= slow_period:
        raise ValueError("快速周期必须小于慢速周期")
    
    # 计算快速和慢速EMA
    ema_fast = data[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data[column].ewm(span=slow_period, adjust=False).mean()
    
    # 计算MACD线
    macd_line = ema_fast - ema_slow
    
    # 计算信号线
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # 计算柱状图
    histogram = macd_line - signal_line
    
    # 返回结果DataFrame
    result = pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }, index=data.index)
    
    return result


def bollinger_bands(
    data: pd.DataFrame, 
    period: int = 20, 
    num_std: float = 2.0, 
    column: str = 'close'
) -> pd.DataFrame:
    """
    计算布林带 (Bollinger Bands)
    
    Args:
        data: 股票数据DataFrame，应包含指定的列
        period: 计算周期，默认为20
        num_std: 标准差倍数，默认为2.0
        column: 计算布林带的列名，默认为'close'
        
    Returns:
        pd.DataFrame: 包含上轨、中轨和下轨的DataFrame
        
    Raises:
        ValueError: 当输入数据不合法时抛出
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("数据必须是pandas DataFrame类型")
    
    if column not in data.columns:
        raise ValueError(f"列 '{column}' 不存在于数据中")
    
    if period <= 0 or period > len(data):
        raise ValueError("周期必须大于0且小于等于数据长度")
    
    if num_std <= 0:
        raise ValueError("标准差倍数必须大于0")
    
    # 计算中轨（移动平均线）
    middle_band = data[column].rolling(window=period).mean()
    
    # 计算标准差
    std_dev = data[column].rolling(window=period).std()
    
    # 计算上下轨
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    # 返回结果DataFrame
    result = pd.DataFrame({
        'upper_band': upper_band,
        'middle_band': middle_band,
        'lower_band': lower_band
    }, index=data.index)
    
    return result


def stochastic_oscillator(
    data: pd.DataFrame, 
    k_period: int = 14, 
    d_period: int = 3
) -> pd.DataFrame:
    """
    计算随机指标 (Stochastic Oscillator)
    
    Args:
        data: 股票数据DataFrame，应包含'high', 'low', 'close'列
        k_period: %K线计算周期，默认为14
        d_period: %D线计算周期，默认为3
        
    Returns:
        pd.DataFrame: 包含%K线和%D线的DataFrame
        
    Raises:
        ValueError: 当输入数据不合法时抛出
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("数据必须是pandas DataFrame类型")
    
    required_columns = ['high', 'low', 'close']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"列 '{col}' 不存在于数据中")
    
    if k_period <= 0 or d_period <= 0:
        raise ValueError("周期参数必须大于0")
    
    if k_period > len(data):
        raise ValueError("K周期必须小于等于数据长度")
    
    # 计算过去k_period天的最高价和最低价
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    
    # 计算%K值
    k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
    
    # 计算%D值（%K的移动平均）
    d_percent = k_percent.rolling(window=d_period).mean()
    
    # 返回结果DataFrame
    result = pd.DataFrame({
        'k_percent': k_percent,
        'd_percent': d_percent
    }, index=data.index)
    
    return result


def on_balance_volume(data: pd.DataFrame) -> pd.Series:
    """
    计算能量潮指标 (On Balance Volume, OBV)
    
    Args:
        data: 股票数据DataFrame，应包含'close'和'volume'列
        
    Returns:
        pd.Series: OBV值
        
    Raises:
        ValueError: 当输入数据不合法时抛出
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("数据必须是pandas DataFrame类型")
    
    required_columns = ['close', 'volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"列 '{col}' 不存在于数据中")
    
    # Vectorized OBV calculation
    price_change = data['close'].diff()
    direction = np.sign(price_change).fillna(0)
    
    # For the first row, OBV is just the volume
    # The direction of the first row is 0, so we need to set it correctly
    obv = (direction * data['volume']).cumsum()
    
    # Adjust for the first element being 0 after diff and sign
    obv.iloc[0] = float(data['volume'].iloc[0])
    # Recalculate cumsum starting with the first volume
    # Actually, the standard OBV starts with 0 or the first volume. 
    # Let's align with common implementations:
    
    obv = direction * data['volume']
    obv.iloc[0] = data['volume'].iloc[0]
    obv = obv.cumsum()
    
    return obv


def volume_weighted_average_price(
    data: pd.DataFrame, 
    period: int = 20
) -> pd.Series:
    """
    计算成交量加权平均价格 (Volume Weighted Average Price, VWAP)
    
    Args:
        data: 股票数据DataFrame，应包含'high', 'low', 'close', 'volume'列
        period: 计算周期，默认为20
        
    Returns:
        pd.Series: VWAP值
        
    Raises:
        ValueError: 当输入数据不合法时抛出
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("数据必须是pandas DataFrame类型")
    
    required_columns = ['high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"列 '{col}' 不存在于数据中")
    
    if period <= 0 or period > len(data):
        raise ValueError("周期必须大于0且小于等于数据长度")
    
    # 计算典型价格
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    
    # 计算VWAP
    vwap = (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
    
    return vwap


def chaikin_money_flow(
    data: pd.DataFrame, 
    period: int = 20
) -> pd.Series:
    """
    计算蔡金资金流量指标 (Chaikin Money Flow, CMF)
    
    Args:
        data: 股票数据DataFrame，应包含'high', 'low', 'close', 'volume'列
        period: 计算周期，默认为20
        
    Returns:
        pd.Series: CMF值
        
    Raises:
        ValueError: 当输入数据不合法时抛出
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("数据必须是pandas DataFrame类型")
    
    required_columns = ['high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"列 '{col}' 不存在于数据中")
    
    if period <= 0 or period > len(data):
        raise ValueError("周期必须大于0且小于等于数据长度")
    
    # 计算货币流量乘数
    money_flow_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    money_flow_multiplier = money_flow_multiplier.fillna(0)  # 处理除零情况
    
    # 计算货币流量体积
    money_flow_volume = money_flow_multiplier * data['volume']
    
    # 计算CMF
    cmf = money_flow_volume.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
    
    return cmf


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    print("测试技术指标计算:")
    print("测试数据:")
    print(test_data)
    
    # 测试简单移动平均线
    sma = simple_moving_average(test_data, period=5)
    print("\n简单移动平均线 (5日):")
    print(sma)
    
    # 测试指数移动平均线
    ema = exponential_moving_average(test_data, period=5)
    print("\n指数移动平均线 (5日):")
    print(ema)
    
    # 测试RSI
    rsi = relative_strength_index(test_data, period=5)
    print("\n相对强弱指数 (5日):")
    print(rsi)
    
    # 测试MACD
    macd = moving_average_convergence_divergence(test_data)
    print("\n异同移动平均线:")
    print(macd)
    
    # 测试布林带
    bb = bollinger_bands(test_data, period=5)
    print("\n布林带 (5日):")
    print(bb)
    
    # 测试随机指标
    stoch = stochastic_oscillator(test_data, k_period=5)
    print("\n随机指标 (5日):")
    print(stoch)
    
    # 测试OBV
    obv = on_balance_volume(test_data)
    print("\n能量潮指标:")
    print(obv)
    
    # 测试VWAP
    vwap = volume_weighted_average_price(test_data, period=5)
    print("\n成交量加权平均价格 (5日):")
    print(vwap)
    
    # 测试CMF
    cmf = chaikin_money_flow(test_data, period=5)
    print("\n蔡金资金流量指标 (5日):")
    print(cmf)