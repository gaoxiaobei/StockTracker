#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险评估模块

本模块提供了股票风险评估的各种指标计算函数，用于量化投资风险。
包括波动率、VaR、最大回撤、夏普比率、贝塔系数、Alpha值等指标。
"""

import pandas as pd
import numpy as np
import warnings
from typing import Union, Optional, Dict, Any
from scipy import stats
from data.fetcher import get_stock_data as data_fetcher


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    计算波动率（标准差）
    
    Args:
        returns: 收益率序列
        annualize: 是否年化，默认为True
        
    Returns:
        float: 波动率
    """
    if len(returns) < 2:
        raise ValueError("收益率序列长度必须大于等于2")
    
    volatility = float(returns.std())
    
    # 年化波动率（假设252个交易日）
    if annualize:
        volatility = volatility * np.sqrt(252)
    
    return volatility


def calculate_var_historical(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    计算历史模拟法VaR（风险价值）
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平，默认为0.05（95%置信度）
        
    Returns:
        float: VaR值
    """
    if len(returns) == 0:
        raise ValueError("收益率序列不能为空")
    
    if not 0 < confidence_level < 1:
        raise ValueError("置信水平必须在0到1之间")
    
    # 计算分位数
    var = float(returns.quantile(confidence_level))
    
    return var


def calculate_var_parametric(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    计算参数法VaR（风险价值）
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平，默认为0.05（95%置信度）
        
    Returns:
        float: VaR值
    """
    if len(returns) < 2:
        raise ValueError("收益率序列长度必须大于等于2")
    
    if not 0 < confidence_level < 1:
        raise ValueError("置信水平必须在0到1之间")
    
    # 计算收益率的均值和标准差
    mean_return = returns.mean()
    std_return = returns.std()
    
    # 计算置信水平对应的Z值
    z_score = stats.norm.ppf(confidence_level)
    
    # 计算参数法VaR
    var = float(mean_return + z_score * std_return)
    
    return var


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    计算最大回撤（Max Drawdown）
    
    Args:
        prices: 价格序列
        
    Returns:
        float: 最大回撤值
    """
    if len(prices) < 2:
        raise ValueError("价格序列长度必须大于等于2")
    
    # 计算累积最大值
    cumulative_max = prices.expanding().max()
    
    # 计算回撤
    drawdown = (prices - cumulative_max) / cumulative_max
    
    # 计算最大回撤
    max_drawdown = float(drawdown.min())
    
    return abs(max_drawdown)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率，默认为0.02（2%）
        
    Returns:
        float: 夏普比率
    """
    if len(returns) < 2:
        raise ValueError("收益率序列长度必须大于等于2")
    
    # 计算超额收益
    excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
    
    # 计算夏普比率
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe_ratio = float(excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    return sharpe_ratio


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    计算贝塔系数（相对于市场指数）
    
    Args:
        stock_returns: 股票收益率序列
        market_returns: 市场指数收益率序列
        
    Returns:
        float: 贝塔系数
    """
    if len(stock_returns) != len(market_returns):
        raise ValueError("股票收益率和市场收益率序列长度必须相同")
    
    if len(stock_returns) < 2:
        raise ValueError("收益率序列长度必须大于等于2")
    
    # 计算协方差和市场方差
    covariance = np.cov(stock_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    
    # 计算贝塔系数
    if market_variance == 0:
        return 0.0
    
    beta = covariance / market_variance
    
    return float(beta)


def calculate_alpha(stock_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    计算Alpha值
    
    Args:
        stock_returns: 股票收益率序列
        market_returns: 市场指数收益率序列
        risk_free_rate: 无风险利率，默认为0.02（2%）
        
    Returns:
        float: Alpha值
    """
    if len(stock_returns) != len(market_returns):
        raise ValueError("股票收益率和市场收益率序列长度必须相同")
    
    if len(stock_returns) < 2:
        raise ValueError("收益率序列长度必须大于等于2")
    
    # 计算贝塔系数
    beta = calculate_beta(stock_returns, market_returns)
    
    # 计算年化收益率
    stock_return_annual = (1 + stock_returns.mean()) ** 252 - 1
    market_return_annual = (1 + market_returns.mean()) ** 252 - 1
    
    # 计算Alpha值
    alpha = stock_return_annual - (risk_free_rate + beta * (market_return_annual - risk_free_rate))
    
    return alpha


def calculate_correlation(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    计算股票与市场指数的相关性
    
    Args:
        stock_returns: 股票收益率序列
        market_returns: 市场指数收益率序列
        
    Returns:
        float: 相关性系数
    """
    if len(stock_returns) != len(market_returns):
        raise ValueError("股票收益率和市场收益率序列长度必须相同")
    
    if len(stock_returns) < 2:
        raise ValueError("收益率序列长度必须大于等于2")
    
    # 计算相关性系数
    correlation = float(np.corrcoef(stock_returns, market_returns)[0, 1])
    
    return correlation


def get_stock_returns(stock_data: pd.DataFrame, column: str = 'close') -> pd.Series:
    """
    计算股票收益率序列
    
    Args:
        stock_data: 股票数据DataFrame
        column: 用于计算收益率的列名，默认为'close'
        
    Returns:
        pd.Series: 收益率序列
    """
    if column not in stock_data.columns:
        raise ValueError(f"列 '{column}' 不存在于数据中")
    
    if len(stock_data) < 2:
        raise ValueError("股票数据长度必须大于等于2")
    
    # 计算收益率
    returns = stock_data[column].pct_change().dropna()
    
    return returns


def get_market_returns(market_symbol: str = "sh000001", start_date: Optional[str] = None) -> pd.Series:
    """
    获取市场指数收益率序列
    
    Args:
        market_symbol: 市场指数代码，默认为上证指数"sh000001"
        start_date: 开始日期
        
    Returns:
        pd.Series: 市场指数收益率序列
    """
    # 获取市场指数数据
    # 尝试不同的市场指数格式
    market_data = pd.DataFrame()  # 初始化为空DataFrame
    market_symbols_to_try = [market_symbol]
    
    # 如果是上证指数，尝试不同的格式
    if market_symbol == "sh000001":
        market_symbols_to_try = ["000001", "sh000001", "szzs"]
    
    for symbol in market_symbols_to_try:
        try:
            market_data = data_fetcher(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                adjust="qfq"
            )
            if not market_data.empty:
                break
        except Exception as e:
            continue
    
    if market_data.empty:
        raise ValueError(f"无法获取市场指数 {market_symbol} 的数据")
    
    # 计算市场指数收益率
    market_returns = get_stock_returns(market_data, 'close')
    
    return market_returns


def assess_risk_level(volatility: float, max_drawdown: float, sharpe_ratio: float,
                     beta: float, var: float) -> Dict[str, Union[str, float, tuple]]:
    """
    根据计算的风险指标对股票进行风险评级
    
    Args:
        volatility: 波动率
        max_drawdown: 最大回撤
        sharpe_ratio: 夏普比率
        beta: 贝塔系数
        var: 风险价值
        
    Returns:
        dict: 风险评级和解释
    """
    # 风险评级标准
    risk_level = "中等"
    risk_explanation = []
    
    # 根据波动率评估风险
    if volatility > 0.3:
        risk_level = "高"
        risk_explanation.append("波动率较高，价格波动剧烈")
    elif volatility < 0.15:
        risk_level = "低"
        risk_explanation.append("波动率较低，价格相对稳定")
    else:
        risk_explanation.append("波动率适中")
    
    # 根据最大回撤评估风险
    if max_drawdown > 0.3:
        if risk_level != "高":
            risk_level = "高"
        risk_explanation.append("最大回撤较大，存在较大下行风险")
    elif max_drawdown < 0.1:
        if risk_level == "高":
            risk_level = "中等"
        risk_explanation.append("最大回撤较小，下行风险有限")
    
    # 根据夏普比率评估风险调整后收益
    if sharpe_ratio < 0:
        if risk_level != "高":
            risk_level = "高"
        risk_explanation.append("夏普比率为负，风险调整后收益不佳")
    elif sharpe_ratio > 1:
        if risk_level == "高":
            risk_level = "中等"
        risk_explanation.append("夏普比率较高，风险调整后收益良好")
    
    # 根据贝塔系数评估系统性风险
    if beta > 1.2:
        if risk_level != "高":
            risk_level = "高"
        risk_explanation.append("贝塔系数大于1.2，系统性风险较高")
    elif beta < 0.8:
        if risk_level == "高":
            risk_level = "中等"
        risk_explanation.append("贝塔系数小于0.8，系统性风险较低")
    
    # 根据VaR评估极端风险
    if var < -0.05:
        if risk_level != "高":
            risk_level = "高"
        risk_explanation.append("VaR值较低，存在较大的极端损失风险")
    elif var > -0.02:
        if risk_level == "高":
            risk_level = "中等"
        risk_explanation.append("VaR值较高，极端损失风险相对较小")
    
    # 提供投资建议
    if risk_level == "低":
        investment_advice = "该股票风险较低，适合风险厌恶型投资者"
    elif risk_level == "中等":
        investment_advice = "该股票风险适中，适合平衡型投资者"
    else:
        investment_advice = "该股票风险较高，适合风险偏好型投资者"
    
    return {
        "risk_level": risk_level,
        "risk_score": (volatility, max_drawdown, sharpe_ratio, beta, var),
        "explanation": "; ".join(risk_explanation),
        "investment_advice": investment_advice
    }


def monte_carlo_simulation(returns: pd.Series, num_simulations: int = 1000, 
                          time_horizon: int = 252) -> Dict[str, Union[np.ndarray, float]]:
    """
    蒙特卡洛模拟预测不同风险情景下的潜在损失
    
    Args:
        returns: 历史收益率序列
        num_simulations: 模拟次数，默认为1000
        time_horizon: 时间范围（交易日），默认为252（一年）
        
    Returns:
        dict: 模拟结果，包括置信区间估计
    """
    if len(returns) < 2:
        raise ValueError("收益率序列长度必须大于等于2")
    
    if num_simulations <= 0:
        raise ValueError("模拟次数必须大于0")
    
    if time_horizon <= 0:
        raise ValueError("时间范围必须大于0")
    
    # 计算收益率的均值和标准差
    mean_return = returns.mean()
    std_return = returns.std()
    
    # 进行蒙特卡洛模拟
    simulation_results = []
    
    for _ in range(num_simulations):
        # 生成随机收益率序列
        random_returns = np.random.normal(mean_return, std_return, time_horizon)
        # 计算累积收益
        cumulative_returns = np.cumprod(1 + random_returns)
        # 计算最终损失（相对于初始值）
        final_loss = 1 - cumulative_returns[-1]
        simulation_results.append(final_loss)
    
    # 转换为numpy数组
    simulation_results = np.array(simulation_results)
    
    # 计算置信区间
    var_95 = np.percentile(simulation_results, 95)
    var_99 = np.percentile(simulation_results, 99)
    expected_loss = np.mean(simulation_results)
    
    # 计算其他统计量
    min_loss = np.min(simulation_results)
    max_loss = np.max(simulation_results)
    std_loss = np.std(simulation_results)
    
    return {
        "simulation_results": simulation_results,
        "var_95": float(var_95),
        "var_99": float(var_99),
        "expected_loss": float(expected_loss),
        "min_loss": float(min_loss),
        "max_loss": float(max_loss),
        "std_loss": float(std_loss)
    }


def calculate_stock_correlations(stock_symbols: list, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    计算多个股票间的相关性矩阵
    
    Args:
        stock_symbols: 股票代码列表
        start_date: 开始日期
        
    Returns:
        pd.DataFrame: 相关性矩阵
    """
    if len(stock_symbols) < 2:
        raise ValueError("股票代码列表长度必须大于等于2")
    
    # 获取所有股票数据
    stock_data_dict = {}
    for symbol in stock_symbols:
        try:
            stock_data = data_fetcher(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                adjust="qfq"
            )
            if not stock_data.empty:
                stock_data_dict[symbol] = stock_data
        except Exception as e:
            warnings.warn(f"无法获取股票 {symbol} 的数据: {e}")
    
    if len(stock_data_dict) < 2:
        raise ValueError("有效股票数据不足，无法计算相关性")
    
    # 计算各股票收益率
    returns_dict = {}
    for symbol, data in stock_data_dict.items():
        try:
            returns = get_stock_returns(data, 'close')
            returns_dict[symbol] = returns
        except Exception as e:
            warnings.warn(f"无法计算股票 {symbol} 的收益率: {e}")
    
    if len(returns_dict) < 2:
        raise ValueError("有效收益率数据不足，无法计算相关性")
    
    # 对齐日期索引
    common_dates = set()
    is_first = True
    for symbol, returns in returns_dict.items():
        if is_first:
            common_dates = set(returns.index)
            is_first = False
        else:
            common_dates = common_dates.intersection(set(returns.index))
    
    if len(common_dates) < 2:
        raise ValueError("共同交易日期不足，无法计算相关性")
    
    common_dates_list = sorted(list(common_dates))
    
    # 构建收益率DataFrame
    returns_df = pd.DataFrame()
    for symbol, returns in returns_dict.items():
        returns_df[symbol] = returns.loc[common_dates]
    
    # 计算相关性矩阵
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix


def comprehensive_risk_assessment(stock_symbol: str, market_symbol: str = "sh000001",
                                 start_date: Optional[str] = None) -> Dict[str, Any]:
    """
    综合风险评估
    
    Args:
        stock_symbol: 股票代码
        market_symbol: 市场指数代码，默认为上证指数
        start_date: 开始日期
        
    Returns:
        dict: 包含所有风险指标的综合评估结果
    """
    # 获取股票数据
    stock_data = data_fetcher(
        symbol=stock_symbol,
        period="daily",
        start_date=start_date,
        adjust="qfq"
    )
    
    if stock_data.empty:
        raise ValueError(f"无法获取股票 {stock_symbol} 的数据")
    
    # 获取市场指数数据
    # 尝试不同的市场指数格式
    market_data = pd.DataFrame()  # 初始化为空DataFrame
    market_symbols_to_try = [market_symbol]
    
    # 如果是上证指数，尝试不同的格式
    if market_symbol == "sh000001":
        market_symbols_to_try = ["000001", "sh000001", "szzs"]
    
    for symbol in market_symbols_to_try:
        try:
            market_data = data_fetcher(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                adjust="qfq"
            )
            if not market_data.empty:
                break
        except Exception as e:
            continue
    
    if market_data.empty:
        raise ValueError(f"无法获取市场指数 {market_symbol} 的数据")
    
    # 计算收益率序列
    stock_returns = get_stock_returns(stock_data, 'close')
    market_returns = get_stock_returns(market_data, 'close')
    
    # 对齐收益率序列的日期
    common_dates = list(set(stock_returns.index) & set(market_returns.index))
    common_dates = sorted(common_dates)
    if len(common_dates) == 0:
        raise ValueError("股票和市场指数数据没有共同的交易日期")
    stock_returns = stock_returns.loc[common_dates]
    market_returns = market_returns.loc[common_dates]
    
    if len(stock_returns) < 2 or len(market_returns) < 2:
        raise ValueError("共同交易日期不足，无法进行风险评估")
    
    # 计算各项风险指标
    volatility = calculate_volatility(stock_returns)
    var_historical = calculate_var_historical(stock_returns)
    var_parametric = calculate_var_parametric(stock_returns)
    max_drawdown = calculate_max_drawdown(stock_data['close'])
    sharpe_ratio = calculate_sharpe_ratio(stock_returns)
    beta = calculate_beta(stock_returns, market_returns)
    alpha = calculate_alpha(stock_returns, market_returns)
    correlation = calculate_correlation(stock_returns, market_returns)
    
    # 风险评级
    risk_assessment = assess_risk_level(volatility, max_drawdown, sharpe_ratio, beta, var_historical)
    
    # 蒙特卡洛模拟
    mc_results = monte_carlo_simulation(stock_returns)
    
    return {
        "stock_symbol": stock_symbol,
        "market_symbol": market_symbol,
        "volatility": volatility,
        "var_historical": var_historical,
        "var_parametric": var_parametric,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "beta": beta,
        "alpha": alpha,
        "correlation_with_market": correlation,
        "risk_level": risk_assessment,
        "monte_carlo_simulation": mc_results,
        "data_points": len(stock_returns)
    }


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    test_prices = pd.Series([100, 102, 98, 105, 103, 107, 104, 108, 106, 110])
    test_returns = test_prices.pct_change().dropna()
    
    print("测试风险指标计算:")
    print("测试价格序列:")
    print(test_prices.values)
    print("测试收益率序列:")
    print(test_returns.values)
    
    # 测试波动率
    volatility = calculate_volatility(test_returns, annualize=False)
    print(f"\n波动率: {volatility:.4f}")
    
    # 测试历史VaR
    var_hist = calculate_var_historical(test_returns)
    print(f"历史VaR (95%置信度): {var_hist:.4f}")
    
    # 测试参数VaR
    var_param = calculate_var_parametric(test_returns)
    print(f"参数VaR (95%置信度): {var_param:.4f}")
    
    # 测试最大回撤
    max_dd = calculate_max_drawdown(test_prices)
    print(f"最大回撤: {max_dd:.4f}")
    
    # 测试夏普比率
    sharpe = calculate_sharpe_ratio(test_returns)
    print(f"夏普比率: {sharpe:.4f}")
    
    # 创建测试市场数据
    market_prices = pd.Series([1000, 1010, 990, 1020, 1005, 1030, 1015, 1035, 1025, 1040])
    market_returns = market_prices.pct_change().dropna()
    
    # 测试贝塔系数
    beta = calculate_beta(test_returns, market_returns)
    print(f"贝塔系数: {beta:.4f}")
    
    # 测试Alpha值
    alpha = calculate_alpha(test_returns, market_returns)
    print(f"Alpha值: {alpha:.4f}")
    
    # 测试相关性
    corr = calculate_correlation(test_returns, market_returns)
    print(f"相关性: {corr:.4f}")
    
    # 测试风险评级
    risk_level = assess_risk_level(volatility, max_dd, sharpe, beta, var_hist)
    print(f"\n风险评级: {risk_level['risk_level']}")
    print(f"风险解释: {risk_level['explanation']}")
    print(f"投资建议: {risk_level['investment_advice']}")