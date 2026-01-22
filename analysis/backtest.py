#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测模块
实现股票交易策略的回测功能，包括策略框架、交易成本模拟、业绩评估等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import random
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """交易记录"""
    date: pd.Timestamp
    symbol: str
    direction: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """持仓记录"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float = 0.0


class BacktestEngine:
    """
    回测引擎
    实现策略回测的核心功能
    """
    
    def __init__(self, initial_capital: float = 1000000.0, 
                 commission_rate: float = 0.0003, 
                 slippage: float = 0.001):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 佣金费率
            slippage: 滑点比率
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # 回测状态
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[Tuple[pd.Timestamp, float]] = []
        self.dates: List[pd.Timestamp] = []
        
        # 性能指标
        self.returns: List[float] = []
        self.benchmark_returns: List[float] = []
        
        # 止损止盈设置
        self.stop_loss_limits: Dict[str, float] = {}
        self.take_profit_limits: Dict[str, float] = {}
    
    def reset(self):
        """重置回测引擎状态"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        self.returns = []
        self.benchmark_returns = []
        self.stop_loss_limits = {}
        self.take_profit_limits = {}
    
    def set_stop_loss(self, symbol: str, stop_loss_price: float):
        """
        设置止损价格
        
        Args:
            symbol: 股票代码
            stop_loss_price: 止损价格
        """
        self.stop_loss_limits[symbol] = stop_loss_price
    
    def set_take_profit(self, symbol: str, take_profit_price: float):
        """
        设置止盈价格
        
        Args:
            symbol: 股票代码
            take_profit_price: 止盈价格
        """
        self.take_profit_limits[symbol] = take_profit_price
    
    def buy(self, symbol: str, quantity: float, price: float, date: pd.Timestamp):
        """
        买入股票
        
        Args:
            symbol: 股票代码
            quantity: 数量
            price: 价格
            date: 交易日期
        """
        # 边界检查
        if quantity <= 0 or price <= 0:
            return  # 无效的交易参数
        
        # 计算实际交易价格（考虑滑点）
        actual_price = price * (1 + self.slippage)
        
        # 计算交易金额和佣金
        amount = quantity * actual_price
        commission = amount * self.commission_rate
        total_cost = amount + commission
        
        # 检查资金是否足够
        if total_cost > self.current_capital:
            # 如果资金不足，按最大可购买数量买入
            available_capital = self.current_capital / (1 + self.commission_rate)
            quantity = int(available_capital / (actual_price * 100)) * 100  # 按手计算
            if quantity <= 0:
                return  # 资金不足，无法买入
        
        # 更新资金
        amount = quantity * actual_price
        commission = amount * self.commission_rate
        total_cost = amount + commission
        self.current_capital -= total_cost
        
        # 更新持仓
        if symbol in self.positions:
            position = self.positions[symbol]
            total_quantity = position.quantity + quantity
            total_cost = position.avg_price * position.quantity + amount
            position.quantity = total_quantity
            position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
        else:
            self.positions[symbol] = Position(symbol, quantity, actual_price)
        
        # 记录交易
        trade = Trade(date, symbol, 'buy', quantity, actual_price, commission, self.slippage)
        self.trades.append(trade)
    
    def sell(self, symbol: str, quantity: float, price: float, date: pd.Timestamp):
        """
        卖出股票
        
        Args:
            symbol: 股票代码
            quantity: 数量
            price: 价格
            date: 交易日期
        """
        # 边界检查
        if quantity <= 0 or price <= 0:
            return  # 无效的交易参数
        
        # 检查是否有足够持仓
        if symbol not in self.positions or self.positions[symbol].quantity < quantity:
            return  # 持仓不足，无法卖出
        
        # 计算实际交易价格（考虑滑点）
        actual_price = price * (1 - self.slippage)
        
        # 计算交易金额和佣金
        amount = quantity * actual_price
        commission = amount * self.commission_rate
        total_gain = amount - commission
        
        # 更新资金
        self.current_capital += total_gain
        
        # 更新持仓
        position = self.positions[symbol]
        position.quantity -= quantity
        if position.quantity <= 0:
            del self.positions[symbol]
            # 移除止损止盈设置
            if symbol in self.stop_loss_limits:
                del self.stop_loss_limits[symbol]
            if symbol in self.take_profit_limits:
                del self.take_profit_limits[symbol]
        
        # 记录交易
        trade = Trade(date, symbol, 'sell', quantity, actual_price, commission, self.slippage)
        self.trades.append(trade)
    
    def check_stop_loss_take_profit(self, symbol: str, current_price: float, date: pd.Timestamp):
        """
        检查是否触发止损或止盈
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            date: 日期
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # 检查止损
        if symbol in self.stop_loss_limits and current_price <= self.stop_loss_limits[symbol]:
            # 触发止损，卖出所有持仓
            self.sell(symbol, position.quantity, current_price, date)
            return
        
        # 检查止盈
        if symbol in self.take_profit_limits and current_price >= self.take_profit_limits[symbol]:
            # 触发止盈，卖出所有持仓
            self.sell(symbol, position.quantity, current_price, date)
    
    def update_portfolio_value(self, date: pd.Timestamp, prices: Dict[str, float]):
        """
        更新投资组合价值
        
        Args:
            date: 日期
            prices: 股票价格字典
        """
        # 检查止损止盈
        for symbol, price in prices.items():
            self.check_stop_loss_take_profit(symbol, price, date)
        
        # 计算持仓市值
        total_value = self.current_capital
        for symbol, position in self.positions.items():
            if symbol in prices:
                market_price = prices[symbol] * (1 - self.slippage)  # 卖出价格
                position.market_value = position.quantity * market_price
                total_value += position.market_value
        
        # 记录投资组合价值
        self.portfolio_values.append((date, total_value))
        self.dates.append(date)
    
    def calculate_returns(self):
        """计算收益率序列"""
        if len(self.portfolio_values) < 2:
            return []
        
        values = [pv[1] for pv in self.portfolio_values]
        returns = []
        for i in range(1, len(values)):
            ret = (values[i] - values[i-1]) / values[i-1]
            returns.append(ret)
        
        self.returns = returns
        return returns
    
    def set_benchmark_returns(self, benchmark_returns: List[float]):
        """
        设置基准收益率序列
        
        Args:
            benchmark_returns: 基准收益率序列
        """
        self.benchmark_returns = benchmark_returns


class Strategy(ABC):
    """
    策略基类
    所有交易策略都需要继承此类并实现generate_signals方法
    """
    
    def __init__(self, name: str):
        """
        初始化策略
        
        Args:
            name: 策略名称
        """
        self.name = name
        self.signals: Dict[str, pd.Series] = {}
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 股票数据字典
            
        Returns:
            交易信号字典
        """
        pass


class MovingAverageCrossoverStrategy(Strategy):
    """
    移动平均线交叉策略
    当短期均线向上穿越长期均线时买入，向下穿越时卖出
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        初始化策略
        
        Args:
            short_window: 短期均线窗口
            long_window: 长期均线窗口
        """
        super().__init__(f"MA_Crossover_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 股票数据字典
            
        Returns:
            交易信号字典
        """
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < self.long_window:
                signals[symbol] = pd.Series(0, index=df.index)
                continue
            
            # 计算移动平均线
            short_ma = df['close'].rolling(window=self.short_window).mean()
            long_ma = df['close'].rolling(window=self.long_window).mean()
            
            # 生成信号
            signal = pd.Series(0, index=df.index)
            signal[short_ma > long_ma] = 1   # 买入信号
            signal[short_ma < long_ma] = -1  # 卖出信号
            
            # 处理前long_window个数据点
            signal.iloc[:self.long_window] = 0
            
            signals[symbol] = signal
        
        return signals


class RSIStrategy(Strategy):
    """
    RSI超买超卖策略
    当RSI低于超卖线时买入，高于超买线时卖出
    """
    
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        """
        初始化策略
        
        Args:
            period: RSI计算周期
            overbought: 超买线
            oversold: 超卖线
        """
        super().__init__(f"RSI_{period}_{overbought}_{oversold}")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 股票数据字典
            
        Returns:
            交易信号字典
        """
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < self.period:
                signals[symbol] = pd.Series(0, index=df.index)
                continue
            
            # 计算RSI
            delta = df['close'].diff()
            gain = delta.clip(lower=0).rolling(window=self.period).mean()
            loss = (-delta).clip(lower=0).rolling(window=self.period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 生成信号
            signal = pd.Series(0, index=df.index)
            signal[rsi < self.oversold] = 1   # 买入信号
            signal[rsi > self.overbought] = -1  # 卖出信号
            
            # 处理前period个数据点
            signal.iloc[:self.period] = 0
            
            signals[symbol] = signal
        
        return signals


class BollingerBandsStrategy(Strategy):
    """
    布林带策略
    当价格突破上轨时卖出，跌破下轨时买入
    """
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        """
        初始化策略
        
        Args:
            period: 布林带计算周期
            num_std: 标准差倍数
        """
        super().__init__(f"Bollinger_Bands_{period}_{num_std}")
        self.period = period
        self.num_std = num_std
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 股票数据字典
            
        Returns:
            交易信号字典
        """
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < self.period:
                signals[symbol] = pd.Series(0, index=df.index)
                continue
            
            # 计算布林带
            middle_band = df['close'].rolling(window=self.period).mean()
            std_dev = df['close'].rolling(window=self.period).std()
            upper_band = middle_band + (std_dev * self.num_std)
            lower_band = middle_band - (std_dev * self.num_std)
            
            # 生成信号
            signal = pd.Series(0, index=df.index)
            signal[df['close'] < lower_band] = 1   # 买入信号
            signal[df['close'] > upper_band] = -1  # 卖出信号
            
            # 处理前period个数据点
            signal.iloc[:self.period] = 0
            
            signals[symbol] = signal
        
        return signals


class MomentumStrategy(Strategy):
    """
    动量策略
    当价格持续上涨时买入，持续下跌时卖出
    """
    
    def __init__(self, period: int = 20):
        """
        初始化策略
        
        Args:
            period: 动量计算周期
        """
        super().__init__(f"Momentum_{period}")
        self.period = period
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 股票数据字典
            
        Returns:
            交易信号字典
        """
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < self.period:
                signals[symbol] = pd.Series(0, index=df.index)
                continue
            
            # 计算动量
            momentum = df['close'] / df['close'].shift(self.period) - 1
            
            # 生成信号（简单阈值）
            signal = pd.Series(0, index=df.index)
            signal[momentum > 0.05] = 1   # 买入信号（上涨超过5%）
            signal[momentum < -0.05] = -1  # 卖出信号（下跌超过5%）
            
            # 处理前period个数据点
            signal.iloc[:self.period] = 0
            
            signals[symbol] = signal
        
        return signals


class MeanReversionStrategy(Strategy):
    """
    均值回归策略
    当价格偏离均值过多时，预期会回归均值
    """
    
    def __init__(self, period: int = 20, threshold: float = 2.0):
        """
        初始化策略
        
        Args:
            period: 均值计算周期
            threshold: 标准差阈值
        """
        super().__init__(f"Mean_Reversion_{period}_{threshold}")
        self.period = period
        self.threshold = threshold
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 股票数据字典
            
        Returns:
            交易信号字典
        """
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < self.period:
                signals[symbol] = pd.Series(0, index=df.index)
                continue
            
            # 计算均值和标准差
            mean_price = df['close'].rolling(window=self.period).mean()
            std_price = df['close'].rolling(window=self.period).std()
            
            # 计算z-score
            z_score = (df['close'] - mean_price) / std_price
            
            # 生成信号
            signal = pd.Series(0, index=df.index)
            signal[z_score < -self.threshold] = 1   # 买入信号（价格低于均值过多）
            signal[z_score > self.threshold] = -1   # 卖出信号（价格高于均值过多）
            
            # 处理前period个数据点
            signal.iloc[:self.period] = 0
            
            signals[symbol] = signal
        
        return signals


def calculate_performance_metrics(returns: List[float], 
                                benchmark_returns: Optional[List[float]] = None,
                                risk_free_rate: float = 0.03) -> Dict[str, float]:
    """
    计算业绩评估指标
    
    Args:
        returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        risk_free_rate: 无风险利率
        
    Returns:
        业绩指标字典
    """
    if len(returns) == 0:
        return {}
    
    # 转换为numpy数组
    returns_np = np.array(returns)
    
    # 累计收益
    cumulative_return = np.prod(1 + returns_np) - 1
    
    # 年化收益（假设252个交易日）
    annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
    
    # 波动率（年化）
    volatility = np.std(returns_np) * np.sqrt(252)
    
    # 夏普比率
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
    
    # 最大回撤
    cumulative_values = np.cumprod(1 + returns_np)
    peak = np.maximum.accumulate(cumulative_values)
    drawdown = (cumulative_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    metrics = {
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': len(returns)
    }
    
    # 如果有基准收益率，计算额外指标
    if benchmark_returns and len(benchmark_returns) == len(returns):
        benchmark_returns_np = np.array(benchmark_returns)
        
        # Alpha和Beta
        if np.std(benchmark_returns_np) != 0:
            beta = np.cov(returns_np, benchmark_returns_np)[0, 1] / np.var(benchmark_returns_np)
        else:
            beta = 0
        
        alpha = annualized_return - (risk_free_rate + beta * (np.mean(benchmark_returns_np) * 252 - risk_free_rate))
        
        # 信息比率
        tracking_error = np.std(returns_np - benchmark_returns_np) * np.sqrt(252)
        information_ratio = (annualized_return - np.mean(benchmark_returns_np) * 252) / tracking_error if tracking_error != 0 else 0
        
        metrics.update({
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'correlation_with_benchmark': np.corrcoef(returns_np, benchmark_returns_np)[0, 1]
        })
    
    return metrics


def run_backtest(data: Dict[str, pd.DataFrame],
                strategy: Strategy,
                initial_capital: float = 1000000.0,
                commission_rate: float = 0.0003,
                slippage: float = 0.001,
                benchmark_data: Optional[pd.Series] = None) -> Dict:
    """
    运行回测
    
    Args:
        data: 股票数据字典
        strategy: 交易策略
        initial_capital: 初始资金
        commission_rate: 佣金费率
        slippage: 滑点比率
        benchmark_data: 基准数据
        
    Returns:
        回测结果字典
    """
    try:
        # 边界检查
        if not data:
            return {"error": "股票数据为空"}
        
        # 生成交易信号
        signals = strategy.generate_signals(data)
        
        # 初始化回测引擎
        engine = BacktestEngine(initial_capital, commission_rate, slippage)
        
        # 获取所有交易日期
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        sorted_dates = sorted(list(all_dates))
        
        # 按日期进行回测
        for date in sorted_dates:
            # 获取当日价格
            prices = {}
            for symbol, df in data.items():
                if date in df.index:
                    prices[symbol] = df.loc[date, 'close']
            
            # 执行交易信号
            for symbol, signal_series in signals.items():
                if symbol in prices and date in signal_series.index:
                    signal = signal_series.loc[date]
                    price = prices[symbol]
                    
                    if signal == 1:  # 买入信号
                        # 简单策略：买入1手（100股）
                        engine.buy(symbol, 100, price, date)
                    elif signal == -1:  # 卖出信号
                        # 卖出所有持仓
                        if symbol in engine.positions:
                            quantity = engine.positions[symbol].quantity
                            engine.sell(symbol, quantity, price, date)
            
            # 更新投资组合价值
            engine.update_portfolio_value(date, prices)
        
        # 计算收益率
        returns = engine.calculate_returns()
        
        # 设置基准收益率
        if benchmark_data is not None:
            benchmark_returns = []
            for i in range(1, len(benchmark_data)):
                ret = (benchmark_data.iloc[i] - benchmark_data.iloc[i-1]) / benchmark_data.iloc[i-1]
                benchmark_returns.append(ret)
            engine.set_benchmark_returns(benchmark_returns[:len(returns)])
        
        # 计算业绩指标
        metrics = calculate_performance_metrics(returns, engine.benchmark_returns)
        
        return {
            'engine': engine,
            'metrics': metrics,
            'signals': signals
        }
    except Exception as e:
        return {"error": f"回测过程中发生错误: {str(e)}"}


def plot_backtest_results(backtest_result: Dict, save_path: Optional[str] = None):
    """
    绘制回测结果图表
    
    Args:
        backtest_result: 回测结果
        save_path: 保存路径，如果为None则显示图表
    """
    engine = backtest_result['engine']
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制投资组合价值曲线
    dates = [pv[0] for pv in engine.portfolio_values]
    values = [pv[1] for pv in engine.portfolio_values]
    
    ax1.plot(dates, values, label='投资组合价值', linewidth=2)
    ax1.set_title('投资组合价值曲线')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('价值 (元)')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制收益率曲线
    if len(engine.returns) > 0:
        ax2.plot(dates[1:], engine.returns, label='策略收益率', alpha=0.7)
        if len(engine.benchmark_returns) > 0:
            ax2.plot(dates[1:len(engine.benchmark_returns)+1], engine.benchmark_returns, 
                    label='基准收益率', alpha=0.7)
        ax2.set_title('收益率曲线')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('收益率')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


# 参数优化相关函数
def grid_search_optimization(data: Dict[str, pd.DataFrame],
                           strategy_class: type,
                           param_grid: Dict[str, List],
                           initial_capital: float = 1000000.0,
                           commission_rate: float = 0.0003,
                           slippage: float = 0.001) -> Dict:
    """
    网格搜索参数优化
    
    Args:
        data: 股票数据字典
        strategy_class: 策略类
        param_grid: 参数网格
        initial_capital: 初始资金
        commission_rate: 佣金费率
        slippage: 滑点比率
        
    Returns:
        优化结果字典
    """
    try:
        import itertools
        
        # 边界检查
        if not data:
            return {"error": "股票数据为空"}
        
        if not param_grid:
            return {"error": "参数网格为空"}
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        best_params = None
        best_sharpe_ratio = -np.inf
        results = []
        
        for params in param_combinations:
            try:
                # 创建策略实例
                param_dict = dict(zip(param_names, params))
                strategy = strategy_class(**param_dict)
                
                # 运行回测
                result = run_backtest(data, strategy, initial_capital, commission_rate, slippage)
                
                # 检查回测是否成功
                if "error" in result:
                    continue
                
                # 记录结果
                sharpe_ratio = result['metrics'].get('sharpe_ratio', -np.inf)
                results.append({
                    'params': param_dict,
                    'sharpe_ratio': sharpe_ratio,
                    'metrics': result['metrics']
                })
                
                # 更新最佳参数
                if sharpe_ratio > best_sharpe_ratio:
                    best_sharpe_ratio = sharpe_ratio
                    best_params = param_dict
            except Exception as e:
                # 跳过出错的参数组合
                continue
        
        return {
            'best_params': best_params,
            'best_sharpe_ratio': best_sharpe_ratio,
            'all_results': results
        }
    except Exception as e:
        return {"error": f"参数优化过程中发生错误: {str(e)}"}


class GeneticAlgorithmOptimizer:
    """
    遗传算法参数优化器
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame],
                 strategy_class: type,
                 param_ranges: Dict[str, Tuple],
                 initial_capital: float = 1000000.0,
                 commission_rate: float = 0.0003,
                 slippage: float = 0.001,
                 population_size: int = 20,
                 generations: int = 10,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        """
        初始化遗传算法优化器
        
        Args:
            data: 股票数据字典
            strategy_class: 策略类
            param_ranges: 参数范围字典
            initial_capital: 初始资金
            commission_rate: 佣金费率
            slippage: 滑点比率
            population_size: 种群大小
            generations: 迭代代数
            mutation_rate: 变异率
            crossover_rate: 交叉率
        """
        self.data = data
        self.strategy_class = strategy_class
        self.param_ranges = param_ranges
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def _generate_individual(self) -> Dict[str, Union[int, float]]:
        """
        生成个体（参数组合）
        
        Returns:
            参数字典
        """
        individual = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                individual[param] = random.randint(min_val, max_val)
            else:
                individual[param] = random.uniform(min_val, max_val)
        return individual
    
    def _evaluate_individual(self, individual: Dict[str, Union[int, float]]) -> float:
        """
        评估个体适应度（夏普比率）
        
        Args:
            individual: 个体参数
            
        Returns:
            适应度值
        """
        try:
            strategy = self.strategy_class(**individual)
            result = run_backtest(self.data, strategy, self.initial_capital, 
                                self.commission_rate, self.slippage)
            return result['metrics'].get('sharpe_ratio', -np.inf)
        except:
            return -np.inf
    
    def _crossover(self, parent1: Dict[str, Union[int, float]], 
                   parent2: Dict[str, Union[int, float]]) -> Tuple[Dict[str, Union[int, float]], Dict[str, Union[int, float]]]:
        """
        交叉操作
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            两个子代
        """
        child1, child2 = parent1.copy(), parent2.copy()
        
        for param in self.param_ranges.keys():
            if random.random() < self.crossover_rate:
                child1[param], child2[param] = child2[param], child1[param]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
        """
        变异操作
        
        Args:
            individual: 个体
            
        Returns:
            变异后的个体
        """
        mutated = individual.copy()
        
        for param, (min_val, max_val) in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                if isinstance(min_val, int) and isinstance(max_val, int):
                    mutated[param] = random.randint(min_val, max_val)
                else:
                    mutated[param] = random.uniform(min_val, max_val)
        
        return mutated
    
    def optimize(self) -> Dict:
        """
        执行遗传算法优化
        
        Returns:
            优化结果
        """
        # 初始化种群
        population = [self._generate_individual() for _ in range(self.population_size)]
        
        best_individual = {}  # 初始化为空字典
        best_fitness = -np.inf
        history = []
        
        for generation in range(self.generations):
            # 评估种群
            fitness_scores = [self._evaluate_individual(ind) for ind in population]
            
            # 记录最佳个体
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
            
            # 确保best_individual不是None
            if best_individual is None:
                best_individual = {}
            
            history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'best_individual': best_individual.copy() if hasattr(best_individual, 'copy') else best_individual
            })
            
            # 选择（锦标赛选择）
            new_population = []
            for _ in range(self.population_size // 2):
                # 选择两个父代
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population
        
        return {
            'best_params': best_individual,
            'best_sharpe_ratio': best_fitness,
            'history': history
        }
    
    def _tournament_selection(self, population: List[Dict[str, Union[int, float]]], 
                             fitness_scores: List[float], 
                             tournament_size: int = 3) -> Dict[str, Union[int, float]]:
        """
        锦标赛选择
        
        Args:
            population: 种群
            fitness_scores: 适应度分数
            tournament_size: 锦标赛大小
            
        Returns:
            选中的个体
        """
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]


def monte_carlo_simulation(returns: List[float],
                          n_simulations: int = 1000,
                          n_days: int = 252) -> Dict[str, Union[List[float], float]]:
    """
    蒙特卡洛模拟
    
    Args:
        returns: 历史收益率序列
        n_simulations: 模拟次数
        n_days: 模拟天数
        
    Returns:
        模拟结果字典
    """
    if len(returns) == 0:
        return {}
    
    # 计算收益率的统计特征
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Vectorized Monte Carlo simulation
    # Generate all random returns at once: (n_simulations, n_days)
    random_returns = np.random.normal(mean_return, std_return, (n_simulations, n_days))
    
    # Calculate cumulative returns for each simulation
    # (1 + random_returns) cumprod along axis 1 (n_days)
    cumulative_returns = np.cumprod(1 + random_returns, axis=1) - 1
    
    # Final cumulative return for each simulation
    simulations = cumulative_returns[:, -1]
    
    # 计算统计指标
    mean_simulation = float(np.mean(simulations))
    std_simulation = float(np.std(simulations))
    percentile_5 = float(np.percentile(simulations, 5))
    percentile_95 = float(np.percentile(simulations, 95))
    
    return {
        'simulations': [float(s) for s in simulations],
        'mean': mean_simulation,
        'std': std_simulation,
        'percentile_5': percentile_5,
        'percentile_95': percentile_95,
        'value_at_risk': -percentile_5  # VaR 95%
    }


def generate_backtest_report(backtest_result: Dict, strategy: Strategy) -> str:
    """
    生成回测报告
    
    Args:
        backtest_result: 回测结果
        strategy: 交易策略
        
    Returns:
        回测报告字符串
    """
    engine = backtest_result['engine']
    metrics = backtest_result['metrics']
    
    # 处理最终资金的显示
    final_value = engine.portfolio_values[-1][1] if engine.portfolio_values else 0
    
    report = f"""
回测报告
========

策略名称: {strategy.name}
回测期间: {engine.dates[0] if engine.dates else 'N/A'} 至 {engine.dates[-1] if engine.dates else 'N/A'}
初始资金: {engine.initial_capital:,.2f} 元
最终资金: {final_value:,.2f} 元

业绩指标:
--------
累计收益: {metrics.get('cumulative_return', 0)*100:.2f}%
年化收益: {metrics.get('annualized_return', 0)*100:.2f}%
波动率: {metrics.get('volatility', 0)*100:.2f}%
夏普比率: {metrics.get('sharpe_ratio', 0):.2f}
最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%

交易统计:
--------
总交易次数: {len(engine.trades)}
持仓股票数: {len(engine.positions)}

"""
    
    # 如果有基准比较
    if 'alpha' in metrics:
        report += f"""
基准比较:
--------
Alpha: {metrics.get('alpha', 0):.4f}
Beta: {metrics.get('beta', 0):.4f}
信息比率: {metrics.get('information_ratio', 0):.2f}
与基准相关性: {metrics.get('correlation_with_benchmark', 0):.4f}
"""
    
    # 添加交易记录
    report += "\n最近10笔交易:\n"
    report += "日期\t\t股票\t方向\t数量\t价格\t\t佣金\n"
    report += "-" * 60 + "\n"
    
    for trade in engine.trades[-10:]:
        report += f"{trade.date.strftime('%Y-%m-%d')}\t{trade.symbol}\t{trade.direction}\t{trade.quantity}\t{trade.price:.2f}\t\t{trade.commission:.2f}\n"
    
    return report


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_data = {
        '000001': pd.DataFrame({
            'open': np.random.rand(100) * 10 + 10,
            'close': np.random.rand(100) * 10 + 10,
            'high': np.random.rand(100) * 10 + 11,
            'low': np.random.rand(100) * 10 + 9,
            'volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
    }
    
    # 测试移动平均线交叉策略
    print("测试移动平均线交叉策略...")
    strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)
    result = run_backtest(test_data, strategy)
    print(f"回测完成，夏普比率: {result['metrics'].get('sharpe_ratio', 0):.2f}")
    
    # 生成报告
    report = generate_backtest_report(result, strategy)
    print(report)