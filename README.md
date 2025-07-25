# StockTracker - 股票价格预测系统

StockTracker 是一个基于机器学习的股票价格预测系统，使用 akshare 获取股票数据，支持多种先进的机器学习模型进行预测，包括 LSTM、GRU、Transformer 以及集成学习模型（随机森林、XGBoost）。

## 功能特性

- 使用 akshare 获取实时股票数据
- 支持多种先进的机器学习模型进行股票价格预测：
  - LSTM（长短期记忆网络）
  - GRU（门控循环单元）
  - Transformer（变压器模型）
  - 随机森林（Random Forest）
  - XGBoost（极端梯度提升）
- 支持超参数调优（网格搜索、贝叶斯优化）
- 支持模型持久化和版本控制
- 支持特征工程（技术指标、时间特征等）
- 支持模型验证和评估（MAE、RMSE、MAPE等指标）
- 支持回测功能评估模型历史表现
- 支持策略回测功能，包括多种交易策略
- 支持交易成本模拟（佣金、滑点等）
- 支持业绩评估指标（累计收益、年化收益、最大回撤、夏普比率等）
- 支持基准比较功能
- 支持仓位管理功能
- 支持止损和止盈机制
- 支持参数优化功能（网格搜索、遗传算法）
- 支持蒙特卡洛模拟评估策略稳健性
- 支持股票价格趋势预测
- 提供简单的投资建议
- 支持可视化股票价格预测结果
- 提供高级可视化功能：
  - 交互式图表（使用Plotly）
  - K线图（蜡烛图）可视化
  - 技术指标叠加图（支持置信区间）
  - 热力图（相关性矩阵、行业表现等，支持聚类）
  - 3D可视化（风险-收益-时间立方体，支持颜色和大小映射）
  - 动画图表（价格变化动画，支持成交量显示）
  - 置信区间可视化
  - 模型性能对比图表
  - 风险指标可视化
  - 实时数据可视化
  - 综合仪表板
- 提供常用技术指标计算功能
- 提供全面的风险评估功能：
  - 波动率（标准差）计算
  - VaR（风险价值）计算 - 历史模拟法和参数法
  - 最大回撤（Max Drawdown）计算
  - 夏普比率计算
  - 贝塔系数计算（相对于市场指数）
  - Alpha值计算
  - 风险等级评估和投资建议
  - 蒙特卡洛模拟预测潜在损失
  - 相关性分析（股票与市场、股票间相关性）
- 提供投资组合分析功能：
  - 投资组合构建和权重分配
  - 投资组合预期收益和风险计算
  - 现代投资组合理论（MPT）相关计算
  - 有效前沿计算和可视化
  - 资本资产定价模型（CAPM）相关计算
  - 夏普比率最大化优化算法
  - 投资组合风险贡献分析
  - 投资组合优化（均值-方差优化、最小方差组合、风险平价组合、Black-Litterman模型）
  - 投资组合绩效评估（Alpha、Beta、信息比率、特雷诺比率、业绩归因分析）
  - 蒙特卡洛模拟用于投资组合优化和情景分析

## 项目结构

```
StockTracker/
├── analysis/           # 分析模块
│   ├── backtest.py     # 策略回测模块
│   ├── technical.py    # 技术指标计算模块
│   ├── portfolio.py    # 投资组合分析模块
│   └── risk.py         # 风险评估模块
├── data/               # 数据模块
│   └── fetcher.py      # 数据获取模块
├── examples/           # 示例脚本
│   └── demo.py         # 演示脚本
├── models/             # 模型模块
│   ├── advanced.py     # 高级机器学习模型模块
│   ├── base.py         # 基础机器学习模型模块
│   └── predictors.py   # 预测器模块
├── tests/              # 测试模块
│   ├── test_all.py     # 综合测试脚本
│   ├── test_fixes.py   # 修复测试脚本
│   ├── test_portfolio.py # 投资组合测试脚本
│   └── test_transformer.py # Transformer模型测试脚本
├── ui/                 # 用户界面模块
│   └── web.py          # Web界面应用
├── visualization/      # 可视化模块
│   └── charts.py       # 图表可视化模块
├── app.py              # Web应用入口
├── data_fetcher.py     # 数据获取模块
├── predictor.py        # 预测器模块
├── pyproject.toml      # 项目依赖配置
├── README.md           # 项目说明文档
└── requirements.txt    # 项目依赖列表
```

## 安装依赖

项目使用 `uv` 作为包管理器，依赖项已在 `pyproject.toml` 中配置。

```bash
# 安装依赖
uv sync
```

## 使用方法

### 1. 运行演示脚本

```bash
python examples/demo.py
```

### 2. 使用增强的可视化功能

```python
import models.predictors as predictor
import data.fetcher as data_fetcher
import visualization.charts as visualization

# 获取股票数据
stock_data = data_fetcher.get_stock_data("002607", period="daily", start_date="20240101", adjust="qfq")

# 创建可视化器
visualizer = visualization.StockVisualizer()

# 绘制带置信区间的技术指标图
from analysis import technical as indicators
sma_20 = indicators.simple_moving_average(stock_data, period=20)
sma_50 = indicators.simple_moving_average(stock_data, period=50)
rsi = indicators.relative_strength_index(stock_data, period=14)

# 计算简单置信区间
sma_20_upper = sma_20 + (stock_data['close'].std() * 0.5)
sma_20_lower = sma_20 - (stock_data['close'].std() * 0.5)

indicators_dict = {
    'SMA 20': sma_20,
    'SMA 50': sma_50,
    'RSI': rsi
}

confidence_intervals = {
    'SMA 20': (sma_20_lower, sma_20_upper)
}

fig = visualizer.plot_technical_indicators(stock_data, indicators_dict, "002607", "Technical Indicators with Confidence Intervals", confidence_intervals)
fig.show()

# 绘制聚类相关性热力图
data_dict = {
    "002607": stock_data,
    "000001": data_fetcher.get_stock_data("000001", period="daily", start_date="20240101", adjust="qfq"),
    "600036": data_fetcher.get_stock_data("600036", period="daily", start_date="20240101", adjust="qfq")
}

fig = visualizer.plot_correlation_heatmap(data_dict, 'close', cluster=True)
fig.show()

# 绘制带成交量的动画价格图
fig = visualizer.plot_animated_price(stock_data.tail(30), "002607", "Animated Price with Volume", show_volume=True)
fig.show()

# 绘制行业表现热力图
industry_data = {
    'Technology': 5.2,
    'Finance': -2.1,
    'Healthcare': 3.8,
    'Energy': -1.5,
    'Consumer': 2.7
}

fig = visualizer.plot_industry_performance_heatmap(industry_data, "Industry Performance")
fig.show()

# 绘制多指标实时图表
fig = visualization.initialize_multi_metric_realtime_chart("002607", ["price", "volume", "rsi"])
# 更新图表数据 (示例数据)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 生成示例数据
now = datetime.now()
sample_data = {
    'time': now,
    'price': 100.0,
    'volume': 1000000,
    'rsi': 50.0
}

fig = visualization.update_multi_metric_realtime_chart(fig, sample_data)
fig.show()

# 绘制综合仪表板
stock_data = data_fetcher.get_stock_data("002607", period="daily", start_date="20240101", adjust="qfq")
prediction_data = predictor.predict_stock_price("002607")
risk_data = predictor.assess_stock_risk("002607")

fig = visualization.create_comprehensive_dashboard(stock_data, prediction_data, risk_data)
fig.show()

# 绘制多股票比较图
data_dict = {
    "002607": stock_data,
    "000001": data_fetcher.get_stock_data("000001", period="daily", start_date="20240101", adjust="qfq"),
    "600036": data_fetcher.get_stock_data("600036", period="daily", start_date="20240101", adjust="qfq")
}

fig = visualization.plot_multi_stock_comparison(data_dict, 'close')
fig.show()

# 绘制投资组合分析图
import numpy as np
weights = [0.4, 0.3, 0.3]
returns_data = {}
for symbol, data in data_dict.items():
    returns_data[symbol] = data['close'].pct_change().dropna()

returns_df = pd.DataFrame(returns_data)

fig = visualization.plot_portfolio_analysis(weights, returns_df, "Portfolio Analysis")
fig.show()

# 绘制回测结果图
# 首先运行一个简单的回测
backtest_result = predictor.run_strategy_backtest("002607", strategy_type="ma_crossover", short_window=10, long_window=30)
if "error" not in backtest_result and backtest_result["success"]:
    fig = visualization.plot_backtest_results(backtest_result["result"], "Backtest Results")
    fig.show()

# 绘制实时价格图
fig = visualization.initialize_realtime_chart("002607")
# 更新图表数据 (示例数据)
sample_data = {
    'time': pd.Timestamp.now(),
    'price': 100.0
}

fig = visualization.update_realtime_chart(fig, sample_data)
fig.show()
```
### 2. 使用预测功能

```python
import models.predictors as predictor

# 使用默认的LSTM模型预测指定股票的价格
result = predictor.predict_stock_price("002607")
print(result)

# 使用GRU模型预测
result = predictor.predict_stock_price("002607", model_type="gru")
print(result)

# 使用Transformer模型预测
result = predictor.predict_stock_price("002607", model_type="transformer")
print(result)

# 使用随机森林模型预测
result = predictor.predict_stock_price("002607", model_type="rf")
print(result)

# 使用XGBoost模型预测
result = predictor.predict_stock_price("002607", model_type="xgboost")
print(result)
```

### 3. 绘制预测图表

```python
import models.predictors as predictor

# 使用默认的LSTM模型绘制股票价格预测图表
predictor.plot_stock_predictions("002607")

# 使用GRU模型绘制图表
predictor.plot_stock_predictions("002607", model_type="gru")

# 使用Transformer模型绘制图表
predictor.plot_stock_predictions("002607", model_type="transformer")
```

### 4. 使用风险评估功能

```python
import models.predictors as predictor

# 评估股票风险
risk_result = predictor.assess_stock_risk("002607")
print(risk_result)

# 预测股票价格并评估风险
prediction_with_risk = predictor.predict_stock_price_with_risk("002607")
print(prediction_with_risk)
```

### 5. 使用技术指标功能

```python
import data.fetcher as data_fetcher
from analysis import technical as indicators

# 获取股票数据
stock_data = data_fetcher.get_stock_data("002607", period="daily", start_date="20240101", adjust="qfq")

# 计算技术指标
sma_20 = indicators.simple_moving_average(stock_data, period=20)  # 20日简单移动平均线
ema_20 = indicators.exponential_moving_average(stock_data, period=20)  # 20日指数移动平均线
rsi_14 = indicators.relative_strength_index(stock_data, period=14)  # 14日相对强弱指数
macd_data = indicators.moving_average_convergence_divergence(stock_data)  # MACD指标
bb_data = indicators.bollinger_bands(stock_data, period=20)  # 20日布林带
stoch_data = indicators.stochastic_oscillator(stock_data, k_period=14, d_period=3)  # 随机指标
obv = indicators.on_balance_volume(stock_data)  # 能量潮指标
vwap = indicators.volume_weighted_average_price(stock_data, period=20)  # 20日成交量加权平均价格
cmf = indicators.chaikin_money_flow(stock_data, period=20)  # 20日蔡金资金流量指标
```

### 6. 使用风险评估功能

```python
import analysis.risk as risk_assessment
import data.fetcher as data_fetcher

# 获取股票数据
stock_data = data_fetcher.get_stock_data("002607", period="daily", start_date="20200101", adjust="qfq")
market_data = data_fetcher.get_stock_data("sh000001", period="daily", start_date="20200101", adjust="qfq")

# 计算收益率序列
stock_returns = risk_assessment.get_stock_returns(stock_data, 'close')
market_returns = risk_assessment.get_stock_returns(market_data, 'close')

# 计算各项风险指标
volatility = risk_assessment.calculate_volatility(stock_returns)  # 波动率
var_hist = risk_assessment.calculate_var_historical(stock_returns)  # 历史VaR
var_param = risk_assessment.calculate_var_parametric(stock_returns)  # 参数VaR
max_drawdown = risk_assessment.calculate_max_drawdown(stock_data['close'])  # 最大回撤
sharpe_ratio = risk_assessment.calculate_sharpe_ratio(stock_returns)  # 夏普比率
beta = risk_assessment.calculate_beta(stock_returns, market_returns)  # 贝塔系数
alpha = risk_assessment.calculate_alpha(stock_returns, market_returns)  # Alpha值
correlation = risk_assessment.calculate_correlation(stock_returns, market_returns)  # 相关性

# 执行综合风险评估
risk_result = risk_assessment.comprehensive_risk_assessment("002607", "sh000001")

# 风险评级和建议
risk_level = risk_result['risk_level']
print(f"风险等级: {risk_level['risk_level']}")
print(f"风险解释: {risk_level['explanation']}")
print(f"投资建议: {risk_level['investment_advice']}")

# 蒙特卡洛模拟
mc_results = risk_result['monte_carlo_simulation']
print(f"蒙特卡洛模拟 - 预期损失: {mc_results['expected_loss']:.4f}")
print(f"蒙特卡洛模拟 - VaR 95%: {mc_results['var_95']:.4f}")
print(f"蒙特卡洛模拟 - VaR 99%: {mc_results['var_99']:.4f}")

# 相关性分析
correlation_matrix = risk_assessment.calculate_stock_correlations(["002607", "000001", "000002"])
print("股票间相关性矩阵:")
print(correlation_matrix)
```

## 模块说明

### data_fetcher.py - 数据获取模块

提供从 akshare 获取股票数据的功能：

- `get_stock_data()`: 获取股票历史数据
- `get_stock_info()`: 获取股票基本信息

### model.py - 机器学习模型模块

实现基于 LSTM 的股票价格预测模型：

- `StockPredictor`: 股票预测器类，包含模型训练和预测功能

### advanced_model.py - 高级机器学习模型模块

实现多种先进的股票价格预测模型：

- `AdvancedStockPredictor`: 高级股票预测器类，支持多种模型
  - LSTM（长短期记忆网络）模型
  - GRU（门控循环单元）模型
  - Transformer（变压器模型）
  - 随机森林（Random Forest）模型
  - XGBoost（极端梯度提升）模型
- `TimeSeriesTransformer`: 专门用于时间序列预测的变压器模型
- `HyperparameterTuner`: 超参数调优器，支持网格搜索
- `compare_models`: 模型比较函数，用于评估不同模型的性能

高级功能：
- 超参数调优（网格搜索）
- 模型持久化和版本控制
- 特征工程（技术指标、时间特征等）
- 模型验证和评估（MAE、RMSE、MAPE等指标）
- 回测功能

### predictor.py - 预测器模块

整合数据获取和模型预测功能：

- `predict_stock_price()`: 预测股票价格

使用投资组合分析功能：

```python
import models.predictors as predictor

# 定义股票组合
stocks_dict = {
    "002607": {"symbol": "002607", "name": "中公教育"},
    "000001": {"symbol": "000001", "name": "平安银行"},
    "600036": {"symbol": "600036", "name": "招商银行"}
}

# 分析投资组合
portfolio_result = predictor.analyze_portfolio(stocks_dict)
print(portfolio_result)

# 均值-方差优化
optimized_result = predictor.optimize_portfolio(stocks_dict, method='mean_variance')
print(optimized_result)

# 最小方差组合优化
min_variance_result = predictor.optimize_portfolio(stocks_dict, method='minimum_variance')
print(min_variance_result)

# 风险平价组合优化
risk_parity_result = predictor.optimize_portfolio(stocks_dict, method='risk_parity')
print(risk_parity_result)

# 蒙特卡洛模拟
mc_result = predictor.monte_carlo_portfolio_simulation(stocks_dict, n_simulations=10000)
print(mc_result)
```
- `plot_stock_predictions()`: 绘制股票价格预测图表
- `assess_stock_risk()`: 评估股票风险
- `predict_stock_price_with_risk()`: 预测股票价格并评估风险
- `analyze_portfolio()`: 分析投资组合
- `optimize_portfolio()`: 优化投资组合
- `monte_carlo_portfolio_simulation()`: 蒙特卡洛投资组合模拟

### technical.py - 技术指标计算模块

提供常用的技术指标计算功能：

- `simple_moving_average()`: 简单移动平均线 (SMA)
- `exponential_moving_average()`: 指数移动平均线 (EMA)
- `relative_strength_index()`: 相对强弱指数 (RSI)
- `moving_average_convergence_divergence()`: 异同移动平均线 (MACD)
- `bollinger_bands()`: 布林带
- `stochastic_oscillator()`: 随机指标
- `on_balance_volume()`: 能量潮指标 (OBV)
- `volume_weighted_average_price()`: 成交量加权平均价格 (VWAP)
- `chaikin_money_flow()`: 蔡金资金流量指标 (CMF)

### risk_assessment.py - 风险评估模块

提供全面的风险评估功能：

- `calculate_volatility()`: 计算波动率（标准差）
- `calculate_var_historical()`: 计算历史模拟法VaR（风险价值）
- `calculate_var_parametric()`: 计算参数法VaR（风险价值）
- `calculate_max_drawdown()`: 计算最大回撤（Max Drawdown）
- `calculate_sharpe_ratio()`: 计算夏普比率
- `calculate_beta()`: 计算贝塔系数（相对于市场指数）
- `calculate_alpha()`: 计算Alpha值
- `calculate_correlation()`: 计算相关性
- `get_stock_returns()`: 计算股票收益率序列
- `get_market_returns()`: 获取市场指数收益率序列
- `assess_risk_level()`: 风险等级评估
- `monte_carlo_simulation()`: 蒙特卡洛模拟
- `calculate_stock_correlations()`: 计算股票间相关性矩阵
- `comprehensive_risk_assessment()`: 综合风险评估

## 依赖项

- akshare >= 1.17.26: 用于获取股票数据
- tensorflow >= 2.19.0: 用于构建机器学习模型
- scikit-learn >= 1.5.0: 用于数据预处理和集成学习模型

### backtest.py - 策略回测模块

实现股票交易策略的回测功能，包括策略框架、交易成本模拟、业绩评估等

- `BacktestEngine`: 回测引擎类
  - 策略回测框架
  - 交易成本模拟（佣金、滑点等）
  - 业绩评估指标（累计收益、年化收益、最大回撤、夏普比率等）
  - 基准比较功能
  - 仓位管理功能
  - 止损和止盈机制
- `Strategy`: 策略基类
- `MovingAverageCrossoverStrategy`: 移动平均线交叉策略
- `RSIStrategy`: RSI超买超卖策略
- `BollingerBandsStrategy`: 布林带策略
- `MomentumStrategy`: 动量策略
- `MeanReversionStrategy`: 均值回归策略
- `calculate_performance_metrics`: 计算业绩评估指标
- `run_backtest`: 运行回测
- `plot_backtest_results`: 绘制回测结果图表
- `grid_search_optimization`: 网格搜索参数优化
- `GeneticAlgorithmOptimizer`: 遗传算法参数优化器
- `monte_carlo_simulation`: 蒙特卡洛模拟
- `generate_backtest_report`: 生成回测报告

使用示例：

```python
import analysis.backtest as backtest
import data.fetcher as data_fetcher

# 获取股票数据
stock_data = data_fetcher.get_stock_data("002607", period="daily", start_date="20200101", adjust="qfq")
data_dict = {"002607": stock_data}

# 创建策略实例
strategy = backtest.MovingAverageCrossoverStrategy(short_window=10, long_window=30)

# 运行回测
result = backtest.run_backtest(data_dict, strategy)

# 计算业绩指标
metrics = backtest.calculate_performance_metrics(result['engine'].returns)

# 生成回测报告
report = backtest.generate_backtest_report(result, strategy)
print(report)

# 参数优化
param_grid = {
    "short_window": [5, 10, 20],
    "long_window": [30, 50, 100]
}
optimization_result = backtest.grid_search_optimization(data_dict, backtest.MovingAverageCrossoverStrategy, param_grid)

# 蒙特卡洛模拟
mc_result = backtest.monte_carlo_simulation(result['engine'].returns, n_simulations=1000)
```

### portfolio.py - 投资组合分析模块

实现现代投资组合理论(MPT)相关计算和分析功能：

- `PortfolioAnalyzer`: 投资组合分析器类
  - 投资组合构建和权重分配
  - 投资组合预期收益和风险计算
  - 均值-方差优化
  - 最小方差组合优化
  - 有效前沿计算和可视化
  - 资本资产定价模型(CAPM)相关计算
  - 风险贡献分析
  - 风险平价组合优化
  - Black-Litterman模型
  - 投资组合绩效评估(Alpha、Beta、信息比率、特雷诺比率、业绩归因分析)
  - 蒙特卡洛模拟用于投资组合优化和情景分析

- `analyze_portfolio`: 投资组合分析函数
- `optimize_portfolio`: 投资组合优化函数
- `monte_carlo_portfolio_simulation`: 蒙特卡洛投资组合模拟函数

使用示例：

```python
import data.fetcher as data_fetcher
import analysis.portfolio as portfolio

# 定义股票组合
stocks_dict = {
    "002607": {"symbol": "002607", "name": "中公教育"},
    "000001": {"symbol": "000001", "name": "平安银行"},
    "600036": {"symbol": "600036", "name": "招商银行"}
}

# 分析投资组合
portfolio_result = portfolio.analyze_portfolio(stocks_dict)

# 均值-方差优化
optimized_result = portfolio.optimize_portfolio(stocks_dict, method='mean_variance')

# 蒙特卡洛模拟
mc_result = portfolio.monte_carlo_portfolio_simulation(stocks_dict, n_simulations=10000)
```
- matplotlib >= 3.9.0: 用于数据可视化
- xgboost >= 2.0.0: 用于XGBoost集成学习模型
- joblib >= 1.4.0: 用于模型持久化
- plotly >= 5.18.0: 用于高级交互式可视化
- scipy >= 1.10.0: 用于聚类分析和统计计算

### charts.py - 高级可视化模块

实现高级股票数据可视化功能，包括交互式图表、K线图、技术指标叠加图、热力图、3D可视化和动画图表等。

- `StockVisualizer`: 高级股票数据可视化器类
  - `plot_interactive_price_chart()`: 创建交互式价格图表
  - `plot_candlestick_chart()`: 创建K线图（蜡烛图）
  - `plot_technical_indicators()`: 绘制技术指标叠加图（支持置信区间）
  - `plot_correlation_heatmap()`: 创建相关性热力图（支持聚类）
  - `plot_industry_performance_heatmap()`: 创建行业表现热力图
  - `plot_3d_risk_return()`: 创建3D风险-收益可视化图（支持颜色和大小映射）
  - `plot_animated_price()`: 创建价格变化动画图（支持成交量显示）
  - `plot_prediction_with_confidence()`: 绘制带置信区间的预测结果图
  - `plot_model_comparison()`: 绘制模型性能对比图
  - `plot_risk_metrics()`: 绘制风险指标可视化图
- `create_dashboard_summary()`: 创建综合仪表板摘要
- `create_comprehensive_dashboard()`: 创建综合仪表板
- `plot_multi_stock_comparison()`: 比较多只股票
- `plot_portfolio_analysis()`: 可视化投资组合分析结果
- `plot_backtest_results()`: 可视化回测结果
- `initialize_realtime_chart()`: 初始化实时图表
- `update_realtime_chart()`: 更新实时图表
- `initialize_multi_metric_realtime_chart()`: 初始化多指标实时图表
- `update_multi_metric_realtime_chart()`: 更新多指标实时图表

## 注意事项

1. 股票预测仅用于学习和研究目的，不构成投资建议
2. 机器学习模型的预测结果可能存在误差
3. 投资有风险，入市需谨慎

## 许可证

本项目仅供学习和研究使用。

## Web界面使用说明

StockTracker提供了一个基于Streamlit的Web界面，方便用户进行股票分析和预测。

### 安装Web界面依赖

在运行Web界面之前，请确保已安装所有依赖项：

```bash
# 安装项目依赖
uv sync
```

### 启动Web界面

```bash
# 运行Streamlit应用
streamlit run app.py
```

启动后，Web界面将在默认浏览器中打开，通常地址为 http://localhost:8501

### Web界面功能

Web界面提供以下功能模块：

1. **首页** - 项目介绍和快速开始指南
2. **股票分析** - 股票数据加载和基本信息查看
3. **技术指标** - 各种技术指标的计算和可视化
4. **价格预测** - 使用机器学习模型进行股票价格预测
5. **风险评估** - 全面的风险指标计算和评估
6. **投资组合** - 投资组合分析、优化和模拟
7. **回测分析** - 交易策略的回测和性能评估
8. **参数设置** - 用户偏好设置
9. **帮助文档** - 使用说明和键盘快捷键

### 键盘快捷键

Web界面支持以下键盘快捷键以提高操作效率：

- `Alt` + `1` : 跳转到首页
- `Alt` + `2` : 跳转到股票分析页面
- `Alt` + `3` : 跳转到技术指标页面
- `Alt` + `4` : 跳转到价格预测页面
- `Alt` + `5` : 跳转到风险评估页面
- `Alt` + `6` : 跳转到投资组合页面
- `Alt` + `7` : 跳转到回测分析页面
- `Alt` + `8` : 跳转到参数设置页面
- `Alt` + `9` : 跳转到帮助文档页面
- `Ctrl` + `R` : 刷新当前页面
- `F1` : 显示帮助文档
- `Esc` : 关闭当前对话框或弹出窗口

### 数据导出

Web界面支持将分析结果导出为JSON格式，报告可导出为TXT格式，图表可直接在浏览器中保存。

### 注意事项

1. 请确保网络连接正常以获取实时股票数据
2. 键盘快捷键在不同浏览器和操作系统中可能有所不同
3. 部分可视化功能需要在本地环境中运行才能显示
4. 股票预测仅用于学习和研究目的，不构成投资建议