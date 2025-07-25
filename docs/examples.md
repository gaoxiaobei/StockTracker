# 使用示例

本文件提供了StockTracker项目的各种使用示例，帮助您快速上手。

## 1. 股票价格预测

使用 `models.predictors` 模块进行股票价格预测。

```python
import models.predictors as predictor

# 使用默认的LSTM模型预测指定股票的价格
result_lstm = predictor.predict_stock_price("002607", model_type="lstm")
print("LSTM 预测结果:", result_lstm)

# 使用GRU模型预测
result_gru = predictor.predict_stock_price("002607", model_type="gru")
print("GRU 预测结果:", result_gru)

# 使用Transformer模型预测
result_transformer = predictor.predict_stock_price("002607", model_type="transformer")
print("Transformer 预测结果:", result_transformer)

# 使用随机森林模型预测
result_rf = predictor.predict_stock_price("002607", model_type="rf")
print("随机森林预测结果:", result_rf)

# 使用XGBoost模型预测
result_xgboost = predictor.predict_stock_price("002607", model_type="xgboost")
print("XGBoost 预测结果:", result_xgboost)
```

## 2. 风险评估

使用 `models.predictors` 模块评估股票风险。

```python
import models.predictors as predictor

# 评估股票风险
risk_result = predictor.assess_stock_risk("002607")
print("风险评估结果:", risk_result)

# 预测股票价格并评估风险
prediction_with_risk = predictor.predict_stock_price_with_risk("002607")
print("预测与风险评估结果:", prediction_with_risk)
```

## 3. 技术指标计算

使用 `analysis.technical` 模块计算常用技术指标。

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

print("SMA 20 (最近5个值):", sma_20.tail())
print("RSI 14 (最近5个值):", rsi_14.tail())
print("MACD (最近5个值):", macd_data.tail())
```

## 4. 投资组合分析

使用 `models.predictors` 模块进行投资组合分析和优化。

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
print("投资组合分析结果:", portfolio_result)

# 均值-方差优化
optimized_result = predictor.optimize_portfolio(stocks_dict, method='mean_variance')
print("均值-方差优化结果:", optimized_result)

# 蒙特卡洛模拟
mc_result = predictor.monte_carlo_portfolio_simulation(stocks_dict, n_simulations=1000)
print("蒙特卡洛模拟结果:", mc_result)
```

## 5. 策略回测

使用 `models.predictors` 模块运行交易策略回测。

```python
import models.predictors as predictor

# 运行移动平均线交叉策略回测
backtest_result = predictor.run_strategy_backtest("002607", strategy_type="ma_crossover", short_window=10, long_window=30)
if backtest_result.get("success"):
    print("回测成功！")
    print("回测报告:", backtest_result["report"])
else:
    print("回测失败:", backtest_result.get("error"))

# 优化策略参数
optimization_result = predictor.optimize_strategy_parameters("002607", strategy_type="ma_crossover", optimizer_type="grid_search")
print("参数优化结果:", optimization_result)
```

## 6. 高级可视化

使用 `visualization.charts` 模块绘制各种图表。

```python
import visualization.charts as visualization
import data.fetcher as data_fetcher
import models.predictors as predictor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 获取股票数据
stock_data = data_fetcher.get_stock_data("002607", period="daily", start_date="20240101", adjust="qfq")

# 创建可视化器
visualizer = visualization.StockVisualizer()

# 绘制K线图
fig_candlestick = visualizer.plot_candlestick_chart(stock_data, "002607", "K线图")
fig_candlestick.show()

# 绘制带置信区间的技术指标图
from analysis import technical as indicators
sma_20 = indicators.simple_moving_average(stock_data, period=20)
sma_50 = indicators.simple_moving_average(stock_data, period=50)
rsi = indicators.relative_strength_index(stock_data, period=14)

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

fig_tech_indicators = visualizer.plot_technical_indicators(stock_data, indicators_dict, "002607", "带置信区间的技术指标", confidence_intervals)
fig_tech_indicators.show()

# 绘制聚类相关性热力图
data_dict_corr = {
    "002607": stock_data,
    "000001": data_fetcher.get_stock_data("000001", period="daily", start_date="20240101", adjust="qfq"),
    "600036": data_fetcher.get_stock_data("600036", period="daily", start_date="20240101", adjust="qfq")
}
fig_correlation = visualizer.plot_correlation_heatmap(data_dict_corr, 'close', cluster=True)
fig_correlation.show()

# 绘制综合仪表板
prediction_data = predictor.predict_stock_price("002607")
risk_data = predictor.assess_stock_risk("002607")
fig_dashboard = visualization.create_comprehensive_dashboard(stock_data, prediction_data, risk_data)
fig_dashboard.show()

# 绘制实时价格图 (初始化和更新示例)
# fig_realtime = visualization.initialize_realtime_chart("002607")
# # 模拟实时数据更新
# for i in range(5):
#     new_data = {
#         'time': pd.Timestamp.now() + pd.Timedelta(seconds=i),
#         'price': 100.0 + np.random.randn() * 0.5
#     }
#     fig_realtime = visualization.update_realtime_chart(fig_realtime, new_data)
#     # fig_realtime.show() # 在实际应用中，这里会持续更新图表
# print("实时图表示例已运行，请在支持实时更新的环境中查看。")