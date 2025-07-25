# 模块详细说明

本文件详细介绍了StockTracker项目中的各个模块及其功能。

## `analysis/` - 分析模块

该模块包含用于股票数据分析的各种子模块，包括回测、投资组合分析、风险评估和技术指标计算。

### `analysis/backtest.py` - 策略回测模块

提供股票交易策略的回测功能，包括策略框架、交易成本模拟、业绩评估等。

- `BacktestEngine`: 回测引擎类，用于执行策略回测。
- `Strategy`: 策略基类，定义了策略的基本接口。
- `MovingAverageCrossoverStrategy`: 移动平均线交叉策略的实现。
- `RSIStrategy`: RSI超买超卖策略的实现。
- `BollingerBandsStrategy`: 布林带策略的实现。
- `MomentumStrategy`: 动量策略的实现。
- `MeanReversionStrategy`: 均值回归策略的实现。
- `calculate_performance_metrics`: 计算回测业绩评估指标。
- `run_backtest`: 运行回测的主函数。
- `plot_backtest_results`: 绘制回测结果图表。
- `grid_search_optimization`: 网格搜索参数优化。
- `GeneticAlgorithmOptimizer`: 遗传算法参数优化器。
- `monte_carlo_simulation`: 蒙特卡洛模拟，用于评估策略稳健性。
- `generate_backtest_report`: 生成回测报告。

### `analysis/portfolio.py` - 投资组合分析模块

实现现代投资组合理论(MPT)相关计算和分析功能。

- `PortfolioAnalyzer`: 投资组合分析器类，提供投资组合构建、优化和分析功能。
- `analyze_portfolio`: 投资组合分析函数。
- `optimize_portfolio`: 投资组合优化函数（均值-方差优化、最小方差组合、风险平价组合、Black-Litterman模型）。
- `monte_carlo_portfolio_simulation`: 蒙特卡洛投资组合模拟函数。

### `analysis/risk.py` - 风险评估模块

提供全面的风险评估功能。

- `calculate_volatility()`: 计算波动率（标准差）。
- `calculate_var_historical()`: 计算历史模拟法VaR（风险价值）。
- `calculate_var_parametric()`: 计算参数法VaR（风险价值）。
- `calculate_max_drawdown()`: 计算最大回撤（Max Drawdown）。
- `calculate_sharpe_ratio()`: 计算夏普比率。
- `calculate_beta()`: 计算贝塔系数（相对于市场指数）。
- `calculate_alpha()`: 计算Alpha值。
- `calculate_correlation()`: 计算相关性。
- `get_stock_returns()`: 计算股票收益率序列。
- `get_market_returns()`: 获取市场指数收益率序列。
- `assess_risk_level()`: 风险等级评估。
- `monte_carlo_simulation()`: 蒙特卡洛模拟。
- `calculate_stock_correlations()`: 计算股票间相关性矩阵。
- `comprehensive_risk_assessment()`: 综合风险评估。

### `analysis/technical.py` - 技术指标计算模块

提供常用的技术指标计算功能。

- `simple_moving_average()`: 简单移动平均线 (SMA)。
- `exponential_moving_average()`: 指数移动平均线 (EMA)。
- `relative_strength_index()`: 相对强弱指数 (RSI)。
- `moving_average_convergence_divergence()`: 异同移动平均线 (MACD)。
- `bollinger_bands()`: 布林带。
- `stochastic_oscillator()`: 随机指标。
- `on_balance_volume()`: 能量潮指标 (OBV)。
- `volume_weighted_average_price()`: 成交量加权平均价格 (VWAP)。
- `chaikin_money_flow()`: 蔡金资金流量指标 (CMF)。

## `configs/` - 配置模块

该模块用于存放项目的配置信息。

## `data/` - 数据模块

该模块负责股票数据的获取和处理。

### `data/fetcher.py` - 数据获取模块

提供从 `akshare` 获取股票数据的功能。

- `get_stock_data()`: 获取股票历史数据。
- `get_stock_info()`: 获取股票基本信息。

## `examples/` - 示例脚本

该模块包含演示项目功能的示例脚本。

### `examples/demo.py` - 演示脚本

展示了如何使用项目中的各个功能。

## `models/` - 模型模块

该模块包含用于股票价格预测的机器学习模型。

### `models/base.py` - 基础机器学习模型模块

实现了基础的机器学习模型。

- `StockPredictor`: 股票预测器类，包含模型训练和预测功能。

### `models/advanced.py` - 高级机器学习模型模块

实现了多种先进的股票价格预测模型。

- `AdvancedStockPredictor`: 高级股票预测器类，支持多种模型（LSTM, GRU, Transformer, 随机森林, XGBoost）。
- `TimeSeriesTransformer`: 专门用于时间序列预测的变压器模型。
- `HyperparameterTuner`: 超参数调优器，支持网格搜索。
- `compare_models`: 模型比较函数，用于评估不同模型的性能。

### `models/predictors.py` - 预测器模块

整合了数据获取和模型预测功能，并提供了高级API。

- `predict_stock_price()`: 预测股票价格。
- `assess_stock_risk()`: 评估股票风险。
- `plot_stock_predictions()`: 绘制股票价格预测图表。
- `predict_stock_price_with_risk()`: 预测股票价格并评估风险。
- `analyze_portfolio()`: 分析投资组合。
- `optimize_portfolio()`: 优化投资组合。
- `monte_carlo_portfolio_simulation()`: 蒙特卡洛投资组合模拟。
- `run_strategy_backtest()`: 运行策略回测。
- `optimize_strategy_parameters()`: 优化策略参数。
- `monte_carlo_strategy_simulation()`: 策略蒙特卡洛模拟。

## `tests/` - 测试模块

该模块包含项目的测试脚本。

## `ui/` - 用户界面模块

该模块包含项目的用户界面相关代码。

### `ui/web.py` - Web界面应用

基于Streamlit的Web界面应用。

## `utils/` - 工具模块

该模块包含项目中的通用工具函数。

## `visualization/` - 可视化模块

该模块提供高级可视化功能。

### `visualization/charts.py` - 图表可视化模块

实现高级股票数据可视化功能，包括交互式图表、K线图、技术指标叠加图、热力图、3D可视化和动画图表等。

- `StockVisualizer`: 高级股票数据可视化器类。
- `plot_interactive_price_chart()`: 创建交互式价格图表。
- `plot_candlestick_chart()`: 创建K线图（蜡烛图）。
- `plot_technical_indicators()`: 绘制技术指标叠加图（支持置信区间）。
- `plot_correlation_heatmap()`: 创建相关性热力图（支持聚类）。
- `plot_industry_performance_heatmap()`: 创建行业表现热力图。
- `plot_3d_risk_return()`: 创建3D风险-收益可视化图（支持颜色和大小映射）。
- `plot_animated_price()`: 创建价格变化动画图（支持成交量显示）。
- `plot_prediction_with_confidence()`: 绘制带置信区间的预测结果图。
- `plot_model_comparison()`: 绘制模型性能对比图。
- `plot_risk_metrics()`: 绘制风险指标可视化图。
- `create_dashboard_summary()`: 创建综合仪表板摘要。
- `create_comprehensive_dashboard()`: 创建综合仪表板。
- `plot_multi_stock_comparison()`: 比较多只股票。
- `plot_portfolio_analysis()`: 可视化投资组合分析结果。
- `plot_backtest_results()`: 可视化回测结果。
- `initialize_realtime_chart()`: 初始化实时图表。
- `update_realtime_chart()`: 更新实时图表。
- `initialize_multi_metric_realtime_chart()`: 初始化多指标实时图表。
- `update_multi_metric_realtime_chart()`: 更新多指标实时图表。