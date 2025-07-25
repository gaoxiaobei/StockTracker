# API参考

本文件提供了StockTracker项目主要API的详细参考。

## `models.predictors` 模块

`models/predictors.py` 模块整合了数据获取和模型预测功能，并提供了高级API，是与项目核心功能交互的主要接口。

### `predict_stock_price(symbol: str, days: int = 5, model_type: str = 'lstm') -> Dict[str, Any]`

预测股票价格。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
    - `days` (`int`, 可选): 预测天数，默认为5。
    - `model_type` (`str`, 可选): 模型类型，可选值包括 `'lstm'`, `'gru'`, `'transformer'`, `'rf'`, `'xgboost'`，默认为`'lstm'`。
- **返回**:
    - `Dict[str, Any]`: 包含预测结果的字典，例如：
        ```json
        {
            "symbol": "002607",
            "stock_name": "中公教育",
            "current_price": 5.00,
            "predicted_price": 5.15,
            "price_change": 0.15,
            "price_change_percent": 3.0,
            "prediction_days": 5,
            "model_type": "lstm"
        }
        ```
- **示例**:
    ```python
    import models.predictors as predictor

    result = predictor.predict_stock_price("002607", model_type="transformer")
    print(result)
    ```

### `assess_stock_risk(symbol: str, market_symbol: str = "sh000001") -> Dict[str, Any]`

评估股票风险。

- **参数**:
    - `symbol` (`str`): 股票代码。
    - `market_symbol` (`str`, 可选): 市场指数代码，默认为上证指数"sh000001"。
- **返回**:
    - `Dict[str, Any]`: 包含风险评估结果的字典，例如：
        ```json
        {
            "symbol": "002607",
            "volatility": 0.025,
            "var_historical": -0.03,
            "max_drawdown": -0.10,
            "sharpe_ratio": 0.8,
            "beta": 1.2,
            "alpha": 0.005,
            "correlation_with_market": 0.7,
            "data_points": 1000,
            "risk_level": {
                "risk_level": "中等风险",
                "explanation": "该股票具有中等波动性和回撤，夏普比率良好。",
                "investment_advice": "适合风险承受能力中等的投资者。"
            },
            "monte_carlo_simulation": {
                "expected_loss": -0.02,
                "var_95": -0.04,
                "var_99": -0.06,
                "min_loss": -0.15,
                "max_loss": 0.05
            }
        }
        ```
- **示例**:
    ```python
    import models.predictors as predictor

    risk_result = predictor.assess_stock_risk("002607")
    print(risk_result)
    ```

### `predict_stock_price_with_risk(symbol: str, days: int = 5, model_type: str = 'lstm') -> Dict[str, Any]`

预测股票价格并评估风险。

- **参数**:
    - `symbol` (`str`): 股票代码。
    - `days` (`int`, 可选): 预测天数，默认为5。
    - `model_type` (`str`, 可选): 模型类型，默认为`'lstm'`。
- **返回**:
    - `Dict[str, Any]`: 包含预测结果和风险评估的合并字典。
- **示例**:
    ```python
    import models.predictors as predictor

    prediction_with_risk = predictor.predict_stock_price_with_risk("002607")
    print(prediction_with_risk)
    ```

### `analyze_portfolio(stocks_dict: Dict[str, Dict], weights: Optional[List[float]] = None) -> Dict[str, Any]`

分析投资组合。

- **参数**:
    - `stocks_dict` (`Dict[str, Dict]`): 股票数据字典，键为股票代码，值为股票信息字典（包含symbol键）。
    - `weights` (`Optional[List[float]]`, 可选): 投资组合权重列表，如果为None则使用等权重。
- **返回**:
    - `Dict[str, Any]`: 投资组合分析结果。
- **示例**:
    ```python
    import models.predictors as predictor

    stocks_to_analyze = {
        "002607": {"symbol": "002607", "name": "中公教育"},
        "000001": {"symbol": "000001", "name": "平安银行"}
    }
    portfolio_analysis_result = predictor.analyze_portfolio(stocks_to_analyze)
    print(portfolio_analysis_result)
    ```

### `optimize_portfolio(stocks_dict: Dict[str, Dict], method: str = 'mean_variance') -> Dict[str, Any]`

优化投资组合。

- **参数**:
    - `stocks_dict` (`Dict[str, Dict]`): 股票数据字典。
    - `method` (`str`, 可选): 优化方法，可选值包括`'mean_variance'` (均值-方差优化), `'minimum_variance'` (最小方差组合优化), `'risk_parity'` (风险平价组合)，默认为`'mean_variance'`。
- **返回**:
    - `Dict[str, Any]`: 投资组合优化结果。
- **示例**:
    ```python
    import models.predictors as predictor

    stocks_to_optimize = {
        "002607": {"symbol": "002607", "name": "中公教育"},
        "000001": {"symbol": "000001", "name": "平安银行"}
    }
    optimized_portfolio = predictor.optimize_portfolio(stocks_to_optimize, method='minimum_variance')
    print(optimized_portfolio)
    ```

### `monte_carlo_portfolio_simulation(stocks_dict: Dict[str, Dict], n_simulations: int = 10000) -> Dict[str, Any]`

蒙特卡洛投资组合模拟。

- **参数**:
    - `stocks_dict` (`Dict[str, Dict]`): 股票数据字典。
    - `n_simulations` (`int`, 可选): 模拟次数，默认为10000。
- **返回**:
    - `Dict[str, Any]`: 蒙特卡洛模拟结果。
- **示例**:
    ```python
    import models.predictors as predictor

    stocks_for_simulation = {
        "002607": {"symbol": "002607", "name": "中公教育"},
        "000001": {"symbol": "000001", "name": "平安银行"}
    }
    mc_result = predictor.monte_carlo_portfolio_simulation(stocks_for_simulation, n_simulations=5000)
    print(mc_result)
    ```

### `run_strategy_backtest(symbol: str, strategy_type: str = "ma_crossover", start_date: str = "20200101", **strategy_params) -> Dict[str, Any]`

运行策略回测。

- **参数**:
    - `symbol` (`str`): 股票代码。
    - `strategy_type` (`str`, 可选): 策略类型，可选值包括`'ma_crossover'` (移动平均线交叉), `'rsi'` (RSI超买超卖), `'bollinger'` (布林带), `'momentum'` (动量), `'mean_reversion'` (均值回归)，默认为`'ma_crossover'`。
    - `start_date` (`str`, 可选): 回测开始日期 (格式: "YYYYMMDD")，默认为"20200101"。
    - `**strategy_params`: 策略特定参数，例如`short_window`, `long_window` (for `ma_crossover`) 或 `period`, `overbought`, `oversold` (for `rsi`)。
- **返回**:
    - `Dict[str, Any]`: 回测结果，包含成功状态、结果数据、报告和策略名称。
- **示例**:
    ```python
    import models.predictors as predictor

    backtest_result = predictor.run_strategy_backtest("002607", strategy_type="ma_crossover", short_window=10, long_window=30)
    print(backtest_result)
    ```

### `optimize_strategy_parameters(symbol: str, strategy_type: str = "ma_crossover", start_date: str = "20200101", optimizer_type: str = "grid_search") -> Dict[str, Any]`

优化策略参数。

- **参数**:
    - `symbol` (`str`): 股票代码。
    - `strategy_type` (`str`, 可选): 策略类型，默认为`'ma_crossover'`。
    - `start_date` (`str`, 可选): 开始日期，默认为"20200101"。
    - `optimizer_type` (`str`, 可选): 优化器类型，可选值包括`'grid_search'` (网格搜索), `'genetic_algorithm'` (遗传算法)，默认为`'grid_search'`。
- **返回**:
    - `Dict[str, Any]`: 优化结果。
- **示例**:
    ```python
    import models.predictors as predictor

    optimization_result = predictor.optimize_strategy_parameters("002607", strategy_type="ma_crossover", optimizer_type="grid_search")
    print(optimization_result)
    ```

### `monte_carlo_strategy_simulation(symbol: str, strategy_type: str = "ma_crossover", start_date: str = "20200101", n_simulations: int = 1000) -> Dict[str, Any]`

策略蒙特卡洛模拟。

- **参数**:
    - `symbol` (`str`): 股票代码。
    - `strategy_type` (`str`, 可选): 策略类型，默认为`'ma_crossover'`。
    - `start_date` (`str`, 可选): 开始日期，默认为"20200101"。
    - `n_simulations` (`int`, 可选): 模拟次数，默认为1000。
- **返回**:
    - `Dict[str, Any]`: 模拟结果。
- **示例**:
    ```python
    import models.predictors as predictor

    mc_strategy_result = predictor.monte_carlo_strategy_simulation("002607", strategy_type="ma_crossover", n_simulations=500)
    print(mc_strategy_result)
    ```
### `plot_interactive_stock_chart(symbol: str, days: int = 60)`

绘制交互式股票图表。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
    - `days` (`int`, 可选): 显示天数，默认为60天。
- **返回**:
    - 无返回值，直接显示交互式图表。
- **示例**:
    ```python
    import models.predictors as predictor

    predictor.plot_interactive_stock_chart("002607", days=30)
    ```

### `plot_candlestick_chart(symbol: str, days: int = 60)`

绘制K线图（蜡烛图）。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
    - `days` (`int`, 可选): 显示天数，默认为60天。
- **返回**:
    - 无返回值，直接显示K线图。
- **示例**:
    ```python
    import models.predictors as predictor

    predictor.plot_candlestick_chart("002607", days=30)
    ```

### `plot_technical_indicators_chart(symbol: str, days: int = 60)`

绘制技术指标叠加图，包括简单移动平均线、相对强弱指数(RSI)和指数平滑异同移动平均线(MACD)等指标。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
    - `days` (`int`, 可选): 显示天数，默认为60天。
- **返回**:
    - 无返回值，直接显示技术指标图。
- **示例**:
    ```python
    import models.predictors as predictor

    predictor.plot_technical_indicators_chart("002607", days=90)
    ```

### `plot_stock_correlation_heatmap(symbols_list: List[str], cluster: bool = False)`

绘制股票相关性热力图，用于分析多只股票之间的相关性关系。

- **参数**:
    - `symbols_list` (`List[str]`): 股票代码列表 (例如: ["002607", "000001", "600000"])。
    - `cluster` (`bool`, 可选): 是否对股票进行聚类排序，默认为False。
- **返回**:
    - 无返回值，直接显示相关性热力图。
- **示例**:
    ```python
    import models.predictors as predictor

    symbols = ["002607", "000001", "600000"]
    predictor.plot_stock_correlation_heatmap(symbols, cluster=True)
    ```

### `plot_3d_risk_return_visualization(portfolio_results: Dict[str, Any])`

绘制3D风险-收益可视化图，用于展示投资组合的风险、收益和夏普比率之间的关系。

- **参数**:
    - `portfolio_results` (`Dict[str, Any]`): 投资组合结果数据，通常包含收益率、风险和夏普比率等指标。
- **返回**:
    - 无返回值，直接显示3D风险-收益可视化图。
- **示例**:
    ```python
    import models.predictors as predictor

    # 假设已有投资组合结果数据
    portfolio_results = {
        "portfolio1": {"return": 0.12, "risk": 0.15, "sharpe_ratio": 0.8},
        "portfolio2": {"return": 0.08, "risk": 0.10, "sharpe_ratio": 0.8},
        "portfolio3": {"return": 0.15, "risk": 0.20, "sharpe_ratio": 0.75}
    }
    predictor.plot_3d_risk_return_visualization(portfolio_results)
    ```

### `plot_animated_price_chart(symbol: str, days: int = 30, show_volume: bool = False)`

绘制价格变化动画图，动态展示股票价格的变化过程。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
    - `days` (`int`, 可选): 显示天数，默认为30天。
    - `show_volume` (`bool`, 可选): 是否显示成交量，默认为False。
- **返回**:
    - 无返回值，直接显示动画价格图。
- **示例**:
    ```python
    import models.predictors as predictor

    predictor.plot_animated_price_chart("002607", days=20, show_volume=True)
    ```

### `plot_prediction_with_confidence_interval(symbol: str, model_type: str = 'lstm', days: int = 5)`

绘制带置信区间的预测结果图，展示股票价格预测及其不确定性范围。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
    - `model_type` (`str`, 可选): 模型类型，默认为'lstm'。可选值包括 'lstm', 'gru', 'transformer', 'rf', 'xgboost'。
    - `days` (`int`, 可选): 预测天数，默认为5天。
- **返回**:
    - 无返回值，直接显示带置信区间的预测图。
- **示例**:
    ```python
    import models.predictors as predictor

    predictor.plot_prediction_with_confidence_interval("002607", model_type="transformer", days=7)
    ```

### `plot_model_comparison_chart(symbol: str, model_types: Optional[List[str]] = None)`

绘制模型性能对比图，比较不同模型对同一股票的预测表现。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
    - `model_types` (`Optional[List[str]]`, 可选): 模型类型列表，默认为None，将使用所有支持的模型类型 ['lstm', 'gru', 'transformer', 'rf', 'xgboost']。
- **返回**:
    - 无返回值，直接显示模型性能对比图。
- **示例**:
    ```python
    import models.predictors as predictor

    predictor.plot_model_comparison_chart("002607", model_types=['lstm', 'gru', 'rf'])
    ```

### `plot_risk_metrics_chart(symbol: str)`

绘制风险指标可视化图，展示股票的各种风险指标，如波动率、最大回撤、VaR、贝塔系数和夏普比率等。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
- **返回**:
    - 无返回值，直接显示风险指标可视化图。
- **示例**:
    ```python
    import models.predictors as predictor

    predictor.plot_risk_metrics_chart("002607")
    ```

### `create_comprehensive_dashboard(symbol: str, model_type: str = 'lstm')`

创建综合仪表板，整合股票价格趋势、预测结果、风险评估等多维度信息。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
    - `model_type` (`str`, 可选): 模型类型，默认为'lstm'。可选值包括 'lstm', 'gru', 'transformer', 'rf', 'xgboost'。
- **返回**:
    - 无返回值，直接显示综合仪表板。
- **示例**:
    ```python
    import models.predictors as predictor

    predictor.create_comprehensive_dashboard("002607", model_type="gru")
    ```

### `plot_multi_stock_comparison(symbols_list: List[str], metric: str = 'close', days: int = 60)`

绘制多股票比较图，用于比较多个股票在同一指标上的表现差异。

- **参数**:
    - `symbols_list` (`List[str]`): 股票代码列表 (例如: ["002607", "000001", "600000"])。
    - `metric` (`str`, 可选): 比较的指标，默认为'close'（收盘价）。
    - `days` (`int`, 可选): 显示天数，默认为60天。
- **返回**:
    - 无返回值，直接显示多股票比较图。
- **示例**:
    ```python
    import models.predictors as predictor

    symbols = ["002607", "000001", "600000"]
    predictor.plot_multi_stock_comparison(symbols, metric='close', days=90)
    ```

### `plot_portfolio_analysis_chart(stocks_dict: Dict[str, Dict], weights: Optional[List[float]] = None)`

绘制投资组合分析图，展示投资组合的收益、累计收益、权重分布和风险贡献等信息。

- **参数**:
    - `stocks_dict` (`Dict[str, Dict]`): 股票数据字典，键为股票代码，值为股票信息字典（包含symbol键）。
    - `weights` (`Optional[List[float]]`, 可选): 投资组合权重列表，如果为None则使用等权重，默认为None。
- **返回**:
    - 无返回值，直接显示投资组合分析图。
- **示例**:
    ```python
    import models.predictors as predictor

    stocks = {
        "002607": {"symbol": "002607", "name": "中公教育"},
        "000001": {"symbol": "000001", "name": "平安银行"}
    }
    weights = [0.6, 0.4]
    predictor.plot_portfolio_analysis_chart(stocks, weights)
    ```

### `plot_backtest_results_chart(symbol: str, strategy_type: str = "ma_crossover", **strategy_params)`

绘制回测结果图，展示交易策略的回测表现，包括投资组合价值、基准价值和交易点位等信息。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
    - `strategy_type` (`str`, 可选): 策略类型，默认为"ma_crossover"（移动平均线交叉策略）。
    - `**strategy_params`: 策略特定参数，例如`short_window`, `long_window` (for `ma_crossover`) 或 `period`, `overbought`, `oversold` (for `rsi`)。
- **返回**:
    - 无返回值，直接显示回测结果图。
- **示例**:
    ```python
    import models.predictors as predictor

    predictor.plot_backtest_results_chart("002607", strategy_type="ma_crossover", short_window=10, long_window=30)
    ```

### `initialize_realtime_chart(symbol: str, window_size: int = 100)`

初始化实时图表，用于创建一个空的实时数据可视化图表框架。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
    - `window_size` (`int`, 可选): 显示数据点数量，默认为100个点。
- **返回**:
    - `go.Figure`: 实时图表对象，可用于后续更新数据。
- **示例**:
    ```python
    import models.predictors as predictor

    fig = predictor.initialize_realtime_chart("002607", window_size=50)
    ```

### `update_realtime_chart(fig: go.Figure, new_data: Dict[str, Any])`

更新实时图表，向已有的实时图表中添加新的数据点并更新显示。

- **参数**:
    - `fig` (`go.Figure`): 实时图表对象，由`initialize_realtime_chart`函数创建。
    - `new_data` (`Dict[str, Any]`): 新数据点，应包含'time'和'price'键的字典 (例如: {"time": "2023-01-01 10:00:00", "price": 10.5})。
- **返回**:
    - `go.Figure`: 更新后的图表对象。
- **示例**:
    ```python
    import models.predictors as predictor

    fig = predictor.initialize_realtime_chart("002607")
    new_point = {"time": "2023-01-01 10:00:00", "price": 10.5}
    updated_fig = predictor.update_realtime_chart(fig, new_point)
    ```

## 其他常用函数

### `data.fetcher.get_stock_data(symbol: str, period: str = "daily", start_date: Optional[str] = None, end_date: Optional[str] = None, adjust: str = "qfq") -> pd.DataFrame`

获取股票数据。

- **参数**:
    - `symbol` (`str`): 股票代码 (例如: "002607")。
    - `period` (`str`, 可选): 数据周期 ("daily", "weekly", "monthly")，默认为"daily"。
    - `start_date` (`Optional[str]`, 可选): 开始日期 (格式: "YYYYMMDD")。
    - `end_date` (`Optional[str]`, 可选): 结束日期 (格式: "YYYYMMDD")。
    - `adjust` (`str`, 可选): 复权类型 ("qfq": 前复权, "hfq": 后复权, "": 不复权)，默认为"qfq"。
- **返回**:
    - `pd.DataFrame`: 股票数据DataFrame。
- **示例**:
    ```python
    import data.fetcher as data_fetcher
    stock_data = data_fetcher.get_stock_data("002607", start_date="20230101", end_date="20241231")
    print(stock_data.head())
    ```

### `data.fetcher.get_stock_info(symbol: str) -> dict`

获取股票基本信息。

- **参数**:
    - `symbol` (`str`): 股票代码。
- **返回**:
    - `dict`: 股票基本信息字典。
- **示例**:
    ```python
    import data.fetcher as data_fetcher
    stock_info = data_fetcher.get_stock_info("002607")
    print(stock_info)
    ```

### `visualization.charts.StockVisualizer` 类

高级股票数据可视化器，提供了多种图表绘制方法。

- **常用方法**:
    - `plot_interactive_price_chart()`: 创建交互式价格图表。
    - `plot_candlestick_chart()`: 创建K线图（蜡烛图）。
    - `plot_technical_indicators()`: 绘制技术指标叠加图。
    - `plot_correlation_heatmap()`: 创建相关性热力图。
    - `create_comprehensive_dashboard()`: 创建综合仪表板。
- **示例**:
    ```python
    import visualization.charts as visualization
    import data.fetcher as data_fetcher

    stock_data = data_fetcher.get_stock_data("002607", period="daily", start_date="20240101", adjust="qfq")
    visualizer = visualization.StockVisualizer()
    fig = visualizer.plot_candlestick_chart(stock_data, "002607")
    fig.show()