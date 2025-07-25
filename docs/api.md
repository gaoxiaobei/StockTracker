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