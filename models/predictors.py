import data.fetcher as data_fetcher
import models.base as model
import models.advanced as advanced_model
import analysis.risk as risk_assessment
import analysis.portfolio as portfolio
import analysis.backtest as backtest
import visualization.charts as visualization
import pandas as pd
from typing import Dict, Any, List, Optional


def predict_stock_price(symbol, days=5, model_type='lstm') -> Dict[str, Any]:
    """
    预测股票价格
    
    Args:
        symbol: 股票代码
        days: 预测天数
        model_type: 模型类型 ('lstm', 'gru', 'transformer', 'rf', 'xgboost')
        
    Returns:
        dict: 预测结果
    """
    # 获取股票数据
    print(f"正在获取股票 {symbol} 的数据...")
    stock_data = data_fetcher.get_stock_data(symbol, period="daily", start_date="20200101", adjust="qfq")
    
    if stock_data.empty:
        return {"error": "无法获取股票数据"}
    
    print(f"获取到 {len(stock_data)} 条数据记录")
    
    # 创建预测器
    if model_type in ['lstm', 'gru', 'transformer', 'rf', 'xgboost']:
        # 使用新的高级预测器
        predictor = advanced_model.AdvancedStockPredictor(look_back=60, model_type=model_type)
    else:
        # 使用原有的LSTM预测器
        predictor = model.StockPredictor(look_back=60)
    
    # 训练模型
    print(f"正在训练 {model_type.upper()} 模型...")
    if model_type in ['lstm', 'gru', 'transformer']:
        history = predictor.train(stock_data, epochs=50, batch_size=32)
    else:
        history = predictor.train(stock_data)
    
    # 预测未来价格
    print("正在预测未来价格...")
    predicted_price = predictor.predict(stock_data)
    
    # 获取当前价格
    current_price = stock_data['close'].iloc[-1]
    
    # 计算预测变化
    price_change = predicted_price - current_price
    price_change_percent = (price_change / current_price) * 100
    
    # 获取股票信息
    stock_info = data_fetcher.get_stock_info(symbol)
    
    result = {
        "symbol": symbol,
        "stock_name": stock_info.get("股票简称", "未知"),
        "current_price": float(current_price),
        "predicted_price": float(predicted_price),
        "price_change": float(price_change),
        "price_change_percent": float(price_change_percent),
        "prediction_days": days,
        "model_type": model_type
    }
    
    return result


def assess_stock_risk(symbol, market_symbol="sh000001") -> Dict[str, Any]:
    """
    评估股票风险
    
    Args:
        symbol: 股票代码
        market_symbol: 市场指数代码，默认为上证指数"sh000001"
        
    Returns:
        dict: 风险评估结果
    """
    try:
        # 执行综合风险评估
        risk_result = risk_assessment.comprehensive_risk_assessment(
            stock_symbol=symbol,
            market_symbol=market_symbol,
            start_date="20200101"
        )
        
        return risk_result
    except Exception as e:
        print(f"风险评估出错: {e}")
        return {"error": f"风险评估失败: {str(e)}"}


def plot_stock_predictions(symbol, days=30, model_type='lstm'):
    """
    绘制股票预测图表
    
    Args:
        symbol: 股票代码
        days: 显示天数
        model_type: 模型类型 ('lstm', 'gru', 'transformer', 'rf', 'xgboost')
    """
    # 获取股票数据
    stock_data = data_fetcher.get_stock_data(symbol, period="daily", start_date="20200101", adjust="qfq")
    
    if stock_data.empty:
        print("无法获取股票数据")
        return
    
    # 创建预测器
    if model_type in ['lstm', 'gru', 'transformer', 'rf', 'xgboost']:
        # 使用新的高级预测器
        predictor = advanced_model.AdvancedStockPredictor(look_back=60, model_type=model_type)
    else:
        # 使用原有的LSTM预测器
        predictor = model.StockPredictor(look_back=60)
    
    # 训练模型
    print(f"正在训练 {model_type.upper()} 模型...")
    if model_type in ['lstm', 'gru', 'transformer']:
        history = predictor.train(stock_data, epochs=50, batch_size=32)
    else:
        history = predictor.train(stock_data)
    
    # 绘制预测结果
    print("正在绘制预测结果...")
    predictor.plot_predictions(stock_data, days=days)

def plot_interactive_stock_chart(symbol, days=60):
    """
    绘制交互式股票图表
    
    Args:
        symbol: 股票代码
        days: 显示天数
    """
    # 获取股票数据
    stock_data = data_fetcher.get_stock_data(symbol, period="daily", start_date="20200101", adjust="qfq")
    
    if stock_data.empty:
        print("无法获取股票数据")
        return
    
    # 只保留最近指定天数的数据
    stock_data = stock_data.tail(days)
    
    # 创建可视化器
    visualizer = visualization.StockVisualizer()
    
    # 绘制交互式价格图表
    fig = visualizer.plot_interactive_price_chart(stock_data, symbol, "Interactive Stock Price")
    fig.show()


def plot_candlestick_chart(symbol, days=60):
    """
    绘制K线图（蜡烛图）
    
    Args:
        symbol: 股票代码
        days: 显示天数
    """
    # 获取股票数据
    stock_data = data_fetcher.get_stock_data(symbol, period="daily", start_date="20200101", adjust="qfq")
    
    if stock_data.empty:
        print("无法获取股票数据")
        return
    
    # 只保留最近指定天数的数据
    stock_data = stock_data.tail(days)
    
    # 创建可视化器
    visualizer = visualization.StockVisualizer()
    
    # 绘制K线图
    fig = visualizer.plot_candlestick_chart(stock_data, symbol, "Candlestick Chart")
    fig.show()


def plot_technical_indicators_chart(symbol, days=60):
    """
    绘制技术指标叠加图
    
    Args:
        symbol: 股票代码
        days: 显示天数
    """
    # 获取股票数据
    stock_data = data_fetcher.get_stock_data(symbol, period="daily", start_date="20200101", adjust="qfq")
    
    if stock_data.empty:
        print("无法获取股票数据")
        return
    
    # 只保留最近指定天数的数据
    stock_data = stock_data.tail(days)
    
    # 计算技术指标
    from analysis import technical as indicators
    sma_20 = indicators.simple_moving_average(stock_data, period=20)
    sma_50 = indicators.simple_moving_average(stock_data, period=50)
    rsi = indicators.relative_strength_index(stock_data, period=14)
    macd_data = indicators.moving_average_convergence_divergence(stock_data)
    
    # 计算简单置信区间（实际应用中应使用更复杂的统计方法）
    sma_20_upper = sma_20 + (stock_data['close'].std() * 0.5)
    sma_20_lower = sma_20 - (stock_data['close'].std() * 0.5)
    sma_50_upper = sma_50 + (stock_data['close'].std() * 0.8)
    sma_50_lower = sma_50 - (stock_data['close'].std() * 0.8)
    
    # 准备指标数据
    indicators_dict = {
        'SMA 20': sma_20,
        'SMA 50': sma_50,
        'RSI': rsi,
        'MACD': macd_data['macd_line']
    }
    
    # 准备置信区间数据
    confidence_intervals = {
        'SMA 20': (sma_20_lower, sma_20_upper),
        'SMA 50': (sma_50_lower, sma_50_upper)
    }
    
    # 创建可视化器
    visualizer = visualization.StockVisualizer()
    
    # 绘制技术指标图
    fig = visualizer.plot_technical_indicators(stock_data, indicators_dict, symbol, "Technical Indicators", confidence_intervals)
    fig.show()


def plot_stock_correlation_heatmap(symbols_list, cluster=False):
    """
    绘制股票相关性热力图
    
    Args:
        symbols_list: 股票代码列表
        cluster: 是否对股票进行聚类排序
    """
    # 获取股票数据
    data_dict = {}
    for symbol in symbols_list:
        stock_data = data_fetcher.get_stock_data(symbol, period="daily", start_date="20200101", adjust="qfq")
        if not stock_data.empty:
            data_dict[symbol] = stock_data
    
    if not data_dict:
        print("无法获取任何股票数据")
        return
    
    # 创建可视化器
    visualizer = visualization.StockVisualizer()
    
    # 绘制相关性热力图
    fig = visualizer.plot_correlation_heatmap(data_dict, 'close', cluster)
    fig.show()


def plot_3d_risk_return_visualization(portfolio_results):
    """
    绘制3D风险-收益可视化图
    
    Args:
        portfolio_results: 投资组合结果数据
    """
    # 创建可视化器
    visualizer = visualization.StockVisualizer()
    
    # 转换数据格式
    import pandas as pd
    portfolio_df = pd.DataFrame(portfolio_results)
    
    # 添加额外的可视化数据
    if 'sortino_ratio' in portfolio_df.columns:
        color_col = 'sortino_ratio'
    else:
        color_col = 'sharpe_ratio'
    
    # 绘制3D风险-收益图
    fig = visualizer.plot_3d_risk_return(portfolio_df, 'return', 'risk', 'sharpe_ratio',
                                        "Risk-Return-Performance 3D Visualization",
                                        color_col=color_col)
    fig.show()


def plot_animated_price_chart(symbol, days=30, show_volume=False):
    """
    绘制价格变化动画图
    
    Args:
        symbol: 股票代码
        days: 显示天数
        show_volume: 是否显示成交量
    """
    # 获取股票数据
    stock_data = data_fetcher.get_stock_data(symbol, period="daily", start_date="20200101", adjust="qfq")
    
    if stock_data.empty:
        print("无法获取股票数据")
        return
    
    # 只保留最近指定天数的数据
    stock_data = stock_data.tail(days)
    
    # 创建可视化器
    visualizer = visualization.StockVisualizer()
    
    # 绘制动画价格图
    fig = visualizer.plot_animated_price(stock_data, symbol, "Animated Price Movement", show_volume=show_volume)
    fig.show()


def plot_prediction_with_confidence_interval(symbol, model_type='lstm', days=5):
    """
    绘制带置信区间的预测结果图

    Args:
        symbol: 股票代码
        model_type: 模型类型
        days: 预测天数
    """
    # 获取股票数据
    stock_data = data_fetcher.get_stock_data(symbol, period="daily", start_date="20200101", adjust="qfq")

    if stock_data.empty:
        print("无法获取股票数据")
        return

    # 创建预测器
    if model_type in ['lstm', 'gru', 'transformer', 'rf', 'xgboost']:
        # 使用新的高级预测器
        predictor = advanced_model.AdvancedStockPredictor(look_back=60, model_type=model_type)
    else:
        # 使用原有的LSTM预测器
        predictor = model.StockPredictor(look_back=60)

    # 训练模型
    print(f"正在训练 {model_type.upper()} 模型...")
    if model_type in ['lstm', 'gru', 'transformer']:
        predictor.train(stock_data, epochs=50, batch_size=32)
    else:
        predictor.train(stock_data)

    # 预测未来价格
    print("正在预测未来价格...")
    predicted_price = predictor.predict(stock_data)

    # 创建预测数据（这里简化处理，实际应该生成未来几天的预测）
    future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=days)
    predictions = pd.Series([predicted_price] * days, index=future_dates)

    # 创建可视化器
    visualizer = visualization.StockVisualizer()

    # 绘制预测图
    fig = visualizer.plot_prediction_with_confidence(stock_data.tail(30), predictions,
                                                    symbol=symbol,
                                                    title=f"Price Prediction with Confidence Interval - {model_type.upper()}")
    fig.show()


def plot_model_comparison_chart(symbol, model_types=None):
    """
    绘制模型性能对比图
    
    Args:
        symbol: 股票代码
        model_types: 模型类型列表
    """
    if model_types is None:
        model_types = ['lstm', 'gru', 'transformer', 'rf', 'xgboost']
    
    # 获取模型预测结果
    model_results = {}
    for model_type in model_types:
        try:
            result = predict_stock_price(symbol, days=5, model_type=model_type)
            if "error" not in result:
                model_results[model_type.upper()] = result
        except Exception as e:
            print(f"模型 {model_type} 预测失败: {e}")
    
    if not model_results:
        print("没有有效的模型预测结果")
        return
    
    # 创建可视化器
    visualizer = visualization.StockVisualizer()
    
    # 绘制模型对比图
    fig = visualizer.plot_model_comparison(model_results, "price_change_percent", 
                                          "Model Performance Comparison")
    fig.show()


def plot_risk_metrics_chart(symbol):
    """
    绘制风险指标可视化图
    
    Args:
        symbol: 股票代码
    """
    # 获取风险评估结果
    risk_result = assess_stock_risk(symbol)
    
    if "error" in risk_result:
        print(f"风险评估失败: {risk_result['error']}")
        return
    
    # 创建可视化器
    visualizer = visualization.StockVisualizer()
    
    # 绘制风险指标图
    fig = visualizer.plot_risk_metrics(risk_result, symbol, "Risk Metrics Visualization")
    fig.show()
def create_comprehensive_dashboard(symbol, model_type='lstm'):
    """
    创建综合仪表板
    
    Args:
        symbol: 股票代码
        model_type: 模型类型
    """
    # 获取股票数据
    stock_data = data_fetcher.get_stock_data(symbol, period="daily", start_date="20200101", adjust="qfq")
    
    if stock_data.empty:
        print("无法获取股票数据")
        return
    
    # 获取预测结果
    prediction_result = predict_stock_price(symbol, model_type=model_type)

    # 获取风险评估结果
    risk_result = assess_stock_risk(symbol)

    # 创建综合仪表板
    if hasattr(visualization, 'create_comprehensive_dashboard'):
        fig = visualization.create_comprehensive_dashboard(stock_data, prediction_result, risk_result)
        fig.show()
    else:
        print("创建综合仪表板功能暂不可用")


def plot_multi_stock_comparison(symbols_list, metric='close', days=60):
    """
    绘制多股票比较图
    
    Args:
        symbols_list: 股票代码列表
        metric: 比较的指标
        days: 显示天数
    """
    # 获取股票数据
    data_dict = {}
    for symbol in symbols_list:
        stock_data = data_fetcher.get_stock_data(symbol, period="daily", start_date="20200101", adjust="qfq")
        if not stock_data.empty:
            # 只保留最近指定天数的数据
            stock_data = stock_data.tail(days)
            data_dict[symbol] = stock_data
    
    if not data_dict:
        print("无法获取任何股票数据")
        return
    
    # 绘制多股票比较图
    fig = visualization.plot_multi_stock_comparison(data_dict, metric)
    fig.show()


def plot_portfolio_analysis_chart(stocks_dict, weights=None):
    """
    绘制投资组合分析图
    
    Args:
        stocks_dict: 股票数据字典
        weights: 投资组合权重
    """
    try:
        # 获取股票数据
        stocks_data = {}
        for symbol, stock_info in stocks_dict.items():
            symbol_code = stock_info.get('symbol', symbol)
            stock_data = data_fetcher.get_stock_data(
                symbol_code,
                period="daily",
                start_date="20200101",
                adjust="qfq"
            )
            if not stock_data.empty:
                stocks_data[symbol_code] = stock_data
        
        if not stocks_data:
            print("无法获取任何股票数据")
            return
        
        # 计算收益率
        returns_data = {}
        for symbol, data in stocks_data.items():
            returns_data[symbol] = data['close'].pct_change().dropna()
        
        returns_df = pd.DataFrame(returns_data)
        
        # 如果没有提供权重，则使用等权重
        if weights is None:
            weights = [1/len(stocks_data)] * len(stocks_data)
        
        # 绘制投资组合分析图
        fig = visualization.plot_portfolio_analysis(weights, returns_df, "Portfolio Analysis")
        fig.show()
    except Exception as e:
        print(f"绘制投资组合分析图失败: {e}")


def plot_backtest_results_chart(symbol, strategy_type="ma_crossover", **strategy_params):
    """
    绘制回测结果图
    
    Args:
        symbol: 股票代码
        strategy_type: 策略类型
        **strategy_params: 策略参数
    """
    try:
        # 运行回测
        backtest_result = run_strategy_backtest(symbol, strategy_type, **strategy_params)
        
        if "error" in backtest_result:
            print(f"回测失败: {backtest_result['error']}")
            return
        
        # 绘制回测结果图
        fig = visualization.plot_backtest_results(backtest_result['result'], "Backtest Results")
        fig.show()
    except Exception as e:
        print(f"绘制回测结果图失败: {e}")


def initialize_realtime_chart(symbol, window_size=100):
    """
    初始化实时图表
    
    Args:
        symbol: 股票代码
        window_size: 显示数据点数量
        
    Returns:
        go.Figure: 实时图表
    """
    # 创建可视化器
    visualizer = visualization.StockVisualizer()
    
    # 初始化实时图表
    fig = visualization.initialize_realtime_chart(symbol, window_size)
    return fig


def update_realtime_chart(fig, new_data):
    """
    更新实时图表
    
    Args:
        fig: 实时图表
        new_data: 新数据点
        
    Returns:
        go.Figure: 更新后的图表
    """
    # 更新实时图表
    updated_fig = visualization.update_realtime_chart(fig, new_data)
    return updated_fig


def predict_stock_price_with_risk(symbol, days=5, model_type='lstm') -> Dict[str, Any]:
    """
    预测股票价格并评估风险
    
    Args:
        symbol: 股票代码
        days: 预测天数
        model_type: 模型类型 ('lstm', 'gru', 'transformer', 'rf', 'xgboost')
        
    Returns:
        dict: 预测结果和风险评估
    """
    # 获取价格预测
    price_prediction = predict_stock_price(symbol, days, model_type)
    
    # 如果价格预测失败，直接返回错误
    if "error" in price_prediction:
        return price_prediction
    
    # 获取风险评估
    risk_assessment_result = assess_stock_risk(symbol)
    
    # 如果风险评估失败，返回价格预测结果
    if "error" in risk_assessment_result:
        price_prediction["risk_assessment"] = risk_assessment_result
        return price_prediction
    
    # 合并结果
    result = price_prediction.copy()
    result["risk_assessment"] = risk_assessment_result
    
    return result


def analyze_portfolio(stocks_dict: Dict[str, Dict], weights: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    分析投资组合
    
    Args:
        stocks_dict: 股票数据字典，键为股票代码，值为股票信息字典（包含symbol键）
        weights: 投资组合权重列表
        
    Returns:
        dict: 投资组合分析结果
    """
    try:
        # 获取股票数据
        stocks_data = {}
        symbols = []
        
        for symbol, stock_info in stocks_dict.items():
            symbol_code = stock_info.get('symbol', symbol)
            symbols.append(symbol_code)
            stock_data = data_fetcher.get_stock_data(
                symbol_code, 
                period="daily", 
                start_date="20200101", 
                adjust="qfq"
            )
            if not stock_data.empty:
                stocks_data[symbol_code] = stock_data
            else:
                return {"error": f"无法获取股票 {symbol_code} 的数据"}
        
        if not stocks_data:
            return {"error": "无法获取任何股票数据"}
        
        # 调用portfolio模块进行分析
        result = portfolio.analyze_portfolio(stocks_data, weights)
        return result
    except Exception as e:
        return {"error": f"投资组合分析失败: {str(e)}"}


def optimize_portfolio(stocks_dict: Dict[str, Dict], method: str = 'mean_variance') -> Dict[str, Any]:
    """
    优化投资组合
    
    Args:
        stocks_dict: 股票数据字典
        method: 优化方法 ('mean_variance', 'minimum_variance', 'risk_parity')
        
    Returns:
        dict: 投资组合优化结果
    """
    try:
        # 获取股票数据
        stocks_data = {}
        symbols = []
        
        for symbol, stock_info in stocks_dict.items():
            symbol_code = stock_info.get('symbol', symbol)
            symbols.append(symbol_code)
            stock_data = data_fetcher.get_stock_data(
                symbol_code, 
                period="daily", 
                start_date="20200101", 
                adjust="qfq"
            )
            if not stock_data.empty:
                stocks_data[symbol_code] = stock_data
            else:
                return {"error": f"无法获取股票 {symbol_code} 的数据"}
        
        if not stocks_data:
            return {"error": "无法获取任何股票数据"}
        
        # 调用portfolio模块进行优化
        result = portfolio.optimize_portfolio(stocks_data, method)
        return result
    except Exception as e:
        return {"error": f"投资组合优化失败: {str(e)}"}


def monte_carlo_portfolio_simulation(stocks_dict: Dict[str, Dict], n_simulations: int = 10000) -> Dict[str, Any]:
    """
    蒙特卡洛投资组合模拟
    
    Args:
        stocks_dict: 股票数据字典
        n_simulations: 模拟次数
        
    Returns:
        dict: 蒙特卡洛模拟结果
    """
    try:
        # 获取股票数据
        stocks_data = {}
        symbols = []
        
        for symbol, stock_info in stocks_dict.items():
            symbol_code = stock_info.get('symbol', symbol)
            symbols.append(symbol_code)
            stock_data = data_fetcher.get_stock_data(
                symbol_code,
                period="daily",
                start_date="20200101",
                adjust="qfq"
            )
            if not stock_data.empty:
                stocks_data[symbol_code] = stock_data
            else:
                return {"error": f"无法获取股票 {symbol_code} 的数据"}
        
        if not stocks_data:
            return {"error": "无法获取任何股票数据"}
        
        # 构建投资组合
        analyzer = portfolio.PortfolioAnalyzer()
        portfolio_info = analyzer.construct_portfolio(stocks_data)
        
        # 执行蒙特卡洛模拟
        result = analyzer.monte_carlo_simulation(portfolio_info, n_simulations)
        return result
    except Exception as e:
        return {"error": f"蒙特卡洛模拟失败: {str(e)}"}


def run_strategy_backtest(symbol: str, strategy_type: str = "ma_crossover",
                         start_date: str = "20200101", **strategy_params) -> Dict[str, Any]:
    """
    运行策略回测
    
    Args:
        symbol: 股票代码
        strategy_type: 策略类型 ('ma_crossover', 'rsi', 'bollinger', 'momentum', 'mean_reversion')
        start_date: 开始日期
        **strategy_params: 策略参数
        
    Returns:
        dict: 回测结果
    """
    try:
        # 获取股票数据
        stock_data = data_fetcher.get_stock_data(
            symbol,
            period="daily",
            start_date=start_date,
            adjust="qfq"
        )
        
        if stock_data.empty:
            return {"error": f"无法获取股票 {symbol} 的数据"}
        
        # 构造数据字典
        data_dict = {symbol: stock_data}
        
        # 创建策略实例
        if strategy_type == "ma_crossover":
            strategy = backtest.MovingAverageCrossoverStrategy(
                short_window=strategy_params.get("short_window", 20),
                long_window=strategy_params.get("long_window", 50)
            )
        elif strategy_type == "rsi":
            strategy = backtest.RSIStrategy(
                period=strategy_params.get("period", 14),
                overbought=strategy_params.get("overbought", 70),
                oversold=strategy_params.get("oversold", 30)
            )
        elif strategy_type == "bollinger":
            strategy = backtest.BollingerBandsStrategy(
                period=strategy_params.get("period", 20),
                num_std=strategy_params.get("num_std", 2.0)
            )
        elif strategy_type == "momentum":
            strategy = backtest.MomentumStrategy(
                period=strategy_params.get("period", 20)
            )
        elif strategy_type == "mean_reversion":
            strategy = backtest.MeanReversionStrategy(
                period=strategy_params.get("period", 20),
                threshold=strategy_params.get("threshold", 2.0)
            )
        else:
            return {"error": f"不支持的策略类型: {strategy_type}"}
        
        # 运行回测
        result = backtest.run_backtest(data_dict, strategy)
        
        # 生成报告
        report = backtest.generate_backtest_report(result, strategy)
        
        return {
            "success": True,
            "result": result,
            "report": report,
            "strategy_name": strategy.name
        }
    except Exception as e:
        return {"error": f"回测失败: {str(e)}"}


def optimize_strategy_parameters(symbol: str, strategy_type: str = "ma_crossover",
                                start_date: str = "20200101",
                                optimizer_type: str = "grid_search") -> Dict[str, Any]:
    """
    优化策略参数
    
    Args:
        symbol: 股票代码
        strategy_type: 策略类型
        start_date: 开始日期
        optimizer_type: 优化器类型 ('grid_search', 'genetic_algorithm')
        
    Returns:
        dict: 优化结果
    """
    try:
        # 获取股票数据
        stock_data = data_fetcher.get_stock_data(
            symbol,
            period="daily",
            start_date=start_date,
            adjust="qfq"
        )
        
        if stock_data.empty:
            return {"error": f"无法获取股票 {symbol} 的数据"}
        
        # 构造数据字典
        data_dict = {symbol: stock_data}
        
        # 根据策略类型设置参数范围
        if strategy_type == "ma_crossover":
            strategy_class = backtest.MovingAverageCrossoverStrategy
            if optimizer_type == "grid_search":
                param_grid = {
                    "short_window": [5, 10, 20],
                    "long_window": [30, 50, 100]
                }
                result = backtest.grid_search_optimization(
                    data_dict, strategy_class, param_grid
                )
            else:  # genetic_algorithm
                param_ranges = {
                    "short_window": (5, 30),
                    "long_window": (30, 100)
                }
                optimizer = backtest.GeneticAlgorithmOptimizer(
                    data_dict, strategy_class, param_ranges
                )
                result = optimizer.optimize()
        elif strategy_type == "rsi":
            strategy_class = backtest.RSIStrategy
            if optimizer_type == "grid_search":
                param_grid = {
                    "period": [14, 20, 30],
                    "overbought": [70, 80],
                    "oversold": [20, 30]
                }
                result = backtest.grid_search_optimization(
                    data_dict, strategy_class, param_grid
                )
            else:  # genetic_algorithm
                param_ranges = {
                    "period": (10, 30),
                    "overbought": (70, 90),
                    "oversold": (10, 30)
                }
                optimizer = backtest.GeneticAlgorithmOptimizer(
                    data_dict, strategy_class, param_ranges
                )
                result = optimizer.optimize()
        else:
            return {"error": f"不支持的策略类型: {strategy_type}"}
        
        return {
            "success": True,
            "result": result,
            "optimizer_type": optimizer_type,
            "strategy_type": strategy_type
        }
    except Exception as e:
        return {"error": f"参数优化失败: {str(e)}"}


def monte_carlo_strategy_simulation(symbol: str, strategy_type: str = "ma_crossover",
                                  start_date: str = "20200101",
                                  n_simulations: int = 1000) -> Dict[str, Any]:
    """
    策略蒙特卡洛模拟
    
    Args:
        symbol: 股票代码
        strategy_type: 策略类型
        start_date: 开始日期
        n_simulations: 模拟次数
        
    Returns:
        dict: 模拟结果
    """
    try:
        # 获取股票数据
        stock_data = data_fetcher.get_stock_data(
            symbol,
            period="daily",
            start_date=start_date,
            adjust="qfq"
        )
        
        if stock_data.empty:
            return {"error": f"无法获取股票 {symbol} 的数据"}
        
        # 构造数据字典
        data_dict = {symbol: stock_data}
        
        # 创建策略实例
        if strategy_type == "ma_crossover":
            strategy = backtest.MovingAverageCrossoverStrategy()
        elif strategy_type == "rsi":
            strategy = backtest.RSIStrategy()
        else:
            strategy = backtest.MovingAverageCrossoverStrategy()
        
        # 运行回测获取收益率
        result = backtest.run_backtest(data_dict, strategy)
        returns = result['engine'].returns
        
        if len(returns) == 0:
            return {"error": "回测未产生收益率数据"}
        
        # 进行蒙特卡洛模拟
        mc_result = backtest.monte_carlo_simulation(returns, n_simulations)
        
        return {
            "success": True,
            "result": mc_result,
            "strategy_name": strategy.name
        }
    except Exception as e:
        return {"error": f"蒙特卡洛模拟失败: {str(e)}"}


if __name__ == "__main__":
    # 测试代码
    symbol = "002607"
    print(f"预测股票 {symbol} 的价格...")
    
    # 测试不同的模型
    model_types = ['lstm', 'gru', 'transformer', 'rf', 'xgboost']
    
    for model_type in model_types:
        print(f"\n=== 使用 {model_type.upper()} 模型 ===")
        result = predict_stock_price(symbol, model_type=model_type)
        
        if "error" in result:
            print(f"预测失败: {result['error']}")
        else:
            print("\n预测结果:")
            print(f"  股票代码: {result['symbol']}")
            print(f"  股票名称: {result['stock_name']}")
            print(f"  当前价格: {result['current_price']:.2f}元")
            print(f"  预测价格: {result['predicted_price']:.2f}元")
            print(f"  价格变化: {result['price_change']:.2f}元 ({result['price_change_percent']:.2f}%)")
            print(f"  模型类型: {result['model_type']}")