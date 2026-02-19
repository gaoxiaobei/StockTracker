#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新的股票预测演示脚本
使用优化后的StockTracker系统进行演示
"""

import models.predictors as predictor
import data.fetcher as data_fetcher
import analysis.technical as indicators
import models.advanced as advanced_model
import analysis.risk as risk_assessment
import analysis.portfolio as portfolio
import visualization.charts as visualization
import time
import matplotlib.pyplot as plt
import pandas as pd
from performance_optimizer import optimize_tensorflow, model_cache, data_loader, memory_optimizer


def demonstrate_technical_indicators(stock_symbol: str, stock_name: str):
    """
    演示技术指标计算功能
    """
    print("\n4. 技术指标分析...")

    # 获取股票数据
    stock_data = data_fetcher.get_stock_data(
        stock_symbol,
        period="daily",
        start_date="20240101",
        adjust="qfq"
    )

    if stock_data.empty:
        print("   无法获取股票数据，跳过技术指标演示")
        return

    # 只保留最近60天的数据用于演示
    stock_data = stock_data.tail(60)
    print(f"   使用最近 {len(stock_data)} 天的数据进行技术指标分析")

    try:
        # 计算并显示简单移动平均线
        print("\n   (1) 移动平均线 (MA):")
        sma_20 = indicators.simple_moving_average(stock_data, period=20)
        sma_50 = indicators.simple_moving_average(stock_data, period=50)
        print(f"       20日简单移动平均线: {sma_20.iloc[-1]:.2f}")
        print(f"       50日简单移动平均线: {sma_50.iloc[-1]:.2f}")

        # 计算并显示指数移动平均线
        print("\n   (2) 指数移动平均线 (EMA):")
        ema_20 = indicators.exponential_moving_average(stock_data, period=20)
        ema_50 = indicators.exponential_moving_average(stock_data, period=50)
        print(f"       20日指数移动平均线: {ema_20.iloc[-1]:.2f}")
        print(f"       50日指数移动平均线: {ema_50.iloc[-1]:.2f}")

        # 计算并显示相对强弱指数
        print("\n   (3) 相对强弱指数 (RSI):")
        rsi_14 = indicators.relative_strength_index(stock_data, period=14)
        print(f"       14日RSI: {rsi_14.iloc[-1]:.2f}")
        if rsi_14.iloc[-1] > 70:
            print("       RSI > 70: 股票可能超买")
        elif rsi_14.iloc[-1] < 30:
            print("       RSI < 30: 股票可能超卖")
        else:
            print("       RSI在30-70之间: 正常波动范围")

        # 计算并显示MACD
        print("\n   (4) 异同移动平均线 (MACD):")
        macd_data = indicators.moving_average_convergence_divergence(stock_data)
        macd_line = macd_data['macd_line'].iloc[-1]
        signal_line = macd_data['signal_line'].iloc[-1]
        histogram = macd_data['histogram'].iloc[-1]
        print(f"       MACD线: {macd_line:.2f}")
        print(f"       信号线: {signal_line:.2f}")
        print(f"       柱状图: {histogram:.2f}")
        if histogram > 0:
            print("       柱状图为正: 多头市场")
        else:
            print("       柱状图为负: 空头市场")

        # 计算并显示布林带
        print("\n   (5) 布林带 (Bollinger Bands):")
        bb_data = indicators.bollinger_bands(stock_data, period=20)
        upper_band = bb_data['upper_band'].iloc[-1]
        middle_band = bb_data['middle_band'].iloc[-1]
        lower_band = bb_data['lower_band'].iloc[-1]
        current_price = stock_data['close'].iloc[-1]
        print(f"       上轨: {upper_band:.2f}")
        print(f"       中轨: {middle_band:.2f}")
        print(f"       下轨: {lower_band:.2f}")
        print(f"       当前价格: {current_price:.2f}")
        if current_price > upper_band:
            print("       价格突破上轨: 可能超买")
        elif current_price < lower_band:
            print("       价格跌破下轨: 可能超卖")
        else:
            print("       价格在布林带通道内: 正常波动")

        # 计算并显示随机指标
        print("\n   (6) 随机指标 (Stochastic Oscillator):")
        stoch_data = indicators.stochastic_oscillator(stock_data, k_period=14, d_period=3)
        k_percent = stoch_data['k_percent'].iloc[-1]
        d_percent = stoch_data['d_percent'].iloc[-1]
        print(f"       %K: {k_percent:.2f}")
        print(f"       %D: {d_percent:.2f}")
        if k_percent > 80 and d_percent > 80:
            print("       随机指标在超买区: 可能回调")
        elif k_percent < 20 and d_percent < 20:
            print("       随机指标在超卖区: 可能反弹")
        else:
            print("       随机指标在正常区间: 趋势持续")

        # 计算并显示成交量指标
        print("\n   (7) 成交量指标:")
        obv = indicators.on_balance_volume(stock_data)
        vwap = indicators.volume_weighted_average_price(stock_data, period=20)
        cmf = indicators.chaikin_money_flow(stock_data, period=20)
        print(f"       能量潮 (OBV): {obv.iloc[-1]:.0f}")
        print(f"       成交量加权平均价格 (VWAP): {vwap.iloc[-1]:.2f}")
        print(f"       蔡金资金流量 (CMF): {cmf.iloc[-1]:.2f}")
        if cmf.iloc[-1] > 0:
            print("       CMF为正: 资金流入")
        else:
            print("       CMF为负: 资金流出")

    except Exception as e:
        print(f"   技术指标计算出错: {e}")


def demonstrate_risk_assessment(stock_symbol: str, stock_name: str):
    """
    演示风险评估功能
    """
    print("\n7. 风险评估分析...")

    try:
        # 执行综合风险评估
        risk_result = risk_assessment.comprehensive_risk_assessment(
            stock_symbol=stock_symbol,
            market_symbol="sh000001",  # 上证指数
            start_date="20200101"
        )

        if "error" in risk_result:
            print(f"   风险评估失败: {risk_result['error']}")
            return

        # 显示风险指标
        print(f"   股票代码: {risk_result['stock_symbol']}")
        print(f"   市场指数: {risk_result['market_symbol']}")
        print(f"   数据点数: {risk_result['data_points']}")

        print("\n   (1) 风险指标:")
        print(f"       波动率: {risk_result['volatility']:.4f}")
        print(f"       历史VaR (95%置信度): {risk_result['var_historical']:.4f}")
        print(f"       参数VaR (95%置信度): {risk_result['var_parametric']:.4f}")
        print(f"       最大回撤: {risk_result['max_drawdown']:.4f}")
        print(f"       夏普比率: {risk_result['sharpe_ratio']:.4f}")
        print(f"       贝塔系数: {risk_result['beta']:.4f}")
        print(f"       Alpha值: {risk_result['alpha']:.4f}")
        print(f"       与市场相关性: {risk_result['correlation_with_market']:.4f}")

        # 显示风险评级
        risk_level = risk_result['risk_level']
        print("\n   (2) 风险评级:")
        print(f"       风险等级: {risk_level['risk_level']}")
        print(f"       风险解释: {risk_level['explanation']}")
        print(f"       投资建议: {risk_level['investment_advice']}")

        # 显示蒙特卡洛模拟结果
        mc_results = risk_result['monte_carlo_simulation']
        print("\n   (3) 蒙特卡洛模拟:")
        print(f"       预期损失: {mc_results['expected_loss']:.4f}")
        print(f"       VaR 95%: {mc_results['var_95']:.4f}")
        print(f"       VaR 99%: {mc_results['var_99']:.4f}")
        print(f"       最小损失: {mc_results['min_loss']:.4f}")
        print(f"       最大损失: {mc_results['max_loss']:.4f}")
        print(f"       损失标准差: {mc_results['std_loss']:.4f}")

    except Exception as e:
        print(f"   风险评估出错: {e}")


def main():
    # 优化TensorFlow性能
    print("正在优化TensorFlow性能...")
    optimize_tensorflow()
    print("TensorFlow性能优化完成\n")
    
    # 指定要预测的股票代码
    stock_symbol = "002607"  # 中公教育

    print("=" * 60)
    print(f"股票价格预测演示 - 股票代码: {stock_symbol}")
    print("=" * 60)

    # 获取股票基本信息
    print("\n1. 获取股票基本信息...")
    stock_info = data_fetcher.get_stock_info(stock_symbol)
    if stock_info:
        print(f"   股票名称: {stock_info.get('股票简称', '未知')}")
        print(f"   行业: {stock_info.get('行业', '未知')}")
        print(f"   总市值: {stock_info.get('总市值', '未知')}")
        print(f"   流通市值: {stock_info.get('流通市值', '未知')}")
    else:
        print("   无法获取股票基本信息")

    # 预测股票价格 - 模型比较
    print("\n2. 模型性能比较...")
    model_types = ['lstm', 'gru', 'transformer', 'rf', 'xgboost']
    results = {}

    for model_type in model_types:
        print(f"   训练 {model_type.upper()} 模型...")
        start_time = time.time()
        result = predictor.predict_stock_price(stock_symbol, model_type=model_type)
        end_time = time.time()

        if "error" in result:
            print(f"   {model_type.upper()} 模型预测失败: {result['error']}")
            results[model_type] = None
        else:
            print(f"   {model_type.upper()} 模型预测耗时: {end_time - start_time:.2f}秒")
            print(f"   预测价格: {result['predicted_price']:.2f}元")
            print(f"   价格变化: {result['price_change']:.2f}元 ({result['price_change_percent']:.2f}%)")
            results[model_type] = result

    # 显示模型比较结果
    print("\n   模型预测结果比较:")
    print("   {:<12} {:<10} {:<10} {:<10}".format("模型类型", "预测价格", "价格变化", "变化百分比"))
    print("   " + "-" * 50)
    for model_type, result in results.items():
        if result:
            print("   {:<12} {:<10.2f} {:<10.2f} {:<10.2f}%".format(
                model_type.upper(),
                result['predicted_price'],
                result['price_change'],
                result['price_change_percent']
            ))
        else:
            print(f"   {model_type.upper():<12} 失败")

    # 根据预测结果给出建议
    print("\n3. 投资建议:")
    # 使用LSTM模型的结果作为投资建议的依据
    if 'lstm' in results and results['lstm']:
        price_change_percent = float(results['lstm']['price_change_percent'])
        if price_change_percent > 5:
            print("   建议: 强烈买入 - 预测价格上涨超过5%")
        elif price_change_percent > 2:
            print("   建议: 买入 - 预测价格上涨超过2%")
        elif price_change_percent > 0:
            print("   建议: 持有 - 预测价格略有上涨")
        elif price_change_percent > -2:
            print("   建议: 持有 - 预测价格基本持平")
        elif price_change_percent > -5:
            print("   建议: 减持 - 预测价格略有下跌")
        else:
            print("   建议: 卖出 - 预测价格大幅下跌超过5%")
    else:
        print("   无法提供投资建议 - LSTM模型预测失败")

    # 技术指标演示
    stock_name = stock_info.get('股票简称', '未知') if stock_info else '未知'
    demonstrate_technical_indicators(stock_symbol, stock_name)

    # 超参数调优演示
    print("\n5. 超参数调优演示...")
    try:
        # 为随机森林模型进行超参数调优
        print("   为随机森林模型进行超参数调优...")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }

        tuner = advanced_model.HyperparameterTuner('rf', param_grid)
        stock_data = data_fetcher.get_stock_data(
            stock_symbol, period="daily", start_date="20200101", adjust="qfq"
        )

        if not stock_data.empty:
            # 使用较少的折叠数和参数组合以加快演示速度
            tuning_result = tuner.grid_search(stock_data, cv=2)
            print(f"   最佳参数: {tuning_result['best_params']}")
            print(f"   最佳得分: {tuning_result['best_score']:.4f}")
        else:
            print("   无法获取股票数据，跳过超参数调优演示")
    except Exception as e:
        print(f"   超参数调优演示出错: {e}")

    # 模型持久化演示
    print("\n6. 模型持久化演示...")
    try:
        print("   训练并保存模型...")
        # 训练一个简单的模型并保存
        stock_data = data_fetcher.get_stock_data(
            stock_symbol, period="daily", start_date="20200101", adjust="qfq"
        )

        if not stock_data.empty:
            # 创建并训练模型
            model_predictor = advanced_model.AdvancedStockPredictor(
                look_back=30, model_type='rf'
            )
            model_predictor.train(stock_data)

            # 保存模型
            model_name = model_predictor.save_model(f"rf_demo_{stock_symbol}")
            print(f"   模型已保存到: {model_name}")

            # 加载模型
            print("   加载模型...")
            loaded_predictor = advanced_model.AdvancedStockPredictor()
            loaded_predictor.load_model(f"rf_demo_{stock_symbol}")
            print("   模型加载成功")

            # 验证模型
            print("   验证模型...")
            original_pred = model_predictor.predict(stock_data)
            loaded_pred = loaded_predictor.predict(stock_data)
            print(f"   原始模型预测: {original_pred:.2f}")
            print(f"   加载模型预测: {loaded_pred:.2f}")
            print(f"   预测结果一致: {abs(original_pred - loaded_pred) < 0.01}")
        else:
            print("   无法获取股票数据，跳过模型持久化演示")
    except Exception as e:
        print(f"   模型持久化演示出错: {e}")

    # 风险评估演示
    print("\n7. 风险评估演示...")
    try:
        demonstrate_risk_assessment(stock_symbol, stock_name)
    except Exception as e:
        print(f"   风险评估演示出错: {e}")

    # 模型验证和评估演示
    print("\n8. 模型验证和评估演示...")
    try:
        stock_data = data_fetcher.get_stock_data(
            stock_symbol, period="daily", start_date="20200101", adjust="qfq"
        )

        if not stock_data.empty:
            # 使用不同的模型进行评估
            model_types = ['lstm', 'rf']
            for model_type in model_types:
                print(f"   评估 {model_type.upper()} 模型...")
                try:
                    predictor_eval = advanced_model.AdvancedStockPredictor(
                        look_back=30, model_type=model_type
                    )

                    if model_type in ['lstm', 'gru', 'transformer']:
                        predictor_eval.train(stock_data, epochs=10)  # 减少epoch以加快演示
                    else:
                        predictor_eval.train(stock_data)

                    # 评估模型
                    eval_results = predictor_eval.evaluate_model(
                        stock_data, metrics=['mae', 'rmse', 'mape']
                    )
                    print(f"   {model_type.upper()} 模型评估结果:")
                    for metric, value in eval_results.items():
                        print(f"     {metric.upper()}: {value:.4f}")
                except Exception as e:
                    print(f"   {model_type.upper()} 模型评估出错: {e}")
        else:
            print("   无法获取股票数据，跳过模型验证和评估演示")
    except Exception as e:
        print(f"   模型验证和评估演示出错: {e}")

    # 投资组合分析演示
    print("\n9. 投资组合分析演示...")
    try:
        demonstrate_portfolio_analysis()
    except Exception as e:
        print(f"   投资组合分析演示出错: {e}")

    # 投资组合优化演示
    print("\n10. 投资组合优化演示...")
    try:
        demonstrate_portfolio_optimization()
    except Exception as e:
        print(f"   投资组合优化演示出错: {e}")

    # 蒙特卡洛模拟演示
    print("\n11. 蒙特卡洛模拟演示...")
    try:
        demonstrate_monte_carlo_simulation()
    except Exception as e:
        print(f"   蒙特卡洛模拟演示出错: {e}")

    # 回测演示
    print("\n12. 回测演示...")
    try:
        demonstrate_backtest()
    except Exception as e:
        print(f"   回测演示出错: {e}")

    # 参数优化演示
    print("\n13. 参数优化演示...")
    try:
        demonstrate_parameter_optimization()
    except Exception as e:
        print(f"   参数优化演示出错: {e}")

    # 新增的可视化功能演示
    demonstrate_interactive_charts(stock_symbol, stock_name)
    demonstrate_candlestick_chart(stock_symbol, stock_name)
    demonstrate_technical_indicators_overlay(stock_symbol, stock_name)
    demonstrate_correlation_heatmap()
    demonstrate_3d_risk_return()
    demonstrate_animated_price_chart(stock_symbol, stock_name)
    demonstrate_prediction_with_confidence_interval(stock_symbol, stock_name)
    demonstrate_model_comparison_chart(stock_symbol, stock_name)
    demonstrate_risk_metrics_chart(stock_symbol, stock_name)

    # 新增的高级可视化功能演示
    demonstrate_comprehensive_dashboard(stock_symbol, stock_name)
    demonstrate_multi_stock_comparison()
    demonstrate_portfolio_analysis_visualization()
    demonstrate_backtest_results_visualization()
    demonstrate_realtime_visualization(stock_symbol, stock_name)

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


def demonstrate_portfolio_analysis():
    """
    演示投资组合分析功能
    """
    print("\n9. 投资组合分析...")

    # 定义股票组合
    stocks_dict = {
        "002607": {"symbol": "002607", "name": "中公教育"},
        "000001": {"symbol": "000001", "name": "平安银行"},
        "600036": {"symbol": "600036", "name": "招商银行"}
    }

    try:
        # 分析投资组合
        print("   分析投资组合...")
        portfolio_result = predictor.analyze_portfolio(stocks_dict)

        if "error" in portfolio_result:
            print(f"   投资组合分析失败: {portfolio_result['error']}")
            return

        if portfolio_result["success"]:
            metrics = portfolio_result["metrics"]
            print(f"   投资组合预期收益: {metrics['expected_return']:.4f}")
            print(f"   投资组合风险(波动率): {metrics['volatility']:.4f}")
            print(f"   夏普比率: {metrics['sharpe_ratio']:.4f}")

            # 风险贡献分析
            risk_contribution = portfolio_result["risk_contribution"]
            if "error" not in risk_contribution:
                print("\n   风险贡献分析:")
                for i, symbol in enumerate(risk_contribution["symbols"]):
                    print(f"     {symbol}: {risk_contribution['percentage_contributions'][i]:.2f}%")
        else:
            print("   投资组合分析失败")

    except Exception as e:
        print(f"   投资组合分析出错: {e}")


def demonstrate_portfolio_optimization():
    """
    演示投资组合优化功能
    """
    print("\n10. 投资组合优化...")

    # 定义股票组合
    stocks_dict = {
        "002607": {"symbol": "002607", "name": "中公教育"},
        "000001": {"symbol": "000001", "name": "平安银行"},
        "600036": {"symbol": "600036", "name": "招商银行"}
    }

    try:
        # 均值-方差优化
        print("   均值-方差优化...")
        mv_result = predictor.optimize_portfolio(stocks_dict, method='mean_variance')

        if "error" in mv_result:
            print(f"   均值-方差优化失败: {mv_result['error']}")
        elif mv_result["success"]:
            print(f"   优化后预期收益: {mv_result['expected_return']:.4f}")
            print(f"   优化后风险(波动率): {mv_result['volatility']:.4f}")
            print(f"   优化后夏普比率: {mv_result['sharpe_ratio']:.4f}")
        else:
            print("   均值-方差优化失败")

        # 最小方差组合优化
        print("\n   最小方差组合优化...")
        min_var_result = predictor.optimize_portfolio(stocks_dict, method='minimum_variance')

        if "error" in min_var_result:
            print(f"   最小方差组合优化失败: {min_var_result['error']}")
        elif min_var_result["success"]:
            print(f"   优化后预期收益: {min_var_result['expected_return']:.4f}")
            print(f"   优化后风险(波动率): {min_var_result['volatility']:.4f}")
            print(f"   优化后夏普比率: {min_var_result['sharpe_ratio']:.4f}")
        else:
            print("   最小方差组合优化失败")

    except Exception as e:
        print(f"   投资组合优化出错: {e}")


def demonstrate_monte_carlo_simulation():
    """
    演示蒙特卡洛模拟功能
    """
    print("\n11. 蒙特卡洛模拟...")

    # 定义股票组合
    stocks_dict = {
        "002607": {"symbol": "002607", "name": "中公教育"},
        "000001": {"symbol": "000001", "name": "平安银行"},
        "600036": {"symbol": "600036", "name": "招商银行"}
    }

    try:
        # 蒙特卡洛模拟
        print("   执行蒙特卡洛模拟...")
        mc_result = predictor.monte_carlo_portfolio_simulation(stocks_dict, n_simulations=1000)

        if "error" in mc_result:
            print(f"   蒙特卡洛模拟失败: {mc_result['error']}")
        else:
            print(f"   模拟次数: {mc_result['n_simulations']}")
            print(f"   最大夏普比率: {mc_result['max_sharpe_ratio']:.4f}")
            print(f"   对应预期收益: {mc_result['return_for_max_sharpe']:.4f}")
            print(f"   对应风险(波动率): {mc_result['volatility_for_max_sharpe']:.4f}")
            print(f"   最小波动率: {mc_result['min_volatility']:.4f}")

    except Exception as e:
        print(f"   蒙特卡洛模拟出错: {e}")


def demonstrate_backtest():
    """
    演示回测功能
    """
    print("\n12. 策略回测...")

    symbol = "002607"
    stock_name = "中公教育"

    try:
        # 运行移动平均线交叉策略回测
        print(f"   运行 {symbol} ({stock_name}) 的移动平均线交叉策略回测...")
        result = predictor.run_strategy_backtest(
            symbol,
            strategy_type="ma_crossover",
            short_window=10,
            long_window=30
        )

        if "error" in result:
            print(f"   回测失败: {result['error']}")
        elif result["success"]:
            metrics = result["result"]["metrics"]
            print(f"   策略名称: {result['strategy_name']}")
            print(f"   累计收益: {metrics.get('cumulative_return', 0)*100:.2f}%")
            print(f"   年化收益: {metrics.get('annualized_return', 0)*100:.2f}%")
            print(f"   夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"   总交易次数: {len(result['result']['engine'].trades)}")
        else:
            print("   回测失败")

    except Exception as e:
        print(f"   回测出错: {e}")


def demonstrate_parameter_optimization():
    """
    演示参数优化功能
    """
    print("\n13. 参数优化...")

    symbol = "002607"
    stock_name = "中公教育"

    try:
        # 网格搜索优化参数
        print(f"   使用网格搜索优化 {symbol} ({stock_name}) 的策略参数...")
        result = predictor.optimize_strategy_parameters(
            symbol,
            strategy_type="ma_crossover",
            optimizer_type="grid_search"
        )

        if "error" in result:
            print(f"   参数优化失败: {result['error']}")
        elif result["success"]:
            best_params = result["result"]["best_params"]
            best_sharpe = result["result"]["best_sharpe_ratio"]
            print(f"   最佳参数: {best_params}")
            print(f"   最佳夏普比率: {best_sharpe:.4f}")
        else:
            print("   参数优化失败")

    except Exception as e:
       print(f"   参数优化出错: {e}")

# 新增的可视化功能演示函数
def demonstrate_interactive_charts(stock_symbol: str, stock_name: str):
   """
   演示交互式图表功能
   """
   print("\n14. 交互式图表演示...")
   try:
       print(f"   绘制 {stock_name} ({stock_symbol}) 的交互式价格图表...")
       predictor.plot_interactive_stock_chart(stock_symbol, days=60)
       print("   交互式图表绘制完成")
   except Exception as e:
       print(f"   交互式图表演示出错: {e}")


def demonstrate_candlestick_chart(stock_symbol: str, stock_name: str):
   """
   演示K线图功能
   """
   print("\n15. K线图演示...")
   try:
       print(f"   绘制 {stock_name} ({stock_symbol}) 的K线图...")
       predictor.plot_candlestick_chart(stock_symbol, days=60)
       print("   K线图绘制完成")
   except Exception as e:
       print(f"   K线图演示出错: {e}")


def demonstrate_technical_indicators_overlay(stock_symbol: str, stock_name: str):
   """
   演示技术指标叠加图功能
   """
   print("\n16. 技术指标叠加图演示...")
   try:
       print(f"   绘制 {stock_name} ({stock_symbol}) 的技术指标叠加图...")
       predictor.plot_technical_indicators_chart(stock_symbol, days=90)
       print("   技术指标叠加图绘制完成")
   except Exception as e:
       print(f"   技术指标叠加图演示出错: {e}")


def demonstrate_correlation_heatmap():
    """
    演示相关性热力图功能
    """
    print("\n17. 相关性热力图演示...")
    try:
        symbols = ["002607", "000001", "600036"]  # 中公教育, 平安银行, 招商银行
        print(f"   绘制股票 {symbols} 的相关性热力图...")
        # 先绘制普通相关性热力图
        predictor.plot_stock_correlation_heatmap(symbols)
        # 再绘制聚类相关性热力图
        print(f"   绘制股票 {symbols} 的聚类相关性热力图...")
        predictor.plot_stock_correlation_heatmap(symbols, cluster=True)
        print("   相关性热力图绘制完成")
    except Exception as e:
        print(f"   相关性热力图演示出错: {e}")


def demonstrate_3d_risk_return():
    """
    演示3D风险-收益可视化功能
    """
    print("\n18. 3D风险-收益可视化演示...")
    try:
        # 创建示例数据
        portfolio_results = {
            'return': [0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25],
            'risk': [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28],
            'sharpe_ratio': [0.5, 0.67, 0.8, 0.83, 0.9, 0.91, 0.88, 0.89],
            'sortino_ratio': [0.6, 0.75, 0.9, 0.95, 1.0, 1.05, 1.02, 1.01]
        }
        print("   绘制3D风险-收益可视化图...")
        predictor.plot_3d_risk_return_visualization(portfolio_results)
        print("   3D风险-收益可视化绘制完成")
    except Exception as e:
        print(f"   3D风险-收益可视化演示出错: {e}")


def demonstrate_animated_price_chart(stock_symbol: str, stock_name: str):
    """
    演示价格变化动画功能
    """
    print("\n19. 价格变化动画演示...")
    try:
        print(f"   绘制 {stock_name} ({stock_symbol}) 的价格变化动画...")
        # 先绘制普通动画图
        predictor.plot_animated_price_chart(stock_symbol, days=30)
        # 再绘制带成交量的动画图
        print(f"   绘制 {stock_name} ({stock_symbol}) 的带成交量价格变化动画...")
        predictor.plot_animated_price_chart(stock_symbol, days=30, show_volume=True)
        print("   价格变化动画绘制完成")
    except Exception as e:
        print(f"   价格变化动画演示出错: {e}")


def demonstrate_prediction_with_confidence_interval(stock_symbol: str, stock_name: str):
   """
   演示带置信区间的预测结果图功能
   """
   print("\n20. 带置信区间的预测结果图演示...")
   try:
       print(f"   绘制 {stock_name} ({stock_symbol}) 的带置信区间预测图...")
       predictor.plot_prediction_with_confidence_interval(stock_symbol, model_type='lstm')
       print("   带置信区间的预测图绘制完成")
   except Exception as e:
       print(f"   带置信区间的预测图演示出错: {e}")


def demonstrate_model_comparison_chart(stock_symbol: str, stock_name: str):
   """
   演示模型性能对比图功能
   """
   print("\n21. 模型性能对比图演示...")
   try:
       print(f"   绘制 {stock_name} ({stock_symbol}) 的模型性能对比图...")
       predictor.plot_model_comparison_chart(stock_symbol)
       print("   模型性能对比图绘制完成")
   except Exception as e:
       print(f"   模型性能对比图演示出错: {e}")


def demonstrate_risk_metrics_chart(stock_symbol: str, stock_name: str):
    """
    演示风险指标可视化图功能
    """
    print("\n22. 风险指标可视化图演示...")
    try:
        print(f"   绘制 {stock_name} ({stock_symbol}) 的风险指标可视化图...")
        predictor.plot_risk_metrics_chart(stock_symbol)
        print("   风险指标可视化图绘制完成")
    except Exception as e:
        print(f"   风险指标可视化图演示出错: {e}")


def demonstrate_comprehensive_dashboard(stock_symbol: str, stock_name: str):
    """
    演示综合仪表板功能
    """
    print("\n23. 综合仪表板演示...")
    try:
        print(f"   创建 {stock_name} ({stock_symbol}) 的综合仪表板...")
        predictor.create_comprehensive_dashboard(stock_symbol)
        print("   综合仪表板创建完成")
    except Exception as e:
        print(f"   综合仪表板演示出错: {e}")


def demonstrate_multi_stock_comparison():
    """
    演示多股票比较视图功能
    """
    print("\n24. 多股票比较视图演示...")
    try:
        symbols = ["002607", "000001", "600036"]  # 中公教育, 平安银行, 招商银行
        names = ["中公教育", "平安银行", "招商银行"]
        print(f"   绘制股票 {', '.join(names)} 的多股票比较图...")
        predictor.plot_multi_stock_comparison(symbols)
        print("   多股票比较图绘制完成")
    except Exception as e:
        print(f"   多股票比较视图演示出错: {e}")


def demonstrate_portfolio_analysis_visualization():
    """
    演示投资组合分析可视化功能
    """
    print("\n25. 投资组合分析可视化演示...")
    try:
        # 定义股票组合
        stocks_dict = {
            "002607": {"symbol": "002607", "name": "中公教育"},
            "000001": {"symbol": "000001", "name": "平安银行"},
            "600036": {"symbol": "600036", "name": "招商银行"}
        }
        print("   绘制投资组合分析图...")
        predictor.plot_portfolio_analysis_chart(stocks_dict)
        print("   投资组合分析图绘制完成")
    except Exception as e:
        print(f"   投资组合分析可视化演示出错: {e}")


def demonstrate_backtest_results_visualization():
    """
    演示回测结果可视化功能
    """
    print("\n26. 回测结果可视化演示...")
    try:
        symbol = "002607"
        stock_name = "中公教育"
        print(f"   绘制 {stock_name} ({symbol}) 的回测结果图...")
        # 运行一个简单的回测以获取结果
        backtest_result = predictor.run_strategy_backtest(
            symbol,
            strategy_type="ma_crossover",
            short_window=10,
            long_window=30
        )
        if "error" not in backtest_result and backtest_result["success"]:
            predictor.plot_backtest_results_chart(backtest_result["result"])
            print("   回测结果图绘制完成")
        else:
            print("   回测执行失败，无法绘制结果图")
    except Exception as e:
        print(f"   回测结果可视化演示出错: {e}")


def demonstrate_realtime_visualization(stock_symbol: str, stock_name: str):
    """
    演示实时数据可视化功能
    """
    print("\n27. 实时数据可视化演示...")
    try:
        print(f"   初始化 {stock_name} ({stock_symbol}) 的实时数据可视化...")
        # 初始化实时图表
        predictor.initialize_realtime_chart(stock_symbol)
        print("   实时数据可视化初始化完成")
        print("   注意: 实时更新需要连接到数据源，此处仅演示初始化过程")
    except Exception as e:
        print(f"   实时数据可视化演示出错: {e}")


if __name__ == "__main__":
    main()