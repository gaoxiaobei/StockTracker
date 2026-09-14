#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockTracker - 股票价格预测系统 主入口点
"""

import sys
import argparse
import time

# Import our modules
from data import fetcher as data_fetcher
from models import predictors as predictor
from analysis import technical as indicators
from analysis import risk as risk_module
from analysis import portfolio as portfolio_module
from analysis import backtest as backtest_module

# 导入性能优化模块
from performance_optimizer import optimize_tensorflow, memory_optimizer


def show_welcome():
    """显示欢迎信息"""
    print("=" * 60)
    print("📈 StockTracker - 股票价格预测系统")
    print("=" * 60)
    print("基于机器学习的股票分析和预测工具")
    print("支持多种模型: LSTM, GRU, Transformer, 随机森林, XGBoost")
    print("提供技术指标分析、风险评估、投资组合优化等功能")
    print()


def show_menu():
    """显示主菜单"""
    print("请选择功能:")
    print("1. 股票价格预测")
    print("2. 技术指标分析")
    print("3. 风险评估")
    print("4. 投资组合分析")
    print("5. 策略回测")
    print("6. 启动Web界面")
    print("0. 退出")
    print()


def predict_stock_price():
    """股票价格预测功能"""
    print("=== 股票价格预测 ===")
    symbol = input("请输入股票代码 (例如: 002607): ").strip()
    if not symbol:
        print("股票代码不能为空")
        return
    
    print("可选模型:")
    print("1. LSTM")
    print("2. GRU")
    print("3. Transformer")
    print("4. 随机森林")
    print("5. XGBoost")
    
    model_choice = input("请选择模型 (默认为LSTM): ").strip()
    model_map = {
        "1": "lstm",
        "2": "gru",
        "3": "transformer",
        "4": "rf",
        "5": "xgboost"
    }
    
    model_type = model_map.get(model_choice, "lstm")
    
    try:
        print(f"正在使用 {model_type.upper()} 模型预测 {symbol} 的价格...")
        result = predictor.predict_stock_price(symbol, model_type=model_type)
        
        if "error" not in result:
            print("\n预测结果:")
            print(f"  股票代码: {result['symbol']}")
            print(f"  股票名称: {result['stock_name']}")
            print(f"  当前价格: {result['current_price']:.2f}元")
            print(f"  预测价格: {result['predicted_price']:.2f}元")
            print(f"  价格变化: {result['price_change']:.2f}元 ({result['price_change_percent']:.2f}%)")
            
            # 投资建议
            if result['price_change_percent'] > 5:
                print("  投资建议: 📈 强烈买入 - 预测价格上涨超过5%")
            elif result['price_change_percent'] > 2:
                print("  投资建议: 📈 买入 - 预测价格上涨超过2%")
            elif result['price_change_percent'] > 0:
                print("  投资建议: ➡️ 持有 - 预测价格略有上涨")
            elif result['price_change_percent'] > -2:
                print("  投资建议: ↔️ 持有 - 预测价格基本持平")
            elif result['price_change_percent'] > -5:
                print("  投资建议: 📉 减持 - 预测价格略有下跌")
            else:
                print("  投资建议: 🚨 卖出 - 预测价格大幅下跌超过5%")
        else:
            print(f"预测失败: {result['error']}")
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")


def technical_analysis():
    """技术指标分析功能"""
    print("=== 技术指标分析 ===")
    symbol = input("请输入股票代码 (例如: 002607): ").strip()
    if not symbol:
        print("股票代码不能为空")
        return
    
    try:
        print(f"正在获取 {symbol} 的数据...")
        stock_data = data_fetcher.get_stock_data(symbol, period="daily", start_date="20240101", adjust="qfq")
        
        if stock_data.empty:
            print("无法获取股票数据")
            return
        
        print("计算技术指标...")
        sma_20 = indicators.simple_moving_average(stock_data, period=20)
        rsi = indicators.relative_strength_index(stock_data, period=14)
        macd_data = indicators.moving_average_convergence_divergence(stock_data)
        
        print("\n技术指标结果:")
        print(f"  最新收盘价: {stock_data['close'].iloc[-1]:.2f}")
        print(f"  20日简单移动平均线: {sma_20.iloc[-1]:.2f}")
        print(f"  14日相对强弱指数: {rsi.iloc[-1]:.2f}")
        print(f"  MACD: {macd_data['macd_line'].iloc[-1]:.2f}")
        print(f"  MACD信号线: {macd_data['signal_line'].iloc[-1]:.2f}")
        
    except Exception as e:
        print(f"技术指标分析过程中出错: {str(e)}")


def run_risk_assessment():
    """风险评估功能"""
    print("=== 风险评估 ===")
    symbol = input("请输入股票代码 (例如: 002607): ").strip()
    if not symbol:
        print("股票代码不能为空")
        return
    
    try:
        print(f"正在评估 {symbol} 的风险...")
        result = predictor.assess_stock_risk(symbol)
        
        if "error" not in result:
            print("\n风险评估结果:")
            print(f"  波动率: {result['volatility']:.4f}")
            print(f"  历史VaR(95%): {result['var_historical']:.4f}")
            print(f"  最大回撤: {result['max_drawdown']:.4f}")
            print(f"  夏普比率: {result['sharpe_ratio']:.4f}")
            print(f"  贝塔系数: {result['beta']:.4f}")
            print(f"  Alpha值: {result['alpha']:.4f}")
            
            risk_level = result['risk_level']
            print(f"\n风险等级: {risk_level['risk_level']}")
            print(f"风险解释: {risk_level['explanation']}")
            print(f"投资建议: {risk_level['investment_advice']}")
        else:
            print(f"风险评估失败: {result['error']}")
    except Exception as e:
        print(f"风险评估过程中出错: {str(e)}")


def run_portfolio_analysis():
    """投资组合分析功能"""
    print("=== 投资组合分析 ===")
    print("请输入投资组合中的股票代码和权重:")
    print("(例如: 002607,0.4)")
    print("输入空行结束输入")
    
    stocks_dict = {}
    weights = []
    
    while True:
        line = input("股票代码和权重 (例如: 002607,0.4): ").strip()
        if not line:
            break
        
        try:
            symbol, weight = line.split(',')
            symbol = symbol.strip()
            weight = float(weight.strip())
            
            # 获取股票名称
            stock_info = data_fetcher.get_stock_info(symbol)
            stock_name = stock_info.get("股票简称", symbol) if stock_info else symbol
            
            stocks_dict[symbol] = {"symbol": symbol, "name": stock_name}
            weights.append(weight)
        except Exception as e:
            print(f"输入格式错误: {str(e)}")
            continue
    
    if not stocks_dict:
        print("投资组合不能为空")
        return
    
    # 检查权重和是否为1
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 0.001:
        print(f"警告: 权重和为 {weight_sum:.3f}，不等于1.0")
        normalize = input("是否自动归一化权重? (y/n): ").strip().lower()
        if normalize == 'y':
            weights = [w / weight_sum for w in weights]
            print("权重已归一化")
    
    try:
        print("正在分析投资组合...")
        result = predictor.analyze_portfolio(stocks_dict, weights)
        
        if "error" not in result and result.get("success"):
            metrics = result["metrics"]
            print("\n投资组合分析结果:")
            print(f"  预期收益: {metrics['expected_return']:.4f}")
            print(f"  风险(波动率): {metrics['volatility']:.4f}")
            print(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
            
            # 风险贡献分析
            risk_contrib = result["risk_contribution"]
            if "error" not in risk_contrib:
                print("\n风险贡献分析:")
                for i, symbol in enumerate(risk_contrib["symbols"]):
                    percentage = risk_contrib["percentage_contributions"][i]
                    print(f"  {symbol}: {percentage:.2f}%")
        else:
            print(f"投资组合分析失败: {result.get('error', '未知错误')}")
    except Exception as e:
        print(f"投资组合分析过程中出错: {str(e)}")


def backtest_strategy():
    """策略回测功能"""
    print("=== 策略回测 ===")
    symbol = input("请输入股票代码 (例如: 002607): ").strip()
    if not symbol:
        print("股票代码不能为空")
        return
    
    print("可选策略:")
    print("1. 移动平均线交叉策略")
    print("2. RSI超买超卖策略")
    
    strategy_choice = input("请选择策略 (默认为移动平均线交叉策略): ").strip()
    
    try:
        if strategy_choice == "2":
            print("正在运行RSI超买超卖策略回测...")
            result = predictor.run_strategy_backtest(
                symbol, 
                strategy_type="rsi",
                period=14,
                overbought=70,
                oversold=30
            )
        else:
            print("正在运行移动平均线交叉策略回测...")
            result = predictor.run_strategy_backtest(
                symbol, 
                strategy_type="ma_crossover",
                short_window=20,
                long_window=50
            )
        
        if "error" not in result and result.get("success"):
            metrics = result["result"]["metrics"]
            print("\n回测结果:")
            print(f"  累计收益: {metrics.get('cumulative_return', 0)*100:.2f}%")
            print(f"  年化收益: {metrics.get('annualized_return', 0)*100:.2f}%")
            print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"  交易次数: {len(result['result']['engine'].trades)}")
        else:
            print(f"回测失败: {result.get('error', '未知错误')}")
    except Exception as e:
        print(f"回测过程中出错: {str(e)}")


def launch_web_interface():
    """启动Web界面"""
    print("=== 启动Web界面 ===")
    print("正在启动StockTracker Web界面...")
    print("请在浏览器中访问: http://localhost:8501")
    print("按 Ctrl+C 停止服务")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nWeb服务已停止")
    except Exception as e:
        print(f"启动Web界面失败: {str(e)}")
        print("请确保已安装streamlit: pip install streamlit")


def main():
    """主函数"""
    # 优化TensorFlow性能
    print("正在优化TensorFlow性能...")
    optimize_tensorflow()
    print("TensorFlow性能优化完成\n")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="StockTracker - 股票价格预测系统")
    parser.add_argument("--symbol", help="股票代码")
    parser.add_argument("--function", choices=["predict", "tech", "risk", "portfolio", "backtest", "web"],
                       help="功能选择")

    args = parser.parse_args()

    # 如果提供了命令行参数，直接执行相应功能
    if args.symbol and args.function:
        if args.function == "predict":
            start_time = time.time()
            result = predictor.predict_stock_price(args.symbol)
            end_time = time.time()
            print(result)
            print(f"预测耗时: {end_time - start_time:.2f}秒")
            return
        elif args.function == "tech":
            start_time = time.time()
            # 获取股票数据
            stock_data = data_fetcher.get_stock_data(args.symbol, period="daily", start_date="20240101", adjust="qfq")
            if not stock_data.empty:
                # 计算技术指标
                sma_20 = indicators.simple_moving_average(stock_data, period=20)
                rsi = indicators.relative_strength_index(stock_data, period=14)
                macd = indicators.moving_average_convergence_divergence(stock_data)

                print(f"SMA20: {sma_20.iloc[-1]:.2f}")
                print(f"RSI: {rsi.iloc[-1]:.2f}")
                print(f"MACD: {macd['macd_line'].iloc[-1]:.2f}")
            else:
                print("无法获取股票数据")
            end_time = time.time()
            print(f"技术指标分析耗时: {end_time - start_time:.2f}秒")
            return
        elif args.function == "risk":
            start_time = time.time()
            result = predictor.assess_stock_risk(args.symbol)
            end_time = time.time()
            print(result)
            print(f"风险评估耗时: {end_time - start_time:.2f}秒")
            return
        elif args.function == "web":
            launch_web_interface()
            return
        elif args.function == "portfolio":
            start_time = time.time()
            # 这里可以添加投资组合分析的命令行接口
            print("投资组合分析功能")
            end_time = time.time()
            print(f"投资组合分析耗时: {end_time - start_time:.2f}秒")
            return
        elif args.function == "backtest":
            start_time = time.time()
            # 这里可以添加策略回测的命令行接口
            print("策略回测功能")
            end_time = time.time()
            print(f"策略回测耗时: {end_time - start_time:.2f}秒")
            return
        # 其他功能类似处理

    # 交互式模式
    show_welcome()

    while True:
        show_menu()
        choice = input("请选择功能 (0-6): ").strip()

        if choice == "0":
            print("感谢使用StockTracker，再见！")
            # 清理内存资源
            memory_optimizer.clear_session()
            break
        elif choice == "1":
            start_time = time.time()
            predict_stock_price()
            end_time = time.time()
            print(f"股票价格预测耗时: {end_time - start_time:.2f}秒")
        elif choice == "2":
            start_time = time.time()
            technical_analysis()
            end_time = time.time()
            print(f"技术指标分析耗时: {end_time - start_time:.2f}秒")
        elif choice == "3":
            start_time = time.time()
            run_risk_assessment()
            end_time = time.time()
            print(f"风险评估耗时: {end_time - start_time:.2f}秒")
        elif choice == "4":
            start_time = time.time()
            run_portfolio_analysis()
            end_time = time.time()
            print(f"投资组合分析耗时: {end_time - start_time:.2f}秒")
        elif choice == "5":
            start_time = time.time()
            backtest_strategy()
            end_time = time.time()
            print(f"策略回测耗时: {end_time - start_time:.2f}秒")
        elif choice == "6":
            launch_web_interface()
        else:
            print("无效选择，请重新输入")

        input("\n按回车键继续...")


if __name__ == "__main__":
    main()
