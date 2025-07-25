#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockTracker 综合测试脚本
全面测试StockTracker的所有功能，确保之前的错误已修复
"""

import json
import numpy as np
import pandas as pd
import warnings
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加当前目录到Python路径
sys.path.append('.')

# 导入项目模块
import data.fetcher as data_fetcher
import models.predictors as predictor
import analysis.technical as indicators
import analysis.risk as risk_assessment
import analysis.portfolio as portfolio
import analysis.backtest as backtest
import visualization.charts as visualization
import models.advanced as advanced_model

# 忽略警告信息
warnings.filterwarnings('ignore')

class StockTrackerTestSuite:
    """StockTracker综合测试套件"""
    
    def __init__(self):
        """初始化测试套件"""
        self.test_results = []
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """
        记录测试结果
        
        Args:
            test_name: 测试名称
            success: 是否成功
            details: 详细信息
        """
        result = {
            'test_name': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        self.test_results.append(result)
        self.report['test_details'].append(result)
        
        # 更新统计信息
        self.report['total_tests'] += 1
        if success:
            self.report['passed_tests'] += 1
            print(f"✓ {test_name}: 通过")
        else:
            self.report['failed_tests'] += 1
            print(f"✗ {test_name}: 失败 - {details}")
    
    def test_data_fetching(self):
        """测试数据获取功能"""
        print("\n=== 测试数据获取功能 ===")
        
        try:
            # 测试获取股票数据
            stock_data = data_fetcher.get_stock_data(
                '000001', 
                period='daily', 
                start_date='20240101', 
                adjust='qfq'
            )
            
            if not stock_data.empty:
                # 检查必要的列是否存在
                required_columns = ['close', 'open', 'high', 'low', 'volume']
                missing_columns = [col for col in required_columns if col not in stock_data.columns]
                
                if not missing_columns:
                    self.log_result("数据获取功能", True, f"成功获取{len(stock_data)}条记录")
                else:
                    self.log_result("数据获取功能", False, f"缺少必要列: {missing_columns}")
            else:
                self.log_result("数据获取功能", False, "返回空数据")
                
        except Exception as e:
            self.log_result("数据获取功能", False, f"异常: {str(e)}")
        
        try:
            # 测试获取股票信息
            stock_info = data_fetcher.get_stock_info('000001')
            
            if stock_info and isinstance(stock_info, dict):
                self.log_result("股票信息获取功能", True, f"成功获取股票信息，包含{len(stock_info)}个字段")
            else:
                self.log_result("股票信息获取功能", False, "返回数据格式不正确")
                
        except Exception as e:
            self.log_result("股票信息获取功能", False, f"异常: {str(e)}")
    
    def test_price_prediction(self):
        """测试股票价格预测功能"""
        print("\n=== 测试股票价格预测功能 ===")
        
        # 测试不同的模型
        model_types = ['lstm', 'gru', 'transformer', 'rf', 'xgboost']
        
        for model_type in model_types:
            try:
                print(f"  测试 {model_type.upper()} 模型...")
                result = predictor.predict_stock_price('000001', days=1, model_type=model_type)
                
                if isinstance(result, dict) and 'error' not in result:
                    self.log_result(f"{model_type.upper()}预测模型", True, 
                                  f"预测价格: {result['predicted_price']:.2f}")
                else:
                    error_msg = result.get('error', '未知错误') if isinstance(result, dict) else '返回格式错误'
                    self.log_result(f"{model_type.upper()}预测模型", False, error_msg)
                    
            except Exception as e:
                self.log_result(f"{model_type.upper()}预测模型", False, f"异常: {str(e)}")
    
    def test_technical_indicators(self):
        """测试技术指标分析功能"""
        print("\n=== 测试技术指标分析功能 ===")
        
        try:
            # 获取测试数据
            stock_data = data_fetcher.get_stock_data(
                '000001', 
                period='daily', 
                start_date='20240101', 
                adjust='qfq'
            )
            
            if stock_data.empty:
                self.log_result("技术指标分析功能", False, "无法获取测试数据")
                return
                
            # 只使用最近30天的数据以加快测试
            stock_data = stock_data.tail(30)
            
            # 测试各种技术指标
            indicators_to_test = [
                ('简单移动平均线', lambda: indicators.simple_moving_average(stock_data, period=10)),
                ('指数移动平均线', lambda: indicators.exponential_moving_average(stock_data, period=10)),
                ('相对强弱指数', lambda: indicators.relative_strength_index(stock_data, period=14)),
                ('MACD', lambda: indicators.moving_average_convergence_divergence(stock_data)),
                ('布林带', lambda: indicators.bollinger_bands(stock_data, period=20)),
                ('随机指标', lambda: indicators.stochastic_oscillator(stock_data)),
                ('能量潮指标', lambda: indicators.on_balance_volume(stock_data)),
                ('成交量加权平均价格', lambda: indicators.volume_weighted_average_price(stock_data, period=10)),
                ('蔡金资金流量', lambda: indicators.chaikin_money_flow(stock_data, period=10))
            ]
            
            for indicator_name, indicator_func in indicators_to_test:
                try:
                    result = indicator_func()
                    if result is not None and not result.empty:
                        self.log_result(f"{indicator_name}", True, f"计算成功，返回{len(result)}个数据点")
                    else:
                        self.log_result(f"{indicator_name}", False, "返回空结果")
                except Exception as e:
                    self.log_result(f"{indicator_name}", False, f"计算异常: {str(e)}")
                    
        except Exception as e:
            self.log_result("技术指标分析功能", False, f"异常: {str(e)}")
    
    def test_risk_assessment(self):
        """测试风险评估功能"""
        print("\n=== 测试风险评估功能 ===")
        
        try:
            # 执行综合风险评估
            risk_result = risk_assessment.comprehensive_risk_assessment(
                stock_symbol='000001',
                market_symbol='sh000001',
                start_date='20240101'
            )
            
            if 'error' in risk_result:
                self.log_result("风险评估功能", False, risk_result['error'])
            else:
                # 检查必要的风险指标
                required_metrics = [
                    'volatility', 'var_historical', 'max_drawdown', 
                    'sharpe_ratio', 'beta', 'alpha'
                ]
                
                missing_metrics = [metric for metric in required_metrics if metric not in risk_result]
                
                if not missing_metrics:
                    self.log_result("风险评估功能", True, 
                                  f"成功计算{len(required_metrics)}个风险指标")
                else:
                    self.log_result("风险评估功能", False, f"缺少指标: {missing_metrics}")
                    
        except Exception as e:
            self.log_result("风险评估功能", False, f"异常: {str(e)}")
    
    def test_portfolio_analysis(self):
        """测试投资组合分析功能"""
        print("\n=== 测试投资组合分析功能 ===")
        
        # 定义测试投资组合
        stocks_dict = {
            '000001': {'symbol': '000001', 'name': '平安银行'},
            '000002': {'symbol': '000002', 'name': '万科A'}
        }
        
        try:
            # 测试投资组合分析
            portfolio_result = predictor.analyze_portfolio(stocks_dict)
            
            if 'error' in portfolio_result:
                self.log_result("投资组合分析功能", False, portfolio_result['error'])
            elif portfolio_result.get('success'):
                self.log_result("投资组合分析功能", True, "分析成功完成")
            else:
                self.log_result("投资组合分析功能", False, "分析失败")
                
        except Exception as e:
            self.log_result("投资组合分析功能", False, f"异常: {str(e)}")
        
        try:
            # 测试投资组合优化
            optimize_result = predictor.optimize_portfolio(stocks_dict, method='mean_variance')
            
            if 'error' in optimize_result:
                self.log_result("投资组合优化功能", False, optimize_result['error'])
            elif optimize_result.get('success'):
                self.log_result("投资组合优化功能", True, "优化成功完成")
            else:
                self.log_result("投资组合优化功能", False, "优化失败")
                
        except Exception as e:
            self.log_result("投资组合优化功能", False, f"异常: {str(e)}")
    
    def test_backtest(self):
        """测试回测功能"""
        print("\n=== 测试回测功能 ===")
        
        try:
            # 测试移动平均线交叉策略回测
            backtest_result = predictor.run_strategy_backtest(
                '000001',
                strategy_type='ma_crossover',
                short_window=5,
                long_window=10,
                start_date='20240101'
            )
            
            if 'error' in backtest_result:
                self.log_result("回测功能", False, backtest_result['error'])
            elif backtest_result.get('success'):
                metrics = backtest_result['result'].get('metrics', {})
                if metrics:
                    self.log_result("回测功能", True, 
                                  f"回测成功，夏普比率: {metrics.get('sharpe_ratio', 0):.4f}")
                else:
                    self.log_result("回测功能", False, "缺少性能指标")
            else:
                self.log_result("回测功能", False, "回测失败")
                
        except Exception as e:
            self.log_result("回测功能", False, f"异常: {str(e)}")
    
    def test_visualization(self):
        """测试可视化功能"""
        print("\n=== 测试可视化功能 ===")
        
        try:
            # 创建测试数据
            dates = pd.date_range('2024-01-01', periods=30, freq='D')
            test_data = pd.DataFrame({
                'open': np.random.rand(30) * 10 + 100,
                'high': np.random.rand(30) * 12 + 105,
                'low': np.random.rand(30) * 8 + 95,
                'close': np.random.rand(30) * 10 + 100,
                'volume': np.random.randint(1000, 10000, 30)
            }, index=dates)
            
            # 初始化可视化器
            visualizer = visualization.StockVisualizer()
            
            # 测试交互式价格图表
            try:
                fig1 = visualizer.plot_interactive_price_chart(test_data, "TEST", "交互式价格图表测试")
                self.log_result("交互式价格图表", True, "创建成功")
            except Exception as e:
                self.log_result("交互式价格图表", False, f"创建失败: {str(e)}")
            
            # 测试K线图
            try:
                fig2 = visualizer.plot_candlestick_chart(test_data, "TEST", "K线图测试")
                self.log_result("K线图", True, "创建成功")
            except Exception as e:
                self.log_result("K线图", False, f"创建失败: {str(e)}")
                
            # 测试技术指标图表
            try:
                sma = test_data['close'].rolling(10).mean()
                indicators_dict = {'SMA 10': sma}
                fig3 = visualizer.plot_technical_indicators(test_data, indicators_dict, "TEST", "技术指标图表测试")
                self.log_result("技术指标图表", True, "创建成功")
            except Exception as e:
                self.log_result("技术指标图表", False, f"创建失败: {str(e)}")
                
        except Exception as e:
            self.log_result("可视化功能", False, f"异常: {str(e)}")
    
    def test_web_interface(self):
        """测试Web界面功能"""
        print("\n=== 测试Web界面功能 ===")
        
        # 检查必要的文件是否存在
        required_files = ['app.py', 'requirements.txt']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            self.log_result("Web界面功能", False, f"缺少必要文件: {missing_files}")
            return
        
        try:
            # 检查Streamlit是否可用
            import streamlit as st
            self.log_result("Web界面功能", True, "Streamlit可用")
        except ImportError:
            self.log_result("Web界面功能", False, "Streamlit不可用")
        
        try:
            # 检查Plotly是否可用
            import plotly
            self.log_result("Plotly可视化", True, "Plotly可用")
        except ImportError:
            self.log_result("Plotly可视化", False, "Plotly不可用")
    
    def test_json_serialization(self):
        """测试JSON序列化功能"""
        print("\n=== 测试JSON序列化功能 ===")
        
        try:
            # 测试numpy数组序列化
            test_data = {
                'numpy_array': np.array([1.1, 2.2, 3.3]),
                'float_values': [float(1.5), float(2.7)],
                'int_values': [int(42), int(100)],
                'string_value': 'test'
            }
            
            # 使用自定义序列化器
            def numpy_converter(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (int, np.integer)):
                    return int(obj)
                elif isinstance(obj, (float, np.floating)):
                    return float(obj)
                elif isinstance(obj, (bool, np.bool_)):
                    return bool(obj)
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            json_str = json.dumps(test_data, default=numpy_converter)
            parsed = json.loads(json_str)
            
            self.log_result("JSON序列化功能", True, "numpy数组序列化成功")
        except Exception as e:
            self.log_result("JSON序列化功能", False, f"序列化失败: {str(e)}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始StockTracker综合测试...")
        print("=" * 50)
        
        # 运行各个测试
        self.test_json_serialization()
        self.test_data_fetching()
        self.test_price_prediction()
        self.test_technical_indicators()
        self.test_risk_assessment()
        self.test_portfolio_analysis()
        self.test_backtest()
        self.test_visualization()
        self.test_web_interface()
        
        # 生成测试报告
        self.generate_report()
        
        return self.report['failed_tests'] == 0
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "=" * 50)
        print("测试报告")
        print("=" * 50)
        
        print(f"测试时间: {self.report['timestamp']}")
        print(f"总测试数: {self.report['total_tests']}")
        print(f"通过测试: {self.report['passed_tests']}")
        print(f"失败测试: {self.report['failed_tests']}")
        print(f"通过率: {self.report['passed_tests']/self.report['total_tests']*100:.1f}%" if self.report['total_tests'] > 0 else "通过率: 0%")
        
        if self.report['failed_tests'] > 0:
            print("\n失败的测试:")
            for test in self.report['test_details']:
                if not test['success']:
                    print(f"  - {test['test_name']}: {test['details']}")
        
        # 保存测试报告到文件
        try:
            report_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(self.report, f, ensure_ascii=False, indent=2)
            print(f"\n详细测试报告已保存到: {report_filename}")
        except Exception as e:
            print(f"\n保存测试报告失败: {e}")

def main():
    """主函数"""
    # 创建测试套件
    test_suite = StockTrackerTestSuite()
    
    # 运行所有测试
    all_passed = test_suite.run_all_tests()
    
    # 根据测试结果返回退出码
    if all_passed:
        print("\n🎉 所有测试通过！StockTracker系统运行正常。")
        return 0
    else:
        print(f"\n⚠️  有 {test_suite.report['failed_tests']} 个测试失败，请检查日志。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)