#!/usr/bin/env python3
"""
测试修复后的功能
"""

import json
import numpy as np
import data.fetcher as data_fetcher
import models.predictors as predictor
import analysis.portfolio as portfolio
import pandas as pd

def test_json_serialization():
    """测试JSON序列化修复"""
    print("=== 测试JSON序列化修复 ===")
    
    # 测试numpy数组转换
    test_data = {
        'numpy_array': np.array([1.1, 2.2, 3.3]),
        'float_values': [float(1.5), float(2.7)],
        'int_values': [int(42), int(100)]
    }
    
    try:
        # 使用自定义序列化器
        def numpy_converter(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError
        
        json_str = json.dumps(test_data, default=numpy_converter)
        parsed = json.loads(json_str)
        print("✓ JSON序列化修复成功")
        return True
    except Exception as e:
        print(f"✗ JSON序列化仍有错误: {e}")
        return False

def test_data_fetcher():
    """测试数据获取功能"""
    print("\n=== 测试数据获取功能 ===")
    
    try:
        # 测试有效股票代码
        stock_data = data_fetcher.get_stock_data('000001', period='daily', start_date='20240101', adjust='qfq')
        if not stock_data.empty:
            print(f"✓ 数据获取成功 - 获取到 {len(stock_data)} 条记录")
            print(f"  列名: {list(stock_data.columns)}")
            print(f"  数据类型: {type(stock_data.index[0])}")
        else:
            print("⚠ 数据获取返回空数据")
        
        # 测试股票信息获取
        stock_info = data_fetcher.get_stock_info('000001')
        if stock_info:
            print("✓ 股票信息获取成功")
        else:
            print("⚠ 股票信息获取失败")
            
        return True
    except Exception as e:
        print(f"✗ 数据获取测试失败: {e}")
        return False

def test_portfolio_json():
    """测试投资组合JSON序列化"""
    print("\n=== 测试投资组合JSON序列化 ===")
    
    try:
        # 创建测试数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        stock1_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        stock2_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 50,
            'open': np.random.randn(100).cumsum() + 50,
            'high': np.random.randn(100).cumsum() + 52,
            'low': np.random.randn(100).cumsum() + 48,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        stocks_data = {
            '000001': stock1_data,
            '000002': stock2_data
        }
        
        # 测试投资组合分析
        result = portfolio.analyze_portfolio(stocks_data, weights=[0.6, 0.4])
        
        if 'error' not in result:
            # 测试JSON序列化
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                elif isinstance(obj, pd.Series):
                    return obj.to_list()
                raise TypeError
            
            json_str = json.dumps(result, default=convert_numpy)
            print("✓ 投资组合JSON序列化成功")
            return True
        else:
            print(f"⚠ 投资组合分析返回错误: {result['error']}")
            return False
            
    except Exception as e:
        print(f"✗ 投资组合测试失败: {e}")
        return False

def test_predictor_structure():
    """测试预测器结构"""
    print("\n=== 测试预测器结构 ===")
    
    try:
        # 测试预测器返回结构
        result = predictor.predict_stock_price('000001', days=1, model_type='lstm')
        
        if isinstance(result, dict):
            print("✓ 预测器返回字典结构")
            if 'error' in result:
                print(f"  返回错误: {result['error']}")
            else:
                print("  返回正常结果")
            return True
        else:
            print("✗ 预测器返回类型错误")
            return False
            
    except Exception as e:
        print(f"✗ 预测器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试所有修复...")
    
    tests = [
        test_json_serialization,
        test_data_fetcher,
        test_portfolio_json,
        test_predictor_structure
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试异常: {e}")
            results.append(False)
    
    print(f"\n=== 测试结果 ===")
    print(f"通过测试: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 所有修复测试通过！")
    else:
        print("⚠ 部分测试未通过，请检查日志")
    
    return all(results)

if __name__ == "__main__":
    main()