#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockTracker 优化演示
展示性能优化和改进功能
"""

import time
import pandas as pd
import sys
import os
# Add the parent directory to the path so we can import from performance_optimizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from performance_optimizer import optimize_tensorflow, model_cache, data_loader, memory_optimizer

def main():
    print("=" * 60)
    print("StockTracker 性能优化演示")
    print("=" * 60)
    
    print("\n1. TensorFlow 性能优化...")
    start_time = time.time()
    optimize_tensorflow()
    end_time = time.time()
    print(f"   TensorFlow 优化完成，耗时: {end_time - start_time:.2f}秒")
    
    print("\n2. 数据缓存功能演示...")
    from data.fetcher import get_stock_data
    start_time = time.time()
    # 第一次获取数据（从网络）
    data1 = get_stock_data("002607", start_date="20240101")
    first_time = time.time()
    print(f"   首次获取数据耗时: {first_time - start_time:.2f}秒")
    
    # 第二次获取相同数据（从缓存）
    data2 = get_stock_data("002607", start_date="20240101")
    second_time = time.time()
    print(f"   缓存获取数据耗时: {second_time - first_time:.2f}秒")
    print(f"   数据一致性检查: {'通过' if len(data1) == len(data2) else '失败'}")
    
    print("\n3. 模型预测功能演示...")
    import models.predictors as predictor
    
    # 演示不同模型的预测
    models_to_test = ['lstm', 'rf']
    for model_type in models_to_test:
        print(f"   测试 {model_type.upper()} 模型...")
        start_time = time.time()
        result = predictor.predict_stock_price("002607", model_type=model_type, days=1)
        end_time = time.time()
        if "error" not in result:
            print(f"     预测价格: {result['predicted_price']:.2f}元")
            print(f"     预测耗时: {end_time - start_time:.2f}秒")
        else:
            print(f"     预测失败: {result['error']}")
    
    print("\n4. 风险评估功能演示...")
    start_time = time.time()
    risk_result = predictor.assess_stock_risk("002607")
    end_time = time.time()
    if "error" not in risk_result:
        print(f"   风险评估完成，耗时: {end_time - start_time:.2f}秒")
        print(f"   波动率: {risk_result['volatility']:.4f}")
        print(f"   夏普比率: {risk_result['sharpe_ratio']:.4f}")
        print(f"   风险等级: {risk_result['risk_level']['risk_level']}")
    else:
        print(f"   风险评估失败: {risk_result['error']}")
    
    print("\n5. 技术指标计算演示...")
    import analysis.technical as tech
    stock_data = get_stock_data("002607", start_date="20240101")
    if not stock_data.empty:
        start_time = time.time()
        rsi = tech.relative_strength_index(stock_data, period=14)
        sma = tech.simple_moving_average(stock_data, period=20)
        end_time = time.time()
        print(f"   技术指标计算完成，耗时: {end_time - start_time:.2f}秒")
        print(f"   最新RSI值: {rsi.iloc[-1]:.2f}")
        print(f"   最新SMA20值: {sma.iloc[-1]:.2f}")
    
    print("\n6. 内存优化演示...")
    # 演示内存优化功能
    print("   应用内存优化策略...")
    memory_optimizer.clear_session()
    print("   TensorFlow会话已清理")
    
    print("\n7. 清理缓存演示...")
    # 展示如何清理缓存
    print("   演示缓存清理功能...")
    try:
        from models.advanced import clear_model_cache
        clear_model_cache()
    except ImportError:
        print("   模型缓存清理功能未找到")
    
    print("\n" + "=" * 60)
    print("StockTracker 优化演示完成！")
    print("主要改进：")
    print("- 数据缓存减少网络请求")
    print("- 模型缓存避免重复训练")
    print("- TensorFlow性能优化")
    print("- 内存管理优化")
    print("- 错误处理改进")
    print("=" * 60)


if __name__ == "__main__":
    main()