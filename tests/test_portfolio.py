#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资组合功能测试脚本
"""

import models.predictors as predictor
import data.fetcher as data_fetcher


def test_portfolio_functions():
    """
    测试投资组合功能
    """
    print("测试投资组合功能...")
    
    # 定义股票组合
    stocks_dict = {
        "002607": {"symbol": "002607", "name": "中公教育"},
        "000001": {"symbol": "000001", "name": "平安银行"},
        "600036": {"symbol": "600036", "name": "招商银行"}
    }
    
    # 测试投资组合分析
    print("\n1. 测试投资组合分析...")
    try:
        portfolio_result = predictor.analyze_portfolio(stocks_dict)
        if "error" in portfolio_result:
            print(f"   投资组合分析失败: {portfolio_result['error']}")
        elif portfolio_result["success"]:
            print("   投资组合分析成功")
            metrics = portfolio_result["metrics"]
            print(f"   预期收益: {metrics['expected_return']:.4f}")
            print(f"   风险(波动率): {metrics['volatility']:.4f}")
            print(f"   夏普比率: {metrics['sharpe_ratio']:.4f}")
        else:
            print("   投资组合分析失败")
    except Exception as e:
        print(f"   投资组合分析出错: {e}")
    
    # 测试均值-方差优化
    print("\n2. 测试均值-方差优化...")
    try:
        mv_result = predictor.optimize_portfolio(stocks_dict, method='mean_variance')
        if "error" in mv_result:
            print(f"   均值-方差优化失败: {mv_result['error']}")
        elif mv_result["success"]:
            print("   均值-方差优化成功")
            print(f"   优化后预期收益: {mv_result['expected_return']:.4f}")
            print(f"   优化后风险(波动率): {mv_result['volatility']:.4f}")
            print(f"   优化后夏普比率: {mv_result['sharpe_ratio']:.4f}")
        else:
            print("   均值-方差优化失败")
    except Exception as e:
        print(f"   均值-方差优化出错: {e}")
    
    # 测试最小方差组合优化
    print("\n3. 测试最小方差组合优化...")
    try:
        min_var_result = predictor.optimize_portfolio(stocks_dict, method='minimum_variance')
        if "error" in min_var_result:
            print(f"   最小方差组合优化失败: {min_var_result['error']}")
        elif min_var_result["success"]:
            print("   最小方差组合优化成功")
            print(f"   优化后预期收益: {min_var_result['expected_return']:.4f}")
            print(f"   优化后风险(波动率): {min_var_result['volatility']:.4f}")
            print(f"   优化后夏普比率: {min_var_result['sharpe_ratio']:.4f}")
        else:
            print("   最小方差组合优化失败")
    except Exception as e:
        print(f"   最小方差组合优化出错: {e}")
    
    # 测试风险平价组合优化
    print("\n4. 测试风险平价组合优化...")
    try:
        risk_parity_result = predictor.optimize_portfolio(stocks_dict, method='risk_parity')
        if "error" in risk_parity_result:
            print(f"   风险平价组合优化失败: {risk_parity_result['error']}")
        elif risk_parity_result["success"]:
            print("   风险平价组合优化成功")
            metrics = risk_parity_result["metrics"]
            print(f"   优化后预期收益: {metrics['expected_return']:.4f}")
            print(f"   优化后风险(波动率): {metrics['volatility']:.4f}")
            print(f"   优化后夏普比率: {metrics['sharpe_ratio']:.4f}")
        else:
            print("   风险平价组合优化失败")
    except Exception as e:
        print(f"   风险平价组合优化出错: {e}")
    
    # 测试蒙特卡洛模拟
    print("\n5. 测试蒙特卡洛模拟...")
    try:
        mc_result = predictor.monte_carlo_portfolio_simulation(stocks_dict, n_simulations=1000)
        if "error" in mc_result:
            print(f"   蒙特卡洛模拟失败: {mc_result['error']}")
        else:
            print("   蒙特卡洛模拟成功")
            print(f"   模拟次数: {mc_result['n_simulations']}")
            print(f"   最大夏普比率: {mc_result['max_sharpe_ratio']:.4f}")
            print(f"   最小波动率: {mc_result['min_volatility']:.4f}")
    except Exception as e:
        print(f"   蒙特卡洛模拟出错: {e}")
    
    print("\n测试完成")


if __name__ == "__main__":
    test_portfolio_functions()