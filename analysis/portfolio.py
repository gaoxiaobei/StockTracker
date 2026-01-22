import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class PortfolioAnalyzer:
    """
    投资组合分析器
    实现现代投资组合理论(MPT)相关计算和分析功能
    """

    def __init__(self):
        """初始化投资组合分析器"""
        pass

    def construct_portfolio(self, stocks_data: Dict[str, pd.DataFrame],
                          weights: Optional[List[float]] = None) -> Dict:
        """
        构建投资组合
        
        Args:
            stocks_data: 股票数据字典，键为股票代码，值为包含价格数据的DataFrame
            weights: 投资组合权重，如果为None则使用等权重
            
        Returns:
            dict: 投资组合信息
        """
        symbols = list(stocks_data.keys())
        n_assets = len(symbols)
        
        if weights is None:
            weights = [1/n_assets] * n_assets
        
        # 确保权重和为1
        weights_array = np.array(weights, dtype=float)
        weights_array = weights_array / np.sum(weights_array)
        
        # 计算收益率
        returns = {}
        for symbol in symbols:
            data = stocks_data[symbol]
            if 'close' in data.columns:
                returns[symbol] = data['close'].pct_change().dropna()
            else:
                raise ValueError(f"股票 {symbol} 数据中缺少 'close' 列")
        
        # 合并收益率数据
        returns_df = pd.DataFrame(returns)
        
        # 计算协方差矩阵
        cov_matrix = returns_df.cov()
        
        portfolio_info = {
            'symbols': symbols,
            'weights': weights_array.tolist(),
            'returns': returns_df,
            'cov_matrix': cov_matrix,
            'n_assets': n_assets
        }
        
        return portfolio_info

    def calculate_portfolio_metrics(self, portfolio_info: Dict) -> Dict:
        """
        计算投资组合的预期收益和风险
        
        Args:
            portfolio_info: 投资组合信息字典
            
        Returns:
            dict: 投资组合指标
        """
        weights = portfolio_info['weights']
        returns_df = portfolio_info['returns']
        cov_matrix = portfolio_info['cov_matrix']
        
        # 计算预期收益率（年化）
        expected_returns = returns_df.mean() * 252  # 年化假设252个交易日
        
        # 计算投资组合预期收益率
        portfolio_return = np.sum(weights * expected_returns)
        
        # 计算投资组合方差
        # 确保weights是numpy数组
        weights_array = np.array(weights)
        portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
        
        # 计算投资组合标准差（风险）
        portfolio_std = np.sqrt(portfolio_variance) * np.sqrt(252)  # 年化标准差
        
        # 计算夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std != 0 else 0
        
        metrics = {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_std),
            'variance': float(portfolio_variance * 252),
            'sharpe_ratio': float(sharpe_ratio),
            'expected_returns_by_asset': expected_returns.to_dict(),
            'weights': [float(w) for w in weights]
        }
        
        return metrics

    def mean_variance_optimization(self, portfolio_info: Dict, 
                                 target_return: Optional[float] = None) -> Dict:
        """
        均值-方差优化
        
        Args:
            portfolio_info: 投资组合信息
            target_return: 目标收益率，如果为None则寻找最大化夏普比率的组合
            
        Returns:
            dict: 优化结果
        """
        returns_df = portfolio_info['returns']
        expected_returns = returns_df.mean() * 252
        cov_matrix = portfolio_info['cov_matrix']
        n_assets = portfolio_info['n_assets']
        
        # 定义约束条件
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # 权重和为1
        
        # 如果指定了目标收益率，添加收益率约束
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x * expected_returns) - target_return
            })
        
        # 权重边界（允许卖空设为-1到1，不允许卖空设为0到1）
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # 初始权重
        init_weights = np.array([1/n_assets] * n_assets)
        
        # 优化目标函数
        if target_return is None:
            # 最大化夏普比率
            def sharpe_objective(weights):
                port_return = np.sum(weights * expected_returns)
                # 确保weights是numpy数组
                weights_array = np.array(weights)
                port_std = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array))) * np.sqrt(252)
                risk_free_rate = 0.03
                sharpe = (port_return - risk_free_rate) / port_std if port_std != 0 else 0
                return -sharpe  # 最小化负夏普比率即最大化夏普比率
            objective_func = sharpe_objective
        else:
            # 最小化方差
            def variance_objective(weights):
                # 确保weights是numpy数组
                weights_array = np.array(weights)
                return np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
            objective_func = variance_objective
        
        # 执行优化
        result = minimize(objective_func, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            port_return = np.sum(optimal_weights * expected_returns)
            # 确保optimal_weights是numpy数组
            optimal_weights_array = np.array(optimal_weights)
            port_variance = np.dot(optimal_weights_array.T, np.dot(cov_matrix, optimal_weights_array))
            port_std = np.sqrt(port_variance) * np.sqrt(252)
            
            # 计算夏普比率
            risk_free_rate = 0.03
            sharpe_ratio = (port_return - risk_free_rate) / port_std if port_std != 0 else 0
            
            return {
                'success': True,
                'weights': [float(w) for w in optimal_weights],
                'expected_return': float(port_return),
                'volatility': float(port_std),
                'sharpe_ratio': float(sharpe_ratio),
                'message': '优化成功'
            }
        else:
            return {
                'success': False,
                'message': f"优化失败: {result.message}"
            }

    def minimum_variance_portfolio(self, portfolio_info: Dict) -> Dict:
        """
        最小方差组合优化
        
        Args:
            portfolio_info: 投资组合信息
            
        Returns:
            dict: 最小方差组合结果
        """
        return self.mean_variance_optimization(portfolio_info, target_return=None)

    def efficient_frontier(self, portfolio_info: Dict, 
                          n_portfolios: int = 100) -> Dict:
        """
        计算有效前沿
        
        Args:
            portfolio_info: 投资组合信息
            n_portfolios: 生成的投资组合数量
            
        Returns:
            dict: 有效前沿数据
        """
        returns_df = portfolio_info['returns']
        expected_returns = returns_df.mean() * 252
        cov_matrix = portfolio_info['cov_matrix']
        n_assets = portfolio_info['n_assets']
        
        # 计算单个资产的最小和最大预期收益率
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        
        # 生成目标收益率范围
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        # 存储结果
        portfolios = []
        
        for target in target_returns:
            result = self.mean_variance_optimization(portfolio_info, target_return=target)
            if result['success']:
                portfolios.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'weights': result['weights']
                })
        
        return {
            'portfolios': portfolios,
            'returns': [float(p['return']) for p in portfolios],
            'volatilities': [float(p['volatility']) for p in portfolios],
            'sharpe_ratios': [float(p['sharpe_ratio']) for p in portfolios]
        }

    def plot_efficient_frontier(self, efficient_frontier_data: Dict, 
                              save_path: Optional[str] = None):
        """
        绘制有效前沿图表
        
        Args:
            efficient_frontier_data: 有效前沿数据
            save_path: 保存路径，如果为None则显示图表
        """
        returns = efficient_frontier_data['returns']
        volatilities = efficient_frontier_data['volatilities']
        sharpe_ratios = efficient_frontier_data['sharpe_ratios']
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis')
        plt.colorbar(scatter, label='夏普比率')
        plt.xlabel('风险 (年化波动率)')
        plt.ylabel('预期收益 (年化)')
        plt.title('投资组合有效前沿')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

    def calculate_capm_metrics(self, stock_returns: pd.Series, 
                             market_returns: pd.Series) -> Dict:
        """
        计算资本资产定价模型(CAPM)相关指标
        
        Args:
            stock_returns: 股票收益率序列
            market_returns: 市场收益率序列
            
        Returns:
            dict: CAPM指标
        """
        # 合并数据并删除NaN
        data = pd.DataFrame({'stock': stock_returns, 'market': market_returns}).dropna()
        
        if len(data) < 2:
            return {'error': '数据不足，无法计算CAPM指标'}
        
        stock_ret = data['stock']
        market_ret = data['market']
        
        # 计算贝塔系数
        covariance = np.cov(stock_ret, market_ret)[0, 1]
        market_variance = np.var(market_ret)
        beta = covariance / market_variance if market_variance != 0 else 0
        
        # 计算Alpha（假设无风险利率为3%）
        risk_free_rate = 0.03 / 252  # 日化无风险利率
        expected_market_return = market_ret.mean()
        expected_stock_return = stock_ret.mean()
        alpha = expected_stock_return - (risk_free_rate + beta * (expected_market_return - risk_free_rate))
        
        # 年化Alpha
        alpha_annualized = alpha * 252
        
        return {
            'beta': beta,
            'alpha': alpha_annualized,
            'correlation': np.corrcoef(stock_ret, market_ret)[0, 1],
            'r_squared': np.corrcoef(stock_ret, market_ret)[0, 1] ** 2
        }

    def risk_contribution_analysis(self, portfolio_info: Dict) -> Dict:
        """
        投资组合风险贡献分析
        
        Args:
            portfolio_info: 投资组合信息
            
        Returns:
            dict: 风险贡献分析结果
        """
        weights = np.array(portfolio_info['weights'])
        cov_matrix = portfolio_info['cov_matrix']
        symbols = portfolio_info['symbols']
        
        # 计算投资组合方差
        # 确保weights是numpy数组
        weights_array = np.array(weights)
        portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
        portfolio_std = np.sqrt(portfolio_variance)
        
        if portfolio_std == 0:
            return {'error': '投资组合标准差为0，无法计算风险贡献'}
        
        # 计算各资产的风险贡献
        marginal_risk_contribution = np.dot(cov_matrix, weights) / portfolio_std
        risk_contribution = weights * marginal_risk_contribution
        risk_contribution_ratio = risk_contribution / portfolio_std
        
        # 计算百分比贡献
        percentage_contribution = risk_contribution_ratio * 100
        
        return {
            'symbols': symbols,
            'weights': [float(w) for w in weights],
            'risk_contributions': [float(rc) for rc in risk_contribution],
            'percentage_contributions': [float(pc) for pc in percentage_contribution],
            'total_risk': float(portfolio_std),
            'marginal_contributions': [float(mc) for mc in marginal_risk_contribution]
        }

    def risk_parity_portfolio(self, portfolio_info: Dict) -> Dict:
        """
        风险平价组合优化
        
        Args:
            portfolio_info: 投资组合信息
            
        Returns:
            dict: 风险平价组合结果
        """
        cov_matrix = portfolio_info['cov_matrix']
        n_assets = portfolio_info['n_assets']
        
        # 目标：使各资产的风险贡献相等
        def objective(weights):
            weights_array = np.array(weights)
            portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
            portfolio_std = np.sqrt(portfolio_variance)
            
            if portfolio_std == 0:
                return np.inf
            
            # 计算各资产的风险贡献
            marginal_contribution = np.dot(cov_matrix, weights_array) / portfolio_std
            risk_contribution = weights_array * marginal_contribution
            
            # 目标是最小化各资产风险贡献的方差
            return np.var(risk_contribution)
        
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        # 优化
        result = minimize(objective, init_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            # 计算优化后的指标
            portfolio_metrics = self.calculate_portfolio_metrics({
                'weights': optimal_weights,
                'returns': portfolio_info['returns'],
                'cov_matrix': cov_matrix,
                'symbols': portfolio_info['symbols'],
                'n_assets': n_assets
            })
            
            return {
                'success': True,
                'weights': [float(w) for w in optimal_weights],
                'metrics': portfolio_metrics,
                'message': '风险平价优化成功'
            }
        else:
            return {
                'success': False,
                'message': f"风险平价优化失败: {result.message}"
            }

    def black_litterman_model(self, portfolio_info: Dict, 
                            views: List[Dict], 
                            tau: float = 0.05) -> Dict:
        """
        Black-Litterman模型
        
        Args:
            portfolio_info: 投资组合信息
            views: 观点列表，每个观点为字典包含'assets', 'weights', 'view_return'
            tau: 不确定性参数
            
        Returns:
            dict: Black-Litterman模型结果
        """
        cov_matrix = portfolio_info['cov_matrix']
        symbols = portfolio_info['symbols']
        n_assets = portfolio_info['n_assets']
        
        # 市场均衡收益率（假设市值权重）
        market_weights = np.array([1/n_assets] * n_assets)
        risk_free_rate = 0.03
        market_risk_premium = 0.05  # 市场风险溢价假设为5%
        equilibrium_returns = risk_free_rate + market_risk_premium * market_weights
        
        # 观点矩阵
        n_views = len(views)
        if n_views == 0:
            return {
                'success': False,
                'message': '需要提供至少一个观点'
            }
        
        P = np.zeros((n_views, n_assets))  # 观点矩阵
        Q = np.zeros(n_views)  # 观点收益向量
        Omega = np.zeros((n_views, n_views))  # 观点不确定性矩阵
        
        for i, view in enumerate(views):
            # 设置观点矩阵P
            for j, asset in enumerate(view['assets']):
                if asset in symbols:
                    asset_idx = symbols.index(asset)
                    P[i, asset_idx] = view['weights'][j]
            
            # 设置观点收益
            Q[i] = view['view_return']
            
            # 设置观点不确定性（简单假设为观点收益的10%）
            Omega[i, i] = (view['view_return'] * 0.1) ** 2
        
        # Black-Litterman公式
        try:
            # 计算后验收益率
            cov_inv = np.linalg.inv(cov_matrix)
            Omega_inv = np.linalg.inv(Omega)
            
            # 后验收益率
            posterior_returns = np.linalg.inv(cov_inv + tau * P.T @ Omega_inv @ P) @ \
                              (cov_inv @ equilibrium_returns + tau * P.T @ Omega_inv @ Q)
            
            # 后验协方差矩阵
            posterior_cov = np.linalg.inv(cov_inv + tau * P.T @ Omega_inv @ P)
            
            return {
                'success': True,
                'posterior_returns': [float(pr) for pr in posterior_returns],
                'posterior_covariance': [[float(val) for val in row] for row in posterior_cov],
                'equilibrium_returns': [float(er) for er in equilibrium_returns],
                'views': views,
                'message': 'Black-Litterman模型计算成功'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Black-Litterman模型计算失败: {str(e)}"
            }

    def calculate_performance_metrics(self, portfolio_returns: pd.Series, 
                                    benchmark_returns: pd.Series) -> Dict:
        """
        计算投资组合绩效评估指标
        
        Args:
            portfolio_returns: 投资组合收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            dict: 绩效评估指标
        """
        # 合并数据并删除NaN
        data = pd.DataFrame({
            'portfolio': portfolio_returns, 
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(data) < 2:
            return {'error': '数据不足，无法计算绩效指标'}
        
        port_ret = data['portfolio']
        bench_ret = data['benchmark']
        
        # 计算Alpha和Beta
        capm_metrics = self.calculate_capm_metrics(port_ret, bench_ret)
        
        # 计算信息比率
        tracking_error = np.std(port_ret - bench_ret) * np.sqrt(252)  # 年化跟踪误差
        information_ratio = (port_ret.mean() - bench_ret.mean()) * 252 / tracking_error if tracking_error != 0 else 0
        
        # 计算特雷诺比率
        treynor_ratio = (port_ret.mean() * 252 - 0.03) / capm_metrics.get('beta', 0) if capm_metrics.get('beta', 0) != 0 else 0
        
        # 计算夏普比率
        sharpe_ratio = (port_ret.mean() * 252 - 0.03) / (port_ret.std() * np.sqrt(252)) if port_ret.std() != 0 else 0
        
        return {
            'alpha': capm_metrics.get('alpha', 0),
            'beta': capm_metrics.get('beta', 0),
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'sharpe_ratio': sharpe_ratio,
            'correlation_with_benchmark': np.corrcoef(port_ret, bench_ret)[0, 1]
        }

    def performance_attribution(self, portfolio_info: Dict, 
                              sector_weights: Dict[str, float],
                              sector_returns: Dict[str, float]) -> Dict:
        """
        业绩归因分析
        
        Args:
            portfolio_info: 投资组合信息
            sector_weights: 各行业的权重
            sector_returns: 各行业的收益率
            
        Returns:
            dict: 业绩归因分析结果
        """
        # 计算总收益率贡献
        total_contribution = 0
        contributions = {}
        
        for sector, weight in sector_weights.items():
            sector_return = sector_returns.get(sector, 0)
            contribution = weight * sector_return
            contributions[sector] = contribution
            total_contribution += contribution
        
        return {
            'sector_contributions': contributions,
            'total_contribution': total_contribution,
            'sector_weights': sector_weights,
            'sector_returns': sector_returns
        }

    def monte_carlo_simulation(self, portfolio_info: Dict, 
                             n_simulations: int = 10000) -> Dict:
        """
        蒙特卡洛模拟用于投资组合优化
        
        Args:
            portfolio_info: 投资组合信息
            n_simulations: 模拟次数
            
        Returns:
            dict: 模拟结果
        """
        n_assets = portfolio_info['n_assets']
        returns_df = portfolio_info['returns']
        expected_returns = returns_df.mean().values * 252
        cov_matrix = portfolio_info['cov_matrix'].values
        
        # Vectorized Monte Carlo simulation
        # Generate all random weights at once: (n_simulations, n_assets)
        weights_record = np.random.random((n_simulations, n_assets))
        weights_record = weights_record / np.sum(weights_record, axis=1)[:, np.newaxis]
        
        # Calculate portfolio returns: (n_simulations,)
        port_returns = np.dot(weights_record, expected_returns)
        
        # Calculate portfolio volatilities: (n_simulations,)
        # variance = w.T * Cov * w
        # Using einsum for vectorized quadratic form: sum_j sum_k w_ij * Cov_jk * w_ik
        port_vars = np.einsum('ij,jk,ik->i', weights_record, cov_matrix, weights_record)
        port_std = np.sqrt(port_vars) * np.sqrt(252)
        
        # Calculate Sharpe Ratios
        risk_free_rate = 0.03
        sharpe_ratios = (port_returns - risk_free_rate) / port_std
        # Handle zero volatility
        sharpe_ratios = np.where(port_std == 0, 0, sharpe_ratios)
        
        # Find optimal portfolios
        max_sharpe_idx = np.argmax(sharpe_ratios)
        min_vol_idx = np.argmin(port_std)
        
        return {
            'simulated_returns': [float(r) for r in port_returns],
            'simulated_volatilities': [float(v) for v in port_std],
            'simulated_sharpe_ratios': [float(s) for s in sharpe_ratios],
            'max_sharpe_ratio': float(sharpe_ratios[max_sharpe_idx]),
            'return_for_max_sharpe': float(port_returns[max_sharpe_idx]),
            'volatility_for_max_sharpe': float(port_std[max_sharpe_idx]),
            'max_sharpe_weights': [float(w) for w in weights_record[max_sharpe_idx]],
            'min_volatility': float(port_std[min_vol_idx]),
            'return_for_min_vol': float(port_returns[min_vol_idx]),
            'sharpe_for_min_vol': float(sharpe_ratios[min_vol_idx]),
            'min_vol_weights': [float(w) for w in weights_record[min_vol_idx]],
            'n_simulations': int(n_simulations)
        }

    def plot_monte_carlo_results(self, monte_carlo_results: Dict, 
                               save_path: Optional[str] = None):
        """
        绘制蒙特卡洛模拟结果
        
        Args:
            monte_carlo_results: 蒙特卡洛模拟结果
            save_path: 保存路径，如果为None则显示图表
        """
        returns = monte_carlo_results['simulated_returns']
        volatilities = monte_carlo_results['simulated_volatilities']
        sharpe_ratios = monte_carlo_results['simulated_sharpe_ratios']
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='夏普比率')
        
        # 标记最优投资组合
        max_sharpe_return = monte_carlo_results['return_for_max_sharpe']
        max_sharpe_vol = monte_carlo_results['volatility_for_max_sharpe']
        plt.scatter(max_sharpe_vol, max_sharpe_return, 
                   marker='*', s=200, c='red', label='最大夏普比率组合')
        
        min_vol_return = monte_carlo_results['return_for_min_vol']
        min_vol_vol = monte_carlo_results['min_volatility']
        plt.scatter(min_vol_vol, min_vol_return, 
                   marker='*', s=200, c='green', label='最小波动率组合')
        
        plt.xlabel('风险 (年化波动率)')
        plt.ylabel('预期收益 (年化)')
        plt.title('蒙特卡洛模拟投资组合优化结果')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def analyze_portfolio(stocks_data: Dict[str, pd.DataFrame], 
                     weights: Optional[List[float]] = None) -> Dict:
    """
    分析投资组合的主要入口函数
    
    Args:
        stocks_data: 股票数据字典
        weights: 投资组合权重
        
    Returns:
        dict: 投资组合分析结果
    """
    analyzer = PortfolioAnalyzer()
    
    try:
        # 构建投资组合
        portfolio_info = analyzer.construct_portfolio(stocks_data, weights)
        
        # 计算基础指标
        metrics = analyzer.calculate_portfolio_metrics(portfolio_info)
        
        # 风险贡献分析
        risk_contribution = analyzer.risk_contribution_analysis(portfolio_info)
        
        # 有效前沿计算
        efficient_frontier = analyzer.efficient_frontier(portfolio_info, n_portfolios=50)
        
        return {
            'success': True,
            'portfolio_info': portfolio_info,
            'metrics': metrics,
            'risk_contribution': risk_contribution,
            'efficient_frontier': efficient_frontier
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def optimize_portfolio(stocks_data: Dict[str, pd.DataFrame], 
                      method: str = 'mean_variance',
                      target_return: Optional[float] = None) -> Dict:
    """
    投资组合优化函数
    
    Args:
        stocks_data: 股票数据字典
        method: 优化方法 ('mean_variance', 'minimum_variance', 'risk_parity')
        target_return: 目标收益率（仅用于均值-方差优化）
        
    Returns:
        dict: 优化结果
    """
    analyzer = PortfolioAnalyzer()
    
    try:
        # 构建投资组合
        portfolio_info = analyzer.construct_portfolio(stocks_data)
        
        # 根据方法进行优化
        if method == 'mean_variance':
            result = analyzer.mean_variance_optimization(portfolio_info, target_return)
        elif method == 'minimum_variance':
            result = analyzer.minimum_variance_portfolio(portfolio_info)
        elif method == 'risk_parity':
            result = analyzer.risk_parity_portfolio(portfolio_info)
        else:
            return {
                'success': False,
                'error': f"不支持的优化方法: {method}"
            }
        
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # 这里可以添加测试代码
    print("投资组合分析模块已定义")