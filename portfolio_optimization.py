import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_portfolio_return(weights, returns):
    return np.dot(weights, returns)

def calculate_portfolio_variance(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

def calculate_sharpe_ratio(returns, risk_free_rate, std_dev):
    return (returns - risk_free_rate) / std_dev

def mean_variance_optimization(returns, covariance_matrix, risk_free_rate):
    num_assets = len(returns)
    
    def objective_function(weights):
        portfolio_return = calculate_portfolio_return(weights, returns)
        portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)
        sharpe_ratio = calculate_sharpe_ratio(portfolio_return, risk_free_rate, np.sqrt(portfolio_variance))
        return -sharpe_ratio
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    initial_weights = np.array([1/num_assets] * num_assets)
    
    optimized_weights = minimize(objective_function, initial_weights, method='SLSQP',
                                 bounds=bounds, constraints=constraints)
    
    return optimized_weights.x

# Example usage
if __name__ == '__main__':
    # Assuming you have a CSV file with stock returns
    stock_returns = pd.read_csv('stock_returns.csv', index_col=0)
    
    returns = stock_returns.mean().values
    covariance_matrix = stock_returns.cov().values
    risk_free_rate = 0.02
    
    optimized_weights = mean_variance_optimization(returns, covariance_matrix, risk_free_rate)
    print("Optimized Portfolio Weights:")
    print(optimized_weights)