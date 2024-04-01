from data_retrieval import retrieve_stock_data, preprocess_data, store_data
from portfolio_optimization import mean_variance_optimization
import pandas as pd

def main():
    # Retrieve and preprocess stock data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    
    stock_data = {}
    for ticker in tickers:
        data = retrieve_stock_data(ticker, start_date, end_date)
        preprocessed_data = preprocess_data(data)
        stock_data[ticker] = preprocessed_data['Adj Close']  # Access 'Adj Close' column
    
    # Combine stock data into a single DataFrame
    combined_data = pd.concat(stock_data, axis=1)
    
    # Calculate stock returns
    stock_returns = combined_data.pct_change()
    
    # Portfolio optimization
    returns = stock_returns.mean().values
    covariance_matrix = stock_returns.cov().values
    risk_free_rate = 0.02
    
    optimized_weights = mean_variance_optimization(returns, covariance_matrix, risk_free_rate)
    
    # Print the optimized portfolio weights
    print("Optimized Portfolio Weights:")
    for ticker, weight in zip(tickers, optimized_weights):
        print(f"{ticker}: {weight:.2f}")

if __name__ == '__main__':
    main()