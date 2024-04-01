import yfinance as yf
import pandas as pd

def retrieve_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def preprocess_data(stock_data):
    # Handle missing data
    stock_data.fillna(method='ffill', inplace=True)
    stock_data.fillna(method='bfill', inplace=True)
    
    # Normalize the data
    stock_data_norm = (stock_data - stock_data.mean()) / stock_data.std()
    
    return stock_data_norm

def store_data(stock_data, filename):
    stock_data.to_csv(filename)

# Example usage
if __name__ == '__main__':
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    
    stock_data = retrieve_stock_data(ticker, start_date, end_date)
    preprocessed_data = preprocess_data(stock_data)
    store_data(preprocessed_data, 'preprocessed_data.csv')