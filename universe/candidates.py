import yfinance as yf
import pandas as pd
from evaluation import metrics
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

START = config['start']
END = config['end']
ANNUALIZED_THRESHOLD = config['annualized_threshold']
ANNUALIZED_START = config['annualized_start']
ANNUALIZED_END = config['annualized_end']

def load_data(start = START, end = END):
    
    tickers = ['QQQ', 
               'IWM', 'IJH', 
               'VUG', 'VTV', 
               'XLK', 'XLE', 'XLF', 'XLV', 'XLU', 'XLI', 'XLB', 'XLP', 'XLY',
               'EFA', 'VGK', 'EWJ',
               'EEM', 'VWO', 'EWZ', 'FXI',
               'IBB', 'VNQ', 'SOXX',
               'GLD', 'USO', 'DBC', 'SLV']

    df = yf.download(tickers, start = start, end = end, auto_adjust = False)['Adj Close']
    returns = df.pct_change().dropna()

    return returns, tickers

def filter_annualized(start = ANNUALIZED_START, end = ANNUALIZED_END, threshold = ANNUALIZED_THRESHOLD):
    returns, tickers = load_data(start = start, end = end)
    
    filtered_tickers = []

    for ticker in tickers:
        if metrics.calculate_annualized_returns(returns[ticker], start = start, end = end) > threshold:
            filtered_tickers.append(ticker)

    return returns[filtered_tickers], filtered_tickers

def final_asset_drift(start = START, end = END):
    
    tickers = pd.read_csv('results/universe_selection/final_assets.csv', skiprows = 1, header = None)[0].tolist()

    df = yf.download(tickers, start = start, end = end, auto_adjust = False)['Adj Close']
    returns = df.pct_change().dropna()

    return returns, tickers