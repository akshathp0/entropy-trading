import yfinance as yf
import pandas as pd
from evaluation import metrics
from pipeline import run
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

START = config['start']
END = config['end']
SHARPE_THRESHOLD = config['sharpe_threshold']
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

def filter_asset_sharpes(returns, tickers):
    sharpes = pd.DataFrame(columns = ['Ticker', 'Sharpe Ratio'])

    for ticker in tickers:
        sharpes.loc[len(sharpes)] = {'Ticker': ticker,
                                'Sharpe Ratio': metrics.calculate_sharpe(returns[ticker])}

    return sharpes

def filter_strat_sharpes(tickers):
    sharpes = pd.DataFrame(columns = ['Ticker', 'Sharpe Ratio'])
    results = {}

    for ticker in tickers:
        df = run.run_pipeline(ticker)
        sharpes.loc[len(sharpes)] = {'Ticker': ticker,
                                'Sharpe Ratio': metrics.calculate_sharpe(df['Blend Return'])}
        results[ticker] = df

    return sharpes, results

def apply_sharpe_threshold(sharpes, threshold = SHARPE_THRESHOLD):
    sharpes = sharpes[sharpes['Sharpe Ratio'] >= threshold]

    return sharpes['Ticker']