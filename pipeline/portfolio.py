from pipeline import run
import pandas as pd
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

GAMMA = config['entropy_confidence']
ENTROPY_MODE = config['entropy_mode']
ENTROPY_WINDOW = config['rolling_window']
INSAMPLE_START = config['start']
INSAMPLE_END = config['end']

def aggregate_results(tickers, start = INSAMPLE_START, end = INSAMPLE_END, gamma = None, mode = None, window = None):
    
    gamma = gamma or GAMMA
    mode = mode or ENTROPY_MODE
    window = window or ENTROPY_WINDOW

    results = {}

    for ticker in tickers:
        df = run.run_pipeline(ticker, start = start, end = end, gamma = gamma, mode = mode, window = window)
        results[ticker] = df

    return results

def build_portfolio(tickers, results, mode = ENTROPY_MODE):
    portfolio_df = pd.DataFrame()

    if mode == 'portfolio_only' or mode == 'both':
        returns = pd.DataFrame({ticker: results[ticker]['Blend Return'] for ticker in tickers})
        confidence = pd.DataFrame({ticker: results[ticker]['Confidence'] for ticker in tickers})
        weights = confidence.div(confidence.sum(axis = 1), axis = 0)
        portfolio_df['Blend Return'] = (returns * weights).sum(axis = 1)
    else:
        portfolio_df['Blend Return'] = sum(results[ticker]['Blend Return'] for ticker in tickers)
        portfolio_df['Blend Return'] = portfolio_df['Blend Return'] / len(tickers)

    portfolio_df['Blend Equity Curve'] = (1 + portfolio_df['Blend Return']).cumprod()

    return portfolio_df

def average_exposure(tickers, results):
    signals = pd.DataFrame({ticker: results[ticker]['Blend Signal'] for ticker in tickers})
    return signals.mean().mean()