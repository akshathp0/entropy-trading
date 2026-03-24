from pipeline import portfolio
from evaluation import metrics
from itertools import product
import pandas as pd
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

ALT_GAMMAS = config['alt_gammas']
ALT_ENTROPY_MODES = config['alt_entropy_modes']
ALT_ENTROPY_WINDOWS = config['alt_entropy_windows']

def set_parameters(tickers, gamma, mode, window):
    results = portfolio.aggregate_results(tickers, gamma = gamma, mode = mode, window = window)
    portfolio_df = portfolio.build_portfolio(tickers, results, mode = mode)

    return portfolio_df

def iterate_pairs(tickers, gammas = ALT_GAMMAS, modes = ALT_ENTROPY_MODES, windows = ALT_ENTROPY_WINDOWS):
    results = pd.DataFrame(columns = ['gamma', 'mode', 'window', 'sharpe'])

    for gamma, mode, window in product(gammas, modes, windows):
        portfolio_df = set_parameters(tickers, gamma, mode, window)
        sharpe = metrics.calculate_sharpe(portfolio_df['Blend Return'])
        results.loc[len(results)] = {'gamma': gamma, 'mode': mode, 'window': window, 'sharpe': sharpe}
        print(f'gamma: {gamma} mode: {mode} window: {window} sharpe: {sharpe}')

    return results

def find_parameters(results):
    best = results.loc[results['sharpe'].idxmax()]

    gamma = best['gamma']
    mode = best['mode']
    window = best['window']

    return gamma, mode, window