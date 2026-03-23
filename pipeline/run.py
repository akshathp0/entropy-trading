from backtests import blend_signal, mean_reversion, exposure
from data import data_loader
from features import tstat, volatility, zscore
from regime import entropy, matrix, state_labels
from evaluation import metrics, plot
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_pipeline(ticker):
    df = data_loader.get_data(ticker)
    df = tstat.compute_tstat(df)
    df = zscore.calculate_zscore(df)
    df = volatility.compute_volatility(df)
    df = state_labels.label_volatility(df)
    df = state_labels.label_regime(df)
    df = state_labels.smooth_regime(df)
    df = matrix.initialize_state(df)
    df = entropy.build_rolling_entropy(df)
    df = exposure.normalize_entropy(df)
    df = mean_reversion.generate_signal(df)
    df = blend_signal.generate_signal(df)
    df = blend_signal.blended_backtest(df)

    return df

def calculate_metrics(df, ticker):
    annualized_returns = metrics.calculate_annualized_returns(df['Blend Return'])
    sharpe = metrics.calculate_sharpe(df['Blend Return'])
    sortino = metrics.calculate_sortino(df['Blend Return'])
    calmar = metrics.calculate_calmar(df['Blend Return'])
    max_drawdown = metrics.calculate_max_drawdown(df['Blend Return'])
    # state_counts = metrics.state_counts(df)
    # entropy_stats = metrics.describe_entropy(df)
    asset = ticker.lower()

    pd.DataFrame({'Annualized Returns': [annualized_returns],
                  'Sharpe': [sharpe],
                  'Sortino': [sortino],
                  'Calmar': [calmar],
                  'Max Drawdown': [max_drawdown],
                    }).to_csv(f'results/assets/{asset}/metrics.csv', index = False)
    
    # state_counts.to_csv(f'results/assets/{asset}/state_counts.csv', index = False)
    # entropy_stats.to_csv(f'results/assets/{asset}/entropy_stats.csv', index = False)

def generate_plots(df, ticker):
    spy_curve = load_spy()

    asset = ticker.lower()
    os.makedirs(f'results/assets/{asset}', exist_ok = True)

    equity_curve = plot.plot_equity_curve(df, ticker, spy_curve)
    equity_curve.savefig(f'results/assets/{asset}/equity_curve.png')

    regime_overlay = plot.plot_regime_overlay(df, ticker)
    regime_overlay.savefig(f'results/assets/{asset}/regime_overlay.png')

    rolling_entropy = plot.plot_rolling_entropy(df, ticker)
    rolling_entropy.savefig(f'results/assets/{asset}/rolling_entropy.png')

    # rolling_sharpe = plot.plot_rolling_sharpe(df, ticker, spy_curve)
    # rolling_sharpe.savefig(f'results/assets/{asset}/rolling_sharpe')

    plt.close("all")

def generate_portfolio(df, ticker):
    spy_curve = load_spy()

    os.makedirs(f'results/portfolio', exist_ok = True)

    equity_curve = plot.plot_equity_curve(df, 'Portfolio', spy_curve)
    equity_curve.savefig('results/assets/portfolio/equity_curve.png')

    plt.close("all")

def calculate_portfolio(df):
    annualized_returns = metrics.calculate_annualized_returns(df['Blend Return'])
    sharpe = metrics.calculate_sharpe(df['Blend Return'])
    sortino = metrics.calculate_sortino(df['Blend Return'])
    calmar = metrics.calculate_calmar(df['Blend Return'])
    max_drawdown = metrics.calculate_max_drawdown(df['Blend Return'])

    pd.DataFrame({'Annualized Returns': [annualized_returns],
                  'Sharpe': [sharpe],
                  'Sortino': [sortino],
                  'Calmar': [calmar],
                  'Max Drawdown': [max_drawdown],
                    }).to_csv(f'results/portfolio/metrics.csv', index = False)
    
def analyze_ticker(df, ticker):
    generate_plots(df, ticker)
    calculate_metrics(df, ticker)

def analyze_portfolio(df):
    generate_portfolio()
    calculate_portfolio(df)
   
def load_spy():
    spy_df = data_loader.get_data('SPY')
    spy_df['Pct Return'] = spy_df['Price'].pct_change()
    spy_curve = (1 + spy_df['Pct Return']).cumprod()
    spy_curve.name = 'SPY Return Curve'

    return spy_curve