from backtests import blend_signal, mean_reversion, momentum_backtest, exposure
from data import data_loader
from features import momentum, tstat, volatility, zscore
from regime import entropy, matrix, state_labels
from evaluation import metrics, plot
import pandas as pd
import os

def run_pipeline(ticker):
    df = data_loader.get_data(ticker)
    df = tstat.compute_tstat(df)
    df = momentum.calculate_momentum(df)
    df = zscore.calculate_zscore(df)
    df = volatility.compute_volatility(df)
    df = state_labels.label_volatility(df)
    df = state_labels.label_regime(df)
    df = state_labels.smooth_regime(df)
    df = matrix.initialize_state(df)
    df = entropy.build_expanding_entropy(df)
    df = exposure.normalize_entropy(df)
    df = mean_reversion.generate_signal(df)
    df = momentum_backtest.generate_signal(df)
    df = blend_signal.generate_signal(df)
    df = blend_signal.blended_backtest(df)

    return df

def calculate_metrics(df, ticker):
    annualized_returns = metrics.calculate_annualized_returns(df['Blend Return'])
    sharpe = metrics.calculate_sharpe(df['Blend Return'])
    sortino = metrics.calculate_sharpe(df['Blend Return'])
    calmar = metrics.calculate_calmar(df['Blend Return'])
    max_drawdown = metrics.calculate_sharpe(df['Blend Return'])
    state_counts = metrics.state_counts(df)
    entropy_stats = metrics.describe_entropy(df)
    asset = ticker.lower()

    pd.DataFrame({'Annualized Returns': [annualized_returns],
                  'Sharpe': [sharpe],
                  'Sortino': [sortino],
                  'Calmar': [calmar],
                  'Max Drawdown': [max_drawdown],
                    }).to_csv(f'results/{asset}/metrics.csv', index = False)
    
    state_counts.to_csv(f'results/{asset}/state_counts.csv', index = False)
    entropy_stats.to_csv(f'results/{asset}/entropy_stats.csv', index = False)

def generate_plots(df, ticker):
    asset = ticker.lower()
    os.makedirs(f'results/{asset}', exist_ok = True)

    equity_curve = plot.plot_equity_curve(df, ticker)
    equity_curve.savefig(f'results/{asset}/equity_curve.png')

    regime_overlay = plot.plot_regime_overlay(df, ticker)
    regime_overlay.savefig(f'results/{asset}/regime_overlay.png')

    expanding_entropy = plot.plot_expanding_entropy(df, ticker)
    expanding_entropy.savefig(f'results/{asset}/expanding_entropy.png')

    rolling_sharpe = plot.plot_rolling_sharpe(df, ticker)
    rolling_sharpe.savefig(f'results/{asset}/rolling_sharpe')

def analyze_ticker(df, ticker):
    generate_plots(df, ticker)
    calculate_metrics(df, ticker)