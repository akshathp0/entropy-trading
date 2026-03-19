import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from evaluation import metrics
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

ROLLING_SHARPE_WINDOW = config['rolling_sharpe_window']

def plot_correlation_heatmap(matrix):
    fig, ax = plt.subplots(figsize = (12, 10))
    sns.heatmap(matrix, annot = False, cmap = 'coolwarm', ax = ax)
    ax.set_title('Initial ETF Universe Heatmap')
    plt.tight_layout()

    return fig

def plot_pca_bars(loadings, n_components):
    fig, axes = plt.subplots(n_components, 1, figsize = (12, n_components * 5))
    for i, ax in enumerate(axes):
        sns.barplot(x = loadings.index, y = loadings[f'PC{i + 1}'], ax = ax)
        ax.set_title(f'PC{i + 1} Bars')

    plt.tight_layout()

    return fig

def plot_equity_curve(df, ticker):
    fig, ax = plt.subplots(figsize = (10, 5))
    sns.lineplot(data = df[['Blend Equity Curve', 'SPY Return Curve']])
    ax.set_title(f'{ticker} Equity Curve vs. SPY')
    plt.tight_layout()

    return fig

def plot_regime_overlay(df, ticker):
    fig, ax = plt.subplots(figsize = (10, 5))
    sns.lineplot(data = df['Price'], ax = ax)
    ax.fill_between(df.index, df['Price'].min(), df['Price'].max(), where = (df['Regime'] == 1), alpha = 0.1, label = 'Bull')
    ax.fill_between(df.index, df['Price'].min(), df['Price'].max(), where = (df['Regime'] == -1), alpha = 0.15, label = 'Bear')
    ax.set_title(f'{ticker} Regime Overlay')
    plt.tight_layout()

    return fig

def plot_expanding_entropy(df, ticker):
    fig, ax = plt.subplots(figsize = (10, 5))
    sns.lineplot(data = df['Expanding Entropy'])
    ax.set_title(f'{ticker} Expanding Entropy Over Time')
    plt.tight_layout()

    return fig

def plot_rolling_sharpe(df, ticker, window = ROLLING_SHARPE_WINDOW):
    fig, ax = plt.subplots(figsize = (10, 5))
    df['Blend Rolling Sharpe'] = df['Blend Return'].rolling(window).apply(lambda x: metrics.calculate_sharpe(pd.Series(x)))
    df['SPY Rolling Sharpe'] = df['Pct Return'].rolling(window).apply(lambda x: metrics.calculate_sharpe(pd.Series(x)))
    sns.lineplot(data = df[['Blend Rolling Sharpe', 'SPY Rolling Sharpe']])
    ax.set_title(f'{ticker} Rolling Sharpe (252 Days) vs. SPY')
    plt.tight_layout()

    return fig