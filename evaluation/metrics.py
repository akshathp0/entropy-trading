import numpy as np
import pandas as pd
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

RISK_FREE_RATE = config['risk_free_rate']
START = config['start']
END = config['end']

def calculate_annualized_returns(returns, start = START, end = END):
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    cum_returns = (1 + returns).cumprod().iloc[-1]

    return cum_returns ** (1 / years) - 1

def calculate_sharpe(returns, rf_rate = RISK_FREE_RATE):
    excess_returns = returns - (rf_rate / 252)

    mean = excess_returns.mean()
    std = excess_returns.std()

    sharpe = np.sqrt(252) * (mean / std)

    return sharpe

def calculate_sortino(returns, rf_rate = RISK_FREE_RATE):
    excess_returns = returns - (rf_rate / 252)

    mean = excess_returns.mean()
    downside_std = excess_returns[excess_returns < 0].std()

    sortino = np.sqrt(252) * (mean / downside_std)

    return sortino

def calculate_max_drawdown(returns):
    equity = (1 + returns).cumprod()
    
    peaks = equity.cummax()
    drawdowns = (equity - peaks) / peaks
    max_drawdown = drawdowns.min()

    return max_drawdown

def calculate_calmar(returns):
    return calculate_annualized_returns(returns) / abs(calculate_max_drawdown(returns))

def state_counts(df):
    return df['State'].value_counts()

def describe_entropy(df):
    return df['Expanding Entropy'].describe()