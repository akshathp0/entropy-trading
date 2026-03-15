import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

RISK_FREE_RATE = config['risk_free_rate']

def calculate_sharpe(returns, rf_rate = RISK_FREE_RATE):
    excess_returns = returns - (rf_rate / 252)

    mean = excess_returns.mean()
    std = excess_returns.std()

    sharpe = np.sqrt(252) * (mean / std)

    return sharpe

def calculate_max_drawdown(returns):
    equity = (1 + returns).cumprod()
    
    peaks = equity.cummax()
    drawdowns = (equity - peaks) / peaks
    max_drawdown = drawdowns.min()

    return max_drawdown