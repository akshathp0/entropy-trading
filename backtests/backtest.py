import pandas as pd
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

def run_backtest(df):
    df['Pct Return'] = df['Price'].pct_change()
    df['SPY Return Curve'] = (1 + df['Pct Return']).cumprod()

    df['Strategy Return'] = df['Exposure'].shift(1) * df['Pct Return']
    df['Strategy Return'] = df['Strategy Return'].fillna(0)

    df['Equity Curve'] = (1 + df['Strategy Return']).cumprod()

    return df