import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

SCALING_FACTOR = config['momentum_scaler']

def generate_signal(df, scaler = SCALING_FACTOR):
    df['Momentum Signal'] = (df['Momentum'] / scaler).clip(0, 1)

    return df

def momentum_backtest(df):
    df['Momentum Return'] = df['Momentum Signal'] * df['Pct Return']
    df['Momentum Return'] = df['Momentum Return'].fillna(0)

    df['Momentum Equity Curve'] = (1 + df['Momentum Return']).cumprod()

    return df