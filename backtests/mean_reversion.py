import numpy as np
from evaluation import metrics
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

LONG_SHORT_THRESHOLD = config['long_short_threshold']
TANH_SCALER = config['tanh_scaler']

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))

    return s

def generate_long_short_signal(df, scaler = TANH_SCALER):
    df['MR Signal'] = np.tanh(-df['Z-Score'] / scaler).clip(-1, 1)

    return df

def generate_long_signal(df):
    df['MR Signal'] = sigmoid(-df['Z-Score']).clip(0, 1)

    return df

def generate_signal(df, threshold = LONG_SHORT_THRESHOLD):
    if (metrics.calculate_annualized_returns(df['Return']) < threshold):
        df = generate_long_short_signal(df)
    else:
        df = generate_long_signal(df)

    return df

def mr_backtest(df):
    df['MR Return'] = df['MR Signal'] * df['Pct Return']
    df['MR Return'] = df['MR Return'].fillna(0)

    df['MR Equity Curve'] = (1 + df['MR Return']).cumprod()

    return df