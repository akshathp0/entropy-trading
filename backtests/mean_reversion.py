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

def generate_signal(df):
    df['MR Signal'] = sigmoid(-df['Z-Score']).clip(0, 1)

    return df

def mr_backtest(df):
    df['MR Return'] = df['MR Signal'] * df['Pct Return']
    df['MR Return'] = df['MR Return'].fillna(0)

    df['MR Equity Curve'] = (1 + df['MR Return']).cumprod()

    return df