import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))

    return s

def generate_signal(df):
    df['MR Exposure'] = sigmoid(-df['Z-Score']).clip(0, 1)

    return df

def mr_backtest(df):
    df['MR Return'] = df['MR Exposure'] * df['Pct Return']
    df['MR Return'] = df['MR Return'].fillna(0)

    df['MR Equity Curve'] = (1 + df['MR Return']).cumprod()

    return df