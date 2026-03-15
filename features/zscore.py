import pandas as pd
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

ZSCORE_WINDOW = config['zscore_window']

def calculate_zscore(df, window = ZSCORE_WINDOW):
    rolling_window = df['Log Return'].rolling(window)
    mean = rolling_window.mean()
    std = rolling_window.std()

    df['Z-Score'] = ((df['Log Return'] - mean) / std).shift(1)

    return df