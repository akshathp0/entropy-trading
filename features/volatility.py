import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

VOLATILITY_FEATURE_WINDOW = config['volatility_feature_window']

def compute_volatility(df, window = VOLATILITY_FEATURE_WINDOW):
    df['Volatility'] = df['Log Return'].rolling(window = window).std().shift(1)

    return df