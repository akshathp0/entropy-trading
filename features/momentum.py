import pandas as pd
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

MOMENTUM_WINDOW = config['momentum_window']

def calculate_momentum(df, window = MOMENTUM_WINDOW):
    df['Momentum'] = df['Log Return'].rolling(window).sum().shift(1)

    return df