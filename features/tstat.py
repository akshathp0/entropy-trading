import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

TSTAT_FEATURE_WINDOW = config['tstat_feature_window']

def compute_tstat(df, window = TSTAT_FEATURE_WINDOW):
    rolling_window = df['Log Return'].rolling(window = window)
    mean = rolling_window.mean()
    std = rolling_window.std(ddof = 1)
    denominator = np.sqrt(window)

    t = mean / (std / denominator)
    df[f'T-Stat_{window}'] = t.shift(1)

    return df