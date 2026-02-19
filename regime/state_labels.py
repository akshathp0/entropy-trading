import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

VOL_PERCENTILE = config['vol_percentile']
VOL_WINDOW = config['vol_window']
REGIME_THRESHOLD = config['regime_threshold']
SMOOTHER = config['smoother']
TSTAT_FEATURE_WINDOW = config['tstat_feature_window']

def label_volatility(df, percentile = VOL_PERCENTILE, window = VOL_WINDOW):
    threshold = df['Volatility'].rolling(window = window).quantile(percentile / 100)
    df['Vol State'] = (df['Volatility'] > threshold).astype(int)

    return df

def label_regime(df, threshold = REGIME_THRESHOLD, window = TSTAT_FEATURE_WINDOW):
    thresholds = [
        (df[f'T-Stat_{window}'] >= threshold),
        (df[f'T-Stat_{window}'] <= -threshold)
    ]

    labels = [1, -1]
    
    df['Regime'] = np.select(thresholds, labels, default = 0)

    return df

