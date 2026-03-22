import pandas as pd
import numpy as np
from features import tstat
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

ENTROPY_CONFIDENCE = config['entropy_confidence']
TSTAT_EXPOSURE_WINDOW = config['tstat_exposure_window']
C = config['c']

def normalize_entropy(df, gamma = ENTROPY_CONFIDENCE):
    df['Confidence'] = 1 - (df['Rolling Entropy'] / np.log(4))
    df['Confidence'] = df['Confidence'].clip(0, 1)
    df['Confidence'] = df['Confidence'] ** gamma
    df['Confidence'] = df['Confidence'].clip(0, 1)
    df['Confidence'] = df['Confidence'].fillna(1.0)

    return df

def normalize_tstat(df, window = TSTAT_EXPOSURE_WINDOW, c = C):
    df = tstat.compute_tstat(df, window)
    df['Trend'] = np.tanh(df[f'T-Stat_{window}'] / c)
    df['Trend'] = df['Trend'].clip(-1, 1)

    return df

def create_exposure(df):
    df['Exposure'] = (df['Confidence'] * df['Trend']).clip(-1, 1)

    return df