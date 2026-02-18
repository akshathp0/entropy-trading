import yfinance as yf
import pandas as pd
import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

TICKER = config['ticker']
START = config['start']
END = config['end']

spy = yf.download(TICKER, start = START, end = END)['Close']

spy['Log Return'] = np.log(spy / spy.shift(1))
spy.dropna(inplace = True)

print(spy.head())