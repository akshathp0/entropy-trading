import yfinance as yf
import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

TICKER = config['ticker']
START = config['start']
END = config['end']

def get_data(ticker = TICKER, start = START, end = END):

    spy = yf.download(ticker, start = start, end = end, auto_adjust = False)['Adj Close']

    spy['Log Return'] = np.log(spy / spy.shift(1))
    spy.dropna(inplace = True)

    return spy