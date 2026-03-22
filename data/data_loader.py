import yfinance as yf
import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

TICKER = config['ticker']
START = config['start']
END = config['end']

def get_data(ticker = TICKER, start = START, end = END):

    df = yf.download(ticker, start = start, end = end, auto_adjust = False)['Adj Close']
    df.columns = ['Price']

    df['Log Return'] = np.log(df / df.shift(1))
    df.dropna(inplace = True)

    df['Return'] = df['Price'].pct_change().dropna()

    return df