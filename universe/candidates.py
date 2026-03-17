import yfinance as yf
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

START = config['start']
END = config['end']

def load_data(start = START, end = END):
    tickers = ['QQQ', 'IWM', 'VTV',
               'EFA', 'EEM',
               'TLT', 'IEF', 'HYG', 'LQD',
               'GLD', 'USO', 'DBC',
               'VNQ',
               'AGG',
               'XLI', 'XLB', 'XLP', 'XLY',
               'XLF', 'XLE', 'XLK', 'XLV', 'XLU']
    
    df = yf.download(tickers, start = start, end = end, auto_adjust = False)['Adj Close']
    returns = df.pct_change().dropna()

    return returns, tickers