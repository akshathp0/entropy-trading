import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

ENTROPY_MODE = config['entropy_mode']

def generate_signal(df, mode = ENTROPY_MODE):
    target_vol = df['Volatility'].expanding().mean()
    
    if mode == 'signal_only' or 'both':
        df['Blend Signal'] = (df['Confidence'] * df['MR Signal'] * (target_vol / df['Volatility'])).clip(0, 1)
    else:
        df['Blend Signal'] = (df['MR Signal'] * (target_vol / df['Volatility'])).clip(0, 1)

    return df

def blended_backtest(df):
    df['Pct Return'] = df['Price'].pct_change()
    df['SPY Return Curve'] = (1 + df['Pct Return']).cumprod()

    df['Blend Return'] = df['Blend Signal'] * df['Pct Return']
    df['Blend Return'] = df['Blend Return'].fillna(0)

    df['Blend Equity Curve'] = (1 + df['Blend Return']).cumprod()

    return df