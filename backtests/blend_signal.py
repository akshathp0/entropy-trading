def generate_signal(df):
    df['Blend Signal'] = (df['Confidence'] * df['MR Signal']).clip(0, 1)

    return df

def blended_backtest(df):
    df['Pct Return'] = df['Price'].pct_change()
    df['SPY Return Curve'] = (1 + df['Pct Return']).cumprod()

    df['Blend Return'] = df['Blend Signal'] * df['Pct Return']
    df['Blend Return'] = df['Blend Return'].fillna(0)

    df['Blend Equity Curve'] = (1 + df['Blend Return']).cumprod()

    return df