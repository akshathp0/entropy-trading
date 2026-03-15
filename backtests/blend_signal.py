MOMENTUM_WEIGHTS = {"1_0": 0.8, "1_1": 0.6,
                    "0_0:": 0.4, "0_1": 0.3,
                    "-1_0": 0.2, "-1_1": 0.1
                    }

def generate_signal(df):
    df['w_mom'] = df['State'].map(MOMENTUM_WEIGHTS).fillna(0.5)
    df['w_mr'] = 1 - df['w_mom']

    df['Blend Signal'] = (df['Confidence'] * 
                          df['w_mom'] * df['Momentum Signal'] +
                          df['w_mr'] * df['MR Signal']).clip(0, 1)

    return df

def blended_backtest(df):
    df['Blend Return'] = df['Blend Signal'] * df['Pct Return']
    df['Blend Return'] = df['Blend Return'].fillna(0)

    df['Blend Equity Curve'] = (1 + df['Blend Return']).cumprod()

    return df