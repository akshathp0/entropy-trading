import pandas as pd

def matrix_template(df):
    window = df.dropna(subset = ['State', 'Next State'])

    transitions = pd.crosstab(window['State'], window['Next State'])
    matrix = transitions.div(transitions.sum(axis = 1), axis = 0)

    return matrix, transitions

def initialize_state(df):
    df['State'] = df['Regime'].astype(str) + '_' + df['Vol State'].astype(str)
    df['Next State'] = df['State'].shift(-1)

    return df

def build_stationary_matrix(df):
    matrix, _ = matrix_template(df)

    return matrix