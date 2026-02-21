import pandas as pd

def build_matrix(df):
    df['State'] = df['Regime'].astype(str) + '_' + df['Vol State'].astype(str)

    transitions = pd.crosstab(df['State'], df['State'].shift(-1))
    transition_matrix = transitions.div(transitions.sum(axis = 1), axis = 0)

    return transition_matrix