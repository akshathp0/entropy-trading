import numpy as np

def find_entropy(transition_matrix):
    transition_matrix = transition_matrix.replace(0, np.nan)
    entropy = -(transition_matrix * np.log(transition_matrix)).sum(axis = 1)

    return entropy

def build_entropy(df, vector):
    df['Entropy'] = df['State'].map(vector)

    return df